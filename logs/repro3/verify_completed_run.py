"""Verify the saved third-round artifacts and compare the two frozen rounds."""
import argparse
from datetime import datetime
import gzip
import json
from pathlib import Path
from statistics import median
import subprocess
import sys

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path.cwd() / 'scripts'))
from analyze_inference_review import resampled_means
from calibrate import write_json_atomic
from report import PAPER
from select_validation_postprocess import verified_metrics
from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS
from ocoot.postprocess import primary_score
from ocoot.runtime import file_sha256, timestamp

RUN = Path('runs/repro3_main')
BASE = Path('runs/repro2_main')


def load(path):
    return json.loads(path.read_text())


def certify_training():
    finished = load(RUN / 'finished.json')
    cfg = yaml.safe_load((RUN / 'config.yaml').read_text())
    plan_path = Path('configs/repro3_main.yaml')
    adoption = load(Path('runs/repro3_pilots/main_adoption.json'))
    assert cfg == yaml.safe_load(plan_path.read_text())
    assert adoption['config_sha256'] == file_sha256(RUN / 'config.yaml')
    data = BenchmarkData(cfg['data_dir'])
    np.testing.assert_array_equal(np.load(RUN / 'train_indices.npy'), data.train)
    assert len(data.train) == 572545
    execution = load(RUN / 'training_execution.json')
    assert execution['world_size'] == 4
    assert execution['global_batch'] == 512
    assert execution['local_micro_batch'] == 16
    assert execution['accumulation_by_rank'] == [8, 8, 8, 8]
    assert execution['dynamo_ddp_optimizer'] is False
    event_path = RUN / 'events.jsonl'
    event_file = event_path.open() if event_path.exists() else gzip.open(event_path.with_suffix('.jsonl.gz'), 'rt')
    with event_file as stream:
        events = [json.loads(line) for line in stream]
    starts = [r for r in events if r['event'] == 'start']
    assert starts[0]['step'] == 0 and all(row['training_cells'] == len(data.train) for row in starts)
    periodic = [r for r in events if r['event'] == 'validation_generation']
    assert [r['epoch'] for r in periodic] == list(range(5, int(finished['epoch'])+1, 5))
    for row in periodic:
        np.testing.assert_allclose(row['selection_score'], primary_score(row) - 10 * row['MSE'], rtol=0, atol=1e-10)
    best = max(periodic, key=lambda r: r['selection_score'])
    assert finished['step'] == int(finished['epoch']) * 1118 and finished['step'] >= best['step']
    paths = [RUN/'last.pt', RUN/'best_generation.pt', RUN/'final_ema.pt',
             RUN/f'validation_epoch_{best["epoch"]:03d}/model.pt']
    last, selected, final, chosen = [torch.load(path, map_location='cpu', mmap=True, weights_only=False) for path in paths]
    assert last['step'] == final['step'] == finished['step'] and last['pending_generation_epoch'] is None
    assert len(last['rng_by_rank']) == 4 and set(last['ema']) == {'0.9999','0.9996'}
    assert selected['step'] == chosen['step'] == best['step']
    assert selected['weights'] == chosen['weights'] == 'ema_0.9996'
    assert selected['criterion'] == 'validation_PDCorr_plus_mean_PDS_minus_weighted_MSE'
    assert selected['selection_score'] == best['selection_score'] == finished['best_validation_generation']
    assert all(torch.equal(value, chosen['model'][name]) for name, value in selected['model'].items())
    result = {'time': timestamp(), 'training_start': starts[0]['time'],
              'elapsed_seconds': (datetime.fromisoformat(finished['time']) - datetime.fromisoformat(starts[0]['time'])).total_seconds(),
              'finished': finished, 'best_epoch': best['epoch'], 'best_step': selected['step'],
              'best_metrics': selected['metrics'], 'periodic': periodic,
              'ranks_saved': len(last['rng_by_rank']), 'optimizer_slots': len(last['optimizer']['state']),
              'median_cells_per_second': median(row['cells_per_sec'] for row in events if row['event'] == 'train' and row['step'] >= 40),
              'max_rank0_gb': max(row['peak_gpu_GB'] for row in events if row['event'] == 'train'),
              'training_launch_count': len(starts), 'config_sha256_at_launch': adoption['config_sha256'],
              'config_sha256_after_annotations': file_sha256(plan_path), 'sha256': {str(path): file_sha256(path) for path in paths}}
    write_json_atomic(RUN / 'training_completion_verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'periodic'}, indent=2))


def paired_results(reference, candidate, split, data):
    paths = [reference, candidate]
    official = data.evaluation_indices(split)
    perturbations = {data.vocab['all_genes'][int(i)] for i in np.unique(data.meta['pert'][official])}
    for folder in paths:
        np.testing.assert_array_equal(np.load(folder/'indices.npy'), official)
    summaries = [verified_metrics(folder/'metrics', conditions=len(perturbations), labels=perturbations)
                 for folder in paths]
    frames = [pd.read_csv(folder/'metrics/results.csv').set_index('perturbation').sort_index() for folder in paths]
    assert frames[0].index.equals(frames[1].index)
    columns, metrics = list(METRICS), list(METRICS.values())
    values = [frame[columns].to_numpy(dtype=float) for frame in frames]
    draws = np.random.default_rng(9232).integers(0, len(perturbations), (2000, len(perturbations)))
    a, b = (resampled_means(value, draws) for value in values)
    ranges = np.nanquantile(b-a, [.025,.975], axis=0)
    rows = [{'metric': k, 'repro2': summaries[0][k], 'repro3': summaries[1][k],
             'delta_from_repro2': summaries[1][k]-summaries[0][k],
             'paired_ci95': ranges[:, i].tolist()} for i,k in enumerate(metrics)]
    if split=='test':
        for row in rows:
            row['paper'] = PAPER[row['metric']]
            row['delta_from_paper'] = row['repro3']-row['paper']
    return {'time': timestamp(), 'split': split, 'perturbations': len(perturbations), 'cells': len(official),
            'replicates': len(draws), 'seed': 9232, 'rows': rows,
            'source_sha256': {str(folder/'metrics/results.csv'): file_sha256(folder/'metrics/results.csv') for folder in paths}}


def certify_test():
    test = RUN/'test_full_postprocess'
    clamp = RUN/'test_full_clamp0'
    validation = RUN/'validation_full_postprocess'
    state = load(RUN/'pipeline_state.json')
    completed = load(test/'postprocess_test_completed.json')
    protocol = load(test/'test_protocol.json')
    selection_file = RUN/'calibration/postprocessing/selected.json'
    stats = load(RUN/'training_completion_verification.json')
    assert state['stage'] == 'evaluated' and state['test_metrics'] == completed['metrics']
    assert completed['protocol'] == protocol
    assert protocol['selection_sha256'] == file_sha256(selection_file)
    assert protocol['validation_completion_sha256'] == file_sha256(validation/'promotion_completed.json')
    assert protocol['checkpoint_sha256'] == stats['sha256'][str(RUN/'best_generation.pt')]
    assert not completed['selection_changed_after_test'] and completed['new_postprocess_generated_samples'] == 0
    for file, expected in completed['output_sha256'].items():
        assert file_sha256(file) == expected, file
    raw_sha = file_sha256(test/'prediction_raw.npy')
    assert raw_sha == file_sha256(clamp/'prediction_raw.npy')
    data = BenchmarkData(yaml.safe_load((RUN / 'config.yaml').read_text())['data_dir'])
    official = data.evaluation_indices('test')
    labels = {data.vocab['all_genes'][int(i)] for i in np.unique(data.meta['pert'][official])}
    assert len(official) == 26878 and len(labels) == 380
    for folder in (test, clamp):
        np.testing.assert_array_equal(np.load(folder/'indices.npy'), official)
        counts = load(folder/'metrics/population_counts.json')
        assert counts == {'real_perturbed_cells': 26878, 'generated_cells': 26878, 'shared_real_controls': 4976}
        verified_metrics(folder/'metrics', conditions=380, labels=labels)
    paired = paired_results(BASE/'test_full_postprocess', test, 'test', data)
    write_json_atomic(RUN/'results/paired_repro2_comparison.json', paired)
    pd.DataFrame(paired['rows']).to_csv(RUN/'results/paired_repro2_comparison.csv', index=False)
    validation_comparison = paired_results(BASE/'validation_full_postprocess', validation, 'validation', data)
    write_json_atomic(RUN/'full_validation_comparison.json', validation_comparison)
    gpu_rows = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                       text=True).strip().splitlines()
    result = {'time': timestamp(), 'evaluation_finished': state['time'], 'population': counts,
              'perturbations': len(labels), 'all_output_hashes_verified': len(completed['output_sha256']),
              'frozen_selection_sha256': protocol['selection_sha256'], 'shared_raw_sha256': raw_sha,
              'both_macro_means_verified': True, 'gpu_processes_at_verification': gpu_rows}
    write_json_atomic(RUN/'test_completion_verification.json', result)
    print(json.dumps(result, indent=2))
    print(pd.DataFrame(paired['rows']).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('train', 'test'))
    args = parser.parse_args()
    torch.set_num_threads(2)
    (certify_training if args.phase == 'train' else certify_test)()
