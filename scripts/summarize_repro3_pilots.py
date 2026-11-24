"""Verify the registered pilots and select the full-training configuration."""
from datetime import datetime
import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
import yaml

from analyze_inference_review import resampled_means
from calibrate import write_json_atomic
from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp


COLUMNS = list(METRICS)
NAMES = list(METRICS.values())


def main():
    plan_path = Path('configs/repro3_pilots/plan.json')
    plan = json.loads(plan_path.read_text())
    root = Path(plan['run_dir'])
    case_state = json.loads((root / 'pilot_state.json').read_text())
    if case_state['stage'] != 'complete' or case_state['plan_sha256'] != file_sha256(plan_path):
        raise ValueError('All registered pilots must be complete before selection')
    configs = {name: yaml.safe_load(Path(case['config']).read_text()) for name, case in plan['cases'].items()}
    baseline = configs['control']
    data = BenchmarkData(baseline['data_dir'])
    train_indices = np.sort(np.random.default_rng(plan['seed']).choice(data.train, plan['training_subset_cells'], replace=False))
    validation_indices = data.evaluation_indices(cap=baseline['training']['val_generation_cap'])
    labels = sorted(data.vocab['all_genes'][i] for i in np.unique(data.meta['pert'][validation_indices]))
    if len(validation_indices) != plan['validation_cells'] or len(labels) != plan['validation_perturbations']:
        raise ValueError('Validation population differs from the registered plan')
    draws = np.random.default_rng(plan['bootstrap_seed']).integers(0, len(labels), (plan['bootstrap_draws'], len(labels)))
    reference = {}
    rows = []
    training = {}
    source_hashes = {}
    weights = np.array([{'PDCorr': 1., 'PDS_L1': 1/3, 'PDS_L2': 1/3, 'PDS_cos': 1/3,
                         'MSE': -plan['mse_weight']}.get(name, 0.) for name in NAMES])

    for name, case in plan['cases'].items():
        run = root / name
        config = configs[name]
        expected = {**baseline, 'run_dir': str(run),
                    'model': {**baseline['model'], 'null_condition_init': case['null_condition_init']},
                    'training': {**baseline['training'], 'noise_std': case['noise_std']}}
        if (file_sha256(case['config']) != case['config_sha256'] or config != expected
                or yaml.safe_load((run / 'config.yaml').read_text()) != expected):
            raise ValueError(f'Unregistered settings for {name}')
        np.testing.assert_array_equal(np.load(run / 'train_indices.npy'), train_indices)
        generation = run / f'validation_epoch_{plan["epochs"]:03d}'
        np.testing.assert_array_equal(np.load(generation / 'indices.npy'), validation_indices)
        state = json.loads((generation / 'generation_state.json').read_text())
        options = {**config['sampling'], 'noise_std': case['noise_std'],
                   'batch_size': config['training']['val_generation_batch'], 'seed': plan['validation_generation_seed']}
        if state != {'total': len(validation_indices), 'completed': len(validation_indices), 'options': options}:
            raise ValueError(f'Unregistered or incomplete generation for {name}')
        model_path = generation / 'model.pt'
        checkpoint = torch.load(model_path, map_location='cpu', mmap=True, weights_only=False)
        finished = json.loads((run / 'finished.json').read_text())
        if (checkpoint['config'] != expected or checkpoint['weights'] != f'ema_{plan["ema_decay"]}'
                or checkpoint['step'] != plan['optimizer_updates'] or finished['step'] != plan['optimizer_updates']):
            raise ValueError(f'Mismatched EMA checkpoint or training budget for {name}')
        del checkpoint
        events = [json.loads(line) for line in (run / 'events.jsonl').read_text().splitlines()]
        start = next(event for event in events if event['event'] == 'start')
        if start['world_size'] != plan['gpu_workers']:
            raise ValueError(f'GPU count differs for {name}')
        samples = [event for event in events if event['event'] == 'train' and event['step'] >= 40]
        probe = next(event for event in reversed(events) if event['event'] == 'validation')
        training[name] = {'start': start['time'], 'finish': finished['time'],
                          'elapsed_seconds': (datetime.fromisoformat(finished['time']) - datetime.fromisoformat(start['time'])).total_seconds(),
                          'median_cells_per_second': median(event['cells_per_sec'] for event in samples),
                          'max_allocated_gb_on_rank0': max(event['peak_gpu_GB'] for event in samples),
                          'last_denoising_probe': probe}
        source_hashes[str(model_path)] = file_sha256(model_path)
        for floor in plan['floors']:
            tag = 'floor_' + format(floor, '.12g').replace('.', 'p')
            output = run / 'zero_floor_validation' / tag
            summary = json.loads((output / 'completed.json').read_text())
            metrics = summary['metrics']
            csv = output / 'metrics/results.csv'
            frame = pd.read_csv(csv).set_index('perturbation').sort_index()
            if list(frame.index) != labels:
                raise ValueError(f'Mismatched validation perturbations for {name}/{tag}')
            values = frame[COLUMNS].to_numpy(dtype=np.float64)
            means = np.array([metrics[key] for key in NAMES])
            np.testing.assert_allclose(np.nanmean(values, axis=0), means, atol=1e-10, rtol=0)
            bootstraps = resampled_means(values, draws)
            if name == 'control':
                reference[floor] = means, bootstraps
            reference_means, reference_boots = reference[floor]
            score_columns = weights != 0
            score_deltas = (bootstraps - reference_boots)[:, score_columns] @ weights[score_columns]
            rows.append({'case': name, 'floor': floor, **metrics, 'score': float(means @ weights),
                         'delta_score': float((means - reference_means) @ weights),
                         'delta_score_ci95': np.quantile(score_deltas[np.isfinite(score_deltas)], [.025, .975]).tolist(),
                         'delta_pdcorr': float(means[NAMES.index('PDCorr')] - reference_means[NAMES.index('PDCorr')]),
                         'delta_deprec': float(means[NAMES.index('DEPrec')] - reference_means[NAMES.index('DEPrec')])})
            source_hashes[str(csv)] = file_sha256(csv)

    rules = plan['adoption']
    checks = {}
    for name in plan['cases']:
        if name == 'control':
            continue
        clamped, floored = [row for row in rows if row['case'] == name]
        checks[name] = {'positive_ci95': clamped['delta_score_ci95'][0] > rules['minimum_score_ci95_low_exclusive'],
                        'pdcorr': clamped['delta_pdcorr'] >= rules['minimum_delta_pdcorr'],
                        'deprec': floored['delta_deprec'] >= rules['minimum_delta_deprec_at_0p05']}
    feasible = [row for row in rows if row['floor'] == 0 and row['case'] in checks and all(checks[row['case']].values())]
    selected = max(feasible, key=lambda row: row['delta_score'])['case'] if feasible else 'control'
    report = {'time': timestamp(), 'plan_sha256': file_sha256(plan_path), 'selected': selected,
              'candidate_checks': checks, 'training': training, 'rows': rows, 'artifact_sha256': source_hashes,
              'bootstrap_scope': 'Paired over validation perturbations, with one training and generation seed.'}
    write_json_atomic(root / 'summary.json', report)
    pd.DataFrame(rows).to_csv(root / 'summary.csv', index=False)
    print(json.dumps({'selected': selected, 'candidate_checks': checks, 'summary': str(root / 'summary.json')}, indent=2))


if __name__ == '__main__':
    main()
