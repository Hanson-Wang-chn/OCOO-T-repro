"""Compare the two saved E45 EMA weights on identical validation batches."""
import argparse
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.postprocess import apply_zero_floor
from ocoot.runtime import exclusive_run, file_sha256, timestamp
from guidance_followup import OUT as GUIDANCE, write
from prefetch_validation import CALIBRATION, read


OUT = CALIBRATION.parent / 'postgate_ema_study'
MODES = {'conditional': ('screen_00', None, .1, .9),
         'guided': ('screen_full_interval', 2., 0., 1.)}
PIECES = ('prediction.npy', 'indices.npy', 'generation_state.json')


def hashes(path, names):
    return {name: file_sha256(path / name) for name in names}


def register():
    protocol = read(CALIBRATION / 'calibration_config.json')
    preserved = read(CALIBRATION.parent / 'snapshots/step_50310/preserved.json')
    source = Path(preserved['archive_path'])
    canonical = Path(protocol['checkpoint'])
    archive_sha = file_sha256(source)
    assert archive_sha == preserved['sha256'] and file_sha256(canonical) == protocol['checkpoint_sha256']
    archive = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    default = torch.load(canonical, map_location='cpu', mmap=True, weights_only=False)
    assert archive['step'] == default['step'] == 50310 and archive['config'] == default['config']
    assert set(archive['model']) == set(default['model']) and len(default['model']) == 193
    standard, slower = archive['ema']['0.9996'], archive['ema']['0.9999']
    assert set(standard) == set(slower) and len(standard) == 189
    for name, tensor in default['model'].items():
        original = standard[name] if name in standard else archive['model'][name]
        assert torch.equal(tensor, original), name
    assert any(not torch.equal(standard[name], slower[name]) for name in standard)
    assert all(torch.isfinite(tensor).all() for tensor in slower.values())
    paths = [source, canonical, CALIBRATION / 'calibration_config.json', GUIDANCE / 'report.json',
             CALIBRATION / 'postprocessing/review.json', Path('scripts/evaluate.py'), Path(__file__)]
    plan = {'time': timestamp(), 'step': 50310, 'ema': ['0.9996', '0.9999'], 'split': 'validation',
            'seed': 4242, 'steps': 50, 'batch_size': 32, 'cap': 16, 'cells': 960,
            'floors': [0., .05], 'test_access': False, 'archive': str(source),
            'modes': {mode: list(values) for mode, values in MODES.items()},
            'inputs_sha256': {str(source): archive_sha, str(canonical): protocol['checkpoint_sha256'],
                              **{str(path): file_sha256(path) for path in paths[2:]}},
            'promotion_rule': 'Review only if the fixed guided comparison shows a material precision gain '
                              'with no convincing degradation in the primary score, DEOver or expression errors. '
                              'The paper default remains unchanged in the full training and original evaluation.'}
    assert read(paths[3])['test_access'] is False and read(paths[4])['selected'] is None
    path = OUT / 'plan.json'
    if not path.exists():
        write(path, plan)
    else:
        assert {k: v for k, v in read(path).items() if k != 'time'} == {k: v for k, v in plan.items() if k != 'time'}
    print(source, flush=True)


def analyze():
    plan = read(OUT / 'plan.json')
    assert all(file_sha256(path) == digest for path, digest in plan['inputs_sha256'].items())
    protocol = read(CALIBRATION / 'calibration_config.json')
    data = BenchmarkData(protocol['data_dir'])
    indices = data.evaluation_indices('validation', 16)
    assert len(indices) == 960
    tools = runpy.run_path('scripts/analyze_full_calibration.py')
    labels = sorted({data.vocab['all_genes'][number] for number in data.meta['pert'][indices]})
    assert len(labels) == 60
    draws = np.random.default_rng(9232).integers(0, len(labels), (2000, len(labels)))
    output = {'time': timestamp(), 'test_access': False, 'bootstrap': 2000, 'by_mode_and_floor': {}}
    for mode, (tag, guidance, lower, upper) in MODES.items():
        reference, alternative = CALIBRATION / tag, OUT / mode
        expected = {'noise_std': 1., 'batch_size': 32, 'steps': 50, 'guidance': guidance,
                    'cfg_min': lower, 'cfg_max': upper, 'seed': 4242}
        assert all(read(path / 'generation_state.json') == {'completed': 960, 'total': 960,
                   'options': {**expected, 'clamp': clamp}} for path, clamp in ((reference, False), (alternative, True)))
        for path in (reference, alternative):
            np.testing.assert_array_equal(np.load(path / 'indices.npy'), indices)
        evaluation = read(alternative / 'evaluation_config.json')
        assert evaluation['checkpoint_sha256'] == plan['inputs_sha256'][plan['archive']]
        assert (evaluation['weights'], evaluation['step'], evaluation['split'], evaluation['cap']) == ('ema_0.9999', 50310, 'validation', 16)
        raw_paths = (reference / 'prediction.npy', alternative / 'prediction_raw.npy')
        output[mode + '_inputs_sha256'] = {str(path): file_sha256(path) for path in raw_paths}
        for floor in plan['floors']:
            metric_dirs = [reference / 'metrics_clamped', alternative / 'metrics']
            if floor:
                metric_dirs = [OUT / mode / ('floor_0p05_' + decay) for decay in plan['ema']]
                for raw_path, metrics in zip(raw_paths, metric_dirs):
                    if not (metrics / 'summary.json').exists():
                        official_metrics(data, indices, apply_zero_floor(np.load(raw_path, mmap_mode='r'), floor), metrics, de=True)
            values = {ema: tools['load_metrics'](metrics, labels, read(metrics / 'summary.json'))
                      for ema, metrics in zip(plan['ema'], metric_dirs)}
            cases, pairs = tools['bootstrap_candidates'](values, draws)
            output['by_mode_and_floor'][f'{mode}/{floor:g}'] = {'cases': cases, 'paired': pairs[0]}
    write(OUT / 'report.json', output)


if __name__ == '__main__':
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('register', 'analyze'))
    stage = parser.parse_args().stage
    with exclusive_run(OUT, 'review'):
        register() if stage == 'register' else analyze()
