"""Compare matched-budget prediction-space or scalar-initialization pilots."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch
import yaml

from ocoot.evaluate import METRICS
from ocoot.model import ModelConfig
from ocoot.runtime import file_sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', default='runs/pilot_sigma_1.0_cellinit_1.0')
    parser.add_argument('--candidate', default='runs/pilot_xpred_vloss')
    parser.add_argument('--outdir', default='runs/xpred_vloss_analysis')
    parser.add_argument('--factor', choices=('prediction_space', 'value_input_init'), default='prediction_space')
    args = parser.parse_args()
    runs = [Path(args.reference), Path(args.candidate)]
    configs = [yaml.safe_load((p / 'config.yaml').read_text()) for p in runs]
    normalized = [{**c, 'model': asdict(ModelConfig(**c['model']))} for c in configs]
    expected = ('v', 'x') if args.factor == 'prediction_space' else ('xavier', 'fan_in')
    for config, value in zip(normalized, expected):
        if config['model'].pop(args.factor) != value:
            raise ValueError(f'Expected {args.factor}={value}')
        config.pop('run_dir')
    if normalized[0] != normalized[1]:
        raise ValueError(f'Pilots differ beyond {args.factor} and run directory')
    sources = []
    for run in runs:
        finished = json.loads((run / 'finished.json').read_text())
        if finished['step'] != 384 or finished['epoch'] != 12:
            raise ValueError('Incomplete or unequal pilot budget')
        sources.extend(run / name for name in ('config.yaml', 'train_indices.npy', 'finished.json',
                                                'source_manifest_step_0.json', 'source_snapshot_step_0.tar.gz'))
    np.testing.assert_array_equal(*[np.load(p / 'train_indices.npy') for p in runs])
    folders = [p / 'validation_epoch_012' for p in runs]
    np.testing.assert_array_equal(*[np.load(p / 'indices.npy') for p in folders])
    markers = [json.loads((p / 'generation_state.json').read_text()) for p in folders]
    if markers[0] != markers[1] or markers[0]['completed'] != 960:
        raise ValueError('Sampling populations/options differ')
    options = markers[0]['options']
    if options['guidance'] is not None or options['steps'] != 50 or options['batch_size'] != 32:
        raise ValueError('Requires original ordinary conditional Heun50 pilot settings')
    checkpoints = [torch.load(p / 'model.pt', map_location='cpu', mmap=True, weights_only=False) for p in folders]
    if any(c['step'] != 384 or c['weights'] != 'raw_pilot_only' for c in checkpoints):
        raise ValueError('Requires equal-budget raw diagnostic weights')
    for config, checkpoint in zip(configs, checkpoints):
        if checkpoint['config'] != config:
            raise ValueError('Saved weights and recorded config disagree')
    states = [checkpoint['model'] for checkpoint in checkpoints]
    if states[0].keys() != states[1].keys():
        raise ValueError('Parameter/buffer names differ')
    if any(states[0][key].shape != states[1][key].shape for key in states[0]):
        raise ValueError('Parameter/buffer shapes differ')
    for key in ('esm', 'gene_indices', 'missing', 'position'):
        if not torch.equal(states[0][key], states[1][key]):
            raise ValueError(f'Frozen input buffers differ: {key}')
    sources.extend(p / name for p in folders for name in ('model.pt', 'indices.npy', 'generation_state.json',
                                                          'prediction.npy', 'prediction_raw.npy'))
    # Some historical raw pilots have no separate prediction_raw.npy: prediction.npy is
    # already unclamped. Retain the real artifact names without manufacturing a cache.
    sources = [p for p in sources if p.name != 'prediction_raw.npy' or p.exists()]
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    available = {'raw': [k for k in helpers['ORDER'] if k not in ('DEOver', 'DEPrec', 'DirAgr')],
                 'clamped': helpers['ORDER']}
    columns = {v: k for k, v in METRICS.items()}
    results = {}
    perturbations = None
    for mode, order in available.items():
        metric_dir = 'metrics' if mode == 'raw' else 'metrics_clamped_de'
        frames = [pd.read_csv(p / metric_dir / 'results.csv').set_index('perturbation').sort_index() for p in folders]
        if len(frames[0]) != 60 or not frames[0].index.equals(frames[1].index) or not frames[0].index.is_unique:
            raise ValueError('Perturbation library differs')
        library = frames[0].index.tolist()
        if perturbations is not None and perturbations != library:
            raise ValueError('Raw/nonnegative perturbation libraries differ')
        perturbations = library
        values = [f[[columns[k] for k in order]].to_numpy(dtype=float) for f in frames]
        if any(np.isinf(value).any() for value in values):
            raise ValueError('Infinite metric in a completed pilot')
        for p, array in zip(folders, values):
            summary = json.loads((p / metric_dir / 'summary.json').read_text())
            np.testing.assert_allclose(np.nanmean(array, axis=0), [summary[k] for k in order], atol=1e-10, rtol=0)
            sources.extend([p / metric_dir / 'results.csv', p / metric_dir / 'summary.json'])
        draws = np.random.default_rng(9232).integers(0, 60, (2000, 60))
        boot = [helpers['resampled_means'](value, draws) for value in values]
        delta = boot[1] - boot[0]
        means = [np.nanmean(value, axis=0) for value in values]
        rows = [{'metric': name, 'reference': float(means[0][i]), 'candidate': float(means[1][i]),
                 'difference': float(means[1][i] - means[0][i]),
                 'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()}
                for i, name in enumerate(order)]
        pd_id = order.index('PDCorr')
        pds = [order.index(key) for key in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
        scores = [float(mean[pd_id] + mean[pds].mean()) for mean in means]
        score_delta = delta[:, pd_id] + delta[:, pds].mean(1)
        results[mode] = {'rows': rows, 'reference_score': scores[0], 'candidate_score': scores[1],
                         'score_difference': scores[1] - scores[0],
                         'score_difference_ci95': np.nanquantile(score_delta, [.025, .975]).tolist()}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    note = ('Paper Eqs.5/6: initial neural weights match, but initial velocity functions/losses differ. '
            'This unpatched diagnostic is not the patched Fig.5 experiment.' if args.factor == 'prediction_space'
            else 'Only the scalar input weight initialization changes; zero output heads give the same initial '
                 'velocity function. Initialization is not specified by the paper.')
    result = {'time': timestamp(), 'reference': str(runs[0]), 'candidate': str(runs[1]), 'factor': args.factor,
              'updates': 384, 'generated_cells': 960, 'perturbations': perturbations,
              'sampling': options, 'results': results,
              'source_sha256': {str(p): file_sha256(p) for p in sources},
              'script_sha256': file_sha256(__file__), 'automatic_main_switch': False,
              'note': note + ' Same data, architecture, budget and sampling noise. Fixed-library paired '
                            'perturbation intervals do not measure independent training-seed uncertainty. '
                            'Short raw diagnostics are not mature EMA or test results.'}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame([{'mode': mode, **row} for mode, r in results.items() for row in r['rows']]).to_csv(out / 'comparison.csv', index=False)
    print(json.dumps({mode: {key: value for key, value in r.items() if key != 'rows'}
                      for mode, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
