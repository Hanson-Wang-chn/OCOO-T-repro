"""Pair two completed periodic validation caches under an identical protocol."""
import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', default='runs/main_v1')
    parser.add_argument('--reference-epoch', type=int, required=True)
    parser.add_argument('--candidate-epoch', type=int, required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    epochs = [args.reference_epoch, args.candidate_epoch]
    if epochs[0] >= epochs[1]:
        parser.error('candidate epoch must be later than reference epoch')
    folders = [Path(args.run_dir) / f'validation_epoch_{epoch:03d}' for epoch in epochs]
    indices = [np.load(folder / 'indices.npy') for folder in folders]
    np.testing.assert_array_equal(*indices)
    markers = [json.loads((folder / 'generation_state.json').read_text()) for folder in folders]
    if markers[0]['options'] != markers[1]['options']:
        raise ValueError('Sampling options differ; no matched-noise comparison')
    if not all(marker['completed'] == marker['total'] == len(indices[0]) for marker in markers):
        raise ValueError('Incomplete generation')
    protocols = [json.loads((folder / 'metrics_de_diagnostic/diagnostic_config.json').read_text()) for folder in folders]
    if any(c['weights'] != 'ema_0.9996' for c in protocols):
        raise ValueError('Periodic comparison requires the same EMA variant')
    for key in ('metadata_sha256', 'metric_source_sha256', 'sampling'):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f'Protocols differ: {key}')
    checkpoints = []
    sources = []
    for folder, protocol in zip(folders, protocols):
        for name, key in [('model.pt', 'checkpoint_sha256'), ('prediction.npy', 'prediction_sha256'),
                          ('indices.npy', 'indices_sha256')]:
            if file_sha256(folder / name) != protocol[key]:
                raise ValueError(f'Cached source changed: {folder / name}')
        checkpoint = torch.load(folder / 'model.pt', map_location='cpu', mmap=True, weights_only=False)
        if checkpoint['step'] != protocol['step'] or checkpoint['weights'] != protocol['weights']:
            raise ValueError('Checkpoint step/weights differ from metric protocol')
        checkpoints.append(checkpoint)
        sources.extend(folder / name for name in ('model.pt', 'prediction.npy', 'indices.npy', 'generation_state.json',
                                                  'metrics_de_diagnostic/diagnostic_config.json',
                                                  'metrics_de_diagnostic/results.csv', 'metrics_de_diagnostic/summary.json'))
    if checkpoints[0]['config'] != checkpoints[1]['config'] or protocols[0]['step'] >= protocols[1]['step']:
        raise ValueError('Configuration changed or checkpoint order is invalid')
    for key in ('esm', 'gene_indices', 'missing', 'position'):
        if not torch.equal(checkpoints[0]['model'][key], checkpoints[1]['model'][key]):
            raise ValueError(f'Frozen buffers differ: {key}')
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    order = helpers['ORDER']
    columns = {value: key for key, value in METRICS.items()}
    frames = [pd.read_csv(folder / 'metrics_de_diagnostic/results.csv').set_index('perturbation').sort_index()
              for folder in folders]
    if not frames[0].index.equals(frames[1].index) or not frames[0].index.is_unique:
        raise ValueError('Perturbation libraries differ')
    values = [frame[[columns[key] for key in order]].to_numpy(dtype=float) for frame in frames]
    means = [np.nanmean(value, axis=0) for value in values]
    for folder, mean in zip(folders, means):
        summary = json.loads((folder / 'metrics_de_diagnostic/summary.json').read_text())
        np.testing.assert_allclose(mean, [summary[key] for key in order], atol=1e-10, rtol=0)
    draws = np.random.default_rng(9232).integers(0, len(frames[0]), (2000, len(frames[0])))
    boot = [helpers['resampled_means'](value, draws) for value in values]
    delta = boot[1] - boot[0]
    rows = [{'metric': metric, 'reference': float(means[0][i]), 'candidate': float(means[1][i]),
             'difference': float(means[1][i] - means[0][i]),
             'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()} for i, metric in enumerate(order)]
    pd_id = order.index('PDCorr')
    pds = [order.index(key) for key in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
    scores = [float(mean[pd_id] + mean[pds].mean()) for mean in means]
    score_delta = delta[:, pd_id] + delta[:, pds].mean(1)
    result = {'time': timestamp(), 'epochs': epochs, 'steps': [c['step'] for c in protocols],
              'perturbations': len(frames[0]), 'generated_cells': len(indices[0]), 'sampling': markers[0]['options'],
              'rows': rows, 'reference_score': scores[0], 'candidate_score': scores[1],
              'score_difference': {'value': scores[1] - scores[0],
                                   'ci95': np.nanquantile(score_delta, [.025, .975]).tolist()},
              'bootstrap': 2000, 'seed': 9232, 'source_sha256': {str(p): file_sha256(p) for p in sources},
              'script_sha256': file_sha256(__file__),
              'note': 'Fixed-library paired perturbation intervals under identical sampling indices/noise/options. '
                      'Does not measure independent training-seed or generation-noise uncertainty. '
                      'Periodic validation diagnostic, not formal calibrated full validation or test.'}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    temporary = out / 'comparison.json.tmp'
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
    temporary.replace(out / 'comparison.json')
    pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
    print(json.dumps({'epochs': epochs, 'score_difference': result['score_difference']}, indent=2))


if __name__ == '__main__':
    main()
