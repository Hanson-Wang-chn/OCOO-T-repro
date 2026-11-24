"""Audit and compare the shared versus copied-then-untied projection pilots."""
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
    parser.add_argument('--candidate', default='runs/pilot_pert_projection_separate')
    parser.add_argument('--outdir', default='runs/pert_projection_analysis')
    args = parser.parse_args()
    runs = [Path(args.reference), Path(args.candidate)]
    configs = [yaml.safe_load((p / 'config.yaml').read_text()) for p in runs]
    normalized = [{**c, 'model': asdict(ModelConfig(**c['model']))} for c in configs]
    assert normalized[0]['model'].pop('pert_projection') == 'shared'
    assert normalized[1]['model'].pop('pert_projection') == 'separate'
    for cfg in normalized:
        cfg.pop('run_dir')
    if normalized[0] != normalized[1]:
        raise ValueError('Pilots differ beyond projection tying and run directory')
    np.testing.assert_array_equal(*[np.load(p / 'train_indices.npy') for p in runs])
    folders = [p / 'validation_epoch_012' for p in runs]
    np.testing.assert_array_equal(*[np.load(p / 'indices.npy') for p in folders])
    markers = [json.loads((p / 'generation_state.json').read_text()) for p in folders]
    if markers[0] != markers[1] or markers[0]['completed'] != 960:
        raise ValueError('Sampling populations/options differ')
    checkpoints = [torch.load(p / 'model.pt', map_location='cpu', mmap=True, weights_only=False) for p in folders]
    if any(c['step'] != 384 or c['weights'] != 'raw_pilot_only' for c in checkpoints):
        raise ValueError('Requires equal-budget raw diagnostic weights')
    for key in ('esm', 'gene_indices', 'missing', 'position'):
        if not torch.equal(checkpoints[0]['model'][key], checkpoints[1]['model'][key]):
            raise ValueError(f'Frozen input buffers differ: {key}')
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    available = {'raw': [k for k in helpers['ORDER'] if k not in ('DEOver', 'DEPrec', 'DirAgr')],
                 'clamped': helpers['ORDER']}
    columns = {v: k for k, v in METRICS.items()}
    sources = [p / file for p in runs for file in ('config.yaml', 'train_indices.npy', 'finished.json')]
    sources += [p / file for p in folders for file in ('model.pt', 'indices.npy', 'generation_state.json')]
    results = {}
    for mode, order in available.items():
        metric_dir = 'metrics' if mode == 'raw' else 'metrics_clamped_de'
        frames = [pd.read_csv(p / metric_dir / 'results.csv').set_index('perturbation').sort_index() for p in folders]
        if not frames[0].index.equals(frames[1].index) or not frames[0].index.is_unique:
            raise ValueError('Perturbation library differs')
        values = [f[[columns[k] for k in order]].to_numpy(dtype=float) for f in frames]
        for p, array in zip(folders, values):
            summary = json.loads((p / metric_dir / 'summary.json').read_text())
            np.testing.assert_allclose(np.nanmean(array, axis=0), [summary[k] for k in order], atol=1e-10, rtol=0)
            sources.extend([p / metric_dir / 'results.csv', p / metric_dir / 'summary.json'])
        draws = np.random.default_rng(9232).integers(0, len(frames[0]), (2000, len(frames[0])))
        boot = [helpers['resampled_means'](v, draws) for v in values]
        delta = boot[1] - boot[0]
        rows = []
        for i, name in enumerate(order):
            rows.append({'metric': name, 'reference': float(np.nanmean(values[0][:, i])),
                         'candidate': float(np.nanmean(values[1][:, i])),
                         'difference': float(np.nanmean(values[1][:, i]) - np.nanmean(values[0][:, i])),
                         'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()})
        pd_id = order.index('PDCorr')
        pds = [order.index(k) for k in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
        score_delta = delta[:, pd_id] + delta[:, pds].mean(1)
        means = [np.nanmean(v, axis=0) for v in values]
        scores = [float(m[pd_id] + m[pds].mean()) for m in means]
        results[mode] = {'rows': rows, 'reference_score': scores[0], 'candidate_score': scores[1],
                         'score_difference': scores[1] - scores[0],
                         'score_difference_ci95': np.nanquantile(score_delta, [.025, .975]).tolist()}
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    result = {'time': timestamp(), 'reference': str(runs[0]), 'candidate': str(runs[1]),
              'updates': 384, 'generated_cells': 960, 'perturbations': 60,
              'sampling': markers[0]['options'], 'results': results,
              'source_sha256': {str(p): file_sha256(p) for p in sources},
              'script_sha256': file_sha256(__file__),
              'note': 'Copied initial projection, same data/budget/noise; paired fixed-library perturbation intervals. '
                      'Short raw diagnostics do not establish mature-EMA or independent-training-seed behavior; no automatic main switch.'}
    (out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame([{'mode': mode, **row} for mode, r in results.items() for row in r['rows']]).to_csv(out / 'comparison.csv', index=False)
    print(json.dumps({mode: {k: v for k, v in r.items() if k != 'rows'} for mode, r in results.items()}, indent=2))


if __name__ == '__main__':
    main()
