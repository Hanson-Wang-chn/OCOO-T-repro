"""Paired, fixed-protocol epoch10-to15 validation diagnostic comparison."""
import json
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp

helpers = runpy.run_path('scripts/analyze_inference_review.py')
order = helpers['ORDER']
resampled_means = helpers['resampled_means']
folders = [Path(f'runs/main_v1/validation_epoch_{epoch:03d}') for epoch in (10, 15)]
indices = [np.load(folder / 'indices.npy') for folder in folders]
np.testing.assert_array_equal(*indices)
markers = [json.loads((folder / 'generation_state.json').read_text()) for folder in folders]
assert markers[0]['options'] == markers[1]['options']
assert all(m['completed'] == m['total'] == len(indices[0]) for m in markers)
configurations = [json.loads((folder / 'metrics_de_diagnostic/diagnostic_config.json').read_text()) for folder in folders]
assert all(c['weights'] == 'ema_0.9996' for c in configurations)
assert configurations[0]['metadata_sha256'] == configurations[1]['metadata_sha256']
columns = {v: k for k, v in METRICS.items()}
frames = [pd.read_csv(folder / 'metrics_de_diagnostic/results.csv').set_index('perturbation').sort_index() for folder in folders]
assert frames[0].index.equals(frames[1].index) and frames[0].index.is_unique
values = [frame[[columns[k] for k in order]].to_numpy(dtype=float) for frame in frames]
for folder, array in zip(folders, values):
    summary = json.loads((folder / 'metrics_de_diagnostic/summary.json').read_text())
    np.testing.assert_allclose(np.nanmean(array, axis=0), [summary[k] for k in order], atol=1e-10, rtol=0)
draws = np.random.default_rng(9232).integers(0, len(frames[0]), (2000, len(frames[0])))
boot = [resampled_means(array, draws) for array in values]
delta = boot[1] - boot[0]
rows = []
for i, metric in enumerate(order):
    low, high = np.nanquantile(delta[:, i], [.025, .975])
    rows.append({'metric': metric, 'epoch10': float(np.nanmean(values[0][:, i])),
                 'epoch15': float(np.nanmean(values[1][:, i])),
                 'difference': float(np.nanmean(values[1][:, i]) - np.nanmean(values[0][:, i])),
                 'ci95': [float(low), float(high)]})
score_draws = delta[:, 0] + delta[:, [4, 5, 6]].mean(axis=1)
score_difference = rows[0]['difference'] + np.mean([rows[i]['difference'] for i in (4, 5, 6)])
files = [folder / suffix for folder in folders for suffix in ('indices.npy', 'generation_state.json',
         'metrics_de_diagnostic/diagnostic_config.json', 'metrics_de_diagnostic/results.csv')]
result = {'time': timestamp(), 'epochs': [10, 15], 'steps': [c['step'] for c in configurations],
          'perturbations': len(frames[0]), 'generated_cells': len(indices[0]),
          'sampling': markers[0]['options'], 'rows': rows,
          'score_difference': {'value': float(score_difference), 'ci95': np.nanquantile(score_draws, [.025, .975]).tolist()},
          'bootstrap': 2000, 'seed': 9232, 'source_sha256': {str(p): file_sha256(p) for p in files},
          'script_sha256': file_sha256(__file__),
          'note': 'Paired60-perturbation diagnostic with fixed reference library/indices/noise/sampler. '
                  'Intervals are not independent-training-seed or generation-noise uncertainty. Ordinary conditional field, not formal CFG or test.'}
out = Path('runs/main_v1/observation/epoch010_to015')
out.mkdir(exist_ok=True)
(out / 'comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
pd.DataFrame(rows).to_csv(out / 'comparison.csv', index=False)
print(json.dumps(result, indent=2, allow_nan=False))
