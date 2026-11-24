"""Paired nine-metric comparisons for a completed cutoff sensitivity study."""
import argparse
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', required=True)
    args = parser.parse_args()
    study = Path(args.study_dir)
    state = json.loads((study / 'study_state.json').read_text())
    if state['stage'] != 'complete':
        raise ValueError('Sensitivity evaluation is incomplete')
    records = json.loads((study / 'comparison.json').read_text())
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    order = helpers['ORDER']
    columns = {value: key for key, value in METRICS.items()}
    tags = sorted(records)
    frames = [pd.read_csv(study / tag / 'metrics/results.csv').set_index('perturbation').sort_index()
              for tag in tags]
    if any(not f.index.equals(frames[0].index) or not f.index.is_unique for f in frames):
        raise ValueError('Perturbation libraries differ')
    values = [f[[columns[k] for k in order]].to_numpy(dtype=float) for f in frames]
    means = [np.nanmean(v, axis=0) for v in values]
    for tag, mean in zip(tags, means):
        np.testing.assert_allclose(mean, [records[tag]['metrics'][k] for k in order], rtol=0, atol=1e-10)
    draws = np.random.default_rng(9232).integers(0, len(frames[0]), (2000, len(frames[0])))
    boot = [helpers['resampled_means'](v, draws) for v in values]
    pd_id = order.index('PDCorr')
    pds = [order.index(k) for k in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
    scores = [float(v[pd_id] + v[pds].mean()) for v in means]
    def compare(first, second):
        delta = boot[second] - boot[first]
        primary = delta[:, pd_id] + delta[:, pds].mean(1)
        return {'reference': tags[first], 'candidate': tags[second],
                'reference_floor': records[tags[first]]['floor'],
                'candidate_floor': records[tags[second]]['floor'],
                'primary_difference': {'value': scores[second] - scores[first],
                                       'ci95': np.nanquantile(primary, [.025, .975]).tolist()},
                'rows': [{'metric': key, 'reference': float(means[first][i]),
                          'candidate': float(means[second][i]),
                          'difference': float(means[second][i] - means[first][i]),
                          'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()}
                         for i, key in enumerate(order)]}
    selected = next(i for i, tag in enumerate(tags) if records[tag]['floor'] == .05)
    seven = next(i for i, tag in enumerate(tags) if records[tag]['floor'] == .07)
    comparisons = [compare(0, i) for i in range(1, len(tags))]
    direct = [compare(selected, seven), compare(selected, len(tags)-1)]
    paths = [study / name for name in ('study_config.json', 'support_audit.json', 'comparison.json')]
    paths += [study / tag / 'metrics/results.csv' for tag in tags]
    result = {'time': timestamp(), 'comparisons_to_zero': comparisons, 'direct_comparisons': direct,
              'perturbations': len(frames[0]), 'bootstrap': 2000, 'seed': 9232,
              'source_sha256': {str(p): file_sha256(p) for p in paths},
              'script_sha256': file_sha256(__file__),
              'bootstrap_helper_sha256': file_sha256('scripts/analyze_inference_review.py'),
              'automatic_formal_selection': False,
              'note': 'Exploratory paired intervals for one fixed generation cache and perturbation library; '
                      'not independent training seeds or fresh generation noise. Original metric definitions retained.'}
    (study / 'paired_comparison.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    pd.DataFrame([{'reference_floor': c['reference_floor'], 'candidate_floor': c['candidate_floor'],
                   **{k: v for k, v in row.items() if k != 'ci95'},
                   'ci95_low': row['ci95'][0], 'ci95_high': row['ci95'][1]}
                  for c in comparisons + direct for row in c['rows']]).to_csv(study / 'paired_comparison.csv', index=False)
    floors = [records[tag]['floor'] for tag in tags]
    audit = json.loads((study / 'support_audit.json').read_text())
    fig, axes = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    for i, (key, ax) in enumerate(zip(order, axes.flat)):
        ax.plot(floors, [v[i] for v in means], marker='o', markersize=4)
        ax.axvline(audit['minimum_positive_training_value'], color='grey', linestyle='--', linewidth=1)
        ax.set_title(key + (' (lower is better)' if key in ('MSE', 'MAE') else ''))
        ax.set_xlabel('Zero cutoff (log-expression units)')
        ax.grid(alpha=.2)
    fig.suptitle('Epoch25 fixed-CFG validation sensitivity; dashed line: minimum positive training value')
    fig.savefig(study / 'metric_sensitivity.png', dpi=180)
    plt.close(fig)
    print(json.dumps({'zero_to_0p05': comparisons[selected-1], '0p05_to_maximum': direct[-1]}, indent=2))


if __name__ == '__main__':
    main()
