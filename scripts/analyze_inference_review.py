"""Paired perturbation bootstrap for completed validation inference cases.

Intervals describe variation across the held-out perturbations. They do not
measure independent training seeds or repeated generation noise.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from ocoot.evaluate import METRICS
from ocoot.runtime import file_sha256, timestamp


ORDER = ['PDCorr', 'DEOver', 'DEPrec', 'DirAgr', 'PDS_L1', 'PDS_L2',
         'PDS_cos', 'MSE', 'MAE']


def resampled_means(values, draws):
    selected = values[draws]
    counts = np.isfinite(selected).sum(axis=1)
    return np.divide(np.nansum(selected, axis=1), counts,
                     out=np.full(counts.shape, np.nan), where=counts > 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--review-dir', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=9232)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('bootstrap must be positive')
    out = Path(args.review_dir)
    protocol = json.loads((out / 'review_config.json').read_text())
    if protocol['split'] != 'validation':
        raise ValueError('This diagnostic analysis must not select settings using test data')
    comparison = json.loads((out / 'comparison.json').read_text())
    reference = 'ema_none'
    if reference not in comparison or len(comparison) < 2:
        raise ValueError('At least one completed guided case and the reference are required')
    columns = {v: k for k, v in METRICS.items()}
    source_files = [out / 'review_config.json', out / 'comparison.json']
    frames = {}
    reference_indices = np.load(out / reference / 'indices.npy')
    for tag, case in comparison.items():
        folder = out / tag
        np.testing.assert_array_equal(np.load(folder / 'indices.npy'), reference_indices)
        configuration = json.loads((folder / 'evaluation_config.json').read_text())
        for key in ['checkpoint_sha256', 'split', 'cap', 'batch_size', 'seed',
                    'steps', 'cfg_min', 'cfg_max', 'clamp', 'de']:
            if configuration[key] != protocol[key]:
                raise ValueError(f'Paired configuration mismatch: {tag}/{key}')
        if configuration['guidance'] != case['guidance']:
            raise ValueError(f'Guidance mismatch: {tag}')
        csv_path = folder / 'metrics/results.csv'
        source_files.extend([csv_path, folder / 'evaluation_config.json', folder / 'indices.npy'])
        frame = pd.read_csv(csv_path).set_index('perturbation').sort_index()
        if not frame.index.is_unique:
            raise ValueError(f'Duplicate perturbations: {tag}')
        values = frame[[columns[key] for key in ORDER]].to_numpy(dtype=np.float64)
        np.testing.assert_allclose(np.nanmean(values, axis=0),
                                   [case['metrics'][key] for key in ORDER], atol=1e-10, rtol=0)
        frames[tag] = (frame.index, values)
    labels = frames[reference][0]
    for tag, (index, _) in frames.items():
        if not index.equals(labels):
            raise ValueError(f'Perturbation order mismatch: {tag}')
    draws = np.random.default_rng(args.seed).integers(0, len(labels), (args.bootstrap, len(labels)))
    bootstraps = {tag: resampled_means(values, draws) for tag, (_, values) in frames.items()}
    reference_means = np.nanmean(frames[reference][1], axis=0)
    rows = []
    score_differences = {}
    for tag, (_, values) in frames.items():
        if tag == reference:
            continue
        means = np.nanmean(values, axis=0)
        delta_draws = bootstraps[tag] - bootstraps[reference]
        score_draws = delta_draws[:, 0] + delta_draws[:, [4, 5, 6]].mean(axis=1)
        score_lo, score_hi = np.quantile(score_draws[np.isfinite(score_draws)], [.025, .975])
        score_differences[tag] = {
            'delta': comparison[tag]['score'] - comparison[reference]['score'],
            'ci95': [float(score_lo), float(score_hi)],
        }
        for i, metric in enumerate(ORDER):
            valid = delta_draws[:, i][np.isfinite(delta_draws[:, i])]
            lo, hi = np.quantile(valid, [.025, .975])
            rows.append({'case': tag, 'guidance': comparison[tag]['guidance'], 'metric': metric,
                         'reference': float(reference_means[i]), 'candidate': float(means[i]),
                         'delta': float(means[i] - reference_means[i]),
                         'delta_ci95_low': float(lo), 'delta_ci95_high': float(hi),
                         'valid_bootstrap_draws': len(valid),
                         'higher_is_better': metric not in ['MSE', 'MAE']})
    pd.DataFrame(rows).to_csv(out / 'paired_differences.csv', index=False)
    result = {'time': timestamp(), 'reference': reference, 'perturbations': len(labels),
              'bootstrap_draws': args.bootstrap, 'seed': args.seed,
              'completed_cases': list(comparison), 'rows': rows,
              'selection_score_differences': score_differences,
              'source_sha256': {str(path): file_sha256(path) for path in source_files},
              'script_sha256': file_sha256(__file__),
              'note': 'Validation diagnostic only; candidate minus ordinary conditional reference. '
                      'Matched perturbation bootstrap, not training-seed or generation-noise uncertainty. '
                      'The ordinary conditional field is excluded from final formal CFG selection.'}
    (out / 'paired_differences.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    fig, axes = plt.subplots(3, 3, figsize=(11, 8), layout='constrained')
    for ax, metric in zip(axes.flat, ORDER):
        selected = [row for row in rows if row['metric'] == metric]
        for i, row in enumerate(selected):
            ax.plot([row['delta_ci95_low'], row['delta_ci95_high']], [i, i], color='#3478a6')
            ax.plot(row['delta'], i, 'o', color='#3478a6')
        ax.axvline(0, color='#858d99', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(selected)), [f'w={row["guidance"]:g}' for row in selected])
        ax.set_title(metric + (' (lower is better)' if metric in ['MSE', 'MAE'] else ''))
        ax.set_xlabel('Difference from conditional-only reference')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
    fig.suptitle(f'Paired validation diagnostics; cap={protocol["cap"]}; 95% perturbation intervals')
    fig.savefig(out / 'paired_differences.png', dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(json.dumps({'cases': list(comparison), 'perturbations': len(labels),
                      'output': str(out / 'paired_differences.json')}, indent=2))


if __name__ == '__main__':
    main()
