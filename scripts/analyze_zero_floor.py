"""Audit and pair completed validation-only zero-floor sensitivity cases.

This analysis reads one immutable prediction population. It neither changes
postprocessing defaults nor selects a final inference configuration.
"""
import argparse
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def analyze(study, bootstrap=2000, seed=9232):
    study = Path(study)
    if bootstrap < 1:
        raise ValueError('bootstrap must be positive')
    state = json.loads((study / 'study_state.json').read_text())
    config = json.loads((study / 'study_config.json').read_text())
    settings = config['time_independent_settings']
    records = json.loads((study / 'comparison.json').read_text())
    if state['stage'] != 'complete' or set(state['cases']) != set(records):
        raise ValueError('Sensitivity evaluation is incomplete')
    tags = sorted(records, key=lambda tag: records[tag]['floor'])
    floors = [records[tag]['floor'] for tag in tags]
    if (len(tags) < 2 or floors[0] != 0 or len(set(floors)) != len(floors)
            or sorted(settings['floors']) != floors
            or any(not np.isfinite(v) or not 0 <= v < settings['minimum_positive_training_expression']
                   for v in floors)):
        raise ValueError('Expected distinct registered floors below the training positive support, including zero')
    generation = Path(settings['generation_dir'])
    extension = 'extension_manifest_sha256' in settings
    prediction = generation / ('stitched_predictions.npy' if extension else 'prediction.npy')
    indices_path = generation / 'indices.npy'
    checkpoint_path = Path(settings['checkpoint'])
    pinned = {checkpoint_path: settings['checkpoint_sha256'],
              prediction: settings['prediction_sha256'],
              indices_path: settings['indices_sha256'],
              Path('src/ocoot/evaluate.py'): settings['metric_source_sha256'],
              Path('scripts/diagnose_zero_floor.py'): settings['source_sha256']}
    if extension:
        pinned[generation / 'completed.json'] = settings['extension_manifest_sha256']
    for path, digest in pinned.items():
        if file_sha256(path) != digest:
            raise ValueError(f'Sensitivity source changed: {path}')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    if checkpoint['step'] != settings['step'] or checkpoint.get('weights') != settings['weights']:
        raise ValueError('Checkpoint step or weights disagree with sensitivity protocol')
    data = BenchmarkData(checkpoint['config']['data_dir'])
    if file_sha256(data.path / 'metadata.npz') != settings['metadata_sha256']:
        raise ValueError('Benchmark metadata changed')
    np.testing.assert_array_equal(checkpoint['model']['gene_indices'].numpy(), data.meta['gene_indices'])
    indices = np.load(indices_path)
    if len(np.unique(indices)) != len(indices) or not np.isin(indices, data.val).all():
        raise ValueError('Population must contain distinct validation cells only')
    if extension:
        np.testing.assert_array_equal(np.sort(indices), data.val)
    predicted = np.load(prediction, mmap_mode='r')
    if (predicted.shape != (len(indices), data.x.shape[1])
            or not np.isfinite(predicted).all() or np.min(predicted) < 0):
        raise ValueError('Invalid nonnegative prediction population')
    labels = sorted(data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][indices]))
    helpers = runpy.run_path('scripts/analyze_inference_review.py')
    order = helpers['ORDER']
    columns = {value: key for key, value in METRICS.items()}
    paths = [study / name for name in ('study_state.json', 'study_config.json', 'comparison.json')]
    paths += list(pinned) + [data.path / 'metadata.npz', data.path / 'vocab.json']
    values, means = [], []
    for tag, floor in zip(tags, floors):
        folder = study / tag
        record = json.loads((folder / 'completed.json').read_text())
        if record != records[tag]:
            raise ValueError(f'Case completion differs from comparison: {tag}')
        frame = pd.read_csv(folder / 'metrics/results.csv').set_index('perturbation').sort_index()
        if not frame.index.is_unique or frame.index.tolist() != labels:
            raise ValueError(f'Perturbation library mismatch: {tag}')
        array = frame[[columns[key] for key in order]].to_numpy(dtype=np.float64)
        mean = np.nanmean(array, axis=0)
        np.testing.assert_allclose(mean, [record['metrics'][k] for k in order], atol=1e-10, rtol=0)
        summary = json.loads((folder / 'metrics/summary.json').read_text())
        np.testing.assert_allclose(mean, [summary[k] for k in order], atol=1e-10, rtol=0)
        projected = np.where(predicted < floor, 0, predicted)
        np.testing.assert_allclose(float(np.mean(projected == 0)), record['zero_fraction'], atol=1e-15, rtol=0)
        np.testing.assert_allclose(float(np.mean(np.abs(projected - predicted), dtype=np.float64)),
                                   record['mean_absolute_change'], atol=1e-15, rtol=0)
        values.append(array)
        means.append(mean)
        paths.extend(folder / name for name in ('completed.json', 'metrics/results.csv', 'metrics/summary.json'))
    draws = np.random.default_rng(seed).integers(0, len(labels), (bootstrap, len(labels)))
    boot = [helpers['resampled_means'](array, draws) for array in values]
    comparisons = []
    for case in range(1, len(tags)):
        delta = boot[case] - boot[0]
        primary = delta[:, 0] + delta[:, [4, 5, 6]].mean(axis=1)
        precision_delta = values[case][:, 2] - values[0][:, 2]
        finite = precision_delta[np.isfinite(precision_delta)]
        largest = np.sort(finite)[-3:].sum()
        total = finite.sum()
        comparisons.append({
            'reference': tags[0], 'candidate': tags[case],
            'reference_floor': floors[0], 'candidate_floor': floors[case],
            'primary_difference': {
                'value': float(means[case][0] + means[case][[4, 5, 6]].mean()
                               - means[0][0] - means[0][[4, 5, 6]].mean()),
                'ci95': np.nanquantile(primary, [.025, .975]).tolist()},
            'rows': [{'metric': key, 'reference': float(means[0][i]),
                      'candidate': float(means[case][i]),
                      'difference': float(means[case][i] - means[0][i]),
                      'ci95': np.nanquantile(delta[:, i], [.025, .975]).tolist()}
                     for i, key in enumerate(order)],
            'precision_heterogeneity': {
                'increased': int(np.sum(finite > 0)), 'decreased': int(np.sum(finite < 0)),
                'equal': int(np.sum(finite == 0)), 'median_difference': float(np.median(finite)),
                'top_three_share_of_net_change': float(largest / total) if total != 0 else None,
                'per_condition': [{'perturbation': label, 'difference': float(value)}
                                  for label, value in zip(labels, precision_delta) if np.isfinite(value)]}})
    result = {'time': timestamp(), 'validation_cells': len(indices), 'perturbations': len(labels),
              'full_validation_population': extension, 'bootstrap': bootstrap, 'seed': seed,
              'comparisons_to_zero': comparisons,
              'source_sha256': {str(path): file_sha256(path) for path in paths},
              'script_sha256': file_sha256(__file__),
              'bootstrap_helper_sha256': file_sha256('scripts/analyze_inference_review.py'),
              'automatic_formal_selection': False,
              'note': 'Exploratory paired intervals for fixed validation predictions and perturbations. '
                      'Not independent training seeds or generation-noise uncertainty. '
                      'Original metric definitions and inference defaults remain unchanged.'}
    atomic_json(study / 'paired_comparison.json', result)
    pd.DataFrame([{'reference_floor': item['reference_floor'], 'candidate_floor': item['candidate_floor'],
                   **{key: value for key, value in row.items() if key != 'ci95'},
                   'ci95_low': row['ci95'][0], 'ci95_high': row['ci95'][1]}
                  for item in comparisons for row in item['rows']]).to_csv(study / 'paired_comparison.csv', index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study-dir', required=True)
    parser.add_argument('--bootstrap', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=9232)
    args = parser.parse_args()
    torch.set_num_threads(1)
    with exclusive_run(args.study_dir, 'zero_floor_analysis'):
        result = analyze(args.study_dir, args.bootstrap, args.seed)
    print(json.dumps({key: result[key] for key in
                      ('validation_cells', 'perturbations', 'comparisons_to_zero')}, indent=2))


if __name__ == '__main__':
    main()
