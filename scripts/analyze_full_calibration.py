"""Audit all four complete validation candidates and their paired uncertainty.

This reads completed calibration/CPU-selection outputs. It never recomputes
official metrics, changes the selection, or reads the test population.
"""
import argparse
import itertools
import json
import os
from pathlib import Path
import runpy

import numpy as np
import pandas as pd

from ocoot.runtime import exclusive_run, file_sha256, timestamp


ORDER = ('PDCorr', 'DEOver', 'DEPrec', 'DirAgr', 'PDS_L1', 'PDS_L2', 'PDS_cos', 'MSE', 'MAE')
COLUMNS = ('pearson_delta', 'overlap_at_N', 'precision_at_N', 'de_direction_match',
           'discrimination_score_l1', 'discrimination_score_l2', 'discrimination_score_cosine', 'mse', 'mae')


def read(path):
    return json.loads(Path(path).read_text())


def load_metrics(directory, labels, expected):
    directory = Path(directory)
    frame = pd.read_csv(directory / 'results.csv').set_index('perturbation').sort_index()
    if not frame.index.is_unique or frame.index.tolist() != labels:
        raise ValueError('Expected the exact complete validation perturbation library')
    values = frame[list(COLUMNS)].to_numpy(float)
    if np.isinf(values).any() or any(np.isnan(values[:, i]).any() for i in range(9) if i != 3):
        raise ValueError('Only the official undefined direction agreement may be missing')
    means = np.nanmean(values, axis=0)
    for metrics in (read(directory / 'summary.json'), expected):
        if set(metrics) != set(ORDER):
            raise ValueError('All nine metrics are required')
        np.testing.assert_allclose(means, [metrics[k] for k in ORDER], rtol=0, atol=1e-10)
    return values


def bootstrap_candidates(values, draws):
    # Direct three-dimensional indexing, independent of the selection code and
    # the earlier resampled_means helper. Preserve each case's NaN convention.
    means = {key: np.nanmean(array, axis=0) for key, array in values.items()}
    boot = {key: np.nanmean(array[draws], axis=1) for key, array in values.items()}
    cases = {}
    for key in values:
        cases[key] = {
            'metrics': {metric: {'value': float(means[key][i]),
                                'ci95': np.nanquantile(boot[key][:, i], [.025, .975]).tolist(),
                                'finite_perturbations': int(np.isfinite(values[key][:, i]).sum())}
                        for i, metric in enumerate(ORDER)},
            'primary': {'value': float(means[key][0] + means[key][4:7].mean()),
                        'ci95': np.nanquantile(boot[key][:, 0] + boot[key][:, 4:7].mean(1), [.025, .975]).tolist()}}
    pairs = []
    for reference, candidate in itertools.combinations(values, 2):
        change = boot[candidate] - boot[reference]
        pairs.append({'reference': reference, 'candidate': candidate,
                      'rows': [{'metric': metric, 'reference': float(means[reference][i]),
                                'candidate': float(means[candidate][i]),
                                'difference': float(means[candidate][i] - means[reference][i]),
                                'ci95': np.nanquantile(change[:, i], [.025, .975]).tolist()}
                               for i, metric in enumerate(ORDER)],
                      'primary_difference': {
                          'value': float(means[candidate][0] + means[candidate][4:7].mean()
                                         - means[reference][0] - means[reference][4:7].mean()),
                          'ci95': np.nanquantile(change[:, 0] + change[:, 4:7].mean(1), [.025, .975]).tolist()}})
    return cases, pairs


def plot(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    limits = result['thresholds']
    thresholds = dict(zip(ORDER, (limits['min_val_pdcorr'], limits['min_val_deover'],
        limits['min_val_deprec'], limits['min_val_diragr'], limits['min_val_pds_each'],
        limits['min_val_pds_each'], limits['min_val_pds_each'], limits['max_val_mse'], limits['max_val_mae'])))
    keys = list(result['cases'])
    labels = [f"R{result['cases'][k]['tag'][-2:]} / floor {result['cases'][k]['zero_floor']:g}" for k in keys]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), layout='constrained')
    for ax, metric in zip(axes.flat, ORDER):
        for i, key in enumerate(keys):
            point = result['cases'][key]['metrics'][metric]
            color = '#3478a6' if result['cases'][key]['zero_floor'] == 0 else '#c07628'
            ax.plot(point['ci95'], [i, i], color=color, linewidth=1.5)
            ax.plot(point['value'], i, 'o', color=color, markersize=4)
        ax.axvline(thresholds[metric], color='#777777', linestyle='--', linewidth=1)
        ax.set_yticks(range(len(keys)), labels)
        ax.set_title(metric + (' (lower is better)' if metric in ('MSE', 'MAE') else ''))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-3, 3))
        ax.set_xlabel('Mean and 95% perturbation interval')
    configs = []
    for key in keys[::2]:
        case = result['cases'][key]
        configs.append(f"R{case['tag'][-2:]}: w={case['guidance']:g}, t=[{case['cfg_min']:g},{case['cfg_max']:g}]")
    fig.suptitle('Complete E35 validation, 4825 cells / 60 perturbations\n' + '; '.join(configs) +
                 '\nDashed lines: unchanged individual quality checks; all four candidates shown')
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'full_candidates.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def analyze(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    for group in ('source_sha256', 'prerequisite_sha256'):
        for path, digest in plan[group].items():
            if file_sha256(path) != digest:
                raise ValueError('Registered analysis input changed: ' + path)
    root = Path(plan['validation_root'])
    complete = read(root / 'completed.json')
    if (complete['test_access'] or complete['final_test_protocol_adopted']
            or complete['status'] != 'completed_validation_review_required'):
        raise ValueError('A completed intermediate validation-only review is required')
    if complete['plan_sha256'] != file_sha256(root / 'plan.json'):
        raise ValueError('Intermediate calibration plan changed')
    for path, digest in complete['output_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Completed calibration output changed: ' + path)
    selector = runpy.run_path('scripts/select_validation_postprocess.py')
    calibration, provenance, original, cfg, data, indices, refinements = selector['load_calibration'](root / 'calibration')
    if len(indices) != 4825 or provenance['step'] != 39130:
        raise ValueError('This registered audit is for the frozen E35 complete validation')
    review_path = calibration / 'postprocessing/review.json'
    review = read(review_path)
    if review['test_access'] or review['cells_each'] != 4825 or review['perturbations'] != 60:
        raise ValueError('Expected complete validation-only CPU postprocessing')
    labels = sorted(data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][indices]))
    cases, values, paths = {}, {}, [plan_path, root / 'completed.json', root / 'plan.json', review_path,
                                  calibration / 'calibration_config.json', calibration / 'all_results.json',
                                  calibration / 'selected.json']
    if len(review['candidates']) != 4:
        raise ValueError('Show both registered floors of both full refinements')
    for record in sorted(refinements, key=lambda r: r['tag']):
        source, raw = selector['validate_refinement'](calibration, provenance, cfg, data, indices, record)
        paths.extend(source / name for name in ('prediction.npy', 'indices.npy', 'generation_state.json'))
        for floor in (0., .05):
            candidate = [c for c in review['candidates'] if (c['tag'], c['zero_floor']) == (record['tag'], floor)]
            if len(candidate) != 1:
                raise ValueError('Missing or duplicate full candidate')
            candidate = candidate[0]
            key = record['tag'] + ('_floor0' if floor == 0 else '_floor0p05')
            metric_dir = Path(candidate['candidate_metrics_path']).parent
            expected_dir = source / 'metrics_clamped' if floor == 0 else calibration / ('postprocessing/' + record['tag'] + '_floor_0p05/metrics')
            if metric_dir.resolve() != expected_dir.resolve():
                raise ValueError('Candidate metrics must come from the registered unchanged raw refinement')
            values[key] = load_metrics(metric_dir, labels, candidate['metrics'])
            cases[key] = {k: candidate[k] for k in ('tag', 'zero_floor', 'guidance', 'cfg_min', 'cfg_max', 'checks')}
            cases[key]['generated_zero_fraction'] = float(np.mean(raw < floor)) if floor else float(np.mean(raw <= 0))
            cases[key]['raw_negative_fraction'] = float(np.mean(raw < 0))
            paths.extend(metric_dir / name for name in ('summary.json', 'results.csv', 'population_counts.json'))
            population = read(metric_dir / 'population_counts.json')
            if population != {'real_perturbed_cells': 4825, 'generated_cells': 4825, 'shared_real_controls': 4976}:
                raise ValueError('Complete validation/reference/control counts differ')
    draws = np.random.default_rng(plan['seed']).integers(0, 60, (plan['bootstrap'], 60))
    summaries, pairs = bootstrap_candidates(values, draws)
    for key in cases:
        cases[key].update(summaries[key])
    result = {'time': timestamp(), 'cases': cases, 'all_six_paired_comparisons': pairs,
              'bootstrap': plan['bootstrap'], 'seed': plan['seed'], 'thresholds': review['thresholds'],
              'existing_selector_choice': review['selected'], 'test_access': False,
              'changes_to_selection_or_training': False,
              'note': 'Unadjusted descriptive perturbation bootstrap; fixed trained model and generation seed. '
                      'All four variants and all six pairwise contrasts are retained. '
                      'No independent training-seed or generation-noise uncertainty; no official-metric recomputation.',
              'source_sha256': {str(path): file_sha256(path) for path in paths}}
    out = Path(plan['outdir'])
    with exclusive_run(out, 'calibration_audit'):
        target = out / 'analysis.json'
        if target.exists():
            raise FileExistsError('Keep the completed audit; inspect its existing provenance')
        target.write_text(json.dumps(result, indent=2, allow_nan=False))
        pd.DataFrame([{'case': key, 'metric': metric, **row, 'ci95_low': row['ci95'][0],
                       'ci95_high': row['ci95'][1]} for key in cases for metric, row in cases[key]['metrics'].items()]
                     ).drop(columns='ci95').to_csv(out / 'full_candidates.csv', index=False)
        plot(result, out)
        outputs = [out / name for name in ('analysis.json', 'full_candidates.csv', 'full_candidates.png', 'full_candidates.pdf')]
        (out / 'completed.json').write_text(json.dumps({'time': timestamp(), 'cpu_only': True, 'test_access': False,
            'plan_sha256': file_sha256(plan_path), 'output_sha256': {str(p): file_sha256(p) for p in outputs}}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    result = analyze(args.plan)
    print(json.dumps({'cases': list(result['cases']), 'test_access': False}))


if __name__ == '__main__':
    main()
