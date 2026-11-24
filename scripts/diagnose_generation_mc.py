"""CPU variance budget of immutable validation predictions, with fixed strata.

Official MSE is retained. Conditional variances of singleton strata cannot be
identified; extrapolating a full MC contribution requires an explicit common-
variance assumption. This produces no corrected predictions or benchmark score.
"""
import argparse
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.mc_diagnostics import mean_sampling_variance
from ocoot.postprocess import apply_zero_floor
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def figures(out, rows, summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    names = list(summaries)
    labels = {'raw': 'Raw', 'clamp0': 'Clamp0', 'floor0p05': 'Floor0.05'}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
    points = [
        ('mean_official_MSE_same_estimable_conditions', 'Observed official MSE', '#333333'),
        ('mean_repeated_strata_MC_contribution', 'Repeated strata MC component', '#267e91'),
        ('mean_common_variance_pooled_proxy', 'Common-variance proxy', '#bd772c'),
    ]
    for offset, (key, label, color) in zip([-.16, 0., .16], points):
        axes[0].plot(np.arange(len(names)) + offset,
                     [summaries[name][key] for name in names], 'o', label=label, color=color)
    axes[0].set_xticks(range(len(names)), [labels[name] for name in names])
    axes[0].set(ylabel='Squared expression units', title='Macro mean on estimable conditions')
    axes[0].legend(fontsize=7)
    selected = [r for r in rows if r['case'] == 'clamp0']
    valid = [r for r in selected if r['pooled_common_within_stratum_variance_proxy'] is not None]
    axes[1].scatter([r['cells'] for r in valid],
                    [r['pooled_common_within_stratum_variance_proxy'] / r['official_MSE'] for r in valid],
                    s=16, color='#267e91', alpha=.75)
    axes[1].set(xlabel='Generated cells per perturbation', ylabel='Pooled proxy / observed MSE',
                title='Assumption-based ratio, clamp0')
    axes[2].scatter([r['cells'] for r in selected], [r['repeated_cell_fraction'] for r in selected],
                    s=16, color='#596e91', alpha=.75)
    axes[2].set(xlabel='Generated cells per perturbation', ylabel='Cells in repeated strata / all cells',
                title='Conditional variance estimation coverage', ylim=(-.03, 1.03))
    for ax in axes:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('E25 fixed validation cache: Monte Carlo diagnostics\n'
                 'Strata = biological batch × original RNG/batch-size segment; genes may be correlated')
    for suffix in ('png', 'pdf'):
        fig.savefig(out / f'mc_variance_diagnostic.{suffix}', dpi=180, bbox_inches='tight')
    plt.close(fig)


def diagnose(plan_path):
    plan_path = Path(plan_path)
    plan = read(plan_path)
    for group in ('source_sha256', 'input_sha256'):
        for path, digest in plan[group].items():
            if file_sha256(path) != digest:
                raise ValueError('Registered MC diagnostic input changed: ' + path)
    torch.set_num_threads(1)
    data = BenchmarkData(plan['data_dir'])
    assert file_sha256(data.path / 'expression.npy') == plan['expression_sha256']
    indices = np.load(plan['indices_file'])
    assert len(indices) == len(np.unique(indices)) == 4825
    np.testing.assert_array_equal(np.sort(indices), data.val)
    assert len(np.unique(data.meta['cell'][indices])) == 1
    assert len(np.unique(data.meta['batch'][indices])) == 56
    raw = np.load(plan['raw_prediction_file'], mmap_mode='r')
    clamped = np.load(plan['clamped_prediction_file'], mmap_mode='r')
    assert raw.shape == clamped.shape == (4825, 2000)
    assert raw.dtype == clamped.dtype == np.float32 and np.isfinite(raw).all()
    np.testing.assert_array_equal(clamped, np.maximum(raw, 0))
    completion = read(plan['cache_completed_file'])
    assert completion['step'] == 27950 and completion['weights'] == 'ema_0.9996'
    assert completion['base_cells'] == 960 and completion['extra_cells'] == 3865
    # Keep the original mixed batch sizes/RNG segments explicit. Never reuse
    # this old diagnostic as a formal, homogeneous calibration refinement.
    segment = (np.arange(len(indices)) >= 960).astype(np.int64)
    stratum = data.meta['batch'][indices] * 2 + segment
    perturbations = np.unique(data.meta['pert'][indices])
    labels = sorted(data.vocab['all_genes'][int(p)] for p in perturbations)
    assert len(labels) == 60
    case_arrays = {'raw': raw, 'clamp0': clamped, 'floor0p05': apply_zero_floor(raw, .05)}
    rows, summaries = [], {}
    for case, array in case_arrays.items():
        folder = Path(plan['metric_dirs'][case])
        frame = pd.read_csv(folder / 'results.csv').set_index('perturbation').sort_index()
        assert frame.index.is_unique and frame.index.tolist() == labels
        original = read(folder / 'summary.json')
        assert abs(float(frame['mse'].mean()) - original['MSE']) < 1e-10
        per_case = []
        for perturbation in perturbations:
            label = data.vocab['all_genes'][int(perturbation)]
            mask = data.meta['pert'][indices] == perturbation
            real_indices = data.val[data.meta['pert'][data.val] == perturbation]
            assert len(real_indices) == int(mask.sum())
            values = np.asarray(array[mask], dtype=np.float64)
            # Cell Eval's MSE generic_evaluation uses all2,000 genes. Its
            # target-gene exclusion is a PDS option, not an MSE operation.
            real_mean = np.asarray(data.x[real_indices], dtype=np.float64).mean(axis=0)
            recomputed = float(np.square(values.mean(axis=0) - real_mean).mean())
            official = float(frame.loc[label, 'mse'])
            if abs(recomputed - official) > plan['mse_scope_check_tolerance']:
                raise ValueError('Independent all-gene MSE disagrees with the original official CSV: ' + label)
            estimated = mean_sampling_variance(values, stratum[mask])
            item = {'case': case, 'perturbation': label, 'official_MSE': official,
                    'independent_MSE_for_scope_check': recomputed,
                    'absolute_scope_check_difference': abs(recomputed - official), **estimated}
            proxy = estimated['pooled_common_within_stratum_variance_proxy']
            item['observed_MSE_minus_assumption_based_proxy'] = official - proxy if proxy is not None else None
            rows.append(item)
            per_case.append(item)
        estimable = [r for r in per_case if r['mc_contribution_repeated_strata_only'] is not None]
        if not estimable:
            raise ValueError('No repeated strata support this variance diagnostic')
        mean = lambda key: float(np.mean([r[key] for r in estimable]))
        observed = mean('official_MSE')
        partial = mean('mc_contribution_repeated_strata_only')
        pooled = mean('pooled_common_within_stratum_variance_proxy')
        summaries[case] = {
            'official_MSE_all60_unchanged': original['MSE'], 'estimable_conditions': len(estimable),
            'mean_official_MSE_same_estimable_conditions': observed,
            'mean_repeated_strata_MC_contribution': partial,
            'mean_common_variance_pooled_proxy': pooled,
            'repeated_component_to_observed_ratio': partial / observed,
            'pooled_proxy_to_observed_ratio': pooled / observed,
            'mean_repeated_cell_fraction': mean('repeated_cell_fraction'),
            'mean_observed_MSE_minus_assumption_based_proxy': mean('observed_MSE_minus_assumption_based_proxy'),
            'maximum_independent_MSE_scope_check_difference': max(r['absolute_scope_check_difference'] for r in per_case),
            'all_conditions_all_strata_repeated': all(r['all_strata_have_repeats'] for r in per_case),
        }
    result = {'time': timestamp(), 'validation_cells': 4825, 'perturbations': 60, 'genes_in_MSE': 2000,
              'summaries': summaries, 'rows': rows,
              'assumptions': plan['assumptions'], 'interpretation_limits': plan['interpretation_limits'],
              'original_mixed_RNG_cache_retained_as_diagnostic_only': True,
              'official_metrics_or_predictions_changed': False, 'new_generated_samples': 0,
              'model_or_training_or_test_protocol_changed': False, 'test_access': False,
              'plan_sha256': file_sha256(plan_path), 'source_sha256': plan['source_sha256'],
              'input_sha256': plan['input_sha256']}
    out = Path(plan['outdir'])
    with exclusive_run(out, 'mc_variance_diagnostic'):
        if (out / 'completed.json').exists():
            proof = read(out / 'completed.json')
            assert proof['plan_sha256'] == result['plan_sha256']
            for path, digest in proof['output_sha256'].items():
                assert file_sha256(path) == digest
            return read(out / 'summary.json')
        atomic(out / 'summary.json', result)
        pd.DataFrame(rows).to_csv(out / 'per_condition.csv', index=False)
        figures(out, rows, summaries)
        paths = [out / name for name in ('summary.json', 'per_condition.csv', 'mc_variance_diagnostic.png', 'mc_variance_diagnostic.pdf')]
        atomic(out / 'completed.json', {'time': timestamp(), 'cpu_only': True, 'test_access': False,
            'plan_sha256': result['plan_sha256'], 'output_sha256': {str(p): file_sha256(p) for p in paths}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    result = diagnose(args.plan)
    print(json.dumps({'summaries': result['summaries'], 'test_access': False}, indent=2))


if __name__ == '__main__':
    main()
