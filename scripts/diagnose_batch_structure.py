"""CPU-only, perturbation-adjusted batch associations in saved validation cells.

These decompositions are descriptive diagnostics, not benchmark metrics or a
postprocessing model. No corrected expression prediction is produced.
"""
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def group_mean(values, inverse, groups):
    result = np.zeros((groups, values.shape[1]), dtype=np.float64)
    np.add.at(result, inverse, values)
    counts = np.bincount(inverse, minlength=groups)
    if np.any(counts == 0):
        raise ValueError('Every fitted group must have observations')
    return result / counts[:, None]


@dataclass
class BatchDesign:
    perturbations: np.ndarray
    batches: np.ndarray
    p_inverse: np.ndarray
    b_inverse: np.ndarray
    cross_counts: np.ndarray
    p_counts: np.ndarray
    b_counts: np.ndarray
    gram_pinv: np.ndarray
    rank: int


def batch_design(pert, batch):
    pert, batch = np.asarray(pert), np.asarray(batch)
    if pert.ndim != 1 or batch.shape != pert.shape or not len(pert):
        raise ValueError('Aligned nonempty label vectors required')
    p, pi = np.unique(pert, return_inverse=True)
    b, bi = np.unique(batch, return_inverse=True)
    cross = np.zeros((len(p), len(b)), dtype=np.float64)
    np.add.at(cross, (pi, bi), 1)
    np_counts, nb_counts = cross.sum(1), cross.sum(0)
    gram = np.diag(nb_counts) - cross.T @ (cross / np_counts[:, None])
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    tolerance = max(float(eigenvalues[-1]), 1.) * max(gram.shape) * np.finfo(float).eps * 10
    keep = eigenvalues > tolerance
    rank = int(keep.sum())
    if rank != len(b) - 1:
        raise ValueError('Batch and perturbation effects are not jointly identifiable')
    pinv = (eigenvectors[:, keep] / eigenvalues[keep]) @ eigenvectors[:, keep].T
    return BatchDesign(p, b, pi, bi, cross, np_counts, nb_counts, pinv, rank)


def fit_batch(values, design):
    """FWL estimate of batch effects after removing perturbation intercepts."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 2 or len(y) != len(design.p_inverse) or not np.isfinite(y).all():
        raise ValueError('Aligned finite cell by gene array required')
    p_mean = group_mean(y, design.p_inverse, len(design.perturbations))
    by_batch = np.zeros((len(design.batches), y.shape[1]), dtype=np.float64)
    np.add.at(by_batch, design.b_inverse, y)
    rhs = by_batch - design.cross_counts.T @ p_mean
    effects = design.gram_pinv @ rhs
    effects -= (effects * design.b_counts[:, None]).sum(0) / len(y)
    p_batch_mean = (design.cross_counts @ effects) / design.p_counts[:, None]
    fitted = effects[design.b_inverse] - p_batch_mean[design.p_inverse]
    centered = y - p_mean[design.p_inverse]
    null_ss = float(np.square(centered).sum())
    residual_ss = float(np.square(centered - fitted).sum())
    return effects, {'within_perturbation_cell_variance': null_ss / y.size,
                     'residualized_batch_component_rms': float(np.sqrt(np.square(fitted).mean())),
                     'in_sample_partial_explained_fraction': 1 - residual_ss / null_ss if null_ss > 1e-20 else None}


def blocked_cv(values, pert, batch, folds=5, seed=9831):
    y = np.asarray(values, dtype=np.float64)
    pert, batch = np.asarray(pert), np.asarray(batch)
    labels = np.unique(pert)
    if not 2 <= folds <= len(labels):
        raise ValueError('At least two nonempty perturbation folds required')
    partitions = np.array_split(np.random.default_rng(seed).permutation(labels), folds)
    rows, total_null, total_error = [], 0., 0.
    for heldout in partitions:
        mask = np.isin(pert, heldout)
        training = batch_design(pert[~mask], batch[~mask])
        effects, _ = fit_batch(y[~mask], training)
        locations = np.searchsorted(training.batches, batch[mask])
        if (np.any(locations >= len(training.batches)) or
                not np.array_equal(training.batches[locations], batch[mask])):
            raise ValueError('A held-out batch has no training support')
        _, pi = np.unique(pert[mask], return_inverse=True)
        groups = len(heldout)
        heldout_y = y[mask]
        centered = heldout_y - group_mean(heldout_y, pi, groups)[pi]
        fitted = effects[locations]
        fitted = fitted - group_mean(fitted, pi, groups)[pi]
        null_ss = float(np.square(centered).sum())
        error_ss = float(np.square(centered - fitted).sum())
        total_null += null_ss
        total_error += error_ss
        rows.append({'heldout_perturbations': heldout.tolist(), 'cells': int(mask.sum()),
                     'training_batch_rank': training.rank,
                     'explained_fraction': 1 - error_ss / null_ss if null_ss > 1e-20 else None})
    return {'folds': rows, 'cells': len(y), 'seed': seed,
            'out_of_fold_explained_fraction': 1 - total_error / total_null if total_null > 1e-20 else None,
            'note': 'Within-perturbation centered structural diagnostic. Held-out perturbation means are removed for this variance decomposition; this is not an expression prediction benchmark.'}


def compare_vectors(reference, candidate, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / weights.sum()
    x = reference - np.sum(reference * weights[:, None], axis=0)
    y = candidate - np.sum(candidate * weights[:, None], axis=0)
    xx = float(np.sum(np.square(x) * weights[:, None]) / x.shape[1])
    yy = float(np.sum(np.square(y) * weights[:, None]) / y.shape[1])
    xy = float(np.sum(x * y * weights[:, None]) / x.shape[1])
    return {'reference_rms': float(np.sqrt(xx)), 'candidate_rms': float(np.sqrt(yy)),
            'slope_candidate_on_reference': xy / xx if xx > 0 else None,
            'cosine': xy / np.sqrt(xx * yy) if xx * yy > 0 else None,
            'difference_rms': float(np.sqrt(max(0., xx + yy - 2 * xy)))}


def plot_results(out, effects, reliable, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    true = effects['true'][reliable].ravel()
    pred = effects['clamp0_prediction'][reliable].ravel()
    limit = float(np.quantile(np.abs(np.concatenate([true, pred])), .995))
    with plt.rc_context({'font.size': 10, 'axes.titlesize': 11}):
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), layout='constrained')
        ax = axes[0]
        ax.hexbin(true, pred, gridsize=55, bins='log', mincnt=1,
                  extent=(-limit, limit, -limit, limit), cmap='Blues')
        ax.plot([-limit, limit], [-limit, limit], '--', color='.5')
        ax.set(xlabel='Real batch effect (log1p)', ylabel='Generated batch effect (log1p)',
               title='Perturbation-adjusted effects', xlim=(-limit, limit), ylim=(-limit, limit))
        ax = axes[1]
        for key, color in [('true', '#28648a'), ('clamp0_prediction', '#a85d20'), ('clamp0_error', '#745691')]:
            values = np.sqrt(np.mean(np.square(effects[key][reliable]), axis=1))
            ax.plot(np.arange(len(values)), values, '.', label=key, color=color)
        ax.set(xlabel='Reliable batch (sorted ID)', ylabel='Effect RMS (log1p)', title='Batch effect magnitude')
        ax.legend(fontsize=8)
        ax = axes[2]
        keys = ['true', 'raw_prediction', 'clamp0_prediction', 'clamp0_error']
        values = [summary['crossvalidation'][k]['out_of_fold_explained_fraction'] for k in keys]
        ax.bar(range(4), values, color=['#28648a', '#709bb5', '#a85d20', '#745691'])
        ax.axhline(0, color='.5', linewidth=1)
        ax.set_xticks(range(4), ['Real', 'Raw pred.', 'Clamp0 pred.', 'Error'], rotation=20)
        ax.set(ylabel='Explained fraction', title='Held-out perturbations; five folds')
        fig.suptitle('E25 batch-associated structure diagnostic; no benchmark scores or fitted corrections')
        fig.savefig(out / 'batch_structure.png', dpi=180)
        fig.savefig(out / 'batch_structure.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    registration = json.loads((out / 'execution_config.json').read_text())
    assert registration['plan_sha256'] == sha(plan_path)
    assert registration['script_sha256'] == sha(__file__)
    assert registration['math_verification_passed']
    assert registration['math_verification_sha256'] == sha(out / 'math_verification.json')
    assert registration['math_test_source_sha256'] == sha('tests/test_batch_structure.py')
    for name, digest in plan['source_sha256'].items():
        assert sha(name) == digest, name
    expression = Path(plan['expression_file']['path'])
    assert expression.stat().st_size == plan['expression_file']['bytes']
    assert expression.stat().st_mtime_ns == plan['expression_file']['mtime_ns']
    with (out / 'analysis.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out / 'completed.json').exists():
            assert json.loads((out / 'completed.json').read_text())['execution_sha256'] == sha(out / 'execution_config.json')
            return
        meta = dict(np.load(Path(plan['data_dir']) / 'metadata.npz'))
        indices = np.load(plan['indices_file'])
        np.testing.assert_array_equal(np.sort(indices), np.flatnonzero(meta['split'] == 1))
        assert len(indices) == plan['expected_cells']
        cell = np.unique(meta['cell'][indices])
        assert len(cell) == 1
        vocab = json.loads((Path(plan['data_dir']) / 'vocab.json').read_text())
        assert vocab['cell_lines'][int(cell[0])] == plan['cell_line']
        control_idx = np.flatnonzero((meta['split'] == 3) & (meta['cell'] == cell[0]))
        x = np.load(expression, mmap_mode='r')
        values = {'true': np.asarray(x[indices], dtype=np.float64),
                  'raw_prediction': np.load(plan['raw_prediction_file']),
                  'clamp0_prediction': np.load(plan['clamped_prediction_file'])}
        assert all(y.shape == (len(indices), 2000) and np.isfinite(y).all() for y in values.values())
        np.testing.assert_array_equal(np.maximum(values['raw_prediction'], 0), values['clamp0_prediction'])
        values['clamp0_error'] = values['true'] - values['clamp0_prediction']
        pert, batch = meta['pert'][indices], meta['batch'][indices]
        design = batch_design(pert, batch)
        assert len(design.perturbations) == plan['expected_perturbations']
        assert len(design.batches) == plan['expected_batches']
        effects, descriptions = {}, {}
        for key, y in values.items():
            effects[key], descriptions[key] = fit_batch(y, design)
        np.testing.assert_allclose(effects['clamp0_error'], effects['true'] - effects['clamp0_prediction'], atol=1e-11, rtol=0)
        control_batches = meta['batch'][control_idx]
        control_counts = np.array([np.sum(control_batches == b) for b in design.batches])
        control_mean = np.stack([np.mean(np.asarray(x[control_idx[control_batches == b]], dtype=np.float64), axis=0)
                                 for b in design.batches])
        reliable = ((design.b_counts >= plan['reliable_min_validation_cells']) &
                    (control_counts >= plan['reliable_min_control_cells']))
        reliable_cells = np.isin(batch, design.batches[reliable])
        weights = design.b_counts[reliable]
        comparisons = {'clamp0_vs_true': compare_vectors(effects['true'][reliable], effects['clamp0_prediction'][reliable], weights),
                       'raw_vs_true': compare_vectors(effects['true'][reliable], effects['raw_prediction'][reliable], weights),
                       'true_vs_controls': compare_vectors(control_mean[reliable], effects['true'][reliable], weights),
                       'clamp0_error_vs_controls': compare_vectors(control_mean[reliable], effects['clamp0_error'][reliable], weights)}
        cv = {key: blocked_cv(y[reliable_cells], pert[reliable_cells], batch[reliable_cells],
                             plan['crossvalidation_folds'], plan['crossvalidation_seed']) for key, y in values.items()}
        result = {'time': datetime.now().astimezone().isoformat(timespec='seconds'),
                  'cells': len(indices), 'controls': len(control_idx), 'perturbations': len(design.perturbations),
                  'batches': len(design.batches), 'batch_rank': design.rank,
                  'reliable_batches': int(reliable.sum()), 'crossvalidation_cells': int(reliable_cells.sum()),
                  'excluded_comparison_batch_ids': design.batches[~reliable].tolist(),
                  'descriptive_fits': descriptions, 'effect_vector_comparisons': comparisons, 'crossvalidation': cv,
                  'plan_sha256': sha(plan_path), 'script_sha256': sha(__file__), 'cpu_only': True,
                  'test_access': False, 'official_benchmark_metrics': False, 'corrected_predictions_produced': False,
                  'note': 'Perturbation-adjusted associations, not causal technical effects. Finite real and generated sample noise affects effect estimates. Crossvalidation is a structural variance decomposition, not an expression-model benchmark.'}
        atomic_json(out / 'summary.json', result)
        with (out / 'batch_effects.npz').open('wb') as stream:
            np.savez_compressed(stream, batch_ids=design.batches, validation_counts=design.b_counts,
                                control_counts=control_counts, reliable=reliable, control_means=control_mean, **effects)
        with (out / 'per_batch.csv').open('w', newline='') as stream:
            rows = [{'batch_id': int(b), 'validation_cells': int(design.b_counts[i]),
                     'control_cells': int(control_counts[i]), 'reliable_comparison': bool(reliable[i]),
                     **{key + '_effect_rms': float(np.sqrt(np.square(value[i]).mean())) for key, value in effects.items()}}
                    for i, b in enumerate(design.batches)]
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        plot_results(out, effects, reliable, result)
        atomic_json(out / 'completed.json', {'time': result['time'], 'stage': 'completed_review_required',
                    'execution_sha256': sha(out / 'execution_config.json'), 'summary_sha256': sha(out / 'summary.json'),
                    'cpu_only': True, 'test_access': False, 'training_changed': False})
        print(json.dumps({'descriptive_fits': descriptions, 'effect_vector_comparisons': comparisons,
                          'crossvalidated_fractions': {key: value['out_of_fold_explained_fraction'] for key, value in cv.items()}}))


if __name__ == '__main__':
    main()
