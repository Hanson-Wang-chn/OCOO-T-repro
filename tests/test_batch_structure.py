"""Independent linear-model checks for the explanatory batch decomposition."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location('batch_structure_diagnostic',
    Path(__file__).resolve().parents[1] / 'scripts/diagnose_batch_structure.py')
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)


def example(perturbations=9, batches=4, genes=7):
    rng = np.random.default_rng(42)
    labels = [(p, b) for p in range(perturbations) for b in range(batches)
              for _ in range(int(rng.integers(1, 6)))]
    pert, batch = np.asarray(labels).T
    perturbation_means = rng.normal(size=(perturbations, genes))
    batch_effects = rng.normal(size=(batches, genes))
    values = perturbation_means[pert] + batch_effects[batch]
    return pert, batch, values, rng


def test_fwl_matches_independent_cell_level_ordinary_least_squares():
    pert, batch, values, rng = example()
    values = values + rng.normal(scale=.05, size=values.shape)
    design = diagnostic.batch_design(pert, batch)
    effects, description = diagnostic.fit_batch(values, design)
    # Independent reference fits the full cell-level design, rather than FWL
    # normal equations. The omitted batch sets an identifiable reference level.
    p_matrix = np.eye(len(design.perturbations))[design.p_inverse]
    b_matrix = np.eye(len(design.batches))[design.b_inverse]
    cell_design = np.column_stack([p_matrix, b_matrix[:, 1:]])
    coefficients = np.linalg.lstsq(cell_design, values, rcond=None)[0]
    expected = np.vstack([np.zeros(values.shape[1]), coefficients[len(design.perturbations):]])
    expected -= np.average(expected, axis=0, weights=design.b_counts)
    np.testing.assert_allclose(effects, expected, atol=2e-12, rtol=0)
    np.testing.assert_allclose(np.average(effects, axis=0, weights=design.b_counts), 0, atol=2e-14)
    residual = values - cell_design @ coefficients
    centered = values - p_matrix @ np.linalg.lstsq(p_matrix, values, rcond=None)[0]
    reference_fraction = 1 - np.square(residual).sum() / np.square(centered).sum()
    assert description['in_sample_partial_explained_fraction'] == pytest.approx(reference_fraction, abs=2e-13)


def test_batch_estimates_ignore_perturbation_baselines_and_row_order():
    pert, batch, values, rng = example()
    effects, _ = diagnostic.fit_batch(values, diagnostic.batch_design(pert, batch))
    shifts = rng.normal(scale=50, size=(len(np.unique(pert)), values.shape[1]))
    order = rng.permutation(len(pert))
    shifted, _ = diagnostic.fit_batch((values + shifts[pert])[order],
        diagnostic.batch_design(pert[order], batch[order]))
    np.testing.assert_allclose(effects, shifted, atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match='not jointly identifiable'):
        diagnostic.batch_design([0, 0, 1, 1], [0, 0, 1, 1])


def test_perturbation_blocked_validation_recovers_shared_known_batch_signal():
    pert, batch, values, _ = example()
    result = diagnostic.blocked_cv(values, pert, batch, folds=3, seed=9831)
    assert result['out_of_fold_explained_fraction'] == pytest.approx(1, abs=2e-13)
    seen = []
    for fold in result['folds']:
        assert fold['explained_fraction'] == pytest.approx(1, abs=2e-13)
        seen.extend(fold['heldout_perturbations'])
    assert sorted(seen) == np.unique(pert).tolist()


def test_vector_comparison_removes_gene_offsets_and_respects_batch_weights():
    reference = np.array([[1., 2., 3.], [5., -1., 0.], [2., 3., 7.]])
    candidate = 2 * reference + np.array([100., -20., 70.])
    result = diagnostic.compare_vectors(reference, candidate, [1, 2, 4])
    assert result['cosine'] == pytest.approx(1, abs=2e-14)
    assert result['slope_candidate_on_reference'] == pytest.approx(2, abs=2e-14)
    assert result['candidate_rms'] == pytest.approx(2 * result['reference_rms'], abs=2e-14)
