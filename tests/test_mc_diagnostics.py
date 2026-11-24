import numpy as np
import pytest

from ocoot.mc_diagnostics import mean_sampling_variance


def test_known_stratum_sums_and_arbitrary_between_stratum_offsets():
    values = np.array([-1., 1., 10 - np.sqrt(3), 10., 10 + np.sqrt(3)])[:, None]
    labels = np.array([0, 0, 1, 1, 1])
    result = mean_sampling_variance(values, labels)
    assert result['all_strata_have_repeats']
    np.testing.assert_allclose(result['mc_contribution_repeated_strata_only'], 13 / 25)
    np.testing.assert_allclose(result['pooled_common_within_stratum_variance_proxy'], 8 / 15)
    np.testing.assert_allclose(result['naive_iid_variance_over_n_includes_stratum_mean_spread'], 6.4)
    moved = values + (labels * 100)[:, None]
    after = mean_sampling_variance(moved, labels)
    np.testing.assert_allclose(after['mc_contribution_repeated_strata_only'], 13 / 25, atol=1e-12)
    assert after['naive_iid_variance_over_n_includes_stratum_mean_spread'] > 100


def test_heteroskedastic_independent_draws_recover_true_mean_sampling_variance():
    rng = np.random.default_rng(3111)
    samples = rng.normal(size=(4000, 5, 1)) * np.array([1, 1, 3, 3, 3])[None, :, None]
    samples[:, 2:] += 20
    labels = np.array([0, 0, 1, 1, 1])
    estimates = [mean_sampling_variance(sample, labels) for sample in samples]
    # Analytic variance of the weighted mean: (2*1 + 3*9)/5^2.
    expected = 29 / 25
    mean_estimate = np.mean([r['mc_contribution_repeated_strata_only'] for r in estimates])
    empirical = np.var(samples.mean(axis=1), ddof=1)
    assert abs(mean_estimate - expected) < .07
    assert abs(empirical - expected) < .10
    assert abs(mean_estimate - empirical) < .12
    assert np.mean([r['naive_iid_variance_over_n_includes_stratum_mean_spread'] for r in estimates]) > 10


def test_row_order_label_names_and_global_gene_shifts_are_irrelevant():
    rng = np.random.default_rng(50)
    values = rng.normal(size=(12, 4))
    labels = np.repeat([0, 1, 2], 4)
    order = rng.permutation(len(labels))
    original = mean_sampling_variance(values, labels)
    changed = mean_sampling_variance(values[order] + np.arange(4)[None, :] * 100,
                                     np.array(['A', 'B', 'C'])[labels[order]])
    for key in ('mc_contribution_repeated_strata_only', 'pooled_common_within_stratum_variance_proxy',
                'naive_iid_variance_over_n_includes_stratum_mean_spread'):
        np.testing.assert_allclose(original[key], changed[key], rtol=0, atol=1e-12)


def test_singletons_are_flagged_and_missing_variances_are_not_invented():
    result = mean_sampling_variance(np.array([[100.], [-1.], [1.]]), ['A', 'B', 'B'])
    assert not result['all_strata_have_repeats']
    assert result['singleton_strata'] == 1 and result['repeated_cell_fraction'] == 2 / 3
    np.testing.assert_allclose(result['mc_contribution_repeated_strata_only'], 4 / 9)
    alone = mean_sampling_variance(np.array([[0.], [10.], [30.]]), ['A', 'B', 'C'])
    assert alone['mc_contribution_repeated_strata_only'] is None
    assert alone['pooled_common_within_stratum_variance_proxy'] is None
    assert alone['within_stratum_degrees_of_freedom'] == 0
    with pytest.raises(ValueError):
        mean_sampling_variance(np.empty((0, 2)), [])
