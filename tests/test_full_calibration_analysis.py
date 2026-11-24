import runpy

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def analysis():
    return runpy.run_path('scripts/analyze_full_calibration.py')


def test_paired_constant_shift_cancels_between_perturbation_variation(analysis):
    values = np.arange(60, dtype=float)[:, None] * np.arange(1, 10)[None, :] / 1000
    changes = np.arange(1, 10) / 100
    draws = np.random.default_rng(9232).integers(0, 60, (2000, 60))
    _, pairs = analysis['bootstrap_candidates']({'reference': values, 'candidate': values + changes}, draws)
    pair = pairs[0]
    for i, row in enumerate(pair['rows']):
        np.testing.assert_allclose([row['difference'], *row['ci95']], changes[i], atol=1e-12)
    expected = changes[0] + changes[4:7].mean()
    np.testing.assert_allclose([pair['primary_difference']['value'], *pair['primary_difference']['ci95']], expected, atol=1e-12)


def test_undefined_direction_preserves_each_case_official_nanmean(analysis):
    values = np.arange(54, dtype=float).reshape(6, 9) / 100
    other = values + .02
    values[0, 3] = np.nan
    other[1, 3] = np.nan
    draws = np.array([[0, 1, 2, 3, 4, 5], [0, 0, 2, 2, 4, 5], [1, 1, 3, 3, 4, 5]])
    cases, pairs = analysis['bootstrap_candidates']({'reference': values, 'candidate': other}, draws)
    expected = []
    for indices in draws:
        before = [values[i, 3] for i in indices if np.isfinite(values[i, 3])]
        after = [other[i, 3] for i in indices if np.isfinite(other[i, 3])]
        expected.append(sum(after) / len(after) - sum(before) / len(before))
    np.testing.assert_allclose(pairs[0]['rows'][3]['ci95'], np.quantile(expected, [.025, .975]))
    assert cases['reference']['metrics']['DirAgr']['finite_perturbations'] == 5
    assert cases['candidate']['metrics']['DirAgr']['finite_perturbations'] == 5


def test_metric_library_and_summary_are_checked_before_comparison(analysis, tmp_path):
    values = np.arange(18, dtype=float).reshape(2, 9) / 100
    frame = pd.DataFrame(values, columns=analysis['COLUMNS'], index=['B', 'A'])
    frame.index.name = 'perturbation'
    frame.to_csv(tmp_path / 'results.csv')
    means = dict(zip(analysis['ORDER'], values.mean(0)))
    import json
    (tmp_path / 'summary.json').write_text(json.dumps(means))
    result = analysis['load_metrics'](tmp_path, ['A', 'B'], means)
    np.testing.assert_array_equal(result, values[::-1])
    with pytest.raises(ValueError, match='exact complete'):
        analysis['load_metrics'](tmp_path, ['A', 'C'], means)
    means['PDCorr'] += 1
    with pytest.raises(AssertionError):
        analysis['load_metrics'](tmp_path, ['A', 'B'], means)
