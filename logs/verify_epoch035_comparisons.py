"""Independent CPU audit of the completed E30-to-E35 paired arithmetic."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


columns = {
    'PDCorr': 'pearson_delta', 'DEOver': 'overlap_at_N', 'DEPrec': 'precision_at_N',
    'DirAgr': 'de_direction_match', 'PDS_L1': 'discrimination_score_l1',
    'PDS_L2': 'discrimination_score_l2', 'PDS_cos': 'discrimination_score_cosine',
    'MSE': 'mse', 'MAE': 'mae',
}
root = Path('runs/main_v1/interim_epoch035_fixed_cfg')
specifications = {
    'ordinary': ('runs/main_v1/observation/epoch030_to035/comparison.json',
                 'runs/main_v1/validation_epoch_030/metrics_de_diagnostic/results.csv',
                 'runs/main_v1/validation_epoch_035/metrics_de_diagnostic/results.csv'),
    'fixed_cfg': (str(root / 'paired/comparison.json'),
                  'runs/main_v1/interim_epoch030_fixed_cfg/candidate/metrics/results.csv',
                  str(root / 'candidate/metrics/results.csv')),
}
sources, audits = {}, {}
for name, (analysis, reference, candidate) in specifications.items():
    result = json.loads(Path(analysis).read_text())
    for path in (analysis, reference, candidate):
        sources[path] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    frames = [pd.read_csv(path).set_index('perturbation').sort_index()
              for path in (reference, candidate)]
    assert frames[0].index.equals(frames[1].index) and frames[0].index.is_unique
    assert len(frames[0]) == result['perturbations'] == 60
    order = [row['metric'] for row in result['rows']]
    values = [frame[[columns[key] for key in order]].to_numpy(float) for frame in frames]
    assert all(not np.isinf(value).any() for value in values)
    # Cell Eval leaves direction agreement undefined for seven populations
    # without a valid DE intersection. Preserve its nanmean convention;
    # never replace these observations with invented zeros.
    for value in values:
        assert {order[i] for i in np.flatnonzero(np.isnan(value).any(0))} <= {'DirAgr'}
    np.testing.assert_array_equal(np.isnan(values[0]), np.isnan(values[1]))
    draws = np.random.default_rng(result['seed']).integers(0, 60, (result['bootstrap'], 60))
    # Deliberately use direct 3-D indexing/means instead of the original
    # resampled_means helper, and do not import either comparison script.
    means = [np.nanmean(value, axis=0) for value in values]
    changes = np.nanmean(values[1][draws], axis=1) - np.nanmean(values[0][draws], axis=1)
    differences = []
    for i, row in enumerate(result['rows']):
        computed = [means[0][i], means[1][i], means[1][i] - means[0][i],
                    *np.quantile(changes[:, i], [.025, .975])]
        recorded = [row['reference'], row['candidate'], row['difference'], *row['ci95']]
        differences.extend(np.abs(np.asarray(computed) - recorded))
    pd_index = order.index('PDCorr')
    pds_indices = [order.index(key) for key in ('PDS_L1', 'PDS_L2', 'PDS_cos')]
    score_delta = changes[:, pd_index] + changes[:, pds_indices].mean(1)
    score_means = [value[pd_index] + value[pds_indices].mean() for value in means]
    differences.extend(np.abs(np.asarray([score_means[1] - score_means[0],
                                           *np.quantile(score_delta, [.025, .975])]) -
                              [result['score_difference']['value'], *result['score_difference']['ci95']]))
    maximum = float(max(differences))
    assert maximum < 1e-12, (name, maximum)
    audits[name] = {'perturbations': 60, 'bootstrap': result['bootstrap'],
                    'finite_observations_per_metric': dict(zip(order, np.isfinite(values[0]).sum(0).tolist())),
                    'seed': result['seed'], 'maximum_absolute_difference': maximum}

completed = json.loads((root / 'completed.json').read_text())
assert completed['fixed_comparison_sha256'] == sources[str(root / 'paired/comparison.json')]
assert completed['periodic_comparison_sha256'] == sources[specifications['ordinary'][0]]
assert completed['plan_sha256'] == hashlib.sha256((root / 'plan.json').read_bytes()).hexdigest()
proof = {'passed': True, 'cpu_only': True, 'recomputed_predictions_or_official_metrics': False,
         'audits': audits, 'source_sha256': sources,
         'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
out = root / 'independent_comparison_verification.json'
assert not out.exists()
out.write_text(json.dumps(proof, indent=2))
print(json.dumps(proof))
