"""Registered near-zero candidates and the unchanged validation quality policy.

The paper does not specify a zero floor. Positive floors are explicit optional
reproduction choices, selected using full validation before any test generation.
"""
import json
import math
from pathlib import Path

import numpy as np

from .runtime import file_sha256


SUPPORTED_FLOORS = (0., .05)
METRIC_NAMES = ('PDCorr', 'DEOver', 'DEPrec', 'DirAgr', 'PDS_L1', 'PDS_L2', 'PDS_cos', 'MSE', 'MAE')
DEFAULT_THRESHOLDS = {
    'min_val_pdcorr': .35, 'min_val_pds': .75, 'min_val_pds_each': .70,
    'max_val_mse': .02, 'max_val_mae': .09, 'min_val_deover': .18,
    'min_val_deprec': .18, 'min_val_diragr': .70,
}


def zero_floor(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError('A registered numerical zero floor is required')
    value = float(value)
    if not math.isfinite(value) or value not in SUPPORTED_FLOORS:
        raise ValueError('Only the registered zero floors0 and0.05 are supported')
    return value


def apply_zero_floor(raw, floor):
    floor = zero_floor(floor)
    raw = np.asarray(raw)
    if raw.dtype not in (np.float32, np.float64) or not np.isfinite(raw).all():
        raise ValueError('Finite floating-point predictions required')
    # Preserve the legacy default operation exactly, including its dtype.
    return np.maximum(raw, 0) if floor == 0 else np.where(raw < floor, 0, raw)


def primary_score(metrics):
    return metrics['PDCorr'] + sum(metrics[k] for k in ('PDS_L1', 'PDS_L2', 'PDS_cos')) / 3


def quality_checks(metrics, thresholds=None):
    thresholds = DEFAULT_THRESHOLDS if thresholds is None else thresholds
    if set(thresholds) != set(DEFAULT_THRESHOLDS) or any(
            isinstance(v, bool) or not math.isfinite(v) or v < 0 for v in thresholds.values()):
        raise ValueError('Complete finite validation thresholds required')
    if set(metrics) != set(METRIC_NAMES) or any(not math.isfinite(float(v)) for v in metrics.values()):
        raise ValueError('All nine finite validation metrics required')
    return {key: margin >= 0 for key, (margin, _) in _quality_margins(metrics, thresholds).items()}


def _quality_margins(metrics, thresholds):
    pds_keys = ('PDS_L1', 'PDS_L2', 'PDS_cos')
    values = {**metrics, 'mean_PDS': sum(metrics[k] for k in pds_keys) / len(pds_keys)}
    minimum = {'PDCorr': thresholds['min_val_pdcorr'], 'mean_PDS': thresholds['min_val_pds'],
               'DEOver': thresholds['min_val_deover'], 'DEPrec': thresholds['min_val_deprec'],
               'DirAgr': thresholds['min_val_diragr'], **{k: thresholds['min_val_pds_each'] for k in pds_keys}}
    maximum = {'MSE': thresholds['max_val_mse'], 'MAE': thresholds['max_val_mae']}
    return {**{key: (values[key] - bound, bound) for key, bound in minimum.items()},
            **{key: (bound - values[key], bound) for key, bound in maximum.items()}}


def choose_feasible(candidates, thresholds=None):
    feasible = []
    for candidate in candidates:
        w = candidate.get('guidance')
        if (candidate.get('cap', 'missing') is not None or w is None
                or not math.isfinite(w) or w <= 0 or not candidate.get('clamp')):
            raise ValueError('Postprocessing selection requires full positive-CFG candidates')
        floor = zero_floor(candidate['zero_floor'])
        if all(quality_checks(candidate['metrics'], thresholds).values()):
            feasible.append(candidate)
    # Prefer the simpler original transformation when primary scores tie.
    return max(feasible, key=lambda r: (primary_score(r['metrics']), -r['zero_floor'])) if feasible else None


def allow_validation_shortfall(plan):
    value = plan.get('allow_validation_shortfall', False)
    if not isinstance(value, bool):
        raise ValueError('Validation shortfall policy must be boolean')
    return value


def choose_validation_candidate(candidates, thresholds, report_shortfall=False):
    feasible = choose_feasible(candidates, thresholds)
    if feasible is not None or not report_shortfall or not candidates:
        return feasible

    def rank(candidate):
        metrics = candidate['metrics']
        shortfall = sum(max(0., -margin) / max(bound, 1e-12)
                        for margin, bound in _quality_margins(metrics, thresholds).values())
        return shortfall, -primary_score(metrics), candidate['zero_floor']

    return min(candidates, key=rank)


def verify_selection_postprocessing(selected, require_review=False):
    """Verify the optional layer added to the original calibration provenance."""
    floor = zero_floor(selected.get('zero_floor', 0.))
    proof = selected.get('postprocess_selection')
    if proof is None:
        if floor > 0 or require_review:
            raise ValueError('Registered full-validation postprocessing selection required')
        return floor
    for name in ('plan', 'review', 'candidate_metrics'):
        if file_sha256(proof[name + '_path']) != proof[name + '_sha256']:
            raise ValueError('Postprocessing selection evidence changed: ' + name)
    if (file_sha256('scripts/select_validation_postprocess.py') != proof['selector_sha256']
            or file_sha256(__file__) != proof['module_sha256']):
        raise ValueError('Postprocessing selection implementation changed')
    plan = json.loads(Path(proof['plan_path']).read_text())
    if plan['floors'] != list(SUPPORTED_FLOORS) or plan['quality_thresholds'] != DEFAULT_THRESHOLDS:
        raise ValueError('Postprocessing plan changed candidates or original quality thresholds')
    review = json.loads(Path(proof['review_path']).read_text())
    choice = review['selected']
    if choice is None or (choice['tag'], choice['zero_floor']) != (selected['tag'], floor):
        raise ValueError('Selection does not match the complete validation review')
    if review['thresholds'] != plan['quality_thresholds']:
        raise ValueError('Postprocessing review thresholds differ')
    report_shortfall = allow_validation_shortfall(plan)
    if review.get('allow_validation_shortfall', False) != report_shortfall:
        raise ValueError('Validation shortfall policy differs')
    best = choose_validation_candidate(review['candidates'], review['thresholds'], report_shortfall)
    if best is None or (best['tag'], best['zero_floor']) != (choice['tag'], floor):
        raise ValueError('Review did not select the best registered candidate')
    for key in ('guidance', 'cfg_min', 'cfg_max', 'cap', 'clamp'):
        if choice[key] != selected[key]:
            raise ValueError('Reviewed sampling settings changed: ' + key)
    checks = quality_checks(selected['metrics'], review['thresholds'])
    if choice['checks'] != checks or (not report_shortfall and not all(checks.values())):
        raise ValueError('Selected postprocessing is inconsistent with the full-validation checks')
    saved = json.loads(Path(proof['candidate_metrics_path']).read_text())
    if set(saved) != set(METRIC_NAMES):
        raise ValueError('Selected metrics are incomplete')
    np.testing.assert_allclose([saved[k] for k in METRIC_NAMES],
                               [selected['metrics'][k] for k in METRIC_NAMES], rtol=0, atol=1e-10)
    np.testing.assert_allclose([choice['metrics'][k] for k in METRIC_NAMES],
                               [selected['metrics'][k] for k in METRIC_NAMES], rtol=0, atol=1e-10)
    return floor
