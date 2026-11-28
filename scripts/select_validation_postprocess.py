"""Evaluate registered zero floors on complete calibration refinements, on CPU."""
import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS, official_metrics
from ocoot.postprocess import (DEFAULT_THRESHOLDS, SUPPORTED_FLOORS, allow_validation_shortfall,
                              apply_zero_floor, choose_validation_candidate, primary_score, quality_checks,
                              selection_score)
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def verified_metrics(directory, expected=None, conditions=None, labels=None):
    directory = Path(directory)
    metrics = json.loads((directory / 'summary.json').read_text())
    if set(metrics) != set(METRICS.values()):
        raise ValueError('Postprocessing comparison requires all nine metrics')
    frame = pd.read_csv(directory / 'results.csv')
    if not frame['perturbation'].is_unique or (conditions is not None and len(frame) != conditions):
        raise ValueError('Metrics do not contain the complete perturbation library')
    if labels is not None and set(frame['perturbation'].astype(str)) != labels:
        raise ValueError('Metric perturbation identities differ from the complete validation split')
    means = {display: float(frame[key].mean()) for key, display in METRICS.items()}
    np.testing.assert_allclose([means[k] for k in metrics], [metrics[k] for k in metrics], rtol=0, atol=1e-10)
    if expected is not None:
        np.testing.assert_allclose([metrics[k] for k in metrics], [expected[k] for k in metrics], rtol=0, atol=1e-10)
    quality_checks(metrics)
    return metrics


def load_calibration(calibration):
    calibration = Path(calibration).resolve()
    if not all((calibration / name).exists() for name in
               ('calibration_config.json', 'selected.json', 'all_results.json')):
        raise ValueError('A complete formal calibration is required; interim stitched caches are not eligible')
    provenance = json.loads((calibration / 'calibration_config.json').read_text())
    original = json.loads((calibration / 'selected.json').read_text())
    if (provenance['refine_cap'] is not None or original['cap'] is not None
            or provenance['weights'] != 'ema_0.9996'
            or provenance['formal_guidance_policy'] != 'literal_dual_cfg_positive_scale'):
        raise ValueError('Requires the original complete default-EMA positive-CFG calibration')
    for key in ('checkpoint', 'checkpoint_sha256', 'step', 'weights', 'data_dir', 'source_sha256'):
        if original[key] != provenance[key]:
            raise ValueError('Original calibration provenance differs: ' + key)
    for path, digest in provenance['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Calibration implementation changed: ' + path)
    if file_sha256(provenance['checkpoint']) != provenance['checkpoint_sha256']:
        raise ValueError('Calibrated checkpoint changed')
    state = torch.load(provenance['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    cfg = state['config']
    weights = 'ema_0.9996' if '0.9996' in state.get('ema', {}) else state.get('weights')
    if (state['step'] != provenance['step'] or weights != 'ema_0.9996'
            or cfg['training'].get('train_subset') or cfg['data_dir'] != provenance['data_dir']):
        raise ValueError('Requires the full-data calibrated default EMA')
    data = BenchmarkData(cfg['data_dir'])
    indices = data.evaluation_indices('validation', None)
    records = json.loads((calibration / 'all_results.json').read_text())
    refinements = [r for r in records if r['tag'].startswith('refine_')]
    count = len(refinements)
    if (count < 2 or len({r['tag'] for r in refinements}) != count
            or len({(r['guidance'], r['cfg_min'], r['cfg_max']) for r in refinements}) != count):
        raise ValueError('The registered calibration requires at least two distinct full refinements')
    return calibration, provenance, original, cfg, data, indices, refinements


def validate_refinement(calibration, provenance, cfg, data, indices, record):
    w = record['guidance']
    if (record['cap'] is not None or not record['clamp'] or w is None
            or not math.isfinite(w) or w <= 0):
        raise ValueError('Capped and conditional-only cases cannot enter formal postprocessing selection')
    if not 0 <= record['cfg_min'] < record['cfg_max'] <= 1:
        raise ValueError('Invalid refinement CFG interval')
    cache = record['cache_provenance']
    source = Path(cache['directory']).resolve()
    if (source != calibration / record['tag'] or cache['split'] != 'validation'
            or not cache['raw_unclamped'] or set(cache['files']) !=
            {'prediction.npy', 'indices.npy', 'generation_state.json'}):
        raise ValueError('Requires an original, unmixed full refinement cache')
    for name, digest in cache['files'].items():
        if file_sha256(source / name) != digest:
            raise ValueError('Raw calibration cache changed: ' + name)
    np.testing.assert_array_equal(np.load(source / 'indices.npy'), indices)
    expected = {'noise_std': cfg['training']['noise_std'], 'batch_size': provenance['batch_size'],
                'steps': 50, 'guidance': w, 'cfg_min': record['cfg_min'], 'cfg_max': record['cfg_max'],
                'clamp': False, 'seed': 4242}
    if json.loads((source / 'generation_state.json').read_text()) != {
            'completed': len(indices), 'total': len(indices), 'options': expected}:
        raise ValueError('Incomplete or mismatched refinement generation')
    raw = np.load(source / 'prediction.npy', mmap_mode='r')
    if raw.dtype != np.float32 or raw.shape != (len(indices), data.x.shape[1]) or not np.isfinite(raw).all():
        raise ValueError('Invalid complete raw refinement array')
    return source, raw


def select(calibration, destination, plan_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    if plan['floors'] != list(SUPPORTED_FLOORS) or plan['quality_thresholds'] != DEFAULT_THRESHOLDS:
        raise ValueError('Use the registered two floors and unchanged quality checks')
    report_shortfall = allow_validation_shortfall(plan)
    calibration, provenance, original, cfg, data, indices, refinements = load_calibration(calibration)
    mse_weight = provenance.get('mse_weight', 0.0)
    if (not math.isfinite(mse_weight) or mse_weight < 0
            or original.get('mse_weight', 0.0) != mse_weight
            or plan.get('calibration', {}).get('mse_weight', mse_weight) != mse_weight):
        raise ValueError('The registered postprocessing and calibration MSE weights differ')
    if plan.get('calibration', {}).get('refine_top', len(refinements)) != len(refinements):
        raise ValueError('The registered full refinement count differs from calibration')
    destination = Path(destination).resolve()
    sources = [Path(r['cache_provenance']['directory']).resolve() for r in refinements]
    if (destination == calibration or any(source == destination or source in destination.parents
                                         or destination in source.parents for source in sources)):
        raise ValueError('Keep postprocessing results separate from original calibration files')
    protocol = {'plan_sha256': file_sha256(plan_path),
                'calibration_config_sha256': file_sha256(calibration / 'calibration_config.json'),
                'base_results_sha256': file_sha256(calibration / 'all_results.json'),
                'original_selected_sha256': file_sha256(calibration / 'selected.json'),
                'checkpoint_sha256': provenance['checkpoint_sha256'],
                'data_files_sha256': {name: file_sha256(data.path / name)
                                     for name in ('expression.npy', 'metadata.npz', 'vocab.json')},
                'source_sha256': {name: file_sha256(name) for name in
                                 ('scripts/select_validation_postprocess.py', 'src/ocoot/postprocess.py')}}
    with exclusive_run(destination, 'postprocess_selection'):
        execution = destination / 'execution_config.json'
        if execution.exists() and json.loads(execution.read_text()) != protocol:
            raise ValueError('Postprocessing output belongs to another source or protocol')
        if not execution.exists():
            if any(p.name != 'postprocess_selection.lock' for p in destination.iterdir()):
                raise FileExistsError('Retain existing output; use its registered workflow')
            atomic_json(execution, protocol)
        completed_path = destination / 'completed.json'
        if completed_path.exists():
            completed = json.loads(completed_path.read_text())
            if completed['protocol'] != protocol:
                raise ValueError('Completed postprocessing selection has another protocol')
            for path, digest in completed['output_sha256'].items():
                if file_sha256(path) != digest:
                    raise ValueError('Completed postprocessing result changed: ' + path)
            for record in refinements:
                validate_refinement(calibration, provenance, cfg, data, indices, record)
            return (json.loads((destination / 'selected.json').read_text()),
                    json.loads((destination / 'review.json').read_text()))
        candidates = []
        conditions = len(np.unique(data.meta['pert'][indices]))
        labels = {data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][indices])}
        for record in refinements:
            source, raw = validate_refinement(calibration, provenance, cfg, data, indices, record)
            for floor in SUPPORTED_FLOORS:
                if floor == 0:
                    metric_dir = source / 'metrics_clamped'
                    metrics = verified_metrics(metric_dir, record['metrics'], conditions, labels)
                else:
                    case = destination / (record['tag'] + '_floor_0p05')
                    case.mkdir(exist_ok=True)
                    metric_dir = case / 'metrics'
                    if not (metric_dir / 'summary.json').exists():
                        official_metrics(data, indices, apply_zero_floor(raw, floor), metric_dir, de=True)
                    metrics = verified_metrics(metric_dir, conditions=conditions, labels=labels)
                candidate = {**copy.deepcopy(record), 'zero_floor': floor, 'metrics': metrics,
                             'score': primary_score(metrics), 'mse_weight': mse_weight,
                             'selection_score': selection_score(metrics, mse_weight),
                             'checks': quality_checks(metrics, plan['quality_thresholds']),
                             'candidate_metrics_path': str((metric_dir / 'summary.json').resolve())}
                candidates.append(candidate)
        chosen = choose_validation_candidate(candidates, plan['quality_thresholds'], report_shortfall, mse_weight)
        review = {'time': timestamp(), 'scope': 'complete validation only',
                  'cells_each': len(indices), 'perturbations': conditions,
                  'thresholds': plan['quality_thresholds'], 'candidates': candidates,
                  'selected': chosen, 'protocol': protocol, 'mse_weight': mse_weight,
                  'score': chosen['score'] if chosen is not None else None,
                  'selection_score': chosen['selection_score'] if chosen is not None else None,
                  'test_access': False, 'new_generated_samples': 0,
                  'allow_validation_shortfall': report_shortfall,
                  'decision_policy': ('Highest PDCorr plus mean PDS minus weighted MSE among candidates passing all original checks. With explicit'
                                      ' report mode and no passing case, choose the lowest total relative shortfall.')}
        review_path = destination / 'review.json'
        atomic_json(review_path, review)
        selected_path = destination / 'selected.json'
        if chosen is None:
            if selected_path.exists():
                raise ValueError('A previous selection exists despite no passing candidate; preserve and review')
            return None, review
        selected = copy.deepcopy(original)
        selected.update({k: copy.deepcopy(v) for k, v in chosen.items() if k not in ('checks', 'candidate_metrics_path')})
        selected['postprocess_selection'] = {
            'plan_path': str(plan_path), 'plan_sha256': protocol['plan_sha256'],
            'review_path': str(review_path), 'review_sha256': file_sha256(review_path),
            'candidate_metrics_path': chosen['candidate_metrics_path'],
            'candidate_metrics_sha256': file_sha256(chosen['candidate_metrics_path']),
            'selector_sha256': protocol['source_sha256']['scripts/select_validation_postprocess.py'],
            'module_sha256': protocol['source_sha256']['src/ocoot/postprocess.py'],
            'base_results_sha256': protocol['base_results_sha256'],
            'calibration_config_sha256': protocol['calibration_config_sha256']}
        selected['note'] = ('Selected on full validation from registered floors 0 and 0.05. '
                            f'All validation checks passed: {all(chosen["checks"].values())}. '
                            'The paper does not specify a zero floor.')
        atomic_json(selected_path, selected)
        outputs = [selected_path, review_path]
        for candidate in candidates:
            summary = Path(candidate['candidate_metrics_path'])
            outputs.extend([summary, summary.parent / 'results.csv'])
        atomic_json(completed_path, {'time': timestamp(), 'cells_each': len(indices),
                    'selected_sha256': file_sha256(selected_path), 'review_sha256': file_sha256(review_path),
                    'output_sha256': {str(path): file_sha256(path) for path in outputs},
                    'protocol': protocol, 'cpu_only': True, 'test_access': False, 'new_generated_samples': 0})
        return selected, review


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    selected, review = select(args.calibration, args.outdir, args.plan)
    print(json.dumps({'selected': selected, 'full_candidates': len(review['candidates'])}, indent=2), flush=True)
    if selected is None:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
