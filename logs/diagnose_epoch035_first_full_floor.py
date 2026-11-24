"""Earlier CPU-only review of the already registered first-full floor0.05.

The original two-refinement calibration and four-candidate selector continue
unchanged. This separate diagnostic cannot select or promote a test protocol.
"""
import argparse
import json
import os
from pathlib import Path
import runpy

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import official_metrics
from ocoot.postprocess import apply_zero_floor, quality_checks, primary_score
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = read(plan_path)
    for group in ('source_sha256', 'prerequisite_sha256'):
        for path, digest in plan[group].items():
            if file_sha256(path) != digest:
                raise ValueError('Registered first-full diagnostic input changed: ' + path)
    torch.set_num_threads(8)
    calibration = Path(plan['calibration']).resolve()
    provenance = read(calibration / 'calibration_config.json')
    assert provenance['step'] == 39130 and provenance['weights'] == 'ema_0.9996'
    assert provenance['refine_cap'] is None and provenance['batch_size'] == 32
    original_plan = read('logs/postprocess_workflow_update/plan.json')
    assert original_plan['floors'] == [0., .05]
    assert original_plan['quality_thresholds'] == plan['quality_thresholds']
    records = read(calibration / 'all_results.json')
    matches = [r for r in records if r['tag'] == 'refine_00']
    if len(matches) != 1:
        raise ValueError('Wait for the first complete original full refinement and its official nine metrics')
    record = matches[0]
    assert record['guidance'] > 0 and record['cap'] is None and record['clamp']
    for key in ('guidance', 'cfg_min', 'cfg_max'):
        assert record[key] == plan['expected_sampling'][key]
    checkpoint = Path(provenance['checkpoint'])
    assert file_sha256(checkpoint) == plan['checkpoint_sha256'] == provenance['checkpoint_sha256']
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    assert state['weights'] == 'ema_0.9996' and state['step'] == 39130
    cfg = state['config']
    assert not cfg['training'].get('train_subset') and cfg['data_dir'] == provenance['data_dir']
    data = BenchmarkData(cfg['data_dir'])
    assert file_sha256(data.path / 'expression.npy') == plan['data_expression_sha256']
    indices = data.evaluation_indices('validation', None)
    assert len(indices) == 4825
    labels = {data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][indices])}
    assert len(labels) == 60
    helpers = runpy.run_path('scripts/select_validation_postprocess.py')
    source, raw = helpers['validate_refinement'](calibration, provenance, cfg, data, indices, record)
    baseline = helpers['verified_metrics'](source / 'metrics_clamped', record['metrics'], 60, labels)
    out = Path(plan['outdir'])
    protocol = {'plan_sha256': file_sha256(plan_path),
                'calibration_config_sha256': file_sha256(calibration / 'calibration_config.json'),
                'checkpoint_sha256': provenance['checkpoint_sha256'],
                'data_expression_sha256': plan['data_expression_sha256'],
                'complete_refinement_record': record, 'zero_floor': .05,
                'independent_of_formal_selector_outputs': True}
    with exclusive_run(out, 'first_full_floor_diagnostic'):
        execution = out / 'execution_config.json'
        if execution.exists() and read(execution) != protocol:
            raise ValueError('Retain this diagnostic; its registered input differs')
        if not execution.exists():
            atomic(execution, protocol)
        completed = out / 'completed.json'
        if completed.exists():
            proof = read(completed)
            assert proof['protocol'] == protocol
            for path, digest in proof['output_sha256'].items():
                assert file_sha256(path) == digest
            print(json.dumps(proof))
            return
        metrics_dir = out / 'metrics_floor_0p05'
        if not (metrics_dir / 'summary.json').exists():
            official_metrics(data, indices, apply_zero_floor(raw, .05), metrics_dir, de=True)
        metrics = helpers['verified_metrics'](metrics_dir, conditions=60, labels=labels)
        result = {'time': timestamp(), 'scope': 'Early single-candidate complete validation diagnostic only',
                  'cells': 4825, 'conditions': 60, 'guidance': record['guidance'],
                  'cfg_min': record['cfg_min'], 'cfg_max': record['cfg_max'],
                  'baseline_floor0': {'metrics': baseline, 'primary': primary_score(baseline),
                                      'checks': quality_checks(baseline, plan['quality_thresholds'])},
                  'diagnostic_floor0p05': {'metrics': metrics, 'primary': primary_score(metrics),
                                         'checks': quality_checks(metrics, plan['quality_thresholds'])},
                  'raw_negative_fraction': float(np.mean(raw < 0)),
                  'zero_fraction_floor0': float(np.mean(raw <= 0)),
                  'zero_fraction_floor0p05': float(np.mean(raw < .05)),
                  'full_four_candidate_review_still_required': True,
                  'zero_floor_unreported_by_paper': True, 'final_protocol_adopted': False,
                  'training_or_original_calibration_changed': False, 'new_generated_samples': 0,
                  'test_access': False, 'protocol': protocol}
        atomic(out / 'review.json', result)
        outputs = [out / 'review.json', metrics_dir / 'summary.json', metrics_dir / 'results.csv']
        atomic(completed, {'time': timestamp(), 'cpu_only': True, 'protocol': protocol,
                          'output_sha256': {str(p): file_sha256(p) for p in outputs},
                          'final_protocol_adopted': False, 'test_access': False})
        print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
