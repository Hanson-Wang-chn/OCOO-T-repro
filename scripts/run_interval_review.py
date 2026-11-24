"""Generate one CFG-interval variant, then audit the controlled difference."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--cfg-min', type=float, default=0.)
    parser.add_argument('--cfg-max', type=float, default=1.)
    args = parser.parse_args()
    if not 0 <= args.cfg_min < args.cfg_max <= 1:
        parser.error('invalid CFG interval')
    reference = Path(args.reference).resolve()
    out = Path(args.outdir)
    with exclusive_run(out, 'interval_review'):
        ref = json.loads((reference / 'evaluation_config.json').read_text())
        if ref['split'] != 'validation' or not ref['clamp'] or not ref['de'] or ref['guidance'] is None:
            raise ValueError('Requires a nonnegative, positive-CFG nine-metric validation reference')
        if ref['steps'] != 50 or ref['cap'] is None or ref['seed'] != 4242:
            raise ValueError('This runner is for the fixed Heun50 capped intermediate protocol')
        if [ref['cfg_min'], ref['cfg_max']] == [args.cfg_min, args.cfg_max]:
            raise ValueError('Candidate interval equals reference')
        if not (reference / 'metrics/summary.json').exists():
            raise ValueError('Reference metrics are incomplete')
        if file_sha256(ref['checkpoint']) != ref['checkpoint_sha256']:
            raise ValueError('Frozen checkpoint changed')
        config = {'reference': str(reference), 'reference_config': ref,
                  'reference_sha256': {name: file_sha256(reference / name) for name in (
                      'evaluation_config.json', 'indices.npy', 'generation_state.json',
                      'prediction.npy', 'prediction_raw.npy', 'metrics/summary.json', 'metrics/results.csv')},
                  'candidate_interval': [args.cfg_min, args.cfg_max],
                  'source_sha256': {name: file_sha256(name) for name in (
                      __file__, 'scripts/evaluate.py', 'scripts/compare_inference_cases.py',
                      'src/ocoot/model.py', 'src/ocoot/flow.py', 'src/ocoot/evaluate.py')},
                  'purpose': 'One-factor interval diagnostic, no formal settings or test selection'}
        protocol = out / 'interval_config.json'
        if protocol.exists() and json.loads(protocol.read_text()) != config:
            raise ValueError('Output is pinned to a different interval comparison')
        atomic_json(protocol, config)
        if (out / 'completed.json').exists():
            return
        def state(stage, **details):
            atomic_json(out / 'interval_state.json', {'time': timestamp(), 'pid': os.getpid(),
                        'stage': stage, 'test_access': False, 'main_training_changed': False, **details})
        try:
            candidate = out / 'candidate'
            command = [sys.executable, 'scripts/evaluate.py', '--checkpoint', ref['checkpoint'],
                       '--split', 'validation', '--outdir', str(candidate), '--cap', str(ref['cap']),
                       '--batch-size', str(ref['batch_size']), '--guidance', str(ref['guidance']),
                       '--cfg-min', str(args.cfg_min), '--cfg-max', str(args.cfg_max), '--clamp', '--de']
            with (out / 'evaluation.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=log)
                state('generating_and_evaluating', child_pid=child.pid, command=command)
                code = child.wait()
            if code:
                raise RuntimeError(f'Evaluation exited with {code}; cache retained')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
                               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
            state('paired_comparison')
            subprocess.run([sys.executable, 'scripts/compare_inference_cases.py', '--reference', str(reference),
                            '--candidate', str(candidate), '--factor', 'cfg_interval',
                            '--outdir', str(out / 'paired')], env=environment, check=True)
            state('context_and_variance_analysis')
            subprocess.run([sys.executable, 'scripts/diagnose_pilots.py', str(reference), str(candidate),
                            '--outdir', str(out / 'context')], env=environment, check=True)
            result = {'time': timestamp(), 'stage': 'completed_review_required',
                      'comparison': str(out / 'paired/comparison.json'),
                      'context': str(out / 'context/context_diagnostics.json'),
                      'script_sha256': file_sha256(__file__), 'test_access': False,
                      'main_training_changed': False, 'formal_setting_selected': False}
            atomic_json(out / 'completed.json', result)
            state('completed_review_required', result=result)
            readme_event(f'冻结断点CFG区间对照及九指标配对分析完成：`{out}/paired/comparison.json`；'
                         '原主训练继续，正式推理配置尚未锁定，未计算测试结果。')
        except Exception as error:
            state('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
