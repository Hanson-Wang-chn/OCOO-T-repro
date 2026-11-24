"""Audit completed validation CFG cases on CPU while the main model trains."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-dir', required=True)
    parser.add_argument('--controller-pid', required=True, type=int)
    args = parser.parse_args()
    out = Path(args.review_dir)
    with exclusive_run(out, 'post_analysis'):
        complete = out / 'post_analysis_complete.json'
        if complete.exists():
            return
        status = out / 'post_analysis_state.json'
        def state(stage, **details):
            atomic_json(status, {'time': timestamp(), 'pid': os.getpid(), 'stage': stage,
                                 'gpu_disabled': True, 'test_access': False, **details})
        try:
            while True:
                path = out / 'review_state.json'
                current = json.loads(path.read_text()) if path.exists() else {}
                if current.get('stage') == 'failed':
                    raise RuntimeError(f'Inference controller failed: {current.get("error")}')
                if current.get('stage') == 'complete':
                    break
                os.kill(args.controller_pid, 0)
                state('waiting_for_cases', controller_pid=args.controller_pid)
                time.sleep(30)
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8',
                               MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
            state('paired_metric_analysis')
            subprocess.run([sys.executable, 'scripts/analyze_inference_review.py', '--review-dir', str(out)],
                           env=environment, check=True)
            cases = json.loads((out / 'comparison.json').read_text())
            state('context_and_variance_analysis')
            subprocess.run([sys.executable, 'scripts/diagnose_pilots.py',
                            *[str(out / tag) for tag in cases], '--outdir', str(out / 'context')],
                           env=environment, check=True)
            result = {'time': timestamp(), 'stage': 'completed_review_required',
                      'paired_analysis': str(out / 'paired_differences.json'),
                      'context': str(out / 'context/context_diagnostics.json'),
                      'script_sha256': file_sha256(__file__), 'test_access': False,
                      'main_training_changed': False, 'formal_settings_selected': False}
            atomic_json(complete, result)
            state('completed_review_required', result=result)
            readme_event(f'冻结断点的验证CFG对照及CPU九指标配对/条件诊断已完成：`{out}/comparison.json`；'
                         '主训练继续，尚未锁定正式推理配置，未计算测试结果。')
        except Exception as error:
            state('failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
