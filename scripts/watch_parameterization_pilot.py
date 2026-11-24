"""Finish CPU-only audits and lossless archival after a registered pilot."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def write_state(path, stage, **details):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps({'time': timestamp(), 'pid': os.getpid(),
                                    'stage': stage, **details}, indent=2))
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--controller-pid', type=int, required=True)
    parser.add_argument('--factor', choices=('prediction_space', 'value_input_init'), default='prediction_space')
    args = parser.parse_args()
    trial = 'xpred_vloss' if args.factor == 'prediction_space' else 'value_fanin'
    label = 'x-prediction+v-loss' if args.factor == 'prediction_space' else '标量值MLP fan-in初始化'
    run = Path(f'runs/pilot_{trial}')
    out = Path(f'runs/{trial}_analysis')
    out.mkdir(parents=True, exist_ok=True)
    state = out / 'watch_state.json'
    with exclusive_run(out, 'post_analysis'):
        if (out / 'post_analysis_complete.json').exists():
            return
        try:
            comparison = Path(f'runs/{trial}_trial_comparison.json')
            while not comparison.exists():
                os.kill(args.controller_pid, 0)
                write_state(state, 'waiting_for_training_and_metrics', controller_pid=args.controller_pid,
                            main_model_mutations=False, test_access=False)
                time.sleep(30)
            if not (run / 'finished.json').exists():
                raise ValueError('Comparison appeared before pilot training completed')
            env = os.environ.copy()
            env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='8')
            tasks = [
                ('paired_analysis', ['scripts/analyze_parameterization_trial.py', '--candidate', str(run),
                                     '--outdir', str(out), '--factor', args.factor]),
                ('cell_context_diagnostics', ['scripts/diagnose_pilots.py',
                                             'runs/pilot_sigma_1.0_cellinit_1.0', str(run),
                                             '--outdir', f'runs/{trial}_context']),
                ('lossless_checkpoint_archive', ['scripts/archive_finished_checkpoints.py', str(run),
                                                 '--destination', '/home/dataset-assist-0/whs/ocoo-t-artifacts/completed-pilots',
                                                 '--audit-dir', f'logs/storage_migration_{trial}']),
            ]
            for stage, command in tasks:
                write_state(state, stage, command=command, main_model_mutations=False, test_access=False)
                subprocess.run([sys.executable, *command], env=env, check=True)
            result = {'time': timestamp(), 'stage': 'completed_review_required',
                      'comparison': str(out / 'comparison.json'),
                      'context': f'runs/{trial}_context/context_diagnostics.json',
                      'storage': f'logs/storage_migration_{trial}/state.json',
                      'main_model_mutations': False, 'automatic_main_switch': False, 'test_access': False,
                      'script_sha256': file_sha256(__file__)}
            (out / 'post_analysis_complete.json').write_text(json.dumps(result, indent=2))
            write_state(state, 'completed_review_required', result=result)
            readme_event(f'{label}短对照、CPU九指标配对分析与完整断点SHA归档均完成；原全量v模型继续。'
                         f'结果 `{out}/comparison.json` 等待综合复核，不自动换主模型，未访问测试表达。')
        except Exception as error:
            write_state(state, 'failed', error=repr(error), main_model_mutations=False, test_access=False)
            raise


if __name__ == '__main__':
    main()
