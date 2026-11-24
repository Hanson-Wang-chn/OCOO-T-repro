"""Track the recovery job and preserve its two diagnostic checkpoints."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from ocoot.distributed import live_worker_pids
from ocoot.runtime import exclusive_run, readme_event, timestamp


RUN = Path('runs/main_ddp_recovery')
PRIOR = Path('runs/main_v1')
LOG = Path('logs/epoch040_recovery')
ARTIFACTS = Path('/home/dataset-assist-0/whs/ocoo-t-artifacts/main-snapshots')
MILESTONES = {42250: 'recovery_step042250', 44720: 'epoch040_ddp_recovery'}


def read(path):
    return json.loads(path.read_text())


def write_status(value):
    path = LOG / 'monitor_state.json'
    tmp = path.with_suffix('.json.tmp')
    tmp.write_text(json.dumps({'time': timestamp(), **value}, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def events(directory):
    rows = []
    for line in (directory / 'events.jsonl').read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def summary(rows):
    validation = next((row for row in reversed(rows) if row['event'] == 'validation'), None)
    if validation is not None:
        raw = validation['raw']['endpoint_mse']
        ema = validation['ema_0.9996']['endpoint_mse']
        validation = {'step': validation['step'], 'raw_endpoint_mse': raw, 'ema_endpoint_mse': ema,
                      'ema_relative_gap': ema / raw - 1}
    train = next((row for row in reversed(rows) if row['event'] == 'train'), None)
    return validation, train


def advance(launch):
    rows = events(RUN)
    validation, train = summary(rows)
    halt = next((row for row in reversed(rows) if row['event'] in ('ema_health_stop', 'start')), None)
    checkpoint = RUN / 'checkpoint_state.json'
    preserved = []
    for step, name in MILESTONES.items():
        local = RUN / 'snapshots' / f'step_{step}'
        if (local / 'full.pt').exists() and not (local / 'preserved.json').exists():
            subprocess.run([sys.executable, 'scripts/preserve_checkpoint.py', '--run-dir', str(RUN),
                            '--step', str(step), '--destination', str(ARTIFACTS / name / 'full.pt')],
                           check=True)
        if (local / 'preserved.json').exists():
            preserved.append(step)
    comparison = LOG / 'repaired_e35_e40/comparison.json'
    if (RUN / 'observation/epoch_040.json').exists() and not comparison.exists():
        subprocess.run([sys.executable, 'scripts/compare_generation_epochs.py',
                        '--run-dir', str(RUN), '--reference-run-dir', str(PRIOR),
                        '--reference-epoch', '35', '--candidate-epoch', '40',
                        '--outdir', str(comparison.parent)], check=True)
        result = read(comparison)
        readme_event(f'修复版 E40 的九项匹配验证已与 E35 配对。综合分变化 {result["score_difference"]["value"]:+.4f}。'
                     f'均使用原普通条件、相同索引与噪声。完整结果见 `{comparison}`。')
    finished = (RUN / 'finished.json').exists()
    stage = 'training'
    if finished:
        stage = 'training_complete'
    elif halt and halt['event'] == 'ema_health_stop':
        stage = 'ema_health_stop'
    else:
        try:
            live_worker_pids(launch['workers'])
        except (FileNotFoundError, RuntimeError):
            # Wait for rank zero to finalize a cooperative stop or a new
            # launcher to publish its identity before classifying the state.
            stage = 'workers_exited'
    return {'stage': stage, 'training': train, 'last_validation': validation,
            'checkpoint': read(checkpoint) if checkpoint.exists() else None,
            'preserved_steps': preserved, 'e35_e40_comparison': str(comparison) if comparison.exists() else None,
            'free_gib': round(shutil.disk_usage(RUN).free / 2**30, 1)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-final-workflow', action='store_true')
    args = parser.parse_args()
    with exclusive_run(LOG, 'monitor'):
        (LOG / 'monitor.pid').write_text(str(os.getpid()))
        while True:
            try:
                launch = read(RUN / 'training_execution.json')
                result = advance(launch)
            except subprocess.CalledProcessError as error:
                # A failed archive or comparison is retried while the immutable
                # input remains pinned. The training process owns its checkpoint.
                result = {'stage': 'artifact_retry', 'error': str(error)}
            write_status(result)
            if result['stage'] == 'ema_health_stop':
                return
            if result['stage'] == 'training_complete':
                if not args.run_final_workflow:
                    return
                if result['e35_e40_comparison']:
                    run_final_workflow(launch, result)
                    return
            time.sleep(45)


def run_final_workflow(launch, result):
    config = 'configs/main_ddp_recovery.yaml'
    plan = 'logs/postprocess_workflow_update/plan.json'
    device = launch['cuda_visible_devices'].split(',')[0]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device)
    write_status({**result, 'stage': 'final_workflow_running', 'final_gpu': device})
    process = subprocess.run([sys.executable, 'scripts/run_experiment.py', '--config', config,
                              '--postprocess-plan', plan], env=env)
    state = RUN / 'pipeline_state.json'
    pipeline = read(state) if state.exists() else {}
    write_status({**result, 'stage': pipeline.get('stage', 'final_workflow_exited'),
                  'pipeline_exit_code': process.returncode, 'pipeline_state': pipeline})
    if process.returncode not in (0, 2):
        process.check_returncode()


if __name__ == '__main__':
    main()
