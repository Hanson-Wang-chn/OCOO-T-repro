"""Run registered validation-only calibration of the immutable E35 model."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from ocoot.distributed import review_trainer_pids
from ocoot.runtime import exclusive_run, file_sha256, readme_event, timestamp


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def guard_main(plan):
    run = Path(plan['main_run'])
    if 'trainer_workers' in plan:
        review_trainer_pids(plan['trainer_workers'], run)
    else:
        fields = Path(f'/proc/{plan["trainer_pid"]}/stat').read_text().split(') ')[1].split()
        if fields[0] == 'Z' or fields[19] != plan['trainer_start_tick']:
            raise RuntimeError('Original trainer changed or exited; review the interim validation job')
        if (run / 'finished.json').exists():
            raise RuntimeError('Training finished; review before further interim selection')
    if list(run.glob('test_full*')):
        raise RuntimeError('A test directory exists; review before further interim selection')
    if file_sha256('configs/main_selected.yaml') != plan['main_config_sha256']:
        raise ValueError('Main configuration changed')


def commands(plan):
    root = Path(plan['outdir'])
    calibration = root / 'calibration'
    return {
        'calibration': [sys.executable, 'scripts/calibrate.py', '--checkpoint', plan['checkpoint'],
                        '--outdir', str(calibration), '--screen-cap', '16', '--refine-cap', 'full',
                        '--batch-size', '32', '--scales', '0.6666666667,1,1.5,2'],
        'postprocessing': [sys.executable, 'scripts/select_validation_postprocess.py',
                           '--calibration', str(calibration), '--outdir', str(calibration / 'postprocessing'),
                           '--plan', 'logs/postprocess_workflow_update/plan.json'],
        'promotion': [sys.executable, 'scripts/promote_validation_cache.py',
                      '--calibration', str(calibration), '--selection', str(calibration / 'postprocessing/selected.json'),
                      '--outdir', str(root / 'validation_full')],
    }


class Controller:
    def __init__(self, plan, root):
        self.plan, self.root = plan, Path(root)
        self.stop = False

    def state(self, stage, **extra):
        atomic(self.root / 'process.json', {'time': timestamp(), 'pid': os.getpid(), 'stage': stage,
               'test_access': False, 'main_training_changed': False, **extra})

    def execute(self, command, stage, cpu=False, allowed=(0,), poll_seconds=2):
        environment = dict(os.environ)
        if cpu:
            environment['CUDA_VISIBLE_DEVICES'] = ''
        child = None
        try:
            with (self.root / f'{stage}.log').open('a') as stream:
                child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                self.state(stage, child_pid=child.pid, command=command, cpu_only=cpu)
                deadline = time.monotonic() + 48 * 3600
                while child.poll() is None:
                    guard_main(self.plan)
                    if self.stop:
                        raise InterruptedError('Interim controller interrupted; completed caches retained')
                    if time.monotonic() > deadline:
                        raise TimeoutError('Interim validation stage exceeded48h; inspect cached progress')
                    time.sleep(poll_seconds)
                if child.returncode not in allowed:
                    raise RuntimeError(f'{stage} exited with{child.returncode}; caches retained')
                return child.returncode
        finally:
            if child is not None and child.poll() is None:
                # Only this validation child is signalled, never the trainer
                # or its process group. Resume uses the existing batch caches.
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    plan_path = Path(args.plan)
    plan = read(plan_path)
    guard_main(plan)
    for group in ('source_sha256', 'prerequisite_sha256'):
        for path, digest in plan[group].items():
            if file_sha256(path) != digest:
                raise ValueError('Registered input changed: ' + path)
    if file_sha256(plan['checkpoint']) != plan['checkpoint_sha256']:
        raise ValueError('FrozenE35 checkpoint changed')
    proof = read('runs/main_v1/interim_epoch035_fixed_cfg/checkpoint_verification.json')
    if (proof['step'] != 39130 or proof['checkpoint_sha256'] != plan['checkpoint_sha256']
            or not proof['all193_tensors_equal_default_ema_and_buffers']):
        raise ValueError('E35 default-EMA tensor proof is required')
    original = read('logs/postprocess_workflow_update/plan.json')
    if original['quality_thresholds'] != plan['quality_thresholds'] or original['floors'] != plan['floors']:
        raise ValueError('Keep the registered floors and every original quality threshold')
    job_commands = commands(plan)
    if job_commands != plan['commands']:
        raise ValueError('Only the three original validation-only commands are allowed')
    gpu = sorted({int(x) for x in subprocess.check_output(
        ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).split()})
    trainer_pids = {worker['pid'] for worker in plan.get('trainer_workers', [])} or {plan['trainer_pid']}
    if not set(gpu).issubset(trainer_pids):
        raise RuntimeError('Wait until only the original trainer occupies the GPU')
    if args.check_only:
        print(json.dumps({'passed': True, 'commands_validation_only': True, 'gpu_pids': gpu,
                          'checkpoint_sha256': plan['checkpoint_sha256'], 'test_access': False}))
        return
    root = Path(plan['outdir'])
    root.mkdir(exist_ok=True, parents=True)
    with exclusive_run(root, 'validation_calibration_controller'):
        control = Controller(plan, root)
        def stop(signum, frame):
            control.stop = True
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            if (root / 'completed.json').exists():
                completed = read(root / 'completed.json')
                if completed['plan_sha256'] != file_sha256(plan_path):
                    raise ValueError('Completed validation has another plan')
                for path, digest in completed['output_sha256'].items():
                    if file_sha256(path) != digest:
                        raise ValueError('Completed validation result changed: ' + path)
                print(json.dumps(completed))
                return
            control.state('registered_validation_calibration_starting', plan_sha256=file_sha256(plan_path))
            readme_event('冻结E35默认EMA开始中途完整验证校准：沿用原正CFG搜索和两个全量4825细化，随后检查0/.05；主训练及质量线保持，不执行测试。见 `runs/main_v1/interim_epoch035_calibration/plan.json`。')
            control.execute(job_commands['calibration'], 'validation_cfg_calibration')
            guard_main(plan)
            code = control.execute(job_commands['postprocessing'], 'validation_postprocessing', cpu=True, allowed=(0, 2))
            guard_main(plan)
            calibration = root / 'calibration'
            selected = read(calibration / 'postprocessing/review.json')['selected']
            if code == 2 and selected is not None:
                raise ValueError('Infeasible selector exit does not agree with its review')
            outputs = [calibration / name for name in ('calibration_config.json', 'all_results.json', 'selected.json',
                       'postprocessing/review.json', 'postprocessing/execution_config.json')]
            if code == 0:
                if selected is None:
                    raise ValueError('Passing selector exit has no selected candidate')
                control.execute(job_commands['promotion'], 'validation_cache_verification', cpu=True)
                outputs += [calibration / 'postprocessing/completed.json', calibration / 'postprocessing/selected.json',
                            root / 'validation_full/metrics/summary.json',
                            root / 'validation_full/promotion_config.json', root / 'validation_full/promotion_completed.json']
            guard_main(plan)
            completed = {'time': timestamp(), 'status': 'completed_validation_review_required',
                         'plan_sha256': file_sha256(plan_path), 'checkpoint_sha256': plan['checkpoint_sha256'],
                         'any_original_gate_feasible_candidate': code == 0, 'selected_validation_case': selected,
                         'final_test_protocol_adopted': False, 'test_access': False, 'main_training_changed': False,
                         'output_sha256': {str(p): file_sha256(p) for p in outputs}}
            atomic(root / 'completed.json', completed)
            control.state('completed_validation_review_required', result=completed, cpu_only=True)
            readme_event('冻结E35中途完整验证校准及原质量检查完成，等待复核；训练未改，验证候选尚未采用为最终测试设置，未执行测试。见 `runs/main_v1/interim_epoch035_calibration/completed.json`。')
        except BaseException as error:
            control.state('interrupted' if isinstance(error, InterruptedError) else 'failed', error=repr(error))
            raise


if __name__ == '__main__':
    main()
