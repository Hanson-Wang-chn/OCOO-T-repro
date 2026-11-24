"""Replace the waiting workflow parent; never signal or restart the trainer."""
from datetime import datetime, timezone, timedelta
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


ROOT = Path('/home/whs/ocoo-t')
OUT = ROOT / 'logs/postprocess_workflow_update'
TRAINER = 656512
OLD_PARENT = 961459
TICK = '783674341'
os.chdir(ROOT)


def now():
    return datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds')


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def process(pid):
    fields = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()
    return {'pid': pid, 'state': fields[0], 'start_tick': fields[19],
            'command': Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode().strip(),
            'cwd': os.readlink(f'/proc/{pid}/cwd')}


def lock_held(path):
    with Path(path).open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def start_count():
    return sum(json.loads(line).get('event') == 'start' for line in
               Path('runs/main_v1/events.jsonl').read_text().splitlines())


def main():
    assert not (OUT / 'deployment.json').exists()
    assert not (OUT / 'deployment_plan.json').exists(), 'Inspect any interrupted handoff first'
    verified = read(OUT / 'verification.json')
    assert verified['passed'] and verified['unique_cpu_checks_passed'] == 35
    assert all(sha(path) == digest for path, digest in verified['source_sha256'].items())
    decision = read('runs/main_v1/interim_epoch030_fixed_cfg/decision.json')
    assert decision['status'] == 'jointly_reviewed_continue_original_training'
    before = process(TRAINER)
    old_parent = process(OLD_PARENT)
    assert before['start_tick'] == TICK and before['state'] != 'Z'
    assert old_parent['cwd'] == before['cwd'] == str(ROOT)
    assert old_parent['command'].endswith('scripts/run_experiment.py --config configs/main_selected.yaml')
    assert read('runs/main_v1/pipeline_state.json')['pid'] == OLD_PARENT
    assert lock_held('runs/main_v1/training.lock') and lock_held('runs/main_v1/pipeline.lock')
    assert not Path('runs/main_v1/finished.json').exists()
    assert not Path('runs/main_v1/calibration/selected.json').exists()
    assert not list(Path('runs/main_v1').glob('test_full*'))
    assert sha('configs/main_selected.yaml') == verified['main_config_sha256']
    starts = start_count()
    assert starts == 2
    checkpoint_before = read('runs/main_v1/checkpoint_state.json')
    monitor = Path('logs/watch_main_training.py')
    monitor_text = monitor.read_text()
    assert monitor_text.count('alive(961459)') == 1
    (OUT / 'watch_main_training_before.py').write_text(monitor_text)
    command = [sys.executable, 'scripts/run_experiment.py', '--config', 'configs/main_selected.yaml',
               '--postprocess-plan', 'logs/postprocess_workflow_update/plan.json']
    plan = {'time': now(), 'trainer_before': before, 'old_parent': old_parent,
            'command': command, 'verified_implementation_sha256': sha(OUT / 'verification.json'),
            'epoch30_decision_sha256': sha('runs/main_v1/interim_epoch030_fixed_cfg/decision.json'),
            'checkpoint_before': checkpoint_before, 'training_start_events': starts,
            'main_config_sha256': sha('configs/main_selected.yaml'),
            'monitor_before_sha256': sha(monitor), 'deploy_script_sha256': sha(__file__),
            'scope': 'Only terminate the waiting parent. Retain the original running trainer, config, optimizer and schedule.'}
    write(OUT / 'deployment_plan.json', plan)
    # Identity checks immediately precede the single PID-targeted signal.
    assert process(OLD_PARENT) == old_parent
    os.kill(OLD_PARENT, signal.SIGTERM)
    deadline = time.monotonic() + 15
    while lock_held('runs/main_v1/pipeline.lock'):
        if time.monotonic() > deadline:
            raise TimeoutError('Old pipeline lock did not release; trainer remains unchanged')
        time.sleep(.2)
    assert process(TRAINER)['start_tick'] == TICK
    environment = dict(os.environ)
    # Preserve the trainer's CUDA visibility without displaying its environment.
    trainer_environment = dict(entry.split(b'=', 1) for entry in
                               Path(f'/proc/{TRAINER}/environ').read_bytes().split(b'\0') if b'=' in entry)
    environment.pop('CUDA_VISIBLE_DEVICES', None)
    if b'CUDA_VISIBLE_DEVICES' in trainer_environment:
        environment['CUDA_VISIBLE_DEVICES'] = trainer_environment[b'CUDA_VISIBLE_DEVICES'].decode()
    with Path('logs/main_pipeline_postprocess_20261121.log').open('a') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    write(OUT / 'deployment_launch.json', {'time': now(), 'new_parent_pid': child.pid, 'command': command})
    deadline = time.monotonic() + 25
    while True:
        state = read('runs/main_v1/pipeline_state.json')
        if state.get('pid') == child.pid and state.get('stage') == 'training':
            break
        if child.poll() is not None or time.monotonic() > deadline:
            raise RuntimeError('New waiting parent failed to attach; inspect launch log')
        time.sleep(.3)
    assert state['attached_existing_training']
    assert state['postprocess_plan_sha256'] == sha(OUT / 'plan.json')
    assert lock_held('runs/main_v1/training.lock') and lock_held('runs/main_v1/pipeline.lock')
    assert process(TRAINER)['start_tick'] == TICK and start_count() == starts
    assert sha('configs/main_selected.yaml') == plan['main_config_sha256']
    assert '--postprocess-plan' in process(child.pid)['command']
    assert read('runs/main_v1/checkpoint_state.json')['step'] >= checkpoint_before['step']
    assert not Path('runs/main_v1/finished.json').exists()
    assert not list(Path('runs/main_v1').glob('test_full*'))
    monitor.write_text(monitor_text.replace('alive(961459)', f'alive({child.pid})'))
    result = {'time': now(), 'status': 'optional_full_validation_workflow_activated_waiting_for_training',
              'old_parent_pid': OLD_PARENT, 'new_parent_pid': child.pid, 'trainer_pid_unchanged': TRAINER,
              'trainer_start_tick_unchanged': TICK, 'training_start_events_unchanged': starts,
              'main_config_sha256_unchanged': plan['main_config_sha256'],
              'new_pipeline_state': state, 'positive_floor_adopted': False,
              'quality_thresholds_relaxed': False, 'new_training_or_gpu_jobs': 0,
              'formal_calibration_started': False, 'test_access': False,
              'monitor_after_sha256': sha(monitor), 'deployment_plan_sha256': sha(OUT / 'deployment_plan.json')}
    write(OUT / 'deployment.json', result)
    snapshot = read('logs/active_work_snapshot.json')
    snapshot['time'] = result['time']
    snapshot['objective_status'] = 'unfinished; E30 joint review completed; original main continues towardE35; optional full-validation0/.05 selection deployed but no floor adopted; no formal calibration/test'
    snapshot['main']['pipeline_pid'] = child.pid
    snapshot['main']['best_generation_epoch'] = 30
    snapshot['main']['early_reference_epoch'] = 30
    snapshot['main']['earliest_possible_original_early_stop_epoch'] = 40
    snapshot['epoch30_joint_decision'] = 'runs/main_v1/interim_epoch030_fixed_cfg/decision.json'
    snapshot['optional_postprocess_workflow'] = {
        'status': result['status'], 'plan': 'logs/postprocess_workflow_update/plan.json',
        'verification': 'logs/postprocess_workflow_update/verification.json',
        'deployment': 'logs/postprocess_workflow_update/deployment.json',
        'source_archive': 'logs/postprocess_workflow_update/source_snapshot.tar.gz',
        'cpu_checks_passed': 35, 'previous_waiting_parent': OLD_PARENT,
        'active_waiting_parent': child.pid, 'formal_choice_not_adopted': True}
    snapshot['next_actions'] = [
        'Register passive exactE35 preservation and one matched fixedCFG followup, then continue the original main and jointly review E30-to-E35.',
        'Original NoCFG checkpoint/early-stopping policy stays unchanged; both E30 point rankings preferE30 but primary intervals crosszero and CFG errors increase.',
        'After training completion, original positiveCFG calibration and two full4825 refinements feed the verified optional0/.05 selector; no candidate passing all unchanged checks means stop before test.',
        'Only then freeze the validation-selected protocol for full26878-cell/380-condition test; retain clamp0 results and label any unreported positive floor.'
    ]
    write('logs/active_work_snapshot.json', snapshot)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
