"""Read-only main-run health and registered periodic-review status snapshot."""
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time


def read(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def alive(pid, start_tick=None):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().split(') ')[1].split()
        return fields[0] != 'Z' and (start_tick is None or fields[19] == start_tick)
    except (OSError, IndexError):
        return False


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
    temporary.replace(path)


now = datetime.now(timezone(timedelta(hours=8)))
run = Path('runs/main_v1')
records = []
for line in (run / 'events.jsonl').read_text().splitlines()[-150:]:
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        pass


def latest(kind):
    return next((r for r in reversed(records) if r.get('event') == kind), {})


finished = read(run / 'finished.json')
observer = read(run / 'observation/state.json')
epoch30 = run / 'interim_epoch030_fixed_cfg'
follow = read(epoch30 / 'process.json')
complete30 = read(epoch30 / 'completed.json')
plan30 = read(epoch30 / 'plan.json')
epoch35 = run / 'interim_epoch035_fixed_cfg'
follow35 = read(epoch35 / 'process.json')
complete35 = read(epoch35 / 'completed.json')
plan35 = read(epoch35 / 'plan.json')
epoch40 = run / 'interim_epoch040_fixed_cfg'
follow40 = read(epoch40 / 'process.json')
complete40 = read(epoch40 / 'completed.json')
plan40 = read(epoch40 / 'plan.json')
calibration35_root = run / 'interim_epoch035_calibration'
calibration35 = read(calibration35_root / 'process.json')
calibration35_complete = read(calibration35_root / 'completed.json')
calibration35_plan = read(calibration35_root / 'plan.json')
calibration35_progress = {p.parent.name: read(p) for p in
                          (calibration35_root / 'calibration').glob('*/generation_state.json')}
main = latest('train')
memory = {key: int(value) for key, value in [line.split() for line in
          Path('/sys/fs/cgroup/memory/memory.stat').read_text().splitlines()]}
oom = Path('/sys/fs/cgroup/memory/memory.oom_control').read_text()
state = {'time': now.isoformat(timespec='seconds'), 'main_train': main,
         'main_validation': latest('validation'), 'main_generation': latest('validation_generation'),
         'main_checkpoint': read(run / 'checkpoint_state.json'),
         'main_alive': alive(656512, '783674341'), 'main_finished': bool(finished),
         'pipeline_alive': alive(1100222), 'pipeline': read(run / 'pipeline_state.json'),
         'observer': observer, 'observer_alive': alive(observer.get('pid', -1)),
         'epoch30_preservation': read('logs/preservation_epoch030/state.json'),
         'epoch30_follower': follow, 'epoch30_completed': complete30,
         'epoch35_preservation': read('logs/preservation_epoch035/state.json'),
         'epoch35_follower': follow35, 'epoch35_completed': complete35,
         'epoch40_preservation': read('logs/preservation_epoch040/state.json'),
         'epoch40_follower': follow40, 'epoch40_completed': complete40,
         'epoch35_calibration': calibration35, 'epoch35_calibration_completed': calibration35_complete,
         'epoch35_calibration_generation_progress': calibration35_progress,
         'main_config_unchanged': sha('configs/main_selected.yaml') ==
             'e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52',
         'local_free_GiB': shutil.disk_usage('.').free / 2**30,
         'memory_anon_GiB': memory['rss'] / 2**30,
         'memory_file_cache_GiB': (memory['inactive_file'] + memory['active_file']) / 2**30,
         'oom_control': oom, 'README_age_seconds': time.time() - Path('README.md').stat().st_mtime}
reasons = []
if not state['main_finished'] and not state['main_alive']:
    reasons.append('main_trainer_changed_or_exited')
if not state['pipeline_alive']:
    reasons.append('pipeline_exited_review')
if not state['main_config_unchanged']:
    reasons.append('main_config_changed')
if (not state['main_finished'] and not state['observer_alive']) or observer.get('failures'):
    reasons.append('observer_review')
if state['local_free_GiB'] < 10:
    reasons.append('low_disk')
if state['memory_anon_GiB'] > 90:
    reasons.append('high_anonymous_memory')
if '\noom_kill 0\n' not in '\n' + oom:
    reasons.append('oom_kill_count')
if state['README_age_seconds'] > 1600:
    reasons.append('readme_stale')
if main and not state['main_finished'] and (now - datetime.fromisoformat(main['time'])).total_seconds() > 1800:
    reasons.append('main_training_stale')
if main and any(not math.isfinite(main[key]) for key in ('loss', 'lr', 'grad_norm')):
    reasons.append('nonfinite_main_training')
if follow.get('stage') == 'failed':
    reasons.append('epoch30_follower_failed')
elif follow and not complete30 and not alive(follow.get('pid', -1)):
    reasons.append('epoch30_follower_exited')
if state['epoch30_preservation'].get('stage') == 'failed':
    reasons.append('epoch30_preserver_failed')
elif (state['epoch30_preservation'].get('stage') in ('waiting_step33540', 'archiving')
      and not alive(state['epoch30_preservation'].get('pid', -1))):
    reasons.append('epoch30_preserver_exited')
if plan30 and not complete30:
    if any(sha(path) != digest for path, digest in plan30['source_sha256'].items()):
        reasons.append('registered_epoch30_source_changed')
if complete30 and not (epoch30 / 'decision.json').exists():
    reasons.append('epoch30_complete_review_required')
if follow35.get('stage') == 'failed':
    reasons.append('epoch35_follower_failed')
elif follow35 and not complete35 and not alive(follow35.get('pid', -1)):
    reasons.append('epoch35_follower_exited')
preservation35 = state['epoch35_preservation']
if preservation35.get('stage') == 'failed':
    reasons.append('epoch35_preserver_failed')
elif (preservation35.get('stage') in ('waiting_step39130', 'archiving')
      and not alive(preservation35.get('pid', -1))):
    reasons.append('epoch35_preserver_exited')
if plan35 and not complete35:
    if any(sha(path) != digest for path, digest in plan35['source_sha256'].items()):
        reasons.append('registered_epoch35_source_changed')
if complete35 and not (epoch35 / 'decision.json').exists():
    reasons.append('epoch35_complete_review_required')
if calibration35.get('stage') in ('failed', 'interrupted'):
    reasons.append('epoch35_calibration_' + calibration35['stage'])
elif calibration35 and not calibration35_complete:
    if not alive(calibration35.get('pid', -1)):
        reasons.append('epoch35_calibration_controller_exited')
    elif calibration35.get('child_pid') and not alive(calibration35['child_pid']):
        if time.time() - (calibration35_root / 'process.json').stat().st_mtime > 60:
            reasons.append('epoch35_calibration_child_exited_review')
if calibration35_plan and not calibration35_complete:
    if any(sha(path) != digest for path, digest in calibration35_plan['source_sha256'].items()):
        reasons.append('registered_epoch35_calibration_source_changed')
if calibration35_complete and not (calibration35_root / 'decision.json').exists():
    reasons.append('epoch35_full_calibration_complete_review_required')
if follow40.get('stage') == 'failed':
    reasons.append('epoch40_follower_failed')
elif follow40 and not complete40 and not alive(follow40.get('pid', -1)):
    reasons.append('epoch40_follower_exited')
preservation40 = state['epoch40_preservation']
if preservation40.get('stage') == 'failed':
    reasons.append('epoch40_preserver_failed')
elif (preservation40.get('stage') in ('waiting_step44720', 'archiving')
      and not alive(preservation40.get('pid', -1))):
    reasons.append('epoch40_preserver_exited')
if plan40 and not complete40:
    if any(sha(path) != digest for path, digest in plan40['source_sha256'].items()):
        reasons.append('registered_epoch40_source_changed')
if complete40 and not (epoch40 / 'decision.json').exists():
    reasons.append('epoch40_complete_review_required')
if state['main_finished']:
    reasons.append('main_training_finished_next_phase_review')
state['break_reason'] = reasons
write('logs/main_watch_latest.json', state)
snapshot = read('logs/active_work_snapshot.json')
snapshot['time'] = state['time']
snapshot['main'].update(latest_train=main, latest_checkpoint=state['main_checkpoint'],
                        latest_observer_state=observer, completed_epoch=observer.get('completed_epoch', 27))
if state['main_validation']:
    snapshot['main']['latest_denoise'] = state['main_validation']
if state['main_generation']:
    generation = state['main_generation']
    snapshot['main']['last_epoch_checkpoint_step'] = generation['step']
    snapshot['main']['latest_generation'] = {k: v for k, v in generation.items()
                                            if k not in ('time', 'event', 'step', 'epoch')}
snapshot['epoch30_fixed_cfg_followup'] = follow
snapshot['epoch35_fixed_cfg_followup'] = follow35
snapshot['epoch35_preservation'] = preservation35
snapshot['epoch40_fixed_cfg_followup'] = follow40
snapshot['epoch40_preservation'] = preservation40
snapshot['epoch35_complete_validation_calibration'] = calibration35
snapshot['epoch35_calibration_generation_progress'] = calibration35_progress
snapshot['latest_main_health'] = 'logs/main_watch_latest.json'
publications = list(Path('logs').glob('readme_*.published.json'))
last_publication = 0.
if publications:
    newest = max(publications, key=lambda p: p.stat().st_mtime)
    snapshot['readme_last_structured_publication'] = str(newest)
    last_publication = newest.stat().st_mtime
pending = [p for p in Path('logs').glob('readme_*update.json')
           if not p.with_suffix('.published.json').exists() and p.stat().st_mtime > last_publication]
snapshot['readme_pending_edit'] = str(max(pending, key=lambda p: p.stat().st_mtime)) if pending else None
write('logs/active_work_snapshot.json', snapshot)
print(json.dumps({'time': state['time'], 'epoch': main.get('epoch'), 'step': main.get('step'),
                  'rate': main.get('cells_per_sec'),
                  'ema_endpoint_mse': state['main_validation'].get('ema_0.9996', {}).get('endpoint_mse'),
                  'epoch30_followup': follow.get('stage'),
                  'epoch35_followup': follow35.get('stage'),
                  'epoch40_followup': follow40.get('stage'),
                  'epoch35_calibration': calibration35.get('stage'),
                  'epoch35_calibration_progress': {tag: [v.get('completed'), v.get('total')]
                                                   for tag, v in calibration35_progress.items()},
                  'free_GiB': round(state['local_free_GiB'], 2),
                  'anon_GiB': round(state['memory_anon_GiB'], 2),
                  'readme_age': round(state['README_age_seconds']), 'break_reason': reasons}))
