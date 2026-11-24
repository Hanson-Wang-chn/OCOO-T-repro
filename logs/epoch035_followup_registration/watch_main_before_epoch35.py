"""Read-only main-run health and registered E30-review status snapshot."""
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
                  'free_GiB': round(state['local_free_GiB'], 2),
                  'anon_GiB': round(state['memory_anon_GiB'], 2),
                  'readme_age': round(state['README_age_seconds']), 'break_reason': reasons}))
