"""Wait passively for E40, then archive the exact complete checkpoint."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from ocoot.runtime import exclusive_run, timestamp

OUT = Path('logs/preservation_epoch040')
RUN = Path('runs/main_v1')
TARGET_STEP = 44720
TRAINER_PID = 656512
TRAINER_START_TICK = '783674341'
CONFIG_SHA = 'e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52'
PRESERVER_SHA = 'dc28700d86c5c16ce260aad4aca03e121155aadd8aa2264115b2af75bda5c177'


def state(stage, **extra):
    value = {'time': timestamp(), 'pid': os.getpid(), 'stage': stage,
             'target_step': TARGET_STEP, 'training_changed': False, **extra}
    path = OUT / 'state.json'
    temp = path.with_suffix('.json.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    with exclusive_run(OUT, 'watcher'):
        if (RUN / 'snapshots/step_44720/preserved.json').exists():
            state('already_preserved')
            return
        state('waiting_step44720', purpose='Preserve an exact mature checkpoint; no new experiment or policy change')
        deadline = time.monotonic() + 48 * 3600
        last_heartbeat = time.monotonic()
        while time.monotonic() < deadline:
            try:
                cp = json.loads((RUN / 'checkpoint_state.json').read_text())
            except (FileNotFoundError, json.JSONDecodeError):
                time.sleep(2)
                continue
            if cp['step'] > TARGET_STEP:
                raise RuntimeError('Exact E40 checkpoint no longer current; do not substitute another step')
            if cp['step'] == TARGET_STEP:
                assert digest('configs/main_selected.yaml') == CONFIG_SHA
                assert digest('scripts/preserve_checkpoint.py') == PRESERVER_SHA
                command = [sys.executable, 'scripts/preserve_checkpoint.py', '--run-dir', str(RUN),
                           '--step', str(TARGET_STEP), '--destination',
                           '/home/dataset-assist-0/whs/ocoo-t-artifacts/main-snapshots/epoch040/full.pt']
                state('archiving', command=command)
                with (OUT / 'archive.log').open('a') as stream:
                    subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
                state('completed', record='runs/main_v1/snapshots/step_44720/preserved.json')
                return
            fields = Path(f'/proc/{TRAINER_PID}/stat').read_text().split(') ')[1].split()
            if fields[0] == 'Z' or fields[19] != TRAINER_START_TICK:
                raise RuntimeError('Original trainer exited or changed before E40; review without relaunching')
            if time.monotonic() - last_heartbeat >= 3600:
                state('waiting_step44720', current_step=cp['step'])
                last_heartbeat = time.monotonic()
            time.sleep(20)
        raise TimeoutError('E40 not reached within48h; no training mutation')


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        OUT.mkdir(exist_ok=True)
        state('failed', error=repr(exc))
        raise
