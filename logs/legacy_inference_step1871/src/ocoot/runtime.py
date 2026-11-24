"""Deterministic run state, atomic checkpointing and append-only progress log."""
import datetime as dt
from contextlib import contextmanager
import json
import os
from pathlib import Path
import random
import time
import fcntl
import hashlib
import numpy as np
import torch


@contextmanager
def exclusive_run(directory,kind):
    """A process crash releases the OS lock; PID files alone cannot do this."""
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    with open(directory/f'{kind}.lock','a+') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            lock.seek(0);owner=lock.read().strip()
            raise RuntimeError(f'{kind} is already running in {directory} ({owner})') from None
        lock.seek(0);lock.truncate();lock.write(f'pid={os.getpid()}');lock.flush()
        try:yield
        finally:fcntl.flock(lock,fcntl.LOCK_UN)


def timestamp():
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec='seconds')


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def file_sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(16<<20),b''):h.update(block)
    return h.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(s):
    random.setstate(s['python']); np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state_all(s['cuda'])


def atomic_save(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with open(tmp, 'wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def log_event(path, event):
    event = {'time': timestamp(), **event}
    with open(path, 'a') as f: f.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + '\n')
    print(json.dumps(event, ensure_ascii=False, allow_nan=False), flush=True)


def edit_readme(transform):
    Path('logs').mkdir(exist_ok=True)
    with open('logs/readme.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        p=Path('README.md');tmp=p.with_suffix('.md.tmp')
        tmp.write_text(transform(p.read_text()));tmp.replace(p)


def readme_event(message,important=True):
    entry=f'\n- {timestamp()}：{message}\n'
    Path('logs').mkdir(exist_ok=True)
    with open('logs/progress_history.md','a') as history:history.write(entry)
    if important:_queue_readme(entry=entry)


def update_readme_live(content):
    _queue_readme(live=content)


def flush_readme(force=False):
    """Publish queued progress at most once per 15 minutes (user preference)."""
    return _queue_readme(force=force)


def _queue_readme(live=None,entry=None,force=False):
    Path('logs').mkdir(exist_ok=True)
    with open('logs/readme.lock','a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        pending_path=Path('logs/readme_pending.json')
        pending=json.loads(pending_path.read_text()) if pending_path.exists() else {'events':[]}
        if live is not None:pending['live']=live
        if entry is not None:pending['events'].append(entry)
        p=Path('README.md')
        due=not p.exists() or time.time()-p.stat().st_mtime>=15*60
        if (force or due) and (pending.get('live') is not None or pending['events']):
            text=p.read_text() if p.exists() else '# OCOO-T 复现\n'
            if pending.get('live') is not None:
                start='<!-- LIVE_TRAINING_START -->';end='<!-- LIVE_TRAINING_END -->'
                block=f'{start}\n\n{pending["live"]}\n\n{end}'
                text=(text[:text.index(start)]+block+text[text.index(end)+len(end):]
                      if start in text and end in text else text+'\n'+block+'\n')
            for event in pending['events']:
                if event.strip() not in text:text+=event
            tmp=p.with_suffix('.md.tmp');tmp.write_text(text);tmp.replace(p)
            if pending_path.exists():pending_path.unlink()
            return True
        tmp=pending_path.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(pending,ensure_ascii=False));tmp.replace(pending_path)
        return False


class DualEMA:
    def __init__(self, model, decays=(0.9999, 0.9996)):
        self.decays = decays
        self.weights = [{n: p.detach().clone() for n, p in model.named_parameters()} for _ in decays]

    @torch.no_grad()
    def update(self, model):
        params = dict(model.named_parameters())
        for decay, weights in zip(self.decays, self.weights):
            torch._foreach_lerp_(list(weights.values()), [params[n] for n in weights], 1 - decay)

    def state_dict(self):
        return {str(d): {n: p.cpu() for n, p in w.items()} for d, w in zip(self.decays, self.weights)}

    def load_state_dict(self, states, device):
        self.weights = [{n: p.to(device) for n, p in states[str(d)].items()} for d in self.decays]

    @torch.no_grad()
    def copy_to(self, model, index=1):
        for n, p in model.named_parameters(): p.copy_(self.weights[index][n])

