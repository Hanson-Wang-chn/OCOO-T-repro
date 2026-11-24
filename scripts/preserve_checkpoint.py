"""Preserve an exact-step full checkpoint without touching active training.

The trainer atomically replaces last.pt. A separate hard link pins one completed
inode, so subsequent trainer saves cannot change the bytes being archived.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import torch

from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--step', type=int, required=True)
    parser.add_argument('--destination', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run = Path(args.run_dir).resolve()
    out = run / 'snapshots' / f'step_{args.step}'
    target = Path(args.destination).resolve()
    if target == run / 'last.pt' or run in target.parents:
        raise ValueError('Archive destination must be outside the active run')
    with exclusive_run(out, 'preservation'):
        pinned = out / 'full.pt'
        if not pinned.exists():
            if pinned.is_symlink():
                raise FileNotFoundError('Existing archive link is broken')
            os.link(run / 'last.pt', pinned)
        before = pinned.stat()
        state = torch.load(pinned, map_location='cpu', mmap=True, weights_only=False)
        if state['step'] != args.step:
            raise ValueError(f'Pinned step {state["step"]} differs from requested {args.step}; '
                             'active training was not changed; inspect the pinned copy')
        if not all(key in state for key in ('model', 'optimizer', 'ema', 'rng', 'config')):
            raise ValueError('Incomplete full checkpoint')
        if set(state['ema']) != {'0.9999', '0.9996'} or not state['optimizer']['state']:
            raise ValueError('Dual EMA or populated optimizer state is missing')
        digest = file_sha256(pinned)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_sha256(target) != digest:
                raise FileExistsError('Archive destination contains different bytes')
        else:
            temporary = target.with_suffix('.pt.copying')
            with pinned.open('rb') as reader, temporary.open('wb') as writer:
                shutil.copyfileobj(reader, writer, length=16 << 20)
                writer.flush()
                os.fsync(writer.fileno())
            if file_sha256(temporary) != digest:
                raise ValueError('Archive SHA256 mismatch; pinned and active files retained')
            temporary.replace(target)
            directory = os.open(target.parent, os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        after = pinned.stat()
        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('Pinned bytes changed; retain both copies for inspection')
        if pinned.is_symlink():
            if pinned.resolve() != target:
                raise ValueError('Existing snapshot link points to another archive')
        else:
            link = out / 'full.pt.link.tmp'
            if link.exists() or link.is_symlink():
                raise FileExistsError('Temporary snapshot link already exists')
            link.symlink_to(target)
            os.replace(link, pinned)
        result = {'time': timestamp(), 'step': state['step'],
                  'source_run': str(run), 'snapshot_path': str(pinned),
                  'archive_path': str(target), 'sha256': digest, 'bytes': target.stat().st_size,
                  'ema_decays': list(state['ema']),
                  'pending_generation_epoch': state.get('pending_generation_epoch'),
                  'full_model_optimizer_ema_rng_retained': True,
                  'active_training_or_last_checkpoint_modified': False,
                  'script_sha256': file_sha256(__file__)}
        atomic_json(out / 'preserved.json', result)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
