"""Byte-preserving relocation of completed pilots' full checkpoints.

Copy, fsync, verify SHA256, then atomically replace the original path with a
symlink. Only explicitly named, finished subset pilots are accepted. All model,
optimizer, EMA and RNG state remains intact; active runs are protected by locks.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import torch
import yaml

from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--audit-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    destination, audit = Path(args.destination).resolve(), Path(args.audit_dir)
    destination.mkdir(parents=True, exist_ok=True)
    with exclusive_run(audit, 'migration'):
        records = []
        for name in args.runs:
            run = Path(name)
            if not (run / 'finished.json').exists():
                raise ValueError(f'Unfinished run is ineligible: {run}')
            with exclusive_run(run, 'training'):
                config = yaml.safe_load((run / 'config.yaml').read_text())
                if not config['training'].get('train_subset'):
                    raise ValueError('This archiver is restricted to completed subset pilots')
                finished = json.loads((run / 'finished.json').read_text())
                source = run / 'last.pt'
                target = destination / run.name / 'last.pt'
                target.parent.mkdir(exist_ok=True)
                if source.is_symlink():
                    if source.resolve() != target or not target.is_file():
                        raise ValueError('Existing checkpoint link points elsewhere')
                    digest = file_sha256(target)
                    state = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
                else:
                    before = source.stat()
                    digest = file_sha256(source)
                    state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
                    if state['step'] != finished['step'] or not all(k in state for k in ('model', 'optimizer', 'ema', 'rng')):
                        raise ValueError('Checkpoint is incomplete or not the finished step')
                    if target.exists():
                        if file_sha256(target) != digest:
                            raise FileExistsError('Destination contains different checkpoint bytes')
                    else:
                        temporary = target.with_suffix('.pt.copying')
                        with source.open('rb') as reader, temporary.open('wb') as writer:
                            shutil.copyfileobj(reader, writer, length=16 << 20)
                            writer.flush()
                            os.fsync(writer.fileno())
                        if file_sha256(temporary) != digest:
                            raise ValueError('Copied checkpoint SHA256 mismatch; local checkpoint retained')
                        temporary.replace(target)
                    after = source.stat()
                    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                        raise ValueError('Source changed during copy; local checkpoint retained')
                    link = source.with_suffix('.pt.link.tmp')
                    if link.exists() or link.is_symlink():
                        raise FileExistsError('Temporary checkpoint link already exists')
                    link.symlink_to(target)
                    # Persist the verified destination before replacing local bytes.
                    descriptor = os.open(target.parent, os.O_DIRECTORY)
                    try: os.fsync(descriptor)
                    finally: os.close(descriptor)
                    os.replace(link, source)
                if state['step'] != finished['step'] or 'optimizer' not in state:
                    raise ValueError('Archived checkpoint state does not match finished run')
                record = {'time': timestamp(), 'run': str(run), 'original_path': str(source.absolute()),
                          'archive_path': str(target), 'bytes': target.stat().st_size, 'sha256': digest,
                          'step': state['step'], 'full_model_optimizer_ema_rng_retained': True,
                          'original_path_is_symlink': source.is_symlink()}
                del state
                atomic_json(run / 'checkpoint_storage.json', record)
                records.append(record)
                log_event(audit / 'events.jsonl', {'event': 'checkpoint_archived', **record})
                atomic_json(audit / 'state.json', {'time': timestamp(), 'stage': 'running', 'records': records})
        atomic_json(audit / 'state.json', {'time': timestamp(), 'stage': 'completed', 'records': records,
                                          'archived_bytes': sum(r['bytes'] for r in records),
                                          'script_sha256': file_sha256(__file__)})


if __name__ == '__main__':
    main()
