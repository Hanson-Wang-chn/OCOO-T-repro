"""Relocate two immutable inference artifacts of explicitly finished pilots.

Full resumable checkpoints are handled separately. Active/full-data runs are
refused. Copy and fsync the destination, verify its SHA256, then atomically
replace the original path with a symlink. No prediction or metric is changed.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

import torch
import yaml

from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


ARTIFACTS = ('best_denoise.pt', 'final_ema.pt')


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def relocate(run_names, destination, audit):
    destination, audit = Path(destination).resolve(), Path(audit)
    records = []
    with exclusive_run(audit, 'inference_migration'):
        for name in run_names:
            run = Path(name).absolute()
            if not (run / 'finished.json').is_file():
                raise ValueError(f'Unfinished run is ineligible: {run}')
            with exclusive_run(run, 'training'):
                config = yaml.safe_load((run / 'config.yaml').read_text())
                if not config['training'].get('train_subset'):
                    raise ValueError('Only finished subset pilots are eligible')
                finished = json.loads((run / 'finished.json').read_text())
                for filename in ARTIFACTS:
                    source = run / filename
                    target = destination / run.name / filename
                    if target == source or run in target.parents:
                        raise ValueError('Archive destination must be outside the pilot run')
                    if source.is_symlink() and (source.resolve() != target or not target.is_file()):
                        raise ValueError('Existing artifact link points elsewhere')
                    before = source.stat()
                    digest = file_sha256(source)
                    state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
                    if (state.get('config') != config or not state.get('model')
                            or not 0 < state['step'] <= finished['step']
                            or (filename == 'final_ema.pt' and state['step'] != finished['step'])):
                        raise ValueError(f'Artifact does not belong to the finished pilot: {source}')
                    newly_relocated = not source.is_symlink()
                    if newly_relocated:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if target.exists():
                            if file_sha256(target) != digest:
                                raise FileExistsError('Archive destination contains different bytes')
                        else:
                            temporary = target.with_suffix('.pt.copying')
                            with source.open('rb') as reader, temporary.open('wb') as writer:
                                shutil.copyfileobj(reader, writer, length=16 << 20)
                                writer.flush()
                                os.fsync(writer.fileno())
                            if file_sha256(temporary) != digest:
                                raise ValueError('Copied artifact SHA256 mismatch; local original retained')
                            temporary.replace(target)
                        descriptor = os.open(target.parent, os.O_DIRECTORY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                        after = source.stat()
                        if (before.st_ino, before.st_size, before.st_mtime_ns) != (
                                after.st_ino, after.st_size, after.st_mtime_ns):
                            raise RuntimeError('Source changed during copy; local original retained')
                        link = source.with_suffix('.pt.link.tmp')
                        if link.exists() or link.is_symlink():
                            raise FileExistsError('Temporary artifact link already exists')
                        link.symlink_to(target)
                        os.replace(link, source)
                        descriptor = os.open(run, os.O_DIRECTORY)
                        try:
                            os.fsync(descriptor)
                        finally:
                            os.close(descriptor)
                    record = {'time': timestamp(), 'run': str(run), 'artifact': filename,
                              'archive_path': str(target), 'original_path': str(source),
                              'sha256': digest, 'bytes': before.st_size, 'step': state['step'],
                              'newly_relocated': newly_relocated, 'original_path_is_symlink': True}
                    del state
                    records.append(record)
                    log_event(audit / 'events.jsonl', {'event': 'inference_artifact_archived', **record})
                    atomic_json(audit / 'state.json', {'time': timestamp(), 'stage': 'running', 'records': records})
        result = {'time': timestamp(), 'stage': 'completed', 'records': records,
                  'bytes_newly_relocated': sum(r['bytes'] for r in records if r['newly_relocated']),
                  'script_sha256': file_sha256(__file__)}
        atomic_json(audit / 'state.json', result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+')
    parser.add_argument('--destination', required=True)
    parser.add_argument('--audit-dir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = relocate(args.runs, args.destination, args.audit_dir)
    print(json.dumps({'artifacts': len(result['records']),
                      'GiB_newly_relocated': result['bytes_newly_relocated'] / 2**30}, indent=2))


if __name__ == '__main__':
    main()
