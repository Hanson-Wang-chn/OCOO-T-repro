"""Reuse a selected full-validation calibration cache, without new sampling.

Verify its checkpoint, source code, options, indices and sample bytes, retain
independent raw/nonnegative arrays, then recompute all nine official metrics on
CPU. This changes neither the selection objective nor validation quality gates.
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS, official_metrics
from ocoot.runtime import exclusive_run, file_sha256, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def copy_checked(source, destination, digest):
    if destination.exists():
        if file_sha256(destination) != digest:
            raise ValueError(f'Existing destination differs: {destination}')
        return
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with source.open('rb') as reader, temporary.open('wb') as writer:
        shutil.copyfileobj(reader, writer, length=16 << 20)
        writer.flush()
        os.fsync(writer.fileno())
    if file_sha256(temporary) != digest:
        raise ValueError('Sample copy SHA256 mismatch; source retained')
    temporary.replace(destination)


def promote(calibration, destination):
    calibration, destination = Path(calibration).resolve(), Path(destination).resolve()
    selected_path = calibration / 'selected.json'
    selected = json.loads(selected_path.read_text())
    provenance_path = calibration / 'calibration_config.json'
    provenance = json.loads(provenance_path.read_text())
    if (selected['cap'] is not None or not selected['clamp']
            or selected.get('formal_guidance_policy') != 'literal_dual_cfg_positive_scale'
            or selected.get('weights') != 'ema_0.9996'):
        raise ValueError('Reuse requires full-validation default-EMA positive-CFG refinement')
    if (not math.isfinite(selected['guidance']) or selected['guidance'] <= 0
            or selected['sampling_steps'] != 50 or selected['seed'] != 4242
            or not 0 <= selected['cfg_min'] <= selected['cfg_max'] <= 1):
        raise ValueError('Selected sampling protocol is invalid')
    for key in ('checkpoint', 'checkpoint_sha256', 'step', 'weights', 'data_dir', 'source_sha256'):
        if selected[key] != provenance[key]:
            raise ValueError(f'Selection and calibration provenance differ: {key}')
    if provenance['formal_guidance_policy'] != selected['formal_guidance_policy']:
        raise ValueError('Formal guidance policies differ')
    for path, digest in selected['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError(f'Calibration source changed: {path}')
    checkpoint = Path(selected['checkpoint'])
    if file_sha256(checkpoint) != selected['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
    cfg = state['config']
    weights = 'ema_0.9996' if '0.9996' in state.get('ema', {}) else state.get('weights')
    if (state['step'] != selected['step'] or weights != selected['weights']
            or cfg['training'].get('train_subset') or cfg['data_dir'] != selected['data_dir']):
        raise ValueError('Checkpoint does not match full-data refinement')
    del state
    cache = selected['cache_provenance']
    source = Path(cache['directory']).resolve()
    if (not selected['tag'].startswith('refine_') or source != calibration / selected['tag']
            or not cache['raw_unclamped'] or cache['split'] != 'validation'):
        raise ValueError('Selected cache is not the registered refinement')
    if destination == source or source in destination.parents or destination in source.parents:
        raise ValueError('Source and destination must be independent directories')
    expected_files = {'prediction.npy', 'indices.npy', 'generation_state.json'}
    if set(cache['files']) != expected_files:
        raise ValueError('Incomplete calibration sample provenance')
    for name, digest in cache['files'].items():
        if file_sha256(source / name) != digest:
            raise ValueError(f'Calibration cache changed: {name}')
    options = {'noise_std': cfg['training']['noise_std'], 'batch_size': selected['batch_size'],
               'steps': 50, 'guidance': selected['guidance'], 'cfg_min': selected['cfg_min'],
               'cfg_max': selected['cfg_max'], 'clamp': False, 'seed': 4242}
    marker = json.loads((source / 'generation_state.json').read_text())
    data = BenchmarkData(cfg['data_dir'])
    indices = data.evaluation_indices('validation', None)
    np.testing.assert_array_equal(np.load(source / 'indices.npy'), indices)
    if marker != {'completed': len(indices), 'total': len(indices), 'options': options}:
        raise ValueError('Calibration generation is incomplete or has different options')
    raw = np.load(source / 'prediction.npy', mmap_mode='r')
    if raw.dtype != np.float32 or raw.shape != (len(indices), data.x.shape[1]):
        raise ValueError('Full-validation prediction schema differs')
    protocol = {'selected_sha256': file_sha256(selected_path),
                'calibration_config_sha256': file_sha256(provenance_path),
                'source': str(source), 'source_files_sha256': cache['files'],
                'checkpoint_sha256': selected['checkpoint_sha256'],
                'data_files_sha256': {name: file_sha256(data.path / name)
                                      for name in ('expression.npy', 'metadata.npz', 'vocab.json')},
                'script_sha256': file_sha256(__file__)}
    with exclusive_run(destination, 'promotion'):
        protocol_file = destination / 'promotion_config.json'
        completed_file = destination / 'promotion_completed.json'
        if protocol_file.exists():
            if json.loads(protocol_file.read_text()) != protocol:
                raise ValueError('Destination is pinned to another promotion')
        else:
            if any(p.name != 'promotion.lock' for p in destination.iterdir()):
                raise FileExistsError('Retain existing validation artifacts; use their original workflow')
            atomic_json(protocol_file, protocol)
        if completed_file.exists():
            completed = json.loads(completed_file.read_text())
            if completed['protocol'] != protocol:
                raise ValueError('Completed promotion has different provenance')
            for name, digest in completed['output_sha256'].items():
                if file_sha256(destination / name) != digest:
                    raise ValueError(f'Promoted output changed: {name}')
            return completed
        copy_checked(source / 'prediction.npy', destination / 'prediction_raw.npy',
                     cache['files']['prediction.npy'])
        copy_checked(source / 'indices.npy', destination / 'indices.npy', cache['files']['indices.npy'])
        temporary = destination / 'prediction.npy.tmp'
        clipped = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=raw.shape)
        for offset in range(0, len(raw), 4096):
            chunk = np.asarray(raw[offset:offset + 4096])
            if not np.isfinite(chunk).all():
                raise ValueError('Nonfinite calibration predictions')
            clipped[offset:offset + len(chunk)] = np.maximum(chunk, 0)
        clipped.flush()
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        del clipped
        temporary.replace(destination / 'prediction.npy')
        options['clamp'] = True
        atomic_json(destination / 'generation_state.json',
                    {'completed': len(indices), 'total': len(indices), 'options': options})
        evaluation = {'checkpoint': str(checkpoint.resolve()),
                      'checkpoint_sha256': selected['checkpoint_sha256'], 'step': selected['step'],
                      'weights': weights, 'split': 'validation', 'cap': None,
                      'guidance': selected['guidance'], 'cfg_min': selected['cfg_min'],
                      'cfg_max': selected['cfg_max'], 'clamp': True, 'de': True,
                      'seed': 4242, 'steps': 50, 'batch_size': selected['batch_size'],
                      'noise_std': cfg['training']['noise_std'], 'model': cfg['model']}
        atomic_json(destination / 'evaluation_config.json', evaluation)
        prediction = np.load(destination / 'prediction.npy', mmap_mode='r')
        metrics = official_metrics(data, indices, prediction, destination / 'metrics', de=True)
        if set(metrics) != set(METRICS.values()):
            raise ValueError('Full validation did not return all nine metrics')
        np.testing.assert_allclose([metrics[k] for k in METRICS.values()],
                                   [selected['metrics'][k] for k in METRICS.values()], rtol=0, atol=1e-10)
        completed = {'time': timestamp(), 'source': str(source), 'cells': len(indices),
                     'new_generated_samples': 0, 'gpu_used': False, 'test_access': False,
                     'raw_samples_retained_bitwise': True, 'all9_metrics_recomputed_and_agree_atol': 1e-10,
                     'metrics': metrics, 'protocol': protocol,
                     'output_sha256': {name: file_sha256(destination / name) for name in
                                      ('prediction_raw.npy', 'prediction.npy', 'indices.npy',
                                       'generation_state.json', 'evaluation_config.json',
                                       'metrics/summary.json', 'metrics/results.csv')}}
        atomic_json(completed_file, completed)
        return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', required=True)
    parser.add_argument('--outdir', required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    print(json.dumps(promote(args.calibration, args.outdir), indent=2), flush=True)


if __name__ == '__main__':
    main()
