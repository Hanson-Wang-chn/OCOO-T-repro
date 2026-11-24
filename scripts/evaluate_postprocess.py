"""Run a locked, validated full test once, then derive its selected zero floor.

The original evaluator generates and evaluates the complete clamp0 baseline.
Its raw samples are retained. The optional transformation runs only on CPU;
test measurements cannot change the already frozen selection.
"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import runpy
import subprocess

import numpy as np
import pandas as pd
import torch

from ocoot.data import BenchmarkData
from ocoot.evaluate import METRICS, official_metrics
from ocoot.postprocess import (DEFAULT_THRESHOLDS, apply_zero_floor, quality_checks,
                              verify_selection_postprocessing)
from ocoot.runtime import exclusive_run, file_sha256, gpu_command, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def training_locked(run):
    lock = Path(run) / 'training.lock'
    if not lock.exists():
        return False
    with lock.open('a+') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
    return False


def verified_validation(selection_path, validation):
    selection_path, validation = Path(selection_path).resolve(), Path(validation).resolve()
    selected = json.loads(selection_path.read_text())
    floor = verify_selection_postprocessing(selected, require_review=True)
    if (selected['sampling_steps'] != 50 or selected['seed'] != 4242
            or selected['formal_guidance_policy'] != 'literal_dual_cfg_positive_scale'
            or selected['guidance'] is None or selected['guidance'] <= 0):
        raise ValueError('A frozen positive-CFG paper sampler is required')
    completion = json.loads((selection_path.parent / 'completed.json').read_text())
    if completion['selected_sha256'] != file_sha256(selection_path):
        raise ValueError('Postprocessing selection is not complete and immutable')
    promotion = json.loads((validation / 'promotion_completed.json').read_text())
    if promotion['protocol']['selected_sha256'] != file_sha256(selection_path):
        raise ValueError('Complete validation belongs to another selection')
    if promotion['protocol']['script_sha256'] != file_sha256('scripts/promote_validation_cache.py'):
        raise ValueError('Validation promotion implementation changed')
    for name, digest in promotion['output_sha256'].items():
        if file_sha256(validation / name) != digest:
            raise ValueError('Validated output changed: ' + name)
    for name, digest in promotion['protocol']['data_files_sha256'].items():
        if file_sha256(Path(selected['data_dir']) / name) != digest:
            raise ValueError('Validated data changed: ' + name)
    metrics = json.loads((validation / 'metrics/summary.json').read_text())
    np.testing.assert_allclose([metrics[k] for k in METRICS.values()],
                               [selected['metrics'][k] for k in METRICS.values()], rtol=0, atol=1e-10)
    for path, digest in selected['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Calibrated implementation changed: ' + path)
    if file_sha256(selected['checkpoint']) != selected['checkpoint_sha256']:
        raise ValueError('Validated checkpoint changed')
    checkpoint = torch.load(selected['checkpoint'], map_location='cpu', mmap=True, weights_only=False)
    cfg = checkpoint['config']
    weights = 'ema_0.9996' if '0.9996' in checkpoint.get('ema', {}) else checkpoint.get('weights')
    if (cfg['training'].get('train_subset') or checkpoint['step'] != selected['step']
            or weights != 'ema_0.9996' or cfg['data_dir'] != selected['data_dir']):
        raise ValueError('Test requires the validated full-data default EMA')
    run = Path(cfg['run_dir'])
    if training_locked(run) or not (run / 'finished.json').exists():
        raise ValueError('Full training must finish before formal test generation')
    data = BenchmarkData(cfg['data_dir'])
    np.testing.assert_array_equal(np.load(validation / 'indices.npy'), data.evaluation_indices('validation', None))
    evaluation = json.loads((validation / 'evaluation_config.json').read_text())
    if (evaluation['split'] != 'validation' or evaluation['cap'] is not None
            or not evaluation['de'] or not evaluation['clamp']
            or evaluation.get('zero_floor', 0.) != floor):
        raise ValueError('Requires the selected complete validation transformation')
    for key in ('checkpoint_sha256', 'step', 'weights', 'guidance', 'cfg_min', 'cfg_max', 'batch_size', 'seed'):
        if evaluation[key] != selected[key]:
            raise ValueError('Validation sampling settings differ: ' + key)
    if evaluation['steps'] != 50 or evaluation['noise_std'] != selected['noise_std']:
        raise ValueError('Validation sampler or noise differs')
    return selected, floor, cfg, data


def run_test(selection_path, validation, baseline, destination, gpu_workers=1):
    selection_path = Path(selection_path).resolve()
    selected, floor, cfg, data = verified_validation(selection_path, validation)
    validation, baseline, destination = map(lambda p: Path(p).resolve(), (validation, baseline, destination))
    paths = [validation, baseline, destination]
    if any(a == b or a in b.parents or b in a.parents for i, a in enumerate(paths) for b in paths[i + 1:]):
        raise ValueError('Keep validation, baseline test and selected test directories independent')
    indices = data.evaluation_indices('test', None)
    expected_options = {'noise_std': selected['noise_std'], 'batch_size': selected['batch_size'],
                        'steps': 50, 'guidance': selected['guidance'], 'cfg_min': selected['cfg_min'],
                        'cfg_max': selected['cfg_max'], 'clamp': True, 'seed': 4242}
    expected_evaluation = {'checkpoint': str(Path(selected['checkpoint']).resolve()),
                           'checkpoint_sha256': selected['checkpoint_sha256'], 'step': selected['step'],
                           'weights': 'ema_0.9996', 'split': 'test', 'cap': None, 'de': True,
                           'model': cfg['model'], **{k: v for k, v in expected_options.items()}}
    protocol = {'selection_sha256': file_sha256(selection_path),
                'validation_completion_sha256': file_sha256(validation / 'promotion_completed.json'),
                'checkpoint_sha256': selected['checkpoint_sha256'], 'zero_floor': floor,
                'baseline': str(baseline), 'test_cells': len(indices),
                'test_perturbations': len(np.unique(data.meta['pert'][indices])),
                'validation_checks': quality_checks(selected['metrics'], DEFAULT_THRESHOLDS),
                'expected_baseline_evaluation': expected_evaluation,
                'source_sha256': {name: file_sha256(name) for name in
                    ('scripts/evaluate_postprocess.py', 'scripts/evaluate.py', 'src/ocoot/postprocess.py')}}
    with exclusive_run(destination, 'postprocess_test'), exclusive_run(baseline, 'postprocess_baseline_test'):
        frozen = destination / 'test_protocol.json'
        if frozen.exists() and json.loads(frozen.read_text()) != protocol:
            raise ValueError('Test settings changed after freezing; retain the original experiment')
        if not frozen.exists():
            if any(p.name != 'postprocess_test.lock' for p in destination.iterdir()):
                raise FileExistsError('Keep existing test artifacts and their original workflow')
            atomic_json(frozen, protocol)
        complete = destination / 'postprocess_test_completed.json'
        if complete.exists():
            result = json.loads(complete.read_text())
            if result['protocol'] != protocol:
                raise ValueError('Completed test has different settings')
            for path, digest in result['output_sha256'].items():
                if file_sha256(path) != digest:
                    raise ValueError('Completed test result changed: ' + path)
            return result
        if not (baseline / 'metrics/summary.json').exists():
            command = gpu_command(['scripts/evaluate.py', '--checkpoint', selected['checkpoint'],
                                   '--split', 'test', '--outdir', str(baseline), '--batch-size', str(selected['batch_size']),
                                   '--guidance', str(selected['guidance']), '--cfg-min', str(selected['cfg_min']),
                                   '--cfg-max', str(selected['cfg_max']), '--clamp', '--de'], gpu_workers)
            with (destination / 'baseline_evaluation.log').open('a') as log:
                child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(destination / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                            'stage': 'locked_full_baseline_test_generation', 'child_pid': child.pid,
                            'command': command, 'zero_floor_selected_before_test': floor})
                code = child.wait()
            if code:
                raise RuntimeError(f'Full baseline test stopped with{code}; batches and frozen settings retained')
        if json.loads((baseline / 'evaluation_config.json').read_text()) != expected_evaluation:
            raise ValueError('Baseline test does not match the pre-test frozen protocol')
        if json.loads((baseline / 'generation_state.json').read_text()) != {
                'completed': len(indices), 'total': len(indices), 'options': expected_options}:
            raise ValueError('Baseline test generation is incomplete or mismatched')
        np.testing.assert_array_equal(np.load(baseline / 'indices.npy'), indices)
        frame = pd.read_csv(baseline / 'metrics/results.csv')
        expected_conditions = {data.vocab['all_genes'][int(p)] for p in np.unique(data.meta['pert'][indices])}
        if not frame['perturbation'].is_unique or set(frame['perturbation']) != expected_conditions:
            raise ValueError('Baseline metrics do not contain every test perturbation')
        baseline_metrics = json.loads((baseline / 'metrics/summary.json').read_text())
        if set(baseline_metrics) != set(METRICS.values()):
            raise ValueError('Baseline test metrics are incomplete')
        np.testing.assert_allclose([float(frame[key].mean()) for key in METRICS],
                                   [baseline_metrics[display] for display in METRICS.values()], rtol=0, atol=1e-10)
        lines = np.unique(data.meta['cell'][indices])
        if len(lines) != 1:
            raise ValueError('This formal benchmark test must contain one cell line')
        controls = data.control[data.meta['cell'][data.control] == lines[0]]
        counts = json.loads((baseline / 'metrics/population_counts.json').read_text())
        if counts != {'real_perturbed_cells': len(indices), 'generated_cells': len(indices),
                      'shared_real_controls': len(controls)}:
            raise ValueError('Baseline test population differs from the complete split')
        raw = np.load(baseline / 'prediction_raw.npy', mmap_mode='r')
        clipped = np.load(baseline / 'prediction.npy', mmap_mode='r')
        if raw.dtype != np.float32 or raw.shape != (len(indices), data.x.shape[1]) or clipped.shape != raw.shape:
            raise ValueError('Baseline test array schema differs')
        for offset in range(0, len(raw), 4096):
            chunk = np.asarray(raw[offset:offset + 4096])
            np.testing.assert_array_equal(apply_zero_floor(chunk, 0), clipped[offset:offset + len(chunk)])
        baseline_hashes = {name: file_sha256(baseline / name) for name in
                          ('prediction.npy', 'prediction_raw.npy', 'indices.npy', 'generation_state.json',
                           'evaluation_config.json', 'metrics/summary.json', 'metrics/results.csv',
                           'metrics/population_counts.json')}
        baseline_record = destination / 'baseline_completed.json'
        if baseline_record.exists() and json.loads(baseline_record.read_text()) != baseline_hashes:
            raise ValueError('Completed baseline samples changed during test resumption')
        atomic_json(baseline_record, baseline_hashes)
        helpers = runpy.run_path(str(Path(__file__).with_name('promote_validation_cache.py')))
        helpers['copy_checked'](baseline / 'prediction_raw.npy', destination / 'prediction_raw.npy', baseline_hashes['prediction_raw.npy'])
        helpers['copy_checked'](baseline / 'indices.npy', destination / 'indices.npy', baseline_hashes['indices.npy'])
        temporary = destination / 'prediction.npy.tmp'
        processed = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=raw.shape)
        for offset in range(0, len(raw), 4096):
            chunk = np.asarray(raw[offset:offset + 4096])
            processed[offset:offset + len(chunk)] = apply_zero_floor(chunk, floor)
        processed.flush()
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        del processed
        temporary.replace(destination / 'prediction.npy')
        evaluation = {**expected_evaluation, 'zero_floor': floor, 'baseline_evaluation_dir': str(baseline),
                      'postprocess_selection_sha256': protocol['selection_sha256']}
        atomic_json(destination / 'evaluation_config.json', evaluation)
        options = {**expected_options}
        if floor > 0:
            options['zero_floor'] = floor
        atomic_json(destination / 'generation_state.json', {'completed': len(indices), 'total': len(indices), 'options': options})
        prediction = np.load(destination / 'prediction.npy', mmap_mode='r')
        metrics = official_metrics(data, indices, prediction, destination / 'metrics', de=True)
        raw_metrics = official_metrics(data, indices, raw, destination / 'metrics_raw', de=False)
        if set(metrics) != set(METRICS.values()):
            raise ValueError('Test did not produce all nine official metrics')
        files = [destination / name for name in ('prediction.npy', 'prediction_raw.npy', 'indices.npy',
                 'evaluation_config.json', 'generation_state.json', 'metrics/summary.json', 'metrics/results.csv',
                 'metrics/population_counts.json', 'metrics_raw/summary.json', 'metrics_raw/results.csv',
                 'metrics_raw/population_counts.json')]
        files.extend(baseline / name for name in baseline_hashes)
        result = {'time': timestamp(), 'protocol': protocol, 'metrics': metrics, 'raw_metrics': raw_metrics,
                  'baseline_metrics': json.loads((baseline / 'metrics/summary.json').read_text()),
                  'new_postprocess_generated_samples': 0, 'selection_changed_after_test': False,
                  'output_sha256': {str(path): file_sha256(path) for path in files}}
        atomic_json(complete, result)
        atomic_json(destination / 'process.json', {'time': timestamp(), 'stage': 'completed_review_required',
                    'metrics': metrics, 'selection_changed_after_test': False})
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True)
    parser.add_argument('--validation-dir', required=True)
    parser.add_argument('--baseline-outdir', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--gpu-workers', type=int, default=1)
    args = parser.parse_args()
    if args.gpu_workers < 1:
        parser.error('gpu-workers must be positive')
    torch.set_num_threads(8)
    print(json.dumps(run_test(args.selection, args.validation_dir, args.baseline_outdir, args.outdir, args.gpu_workers),
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
