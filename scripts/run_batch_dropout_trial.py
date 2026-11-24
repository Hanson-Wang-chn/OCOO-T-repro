"""A registered, validation-only continuation from a mature complete checkpoint.

The main trainer is untouched. A trial-local AdamW subclass requests the
existing safe-save signal after exactly the registered number of updates.
The original epoch count, learning-rate schedule, optimizer and EMA rules stay
unchanged. This is an interpretation trial, not a completed full training run.
"""
import argparse
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import numpy as np
import torch
import yaml

from ocoot.model import ModelConfig
from ocoot.runtime import atomic_save, exclusive_run, file_sha256, readme_event, timestamp


def atomic_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def assert_tree_equal(first, second):
    if isinstance(first, torch.Tensor):
        if first.dtype != second.dtype or not torch.equal(first, second):
            raise ValueError('Checkpoint tensor differs')
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    elif isinstance(first, dict):
        if first.keys() != second.keys():
            raise ValueError('Checkpoint keys differ')
        for key in first:
            assert_tree_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        if type(first) is not type(second) or len(first) != len(second):
            raise ValueError('Checkpoint sequence differs')
        for a, b in zip(first, second):
            assert_tree_equal(a, b)
    elif first != second:
        raise ValueError('Checkpoint metadata differs')


def configs_match(first, second):
    normalized = [{**cfg, 'model': asdict(ModelConfig(**cfg['model']))} for cfg in (first, second)]
    if normalized[0]['model'].pop('drop_batch_with_cell') is not True:
        raise ValueError('Control must use original combined dropout')
    if normalized[1]['model'].pop('drop_batch_with_cell') is not False:
        raise ValueError('Candidate must retain the batch embedding')
    for cfg in normalized:
        cfg.pop('run_dir')
    if normalized[0] != normalized[1]:
        raise ValueError('Configurations differ beyond batch dropout and run directory')


def bounded_adamw_class(original, updates, callback):
    if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
        raise ValueError('A positive integer update budget is required')

    class BoundedAdamW(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.completed_updates = 0

        def step(self, *args, **kwargs):
            if self.completed_updates >= updates:
                raise RuntimeError('Refusing an optimizer update beyond the registered budget')
            result = super().step(*args, **kwargs)
            self.completed_updates += 1
            if self.completed_updates == updates:
                callback()
            return result

    return BoundedAdamW


def request_safe_stop():
    handler = signal.getsignal(signal.SIGTERM)
    if not callable(handler) or handler.__name__ != 'on_signal':
        raise RuntimeError('Trainer safe-save signal handler is not installed')
    os.kill(os.getpid(), signal.SIGTERM)


def prepare(plan, out):
    source = Path(plan['base_checkpoint'])
    reference = Path(plan['reference_checkpoint'])
    if (file_sha256(source) != plan['base_checkpoint_sha256']
            or file_sha256(reference) != plan['reference_checkpoint_sha256']):
        raise ValueError('Registered mature checkpoint changed')
    for name, digest in plan['original_source_sha256'].items():
        path = Path(name) if name != 'src/ocoot/model.py' else Path(
            'logs/source_archive/model_before_batch_retention_1f83d4d27de5.py')
        if file_sha256(path) != digest:
            raise ValueError(f'Original implementation changed: {path}')
    if file_sha256('configs/main_selected.yaml') != plan['main_config_sha256']:
        raise ValueError('Main training configuration changed')
    base = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if base['step'] != plan['base_step'] or plan['reference_step'] - base['step'] != plan['updates']:
        raise ValueError('Wrong continuation budget')
    if (base['config']['training'].get('train_subset')
            or set(base['ema']) != {'0.9999', '0.9996'} or not base['optimizer']['state']):
        raise ValueError('Requires a populated full-data complete checkpoint')
    cfg = copy.deepcopy(base['config'])
    cfg['run_dir'] = plan['candidate_run']
    cfg['model']['drop_batch_with_cell'] = False
    configs_match(base['config'], cfg)
    run = Path(cfg['run_dir'])
    run.mkdir(parents=True, exist_ok=True)
    config_path = Path(plan['candidate_config'])
    if config_path.exists() and yaml.safe_load(config_path.read_text()) != cfg:
        raise ValueError('Candidate YAML belongs to another experiment')
    config_path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    checkpoint_path = run / 'last.pt'
    provenance = out / 'fork_provenance.json'
    if not checkpoint_path.exists():
        # Epoch25 generation was already completed by the original run. Its RNG
        # isolation restores this same stored training RNG before epoch26.
        if base.get('pending_generation_epoch') not in (None, 25):
            raise ValueError('Unexpected pending work at the source checkpoint')
        fork = {**base, 'config': cfg, 'pending_generation_epoch': None}
        atomic_save(fork, checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
    if saved['config'] != cfg or not base['step'] <= saved['step'] <= plan['stop_after_step']:
        raise ValueError('Candidate checkpoint has a different config or step')
    if not provenance.exists():
        if saved['step'] != base['step']:
            raise ValueError('Missing fork provenance for an advanced candidate')
        for key in base:
            if key not in ('config', 'pending_generation_epoch'):
                assert_tree_equal(base[key], saved[key])
        if saved['pending_generation_epoch'] is not None:
            raise ValueError('Pending source evaluation must be explicitly cleared')
        atomic_json(provenance, {'time': timestamp(), 'base_checkpoint_sha256': plan['base_checkpoint_sha256'],
                                'fork_initial_checkpoint_sha256': file_sha256(checkpoint_path),
                                'base_step': base['step'], 'candidate_config': cfg,
                                'all_model_optimizer_dualEMA_rng_and_other_metadata_equal': True,
                                'only_changes': ['config.run_dir', 'config.model.drop_batch_with_cell',
                                                 'pending_generation_epoch'],
                                'original_main_changed': False})
    elif json.loads(provenance.read_text())['candidate_config'] != cfg:
        raise ValueError('Fork provenance configuration mismatch')
    return cfg, saved['step']


def export_ema(source, target, expected_step):
    state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
    if state['step'] != expected_step:
        raise ValueError('Unequal endpoint optimization budgets')
    ema = state['ema']['0.9996']
    if not ema or not set(ema).issubset(state['model']):
        raise ValueError('Invalid default EMA state')
    model = {key: ema[key] if key in ema else value for key, value in state['model'].items()}
    weights = {'config': state['config'], 'model': model, 'step': state['step'],
               'weights': 'ema_0.9996', 'criterion': 'matched_mature_batch_dropout_diagnostic',
               'source_checkpoint_sha256': file_sha256(source)}
    if target.exists():
        existing = torch.load(target, map_location='cpu', mmap=True, weights_only=False)
        assert_tree_equal(weights, existing)
    else:
        atomic_save(weights, target)
    return {'source_checkpoint_sha256': weights['source_checkpoint_sha256'],
            'export_sha256': file_sha256(target), 'step': state['step'], 'weights': weights['weights']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--prepare-only', action='store_true')
    mode.add_argument('--train-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    plan_path = Path(args.plan)
    out = plan_path.parent
    plan = json.loads(plan_path.read_text())
    if args.train_only:
        cfg, step = prepare(plan, out)
        if step == plan['stop_after_step']:
            return
        original = torch.optim.AdamW
        torch.optim.AdamW = bounded_adamw_class(original, plan['stop_after_step'] - step, request_safe_stop)
        try:
            from ocoot.train import train
            train(cfg, resume=True)
        finally:
            torch.optim.AdamW = original
        saved = torch.load(Path(cfg['run_dir']) / 'last.pt', map_location='cpu', mmap=True, weights_only=False)
        if saved['step'] != plan['stop_after_step'] or saved.get('pending_generation_epoch') is not None:
            raise RuntimeError('Bounded trial stopped before its exact completed update budget')
        if (Path(cfg['run_dir']) / 'finished.json').exists():
            raise RuntimeError('A bounded continuation must not claim completed full training')
        atomic_json(out / 'bounded_training_completed.json', {'time': timestamp(), 'step': saved['step'],
                    'updates_from_base': saved['step'] - plan['base_step'], 'full_state_retained': True,
                    'training_source_unchanged': True, 'main_training_changed': False})
        return
    with exclusive_run(out, 'controller'):
        cfg, step = prepare(plan, out)
        if args.prepare_only:
            print(json.dumps({'prepared_step': step, 'config': cfg, 'gpu_used': False}))
            return
        verification = json.loads((out / 'prelaunch_verification.json').read_text())
        files = ['src/ocoot/model.py', 'src/ocoot/train.py', 'scripts/run_batch_dropout_trial.py',
                 'tests/test_batch_dropout_trial.py', 'src/ocoot/flow.py', 'src/ocoot/runtime.py',
                 'scripts/evaluate.py', 'scripts/diagnose_zero_floor.py', 'scripts/analyze_zero_floor.py']
        sources = {name: file_sha256(name) for name in files}
        if not verification['passed'] or any(verification['source_sha256'].get(p) != h for p, h in sources.items()):
            raise ValueError('Prelaunch CPU verification does not cover the current sources')
        protocol = {'plan_sha256': file_sha256(plan_path), 'source_sha256': sources,
                    'verification_sha256': file_sha256(out / 'prelaunch_verification.json')}
        registered = out / 'execution_config.json'
        if registered.exists() and json.loads(registered.read_text()) != protocol:
            raise ValueError('Registered execution source or protocol changed')
        atomic_json(registered, protocol)

        def execute(command, stage, cpu=False):
            env = os.environ.copy()
            if cpu:
                env.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            with (out / f'{stage}.log').open('a') as log:
                child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                atomic_json(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(), 'stage': stage,
                                                  'child_pid': child.pid, 'command': command,
                                                  'main_training_changed': False, 'test_access': False})
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage} exited with {code}; resumable artifacts retained')

        if step < plan['stop_after_step']:
            execute([sys.executable, __file__, '--plan', str(plan_path), '--train-only'], 'training')
        checkpoints = {'reference': Path(plan['reference_checkpoint']), 'candidate': Path(cfg['run_dir']) / 'last.pt'}
        models = out / 'ema_models'
        models.mkdir(exist_ok=True)
        exports = {tag: export_ema(path, models / f'{tag}.pt', plan['stop_after_step'])
                   for tag, path in checkpoints.items()}
        atomic_json(out / 'exports.json', exports)
        from ocoot.data import BenchmarkData
        from ocoot.evaluate import official_metrics
        data = BenchmarkData(cfg['data_dir'])
        expected_indices = data.evaluation_indices('validation', 16)
        case_hashes = {}
        for tag in ('reference', 'candidate'):
            case = out / tag
            model_path = (models / f'{tag}.pt').resolve()
            if not (case / 'metrics/summary.json').exists():
                command = [sys.executable, 'scripts/evaluate.py', '--checkpoint', str(model_path),
                           '--split', 'validation', '--outdir', str(case), '--cap', '16', '--batch-size', '16',
                           '--guidance', '2', '--cfg-min', '.1', '--cfg-max', '.9', '--clamp', '--de']
                execute(command, f'evaluate_{tag}')
            evaluation = json.loads((case / 'evaluation_config.json').read_text())
            expected = {'checkpoint_sha256': exports[tag]['export_sha256'], 'step': plan['stop_after_step'],
                        'weights': 'ema_0.9996', **{key: plan['evaluation'][key] for key in
                        ('split', 'cap', 'batch_size', 'seed', 'guidance', 'cfg_min', 'cfg_max', 'steps', 'clamp', 'de')}}
            if any(evaluation.get(key) != value for key, value in expected.items()):
                raise ValueError(f'Completed evaluation provenance differs: {tag}')
            indices = np.load(case / 'indices.npy')
            np.testing.assert_array_equal(indices, expected_indices)
            marker = json.loads((case / 'generation_state.json').read_text())
            if marker['completed'] != len(indices) or marker['total'] != len(indices):
                raise ValueError('Evaluation generation is incomplete')
            raw = np.load(case / 'prediction_raw.npy', mmap_mode='r')
            clipped = np.load(case / 'prediction.npy', mmap_mode='r')
            if not np.isfinite(raw).all():
                raise ValueError('Nonfinite cached predictions')
            np.testing.assert_array_equal(clipped, np.maximum(raw, 0))
            official_metrics(data, indices, raw, case / 'metrics_raw')
            link = case / 'model.pt'
            if not link.exists():
                link.symlink_to(model_path)
            elif link.resolve() != model_path:
                raise ValueError('Evaluation model link differs')
            execute([sys.executable, 'scripts/diagnose_zero_floor.py', '--generation-dir', str(case),
                     '--outdir', str(case / 'zero_floor'), '--floors', '0,.05'], f'floor_{tag}', cpu=True)
            execute([sys.executable, 'scripts/analyze_zero_floor.py', '--study-dir', str(case / 'zero_floor')],
                    f'floor_analysis_{tag}', cpu=True)
            case_hashes[tag] = {name: file_sha256(case / name) for name in
                               ('indices.npy', 'prediction.npy', 'prediction_raw.npy', 'generation_state.json',
                                'evaluation_config.json', 'metrics/summary.json', 'metrics_raw/summary.json',
                                'zero_floor/comparison.json')}
        atomic_json(out / 'completed.json', {'time': timestamp(), 'stage': 'completed_review_required',
                    'step': plan['stop_after_step'], 'exports': exports, 'case_sha256': case_hashes,
                    'automatic_main_switch': False,
                    'formal_selection_changed': False, 'test_access': False})
        atomic_json(out / 'process.json', {'time': timestamp(), 'pid': os.getpid(),
                    'stage': 'completed_review_required', 'main_training_changed': False, 'test_access': False})
        readme_event(f'成熟batch dropout单因素短续训及两组固定CFG九指标完成，均为step{plan["stop_after_step"]}；'
                     f'待配对复核，不自动更改主模型；结果 `{out}/completed.json`。')


if __name__ == '__main__':
    main()
