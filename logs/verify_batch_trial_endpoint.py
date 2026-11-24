"""Independent CPU-only audit of the registered mature trial endpoint."""
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    torch.set_num_threads(1)
    root = Path('runs/batch_dropout_mature_trial')
    plan = json.loads((root / 'plan.json').read_text())
    execution = json.loads((root / 'execution_config.json').read_text())
    completion = json.loads((root / 'bounded_training_completed.json').read_text())
    exports = json.loads((root / 'exports.json').read_text())
    assert execution['plan_sha256'] == digest(root / 'plan.json')
    assert all(digest(p) == h for p, h in execution['source_sha256'].items())
    assert digest('configs/main_selected.yaml') == plan['main_config_sha256']
    assert completion['step'] == plan['stop_after_step'] == 29068
    assert completion['updates_from_base'] == plan['updates'] == 1118
    assert not (Path(plan['candidate_run']) / 'finished.json').exists()
    paths = {'reference': Path(plan['reference_checkpoint']),
             'candidate': Path(plan['candidate_run']) / 'last.pt'}
    configs, audit = {}, {}
    frozen_buffers = None
    for tag, source in paths.items():
        state = torch.load(source, map_location='cpu', mmap=True, weights_only=False)
        export_path = root / 'ema_models' / f'{tag}.pt'
        export = torch.load(export_path, map_location='cpu', mmap=True, weights_only=False)
        assert state['step'] == export['step'] == plan['stop_after_step']
        assert state.get('pending_generation_epoch') is None
        assert set(state['ema']) == {'0.9999', '0.9996'}
        assert state['rng']
        slots = state['optimizer']['state']
        assert len(slots) == len(state['ema']['0.9996']) == 189
        assert all(int(slot['step'].item()) == plan['stop_after_step'] for slot in slots.values())
        assert all('exp_avg' in slot and 'exp_avg_sq' in slot for slot in slots.values())
        assert len(state['model']) == len(export['model']) == 193
        assert export['weights'] == exports[tag]['weights'] == 'ema_0.9996'
        source_hash, export_hash = digest(source), digest(export_path)
        assert source_hash == exports[tag]['source_checkpoint_sha256'] == export['source_checkpoint_sha256']
        assert export_hash == exports[tag]['export_sha256']
        assert state['model'].keys() == export['model'].keys()
        for key, raw in state['model'].items():
            expected = state['ema']['0.9996'].get(key, raw)
            assert torch.equal(expected, export['model'][key]), (tag, key)
        buffers = {key: value for key, value in export['model'].items()
                   if key not in state['ema']['0.9996']}
        assert len(buffers) == 4
        if frozen_buffers is None:
            frozen_buffers = buffers
        else:
            assert buffers.keys() == frozen_buffers.keys()
            assert all(torch.equal(value, frozen_buffers[key]) for key, value in buffers.items())
        configs[tag] = state['config']
        audit[tag] = {'checkpoint': str(source), 'checkpoint_sha256': source_hash,
                      'export_sha256': export_hash, 'step': state['step'],
                      'optimizer_parameters': len(slots), 'export_tensors': len(export['model']),
                      'all_export_tensors_equal_default_ema_and_buffers': True,
                      'dual_ema_rng_and_adamw_slots_retained': True}
        del state, export
    from dataclasses import asdict
    from ocoot.model import ModelConfig
    normalized = []
    for tag in ('reference', 'candidate'):
        cfg = {**configs[tag], 'model': asdict(ModelConfig(**configs[tag]['model']))}
        expected = tag == 'reference'
        assert cfg['model'].pop('drop_batch_with_cell') is expected
        cfg.pop('run_dir')
        normalized.append(cfg)
    assert normalized[0] == normalized[1]
    result = {'time': datetime.now().astimezone().isoformat(timespec='seconds'),
              'passed': True, 'cpu_only': True, 'test_access': False,
              'plan_and_frozen_sources_unchanged': True,
              'exact_budget_updates': 1118, 'full_training_claimed': False,
              'only_config_difference_beyond_run_dir': 'model.drop_batch_with_cell',
              'four_frozen_buffers_equal_between_endpoints': True, 'endpoints': audit}
    path = root / 'endpoint_independent_verification.json'
    assert not path.exists(), 'Keep completed audits immutable'
    temporary = path.with_suffix('.json.tmp')
    with temporary.open('w') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
