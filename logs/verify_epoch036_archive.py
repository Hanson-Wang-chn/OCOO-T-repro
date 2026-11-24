"""CPU audit of the registered post-E35-evaluation E36 recovery archive."""
import json
from pathlib import Path

import torch
import yaml

from ocoot.runtime import file_sha256, timestamp


def main():
    torch.set_num_threads(1)
    out = Path('logs/preservation_epoch036')
    plan = json.loads((out / 'verification_plan.json').read_text())
    for path, digest in plan['source_sha256'].items():
        if file_sha256(path) != digest:
            raise ValueError('Registered archive verification input changed: ' + path)
    if (out / 'verification.json').exists():
        raise FileExistsError('Inspect the existing independent archive proof')
    record = json.loads(Path('runs/main_v1/snapshots/step_40248/preserved.json').read_text())
    path = Path(record['archive_path'])
    assert path == Path('/home/dataset-assist-0/whs/ocoo-t-artifacts/main-snapshots/epoch036/full.pt')
    assert record['step'] == 40248 and record['bytes'] == path.stat().st_size
    assert record['script_sha256'] == file_sha256('scripts/preserve_checkpoint.py')
    assert record['pending_generation_epoch'] is None
    assert file_sha256(path) == record['sha256']
    state = torch.load(path, mmap=True, map_location='cpu', weights_only=False)
    assert state['step'] == 40248 == 36 * 1118
    assert state['config'] == yaml.safe_load(Path('configs/main_selected.yaml').read_text())
    assert state['pending_generation_epoch'] is None
    bridge = json.loads(Path('logs/epoch035_periodic_selection_metadata.json').read_text())['metadata']
    for name in ('best_generation', 'best_generation_epoch', 'early_reference', 'early_reference_epoch'):
        assert state[name] == bridge[name], name
    assert state['best_generation_epoch'] == state['early_reference_epoch'] == 35
    assert len(state['model']) == 193 and set(state['ema']) == {'0.9999', '0.9996'}
    parameter_names = list(state['ema']['0.9996'])
    assert len(parameter_names) == len(set(parameter_names)) == 189
    for decay, parameters in state['ema'].items():
        assert list(parameters) == parameter_names
        for name, value in parameters.items():
            assert value.shape == state['model'][name].shape
            assert torch.isfinite(value).all(), (decay, name)
    for name, value in state['model'].items():
        if value.is_floating_point():
            assert torch.isfinite(value).all(), name
    optimizer = state['optimizer']
    assert len(optimizer['param_groups']) == 1
    indices = optimizer['param_groups'][0]['params']
    assert len(indices) == len(set(indices)) == len(optimizer['state']) == 189
    for index, name in zip(indices, parameter_names):
        values = optimizer['state'][index]
        assert int(values['step'].item()) == 40248
        for key in ('exp_avg', 'exp_avg_sq'):
            assert values[key].shape == state['model'][name].shape
            assert torch.isfinite(values[key]).all(), (name, key)
    rng = state['rng']
    assert set(rng) == {'python', 'numpy', 'torch', 'cuda'}
    assert isinstance(rng['python'], tuple) and len(rng['python']) == 3
    assert isinstance(rng['numpy'], tuple) and len(rng['numpy']) == 5
    assert rng['torch'].dtype == torch.uint8 and rng['torch'].numel() > 0
    assert len(rng['cuda']) == 1 and all(v.dtype == torch.uint8 and v.numel() > 0 for v in rng['cuda'])
    proof = {'time': timestamp(), 'passed': True, 'step': 40248, 'cpu_only': True,
             'full_checkpoint_sha256': record['sha256'], 'bytes': record['bytes'],
             'raw_model_and_both_ema_finite': True, 'all189_optimizer_slots_step40248_and_finite': True,
             'optimizer_parameter_and_ema_shapes_verified': True,
             'original_full_configuration_equal': True,
             'pending_generation_epoch': None, 'best_generation_epoch': 35, 'early_reference_epoch': 35,
             'post_epoch35_selection_metadata_equal': True, 'all_four_rng_state_types_retained': True,
             'resume_metadata_schema_verified': True, 'resumed_or_changed_training': False,
             'test_access': False, 'plan_sha256': file_sha256(out / 'verification_plan.json'),
             'source_sha256': plan['source_sha256']}
    target = out / 'verification.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(proof, indent=2))
    temporary.replace(target)
    print(json.dumps(proof))


if __name__ == '__main__':
    main()
