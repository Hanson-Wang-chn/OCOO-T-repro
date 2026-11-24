"""CPU-only integrity check of the explicitly stopped migration checkpoint."""
import argparse
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''

import torch
import yaml

from ocoot.runtime import file_sha256, timestamp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', default='logs/server_shutdown_20261121/migration_state.json')
    parser.add_argument('--report', default='logs/migration_preflight.json')
    args = parser.parse_args()
    expected = json.loads(Path(args.manifest).read_text())
    path = Path(expected['checkpoint_relative'])
    torch.set_num_threads(1)
    if path.stat().st_size != expected['checkpoint_bytes']:
        raise ValueError('Full checkpoint byte count differs; do not start training')
    digest = file_sha256(path)
    if digest != expected['checkpoint_sha256']:
        raise ValueError('Full checkpoint SHA differs; this preflight is for stopped step41374')
    if file_sha256(expected['snapshot_relative']) != digest:
        raise ValueError('External redundant checkpoint differs')
    config_path = Path('configs/main_selected.yaml')
    if file_sha256(config_path) != expected['config_sha256']:
        raise ValueError('Original training config changed')
    config = yaml.safe_load(config_path.read_text())
    state = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    if state['step'] != 41374 or state['config'] != config:
        raise ValueError('Stopped checkpoint step/config mismatch')
    first, second = state['ema']['0.9996'], state['ema']['0.9999']
    parameters = list(first)
    model = state['model']
    if (len(model) != 193 or len(first) != 189 or set(first) != set(second)
            or set(model) - set(first) != {'esm', 'gene_indices', 'missing', 'position'}):
        raise ValueError('Model/dual EMA schema mismatch')
    for name, value in model.items():
        if not torch.isfinite(value).all():
            raise ValueError('Nonfinite raw model tensor: ' + name)
    for ema in (first, second):
        for name in parameters:
            if (ema[name].shape != model[name].shape or ema[name].dtype != model[name].dtype
                    or not torch.isfinite(ema[name]).all()):
                raise ValueError('EMA shape/type/finiteness mismatch: ' + name)
    optimizer = state['optimizer']
    ids = [key for group in optimizer['param_groups'] for key in group['params']]
    if len(ids) != 189 or len(set(ids)) != 189 or len(optimizer['state']) != 189:
        raise ValueError('Optimizer slots are incomplete')
    for name, key in zip(parameters, ids):
        slot = optimizer['state'][key]
        if int(slot['step'].item()) != 41374:
            raise ValueError('Optimizer step mismatch: ' + name)
        for field in ('exp_avg', 'exp_avg_sq'):
            if slot[field].shape != model[name].shape or not torch.isfinite(slot[field]).all():
                raise ValueError('Invalid optimizer moment: ' + name)
    if set(state['rng']) != {'python', 'numpy', 'torch', 'cuda'}:
        raise ValueError('Incomplete RNG state')
    if (not state['rng']['cuda'] or state['rng']['torch'].dtype != torch.uint8
            or any(value.dtype != torch.uint8 or not value.numel() for value in state['rng']['cuda'])):
        raise ValueError('Invalid CPU/CUDA RNG representation')
    if (state.get('pending_generation_epoch') is not None
            or state['best_generation_epoch'] != 35 or state['early_reference_epoch'] != 35):
        raise ValueError('Post-generation selection/recovery metadata differs')
    expression = Path(config['data_dir']) / 'expression.npy'
    if file_sha256(expression) != expected['data_expression_sha256']:
        raise ValueError('Processed expression SHA differs')
    broken, count = [], 0
    for base, directories, files in os.walk('.', followlinks=False):
        directories[:] = [d for d in directories if d not in ('.git', '__pycache__', '.pytest_cache')]
        for name in [*directories, *files]:
            link = Path(base) / name
            if link.is_symlink():
                count += 1
                if not link.exists():
                    broken.append(str(link))
    if broken:
        raise FileNotFoundError('Missing external artifacts; restore/relocate links: ' + json.dumps(broken))
    report = {
        'time': timestamp(), 'status': 'passed', 'cpu_only': True,
        'checkpoint_sha256': digest, 'checkpoint_bytes': path.stat().st_size,
        'step': state['step'], 'epoch': state['step'] / 1118,
        'raw_tensors_finite': 193, 'each_EMA_parameter_tensors_finite': 189,
        'AdamW_slots_at_step41374': 189, 'four_RNG_types_present': True,
        'CUDA_RNG_states': len(state['rng']['cuda']),
        'best_generation_epoch': 35, 'early_reference_epoch': 35,
        'pending_generation_epoch': None, 'config_exact': True,
        'expression_SHA_verified': True, 'all_symlinks_resolve': count,
        'torch_version': torch.__version__, 'torch_CUDA_build': torch.version.cuda,
        'GPU_initialized': torch.cuda.is_initialized(),
        'note': 'Byte/schema/state verification only; no actual GPU optimizer-resume test.',
    }
    out = Path(args.report)
    out.parent.mkdir(exist_ok=True, parents=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
