"""Measure the four conditional velocity fields on paired HepG2 samples."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.runtime import exclusive_run, file_sha256, timestamp


BRANCHES = ('vu', 'vc', 'vp', 'vpc')
CELL_SEED = 1738
NOISE_SEED = 9821


@torch.inference_mode()
def velocity_mse(model, data, indices, noise, times, batch_size):
    x, pert, cell, batch = data.batch(indices)
    target = x - noise
    errors = torch.zeros((len(times), len(BRANCHES)), dtype=torch.float64, device=x.device)
    for start in range(0, len(indices), batch_size):
        rows = slice(start, start + batch_size)
        n = len(x[rows])
        on = torch.ones(n, dtype=torch.bool, device=x.device)
        off = ~on
        drop_pert = torch.cat((on, on, off, off))
        drop_cell = torch.cat((on, off, on, off))
        labels = [values[rows].repeat(4) for values in (pert, cell, batch)]
        for i, t in enumerate(times):
            z = t * x[rows] + (1 - t) * noise[rows]
            time = torch.full((4 * n,), t, device=x.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                prediction = model(z.repeat(4, 1), time, *labels, drop_pert, drop_cell)
            squared = (prediction.float().reshape(4, n, -1) - target[rows]).square()
            errors[i] += squared.sum(dim=(1, 2), dtype=torch.float64)
    return (errors / x.numel()).tolist()


def summarize(times, train, validation):
    rows = []
    for t, train_errors, val_errors in zip(times, train, validation, strict=True):
        train_mse = dict(zip(BRANCHES, train_errors, strict=True))
        val_mse = dict(zip(BRANCHES, val_errors, strict=True))
        gain_train = train_mse['vc'] - train_mse['vpc']
        gain_val = val_mse['vc'] - val_mse['vpc']
        rows.append({'t': t, 'train_mse': train_mse, 'validation_mse': val_mse,
                     'gain_train': gain_train, 'gain_val': gain_val,
                     'overfit_ratio': gain_train / max(gain_val, 1e-6), 'vu_mse_val': val_mse['vu']})
    cutoff = next((row['t'] for row in rows if row['gain_val'] <= 0), None)
    return rows, cutoff


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--outdir', type=Path, required=True)
    parser.add_argument('--n-cells', type=int, default=512)
    parser.add_argument('--timesteps', default='0.05,0.25,0.50,0.75,0.90,0.96')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    try:
        times = [float(value) for value in args.timesteps.split(',')]
    except ValueError:
        parser.error('timesteps must be comma-separated numbers')
    if times != sorted(set(times)) or any(not 0 <= t <= 1 for t in times):
        parser.error('timesteps must be strictly increasing in [0,1]')
    if args.n_cells <= 0 or args.batch_size <= 0:
        parser.error('n-cells and batch-size must be positive')

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    with exclusive_run(args.outdir, 'velocity_probe'):
        checkpoint = torch.load(args.checkpoint, map_location='cpu', mmap=True, weights_only=False)
        cfg = checkpoint['config']
        weights = 'ema_0.9996' if '0.9996' in checkpoint.get('ema', {}) else checkpoint.get('weights')
        if weights != 'ema_0.9996':
            raise ValueError('The velocity probe requires default EMA weights')
        data = BenchmarkData(cfg['data_dir'])
        line = next(i for i, name in enumerate(data.vocab['cell_lines']) if name.lower() == 'hepg2')
        populations = [split[data.meta['cell'][split] == line] for split in (data.train, data.val)]
        if min(map(len, populations)) < args.n_cells:
            raise ValueError('Not enough HepG2 training or validation cells for the requested probe')
        rng = np.random.default_rng(CELL_SEED)
        train, val = [np.sort(rng.choice(split, args.n_cells, replace=False)) for split in populations]
        report_path = args.outdir / 'summary.json'
        protocol = {'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': file_sha256(args.checkpoint),
                    'step': checkpoint['step'], 'weights': weights, 'cells_each': args.n_cells,
                    'times': times, 'cell_line': 'HepG2', 'cell_seed': CELL_SEED, 'noise_seed': NOISE_SEED,
                    'noise_std': cfg['training']['noise_std'], 'batch_size': args.batch_size, 'precision': 'BF16',
                    'metadata_sha256': file_sha256(data.path / 'metadata.npz'),
                    'source_sha256': {str(path): file_sha256(path) for path in
                                      (Path(__file__), Path('src/ocoot/model.py'), Path('src/ocoot/data.py'))},
                    'test_access': False}
        if report_path.exists():
            existing = json.loads(report_path.read_text())
            if existing['protocol'] != protocol or existing['indices_sha256'] != file_sha256(args.outdir / 'indices.npz'):
                raise ValueError('Probe output belongs to a different checkpoint or protocol')
            return

        model = data.build_model(cfg['model'])
        model.load_state_dict(checkpoint['model'])
        if 'ema' in checkpoint:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    parameter.copy_(checkpoint['ema']['0.9996'][name])
        model = model.cuda().eval()
        del checkpoint
        generator = torch.Generator(device='cuda').manual_seed(NOISE_SEED)
        noise = torch.randn((args.n_cells, data.x.shape[1]), device='cuda', generator=generator) * protocol['noise_std']
        errors_train = velocity_mse(model, data, train, noise, times, args.batch_size)
        errors_val = velocity_mse(model, data, val, noise, times, args.batch_size)
        rows, cutoff = summarize(times, errors_train, errors_val)

        indices = args.outdir / 'indices.npz'
        np.savez(indices, train_hepg2=train, validation_hepg2=val)
        report = {'time': timestamp(), 'protocol': protocol, 'rows': rows, 'val_t_cutoff': cutoff,
                  'indices_sha256': file_sha256(indices)}
        temporary = report_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        temporary.replace(report_path)
        print(json.dumps(report, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
