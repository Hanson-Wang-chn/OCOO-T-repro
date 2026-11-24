"""Paired condition interventions on a frozen full-training checkpoint.

This measures denoising dependence, not generation quality. No test cells are
used. The running trainer, its optimizer, and its RNG are never accessed.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from ocoot.data import BenchmarkData
from ocoot.flow import flow_batch
from ocoot.runtime import exclusive_run, file_sha256, log_event, timestamp


CASES = ('matched', 'shuffled_pert', 'null_pert', 'shuffled_time')


def atomic_json(value, path):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def atomic_npz(path, **arrays):
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def summarize(arrays, bootstrap, seed):
    losses = arrays['velocity_losses'].astype(np.float64)
    endpoint = losses * np.square(1 - arrays['t'].astype(np.float64))[:, None]
    _, inverse = np.unique(arrays['pert'], return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(seed).integers(0, len(counts), (bootstrap, len(counts)))

    def paired_delta(values):
        delta = values - values[:, 0, None]
        totals = np.stack([np.bincount(inverse, weights=d, minlength=len(counts)) for d in delta.T], axis=1)
        resampled = totals[draws].sum(1) / counts[draws].sum(1)[:, None]
        return delta.mean(0), np.quantile(resampled, [.025, .975], axis=0)

    velocity_delta, velocity_ci = paired_delta(losses)
    endpoint_delta, endpoint_ci = paired_delta(endpoint)
    return {
        'cells': len(losses), 'perturbations': len(counts),
        'changed_perturbation_fraction': float(np.mean(arrays['pert'] != arrays['shuffled_pert'])),
        'changed_time_fraction': float(np.mean(arrays['t'] != arrays['shuffled_t'])),
        'cases': {
            case: {
                'velocity_mse': float(losses[:, i].mean()),
                'endpoint_mse': float(endpoint[:, i].mean()),
                'delta_velocity_mse': float(velocity_delta[i]),
                'delta_velocity_mse_cluster_ci95': velocity_ci[:, i].tolist(),
                'delta_endpoint_mse': float(endpoint_delta[i]),
                'delta_endpoint_mse_cluster_ci95': endpoint_ci[:, i].tolist(),
                'velocity_response_rms': float(np.sqrt(arrays['response_mse'][:, i].mean(dtype=np.float64))),
            } for i, case in enumerate(CASES)
        },
    }


@torch.no_grad()
def probe(forward, data, indices, batch_size, seed, progress):
    x, pert, cell, batch = data.batch(indices)
    generator = torch.Generator(device=x.device).manual_seed(seed)
    z, t, target = flow_batch(x, generator=generator, **progress['flow'])
    rng = np.random.default_rng(seed + 1)
    wrong_pert = pert.cpu().numpy().copy()
    batch_ids = batch.cpu().numpy()
    for batch_id in np.unique(batch_ids):
        positions = np.flatnonzero(batch_ids == batch_id)
        wrong_pert[positions] = wrong_pert[rng.permutation(positions)]
    wrong_pert = torch.from_numpy(wrong_pert).to(x.device)
    wrong_t = t[torch.from_numpy(rng.permutation(len(t))).to(x.device)]
    losses = np.empty((len(x), len(CASES)), dtype=np.float32)
    responses = np.empty_like(losses)
    matched = torch.empty_like(target)
    for case_id, case in enumerate(CASES):
        for a in range(0, len(x), batch_size):
            sl = slice(a, min(a + batch_size, len(x)))
            off = torch.zeros(len(z[sl]), dtype=torch.bool, device=x.device)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                pred = forward(
                    z[sl], wrong_t[sl] if case == 'shuffled_time' else t[sl],
                    wrong_pert[sl] if case == 'shuffled_pert' else pert[sl],
                    cell[sl], batch[sl], ~off if case == 'null_pert' else off, off,
                ).float()
            if case == 'matched':
                matched[sl] = pred
            losses[sl, case_id] = (pred - target[sl]).square().mean(-1).cpu().numpy()
            responses[sl, case_id] = (pred - matched[sl]).square().mean(-1).cpu().numpy()
        if not np.isfinite(losses[:, case_id]).all():
            raise FloatingPointError(f'Nonfinite intervention loss: {case}')
        log_event(progress['events'], {'event': 'intervention_case', 'weights': progress['weights'],
                  'split': progress['split'], 'case': case, 'velocity_mse': float(losses[:, case_id].mean())})
    return {
        'indices': indices, 'pert': pert.cpu().numpy(), 'shuffled_pert': wrong_pert.cpu().numpy(),
        'batch': batch_ids, 't': t.cpu().numpy(), 'shuffled_t': wrong_t.cpu().numpy(),
        'velocity_losses': losses, 'response_mse': responses,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--outdir', required=True)
    parser.add_argument('--cells', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--seed', type=int, default=9231)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--no-compile', action='store_true')
    args = parser.parse_args()
    if min(args.cells, args.batch_size, args.bootstrap) < 1:
        parser.error('cells, batch-size and bootstrap must be positive')
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    with exclusive_run(out, 'conditioning_probe'):
        # Atomic hard-linking pins the inode while last.pt continues to be
        # replaced by training. Never mutate this immutable snapshot.
        snapshot = out / 'source_snapshot.pt'
        if not snapshot.exists():
            os.link(Path(args.checkpoint).resolve(), snapshot)
        checkpoint = torch.load(snapshot, map_location='cpu', mmap=True, weights_only=False)
        cfg = checkpoint['config']
        if cfg['training'].get('train_subset') or 'ema' not in checkpoint:
            raise ValueError('This probe requires a full-data training checkpoint with both raw and EMA weights')
        data = BenchmarkData(cfg['data_dir'])
        torch.set_num_threads(4)
        torch.backends.cuda.matmul.allow_tf32 = True
        protocol = {
            **vars(args), 'checkpoint': str(Path(args.checkpoint).resolve()),
            'checkpoint_sha256': file_sha256(snapshot), 'step': checkpoint['step'],
            'script_sha256': file_sha256(__file__),
            'model_sha256': file_sha256('src/ocoot/model.py'),
            'flow_sha256': file_sha256('src/ocoot/flow.py'),
            'data_code_sha256': file_sha256('src/ocoot/data.py'),
            'metadata_sha256': file_sha256(data.path / 'metadata.npz'),
            'vocab_sha256': file_sha256(data.path / 'vocab.json'),
            'cases': list(CASES), 'ema_decay': '0.9996',
            'note': 'Paired denoising interventions only; no test cells or generation metrics. '
                    'Perturbation labels are shuffled within batch and cell line. '
                    'Positive loss delta favors matched conditions. Confidence intervals bootstrap perturbations, not training seeds.',
        }
        config_file = out / 'probe_config.json'
        if config_file.exists() and json.loads(config_file.read_text()) != protocol:
            raise ValueError('Probe output directory is pinned to a different protocol')
        atomic_json(protocol, config_file)
        if (out / 'summary.json').exists():
            print((out / 'summary.json').read_text())
            return
        target_line = np.unique(data.meta['cell'][data.val])
        assert len(target_line) == 1
        train_pool = data.train[data.meta['cell'][data.train] == target_line[0]]
        pools = {'train_hepg2': train_pool, 'validation': data.val}
        selected = {
            split: np.sort(np.random.default_rng(args.seed + i).choice(pool, min(args.cells, len(pool)), replace=False))
            for i, (split, pool) in enumerate(pools.items())
        }
        assert all(np.all(data.meta['split'][idx] == (0 if name == 'train_hepg2' else 1)) for name, idx in selected.items())
        model = data.build_model(cfg['model']).cuda().eval()
        forward = model if args.no_compile else torch.compile(model, fullgraph=True)
        result = {'time': timestamp(), 'protocol': protocol, 'weights': {}}
        states = {'raw_diagnostic': checkpoint['model'],
                  'ema_0.9996': {**checkpoint['model'], **checkpoint['ema']['0.9996']}}
        tr = cfg['training']
        flow_options = {key: tr[key] for key in ('noise_std', 'p_mean', 'p_std')}
        for weights, state in states.items():
            model.load_state_dict(state)
            result['weights'][weights] = {}
            for i, (split, indices) in enumerate(selected.items()):
                cache = out / f'{weights}_{split}.npz'
                if cache.exists():
                    with np.load(cache) as saved:
                        arrays = {key: saved[key] for key in saved.files}
                    np.testing.assert_array_equal(arrays['indices'], indices)
                else:
                    arrays = probe(forward, data, indices, args.batch_size, args.seed + 1000 * (i + 1),
                                   {'flow': flow_options, 'events': out / 'events.jsonl', 'weights': weights, 'split': split})
                    atomic_npz(cache, **arrays)
                result['weights'][weights][split] = summarize(arrays, args.bootstrap, args.seed + 123)
        atomic_json(result, out / 'summary.json')
        print(json.dumps(result, indent=2, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
