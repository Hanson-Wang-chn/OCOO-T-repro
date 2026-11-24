"""Compare the actual mature model's one-step update on a fixed global batch."""
import argparse
import contextlib
import json
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from ocoot.data import BenchmarkData
from ocoot.distributed import TrainingRanks
from ocoot.flow import flow_batch


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--eager', action='store_true')
    p.add_argument('--compare', type=Path)
    p.add_argument('--no-bucket-view', action='store_true')
    p.add_argument('--manual-reduce', action='store_true')
    p.add_argument('--disable-ddp-optimizer', action='store_true')
    p.add_argument('--simulate-world', type=int)
    p.add_argument('--simulate-rank', type=int)
    a = p.parse_args()
    ranks = TrainingRanks()
    with ranks.initialize():
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        if a.disable_ddp_optimizer:
            torch._dynamo.config.optimize_ddp = False
        src = Path('runs/main_v1/snapshots/step_41374/full.pt')
        s = torch.load(src, map_location='cpu', mmap=True, weights_only=False)
        cfg = s['config']
        tr = cfg['training']
        data = BenchmarkData(cfg['data_dir'])
        model = data.build_model(cfg['model']).to(ranks.device).train()
        model.load_state_dict(s['model'])
        opt = torch.optim.AdamW(model.parameters(), lr=tr['lr'], betas=tuple(tr['betas']), weight_decay=tr['weight_decay'], fused=True)
        opt.load_state_dict(s['optimizer'])
        forward = model if a.eager else torch.compile(model, fullgraph=True)
        if ranks.size > 1 and not a.manual_reduce:
            forward = DistributedDataParallel(forward, device_ids=[ranks.local_rank],
                broadcast_buffers=False, gradient_as_bucket_view=not a.no_bucket_view)
        steps_per_epoch = len(data.train) // 512
        for offset in range(2):
            step = s['step'] + offset
            epoch, within = divmod(step, steps_per_epoch)
            perm = np.random.default_rng(cfg['seed'] + 10007*epoch).permutation(data.train)
            size, rank = a.simulate_world or ranks.size, ranks.rank
            if a.simulate_rank is not None:
                rank = a.simulate_rank
            batches = np.array_split(np.arange(32), size)[rank] if a.simulate_rank is not None else (
                np.array_split(np.arange(32), ranks.size)[ranks.rank])
            started = time.monotonic()
            opt.zero_grad(set_to_none=True)
            total = torch.zeros((), device=ranks.device)
            for j in batches:
                i = within*512+int(j)*16
                x, p, c, b = data.batch(perm[i:i+16], device=ranks.device)
                generator = torch.Generator(device=ranks.device).manual_seed(1234567 + int(j) + offset * 32)
                z, t, v = flow_batch(x, tr['noise_std'], tr['p_mean'], tr['p_std'], generator)
                drop = [torch.rand(16, device=ranks.device, generator=generator)<cfg['model']['drop_prob'] for _ in range(2)]
                scope = forward.no_sync() if isinstance(forward, DistributedDataParallel) and j!=batches[-1] else contextlib.nullcontext()
                with scope:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        loss = (forward(z, t, p, c, b, *drop).float()-v).square().mean() * (size/32)
                    loss.backward()
                total += loss.detach()
            if ranks.size>1:
                dist.all_reduce(total)
                if a.manual_reduce:
                    for param in model.parameters():
                        dist.all_reduce(param.grad)
                        param.grad.div_(ranks.size)
                checksum = torch.stack([p.grad.sum() for p in model.parameters()])
                rank_zero = checksum.clone()
                dist.broadcast(rank_zero, src=0)
                discrepancy = (checksum-rank_zero).abs().max()
                dist.all_reduce(discrepancy, op=dist.ReduceOp.MAX)
            elif a.simulate_rank is None and a.simulate_world:
                for param in model.parameters():
                    param.grad.div_(size)
            opt.step()
            if ranks.primary:
                report = dict(step=step+1, loss=total.item()/ranks.size, world=ranks.size,
                              eager=a.eager, bucket_view=not a.no_bucket_view, seconds=time.monotonic()-started)
                params = {n: p for n, p in model.named_parameters()}
                print('partial', report, flush=True)
                grads = {n:p.grad.cpu() for n,p in params.items()}
                raw = {n:p.detach().cpu() for n,p in params.items()}
                report['grad_norm'] = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g) for g in grads.values()])).item()
                if a.compare:
                    ref = torch.load(a.compare/f'{offset}.pt', map_location='cpu', weights_only=True, mmap=True)
                    ref_grad = ref['grads']
                    sum_sq = sum(float((g-ref_grad[n]).square().sum()) for n,g in grads.items())
                    ref_sq = sum(float(g.square().sum()) for g in ref_grad.values())
                    report['relative_grad_l2_error'] = (sum_sq / ref_sq) ** .5
                    report['max_param_error'] = max(float((p-ref['weights'][n]).abs().max()) for n,p in raw.items())
                    report['time_bias_grad_norm'] = float(torch.linalg.vector_norm(grads['time.0.bias']))
                    report['time_bias_reference_grad_norm'] = float(torch.linalg.vector_norm(ref_grad['time.0.bias']))
                    report['max_gradient_error_keys'] = sorted(((float((g-ref_grad[n]).abs().max()),n)
                                               for n,g in grads.items()),reverse=True)[:8]
                else:
                    a.out.mkdir(parents=True,exist_ok=True)
                    torch.save({'weights': raw,'grads':grads}, a.out/f'{offset}.pt')
                if ranks.size>1:
                    report['cross_rank_grad_checksum_max_error'] = discrepancy.item()
                print(json.dumps(report),flush=True)
            ranks.barrier()

if __name__ == '__main__':
    main()
