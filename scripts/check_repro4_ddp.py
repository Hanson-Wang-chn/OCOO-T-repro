"""Run a bounded native training smoke test or replay matched compiled gradients."""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import signal

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import yaml

from ocoot.data import BenchmarkData
from ocoot.distributed import TrainingRanks, microbatch_shard
from ocoot.flow import flow_batch
from ocoot.runtime import atomic_save, file_sha256
import ocoot.train as trainer


def bounded_smoke(config, steps):
    run = Path(config['run_dir'])
    marker = run / 'checkpoint_state.json'
    if marker.exists() and json.loads(marker.read_text())['step'] == steps:
        return

    def observe(path, event):
        original(path, event)
        if event.get('event') == 'train' and event['step'] == steps - 1:
            os.kill(os.getpid(), signal.SIGTERM)

    original = trainer.log_event
    trainer.log_event = observe
    try:
        trainer.train(config)
    finally:
        trainer.log_event = original
    if TrainingRanks().primary and json.loads(marker.read_text())['step'] != steps:
        raise RuntimeError(f'Smoke test did not complete exactly {steps} optimizer steps')


def replay(config, checkpoint, reference, compare, output):
    ranks = TrainingRanks()
    with ranks.initialize(), torch._dynamo.config.patch(optimize_ddp=False):
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = True
        training = config['training']
        data = BenchmarkData(config['data_dir'])
        state = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=False)
        model = data.build_model(config['model'])
        model.load_state_dict(state['model'])
        model = model.to(ranks.device).train()
        step = state['step']
        del state
        forward = torch.compile(model, fullgraph=True, mode=training.get('compile_mode', 'default'))
        if ranks.size > 1:
            forward = DistributedDataParallel(forward, device_ids=[ranks.local_rank], broadcast_buffers=False,
                                              gradient_as_bucket_view=True)
        micro = training['micro_batch']
        effective = training['effective_batch']
        shard = microbatch_shard(effective, micro, ranks.size, ranks.rank)
        epoch, within = divmod(step, len(data.train) // effective)
        order = np.random.default_rng(config['seed'] + 10007 * epoch).permutation(data.train)
        loss_total = torch.zeros((), device=ranks.device)
        for batch in shard:
            first = within * effective + batch * micro
            x, pert, cell, batch_labels = data.batch(order[first:first + micro], device=ranks.device)
            generator = torch.Generator(device=ranks.device).manual_seed(1234567 + batch)
            z, t, v = flow_batch(x, training['noise_std'], training['p_mean'], training['p_std'], generator)
            # Key the CUDA RNG by global microbatch so compiled condition and
            # attention dropout see identical streams in single and DDP replay.
            torch.cuda.manual_seed(64713 + batch)
            scope = forward.no_sync() if ranks.size > 1 and batch != shard.stop - 1 else nullcontext()
            with scope:
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = (forward(z, t, pert, cell, batch_labels).float() - v).square().mean()
                    loss = loss * (ranks.size * micro / effective)
                loss.backward()
            loss_total += loss.detach()
        gradients = {name: param.grad for name, param in model.named_parameters()}
        norm = torch.linalg.vector_norm(torch.stack(torch._foreach_norm(list(gradients.values())))).item()
        if not 0 < norm < float('inf'):
            raise FloatingPointError('Nonfinite or zero replay gradient')
        cross_rank_error = 0.
        if ranks.size > 1:
            dist.all_reduce(loss_total)
            checksums = torch.stack([grad.sum() for grad in gradients.values()])
            first_rank = checksums.clone()
            dist.broadcast(first_rank, src=0)
            discrepancy = (checksums - first_rank).abs().max()
            dist.all_reduce(discrepancy, op=dist.ReduceOp.MAX)
            cross_rank_error = discrepancy.item()
        if ranks.primary:
            provenance = {'checkpoint_sha256': file_sha256(checkpoint), 'step': step,
                          'effective_batch': effective, 'micro_batch': micro, 'model': config['model'],
                          'noise_std': training['noise_std'], 'p_mean': training['p_mean'], 'p_std': training['p_std'],
                          'seed': config['seed'], 'optimize_ddp': False}
            result = {'protocol': provenance, 'world_size': ranks.size, 'loss': float(loss_total) / ranks.size,
                      'grad_norm': norm, 'cross_rank_grad_checksum_max_error': cross_rank_error}
            cpu_gradients = {name: grad.cpu() for name, grad in gradients.items()}
            if compare:
                expected = torch.load(reference, map_location='cpu', mmap=True, weights_only=True)
                if provenance != expected['protocol'] or ranks.size != 4:
                    raise ValueError('Gradient comparison requires the same replay protocol on four ranks')
                expected = expected['gradients']
                squared = sum(float((grad - expected[name]).square().sum()) for name, grad in cpu_gradients.items())
                reference_squared = sum(float(grad.square().sum()) for grad in expected.values())
                result['relative_grad_l2_error'] = (squared / reference_squared) ** .5
                result['passed'] = result['relative_grad_l2_error'] < .01 and cross_rank_error == 0
            else:
                if ranks.size != 1:
                    raise ValueError('Use one GPU to register the gradient reference')
                atomic_save({'protocol': provenance, 'gradients': cpu_gradients}, reference)
            output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
            print(json.dumps(result, indent=2), flush=True)
        ranks.barrier()
        if ranks.primary and compare and not result['passed']:
            raise RuntimeError('Four-GPU gradient replay exceeded the 1% relative L2 tolerance')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke-steps', type=int)
    mode.add_argument('--checkpoint', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    if args.smoke_steps is not None:
        if args.smoke_steps < 2 or config['training']['log_steps'] != 1 or TrainingRanks().size != 4:
            parser.error('The training smoke test requires four GPUs, log_steps=1 and at least two steps')
        bounded_smoke(config, args.smoke_steps)
    else:
        if args.reference is None or args.output is None:
            parser.error('Gradient replay requires --reference and --output')
        replay(config, args.checkpoint, args.reference, args.compare, args.output)


if __name__ == '__main__':
    main()
