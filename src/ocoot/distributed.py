"""Single-node DDP setup and checkpointable per-rank random streams."""
from contextlib import contextmanager
from datetime import timedelta
import json
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist


def microbatch_shard(effective_batch, micro_batch, size, rank):
    """Return a contiguous range of whole microbatches in one global batch."""
    if micro_batch <= 0 or effective_batch <= 0 or effective_batch % micro_batch:
        raise ValueError('Effective batch must be a positive multiple of microbatch')
    total = effective_batch // micro_batch
    if not 0 <= rank < size <= total:
        raise ValueError('Every worker needs at least one microbatch')
    base, extra = divmod(total, size)
    first = rank * base + min(rank, extra)
    return range(first, first + base + (rank < extra))


def live_worker_pids(workers):
    pids = set()
    for worker in workers:
        pid = worker['pid']
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z' or int(fields[19]) != worker['start_tick']:
            raise RuntimeError(f'Training worker {pid} has changed or exited')
        pids.add(pid)
    return pids


def review_trainer_pids(workers, run):
    run = Path(run)
    if json.loads((run / 'training_execution.json').read_text())['workers'] != workers:
        raise RuntimeError('The review is registered to another launch')
    if (run / 'finished.json').exists():
        return {worker['pid'] for worker in workers}
    return live_worker_pids(workers)


class TrainingRanks:
    def __init__(self):
        self.size = int(os.environ.get('WORLD_SIZE', 1))
        self.rank = int(os.environ.get('RANK', 0))
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.device = torch.device('cuda', self.local_rank)
        self.primary = self.rank == 0
        self.control = None

    @contextmanager
    def initialize(self):
        torch.cuda.set_device(self.device)
        if self.size == 1:
            yield self
            return
        if int(os.environ['LOCAL_WORLD_SIZE']) != self.size:
            raise ValueError('Training supports a single node')
        dist.init_process_group('nccl')
        try:
            # Generation runs on rank zero and can exceed the NCCL timeout.
            self.control = dist.new_group(backend='gloo', timeout=timedelta(hours=6))
            yield self
        finally:
            dist.destroy_process_group()

    def barrier(self):
        if self.size > 1:
            dist.barrier(group=self.control)

    def from_primary(self, value):
        if self.size > 1:
            values = [value if self.primary else None]
            dist.broadcast_object_list(values, group=self.control)
            return values[0]
        return value

    def local_rng(self):
        return {'python': random.getstate(), 'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(), 'cuda': [torch.cuda.get_rng_state(self.device)]}

    def set_local_rng(self, state):
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'][0], self.device)

    def gather_rng(self):
        return self.gather(self.local_rng())

    def gather(self, value):
        if self.size == 1:
            return [value]
        gathered = [None] * self.size if self.primary else None
        dist.gather_object(value, gathered, dst=0, group=self.control)
        return gathered

    def restore_rng(self, checkpoint, seed):
        states = checkpoint.get('rng_by_rank', [checkpoint['rng']])
        if len(states) == self.size or self.rank == 0:
            self.set_local_rng(states[self.rank])
        else:
            self.seed(seed, step=checkpoint['step'])

    def seed(self, seed, step=0):
        rank_seed = seed
        if self.rank:
            rank_seed = int(np.random.SeedSequence([seed, step, self.rank]).generate_state(1)[0])
        random.seed(rank_seed)
        np.random.seed(rank_seed)
        torch.random.default_generator.manual_seed(rank_seed)
        torch.cuda.manual_seed(rank_seed)
