from contextlib import nullcontext

import pytest
import torch

from ocoot.distributed import microbatch_shard
from ocoot import train as trainer


@pytest.mark.parametrize('world_size, counts', [(1, [32]), (2, [16, 16]),
                                               (3, [11, 11, 10]), (4, [8, 8, 8, 8])])
def test_global_batch_shards_cover_each_cell_with_uniform_weight(world_size, counts):
    shards = [microbatch_shard(512, 16, world_size, rank) for rank in range(world_size)]
    assert [len(shard) for shard in shards] == counts
    assert [batch for shard in shards for batch in shard] == list(range(32))
    # DDP averages rank losses, so rank loss sums need the world-size factor
    # regardless of how many microbatches that rank was assigned.
    expected = sum(range(32)) / 32
    actual = sum(sum(shard) * world_size / 32 for shard in shards) / world_size
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('world_size, compile_model, expected', [(4, True, False), (1, True, True),
                                                                 (4, False, True)])
def test_ddp_optimizer_partition_is_disabled_only_for_compiled_ddp(monkeypatch, world_size,
                                                                     compile_model, expected):
    class Worker:
        size = world_size
        primary = False

        def initialize(self):
            return nullcontext()

        def barrier(self):
            pass

    monkeypatch.setattr(trainer, 'TrainingRanks', Worker)
    monkeypatch.setattr(trainer, '_train_locked', lambda *args: torch._dynamo.config.optimize_ddp)
    with torch._dynamo.config.patch(optimize_ddp=True):
        assert trainer.train({'training': {'compile': compile_model}}) is expected
        assert torch._dynamo.config.optimize_ddp is True
