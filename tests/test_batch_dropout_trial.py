"""CPU checks for a mature-state nuisance-batch interpretation trial."""
import copy
import importlib.util
import json
from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
import torch

from ocoot.model import ModelConfig, OCOOT
from ocoot.runtime import file_sha256


OPTIONS = dict(dim=32, depth=3, heads=4, value_hidden=32, time_dim=16,
               context_tokens=3, context_start=1, grad_checkpoint=False)
trial = runpy.run_path('scripts/run_batch_dropout_trial.py')


def legacy():
    path = Path(__file__).parent / 'model_before_batch_retention.py'
    assert file_sha256(path) == '1f83d4d27de5a0e72ffc06647443c6490b1442ecb52ddcbdc72f0794cb269a09'
    spec = importlib.util.spec_from_file_location('model_before_batch_retention', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build(cls=OCOOT, cfg=None):
    embeddings = torch.randn(9, 16, generator=torch.Generator().manual_seed(21))
    return cls(cfg or ModelConfig(**OPTIONS), embeddings, torch.arange(6), 2, 3)


def activate(model):
    rng = torch.Generator().manual_seed(320)
    with torch.no_grad():
        for block in model.blocks:
            block.ada[-1].weight.normal_(0, .04, generator=rng)
            block.ada[-1].bias.normal_(0, .04, generator=rng)
        model.final_ada[-1].weight.normal_(0, .04, generator=rng)
        model.output.weight.normal_(0, .04, generator=rng)


def inputs():
    return (torch.randn(4, 6, generator=torch.Generator().manual_seed(13)),
            torch.tensor([.2, .3, .4, .7]), torch.tensor([0, 1, 2, 3]),
            torch.tensor([0, 0, 1, 1]), torch.tensor([0, 1, 2, 1]))


def test_default_matches_previous_nonzero_forward_gradients_and_rng_for_all_modes():
    old = legacy()
    torch.manual_seed(83)
    first = build(old.OCOOT, old.ModelConfig(**OPTIONS))
    state = torch.get_rng_state()
    torch.manual_seed(83)
    second = build()
    assert torch.equal(state, torch.get_rng_state())
    assert list(dict(first.named_parameters())) == list(dict(second.named_parameters()))
    trial['assert_tree_equal'](first.state_dict(), second.state_dict())
    activate(first)
    second.load_state_dict(first.state_dict())
    pdrop = torch.tensor([False, True, False, True])
    cdrop = torch.tensor([False, False, True, True])
    a, b = first(*inputs(), pdrop, cdrop), second(*inputs(), pdrop, cdrop)
    assert a.abs().sum() > 0
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward(); b.square().sum().backward()
    for name, p in first.named_parameters():
        q = dict(second.named_parameters())[name]
        assert (p.grad is None) == (q.grad is None)
        if p.grad is not None:
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
    torch.manual_seed(991)
    a = first(*inputs())
    rng = torch.get_rng_state()
    torch.manual_seed(991)
    b = second(*inputs())
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_batch_retention_changes_only_cell_dropped_context_and_keeps_conditional_path():
    torch.manual_seed(99)
    control = build().eval()
    activate(control)
    candidate = build(cfg=ModelConfig(**OPTIONS, drop_batch_with_cell=False)).eval()
    candidate.load_state_dict(control.state_dict())
    args = inputs()
    off = torch.zeros(4, dtype=torch.bool)
    torch.testing.assert_close(control(*args, off, off), candidate(*args, off, off), rtol=0, atol=0)
    captured = []
    handles = [m.blocks[0].register_forward_pre_hook(lambda _, values: captured.append(values[1].detach().clone()))
               for m in (control, candidate)]
    pdrop, cdrop = torch.tensor([False, True, False, True]), torch.tensor([False, False, True, True])
    a, b = control(*args, pdrop, cdrop), candidate(*args, pdrop, cdrop)
    for handle in handles:
        handle.remove()
    expected = candidate.batch(args[4]).detach() * cdrop[:, None]
    torch.testing.assert_close(captured[1] - captured[0], expected, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(a[:2], b[:2], rtol=0, atol=0)
    assert not torch.equal(a[2:], b[2:])
    on = ~off
    control.zero_grad(set_to_none=True); candidate.zero_grad(set_to_none=True)
    control(*args, off, on).square().sum().backward()
    candidate(*args, off, on).square().sum().backward()
    assert control.batch.weight.grad.abs().sum() == 0
    assert candidate.batch.weight.grad.abs().sum() > 0


def test_bounded_adamw_preserves_updates_slots_rng_and_stops_before_an_extra_update():
    a = torch.nn.Parameter(torch.tensor([.1, -.3, .7]))
    b = torch.nn.Parameter(a.detach().clone())
    seen = []
    bounded = trial['bounded_adamw_class'](torch.optim.AdamW, 5, lambda: seen.append(b.detach().clone()))
    reference = torch.optim.AdamW([a], lr=1e-4, betas=(.9, .95), weight_decay=0)
    candidate = bounded([b], lr=1e-4, betas=(.9, .95), weight_decay=0)
    rng = torch.get_rng_state()
    for i in range(5):
        gradient = torch.tensor([1., .3, -.4]) * (i + 1)
        a.grad = gradient.clone(); b.grad = gradient.clone()
        reference.step(); candidate.step()
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        trial['assert_tree_equal'](reference.state_dict(), candidate.state_dict())
        assert len(seen) == int(i == 4)
    assert torch.equal(rng, torch.get_rng_state())
    before = b.detach().clone()
    with pytest.raises(RuntimeError, match='beyond'):
        candidate.step()
    torch.testing.assert_close(b, before, rtol=0, atol=0)


def test_safe_stop_refuses_a_missing_trainer_handler(monkeypatch):
    function = trial['request_safe_stop']
    monkeypatch.setattr(function.__globals__['signal'], 'getsignal', lambda _: 0)
    with pytest.raises(RuntimeError, match='handler'):
        function()


def test_full_fork_preserves_optimizer_dual_ema_rng_and_rejects_config_drift(tmp_path):
    source, reference = tmp_path / 'source.pt', tmp_path / 'control.pt'
    cfg = {'seed': 42, 'data_dir': 'data/processed', 'run_dir': 'original',
           'model': OPTIONS, 'training': {'train_subset': None, 'epochs': 100}}
    state = {'config': cfg, 'step': 27950, 'model': {'w': torch.tensor([1., 2.])},
             'optimizer': {'state': {0: {'exp_avg': torch.tensor([.1, .2]), 'step': torch.tensor(37.)}},
                           'param_groups': [{'lr': .000088, 'params': [0]}]},
             'ema': {tag: {'w': torch.tensor([.5, 1.5])} for tag in ('0.9999', '0.9996')},
             'rng': {'torch': torch.get_rng_state(), 'numpy': np.random.get_state()},
             'pending_generation_epoch': 25, 'best_generation': .92}
    torch.save(state, source); torch.save({**state, 'step': 29068}, reference)
    paths = ['src/ocoot/train.py', 'scripts/train.py', 'src/ocoot/flow.py', 'src/ocoot/runtime.py']
    plan = {'base_checkpoint': str(source), 'reference_checkpoint': str(reference),
            'base_checkpoint_sha256': file_sha256(source), 'reference_checkpoint_sha256': file_sha256(reference),
            'base_step': 27950, 'reference_step': 29068, 'updates': 1118, 'stop_after_step': 29068,
            'candidate_run': str(tmp_path / 'candidate'), 'candidate_config': str(tmp_path / 'candidate.yaml'),
            'main_config_sha256': file_sha256('configs/main_selected.yaml'),
            'original_source_sha256': {p: file_sha256(p) for p in paths}}
    fork_cfg, step = trial['prepare'](plan, tmp_path)
    assert step == 27950 and fork_cfg['model']['drop_batch_with_cell'] is False
    fork = torch.load(tmp_path / 'candidate/last.pt', weights_only=False, mmap=True)
    for key in ('model', 'optimizer', 'ema', 'rng', 'best_generation'):
        trial['assert_tree_equal'](state[key], fork[key])
    assert fork['pending_generation_epoch'] is None
    assert not (tmp_path / 'candidate/finished.json').exists()
    trial['prepare'](plan, tmp_path)  # idempotent preparation
    changed = copy.deepcopy(fork_cfg); changed['training']['epochs'] = 26
    with pytest.raises(ValueError, match='beyond'):
        trial['configs_match'](cfg, changed)


def test_ema_export_uses_default_ema_and_preserves_buffers(tmp_path):
    source, output = tmp_path / 'full.pt', tmp_path / 'ema.pt'
    state = {'step': 29068, 'config': {'model': {}},
             'model': {'w': torch.ones(2), 'buffer': torch.arange(3)},
             'ema': {'0.9996': {'w': torch.zeros(2)}, '0.9999': {'w': torch.full((2,), .5)}}}
    torch.save(state, source)
    trial['export_ema'](source, output, 29068)
    exported = torch.load(output, weights_only=False)
    assert exported['weights'] == 'ema_0.9996'
    assert torch.equal(exported['model']['w'], torch.zeros(2))
    assert torch.equal(exported['model']['buffer'], torch.arange(3))
    trial['export_ema'](source, output, 29068)
    with pytest.raises(ValueError, match='budgets'):
        trial['export_ema'](source, output, 29069)
