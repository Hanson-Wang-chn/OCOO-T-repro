"""Protect ongoing shared-projection runs when adding an untied trial."""
import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
import torch

from ocoot.model import ModelConfig, OCOOT


def legacy():
    path = Path(__file__).parent / 'model_before_separate_projection.py'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == '42dfbc953b20c4584a6e53bb6da003db33c0c2e6d432c3c386b238e801b890f1'
    spec = importlib.util.spec_from_file_location('ocoot_model_before_projection', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


OPTIONS = dict(dim=32, depth=3, heads=4, value_hidden=32, time_dim=16,
               context_tokens=3, context_start=1, grad_checkpoint=False)


def build(cls, cfg):
    embedding = torch.randn(7, 16, generator=torch.Generator().manual_seed(14))
    return cls(cfg, embedding, torch.arange(6), 2, 3,
               torch.tensor([False, False, True, False, False, True, False]))


def activate_conditioning(model):
    # Zero initial output/adaLN would make a forward-equivalence test vacuous.
    generator = torch.Generator().manual_seed(319)
    with torch.no_grad():
        for block in model.blocks:
            block.ada[-1].weight.normal_(0, .03, generator=generator)
            block.ada[-1].bias.normal_(0, .03, generator=generator)
        model.final_ada[-1].weight.normal_(0, .03, generator=generator)
        model.output.weight.normal_(0, .03, generator=generator)


def forward(model):
    generator = torch.Generator().manual_seed(901)
    z = torch.randn(2, 6, generator=generator)
    labels = torch.tensor([0, 2])
    cells, batches = torch.tensor([0, 1]), torch.tensor([1, 2])
    off = torch.zeros(2, dtype=torch.bool)
    return model(z, torch.tensor([.2, .7]), labels, cells, batches, off, off)


def test_default_preserves_legacy_initialization_forward_backward_and_parameter_order():
    old = legacy()
    torch.manual_seed(726)
    first = build(old.OCOOT, old.ModelConfig(**OPTIONS)).eval()
    old_rng = torch.get_rng_state()
    torch.manual_seed(726)
    second = build(OCOOT, ModelConfig(**OPTIONS)).eval()
    assert torch.equal(old_rng, torch.get_rng_state())
    assert list(dict(first.named_parameters())) == list(dict(second.named_parameters()))
    for name, tensor in first.state_dict().items():
        torch.testing.assert_close(tensor, second.state_dict()[name], rtol=0, atol=0)
    activate_conditioning(first)
    second.load_state_dict(first.state_dict(), strict=True)
    a, b = forward(first), forward(second)
    assert a.abs().sum() > 0
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    for name, param in first.named_parameters():
        other = dict(second.named_parameters())[name]
        assert (param.grad is None) == (other.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=0, atol=0)


def test_separate_starts_identically_without_rng_use_but_splits_projection_gradients():
    torch.manual_seed(826)
    shared = build(OCOOT, ModelConfig(**OPTIONS)).eval()
    shared_rng = torch.get_rng_state()
    torch.manual_seed(826)
    separate = build(OCOOT, ModelConfig(**OPTIONS, pert_projection='separate')).eval()
    assert torch.equal(shared_rng, torch.get_rng_state())
    assert separate.pert_proj is not separate.gene_proj
    for name, tensor in shared.state_dict().items():
        torch.testing.assert_close(tensor, separate.state_dict()[name], rtol=0, atol=0)
    torch.testing.assert_close(separate.gene_proj.weight, separate.pert_proj.weight, rtol=0, atol=0)
    activate_conditioning(shared)
    result = separate.load_state_dict(shared.state_dict(), strict=False)
    assert set(result.missing_keys) == {'pert_proj.weight', 'pert_proj.bias'}
    assert not result.unexpected_keys
    a, b = forward(shared), forward(separate)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    a.square().sum().backward()
    b.square().sum().backward()
    assert separate.gene_proj.weight.grad.abs().sum() > 0
    assert separate.pert_proj.weight.grad.abs().sum() > 0
    torch.testing.assert_close(shared.gene_proj.weight.grad,
                               separate.gene_proj.weight.grad + separate.pert_proj.weight.grad,
                               rtol=2e-5, atol=2e-7)
    torch.testing.assert_close(shared.gene_proj.bias.grad,
                               separate.gene_proj.bias.grad + separate.pert_proj.bias.grad,
                               rtol=2e-5, atol=2e-7)
    assert not separate.esm.requires_grad and separate.esm.grad is None


def test_input_scales_do_not_change_other_weights_or_the_initial_rng_stream():
    torch.manual_seed(726)
    original = build(OCOOT, ModelConfig(**OPTIONS))
    original_rng = torch.get_rng_state()
    torch.manual_seed(726)
    candidate = build(OCOOT, ModelConfig(**OPTIONS, position_scale=.125, gene_proj_init_scale=2.))
    assert torch.equal(original_rng, torch.get_rng_state())
    for key, value in original.state_dict().items():
        expected = value * .125 if key == 'position' else value * 2 if key == 'gene_proj.weight' else value
        torch.testing.assert_close(candidate.state_dict()[key], expected, rtol=0, atol=0)


@pytest.mark.parametrize('projection,drop_batch', [('shared', True), ('separate', False)])
@torch.no_grad()
def test_null_initialization_uses_only_the_dropped_conditions(projection, drop_batch):
    cfg = ModelConfig(**OPTIONS, pert_projection=projection, drop_batch_with_cell=drop_batch,
                      null_condition_init='training_mean')
    model = build(OCOOT, cfg)
    if projection == 'separate':
        model.pert_proj.weight.mul_(2)
    before = {name: value.clone() for name, value in model.state_dict().items()}
    pert = torch.tensor([1, 1, 1, 2, 4])
    cell = torch.tensor([0, 0, 1, 1, 1])
    batch = torch.tensor([0, 1, 1, 1, 2])
    expected_pert = model.gene_table(model.pert_proj)[pert].mean(0)
    expected_cell = model.cell(cell)
    if drop_batch:
        expected_cell += model.batch(batch)
    rng = torch.get_rng_state()
    model.initialize_null_conditions(pert, cell, batch)
    assert torch.equal(torch.get_rng_state(), rng)
    before.update(null_pert=expected_pert, null_cell=expected_cell.mean(0))
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
