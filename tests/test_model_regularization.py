from pathlib import Path
import runpy
from unittest.mock import patch

import pytest
import torch

from ocoot.model import ModelConfig, OCOOT


def make_model(config_class=ModelConfig, model_class=OCOOT, **choices):
    options = dict(dim=32, depth=4, heads=4, value_hidden=32, time_dim=16,
                   context_tokens=2, context_start=1, grad_checkpoint=False)
    options.update(choices)
    torch.manual_seed(19)
    model = model_class(config_class(**options), torch.randn(9, 16), torch.arange(8), 2, 3)
    with torch.no_grad():
        for block in model.blocks:
            block.ada[-1].weight.normal_(std=.02)
        model.output.weight.normal_(std=.02)
    return model


def inputs(n=8, device='cpu'):
    return (torch.randn(n, 8, device=device), torch.full((n,), .3, device=device),
            torch.arange(n, device=device) % 9, torch.arange(n, device=device) % 2,
            torch.arange(n, device=device) % 3)


def test_dropout_defaults_match_legacy_training_and_state():
    legacy = runpy.run_path(str(Path(__file__).with_name('model_before_batch_retention.py')))
    old = make_model(legacy['ModelConfig'], legacy['OCOOT'])
    new = make_model()
    assert set(old.state_dict()) == set(new.state_dict())
    new.load_state_dict(old.state_dict(), strict=True)
    values = inputs()
    torch.manual_seed(57)
    expected = old(*values)
    torch.manual_seed(57)
    torch.testing.assert_close(new(*values), expected, atol=0, rtol=0)


def test_jit_middle_dropout_is_disabled_at_eval():
    regularized = make_model(depth=12, attn_drop=.1, proj_drop=.1)
    plain = make_model(depth=12)
    assert [b.attn.attn_drop for b in regularized.blocks] == [0.] * 3 + [.1] * 6 + [0.] * 3
    assert [b.attn.proj_drop.p for b in regularized.blocks] == [0.] * 3 + [.1] * 6 + [0.] * 3
    values = inputs()
    expected = plain.eval()(*values)
    torch.testing.assert_close(regularized.eval()(*values), expected, atol=0, rtol=0)
    torch.testing.assert_close(regularized(*values), expected, atol=0, rtol=0)
    assert all(b.attn.proj_drop.p == .1 for b in make_model(proj_drop=.1, dropout_layer_policy='all').blocks)


def test_balanced_condition_dropout_frequencies_and_explicit_masks():
    model = make_model(depth=0, condition_dropout_mode='balanced_3mode')
    values = inputs(16384)
    with patch('ocoot.model.torch.where', wraps=torch.where) as where:
        model(*values)
    drop_pert, drop_cell = [call.args[0].flatten() for call in where.call_args_list]
    for mask, probability in ((drop_pert & drop_cell, .06), (~drop_pert & drop_cell, .08),
                              (drop_pert & ~drop_cell, .08), (~drop_pert & ~drop_cell, .78)):
        assert abs(mask.float().mean().item() - probability) < .01
    with patch('ocoot.model.torch.rand', side_effect=AssertionError('unexpected random sampling')):
        model(*values, drop_pert, drop_cell)
        model.eval()(*values)


@pytest.mark.parametrize('options', [dict(attn_drop=1.), dict(proj_drop=-.1), dict(dropout_layer_policy='other'),
                                     dict(condition_dropout_mode='other'),
                                     dict(condition_dropout_mode='balanced_3mode', drop_both_prob=.9)])
def test_invalid_dropout_config_is_rejected(options):
    with pytest.raises(ValueError):
        make_model(**options)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA compiler integration')
def test_dropout_compiles_with_fullgraph_on_cuda():
    model = make_model(attn_drop=.1, proj_drop=.1, condition_dropout_mode='balanced_3mode').cuda().train()
    compiled = torch.compile(model, fullgraph=True)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        output = compiled(*inputs(device='cuda'))
    output.float().square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
