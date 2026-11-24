"""The scalar fan-in trial must alter only its registered initialization choice."""
import math

import torch

from ocoot.model import ModelConfig, OCOOT


def network(initialization):
    torch.manual_seed(78)
    embeddings = torch.ones(7, 16)
    cfg = ModelConfig(dim=32, depth=3, heads=4, value_hidden=32, time_dim=16,
                      context_tokens=3, context_start=1, grad_checkpoint=False,
                      value_input_init=initialization)
    return OCOOT(cfg, embeddings, torch.arange(6), 2, 3)


def test_fan_in_changes_only_scalar_weight_scale_without_consuming_rng():
    reference = network('xavier')
    rng = torch.get_rng_state().clone()
    candidate = network('fan_in')
    assert torch.equal(torch.get_rng_state(), rng)
    assert list(dict(reference.named_parameters())) == list(dict(candidate.named_parameters()))
    factor = math.sqrt(33 / 6)
    for name, parameter in reference.state_dict().items():
        expected = parameter * factor if name == 'value.0.weight' else parameter
        torch.testing.assert_close(candidate.state_dict()[name], expected, rtol=0, atol=0)
    assert float(candidate.value[0].weight.detach().abs().max()) <= 1
    assert torch.count_nonzero(candidate.value[0].bias) == 0
    zero, direction = torch.zeros(1, 1), torch.ones(1, 1)
    _, derivative_reference = torch.autograd.functional.jvp(reference.value, zero, direction)
    _, derivative_candidate = torch.autograd.functional.jvp(candidate.value, zero, direction)
    torch.testing.assert_close(derivative_candidate, derivative_reference * factor, rtol=2e-6, atol=1e-7)
