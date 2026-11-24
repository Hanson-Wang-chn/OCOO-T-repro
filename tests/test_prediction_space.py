"""Paper Eq.6 must agree with the velocity loss and the unchanged ODE solver."""
import torch

from ocoot.flow import sample
from ocoot.model import ModelConfig, OCOOT


def model(space):
    torch.manual_seed(27)
    embedding = torch.randn(7, 16, generator=torch.Generator().manual_seed(41))
    cfg = ModelConfig(dim=32, depth=3, heads=4, value_hidden=32, time_dim=16,
                      context_tokens=3, context_start=1, grad_checkpoint=False,
                      prediction_space=space)
    return OCOOT(cfg, embedding, torch.arange(6), 2, 3).eval()


def inputs():
    x = torch.tensor([[0., .3, 1., 0., 2., .7], [.1, 0., .8, 2., 0., 1.]])
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(72))
    t = torch.tensor([.2, .8])
    z = t[:, None] * x + (1 - t[:, None]) * noise
    return x, noise, t, z, torch.tensor([0, 2]), torch.tensor([0, 1]), torch.tensor([1, 2])


def test_x_prediction_velocity_loss_equals_reweighted_clean_loss_and_gradients():
    raw, converted = model('v'), model('x')
    assert list(dict(raw.named_parameters())) == list(dict(converted.named_parameters()))
    for key, value in raw.state_dict().items():
        torch.testing.assert_close(value, converted.state_dict()[key], rtol=0, atol=0)
    generator = torch.Generator().manual_seed(81)
    with torch.no_grad():
        for block in raw.blocks:
            block.ada[-1].weight.normal_(0, .02, generator=generator)
        raw.final_ada[-1].weight.normal_(0, .02, generator=generator)
        raw.output.weight.normal_(0, .03, generator=generator)
    converted.load_state_dict(raw.state_dict(), strict=True)
    x, noise, t, z, pert, cell, batch = inputs()
    theta = raw(z, t, pert, cell, batch)
    velocity = converted(z, t, pert, cell, batch)
    v_loss = (velocity - (x - noise)).square().mean()
    weighted_x_loss = ((theta - x) / (1 - t[:, None])).square().mean()
    torch.testing.assert_close(v_loss, weighted_x_loss, rtol=1e-6, atol=1e-7)
    v_loss.backward()
    weighted_x_loss.backward()
    for name, param in raw.named_parameters():
        other = dict(converted.named_parameters())[name]
        assert (param.grad is None) == (other.grad is None)
        if param.grad is not None:
            torch.testing.assert_close(param.grad, other.grad, rtol=5e-5, atol=2e-6)


def test_zero_clean_head_generates_zero_under_the_existing_heun50_solver():
    network = model('x')
    x, noise, t, z, pert, cell, batch = inputs()
    velocity = network(z, t, pert, cell, batch)
    torch.testing.assert_close(velocity, -z / (1 - t[:, None]), rtol=0, atol=0)
    generated = sample(network, pert, cell, batch, 6, steps=50, seed=913, clamp=False)
    assert float(generated.abs().max()) < 1e-5
