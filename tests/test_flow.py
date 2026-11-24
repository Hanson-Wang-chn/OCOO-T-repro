import torch
from ocoot.flow import flow_batch, guided_velocity, sample
from ocoot.model import OCOOT, ModelConfig


def test_flow_endpoints_and_velocity():
    x = torch.randn(512, 8)
    z, t, v = flow_batch(x)
    assert torch.allclose(z + (1 - t[:, None]) * v, x, atol=1e-6)
    assert abs(torch.logit(t).mean().item() + .8) < .15


class Field:
    def __call__(self, z, t, p, c, b, drop_pert=None, drop_cell=None):
        if drop_pert is None: return torch.ones_like(z) * 5
        return (1 + 3 * (~drop_pert).float() + (~drop_cell).float())[:, None].expand_as(z)


def test_literal_dual_cfg():
    z = torch.zeros(2, 3); a = torch.zeros(2, dtype=torch.long)
    # vu=1, vc=2, vp=4, vpc=5; Eq11 with w=2 gives 1+1+3+8=13.
    assert torch.all(guided_velocity(Field(), z, a.float(), a, a, a, 2) == 13)
    assert torch.all(guided_velocity(Field(), z, a.float(), a, a, a, None) == 5)


def test_heun_euler_constant_field():
    a = torch.zeros(4, dtype=torch.long)
    z = sample(Field(), a, a, a, 5, steps=50, seed=13)
    noise = torch.randn((4, 5), generator=torch.Generator().manual_seed(13))
    assert torch.allclose(z, noise + 5, atol=1e-5)


def test_zero_init_and_all_layers_get_gradients():
    cfg = ModelConfig(dim=32, depth=4, heads=4, value_hidden=32, time_dim=16, context_tokens=2, context_start=1, grad_checkpoint=True)
    m = OCOOT(cfg, torch.randn(12, 64), torch.arange(8), 2, 3)
    x = torch.randn(4, 8); t = torch.rand(4); p = torch.tensor([1, 2, 3, 4]); c = p % 2; b = p % 3
    assert torch.count_nonzero(m(x, t, p, c, b)) == 0
    opt = torch.optim.AdamW(m.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad(); (m(x, t, p, c, b) - x).square().mean().backward(); opt.step()
    assert all(block.attn.qkv.weight.grad.abs().sum() > 0 for block in m.blocks)
    assert m.esm.grad is None

