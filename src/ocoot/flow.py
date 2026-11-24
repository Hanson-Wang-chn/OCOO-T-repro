"""Rectified flow loss and literal dual CFG from OCOO-T equations 2,3,5,11."""
import torch


def flow_batch(x, noise_std=1.0, p_mean=-0.8, p_std=0.8, generator=None):
    t = torch.sigmoid(torch.randn(len(x), device=x.device, generator=generator) * p_std + p_mean)
    eps = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator) * noise_std
    return t[:, None] * x + (1 - t[:, None]) * eps, t, x - eps


def guided_velocity(model, z, t, pert, cell, batch, w=None):
    # w=None means the ordinary fully conditional field. w=1 in Eq.11 is
    # NOT the ordinary field: the two half-weight terms also remain active.
    if w is None:
        return model(z, t, pert, cell, batch)
    b = len(z)
    off = torch.zeros(b, dtype=torch.bool, device=z.device)
    on = ~off
    velocities = model(z.repeat(4, 1), t.repeat(4), pert.repeat(4), cell.repeat(4), batch.repeat(4),
                       torch.cat([on, on, off, off]), torch.cat([on, off, on, off])).float()
    vu, vc, vp, vpc = velocities.chunk(4)
    return vu + w / 2 * (vc - vu) + w / 2 * (vp - vu) + w * (vpc - vu)


@torch.no_grad()
def sample(model, pert, cell, batch, n_genes, steps=50, noise_std=1., guidance=None,
           cfg_min=0.1, cfg_max=0.9, seed=0, clamp=False):
    device = pert.device
    rng = torch.Generator(device=device).manual_seed(seed)
    z = torch.randn((len(pert), n_genes), generator=rng, device=device) * noise_std
    dt = 1 / steps
    for i in range(steps):
        t = i / steps
        ts = torch.full((len(z),), t, device=device)
        w = guidance if cfg_min <= t <= cfg_max else None
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
            v = guided_velocity(model, z, ts, pert, cell, batch, w).float()
            if i == steps - 1:  # paper: Euler on final interval
                z = z + dt * v
            else:
                proposal = z + dt * v
                t2 = (i + 1) / steps
                w2 = guidance if cfg_min <= t2 <= cfg_max else None
                v2 = guided_velocity(model, proposal, ts + dt, pert, cell, batch, w2).float()
                z = z + 0.5 * dt * (v + v2)
    return z.clamp_min(0) if clamp else z

