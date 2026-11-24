"""Continuous gene-token Transformer, following OCOO-T §3 and Appendix A.

Unspecified implementation decisions are explicit config options, not paper facts.
The default is the unpatched, 2,000-HVG benchmark network.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    swiglu_convention: str = "param_matched"  # JiT's ratio 4.0 -> 2048
    context_tokens: int = 32
    context_start: int = 3  # zero-based: beginning of layer 4
    drop_prob: float = 0.1
    value_hidden: int = 768
    time_dim: int = 256
    time_scale: float = 1.0
    pos_encoding: bool = True  # shown in Figure 1
    batch_condition: bool = True
    cell_init_std: float = 0.02
    batch_init_std: float = 0.02
    grad_checkpoint: bool = True


def sincos(x, dim, max_period=10000):
    freq = torch.exp(-math.log(max_period) * torch.arange(dim // 2, device=x.device).float() / (dim // 2))
    y = x.float()[..., None] * freq
    return torch.cat([y.cos(), y.sin()], -1)


class RMSNorm(nn.Module):
    def __init__(self, dim, affine=True, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight
        return y.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.qnorm = RMSNorm(self.head_dim)
        self.knorm = RMSNorm(self.head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(self.qnorm(q), self.knorm(k), v, dropout_p=0.0)
        return self.proj(y.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim, ratio, convention):
        super().__init__()
        # Paper says expansion ratio 4.0, but does not state parameter matching.
        hidden = int(dim * ratio * (2 / 3 if convention == "param_matched" else 1))
        self.up = nn.Linear(dim, 2 * hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim)
        self.norm2 = RMSNorm(cfg.dim)
        self.attn = Attention(cfg.dim, cfg.heads)
        self.ffn = SwiGLU(cfg.dim, cfg.mlp_ratio, cfg.swiglu_convention)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, 6 * cfg.dim))

    def forward(self, x, condition):
        s1, a1, g1, s2, a2, g2 = self.ada(condition).chunk(6, -1)
        x = x + g1[:, None] * self.attn(modulate(self.norm1(x), s1, a1))
        return x + g2[:, None] * self.ffn(modulate(self.norm2(x), s2, a2))


class OCOOT(nn.Module):
    def __init__(self, cfg: ModelConfig, embeddings, gene_indices, num_cell_lines, num_batches, missing=None):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("esm", embeddings.float())  # frozen pretrained prior
        self.register_buffer("gene_indices", gene_indices.long())
        self.register_buffer("missing", torch.zeros(len(embeddings), dtype=torch.bool) if missing is None else missing.bool())
        self.gene_proj = nn.Linear(embeddings.shape[1], cfg.dim)
        # Only non-protein/missing genes use these trainable fallback identities.
        self.missing_identity = nn.Embedding(len(embeddings), cfg.dim)
        self.value = nn.Sequential(nn.Linear(1, cfg.value_hidden), nn.SiLU(), nn.Linear(cfg.value_hidden, cfg.dim))
        self.time = nn.Sequential(nn.Linear(cfg.time_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.cell = nn.Embedding(num_cell_lines, cfg.dim)
        self.batch = nn.Embedding(num_batches, cfg.dim) if cfg.batch_condition else None
        self.null_pert = nn.Parameter(torch.zeros(cfg.dim))
        self.null_cell = nn.Parameter(torch.zeros(cfg.dim))
        self.context_pos = nn.Parameter(torch.empty(1, cfg.context_tokens, cfg.dim))
        pos = sincos(torch.arange(len(gene_indices)), cfg.dim)
        self.register_buffer("position", pos if cfg.pos_encoding else torch.zeros_like(pos))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.final_norm = RMSNorm(cfg.dim)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, 2 * cfg.dim))
        self.output = nn.Linear(cfg.dim, 1)
        self.initialize()

    def initialize(self):
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        self.apply(init)
        nn.init.zeros_(self.missing_identity.weight)
        nn.init.normal_(self.cell.weight, std=self.cfg.cell_init_std)
        if self.batch is not None: nn.init.normal_(self.batch.weight, std=self.cfg.batch_init_std)
        nn.init.normal_(self.context_pos, std=0.02)
        for m in [self.time[0], self.time[2]]: nn.init.normal_(m.weight, std=0.02)
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def gene_table(self):
        return self.gene_proj(self.esm) + self.missing_identity.weight * self.missing[:, None]

    def forward(self, z, t, pert, cell, batch, drop_pert=None, drop_cell=None):
        table = self.gene_table()
        p = table[pert]
        c = self.cell(cell)
        if self.batch is not None: c = c + self.batch(batch)
        if drop_pert is None:
            drop_pert = torch.rand(len(z), device=z.device) < self.cfg.drop_prob if self.training else torch.zeros(len(z), device=z.device, dtype=torch.bool)
        if drop_cell is None:
            drop_cell = torch.rand(len(z), device=z.device) < self.cfg.drop_prob if self.training else torch.zeros(len(z), device=z.device, dtype=torch.bool)
        p = torch.where(drop_pert[:, None], self.null_pert.to(p.dtype), p)
        c = torch.where(drop_cell[:, None], self.null_cell.to(c.dtype), c)
        cond = self.time(sincos(t * self.cfg.time_scale, self.cfg.time_dim)) + p + c
        x = self.value(z[..., None]) + table[self.gene_indices][None] + self.position[None]
        for i, block in enumerate(self.blocks):
            if i == self.cfg.context_start:
                context = self.context_pos + (p + c)[:, None]
                x = torch.cat([context.expand(len(z), -1, -1), x], dim=1)
            if self.cfg.grad_checkpoint and self.training:
                x = checkpoint(block, x, cond, use_reentrant=False)
            else:
                x = block(x, cond)
        if self.cfg.context_start < self.cfg.depth:
            x = x[:, self.cfg.context_tokens:]
        shift, scale = self.final_ada(cond).chunk(2, -1)
        return self.output(modulate(self.final_norm(x), shift, scale)).squeeze(-1)

