"""Continuous gene-token Transformer, following OCOO-T §3 and Appendix A.

Unspecified implementation decisions are explicit config options, not paper facts.
The default is the unpatched, 2,000-HVG benchmark network.
"""
from dataclasses import dataclass
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint


# 模型的结构与消融开关集中在此数据类中。
# 默认值对应未修改补丁、包含 2000 个高变基因的基准网络。
@dataclass
class ModelConfig:
    # 主干隐藏维度、层数和注意力头数共同决定网络容量。
    dim: int = 768
    depth: int = 12
    heads: int = 12
    mlp_ratio: float = 4.0
    swiglu_convention: str = "param_matched"  # JiT's ratio 4.0 -> 2048
    # 下面的 context 参数控制全局上下文 token 数量及其插入位置。
    # context_start 为零基索引，默认 3 表示从第 4 层之前插入上下文。
    context_tokens: int = 32
    context_start: int = 3  # zero-based: beginning of layer 4
    # 训练时随机丢弃扰动或细胞条件的概率，用于支持分类器自由引导。
    drop_prob: float = 0.1
    value_hidden: int = 768
    time_dim: int = 256
    time_scale: float = 1.0
    # position_scale 缩放基因位置编码；关闭 pos_encoding 时位置编码为全零。
    pos_encoding: bool = True  # shown in Figure 1
    batch_condition: bool = True
    cell_init_std: float = 0.02
    batch_init_std: float = 0.02
    grad_checkpoint: bool = True
    pert_projection: str = "shared"  # Unspecified in paper; default preserves original runs.
    prediction_space: str = "v"  # Eq.5 main model; Eq.6 x-prediction is an explicit ablation.
    prediction_eps: float = 1e-5  # Numerical guard for Eq.6; inactive on the 50-step sampling grid.
    value_input_init: str = "xavier"  # Unspecified; fan_in changes only the scalar input weight scale.
    drop_batch_with_cell: bool = True  # Unspecified; preserve the original combined-drop default.
    position_scale: float = 1.0
    gene_proj_init_scale: float = 1.0
    null_condition_init: str = "zero"


# 为标量时间或位置生成正余弦周期编码。
# 输入的任意前置维度保持不变，最后一维扩展为 dim 个编码分量。
def sincos(x, dim, max_period=10000):
    freq = torch.exp(-math.log(max_period) * torch.arange(dim // 2, device=x.device).float() / (dim // 2))
    y = x.float()[..., None] * freq
    return torch.cat([y.cos(), y.sin()], -1)


# RMSNorm 按最后一维的均方根缩放张量，可选逐维仿射权重。
# 归一化统计量始终以 float32 累计，以提高低精度训练的数值稳定性。
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


# 多头点积注意力；Query 和 Key 在计算前分别经过 RMSNorm。
# 此实现不接收显式 mask，context 与基因 token 进行双向完全注意力。
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


# SwiGLU 门控前馈网络：SiLU(u) 与并行投影 v逐元素相乘。
# param_matched 模式按 2/3 缩减隐藏宽度，使参数量接近参数匹配实现。
class SwiGLU(nn.Module):
    def __init__(self, dim, ratio, convention):
        super().__init__()
        if convention not in ('param_matched', 'literal'):
            raise ValueError('swiglu_convention must be param_matched or literal')
        # Paper says expansion ratio 4.0, but does not state parameter matching.
        hidden = int(dim * ratio * (2 / 3 if convention == "param_matched" else 1))
        self.up = nn.Linear(dim, 2 * hidden)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.up(x).chunk(2, dim=-1)
        return self.down(F.silu(a) * b)


# AdaLN 调制：先缩放后平移；scale 加 1 可以保留初始恒等缩放。
def modulate(x, shift, scale):
    return x * (1 + scale[:, None]) + shift[:, None]


# Transformer block：前置归一化、AdaLN 调制和两条门控残差分支。
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


# 主模型：组合基因先验、扰动与细胞条件，并逐块处理连续表达基因 token。
class OCOOT(nn.Module):
# 构造冻结基因表、离散条件、时间编码、Transformer 主干及输出头。
    def __init__(self, cfg: ModelConfig, embeddings, gene_indices, num_cell_lines, num_batches, missing=None):
        super().__init__()
        self.cfg = cfg
        if cfg.prediction_space not in ('v', 'x'):
            raise ValueError('prediction_space must be v or x')
        if not 0 < cfg.prediction_eps < 1:
            raise ValueError('prediction_eps must lie in (0,1)')
        if cfg.value_input_init not in ('xavier', 'fan_in'):
            raise ValueError('value_input_init must be xavier or fan_in')
        if cfg.null_condition_init not in ('zero', 'training_mean'):
            raise ValueError('null_condition_init must be zero or training_mean')
        for name in ('position_scale', 'gene_proj_init_scale'):
            scale = getattr(cfg, name)
            if not math.isfinite(scale) or scale <= 0:
                raise ValueError(f'{name} must be positive and finite')
# esm 与 gene_indices 是只读模型数据；missing 记录缺少蛋白质先验的基因。
        self.register_buffer("esm", embeddings.float())  # frozen pretrained prior
        self.register_buffer("gene_indices", gene_indices.long())
        self.register_buffer("missing", torch.zeros(len(embeddings), dtype=torch.bool) if missing is None else missing.bool())
# gene_proj 将预训练蛋白质 embedding 映射到 Transformer 隐藏维度。
        self.gene_proj = nn.Linear(embeddings.shape[1], cfg.dim)
        if cfg.pert_projection not in ('shared', 'separate'):
            raise ValueError('pert_projection must be shared or separate')
        self.pert_proj = None
        # Only non-protein/missing genes use these trainable fallback identities.
# 缺失基因使用可学习 fallback embedding；普通基因由 missing mask 屏蔽。
        self.missing_identity = nn.Embedding(len(embeddings), cfg.dim)
# value 编码标量表达值，time 编码连续时间。
        self.value = nn.Sequential(nn.Linear(1, cfg.value_hidden), nn.SiLU(), nn.Linear(cfg.value_hidden, cfg.dim))
        self.time = nn.Sequential(nn.Linear(cfg.time_dim, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
# cell/batch 为离散条件 embedding；null 参数提供 classifier-free guidance 的空条件。
        self.cell = nn.Embedding(num_cell_lines, cfg.dim)
        self.batch = nn.Embedding(num_batches, cfg.dim) if cfg.batch_condition else None
        self.null_pert = nn.Parameter(torch.zeros(cfg.dim))
        self.null_cell = nn.Parameter(torch.zeros(cfg.dim))
        self.context_pos = nn.Parameter(torch.empty(1, cfg.context_tokens, cfg.dim))
# 每个基因依据序列位置获得正余弦编码，可整体缩放或完全关闭。
        pos = sincos(torch.arange(len(gene_indices)), cfg.dim)
        if cfg.position_scale != 1.0:
            pos.mul_(cfg.position_scale)
        self.register_buffer("position", pos if cfg.pos_encoding else torch.zeros_like(pos))
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.depth))
        self.final_norm = RMSNorm(cfg.dim)
        self.final_ada = nn.Sequential(nn.SiLU(), nn.Linear(cfg.dim, 2 * cfg.dim))
        self.output = nn.Linear(cfg.dim, 1)
        self.initialize()
        if cfg.pert_projection == 'separate':
            # Copy after initialization: no extra RNG consumption, and the
            # initial semantic vectors match the shared-projection control.
            self.pert_proj = copy.deepcopy(self.gene_proj)

    # 对网络各模块施加统一初始化，同时保留若干结构的专用初始化。
    def initialize(self):
        # 普通 Linear 使用 Xavier 初始化并清零 bias。
        def init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
        # 通用 Linear 初始化完成后，再覆盖若干需要特殊取值的参数。
        self.apply(init)
        # 缺失基因 identity 初始为零，使初始状态完全依赖静态先验。
        nn.init.zeros_(self.missing_identity.weight)
        nn.init.normal_(self.cell.weight, std=self.cfg.cell_init_std)
        if self.batch is not None: nn.init.normal_(self.batch.weight, std=self.cfg.batch_init_std)
        nn.init.normal_(self.context_pos, std=0.02)
        # 时间网络的首尾 Linear 使用较小正态分布初始化。
        for m in [self.time[0], self.time[2]]: nn.init.normal_(m.weight, std=0.02)
        # AdaLN 门控和最终输出初始为零，训练起点接近恒等残差或零速度。
        for b in self.blocks:
            nn.init.zeros_(b.ada[-1].weight); nn.init.zeros_(b.ada[-1].bias)
        nn.init.zeros_(self.final_ada[-1].weight); nn.init.zeros_(self.final_ada[-1].bias)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)
        if self.cfg.gene_proj_init_scale != 1.0:
            with torch.no_grad():
                self.gene_proj.weight.mul_(self.cfg.gene_proj_init_scale)
        if self.cfg.value_input_init == 'fan_in':
            # Same random draw, rescaled from Xavier's bound to 1/sqrt(fan_in).
            # Retain zero bias, all other weights, and the original RNG stream.
            layer = self.value[0]
            factor = math.sqrt((layer.in_features + layer.out_features) / (6 * layer.in_features))
            with torch.no_grad():
                layer.weight.mul_(factor)

    # 构造每个基因的静态向量：投影后的蛋白质先验加缺失基因身份向量。
    def gene_table(self, projection=None):
        projection = self.gene_proj if projection is None else projection
        return projection(self.esm) + self.missing_identity.weight * self.missing[:, None]

    # 下列方法使用训练数据设置空条件，不参与梯度计算。
    @torch.no_grad()
    # 用训练集的加权均值初始化丢弃条件，而不是使用任意随机值或全零。
    def initialize_null_conditions(self, pert_ids, cell_ids, batch_ids):
        """Start the dropped conditions at the cell-weighted training means."""
        def population_mean(table, ids):
            counts = torch.bincount(ids, minlength=len(table)).to(table)
            return (counts / counts.sum()) @ table

        self.null_pert.copy_(population_mean(self.gene_table(self.pert_proj), pert_ids))
        cell_mean = population_mean(self.cell.weight, cell_ids)
        if self.batch is not None and self.cfg.drop_batch_with_cell:
            cell_mean += population_mean(self.batch.weight, batch_ids)
        self.null_cell.copy_(cell_mean)

    # 正向传播：读取条件、生成 gene token、处理 Transformer，并转换预测空间。
    def forward(self, z, t, pert, cell, batch, drop_pert=None, drop_cell=None):
        """Return velocity for both network prediction spaces (paper Eqs.5/6)."""
        # table 同时用于静态基因特征和扰动特征；pert_proj 可选择独立投影。
        table = self.gene_table()
        p = (table if self.pert_proj is None else self.gene_table(self.pert_proj))[pert]
        c = self.cell(cell)
        batch_embedding = self.batch(batch) if self.batch is not None else None
        if batch_embedding is not None and self.cfg.drop_batch_with_cell:
            c = c + batch_embedding
        # 训练时随机丢弃，推理时默认不丢弃；显式参数可覆盖该行为。
        if drop_pert is None:
            drop_pert = torch.rand(len(z), device=z.device) < self.cfg.drop_prob if self.training else torch.zeros(len(z), device=z.device, dtype=torch.bool)
        if drop_cell is None:
            drop_cell = torch.rand(len(z), device=z.device) < self.cfg.drop_prob if self.training else torch.zeros(len(z), device=z.device, dtype=torch.bool)
        p = torch.where(drop_pert[:, None], self.null_pert.to(p.dtype), p)
        c = torch.where(drop_cell[:, None], self.null_cell.to(c.dtype), c)
        if batch_embedding is not None and not self.cfg.drop_batch_with_cell:
            c = c + batch_embedding
        # cond 汇总连续时间、扰动和细胞/batch 条件，驱动所有 AdaLN 调制。
        cond = self.time(sincos(t * self.cfg.time_scale, self.cfg.time_dim)) + p + c
        # 每个基因 token 组合当前表达值、静态基因 embedding 和位置编码。
        x = self.value(z[..., None]) + table[self.gene_indices][None] + self.position[None]
        # 在指定层插入 context，并逐块计算；训练时可对 block 使用梯度检查点。
        for i, block in enumerate(self.blocks):
            if i == self.cfg.context_start:
                # context token 保留可学习位置，并叠加当前样本的扰动和细胞条件。
                context = self.context_pos + (p + c)[:, None]
                x = torch.cat([context.expand(len(z), -1, -1), x], dim=1)
            if self.cfg.grad_checkpoint and self.training:
                # 检查点会在反向传播时重算前向，从而降低训练显存占用。
                x = checkpoint(block, x, cond, use_reentrant=False)
            else:
                x = block(x, cond)
        if self.cfg.context_start < self.cfg.depth:
            x = x[:, self.cfg.context_tokens:]
        # 最终预测头使用与 block 相同的 AdaLN 模式调制归一化后的基因表示。
        shift, scale = self.final_ada(cond).chunk(2, -1)
        prediction = self.output(modulate(self.final_norm(x), shift, scale)).squeeze(-1)
        # x-prediction 将目标状态预测转换为流速度，以公式中的剩余时间归一化。
        if self.cfg.prediction_space == 'x':
            denominator = (1 - t.float()).clamp_min(self.cfg.prediction_eps)
            return (prediction.float() - z.float()) / denominator[:, None]
        return prediction

