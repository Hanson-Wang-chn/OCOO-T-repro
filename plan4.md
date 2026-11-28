# OCOO-T Replogle–Nadig 第四轮复现改进方案与执行手册（`plan4.md`）

> **文档用途**：本文件面向后续接手的自动化开发与训练监督智能体，提供第四轮复现（分支 `codex/repro4`）的完整根因诊断、代码与配置修改说明书、分阶段小实验与全量重训计划，以及**全程在线监督训练的监测标准与异常处置预案**。
>
> **核心原则**：**严格忠于 OCOO-T v2 原论文方法本身**，不质疑也不改动论文已明确定义的核心算法；本轮所有改动均限定在**修复工程 Bug、纠正代码与论文/JiT 参考实现之间的偏差、在有效训练尺度下重选论文未公开细节，以及修正 CFG 校准与验证选型流程**。

---

## 0. 执行红线与运行环境

### 0.1 必须严格忠于原论文的不可变约束

后续智能体在修改代码、运行小实验及全量重训期间，**不得改动**以下论文核心设定（见 `paper/tex_source_v2/main.tex` §3 与 Appendix A）：

| 模块 | 原论文明确设定（不可更改） |
| --- | --- |
| 流匹配目标 | Rectified flow 线性插值 `z_t = t * x + (1 - t) * eps`、`v-prediction`、速度场 L2 损失（Eqs. 2, 3, 5） |
| 时间采样分布 | Logit-normal：`t = sigmoid(z)`，`z ~ N(-0.8, 0.8²)` |
| 双条件 CFG 公式 | 严格采用 Eq. 11：`v_tilde = vu + (w/2)*(vc - vu) + (w/2)*(vp - vu) + w*(vpc - vu)`，并在指定区间 `[t_min, t_max]` 内启用 |
| ODE 采样器 | Heun 二阶求解器，50 均匀步，最后一步替换为 Euler 步 |
| Transformer 主干 | 12 层 / 宽 768 / 12 头、RMSNorm、QK-RMSNorm、SwiGLU（`mlp_ratio=4.0`，JiT 参数匹配隐层宽 2,048）、AdaLN-zero、32 个 in-context 条件 token、2,000 HVG 不做 patch |
| 优化器与学习率 | AdamW（`betas=(0.9, 0.95)`、`weight_decay=0.0`），全局 `effective_batch=512`，峰值 `lr=1e-4`（`5e-5 × 512 / 256`），前 5 轮线性 warmup 后 cosine 衰减至 0，BF16 混合精度，原生 DDP（保持 `optimize_ddp=False`） |
| EMA 策略 | 维护双 EMA（`0.9999` 和 `0.9996`），正式校准与测试**必须使用默认 `EMA 0.9996`** |
| 数据与评测协议 | 官方 Replogle–Nadig 2,000 HVG 拆分（训练 `572,545` 细胞 / `2,013` 扰动 / 4 细胞系；验证 `4,825` 细胞 / `60` HepG2 扰动；测试 `26,878` 细胞 / `380` HepG2 扰动；共享 `4,976` 个真实 HepG2 对照），使用 Cell Eval 0.8.2 / pdex 0.3.0 计算全部 9 项指标 |

### 0.2 工程与评测纪律

1. **严格测试集隔离**：所有小实验、检查点轮次比选、CFG 尺度/时间窗搜索、近零截断选择，**仅允许在验证集（`validation`）上完成**。正式测试集（`test`）只在第四轮全量重训及验证校准完全冻结后运行**唯一一次**。
2. **向后兼容**：所有代码新增的配置项和 CLI 参数必须保留旧默认值，确保仓库内现有 25 个测试文件（`tests/`）全部通过，不破坏已归档的 `runs/repro3_main/` 核验链。
3. **运行环境与实测吞吐参考**：
   - 分支：`codex/repro4`（基于 `codex/repro3` `983d9b9` 创建）
   - Conda 环境：`/home/whs/miniconda3/envs/ocoo-t`
   - 硬件：4 × NVIDIA A100-PCIE-40GB（GPU 0–3）
   - 实测基线性能（`repro3_main`）：`micro_batch=16`，4 卡各累积 8 次（`effective_batch=512`），`grad_checkpoint=false`，`compile=true`；rank 0 显存峰值 **18.44 GB / 40 GB**，中位吞吐 **265.5 cells/s**，每轮 `1,118` 步约 **36 分钟**（2 卡各累积 16 次约 **70 分钟/轮**，可 2×2 并行同时跑两组小实验）。
   - 验证生成耗时参考（4 卡双 CFG 50 步 Heun，`batch_size=32`）：`cap=16`（960 细胞）约 **6 分钟/组**，全量验证（4,825 细胞）约 **30 分钟/组**，全量测试（26,878 细胞 + DE）约 **2.9 小时**。

---

## 1. 当前复现现状与核心不足诊断

### 1.1 前三轮复现结果与论文 Table 3 对照

| 指标 | 论文 Table 3 | 第一轮（仅验证，未测） | 第二轮（测试 `floor=0.05`） | 第三轮（测试 `floor=0.05`） | 第三轮（测试 `clamp0`） | 第三轮 − 论文 | 状态 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `PDCorr` ↑ | `0.4370` | `0.3651` (val) | `0.4006` | `0.4122` | `0.4132` | `-0.0248` | 待提升 |
| `MSE` ↓ | `0.00790` | `0.01907` (val) | `0.01107` | `0.01010` | `0.01006` | `+0.00220` (+27.8%) | 主要短板 |
| `MAE` ↓ | `0.0610` | `0.08379` (val) | `0.0695` | `0.0673` | `0.0670` | `+0.0063` (+10.3%) | 主要短板 |
| `PDS_L1` ↑ | `0.8200` | `0.8217` (val) | `0.8314` | `0.8359` | `0.8356` | `+0.0159` | 已超论文 |
| `PDS_L2` ↑ | `0.8530` | `0.8344` (val) | `0.8444` | `0.8495` | `0.8495` | `-0.0035` | 接近（372 seen 为 `0.8588`） |
| `PDS_cos` ↑ | `0.8330` | `0.8683` (val) | `0.8052` | `0.8029` | `0.8031` | `-0.0301` | 主要短板 |
| `DEOver` ↑ | `0.2320` | `0.1946` (val) | `0.2389` | `0.2415` | `0.1866` | `+0.0095` | 已超论文 |
| `DEPrec` ↑ | `0.2400` | `0.1586` (val) | `0.2070` | `0.2164` | `0.1127` | `-0.0236` | 主要短板 |
| `DirAgr` ↑ | `0.7660` | `0.8197` (val) | `0.8116` | `0.8211` | `0.8229` | `+0.0551` | 偏高（过强 CFG 伴随特征） |

### 1.2 根因总览：推理侧（CFG）与训练侧（本体过拟合）相互耦合

详细排查表明，剩余差距由**推理侧 CFG 参数/选型失误**和**训练侧模型实现偏差**共同导致，且二者彼此放大：

* **为什么不能只怪训练**：在 `repro3_main` 现有检查点上，CFG 参数过强、未按论文要求限制时间窗，加上选型代码 Bug，直接造成了幅度过冲、细胞系均值偏移、余弦区分度暴跌和 DE 假阳性泛滥。
* **为什么不能只靠调 CFG**：当前模型在**不开 CFG**（`w=None`，`cap=16`）时 `PDCorr` 仅 `0.2408`、`PDS_L2` 仅 `0.6822`；即便换回 `w=1.125`，全量验证 `PDCorr` 仍只有 `0.3942`。正是模型本体跨细胞系泛化偏弱且过早过拟合，才逼迫校准阶段拉高 `w`，最终陷入“`PDCorr/PDS` 与 `MSE/DEPrec/PDS_cos` 二选一”的困境。

### 1.3 已证实的具体问题清单

#### A. 推理与校准侧问题（直接影响 `MSE`、`MAE`、`PDS_cos`、`DEPrec`）

1. **[严重 Bug] `postprocess.py` 忽略 `mse_weight`，静默覆盖了更好的 CFG 选择**：
   - 在 `repro3` 校准中，`scripts/calibrate.py --mse-weight 10` 实际选中的是 `refine_01`（`w=1.125`，`clamp0_selection_score = 1.1146 > 1.1088`）。在全量验证集（`floor=0.05`）上，`w=1.125` 相比 `w=1.375`：`MSE` 从 `0.01068` 降到 `0.00874`（**-18.2%**）、`MAE` 从 `0.06727` 降到 `0.06305`、`DEPrec` 从 `0.1896` 升到 `0.2056`、`PDCorr` 略升（`0.3942` vs `0.3939`），且 DE 基因数与真值高度吻合（`221` vs `211`）。
   - 然而 `src/ocoot/postprocess.py` 第 73–84 行的 `choose_feasible` 硬编码按未惩罚 MSE 的 `primary_score(r['metrics'])` 选优，把 `w=1.375` 硬推入了正式测试。
2. **[严重偏差] 未遵守论文 `[t_min, t_max]` 内部时间窗约束，且 `w` 搜索下限过高**：
   - 论文 §3.3 明确指出需仅在内部时间窗 `[t_min, t_max]` 内开启 CFG，以避免极端噪声端的过度引导；而 `scripts/calibrate.py` 写死了仅比较 `[0.1, 0.9]` 和 `[0.0, 1.0]`，最终在 `repro3` 中选出了全区间 `[0.0, 1.0]`。
   - 在 Eq. 11 下单条件有效外推倍率为 `1.5w`（`w=1.0` 已是 `1.5×`，`w=1.375` 达 `2.06×`），但 `repro3` 从未搜索 `w < 1.125`。
   - **定量实证**：在 `repro3` 测试集上，`w=1.375, [0, 1]` 造成整体扰动幅度过冲 `1.276×`（`pert_spec = 1.304×`）；沿 `HepG2 - AllTrain` 方向 `+4.0%` 的细胞系均值过冲单独拉低 `PDS_cos` `-0.0260`（占总 `PDS_cos` 差距的 86%）；中位 DE 基因数膨胀 `2.3×`（106 vs 46），直接压低 `DEPrec`。GPU 分时间步速度探针进一步证实：验证集在 `t ≥ 0.75` 时全条件速度 `vpc` 反而劣于仅细胞系速度 `vc`（`t=0.96 → 0.98` 速度 MSE 从 `0.125` 跳升至 `0.204`），全程开启 `[0.0, 1.0]` 直接放大了末段噪声。
3. **[未公开细节] 近零连续输出的评估口径**：
   - 真实 log1p 表达含 `55.4%` 精确零且最小非零值为 `0.0728`，连续流匹配生成有 `27.8%` 的微正噪声落在 `(0, 0.05)`，仅 `clamp0` 时触发 Wilcoxon 秩和检验大量假阳性（`DEPrec` 仅 `0.1127`），`floor=0.05` 升至 `0.2164`。本轮仍须在验证集严格比较 `0.0` 和 `0.05`，并在最终报告中同时保留 `clamp0` 对照。

#### B. 训练与模型实现侧问题（直接导致过早过拟合、`PDCorr` 瓶颈及未见扰动崩盘）

4. **[核心训练问题] Epoch 30 后显著过拟合，与论文 Fig. 5 持续提升至 Epoch 60–80 明显背离**：
   - `repro3_main` 的验证 `ema_endpoint_mse` 在 **Epoch 29.52（step 33,000）达到最低点 `0.097652`**，随后单调恶化至 E55 的 `0.09806`；周期验证 `PDCorr` 在 E30 见顶（`0.3351`）后停滞，生成 `MSE` 从 E15（`0.01002`）一路恶化至 E55（`0.01302`）。
   - 分时间步 GPU 探针证实：在训练集 HepG2 上，扰动条件带来的速度误差下降（`mse(vc) - mse(vpc)`）比验证集 HepG2 高 **6.5–42 倍**（`t=0.05/0.25/0.50`：训练 `-0.0075/-0.0071/-0.0042` vs 验证 `-0.0009/-0.0011/-0.0001`），说明模型严重背诵了 `(cell_line, pert)` 组合。
5. **[偏差 1] `cell_init_std = 1.0` 偏离 JiT 默认 `0.02`，破坏条件向量尺度平衡**：
   - 论文所沿用的 JiT 源码（`paper/jit_models.py` 第 299–300 行）对标签嵌入使用 `std = 0.02`。当前 `configs/repro3_main.yaml` 第 32 行因首轮 384 步无效短实验将其设成 `1.0`。
   - 实测 E45 时 `cell.weight` RMS 仍高达 `0.968`（细胞系间距离 `37.6–38.9`），能量约为扰动差异（`0.333`）的 **8.5 倍**、是 `null_cell`（`0.095`）的 **10 倍**、是 `null_pert`（`0.032`）的 **14 倍**，在 `cond = time + p + c` 和 `in-context` 中压制跨细胞系共享扰动语义。
6. **[偏差 2] 漏掉了 JiT 主干中间层的 `attn_drop / proj_drop`**：
   - `paper/jit_models.py` 第 267–272 行在中间一半 Transformer 层（`depth // 4 ≤ i < (depth // 4) * 3`，即 12 层中的第 3–8 层）明确加入 `attn_drop` 和 `proj_drop`；但 `src/ocoot/model.py` 第 89 行将 `dropout_p` 写死为 `0.0` 且无 `proj_drop`。在论文设定 `weight_decay = 0.0` 下，137M 参数模型缺乏主干防过拟合机制。
7. **[偏差 3] `batch`（`gem_group`）被绑进 `drop_cell` 参与 CFG 外推**：
   - 论文 §3.3 Eq. 11 仅对 `y`（扰动）与 `c`（细胞系）做 dropout 和 CFG；§3.4（`main.tex` 第 239 行）明确默认嵌入细胞系/细胞类型，在 PBMC 才额外嵌入 donor。
   - 当前 `src/ocoot/model.py` 第 244–255 行在 Replogle 上既启用 `batch_condition=True`，又默认 `drop_batch_with_cell=True`，使 56 个 `gem_group` 批次嵌入被 Eq. 11 一起放大 `1.5w` 倍——实测生成样本的 batch 效应 RMS 比真实数据膨胀了 **`+41%`**（`0.0837` vs `0.0593`）。
8. **[偏差 4] 共享单层 `gene_proj` 压制扰动表征，8 个全局未见扰动彻底崩盘**：
   - 论文 Fig. 1 将 `Gene encoder` 和 `Perturbation encoder` 画为两个模块；§3.4 描述 Replogle 扰动与基因共享初始 ESM2 语义空间。
   - 当前 `src/ocoot/model.py` 第 154–157 行共享同一个 `Linear(5120, 768)`，每个训练样本承受 **2000:1 的基因 token vs 扰动条件梯度挤压**，且几乎满秩（有效秩 `621/768`）记住了 2,013 个训练扰动。直接后果是测试集中 **8 个全局未见扰动全部崩盘**（`PDCorr = -0.089`、`PDS_L2 = 0.416`、`DEPrec = 0.000`），仅这 8 个扰动就将整体 `PDCorr` 拉低 `-0.0108`、`PDS_L2` 拉低 `-0.0093`。
9. **[偏差 5] 固定 1D `sincos` 位置编码幅度过大**：
   - 论文 Eq. 9 写作 `x_in = h_expr + h_gene`，未给出位置编码项和尺度。当前 `src/ocoot/model.py` 第 171–174 行叠加未缩放 1D `sincos`（RMS = `0.707`），大于训练后的 `h_gene`（`0.440`）与 `value(1)`（`0.396`），初始化阶段更是 `h_expr`（`0.025`）的 28 倍。由于 2,000 个 HVG 顺序无几何拓扑意义，大振幅固定编码容易干扰表达与语义对齐。
10. **[未公开细节 1] 三模式条件 Dropout 下无条件分支 `v_uncond` 训练严重不足**：
    - 论文 §3.3（`main.tex` 第 212 行）要求联合 dropout 覆盖“只丢 `c`、只丢 `y`、两者都丢”三种模式；附录 A 提到 `0.1`。按当前字面独立 `0.1 × 0.1` 执行时，“两者都丢”的 `v_uncond` 仅占 **1%** 训练样本（每 512 batch 仅约 5 个细胞），而它在 Eq. 11 中承担了绝对值最大的权重 `|1 - 2w|`。
11. **[未公开细节 2 & 方法论缺陷] 噪声尺度 `σ_ε`、训练轮数预算与前三轮无效短 Pilot**：
    - 论文 Eq. 2 未公开 `σ_ε`；第三轮仅将 `noise_std` 从 `1.0` 调至 `0.75` 就改善了 8/9 项测试指标，而数据基因内标准差为 `0.457`（全局 `0.680`），更小的 `0.55` 尚未在有效训练长度下验证。
    - 论文 Fig. 5 训练到 80 轮；第三轮预算仅 55 轮。
    - **关键方法论教训**：前三轮全部小实验都在 `16,384` 子集上仅跑 384 或 1,536 步，且第三轮用未预热的 `EMA 0.9996`（`0.9996^1536 = 54.1%` 仍为第 0 步随机初始化！）评估，三组 `PDS ≈ 0.51`（随机水平）。**第四轮严禁再使用 `train_subset` 超短 Pilot！所有小实验必须使用全量 `572,545` 细胞并训练至少 10 full epochs（`11,180` 步，`0.9996^11180 = 1.14%`），跨过扰动学习拐点后再做决策。**

---

## 2. 代码与配置修改说明书（仅供后续智能体按单实施）

> 本章节仅精确说明**哪里有问题、应当如何修改、如何验证**；实施智能体在动手前须先完整阅读对应文件，修改后先跑通全部单元测试再启动任何 GPU 任务。

### 修改项 M1（P0 必改）：修复后处理选型忽略 `mse_weight` 的 Bug

* **涉及文件**：
  - `src/ocoot/postprocess.py`（L44–109）
  - `scripts/select_validation_postprocess.py`（L76–80, L113–191）
  - `scripts/run_experiment.py`（L70–76, L122–140）
* **现有错误**：
  1. `postprocess.py` 的 `choose_feasible`（L84）仅最大化 `primary_score(r['metrics'])`，完全忽略 `calibrate.py` 使用的 `mse_weight`（10.0），导致全量验证阶段把更高 MSE 的 `w=1.375` 顶替了 `w=1.125`。
  2. `select_validation_postprocess.py`（L78–80）硬编码校验 `len(refinements) != 2`，限制了精炼候选扩充到 3 组。
* **修改要求**：
  1. 在 `src/ocoot/postprocess.py` 新增辅助函数：
     ```python
     def selection_score(metrics, mse_weight=0.0):
         return primary_score(metrics) - mse_weight * metrics['MSE']
     ```
     为 `choose_feasible(candidates, thresholds=None, mse_weight=0.0)` 与 `choose_validation_candidate(candidates, thresholds, report_shortfall=False, mse_weight=0.0)` 增加可选参数 `mse_weight=0.0`（默认 `0.0` 保证旧测试 100% 兼容）：
     - `choose_feasible` 排序键改为 `(selection_score(r['metrics'], mse_weight), -r['zero_floor'])`。
     - `choose_validation_candidate` 短缺排序次级键同步改为 `-selection_score(metrics, mse_weight)`。
  2. 在 `scripts/select_validation_postprocess.py` 中：
     - 从 `provenance.get('mse_weight', 0.0)` 读取 `mse_weight`；若 `plan` 包含 `calibration.mse_weight`，强制校验二者相等（防止计划与校准脱节）。
     - 将 `len(refinements) != 2` 放宽为 `len(refinements) < 2`（仍保持 `tag` 与 `(guidance, cfg_min, cfg_max)` 唯一性校验）。
     - 在每个候选和 `review.json` 中同时记录 `score`（原始 `primary_score`）、`mse_weight` 和 `selection_score`，并把 `mse_weight` 透传给 `choose_validation_candidate`。
  3. 同步更新 `scripts/run_experiment.py`，确保 `postprocess_plan` 的 `calibration.mse_weight` 与 `--calibration-mse-weight` 一致。
* **验收测试**：在 `tests/test_postprocess_pipeline.py` 新增测试，用 `repro3` 的真实 4 组候选数字验证：`mse_weight=0.0` 时选 `w=1.375`，`mse_weight=10.0` 时选 `w=1.125`。

### 修改项 M2（P0 必改）：升级 CFG 验证校准支持自定义时间窗与更广 `w` 网格

* **涉及文件**：
  - `scripts/calibrate.py`（L40–163）
  - `scripts/run_experiment.py`（L51–65, L110–120）
* **现有不足**：
  1. `--compare-windows` 硬编码只比较 `[0.1, 0.9]` 和 `[0.0, 1.0]`，无法系统探索避开极端噪声端与 `t ≥ 0.75` 劣化区的论文内部时间窗 `[t_min, t_max]`。
  2. 精炼候选写死只有 2 组，当最优窗口与次优低 MSE 窗口不同时覆盖面不足。
* **修改要求**：
  1. 为 `scripts/calibrate.py` 新增两个向后兼容参数：
     - `--windows`：逗号分隔的 `low:high` 列表，如 `"0.05:0.75,0.05:0.85,0.1:0.8,0.1:0.9,0.0:1.0"`。解析并校验所有 `(low, high)` 满足 `0.0 <= low < high <= 1.0` 且无重复；若未传 `--windows`，则维持旧逻辑（`--compare-windows` 时为 `[0.1, 0.9]` 与 `[0.0, 1.0]`，否则仅 `[0.1, 0.9]` + 最佳尺度补测 `[0.0, 1.0]`）。
     - `--refine-top`：正整数，默认 `2`（兼容旧行为），第四轮设为 `3`。
  2. 精炼候选选取策略（保证多样且不丢失低 MSE 解）：
     - 第 1 组：所有正 `w` 筛查候选中 `selection_score` 最高者；
     - 第 2 组：在距离第 1 组 `selection_score` 不超过 `0.04` 的其余候选中，**`MSE` 最低者**（若无则取次高分）；
     - 第 3 组（当 `refine_top >= 3`）：在前两组未覆盖的候选里，优先选择**不同时间窗**下的最高 `selection_score` 候选（若仅有一个时间窗则取剩余最高分）。
  3. 在 `calibration_config.json` 中完整记录 `screen_windows` 和 `refine_top`，保持所有 SHA256 审计与幂等断点续跑能力。
  4. 注意：`screen_00`（`w=None`）继续保留作为无引导本体性能的诊断基线，不进入正式正 CFG 选优。
  5. 在 `scripts/run_experiment.py` 增加透传参数 `--calibration-windows` 和 `--calibration-refine-top`。
* **验收测试**：在 `tests/test_calibration_policy.py` 新增参数解析、多窗选型与 3-refinement 选择单元测试。

### 修改项 M3（P0 必改）：为模型补齐 JiT 中间层 Dropout 与三模式条件 Dropout 选项

* **涉及文件**：
  - `src/ocoot/model.py`（L18–130, L175, L248–256）
* **现有不足**：
  1. `Attention` 与 `Block` 缺失 JiT 主干中间层的 `attn_drop / proj_drop`（见 `paper/jit_models.py` L107–139, L267–272）。
  2. `OCOOT.forward` 仅支持独立伯努利 dropout，在 `drop_prob=0.1` 时 `v_uncond` 仅覆盖 1% 样本。
* **修改要求（所有新字段默认值必须与旧行为严格等价）**：
  1. 在 `ModelConfig` 增加：
     ```python
     attn_drop: float = 0.0
     proj_drop: float = 0.0
     dropout_layer_policy: str = "jit_middle"  # "jit_middle" (depth//4 <= i < 3*depth//4) 或 "all"
     condition_dropout_mode: str = "independent"  # "independent" 或 "balanced_3mode"
     drop_cell_only_prob: float = 0.08
     drop_pert_only_prob: float = 0.08
     drop_both_prob: float = 0.06
     ```
     在 `OCOOT.__init__` 中校验：`0.0 <= attn_drop < 1.0`、`0.0 <= proj_drop < 1.0`、`dropout_layer_policy in ("jit_middle", "all")`、`condition_dropout_mode in ("independent", "balanced_3mode")`；当为 `"balanced_3mode"` 时，三个概率非负且之和 `< 1.0`。
  2. 在 `Attention.__init__(self, dim, heads, attn_drop=0.0, proj_drop=0.0)` 中：
     - 保存 `self.attn_drop = attn_drop`、`self.proj_drop = nn.Dropout(proj_drop)`；
     - `forward` 中调用 `F.scaled_dot_product_attention(self.qnorm(q), self.knorm(k), v, dropout_p=self.attn_drop if self.training else 0.0)`，并在 `self.proj(...)` 输出后施加 `self.proj_drop(...)`。
  3. 在 `OCOOT.__init__` 构造 `self.blocks` 时，严格按 `paper/jit_models.py` 第 269–270 行的层级区间为每层传入 `attn_drop` 和 `proj_drop`：
     - 若 `dropout_layer_policy == "jit_middle"`：仅在 `cfg.depth // 4 <= i < (cfg.depth // 4) * 3`（12 层即 `i ∈ {3, 4, 5, 6, 7, 8}`）启用非零 dropout，其余层为 `0.0`。
     - 若 `dropout_layer_policy == "all"`：全部层启用。
  4. 在 `OCOOT.forward`（L248–253）实现 `"balanced_3mode"`（**纯张量无 Python 数据依赖分支，确保 `torch.compile(fullgraph=True)` 无 graph break**）：
     - 仅当 `drop_pert is None and drop_cell is None and self.training and self.cfg.condition_dropout_mode == "balanced_3mode"` 时：
       - 采样 `u = torch.rand(len(z), device=z.device)`；
       - `p_both = self.cfg.drop_both_prob`；`p_cell = p_both + self.cfg.drop_cell_only_prob`；`p_pert = p_cell + self.cfg.drop_pert_only_prob`；
       - `drop_both = u < p_both`；
       - `drop_cell = drop_both | ((u >= p_both) & (u < p_cell))`；
       - `drop_pert = drop_both | ((u >= p_cell) & (u < p_pert))`。
     - 当推理阶段显式传入 `drop_pert` / `drop_cell` 时，行为与原实现完全一致。
  5. **复用现有开关（无需改结构代码，直接在 YAML 中切换）**：`cell_init_std`、`batch_condition`、`drop_batch_with_cell`、`pos_encoding`、`position_scale`、`pert_projection`。
* **验收测试**：
  - 在 `tests/` 新增 `test_repro4_model_options.py`：
    1. 默认参数与旧版 `ModelConfig` 前向输出按位/数值等价；
    2. `jit_middle` 模式下仅第 3–8 层有非零 `attn_drop/proj_drop`，`eval()` 模式下输出确定性一致；
    3. `balanced_3mode` 在大样本下四种分支（full / cell-only / pert-only / both）比例符合设定误差 `< 0.01`；
    4. `torch.compile(model, fullgraph=True)` 在 CUDA 上编译前向+反向无报错、无 graph break。

### 修改项 M4（P1 必改）：增强训练可观测性、修复 `best_denoise.pt` 元数据与 EMA 健康检查陷阱

* **涉及文件**：
  - `src/ocoot/train.py`（L62–69, L260–316, L395–423）
  - 新增脚本 `scripts/probe_velocity.py`
* **现有隐患与不足**：
  1. **`best_denoise.pt` 缺少 `weights` 字段**：`train.py` L407 保存 `best_denoise.pt` 时漏写了 `'weights': 'ema_0.9996'`，导致 `scripts/calibrate.py` L71 直接拒绝加载它。
  2. **`--max-ema-endpoint-relative-gap` 缺少预热步数保护**：`train.py` L402–404 从 step 250 就开始比较 `ema_endpoint_mse` 与 `raw_endpoint_mse`；但从头训练时前 10 轮 EMA 尚未追上 raw（例如 `repro3_main` E5 时 EMA 比 raw 高 47%），从头训练若传入 `0.04` 会立刻误停！
  3. **缺少在线过拟合雷达与纯本体（`w=None`）监控**：周期验证只跑单一固定 CFG，看不出模型究竟是“本体变强”还是“只靠强 CFG 推高了 `PDS_cos`”。
* **修改要求**：
  1. 在 `src/ocoot/train.py` 第 407 行保存 `best_denoise.pt` 时补上 `'weights': 'ema_0.9996'`。
  2. 为 `train(...)` 和 `scripts/train.py` 新增可选参数 `ema_health_min_step: int = 12000`（默认 12,000 步，对应约 10.7 轮，`0.9996^12000 = 0.0082`），仅在 `step >= ema_health_min_step` 时才触发 `max_ema_endpoint_relative_gap` 熔断；同时在 `events.jsonl` 持续记录 `ema_to_raw_gap` 供在线监控。
  3. 在 `train.py` 的周期验证中新增两个轻量可选配置（默认关闭，保持兼容）：
     - `val_unguided_epochs`（如设为 `10`）：每 10 轮额外用 `w=None, cap=16` 评估一次纯本体 `EMA 0.9996` 的 `PDCorr / PDS_L2 / MSE`，结果写入 `events.jsonl`（`event: "validation_unguided"`），不干扰 `best_generation.pt` 选型。
     - `pilot_eval_raw_final`（bool，默认 `false`）：全量数据短期小实验（如 10 轮）最后一轮除了评估 `EMA 0.9996`，也额外输出一次 `raw` 权重在相同 `cap=16` 上的指标用于交叉核对（明确标注 `diagnostic_only`，不改变正式流程只认 `ema_0.9996` 的铁律）。
  4. 新增只读诊断脚本 `scripts/probe_velocity.py`：
     - 输入：`--checkpoint`、`--n-cells 512`、`--timesteps 0.05,0.25,0.50,0.75,0.90,0.96`、`--outdir`；
     - 固定种子抽取 512 个训练 HepG2 细胞与 512 个验证 HepG2 细胞（同一批次噪声），计算 4 个 CFG 分支（`vu, vc, vp, vpc`）在每个 `t` 上的速度 MSE，输出 JSON 报告：
       - `gain_train(t) = mse_train(vc) - mse_train(vpc)`
       - `gain_val(t) = mse_val(vc) - mse_val(vpc)`
       - `overfit_ratio(t) = gain_train(t) / max(gain_val(t), 1e-6)`
       - `val_t_cutoff`（验证集 `gain_val(t)` 首次 ≤ 0 的时间点）
       - `vu_mse_val(t)`（衡量无条件分支质量）
     - 单卡运行耗时约 15 秒，用于每个 Pilot 结束后以及正式重训每 10 轮做一次在线过拟合体检。

### 修改项 M5（P1 配置登记）：第四轮配置与后处理计划文件

待 Phase 0 完成后，由执行智能体按以下规范新建配置文件（切勿覆盖 `configs/repro3_*`）：

* `configs/repro4_postprocess.json`：
  - `floors`: `[0.0, 0.05]`（保持不变）
  - `quality_thresholds`: 与 `DEFAULT_THRESHOLDS` 完全一致（不可放松质量门槛）
  - `allow_validation_shortfall`: `false`（第四轮默认必须通过全部验证门槛才放行测试；仅当用户显式批准时才可改为短缺报告）
  - `calibration`: `scales: [0.65, 0.75, 0.85, 0.95, 1.05, 1.15, 1.25]`，`windows: [[0.05, 0.75], [0.05, 0.85], [0.1, 0.8], [0.1, 0.9], [0.0, 1.0]]`，`screen_cap: 16`，`refine_top: 3`，`seed: 4242`，`batch_size: 32`，`mse_weight: 10.0`。
* `configs/repro4_pilots/*.yaml`：详见第 3 节 Pilot Rounds 1–3。
* `configs/repro4_main.yaml`：由 Pilot Round 3 胜出方案填充模型和 `noise_std`，训练参数采用 `epochs: 80`、`warmup_epochs: 5`、`min_epochs: 35`、`early_stop_patience_epochs: 15`、`val_generation_epochs: 5`、`val_unguided_epochs: 10`、`generation_mse_weight: 10.0`，周期验证 `sampling` 采用 Phase 0 锁定的 `(w, cfg_min, cfg_max)`。

---

## 3. 分阶段实验与重训计划

### 总时间与资源预算概览

| 阶段 | 内容 | 数据规模与步数 | GPU 策略 | 预计墙钟耗时 | 核心产出 |
| --- | --- | --- | --- | ---: | --- |
| **Phase 0** | 实施 M1–M4 + 单测 + `repro3_main` 零重训 CFG 校准与探针 | 验证集 only（不碰测试） | 4 卡 | ~3–4 小时 | 量化纯 CFG 收益；锁定后续周期验证默认 `(w, [t_min, t_max])` |
| **Pilot Round 1** | 底座纠偏（`cell_init_std` + `batch` + `pos_encoding`），4 组 | **全量 572,545 细胞**，10 epochs（11,180 步） | 2×2 卡并行（两批） | ~24 小时 | 锁定最佳基础底座 `Base1` |
| **Pilot Round 2** | 抗过拟合与表征（JiT Dropout / `separate` 投影 / 三模式 Dropout），3 组 | **全量 572,545 细胞**，10 epochs（11,180 步） | 2×2 并行 + 4 卡 | ~18 小时 | 锁定改进组合 `Base2` |
| **Pilot Round 3** | 组合集成 + `σ_ε`（`0.75` vs `0.55`）及过拟合延展验证，2 组 | **全量 572,545 细胞**，15 epochs（16,770 步） | 2×2 卡并行 | ~18 小时 | Go/No-Go 门控，锁定 `configs/repro4_main.yaml` |
| **Phase 4** | 正式全量重训（80 epochs）+ 验证校准 + 冻结测试 + 归档 | 全量训练（89,440 步）+ 完整验证 + 单次测试 | 4 卡 DDP | ~54–58 小时 | `runs/repro4_main/results/REPORT.md` 与 `repro4.md` |

> **为什么 Pilot 必须是全量数据 10–15 轮？**
> 依据 `repro3_main` 实测曲线：第 5 轮（5,590 步）时 `EMA 0.9996` 仍残留 `10.7%` 第 0 步随机权重（`PDS_L2 = 0.532`）；到第 10 轮（11,180 步）时仅残留 `1.14%`（`PDCorr = 0.286, PDS_L2 = 0.662, MSE = 0.0102`），到第 15 轮（16,770 步）时 `EMA` 已反超 `raw`（`PDCorr = 0.301, PDS_L2 = 0.739`）。只有在全量数据跨过 10 轮拐点后，对比结果才具备对 80 轮全量训练的预测力。

---

### Phase 0：代码修复、回归验证与零重训 CFG 校准（~3–4 小时）

1. **实施 M1–M4 代码修改并运行回归验证**：
   ```bash
   conda activate ocoo-t
   pytest -q
   ```
   全部旧测试与新增测试必须 100% 通过。随后做一次 10 步 4 卡 `torch.compile` + `attn_drop=0.1, proj_drop=0.1` + `balanced_3mode` 的冒烟测试，并复用 `logs/epoch040_recovery` 的方法核验 4 卡与单卡同批梯度相对 L2 偏差 `< 1%`。
2. **对现有 `repro3_main` 跑分时间步速度探针**：
   - 对 `runs/repro3_main/validation_epoch_030/model.pt`（E30，去噪最优附近）与 `runs/repro3_main/best_generation.pt`（E45）各跑一次 `scripts/probe_velocity.py`，存档于 `runs/repro4_phase0/velocity_probe/` 作为第四轮抗过拟合基准。
3. **对现有 `repro3_main` 跑修复版验证集 CFG 校准（仅验证集，不碰测试集）**：
   - 输出至新目录 `runs/repro4_phase0/repro3_e45_calibration/`（绝不修改 `runs/repro3_main/` 原文件）：
     - `scales`: `0.65,0.75,0.85,0.95,1.05,1.15,1.25`
     - `windows`: `0.05:0.75,0.05:0.85,0.1:0.8,0.1:0.9,0.0:1.0`
     - `mse-weight`: `10.0`，`refine-top`: `3`
   - 紧接着用修复版 `scripts/select_validation_postprocess.py` 在 3 组 full 4,825 验证候选上比较 `floor=0.0` 和 `floor=0.05`。
   - 另用 E45 筛出的前 3 组 `(w, window)`，对 `runs/repro3_main/validation_epoch_030/model.pt`（E30）跑 `cap=16` 验证对照（耗时约 20 分钟），确认“去噪最优轮次 vs 后期轮次”在正确 CFG 下的差异。
4. **Phase 0 决策输出**：
   - 将 Phase 0 全量验证选出的最佳 `(w_0, cfg_min_0, cfg_max_0)` 锁定为后续所有 Pilot 和主训练周期验证的默认采样配置（替代过去过强的 `w=1.5, [0.1, 0.9]`）。
   - 若最优 `w_0` 落在网格下边界 `0.65`，先加测 `w=0.50, 0.55` 确认真正的峰值位置后再锁定。

---

### Pilot Round 1：底座纠偏实验（4 组 × 10 full epochs，~24 小时）

**公共固定设置**：`seed=42`，`train_subset: null`（全量 572,545 细胞），`epochs: 10`（11,180 步），`warmup_epochs: 2`，`effective_batch: 512`（2 卡 × `micro_batch=16` × 16 累积），`lr: 1e-4`，`betas: [0.9, 0.95]`，`weight_decay: 0.0`，`noise_std: 0.75`，`val_generation_epochs: 5`（在 E5、E10 评估），`val_generation_cap: 16`，`pilot_eval_raw_final: true`。

| 编号 | 运行目录 | 相对 `repro3_main` 的改动 | 验证假设 |
| --- | --- | --- | --- |
| `P1_ctrl` | `runs/repro4_pilots/p1_ctrl` | 无（`cell_init_std=1.0`, `batch_condition=true`, `drop_batch_with_cell=true`, `pos_encoding=true, position_scale=1.0`） | 严格同预算基线对照 |
| `P1_cell_nobatch` | `runs/repro4_pilots/p1_cell_nobatch` | `cell_init_std=0.02`, `batch_condition=false` | 恢复 JiT 初始化 + 忠于论文 Fig.1/§3.4 不在 Replogle 注入/放大 batch |
| `P1_cell_batch_noguide` | `runs/repro4_pilots/p1_cell_batch_noguide` | `cell_init_std=0.02`, `batch_condition=true`, `drop_batch_with_cell=false` | 检验保留 batch 作非引导协变量是否优于完全移除 batch |
| `P1_cell_nobatch_nopos` | `runs/repro4_pilots/p1_cell_nobatch_nopos` | `cell_init_std=0.02`, `batch_condition=false`, `pos_encoding=false` | 检验严格遵循 Eq. 9（去掉过大 1D `sincos` 位置编码）的净收益 |

* **并行排程**：第 1 批同时运行 `P1_ctrl`（GPU 0,1）和 `P1_cell_nobatch`（GPU 2,3），约 12 小时；第 2 批同时运行 `P1_cell_batch_noguide`（GPU 0,1）和 `P1_cell_nobatch_nopos`（GPU 2,3），约 12 小时。
* **统一评测包（每组 E10 结束后跑，约 15 分钟）**：
  1. `EMA 0.9996` 在 `cap=16` 验证集上分别评估 **`w=None`** 和 **Phase 0 锁定 CFG**（`floor=0.0` 与 `0.05`）；
  2. `raw` 权重对照（排除 EMA 滞后干扰）；
  3. `scripts/probe_velocity.py` 输出 `gain_val`、`overfit_ratio`、`val_t_cutoff`；
  4. 计算 batch RMS 膨胀率（生成 batch 均值 RMS / 真实 batch 均值 RMS）。
* **晋级规则（锁定 `Base1`）**：
  - 优先采用在 **Phase 0 CFG 选型分 `S = PDCorr + mean(PDS) - 10 * MSE`** 与 **`w=None` 纯本体 `PDCorr/PDS_L2`** 上同时占优、且 `ema_endpoint_mse` 不劣化、batch RMS 膨胀率接近 `1.0` 的配置作为 `Base1`。
  - 若 `P1_cell_batch_noguide` 显著优于 `P1_cell_nobatch` 且 `pos_encoding=false` 亦为正收益，可在进入 Round 2 前将两者自然合并为 `Base1`（若存在交互疑虑，补跑 1 组 10 轮确认）。

---

### Pilot Round 2：抗过拟合与表征泛化实验（3 组 × 10 full epochs，~18 小时）

**公共设置**：全部基于 `Base1`，其余训练参数同 Round 1（全量 572,545 细胞，10 epochs，11,180 步）。

| 编号 | 运行目录 | 相对 `Base1` 的单因素改动 | 核心观察指标 |
| --- | --- | --- | --- |
| `P2_jit_drop` | `runs/repro4_pilots/p2_jit_drop` | `attn_drop=0.1`, `proj_drop=0.1`, `dropout_layer_policy="jit_middle"` | 重点看 **验证 `ema_endpoint_mse`、`overfit_ratio`、`val_t_cutoff` 和 `w=None` 验证 `PDCorr`**（注意：dropout 初期训练 loss 略高属正常，绝不能以 train loss 淘汰！） |
| `P2_sep_pert` | `runs/repro4_pilots/p2_sep_pert` | `pert_projection="separate"` | 重点看整体 `PDCorr/PDS`、**验证集 2 个全局未见扰动得分**，以及 `pert_proj` 有效秩和 `overfit_ratio` |
| `P2_cfg_drop` | `runs/repro4_pilots/p2_cfg_drop` | `condition_dropout_mode="balanced_3mode"`（`0.08 / 0.08 / 0.06`，`v_uncond = 6%`） | 重点看 `vu_mse_val(t)` 是否显著下降，以及 **Phase 0 CFG 下 `S`、`MSE`、`DEPrec` 的权衡**（并顺手快速扫 3 档 `w ∈ {w_0 - 0.15, w_0, w_0 + 0.15}`，因 `v_uncond` 变准后最优 `w` 可能轻微移动） |

* **并行排程**：先 2×2 并行跑 `P2_jit_drop`（GPU 0,1）和 `P2_sep_pert`（GPU 2,3）（~12h），再 4 卡跑 `P2_cfg_drop`（~6h）。
* **采用规则（锁定 `Base2`）**：
  - 每项单因素相对 `Base1` 须满足以下条件才被纳入 `Base2`：
    - `P2_jit_drop`：`overfit_ratio` 明显下降（更接近 1）或 `val_t_cutoff` 提高，且 E10 验证 `S` 不低于 `Base1 - 0.008`（若 10 轮收益胶着，因为正则化优势通常在 E15+ 放大，保留到 Round 3 延展至 15–20 轮再最终拍板，或尝试 `0.05`）。
    - `P2_sep_pert`：E10 验证 `S` ≥ `Base1`（尤其 `PDCorr` 不降、未见扰动有改善），且 `overfit_ratio` 不恶化。
    - `P2_cfg_drop`：`vu_mse_val` 下降，且在同 3 档 `w` 搜索下的最优验证 `S` ≥ `Base1 + 0.005`（或同 `PDCorr/PDS` 下 `MSE` 明显降低）。

---

### Pilot Round 3：组合验证与噪声尺度 `σ_ε` 选型（2 组 × 15 full epochs，~18 小时）

**公共设置**：基于 `Base2` 完整组合，全量 572,545 细胞，**15 full epochs（16,770 步）**，`warmup_epochs: 3`，2×2 卡并行（GPU 0,1 与 GPU 2,3，~18 小时同时完成）。

| 编号 | 运行目录 | 差异 | 目的 |
| --- | --- | --- | --- |
| `P3_combo_s075` | `runs/repro4_pilots/p3_combo_s075` | `noise_std=0.75` | 验证 `Base2` 组合无负交互，并与 `repro3_main` E15 真实历史曲线直接对表 |
| `P3_combo_s055` | `runs/repro4_pilots/p3_combo_s055` | `noise_std=0.55` | 检验更贴近数据真实标准差（基因内 `0.457` / 全局 `0.680`）的 `σ_ε` 是否有进一步净收益 |

> **关键比较原则（避免误杀 `σ_ε = 0.55`）**：
> 1. **不可直接比较不同 `σ_ε` 的 raw `velocity_mse`**（因为目标 `v = x - ε` 的方差随 `σ_ε²` 缩放），比较去噪精度必须看 **`endpoint_mse`**；
> 2. **不同 `σ_ε` 对应的最优 CFG 尺度 `w` 会发生平移**（`σ_ε` 越小时本体信号往往越强、所需 `w` 越低），因此 E15 对比 `P3_combo_s075` 和 `P3_combo_s055` 时，两者各自在 `cap=16` 上快速扫描 4 档 `w ∈ {0.65, 0.85, 1.05, 1.25}`（使用 Phase 0 最佳时间窗，每组仅需约 25 分钟），取各自验证最优 `S` 进行公平对比！

* **进入 Phase 4 全量重训的 Go/No-Go 门控**：
  胜出组合 `Repro4_Final` 在 E15（16,770 步）上必须同时满足：
  1. 相对 `repro3_main` E15 基准（`ema_emse=0.09823`、`PDCorr=0.3005`、`PDS_L2=0.7386`、`MSE=0.01002`），验证生成综合分 `S` 明确更高，且 **`PDCorr` 提升 ≥ +0.010** 或 **`MSE` 下降 ≥ 10% 且 `PDCorr` 不降**；
  2. `probe_velocity.py` 显示 `overfit_ratio` 明显好于 `repro3_main`，且 `val_t_cutoff` ≥ `0.75`；
  3. 全程无非有限梯度、无 EMA 异常漂移。
  - 若未通过门控，**禁止直接启动 50 小时全量重训**：立即按第 4.3 节预案排查具体负交互因素并补做定向短消融。

---

### Phase 4：一轮正式全量重训、完整校准与冻结测试（~54–58 小时）

1. **冻结训练配置 `configs/repro4_main.yaml`**：
   - 模型与 `noise_std`：完整继承 Pilot Round 3 胜出方案 `Repro4_Final`；
   - 训练预算（忠于论文 Fig. 5 曲线）：`epochs: 80`（89,440 步，4 卡约 48 小时），`warmup_epochs: 5`，`min_epochs: 35`，`early_stop_patience_epochs: 15`，`early_stop_min_delta: 0.002`；
   - 周期验证：`val_generation_epochs: 5`，`val_generation_cap: 32`，`val_unguided_epochs: 10`，`generation_mse_weight: 10.0`，`sampling` 使用 Phase 0 锁定的 `(w_0, cfg_min_0, cfg_max_0)`；
   - 健康防护：`--max-ema-endpoint-relative-gap 0.04`（配合 M4 的 `ema_health_min_step=12000`）；
   - 保留每 5 轮的 `validation_epoch_XXX/model.pt`、`best_denoise.pt`、`best_generation.pt` 及 `last.pt`。
2. **启动方式**：
   - 使用 4 卡 DDP 启动训练，并在后台或同会话持续在线监督（详见第 4 节）。
3. **训练完成后的验证轮次复核（仅验证集，~0.5 小时）**：
   - 检查 `best_generation.pt` 所在轮次与 `best_denoise.pt` 所在轮次是否一致；若相差 ≥ 10 轮（例如 E45 vs E70），在**验证集 `cap=16`** 上用 Phase 0 最佳 `(w_0, window_0)` 对两个检查点做配对比较，选择验证 `selection_score` 更高且不过拟合的那个检查点进入正式校准（全程记录在案，绝不触碰测试集）。
4. **完整验证校准 + 后处理选型 + 单次冻结测试（~6 小时）**：
   - 运行升级版 `scripts/calibrate.py`（`scales`、`windows`、`refine-top 3`、`mse-weight 10.0`）→ 修复版 `scripts/select_validation_postprocess.py`（使用 `configs/repro4_postprocess.json`，统一以 `selection_score` 排序并校验全部 8 项原始质量门槛）→ 仅当全部检查通过后，运行唯一一次全量冻结测试（`26,878` 细胞 / `380` 扰动），同时生成 `results/REPORT.md`（验证选定后处理）和 `results_clamp0/REPORT.md`（同批原始样本仅裁负对照）。
5. **归档**：撰写 `repro4.md`（对照表 3 及前 3 轮所有指标、完整改动审计、Pilot 结果、监控日志与代码哈希），在 `codex/repro4` 提交。

---

## 4. 在线训练监督手册与异常处置预案

> 后续智能体在 Pilot 和 Phase 4 训练期间需**全程保持在线监督**，按以下节奏主动巡检、比对基线，并在异常苗头出现时第一时间按预案处置，避免浪费数十小时 GPU 算力。

### 4.1 监督节奏与标准巡检清单

1. **启动后黄金 15 分钟（Step 1–50）必检项**：
   - 核对 `nvidia-smi`：4 张（或 2 张）A100 均处于 ~95–100% 算力占用，单卡显存处于 **18–26 GB**（安全线 `< 36 GB`）；
   - 核对 `runs/<run_dir>/training_execution.json`：`world_size`、`global_batch=512`、`dynamo_ddp_optimizer=false` 全部正确；
   - 核对 `runs/<run_dir>/events.jsonl` 前 10 条 `train` 事件：吞吐正常（4 卡 **240–275 cells/s**；2 卡 **125–145 cells/s**），`loss` 有限且平稳下降，`grad_norm` 有限且无爆炸（初期通常 `0.05–1.5`，进入稳定期后中位数约 `0.08`）。
2. **常规在线巡检（每 30 分钟或每完成 1 个 epoch 巡检一次；严禁写死循环 `sleep` 忙等，应使用定时器或任务通知唤醒）**：
   - 检查训练进程存活（`training.lock` 持有、`training.pid` 存活）与最新 `checkpoint_state.json` 时间戳是否推进（超过 20 分钟无新日志且不在周期验证生成阶段则判定疑似卡死）；
   - 检查最新 `events.jsonl` 中的 `loss`、`lr`、`grad_norm`、`raw_endpoint_mse`、`ema_endpoint_mse`；
   - 检查磁盘剩余空间（确保当前分区剩余 `> 40 GB`，每轮全量重训产物约占 15–20 GB）。
3. **每 5 轮深度体检（周期验证结束后立即执行）**：
   - 读取 `events.jsonl` 最新 `validation_generation` 记录，对照下表 4.2 判断进度是否健康；
   - 每 10 轮在单卡（或暂停间隙 / 空闲显存确认安全后；若 4 卡满载则在训练结束后或复用已落盘 `validation_epoch_XXX/model.pt` 在周期验证完成后立即跑 15 秒）执行 `scripts/probe_velocity.py`，跟踪 `overfit_ratio` 与 `val_t_cutoff` 曲线。

### 4.2 正式全量重训健康里程碑基线（锚定 `repro3_main` 实测曲线）

下表以 `repro3_main`（`noise_std=0.75`）为历史参照，列出第四轮 80 轮主训练在各轮次的预期健康区间与预警红线：

| 轮次（步数） | `repro3` 历史参考（`ema_emse` / `PDCorr` / `PDS_L2` / `MSE`） | 第四轮健康预期（绿灯） | 警告阈值（黄灯，需深度诊断） | 异常阈值（红灯，立即介入） |
| --- | --- | --- | --- | --- |
| **E1** (1,118) | `0.4928` (raw `0.1154`) / — | `raw_emse < 0.118`，`loss` 平滑下降 | `raw_emse > 0.125` 或频繁 `grad_norm > 5` | `NaN/Inf` 或 `raw_emse > 0.15` |
| **E5** (5,590) | `0.1492` (raw `0.1014`) / `0.167` / `0.532` / `0.0396` | `raw_emse < 0.102`，`EMA` 正常追赶（滞后属正常） | `raw_emse > 0.106` | `raw_emse` 不降反升 |
| **E10** (11,180) | `0.1011` (raw `0.0994`) / `0.286` / `0.662` / `0.0102` | `ema_emse ≤ 0.101`，`ema/raw` 差距 `< 2.5%`，`PDCorr ≥ 0.29` | `ema_emse > raw_emse * 1.03` 或 `PDCorr < 0.26` | `ema_emse > raw_emse * 1.04`（触发 EMA 健康停机） |
| **E15** (16,770) | `0.0982` (raw `0.0986`) / `0.301` / `0.739` / `0.0100` | `ema_emse` 反超 `raw_emse`，`PDCorr ≥ 0.31`，`PDS_L2 ≥ 0.74` | `ema_emse` 仍高于 `raw_emse` 或 `PDCorr < 0.29` | `PDS_L2 < 0.65`（条件学习停滞） |
| **E30** (33,540) | `0.0977` (见底) / `0.335` / `0.798` / `0.0118` | `ema_emse ≤ 0.0975` 且**继续下降**，`PDCorr ≥ 0.35`，`MSE ≤ 0.0105` | `ema_emse` 提前见底回升且 `overfit_ratio > 5` | `PDCorr < 0.32` 或 `MSE > 0.014` |
| **E45** (50,310) | `0.0979` (已过拟合) / `0.333` / `0.806` / `0.0125` | `ema_emse` 仍优于 E30 或持平，`PDCorr ≥ 0.365`，`val_t_cutoff ≥ 0.80` | `ema_emse` 连续 3 次周期上升，`MSE` 抬头 | 验证指标全面倒退超过 3% |
| **E60–80** (67k–89k) | `repro3` 未训至此（E55 为 `0.0981` / `0.331`） | 类似论文 Fig. 5 稳步收敛至最佳点，`cap=32` `PDCorr ≥ 0.37–0.40` | 平台期波动（允许正常早停或跑满 cosine） | 末期指标崩塌 |

> 注：若 Pilot Round 3 选用了 `noise_std=0.55`，表中的 `endpoint_mse` 绝对值会与 `0.75` 略有差异，应以 Pilot Round 3 在 E10/E15 实测的 `endpoint_mse` 作为新的相对锚点。

### 4.3 异常处置决策树（If-Then 预案）

当在线监督发现任何异常时，**必须按以下预案处置，禁止拍脑袋魔改论文核心方法**：

| 异常场景 | 判定信号 | 立即处置与调整方案 |
| --- | --- | --- |
| **1. `NaN/Inf` 或梯度爆炸** | 触发 `FloatingPointError`；或稳定期连续出现 `grad_norm > 20` | 1. 保留 `last.pt` 不覆盖，检查异常步前 20 条 `events.jsonl`。<br>2. 先单步复现确认是否由新增 `attn_drop/proj_drop` + `torch.compile` 引发图编译数值错误（关闭 `compile` 复核 20 步）。<br>3. 若为偶发 BF16 梯度尖峰且可稳定复现，从 `last.pt` 续训并在配置中启用保守防护 `max_grad_norm: 1.0`（忠于训练目标，不改 lr/优化器），记录事件。 |
| **2. EMA 异常漂移（DDP 梯度失真复发）** | `step ≥ 12000` 后 `ema_endpoint_mse > raw_endpoint_mse * 1.04` 触发健康停机 | 1. **立即停止续训**，严禁调大阈值硬跑。<br>2. 对照 `logs/epoch040_recovery/README.md`，冻结当前断点运行 1 卡 vs 4 卡同批梯度 L2 偏差核验。<br>3. 核查 `optimize_ddp=False` 是否生效、新增 dropout 是否导致跨 rank RNG 或 DDP bucket 同步不一致；修复并验证偏差 `< 1%` 后再从故障前安全断点接续。 |
| **3. 显存 OOM 或吞吐骤降** | CUDA OOM；或 4 卡吞吐持续 `< 200 cells/s`（2 卡 `< 105 cells/s`） | 1. 检查 `nvidia-smi` 是否有残留僵尸进程占用显存。<br>2. 若因新增 dropout 激活导致峰值显存逼近 40 GB，开启 `grad_checkpoint: true`（保持 `effective_batch=512` 和 `micro_batch=16` 不变；若仍 OOM 再降 `micro_batch=8` 并加倍累积步数，**绝不改变全局 batch=512**）。<br>3. 若吞吐异常低，检查 `TORCH_LOGS=graph_breaks` 确认 `balanced_3mode` 未引入 `torch.compile` 重编译。 |
| **4. 进程卡死（DDP Hang）** | `events.jsonl` 超过 20 分钟无更新且 GPU 利用率归零（排除周期验证/DE 计算时段） | 1. 用 `ps -ef` 和 `nvidia-smi` 确认是 NCCL 通信死锁还是 CPU DE 计算（DE 计算时 GPU 0% 但 CPU 800% 属正常）。<br>2. 确认真卡死后发送 `SIGTERM` 等待优雅保存，超时再终止；核对 `last.pt` 与 `pending_generation_epoch` 完整性后直接幂等续训。 |
| **5. Pilot 阶段出现负交互或收益胶着** | Round 2 的 `P2_jit_drop` 在 E10 与 `Base1` 持平；或 Round 3 的组合分数低于单独某项 | 1. **对 `P2_jit_drop`**：正则化收益往往在 E15–20 才凸显，不要在 E10 草率淘汰——延长至 E20（多花 ~6 小时）看 `ema_endpoint_mse` 与 `overfit_ratio` 拐点，若 `0.1` 偏重则补测 `attn_drop=0.05, proj_drop=0.05`。<br>2. **对 `P3_combo` 负交互**：优先拆开检验 `P2_sep_pert` 与 `P2_cfg_drop` 是否叠加过猛（如退回 `pert_projection=shared` 或降低 `drop_both_prob` 至 `0.04`），确认净收益后再进 Phase 4。 |
| **6. 主训练中期再次出现过拟合** | E30–45 期间 `ema_endpoint_mse` 连续 3 次周期回升，且 `probe_velocity` 显示 `overfit_ratio > 5` 或 `val_t_cutoff < 0.70` | 1. **不立即中断训练**：先完成至少 `min_epochs=35` 并保留全部 5 轮快照 `validation_epoch_XXX/model.pt` 与 `best_denoise.pt`。<br>2. 训练结束后在验证集对比 `best_denoise.pt` 附近轮次与 `best_generation.pt`，择优校准。<br>3. 若最佳轮次校准后全量验证仍显著落后论文且 `overfit_ratio` 居高不下，仅调整 JiT 中间层 `attn_drop/proj_drop`（如 `0.1 → 0.15`）或 `position_scale`，**绝不偷偷加 `weight_decay`**。 |
| **7. 周期验证早停疑似误触发** | 触发 `early_stop_patience_epochs=15` 早停，但日志显示 `ema_endpoint_mse` 和 `w=None` 仍在持续变好 | 说明固定周期验证 CFG（`w_0`）对后期更强模型已经偏强，导致 `MSE` 上升拖累了周期选型分。处置：检查最近 3 个 `validation_epoch_XXX/model.pt` 在更低 `w`（如 `w_0 - 0.15`）下的 `cap=16` 验证分；若确实仍在变好，提高 `early_stop_patience_epochs`（或设 `null` 跑满 80 轮 cosine 预算）从 `last.pt` 继续完成训练。 |
| **8. 校准后验证最优值触及搜索边界，或未通过全部质量门槛** | 最优 `w` 落在 `0.65` 或 `1.25` 边界；或完整验证仍有单项未过 `DEFAULT_THRESHOLDS` | 1. **触及边界**：仅在验证集向外扩充 2 档 `w`（如 `0.50, 0.575` 或 `1.375`）或补充更窄时间窗（如 `[0.05, 0.70]`、`[0.10, 0.70]`），重新运行验证校准与后处理选择。<br>2. **仍未通过质量门槛**：**严禁自动进入测试集！** 停在 `validation_needs_review`，输出完整 9 项验证指标、与 `repro3` 及论文的差异分析、以及失败原因诊断向用户汇报，等待用户批准后再决定是否补实验或做短缺报告测试。 |

---

## 5. 第四轮最终交付验收清单

执行智能体在宣布第四轮复现完成前，须逐项核对以下交付物全部落盘且哈希完整：

- [ ] **代码与测试**：M1–M4 代码修改完成，`pytest -q` 全绿，4 卡 vs 1 卡梯度一致性验证通过。
- [ ] **Phase 0 报告**：`runs/repro4_phase0/` 保存 `repro3_main` E45 升级版校准结果、E30 对照及 `probe_velocity` 基线。
- [ ] **三轮有效小实验记录**：`runs/repro4_pilots/` 保存 Round 1–3 所有全量 10–15 轮实验的 `summary.json`、`probe_velocity` 输出及采纳理由。
- [ ] **全量重训与在线监督归档**：`runs/repro4_main/` 保存完整 `events.jsonl`、`config.yaml`、`best_denoise.pt`、`best_generation.pt`、`last.pt`、每 5 轮验证快照与 `probe_velocity` 轨迹。
- [ ] **纯验证校准与门控记录**：`runs/repro4_main/calibration/` 与 `calibration/postprocessing/` 保存 `all_results.json`、`review.json`、`selected.json`，确认 `test_access: false` 且 `mse_weight` 一致。
- [ ] **单次冻结测试与双后处理报告**：`runs/repro4_main/results/REPORT.md`（验证选定 `zero_floor`）与 `runs/repro4_main/results_clamp0/REPORT.md`（同批原始样本 `clamp0` 对照），附与论文 Table 3 及 `repro2/3` 的配对 bootstrap 比较。
- [ ] **归档文档**：在分支 `codex/repro4` 根目录产出 `repro4.md` 总结全过程，释放全部 GPU（`nvidia-smi` 确认 0 MiB / 0%）。
