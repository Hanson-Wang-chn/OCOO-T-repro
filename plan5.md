# OCOO-T Replogle–Nadig 第五轮复现执行手册（`plan5.md`）

> 分支：`codex/repro5`（基于 `codex/repro3` `983d9b9` 干净分出，第四轮进度完整保留在 `codex/repro4`）。
> 代码与配置已全部修改并完成回归与 4 卡实机验证；监督智能体**无需再改代码或补跑小实验**，直接按本手册执行全量重训、验证校准、单次冻结测试并全程在线监看即可。

---

## 一、本轮改动与已验证状态

所有改动严格忠于 OCOO-T v2 原论文方法（保留 Rectified Flow `v-prediction`、`logit-normal(-0.8, 0.8)`、Eq. 11 双条件 CFG、AdamW 零 weight decay、`effective_batch=512`、`lr=1e-4`、5 轮 warmup + cosine、双 EMA 默认 `0.9996`、50 步 Heun + 末步 Euler、共享 ESM2 投影、独立 `drop_prob=0.1`）：

1. **CFG 选型与校准核心修复（重点）**：
   - 修复 `src/ocoot/postprocess.py` 与 `scripts/select_validation_postprocess.py` 忽略 `mse_weight` 的 Bug，全量验证与筛选统一按 `S = PDCorr + mean(PDS_L1, PDS_L2, PDS_cos) - 10 * MSE` 排序；
   - 升级 `scripts/calibrate.py` 与 `scripts/run_experiment.py`，支持多时间窗 `--windows`、`--screen-cap 24`（降低 `cap=16` 对大 `w` 的有限样本偏高偏差）与 `--refine-top 3`（覆盖最高 `S`、近邻低 MSE、不同时间窗）；
   - `configs/repro5_postprocess.json` 登记 6 档尺度 `[0.75, 0.9, 1.05, 1.15, 1.25, 1.375]` 与 4 个时间窗 `[0.0, 0.75]`、`[0.0, 0.85]`、`[0.05, 0.85]`、`[0.0, 1.0]`（保留 `t=0` 早期 `PDCorr` 增益并切断 `t>0.75/0.85` 末段过冲），默认严格通过全部原始验证门槛后才进入测试；
   - 训练期周期验证改用温和 CFG（`w=1.15, [0.0, 0.85]`），消除 `w=1.5` 偏袒后期过拟合检查点的问题。
2. **忠于论文/JiT 的训练微调（`configs/repro5_main.yaml`）**：
   - `cell_init_std: 0.02`（恢复 JiT 标准）+ `batch_condition: false`（忠于论文 Fig.1/§3.4，避免 CFG 放大批次噪声；第四轮 E5 已实证此两项带来 `PDCorr +44.9%`、`MSE -25.7%`）；
   - JiT 中间六层 `attn_drop: 0.1`、`proj_drop: 0.1`（`dropout_layer_policy: jit_middle`，对齐 `paper/jit_models.py`，缓解零 weight decay 下 E30 后过拟合）；
   - `position_scale: 0.1`（保留 Fig.1 位置编码，避免初始振幅压过表达和 ESM2 语义）；
   - `epochs: 80`（对齐论文 Fig.5）、`min_epochs: 35`、`early_stop_patience_epochs: 15`；每 10 轮附带 `w=None` 无引导诊断；`best_denoise.pt` 补齐 `weights: ema_0.9996`；默认从 `step >= 12000` 起启用 `4%` EMA 健康熔断。
3. **已完成的前置验证**：
   - 单元测试：`pytest -q` **116 项全通过**；
   - 4×A100 实机冒烟与梯度一致性：参数量 `136,876,033`，4 卡 vs 单卡同批梯度相对 L2 偏差 **`0.00000`**（跨 rank 校验和差 `0.000e+00`），实测吞吐 **`270.4 cells/s`**（约 36 分钟/轮），单卡峰值显存 **`18.62 GB / 40 GB`**。

---

## 二、标准执行命令

先确保处于仓库根目录 `/home/dataset-assist-0/whs/ocoo-t`、分支 `codex/repro5`，使用独立环境 `/home/whs/miniconda3/envs/ocoo-t`：

```bash
export PATH=/home/whs/miniconda3/envs/ocoo-t/bin:$PATH
git branch --show-current   # 应为 codex/repro5
nvidia-smi                  # 确认 4 张 A100 空闲
```

### Step 1：启动 4 卡全量重训（预计 ~38–48 小时）

支持断点幂等续训；若中途中断，**直接重跑同一命令**即可从 `last.pt` 无缝接续（切勿加 `--fresh`）：

```bash
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train.py --config configs/repro5_main.yaml
```

> 提示：若希望全程单命令自动衔接校准与测试（不单独做 Step 2 的 `best_denoise.pt` 复核），也可直接运行 Step 3；但**推荐先跑 Step 1，完成后按 Step 2 核对最佳验证轮次，再跑 Step 3**。

### Step 2：训练结束后验证轮次复核与过拟合探针（仅验证集，约 15 分钟）

1. 检查 `best_generation.pt` 与 `best_denoise.pt` 所在步数，并运行速度探针：

```bash
python - <<'PY'
import torch
for name in ("best_generation.pt", "best_denoise.pt"):
    s = torch.load(f"runs/repro5_main/{name}", map_location="cpu", weights_only=False)
    print(name, "step=", s["step"], "epoch=", round(s["step"] / 1118, 2), "weights=", s.get("weights"))
PY

python scripts/probe_velocity.py \
  --checkpoint runs/repro5_main/best_generation.pt \
  --outdir runs/repro5_main/velocity_probe/best_generation
```

2. 若 `best_denoise.pt` 与 `best_generation.pt` 相差 **≥ 5 轮（≥ 5,590 步）**，在验证集 `cap=24` 上用同一套温和 CFG 快速配对比较两者（绝不碰测试集）：

```bash
for NAME in best_generation best_denoise; do
  python -m torch.distributed.run --standalone --nproc-per-node=4 \
    scripts/evaluate.py \
    --checkpoint "runs/repro5_main/${NAME}.pt" \
    --split validation --cap 24 --batch-size 32 \
    --guidance 1.15 --cfg-min 0.0 --cfg-max 0.85 --clamp \
    --outdir "runs/repro5_main/val_ckpt_compare/${NAME}"
done
```

按验证集 `S = PDCorr + (PDS_L1 + PDS_L2 + PDS_cos)/3 - 10 * MSE` 选出更优的 EMA 检查点作为 `SELECTED_CKPT`（默认 `runs/repro5_main/best_generation.pt`）。

### Step 3：验证集 CFG 校准 → 后处理选型 → 质量门控 → 单次冻结测试 → 报告（预计 ~7.5 小时）

`runs/repro5_main/finished.json` 存在时，本命令自动跳过训练，依次完成：24 组 `cap=24` 多窗筛选 + 3 组 4,825 细胞全量精炼（~4.5h）→ CPU `floor=0.0/0.05` 统一 `S` 选型与全部 8 项质量检查 → 单次冻结全量测试 26,878 细胞（~2.9h）→ 生成 `results/REPORT.md` 与 `results_clamp0/REPORT.md`：

```bash
SELECTED_CKPT=runs/repro5_main/best_generation.pt   # 若 Step 2 中 best_denoise 更优则改用之

python scripts/run_experiment.py \
  --config configs/repro5_main.yaml \
  --postprocess-plan configs/repro5_postprocess.json \
  --gpu-workers 4 \
  --calibration-checkpoint "$SELECTED_CKPT" \
  --calibration-scales 0.75,0.9,1.05,1.15,1.25,1.375 \
  --calibration-windows 0.0:0.75,0.0:0.85,0.05:0.85,0.0:1.0 \
  --calibration-screen-cap 24 \
  --calibration-refine-top 3 \
  --calibration-mse-weight 10.0
```

---

## 三、在线监督监测标准与应急处置

### 3.1 常规巡检命令（每 20–30 分钟查看一次，勿忙等 `sleep`）

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
cat runs/repro5_main/checkpoint_state.json
tail -n 8 runs/repro5_main/events.jsonl
```

启动后前 15 分钟确认：4 卡利用率 ~95–100%、单卡显存 ~18–22 GB、吞吐 ~250–275 cells/s、`loss` 与 `grad_norm` 有限且平稳下降。

### 3.2 健康里程碑参考（锚定第三轮 `repro3_main` 实测曲线）

| 轮次（步数） | 第三轮历史参考（`ema_emse` / `PDCorr` / `PDS_L2` / `MSE`） | 第五轮预期健康表现 |
| --- | --- | --- |
| **E5** (5,590) | `0.1492` / `0.167` / `0.532` / `0.0396` | `ema_emse < 0.135`，`PDCorr > 0.20`，`MSE < 0.025`（EMA 滞后属正常） |
| **E10** (11,180) | `0.1011` / `0.286` / `0.662` / `0.0102` | `ema_emse ≤ 0.101`，`ema_to_raw_gap < 3%`，`PDCorr ≥ 0.29` |
| **E15** (16,770) | `0.0982` / `0.301` / `0.739` / `0.0100` | `ema_emse` 反超 `raw_emse`，`PDCorr ≥ 0.31`，`PDS_L2 ≥ 0.74` |
| **E30** (33,540) | `0.0977`（见底）/ `0.335` / `0.798` / `0.0118` | `ema_emse` 持续下降、`PDCorr ≥ 0.345`、`MSE ≤ 0.0105` |
| **E45–80** | 第三轮 E45 起过拟合恶化（`0.0979` → `0.0981`） | 在 JiT dropout 保护下继续平滑收敛，无明显过拟合倒退 |

### 3.3 简明异常处置规则

1. **进程意外退出或卡死**：确认无残留训练进程后，直接重跑 Step 1 同一命令从 `last.pt` 幂等续训。
2. **触发 EMA 健康熔断（`event: "ema_health_stop"`，`step ≥ 12000` 时 `ema_to_raw_gap > 4%`）**：**立即停止，切勿调大阈值硬跑**；检查 `snapshots/before_ema_health_stop_*.pt` 并核查 4 卡梯度一致性后再汇报用户。
3. **周期早停疑似误触发**：若 `finished.json` 显示 `stopped_early: true`，但 `events.jsonl` 中 `ema_endpoint_mse` 与每 10 轮的 `validation_unguided`（`w=None`）仍在持续改善，可用以下命令跑完原 80 轮 cosine 预算：
   ```bash
   python -m torch.distributed.run --standalone --nproc-per-node=4 \
     scripts/train.py --config configs/repro5_main.yaml --finish-schedule
   ```
4. **校准最优 `w` 落到网格边界（`0.75` 或 `1.375`）**：在测试前仅用验证集补测更外侧两档（如 `0.60,0.675` 或 `1.50`）确认峰值，严禁凭测试集调参。
5. **完整验证未通过全部质量门槛（`run_experiment.py` 返回退出码 `2` 并停在测试前）**：**严禁擅自绕过门控或运行测试**；整理 `runs/repro5_main/calibration/postprocessing/review.json` 全部候选指标及差距原因向用户汇报。

---

## 四、完成验收

- 核对 `runs/repro5_main/results/REPORT.md`（验证选定后处理）与 `runs/repro5_main/results_clamp0/REPORT.md`（同批原始样本 `clamp0` 对照），对照论文 Table 3 与 `repro3.md` 全部 9 项指标；
- 产出简洁归档 `repro5.md`，确认全部 GPU 已释放（`nvidia-smi` 为 `0 MiB / 0%`），在 `codex/repro5` 提交结果元数据。
