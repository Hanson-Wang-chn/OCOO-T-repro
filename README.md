# OCOO-T Replogle–Nadig 第三轮复现

本轮固定 Replogle–Nadig 官方划分和 2,000 HVG。第二轮冻结测试的 PDCorr 为 0.40058，MSE 为 0.01107，DEPrec 为 0.20700。论文表 3 分别为 0.437、0.00790、0.240。完整的第二轮结果、参数和当时的 README 已归档于 [repro2.md](repro2.md)，基线提交为 `codex/repro2` 上的 `bfd2ce0`。第三轮在 `codex/repro3` 分支开展。 [论文 v2](https://arxiv.org/html/2606.12838v2)

<!-- LIVE_TRAINING_START -->

三组小实验已完成，主训练采用 σ=0.75 和零初始化 null。正在完成主训练启动前的测试。更新于 2026-11-25T04:47:53+08:00。

<!-- LIVE_TRAINING_END -->

## 改进计划

论文 §3、附录 A.1–A.4 和表 3 分别约束模型、训练、采样和评估。沿用 2,000 HVG 的归一化 log1p 表达、冻结 ESM2、12 层/768 宽/12 头网络、第 4 层起 32 个上下文 token、速度预测 L2、logit-normal `(-0.8, 0.8)`、独立条件 dropout `0.1`、AdamW `β=(0.9,0.95)` 与零权重衰减、全局 batch 512 对应 `lr=1e-4`、5 轮线性 warmup、余弦衰减至 0、双 EMA `0.9999/0.9996`。采样沿用式 11 四个条件分支、50 等步长 Heun 和末步 Euler，生成全部使用 EMA 0.9996。 [论文 v2](https://arxiv.org/html/2606.12838v2)

1. 做三组配对小实验。对照使用第二轮的噪声标准差 1.0 和零初始化的可学习 null embedding。另两组分别将训练和生成的 Gaussian 标准差设为 0.75，或用同一组训练行的条件向量频率加权均值初始化两个现有 null embedding。论文 §3.1 将噪声尺度写为符号，§3.3 规定 null embedding 和 dropout，二者均无指定初始化数值。噪声尺度与训练、生成共用。均值初始化只替换两个参数的起点，且 cell null 的均值包含被同一 dropout 覆盖的 batch 项。
2. 三组各用 seed 42 固定的 16,384 个训练细胞、1,536 次优化更新、四卡 DDP、全局 batch 512。结束时用 EMA 0.9996、`w=1.5`、`[0,1]`、同一组 960 个验证索引与 seed 4242 生成，评估裁负及 0.05 两种设置。1,536 步短于该 EMA 的半衰期，结果只作候选筛选。主筛选用裁负下的 `S = PDCorr + mean(PDS_L1, PDS_L2, PDS_cos) - 10 × MSE`，用 2,000 次按 60 个扰动配对 bootstrap 比较。采用候选须满足 ΔS 的 95% 区间下限大于 0、ΔPDCorr ≥ −0.01、0.05 设置下的 ΔDEPrec ≥ −0.005。若两者达标，采用 ΔS 较大者。否则沿用对照。
3. 选定后从随机初始化做一次全量四卡训练。保留每卡 microbatch 16、累积 8、核心编译和原生 DDP 通信。余弦预算 55 轮，最低 20 轮，周期验证的 `S` 连续 10 轮无 0.002 改善则停。每 5 轮按固定 1,916 个验证细胞做 EMA 生成，周期模型选择使用 `S`。第二轮 E30/E35 的原综合分接近，E30 MSE 低约 0.00035，加入 MSE 权重使模型选择可以反映本轮损失差距。`S` 仅用于验证选择，训练仍优化论文的 L2，正式报告九项 Cell Eval 指标。
4. 用 960 个固定验证样本筛选 `w=1.125,1.25,1.375,1.5,1.625,1.75` 与 `[0.1,0.9]`、`[0,1]` 的组合，普通条件只作诊断。在 `S` 最高与接近最高分且 MSE 最低的两个双 CFG 候选上补齐 4,825 个验证细胞，检查裁负与 0.05，沿用第二轮完整验证门槛及相对短缺规则。仅用验证集冻结权重、采样和输出阈值，再做一次 26,878 个细胞、380 个扰动的完整测试。两个输出阈值共享该次生成的原始预测。

Git 保留源代码、参数、文档、指标和可核验的运行清单。模型权重、原始数据、大数组和逐基因 DE 大表留在工作区。

## 运行配置

- 小实验启动命令为 `CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=6 python scripts/run_repro3_pilots.py`。相同子集的 48 个小轮次每轮 32 步，warmup 为 160 步。完整设置和配置哈希见 [小实验登记](configs/repro3_pilots/plan.json)。
- 独立 dropout 使两项标签条件同时置空的概率为 1%。训练均值初始化直接使用训练细胞的标签频率，核对了两个初值与逐细胞计算的均值一致，且随机数流保持相同。 [模型代码](src/ocoot/model.py)
- CFG 的密集范围参考第二轮完整验证。全时间区间的 `w=2` 在裁负下的 MSE 为 0.01734，`w=1.5` 为 0.01141。较小尺度与更密的网格能检验表达误差和扰动区分之间的取舍。 [第二轮校准](runs/repro2_main/calibration/all_results.json) · [第三轮评测登记](configs/repro3_postprocess.json)

## 小实验结果与主训练配置

三组均完成 1,536 步，四卡训练和末次 EMA 生成各耗时约 64–65 分钟。训练集索引、验证集索引、训练源码、生成种子、完整更新数和 EMA 权重均已核对。表中的截断结果来自每组同一批样本。 [配对比较和运行摘要](runs/repro3_pilots/summary.json) · [六组九指标 CSV](runs/repro3_pilots/summary.csv)

| 指标 | 对照 0 | 对照 0.05 | σ=0.75 0 | σ=0.75 0.05 | 训练均值 null 0 | 训练均值 null 0.05 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDCorr | 0.10877 | 0.10877 | 0.11023 | 0.11022 | 0.11071 | 0.11071 |
| MSE | 0.27225 | 0.27218 | 0.26633 | 0.26635 | 0.30899 | 0.30884 |
| MAE | 0.34989 | 0.34962 | 0.31612 | 0.31588 | 0.40621 | 0.40592 |
| PDS_L1 | 0.50222 | 0.50194 | 0.51111 | 0.51083 | 0.50889 | 0.50944 |
| PDS_L2 | 0.51722 | 0.51722 | 0.51167 | 0.51139 | 0.51417 | 0.51417 |
| PDS_cos | 0.51056 | 0.51083 | 0.51083 | 0.51056 | 0.50778 | 0.50778 |
| DEOver | 0.11465 | 0.11470 | 0.13668 | 0.13666 | 0.09963 | 0.09935 |
| DEPrec | 0.09682 | 0.09806 | 0.10016 | 0.10156 | 0.09319 | 0.09323 |
| DirAgr | 0.50184 | 0.50200 | 0.51828 | 0.51865 | 0.48832 | 0.48840 |

| 候选相对对照 | ΔS（裁负） | 配对 95% 区间 | ΔPDCorr（裁负） | ΔDEPrec（0.05） | 采用 |
| --- | ---: | ---: | ---: | ---: | --- |
| σ=0.75 | +0.06185 | [+0.03795, +0.08648] | +0.00146 | +0.00349 | 是 |
| 训练均值 null | -0.36515 | [-0.39744, -0.33372] | +0.00195 | -0.00483 | 否 |

按登记规则选用 **σ=0.75、零初始化 null**。该候选的初期收益主要来自 MSE，三组的 PDS 都接近随机水平，EMA 仍处于短预算的滞后期。配对区间反映同一训练种子下的验证扰动差异，完整训练和测试将检验最终指标。 [冻结选择](runs/repro3_pilots/main_adoption.json)

从 step 0 开始，覆盖全量 572,545 个训练细胞，每轮 1,118 个优化步。四张 A100 同步训练，采样按原批次在四卡分片。周期 [0.1,0.9]、w=1.5 的 EMA 生成保持与第二轮一致，检查点选择和 CFG 筛选使用预先登记的 MSE 权重 10。 [主训练配置](configs/repro3_main.yaml) · [评测配置](configs/repro3_postprocess.json)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=6 \
  python scripts/run_experiment.py --config configs/repro3_main.yaml --gpu-workers 4 \
  --calibration-scales 1.125,1.25,1.375,1.5,1.625,1.75 --compare-windows \
  --calibration-mse-weight 10 --postprocess-plan configs/repro3_postprocess.json
```

## 实验记录

- 2026-11-25T01:30:49+08:00：启动/续训 `runs/repro3_pilots/control`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro3_pilots/control/config.yaml`。

- 2026-11-25T01:33:31+08:00：CPU 测试 98 项通过，3 项 CUDA 测试因隐藏 GPU 而跳过。共享和独立投影中的 null 初始化、旧模型一致性、CFG 有无 MSE 权重的筛选路径均已覆盖。

- 2026-11-25T01:36:52+08:00：`runs/repro3_pilots/control` 完成第 5/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75203。

- 2026-11-25T01:42:51+08:00：`runs/repro3_pilots/control` 完成第 10/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73612。

- 2026-11-25T01:48:40+08:00：小实验 `control`至 E15.000、step 480，最近吞吐中位数 266.1 cells/s，loss 0.38078，lr 8.73e-05。完整断点 step 448。固定去噪探针 step 448，EMA endpoint MSE 0.72232。

- 2026-11-25T01:48:48+08:00：`runs/repro3_pilots/control` 完成第 15/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.71881。

- 2026-11-25T01:54:47+08:00：`runs/repro3_pilots/control` 完成第 20/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.70074。

- 2026-11-25T02:00:45+08:00：`runs/repro3_pilots/control` 完成第 25/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.68251。

- 2026-11-25T02:03:57+08:00：小实验 `control`至 E27.500、step 880，最近吞吐中位数 265.9 cells/s，loss 0.35545，lr 4.65e-05。完整断点 step 864。固定去噪探针 step 864，EMA endpoint MSE 0.67519。

- 2026-11-25T02:06:43+08:00：`runs/repro3_pilots/control` 完成第 30/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.66404。

- 2026-11-25T02:12:40+08:00：`runs/repro3_pilots/control` 完成第 35/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.64426。

- 2026-11-25T02:18:39+08:00：`runs/repro3_pilots/control` 完成第 40/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.62130。

- 2026-11-25T02:19:25+08:00：小实验 `control`至 E40.625、step 1,300，最近吞吐中位数 265.8 cells/s，loss 0.35392，lr 7.14e-06。完整断点 step 1,280。固定去噪探针 step 1,280，EMA endpoint MSE 0.62130。

- 2026-11-25T02:24:36+08:00：`runs/repro3_pilots/control` 完成第 45/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.59373。

- 2026-11-25T02:34:37+08:00：小实验 `control`至 E47.812、step 1,530，最近吞吐中位数 266.1 cells/s，loss 0.34991，lr 6.39e-09。完整断点 step 1,536。固定去噪探针 step 1,536，EMA endpoint MSE 0.57505。

- 2026-11-25T02:35:17+08:00：`runs/repro3_pilots/control` 第 48 轮验证生成（EMA=0.9996，每扰动最多 16 细胞，50 步）：PDCorr=0.1088，MSE=0.27225，PDS_L2=0.5172。这是验证结果，非测试结果。

- 2026-11-25T02:35:23+08:00：`runs/repro3_pilots/control` 完成第 48/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.57505。

- 2026-11-25T02:35:30+08:00：`runs/repro3_pilots/control` 训练结束（48.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-25T02:36:06+08:00：小实验 `control` 完成 EMA 验证。裁负下 PDCorr=0.10877，MSE=0.27225，0.05 设置下 DEPrec=0.09806。

- 2026-11-25T02:36:12+08:00：启动/续训 `runs/repro3_pilots/sigma_075`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro3_pilots/sigma_075/config.yaml`。

- 2026-11-25T02:42:12+08:00：`runs/repro3_pilots/sigma_075` 完成第 5/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.54619。

- 2026-11-25T02:48:08+08:00：`runs/repro3_pilots/sigma_075` 完成第 10/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.53169。

- 2026-11-25T02:49:38+08:00：小实验 `sigma_075`至 E11.250、step 360，最近吞吐中位数 266.2 cells/s，loss 0.34410，lr 9.49e-05。完整断点 step 352。固定去噪探针 step 352，EMA endpoint MSE 0.52860。

- 2026-11-25T02:54:04+08:00：`runs/repro3_pilots/sigma_075` 完成第 15/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.51613。

- 2026-11-25T03:00:01+08:00：`runs/repro3_pilots/sigma_075` 完成第 20/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.50058。

- 2026-11-25T03:04:26+08:00：小实验 `sigma_075`至 E23.750、step 760，最近吞吐中位数 265.9 cells/s，loss 0.32317，lr 6.01e-05。完整断点 step 736。固定去噪探针 step 736，EMA endpoint MSE 0.49168。

- 2026-11-25T03:05:58+08:00：`runs/repro3_pilots/sigma_075` 完成第 25/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.48602。

- 2026-11-25T03:11:55+08:00：`runs/repro3_pilots/sigma_075` 完成第 30/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.47261。

- 2026-11-25T03:17:52+08:00：`runs/repro3_pilots/sigma_075` 完成第 35/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.46001。

- 2026-11-25T03:19:38+08:00：小实验 `sigma_075`至 E36.250、step 1,160，最近吞吐中位数 266.1 cells/s，loss 0.31779，lr 1.74e-05。完整断点 step 1,152。固定去噪探针 step 1,152，EMA endpoint MSE 0.45751。

- 2026-11-25T03:23:49+08:00：`runs/repro3_pilots/sigma_075` 完成第 40/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.44748。

- 2026-11-25T03:29:46+08:00：`runs/repro3_pilots/sigma_075` 完成第 45/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.43443。

- 2026-11-25T03:34:50+08:00：小实验 `sigma_075`至 E47.812、step 1,530，最近吞吐中位数 266.1 cells/s，loss 0.31229，lr 6.39e-09。完整断点 step 1,536。固定去噪探针 step 1,536，EMA endpoint MSE 0.42630。最近生成 `parts_w4_from0` 分片 32/224 个细胞。

- 2026-11-25T03:40:24+08:00：`runs/repro3_pilots/sigma_075` 第 48 轮验证生成（EMA=0.9996，每扰动最多 16 细胞，50 步）：PDCorr=0.1102，MSE=0.26633，PDS_L2=0.5117。这是验证结果，非测试结果。

- 2026-11-25T03:40:30+08:00：`runs/repro3_pilots/sigma_075` 完成第 48/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.42630。

- 2026-11-25T03:40:36+08:00：`runs/repro3_pilots/sigma_075` 训练结束（48.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-25T03:41:12+08:00：小实验 `sigma_075` 完成 EMA 验证。裁负下 PDCorr=0.11023，MSE=0.26633，0.05 设置下 DEPrec=0.10156。

- 2026-11-25T03:41:18+08:00：启动/续训 `runs/repro3_pilots/null_training_mean`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro3_pilots/null_training_mean/config.yaml`。

- 2026-11-25T03:47:21+08:00：`runs/repro3_pilots/null_training_mean` 完成第 5/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75176。

- 2026-11-25T03:49:57+08:00：小实验 `null_training_mean`至 E6.875、step 220，最近吞吐中位数 265.4 cells/s，loss 0.40488，lr 9.95e-05。完整断点 step 224。固定去噪探针 step 224，EMA endpoint MSE 0.74491。

- 2026-11-25T03:53:20+08:00：`runs/repro3_pilots/null_training_mean` 完成第 10/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73366。

- 2026-11-25T03:59:19+08:00：`runs/repro3_pilots/null_training_mean` 完成第 15/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.71399。

- 2026-11-25T04:05:11+08:00：小实验 `null_training_mean`至 E20.000、step 640，最近吞吐中位数 264.8 cells/s，loss 0.37170，lr 7.3e-05。完整断点 step 608。固定去噪探针 step 640，EMA endpoint MSE 0.69387。

- 2026-11-25T04:05:17+08:00：`runs/repro3_pilots/null_training_mean` 完成第 20/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.69387。

- 2026-11-25T04:11:14+08:00：`runs/repro3_pilots/null_training_mean` 完成第 25/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.67436。

- 2026-11-25T04:17:10+08:00：`runs/repro3_pilots/null_training_mean` 完成第 30/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.65572。

- 2026-11-25T04:20:24+08:00：小实验 `null_training_mean`至 E32.500、step 1,040，最近吞吐中位数 266.2 cells/s，loss 0.35189，lr 2.89e-05。完整断点 step 1,024。固定去噪探针 step 1,024，EMA endpoint MSE 0.64835。

- 2026-11-25T04:23:07+08:00：`runs/repro3_pilots/null_training_mean` 完成第 35/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.63706。

- 2026-11-25T04:29:03+08:00：`runs/repro3_pilots/null_training_mean` 完成第 40/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.61640。

- 2026-11-25T04:35:00+08:00：`runs/repro3_pilots/null_training_mean` 完成第 45/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.59171。

- 2026-11-25T04:35:43+08:00：小实验 `null_training_mean`至 E45.625、step 1,460，最近吞吐中位数 266.1 cells/s，loss 0.34721，lr 7.71e-07。完整断点 step 1,440。固定去噪探针 step 1,440，EMA endpoint MSE 0.59171。

- 2026-11-25T04:45:40+08:00：`runs/repro3_pilots/null_training_mean` 第 48 轮验证生成（EMA=0.9996，每扰动最多 16 细胞，50 步）：PDCorr=0.1107，MSE=0.30899，PDS_L2=0.5142。这是验证结果，非测试结果。

- 2026-11-25T04:45:46+08:00：`runs/repro3_pilots/null_training_mean` 完成第 48/48 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.57450。

- 2026-11-25T04:46:01+08:00：`runs/repro3_pilots/null_training_mean` 训练结束（48.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-25T04:46:36+08:00：小实验 `null_training_mean` 完成 EMA 验证。裁负下 PDCorr=0.11071，MSE=0.30899，0.05 设置下 DEPrec=0.09323。

- 2026-11-25T04:48:21+08:00：主训练前测试完成，101 项通过，包含实际 CUDA 的中断恢复、加权选择及采样缓存复用。详见 `logs/repro3/tests.log`。
