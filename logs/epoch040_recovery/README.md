# E35–E40 四卡故障与恢复

## 已定位的退化

2026-11-22 13:38（北京时间），旧四卡运行在 step 45170 保存完整状态后退出。step 44720 的生成权重与同一步全量断点中的 EMA 0.9996 和四个冻结 buffer 一致。E35 至 E40 的周期验证沿用同一批 1,916 条预测索引、B32、种子 4242 和 Heun 50 步。

| 验证指标 | E35 | 旧 E40 |
| --- | ---: | ---: |
| 普通条件生成 PDCorr | 0.29607 | 0.18819 |
| 普通条件生成 MSE | 0.00943 | 0.09878 |
| 原始权重 endpoint MSE | 0.10441 | 0.10438 |
| 默认 EMA endpoint MSE | 0.10394 | 0.19701 |

从冻结 E40 权重独立加载，在 eager 和编译执行中都复现了 EMA 退化。EMA 0.9999 的 endpoint MSE 约 0.168，亦高于原始权重。替换时间条件模块或主干的混合权重探针用于识别偏差分布。完整记录见 [匹配生成对照](observed_e35_e40/comparison.json)、[E40 冻结复算](probe_step44720.json) 和 [组件探针](probe_components.json)。

## 控制变量回放

使用 step 41374 完整断点、同一训练数据顺序、相同的 t/noise/条件 mask、原 BF16 和 AdamW，回放 41375、41376。各行是相对同机单卡编译参考的全参数梯度 L2 偏差。

| 执行方式 | 41375 | 41376 |
| --- | ---: | ---: |
| 四卡，原 Dynamo DDPOptimizer | 104.93% | 205.40% |
| 两卡，关闭 bucket view | 约 105% | 205.22% |
| 两卡，内层编译及显式规约 | 0.18% | 0.25% |
| 两卡，保留 DDP 并关闭 Dynamo 分图 | 0.20% | 0.22% |
| 三卡，保留 DDP 并关闭 Dynamo 分图 | 0.32% | 0.19% |
| 四卡，保留 DDP 并关闭 Dynamo 分图 | 0.26% | 0.21% |

修复在多卡训练的作用域内关闭 `optimize_ddp`，内层模型仍由 Inductor 编译，原生 DDP 继续规约梯度。关闭 bucket view 单独运行仍复现误差。两卡显式规约、两到四卡关闭分图与单卡参考吻合；修复版两至四卡的梯度校验和在各 rank 间一致。PyTorch 的调试手册也给出了这一分图开关用于定位精度偏差。[四卡原回放](replay_ddp.log) · [修复四卡回放](replay_ddp4_no_ddp_optimizer.log) · [PyTorch 调试手册](https://github.com/pytorch/pytorch/blob/main/.agents/skills/fix-issue/torch_compile_manual.md)

## 重训和检查

配置 `configs/main_ddp_recovery.yaml` 相对原科学配置只更改运行目录。step 41374 的模型、AdamW、双 EMA、随机状态和早停状态均恢复，旧 E35 选型结果随分支保留。当前使用四卡、每卡 microbatch 16、累积 8 次。三卡可分配 11、11、10 次，按全局批次计算梯度均值。

- 对固定的 1,024 个验证细胞比较原始权重与 EMA 的去噪 endpoint MSE。4% 的相对差触发安全保存和暂停，阈值仅用于这次故障监控。
- 对 step 42250 保存单独的完整断点。旧运行在这一步超过 4% 健康阈值。
- step 44720 保存生成前完整断点。生成完成后，CPU 补齐九项并和旧 E35 做相同协议的配对比较。
- 重训结束后，用完整验证集校准最佳检查点，并按预登记质量标准检查九项指标。

[重训来源](../../runs/main_ddp_recovery/recovery_origin.json) · [主训练日志](main_console.log) · [测试日志](pytest_final.log) · [监控状态](monitor_state.json)

重训前四次定点检查的 EMA endpoint MSE 均约 0.104。原运行在 step 42250 的 EMA 为 0.112078，相对 raw 高 7.51%。重训同一步的 EMA 为 0.104006，相对 raw 低 0.27%。完整断点已写入备份盘，并保存同协议健康对照。[逐步数值](health_comparison.json) · [误差曲线](ema_health.png) · [42250 归档](../../runs/main_ddp_recovery/snapshots/step_42250/preserved.json)

监控进程会归档重跑的 E40 完整断点，并在 E40 CPU 九指标齐全后与原 E35 配对。训练完成且配对记录齐全后，调用原校准、全量验证和测试质量门槛。健康告警暂停时会保留完整断点和日志。

## 修复版 E40 核验（15:56）

step 44720 保存前的固定去噪 endpoint MSE 为 raw 0.104294、EMA 0.104050。重跑至 E40 的 16 次匹配探针中，EMA 介于 0.103985 和 0.104051。独立读回 E40 归档和推理模型，189 个 EMA 参数和四个 buffer 全部精确相等。

| 普通条件周期验证 | E35 | 旧 E40 | 修复 E40 |
| --- | ---: | ---: | ---: |
| PDCorr | .29607 | .18819 | .30090 |
| PDS L1/L2/cos | .69806/.71194/.75278 | .57417/.58306/.53806 | .69583/.71111/.74778 |
| MSE | .00943 | .09878 | .00951 |
| MAE | .06781 | .22047 | .06796 |
| DEOver | .09479 | .08986 | .09538 |
| DEPrec | .07845 | .09509 | .07963 |
| DirAgr | .70918 | .58762 | .71209 |

1,916 个预测沿用同一索引、噪声和非负后处理。相对旧 E40，修复 E40 的配对扰动 MSE 差为 −0.08927（95% 区间 −0.09172 至 −0.08695），PDCorr 差为 +0.11271（+0.06106 至 +0.16640）；DEPrec 从旧 E40 的 .09509 降到 .07963。相对 E35，修复 E40 综合分变化 +0.00214，95% 区间 −0.01538 至 +0.01764。结果支持本次故障消除，E35 至修复 E40 的综合改善仍有不确定性。

按训练启动前已有的点估计规则（最小改善 .002），E40 成为新的早停参考轮次。最早可能早停轮次为 E50，继续保持同一训练和验证配置。[逐张量核验](repaired_e40_inference_verification.json) · [旧/新 E40 配对](paired_old_repaired_e40.json) · [E35/修复 E40 配对](repaired_e35_e40/comparison.json)

## E45 后续检查（19:16）

step 50310 的完整状态已归档。EMA 去噪 endpoint MSE 为 0.104207，原始权重为 0.104584。相同 1,916 细胞周期生成的 PDCorr 为 0.30433、MSE 为 0.00943、PDS_L2 为 0.71667，CPU 九指标已齐。E45 相对修复 E40 的综合分差为 +0.00732（95% 配对扰动区间 −0.00412 至 +0.01949）。点估计满足原规则，早停参考更新至 E45，最早可在 E55 停止。E46 训练已开始。[E45 归档](../../runs/main_ddp_recovery/snapshots/step_50310/preserved.json) · [E45 九指标](../../runs/main_ddp_recovery/observation/epoch_045.json) · [E40/E45 配对](repaired_e40_e45/comparison.json)

## E50 后续检查（22:38）

step 55900 完整状态已归档。轮末 raw、EMA 去噪 endpoint MSE 分别为 0.104814、0.104524。普通条件 1,916 细胞生成的 PDCorr 为 0.29683、MSE 为 0.00973，原协议下相对 E45 的综合分差为 −0.02000（95% 配对扰动区间 −0.03586 至 −0.00562）。E45 保持为最佳与早停参考。E50 的 EMA 去噪保持正常范围。E51 继续训练，下一次周期生成在 E55。[E50 归档](../../runs/main_ddp_recovery/snapshots/step_55900/preserved.json) · [E50 九项指标](../../runs/main_ddp_recovery/observation/epoch_050.json) · [E45/E50 配对](repaired_e45_e50/comparison.json)

## 主训练结束（2026-11-23 02:00）

主训练按已有耐心规则在 E55（step 61490）早停，最佳检查点仍为 E45（step 50310）。同协议下 E55 相对 E45 的综合分差 −0.02335（95% 配对扰动区间 −0.04690 至 −0.00058）。E55 默认 EMA 去噪误差 0.104962，仍低于原始权重 0.105244。结束版 last.pt 的 AdamW 步数、双 EMA 和四 rank RNG 已复核。E45 最佳和 E55 最终推理状态的 193 个张量均与各自完整状态精确一致。[训练完成记录](../../runs/main_ddp_recovery/finished.json) · [E55 配对](repaired_e45_e55/comparison.json) · [最终断点核验](final_training_checkpoint_verification.json)

## 完整验证及后续对照（2026-11-23 04:00）

E45 默认 EMA 的两种正尺度 CFG 设置均完成同一 4,825 个细胞、60 个扰动的验证。四个组合中，`w=2`、全区间 `[0,1]`、0.05 截断的综合分最高，PDCorr 为 0.36513，MSE 为 0.01907，DEOver 为 0.19464。DEPrec 为 0.15863，低于预登记的 0.18。验证扰动配对重抽样的 DEPrec 95% 区间为 0.11062 至 0.20493。其余三项组合也有未通过的指标，正式流程记为 `validation_needs_review`。[原检查记录](../../runs/main_ddp_recovery/calibration/postprocessing/review.json) · [独立九指标审计](e45_full_calibration_independent_audit.json)

显著 DE 基因数中位数在实测样本为 28，预测从 0 截断的 1,415.5 降至 0.05 截断的 172。将截断改为 0.07 仅使全区间 DEPrec 增加 0.00034，扰动重抽样区间覆盖零。现登记四组各 960 个验证细胞的对照，检查全引导区间两端和 `w=2` 附近的尺度。全部沿用同一 E45、EMA 0.9996、初始噪声和 Heun 50 步。测试尚未启动。[DE 数量分布](e45_de_count_diagnostic.json) · [0.07 对照](support_floor_diagnostic/refine_00/diagnostic.json) · [引导对照登记](../../runs/main_ddp_recovery/postgate_guidance_study/plan.json)

04:27 四组引导对照已完成。保留前段、去掉末段（`w=2`、`[0,0.9]`）的 DEPrec 为 0.16467，比相同 960 细胞的全区间参考高 0.00146（95% 配对扰动区间 −0.00195 至 +0.00457）。减少起点引导的 DEPrec 下降 0.04387，`w=1.75` 和 `w=2.25` 也没有提升。四个候选均未达到登记的 +0.008 复查标准。继续检查 E45 同一步的两组 EMA，固定普通条件与 `w=2` 全区间各 960 细胞，并保留 0 和 0.05 的结果。[引导对照报告](../../runs/main_ddp_recovery/postgate_guidance_study/report.json) · [双 EMA 对照计划](../../runs/main_ddp_recovery/postgate_ema_study/plan.json)

E35 和 E45 的全区间 `.05` 验证缓存也按 4,825 个相同细胞、B32 和同噪声配对复核。DEPrec 从 0.15419 到 0.15863，变化 +0.00443（95% 区间 −0.00707 至 +0.01798），PDS_L2 变化 +0.02250（+0.00194 至 +0.04778）。DEPrec 缺口在故障前 E35 和修复后 E45 都存在。[全量配对重抽样](e35_to_e45_full_guided.json)

04:56 双 EMA 对照完成。换成 0.9999 后，全区间 `w=2`、0.05 截断在同批验证中的 MSE 从 0.02556 降至 0.02272，DEPrec 从 0.16321 降至 0.13239（配对变化 −0.03082，95% 区间 −0.06477 至 −0.00950），DEOver 同时下降 0.01075。只裁负值时的 DEPrec 变化为 +0.00684，区间为 −0.00065 至 +0.01349。继续保留论文默认 EMA 0.9996。验证质量状态仍为 `validation_needs_review`，预登记检查线保留，未生成正式测试样本。GPU 验证进程已正常退出。[双 EMA 完整报告](../../runs/main_ddp_recovery/postgate_ema_study/report.json) · [正式检查状态](../../runs/main_ddp_recovery/pipeline_state.json)

## 更换 GPU 配额时的续训入口

若需在本机更换 GPU 配额，从 `runs/main_ddp_recovery/last.pt` 续训。核对当前 `training_execution.json` 的 rank 0 PID 和 start tick，向 rank 0 发 SIGTERM，等当前全局步保存、四个 worker 退出后再启动。下例以三卡为例。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=2 \
  NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo \
  TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/ocoot-inductor" \
  "$HOME/miniconda3/envs/ocoo-t/bin/python" -m torch.distributed.run \
  --standalone --nproc-per-node=3 scripts/train.py \
  --config configs/main_ddp_recovery.yaml --max-ema-endpoint-relative-gap .04 \
  --preserve-step 42250 --preserve-step 44720
```

相同 GPU 数从断点逐 rank 恢复 RNG。GPU 数变化时，rank 0 保留已有状态，其他 rank 根据原种子、断点步数和 rank 获得独立随机流。数据顺序、已完成全局步数、学习率日程和 EMA 状态随断点恢复。
