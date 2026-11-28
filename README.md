# OCOO-T Replogle–Nadig 第四轮复现

本轮在 `codex/repro4` 上从 `983d9b9` 开始，按 [plan4.md](plan4.md) 实施改动、验证和重训。三轮训练小实验均使用官方拆分的全部 572,545 个训练细胞，分别训练 10、10 和 15 轮。模型、检查点、CFG 与近零阈值均只用验证集选取。完整验证通过登记的质量门槛后，才会在冻结配置下运行一次测试生成。

## 当前进度

Phase 0 已通过代码回归和梯度核验。第三轮 E45 校准、上界补测和 E30 对照均已完成。按用户要求在 Pilot Round 1 停止。评测使用 Cell Eval 0.8.2 和 pdex 0.3.0，工作分区可用约 2.1 TiB。 [E45 校准进展](runs/repro4_phase0/repro3_e45_calibration/all_results.json)

<!-- LIVE_TRAINING_START -->

**训练已停止。** 两组试验已保存完整断点，调度进程已退出，四张 GPU 已释放。

<!-- LIVE_TRAINING_END -->

## 实施与选择记录

| 阶段 | 比较内容 | 状态 |
| --- | --- | --- |
| Phase 0 | M1–M4 回归、四卡梯度核验、第三轮 E45 CFG 校准与 E30 对照 | [已完成](runs/repro4_phase0/summary.json) |
| Pilot Round 1 | 同预算对照、细胞系初始化、批次条件、位置编码 | 已停止，保留两组 E5 后断点 |
| Pilot Round 2 | JiT 中间层 Dropout、独立扰动投影、三模式条件 Dropout | 待 Round 1 选定 Base1 |
| Pilot Round 3 | 组合验证与噪声标准差 0.75 / 0.55 | 待 Round 2 选定 Base2 |
| 主实验 | 80 轮预算全量重训、验证校准、单次冻结测试 | 待 Round 3 门控 |

## 代码改动与诊断约定

- 完整验证使用与校准一致的 `S = PDCorr + mean(PDS) − 10 × MSE`。第三轮四个真实候选的回归确认，权重 0 选 `w=1.375`，权重 10 选 `w=1.125`。五个时间窗均配合七档尺度进行筛选，三个全量精炼名额分别覆盖最高分、近邻低 MSE 和另一时间窗。
- 模型新增中间六层 JiT Dropout 和三模式条件 Dropout，旧配置的前向与参数格式通过回归。最佳去噪权重添加 EMA 标签，健康熔断从 step 12,000 开始，新增无引导 EMA 监测及末轮 raw 诊断。最终完整回归通过 115 项测试。四卡冒烟在 step 2–10 的吞吐中位数约 271 cells/s，rank 0 分配显存峰值 18.60 GB。成熟 E45 权重配合两类 Dropout 做同批梯度回放，单卡与四卡相对 L2 偏差 0.578%，四个 rank 的梯度校验和一致。 [回归日志](runs/repro4_phase0/pytest_repro4.log) · [梯度回放](runs/repro4_phase0/gradient_ddp4.json)
- 小实验以 `clamp0` 的 S 比选，统一保留 `floor=0.05` 的九项指标和 raw 权重资料。过拟合程度以 `t=0.05/0.25/0.50` 上 `abs(overfit_ratio − 1)` 的中位数比较，下降超过 10% 作为明确改善。速度探针的偏差比较使用相同抽样和噪声种子。批次 RMS 同时记录同批配对的加权中心化均值和控制扰动构成后的版本。

| 第三轮检查点 | 验证扰动条件首次无收益的 t | 早期 t 的 `overfit_ratio` |
| --- | ---: | --- |
| E15 | 0.90 | 2.74 / 2.64 / 3.47 |
| E30 | 0.96 | 4.01 / 3.45 / 6.61 |
| E45 | 0.50 | 5.89 / 7.38 / 4,410.78 |

E45 在 `t=0.50` 的验证增益为 −0.000024，比例的分母使用登记的 `1e-6` 下限。单卡探针实测约 3 分钟，每组保留 512 个训练 HepG2 细胞与 512 个验证 HepG2 细胞。 [E15 探针](runs/repro4_phase0/velocity_probe/e15/summary.json) · [E30 探针](runs/repro4_phase0/velocity_probe/e30/summary.json) · [E45 探针](runs/repro4_phase0/velocity_probe/e45/summary.json)

## Phase 0 验证结果

周期验证锁定 `w=1.25` 和时间窗 [0.0, 1.0]。E45 全量验证选定阈值 0.05，S 为 1.11327。 [完整候选与速度探针](runs/repro4_phase0/summary.json)

| w | 时间窗 | E30 S | E45 S | E30 PDCorr | E45 PDCorr | E30 MSE | E45 MSE |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.25 | [0.0, 1.0] | 0.93590 | 0.93463 | 0.31550 | 0.31120 | 0.01500 | 0.01584 |
| 0.95 | [0.0, 1.0] | 0.90310 | 0.90239 | 0.29516 | 0.29039 | 0.01336 | 0.01382 |
| 1.25 | [0.05, 0.85] | 0.90767 | 0.91262 | 0.29635 | 0.29320 | 0.01465 | 0.01553 |

## 停止与断点

北京时间 14:18:33 收到停止请求。已停止后续调度并让两组训练完成当前优化步后写入完整断点。已核对优化器步数、双 EMA、每个 rank 的随机状态和 SHA256。 [停止核验](runs/repro4_phase0/stop_verification.json)

| 实验 | 保存步数 | 保存轮次 | E5 PDCorr | E5 PDS_L2 | E5 MSE | 断点 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `p1_ctrl` | 5,731 | 5.126 | 0.14159 | 0.58333 | 0.03106 | [last.pt](runs/repro4_pilots/p1_ctrl/last.pt) |
| `p1_cell_nobatch` | 5,784 | 5.174 | 0.20513 | 0.61861 | 0.02308 | [last.pt](runs/repro4_pilots/p1_cell_nobatch/last.pt) |

两个试验的 E10 比选尚未完成。第二批小实验及后续重训与正式测试尚未启动。

## 在线记录

- 2026-11-28 00:11（北京时间）核对分支、运行环境和检查点，开始实现 M1–M4。长时间运行期间每 15 分钟在此记录进度，异常和阶段决策随发现更新。


- 2026-11-28T00:26:35+08:00：启动/续训 `runs/repro4_phase0/ddp_smoke`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro4_phase0/ddp_smoke/config.yaml`。

- 2026-11-28T00:27:37+08:00：`runs/repro4_phase0/ddp_smoke` 收到停止信号，完成当前优化步 10 后安全保存。

- 2026-11-28T00:28:55+08:00：Phase 0 完成回归与 DDP 核验。113 项测试通过。4 卡冒烟完成 10 步，同批梯度 L2 相对偏差 0.578%。首次回归发现后处理模块导入时修改 CUDA 环境，已移除该副作用并全量复测。

- 2026-11-28T00:34:12+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2408，PDS_L2=0.6822，MSE=0.01391（非测试）。

- 2026-11-28T00:40:07+08:00：验证集 CFG 校准 `screen_01`：w=0.65，区间 [0.05,0.75]，PDCorr=0.2485，PDS_L2=0.6886，MSE=0.01350（非测试）。

- 2026-11-28T00:44:02+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2144 GiB。

- 2026-11-28T00:45:27+08:00：验证集 CFG 校准 `screen_02`：w=0.75，区间 [0.05,0.75]，PDCorr=0.2593，PDS_L2=0.7136，MSE=0.01347（非测试）。

- 2026-11-28T00:46:37+08:00：补充覆盖 EMA 健康检查的预热、熔断前断点及末轮 raw / 无引导诊断，最终回归 115 项全通过。日志见 `runs/repro4_phase0/pytest_repro4.log`。

- 2026-11-28T00:50:50+08:00：验证集 CFG 校准 `screen_03`：w=0.85，区间 [0.05,0.75]，PDCorr=0.2688，PDS_L2=0.7367，MSE=0.01358（非测试）。

- 2026-11-28T00:56:15+08:00：验证集 CFG 校准 `screen_04`：w=0.95，区间 [0.05,0.75]，PDCorr=0.2769，PDS_L2=0.7481，MSE=0.01384（非测试）。

- 2026-11-28T00:59:03+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2144 GiB。

- 2026-11-28T01:01:40+08:00：验证集 CFG 校准 `screen_05`：w=1.05，区间 [0.05,0.75]，PDCorr=0.2831，PDS_L2=0.7550，MSE=0.01423（非测试）。

- 2026-11-28T01:07:03+08:00：验证集 CFG 校准 `screen_06`：w=1.15，区间 [0.05,0.75]，PDCorr=0.2883，PDS_L2=0.7700，MSE=0.01478（非测试）。

- 2026-11-28T01:12:25+08:00：验证集 CFG 校准 `screen_07`：w=1.25，区间 [0.05,0.75]，PDCorr=0.2922，PDS_L2=0.7781，MSE=0.01542（非测试）。

- 2026-11-28T01:14:04+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2144 GiB。

- 2026-11-28T01:18:18+08:00：验证集 CFG 校准 `screen_window_01_01`：w=0.65，区间 [0.05,0.85]，PDCorr=0.2488，PDS_L2=0.6897，MSE=0.01349（非测试）。

- 2026-11-28T01:24:10+08:00：验证集 CFG 校准 `screen_window_01_02`：w=0.75，区间 [0.05,0.85]，PDCorr=0.2600，PDS_L2=0.7144，MSE=0.01347（非测试）。

- 2026-11-28T01:29:05+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 0%、GPU 3 11273 MiB / 0%。工作分区可用 2143 GiB。

- 2026-11-28T01:30:02+08:00：验证集 CFG 校准 `screen_window_01_03`：w=0.85，区间 [0.05,0.85]，PDCorr=0.2695，PDS_L2=0.7367，MSE=0.01358（非测试）。

- 2026-11-28T01:35:55+08:00：验证集 CFG 校准 `screen_window_01_04`：w=0.95，区间 [0.05,0.85]，PDCorr=0.2777，PDS_L2=0.7472，MSE=0.01386（非测试）。

- 2026-11-28T01:41:49+08:00：验证集 CFG 校准 `screen_window_01_05`：w=1.05，区间 [0.05,0.85]，PDCorr=0.2840，PDS_L2=0.7558，MSE=0.01427（非测试）。

- 2026-11-28T01:44:06+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T01:47:43+08:00：验证集 CFG 校准 `screen_window_01_06`：w=1.15，区间 [0.05,0.85]，PDCorr=0.2891，PDS_L2=0.7708，MSE=0.01485（非测试）。

- 2026-11-28T01:53:39+08:00：验证集 CFG 校准 `screen_window_01_07`：w=1.25，区间 [0.05,0.85]，PDCorr=0.2932，PDS_L2=0.7811，MSE=0.01553（非测试）。

- 2026-11-28T01:59:06+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 0%、GPU 1 11273 MiB / 0%、GPU 2 11233 MiB / 0%、GPU 3 11273 MiB / 0%。工作分区可用 2143 GiB。

- 2026-11-28T01:59:08+08:00：验证集 CFG 校准 `screen_window_02_01`：w=0.65，区间 [0.1,0.8]，PDCorr=0.2480，PDS_L2=0.6914，MSE=0.01352（非测试）。

- 2026-11-28T02:04:38+08:00：验证集 CFG 校准 `screen_window_02_02`：w=0.75，区间 [0.1,0.8]，PDCorr=0.2554，PDS_L2=0.7103，MSE=0.01355（非测试）。

- 2026-11-28T02:10:08+08:00：验证集 CFG 校准 `screen_window_02_03`：w=0.85，区间 [0.1,0.8]，PDCorr=0.2621，PDS_L2=0.7308，MSE=0.01368（非测试）。

- 2026-11-28T02:14:07+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T02:15:38+08:00：验证集 CFG 校准 `screen_window_02_04`：w=0.95，区间 [0.1,0.8]，PDCorr=0.2690，PDS_L2=0.7450，MSE=0.01392（非测试）。

- 2026-11-28T02:21:07+08:00：验证集 CFG 校准 `screen_window_02_05`：w=1.05，区间 [0.1,0.8]，PDCorr=0.2743，PDS_L2=0.7533，MSE=0.01427（非测试）。

- 2026-11-28T02:26:37+08:00：验证集 CFG 校准 `screen_window_02_06`：w=1.15，区间 [0.1,0.8]，PDCorr=0.2785，PDS_L2=0.7611，MSE=0.01471（非测试）。

- 2026-11-28T02:29:08+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T02:32:07+08:00：验证集 CFG 校准 `screen_window_02_07`：w=1.25，区间 [0.1,0.8]，PDCorr=0.2829，PDS_L2=0.7736，MSE=0.01524（非测试）。

- 2026-11-28T02:38:10+08:00：验证集 CFG 校准 `screen_window_03_01`：w=0.65，区间 [0.1,0.9]，PDCorr=0.2482，PDS_L2=0.6919，MSE=0.01351（非测试）。

- 2026-11-28T02:44:09+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 0%、GPU 1 11273 MiB / 0%、GPU 2 11233 MiB / 0%、GPU 3 11273 MiB / 0%。工作分区可用 2143 GiB。

- 2026-11-28T02:44:10+08:00：验证集 CFG 校准 `screen_window_03_02`：w=0.75，区间 [0.1,0.9]，PDCorr=0.2558，PDS_L2=0.7111，MSE=0.01354（非测试）。

- 2026-11-28T02:49:50+08:00：验证集 CFG 校准 `screen_window_03_03`：w=0.85，区间 [0.1,0.9]，PDCorr=0.2626，PDS_L2=0.7314，MSE=0.01368（非测试）。

- 2026-11-28T02:55:31+08:00：验证集 CFG 校准 `screen_window_03_04`：w=0.95，区间 [0.1,0.9]，PDCorr=0.2696，PDS_L2=0.7450，MSE=0.01392（非测试）。

- 2026-11-28T02:59:10+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T03:01:14+08:00：验证集 CFG 校准 `screen_window_03_05`：w=1.05，区间 [0.1,0.9]，PDCorr=0.2746，PDS_L2=0.7533，MSE=0.01430（非测试）。

- 2026-11-28T03:07:02+08:00：验证集 CFG 校准 `screen_window_03_06`：w=1.15，区间 [0.1,0.9]，PDCorr=0.2790，PDS_L2=0.7614，MSE=0.01475（非测试）。

- 2026-11-28T03:13:03+08:00：验证集 CFG 校准 `screen_window_03_07`：w=1.25，区间 [0.1,0.9]，PDCorr=0.2833，PDS_L2=0.7742，MSE=0.01530（非测试）。

- 2026-11-28T03:14:11+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T03:19:58+08:00：验证集 CFG 校准 `screen_full_01`：w=0.65，区间 [0.0,1.0]，PDCorr=0.2498，PDS_L2=0.6842，MSE=0.01347（非测试）。

- 2026-11-28T03:26:53+08:00：验证集 CFG 校准 `screen_full_02`：w=0.75，区间 [0.0,1.0]，PDCorr=0.2670，PDS_L2=0.7150，MSE=0.01334（非测试）。

- 2026-11-28T03:29:12+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2143 GiB。

- 2026-11-28T03:33:47+08:00：验证集 CFG 校准 `screen_full_03`：w=0.85，区间 [0.0,1.0]，PDCorr=0.2802，PDS_L2=0.7381，MSE=0.01347（非测试）。

- 2026-11-28T03:40:41+08:00：验证集 CFG 校准 `screen_full_04`：w=0.95，区间 [0.0,1.0]，PDCorr=0.2904，PDS_L2=0.7500，MSE=0.01382（非测试）。

- 2026-11-28T03:44:13+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T03:47:33+08:00：验证集 CFG 校准 `screen_full_05`：w=1.05，区间 [0.0,1.0]，PDCorr=0.2986，PDS_L2=0.7700，MSE=0.01438（非测试）。

- 2026-11-28T03:54:22+08:00：验证集 CFG 校准 `screen_full_06`：w=1.15，区间 [0.0,1.0]，PDCorr=0.3062，PDS_L2=0.7847，MSE=0.01507（非测试）。

- 2026-11-28T03:59:14+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T04:01:12+08:00：验证集 CFG 校准 `screen_full_07`：w=1.25，区间 [0.0,1.0]，PDCorr=0.3112，PDS_L2=0.7892，MSE=0.01584（非测试）。

- 2026-11-28T04:09:08+08:00：Phase 0 已完成 36 组 cap=16 筛选。按最高 S、近邻低 MSE 与不同时间窗依次精炼 w=1.25 / [0.0, 1.0]、w=0.95 / [0.0, 1.0]、w=1.25 / [0.05, 0.85]，每组补齐 4,825 个验证细胞。

- 2026-11-28T04:14:15+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T04:29:16+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T04:32:50+08:00：验证集 CFG 校准 `refine_00`：w=1.25，区间 [0.0,1.0]，PDCorr=0.3957，PDS_L2=0.8156，MSE=0.00957（非测试）。

- 2026-11-28T04:44:16+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T04:59:17+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T05:04:13+08:00：验证集 CFG 校准 `refine_01`：w=0.95，区间 [0.0,1.0]，PDCorr=0.3939，PDS_L2=0.7922，MSE=0.00790（非测试）。

- 2026-11-28T05:14:18+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T05:29:19+08:00：巡检。runs/repro4_phase0/repro3_e45_calibration 验证 CFG 校准。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 0%。工作分区可用 2142 GiB。

- 2026-11-28T05:31:15+08:00：验证集 CFG 校准 `refine_02`：w=1.25，区间 [0.05,0.85]，PDCorr=0.3829，PDS_L2=0.8094，MSE=0.00913（非测试）。

- 2026-11-28T05:31:15+08:00：`runs/repro4_phase0/repro3_e45_calibration` 完成验证集校准，共 3 个精炼候选，完整记录见 `all_results.json`。

- 2026-11-28T05:39:14+08:00：完成 `runs/repro4_phase0/repro3_e30_comparison/refine_00` 的 validation 评估（细胞 960，weights=ema_0.9996，CFG=1.25）；指标见 `runs/repro4_phase0/repro3_e30_comparison/refine_00/metrics/summary.json`。PDCorr=0.3155，MSE=0.01500。

- 2026-11-28T05:45:17+08:00：巡检。runs/repro4_phase0/repro3_e30_comparison/refine_01 cap=16 验证生成。GPU 0 10681 MiB / 100%、GPU 1 10681 MiB / 97%、GPU 2 10681 MiB / 0%、GPU 3 10681 MiB / 0%。工作分区可用 2142 GiB。

- 2026-11-28T05:46:23+08:00：完成 `runs/repro4_phase0/repro3_e30_comparison/refine_01` 的 validation 评估（细胞 960，weights=ema_0.9996，CFG=0.95）；指标见 `runs/repro4_phase0/repro3_e30_comparison/refine_01/metrics/summary.json`。PDCorr=0.2952，MSE=0.01336。

- 2026-11-28T05:53:03+08:00：完成 `runs/repro4_phase0/repro3_e30_comparison/refine_02` 的 validation 评估（细胞 960，weights=ema_0.9996，CFG=1.25）；指标见 `runs/repro4_phase0/repro3_e30_comparison/refine_02/metrics/summary.json`。PDCorr=0.2964，MSE=0.01465。

- 2026-11-28T05:53:06+08:00：Phase 0 初始全量验证优选 CFG，w=1.25，时间窗 [0.0, 1.0]。全量验证 S=1.11327，PDCorr=0.39447，MSE=0.00960，近零阈值 0.05。E30、E45 及探针结果见 `runs/repro4_phase0/summary.json`。

- 2026-11-28T05:53:08+08:00：Phase 0 初始全量验证优选 w=1.25 触及上界。依据异常预案，在所选窗口和得分最高的另一窗口 ((0.0, 1.0), (0.05, 0.85)) 补测 w=1.375、1.50。

- 2026-11-28T05:55:15+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2408，PDS_L2=0.6822，MSE=0.01391（非测试）。

- 2026-11-28T06:02:36+08:00：验证集 CFG 校准 `screen_01`：w=1.375，区间 [0.0,1.0]，PDCorr=0.3170，PDS_L2=0.7994，MSE=0.01695（非测试）。

- 2026-11-28T06:08:09+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 2 组，最近一组 screen_01。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T06:09:23+08:00：验证集 CFG 校准 `screen_02`：w=1.5，区间 [0.0,1.0]，PDCorr=0.3214，PDS_L2=0.8022，MSE=0.01818（非测试）。

- 2026-11-28T06:15:22+08:00：验证集 CFG 校准 `screen_window_01_01`：w=1.375，区间 [0.05,0.85]，PDCorr=0.2980，PDS_L2=0.7950，MSE=0.01646（非测试）。

- 2026-11-28T06:21:14+08:00：验证集 CFG 校准 `screen_window_01_02`：w=1.5，区间 [0.05,0.85]，PDCorr=0.3027，PDS_L2=0.8025，MSE=0.01750（非测试）。

- 2026-11-28T06:23:10+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 5 组，最近一组 screen_window_01_02。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T06:38:11+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 5 组，最近一组 screen_window_01_02。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2142 GiB。

- 2026-11-28T06:52:41+08:00：验证集 CFG 校准 `refine_00`：w=1.375，区间 [0.0,1.0]，PDCorr=0.3950，PDS_L2=0.8206，MSE=0.01065（非测试）。

- 2026-11-28T06:53:11+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 6 组，最近一组 refine_00。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2141 GiB。

- 2026-11-28T07:08:12+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 6 组，最近一组 refine_00。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2141 GiB。

- 2026-11-28T07:19:28+08:00：验证集 CFG 校准 `refine_01`：w=1.375，区间 [0.05,0.85]，PDCorr=0.3816，PDS_L2=0.8167，MSE=0.00994（非测试）。

- 2026-11-28T07:23:13+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 7 组，最近一组 refine_01。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2141 GiB。

- 2026-11-28T07:38:14+08:00：巡检。runs/repro4_phase0/repro3_e45_upper_guidance 验证 CFG 校准，已记录 7 组，最近一组 refine_01。GPU 0 11273 MiB / 100%、GPU 1 11273 MiB / 100%、GPU 2 11233 MiB / 100%、GPU 3 11273 MiB / 100%。工作分区可用 2141 GiB。

- 2026-11-28T07:50:30+08:00：验证集 CFG 校准 `refine_02`：w=1.5，区间 [0.0,1.0]，PDCorr=0.3938，PDS_L2=0.8247，MSE=0.01188（非测试）。

- 2026-11-28T07:50:30+08:00：`runs/repro4_phase0/repro3_e45_upper_guidance` 完成验证集校准，共 3 个精炼候选，完整记录见 `all_results.json`。

- 2026-11-28T07:51:22+08:00：Phase 0 上界补测结束，周期 CFG 锁定为 w=1.25，时间窗 [0.0, 1.0]，完整验证 S 1.11327。

- 2026-11-28T07:51:22+08:00：Pilot Round 1 按登记的四组启动。全部使用 572,545 个训练细胞、10 轮、11,180 步，两卡训练各累积 16 次。周期验证与末轮对照使用同一批 cap=16 验证细胞。

- 2026-11-28T07:51:28+08:00：启动/续训 `runs/repro4_pilots/p1_cell_nobatch`，从 step 0 接续，2 卡、每卡 microbatch=16、各卡累积 [16, 16] 次、全局 batch=512，完整配置 `runs/repro4_pilots/p1_cell_nobatch/config.yaml`。

- 2026-11-28T07:51:28+08:00：启动/续训 `runs/repro4_pilots/p1_ctrl`，从 step 0 接续，2 卡、每卡 microbatch=16、各卡累积 [16, 16] 次、全局 batch=512，完整配置 `runs/repro4_pilots/p1_ctrl/config.yaml`。

- 2026-11-28T07:53:22+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 0。E0.022，step 25，loss 1.21225，lr 1.12e-06，grad_norm 13.02337，138.4 cells/s，峰值 18.44 GB。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 0。E0.018，step 20，loss 1.21389，lr 8.94e-07，grad_norm 12.81704，140.1 cells/s，峰值 18.44 GB。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2141 GiB。

- 2026-11-28T07:55:22+08:00：`runs/repro4_pilots/p1_ctrl` 启动核验完成，2 卡，累积 [16, 16]，全局 batch 512，关闭 Dynamo DDP 分图。首批训练记录的稳定吞吐 138.2 cells/s，rank 0 峰值 18.44 GB。

- 2026-11-28T07:55:22+08:00：`runs/repro4_pilots/p1_cell_nobatch` 启动核验完成，2 卡，累积 [16, 16]，全局 batch 512，关闭 Dynamo DDP 分图。首批训练记录的稳定吞吐 139.5 cells/s，rank 0 峰值 18.44 GB。

- 2026-11-28T08:08:23+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 250。E0.228，step 255，loss 0.82902，lr 1.14e-05，grad_norm 2.65377，103.7 cells/s，峰值 18.44 GB。去噪验证 step 250 raw 0.362622，EMA 0.551778，相对差 +52.16%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 250。E0.228，step 255，loss 0.94718，lr 1.14e-05，grad_norm 3.02638，108.1 cells/s，峰值 18.44 GB。去噪验证 step 250 raw 0.435554，EMA 0.551818，相对差 +26.69%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 97%、GPU 3 18009 MiB / 100%。工作分区可用 2131 GiB。

- 2026-11-28T08:23:24+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 450。E0.434，step 485，loss 0.37675，lr 2.17e-05，grad_norm 1.72132，136.3 cells/s，峰值 18.44 GB。去噪验证 step 250 raw 0.362622，EMA 0.551778，相对差 +52.16%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 450。E0.438，step 490，loss 0.42857，lr 2.19e-05，grad_norm 2.09425，138.4 cells/s，峰值 18.44 GB。去噪验证 step 250 raw 0.435554，EMA 0.551818，相对差 +26.69%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 99%、GPU 3 18009 MiB / 99%。工作分区可用 2132 GiB。

- 2026-11-28T08:38:25+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 700。E0.640，step 715，loss 0.35260，lr 3.2e-05，grad_norm 1.70371，136.2 cells/s，峰值 18.44 GB。去噪验证 step 500 raw 0.125359，EMA 0.533738，相对差 +325.77%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 700。E0.648，step 725，loss 0.36363，lr 3.24e-05，grad_norm 1.84363，138.0 cells/s，峰值 18.44 GB。去噪验证 step 500 raw 0.167356，EMA 0.535433，相对差 +219.94%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T08:53:26+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 900。E0.850，step 950，loss 0.34689，lr 4.25e-05，grad_norm 1.41995，135.9 cells/s，峰值 18.44 GB。去噪验证 step 750 raw 0.115094，EMA 0.512536，相对差 +345.32%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 950。E0.854，step 955，loss 0.33717，lr 4.27e-05，grad_norm 1.27505，108.0 cells/s，峰值 18.44 GB。去噪验证 step 750 raw 0.124148，EMA 0.515239，相对差 +315.02%。GPU 0 19021 MiB / 0%、GPU 1 17961 MiB / 0%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2131 GiB。

- 2026-11-28T09:08:28+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 1,150。E1.051，step 1,175，loss 0.33093，lr 5.25e-05，grad_norm 0.68843，136.9 cells/s，峰值 18.44 GB。去噪验证 step 1,118 raw 0.108904，EMA 0.483855，相对差 +344.29%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 1,150。E1.060，step 1,185，loss 0.32776，lr 5.3e-05，grad_norm 2.10755，137.9 cells/s，峰值 18.44 GB。去噪验证 step 1,118 raw 0.111511，EMA 0.487287，相对差 +336.99%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2135 GiB。

- 2026-11-28T09:23:29+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 1,400。E1.257，step 1,405，loss 0.32574，lr 6.28e-05，grad_norm 0.83808，105.9 cells/s，峰值 18.44 GB。去噪验证 step 1,250 raw 0.107865，EMA 0.474842，相对差 +340.22%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 1,400。E1.270，step 1,420，loss 0.32141，lr 6.35e-05，grad_norm 2.47416，137.8 cells/s，峰值 18.44 GB。去噪验证 step 1,250 raw 0.109151，EMA 0.478372，相对差 +338.26%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T09:38:30+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 1,600。E1.462，step 1,635，loss 0.32794，lr 7.31e-05，grad_norm 2.09693，136.6 cells/s，峰值 18.44 GB。去噪验证 step 1,500 raw 0.108893，EMA 0.458223，相对差 +320.80%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 1,650。E1.476，step 1,650，loss 0.31895，lr 7.38e-05，grad_norm 0.83003，137.7 cells/s，峰值 18.44 GB。去噪验证 step 1,500 raw 0.108157，EMA 0.462423，相对差 +327.55%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T09:53:31+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 1,850。E1.668，step 1,865，loss 0.31982，lr 8.34e-05，grad_norm 0.92449，136.8 cells/s，峰值 18.44 GB。去噪验证 step 1,750 raw 0.105235，EMA 0.438361，相对差 +316.56%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 1,850。E1.686，step 1,885，loss 0.31799，lr 8.43e-05，grad_norm 2.31566，137.5 cells/s，峰值 18.44 GB。去噪验证 step 1,750 raw 0.106429，EMA 0.444991，相对差 +318.11%。GPU 0 19021 MiB / 99%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 99%、GPU 3 18009 MiB / 100%。工作分区可用 2133 GiB。

- 2026-11-28T10:08:32+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 2,050。E1.878，step 2,100，loss 0.31696，lr 9.39e-05，grad_norm 0.61774，136.5 cells/s，峰值 18.44 GB。去噪验证 step 2,000 raw 0.104599，EMA 0.410686，相对差 +292.63%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 2,100。E1.896，step 2,120，loss 0.31959，lr 9.48e-05，grad_norm 0.66667，137.6 cells/s，峰值 18.44 GB。去噪验证 step 2,000 raw 0.105359，EMA 0.422277，相对差 +300.80%。GPU 0 19021 MiB / 0%、GPU 1 17961 MiB / 0%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 99%。工作分区可用 2131 GiB。

- 2026-11-28T10:23:34+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 2,300。E2.080，step 2,325，loss 0.31301，lr 0.0001，grad_norm 0.75091，136.8 cells/s，峰值 18.44 GB。去噪验证 step 2,250 raw 0.103748，EMA 0.375276，相对差 +261.72%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 2,300。E2.102，step 2,350，loss 0.31833，lr 0.0001，grad_norm 1.36721，137.8 cells/s，峰值 18.44 GB。去噪验证 step 2,250 raw 0.106651，EMA 0.392994，相对差 +268.49%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 99%、GPU 2 19067 MiB / 0%、GPU 3 18009 MiB / 0%。工作分区可用 2130 GiB。

- 2026-11-28T10:38:35+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 2,550。E2.285，step 2,555，loss 0.31467，lr 9.97e-05，grad_norm 1.84230，106.4 cells/s，峰值 18.44 GB。去噪验证 step 2,500 raw 0.102948，EMA 0.337313，相对差 +227.65%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 2,550。E2.308，step 2,580，loss 0.31595，lr 9.96e-05，grad_norm 0.75741，137.8 cells/s，峰值 18.44 GB。去噪验证 step 2,500 raw 0.103875，EMA 0.360069，相对差 +246.64%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T10:53:36+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 2,750。E2.496，step 2,790，loss 0.31384，lr 9.91e-05，grad_norm 1.43551，136.5 cells/s，峰值 18.44 GB。去噪验证 step 2,750 raw 0.102589，EMA 0.301665，相对差 +194.05%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 2,800。E2.518，step 2,815，loss 0.31587，lr 9.9e-05，grad_norm 0.56114，137.9 cells/s，峰值 18.44 GB。去噪验证 step 2,750 raw 0.103347，EMA 0.327579，相对差 +216.97%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2133 GiB。

- 2026-11-28T11:08:37+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 3,000。E2.701，step 3,020，loss 0.31531，lr 9.81e-05，grad_norm 0.36457，137.1 cells/s，峰值 18.44 GB。去噪验证 step 3,000 raw 0.102983，EMA 0.270410，相对差 +162.58%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 3,000。E2.728，step 3,050，loss 0.31669，lr 9.8e-05，grad_norm 0.59781，138.0 cells/s，峰值 18.44 GB。去噪验证 step 3,000 raw 0.102785，EMA 0.297715，相对差 +189.65%。GPU 0 19021 MiB / 99%、GPU 1 17961 MiB / 99%、GPU 2 19067 MiB / 0%、GPU 3 18009 MiB / 0%。工作分区可用 2134 GiB。

- 2026-11-28T11:23:38+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 3,250。E2.907，step 3,250，loss 0.31551，lr 9.69e-05，grad_norm 0.98983，136.4 cells/s，峰值 18.44 GB。去噪验证 step 3,250 raw 0.101872，EMA 0.245107，相对差 +140.60%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 3,250。E2.934，step 3,280，loss 0.31108，lr 9.67e-05，grad_norm 0.65615，137.8 cells/s，峰值 18.44 GB。去噪验证 step 3,250 raw 0.102683，EMA 0.271363，相对差 +164.27%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 99%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2131 GiB。

- 2026-11-28T11:38:40+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 3,450。E3.113，step 3,480，loss 0.31185，lr 9.53e-05，grad_norm 1.26162，135.8 cells/s，峰值 18.44 GB。去噪验证 step 3,354 raw 0.102325，EMA 0.236058，相对差 +130.69%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 3,500。E3.140，step 3,510，loss 0.31552，lr 9.51e-05，grad_norm 0.37682，138.3 cells/s，峰值 18.44 GB。去噪验证 step 3,500 raw 0.102080，EMA 0.248609，相对差 +143.54%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T11:53:41+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 3,700。E3.318，step 3,710，loss 0.31088，lr 9.35e-05，grad_norm 0.34604，136.3 cells/s，峰值 18.44 GB。去噪验证 step 3,500 raw 0.101955，EMA 0.224857，相对差 +120.55%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 3,700。E3.350，step 3,745，loss 0.30922，lr 9.31e-05，grad_norm 0.42695，137.7 cells/s，峰值 18.44 GB。去噪验证 step 3,500 raw 0.102080，EMA 0.248609，相对差 +143.54%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 99%。工作分区可用 2133 GiB。

- 2026-11-28T12:08:42+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 3,900。E3.524，step 3,940，loss 0.30905，lr 9.13e-05，grad_norm 0.65960，135.6 cells/s，峰值 18.44 GB。去噪验证 step 3,750 raw 0.101321，EMA 0.208832，相对差 +106.11%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 3,950。E3.560，step 3,980，loss 0.31027，lr 9.09e-05，grad_norm 0.24848，137.5 cells/s，峰值 18.44 GB。去噪验证 step 3,750 raw 0.102444，EMA 0.227587，相对差 +122.16%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2135 GiB。

- 2026-11-28T12:23:43+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 4,150。E3.730，step 4,170，loss 0.31222，lr 8.89e-05，grad_norm 0.30771，136.0 cells/s，峰值 18.44 GB。去噪验证 step 4,000 raw 0.101010，EMA 0.195874，相对差 +93.92%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 4,200。E3.766，step 4,210，loss 0.30756，lr 8.85e-05，grad_norm 0.39090，138.1 cells/s，峰值 18.44 GB。去噪验证 step 4,000 raw 0.101695，EMA 0.208148，相对差 +104.68%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T12:38:44+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 4,400。E3.936，step 4,400，loss 0.30637，lr 8.63e-05，grad_norm 0.43712，135.9 cells/s，峰值 18.44 GB。去噪验证 step 4,250 raw 0.100796，EMA 0.184779，相对差 +83.32%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 4,400。E3.976，step 4,445，loss 0.31299，lr 8.57e-05，grad_norm 0.29800，137.6 cells/s，峰值 18.44 GB。去噪验证 step 4,250 raw 0.101492，EMA 0.189843，相对差 +87.05%。GPU 0 19021 MiB / 99%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 99%。工作分区可用 2132 GiB。

- 2026-11-28T12:53:46+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 4,600。E4.137，step 4,625，loss 0.30911，lr 8.34e-05，grad_norm 0.27035，136.5 cells/s，峰值 18.44 GB。去噪验证 step 4,500 raw 0.100798，EMA 0.175221，相对差 +73.83%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 4,650。E4.177，step 4,670，loss 0.30610，lr 8.28e-05，grad_norm 0.68441，137.5 cells/s，峰值 18.44 GB。去噪验证 step 4,500 raw 0.101564，EMA 0.173617，相对差 +70.94%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 99%。工作分区可用 2132 GiB。

- 2026-11-28T13:08:47+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 4,850。E4.343，step 4,855，loss 0.30648，lr 8.03e-05，grad_norm 0.58667，105.5 cells/s，峰值 18.44 GB。去噪验证 step 4,750 raw 0.100853，EMA 0.166297，相对差 +64.89%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 4,900。E4.387，step 4,905，loss 0.30477，lr 7.96e-05，grad_norm 0.57313，104.6 cells/s，峰值 18.44 GB。去噪验证 step 4,750 raw 0.101256，EMA 0.159865，相对差 +57.88%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T13:23:48+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 5,050。E4.548，step 5,085，loss 0.30678，lr 7.7e-05，grad_norm 0.19688，136.7 cells/s，峰值 18.44 GB。去噪验证 step 5,000 raw 0.100417，EMA 0.158351，相对差 +57.69%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 5,100。E4.593，step 5,135，loss 0.30644，lr 7.63e-05，grad_norm 0.15050，137.6 cells/s，峰值 18.44 GB。去噪验证 step 5,000 raw 0.101104，EMA 0.148594，相对差 +46.97%。GPU 0 19021 MiB / 99%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T13:38:49+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 5,300。E4.754，step 5,315，loss 0.30183，lr 7.35e-05，grad_norm 0.55165，136.6 cells/s，峰值 18.44 GB。去噪验证 step 5,250 raw 0.100349，EMA 0.150954，相对差 +50.43%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 5,350。E4.803，step 5,370，loss 0.30619，lr 7.27e-05，grad_norm 0.14104，137.5 cells/s，峰值 18.44 GB。去噪验证 step 5,250 raw 0.101080，EMA 0.139351，相对差 +37.86%。GPU 0 19021 MiB / 100%、GPU 1 17961 MiB / 100%、GPU 2 19067 MiB / 100%、GPU 3 18009 MiB / 100%。工作分区可用 2132 GiB。

- 2026-11-28T13:53:51+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 5,500。E4.964，step 5,550，loss 0.31291，lr 6.98e-05，grad_norm 0.15798，136.4 cells/s，峰值 18.44 GB。去噪验证 step 5,500 raw 0.100128，EMA 0.144179，相对差 +43.99%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 5,590。E5.000，step 5,590，loss 0.30904，lr 6.92e-05，grad_norm 0.51758，137.5 cells/s，峰值 18.44 GB。去噪验证 step 5,590 raw 0.100635，EMA 0.129407，相对差 +28.59%。GPU 0 19021 MiB / 0%、GPU 1 17961 MiB / 0%、GPU 2 19067 MiB / 0%、GPU 3 18009 MiB / 0%。工作分区可用 2132 GiB。

- 2026-11-28T14:06:04+08:00：`runs/repro4_pilots/p1_cell_nobatch` 第 5 轮验证生成（EMA=0.9996，每扰动最多 16 细胞，50 步）：PDCorr=0.2051，MSE=0.02308，PDS_L2=0.6186。这是验证结果，非测试结果。

- 2026-11-28T14:06:09+08:00：`runs/repro4_pilots/p1_cell_nobatch` 完成第 5/10 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.12941。

- 2026-11-28T14:08:52+08:00：巡检。runs/repro4_pilots/p1_ctrl 全量数据训练 已保存 step 5,590。E5.000，step 5,590，loss 0.30822，lr 6.92e-05，grad_norm 0.29150，136.3 cells/s，峰值 18.44 GB。去噪验证 step 5,590 raw 0.100056，EMA 0.141883，相对差 +41.80%。runs/repro4_pilots/p1_cell_nobatch 全量数据训练 已保存 step 5,600。E5.036，step 5,630，loss 0.30716，lr 6.85e-05，grad_norm 0.16196，137.5 cells/s，峰值 18.44 GB。去噪验证 step 5,590 raw 0.100635，EMA 0.129407，相对差 +28.59%。E5 生成 PDCorr 0.2051，PDS_L2 0.6186，MSE 0.02308，S 0.5585。GPU 0 27057 MiB / 100%、GPU 1 25997 MiB / 100%、GPU 2 27103 MiB / 100%、GPU 3 26045 MiB / 100%。工作分区可用 2133 GiB。

- 2026-11-28T14:09:33+08:00：`runs/repro4_pilots/p1_ctrl` 第 5 轮验证生成（EMA=0.9996，每扰动最多 16 细胞，50 步）：PDCorr=0.1416，MSE=0.03106，PDS_L2=0.5833。这是验证结果，非测试结果。

- 2026-11-28T14:09:39+08:00：`runs/repro4_pilots/p1_ctrl` 完成第 5/10 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.14188。

- 2026-11-28T14:18:42+08:00：`runs/repro4_pilots/p1_ctrl` 收到停止信号，完成当前优化步 5731 后安全保存。

- 2026-11-28T14:18:42+08:00：`runs/repro4_pilots/p1_cell_nobatch` 收到停止信号，完成当前优化步 5784 后安全保存。
