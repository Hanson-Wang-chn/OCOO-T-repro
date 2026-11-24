# OCOO-T 首轮复现归档

首轮 Replogle–Nadig 2,000 HVG 训练于北京时间 2026-11-23 02:00 完成，共 55 轮、61,490 个更新。E45 的默认 EMA 是周期生成得分最高的检查点。完整验证覆盖 4,825 个 HepG2 细胞、60 个扰动。首轮正式测试尚未运行，论文表 3 的测试复现差距仍待同拆分测量。附录保存了本轮开始前的 README 全文。[训练结束记录](runs/main_ddp_recovery/finished.json) · [工作流状态](runs/main_ddp_recovery/pipeline_state.json)

## 数据与参数

| 项目 | 首轮设置 |
| --- | --- |
| 数据和拆分 | 复用 STATE / PerturbDiff 发布的 efficacy 过滤、库大小归一化及 log1p、2,000 HVG 顺序和 HepG2 留出列表。训练 572,545 细胞 / 2,013 扰动 / 四细胞系，验证 4,825 / 60，测试 26,878 / 380。 |
| 对照和批次 | 使用 39,165 个真实对照中的 4,976 个 HepG2 对照，真实端与预测端共享。56 个 gem_group 独立标签与 cell-line 嵌入相加，cell dropout 同时置空两者。 |
| 基因条件 | 冻结 5,120 维 ESM2，经共享线性投影用于基因身份及扰动。联合词表 3,618 个，命中 3,545 个，70 个 HVG 缺失。缺失取平均向量并加可训练身份，各官方扰动均保留。 |
| Transformer | 12 层 / 宽 768 / 12 头，RMSNorm、QK-RMSNorm、SwiGLU、adaLN-zero，不做 patch。第 4 层起前置 32 个含扰动及细胞条件的 token。 |
| 首轮架构解释 | SwiGLU 采用 JiT 的 ratio 4 参数匹配惯例，实际中间宽 2,048。连续表达 MLP 为 1→768→768，固定一维 sincos，时间编码 256 维且尺度 1，cell / batch 初始化标准差 1 / 0.02，线性层 Xavier。 |
| Flow 和优化 | 速度预测与 L2，Gaussian 噪声标准差 1，logit-normal 为 N(−0.8, 0.8²)，扰动与细胞标签独立 dropout 0.1。AdamW β=(0.9,0.95)、weight decay 0，BF16，有效 batch 512，峰值 lr 1e−4，5 轮线性 warmup 后 cosine 至 0。 |
| 训练长度 | seed 42，每轮 1,118 个完整批次，129 个尾部细胞当轮丢弃。上限 100 轮、至少 20 轮，周期 EMA 生成每 5 轮。PDCorr + mean(PDS) 连续 10 轮未提升 0.002 时早停。 |
| EMA 和采样 | 双 EMA 0.9999 / 0.9996，0.9996 为默认推理权重。均匀 Heun 50 步、末步 Euler、双 CFG 式 11。验证筛选种子 4242、每批 32、每扰动最多 16 个生成细胞，两个候选均生成全部 4,825 个细胞。 |
| 评测 | Cell Eval 0.8.2 / pdex 0.3.0 的九项指标，按扰动宏平均。保留原始实数预测，DE 计算使用非负预测。候选比较裁负与将小于 0.05 的生成表达置零。 |

论文、数据证据和完整设置见 [论文 v2](https://arxiv.org/html/2606.12838v2)、[数据统计](data/processed/data_report.json)、[扰动覆盖](data/processed/perturbation_support_audit.json)、[设计说明](paper/implementation_notes.md) 和 [最终训练配置](runs/main_ddp_recovery/config.yaml)。

## 运行过程

1. 11 月 17 日启动首轮原训练。训练前用 16,384 个细胞、384 次更新做噪声、时间编码和 cell 初始化的 2×2×2 配对；选择 1 / 1 / 1。随后补三组短试验。分离投影与 x-prediction 的裁负综合分分别下降 0.00741 / 0.13061，fan-in 的点估计上升 0.01556、区间跨零且表达误差增加。E25→26 保留 batch 的独立试验未显示净收益。[八组网格](runs/pilot_grid_complete/pilot_metrics.csv) · [分离投影](runs/pert_projection_analysis/decision.json) · [x-prediction](runs/xpred_vloss_analysis/decision.json) · [fan-in](runs/value_fanin_analysis/decision.json) · [batch dropout](runs/batch_dropout_mature_trial/decision.json)
2. 旧机单卡每次 16 个细胞、累积 32 次。11 月 22 日迁移，四张 A100 PCIe 各 16 个细胞、累积 8 次。单卡 70.51 与四卡 270.81 cells/s 属同断点短基准。旧四卡运行在 E40 出现 EMA 异常；冻结回放显示 PyTorch 2.8 Dynamo DDPOptimizer 分图造成同批梯度偏差。关闭该分图后，四卡对单卡梯度相对 L2 从 104.9% 降到 0.26%。修复从 step 41,374 接续，保留核心模型编译与原生 DDP。 [迁移与基准](MIGRATION.md) · [故障诊断](logs/epoch040_recovery/README.md)
3. 修复后 E40（step 44,720）、E45（50,310）、E50（55,900）、E55（61,490）均保存并验证完整断点。E40 周期生成恢复，E45 更新最佳及早停参考。E55 触发早停。 [重训来源](runs/main_ddp_recovery/recovery_origin.json) · [完整状态校验](logs/epoch040_recovery/final_training_checkpoint_verification.json)

## 周期验证

同一批最多每扰动 32 个生成细胞，共 1,916 个预测，真实端包含所有验证细胞。采样用普通条件场、EMA 0.9996、Heun 50 步及裁负。

| 轮次 | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [E35（main_v1）](runs/main_v1/validation_epoch_035/metrics_de_diagnostic/summary.json) | 0.29607 | 0.69806 | 0.71194 | 0.75278 | 0.00943 | 0.06781 | 0.09479 | 0.07845 | 0.70918 |
| [E40（main_v1）](runs/main_v1/validation_epoch_040/metrics_de_diagnostic/summary.json) | 0.18819 | 0.57417 | 0.58306 | 0.53806 | 0.09878 | 0.22047 | 0.08986 | 0.09509 | 0.58762 |
| [E40（main_ddp_recovery）](runs/main_ddp_recovery/validation_epoch_040/metrics_de_diagnostic/summary.json) | 0.30090 | 0.69583 | 0.71111 | 0.74778 | 0.00951 | 0.06796 | 0.09538 | 0.07963 | 0.71209 |
| [E45（main_ddp_recovery）](runs/main_ddp_recovery/validation_epoch_045/metrics_de_diagnostic/summary.json) | 0.30433 | 0.70250 | 0.71667 | 0.74722 | 0.00943 | 0.06773 | 0.09744 | 0.07975 | 0.70255 |
| [E50（main_ddp_recovery）](runs/main_ddp_recovery/validation_epoch_050/metrics_de_diagnostic/summary.json) | 0.29683 | 0.68861 | 0.70222 | 0.73806 | 0.00973 | 0.06865 | 0.09324 | 0.07989 | 0.69631 |
| [E55（main_ddp_recovery）](runs/main_ddp_recovery/validation_epoch_055/metrics_de_diagnostic/summary.json) | 0.28959 | 0.68556 | 0.70028 | 0.75472 | 0.00982 | 0.06905 | 0.09104 | 0.07798 | 0.69384 |

## 完整验证与论文对照

| 样本与配置 | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 论文表 3 · 测试 | 0.43700 | 0.82000 | 0.85300 | 0.83300 | 0.00790 | 0.06100 | 0.23200 | 0.24000 | 0.76600 |
| E45 验证 · w=2 / [0,1] / 阈值 0 | 0.36642 | 0.81917 | 0.83361 | 0.86528 | 0.01905 | 0.08367 | 0.17310 | 0.10192 | 0.80703 |
| E45 验证 · w=2 / [0,1] / 阈值 0.05 | 0.36513 | 0.82167 | 0.83444 | 0.86833 | 0.01907 | 0.08379 | 0.19464 | 0.15863 | 0.81965 |
| E45 验证 · w=2 / [0.1,0.9] / 阈值 0 | 0.34587 | 0.82056 | 0.83417 | 0.86528 | 0.01577 | 0.07927 | 0.16308 | 0.10002 | 0.81000 |
| E45 验证 · w=2 / [0.1,0.9] / 阈值 0.05 | 0.34482 | 0.82167 | 0.83472 | 0.86639 | 0.01578 | 0.07930 | 0.19168 | 0.15861 | 0.81254 |
| 上述全区间 0.05 验证值减论文测试值 | -0.07187 | +0.00167 | -0.01856 | +0.03533 | +0.01117 | +0.02279 | -0.03736 | -0.08137 | +0.05365 |

验证与论文测试的扰动集合和样本量不同，这些差值仅用于定位误差。全区间 0.05 截断相对同批裁负将 DEPrec 从 0.10192 提至 0.15863，DEOver 从 0.17310 提至 0.19464。该候选仅 DEPrec 未达到项目的 0.18 验证检查线。该检查线由首轮复现设定。补做的四组引导对照中，最佳 DEPrec 改善 0.00146。配对 960 细胞的 EMA 对照中，0.9999 在 w=2 全区间和 0.05 截断下的 DEPrec 比 0.9996 低 0.03082。[完整验证记录](runs/main_ddp_recovery/calibration/postprocessing/review.json) · [引导配对](runs/main_ddp_recovery/postgate_guidance_study/report.json) · [EMA 配对](runs/main_ddp_recovery/postgate_ema_study/report.json)

## 原 README 全文

原文件 SHA-256 为 `276fd9490629e8dfe5a662b27130e62220272c8568bb2bd1fcd8e3ba34a0b5e5`。下方从原一级标题开始按字节保留。

# OCOO-T 复现（Replogle–Nadig）

本项目独立实现 OCOO-T v2，仅使用 Replogle–Nadig。新机使用 4 张 A100 PCIe 40GB，独立 Conda 环境为 `/home/whs/miniconda3/envs/ocoo-t`。[论文](paper/OCOO-T_v2.pdf)

**四卡故障已修复，重训已完成。** `runs/main_ddp_recovery` 从故障前 step **41374** 接续，11 月 23 日 02:00 按原规则在 **E55** 早停，**E45** 为最佳。E40 匹配验证 MSE 回落至 **0.00951**，PDCorr 回升至 **0.30090**。旧运行 `runs/main_v1` 保留到 step 45170 供核查。[训练完成记录](runs/main_ddp_recovery/finished.json) · [E40 配对](logs/epoch040_recovery/paired_old_repaired_e40.json)

完整验证中，E45 的全区间 `w=2` 配合 0.05 截断仅缺 **DEPrec 0.15863**，原检查线为 **0.18**。补充引导对照的最佳改善为 0.00146。换用 EMA 0.9999 后，同批全区间的 DEPrec 下降 0.03082。故保留论文默认 EMA，复现流程停在验证审查，正式测试尚未启动。[完整验证](runs/main_ddp_recovery/calibration/postprocessing/review.json) · [引导对照](runs/main_ddp_recovery/postgate_guidance_study/report.json) · [EMA 对照](runs/main_ddp_recovery/postgate_ema_study/report.json)

旧运行 E35→E40 普通条件验证 MSE 从 **0.00943** 升至 **0.09878**。冻结复算确认原始权重去噪误差稳定，默认 EMA 的误差独立复算仍由约 0.104 升至 0.197。冻结 step 41374 的同批回放把问题定位到 PyTorch 2.8 的 Dynamo DDPOptimizer 分图。仅关闭分图后，四卡梯度相对单卡的 L2 偏差由 **104.9%** 降至 **0.26%**，四个 rank 的梯度校验和一致。继续使用核心模型编译、原生 DDP、BF16、全局 batch 512 和原采样参数。[故障对照与重训记录](logs/epoch040_recovery/README.md)

重训健康检查使用固定的 1,024 验证细胞。若默认 EMA 的去噪 endpoint MSE 比当前原始权重高超过 4%，会先保存完整断点再暂停。三卡接续按 11、11、10 个 microbatch 划分，仍覆盖每个 512 细胞的全局批次。

同机实测单卡 **70.51**、四卡 **270.81 cells/s（3.84 倍）**。从同一断点各预热 2 步、计时 8 个完整优化步，计时段为独占 GPU 的训练更新。并行验证、完整断点写盘和周期生成另占墙钟时间。[单卡基准](logs/local_ddp_migration/bench_single_b16.log) · [四卡基准](logs/local_ddp_migration/bench_ddp_b16.log) · [迁移与恢复](MIGRATION.md)

<!-- LIVE_TRAINING_START -->

当前状态为重训和补充验证均已结束。最佳 E45 已归档，完整验证 DEPrec 为 0.15863，正式测试尚未启动。四张 A100 已释放。

<!-- LIVE_TRAINING_END -->

## 断点与迁移

- 重训末步完整断点为 `runs/main_ddp_recovery/last.pt`，最佳权重为 `best_generation.pt`。E40、E45、E50 和 E55 完整断点均归档并核验。旧运行 step 44720 和 45170 保留用于故障核查。[重训来源](runs/main_ddp_recovery/recovery_origin.json) · [末步和最佳核验](logs/epoch040_recovery/final_training_checkpoint_verification.json) · [旧运行归档](runs/main_v1/snapshots/step_45170/preserved.json)
- 外部目录：/home/dataset-assist-0/whs/ocoo-t-artifacts。已包含ocoo-t-environment.tar（7.92GiB）及SHA文件，迁移可直接解包环境。
- 新机训练使用每卡 microbatch 16 × 8 次累积 × 4 卡，全局 batch 512。模型、学习率日程、早停和采样配置沿用原配置。rank 0 恢复旧机随机流，rank 1–3 由原种子、断点步和 rank 派生独立随机流，四卡断点保存各自状态。[训练入口](scripts/train.py) · [实时进程](runs/main_ddp_recovery/training_execution.json)

## 数据与设计

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，HepG2为4,976。公开数据与Arc原件SHA一致；沿用efficacy过滤、库大小归一化/log1p、固定2,000HVG和官方拆分，不重复归一化。保留56个batch及全部官方条件；验证含2个、测试标签含8个全局未见扰动。冻结5120维ESM2，联合3618基因匹配3545，缺失处理/来源见[设计说明](paper/implementation_notes.md)。

12层/768宽/12头连续基因Transformer，RMSNorm、QK-norm、SwiGLU、adaLN-zero，第4层起32条件token；2,000HVG不patch，v-prediction rectified flow。全局 batch 512、lr1e-4、5轮warmup/cosine/BF16及双EMA。按论文附录默认用 EMA 0.9996 推理，采用原式双 CFG 和均匀 50 步 Heun，末步 Euler。

未公开选择noise_std1/time_scale1/cell_init_std1，共享投影、SwiGLU参数匹配2048、Xavier等均已记录并做短对照。100轮上限/20轮最少、原验证点分数10轮耐心及0.002最小增益属复现选择。原best/early_reference=E35，最早可能E45停止；迁移不改变这些参数。环境与版本见[environment.yml](environment.yml)、[锁文件](environment.lock.txt)。

## 旧机验证记录

冻结 E35 的 w2/[0,1] 已完成 4825 细胞。w2/[.1,.9] 在新机从 1024 接续至 **1056/4825**，沿用 B32、seed 4242 和 Heun 50 步。故障排查暂停了旧运行的辅助队列。E35 缓存完整保留，修复版 E40 使用独立目录重新生成。[原记录](runs/main_v1/interim_epoch035_calibration/plan.json) · [新机采样进度](runs/main_v1/interim_epoch035_calibration_local_ddp/calibration/refine_01/generation_state.json)

首组完整**验证**结果（非测试）：

| 处理 | PDCorr | PDS_L2 | MSE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clamp0 | 0.3614 | 0.8125 | 0.01820 | 0.1770 | 0.1012 | 0.7931 |
| 原文未公开的0.05候选 | 0.3591 | 0.8119 | 0.01822 | 0.1963 | 0.1542 | 0.7951 |

0.05改善DE点估计，但DEPrec仍缺原0.18复现检查线，PD及综合略降，未采用；检查线并非论文公开验证阈值。[完整九项与提前诊断结论](runs/main_v1/interim_epoch035_calibration/first_full_floor_diagnostic/decision.json)。新机应完成第二组、原0/.05共四候选检查、提前/正式.05核对及[独立配对审计](runs/main_v1/interim_epoch035_calibration/analysis_plan.json)。

[成熟双EMA两病例检查](runs/main_v1/ema_epoch035_review/plan.json)已登记，4项CPU协议检查及真实193-tensor装载核对通过；尚未做0.9999 GPU生成或采用另一EMA。须等本轮完整审计后，只补固定普通/全区间CFG各960验证细胞，不新增训练或CFG搜索。[E40精确保存/一次固定CFG后续](runs/main_v1/interim_epoch040_fixed_cfg/plan.json)也须按新PID/路径重新登记。

11组短对照、E20 CFG强度/区间、E25完整诊断及.05、成熟batch-retention、CPU批次/有限采样诊断均已完成；没有可靠依据的替代配置未采用。历史原始结果和详细限制见[context.md](context.md)、[设计及尝试](paper/implementation_notes.md)、[完整时间线](logs/progress_history.md)。E25拼接缓存混合B16/B32种子，仅诊断，不冒充正式完整细化。

最终流程仍为原训练→正CFG搜索/两个全量验证→0/.05完整检查→冻结选择→全26878/380测试→九项报告，测试保留同raw的clamp0对照。后续工具已35项CPU检查；尚无正式测试或最终REPORT。主训练期间每 50 步和轮末原子保存完整断点。验证生成按批保存，README 约每 15 分钟汇总。

## 关键记录（北京时间）

| 时间 | 进展 |
| --- | --- |
| 11-17 00:54–07:00 | 建环境、保存论文/数据、核对官方协议，启动第一版全量训练。 |
| 11-17 10:59 | 改为约15分钟更新README，安全保存后原配置续训。 |
| 11-18–20 | 条件/数量/稀疏性诊断、11组短对照、E20/E25及成熟batch受控试验完成。 |
| 11-21 12:03–13:18 | E35归档与双协议复核完成，启动冻结E35完整验证，登记E40后续。 |
| 11-21 21:46–21:56 | 首组全量/.05检查完成，仍缺DEPrec；登记成熟双EMA并通过CPU准备。 |
| 11-21 22:35 | 按用户要求保存step41374并停止全部实验任务，保留第二组1024批次缓存。 |
| 11-21 22:37–22:48 | 最新断点冗余归档、完整CPU核验与可迁移环境包完成。 |
| 11-22 11:14 | 新机以四卡 DDP 恢复，独立 Conda 环境、完整断点和训练吞吐均已核验。 |
| 11-22 13:47 | 定位 Dynamo DDPOptimizer 梯度偏差，修复后从 step 41374 重训。 |
| 11-23 02:00 | 修复版完成 E55，最佳 E45，完整断点读回核验。 |
| 11-23 04:00 | E45 两组完整验证与四个候选检查完成，登记补充引导对照。 |

范围：目前优先复现可核对数据与拆分的Table3主基准。Table5的18,080基因面板/引用不一致，[来源审计](paper/long_panel_source_audit.json)已保存；control-context/MMD消融未运行，不以其他数据或补零面板冒充。

- 2026-11-22T11:14:17+08:00：启动/续训 `runs/main_v1`，从 step 41374 接续，4 卡、每卡 microbatch=16、累积 8 次、全局 batch=512，完整配置 `runs/main_v1/config.yaml`。

- 2026-11-22T11:18:57+08:00：冻结E35默认EMA开始中途完整验证校准：沿用原正CFG搜索和两个全量4825细化，随后检查0/.05；主训练及质量线保持，不执行测试。见 `runs/main_v1/interim_epoch035_calibration/plan.json`。

- 2026-11-22T11:19:20+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2503，PDS_L2=0.7044，MSE=0.01318（非测试）。

- 2026-11-22T11:19:32+08:00：验证集 CFG 校准 `screen_01`：w=0.6666666667，区间 [0.1,0.9]，PDCorr=0.2552，PDS_L2=0.7161，MSE=0.01283（非测试）。

- 2026-11-22T11:19:47+08:00：验证集 CFG 校准 `screen_02`：w=1.0，区间 [0.1,0.9]，PDCorr=0.2801，PDS_L2=0.7678，MSE=0.01334（非测试）。

- 2026-11-22T11:19:59+08:00：验证集 CFG 校准 `screen_03`：w=1.5，区间 [0.1,0.9]，PDCorr=0.2959，PDS_L2=0.7869，MSE=0.01633（非测试）。

- 2026-11-22T11:20:11+08:00：验证集 CFG 校准 `screen_04`：w=2.0，区间 [0.1,0.9]，PDCorr=0.3003，PDS_L2=0.8039，MSE=0.02134（非测试）。

- 2026-11-22T11:20:23+08:00：验证集 CFG 校准 `screen_full_interval`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3208，PDS_L2=0.8056，MSE=0.02447（非测试）。

- 2026-11-22T11:20:36+08:00：验证集 CFG 校准 `refine_00`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3614，PDS_L2=0.8125，MSE=0.01820（非测试）。

- 2026-11-22T11:24:57+08:00：E35 验证已从 1024 接续并落盘 1056/4825。GPU 3 共享时主训练从约 269 降至约 116 cells/s，余下验证已排在主训练后。E40 完整断点仍在训练期间归档。

- 2026-11-22T13:23:19+08:00：`runs/main_v1` 第 40 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.1882，MSE=0.09878，PDS_L2=0.5831。这是验证结果，非测试结果。

- 2026-11-22T13:23:24+08:00：`runs/main_v1` 完成第 40/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.19701。

- 2026-11-22T13:24:00+08:00：第 40 轮缓存的 CPU 九指标检查完成：DEOver=0.0899、DEPrec=0.0951、DirAgr=0.5876；复用既有预测，无新增生成。完整结果见 `runs/main_v1/validation_epoch_040/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-22T13:38:41+08:00：`runs/main_v1` 收到停止信号，完成当前优化步 45170 后安全保存。

- 2026-11-22T13:47:53+08:00：启动/续训 `runs/main_ddp_recovery`，从 step 41374 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/main_ddp_recovery/config.yaml`。

- 2026-11-22T13:53:50+08:00：关闭 Dynamo DDPOptimizer 分图后，修复版首轮固定验证在 step 41500 完成。原始权重 endpoint MSE=0.104432，默认 EMA=0.103985。首个新完整断点 41400 已读回通过；继续监控旧故障区间 42000–42250 和 E40。

- 2026-11-22T14:18:47+08:00：已重跑至旧版首次明显超出健康阈值的 step 42250。旧版 EMA endpoint MSE 0.112078，相对 raw +7.51%；修复版 0.104006，相对 raw −0.27%。同一固定验证样本，完整断点已保存。继续核验修复后的 E40 生成。

- 2026-11-22T15:53:45+08:00：`runs/main_ddp_recovery` 第 40 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3009，MSE=0.00951，PDS_L2=0.7111。这是验证结果，非测试结果。

- 2026-11-22T15:53:48+08:00：`runs/main_ddp_recovery` 完成第 40/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10405。

- 2026-11-22T15:54:55+08:00：第 40 轮缓存的 CPU 九指标检查完成：DEOver=0.0954、DEPrec=0.0796、DirAgr=0.7121；复用既有预测，无新增生成。完整结果见 `runs/main_ddp_recovery/validation_epoch_040/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-22T15:55:40+08:00：修复版 E40 的九项匹配验证已与 E35 配对。综合分变化 +0.0021。均使用原普通条件、相同索引与噪声。完整结果见 `logs/epoch040_recovery/repaired_e35_e40/comparison.json`。

- 2026-11-22T15:57:11+08:00：修复版 E40 重跑完成，1916 个普通条件验证样本和九指标已保存。MSE=0.009512、PDCorr=0.300903，原故障 E40 分别为 0.098782 和 0.188192。与 E35 的综合分差 +0.00214，95% 区间跨零。按原点估计规则更新早停参考到 E40，最早 E50。完整证据见 logs/epoch040_recovery/README.md。

- 2026-11-22T19:14:48+08:00：`runs/main_ddp_recovery` 第 45 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3043，MSE=0.00943，PDS_L2=0.7167。这是验证结果，非测试结果。

- 2026-11-22T19:14:51+08:00：`runs/main_ddp_recovery` 完成第 45/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10421。

- 2026-11-22T19:15:21+08:00：第 45 轮缓存的 CPU 九指标检查完成：DEOver=0.0974、DEPrec=0.0797、DirAgr=0.7026；复用既有预测，无新增生成。完整结果见 `runs/main_ddp_recovery/validation_epoch_045/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-22T19:16:16+08:00：修复版 E45 已归档完整 step 50310，并完成相同 1916 细胞的周期生成及九项。MSE=0.009429、PDCorr=0.304331；相对修复 E40 的综合分+0.00732，配对 95% 区间跨零。按原定点估计规则早停参考由 E40 更新到 E45，最早 E55。

- 2026-11-22T22:36:20+08:00：`runs/main_ddp_recovery` 第 50 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2968，MSE=0.00973，PDS_L2=0.7022。这是验证结果，非测试结果。

- 2026-11-22T22:36:23+08:00：`runs/main_ddp_recovery` 完成第 50/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10452。

- 2026-11-22T22:36:47+08:00：第 50 轮缓存的 CPU 九指标检查完成：DEOver=0.0932、DEPrec=0.0799、DirAgr=0.6963；复用既有预测，无新增生成。完整结果见 `runs/main_ddp_recovery/validation_epoch_050/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-22T22:38:13+08:00：E50（step 55900）已归档并完成九项固定周期验证。相对 E45 综合分 −0.02000，95% 配对区间 [−0.03586, −0.00562]。E50 默认 EMA 去噪误差 0.104524，保持正常。E45 维持最佳及早停参考，继续按原规则到 E55。

- 2026-11-23T02:00:20+08:00：`runs/main_ddp_recovery` 第 55 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2896，MSE=0.00982，PDS_L2=0.7003。这是验证结果，非测试结果。

- 2026-11-23T02:00:25+08:00：`runs/main_ddp_recovery` 完成第 55/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10496。

- 2026-11-23T02:00:25+08:00：`runs/main_ddp_recovery` 验证生成指标连续 10 轮无改善，按预设验证收敛规则停止。

- 2026-11-23T02:00:32+08:00：`runs/main_ddp_recovery` 训练结束（55.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-23T02:00:53+08:00：第 55 轮缓存的 CPU 九指标检查完成：DEOver=0.0910、DEPrec=0.0780、DirAgr=0.6938；复用既有预测，无新增生成。完整结果见 `runs/main_ddp_recovery/validation_epoch_055/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-23T02:07:28+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2542，PDS_L2=0.7117，MSE=0.01315（非测试）。

- 2026-11-23T02:29:35+08:00：验证集 CFG 校准 `screen_01`：w=0.6666666667，区间 [0.1,0.9]，PDCorr=0.2554，PDS_L2=0.7189，MSE=0.01297（非测试）。

- 2026-11-23T02:29:47+08:00：验证集 CFG 校准 `screen_02`：w=1.0，区间 [0.1,0.9]，PDCorr=0.2755，PDS_L2=0.7664，MSE=0.01370（非测试）。

- 2026-11-23T02:30:00+08:00：验证集 CFG 校准 `screen_03`：w=1.5，区间 [0.1,0.9]，PDCorr=0.2919，PDS_L2=0.7978，MSE=0.01698（非测试）。

- 2026-11-23T02:30:13+08:00：验证集 CFG 校准 `screen_04`：w=2.0，区间 [0.1,0.9]，PDCorr=0.2989，PDS_L2=0.8211，MSE=0.02233（非测试）。

- 2026-11-23T02:54:49+08:00：验证集 CFG 校准 `screen_full_interval`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3229，PDS_L2=0.8186，MSE=0.02546（非测试）。

- 2026-11-23T03:32:25+08:00：正式校准的第二个完整验证候选已由三张空闲 GPU 生成并逐段核验，原校准流程将对同一 raw 缓存计算指标。原 B32 批次、全局种子偏移和双 CFG 选型保持。

- 2026-11-23T03:35:20+08:00：`runs/main_ddp_recovery` 工作流发生错误：CalledProcessError；检查点和已完成采样保留，继续排查时查看 `runs/main_ddp_recovery/pipeline.log`。

- 2026-11-23T03:57:55+08:00：第一组正式全量验证已保留前 1568 个样本并用四卡完成剩余批次。四张卡的 w2/[0,1] 首批逐元素一致，按原索引及种子偏移合并 4825 个 raw 样本。恢复正式校准评分。

- 2026-11-23T03:58:20+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2542，PDS_L2=0.7117，MSE=0.01315（非测试）。

- 2026-11-23T03:58:32+08:00：验证集 CFG 校准 `screen_01`：w=0.6666666667，区间 [0.1,0.9]，PDCorr=0.2554，PDS_L2=0.7189，MSE=0.01297（非测试）。

- 2026-11-23T03:58:44+08:00：验证集 CFG 校准 `screen_02`：w=1.0，区间 [0.1,0.9]，PDCorr=0.2755，PDS_L2=0.7664，MSE=0.01370（非测试）。

- 2026-11-23T03:58:57+08:00：验证集 CFG 校准 `screen_03`：w=1.5，区间 [0.1,0.9]，PDCorr=0.2919，PDS_L2=0.7978，MSE=0.01698（非测试）。

- 2026-11-23T03:59:09+08:00：验证集 CFG 校准 `screen_04`：w=2.0，区间 [0.1,0.9]，PDCorr=0.2989，PDS_L2=0.8211，MSE=0.02233（非测试）。

- 2026-11-23T03:59:21+08:00：验证集 CFG 校准 `screen_full_interval`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3229，PDS_L2=0.8186，MSE=0.02546（非测试）。

- 2026-11-23T03:59:34+08:00：验证集 CFG 校准 `refine_00`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3664，PDS_L2=0.8336，MSE=0.01905（非测试）。

- 2026-11-23T03:59:47+08:00：验证集 CFG 校准 `refine_01`：w=2.0，区间 [0.1,0.9]，PDCorr=0.3459，PDS_L2=0.8342，MSE=0.01577（非测试）。

- 2026-11-23T03:59:47+08:00：完成验证集采样校准；设置已锁定于 `runs/main_ddp_recovery/calibration/selected.json`，尚未据此计算测试结果。

- 2026-11-23T04:00:24+08:00：`runs/main_ddp_recovery` 的完整0/.05验证候选均未通过原质量检查，保留结果并停在测试前；见 `runs/main_ddp_recovery/calibration/postprocessing/review.json`。

- 2026-11-23T04:29:53+08:00：四组补充 CFG 验证已完成。去掉末段引导的 DEPrec 相对同批全区间仅提高 0.00146，低于预登记的 0.008。开始用冻结 E45 同一步两组 EMA 做固定验证对照，见 `runs/main_ddp_recovery/postgate_ema_study/plan.json`。

- 2026-11-23T04:35:46+08:00：完成 `runs/main_ddp_recovery/postgate_ema_study/conditional` 的 validation 评估（细胞 960，weights=ema_0.9999，CFG=None）；指标见 `runs/main_ddp_recovery/postgate_ema_study/conditional/metrics/summary.json`。PDCorr=0.2706，MSE=0.01226。

- 2026-11-23T04:53:59+08:00：完成 `runs/main_ddp_recovery/postgate_ema_study/guided` 的 validation 评估（细胞 960，weights=ema_0.9999，CFG=2.0）；指标见 `runs/main_ddp_recovery/postgate_ema_study/guided/metrics/summary.json`。PDCorr=0.3257，MSE=0.02257。

- 2026-11-23T04:56:41+08:00：E45 双 EMA 验证完成。全区间 w=2 和 0.05 截断下，0.9999 的配对 DEPrec 比默认 0.9996 下降 0.03082。继续保留默认 EMA；完整验证状态为 validation_needs_review，正式测试未启动。
