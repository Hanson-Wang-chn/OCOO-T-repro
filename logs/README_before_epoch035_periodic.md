# OCOO-T 复现（Replogle–Nadig）

在本机 A100 40 GB / conda `ocoo-t` 上独立实现 [OCOO-T v2](https://arxiv.org/abs/2606.12838v2)。仅下载并运行 Replogle–Nadig；Tahoe100M / PBMC 未下载。

**当前：原模型已完成第30轮联合复核，继续训练，下一周期E35。综合增益区间跨0且固定CFG有误差代价；后续完整验证流程已接续，尚未采用新阈值或访问测试。**

<!-- LIVE_TRAINING_START -->

实时生成评估：`runs/main_v1/interim_epoch035_fixed_cfg/candidate`，已生成 **16/960** 个细胞（Heun 50 步）；更新于 2026-11-21T12:18:27+08:00。

<!-- LIVE_TRAINING_END -->

## 数据与模型

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，其中HepG2为4,976个。公开数据与Arc原文件SHA256一致；沿用已发布efficacy过滤、库大小归一化/log1p、固定2,000 HVG及官方拆分，不重复归一化。保留56个batch、全部官方条件；验证含2个、测试标签含8个全局未见扰动。ESM2匹配3,545/3,618个联合基因，缺失处理与来源见[设计说明](paper/implementation_notes.md)。

模型为12层/768宽/12头连续基因Transformer：RMSNorm、QK-norm、SwiGLU、adaLN-zero，从第4层起32个条件token，冻结5120维ESM2，2,000 HVG不patch，v-prediction rectified flow。保持有效batch512（micro16、累积32）、lr1e-4、5轮warmup、cosine、BF16及双EMA；推理用默认EMA0.9996、均匀50步Heun/末步Euler、原式双条件CFG。原文8×H100，本机通过累积保持有效batch。

未公开选择：noise_std=1、time_scale=1、cell_init_std=1；最多100轮、至少20轮，按预登记验证点分数/10轮耐心及0.002最小增益判断。训练长度属于复现选择。每5轮生成验证；单训练进程约69–71 cells/s，每轮约2.3小时。明确参数、解释与试验依据均[单独记录](paper/implementation_notes.md)。

## 当前结果与尝试

下表均是第30轮**限量验证诊断**，不是正式完整验证或测试；两行采样协议不同，绝对分数不直接比较。

| 协议 | 预测细胞 | PDCorr | PDS_L2 | MSE | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通条件场，cap32/B32 | 1,916 | 0.2983 | 0.7031 | 0.00939 | 0.0774 | 0.6984 |
| 固定w2/[0.1,0.9]，cap16/B16 | 960 | 0.2872 | 0.7917 | 0.02206 | 0.0916 | 0.6998 |

E25→30的综合差（PDCorr+meanPDS）：ordinary +0.00880[-0.00799,0.02653]，固定CFG +0.01432[-0.00579,0.03644]，均跨0。固定CFG的DEPrec、PDS_cos增益区间高于0，同时MSE/MAE明确退化，不能宣称稳定整体提升。全部九项均值/差值/区间独立核算最大差0；DirAgr按官方方式保留53个有效条件。见[联合决定](runs/main_v1/interim_epoch030_fixed_cfg/decision.json)、[九指标配对图](runs/main_v1/interim_epoch030_fixed_cfg/joint_review/paired_protocols.png)。

两套点排序目前均偏好E30，保留原训练、选型及收敛规则；原规则的best/early_reference已刷新为E30，最早可能E40停止，后续改善会继续顺延。下一周期E35的完整状态归档及一个固定CFG对照已[登记并自动等待](runs/main_v1/interim_epoch035_fixed_cfg/plan.json)，不重复参考生成、不自动改策略。E30完整模型/优化器/双EMA/RNG已[SHA核验归档](runs/main_v1/snapshots/step_33540/preserved.json)。

已完成的主要尝试：

- 同一16,384训练细胞/384更新/seed42，完成noise×time×cell-init八组及独立投影、x预测、fan-in共11组短对照。保留当前共享投影/v预测/Xavier主模型；fan-in综合点估计略高但区间跨0，误差与DEPrec更差，不据短试跑推断成熟结果。详见[八组网格](runs/pilot_grid_complete/summary.json)、[补充三组](runs/supplemental_trials_complete.json)。这些是raw短预算诊断，正式推理始终用成熟EMA。
- E20七种CFG强度及区间检查显示：w2附近辨别收益趋平，更强引导增加误差；保留两区间供成熟验证，未擅改原公式或50步。见[强度](runs/main_v1/interim_epoch020_all_scales/decision.json)、[区间](runs/main_v1/interim_epoch020_interval_review/decision.json)。
- E25全4,825细胞固定CFG诊断：PD=0.3306、PDS_L2=0.7908、DEPrec=0.0989；同一raw样本用0.05处理后DEPrec=0.1586，增益0.0597[0.0232,0.0928]，但PD降至0.3288，完整检查仍未过。该阈值仅保留为候选，拼接诊断缓存不冒充正式细化。见[全量决定](runs/main_v1/interim_epoch025_full_guided/decision.json)。
- E25→26成熟batch-retention单因素对照综合增益区间跨0，MSE/MAE/方向退化，未采用或追加预算；CPU批次结构检查也不支持简单漏学或均值补丁。见[成熟对照](runs/batch_dropout_mature_trial/decision.json)、[结构检查](runs/main_v1/batch_structure_epoch025/decision.json)。

## 运行与恢复

环境、公开数据和缓存已准备好。首次数据步骤由[scripts/download_data.py](scripts/download_data.py)、[scripts/prepare_data.py](scripts/prepare_data.py)完成；版本锁定于[environment.yml](environment.yml)和[environment.lock.txt](environment.lock.txt)。已有环境与缓存时的接续命令：

```bash
conda activate ocoo-t
python scripts/run_experiment.py --config configs/main_selected.yaml --postprocess-plan logs/postprocess_workflow_update/plan.json
python scripts/status.py --run-dir runs/main_v1
```

流程：训练 → 原正CFG搜索/两个全量验证细化 → 完整0/.05候选检查 → 固定验证选择 → 全量测试/九指标报告。只在通过全部原质量检查的候选中按既有综合目标选择；无合格候选则停在测试前继续分析。原校准selected、raw与clamp0结果保留；任何正阈值均标记为原文未公开选择，测试保留同一raw的clamp0对照，参数先锁定再生成。

后续流程已通过35项CPU检查，并[安全接续原训练](logs/postprocess_workflow_update/deployment.json)，训练进程/优化器/日程保持。**当前尚未正式校准、采用0.05或访问测试。** 新选择器拒绝限量/拼接/来源不一致缓存，实际旧全量缓存0.05九指标重算差0；验证证据和源码归档见[核验记录](logs/postprocess_workflow_update/verification.json)。

同一命令可恢复，进程锁防止重复运行。`last.pt`每50步及每轮原子保存模型、AdamW、双EMA、随机状态和待办评估，生成按批落盘；旧完整断点SHA核验后转存现有挂载。README每15分钟汇总，日志/断点频率保持。进度见[训练曲线](runs/main_v1/progress.png)、`runs/main_v1/events.jsonl`；最终报告将写入`runs/main_v1/results/REPORT.md`，clamp0对照写入`results_clamp0/`。

## 关键记录（北京时间）

| 时间 | 进展 / 决策 |
| --- | --- |
| 11-17 00:54–01:02 | 建立ocoo-t环境，保存论文/数据，核对SHA、HVG及官方拆分。 |
| 11-17 07:00 | 开始第一版全量训练；明确的论文参数保持。 |
| 11-17 10:59 | 改为15分钟更新README；step1871安全保存后原配置续训。 |
| 11-18 | 条件干预、E10/E15生成和数量/近零诊断完成，差距仍在，保留原训练。 |
| 11-19 | 11组短对照、E20验证与CFG强度/区间检查完成，未采用无稳定收益的替代配置。 |
| 11-20 05:38–18:56 | E25全量/0.05诊断、成熟batch对照和CPU结构检查完成；保留主模型，阈值待新完整验证。 |
| 11-20 23:38 | E30完整模型/优化器/双EMA/RNG归档，随后完成周期生成。 |
| 11-21 00:39–00:46 | 两协议联合复核；安全接续验证流程，原训练保持；E35完整归档/单次CFG被动后续已登记。 |

详细设计、未知项与尝试见[implementation_notes.md](paper/implementation_notes.md)，完整时间记录见[progress_history.md](logs/progress_history.md)，重复旧摘要保留于发布备份。

范围：当前优先复现可核对数据与拆分的Table3主基准。Table5的18,080-gene公开引用与Replogle面板不一致，[官方来源审计](paper/long_panel_source_audit.json)已保存；control-context/MMD消融尚未运行，不以其他数据或补零面板冒充长面板实验。

- 2026-11-21T00:38:40+08:00：完成 `runs/main_v1/interim_epoch030_fixed_cfg/candidate` 的 validation 评估（细胞 960，weights=ema_0.9996，CFG=2.0）；指标见 `runs/main_v1/interim_epoch030_fixed_cfg/candidate/metrics/summary.json`。PDCorr=0.2872，MSE=0.02206。

- 2026-11-21T00:38:54+08:00：第30轮ordinary与固定w2的E25→30九指标配对分析完成，等待联合复核；未自动更改断点选择或收敛策略。见 `runs/main_v1/interim_epoch030_fixed_cfg/completed.json`。

- 2026-11-21T12:04:33+08:00：第35轮step39130完整模型/AdamW/双EMA/RNG已SHA核验归档，周期生成开始；固定CFG后续按登记自动等待。原参数/日程保持，目前无新的生成成绩，也未选择阈值或访问测试。见 `runs/main_v1/snapshots/step_39130/preserved.json`。

- 2026-11-21T12:14:59+08:00：`runs/main_v1` 第 35 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2961，MSE=0.00943，PDS_L2=0.7119。这是验证结果，非测试结果。

- 2026-11-21T12:15:05+08:00：`runs/main_v1` 完成第 35/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10394。

- 2026-11-21T12:16:07+08:00：第 35 轮缓存的 CPU 九指标检查完成：DEOver=0.0948、DEPrec=0.0784、DirAgr=0.7092；复用既有预测，无新增生成。完整结果见 `runs/main_v1/validation_epoch_035/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。
