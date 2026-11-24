# OCOO-T 复现（Replogle–Nadig）

在本机 A100 40 GB / conda `ocoo-t` 上独立实现 [OCOO-T v2](https://arxiv.org/abs/2606.12838v2)。仅下载并运行 Replogle–Nadig；Tahoe100M / PBMC 未下载。

**当前：冻结E35首组完整验证及0.05诊断完成，DEPrec仍未过预设检查；第二组全量验证与原主训练继续。最终设置未采用，测试尚未运行。**

<!-- LIVE_TRAINING_START -->

实时训练：`runs/main_v1`，epoch **36.986**，优化步 **41350**，loss **0.33299**，约 **30.5 cells/s**，更新于 2026-11-21T22:28:17+08:00。

<!-- LIVE_TRAINING_END -->

## 数据与模型

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，其中HepG2为4,976个。公开数据与Arc原文件SHA256一致；沿用已发布efficacy过滤、库大小归一化/log1p、固定2,000 HVG及官方拆分，不重复归一化。保留56个batch、全部官方条件；验证含2个、测试标签含8个全局未见扰动。ESM2匹配3,545/3,618个联合基因，缺失处理与来源见[设计说明](paper/implementation_notes.md)。

模型为12层/768宽/12头连续基因Transformer：RMSNorm、QK-norm、SwiGLU、adaLN-zero，从第4层起32个条件token，冻结5120维ESM2，2,000 HVG不patch，v-prediction rectified flow。保持有效batch512（micro16、累积32）、lr1e-4、5轮warmup、cosine、BF16及双EMA；复现默认用EMA0.9996推理（原文未指定两份EMA的选择规则）、均匀50步Heun/末步Euler、原式双条件CFG。原文8×H100，本机通过累积保持有效batch。

未公开选择：noise_std=1、time_scale=1、cell_init_std=1；最多100轮、至少20轮，按预登记验证点分数/10轮耐心及0.002最小增益判断。训练长度属于复现选择。每5轮生成验证；单训练进程约69–71 cells/s，每轮约2.3小时。明确参数、解释与试验依据均[单独记录](paper/implementation_notes.md)。

## 当前结果与尝试

下表是第35轮**限量验证诊断**；两行采样协议不同，绝对分数不直接比较，均非测试成绩。

| 协议 | 预测细胞 | PDCorr | PDS_L2 | MSE | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 普通条件场，cap32/B32 | 1,916 | 0.2961 | 0.7119 | 0.00943 | 0.0784 | 0.7092 |
| 固定w2/[0.1,0.9]，cap16/B16 | 960 | 0.2931 | 0.7967 | 0.02312 | 0.0875 | 0.7046 |

E30→35的综合差（PDCorr+meanPDS）：ordinary +0.00955[-0.01015,0.03235]，固定CFG +0.01116[-0.00762,0.03130]，均跨0。固定CFG的MSE增0.00106[0.00024,0.00211]、MAE增0.00232[0.00110,0.00376]；其余七项差值区间均跨0。全部九项均值/差值/2000次区间独立核算最大差0，DirAgr按官方方式保留53个有效条件。见[联合决定](runs/main_v1/interim_epoch035_fixed_cfg/decision.json)、[配对图](runs/main_v1/interim_epoch035_fixed_cfg/joint_review/paired_protocols.png)。

两套点排序均偏好E35，保持原模型、训练日程和收敛规则；best/early_reference按原规则为E35，最早可能E45停止，后续改善会顺延。完整模型/AdamW/双EMA/RNG已[SHA归档](runs/main_v1/snapshots/step_39130/preserved.json)，193个推理tensor与默认EMA及buffers一致；下一周期E40[完整归档和一次固定CFG比较](runs/main_v1/interim_epoch040_fixed_cfg/plan.json)已登记并被动等待。

冻结E35的[中途校准](runs/main_v1/interim_epoch035_calibration/plan.json)已完成六组粗筛，按原综合目标选出w2/[0,1]与w2/[0.1,0.9]各做完整4,825细胞/B32/seed4242/Heun50。首组全区间及预登记CPU0.05诊断已完成：

| 首组全量处理 | PDCorr | PDS_L2 | MSE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clamp0 | 0.3614 | 0.8125 | 0.01820 | 0.1770 | 0.1012 | 0.7931 |
| 原文未公开的0.05候选 | 0.3591 | 0.8119 | 0.01822 | 0.1963 | 0.1542 | 0.7951 |

0.05改善DE点估计，但仍缺DEPrec的原0.18复现检查线，PD及综合略降，尚未采用；这不是论文测试成绩。完整九指标、来源与限制见[提前诊断结论](runs/main_v1/interim_epoch035_calibration/first_full_floor_diagnostic/decision.json)。第二组部分区间全量生成继续，之后仍须原四候选完整检查、提前/正式0.05结果核对及[独立配对区间审计](runs/main_v1/interim_epoch035_calibration/analysis_plan.json)。

另已登记[成熟双EMA配对检查](runs/main_v1/ema_epoch035_review/plan.json)：CPU验证两份193-tensor推理装载与同一E35完整断点一致；当前校准和审计结束后，只补0.9999普通/全区间CFG各960个验证细胞，复用同索引/噪声的0.9996参考。4项协议检查通过，尚未启动这项GPU生成，不增加训练或自动更换EMA。

原主训练及预设规则保持，第36轮完整恢复备份已[独立核查](logs/preservation_epoch036/verification.json)：189个AdamW slots、双EMA、四类RNG和E35评估后的元数据保留；没有另起恢复训练。不同B16诊断缓存不混入本次B32校准。

已完成的主要尝试：

- 同一16,384训练细胞/384更新/seed42，完成noise×time×cell-init八组及独立投影、x预测、fan-in共11组短对照。保留当前共享投影/v预测/Xavier主模型；fan-in综合点估计略高但区间跨0，误差与DEPrec更差，不据短试跑推断成熟结果。详见[八组网格](runs/pilot_grid_complete/summary.json)、[补充三组](runs/supplemental_trials_complete.json)。这些是raw短预算诊断，正式推理始终用成熟EMA。
- E20七种CFG强度及区间检查显示：w2附近辨别收益趋平，更强引导增加误差；保留两区间供成熟验证，未擅改原公式或50步。见[强度](runs/main_v1/interim_epoch020_all_scales/decision.json)、[区间](runs/main_v1/interim_epoch020_interval_review/decision.json)。
- E25全4,825细胞固定CFG诊断：PD=0.3306、PDS_L2=0.7908、DEPrec=0.0989；同一raw样本用0.05处理后DEPrec=0.1586，增益0.0597[0.0232,0.0928]，但PD降至0.3288，完整检查仍未过。该阈值仅保留为候选，拼接诊断缓存不冒充正式细化。见[全量决定](runs/main_v1/interim_epoch025_full_guided/decision.json)。
- E25→26成熟batch-retention单因素对照综合增益区间跨0，MSE/MAE/方向退化，未采用或追加预算；CPU批次结构检查也不支持简单漏学或均值补丁。见[成熟对照](runs/batch_dropout_mature_trial/decision.json)、[结构检查](runs/main_v1/batch_structure_epoch025/decision.json)。
- 旧E25缓存的[有限采样CPU诊断](runs/main_v1/mc_variance_epoch025/decision.json)仅覆盖部分可识别方差；额外同方差假设下proxy约为原MSE的16%，不能当作误差占比或修正成绩，未据此改生成数量。

## 运行与恢复

环境、公开数据和缓存已准备好。首次数据步骤由[scripts/download_data.py](scripts/download_data.py)、[scripts/prepare_data.py](scripts/prepare_data.py)完成；版本锁定于[environment.yml](environment.yml)和[environment.lock.txt](environment.lock.txt)。已有环境与缓存时的接续命令：

```bash
conda activate ocoo-t
python scripts/run_experiment.py --config configs/main_selected.yaml --postprocess-plan logs/postprocess_workflow_update/plan.json
python scripts/status.py --run-dir runs/main_v1
```

流程：训练 → 原正CFG搜索/两个全量验证细化 → 完整0/.05候选检查 → 固定验证选择 → 全量测试/九指标报告。以本复现预设检查（非论文公开的验证阈值）筛选完整候选，再按既有综合目标选择；无合格候选则停在测试前继续分析。原校准selected、raw与clamp0结果保留；任何正阈值均标记为原文未公开选择，测试保留同一raw的clamp0对照，参数先锁定再生成。

后续流程已通过35项CPU检查，并[安全接续原训练](logs/postprocess_workflow_update/deployment.json)，训练进程/优化器/日程保持。**中途E35完整验证校准已启动；训练结束后的最终选择和测试尚未执行，0.05仍是候选。** 中途控制器另通过4项CPU进程/中断/无合格候选检查；只操作自己的验证子进程，缓存可恢复。新选择器拒绝限量/拼接/来源不一致缓存，实际旧全量缓存0.05九指标重算差0；原核验及源码见[核验记录](logs/postprocess_workflow_update/verification.json)。

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
| 11-21 12:03–13:18 | E35完整归档/联合复核完成；原训练继续，启动冻结E35全量验证校准，登记E40后续。 |
| 11-21 17:13–17:27 | E36完整恢复备份核查通过；E35六组粗筛完成，开始两个全4,825细胞验证细化。 |
| 11-21 18:45 | 旧E25缓存有限采样CPU诊断完成，保留原指标与采样配置。 |
| 11-21 21:46–21:50 | E35首组全量及提前0.05检查完成，仍缺DEPrec；第二组全量继续。 |
| 11-21 21:56 | 成熟双EMA两病例检查登记、CPU核查通过；等本轮完整审计后再运行。 |

详细设计、未知项与尝试见[implementation_notes.md](paper/implementation_notes.md)，完整时间记录见[progress_history.md](logs/progress_history.md)，重复旧摘要保留于发布备份。

范围：当前优先复现可核对数据与拆分的Table3主基准。Table5的18,080-gene公开引用与Replogle面板不一致，[官方来源审计](paper/long_panel_source_audit.json)已保存；control-context/MMD消融尚未运行，不以其他数据或补零面板冒充长面板实验。




- 2026-11-21T21:59:07+08:00：成熟双EMA验证检查已登记，4项协议检查及真实E35两份193-tensor CPU装载核对通过；仅补0.9999普通/全区间CFG各960个验证细胞，复用0.9996同索引/噪声参考。须等当前全量校准及独立审计完成后再启动，尚未新增GPU生成或训练。见 `runs/main_v1/ema_epoch035_review/plan.json`。
