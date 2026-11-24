# OCOO-T 复现（Replogle–Nadig）

在本机 A100 40 GB / conda `ocoo-t` 上独立实现 [OCOO-T v2](https://arxiv.org/abs/2606.12838v2)。仅下载并运行 Replogle–Nadig；Tahoe100M / PBMC 未下载。

**当前：原全量训练已完成27轮并继续，下一周期第30轮。成熟batch dropout对照未显示稳定收益，误差和方向指标退化，保留原设计。正式校准与测试尚未开始。**

<!-- LIVE_TRAINING_START -->

实时训练：`runs/main_v1`，epoch **27.929**，优化步 **31225**，loss **0.33294**，约 **68.3 cells/s**，更新于 2026-11-20T18:43:47+08:00。

<!-- LIVE_TRAINING_END -->

## 数据与模型

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，HepG2为4,976个。数据与Arc原发布文件SHA256一致，沿用公开efficacy过滤、库大小归一化/log1p、固定2,000 HVG与官方拆分，不重复归一化。保留56个batch；ESM2匹配3,545/3,618个联合基因，70个HVG的缺失处理见设计说明。验证含2个、测试标签含8个全局未见扰动，保留全部官方条件。

模型为12层、768宽、12头的连续基因Transformer，包含RMSNorm、QK-norm、SwiGLU、adaLN-zero和第4层起的32个条件token；直接在2,000 HVG上做v-prediction rectified flow，不patch。有效batch512（micro16、累积32）、lr1e-4、5轮warmup、cosine、BF16、双EMA保持论文设置；正式推理用EMA0.9996、50步Heun、末步Euler及原式双条件CFG。原文使用8×H100，本机1×A100通过累积保持相同有效batch。

未公开细节及解释见 [implementation_notes.md](paper/implementation_notes.md)。当前配置noise_std=1、time_scale=1、cell_init_std=1；训练上限100轮、至少20轮后按预登记验证收敛规则停止，属于原文未公开训练长度的复现选择。单训练进程约69–71 cells/s、每轮约2.3小时。每5轮做EMA生成验证，最终只用验证集校准正尺度双CFG，原始与非负预测均保存。

## 小实验与当前观察

相同16,384个训练细胞、seed42、12轮/384个优化步、60个验证条件、每条件最多16细胞、50步Heun。下表为raw权重的非负后处理诊断，**不是成熟EMA或正式测试成绩**。

| 噪声σ | 时间尺度 | cell初始std | PDCorr | PDS_L2 | MSE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 1 | 0.02 | -0.0100 | 0.5039 | 0.07767 |
| 0.5 | 1 | 0.02 | 0.1088 | 0.5175 | 0.04038 |
| 1 | 1000 | 0.02 | -0.0007 | 0.5036 | 0.07374 |
| 0.5 | 1000 | 0.02 | 0.1292 | 0.5281 | 0.04472 |
| 1 | 1 | 1 | 0.2299 | 0.5183 | 0.01357 |
| 0.5 | 1 | 1 | 0.2070 | 0.5308 | 0.01624 |
| 1 | 1000 | 1 | 0.2280 | 0.5197 | 0.01428 |
| 0.5 | 1000 | 1 | 0.2161 | 0.5261 | 0.01649 |

raw/非负综合点估计最高者均仍为noise1/time1/cell std1。补充两个time1000组合未显示稳定综合收益，因此保留正在训练的主模型。完整九指标、固定扰动库上的配对区间、预算审计与输入哈希见 [八组分析](runs/pilot_grid_complete/summary.json) 和 [复核决定](runs/main_selection_after_grid.json)。区间不表示独立训练种子的不确定性。

三个补充对照也已完成：独立投影和x预测未改善综合指标；fan-in初始化的非负综合点估计提高0.0156，但配对区间[-0.0078,0.0388]跨0，MSE/MAE及DEPrec更差。保留已训练的共享投影/v预测/Xavier主模型；不据短实验否定其他设计的成熟性能。原第五组仍是八组网格内最高点估计，全部11组中fan-in点估计略高，不能混淆。见 [11组汇总](runs/supplemental_trials_complete.json)、[投影决定](runs/pert_projection_analysis/decision.json)、[x预测决定](runs/xpred_vloss_analysis/decision.json) 和 [初始化决定](runs/value_fanin_analysis/decision.json)。

第25轮周期EMA验证（普通条件场、cap32，未校准CFG）：PDCorr=0.2987、PDS_L2=0.6967、MSE=0.00921、DEPrec=0.0774。相对第20轮综合仅+0.00279，配对区间[-0.01873,0.02527]跨0；仅PDS_cos的配对区间高于0，不能宣称整体继续明显改善。原模型按预登记收敛规则继续，下一周期第30轮，见 [配对决定](runs/main_v1/observation/epoch020_to025/decision.json)。step27950完整模型/优化器/双EMA/RNG已SHA核验保存，见 [断点记录](runs/main_v1/snapshots/step_27950/preserved.json)。

同协议条件探针中，第25轮EMA验证置乱扰动的velocity损失增量较第12轮增长0.001361[0.000842,0.001983]。随后固定w2/[0.1,0.9]/cap16/B16的E20→25生成比较完成：综合+0.04075[0.01885,0.06420]，PDCorr和三项PDS的配对增量区间均高于0；DEPrec下降0.00551[-0.01162,-0.00022]。这是同一cap16协议内的跨轮比较，与周期cap32的绝对分数不同。原模型继续训练，见 [条件探针决定](runs/main_v1/conditioning_epoch_025/decision.json) 和 [固定CFG决定](runs/main_v1/interim_epoch025_fixed_cfg/decision.json)。

E25固定w2全量诊断已完成：4,825验证细胞的PDCorr/PDS_L2/DEPrec为0.3306/0.7908/0.0989，DEOver=0.1682、DirAgr=0.7120、MSE=0.01446。原960个raw/非负预测逐字节保留，额外3,865个采用预登记B32/seed62525。相同预测设0.05阈值后，DEPrec=0.1586，增量0.0597[0.0232,0.0928]，34个条件升、14个降；PDCorr=0.3288，DEOver=0.1943，综合差的区间跨0。保留该后处理候选待独立成熟验证，正式设置仍为clamp0；有限数量和零支持尚未解决PD/DE精度差距。九指标与配对分析见 [全量决定](runs/main_v1/interim_epoch025_full_guided/decision.json)。

成熟batch dropout单因素对照已完成：同一E25完整状态续训1118更新至E26，比较置空cell-line时是否保留batch。保留batch的综合差+0.00165[-0.00460,0.00786]，未显示稳定收益；MSE增加0.000296、MAE增加0.000982、DirAgr下降0.01040，三项配对区间均显示退化，0.05处理下结论相同。本轮不采用或追加该候选，原主训练继续；不据单轮试验否定所有更长替代训练。见 [决定](runs/batch_dropout_mature_trial/decision.json)、[九指标配对图](runs/batch_dropout_mature_trial/paired_comparison_readable.png)。

冻结E20的七强度对照齐全（含ordinary，cap16/B16，与周期协议不同）：w2相对匹配ordinary的综合+0.1164[0.0523,0.1857]。w2.5较w2仅+0.0043[-0.0052,0.0153]，MSE/MAE明显变差，w3无进一步综合收益，结束强度扩展。固定w2后，全区间较[0.1,0.9]提高PDCorr/DEOver/DEPrec，但MSE/MAE也变差；综合+0.0138[-0.0088,0.0371]。保留两区间待成熟模型正式验证，不自动锁定设置或改门槛。见 [强度决定](runs/main_v1/interim_epoch020_all_scales/decision.json) 和 [区间决定](runs/main_v1/interim_epoch020_interval_review/decision.json)。

第15轮曾将原预测扩展到全量4,825验证细胞，PDCorr/PDS_L2/DEPrec为0.3491/0.6886/0.0880；数量增加未解决DE/PDS差距。零值秩敏感性及阈值诊断也未支持整体修复，未采用额外阈值；合成控制结果不作为模型成绩。见 [第15轮全量诊断](runs/main_v1/interim_epoch015_full_conditional/completed.json)。

## 运行与恢复

环境已经建立。首次准备数据可运行 `scripts/download_data.py`、`scripts/prepare_data.py`；已有缓存直接使用：

```bash
conda activate ocoo-t
python scripts/run_experiment.py --config configs/main_selected.yaml
python scripts/status.py --run-dir runs/main_v1
```

工作流：训练 → 验证CFG校准（九指标）→ 完整验证/质量检查 → 全量测试 → 报告。相同命令恢复，进程锁防止重复写入。`last.pt`每50个优化步及每轮原子保存模型、优化器、双EMA、RNG和待办评估；生成按批落盘。README按要求每15分钟汇总，训练日志/断点频率不变。 正式校准的全量选中样本将经SHA、设置和全量索引核验后复用于完整验证，并在CPU重算九指标，避免重复生成；工作流可接续等待已运行的训练。见 [流程更新](runs/final_validation_cache_reuse_update.json)。

训练前11项核心检查通过；后续校准/缓存复用/训练接续13项CPU检查通过（含全量排名反转、真实DE重算一致、中断恢复与进程锁）；模型可选分支的5项CPU检查通过（含默认数值兼容、Eq6 loss/梯度/Heun及初始化解析检查）。正CFG全量扩展4项CPU检查通过（路由、前缀保留、恢复及坏缓存拒绝）。原六组预测的array/完整DE入口前六项差值全部为0。版本锁定于 `environment.yml` / `environment.lock.txt`。进度见 [曲线](runs/main_v1/progress.png)、`runs/main_v1/events.jsonl`；最终报告路径为 `runs/main_v1/results/REPORT.md`。

## 关键记录（北京时间）

| 时间 | 进展 / 决策 |
| --- | --- |
| 11-17 00:54–01:02 | 建立环境，保存论文与公开数据；核对SHA256、固定HVG及Table1拆分。 |
| 11-17 01:06–06:59 | 完成原六组；排查生成中的细胞条件偏差，比较初始化后选择cell std1。 |
| 11-17 03:10 | 验证压缩包后移除22.30GB重复解压副本，保留原始压缩包与训练缓存。 |
| 11-17 04:38 | 纠正v1 HTML/v2 PDF版本标签，保存两版与差异；主实验参数未变。 |
| 11-17 05:54–06:56 | 修复轮末评估中断恢复，加入进程锁；完成11项核心检查。 |
| 11-17 07:00 | 从头开始第一版全量训练；明确的论文参数保持不变。 |
| 11-17 10:59 | 按要求改为15分钟更新README，在step1871安全保存并原配置续训。 |
| 11-18 02:21 | 配对干预确认模型在利用扰动/时间条件，仍不能等同于生成达标。 |
| 11-18 07:14–09:09 | 完成第10轮EMA生成、DE/采样量/稀疏性诊断与配对CFG；未采用无收益的近零阈值。 |
| 11-18 09:10–13:16 | 顺序补齐两个交互组合，完成完整2×2×2网格及九指标复核；维持原主配置。 |
| 11-18 12:16 | 主训练完成第11轮，保存完整断点并更新曲线；正式校准和测试尚未开始。 |
| 11-18 22:26 | 第15轮生成与九指标完成；PDS相对第10轮改善，保留主训练。 |
| 11-18 23:13–23:27 | 全量验证数量/零值秩/阈值诊断完成；数量和阈值未解决整体差距，不采用额外阈值。 |
| 11-18 23:36–11-19 01:55 | 完成独立投影对照；保留共享投影；已完成断点SHA校验后迁到现有用户挂载。 |
| 11-19 02:00–04:25 | 末段时间/FP32探针及x预测对照完成；保留BF16/time1/v预测。 |
| 11-19 04:25 | 启动预登记fan-in初始化对照；主模型和论文明确设置不变。 |
| 11-19 06:32 | 11组短实验全部完成，fan-in证据不足以重启；GPU恢复仅原全量主训练，等待第20轮生成。 |
| 11-19 14:21–14:26 | 第20轮九指标完成，PDS继续改善；原模型续训，启动冻结断点的5-case验证CFG粗筛。 |
| 11-19 17:51 | CFG粗筛证实辨别收益及误差代价；复用参考，只追加w2.5/3检查上界。 |
| 11-19 19:29–19:34 | 上界已包围，收益接近平顶；原模型完成21轮，固定w2新增一次全区间对照。 |
| 11-19 20:28 | 区间对照完成，未消除指标取舍；结束第20轮推理诊断，GPU恢复仅原主训练，等待第25轮评估。 |
| 11-20 05:38–05:51 | 第25轮生成与九指标完成；完整step27950断点已SHA归档。综合增益变弱，保持原训练并做同协议条件探针。 |
| 11-20 05:56–06:02 | 留出扰动去噪依赖继续增强；只新增一次固定CFG的跨断点生成比较，不重新扫引导。 |
| 11-20 06:14 | 后续流程13项CPU检查通过；部署缓存复用及主进程接续，原训练进程保持运行。 |
| 11-20 06:50–07:15 | 固定CFG跨轮九指标继续显示辨别进展；近零阈值未支持整体修复，启动同一冻结模型的全量验证数量诊断。 |
| 11-20 10:27–10:35 | 全量CFG/0.05配对诊断完成，仍有PD/DE精度差距；原E26完整断点已SHA归档，登记成熟单因素短续训。 |
| 11-20 10:45 | 兼容性/状态继承/预算检查通过，启动同一E25→26成熟短续训；原100轮LR日程和主训练保持。 |
| 11-20 15:56 | 候选精确完成1118更新，raw/AdamW/双EMA/RNG与默认EMA导出核验通过；开始两端配对生成，原主模型已完成27轮。 |
| 11-20 17:32–17:35 | 成熟对照九指标/配对分析完成，误差与方向退化，保留原batch处理；GPU恢复仅主训练，等待E30复核。 |

详细思考与尝试见 [设计说明](paper/implementation_notes.md)，完整时间记录见 [progress_history.md](logs/progress_history.md)。当前先完成可核对数据与划分的Table3主基准；Table5的18,080-gene公开引用与Replogle面板不一致，已保存官方来源审计。control-context/MMD消融尚未运行，未将其他数据或补零面板冒充Replogle长面板。


- 2026-11-20T17:31:05+08:00：完成 `runs/batch_dropout_mature_trial/candidate` 的 validation 评估（细胞 960，weights=ema_0.9996，CFG=2.0）；指标见 `runs/batch_dropout_mature_trial/candidate/metrics/summary.json`。PDCorr=0.2903，MSE=0.02096。

- 2026-11-20T17:31:43+08:00：成熟batch dropout单因素短续训及两组固定CFG九指标完成，均为step29068；待配对复核，不自动更改主模型；结果 `runs/batch_dropout_mature_trial/completed.json`。

- 2026-11-20T17:32:39+08:00：成熟batch dropout两组固定CFG九指标配对分析完成：floor_0综合差+0.00165[-0.00460,+0.00786]；floor_0p05综合差+0.00201[-0.00382,+0.00795]。仍需结合九指标取舍复核，不自动切换主模型或改正式推理；见 `runs/batch_dropout_mature_trial/paired_comparison.json`。

- 2026-11-20T17:35:40+08:00：成熟batch dropout对照完整复核：综合+0.00165[-0.00460,0.00786]未稳定改善，MSE/MAE和DirAgr的区间均显示退化，0.05处理亦同；本轮不采用或追加候选，保留原主训练并等待E30。完整raw/AdamW/双EMA/RNG、两端九指标和图表均保留；见 `runs/batch_dropout_mature_trial/decision.json`。

- 2026-11-20T17:39:56+08:00：第30轮联合复核已登记并设置被动后续：原主训练继续；到E30完整断点与CPU观察就绪且GPU无其他辅助任务后，仅新增一次固定w2/960细胞生成，复用E25参考，同时比较ordinary与CFG的25→30九指标，不自动改选型/收敛规则。见 `runs/main_v1/interim_epoch030_fixed_cfg/plan.json`。

- 2026-11-20T18:38:22+08:00：等待E30期间登记CPU批次结构误差分解：复用E25全量拼接预测，消除扰动均值后检查共享批次误差；4项独立OLS/识别性/留出恢复数学检查通过，输入SHA及源码已固定。无新GPU任务或正式设置更改；见 `runs/main_v1/batch_structure_epoch025/plan.json`。
