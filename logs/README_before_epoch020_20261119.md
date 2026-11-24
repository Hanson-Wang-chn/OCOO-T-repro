# OCOO-T 复现（Replogle–Nadig）

在本机 A100 40 GB / conda `ocoo-t` 上独立实现 [OCOO-T v2](https://arxiv.org/abs/2606.12838v2)。仅下载并运行 Replogle–Nadig；Tahoe100M / PBMC 未下载。

**当前：11组短实验及九指标复核均已完成，保留原主配置。GPU仅用于全量主训练，下一周期评估为第20轮。正式CFG校准、最终验证与测试尚未开始，尚未达到论文成绩。**

<!-- LIVE_TRAINING_START -->

实时训练：`runs/main_v1`，epoch **20.004**，优化步 **22365**，loss **0.34291**，约 **58.6 cells/s**，更新于 2026-11-19T14:21:31+08:00。

<!-- LIVE_TRAINING_END -->

## 数据与模型

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，HepG2为4,976个。数据与Arc原发布文件SHA256一致，沿用公开efficacy过滤、库大小归一化/log1p、固定2,000 HVG与官方拆分，不重复归一化。保留56个batch；ESM2匹配3,545/3,618个联合基因，70个HVG的缺失处理见设计说明。验证含2个、测试标签含8个全局未见扰动，保留全部官方条件。

模型为12层、768宽、12头的连续基因Transformer，包含RMSNorm、QK-norm、SwiGLU、adaLN-zero和第4层起的32个条件token；直接在2,000 HVG上做v-prediction rectified flow，不patch。有效batch512（micro16、累积32）、lr1e-4、5轮warmup、cosine、BF16、双EMA保持论文设置；正式推理用EMA0.9996、50步Heun、末步Euler及原式双条件CFG。

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

第15轮周期EMA验证（普通条件场、cap32，未校准CFG）：PDCorr=0.2976、PDS_L2=0.6808、MSE=0.00904、DEPrec=0.0750。保留这些预测、补足至全量4,825个验证细胞后，PDCorr/PDS_L2/DEPrec为0.3491/0.6886/0.0880，MSE=0.00745，数量增加未消除DE/PDS差距。已完成实测数据的零值秩敏感性控制实验，另复查当前预测的近零处理；这些合成控制结果不作为模型成绩，尚未采用阈值。主训练继续，下一周期评估为第20轮。见 [第15轮完整诊断](runs/main_v1/interim_epoch015_full_conditional/completed.json)。

## 运行与恢复

环境已经建立。首次准备数据可运行 `scripts/download_data.py`、`scripts/prepare_data.py`；已有缓存直接使用：

```bash
conda activate ocoo-t
python scripts/run_experiment.py --config configs/main_selected.yaml
python scripts/status.py --run-dir runs/main_v1
```

工作流：训练 → 验证CFG校准（九指标）→ 完整验证/质量检查 → 全量测试 → 报告。相同命令恢复，进程锁防止重复写入。`last.pt`每50个优化步及每轮原子保存模型、优化器、双EMA、RNG和待办评估；生成按批落盘。README按要求每15分钟汇总，训练日志/断点频率不变。

训练前11项核心检查通过；最终校准流程4项CPU检查通过（含全量细化排名反转）；模型可选分支的5项CPU检查通过（含默认数值兼容、Eq6 loss/梯度/Heun及初始化解析检查）。原六组预测的array/完整DE入口前六项差值全部为0。版本锁定于 `environment.yml` / `environment.lock.txt`。进度见 [曲线](runs/main_v1/progress.png)、`runs/main_v1/events.jsonl`；最终报告路径为 `runs/main_v1/results/REPORT.md`。

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

详细思考与尝试见 [设计说明](paper/implementation_notes.md)，完整时间记录见 [progress_history.md](logs/progress_history.md)。当前先完成可核对数据与划分的Table3主基准；Table5的18,080-gene公开引用与Replogle面板不一致，已保存官方来源审计。control-context/MMD消融尚未运行，未将其他数据或补零面板冒充Replogle长面板。


- 2026-11-19T04:24:30+08:00：`runs/pilot_xpred_vloss` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.0974，MSE=0.05845，PDS_L2=0.5214。这是验证结果，非测试结果。

- 2026-11-19T04:24:36+08:00：`runs/pilot_xpred_vloss` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.54798。

- 2026-11-19T04:24:39+08:00：`runs/pilot_xpred_vloss` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-19T04:25:19+08:00：启动/续训 `runs/pilot_value_fanin`；训练细胞 16,384；有效 batch=512、microbatch=16；计划 12 epoch；噪声标准差=1.0。完整配置 `runs/pilot_value_fanin/config.yaml`。

- 2026-11-19T04:25:25+08:00：x-prediction+v-loss短对照、CPU九指标配对分析与完整断点SHA归档均完成；原全量v模型继续。结果 `runs/xpred_vloss_analysis/comparison.json` 等待综合复核，不自动换主模型，未访问测试表达。

- 2026-11-19T04:27:10+08:00：x-prediction+v-loss短对照完成：非负PDCorr0.0982/MSE0.0565/Dir0.5463，原综合差-0.13061，配对区间[-0.16832,-0.09368]；局部DE升高但整体更差，保留主v模型。完整断点已SHA归档；预登记fan-in v对照已顺序启动，首步目标/noise检查精确匹配，未访问测试。

- 2026-11-19T05:11:52+08:00：`runs/pilot_value_fanin` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75053。

- 2026-11-19T05:58:33+08:00：`runs/pilot_value_fanin` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.72597。

- 2026-11-19T06:31:03+08:00：`runs/pilot_value_fanin` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.2306，MSE=0.01538，PDS_L2=0.5358。这是验证结果，非测试结果。

- 2026-11-19T06:31:09+08:00：`runs/pilot_value_fanin` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.71481。

- 2026-11-19T06:31:13+08:00：`runs/pilot_value_fanin` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-19T06:31:59+08:00：标量值MLP fan-in初始化短对照、CPU九指标配对分析与完整断点SHA归档均完成；原全量v模型继续。结果 `runs/value_fanin_analysis/comparison.json` 等待综合复核，不自动换主模型，未访问测试表达。

- 2026-11-19T06:34:23+08:00：11组短实验全部完成并保留完整断点。fan-in组非负综合点估计+0.01556，但配对区间[-0.00784,0.03883]跨0，MSE/MAE及DEPrec变差，未修复整体性能；保留已成熟主轨迹。GPU恢复仅原全量训练，无额外短任务排队，下一关键评估为第20轮。见 `runs/value_fanin_analysis/decision.json`。

- 2026-11-19T14:20:48+08:00：`runs/main_v1` 第 20 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3001，MSE=0.00901，PDS_L2=0.7014。这是验证结果，非测试结果。

- 2026-11-19T14:20:54+08:00：`runs/main_v1` 完成第 20/100 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10411。
