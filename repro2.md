# OCOO-T Replogle–Nadig 第二轮复现归档

第二轮从随机初始化重训，在北京时间 2026-11-24 19:34 按验证规则于 E45 / step 50,310 停止。E35 / step 39,130 的 EMA 0.9996 入选，正式测试于 2026-11-25 00:51 完成。全部 26,878 个测试细胞及 380 个扰动均已核对，真实端和预测端共用 4,976 个 HepG2 对照，裁负及 0.05 两种评估复用同一批原始预测。 [训练结果](runs/repro2_main/finished.json) · [完整性核验](runs/repro2_main/test_completion_verification.json)

## 参数与实现

| 项目 | 第二轮设置 |
| --- | --- |
| 基准 | 官方 Replogle–Nadig 2,000 HVG，训练 572,545 个细胞 / 2,013 个扰动 / 四种细胞系，验证 4,825 / 60，测试 26,878 / 380。沿用原始高变基因顺序、归一化及 log1p。 |
| 模型 | 136,919,041 参数，12 层 Transformer、宽 768、12 头、RMSNorm、QK-norm、SwiGLU ratio 4（JiT 参数匹配后的宽度 2,048）、adaLN-zero，第 4 层起前置 32 个上下文 token。 |
| 编码 | 1→768→768 表达 MLP，固定一维位置编码幅度 1，时间正弦编码 256 维、时间尺度 1。冻结的 5,120 维 ESM2 通过共享 Xavier 投影为基因身份和扰动条件。细胞系及 56 个 batch 标签嵌入相加，两者的初始化标准差为 1 和 0.02。 |
| 流匹配 | Gaussian 噪声标准差 1，logit-normal N(−0.8, 0.8²)，速度预测与 L2，扰动及细胞条件各自以 0.1 独立置空。 |
| 优化与调度 | seed 42、AdamW β=(0.9,0.95)、weight decay 0、BF16，峰值 lr 1e−4，线性 warmup 5 轮，60 轮预算的 cosine 至 0。最低 20 轮，周期综合分连续 10 轮未达到 0.002 改善时早停。 |
| 四卡执行 | 4×A100 PCIe 40 GB、每卡 microbatch 16×累积 8、全局 batch 512，每轮 1,118 次更新。保留原生 DDP 和核心模型编译，关闭 Dynamo DDPOptimizer 分图。每 50 步和轮末原子存盘。 |
| 验证与采样 | 双 EMA 0.9999 / 0.9996，周期验证用后者和双 CFG w=1.5/[0.1,0.9]，每扰动最多 32 个生成样本。所有正式采样均用论文式 11、均匀 50 步 Heun、末步 Euler。 |
| 冻结测试 | E35 EMA 0.9996，w=1.5、[0,1]、0.05 近零截断，生成 batch 32、seed 4242。Cell Eval 0.8.2 / pdex 0.3.0，按扰动宏平均，保留同 raw 的裁负对照。 |

配置、来源和选择细节见 [全量配置](configs/repro2_main.yaml)、[评测登记](configs/repro2_postprocess.json)、[冻结协议](runs/repro2_main/test_full_postprocess/test_protocol.json) 和 [论文 v2](https://arxiv.org/html/2606.12838v2)。

## 运行及选型

1. 用相同 16,384 个训练细胞、384 次更新及 960 个固定验证索引比较四卡对照、位置幅度 0.125、共享投影权重初始化 ×2、SwiGLU 宽度 3,072。相对对照的综合分差依次为 −0.00347、−0.00321、+0.00305，配对区间均跨零，三个候选的 MSE 均增加。沿用原位置幅度、共享投影和 2,048 维 SwiGLU。 [短实验结果](runs/repro2_pilots/summary.csv)
2. E45 完整断点的四卡短回放测得 B16/B32 为 268.30/269.50 cells/s，峰值分配为 18.47/32.02 GB。选择 B16 留出采样显存。生成用保持批次与噪声种子的分片，单卡与多卡实测缓存逐元素一致。 [微批回放](logs/repro2/bench_summary.json) · [生成核验](runs/repro2_gpu_sharding_smoke/verification.json)
3. 主训练含检查点与周期生成耗时 30 小时 43 分钟。E35 周期综合分 1.13456 为最高。E45 较 E35 低 0.02036，配对区间 [−0.05467,+0.01165]。末步保存了优化器、双 EMA 和各 rank 的 RNG。 [末步权重核验](runs/repro2_main/training_completion_verification.json) · [周期验证配对](runs/repro2_main/observation/pair_e035_e045/comparison.json)
4. 筛选 12 组正尺度双 CFG 和一组普通条件，按规则选择 w=2 和 w=1.5 两个全时间区间候选，分别补齐 4,825 个验证细胞并比较两个阈值。依据相对短缺规则选定 w=1.5 和 0.05。验证 PDCorr 0.37385、PDS_L2 0.81333、MSE 0.01146、DEPrec 0.17334。DEPrec 低于项目 0.18 检查线，冻结记录保留这项短缺，随后执行一次完整测试。 [完整验证](runs/repro2_main/calibration/postprocessing/review.json) · [与首轮验证配对](runs/repro2_main/full_validation_comparison.json)

## 正式测试与论文差距

| 指标 | 论文表 3 | 裁负对照 | 阈值 0.05 | 差值 | 95% 扰动区间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDCorr | 0.43700 | 0.40321 | 0.40058 | -0.03642 | [0.36672, 0.43773] |
| PDS_L1 | 0.82000 | 0.83073 | 0.83136 | +0.01136 | [0.80943, 0.85100] |
| PDS_L2 | 0.85300 | 0.84415 | 0.84435 | -0.00865 | [0.82420, 0.86359] |
| PDS_cos | 0.83300 | 0.80551 | 0.80520 | -0.02780 | [0.77865, 0.83026] |
| MSE | 0.00790 | 0.01099 | 0.01107 | +0.00317 | [0.01030, 0.01190] |
| MAE | 0.06100 | 0.06910 | 0.06949 | +0.00849 | [0.06745, 0.07148] |
| DEOver | 0.23200 | 0.18642 | 0.23892 | +0.00692 | [0.21116, 0.26779] |
| DEPrec | 0.24000 | 0.10347 | 0.20700 | -0.03300 | [0.18368, 0.23045] |
| DirAgr | 0.76600 | 0.80658 | 0.81160 | +0.04560 | [0.78452, 0.83761] |

PDS_L1、DEOver 和 DirAgr 的正式测试点估计高于论文。PDCorr 和 DEPrec 分别低 0.03642、0.03300，MSE 高 0.00317。首轮止于完整验证，跨轮变化见匹配的验证对照。区间由 2,000 次扰动 bootstrap 得到。 [正式报告](runs/repro2_main/results/REPORT.md) · [九指标明细](runs/repro2_main/results/benchmark_comparison.csv) · [论文表 3](https://arxiv.org/html/2606.12838v2)

## 第二轮原 README 全文

原文件共 66,569 字节，SHA-256 为 `bd31c93cc5ccc1f25bba9ce6c6ba15ddc52fb376db380ffecff854880d7baf65`。下方从原一级标题开始逐字节保存。

# OCOO-T Replogle–Nadig 第二轮复现

第一轮训练、四卡故障与恢复、完整验证、九项指标及原 README 全文已收入 [repro1.md](repro1.md)。本轮目标是完成官方 2,000 HVG 短面板的全训练集重训，并用 26,878 个细胞、380 个扰动完成一次冻结的全量测试。

<!-- LIVE_TRAINING_START -->

第二轮已完成。四卡全量训练在 E45 结束，完整验证选定 E35 的正式测试配置。正式测试覆盖 26,878 个细胞及 380 个扰动，两份报告已生成。 [正式报告](runs/repro2_main/results/REPORT.md)

<!-- LIVE_TRAINING_END -->

## 论文约束

保持官方拆分、发布的 library-size normalization 和 log1p、2,000 HVG、速度预测和 L2、logit-normal N(−0.8, 0.8²)、独立 p/c dropout 0.1、12 层 / 768 / 12 头、RMSNorm、QK-norm、SwiGLU ratio 4、adaLN-zero、第 4 层起 32 条件 token、冻结 5,120 维 ESM2 和共享身份投影、AdamW 与有效 batch 512、5 轮 warmup/cosine、BF16、双 EMA、默认 EMA 0.9996、式 11 双 CFG 和 50 步均匀 Heun（末步 Euler）。使用四张 A100 以梯度累积实现同一全局 batch。 [论文 §3、表 3、附录 A–B](https://arxiv.org/html/2606.12838v2)

## 改进计划

1. **输入与条件尺度。** 图 1 要求将位置编码加到基因和表达 token，幅度未公开。首轮初始化中，位置 RMS 0.707、基因先验 0.137、标量表达每单位编码变化约 0.025，扰动间投影 RMS 0.085，cell embedding RMS 0.987。配对比较原设置、位置幅度 0.125、共享 ESM2 投影初始化权重 ×2。每次只动一个量，保持公式、输入和参数类型。
2. **FFN 实现核对。** 附录写 SwiGLU ratio 4。已有实现按 JiT 的参数匹配口径得到 2,048 中间维度。另比较字面 4×768=3,072 的实现，明确记录两种口径与计算开销。保留效果和依据更充分的实现。
3. **短实验与选型。** 四组使用同一 16,384 训练细胞、seed 42、12 轮 / 384 次优化、独立四卡 DDP、全局 batch 512。固定 960 个验证索引与初始噪声，比较九指标及配对扰动 bootstrap。短跑末端原始权重用于诊断，另保存两组 EMA 和固定去噪探针。训练候选综合分为 PDCorr + mean(三种 PDS)，并同时检查 MSE、MAE 与 DE。若收益小于采样不确定性，优先采用论文依据和后续全数据验证更强的选择。
4. **采样与 DE。** 利用首轮的两组完整验证 raw 缓存复查近零支持；阈值候选 0 / 0.025 / 0.05 / 0.065 / 0.07 均低于观测到的 HepG2 训练正值下界。用冻结的验证配置选择 CFG 强度、时间区间和全局近零阈值，保留同 raw 的裁负对照。DE 仍使用相同 Cell Eval 与真实对照。重训后先做固定索引筛选，再按完整验证复核。
5. **四卡重训与正式评测。** 保留原生 DDP，关闭已确认会改变梯度的 Dynamo DDPOptimizer 分图。测量 microbatch、显存、速度、完整断点读回和生成分片一致性。在新运行目录重训，约每 15 分钟汇总 README。保留首轮质量线作为达标检查，由完整验证冻结测试设置，完整报告未达标项及表 3 的实测差距。
   
## 验证冻结规则

模型短跑选型以配对分数、表达误差和 DE 的共同表现为依据。复查中无明确收益的选择使用现有实现。短跑的配对区间只描述这 60 个扰动，最终表现仍由全数据训练的默认 EMA 验证。

重训后的验证筛选预定 w∈{0.75,1,1.25,1.5,1.75,2} 和 [0.1,0.9]、[0,1]。完整生成综合分最高的候选，再完整生成综合分距最高不超过 0.04 的候选中 MSE 最低的一组。没有进入该分差范围的候选时，使用得分第二的配置。两组各生成 4,825 个细胞，逐一检查裁负和 0.05。

选择达成全部首轮检查线的最高综合分候选。如果所有候选均有短缺，选择各检查相对短缺之和最小的候选，并在正式报告列出未达到的检查线。冻结内容包含权重哈希、完整验证指标、CFG、后处理、索引、批次和种子。测试只运行一次，正式比较使用论文表 3 的同一拆分。

## 短实验结果与模型选择

每组 384 次更新。对照和三个候选用同一训练子集、960 个验证索引和同一生成噪声。这里使用裁负后的原始权重生成，验证真实端使用全部 4,825 个细胞。配对区间按同一批 60 个扰动重采样 2,000 次。 [完整记录](runs/repro2_pilots/summary.json) · [CSV](runs/repro2_pilots/summary.csv)

| 配置 | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr | 综合分差和 95% 配对区间 | cells/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 四卡对照 | 0.22211 | 0.51417 | 0.52222 | 0.51056 | 0.01384 | 0.08279 | 0.05122 | 0.05363 | 0.63172 | 对照 | 265.6 |
| 位置幅度 0.125 | 0.21920 | 0.51472 | 0.52000 | 0.51056 | 0.01499 | 0.08639 | 0.05349 | 0.05259 | 0.61987 | -0.00347 [-0.01575, +0.00951] | 265.7 |
| 投影初始化 ×2 | 0.22011 | 0.51417 | 0.52000 | 0.50917 | 0.01414 | 0.08322 | 0.05265 | 0.05210 | 0.61703 | -0.00321 [-0.01209, +0.00560] | 266.0 |
| SwiGLU 3,072 | 0.22730 | 0.51639 | 0.51750 | 0.50667 | 0.01489 | 0.08679 | 0.05469 | 0.05524 | 0.65269 | +0.00305 [-0.00534, +0.01101] | 230.3 |

全量模型沿用位置幅度 1 和共享 ESM2 Xavier 投影。两项改动的末轮综合分点估计下降，表达 MSE 增加。SwiGLU 3,072 在 PDCorr 和 DEPrec 的点估计略好，综合分区间跨零，MSE 增加 0.00106，训练吞吐下降约 13%。因此选择沿用 JiT 口径的 ratio 4 参数匹配实现（中间 2,048、约 1.369 亿参数）。两种实现都保留了 ratio 配置 4，区别是门控后的中间维度。全量阶段的重点是余弦退火终点、使用双 CFG 选取检查点及完整验证校准。

## 全量训练配置

全量训练从 step 0 开始。训练集为 572,545 个细胞，每轮 1,118 次更新。微批量 16，每卡累积 8 次，四卡全局 batch 512。E45 回放中 B16/B32 分别为 268.30/269.50 cells/s，rank 0 峰值分配 18.47/32.02 GB。选择 B16 保留生成时的显存余量。每 50 步和每轮末原子保存完整断点。 [四卡回放](logs/repro2/bench_summary.json)

周期生成使用四卡、Heun 50 步、默认 EMA 0.9996、式 11 的 w=1.5 和 [0.1,0.9]。每次最多 32 个预测细胞/扰动，使用固定验证索引。单、三卡验证样本的 raw 数组经逐元素核验。选出模型后，完整验证和测试同样按全局原批次顺序和种子分片。 [训练配置](configs/repro2_main.yaml) · [并行核验](runs/repro2_gpu_sharding_smoke/verification.json) · [代码回归](logs/repro2/cpu_tests_final.log)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=6 \
  /home/whs/miniconda3/envs/ocoo-t/bin/python scripts/run_experiment.py --config configs/repro2_main.yaml --gpu-workers 4 --calibration-scales 0.75,1,1.25,1.5,1.75,2 --compare-windows --postprocess-plan configs/repro2_postprocess.json
```

## 全量训练结果

从随机初始化在四张 A100 上训练至 E45 / step 50,310，耗时 30 小时 43 分钟，包含断点、去噪检查和周期生成。E35 / step 39,130 的默认 EMA 综合分 1.13456 为最高。E45 相对 E35 的综合分差为 -0.02036，95% 配对区间 [-0.05467, +0.01165]。末步训练状态、四卡 RNG、双 EMA 和冻结的 E35 权重均已读回核验。 [训练完成](runs/repro2_main/finished.json) · [断点核验](runs/repro2_main/training_completion_verification.json) · [E35→E45 配对](runs/repro2_main/observation/pair_e035_e045/comparison.json)

周期验证每次生成 1,916 个细胞，使用固定 w=1.5、[0.1,0.9]、默认 EMA 和裁负。表中的综合分用于选取检查点。

| 轮次 | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr | 综合分 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| [E5](runs/repro2_main/validation_epoch_005/metrics_de_diagnostic/summary.json) | 0.19726 | 0.54139 | 0.54444 | 0.51583 | 0.02962 | 0.11271 | 0.11609 | 0.10776 | 0.62403 | 0.73115 |
| [E10](runs/repro2_main/validation_epoch_010/metrics_de_diagnostic/summary.json) | 0.25966 | 0.64139 | 0.65056 | 0.66917 | 0.01058 | 0.06970 | 0.09582 | 0.07383 | 0.65835 | 0.91336 |
| [E15](runs/repro2_main/validation_epoch_015/metrics_de_diagnostic/summary.json) | 0.28873 | 0.71583 | 0.72667 | 0.73806 | 0.01024 | 0.07048 | 0.10632 | 0.08368 | 0.67585 | 1.01558 |
| [E20](runs/repro2_main/validation_epoch_020/metrics_de_diagnostic/summary.json) | 0.31987 | 0.75361 | 0.77444 | 0.77417 | 0.01057 | 0.07128 | 0.11597 | 0.08827 | 0.69247 | 1.08728 |
| [E25](runs/repro2_main/validation_epoch_025/metrics_de_diagnostic/summary.json) | 0.32539 | 0.76472 | 0.78389 | 0.80444 | 0.01136 | 0.07240 | 0.11904 | 0.08923 | 0.73472 | 1.10974 |
| [E30](runs/repro2_main/validation_epoch_030/metrics_de_diagnostic/summary.json) | 0.32927 | 0.78806 | 0.80389 | 0.81361 | 0.01203 | 0.07302 | 0.12111 | 0.09018 | 0.75406 | 1.13112 |
| [E35](runs/repro2_main/validation_epoch_035/metrics_de_diagnostic/summary.json) | 0.32771 | 0.78722 | 0.80194 | 0.83139 | 0.01237 | 0.07407 | 0.11981 | 0.08922 | 0.73986 | 1.13456 |
| [E40](runs/repro2_main/validation_epoch_040/metrics_de_diagnostic/summary.json) | 0.31966 | 0.78639 | 0.80694 | 0.82306 | 0.01292 | 0.07549 | 0.12177 | 0.08908 | 0.71398 | 1.12513 |
| [E45](runs/repro2_main/validation_epoch_045/metrics_de_diagnostic/summary.json) | 0.32105 | 0.77222 | 0.79389 | 0.81333 | 0.01347 | 0.07692 | 0.12085 | 0.08961 | 0.73184 | 1.11420 |

## 完整验证与正式测试配置

两个引导候选各生成 4,825 个细胞、60 个扰动。表中每个阈值都使用同一个候选的原始样本。 [四组验证](runs/repro2_main/calibration/postprocessing/review.json) · [冻结记录](runs/repro2_main/test_full_postprocess/test_protocol.json)

| w / 区间 / 截断 | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr | 未达到检查线 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 2 / [0,1] / 0 | 0.36376 | 0.81056 | 0.81833 | 0.82778 | 0.01734 | 0.08037 | 0.18508 | 0.10039 | 0.79154 | DEPrec |
| 2 / [0,1] / 0.05 | 0.36069 | 0.81222 | 0.81750 | 0.82667 | 0.01736 | 0.08040 | 0.20259 | 0.16555 | 0.78795 | DEPrec |
| 1.5 / [0,1] / 0 | 0.37715 | 0.80333 | 0.81194 | 0.81972 | 0.01141 | 0.06873 | 0.16140 | 0.09791 | 0.78484 | DEOver、DEPrec |
| 1.5 / [0,1] / 0.05 | 0.37385 | 0.80389 | 0.81333 | 0.81806 | 0.01146 | 0.06897 | 0.19885 | 0.17334 | 0.79818 | DEPrec |

按预先登记的相对短缺规则选定 **E35 EMA 0.9996、w=1.5、[0,1]、阈值 0.05**。DEPrec 为 0.17334，仍低于项目检查线 0.18。完成样本、九指标及配置哈希复核后，于北京时间 2026-11-24 21:57 启动正式测试。正式测试同时保留同一批 raw 样本的裁负对照。

与首轮 E45 的完整验证相比，在匹配 w=2、[0,1]、阈值 0.05 时，第二轮 MSE 下降 0.00171，DEPrec 上升 0.00692，PDCorr 和三种 PDS 均下降。采用最终 w=1.5 后，相对首轮最佳验证候选的 MSE 从 0.01907 降至 0.01146，DEPrec 从 0.15863 升至 0.17334，PDCorr 从 0.36513 升至 0.37385，PDS_L2 从 0.83444 降至 0.81333。最终配置比较同时反映训练和 CFG 选择差异。 [完整验证配对差值及区间](runs/repro2_main/full_validation_comparison.json)

## 正式测试结果

第二轮于北京时间 2026-11-25 00:51 完成。正式评测从冻结协议到报告用时约 2 小时 54 分钟，覆盖 26,878 个生成细胞、380 个扰动和 4,976 个共享真实 HepG2 对照。两种后处理使用同一批原始样本。 [正式报告](runs/repro2_main/results/REPORT.md) · [裁负报告](runs/repro2_main/results_clamp0/REPORT.md) · [完整性核验](runs/repro2_main/test_completion_verification.json)

| 指标 | 论文表 3 | 裁负对照 | 选定阈值 0.05 | 选定值减论文值 |
| --- | ---: | ---: | ---: | ---: |
| PDCorr | 0.43700 | 0.40321 | 0.40058 | -0.03642 |
| PDS_L1 | 0.82000 | 0.83073 | 0.83136 | +0.01136 |
| PDS_L2 | 0.85300 | 0.84415 | 0.84435 | -0.00865 |
| PDS_cos | 0.83300 | 0.80551 | 0.80520 | -0.02780 |
| MSE | 0.00790 | 0.01099 | 0.01107 | +0.00317 |
| MAE | 0.06100 | 0.06910 | 0.06949 | +0.00849 |
| DEOver | 0.23200 | 0.18642 | 0.23892 | +0.00692 |
| DEPrec | 0.24000 | 0.10347 | 0.20700 | -0.03300 |
| DirAgr | 0.76600 | 0.80658 | 0.81160 | +0.04560 |

PDS_L1、DEOver 和 DirAgr 的点估计高于论文。PDCorr 低 0.03642，DEPrec 低 0.03300，MSE 高 0.00317，PDS_cos 低 0.02780。0.05 是论文未公开的后处理选择，由完整验证决定。测试集中，同样本裁负的 DEPrec 为 0.10347，应用 0.05 后为 0.20700。首轮只有完整验证，跨轮改进以匹配验证表为依据。 [论文 v2 表 3](https://arxiv.org/html/2606.12838v2) · [九指标与区间](runs/repro2_main/results/benchmark_comparison.csv)

## 实验和进度记录


- 2026-11-23T11:33:32+08:00：启动/续训 `runs/repro2_pilots/control`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro2_pilots/control/config.yaml`。

- 2026-11-23T11:40:09+08:00：`runs/repro2_pilots/control` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75203。

- 2026-11-23T11:45:17+08:00：短实验对照组已用四卡 DDP 完成热身，普通更新窗口约 265–273 cells/s。每卡峰值分配约 18.44 GB（rank 0）。四组配置、预算和配对索引协议见 [试验登记](runs/repro2_pilots/plan.json)。三组候选的末轮生成按整批分片加速，保留对照组的全局批次及随机种子。

- 2026-11-23T11:45:17+08:00：冻结 E45 两组各 4,825 个 raw 验证样本的 CPU 阈值对照完成。全区间 0/.025/.05/.065/.07 的 DEPrec 依次为 0.10192/0.14862/0.15863/0.15892/0.15896。区间 [0.1,0.9] 的 .05/.065/.07 为 0.15861/0.15887/0.15884。0.05 以上的最高增益约 0.00033，DEOver 同时略降。正式校准继续比较 0 和 0.05。训练最小正值 0.07279，所有结果使用原九项指标。 [全区间](runs/repro2_floor_audit/full_window/comparison.json) · [部分区间](runs/repro2_floor_audit/partial_window/comparison.json)

- 2026-11-23T11:45:17+08:00：CPU 回归通过 91 项，GPU 相关的 3 项留待实机。覆盖旧模型默认值逐元素一致、两项新选项的 RNG、生成分片的种子/顺序/断点合并，以及测试审核的验证短缺记录。 [测试日志](logs/repro2/cpu_tests.log)

- 2026-11-23T11:46:07+08:00：`runs/repro2_pilots/control` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73601。

- 2026-11-23T11:51:13+08:00：完成 `runs/repro2_gpu_sharding_smoke/single` 的 validation 评估（细胞 60，weights=ema_0.9996，CFG=1.5）；指标见 `runs/repro2_gpu_sharding_smoke/single/metrics/summary.json`。PDCorr=0.1049，MSE=0.13797。

- 2026-11-23T11:52:24+08:00：完成 `runs/repro2_gpu_sharding_smoke/sharded` 的 validation 评估（细胞 60，weights=ema_0.9996，CFG=1.5）；指标见 `runs/repro2_gpu_sharding_smoke/sharded/metrics/summary.json`。PDCorr=0.1049，MSE=0.13797。

- 2026-11-23T11:52:26+08:00：A100 实机生成分片检查通过。同一 E45、60 个固定验证样本、B8、原式 CFG 1.5 和 Heun 50 步，单卡与三卡的原始及裁负数组、索引和六项指标逐项一致。 [核验](runs/repro2_gpu_sharding_smoke/verification.json)

- 2026-11-23T11:53:30+08:00：真实 CUDA 的训练与采样回归 5 项通过。另用内存模拟数据完成三卡 DDP 双 EMA 广播、分片合并和完整断点训练。最后一轮生成使用的 EMA 参数与完整断点逐元素一致，保存全部 rank 的 RNG。 [GPU 测试](logs/repro2/gpu_tests.log) · [DDP 检查](logs/repro2/ddp_smoke.log)

- 2026-11-23T11:55:00+08:00：`runs/repro2_pilots/control` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.2221，MSE=0.01384，PDS_L2=0.5222。这是验证结果，非测试结果。

- 2026-11-23T11:55:05+08:00：`runs/repro2_pilots/control` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.72906。

- 2026-11-23T11:55:12+08:00：`runs/repro2_pilots/control` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-23T11:55:21+08:00：启动/续训 `runs/repro2_pilots/position_0p125`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro2_pilots/position_0p125/config.yaml`。

- 2026-11-23T11:55:49+08:00：384 步短实验 `control` 已完成 960 个固定验证样本和九项评估。PDCorr=0.22211，PDS_L2=0.52222，DEPrec=0.05363，MSE=0.01384。综合分相对四卡对照 +0.00000，扰动配对区间 [+0.00000, +0.00000]。 [完整对照](runs/repro2_pilots/summary.json)

- 2026-11-23T11:56:35+08:00：全数据阶段计划从随机初始化重训，最多 60 轮、最少 20 轮、连续 10 轮无达到 0.002 的综合分改善时停止。首轮 E45 后继续以较高学习率训练，E50 和 E55 的固定周期验证均回落。本轮将余弦退火终点设在 E60，以降低成熟阶段的学习率，5 轮 warmup 和峰值 1e-4 按论文。每 5 轮使用默认 EMA 及固定式 11 w=1.5、[0.1,0.9] 生成，采样与正式推理方法一致。最终 w、区间及阈值按完整验证冻结。 [评测登记](configs/repro2_postprocess.json)

- 2026-11-23T12:01:23+08:00：`runs/repro2_pilots/position_0p125` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74576。

- 2026-11-23T12:07:20+08:00：`runs/repro2_pilots/position_0p125` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.71998。

- 2026-11-23T12:11:31+08:00：E45 全量验证的误差分解显示，w=2 时全区间与部分区间的整面板均方误差为 0.01905/0.01577。其中各扰动共享的表达偏差项分别为 0.00082/0.00091。正式细化将同时保留高扰动分数及低表达误差候选。 [误差与 DE 计数](runs/repro2_floor_audit/bias_and_de_counts.json)

- 2026-11-23T12:11:42+08:00：`runs/repro2_pilots/position_0p125` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.2192，MSE=0.01499，PDS_L2=0.5200。这是验证结果，非测试结果。

- 2026-11-23T12:11:47+08:00：`runs/repro2_pilots/position_0p125` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.70899。

- 2026-11-23T12:12:02+08:00：`runs/repro2_pilots/position_0p125` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-23T12:12:11+08:00：启动/续训 `runs/repro2_pilots/gene_init_2`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro2_pilots/gene_init_2/config.yaml`。

- 2026-11-23T12:12:29+08:00：384 步短实验 `position_0p125` 已完成 960 个固定验证样本和九项评估。PDCorr=0.21920，PDS_L2=0.52000，DEPrec=0.05259，MSE=0.01499。综合分相对四卡对照 -0.00347，扰动配对区间 [-0.01575, +0.00951]。 [完整对照](runs/repro2_pilots/summary.json)

- 2026-11-23T12:13:08+08:00：位置幅度 0.125 的优势集中在短跑热身。step 128 的 raw endpoint MSE 为 0.15647，对照为 0.22660。step 384 收窄为 0.12060 对 0.12156。末轮生成的综合分差 −0.00347，95% 配对区间 [−0.01575, +0.00951]，MSE 从 0.01384 变为 0.01499。暂沿用原位置幅度。四卡分片完整覆盖 256/256/224/224 个样本。末轮探针到生成结束用时约 2.1 分钟，对照单卡同形生成约 6.6 分钟，两次编译缓存状态不同。 [配对表](runs/repro2_pilots/summary.csv)

- 2026-11-23T12:13:08+08:00：首轮全区间 w=2 的 DE 调用数中位数在裁负时为 1,415.5，阈值 0.05 时为 172，真实端为 28.5（FDR<0.05）。近零截断明显减少过多显著基因，精确率差距还涉及扰动条件下的表达误差。 [DE 计数](runs/repro2_floor_audit/bias_and_de_counts.json)

- 2026-11-23T12:18:17+08:00：`runs/repro2_pilots/gene_init_2` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75092。

- 2026-11-23T12:24:15+08:00：`runs/repro2_pilots/gene_init_2` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73257。

- 2026-11-23T12:28:25+08:00：`runs/repro2_pilots/gene_init_2` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.2201，MSE=0.01414，PDS_L2=0.5200。这是验证结果，非测试结果。

- 2026-11-23T12:28:30+08:00：`runs/repro2_pilots/gene_init_2` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.72494。

- 2026-11-23T12:28:37+08:00：`runs/repro2_pilots/gene_init_2` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-23T12:28:45+08:00：启动/续训 `runs/repro2_pilots/swiglu_literal`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro2_pilots/swiglu_literal/config.yaml`。

- 2026-11-23T12:29:09+08:00：384 步短实验 `gene_init_2` 已完成 960 个固定验证样本和九项评估。PDCorr=0.22011，PDS_L2=0.52000，DEPrec=0.05210，MSE=0.01414。综合分相对四卡对照 -0.00321，扰动配对区间 [-0.01209, +0.00560]。 [完整对照](runs/repro2_pilots/summary.json)

- 2026-11-23T12:36:13+08:00：`runs/repro2_pilots/swiglu_literal` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75216。

- 2026-11-23T12:43:06+08:00：`runs/repro2_pilots/swiglu_literal` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73669。

- 2026-11-23T12:48:08+08:00：`runs/repro2_pilots/swiglu_literal` 第 12 轮验证生成（原始权重诊断，每扰动最多 16 细胞，50 步）：PDCorr=0.2273，MSE=0.01489，PDS_L2=0.5175。这是验证结果，非测试结果。

- 2026-11-23T12:48:15+08:00：`runs/repro2_pilots/swiglu_literal` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73002。

- 2026-11-23T12:48:23+08:00：`runs/repro2_pilots/swiglu_literal` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-23T12:49:11+08:00：384 步短实验 `swiglu_literal` 已完成 960 个固定验证样本和九项评估。PDCorr=0.22730，PDS_L2=0.51750，DEPrec=0.05524，MSE=0.01489。综合分相对四卡对照 +0.00305，扰动配对区间 [-0.00534, +0.01101]。 [完整对照](runs/repro2_pilots/summary.json)

- 2026-11-23T12:50:30+08:00：E45 完整断点上完成四卡 microbatch 短基准。每项 2 步热身、12 步计时、同一全局 batch 512。B16 268.3 cells/s，峰值分配 18.47 GB。B32 269.5 cells/s，峰值分配 32.02 GB。 [基准记录](logs/repro2/bench_summary.json)

- 2026-11-23T12:51:00+08:00：第二轮全量工作流已启动，训练、完整验证校准、锁定配置与正式评测连续执行。 [启动记录](logs/repro2/main_launch.json)

- 2026-11-23T12:51:07+08:00：启动/续训 `runs/repro2_main`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro2_main/config.yaml`。

- 2026-11-23T12:53:29+08:00：全量训练首个完整断点 step 50 已读回核验。189 个 AdamW 槽位、双 EMA 与四个 rank 的 RNG 均匹配。 [断点核验](logs/repro2/first_checkpoint_verification.json)

- 2026-11-23T13:06:36+08:00：`runs/repro2_main` 训练至 E0.403、step 450。最近窗口中位吞吐 266.0 cells/s、loss 0.63681、lr 8.05e-06。已落盘完整断点 step 450。固定探针 step 250，raw/EMA endpoint MSE 0.68143/0.76162。

- 2026-11-23T13:22:11+08:00：`runs/repro2_main` 训练至 E0.805、step 900。最近窗口中位吞吐 266.5 cells/s、loss 0.39686、lr 1.61e-05。已落盘完整断点 step 900。固定探针 step 750，raw/EMA endpoint MSE 0.13109/0.72917。

- 2026-11-23T13:37:47+08:00：`runs/repro2_main` 训练至 E1.208、step 1,350。最近窗口中位吞吐 266.0 cells/s、loss 0.38150、lr 2.42e-05。已落盘完整断点 step 1,350。固定探针 step 1,250，raw/EMA endpoint MSE 0.11984/0.68646。

- 2026-11-23T13:53:22+08:00：`runs/repro2_main` 训练至 E1.619、step 1,810。最近窗口中位吞吐 266.0 cells/s、loss 0.36891、lr 3.24e-05。已落盘完整断点 step 1,800。固定探针 step 1,750，raw/EMA endpoint MSE 0.11515/0.63891。

- 2026-11-23T14:08:58+08:00：`runs/repro2_main` 训练至 E2.021、step 2,260。最近窗口中位吞吐 265.2 cells/s、loss 0.36055、lr 4.04e-05。已落盘完整断点 step 2,250。固定探针 step 2,250，raw/EMA endpoint MSE 0.11515/0.56334。

- 2026-11-23T14:10:33+08:00：前两轮全量训练含检查点和去噪探针的墙钟用时为 38.6/38.0 分钟。按这段实测速率，从 step 0 跑到上限 E60 约 38.3 小时，周期生成和最终验证/测试另计。实际停止轮数由预设早停规则决定。

- 2026-11-23T14:24:33+08:00：`runs/repro2_main` 训练至 E2.428、step 2,715。最近窗口中位吞吐 265.0 cells/s、loss 0.35692、lr 4.86e-05。已落盘完整断点 step 2,700。固定探针 step 2,500，raw/EMA endpoint MSE 0.11283/0.53418。

- 2026-11-23T14:40:08+08:00：`runs/repro2_main` 训练至 E2.831、step 3,165。最近窗口中位吞吐 265.2 cells/s、loss 0.35058、lr 5.66e-05。已落盘完整断点 step 3,150。固定探针 step 3,000，raw/EMA endpoint MSE 0.11075/0.43577。

- 2026-11-23T14:55:44+08:00：`runs/repro2_main` 训练至 E3.229、step 3,610。最近窗口中位吞吐 265.4 cells/s、loss 0.35219、lr 6.46e-05。已落盘完整断点 step 3,600。固定探针 step 3,500，raw/EMA endpoint MSE 0.10977/0.32575。

- 2026-11-23T15:11:19+08:00：`runs/repro2_main` 训练至 E3.631、step 4,060。最近窗口中位吞吐 266.1 cells/s、loss 0.35209、lr 7.26e-05。已落盘完整断点 step 4,050。固定探针 step 4,000，raw/EMA endpoint MSE 0.10910/0.26219。

- 2026-11-23T15:26:55+08:00：`runs/repro2_main` 训练至 E4.030、step 4,505。最近窗口中位吞吐 266.2 cells/s、loss 0.34713、lr 8.06e-05。已落盘完整断点 step 4,500。固定探针 step 4,500，raw/EMA endpoint MSE 0.10865/0.22388。

- 2026-11-23T15:42:30+08:00：`runs/repro2_main` 训练至 E4.436、step 4,960。最近窗口中位吞吐 266.0 cells/s、loss 0.35047、lr 8.87e-05。已落盘完整断点 step 4,950。固定探针 step 4,750，raw/EMA endpoint MSE 0.10905/0.20882。

- 2026-11-23T15:58:05+08:00：`runs/repro2_main` 训练至 E4.839、step 5,410。最近窗口中位吞吐 266.0 cells/s、loss 0.34913、lr 9.68e-05。已落盘完整断点 step 5,400。固定探针 step 5,250，raw/EMA endpoint MSE 0.10846/0.18459。

- 2026-11-23T16:13:41+08:00：`runs/repro2_main` 训练至 E5.000、step 5,590。最近窗口中位吞吐 265.6 cells/s、loss 0.34587、lr 0.0001。已落盘完整断点 step 5,590。固定探针 step 5,590，raw/EMA endpoint MSE 0.10773/0.17204。

- 2026-11-23T16:15:49+08:00：`runs/repro2_main` 第 5 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.1973，MSE=0.02962，PDS_L2=0.5444。这是验证结果，非测试结果。

- 2026-11-23T16:16:01+08:00：`runs/repro2_main` 完成第 5/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.17204。

- 2026-11-23T16:16:49+08:00：第 5 轮缓存的 CPU 九指标检查完成：DEOver=0.1161、DEPrec=0.1078、DirAgr=0.6240；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_005/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-23T16:28:44+08:00：`runs/repro2_main` 训练至 E5.335、step 5,965。最近窗口中位吞吐 266.2 cells/s、loss 0.34263、lr 0.0001。已落盘完整断点 step 5,950。固定探针 step 5,750，raw/EMA endpoint MSE 0.10773/0.16713。

- 2026-11-23T16:44:20+08:00：`runs/repro2_main` 训练至 E5.747、step 6,425。最近窗口中位吞吐 264.9 cells/s、loss 0.34700、lr 0.0001。已落盘完整断点 step 6,400。固定探针 step 6,250，raw/EMA endpoint MSE 0.10756/0.15348。

- 2026-11-23T16:59:55+08:00：`runs/repro2_main` 训练至 E6.149、step 6,875。最近窗口中位吞吐 264.2 cells/s、loss 0.34173、lr 9.99e-05。已落盘完整断点 step 6,850。固定探针 step 6,750，raw/EMA endpoint MSE 0.10751/0.14231。

- 2026-11-23T17:15:31+08:00：`runs/repro2_main` 训练至 E6.561、step 7,335。最近窗口中位吞吐 264.6 cells/s、loss 0.34455、lr 9.98e-05。已落盘完整断点 step 7,300。固定探针 step 7,250，raw/EMA endpoint MSE 0.10670/0.13334。

- 2026-11-23T17:31:06+08:00：`runs/repro2_main` 训练至 E6.968、step 7,790。最近窗口中位吞吐 264.7 cells/s、loss 0.33976、lr 9.97e-05。已落盘完整断点 step 7,750。固定探针 step 7,750，raw/EMA endpoint MSE 0.10660/0.12622。

- 2026-11-23T17:46:42+08:00：`runs/repro2_main` 训练至 E7.379、step 8,250。最近窗口中位吞吐 264.9 cells/s、loss 0.33841、lr 9.95e-05。已落盘完整断点 step 8,200。固定探针 step 8,000，raw/EMA endpoint MSE 0.10647/0.12330。

- 2026-11-23T18:02:17+08:00：`runs/repro2_main` 训练至 E7.786、step 8,705。最近窗口中位吞吐 264.7 cells/s、loss 0.34403、lr 9.94e-05。已落盘完整断点 step 8,700。固定探针 step 8,500，raw/EMA endpoint MSE 0.10592/0.11863。

- 2026-11-23T18:17:53+08:00：`runs/repro2_main` 训练至 E8.193、step 9,160。最近窗口中位吞吐 265.3 cells/s、loss 0.34992、lr 9.92e-05。已落盘完整断点 step 9,150。固定探针 step 9,000，raw/EMA endpoint MSE 0.10632/0.11521。

- 2026-11-23T18:33:28+08:00：`runs/repro2_main` 训练至 E8.600、step 9,615。最近窗口中位吞吐 265.0 cells/s、loss 0.33660、lr 9.89e-05。已落盘完整断点 step 9,600。固定探针 step 9,500，raw/EMA endpoint MSE 0.10594/0.11271。

- 2026-11-23T18:49:03+08:00：`runs/repro2_main` 训练至 E9.007、step 10,070。最近窗口中位吞吐 265.0 cells/s、loss 0.34281、lr 9.87e-05。已落盘完整断点 step 10,062。固定探针 step 10,062，raw/EMA endpoint MSE 0.10576/0.11058。

- 2026-11-23T19:04:39+08:00：`runs/repro2_main` 训练至 E9.419、step 10,530。最近窗口中位吞吐 265.9 cells/s、loss 0.34180、lr 9.84e-05。已落盘完整断点 step 10,500。固定探针 step 10,500，raw/EMA endpoint MSE 0.10579/0.10935。

- 2026-11-23T19:20:15+08:00：`runs/repro2_main` 训练至 E9.835、step 10,995。最近窗口中位吞吐 264.8 cells/s、loss 0.33857、lr 9.81e-05。已落盘完整断点 step 10,950。固定探针 step 10,750，raw/EMA endpoint MSE 0.10586/0.10879。

- 2026-11-23T19:35:51+08:00：`runs/repro2_main` 训练至 E10.000、step 11,180。最近窗口中位吞吐 265.0 cells/s、loss 0.34062、lr 9.8e-05。已落盘完整断点 step 11,180。固定探针 step 11,180，raw/EMA endpoint MSE 0.10562/0.10795。

- 2026-11-23T19:37:25+08:00：`runs/repro2_main` 第 10 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2597，MSE=0.01058，PDS_L2=0.6506。这是验证结果，非测试结果。

- 2026-11-23T19:37:28+08:00：`runs/repro2_main` 完成第 10/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10795。

- 2026-11-23T19:38:04+08:00：第 10 轮缓存的 CPU 九指标检查完成：DEOver=0.0958、DEPrec=0.0738、DirAgr=0.6584；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_010/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-23T19:50:54+08:00：`runs/repro2_main` 训练至 E10.349、step 11,570。最近窗口中位吞吐 264.8 cells/s、loss 0.33800、lr 9.77e-05。已落盘完整断点 step 11,550。固定探针 step 11,500，raw/EMA endpoint MSE 0.10550/0.10743。

- 2026-11-23T20:06:30+08:00：`runs/repro2_main` 训练至 E10.751、step 12,020。最近窗口中位吞吐 264.9 cells/s、loss 0.33474、lr 9.73e-05。已落盘完整断点 step 12,000。固定探针 step 12,000，raw/EMA endpoint MSE 0.10517/0.10679。

- 2026-11-23T20:22:08+08:00：`runs/repro2_main` 训练至 E11.149、step 12,465。最近窗口中位吞吐 264.6 cells/s、loss 0.33248、lr 9.69e-05。已落盘完整断点 step 12,450。固定探针 step 12,298，raw/EMA endpoint MSE 0.10564/0.10647。

- 2026-11-23T20:37:43+08:00：`runs/repro2_main` 训练至 E11.552、step 12,915。最近窗口中位吞吐 264.7 cells/s、loss 0.34124、lr 9.65e-05。已落盘完整断点 step 12,900。固定探针 step 12,750，raw/EMA endpoint MSE 0.10506/0.10606。

- 2026-11-23T20:53:21+08:00：`runs/repro2_main` 训练至 E11.950、step 13,360。最近窗口中位吞吐 264.7 cells/s、loss 0.34422、lr 9.61e-05。已落盘完整断点 step 13,350。固定探针 step 13,250，raw/EMA endpoint MSE 0.10510/0.10569。

- 2026-11-23T21:09:01+08:00：`runs/repro2_main` 训练至 E12.343、step 13,800。最近窗口中位吞吐 265.2 cells/s、loss 0.34096、lr 9.57e-05。已落盘完整断点 step 13,800。固定探针 step 13,750，raw/EMA endpoint MSE 0.10502/0.10541。

- 2026-11-23T21:24:36+08:00：`runs/repro2_main` 训练至 E12.746、step 14,250。最近窗口中位吞吐 265.9 cells/s、loss 0.33591、lr 9.52e-05。已落盘完整断点 step 14,250。固定探针 step 14,250，raw/EMA endpoint MSE 0.10514/0.10517。

- 2026-11-23T21:40:13+08:00：`runs/repro2_main` 训练至 E13.144、step 14,695。最近窗口中位吞吐 266.0 cells/s、loss 0.33444、lr 9.47e-05。已落盘完整断点 step 14,650。固定探针 step 14,534，raw/EMA endpoint MSE 0.10535/0.10507。

- 2026-11-23T21:55:48+08:00：`runs/repro2_main` 训练至 E13.547、step 15,145。最近窗口中位吞吐 265.2 cells/s、loss 0.33408、lr 9.42e-05。已落盘完整断点 step 15,100。固定探针 step 15,000，raw/EMA endpoint MSE 0.10523/0.10492。

- 2026-11-23T22:11:23+08:00：`runs/repro2_main` 训练至 E13.945、step 15,590。最近窗口中位吞吐 265.2 cells/s、loss 0.34185、lr 9.36e-05。已落盘完整断点 step 15,550。固定探针 step 15,500，raw/EMA endpoint MSE 0.10532/0.10478。

- 2026-11-23T22:27:00+08:00：`runs/repro2_main` 训练至 E14.338、step 16,030。最近窗口中位吞吐 265.0 cells/s、loss 0.33755、lr 9.31e-05。已落盘完整断点 step 16,000。固定探针 step 16,000，raw/EMA endpoint MSE 0.10526/0.10467。

- 2026-11-23T22:42:35+08:00：`runs/repro2_main` 训练至 E14.741、step 16,480。最近窗口中位吞吐 265.1 cells/s、loss 0.33402、lr 9.25e-05。已落盘完整断点 step 16,450。固定探针 step 16,250，raw/EMA endpoint MSE 0.10489/0.10463。

- 2026-11-23T22:58:11+08:00：`runs/repro2_main` 训练至 E15.000、step 16,770。最近窗口中位吞吐 265.4 cells/s、loss 0.34513、lr 9.21e-05。已落盘完整断点 step 16,770。固定探针 step 16,770，raw/EMA endpoint MSE 0.10483/0.10453。

- 2026-11-23T23:03:38+08:00：`runs/repro2_main` 第 15 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2887，MSE=0.01024，PDS_L2=0.7267。这是验证结果，非测试结果。

- 2026-11-23T23:03:43+08:00：`runs/repro2_main` 完成第 15/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10453。

- 2026-11-23T23:04:05+08:00：第 15 轮缓存的 CPU 九指标检查完成：DEOver=0.1063、DEPrec=0.0837、DirAgr=0.6758；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_015/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-23T23:13:15+08:00：`runs/repro2_main` 训练至 E15.246、step 17,045。最近窗口中位吞吐 265.6 cells/s、loss 0.34205、lr 9.17e-05。已落盘完整断点 step 17,000。固定探针 step 17,000，raw/EMA endpoint MSE 0.10527/0.10449。

- 2026-11-23T23:28:51+08:00：`runs/repro2_main` 训练至 E15.653、step 17,500。最近窗口中位吞吐 265.8 cells/s、loss 0.33811、lr 9.1e-05。已落盘完整断点 step 17,450。固定探针 step 17,250，raw/EMA endpoint MSE 0.10490/0.10445。

- 2026-11-23T23:44:27+08:00：`runs/repro2_main` 训练至 E16.047、step 17,940。最近窗口中位吞吐 266.0 cells/s、loss 0.34234、lr 9.04e-05。已落盘完整断点 step 17,900。固定探针 step 17,888，raw/EMA endpoint MSE 0.10480/0.10440。

- 2026-11-24T00:00:02+08:00：`runs/repro2_main` 训练至 E16.453、step 18,395。最近窗口中位吞吐 265.9 cells/s、loss 0.32815、lr 8.97e-05。已落盘完整断点 step 18,350。固定探针 step 18,250，raw/EMA endpoint MSE 0.10484/0.10437。

- 2026-11-24T00:15:38+08:00：`runs/repro2_main` 训练至 E16.856、step 18,845。最近窗口中位吞吐 266.2 cells/s、loss 0.33438、lr 8.9e-05。已落盘完整断点 step 18,800。固定探针 step 18,750，raw/EMA endpoint MSE 0.10463/0.10432。

- 2026-11-24T00:31:14+08:00：`runs/repro2_main` 训练至 E17.250、step 19,285。最近窗口中位吞吐 264.5 cells/s、loss 0.33272、lr 8.83e-05。已落盘完整断点 step 19,250。固定探针 step 19,250，raw/EMA endpoint MSE 0.10503/0.10428。

- 2026-11-24T00:46:50+08:00：`runs/repro2_main` 训练至 E17.657、step 19,740。最近窗口中位吞吐 265.7 cells/s、loss 0.33439、lr 8.75e-05。已落盘完整断点 step 19,700。固定探针 step 19,500，raw/EMA endpoint MSE 0.10466/0.10426。

- 2026-11-24T01:02:30+08:00：`runs/repro2_main` 训练至 E18.055、step 20,185。最近窗口中位吞吐 265.0 cells/s、loss 0.34147、lr 8.67e-05。已落盘完整断点 step 20,150。固定探针 step 20,124，raw/EMA endpoint MSE 0.10491/0.10422。

- 2026-11-24T01:18:05+08:00：`runs/repro2_main` 训练至 E18.453、step 20,630。最近窗口中位吞吐 264.5 cells/s、loss 0.33019、lr 8.6e-05。已落盘完整断点 step 20,600。固定探针 step 20,500，raw/EMA endpoint MSE 0.10494/0.10421。

- 2026-11-24T01:33:41+08:00：`runs/repro2_main` 训练至 E18.855、step 21,080。最近窗口中位吞吐 264.6 cells/s、loss 0.33405、lr 8.51e-05。已落盘完整断点 step 21,050。固定探针 step 21,000，raw/EMA endpoint MSE 0.10466/0.10420。

- 2026-11-24T01:49:17+08:00：`runs/repro2_main` 训练至 E19.249、step 21,520。最近窗口中位吞吐 264.5 cells/s、loss 0.33560、lr 8.43e-05。已落盘完整断点 step 21,500。固定探针 step 21,500，raw/EMA endpoint MSE 0.10478/0.10418。

- 2026-11-24T02:04:53+08:00：`runs/repro2_main` 训练至 E19.656、step 21,975。最近窗口中位吞吐 266.0 cells/s、loss 0.33733、lr 8.35e-05。已落盘完整断点 step 21,950。固定探针 step 21,750，raw/EMA endpoint MSE 0.10476/0.10417。

- 2026-11-24T02:20:29+08:00：`runs/repro2_main` 训练至 E20.000、step 22,360。最近窗口中位吞吐 266.2 cells/s、loss 0.33846、lr 8.27e-05。已落盘完整断点 step 22,360。固定探针 step 22,360，raw/EMA endpoint MSE 0.10450/0.10417。

- 2026-11-24T02:29:00+08:00：`runs/repro2_main` 第 20 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3199，MSE=0.01057，PDS_L2=0.7744。这是验证结果，非测试结果。

- 2026-11-24T02:29:05+08:00：`runs/repro2_main` 完成第 20/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10417。

- 2026-11-24T02:30:03+08:00：第 20 轮缓存的 CPU 九指标检查完成：DEOver=0.1160、DEPrec=0.0883、DirAgr=0.6925；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_020/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T02:35:33+08:00：`runs/repro2_main` 训练至 E20.165、step 22,545。最近窗口中位吞吐 264.3 cells/s、loss 0.33929、lr 8.24e-05。已落盘完整断点 step 22,500。固定探针 step 22,500，raw/EMA endpoint MSE 0.10458/0.10417。

- 2026-11-24T02:51:11+08:00：`runs/repro2_main` 训练至 E20.572、step 23,000。最近窗口中位吞吐 265.4 cells/s、loss 0.33600、lr 8.15e-05。已落盘完整断点 step 22,950。固定探针 step 22,750，raw/EMA endpoint MSE 0.10474/0.10416。

- 2026-11-24T03:06:46+08:00：`runs/repro2_main` 训练至 E20.975、step 23,450。最近窗口中位吞吐 264.3 cells/s、loss 0.33909、lr 8.06e-05。已落盘完整断点 step 23,400。固定探针 step 23,250，raw/EMA endpoint MSE 0.10455/0.10415。

- 2026-11-24T03:22:23+08:00：`runs/repro2_main` 训练至 E21.369、step 23,890。最近窗口中位吞吐 266.1 cells/s、loss 0.33272、lr 7.97e-05。已落盘完整断点 step 23,850。固定探针 step 23,750，raw/EMA endpoint MSE 0.10496/0.10415。

- 2026-11-24T03:37:59+08:00：`runs/repro2_main` 训练至 E21.775、step 24,345。最近窗口中位吞吐 266.1 cells/s、loss 0.33795、lr 7.88e-05。已落盘完整断点 step 24,300。固定探针 step 24,250，raw/EMA endpoint MSE 0.10451/0.10415。

- 2026-11-24T03:53:35+08:00：`runs/repro2_main` 训练至 E22.169、step 24,785。最近窗口中位吞吐 264.4 cells/s、loss 0.33246、lr 7.78e-05。已落盘完整断点 step 24,750。固定探针 step 24,750，raw/EMA endpoint MSE 0.10450/0.10415。

- 2026-11-24T04:09:11+08:00：`runs/repro2_main` 训练至 E22.576、step 25,240。最近窗口中位吞吐 264.7 cells/s、loss 0.33767、lr 7.69e-05。已落盘完整断点 step 25,200。固定探针 step 25,000，raw/EMA endpoint MSE 0.10491/0.10415。

- 2026-11-24T04:24:46+08:00：`runs/repro2_main` 训练至 E22.979、step 25,690。最近窗口中位吞吐 264.6 cells/s、loss 0.34018、lr 7.59e-05。已落盘完整断点 step 25,650。固定探针 step 25,500，raw/EMA endpoint MSE 0.10436/0.10413。

- 2026-11-24T04:40:23+08:00：`runs/repro2_main` 训练至 E23.372、step 26,130。最近窗口中位吞吐 264.3 cells/s、loss 0.33555、lr 7.49e-05。已落盘完整断点 step 26,100。固定探针 step 26,000，raw/EMA endpoint MSE 0.10461/0.10412。

- 2026-11-24T04:55:58+08:00：`runs/repro2_main` 训练至 E23.775、step 26,580。最近窗口中位吞吐 264.2 cells/s、loss 0.33433、lr 7.39e-05。已落盘完整断点 step 26,550。固定探针 step 26,500，raw/EMA endpoint MSE 0.10465/0.10410。

- 2026-11-24T05:11:35+08:00：`runs/repro2_main` 训练至 E24.168、step 27,020。最近窗口中位吞吐 266.1 cells/s、loss 0.33787、lr 7.29e-05。已落盘完整断点 step 27,000。固定探针 step 27,000，raw/EMA endpoint MSE 0.10442/0.10408。

- 2026-11-24T05:27:11+08:00：`runs/repro2_main` 训练至 E24.575、step 27,475。最近窗口中位吞吐 264.6 cells/s、loss 0.33643、lr 7.19e-05。已落盘完整断点 step 27,450。固定探针 step 27,250，raw/EMA endpoint MSE 0.10464/0.10408。

- 2026-11-24T05:42:46+08:00：`runs/repro2_main` 训练至 E24.973、step 27,920。最近窗口中位吞吐 264.5 cells/s、loss 0.33693、lr 7.08e-05。已落盘完整断点 step 27,900。固定探针 step 27,750，raw/EMA endpoint MSE 0.10449/0.10407。

- 2026-11-24T05:54:32+08:00：`runs/repro2_main` 第 25 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3254，MSE=0.01136，PDS_L2=0.7839。这是验证结果，非测试结果。

- 2026-11-24T05:54:37+08:00：`runs/repro2_main` 完成第 25/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10406。

- 2026-11-24T05:55:06+08:00：第 25 轮缓存的 CPU 九指标检查完成：DEOver=0.1190、DEPrec=0.0892、DirAgr=0.7347；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_025/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T05:57:51+08:00：`runs/repro2_main` 训练至 E25.081、step 28,040。最近窗口中位吞吐 264.4 cells/s、loss 0.33777、lr 7.06e-05。已落盘完整断点 step 28,000。固定探针 step 28,000，raw/EMA endpoint MSE 0.10441/0.10406。

- 2026-11-24T06:13:27+08:00：`runs/repro2_main` 训练至 E25.487、step 28,495。最近窗口中位吞吐 264.4 cells/s、loss 0.33599、lr 6.95e-05。已落盘完整断点 step 28,450。固定探针 step 28,250，raw/EMA endpoint MSE 0.10451/0.10405。

- 2026-11-24T06:29:03+08:00：`runs/repro2_main` 训练至 E25.886、step 28,940。最近窗口中位吞吐 264.5 cells/s、loss 0.34070、lr 6.84e-05。已落盘完整断点 step 28,900。固定探针 step 28,750，raw/EMA endpoint MSE 0.10444/0.10404。

- 2026-11-24T06:44:40+08:00：`runs/repro2_main` 训练至 E26.284、step 29,385。最近窗口中位吞吐 264.3 cells/s、loss 0.33404、lr 6.74e-05。已落盘完整断点 step 29,350。固定探针 step 29,250，raw/EMA endpoint MSE 0.10450/0.10404。

- 2026-11-24T07:00:17+08:00：`runs/repro2_main` 训练至 E26.686、step 29,835。最近窗口中位吞吐 266.0 cells/s、loss 0.33248、lr 6.63e-05。已落盘完整断点 step 29,800。固定探针 step 29,750，raw/EMA endpoint MSE 0.10427/0.10404。

- 2026-11-24T07:15:53+08:00：`runs/repro2_main` 训练至 E27.084、step 30,280。最近窗口中位吞吐 264.8 cells/s、loss 0.33421、lr 6.52e-05。已落盘完整断点 step 30,250。固定探针 step 30,250，raw/EMA endpoint MSE 0.10429/0.10403。

- 2026-11-24T07:31:29+08:00：`runs/repro2_main` 训练至 E27.487、step 30,730。最近窗口中位吞吐 264.5 cells/s、loss 0.33081、lr 6.41e-05。已落盘完整断点 step 30,700。固定探针 step 30,500，raw/EMA endpoint MSE 0.10443/0.10402。

- 2026-11-24T07:47:05+08:00：`runs/repro2_main` 训练至 E27.889、step 31,180。最近窗口中位吞吐 264.4 cells/s、loss 0.33294、lr 6.3e-05。已落盘完整断点 step 31,150。固定探针 step 31,000，raw/EMA endpoint MSE 0.10438/0.10401。

- 2026-11-24T08:02:42+08:00：`runs/repro2_main` 训练至 E28.283、step 31,620。最近窗口中位吞吐 264.4 cells/s、loss 0.33250、lr 6.19e-05。已落盘完整断点 step 31,600。固定探针 step 31,500，raw/EMA endpoint MSE 0.10435/0.10400。

- 2026-11-24T08:18:20+08:00：`runs/repro2_main` 训练至 E28.690、step 32,075。最近窗口中位吞吐 266.0 cells/s、loss 0.32883、lr 6.08e-05。已落盘完整断点 step 32,050。固定探针 step 32,000，raw/EMA endpoint MSE 0.10417/0.10399。

- 2026-11-24T08:33:57+08:00：`runs/repro2_main` 训练至 E29.083、step 32,515。最近窗口中位吞吐 264.4 cells/s、loss 0.33410、lr 5.97e-05。已落盘完整断点 step 32,500。固定探针 step 32,500，raw/EMA endpoint MSE 0.10450/0.10398。

- 2026-11-24T08:39:44+08:00：会话恢复后核对了原训练控制器、监控进程和四个 DDP rank。训练在 E29.23 持续更新，最近中位吞吐约 266 cells/s，完整断点已落盘至 step 32,650。训练和监控进程均沿用原启动记录。 [训练进程](runs/repro2_main/training_execution.json)

- 2026-11-24T08:49:33+08:00：`runs/repro2_main` 训练至 E29.495、step 32,975。最近窗口中位吞吐 265.7 cells/s、loss 0.33368、lr 5.85e-05。已落盘完整断点 step 32,950。固定探针 step 32,750，raw/EMA endpoint MSE 0.10414/0.10397。

- 2026-11-24T09:05:09+08:00：`runs/repro2_main` 训练至 E29.897、step 33,425。最近窗口中位吞吐 265.7 cells/s、loss 0.33349、lr 5.74e-05。已落盘完整断点 step 33,400。固定探针 step 33,250，raw/EMA endpoint MSE 0.10429/0.10398。

- 2026-11-24T09:19:56+08:00：`runs/repro2_main` 第 30 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3293，MSE=0.01203，PDS_L2=0.8039。这是验证结果，非测试结果。

- 2026-11-24T09:20:01+08:00：`runs/repro2_main` 完成第 30/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10399。

- 2026-11-24T09:20:45+08:00：`runs/repro2_main` 训练至 E30.013、step 33,555。最近窗口中位吞吐 265.9 cells/s、loss 0.33862、lr 5.71e-05。已落盘完整断点 step 33,550。固定探针 step 33,540，raw/EMA endpoint MSE 0.10436/0.10399。

- 2026-11-24T09:21:09+08:00：第 30 轮缓存的 CPU 九指标检查完成：DEOver=0.1211、DEPrec=0.0902、DirAgr=0.7541；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_030/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T09:35:49+08:00：`runs/repro2_main` 训练至 E30.411、step 34,000。最近窗口中位吞吐 265.6 cells/s、loss 0.33171、lr 5.6e-05。已落盘完整断点 step 33,950。固定探针 step 33,750，raw/EMA endpoint MSE 0.10433/0.10399。

- 2026-11-24T09:51:25+08:00：`runs/repro2_main` 训练至 E30.814、step 34,450。最近窗口中位吞吐 265.7 cells/s、loss 0.33452、lr 5.48e-05。已落盘完整断点 step 34,400。固定探针 step 34,250，raw/EMA endpoint MSE 0.10436/0.10400。

- 2026-11-24T10:07:02+08:00：`runs/repro2_main` 训练至 E31.212、step 34,895。最近窗口中位吞吐 265.7 cells/s、loss 0.33500、lr 5.37e-05。已落盘完整断点 step 34,850。固定探针 step 34,750，raw/EMA endpoint MSE 0.10441/0.10401。

- 2026-11-24T10:22:38+08:00：`runs/repro2_main` 训练至 E31.614、step 35,345。最近窗口中位吞吐 265.7 cells/s、loss 0.33138、lr 5.25e-05。已落盘完整断点 step 35,300。固定探针 step 35,250，raw/EMA endpoint MSE 0.10426/0.10400。

- 2026-11-24T10:38:14+08:00：`runs/repro2_main` 训练至 E32.013、step 35,790。最近窗口中位吞吐 266.0 cells/s、loss 0.33935、lr 5.14e-05。已落盘完整断点 step 35,776。固定探针 step 35,776，raw/EMA endpoint MSE 0.10435/0.10399。

- 2026-11-24T10:53:54+08:00：`runs/repro2_main` 训练至 E32.424、step 36,250。最近窗口中位吞吐 265.9 cells/s、loss 0.34089、lr 5.02e-05。已落盘完整断点 step 36,200。固定探针 step 36,000，raw/EMA endpoint MSE 0.10410/0.10399。

- 2026-11-24T11:09:30+08:00：`runs/repro2_main` 训练至 E32.826、step 36,700。最近窗口中位吞吐 265.8 cells/s、loss 0.32727、lr 4.91e-05。已落盘完整断点 step 36,700。固定探针 step 36,500，raw/EMA endpoint MSE 0.10418/0.10399。

- 2026-11-24T11:25:06+08:00：`runs/repro2_main` 训练至 E33.225、step 37,145。最近窗口中位吞吐 265.8 cells/s、loss 0.33843、lr 4.79e-05。已落盘完整断点 step 37,100。固定探针 step 37,000，raw/EMA endpoint MSE 0.10433/0.10399。

- 2026-11-24T11:40:42+08:00：`runs/repro2_main` 训练至 E33.631、step 37,600。最近窗口中位吞吐 265.5 cells/s、loss 0.33820、lr 4.68e-05。已落盘完整断点 step 37,550。固定探针 step 37,500，raw/EMA endpoint MSE 0.10431/0.10400。

- 2026-11-24T11:56:19+08:00：`runs/repro2_main` 训练至 E34.030、step 38,045。最近窗口中位吞吐 265.5 cells/s、loss 0.33107、lr 4.56e-05。已落盘完整断点 step 38,012。固定探针 step 38,012，raw/EMA endpoint MSE 0.10447/0.10400。

- 2026-11-24T12:11:55+08:00：`runs/repro2_main` 训练至 E34.436、step 38,500。最近窗口中位吞吐 265.9 cells/s、loss 0.33187、lr 4.45e-05。已落盘完整断点 step 38,450。固定探针 step 38,250，raw/EMA endpoint MSE 0.10438/0.10400。

- 2026-11-24T12:27:31+08:00：`runs/repro2_main` 训练至 E34.839、step 38,950。最近窗口中位吞吐 265.7 cells/s、loss 0.33747、lr 4.33e-05。已落盘完整断点 step 38,950。固定探针 step 38,750，raw/EMA endpoint MSE 0.10426/0.10401。

- 2026-11-24T12:43:08+08:00：`runs/repro2_main` 训练至 E35.000、step 39,130。最近窗口中位吞吐 265.5 cells/s、loss 0.33785、lr 4.29e-05。已落盘完整断点 step 39,130。固定探针 step 39,130，raw/EMA endpoint MSE 0.10419/0.10400。

- 2026-11-24T12:44:31+08:00：`runs/repro2_main` 第 35 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3277，MSE=0.01237，PDS_L2=0.8019。这是验证结果，非测试结果。

- 2026-11-24T12:44:36+08:00：`runs/repro2_main` 完成第 35/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10400。

- 2026-11-24T12:45:22+08:00：第 35 轮缓存的 CPU 九指标检查完成：DEOver=0.1198、DEPrec=0.0892、DirAgr=0.7399；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_035/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T12:47:20+08:00：E30→E35 使用相同 1,916 个生成索引、噪声、CFG 和默认 EMA 做配对复核。综合分增加 +0.00344，95% 扰动配对区间 [-0.01219, +0.01971]。按预设点估计规则更新早停参考至 E35。 [周期验证配对](runs/repro2_main/observation/pair_e030_e035/comparison.json)

- 2026-11-24T12:58:13+08:00：`runs/repro2_main` 训练至 E35.349、step 39,520。最近窗口中位吞吐 265.6 cells/s、loss 0.33212、lr 4.19e-05。已落盘完整断点 step 39,500。固定探针 step 39,500，raw/EMA endpoint MSE 0.10428/0.10400。

- 2026-11-24T13:13:55+08:00：`runs/repro2_main` 训练至 E35.760、step 39,980。最近窗口中位吞吐 265.5 cells/s、loss 0.32903、lr 4.07e-05。已落盘完整断点 step 39,950。固定探针 step 39,750，raw/EMA endpoint MSE 0.10415/0.10399。

- 2026-11-24T13:29:31+08:00：`runs/repro2_main` 训练至 E36.158、step 40,425。最近窗口中位吞吐 265.2 cells/s、loss 0.33063、lr 3.96e-05。已落盘完整断点 step 40,400。固定探针 step 40,250，raw/EMA endpoint MSE 0.10414/0.10399。

- 2026-11-24T13:45:07+08:00：`runs/repro2_main` 训练至 E36.565、step 40,880。最近窗口中位吞吐 265.3 cells/s、loss 0.32659、lr 3.85e-05。已落盘完整断点 step 40,850。固定探针 step 40,750，raw/EMA endpoint MSE 0.10431/0.10398。

- 2026-11-24T14:00:44+08:00：`runs/repro2_main` 训练至 E36.968、step 41,330。最近窗口中位吞吐 265.3 cells/s、loss 0.33099、lr 3.74e-05。已落盘完整断点 step 41,300。固定探针 step 41,250，raw/EMA endpoint MSE 0.10428/0.10397。

- 2026-11-24T14:16:20+08:00：`runs/repro2_main` 训练至 E37.366、step 41,775。最近窗口中位吞吐 265.2 cells/s、loss 0.33308、lr 3.63e-05。已落盘完整断点 step 41,750。固定探针 step 41,750，raw/EMA endpoint MSE 0.10441/0.10397。

- 2026-11-24T14:31:57+08:00：`runs/repro2_main` 训练至 E37.773、step 42,230。最近窗口中位吞吐 265.5 cells/s、loss 0.33369、lr 3.52e-05。已落盘完整断点 step 42,200。固定探针 step 42,000，raw/EMA endpoint MSE 0.10422/0.10398。

- 2026-11-24T14:33:47+08:00：第二轮已越过首轮出现 DDP 数值偏差的 step 42,250。固定探针的 raw/默认 EMA endpoint MSE 为 0.104455/0.103985，EMA 相对 raw 低 0.45%。作为对照，首轮故障轨迹同一步的相对差为 +7.51%。四卡同步和检查点保持正常。 [首轮故障对照](logs/epoch040_recovery/README.md)

- 2026-11-24T14:47:33+08:00：`runs/repro2_main` 训练至 E38.171、step 42,675。最近窗口中位吞吐 264.9 cells/s、loss 0.33499、lr 3.41e-05。已落盘完整断点 step 42,650。固定探针 step 42,500，raw/EMA endpoint MSE 0.10442/0.10399。

- 2026-11-24T15:03:10+08:00：`runs/repro2_main` 训练至 E38.573、step 43,125。最近窗口中位吞吐 264.7 cells/s、loss 0.33198、lr 3.3e-05。已落盘完整断点 step 43,100。固定探针 step 43,000，raw/EMA endpoint MSE 0.10433/0.10402。

- 2026-11-24T15:18:46+08:00：`runs/repro2_main` 训练至 E38.976、step 43,575。最近窗口中位吞吐 264.8 cells/s、loss 0.33257、lr 3.19e-05。已落盘完整断点 step 43,550。固定探针 step 43,500，raw/EMA endpoint MSE 0.10429/0.10403。

- 2026-11-24T15:34:23+08:00：`runs/repro2_main` 训练至 E39.374、step 44,020。最近窗口中位吞吐 264.9 cells/s、loss 0.33130、lr 3.09e-05。已落盘完整断点 step 44,000。固定探针 step 44,000，raw/EMA endpoint MSE 0.10430/0.10404。

- 2026-11-24T15:49:59+08:00：`runs/repro2_main` 训练至 E39.781、step 44,475。最近窗口中位吞吐 264.6 cells/s、loss 0.33471、lr 2.98e-05。已落盘完整断点 step 44,450。固定探针 step 44,250，raw/EMA endpoint MSE 0.10430/0.10405。

- 2026-11-24T16:05:36+08:00：`runs/repro2_main` 训练至 E40.000、step 44,720。最近窗口中位吞吐 265.4 cells/s、loss 0.33299、lr 2.92e-05。已落盘完整断点 step 44,720。固定探针 step 44,720，raw/EMA endpoint MSE 0.10426/0.10406。

- 2026-11-24T16:09:13+08:00：`runs/repro2_main` 第 40 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3197，MSE=0.01292，PDS_L2=0.8069。这是验证结果，非测试结果。

- 2026-11-24T16:09:19+08:00：`runs/repro2_main` 完成第 40/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10406。

- 2026-11-24T16:09:41+08:00：第 40 轮缓存的 CPU 九指标检查完成：DEOver=0.1218、DEPrec=0.0891、DirAgr=0.7140；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_040/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T16:11:53+08:00：E35→E40 的同噪声周期验证综合分变化 -0.00944，95% 扰动配对区间 [-0.02889, +0.01076]。E35 继续作为最佳检查点和早停参考。E40 的默认 EMA 去噪误差为 0.104060。 [周期验证配对](runs/repro2_main/observation/pair_e035_e040/comparison.json)

- 2026-11-24T16:20:42+08:00：`runs/repro2_main` 训练至 E40.291、step 45,045。最近窗口中位吞吐 265.2 cells/s、loss 0.32670、lr 2.85e-05。已落盘完整断点 step 45,000。固定探针 step 45,000，raw/EMA endpoint MSE 0.10438/0.10408。

- 2026-11-24T16:36:22+08:00：`runs/repro2_main` 训练至 E40.698、step 45,500。最近窗口中位吞吐 264.4 cells/s、loss 0.33590、lr 2.74e-05。已落盘完整断点 step 45,450。固定探针 step 45,500，raw/EMA endpoint MSE 0.10448/0.10412。

- 2026-11-24T16:52:00+08:00：`runs/repro2_main` 训练至 E41.096、step 45,945。最近窗口中位吞吐 264.5 cells/s、loss 0.33563、lr 2.64e-05。已落盘完整断点 step 45,900。固定探针 step 45,838，raw/EMA endpoint MSE 0.10428/0.10412。

- 2026-11-24T17:07:36+08:00：`runs/repro2_main` 训练至 E41.498、step 46,395。最近窗口中位吞吐 265.2 cells/s、loss 0.32715、lr 2.54e-05。已落盘完整断点 step 46,350。固定探针 step 46,250，raw/EMA endpoint MSE 0.10428/0.10414。

- 2026-11-24T17:23:12+08:00：`runs/repro2_main` 训练至 E41.905、step 46,850。最近窗口中位吞吐 265.2 cells/s、loss 0.33479、lr 2.44e-05。已落盘完整断点 step 46,800。固定探针 step 46,750，raw/EMA endpoint MSE 0.10459/0.10416。

- 2026-11-24T17:38:49+08:00：`runs/repro2_main` 训练至 E42.299、step 47,290。最近窗口中位吞吐 264.3 cells/s、loss 0.32804、lr 2.35e-05。已落盘完整断点 step 47,250。固定探针 step 47,250，raw/EMA endpoint MSE 0.10457/0.10419。

- 2026-11-24T17:54:25+08:00：`runs/repro2_main` 训练至 E42.706、step 47,745。最近窗口中位吞吐 264.7 cells/s、loss 0.33301、lr 2.25e-05。已落盘完整断点 step 47,700。固定探针 step 47,500，raw/EMA endpoint MSE 0.10447/0.10421。

- 2026-11-24T18:10:02+08:00：`runs/repro2_main` 训练至 E43.104、step 48,190。最近窗口中位吞吐 264.9 cells/s、loss 0.33014、lr 2.15e-05。已落盘完整断点 step 48,150。固定探针 step 48,074，raw/EMA endpoint MSE 0.10444/0.10422。

- 2026-11-24T18:25:39+08:00：`runs/repro2_main` 训练至 E43.506、step 48,640。最近窗口中位吞吐 265.4 cells/s、loss 0.32288、lr 2.06e-05。已落盘完整断点 step 48,600。固定探针 step 48,500，raw/EMA endpoint MSE 0.10449/0.10423。

- 2026-11-24T18:41:15+08:00：`runs/repro2_main` 训练至 E43.913、step 49,095。最近窗口中位吞吐 264.8 cells/s、loss 0.33103、lr 1.97e-05。已落盘完整断点 step 49,050。固定探针 step 49,000，raw/EMA endpoint MSE 0.10436/0.10424。

- 2026-11-24T18:56:52+08:00：`runs/repro2_main` 训练至 E44.307、step 49,535。最近窗口中位吞吐 265.0 cells/s、loss 0.32957、lr 1.88e-05。已落盘完整断点 step 49,500。固定探针 step 49,500，raw/EMA endpoint MSE 0.10454/0.10426。

- 2026-11-24T19:12:29+08:00：`runs/repro2_main` 训练至 E44.714、step 49,990。最近窗口中位吞吐 265.1 cells/s、loss 0.33194、lr 1.79e-05。已落盘完整断点 step 49,950。固定探针 step 49,750，raw/EMA endpoint MSE 0.10437/0.10427。

- 2026-11-24T19:28:06+08:00：`runs/repro2_main` 训练至 E45.000、step 50,310。最近窗口中位吞吐 265.2 cells/s、loss 0.33385、lr 1.73e-05。已落盘完整断点 step 50,310。固定探针 step 50,310，raw/EMA endpoint MSE 0.10445/0.10430。

- 2026-11-24T19:34:30+08:00：`runs/repro2_main` 第 45 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3211，MSE=0.01347，PDS_L2=0.7939。这是验证结果，非测试结果。

- 2026-11-24T19:34:35+08:00：`runs/repro2_main` 完成第 45/60 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10430。

- 2026-11-24T19:34:35+08:00：`runs/repro2_main` 验证生成指标连续 10 轮无改善，按预设验证收敛规则停止。

- 2026-11-24T19:34:42+08:00：`runs/repro2_main` 训练结束（45.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-24T19:34:59+08:00：第 45 轮缓存的 CPU 九指标检查完成：DEOver=0.1209、DEPrec=0.0896、DirAgr=0.7318；复用既有预测，无新增生成。完整结果见 `runs/repro2_main/validation_epoch_045/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-24T19:37:05+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2431，PDS_L2=0.6958，MSE=0.01364（非测试）。

- 2026-11-24T19:37:09+08:00：训练完成校验通过，已启动冻结 E35 权重的四卡完整验证流程。

- 2026-11-24T19:43:34+08:00：验证集 CFG 校准 `screen_01`：w=0.75，区间 [0.1,0.9]，PDCorr=0.2565，PDS_L2=0.7242，MSE=0.01321（非测试）。

- 2026-11-24T19:49:29+08:00：验证集 CFG 校准 `screen_02`：w=1.0，区间 [0.1,0.9]，PDCorr=0.2704，PDS_L2=0.7444，MSE=0.01378（非测试）。

- 2026-11-24T19:55:23+08:00：验证集 CFG 校准 `screen_03`：w=1.25，区间 [0.1,0.9]，PDCorr=0.2800，PDS_L2=0.7536，MSE=0.01487（非测试）。

- 2026-11-24T20:01:19+08:00：验证集 CFG 校准 `screen_04`：w=1.5，区间 [0.1,0.9]，PDCorr=0.2865，PDS_L2=0.7719，MSE=0.01649（非测试）。

- 2026-11-24T20:07:15+08:00：验证集 CFG 校准 `screen_05`：w=1.75，区间 [0.1,0.9]，PDCorr=0.2896，PDS_L2=0.7881，MSE=0.01858（非测试）。

- 2026-11-24T20:13:10+08:00：验证集 CFG 校准 `screen_06`：w=2.0，区间 [0.1,0.9]，PDCorr=0.2922，PDS_L2=0.7972，MSE=0.02106（非测试）。

- 2026-11-24T20:19:55+08:00：验证集 CFG 校准 `screen_full_01`：w=0.75，区间 [0.0,1.0]，PDCorr=0.2651，PDS_L2=0.7300，MSE=0.01307（非测试）。

- 2026-11-24T20:26:38+08:00：验证集 CFG 校准 `screen_full_02`：w=1.0，区间 [0.0,1.0]，PDCorr=0.2886，PDS_L2=0.7478，MSE=0.01378（非测试）。

- 2026-11-24T20:33:24+08:00：验证集 CFG 校准 `screen_full_03`：w=1.25，区间 [0.0,1.0]，PDCorr=0.3032，PDS_L2=0.7669，MSE=0.01538（非测试）。

- 2026-11-24T20:40:10+08:00：验证集 CFG 校准 `screen_full_04`：w=1.5，区间 [0.0,1.0]，PDCorr=0.3105，PDS_L2=0.7850，MSE=0.01767（非测试）。

- 2026-11-24T20:46:57+08:00：验证集 CFG 校准 `screen_full_05`：w=1.75，区间 [0.0,1.0]，PDCorr=0.3156，PDS_L2=0.7956，MSE=0.02038（非测试）。

- 2026-11-24T20:53:45+08:00：验证集 CFG 校准 `screen_full_06`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3164，PDS_L2=0.7986，MSE=0.02354（非测试）。

- 2026-11-24T20:55:02+08:00：12 组双 CFG 筛选及普通条件对照已完成。最高综合分来自 w=2、[0,1]（1.10339），表达误差备选为 w=1.5、[0,1]（MSE 0.01767，综合分 1.08324）。按预先登记规则，两组各生成全部 4,825 个验证细胞，再比较裁负和 0.05。 [筛选结果](runs/repro2_main/calibration/all_results.json)

- 2026-11-24T21:24:55+08:00：验证集 CFG 校准 `refine_00`：w=2.0，区间 [0.0,1.0]，PDCorr=0.3638，PDS_L2=0.8183，MSE=0.01734（非测试）。

- 2026-11-24T21:55:53+08:00：验证集 CFG 校准 `refine_01`：w=1.5，区间 [0.0,1.0]，PDCorr=0.3771，PDS_L2=0.8119，MSE=0.01141（非测试）。

- 2026-11-24T21:55:53+08:00：完成验证集采样校准；设置已锁定于 `runs/repro2_main/calibration/selected.json`，尚未据此计算测试结果。

- 2026-11-24T21:56:59+08:00：`runs/repro2_main` 的完整验证未达到 DEPrec 的检查线。配置已按登记规则冻结，正式测试量化表 3 差距。

- 2026-11-24T23:30:30+08:00：正式测试已过半，四卡分片累计落盘 14,688/26,878 个细胞。相同原始预测会用于裁负和 0.05 两套 Cell Eval 指标。 [冻结协议](runs/repro2_main/test_full_postprocess/test_protocol.json)

- 2026-11-25T00:49:38+08:00：完成 `/home/dataset-assist-0/whs/ocoo-t/runs/repro2_main/test_full_clamp0` 的 test 评估（细胞 26,878，weights=ema_0.9996，CFG=1.5）；指标见 `/home/dataset-assist-0/whs/ocoo-t/runs/repro2_main/test_full_clamp0/metrics/summary.json`。PDCorr=0.4032，MSE=0.01099。

- 2026-11-25T00:51:15+08:00：`runs/repro2_main` 的完整测试与报告已产出：`runs/repro2_main/results/REPORT.md`。以下是实测结果，是否达到论文水平需结合全部指标审查；当前不会自动宣称复现成功。

- 2026-11-25T00:52:38+08:00：正式测试及两份报告已完成。核验全部测试索引、380 个扰动、共享真实对照、九指标宏平均、冻结配置及生成产物哈希。核验时四张 GPU 均已释放。 [测试核验](runs/repro2_main/test_completion_verification.json)
