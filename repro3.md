# OCOO-T Replogle–Nadig 第三轮复现归档

第三轮于北京时间 **2026-11-26 23:31** 完成正式评测。四卡全量训练从头跑完 55 轮、61,490 个优化步，周期验证选出第 45 轮的 EMA 0.9996 权重。冻结测试覆盖 26,878 个细胞、380 个扰动，真实和预测端共用 4,976 个 HepG2 对照。相较第二轮，九项指标中八项点估计改善。相较论文表 3，三项点估计更好，表达量误差、差异表达精度和余弦扰动区分仍有差距。 [论文 v2 表 3](https://arxiv.org/html/2606.12838v2#S4.T3) · [测试报告](runs/repro3_main/results/REPORT.md) · [第二轮归档](repro2.md)

## 参数与选择依据

| 项目 | 第三轮设置 |
| --- | --- |
| 数据与预处理 | Replogle–Nadig 官方划分及固定 2,000 HVG，保留公开归一化 log1p 表达与基因顺序。训练 572,545 个细胞、2,013 个扰动、4 个细胞系。完整验证 4,825 个细胞、60 个扰动。 |
| 主干 | 136,919,041 个参数，12 层 Transformer、宽 768、12 头、RMSNorm、QK-norm、AdaLN-zero、SwiGLU ratio 4，按 JiT 参数匹配约定取隐层宽 2,048。第 4 层起注入 32 个上下文 token。 |
| 表达与条件 | 表达 MLP 1→768→768，固定一维位置编码，幅度 1。时间正弦编码维数 256、尺度 1。冻结 5,120 维 ESM2，经共享投影编码基因及扰动。细胞系和 56 个 batch 标签嵌入的初始化标准差分别为 1 和 0.02。 |
| 噪声与目标 | 小实验选定训练和生成共用 Gaussian σ=0.75。保留零初始化的可学习 null embedding，扰动和细胞标签独立以 0.1 概率置空。时间采样为 `sigmoid(N(−0.8, 0.8²))`，采用速度预测和 L2 损失。 |
| 优化器与调度 | 训练 seed 42，AdamW β=(0.9, 0.95)，零权重衰减，峰值 lr 1e−4。前 5 轮线性 warmup，余弦衰减至 0，预算 55 轮。最低 20 轮，若连续 10 轮未取得 0.002 的周期选型分数改善则早停。实际跑满预算。 |
| 四卡执行 | 4×A100 PCIe 40 GB，BF16、默认核心编译、原生 DDP。每卡 microbatch 16，累积 8 次，全局 batch 512，每轮 1,118 步。关闭 Dynamo DDPOptimizer 分图，每 50 步保存完整训练状态。 |
| 周期选型 | 保存 EMA 0.9999 和 0.9996，每 5 轮用 EMA 0.9996 评估同一组 1,916 个验证细胞。周期采样用双 CFG `w=1.5`、区间 `[0.1,0.9]`。用 `S = PDCorr + mean(PDS_L1, PDS_L2, PDS_cos) − 10 × MSE` 选检查点并判断收敛。 |
| 冻结测试 | 第 45 轮、step 50,310、EMA 0.9996。式 11 双 CFG `w=1.375`、区间 `[0,1]`、50 等步 Heun、末步 Euler，生成 batch 32、seed 4242。将小于 0.05 的预测表达置零，另用同一批原始生成计算仅裁负对照。Cell Eval 0.8.2 / pdex 0.3.0。 |

论文约束了主干、训练目标、优化器、EMA 及双 CFG 公式。σ、null 初始化、验证打分、候选尺度及近零阈值为本轮登记并由验证选择的复现参数。近零阈值用于输出处理，训练始终优化速度 L2。 [主训练 YAML](configs/repro3_main.yaml) · [小实验登记](configs/repro3_pilots/plan.json) · [评测登记](configs/repro3_postprocess.json) · [数据清单](data/raw/manifest.json)

## 实际运行与验证过程

1. **三组四卡小实验各跑 1,536 步。** 三组共享 16,384 个训练细胞与 960 个验证生成索引。比较第二轮对照、σ=0.75、训练均值 null 初始化。在仅裁负的 EMA 结果上，σ=0.75 的配对 ΔS 为 +0.06185，95% 区间为 `[+0.03795, +0.08648]`，且满足 PDCorr 和 DEPrec 守门条件。训练均值 null 的 ΔS 为 −0.36515，主跑沿用零初始化。短实验中三组 PDS 均接近随机，收益主要来自早期 MSE。 [小实验全指标与配对区间](runs/repro3_pilots/summary.json)
2. **一次全量重训。** 2026-11-25 04:48:38 至 2026-11-26 18:14:23，耗时 37 小时 25 分 45 秒，含周期生成。日志中位吞吐 265.46 cells/s，rank 0 分配显存峰值 18.44 GB。第 45 轮的 S=1.02233 为周期最佳。末步的优化器状态、双 EMA 和四个 rank 的 RNG 均已核对。 [训练完成核验](runs/repro3_main/training_completion_verification.json) · [周期九指标](runs/repro3_main/observation/generation_history.json) · [完整事件日志](runs/repro3_main/events.jsonl.gz)
3. **只用验证集校准并冻结。** 用 960 个固定索引筛选两种时间窗与六个正指导尺度，另跑普通条件参考。将 S 最高的 `w=1.375/[0,1]` 和邻近候选中 MSE 最低的 `w=1.125/[0,1]` 各补齐至 4,825 个验证细胞。两种尺度的 0.05 候选都通过预先登记的质量门槛，依原综合分选中 `w=1.375`，完整验证的 PDCorr 0.39385、MSE 0.01068、DEPrec 0.18961。 [校准全记录](runs/repro3_main/calibration/all_results.json) · [完整验证审查](runs/repro3_main/calibration/postprocessing/review.json)
4. **冻结后完成一次正式生成。** 2026-11-26 20:36 进入测试，23:31:20 产出报告。复用同一批原始生成计算两种阈值，核对 19 项产物哈希、完整测试索引及两套按扰动宏平均。训练代码启动副本与后续注释变更有单独校验，校准与评测的源码哈希可在 `57631a8` 核对。 [测试协议](runs/repro3_main/test_full_postprocess/test_protocol.json) · [测试完成核验](runs/repro3_main/test_completion_verification.json) · [评测代码核验](runs/repro3_main/evaluation_code_verification.json)

## 九项正式指标与论文差距

差值为第三轮减去论文值。区间来自第三轮测试扰动 bootstrap，反映该次评测的扰动间变异。

| 指标 | 论文表 3 | 第二轮 | 第三轮 | 第三轮 − 论文 | 第三轮 95% 区间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PDCorr ↑ | 0.43700 | 0.40058 | 0.41224 | −0.02476 | [0.37864, 0.44889] |
| DEOver ↑ | 0.23200 | 0.23892 | 0.24153 | +0.00953 | [0.21398, 0.26948] |
| DEPrec ↑ | 0.24000 | 0.20700 | 0.21641 | −0.02359 | [0.19334, 0.23941] |
| DirAgr ↑ | 0.76600 | 0.81160 | 0.82113 | +0.05513 | [0.79615, 0.84563] |
| PDS_L1 ↑ | 0.82000 | 0.83136 | 0.83589 | +0.01589 | [0.81519, 0.85451] |
| PDS_L2 ↑ | 0.85300 | 0.84435 | 0.84951 | −0.00349 | [0.82951, 0.86849] |
| PDS_cos ↑ | 0.83300 | 0.80520 | 0.80286 | −0.03014 | [0.77568, 0.82750] |
| MSE ↓ | 0.00790 | 0.01107 | 0.01010 | +0.00220 | [0.00942, 0.01085] |
| MAE ↓ | 0.06100 | 0.06949 | 0.06727 | +0.00627 | [0.06541, 0.06912] |

表达量重建的差距最明确，MSE 和 MAE 仍比论文高 27.8% 和 10.3%。PDS_L2 已接近论文，PDS_cos 仍低 0.03014，表明按余弦相似度区分扰动响应仍偏弱。DEPrec 低 2.36 个百分点，DirAgr 在成功找回的差异基因上高 5.51 个百分点。第三轮相较第二轮的配对区间支持 PDCorr、MSE、MAE 和 DEPrec 改善，PDS_cos 的下降区间跨零。 [表 3 对照 CSV](runs/repro3_main/results/benchmark_comparison.csv) · [第二、三轮配对 CSV](runs/repro3_main/results/paired_repro2_comparison.csv) · [裁负测试对照](runs/repro3_main/results_clamp0/REPORT.md)

## Git 交付与复现入口

`codex/repro3` 收录训练、数据准备、评测及测试代码，三轮归档文档，第三轮冻结配置、每扰动指标、完整结果与日志、数据和源码哈希。模型检查点、原始表达数据、ESM2 张量及大预测数组留在运行工作区。克隆后可直接查看已归档结果，从头训练需按下列命令准备数据。原始文件清单记录了主跑的校验值。 [环境定义](environment.yml) · [环境快照](environment.lock.txt) · [第三轮正式结果](runs/repro3_main/results/REPORT.md) · [数据来源清单](data/raw/manifest.json)

从新克隆的仓库根目录准备 Python 环境和数据。分支提供主跑所用的 HGNC 表压缩快照，供重跑使用同一份基因别名表。解压后的 SHA256 为 `6df77ac3cbaeb86a6dce5e17d851ebb27b6b3cdcd2471a211a3d8464f00c54c4`。 [压缩快照](data/raw/hgnc_complete_set.tsv.gz) · [HGNC 数据下载说明](https://hgnc.genenames.org/download/) · [HGNC CC0 声明](https://hgnc.genenames.org/about/license/)

```bash
conda env create -f environment.yml
conda activate ocoo-t

if [ ! -e data/raw/hgnc_complete_set.tsv ]; then
  gzip -dc data/raw/hgnc_complete_set.tsv.gz > data/raw/hgnc_complete_set.tsv
fi
printf '%s\n' '6df77ac3cbaeb86a6dce5e17d851ebb27b6b3cdcd2471a211a3d8464f00c54c4  data/raw/hgnc_complete_set.tsv' | sha256sum -c - &&
  python scripts/download_data.py &&
  python scripts/prepare_data.py
```

全量重跑使用新的 `run_dir`。下例将已有的主跑归档保留在 `runs/repro3_main`，并将同一组科学参数复制到新配置。归档数据目录里的 `finished.json` 是旧跑完成标记，新的输出目录应当空置。

```bash
repro3_rerun_config=$(mktemp /tmp/ocoot-repro3.XXXXXXXX.yaml)
sed 's|^run_dir: runs/repro3_main$|run_dir: runs/repro3_main_rerun|' \
  configs/repro3_main.yaml > "$repro3_rerun_config"
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=6 \
  python scripts/run_experiment.py --config "$repro3_rerun_config" --gpu-workers 4 \
  --calibration-scales 1.125,1.25,1.375,1.5,1.625,1.75 --compare-windows \
  --calibration-mse-weight 10 --postprocess-plan configs/repro3_postprocess.json
```

三组小实验的原始命令见下方归档。在独立克隆中重跑小实验时，可先把 `runs/repro3_pilots` 移至邻近备份目录，给登记计划留出同名的空输出位置。运行期间进度脚本会重新创建本地 `README.md`。

## 上传到 GitHub

在 GitHub 创建空仓库，保持 README、许可证和 `.gitignore` 的初始化选项空置。使用已配置的 GitHub HTTPS 凭据，或采用 GitHub 提供的 SSH URL。在项目根目录将 `OWNER/REPO` 替换为自己的账号和仓库名。 [GitHub 导入本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github?platform=linux) · [GitHub 认证方式](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)

```bash
git remote add origin https://github.com/OWNER/REPO.git
git remote -v
git push -u origin codex/repro3
git ls-remote --heads origin refs/heads/codex/repro3
```

推送后在仓库页面选 `codex/repro3`，打开根目录的 `repro3.md`。若 `origin` 已配置为目标仓库，可从核对远端 URL 的命令开始。

---

## 原 README 全文

归档前 `README.md` 的 SHA256 为 `a12b8dd46ae7d8f7d8c1a97007a3053ff811114ffa0d2b3a6bb250382e384d0d`。从下一个一级标题起为原文件全文。

<!-- ARCHIVED_README_START -->

# OCOO-T Replogle–Nadig 第三轮复现

本轮固定 Replogle–Nadig 官方划分和 2,000 HVG。第二轮冻结测试的 PDCorr 为 0.40058，MSE 为 0.01107，DEPrec 为 0.20700。论文表 3 分别为 0.437、0.00790、0.240。完整的第二轮结果、参数和当时的 README 已归档于 [repro2.md](repro2.md)，基线提交为 `codex/repro2` 上的 `bfd2ce0`。第三轮在 `codex/repro3` 分支开展。 [论文 v2](https://arxiv.org/html/2606.12838v2)

<!-- LIVE_TRAINING_START -->

第三轮已完成。正式评测于北京时间 2026-11-26 23:31 结束。PDCorr **0.41224**，MSE **0.01010**，DEPrec **0.21641**。

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

## 全量训练结果

主训练于北京时间 2026-11-25 04:48 启动，2026-11-26 18:14 完成。55 轮共 61,490 个优化步，含周期采样约 37 小时 26 分钟。训练日志的中位吞吐为 265.46 cells/s，rank 0 训练显存分配峰值为 18.44 GB。整个主训练只启动一次。末步检查点的 189 个优化器状态条目、双 EMA 及四个 rank 随机状态均已核对。

按周期验证分数选择 **第 45 轮、step 50,310、EMA 0.9996**，S=1.02233。标准化噪声种子、验证索引和 CFG 周期设置固定。以下是每扰动最多 32 个、实际 1,916 个生成细胞的验证诊断，DE 指标由 CPU 复用对应缓存计算。

| 轮次 | S | PDCorr | PDS_L1 | PDS_L2 | PDS_cos | MSE | MAE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 0.29358 | 0.16714 | 0.52500 | 0.53194 | 0.51056 | 0.03961 | 0.12998 | 0.12514 | 0.11473 | 0.65172 |
| 10 | 0.83531 | 0.28575 | 0.65111 | 0.66194 | 0.64250 | 0.01023 | 0.06819 | 0.09666 | 0.07529 | 0.68017 |
| 15 | 0.93281 | 0.30054 | 0.72500 | 0.73861 | 0.73389 | 0.01002 | 0.06998 | 0.11024 | 0.08369 | 0.67944 |
| 20 | 0.99985 | 0.33140 | 0.76361 | 0.78111 | 0.77167 | 0.01037 | 0.07099 | 0.12270 | 0.08945 | 0.71805 |
| 25 | 1.00708 | 0.33296 | 0.76944 | 0.78833 | 0.79056 | 0.01087 | 0.07181 | 0.12057 | 0.08973 | 0.73153 |
| 30 | 1.01482 | 0.33507 | 0.77917 | 0.79750 | 0.81778 | 0.01184 | 0.07304 | 0.12272 | 0.09014 | 0.73258 |
| 35 | 1.01800 | 0.33018 | 0.77861 | 0.79722 | 0.84194 | 0.01181 | 0.07338 | 0.12216 | 0.09200 | 0.73210 |
| 40 | 1.00460 | 0.32223 | 0.78361 | 0.79806 | 0.83667 | 0.01237 | 0.07445 | 0.11949 | 0.09147 | 0.75715 |
| 45 | 1.02233 | 0.33304 | 0.79278 | 0.80556 | 0.84361 | 0.01247 | 0.07476 | 0.13542 | 0.09112 | 0.74596 |
| 50 | 1.01394 | 0.33206 | 0.79139 | 0.80278 | 0.83972 | 0.01294 | 0.07562 | 0.13630 | 0.09219 | 0.74995 |
| 55 | 1.01534 | 0.33123 | 0.79083 | 0.80472 | 0.84750 | 0.01302 | 0.07596 | 0.13641 | 0.09108 | 0.72751 |

[训练完成性核验](runs/repro3_main/training_completion_verification.json) · [周期九指标与生成设置](runs/repro3_main/observation/generation_history.json) · [训练曲线](runs/repro3_main/progress.png)

## 完整验证与测试冻结

验证校准从北京时间 18:14 开始。12 组双 CFG 与 1 组普通条件参考各用 960 个固定采样索引。在裁负结果中，`w=1.375/[0,1]` 的 S=0.93779 排第一，`w=1.125/[0,1]` 是距最高 S 0.04 以内 MSE 最低的另一组。两组各补齐 4,825 个验证细胞及 60 个扰动。 [全部校准结果](runs/repro3_main/calibration/all_results.json)

| 指标 | w=1.375 裁负 | w=1.375 阈值 0.05 | w=1.125 裁负 | w=1.125 阈值 0.05 |
| --- | ---: | ---: | ---: | ---: |
| PDCorr | 0.39504 | 0.39385 | 0.39536 | 0.39415 |
| PDS_L1 | 0.80972 | 0.81167 | 0.79972 | 0.80000 |
| PDS_L2 | 0.82056 | 0.82000 | 0.81222 | 0.81250 |
| PDS_cos | 0.83056 | 0.83000 | 0.80750 | 0.80472 |
| MSE | 0.01065 | 0.01068 | 0.00872 | 0.00874 |
| MAE | 0.06712 | 0.06727 | 0.06293 | 0.06305 |
| DEOver | 0.18504 | 0.21586 | 0.17202 | 0.20962 |
| DEPrec | 0.10603 | 0.18961 | 0.10110 | 0.20565 |
| DirAgr | 0.81197 | 0.81275 | 0.81215 | 0.81236 |
| 原综合分 | 1.21531 | 1.21441 | 1.20184 | 1.19989 |
| 所有项目检查通过 | 否 | 是 | 否 | 是 |

两个阈值 0.05 候选均通过预登记的完整验证检查，因原综合分 1.21441 较高，选定 **w=1.375、[0,1]、阈值 0.05**。w=1.125 的完整验证 MSE 和 DEPrec 更优。两种指标取舍已保存，正式测试沿用冻结选择。 [完整验证审查](runs/repro3_main/calibration/postprocessing/review.json)

在相同完整验证索引上，相对第二轮，第三轮 PDCorr 增加 0.02000，MSE 下降 0.00078，DEPrec 增加 0.01628。三项的 95% 扰动配对区间分别为 [0.00106,0.04083]、[−0.00149,−0.00013] 和 [0.00306,0.03174]。差异涵盖两轮的训练、检查点和采样选择。 [完整验证配对](runs/repro3_main/full_validation_comparison.json)

冻结协议保存权重、验证和选型摘要。一次四卡采样覆盖全部 26,878 个测试细胞，裁负和阈值 0.05 共享原始生成。
 [测试冻结协议](runs/repro3_main/test_full_postprocess/test_protocol.json)

## 正式测试结果

北京时间 **2026-11-26 23:31** 完成冻结评测及报告。测试包含 26,878 个扰动细胞和 380 个扰动。真实与预测两端共用 4,976 个 HepG2 对照。裁负与 0.05 使用同一批原始预测，已核验冻结选型、19 个测试产物哈希、两套按扰动宏平均和四卡进程退出状态。 [测试报告](runs/repro3_main/results/REPORT.md) · [完成性核验](runs/repro3_main/test_completion_verification.json)

| 指标 | 论文表 3 | 第二轮 | 第三轮阈值 0.05 | Δ论文 | Δ第二轮 | Δ第二轮 95% 配对区间 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PDCorr | 0.43700 | 0.40058 | 0.41224 | -0.02476 | +0.01166 | [+0.00527, +0.01811] |
| PDS_L1 | 0.82000 | 0.83136 | 0.83589 | +0.01589 | +0.00453 | [-0.00340, +0.01251] |
| PDS_L2 | 0.85300 | 0.84435 | 0.84951 | -0.00349 | +0.00516 | [-0.00339, +0.01360] |
| PDS_cos | 0.83300 | 0.80520 | 0.80286 | -0.03014 | -0.00234 | [-0.01567, +0.01085] |
| MSE | 0.00790 | 0.01107 | 0.01010 | +0.00220 | -0.00097 | [-0.00128, -0.00067] |
| MAE | 0.06100 | 0.06949 | 0.06727 | +0.00627 | -0.00222 | [-0.00291, -0.00153] |
| DEOver | 0.23200 | 0.23892 | 0.24153 | +0.00953 | +0.00261 | [-0.00984, +0.01637] |
| DEPrec | 0.24000 | 0.20700 | 0.21641 | -0.02359 | +0.00941 | [+0.00057, +0.01851] |
| DirAgr | 0.76600 | 0.81160 | 0.82113 | +0.05513 | +0.00953 | [-0.00651, +0.02614] |

与第二轮相比，8 项点估计改善。PDCorr、MSE、MAE 和 DEPrec 的同扰动配对区间支持改善。PDS_cos 降低 0.00234，区间跨零。区间来自同一测试库 380 个扰动的 2,000 次 bootstrap，配对结果涵盖两轮的训练和采样配置。 [配对结果 CSV](runs/repro3_main/results/paired_repro2_comparison.csv)

PDCorr 距表 3 为 0.02476，DEPrec 差 0.02359，PDS_cos 差 0.03014。MSE 从第二轮下降约 8.76%，仍比论文高 0.00220。PDS_L1、DEOver 和 DirAgr 的点估计高于表 3。 [论文 v2](https://arxiv.org/html/2606.12838v2)

同一样本从裁负改为验证锁定的 0.05 后，DEPrec 由 0.11274 到 0.21641，PDCorr 由 0.41318 到 0.41224，MSE 由 0.01006 到 0.01010。 [裁负测试对照](runs/repro3_main/results_clamp0/REPORT.md)

完整事件日志在 Git 中采用 gzip，校准与冻结测试所用源码见提交 57631a8。训练和评估期间补加的源码注释单独保留，并核验语法树与原参数一致。 [完整事件日志](runs/repro3_main/events.jsonl.gz) · [核验脚本](logs/repro3/verify_completed_run.py) · [评测源码核验](runs/repro3_main/evaluation_code_verification.json)

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

- 2026-11-25T04:48:38+08:00：启动/续训 `runs/repro3_main`，从 step 0 接续，4 卡、每卡 microbatch=16、各卡累积 [8, 8, 8, 8] 次、全局 batch=512，完整配置 `runs/repro3_main/config.yaml`。

- 2026-11-25T05:03:41+08:00：全量训练至 E0.394、step 440，最近吞吐中位数 267.4 cells/s，loss 0.61147，lr 7.87e-06。完整断点 step 400。固定去噪探针 step 250，EMA endpoint MSE 0.55340。

- 2026-11-25T05:18:51+08:00：全量训练至 E0.783、step 875，最近吞吐中位数 265.3 cells/s，loss 0.35728，lr 1.57e-05。完整断点 step 850。固定去噪探针 step 750，EMA endpoint MSE 0.52200。

- 2026-11-25T05:34:03+08:00：全量训练至 E1.167、step 1,305，最近吞吐中位数 266.3 cells/s，loss 0.34439，lr 2.33e-05。完整断点 step 1,300。固定去噪探针 step 1,250，EMA endpoint MSE 0.48353。

- 2026-11-25T05:49:21+08:00：全量训练至 E1.565、step 1,750，最近吞吐中位数 266.5 cells/s，loss 0.33117，lr 3.13e-05。完整断点 step 1,750。固定去噪探针 step 1,750，EMA endpoint MSE 0.45164。

- 2026-11-25T06:04:28+08:00：全量训练至 E1.963、step 2,195，最近吞吐中位数 266.5 cells/s，loss 0.32749，lr 3.93e-05。完整断点 step 2,150。固定去噪探针 step 2,000，EMA endpoint MSE 0.43023。

- 2026-11-25T06:19:41+08:00：全量训练至 E2.348、step 2,625，最近吞吐中位数 266.2 cells/s，loss 0.32009，lr 4.7e-05。完整断点 step 2,600。固定去噪探针 step 2,500，EMA endpoint MSE 0.36739。

- 2026-11-25T06:34:50+08:00：全量训练至 E2.742、step 3,065，最近吞吐中位数 266.2 cells/s，loss 0.31509，lr 5.48e-05。完整断点 step 3,050。固定去噪探针 step 3,000，EMA endpoint MSE 0.30378。

- 2026-11-25T06:50:01+08:00：全量训练至 E3.131、step 3,500，最近吞吐中位数 266.1 cells/s，loss 0.31349，lr 6.26e-05。完整断点 step 3,450。固定去噪探针 step 3,354，EMA endpoint MSE 0.26042。

- 2026-11-25T07:05:11+08:00：全量训练至 E3.524、step 3,940，最近吞吐中位数 266.2 cells/s，loss 0.31885，lr 7.05e-05。完整断点 step 3,900。固定去噪探针 step 3,750，EMA endpoint MSE 0.22292。

- 2026-11-25T07:20:22+08:00：全量训练至 E3.913、step 4,375，最近吞吐中位数 266.1 cells/s，loss 0.31405，lr 7.83e-05。完整断点 step 4,350。固定去噪探针 step 4,250，EMA endpoint MSE 0.19206。

- 2026-11-25T07:35:37+08:00：全量训练至 E4.302、step 4,810，最近吞吐中位数 266.1 cells/s，loss 0.31392，lr 8.6e-05。完整断点 step 4,800。固定去噪探针 step 4,750，EMA endpoint MSE 0.17201。

- 2026-11-25T07:50:46+08:00：全量训练至 E4.696、step 5,250，最近吞吐中位数 266.4 cells/s，loss 0.30550，lr 9.39e-05。完整断点 step 5,250。固定去噪探针 step 5,250，EMA endpoint MSE 0.15735。

- 2026-11-25T08:06:02+08:00：全量训练至 E5.000、step 5,590，最近吞吐中位数 266.0 cells/s，loss 0.30973，lr 0.0001。完整断点 step 5,590。固定去噪探针 step 5,590，EMA endpoint MSE 0.14915。最近生成 `parts_w4_from0` 分片 128/480 个细胞。

- 2026-11-25T08:13:46+08:00：`runs/repro3_main` 第 5 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.1671，MSE=0.03961，PDS_L2=0.5319。这是验证结果，非测试结果。

- 2026-11-25T08:13:52+08:00：`runs/repro3_main` 完成第 5/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.14915。

- 2026-11-25T08:14:24+08:00：第 5 轮缓存的 CPU 九指标检查完成：DEOver=0.1251、DEPrec=0.1147、DirAgr=0.6517；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_005/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-25T08:21:14+08:00：全量训练至 E5.188、step 5,800，最近吞吐中位数 265.3 cells/s，loss 0.30928，lr 0.0001。完整断点 step 5,750。固定去噪探针 step 5,750，EMA endpoint MSE 0.14562。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T08:21:46+08:00：E5 同阶段复核使用相同的 1,916 个生成索引和 CFG。第三轮 PDCorr 0.16714、MSE 0.03961，第二轮分别为 0.19726、0.02962。较小噪声在短实验中的 MSE 优势在全量 warmup 末未延续。按锁定配置继续，等待更成熟的 EMA 周期验证。

- 2026-11-25T08:36:26+08:00：全量训练至 E5.586、step 6,245，最近吞吐中位数 266.2 cells/s，loss 0.30750，lr 0.0001。完整断点 step 6,200。固定去噪探针 step 6,000，EMA endpoint MSE 0.14066。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T08:51:35+08:00：全量训练至 E5.975、step 6,680，最近吞吐中位数 266.0 cells/s，loss 0.30431，lr 9.99e-05。完整断点 step 6,650。固定去噪探针 step 6,500，EMA endpoint MSE 0.13159。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T09:06:40+08:00：全量训练至 E6.360、step 7,110，最近吞吐中位数 265.8 cells/s，loss 0.30198，lr 9.98e-05。完整断点 step 7,100。固定去噪探针 step 7,000，EMA endpoint MSE 0.12404。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T09:21:44+08:00：全量训练至 E6.749、step 7,545，最近吞吐中位数 266.1 cells/s，loss 0.30421，lr 9.97e-05。完整断点 step 7,500。固定去噪探针 step 7,500，EMA endpoint MSE 0.11789。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T09:36:56+08:00：全量训练至 E7.138、step 7,980，最近吞吐中位数 265.1 cells/s，loss 0.30559，lr 9.96e-05。完整断点 step 7,950。固定去噪探针 step 7,826，EMA endpoint MSE 0.11463。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T09:52:02+08:00：全量训练至 E7.527、step 8,415，最近吞吐中位数 264.8 cells/s，loss 0.30991，lr 9.94e-05。完整断点 step 8,400。固定去噪探针 step 8,250，EMA endpoint MSE 0.11123。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T10:07:07+08:00：全量训练至 E7.916、step 8,850，最近吞吐中位数 264.4 cells/s，loss 0.30513，lr 9.92e-05。完整断点 step 8,850。固定去噪探针 step 8,750，EMA endpoint MSE 0.10807。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T10:22:18+08:00：全量训练至 E8.301、step 9,280，最近吞吐中位数 264.7 cells/s，loss 0.30536，lr 9.89e-05。完整断点 step 9,250。固定去噪探针 step 9,250，EMA endpoint MSE 0.10580。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T10:37:29+08:00：全量训练至 E8.694、step 9,720，最近吞吐中位数 264.6 cells/s，loss 0.30590，lr 9.87e-05。完整断点 step 9,700。固定去噪探针 step 9,500，EMA endpoint MSE 0.10484。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T10:52:35+08:00：全量训练至 E9.079、step 10,150，最近吞吐中位数 265.4 cells/s，loss 0.30448，lr 9.84e-05。完整断点 step 10,100。固定去噪探针 step 10,062，EMA endpoint MSE 0.10315。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T11:07:46+08:00：全量训练至 E9.468、step 10,585，最近吞吐中位数 264.9 cells/s，loss 0.30778，lr 9.8e-05。完整断点 step 10,550。固定去噪探针 step 10,500，EMA endpoint MSE 0.10217。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T11:22:53+08:00：全量训练至 E9.852、step 11,015，最近吞吐中位数 265.2 cells/s，loss 0.30513，lr 9.77e-05。完整断点 step 11,000。固定去噪探针 step 11,000，EMA endpoint MSE 0.10131。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。

- 2026-11-25T11:38:13+08:00：全量训练至 E10.000、step 11,180，最近吞吐中位数 265.0 cells/s，loss 0.30447，lr 9.76e-05。完整断点 step 11,180。固定去噪探针 step 11,180，EMA endpoint MSE 0.10106。最近周期验证 E5，PDCorr=0.16714，MSE=0.03961，S=0.29358。最近生成 `validation_epoch_010` 已完成 1,664/1,916 个细胞。

- 2026-11-25T11:39:15+08:00：`runs/repro3_main` 第 10 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.2858，MSE=0.01023，PDS_L2=0.6619。这是验证结果，非测试结果。

- 2026-11-25T11:39:21+08:00：`runs/repro3_main` 完成第 10/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.10106。

- 2026-11-25T11:40:21+08:00：第 10 轮缓存的 CPU 九指标检查完成：DEOver=0.0967、DEPrec=0.0753、DirAgr=0.6802；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_010/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-25T11:53:29+08:00：全量训练至 E10.358、step 11,580，最近吞吐中位数 264.0 cells/s，loss 0.30585，lr 9.72e-05。完整断点 step 11,550。固定去噪探针 step 11,500，EMA endpoint MSE 0.10064。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T11:53:57+08:00：E10 同阶段比较已核对相同生成索引和 CFG。与第二轮相比，PDCorr 由 0.25966 到 0.28575，MSE 由 0.01058 到 0.01023，PDS_cos 由 0.66917 到 0.64250。周期验证反映不同训练配置的当前进度。

- 2026-11-25T12:45:42+08:00：全量训练至 E11.695、step 13,075，最近吞吐中位数 264.8 cells/s，loss 0.30674，lr 9.56e-05。完整断点 step 13,050。固定去噪探针 step 13,000，EMA endpoint MSE 0.09940。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T12:46:25+08:00：聊天端计时器的一次回调延后，训练日志持续更新。回看三次计划检查时的最近记录，12:08 step 12,010 / loss 0.30582，12:23 step 12,435 / loss 0.30517，12:38 step 12,865 / loss 0.30732。后续用独立本地进程按 15 分钟发布 README 快照，并继续检查训练与评测。

- 2026-11-25T13:00:44+08:00：全量训练至 E12.075、step 13,500，最近吞吐中位数 265.0 cells/s，loss 0.30111，lr 9.51e-05。完整断点 step 13,500。固定去噪探针 step 13,500，EMA endpoint MSE 0.09913。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T13:15:44+08:00：全量训练至 E12.464、step 13,935，最近吞吐中位数 265.1 cells/s，loss 0.30222，lr 9.46e-05。完整断点 step 13,900。固定去噪探针 step 13,750，EMA endpoint MSE 0.09900。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T13:30:44+08:00：全量训练至 E12.853、step 14,370，最近吞吐中位数 265.1 cells/s，loss 0.30674，lr 9.4e-05。完整断点 step 14,350。固定去噪探针 step 14,250，EMA endpoint MSE 0.09881。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T13:45:44+08:00：全量训练至 E13.233、step 14,795，最近吞吐中位数 265.0 cells/s，loss 0.30750，lr 9.35e-05。完整断点 step 14,750。固定去噪探针 step 14,750，EMA endpoint MSE 0.09866。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T14:00:44+08:00：全量训练至 E13.623、step 15,230，最近吞吐中位数 265.3 cells/s，loss 0.29884，lr 9.28e-05。完整断点 step 15,200。固定去噪探针 step 15,000，EMA endpoint MSE 0.09859。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T14:15:44+08:00：全量训练至 E14.003、step 15,655，最近吞吐中位数 266.0 cells/s，loss 0.30489，lr 9.22e-05。完整断点 step 15,652。固定去噪探针 step 15,652，EMA endpoint MSE 0.09843。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T14:30:44+08:00：全量训练至 E14.392、step 16,090，最近吞吐中位数 265.6 cells/s，loss 0.29920，lr 9.15e-05。完整断点 step 16,050。固定去噪探针 step 16,000，EMA endpoint MSE 0.09836。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T14:45:44+08:00：全量训练至 E14.781、step 16,525，最近吞吐中位数 265.0 cells/s，loss 0.30000，lr 9.09e-05。完整断点 step 16,500。固定去噪探针 step 16,500，EMA endpoint MSE 0.09828。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。

- 2026-11-25T15:00:44+08:00：全量训练至 E15.000、step 16,770，最近吞吐中位数 265.0 cells/s，loss 0.30840，lr 9.05e-05。完整断点 step 16,770。固定去噪探针 step 16,770，EMA endpoint MSE 0.09823。最近周期验证 E10，PDCorr=0.28575，MSE=0.01023，S=0.83531。最近生成 `validation_epoch_015` 已完成 1,024/1,916 个细胞。

- 2026-11-25T15:05:17+08:00：`runs/repro3_main` 第 15 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3005，MSE=0.01002，PDS_L2=0.7386。这是验证结果，非测试结果。

- 2026-11-25T15:05:22+08:00：`runs/repro3_main` 完成第 15/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09823。

- 2026-11-25T15:06:21+08:00：第 15 轮缓存的 CPU 九指标检查完成：DEOver=0.1102、DEPrec=0.0837、DirAgr=0.6794；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_015/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-25T15:15:44+08:00：全量训练至 E15.264、step 17,065，最近吞吐中位数 265.1 cells/s，loss 0.30341，lr 9e-05。完整断点 step 17,050。固定去噪探针 step 17,000，EMA endpoint MSE 0.09820。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T15:30:44+08:00：全量训练至 E15.653、step 17,500，最近吞吐中位数 263.7 cells/s，loss 0.30369，lr 8.92e-05。完整断点 step 17,450。固定去噪探针 step 17,500，EMA endpoint MSE 0.09814。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T15:45:44+08:00：全量训练至 E16.033、step 17,925，最近吞吐中位数 264.3 cells/s，loss 0.30348，lr 8.85e-05。完整断点 step 17,900。固定去噪探针 step 17,888，EMA endpoint MSE 0.09809。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T16:00:44+08:00：全量训练至 E16.418、step 18,355，最近吞吐中位数 263.8 cells/s，loss 0.30346，lr 8.77e-05。完整断点 step 18,350。固定去噪探针 step 18,250，EMA endpoint MSE 0.09805。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T16:15:44+08:00：全量训练至 E16.802、step 18,785，最近吞吐中位数 264.1 cells/s，loss 0.30266，lr 8.69e-05。完整断点 step 18,750。固定去噪探针 step 18,750，EMA endpoint MSE 0.09799。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T16:30:44+08:00：全量训练至 E17.182、step 19,210，最近吞吐中位数 264.8 cells/s，loss 0.30356，lr 8.61e-05。完整断点 step 19,200。固定去噪探针 step 19,006，EMA endpoint MSE 0.09796。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T16:45:44+08:00：全量训练至 E17.572、step 19,645，最近吞吐中位数 265.6 cells/s，loss 0.30419，lr 8.52e-05。完整断点 step 19,600。固定去噪探针 step 19,500，EMA endpoint MSE 0.09792。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T17:00:44+08:00：全量训练至 E17.965、step 20,085，最近吞吐中位数 265.4 cells/s，loss 0.29733，lr 8.43e-05。完整断点 step 20,050。固定去噪探针 step 20,000，EMA endpoint MSE 0.09788。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T17:15:44+08:00：全量训练至 E18.354、step 20,520，最近吞吐中位数 265.9 cells/s，loss 0.30110，lr 8.34e-05。完整断点 step 20,500。固定去噪探针 step 20,500，EMA endpoint MSE 0.09785。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T17:30:44+08:00：全量训练至 E18.752、step 20,965，最近吞吐中位数 265.3 cells/s，loss 0.29963，lr 8.25e-05。完整断点 step 20,950。固定去噪探针 step 20,750，EMA endpoint MSE 0.09784。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T17:45:44+08:00：全量训练至 E19.141、step 21,400，最近吞吐中位数 265.7 cells/s，loss 0.29936，lr 8.15e-05。完整断点 step 21,400。固定去噪探针 step 21,250，EMA endpoint MSE 0.09782。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T18:00:44+08:00：全量训练至 E19.535、step 21,840，最近吞吐中位数 265.7 cells/s，loss 0.30171，lr 8.06e-05。完整断点 step 21,800。固定去噪探针 step 21,750，EMA endpoint MSE 0.09780。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T18:15:44+08:00：全量训练至 E19.928、step 22,280，最近吞吐中位数 266.0 cells/s，loss 0.30025，lr 7.96e-05。完整断点 step 22,250。固定去噪探针 step 22,250，EMA endpoint MSE 0.09779。最近周期验证 E15，PDCorr=0.30054，MSE=0.01002，S=0.93281。

- 2026-11-25T18:29:09+08:00：`runs/repro3_main` 第 20 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3314，MSE=0.01037，PDS_L2=0.7811。这是验证结果，非测试结果。

- 2026-11-25T18:29:13+08:00：`runs/repro3_main` 完成第 20/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09779。

- 2026-11-25T18:29:34+08:00：第 20 轮缓存的 CPU 九指标检查完成：DEOver=0.1227、DEPrec=0.0895、DirAgr=0.7181；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_020/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-25T18:30:44+08:00：全量训练至 E20.036、step 22,400，最近吞吐中位数 266.1 cells/s，loss 0.30266，lr 7.93e-05。完整断点 step 22,400。固定去噪探针 step 22,360，EMA endpoint MSE 0.09779。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T18:34:49+08:00：E20 复核 1,916 个相同索引与生成种子。第三轮相对第二轮 PDCorr +0.01153、MSE -0.00020。九项差值及按 60 个扰动配对区间见 `runs/repro3_main/observation/pair_repro2_e020.json`。

- 2026-11-25T18:45:44+08:00：全量训练至 E20.434、step 22,845，最近吞吐中位数 266.0 cells/s，loss 0.30352，lr 7.83e-05。完整断点 step 22,800。固定去噪探针 step 22,750，EMA endpoint MSE 0.09778。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T19:00:44+08:00：全量训练至 E20.832、step 23,290，最近吞吐中位数 265.6 cells/s，loss 0.29949，lr 7.72e-05。完整断点 step 23,250。固定去噪探针 step 23,250，EMA endpoint MSE 0.09777。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T19:15:44+08:00：全量训练至 E21.221、step 23,725，最近吞吐中位数 265.9 cells/s，loss 0.29799，lr 7.62e-05。完整断点 step 23,700。固定去噪探针 step 23,500，EMA endpoint MSE 0.09776。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T19:30:44+08:00：全量训练至 E21.619、step 24,170，最近吞吐中位数 265.9 cells/s，loss 0.30097，lr 7.51e-05。完整断点 step 24,150。固定去噪探针 step 24,000，EMA endpoint MSE 0.09776。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T19:45:44+08:00：全量训练至 E22.008、step 24,605，最近吞吐中位数 265.6 cells/s，loss 0.29391，lr 7.41e-05。完整断点 step 24,600。固定去噪探针 step 24,596，EMA endpoint MSE 0.09775。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T20:00:44+08:00：全量训练至 E22.402、step 25,045，最近吞吐中位数 265.8 cells/s，loss 0.30721，lr 7.3e-05。完整断点 step 25,000。固定去噪探针 step 25,000，EMA endpoint MSE 0.09775。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T20:15:44+08:00：全量训练至 E22.804、step 25,495，最近吞吐中位数 266.0 cells/s，loss 0.30287，lr 7.18e-05。完整断点 step 25,450。固定去噪探针 step 25,250，EMA endpoint MSE 0.09775。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T20:30:44+08:00：全量训练至 E23.193、step 25,930，最近吞吐中位数 265.9 cells/s，loss 0.29907，lr 7.07e-05。完整断点 step 25,900。固定去噪探针 step 25,750，EMA endpoint MSE 0.09772。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T20:45:44+08:00：全量训练至 E23.582、step 26,365，最近吞吐中位数 265.5 cells/s，loss 0.30038，lr 6.96e-05。完整断点 step 26,350。固定去噪探针 step 26,250，EMA endpoint MSE 0.09770。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T21:00:44+08:00：全量训练至 E23.976、step 26,805，最近吞吐中位数 265.7 cells/s，loss 0.30169，lr 6.85e-05。完整断点 step 26,800。固定去噪探针 step 26,750，EMA endpoint MSE 0.09769。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T21:15:44+08:00：全量训练至 E24.369、step 27,245，最近吞吐中位数 265.8 cells/s，loss 0.29979，lr 6.73e-05。完整断点 step 27,200。固定去噪探针 step 27,000，EMA endpoint MSE 0.09769。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T21:30:44+08:00：全量训练至 E24.763、step 27,685，最近吞吐中位数 265.9 cells/s，loss 0.29943，lr 6.62e-05。完整断点 step 27,650。固定去噪探针 step 27,500，EMA endpoint MSE 0.09769。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。

- 2026-11-25T21:45:44+08:00：全量训练至 E25.000、step 27,950，最近吞吐中位数 265.8 cells/s，loss 0.30255，lr 6.55e-05。完整断点 step 27,950。固定去噪探针 step 27,950，EMA endpoint MSE 0.09770。最近周期验证 E20，PDCorr=0.33140，MSE=0.01037，S=0.99985。最近生成 `validation_epoch_025` 已完成 1,024/1,916 个细胞。

- 2026-11-25T21:50:35+08:00：`runs/repro3_main` 第 25 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3330，MSE=0.01087，PDS_L2=0.7883。这是验证结果，非测试结果。

- 2026-11-25T21:50:40+08:00：`runs/repro3_main` 完成第 25/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09770。

- 2026-11-25T21:51:01+08:00：第 25 轮缓存的 CPU 九指标检查完成：DEOver=0.1206、DEPrec=0.0897、DirAgr=0.7315；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_025/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-25T22:00:44+08:00：全量训练至 E25.259、step 28,240，最近吞吐中位数 266.0 cells/s，loss 0.30406，lr 6.47e-05。完整断点 step 28,200。固定去噪探针 step 28,000，EMA endpoint MSE 0.09770。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T22:15:44+08:00：全量训练至 E25.648、step 28,675，最近吞吐中位数 266.0 cells/s，loss 0.29828，lr 6.35e-05。完整断点 step 28,650。固定去噪探针 step 28,500，EMA endpoint MSE 0.09770。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T22:30:44+08:00：全量训练至 E26.029、step 29,100，最近吞吐中位数 265.4 cells/s，loss 0.30194，lr 6.24e-05。完整断点 step 29,100。固定去噪探针 step 29,068，EMA endpoint MSE 0.09769。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T22:45:44+08:00：全量训练至 E26.418、step 29,535，最近吞吐中位数 265.8 cells/s，loss 0.30210，lr 6.12e-05。完整断点 step 29,500。固定去噪探针 step 29,500，EMA endpoint MSE 0.09770。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T23:00:44+08:00：全量训练至 E26.811、step 29,975，最近吞吐中位数 266.0 cells/s，loss 0.29985，lr 6e-05。完整断点 step 29,950。固定去噪探针 step 29,750，EMA endpoint MSE 0.09770。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T23:15:44+08:00：全量训练至 E27.191、step 30,400，最近吞吐中位数 265.7 cells/s，loss 0.29954，lr 5.88e-05。完整断点 step 30,400。固定去噪探针 step 30,250，EMA endpoint MSE 0.09769。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T23:30:44+08:00：全量训练至 E27.581、step 30,835，最近吞吐中位数 265.8 cells/s，loss 0.29940，lr 5.76e-05。完整断点 step 30,800。固定去噪探针 step 30,750，EMA endpoint MSE 0.09769。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-25T23:45:44+08:00：全量训练至 E27.965、step 31,265，最近吞吐中位数 264.9 cells/s，loss 0.29869，lr 5.64e-05。完整断点 step 31,250。固定去噪探针 step 31,250，EMA endpoint MSE 0.09768。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T00:00:44+08:00：全量训练至 E28.354、step 31,700，最近吞吐中位数 264.2 cells/s，loss 0.29607，lr 5.52e-05。完整断点 step 31,700。固定去噪探针 step 31,500，EMA endpoint MSE 0.09768。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T00:15:44+08:00：全量训练至 E28.748、step 32,140，最近吞吐中位数 264.5 cells/s，loss 0.30512，lr 5.39e-05。完整断点 step 32,100。固定去噪探针 step 32,000，EMA endpoint MSE 0.09767。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T00:30:44+08:00：全量训练至 E29.123、step 32,560，最近吞吐中位数 263.8 cells/s，loss 0.30534，lr 5.28e-05。完整断点 step 32,550。固定去噪探针 step 32,500，EMA endpoint MSE 0.09767。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T00:45:44+08:00：全量训练至 E29.517、step 33,000，最近吞吐中位数 264.3 cells/s，loss 0.29962，lr 5.15e-05。完整断点 step 32,950。固定去噪探针 step 32,750，EMA endpoint MSE 0.09766。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T01:00:44+08:00：全量训练至 E29.897、step 33,425，最近吞吐中位数 264.2 cells/s，loss 0.29992，lr 5.03e-05。完整断点 step 33,400。固定去噪探针 step 33,250，EMA endpoint MSE 0.09767。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。

- 2026-11-26T01:15:44+08:00：全量训练至 E30.000、step 33,540，最近吞吐中位数 264.2 cells/s，loss 0.29890，lr 5e-05。完整断点 step 33,540。固定去噪探针 step 33,540，EMA endpoint MSE 0.09768。最近周期验证 E25，PDCorr=0.33296，MSE=0.01087，S=1.00708。最近生成 `validation_epoch_030` 已完成 1,884/1,916 个细胞。

- 2026-11-26T01:15:55+08:00：`runs/repro3_main` 第 30 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3351，MSE=0.01184，PDS_L2=0.7975。这是验证结果，非测试结果。

- 2026-11-26T01:16:00+08:00：`runs/repro3_main` 完成第 30/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09768。

- 2026-11-26T01:17:00+08:00：第 30 轮缓存的 CPU 九指标检查完成：DEOver=0.1227、DEPrec=0.0901、DirAgr=0.7326；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_030/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T01:23:30+08:00：E30 的周期分数 S=1.01482，较 E25 高 0.00774，按登记标准刷新收敛参考轮次。与第二轮 E30 同协议比较，PDCorr 高 0.00580，MSE 低 0.00019，三种 PDS 均值低 0.00370。

- 2026-11-26T01:30:44+08:00：全量训练至 E30.380、step 33,965，最近吞吐中位数 264.4 cells/s，loss 0.30560，lr 4.88e-05。完整断点 step 33,950。固定去噪探针 step 33,750，EMA endpoint MSE 0.09768。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T01:45:44+08:00：全量训练至 E30.765、step 34,395，最近吞吐中位数 264.3 cells/s，loss 0.29709，lr 4.76e-05。完整断点 step 34,350。固定去噪探针 step 34,250，EMA endpoint MSE 0.09768。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T02:00:44+08:00：全量训练至 E31.145、step 34,820，最近吞吐中位数 264.5 cells/s，loss 0.30233，lr 4.64e-05。完整断点 step 34,800。固定去噪探针 step 34,750，EMA endpoint MSE 0.09768。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T02:15:44+08:00：全量训练至 E31.530、step 35,250，最近吞吐中位数 265.7 cells/s，loss 0.29861，lr 4.52e-05。完整断点 step 35,250。固定去噪探针 step 35,250，EMA endpoint MSE 0.09768。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T02:30:44+08:00：全量训练至 E31.923、step 35,690，最近吞吐中位数 265.7 cells/s，loss 0.29570，lr 4.4e-05。完整断点 step 35,650。固定去噪探针 step 35,500，EMA endpoint MSE 0.09768。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T02:45:44+08:00：全量训练至 E32.303、step 36,115，最近吞吐中位数 265.2 cells/s，loss 0.30217，lr 4.28e-05。完整断点 step 36,100。固定去噪探针 step 36,000，EMA endpoint MSE 0.09767。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T03:00:46+08:00：全量训练至 E32.692、step 36,550，最近吞吐中位数 265.6 cells/s，loss 0.29895，lr 4.16e-05。完整断点 step 36,550。固定去噪探针 step 36,500，EMA endpoint MSE 0.09767。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T03:15:44+08:00：全量训练至 E33.077、step 36,980，最近吞吐中位数 265.5 cells/s，loss 0.29931，lr 4.04e-05。完整断点 step 36,950。固定去噪探针 step 36,894，EMA endpoint MSE 0.09766。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T03:30:44+08:00：全量训练至 E33.466、step 37,415，最近吞吐中位数 265.7 cells/s，loss 0.29786，lr 3.92e-05。完整断点 step 37,400。固定去噪探针 step 37,250，EMA endpoint MSE 0.09767。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T03:45:45+08:00：全量训练至 E33.855、step 37,850，最近吞吐中位数 265.7 cells/s，loss 0.30139，lr 3.8e-05。完整断点 step 37,800。固定去噪探针 step 37,750，EMA endpoint MSE 0.09766。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T04:00:44+08:00：全量训练至 E34.235、step 38,275，最近吞吐中位数 266.0 cells/s，loss 0.30238，lr 3.69e-05。完整断点 step 38,250。固定去噪探针 step 38,250，EMA endpoint MSE 0.09767。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T04:15:44+08:00：全量训练至 E34.624、step 38,710，最近吞吐中位数 266.1 cells/s，loss 0.29642，lr 3.57e-05。完整断点 step 38,700。固定去噪探针 step 38,500，EMA endpoint MSE 0.09767。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。

- 2026-11-26T04:30:44+08:00：全量训练至 E35.000、step 39,130，最近吞吐中位数 266.0 cells/s，loss 0.30423，lr 3.46e-05。完整断点 step 39,130。固定去噪探针 step 39,130，EMA endpoint MSE 0.09770。最近周期验证 E30，PDCorr=0.33507，MSE=0.01184，S=1.01482。最近生成 `validation_epoch_030` 已完成 1,916/1,916 个细胞。

- 2026-11-26T04:40:59+08:00：`runs/repro3_main` 第 35 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3302，MSE=0.01181，PDS_L2=0.7972。这是验证结果，非测试结果。

- 2026-11-26T04:41:04+08:00：`runs/repro3_main` 完成第 35/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09770。

- 2026-11-26T04:42:09+08:00：第 35 轮缓存的 CPU 九指标检查完成：DEOver=0.1222、DEPrec=0.0920、DirAgr=0.7321；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_035/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T04:45:44+08:00：全量训练至 E35.116、step 39,260，最近吞吐中位数 264.4 cells/s，loss 0.29636，lr 3.42e-05。完整断点 step 39,250。固定去噪探针 step 39,250，EMA endpoint MSE 0.09770。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T04:55:56+08:00：E35 周期 S=1.01800，较 E30 增加 0.00317，超过预登记的 0.002 收敛刷新阈值。原综合分差 +0.00290，按扰动配对 95% 区间 [−0.01277,+0.01895]。详见 `runs/repro3_main/observation/pair_e030_e035/comparison.json`。

- 2026-11-26T05:00:45+08:00：全量训练至 E35.510、step 39,700，最近吞吐中位数 266.0 cells/s，loss 0.30083，lr 3.3e-05。完整断点 step 39,650。固定去噪探针 step 39,500，EMA endpoint MSE 0.09771。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T05:15:44+08:00：全量训练至 E35.894、step 40,130，最近吞吐中位数 265.7 cells/s，loss 0.30102，lr 3.19e-05。完整断点 step 40,100。固定去噪探针 step 40,000，EMA endpoint MSE 0.09772。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T05:30:44+08:00：全量训练至 E36.275、step 40,555，最近吞吐中位数 265.8 cells/s，loss 0.30003，lr 3.08e-05。完整断点 step 40,550。固定去噪探针 step 40,500，EMA endpoint MSE 0.09773。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T05:45:44+08:00：全量训练至 E36.668、step 40,995，最近吞吐中位数 266.0 cells/s，loss 0.30044，lr 2.97e-05。完整断点 step 40,950。固定去噪探针 step 40,750，EMA endpoint MSE 0.09773。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T06:00:44+08:00：全量训练至 E37.048、step 41,420，最近吞吐中位数 265.8 cells/s，loss 0.29555，lr 2.86e-05。完整断点 step 41,400。固定去噪探针 step 41,366，EMA endpoint MSE 0.09774。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T06:15:44+08:00：全量训练至 E37.437、step 41,855，最近吞吐中位数 265.6 cells/s，loss 0.30367，lr 2.75e-05。完整断点 step 41,850。固定去噪探针 step 41,750，EMA endpoint MSE 0.09775。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T06:30:44+08:00：全量训练至 E37.826、step 42,290，最近吞吐中位数 265.8 cells/s，loss 0.29864，lr 2.64e-05。完整断点 step 42,250。固定去噪探针 step 42,250，EMA endpoint MSE 0.09776。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T06:45:44+08:00：全量训练至 E38.211、step 42,720，最近吞吐中位数 265.7 cells/s，loss 0.29463，lr 2.53e-05。完整断点 step 42,700。固定去噪探针 step 42,500，EMA endpoint MSE 0.09777。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T07:00:44+08:00：全量训练至 E38.596、step 43,150，最近吞吐中位数 265.8 cells/s，loss 0.29799，lr 2.43e-05。完整断点 step 43,150。固定去噪探针 step 43,000，EMA endpoint MSE 0.09779。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T07:15:44+08:00：全量训练至 E38.985、step 43,585，最近吞吐中位数 265.7 cells/s，loss 0.29500，lr 2.33e-05。完整断点 step 43,550。固定去噪探针 step 43,500，EMA endpoint MSE 0.09780。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T07:30:44+08:00：全量训练至 E39.365、step 44,010，最近吞吐中位数 265.9 cells/s，loss 0.29903，lr 2.23e-05。完整断点 step 44,000。固定去噪探针 step 44,000，EMA endpoint MSE 0.09780。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T07:45:44+08:00：全量训练至 E39.758、step 44,450，最近吞吐中位数 266.1 cells/s，loss 0.29531，lr 2.12e-05。完整断点 step 44,450。固定去噪探针 step 44,250，EMA endpoint MSE 0.09780。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。

- 2026-11-26T08:00:44+08:00：全量训练至 E40.000、step 44,720，最近吞吐中位数 266.0 cells/s，loss 0.29927，lr 2.06e-05。完整断点 step 44,720。固定去噪探针 step 44,720，EMA endpoint MSE 0.09782。最近周期验证 E35，PDCorr=0.33018，MSE=0.01181，S=1.01800。最近生成 `validation_epoch_040` 已完成 896/1,916 个细胞。

- 2026-11-26T08:05:45+08:00：`runs/repro3_main` 第 40 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3222，MSE=0.01237，PDS_L2=0.7981。这是验证结果，非测试结果。

- 2026-11-26T08:05:50+08:00：`runs/repro3_main` 完成第 40/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09782。

- 2026-11-26T08:06:21+08:00：第 40 轮缓存的 CPU 九指标检查完成：DEOver=0.1195、DEPrec=0.0915、DirAgr=0.7572；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_040/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T08:15:44+08:00：全量训练至 E40.250、step 45,000，最近吞吐中位数 266.0 cells/s，loss 0.30126，lr 2e-05。完整断点 step 45,000。固定去噪探针 step 45,000，EMA endpoint MSE 0.09783。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T08:30:44+08:00：全量训练至 E40.644、step 45,440，最近吞吐中位数 265.9 cells/s，loss 0.29791，lr 1.9e-05。完整断点 step 45,400。固定去噪探针 step 45,250，EMA endpoint MSE 0.09784。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T08:45:44+08:00：全量训练至 E41.024、step 45,865，最近吞吐中位数 265.5 cells/s，loss 0.30011，lr 1.81e-05。完整断点 step 45,850。固定去噪探针 step 45,838，EMA endpoint MSE 0.09785。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T09:00:44+08:00：全量训练至 E41.413、step 46,300，最近吞吐中位数 265.8 cells/s，loss 0.29998，lr 1.71e-05。完整断点 step 46,300。固定去噪探针 step 46,250，EMA endpoint MSE 0.09786。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T09:15:44+08:00：全量训练至 E41.807、step 46,740，最近吞吐中位数 265.8 cells/s，loss 0.29923，lr 1.62e-05。完整断点 step 46,700。固定去噪探针 step 46,500，EMA endpoint MSE 0.09786。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T09:30:44+08:00：全量训练至 E42.187、step 47,165，最近吞吐中位数 265.7 cells/s，loss 0.29662，lr 1.53e-05。完整断点 step 47,150。固定去噪探针 step 47,000，EMA endpoint MSE 0.09786。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T09:45:44+08:00：全量训练至 E42.576、step 47,600，最近吞吐中位数 265.5 cells/s，loss 0.29859，lr 1.45e-05。完整断点 step 47,600。固定去噪探针 step 47,500，EMA endpoint MSE 0.09787。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T10:00:44+08:00：全量训练至 E42.965、step 48,035，最近吞吐中位数 265.8 cells/s，loss 0.30112，lr 1.36e-05。完整断点 step 48,000。固定去噪探针 step 48,000，EMA endpoint MSE 0.09787。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T10:15:44+08:00：全量训练至 E43.354、step 48,470，最近吞吐中位数 266.0 cells/s，loss 0.30173，lr 1.28e-05。完整断点 step 48,450。固定去噪探针 step 48,250，EMA endpoint MSE 0.09787。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T10:30:44+08:00：全量训练至 E43.748、step 48,910，最近吞吐中位数 265.7 cells/s，loss 0.29578，lr 1.2e-05。完整断点 step 48,900。固定去噪探针 step 48,750，EMA endpoint MSE 0.09787。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T10:45:44+08:00：全量训练至 E44.141、step 49,350，最近吞吐中位数 265.4 cells/s，loss 0.29776，lr 1.12e-05。完整断点 step 49,300。固定去噪探针 step 49,250，EMA endpoint MSE 0.09788。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T11:00:44+08:00：全量训练至 E44.535、step 49,790，最近吞吐中位数 265.4 cells/s，loss 0.30186，lr 1.04e-05。完整断点 step 49,750。固定去噪探针 step 49,750，EMA endpoint MSE 0.09789。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T11:15:44+08:00：全量训练至 E44.933、step 50,235，最近吞吐中位数 265.6 cells/s，loss 0.30097，lr 9.68e-06。完整断点 step 50,200。固定去噪探针 step 50,000，EMA endpoint MSE 0.09790。最近周期验证 E40，PDCorr=0.32223，MSE=0.01237，S=1.00460。

- 2026-11-26T11:29:18+08:00：`runs/repro3_main` 第 45 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3330，MSE=0.01247，PDS_L2=0.8056。这是验证结果，非测试结果。

- 2026-11-26T11:29:21+08:00：`runs/repro3_main` 完成第 45/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09790。

- 2026-11-26T11:29:45+08:00：第 45 轮缓存的 CPU 九指标检查完成：DEOver=0.1354、DEPrec=0.0911、DirAgr=0.7460；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_045/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T11:30:29+08:00：E45 周期 S=1.02233，相对 E35 增加 0.00433，满足 0.002 的收敛参考刷新条件。PDS 均值由 0.80593 到 0.81398，MSE 由 0.01181 到 0.01247。依原 55 轮预算继续，周期最佳仍由预登记 S 选定。

- 2026-11-26T11:30:44+08:00：全量训练至 E45.036、step 50,350，最近吞吐中位数 265.7 cells/s，loss 0.29856，lr 9.48e-06。完整断点 step 50,350。固定去噪探针 step 50,310，EMA endpoint MSE 0.09790。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T11:45:44+08:00：全量训练至 E45.429、step 50,790，最近吞吐中位数 265.6 cells/s，loss 0.29822，lr 8.77e-06。完整断点 step 50,750。固定去噪探针 step 50,750，EMA endpoint MSE 0.09791。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T12:00:44+08:00：全量训练至 E45.832、step 51,240，最近吞吐中位数 265.7 cells/s，loss 0.29590，lr 8.07e-06。完整断点 step 51,200。固定去噪探针 step 51,000，EMA endpoint MSE 0.09792。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T12:15:44+08:00：全量训练至 E46.216、step 51,670，最近吞吐中位数 265.4 cells/s，loss 0.29955，lr 7.42e-06。完整断点 step 51,650。固定去噪探针 step 51,500，EMA endpoint MSE 0.09793。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T12:30:44+08:00：全量训练至 E46.614、step 52,115，最近吞吐中位数 265.3 cells/s，loss 0.29831，lr 6.78e-06。完整断点 step 52,100。固定去噪探针 step 52,000，EMA endpoint MSE 0.09794。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T12:45:44+08:00：全量训练至 E46.999、step 52,545，最近吞吐中位数 265.5 cells/s，loss 0.29793，lr 6.19e-06。完整断点 step 52,546。固定去噪探针 step 52,546，EMA endpoint MSE 0.09795。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T13:00:44+08:00：全量训练至 E47.388、step 52,980，最近吞吐中位数 265.6 cells/s，loss 0.30078，lr 5.61e-06。完整断点 step 52,950。固定去噪探针 step 52,750，EMA endpoint MSE 0.09796。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T13:15:44+08:00：全量训练至 E47.777、step 53,415，最近吞吐中位数 265.8 cells/s，loss 0.30174，lr 5.06e-06。完整断点 step 53,400。固定去噪探针 step 53,250，EMA endpoint MSE 0.09796。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T13:30:44+08:00：全量训练至 E48.162、step 53,845，最近吞吐中位数 265.4 cells/s，loss 0.29659，lr 4.55e-06。完整断点 step 53,800。固定去噪探针 step 53,750，EMA endpoint MSE 0.09797。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T13:45:44+08:00：全量训练至 E48.547、step 54,275，最近吞吐中位数 264.7 cells/s，loss 0.30169，lr 4.06e-06。完整断点 step 54,250。固定去噪探针 step 54,250，EMA endpoint MSE 0.09798。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T14:00:44+08:00：全量训练至 E48.936、step 54,710，最近吞吐中位数 264.7 cells/s，loss 0.29614，lr 3.59e-06。完整断点 step 54,700。固定去噪探针 step 54,500，EMA endpoint MSE 0.09799。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T14:15:44+08:00：全量训练至 E49.316、step 55,135，最近吞吐中位数 264.5 cells/s，loss 0.29709，lr 3.16e-06。完整断点 step 55,100。固定去噪探针 step 55,000，EMA endpoint MSE 0.09800。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T14:30:44+08:00：全量训练至 E49.705、step 55,570，最近吞吐中位数 264.2 cells/s，loss 0.29633，lr 2.74e-06。完整断点 step 55,550。固定去噪探针 step 55,500，EMA endpoint MSE 0.09801。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。

- 2026-11-26T14:45:44+08:00：全量训练至 E50.000、step 55,900，最近吞吐中位数 264.4 cells/s，loss 0.29937，lr 2.45e-06。完整断点 step 55,900。固定去噪探针 step 55,900，EMA endpoint MSE 0.09801。最近周期验证 E45，PDCorr=0.33304，MSE=0.01247，S=1.02233。最近生成 `validation_epoch_050` 已完成 640/1,916 个细胞。

- 2026-11-26T14:52:51+08:00：`runs/repro3_main` 第 50 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3321，MSE=0.01294，PDS_L2=0.8028。这是验证结果，非测试结果。

- 2026-11-26T14:52:55+08:00：`runs/repro3_main` 完成第 50/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09801。

- 2026-11-26T14:54:02+08:00：第 50 轮缓存的 CPU 九指标检查完成：DEOver=0.1363、DEPrec=0.0922、DirAgr=0.7500；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_050/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T15:00:44+08:00：全量训练至 E50.206、step 56,130，最近吞吐中位数 264.3 cells/s，loss 0.29771，lr 2.25e-06。完整断点 step 56,100。固定去噪探针 step 56,000，EMA endpoint MSE 0.09801。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T15:15:44+08:00：全量训练至 E50.595、step 56,565，最近吞吐中位数 264.6 cells/s，loss 0.30065，lr 1.9e-06。完整断点 step 56,550。固定去噪探针 step 56,500，EMA endpoint MSE 0.09802。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T15:30:44+08:00：全量训练至 E50.988、step 57,005，最近吞吐中位数 264.7 cells/s，loss 0.29932，lr 1.58e-06。完整断点 step 57,000。固定去噪探针 step 57,000，EMA endpoint MSE 0.09803。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T15:45:44+08:00：全量训练至 E51.382、step 57,445，最近吞吐中位数 264.6 cells/s，loss 0.29906，lr 1.29e-06。完整断点 step 57,400。固定去噪探针 step 57,250，EMA endpoint MSE 0.09803。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T16:00:44+08:00：全量训练至 E51.775、step 57,885，最近吞吐中位数 264.4 cells/s，loss 0.29830，lr 1.02e-06。完整断点 step 57,850。固定去噪探针 step 57,750，EMA endpoint MSE 0.09804。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T16:15:44+08:00：全量训练至 E52.165、step 58,320，最近吞吐中位数 264.4 cells/s，loss 0.29721，lr 7.92e-07。完整断点 step 58,300。固定去噪探针 step 58,250，EMA endpoint MSE 0.09804。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T16:30:44+08:00：全量训练至 E52.558、step 58,760，最近吞吐中位数 264.6 cells/s，loss 0.29665，lr 5.88e-07。完整断点 step 58,750。固定去噪探针 step 58,750，EMA endpoint MSE 0.09805。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T16:45:44+08:00：全量训练至 E52.956、step 59,205，最近吞吐中位数 264.9 cells/s，loss 0.29871，lr 4.12e-07。完整断点 step 59,200。固定去噪探针 step 59,000，EMA endpoint MSE 0.09805。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T17:00:44+08:00：全量训练至 E53.345、step 59,640，最近吞吐中位数 264.8 cells/s，loss 0.29501，lr 2.7e-07。完整断点 step 59,600。固定去噪探针 step 59,500，EMA endpoint MSE 0.09805。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T17:15:44+08:00：全量训练至 E53.743、step 60,085，最近吞吐中位数 264.7 cells/s，loss 0.29536，lr 1.56e-07。完整断点 step 60,050。固定去噪探针 step 60,000，EMA endpoint MSE 0.09806。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T17:30:44+08:00：全量训练至 E54.132、step 60,520，最近吞吐中位数 264.7 cells/s，loss 0.29530，lr 7.44e-08。完整断点 step 60,500。固定去噪探针 step 60,500，EMA endpoint MSE 0.09806。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T17:45:44+08:00：全量训练至 E54.530、step 60,965，最近吞吐中位数 265.0 cells/s，loss 0.29763，lr 2.18e-08。完整断点 step 60,950。固定去噪探针 step 60,750，EMA endpoint MSE 0.09806。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T18:00:44+08:00：全量训练至 E54.924、step 61,405，最近吞吐中位数 265.6 cells/s，loss 0.29951，lr 5.84e-10。完整断点 step 61,400。固定去噪探针 step 61,250，EMA endpoint MSE 0.09806。最近周期验证 E50，PDCorr=0.33206，MSE=0.01294，S=1.01394。

- 2026-11-26T18:14:16+08:00：`runs/repro3_main` 第 55 轮验证生成（EMA=0.9996，每扰动最多 32 细胞，50 步）：PDCorr=0.3312，MSE=0.01302，PDS_L2=0.8047。这是验证结果，非测试结果。

- 2026-11-26T18:14:19+08:00：`runs/repro3_main` 完成第 55/55 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.09806。

- 2026-11-26T18:14:19+08:00：`runs/repro3_main` 验证生成指标连续 10 轮无改善，按预设验证收敛规则停止。

- 2026-11-26T18:14:23+08:00：`runs/repro3_main` 训练结束（55.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-26T18:14:47+08:00：第 55 轮缓存的 CPU 九指标检查完成：DEOver=0.1364、DEPrec=0.0911、DirAgr=0.7275；复用既有预测，无新增生成。完整结果见 `runs/repro3_main/validation_epoch_055/metrics_de_diagnostic/summary.json`（周期验证诊断，非正式测试）。

- 2026-11-26T18:15:44+08:00：全量工作流处于CFG 校准阶段。已完成 0/15 个生成评估。最近生成 `screen_00` 已完成 608/960 个细胞。

- 2026-11-26T18:16:33+08:00：验证集 CFG 校准 `screen_00`：w=None，区间 [0.1,0.9]，PDCorr=0.2408，PDS_L2=0.6822，MSE=0.01391（非测试）。

- 2026-11-26T18:17:39+08:00：训练期间工作区增加了源码和 YAML 注释。训练副本 SHA256 仍匹配锁定配置。逐文件确认 Python 语法树和 YAML 参数一致，并保留修改前后摘要。后续周期配对复用了既有结果，E55 相对 E45 的 ΔS 为 -0.00699，95% 区间 [-0.02237,+0.00816]。 [源码核验](runs/repro3_main/source_annotations_verification.json) · [配对结果](runs/repro3_main/observation/pair_e045_e055/comparison.json)

- 2026-11-26T18:18:14+08:00：四卡全量训练检查通过，校准使用 E45 EMA 0.9996。E55 与历史启动配置的原始副本已保存，完整验证冻结后开始一次正式测试。

- 2026-11-26T18:23:00+08:00：验证集 CFG 校准 `screen_01`：w=1.125，区间 [0.1,0.9]，PDCorr=0.2780，PDS_L2=0.7581，MSE=0.01463（非测试）。

- 2026-11-26T18:28:57+08:00：验证集 CFG 校准 `screen_02`：w=1.25，区间 [0.1,0.9]，PDCorr=0.2833，PDS_L2=0.7742，MSE=0.01530（非测试）。

- 2026-11-26T18:30:44+08:00：全量工作流处于CFG 校准阶段。已完成 3/15 个生成评估。最近生成 `screen_03` 已完成 256/960 个细胞。

- 2026-11-26T18:34:54+08:00：验证集 CFG 校准 `screen_03`：w=1.375，区间 [0.1,0.9]，PDCorr=0.2878，PDS_L2=0.7861，MSE=0.01612（非测试）。

- 2026-11-26T18:40:53+08:00：验证集 CFG 校准 `screen_04`：w=1.5，区间 [0.1,0.9]，PDCorr=0.2924，PDS_L2=0.7969，MSE=0.01703（非测试）。

- 2026-11-26T18:45:44+08:00：全量工作流处于CFG 校准阶段。已完成 5/15 个生成评估。最近生成 `screen_05` 已完成 768/960 个细胞。

- 2026-11-26T18:46:48+08:00：验证集 CFG 校准 `screen_05`：w=1.625，区间 [0.1,0.9]，PDCorr=0.2947，PDS_L2=0.8047，MSE=0.01810（非测试）。

- 2026-11-26T18:52:45+08:00：验证集 CFG 校准 `screen_06`：w=1.75，区间 [0.1,0.9]，PDCorr=0.2974，PDS_L2=0.8117，MSE=0.01926（非测试）。

- 2026-11-26T18:59:30+08:00：验证集 CFG 校准 `screen_full_01`：w=1.125，区间 [0.0,1.0]，PDCorr=0.3044，PDS_L2=0.7811，MSE=0.01489（非测试）。

- 2026-11-26T19:00:44+08:00：全量工作流处于CFG 校准阶段。已完成 8/15 个生成评估。最近生成 `screen_full_02` 已完成 128/960 个细胞。

- 2026-11-26T19:06:13+08:00：验证集 CFG 校准 `screen_full_02`：w=1.25，区间 [0.0,1.0]，PDCorr=0.3112，PDS_L2=0.7892，MSE=0.01584（非测试）。

- 2026-11-26T19:12:59+08:00：验证集 CFG 校准 `screen_full_03`：w=1.375，区间 [0.0,1.0]，PDCorr=0.3170，PDS_L2=0.7994，MSE=0.01695（非测试）。

- 2026-11-26T19:15:44+08:00：全量工作流处于CFG 校准阶段。已完成 10/15 个生成评估。最近生成 `screen_full_04` 已完成 384/960 个细胞。

- 2026-11-26T19:19:44+08:00：验证集 CFG 校准 `screen_full_04`：w=1.5，区间 [0.0,1.0]，PDCorr=0.3214，PDS_L2=0.8022，MSE=0.01818（非测试）。

- 2026-11-26T19:26:30+08:00：验证集 CFG 校准 `screen_full_05`：w=1.625，区间 [0.0,1.0]，PDCorr=0.3246，PDS_L2=0.8047，MSE=0.01954（非测试）。

- 2026-11-26T19:30:44+08:00：全量工作流处于CFG 校准阶段。已完成 12/15 个生成评估。最近生成 `screen_full_06` 已完成 640/960 个细胞。

- 2026-11-26T19:33:16+08:00：验证集 CFG 校准 `screen_full_06`：w=1.75，区间 [0.0,1.0]，PDCorr=0.3282，PDS_L2=0.8061，MSE=0.02100（非测试）。

- 2026-11-26T19:45:44+08:00：全量工作流处于CFG 校准阶段。已完成 13/15 个生成评估。最近生成 `refine_00` 已完成 1,920/4,825 个细胞。

- 2026-11-26T20:00:44+08:00：全量工作流处于CFG 校准阶段。已完成 13/15 个生成评估。最近生成 `refine_00` 已完成 4,320/4,825 个细胞。

- 2026-11-26T20:04:30+08:00：验证集 CFG 校准 `refine_00`：w=1.375，区间 [0.0,1.0]，PDCorr=0.3950，PDS_L2=0.8206，MSE=0.01065（非测试）。

- 2026-11-26T20:15:44+08:00：全量工作流处于CFG 校准阶段。已完成 14/15 个生成评估。最近生成 `refine_01` 已完成 1,664/4,825 个细胞。

- 2026-11-26T20:30:44+08:00：全量工作流处于CFG 校准阶段。已完成 14/15 个生成评估。最近生成 `refine_01` 已完成 4,096/4,825 个细胞。

- 2026-11-26T20:35:45+08:00：验证集 CFG 校准 `refine_01`：w=1.125，区间 [0.0,1.0]，PDCorr=0.3954，PDS_L2=0.8122，MSE=0.00872（非测试）。

- 2026-11-26T20:35:45+08:00：完成验证集采样校准。两组全量候选和校准选择见 `runs/repro3_main/calibration/selected.json` 与 `runs/repro3_main/calibration/all_results.json`。

- 2026-11-26T20:38:30+08:00：完整验证通过全部项目门槛。冻结 E45 EMA 0.9996、w=1.375 / [0,1] / 0.05 后，四卡正式测试已启动。

- 2026-11-26T20:45:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 1,280/26,878 个细胞。

- 2026-11-26T21:00:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 3,712/26,878 个细胞。

- 2026-11-26T21:15:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 6,048/26,878 个细胞。

- 2026-11-26T21:30:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 8,416/26,878 个细胞。

- 2026-11-26T21:45:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 10,816/26,878 个细胞。

- 2026-11-26T22:00:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 13,120/26,878 个细胞。

- 2026-11-26T22:15:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 15,520/26,878 个细胞。

- 2026-11-26T22:30:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 17,856/26,878 个细胞。

- 2026-11-26T22:45:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 20,256/26,878 个细胞。

- 2026-11-26T23:00:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 22,624/26,878 个细胞。

- 2026-11-26T23:15:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_clamp0` 已完成 25,024/26,878 个细胞。

- 2026-11-26T23:29:42+08:00：完成 `/home/dataset-assist-0/whs/ocoo-t/runs/repro3_main/test_full_clamp0` 的 test 评估（细胞 26,878，weights=ema_0.9996，CFG=1.375）；指标见 `/home/dataset-assist-0/whs/ocoo-t/runs/repro3_main/test_full_clamp0/metrics/summary.json`。PDCorr=0.4132，MSE=0.01006。

- 2026-11-26T23:30:44+08:00：全量工作流处于冻结测试阶段。最近生成 `test_full_postprocess` 已完成 26,878/26,878 个细胞。

- 2026-11-26T23:31:20+08:00：`runs/repro3_main` 的完整测试与报告已产出：`runs/repro3_main/results/REPORT.md`。以下是实测结果，是否达到论文水平需结合全部指标审查；当前不会自动宣称复现成功。

- 2026-11-26T23:33:54+08:00：全量训练及正式测试均结束，预测、测试索引、冻结摘要与指标均通过最终完整性核验。四张 GPU 已释放。
