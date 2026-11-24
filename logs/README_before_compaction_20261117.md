# OCOO-T 复现（Replogle–Nadig）

目标：在本机 A100 40 GB 上复现 [OCOO-T（arXiv:2606.12838v1）](https://arxiv.org/abs/2606.12838)，只下载和运行 Replogle–Nadig，不运行 Tahoe100M / PBMC。使用独立 conda 环境 `ocoo-t`。

当前状态：数据与评估协议已验证，5 组超参数小实验顺序运行中。实时状态和关键节点在下方自动更新。**尚无最终复现指标，不宣称达到论文结果。**

## 原则

- 论文明确的架构、损失、训练和采样参数保持一致；单卡用梯度累积保持有效批量。
- 原文缺失或存在歧义的细节写入 `paper/implementation_notes.md`，通过验证集小实验选择；测试集不参与调参。
- 保存数据来源、拆分、配置、随机种子、日志、优化器及双 EMA 检查点，支持中断续训。
- 指标同时报告计算协议和与论文的可比性限制；未完成的工作如实标明。

## 进度记录（北京时间）

| 时间 | 进展 / 决策 |
| --- | --- |
| 2026-11-17 00:54 | 开始；当前目录为空。检测到 A100-PCIE 40 GB，磁盘可用约 77 GB；检索到论文 arXiv:2606.12838v1，开始保存 PDF/HTML、建立 `ocoo-t` 环境。 |
| 2026-11-17 01:01 | 下载 PerturbDiff 的 2.55 GB 压缩文件，解压为 22.30 GB；SHA256 与 Arc 原发布完全一致。沿用公开 efficacy QC / 归一化 / log1p / 2,000 HVG，避免重复处理。 |
| 2026-11-17 01:02 | 严格验证全部拆分数量：训练 572,545、验证 4,825（60 扰动，HepG2）、测试 26,878（380 扰动，HepG2）、对照 39,165。56 batch 标签；固定 HVG 顺序与 X_hvg 一致。ESM2 匹配 3,545 / 3,618 基因，其中 70 个 HVG 需缺失身份回退。 |
| 2026-11-17 01:03 | 4 项公式与梯度测试通过。完整模型约 1.37 亿参数；A100 microbatch=16：eager 约 41 cells/s、29.7 GB，torch.compile 约 73 cells/s、15.6 GB（合成数据初测）。正在扩大 batch 测量；这是纯执行优化，不改变科学设计。 |

## 目录

`paper/` 原文与设计说明；`src/ocoot/` 实现；`scripts/` 下载、处理、训练与评估入口；`configs/` 配置；`data/` 公开数据与缓存；`runs/` 训练产物；`logs/` 执行日志。

## 运行与恢复

```bash
conda activate ocoo-t
python scripts/download_data.py     # 仅 Replogle，断点下载、SHA256 校验和解压
python scripts/prepare_data.py      # 验证固定 HVG / 拆分，生成缓存
python -m pytest -q
python scripts/train.py --config configs/pilot_sigma_1.0.yaml  # 自动从同目录 last.pt 续训
# 确定配置后进行主训练；最终命令与结果在下方进度中记录
```

`paper/implementation_notes.md` 记录原文参数、缺失细节及选择依据。`data/raw/manifest.json` 为数据校验清单，`data/processed/data_report.json` 为拆分与嵌入覆盖检查。`runs/*/events.jsonl` 为逐步日志，`last.pt` 含模型、优化器、双 EMA 与随机状态；`best_*.pt` / `final_ema.pt` 为推理权重。训练会自动更新本文实时状态和关键节点；不要把小实验或验证指标视为最终测试成绩。

- 2026-11-17T01:06:00+08:00：启动/续训 `runs/engine_smoke`；训练细胞 512；有效 batch=512、microbatch=16；计划 2 epoch；噪声标准差=1.0。完整配置 `runs/engine_smoke/config.yaml`。

<!-- LIVE_TRAINING_START -->

实时训练：`runs/pilot_sigma_1.0_t1000`，epoch **4.375**，优化步 **140**，loss **0.60961**，约 **68.2 cells/s**，更新于 2026-11-17T03:21:50+08:00。

<!-- LIVE_TRAINING_END -->

- 2026-11-17T01:06:01+08:00：`runs/engine_smoke` 完成第 1/2 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74119。

- 2026-11-17T01:06:02+08:00：`runs/engine_smoke` 完成第 2/2 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74118。

- 2026-11-17T01:06:02+08:00：`runs/engine_smoke` 训练结束（2.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-17T01:06:50+08:00：启动/续训 `runs/pilot_sigma_1.0`；训练细胞 16,384；有效 batch=512、microbatch=16；计划 12 epoch；噪声标准差=1.0。完整配置 `runs/pilot_sigma_1.0/config.yaml`。

- 2026-11-17T01:11:34+08:00：`runs/pilot_sigma_1.0` 完成第 1/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76344。

- 2026-11-17T01:15:48+08:00：`runs/pilot_sigma_1.0` 完成第 2/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76265。

- 2026-11-17T01:19:53+08:00：`runs/pilot_sigma_1.0` 完成第 3/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76144。

- 2026-11-17T01:24:01+08:00：`runs/pilot_sigma_1.0` 完成第 4/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75924。

- 2026-11-17T01:28:22+08:00：`runs/pilot_sigma_1.0` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75651。

- 2026-11-17T01:32:27+08:00：`runs/pilot_sigma_1.0` 完成第 6/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75355。

- 2026-11-17T01:36:36+08:00：`runs/pilot_sigma_1.0` 完成第 7/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75047。

- 2026-11-17T01:41:01+08:00：`runs/pilot_sigma_1.0` 完成第 8/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74733。

- 2026-11-17T01:45:09+08:00：`runs/pilot_sigma_1.0` 完成第 9/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74414。

- 2026-11-17T01:49:22+08:00：`runs/pilot_sigma_1.0` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.74090。

- 2026-11-17T01:53:43+08:00：`runs/pilot_sigma_1.0` 完成第 11/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73762。

- 2026-11-17T02:04:30+08:00：`runs/pilot_sigma_1.0` 第 12 轮验证生成（每扰动最多 16 细胞，50 步）：PDCorr=-0.0177，MSE=0.07931，PDS_L2=0.5078。这是验证结果，非测试结果。

- 2026-11-17T02:04:33+08:00：`runs/pilot_sigma_1.0` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.73430。

- 2026-11-17T02:04:37+08:00：`runs/pilot_sigma_1.0` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-17T02:04:46+08:00：启动/续训 `runs/pilot_sigma_0.5`；训练细胞 16,384；有效 batch=512、microbatch=16；计划 12 epoch；噪声标准差=0.5。完整配置 `runs/pilot_sigma_0.5/config.yaml`。

- 2026-11-17T02:09:27+08:00：`runs/pilot_sigma_0.5` 完成第 1/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.40655。

- 2026-11-17T02:10:00+08:00：首组短试跑完成 12 轮：验证生成 PDCorr=-0.0177、PDS_L2=0.5078、MSE=0.07931，未学好目标细胞系，不能作为复现成绩。检查基因顺序、flow/Heun 公式及评估路径后，新增 cell embedding 初始化 std=1.0 对照（原文未公开此参数），检验条件信号强度。σ=0.5 组已在训练。

- 2026-11-17T02:14:05+08:00：`runs/pilot_sigma_0.5` 完成第 2/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.40578。

- 2026-11-17T02:18:21+08:00：`runs/pilot_sigma_0.5` 完成第 3/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.40482。

- 2026-11-17T02:22:33+08:00：`runs/pilot_sigma_0.5` 完成第 4/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.40319。

- 2026-11-17T02:26:46+08:00：`runs/pilot_sigma_0.5` 完成第 5/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.40078。

- 2026-11-17T02:30:59+08:00：`runs/pilot_sigma_0.5` 完成第 6/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.39819。

- 2026-11-17T02:35:12+08:00：`runs/pilot_sigma_0.5` 完成第 7/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.39551。

- 2026-11-17T02:39:25+08:00：`runs/pilot_sigma_0.5` 完成第 8/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.39276。

- 2026-11-17T02:43:39+08:00：`runs/pilot_sigma_0.5` 完成第 9/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.38996。

- 2026-11-17T02:47:52+08:00：`runs/pilot_sigma_0.5` 完成第 10/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.38714。

- 2026-11-17T02:52:05+08:00：`runs/pilot_sigma_0.5` 完成第 11/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.38430。

- 2026-11-17T03:02:59+08:00：`runs/pilot_sigma_0.5` 第 12 轮验证生成（每扰动最多 16 细胞，50 步）：PDCorr=0.1062，MSE=0.04081，PDS_L2=0.5181。这是验证结果，非测试结果。

- 2026-11-17T03:03:02+08:00：`runs/pilot_sigma_0.5` 完成第 12/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.38145。

- 2026-11-17T03:03:05+08:00：`runs/pilot_sigma_0.5` 训练结束（12.00 轮），保存最终 EMA；后续采样与评估以独立结果文件为准。

- 2026-11-17T03:03:11+08:00：启动/续训 `runs/pilot_sigma_1.0_t1000`；训练细胞 16,384；有效 batch=512、microbatch=16；计划 12 epoch；噪声标准差=1.0。完整配置 `runs/pilot_sigma_1.0_t1000/config.yaml`。

- 2026-11-17T03:07:45+08:00：`runs/pilot_sigma_1.0_t1000` 完成第 1/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76344。

- 2026-11-17T03:10:16+08:00：核验压缩原始数据 SHA256 后，移除约 22.30 GB 的重复解压副本以预留检查点空间；原始压缩包、训练缓存及所有实验权重均保留。需要完整 H5AD 时运行 `python scripts/download_data.py` 即可本地恢复。

- 2026-11-17T03:11:56+08:00：`runs/pilot_sigma_1.0_t1000` 完成第 2/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76264。

- 2026-11-17T03:15:00+08:00：σ=0.5 的第二组短试跑完成：验证生成 PDCorr=0.1062、PDS_L2=0.5181、MSE=0.04081，仍未达到要求；开始 time_scale=1000 对照。新增原始预测保留/采样恢复测试后，7 项测试通过；准备全量第 1、2 轮的提前条件诊断，避免等待多轮后才发现问题。

- 2026-11-17T03:16:08+08:00：`runs/pilot_sigma_1.0_t1000` 完成第 3/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.76141。

- 2026-11-17T03:20:20+08:00：`runs/pilot_sigma_1.0_t1000` 完成第 4/12 轮，已保存优化器、双 EMA、随机状态检查点；验证 EMA endpoint MSE=0.75921。
