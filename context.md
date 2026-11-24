# OCOO-T 复现交接上下文

## 2026-11-22 E40 退化排查与重训

- 旧四卡运行的 EMA 自 step 42000 附近逐渐退化。E40 去噪 endpoint MSE 原始权重 0.10438、EMA0.9996 为 0.19701，两组冻结复算均重现。E35→E40 的 matched ordinary 生成 MSE 从 0.00943 升至 0.09878。
- 同一数据、噪声和条件 mask 的两步回放定位到 Dynamo DDPOptimizer 分图。修复为多卡编译时作用域内设置 `optimize_ddp=False`，保留编译后的核心模型及原生 DDP。1/2/3/4 卡回放和旧/新对照见 [排查记录](logs/epoch040_recovery/README.md)。
- 已安全停止旧 `runs/main_v1` 于 45170 并归档。现用 `configs/main_ddp_recovery.yaml` / `runs/main_ddp_recovery` 从 SHA 已核验的 41374 完整状态重跑。模型、数据、BF16、优化器、双 EMA、早停及推理参数保持。EMA endpoint MSE 相对原始权重超过 4% 时保存并暂停，便于及时干预。
- 修复版 E40 精确 step 44720 和原故障时首次超过 4% 的 step 42250 预设保存；与原 E35 的普通条件验证可在不同 run_dir 下做匹配对照。旧 E40 和 E35 的历史缓存单独保留。
- 修复版已重跑并归档 E40。匹配验证 1916 个细胞的 MSE=0.009512、PDCorr=0.300903。旧 E40 分别为 0.098782、0.188192。E40 按原点估计规则更新早停参考，最早可能 E50 停止。完整九项、配对区间和来源均在 [恢复记录](logs/epoch040_recovery/README.md)。
- E45 后续周期验证 MSE=0.009429、PDCorr=0.304331，按原规则成为最佳及早停参考。E50 配对综合分相对 E45 差 −0.02000，去噪仍在正常范围。两轮完整断点已归档，最早可能 E55 停止，持续进度以 [监控状态](logs/epoch040_recovery/monitor_state.json) 为准。
- 11-23 02:00 已按原规则在 E55（step 61490）结束训练，最佳为 E45。末步完整断点、E55 最终 EMA 和 E45 最佳 EMA 均已读回核验。[训练完成记录](runs/main_ddp_recovery/finished.json) · [断点核验](logs/epoch040_recovery/final_training_checkpoint_verification.json)
- 11-23 04:00 已完成两组全量验证及四个后处理候选的检查。`w=2`、`[0,1]`、0.05 截断的 DEPrec 为 0.15863，低于复现预登记 0.18，其余指标通过。正式测试未开始。已登记四项针对性引导对照，沿用同一 E45 和固定验证采样。 [检查记录](runs/main_ddp_recovery/calibration/postprocessing/review.json) · [独立重抽样](logs/epoch040_recovery/e45_full_calibration_independent_audit.json) · [对照计划](runs/main_ddp_recovery/postgate_guidance_study/plan.json)
- 11-23 04:27 引导对照完成。`w=2`、`[0,0.9]` 的 DEPrec 比同批全区间提高 0.00146，区间跨零，低于预设 0.008。四组均未登记新全量候选。已按同 E45、同 960 细胞和同噪声启动 0.9999 的普通条件和全区间配对诊断。[引导对照报告](runs/main_ddp_recovery/postgate_guidance_study/report.json) · [双 EMA 对照计划](runs/main_ddp_recovery/postgate_ema_study/plan.json)
- 11-23 04:56 双 EMA 对照完成。在全区间 `w=2` 和 0.05 截断下，0.9999 的 DEPrec 比 0.9996 下降 0.03082（配对区间 −0.06477 至 −0.00950），保留论文默认 EMA 0.9996。修复版最佳 E45，复现验证仍缺 DEPrec，正式测试未启动，四张卡已释放。[双 EMA 报告](runs/main_ddp_recovery/postgate_ema_study/report.json) · [完整验证结果](runs/main_ddp_recovery/calibration/postprocessing/review.json)

## 2026-11-22 本机续训

- 新项目目录 `/home/dataset-assist-0/whs/ocoo-t`，外部产物 `/home/dataset-assist-0/whs/ocoo-t-artifacts`，Conda 环境 `/home/whs/miniconda3/envs/ocoo-t`。本机已核验 41374 完整断点和冗余副本、表达矩阵、环境包及 73 个符号链接。
- 主训练使用四卡 DDP，`configs/main_selected.yaml` 与旧断点一致。第一新步 41375。每卡 microbatch 16 × 累积 8 × 四卡，维持 512 全局批次和每轮 1118 步。单卡→四卡时保留 rank 0 原随机流，新增 rank 的流按原种子、step、rank 分开派生。step 41450 的完整四卡断点已通过独立 CPU 核验。[迁移流程](MIGRATION.md) · [四卡断点核验](logs/local_ddp_migration/resumed_checkpoint_verification.json)
- 同机独占训练短基准 70.51 → 270.81 cells/s（3.84 倍），含优化器和 EMA 更新，计时不含编译、检查点及验证。进程和实时日志见 `runs/main_v1/training_execution.json` 与 `logs/local_ddp_migration/`。
- E35 校准在 `interim_epoch035_calibration_local_ddp` 从 1024 接续至 1056。与 DDP 共用 GPU 3 时主训练降至约 116 cells/s，已在批次落盘后暂停辅助 GPU 进程。CPU 观察器及队列保持运行，E40 step 44720 到达时归档完整状态，剩余验证依序排在主训练完成后、正式测试前。[重绑定记录](logs/local_ddp_migration/review_registration.json) · [队列状态](logs/local_ddp_migration/reviews_state.json)
- 复核论文 v2 附录 A 原文，0.9996 是论文明确指定的默认推理 EMA。0.9999 的小规模配对仍是额外诊断。

## 2026-11-21 旧机停机记录

更新：2026-11-21，北京时间。**旧服务器已按用户要求安全停止，不要在这里自动重启。复现尚未完成，没有测试集成绩，不能宣称达到论文水平。**

## 用户要求与本次停止

原目标是在当前目录独立复现论文 OCOO-T: A Simple and Scalable Virtual Cell Model for Transcriptional Perturbation Response Prediction。没有找到作者实现，须仔细照论文实现数据、模型、训练和评估。只跑 Replogle–Nadig，因空间不下载 Tahoe100M/PBMC。独立 conda 环境名 ocoo-t；本地GPU；未公开细节要记录、先做小实验，质量不足则诊断、合理优化，不为赶时间缩减论文明确参数。中文 README **约15分钟更新一次**，日志/完整断点/生成批次仍频繁落盘。尽可能自主处理，不反复询问用户。

最新要求因旧服务器到期：保存最新完整断点、结束任务进程、交接上下文、说明新服务器Codex恢复方法。本次22:35:25先停接续父进程/观察器/等待器及辅助校准，再给主训练SIGTERM。主训练完成当前全局步，**22:35:33原子保存step41374后退出**，未强杀训练，未造训练完成标记。

任务进程656512、1100222、785577、1251199、1251202、1251717、1251718已退出。旧PID仅是历史，**不要在新机按旧号码杀进程或认定任务仍在运行**。旧锁文件/PID文件可能仍有文本，但OS锁已释放；不要删除缓存/证明来“修复”它们。停止证据：logs/server_shutdown_20261121/。

## 最重要的文件与断点

| 项目 | 路径或值 |
|---|---|
| 原项目 | /home/whs/ocoo-t |
| 最新完整断点 | runs/main_v1/last.pt，**2,818,999,011 bytes** |
| 最新全局步 | **41374**；每轮1118步，进度**37.007155635轮** |
| 最新SHA256 | **cf418b1615a7ec4c48906f931a21c08a50564f2938816d117b18d1fbedc00df8** |
| 同字节冗余备份 | /home/dataset-assist-0/whs/ocoo-t-artifacts/main-snapshots/shutdown_step041374/full.pt |
| 备份链接与证明 | runs/main_v1/snapshots/step_41374/full.pt、preserved.json |
| 原主配置 | configs/main_selected.yaml |
| 配置SHA256 | e51d2cc9b8bb4ddc8e9dda24861f675b6a37a2486b7faffc69ed71e7f2342b52 |
| 原conda环境 | /home/dataset-assist-0/whs/conda-env/ocoo-t |
| 外部历史产物 | /home/dataset-assist-0/whs/ocoo-t-artifacts，**必须复制或确认新机共享可用** |
| 停机事实 | logs/server_shutdown_20261121/migration_state.json |
| 迁移操作 | MIGRATION.md |

**继续优化必须用完整last.pt，不能用best_generation.pt/final_ema.pt等仅推理文件替代。** 完整状态包含模型、189个AdamW参数状态、双EMA、Python/NumPy/Torch CPU/CUDA RNG、step、最佳分数/早停/待评估信息。

本次pending_generation_epoch=None，best_generation_epoch和early_reference_epoch均为E35，综合分1.016999948852937。原规则下最早可能E45停止，后续改善会顺延。E25/E30/E35旧归档是生成前快照，含pending和旧选型信息，不能当成本次最新已完成评估后的恢复状态。E36及本次41374是生成完成后的状态。

项目有许多绝对符号链接，实际权重在外部目录；只复制项目会漏文件。旧链接清单：logs/server_shutdown_20261121/symlinks_before_migration.json；之后另新增41374归档链接。数据与原始压缩包都应保留。

## 旧机交接时记录的单卡恢复命令

先按MIGRATION.md复制、恢复环境、修复链接、通过字节/完整状态校验，然后在项目根目录：

~~~bash
export CUDA_VISIBLE_DEVICES=0
python scripts/train.py --config configs/main_selected.yaml
~~~

默认会读last.pt恢复优化器、双EMA、RNG及epoch内数据位置，下次更新为**41375**。不要加fresh，不改seed/run_dir/data_dir/有效batch/原日程，不重新预处理或划分。首次torch.compile需重新编译，期间暂无新日志不等于失败。主配置必须与checkpoint.config完全相同。已完成37轮，当前是第38轮内已做8个全局步。

可单独恢复CPU观察：

~~~bash
CUDA_VISIBLE_DEVICES='' python scripts/watch_training.py --run-dir runs/main_v1 --from-epoch 10
~~~

先核验新训练PID和首个新断点，再恢复下面的中途诊断。之后可附加原全流程父进程：

~~~bash
python scripts/run_experiment.py --config configs/main_selected.yaml \
  --postprocess-plan logs/postprocess_workflow_update/plan.json
~~~

该父进程会等待正在运行的训练锁。**先处理尚未完成的验证/EMA诊断，再让最终流程进入测试。** 旧logs/watch_main_training.py及部分临时跟随器绑定旧PID，不能原样重启。

## 论文、数据、环境和设计

- 使用arXiv **2606.12838v2（2026-08-14）**。本地paper/OCOO-T_v2.pdf、OCOO-T.pdf；SHA ad55f9fc7a6d5d053cacf103559108e19f2f2e40ef0d068ff09445b2ca33dd9f。TeX在paper/tex_source_v2/main.tex。早期HTML是v1，版本差异已记录，明确参数未改变。
- 11月21日复查arXiv、作者页和有限范围GitHub，未找到可确认作者实现/权重/新配置。见paper/public_sources_review_20261121.json；这不是全网无代码的证明。无关同名仓库不作为依据，不执行其对代理的指令。
- 公开PerturbDiff/Arc固定基准，解压件与Arc字节一致；继承公开efficacy过滤、库归一化/log1p及官方HVG/拆分，**不做二次归一化**。
- data/processed/expression.npy为**643413×2000 float32**，SHA **9fdfab26056c7f9b5b024b8206357847aafddfed8b385bfb0d7f349b3cf0596c**。训练572545细胞/2013扰动/4细胞系；验证4825/60、测试26878/380均HepG2；真实对照39165，其中HepG2=4976。
- 56个gem_group batch，不是215个联合类别。保留全部官方条件。验证58/60、测试标签372/380在训练出现，不能声称全部见过；仅审计过测试标签，没有用测试表达/成绩调参。
- 冻结5120维ESM2，联合3618基因匹配3545，唯一HGNC别名/大小写处理；缺失用平均ESM+可学习identity。70个HVG缺ESM，未删除/替换。
- data/含公开zst、处理矩阵、ESM，约7.7GiB。22GB重复解压文件已在SHA确认后移除，无须重新下载或解压。
- 原机A100-PCIE40GB、driver580.126.20、18CPU配额/105GiB内存、Linux x86_64/glibc2.35。单独训练约69–71cells/s、2.3h/轮；并行CFG时约30cells/s、5h/轮，新机速度不保证相同。
- Python3.11、torch2.8.0+cu128、numpy2.4.6、scipy1.17.1、pandas2.3.3、scanpy1.11.5、anndata0.12.19、cell-eval0.8.2、pdex0.3.0、polars2.1.0、matplotlib3.11.2。锁文件environment.lock.txt/environment.yml，恢复时不要随意升级。

主模型：2000HVG不patch，12层/768宽/12头64维，RMSNorm/QK-RMSNorm、SwiGLU、adaLN-zero，第4层起32条件token，1D sincos；连续值两层MLP，冻结ESM线性投影，时间/扰动/细胞/batch条件。

RF路径z=t*x+(1−t)*eps，目标v=x−eps，v-pred/v-loss；t=sigmoid(N(−.8,.8²))。AdamW(.9,.95)、wd0、有效batch512、lr1e−4、5轮warmup/cosine/BF16。双EMA .9999/.9996每次全局更新，论文附录 A 指定默认推理用 .9996。50均匀积分区间/51节点，Heun末区间Euler，**99次场求值**。

双CFG严格为：vu+w/2*(vc−vu)+w/2*(vp−vu)+w*(vpc−vu)。**w=1不等于普通场**，普通全条件用guidance=None，不能替换为常规单条件公式。p/c独立drop.1。

未公开选择详见paper/implementation_notes.md：noise_std1、time_scale1、cell_init_std1、batch_init_std.02，共享gene/pert投影；SwiGLU按JiT参数匹配ratio4→2048（literal3072可选但未采用），值MLP Xavier、cell+batch合后drop。100轮上限/20轮最少/耐心10轮/min_delta.002属于复现选择，非论文参数。

实际micro16×累积32，grad_checkpoint=False，torch.compile；每轮1118更新/drop_last129，数据顺序seed42+10007*epoch。每50步及轮末完整原子保存，SIGTERM完成当前全局步后保存。每250步/轮末固定1024验证去噪仅作健康检查；每5轮EMA普通场cap32/B32生成1916细胞用于原选型/早停。

## 待办一：接续冻结E35完整验证

目录runs/main_v1/interim_epoch035_calibration/。固定step39130，推理文件runs/main_v1/validation_epoch_035/model.pt SHA **c51e91a0ce8fb9495b650788b2694f50b7cb036eb4a0b9baaee8926db43216bd**。全部193tensor与完整E35的EMA.9996+4buffers精确一致，证明在interim_epoch035_fixed_cfg/checkpoint_verification.json。

六组cap16/B32/960粗筛已完成，原主分PD+meanPDS选出两个正CFG设置各做全4825验证：

1. refine_00=w2/[0,1]，**4825/4825已完成且九指标齐全**。
2. refine_01=w2/[.1,.9]，**停机时1024/4825已逐批保存**，B32/seed4242/原Heun50。

calibration/all_results.json有7条完整记录。生成按marker恢复；保留raw.npy、indices、generation_state.json；未进入marker的尾部不是有效缓存，续跑会覆盖。不要改seed/B32、混入B16或将部分标全量。process.json显示interrupted是本次用户停机，不是数值故障。

控制器logs/run_epoch035_validation_calibration.py及plan.json绑定旧trainer PID/start tick和Python绝对路径。新机需显式注册运行身份/路径迁移，旧计划/归档/SHA保留，不能直接把旧PID当新进程。原计划SHA f24dcdff987abd0ef788556da569b436cc6ae76a5c3d0ce889494e211263d6cc。

完成第二组后，原select_validation_postprocess.py对两个raw各检查0/.05，共4候选。无合格候选exit2是需复核分支，不要改门槛或造completed；若合格按原主分选择并验证promotion。**此中途流程不执行测试。**

已预登记独立四候选审计：analysis_plan.json SHA **337489792d00d5a3b0a5b78f11cc1065fe949034ebd03681d612456484be0f30**，脚本scripts/analyze_full_calibration.py，3项CPU数学检查通过。对4候选、全部6配对比较保留九指标/主分2000次seed9232区间，DirAgr按各候选自己的NaN语义。完成后看PNG/PDF并写该校准目录独立decision.json，含test_access:false。不要改已被其他计划钉住的interim_epoch035_fixed_cfg/decision.json。

首组全量的**验证**结果如下，非论文测试表格：

| 处理 | PD | PDS L1/L2/cos | MSE | MAE | Over | Prec | Dir |
|---|---:|---|---:|---:|---:|---:|---:|
| clamp0 | .361391 | .800000/.812500/.850556 | .01819736 | .08195292 | .177020 | .101203 | .793105 |
| floor.05 | .359126 | .802222/.811944/.849722 | .01822327 | .08213744 | .196269 | .154194 | .795141 |

21:47已按预登记提前独立CPU检查.05，exit0/SHA通过，目录first_full_floor_diagnostic/。clamp0缺Over/Prec，.05仍缺Prec，均未采用。零比例16.51%→53.44%，DE点增益不等于整体修复；配对区间待4候选完整审计。**正式selector重算同一.05后与提前结果逐项核对。**

原工程检查：PD≥.35、meanPDS≥.75、各PDS≥.70、MSE≤.02、MAE≤.09、Over≥.18、Prec≥.18、Dir≥.70。**不是论文公开的验证阈值，也不是达到论文水平的证明。** 未因结果不理想放宽。

## 待办二：已登记的成熟双EMA检查

runs/main_v1/ema_epoch035_review/plan.json，SHA **cdc72b7b4b4d9427e218b2f6aed41d9083d3dc58bf7f52fb0b72b93a5ec5fd50**；脚本scripts/review_ema_weights.py。21:56登记，4项协议CPU检查1.48s通过，真实两份193tensor装载CPU核对通过。**没有.9999生成或成绩，没有新增训练。**

当前两个full、4候选、独立审计及根结论全部完成后，且只有主训练占GPU，才启动。只补两项960/B32/seed4242/Heun50/clamp0的EMA.9999验证：普通场和w2/[0,1]；复用同E35的screen_00、screen_full_interval之.9996参考（B32，不能混用旧B16固定CFG缓存）。不搜索CFG/阈值，所有9指标/配对区间保留，不自动改变默认EMA、主训练或最终协议。

原入口（须先重新绑定新PID/路径及前序计划SHA）：

~~~bash
python scripts/review_ema_weights.py --plan runs/main_v1/ema_epoch035_review/plan.json
~~~

check-only只做CPU准备。**实际GPU父进程不能继承CPU检查的CUDA_VISIBLE_DEVICES空值**，要匹配新主训练真实可见性。原机未设此变量，新机重新读取。完整E35归档SHA 5bbdfe093771ea2f38deffa7a7ad6d633cd27d3f1a33ff00458abd2d79aafb7a；本次仅推理，不用其pending状态长续训。

固定CFG主分有可信改善且MSE/MAE无明确退化才值得另登记全量验证；弱或混合证据保持默认。2个有限病例不证明某EMA全局最优。

## E40与最终流程

- E40目标step44720。旧保存器/跟随器已停，未发生E40实验。计划interim_epoch040_fixed_cfg/plan.json，脚本logs/preserve_main_epoch040_when_ready.py和follow_epoch040_fixed_cfg.py绑定旧PID，须新机登记。精确保留E40完整状态，等NoCFG周期/CPU九项齐全，再只补一次匹配固定CFG，与E35联合复核。
- 训练完成后：原正CFG搜索→2个完整4825细化→0/.05完整检查→冻结选择→全26878/380测试→9指标/2000区间/报告。同raw保留clamp0对照，正floor标为未公开选择。
- 后续计划logs/postprocess_workflow_update/plan.json SHA **87445b9d673b299bef461ec5cc6ce78a37d3cb64b94ab4b46ed672fe89e79371**，35项CPU检查通过，实际旧全量.05九项重算差0。严格同checkpoint/协议的完整缓存才可复用；拒绝限量及混合RNG。
- 没有finished.json、正式test_protocol、test_full*或最终results/REPORT.md；不能用测试调参或直接宣称成功。完整检查不合格先继续诊断/合理优化。

## 历史结论与证据索引

- 已完成8组16,384训练细胞/384更新/seed42的noise×time×cell-init网格，再做独立投影/x预测/fan-in，共11组短诊断。保留当前共享/v/Xavier；fan-in点主分略高但CI跨0，误差/Prec更差。不要宣称当前在全部11项点分最高。短试跑raw不代替成熟EMA。runs/pilot_grid_complete/summary.json、runs/supplemental_trials_complete.json。
- E20更大CFG2.5/3无稳定收益、误差增加；严格保留原双CFG与50步。interim_epoch020_all_scales/、interim_epoch020_interval_review/。
- E25全4825固定CFG前960 B16/seed4242，后3865 B32/seed62525，**混合RNG仅诊断，不可冒充正式refinement**。clamp PD.330621/Prec.098926；.05 PD.328796/Prec.158591，Prec增.059665 CI[.02320,.09276]仍未过检查。interim_epoch025_full_guided/decision.json。
- E25→26 batch-retention成熟单因素1118更新，从同θ/AdamW/EMA/RNG出发，只改drop-batch。主分CI跨0，MSE/MAE增加、方向下降，**未采用、不追加该预算**。runs/batch_dropout_mature_trial/decision.json。原初始RNG核验不等于编译训练逐样本流已审计。
- CPU批次结构不支持简单漏学/均值补丁，扰动留出误差R²五折均负；描述性分解不是因果归因。batch_structure_epoch025/decision.json。
- 新CPU MC诊断mc_variance_epoch025/decision.json：约50.1%细胞有重复batch×生成段层，单例条件方差不可识别；额外同方差假设下proxy约原MSE16.1%，不是确定下界/因果占比/校正成绩；未改采样数或原成绩。4项数学检查通过。
- **CellEval MSE用全部2000基因；目标基因排除是PDS等特定指标选项，不套到MSE。** 既有调用正确。旧0.7.2时代9核心与0.8.2已核对；保持pdex epsilon0/无额外CPM过滤。两侧共享4976真实对照；cap只限预测，真实验证总为4825。
- E30→35普通1916/B32主分差+.009552 CI[−.010146,.032354]；固定960/B16差+.011156 CI[−.007623,.031301]。均跨0；固定MSE/MAE差CI为正。九项/主分独立核算差0，点排序偏E35但不是稳定增益。E35普通PD.296074/PDS_L2.711944/MSE.00943374/Prec.078446；固定PD.293114/PDS_L2.796667/MSE.02311776/Prec.087518。两协议绝对分不可直接比。
- late-time探针显示原分布很少覆盖t>.98，FP32对该探针只改善约.14%；保留原BF16/时间分布/Heun50，不把探针当已证明Bayes风险或代码错误。
- 论文Table3测试目标PD.437、Over.232、Prec.240、Dir.766、PDS.820/.853/.833、MSE.00790、MAE.061；当前验证不能直接宣称复现。没有VAE/OT/MMD或测试拟合修正。
- 18080基因长面板公开引用与Replogle不一致，已只读Range核查官方merged/四单文件，无对应raw/layers；不拿别的数据/补零冒充。优先Table3；control-context/MMD消融未跑。paper/long_panel_source_audit.json。

## 新机路径、PID与来源证据

主训练配置data_dir/run_dir相对，完整断点可在新目录恢复；中途校准JSON/计划/子命令却有旧绝对路径、PID和相互SHA绑定。

1. 优先保留原绝对项目/外部目录逻辑路径；**即使路径相同仍须重新绑定PID/start tick**。旧守卫不能直接运行。
2. 若路径不同，先用scripts/relocate_artifact_links.py修复链接并恢复主训练；历史JSON/SHA原样保留。给中途任务建立迁移副本/新计划，记录旧→新路径、Python前缀、PID、来源SHA。
3. 只重定位迁移副本的元数据。checkpoint/raw/indices/marker options、B32/seed、CFG、训练/质量参数不变。calibrate.py严格比较calibration_config.json，不能原地覆盖旧JSON绕过不匹配。1024已保存批次要复用，不能标全量。
4. 同步注册新计划之间依赖，包括独立审计、EMA前序分析计划SHA。旧计划、归档和结论留作证据。迁移重绑定不是科学超参数变化。
5. 新PID及CUDA可见性应从实际新进程读取。不同GPU/编译器不保证浮点逐位相同；先查loss、断点、来源和显存，不因此随意改论文参数。

完整设计/未知项/时间线仍在README.md、paper/implementation_notes.md、logs/progress_history.md，全部一起迁移。本文件的索引不是删除那些证据的理由。新迁移工具只查字节/完整状态/链接，不启动GPU。

## 可直接交给新Codex的说明

> 请先读context.md和MIGRATION.md，核验迁移材料与最新完整断点41374。旧服务器已按用户要求停止，不要从零开始或用仅EMA文件续优化。恢复原配置主训练，README保持约15分钟更新；登记新PID/路径映射、保留旧证据，接续E35第二个full的1024/4825缓存，完成原四候选、提前.05逐项核对及独立审计，再按登记检查成熟双EMA。重新绑定E40精确保存/固定CFG跟随计划，继续原论文复现目标。测试尚未运行，不用测试调参，不宣称已达到论文水平。尽可能自主完成，无需反复询问。


## 已完成的停机核验补充

最新完整断点已通过scripts/verify_migration.py独立CPU核验：193个raw tensor、两份189参数EMA、全部189个AdamW slots均有限且step41374，四类RNG/原配置/选型元数据与表达矩阵SHA均保留；冗余归档SHA相等、全部73个符号链接可解析，没有初始化GPU。这不是另起GPU恢复训练测试。报告：logs/server_shutdown_20261121/checkpoint_verification.json。

环境包已完成：外部产物目录下ocoo-t-environment.tar，8,506,265,600 bytes，SHA256 a07fbf5aba814ed447216055430ecb272705c8444f83bfac88cbc9e91fda8032；同目录有sha256校验文件。解包后运行conda-unpack，并用新环境Python在项目目录执行pip install --no-deps --no-build-isolation -e .以修正editable源码路径。具体可复制命令见MIGRATION.md。

附加完整性：迁移后在项目根目录运行sha256sum -c logs/server_shutdown_20261121/critical_small_files.sha256，核对metadata/vocab/embeddings、主配置和关键源码，再执行完整CPU迁移核验。最终进程审计在logs/server_shutdown_20261121/final_process_audit.json：旧任务PID均已退出，NVIDIA计算进程为空，未操作无关系统/用户服务。迁移体积实测项目44GiB、外部目录66GiB（包含环境包），不要漏掉后者。
