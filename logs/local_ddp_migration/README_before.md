# OCOO-T 复现（Replogle–Nadig）

本项目独立实现[OCOO-T v2](https://arxiv.org/abs/2606.12838v2)，仅使用Replogle–Nadig，未下载Tahoe100M/PBMC。原机A10040GB，独立conda环境ocoo-t。

**当前：按用户要求安全停止，等待迁移。最新完整断点step41374（37.007轮）已核验及冗余归档；复现尚未完成，最终设置与测试均未执行。**

交接必读：[context.md](context.md)、[MIGRATION.md](MIGRATION.md)。迁移必须同时携带项目和外部产物目录，后者含历史权重及已打包环境；不要只复制链接。

<!-- LIVE_TRAINING_START -->

已安全停止：runs/main_v1/last.pt，step **41374**，保存于2026-11-21 22:35:33+08:00。模型、AdamW、双EMA、四类RNG及选型信息均保留；没有finished.json，下一优化步为41375。

<!-- LIVE_TRAINING_END -->

## 断点与迁移

- 主断点：runs/main_v1/last.pt，2,818,999,011 bytes；[CPU完整核验](logs/server_shutdown_20261121/checkpoint_verification.json)通过。
- SHA256：cf418b1615a7ec4c48906f931a21c08a50564f2938816d117b18d1fbedc00df8；[归档证明](runs/main_v1/snapshots/step_41374/preserved.json)。
- 外部目录：/home/dataset-assist-0/whs/ocoo-t-artifacts。已包含ocoo-t-environment.tar（7.92GiB）及SHA文件，迁移可直接解包环境。
- 训练/校准/观察器/等待任务均已结束，GPU无项目进程；[停机事实](logs/server_shutdown_20261121/migration_state.json)。不是训练收敛完成或异常丢失。

迁移校验通过后，在新机以原配置接续。路径/PID绑定、环境解包、链接修复及完整命令见[MIGRATION.md](MIGRATION.md)；旧服务器不要重启任务。

## 数据与设计

| 划分 | 扰动细胞 | 扰动数 | 细胞系 |
| --- | ---: | ---: | --- |
| 训练 | 572,545 | 2,013 | 4 |
| 验证 | 4,825 | 60 | HepG2 |
| 测试 | 26,878 | 380 | HepG2 |

另有39,165个真实对照，HepG2为4,976。公开数据与Arc原件SHA一致；沿用efficacy过滤、库大小归一化/log1p、固定2,000HVG和官方拆分，不重复归一化。保留56个batch及全部官方条件；验证含2个、测试标签含8个全局未见扰动。冻结5120维ESM2，联合3618基因匹配3545，缺失处理/来源见[设计说明](paper/implementation_notes.md)。

12层/768宽/12头连续基因Transformer，RMSNorm、QK-norm、SwiGLU、adaLN-zero，第4层起32条件token；2,000HVG不patch，v-prediction rectified flow。有效batch512（micro16×32累积）、lr1e-4、5轮warmup/cosine/BF16及双EMA；复现默认EMA0.9996、原式双CFG和均匀50步Heun/末步Euler，不能以w1代替普通条件场。

未公开选择noise_std1/time_scale1/cell_init_std1，共享投影、SwiGLU参数匹配2048、Xavier等均已记录并做短对照。100轮上限/20轮最少、原验证点分数10轮耐心及0.002最小增益属复现选择。原best/early_reference=E35，最早可能E45停止；迁移不改变这些参数。环境与版本见[environment.yml](environment.yml)、[锁文件](environment.lock.txt)。

## 已有结果与待办

冻结E35的两组完整验证中，w2/[0,1]已完成4825细胞/B32/seed4242/原Heun50；w2/[.1,.9]保存至**1024/4825**，因本次用户要求中断，待新机接续。原六组粗筛及七项完整记录保留。

首组完整**验证**结果（非测试）：

| 处理 | PDCorr | PDS_L2 | MSE | DEOver | DEPrec | DirAgr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clamp0 | 0.3614 | 0.8125 | 0.01820 | 0.1770 | 0.1012 | 0.7931 |
| 原文未公开的0.05候选 | 0.3591 | 0.8119 | 0.01822 | 0.1963 | 0.1542 | 0.7951 |

0.05改善DE点估计，但DEPrec仍缺原0.18复现检查线，PD及综合略降，未采用；检查线并非论文公开验证阈值。[完整九项与提前诊断结论](runs/main_v1/interim_epoch035_calibration/first_full_floor_diagnostic/decision.json)。新机应完成第二组、原0/.05共四候选检查、提前/正式.05核对及[独立配对审计](runs/main_v1/interim_epoch035_calibration/analysis_plan.json)。

[成熟双EMA两病例检查](runs/main_v1/ema_epoch035_review/plan.json)已登记，4项CPU协议检查及真实193-tensor装载核对通过；尚未做0.9999 GPU生成或采用另一EMA。须等本轮完整审计后，只补固定普通/全区间CFG各960验证细胞，不新增训练或CFG搜索。[E40精确保存/一次固定CFG后续](runs/main_v1/interim_epoch040_fixed_cfg/plan.json)也须按新PID/路径重新登记。

11组短对照、E20 CFG强度/区间、E25完整诊断及.05、成熟batch-retention、CPU批次/有限采样诊断均已完成；没有可靠依据的替代配置未采用。历史原始结果和详细限制见[context.md](context.md)、[设计及尝试](paper/implementation_notes.md)、[完整时间线](logs/progress_history.md)。E25拼接缓存混合B16/B32种子，仅诊断，不冒充正式完整细化。

最终流程仍为原训练→正CFG搜索/两个全量验证→0/.05完整检查→冻结选择→全26878/380测试→九项报告，测试保留同raw的clamp0对照。后续工具已35项CPU检查；尚无正式测试或最终REPORT。主断点每50步/轮末完整原子保存，生成按批保存；恢复后README仍约15分钟汇总。

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

范围：目前优先复现可核对数据与拆分的Table3主基准。Table5的18,080基因面板/引用不一致，[来源审计](paper/long_panel_source_audit.json)已保存；control-context/MMD消融未运行，不以其他数据或补零面板冒充。
