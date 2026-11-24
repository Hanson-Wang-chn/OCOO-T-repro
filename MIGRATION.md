# 迁移与恢复

## 当前主训练分支

第 40 轮退化的修复分支为 `runs/main_ddp_recovery`，已从 step 41374 接续并在 E55 完成，E45 为最佳。旧 `runs/main_v1` 已安全停止并归档。当前断点、诊断和 1–4 卡恢复说明见 [E40 恢复记录](logs/epoch040_recovery/README.md)。早期迁移命令保留作历史。

## 本机运行位置

| 项目 | 路径 |
| --- | --- |
| 项目根目录 | `/home/dataset-assist-0/whs/ocoo-t` |
| 外部权重和备份 | `/home/dataset-assist-0/whs/ocoo-t-artifacts` |
| 用户 Conda | `/home/whs/miniconda3` |
| 训练环境 | `/home/whs/miniconda3/envs/ocoo-t` |

环境包已校验、解包和重定位，editable 安装指向本机仓库。四卡进程、CPU 观察器及续训控制器的日志位于 `logs/local_ddp_migration/`。已启动的任务用 `scripts/status.py --run-dir runs/main_v1` 或查看 `runs/main_v1/training_execution.json` 和 `checkpoint_state.json` 确认。

主训练同机实测 70.51 → 270.81 cells/s（3.84 倍）。每卡 batch 16、累积 8 次，每步处理同一置乱批次的 512 个细胞。rank 0 恢复旧机 RNG，新增 rank 的 RNG 从种子、rank 和起点 step 独立派生。后续四卡续训从完整断点逐 rank 恢复 RNG。单卡接续切换到 DDP 时，浮点规约和随机流会使轨迹出现差异。

## 旧机归档

旧机已安全停止于 **step41374 / 37.007轮**；最新完整断点为 runs/main_v1/last.pt，SHA256：

    cf418b1615a7ec4c48906f931a21c08a50564f2938816d117b18d1fbedc00df8

完整上下文见 [context.md](context.md)。这里只给操作步骤。**不要在旧机重新训练；以下恢复命令在新机执行。**

## 需要复制的材料

1. **整个项目**：/home/whs/ocoo-t/，约45GiB，包含代码、论文、公开数据/处理缓存、最新last.pt、日志/记录与未完成采样缓存。
2. **整个外部产物目录**：/home/dataset-assist-0/whs/ocoo-t-artifacts/，含历史完整断点、推理权重及本次冗余断点。项目有大量链接指向这里，不能漏传。
3. 环境已打包到上述外部目录的 **ocoo-t-environment.tar**，不必再复制8GiB原conda目录。包为未压缩tar，约7.92GiB，适合快速内网传输；SHA：

       a07fbf5aba814ed447216055430ecb272705c8444f83bfac88cbc9e91fda8032

同目录有ocoo-t-environment.tar.sha256；打包记录在logs/server_shutdown_20261121/environment_pack_completed.json。使用conda-pack0.8.1，工具仅安装到/tmp，没有修改训练环境。项目是editable安装，迁移后需要重新指向新目录。

建议新存储至少160GiB可用，容纳整个项目、外部历史、环境包/解包及后续原子检查点临时空间。不要在迁移时删除原文件。如果新机已共享外部挂载，可先核对内容/SHA，再决定是否省略重复复制。

## 从新服务器拉取

将OLD_SERVER换成实际SSH地址。下例允许迁移到新的目录；rsync保留符号链接，随后统一修复。保持原绝对目录也可以，但旧PID守卫仍须重新登记。

~~~bash
OLD_SERVER='whs@旧服务器地址'
PROJECT_ROOT=/home/dataset-assist-0/whs/ocoo-t
ARTIFACT_ROOT=/home/dataset-assist-0/whs/ocoo-t-artifacts
ENV_ROOT="$HOME/miniconda3/envs/ocoo-t"

mkdir -p "$PROJECT_ROOT" "$ARTIFACT_ROOT" "$ENV_ROOT"
rsync -aHS --partial --info=progress2 \
  "$OLD_SERVER:/home/whs/ocoo-t/" "$PROJECT_ROOT/"
rsync -aHS --partial --info=progress2 \
  "$OLD_SERVER:/home/dataset-assist-0/whs/ocoo-t-artifacts/" "$ARTIFACT_ROOT/"
~~~

可以中断后重跑相同rsync。不要加delete，也不必用L去跟随并重复复制所有链接。传完可用相同rsync加checksum再核对一次；脚本还会对最新断点与表达矩阵做SHA校验。

## 恢复环境与链接

源环境是Linux x86_64/glibc2.35、Python3.11、PyTorch2.8.0+cu128。目标需要兼容Linux环境和该PyTorch的NVIDIA驱动，建议支持BF16且显存至少40GB。按[conda-pack官方流程](https://conda.github.io/conda-pack/)在选定的最终目录解包、清理prefix；清理后不要直接再次移动环境。

~~~bash
cd "$ARTIFACT_ROOT"
sha256sum -c ocoo-t-environment.tar.sha256
tar -xf ocoo-t-environment.tar -C "$ENV_ROOT"
"$ENV_ROOT/bin/python" "$ENV_ROOT/bin/conda-unpack"

cd "$PROJECT_ROOT"
"$ENV_ROOT/bin/python" -m pip install --no-deps --no-build-isolation -e .
python3 scripts/relocate_artifact_links.py --artifacts-root "$ARTIFACT_ROOT"
python3 scripts/relocate_artifact_links.py --artifacts-root "$ARTIFACT_ROOT" --apply
sha256sum -c logs/server_shutdown_20261121/critical_small_files.sha256
"$ENV_ROOT/bin/python" scripts/verify_migration.py
~~~

链接工具先只读检查所有目标，缺任一文件则在改动前拒绝；apply后写迁移链接记录，只改符号链接，不改历史实验JSON/权重/数据。已在小型实际文件系统案例中验证dry-run、相对链接及缺失目标拒绝。迁移核验脚本是**CPU检查**，验证最新断点/冗余副本/配置/表达矩阵SHA、完整AdamW/双EMA/RNG及所有链接。此检查针对本次停止的41374，须在继续训练前执行。

可直接调用上述完整 Python 路径。交互式终端可先执行 `source "$HOME/miniconda3/etc/profile.d/conda.sh"`，再执行 `conda activate "$ENV_ROOT"`。依赖版本见原 `environment.yml` 和 `environment.lock.txt`。

再单独确认新GPU可用：

~~~bash
CUDA_VISIBLE_DEVICES=0,1,2,3 "$ENV_ROOT/bin/python" -c \
  "import torch; assert torch.cuda.device_count()==4; print(torch.__version__)"
~~~

## 接续主训练

当需要重启主训练时，在项目根目录执行以下命令。先用进程和训练锁核对当前任务状态。

~~~bash
cd "$PROJECT_ROOT"
mkdir -p logs/local_ddp_migration
nohup setsid env CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=2 \
  TORCHINDUCTOR_COMPILE_THREADS=6 TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/ocoot-inductor" \
  NCCL_SOCKET_IFNAME=lo GLOO_SOCKET_IFNAME=lo PYTHONUNBUFFERED=1 \
  "$ENV_ROOT/bin/python" -m torch.distributed.run --standalone --nproc-per-node=4 \
  scripts/train.py --config configs/main_selected.yaml --preserve-step 44720 \
  >> logs/local_ddp_migration/main_console.log 2>&1 < /dev/null &
echo $! > logs/local_ddp_migration/torchrun.pid
~~~

程序从 `last.pt` 恢复最新完整状态。初次新机续训的首步是 41375。查看主控制台、`runs/main_v1/events.jsonl` 和 `checkpoint_state.json` 确认进度及落盘。step 44720 的生成前完整断点会在训练进程内固定，再独立归档到外部目录。需要安全停止时，核对 `training_execution.json` 中的 rank 0 PID 及 start tick，给该进程发 SIGTERM，并等待全局步完成和检查点落盘。

CPU观察器可在另一个终端启动：

~~~bash
cd "$PROJECT_ROOT"
nohup setsid env CUDA_VISIBLE_DEVICES='' "$ENV_ROOT/bin/python" scripts/watch_training.py \
  --run-dir runs/main_v1 --from-epoch 10 \
  >> logs/local_ddp_migration/observer_console.log 2>&1 < /dev/null &
~~~

用户仍要求README约每15分钟更新，日志和检查点频率不变。

## 接续未完成的验证

`scripts/resume_local_reviews.py` 根据实际四卡 worker 的 PID/start tick 注册迁移计划。它保留原始计划与缓存，核验后复制原有生成数组和 marker，再重定位工作副本。E40 完整断点在训练期间归档。GPU 3 共享实测使主训练降至约 116 cells/s，续训控制器默认等待主训练完成，再依次执行 E35 四候选、成熟双 EMA 和 E40 匹配复核。最终完整验证和原有质量检查决定能否进入测试。

~~~bash
nohup setsid env OMP_NUM_THREADS=2 TORCHINDUCTOR_COMPILE_THREADS=6 \
  TORCHINDUCTOR_CACHE_DIR="$HOME/.cache/ocoot-inductor" PYTHONUNBUFFERED=1 \
  "$ENV_ROOT/bin/python" scripts/resume_local_reviews.py \
  --artifacts-root "$ARTIFACT_ROOT" --aux-gpu 3 \
  >> logs/local_ddp_migration/reviews_console.log 2>&1 < /dev/null &
~~~

上述流程须在首个新完整断点落盘后启动。E35 已验证从 1024 接续至 1056，并保留在迁移副本中。若主训练重启，应根据新 worker 身份重绑迁移计划，保留已完成归档和缓存证据。

### 旧机交接细节

**不要直接启动旧硬编码PID的watch_main_training、E35控制器或E40跟随器。** context.md包含源计划、SHA、旧PID、结果及具体待办。

- E35第二项完整验证已有**1024/4825**缓存。原控制器是用户主动interrupted；保留raw/indices/marker以及7项已完成记录。
- 路径变动时，calibration_config.json及计划中的旧绝对路径会触发严格拒绝。保留原始记录，给迁移任务建新副本/运行计划并登记旧→新路径、Python前缀、新PID/start tick及依赖SHA；只重绑定运行身份/路径，证明所有checkpoint/data/数组及采样选项相同。
- 完成2组full后做原0/.05共4候选检查，核对提前.05与正式重算，执行已登记独立审计、看图并写结论。
- 再执行已经登记但尚未做GPU生成的成熟双EMA两病例检查；重新绑定E40精确归档及固定CFG后续。
- 新机GPU可见性取实际主训练环境，不能沿用CPU预检查的空CUDA变量。
- 主训练原配置/优化器/RNG/种子/有效batch/Heun50/质量标准保持；最终测试尚未运行，不能以迁移为由跳过验证或用测试调参。

后续需要接上全流程父进程时，可运行：

~~~bash
"$ENV_ROOT/bin/python" scripts/run_experiment.py \
  --config configs/main_selected.yaml \
  --postprocess-plan logs/postprocess_workflow_update/plan.json
~~~

它会附加等待现有训练，不应重复训练；应在新Codex核实未完成诊断与运行身份绑定后再接上。

可直接给新Codex：

> 阅读context.md和MIGRATION.md，先验证迁移，再从step41374完整断点恢复原配置训练。保留历史来源并登记新PID/路径映射，接续E35剩余全量验证、四候选/提前.05核对/独立审计，再做已登记双EMA与E40后续。README维持15分钟更新，测试未运行，不要从零训练、放宽检查或用测试调参；尽量自主完成原复现目标。
