# Replogle–Nadig 复现结果

正式测试：26,878 个扰动细胞、380 个扰动、2,000 HVG；真实对照与预测端对照一致。

完整验证低于项目检查线的指标为 DEPrec。测试配置按登记计划在完整验证后冻结。

本次采用完整验证锁定的近零阈值 **0.05**：将低于该值的生成表达置0。原文未公开这一处理细节，属于明确记录的复现选择。下面同时保留相同raw样本的原始clamp0对照；未根据测试结果选择阈值。

| 指标 | 原文 Table 3 | 本次 | 95% 区间 |
| --- | ---: | ---: | ---: |
| PDCorr | 0.437 | 0.40058 | [0.36672, 0.43773] |
| MSE | 0.0079 | 0.011066 | [0.010304, 0.011904] |
| MAE | 0.061 | 0.06949 | [0.067445, 0.071477] |
| PDS_L1 | 0.82 | 0.83136 | [0.80943, 0.851] |
| PDS_L2 | 0.853 | 0.84435 | [0.8242, 0.86359] |
| PDS_cos | 0.833 | 0.8052 | [0.77865, 0.83026] |
| DEOver | 0.232 | 0.23892 | [0.21116, 0.26779] |
| DEPrec | 0.24 | 0.207 | [0.18368, 0.23045] |
| DirAgr | 0.766 | 0.8116 | [0.78452, 0.83761] |

同一生成样本的后处理对照：

| 指标 | 原始clamp0 | 验证选定处理 |
| --- | ---: | ---: |
| PDCorr | 0.40321 | 0.40058 |
| MSE | 0.010994 | 0.011066 |
| MAE | 0.069104 | 0.06949 |
| PDS_L1 | 0.83073 | 0.83136 |
| PDS_L2 | 0.84415 | 0.84435 |
| PDS_cos | 0.80551 | 0.8052 |
| DEOver | 0.18642 | 0.23892 |
| DEPrec | 0.10347 | 0.207 |
| DirAgr | 0.80658 | 0.8116 |

区间来自测试扰动的 2,000 次 bootstrap，反映条件间差异，不代表多次独立训练的误差。

检查点与采样设置：

```json
{
  "checkpoint": "/home/dataset-assist-0/whs/ocoo-t/runs/repro2_main/best_generation.pt",
  "checkpoint_sha256": "88180a502e5b08d6ba1d6ba2a03caeb7081fff82d216f5ac6c9d00d012b92d09",
  "step": 39130,
  "weights": "ema_0.9996",
  "split": "test",
  "cap": null,
  "de": true,
  "model": {
    "dim": 768,
    "depth": 12,
    "heads": 12,
    "mlp_ratio": 4.0,
    "swiglu_convention": "param_matched",
    "context_tokens": 32,
    "context_start": 3,
    "drop_prob": 0.1,
    "value_hidden": 768,
    "time_dim": 256,
    "time_scale": 1.0,
    "pos_encoding": true,
    "batch_condition": true,
    "cell_init_std": 1.0,
    "batch_init_std": 0.02,
    "grad_checkpoint": false,
    "position_scale": 1.0,
    "gene_proj_init_scale": 1.0
  },
  "noise_std": 1.0,
  "batch_size": 32,
  "steps": 50,
  "guidance": 1.5,
  "cfg_min": 0.0,
  "cfg_max": 1.0,
  "clamp": true,
  "seed": 4242,
  "zero_floor": 0.05,
  "baseline_evaluation_dir": "/home/dataset-assist-0/whs/ocoo-t/runs/repro2_main/test_full_clamp0",
  "postprocess_selection_sha256": "5a864f552a3f7ee34d118598a9ce2c165502ebcdcdece6f2695c115d2248a32d"
}
```

原文未公开的实现选择与所有尝试见 `paper/implementation_notes.md`；本表由实际 Cell Eval 结果生成，不自动判定复现成功。
