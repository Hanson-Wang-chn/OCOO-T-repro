# Replogle–Nadig 复现结果

正式测试：26,878 个扰动细胞、380 个扰动、2,000 HVG；真实对照与预测端对照一致。

本次采用完整验证锁定的近零阈值 **0.05**：将低于该值的生成表达置0。原文未公开这一处理细节，属于明确记录的复现选择。下面同时保留相同raw样本的原始clamp0对照；未根据测试结果选择阈值。

| 指标 | 原文 Table 3 | 本次 | 95% 区间 |
| --- | ---: | ---: | ---: |
| PDCorr | 0.437 | 0.41224 | [0.37864, 0.44889] |
| MSE | 0.0079 | 0.010097 | [0.0094178, 0.010845] |
| MAE | 0.061 | 0.06727 | [0.065412, 0.069124] |
| PDS_L1 | 0.82 | 0.83589 | [0.81519, 0.85451] |
| PDS_L2 | 0.853 | 0.84951 | [0.82951, 0.86849] |
| PDS_cos | 0.833 | 0.80286 | [0.77568, 0.8275] |
| DEOver | 0.232 | 0.24153 | [0.21398, 0.26948] |
| DEPrec | 0.24 | 0.21641 | [0.19334, 0.23941] |
| DirAgr | 0.766 | 0.82113 | [0.79615, 0.84563] |

同一生成样本的后处理对照：

| 指标 | 原始clamp0 | 验证选定处理 |
| --- | ---: | ---: |
| PDCorr | 0.41318 | 0.41224 |
| MSE | 0.010059 | 0.010097 |
| MAE | 0.067041 | 0.06727 |
| PDS_L1 | 0.83561 | 0.83589 |
| PDS_L2 | 0.84949 | 0.84951 |
| PDS_cos | 0.8031 | 0.80286 |
| DEOver | 0.18655 | 0.24153 |
| DEPrec | 0.11274 | 0.21641 |
| DirAgr | 0.82288 | 0.82113 |

区间来自测试扰动的 2,000 次 bootstrap，反映条件间差异，不代表多次独立训练的误差。

检查点与采样设置：

```json
{
  "checkpoint": "/home/dataset-assist-0/whs/ocoo-t/runs/repro3_main/best_generation.pt",
  "checkpoint_sha256": "854797ba7cc71ae821ab53feeec231dc0f75b67938baf1398aead60267220918",
  "step": 50310,
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
    "gene_proj_init_scale": 1.0,
    "null_condition_init": "zero"
  },
  "noise_std": 0.75,
  "batch_size": 32,
  "steps": 50,
  "guidance": 1.375,
  "cfg_min": 0.0,
  "cfg_max": 1.0,
  "clamp": true,
  "seed": 4242,
  "zero_floor": 0.05,
  "baseline_evaluation_dir": "/home/dataset-assist-0/whs/ocoo-t/runs/repro3_main/test_full_clamp0",
  "postprocess_selection_sha256": "c889b50632e8a84fa332d7c880e9e257cec7f75229f1fdbbd8a1765fe918a87a"
}
```

原文未公开的实现选择与所有尝试见 `paper/implementation_notes.md`；本表由实际 Cell Eval 结果生成，不自动判定复现成功。
