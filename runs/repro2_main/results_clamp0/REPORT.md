# Replogle–Nadig 复现结果

正式测试：26,878 个扰动细胞、380 个扰动、2,000 HVG；真实对照与预测端对照一致。

| 指标 | 原文 Table 3 | 本次 | 95% 区间 |
| --- | ---: | ---: | ---: |
| PDCorr | 0.437 | 0.40321 | [0.36946, 0.44056] |
| MSE | 0.0079 | 0.010994 | [0.010235, 0.011826] |
| MAE | 0.061 | 0.069104 | [0.067082, 0.071099] |
| PDS_L1 | 0.82 | 0.83073 | [0.80871, 0.85032] |
| PDS_L2 | 0.853 | 0.84415 | [0.82385, 0.86347] |
| PDS_cos | 0.833 | 0.80551 | [0.77924, 0.83023] |
| DEOver | 0.232 | 0.18642 | [0.16261, 0.21102] |
| DEPrec | 0.24 | 0.10347 | [0.089848, 0.11826] |
| DirAgr | 0.766 | 0.80658 | [0.77809, 0.83419] |

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
  "guidance": 1.5,
  "cfg_min": 0.0,
  "cfg_max": 1.0,
  "clamp": true,
  "de": true,
  "seed": 4242,
  "steps": 50,
  "batch_size": 32,
  "noise_std": 1.0,
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
  }
}
```

原文未公开的实现选择与所有尝试见 `paper/implementation_notes.md`；本表由实际 Cell Eval 结果生成，不自动判定复现成功。
