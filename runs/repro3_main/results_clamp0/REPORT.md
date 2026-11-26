# Replogle–Nadig 复现结果

正式测试：26,878 个扰动细胞、380 个扰动、2,000 HVG；真实对照与预测端对照一致。

| 指标 | 原文 Table 3 | 本次 | 95% 区间 |
| --- | ---: | ---: | ---: |
| PDCorr | 0.437 | 0.41318 | [0.3796, 0.44978] |
| MSE | 0.0079 | 0.010059 | [0.0093843, 0.010804] |
| MAE | 0.061 | 0.067041 | [0.065201, 0.068867] |
| PDS_L1 | 0.82 | 0.83561 | [0.81505, 0.85459] |
| PDS_L2 | 0.853 | 0.84949 | [0.82974, 0.86856] |
| PDS_cos | 0.833 | 0.8031 | [0.77591, 0.82756] |
| DEOver | 0.232 | 0.18655 | [0.16281, 0.21107] |
| DEPrec | 0.24 | 0.11274 | [0.098081, 0.12814] |
| DirAgr | 0.766 | 0.82288 | [0.79738, 0.84742] |

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
  "guidance": 1.375,
  "cfg_min": 0.0,
  "cfg_max": 1.0,
  "clamp": true,
  "de": true,
  "seed": 4242,
  "steps": 50,
  "batch_size": 32,
  "noise_std": 0.75,
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
  }
}
```

原文未公开的实现选择与所有尝试见 `paper/implementation_notes.md`；本表由实际 Cell Eval 结果生成，不自动判定复现成功。
