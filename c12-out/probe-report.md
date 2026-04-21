# Phase C12 — activation-weighted Marlin drift probe

- Checkpoint: `/workspace/qwen3.5-4b`
- Device: `cuda`, dtype: `bf16`
- Calibration prompts: 32 (595 tokens total)
- Projections audited: 104
- Elapsed: 160.1s
- Benign threshold: weighted_drift_ratio < 0.001

## Top 10 projections by weighted drift ratio

| layer | kind | k (in) | n (out) | weighted_ratio | err_frob | drift_frob |
|------:|:-----|------:|--------:|----------------:|---------:|-----------:|
| 6 | down_proj | 9216 | 2560 | 1.447e-01 | 2.133e+01 | 4.618e+00 |
| 18 | down_proj | 9216 | 2560 | 1.384e-01 | 2.980e+01 | 5.056e+00 |
| 25 | down_proj | 9216 | 2560 | 1.202e-01 | 3.003e+01 | 4.926e+00 |
| 5 | up_proj | 2560 | 9216 | 1.179e-01 | 8.661e+01 | 4.566e+00 |
| 4 | up_proj | 2560 | 9216 | 1.177e-01 | 7.949e+01 | 4.603e+00 |
| 15 | up_proj | 2560 | 9216 | 1.170e-01 | 9.807e+01 | 4.845e+00 |
| 17 | down_proj | 9216 | 2560 | 1.168e-01 | 1.031e+01 | 4.882e+00 |
| 6 | up_proj | 2560 | 9216 | 1.164e-01 | 9.644e+01 | 4.563e+00 |
| 21 | down_proj | 9216 | 2560 | 1.161e-01 | 1.872e+01 | 4.876e+00 |
| 20 | down_proj | 9216 | 2560 | 1.154e-01 | 1.739e+01 | 4.874e+00 |

**Worst projection: `L6.down_proj`, weighted_drift_ratio = 1.447e-01**

**Verdict: NON-TRIVIAL.** At least one projection has an activation-weighted drift ratio >= 1e-03; Marlin drift plausibly lands on high-activation channels. Corroborates PRIMARY-POSITIVE on the C12 fp32-head bench.
