E440 Metal batch transposed GEMV Qwen-shape summary

| label | batch2 | batch3 | batch4 | batch8 |
| --- | ---: | ---: | ---: | ---: |
| mlp_down_proj | `1071.554 us` (row_pair_tile8_shared) | `2024.062 us` (row_pair_tile8_shared) | `2043.938 us` (row_pair_tile8_shared) | `2897.725 us` (row_quad_tile8) |
| gdn_out_proj | `495.569 us` (row_pair_tile8_shared) | `903.560 us` (row_pair_tile8_shared) | `954.365 us` (row_pair_tile8_shared) | `1317.763 us` (row_quad_tile8) |
| attn_output | `334.760 us` (row_pair_tile8_shared) | `607.650 us` (row_pair_tile8_shared) | `552.200 us` (row_pair_tile8_shared) | `996.023 us` (row_quad_tile8) |
| attn_qkv_like | `418.392 us` (row_pair_tile8_shared) | `811.804 us` (row_pair_tile8_shared) | `960.577 us` (row_pair_tile8_shared) | `1235.338 us` (row_quad_tile8) |

All rows matched broadcast matmul exactly in the logged run (max_abs_diff=0, mean_abs_diff=0).
