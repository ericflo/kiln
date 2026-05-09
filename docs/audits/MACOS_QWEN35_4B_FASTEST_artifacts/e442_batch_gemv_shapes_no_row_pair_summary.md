E442 batch transposed-GEMV row-pair policy comparison

Same-source Qwen3.5 batch transposed-GEMV selected-path bench, default versus
`KILN_DISABLE_METAL_TRANSPOSED_COOP_GEMV_ROW_PAIR=1`.

| label | batch | default | no row pair | no/default |
| --- | ---: | ---: | ---: | ---: |
| mlp_down_proj | 2 | `1087.656` row_pair_tile8_shared | `1840.315` rowwise_tile8_shared | `1.692x` |
| mlp_down_proj | 3 | `1909.260` row_pair_tile8_shared | `2791.427` rowwise_tile8_shared | `1.462x` |
| mlp_down_proj | 4 | `2039.400` row_pair_tile8_shared | `3665.152` rowwise_tile8_shared | `1.797x` |
| mlp_down_proj | 8 | `2879.715` row_quad_tile8 | `7008.546` rowwise_tile8_shared | `2.434x` |
| gdn_out_proj | 2 | `493.350` row_pair_tile8_shared | `889.690` rowwise_tile8_shared | `1.803x` |
| gdn_out_proj | 3 | `838.362` row_pair_tile8_shared | `1181.519` rowwise_tile8_shared | `1.409x` |
| gdn_out_proj | 4 | `907.181` row_pair_tile8_shared | `1648.844` rowwise_tile8_shared | `1.818x` |
| gdn_out_proj | 8 | `1433.606` row_quad_tile8 | `3254.529` rowwise_tile8_shared | `2.270x` |
| attn_output | 2 | `334.594` row_pair_tile8_shared | `624.250` rowwise_tile8_shared | `1.866x` |
| attn_output | 3 | `646.496` row_pair_tile8_shared | `997.335` rowwise_tile8_shared | `1.543x` |
| attn_output | 4 | `563.077` row_pair_tile8_shared | `1076.479` rowwise_tile8_shared | `1.912x` |
| attn_output | 8 | `967.652` row_quad_tile8 | `2142.238` rowwise_tile8_shared | `2.214x` |
| attn_qkv_like | 2 | `453.733` row_pair_tile8_shared | `854.610` rowwise_tile8_shared | `1.884x` |
| attn_qkv_like | 3 | `811.052` row_pair_tile8_shared | `1259.850` rowwise_tile8_shared | `1.553x` |
| attn_qkv_like | 4 | `1008.085` row_pair_tile8_shared | `1596.131` rowwise_tile8_shared | `1.583x` |
| attn_qkv_like | 8 | `1234.385` row_quad_tile8 | `3172.929` rowwise_tile8_shared | `2.570x` |

Disabling row-pair is slower for every tested shape and batch. Keep the current
row-pair/row-quad policy; there is no shape-specific rowwise exception to try.
