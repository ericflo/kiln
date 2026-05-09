E441 row-quad tile8 shape-gate comparison

| label | batch8 default1 | batch8 disabled | batch8 default2 | verdict |
| --- | ---: | ---: | ---: | --- |
| mlp_down_proj | `2897.725` row_quad_tile8 | `3108.740` row_quad_tile4_shared | `2718.700` row_quad_tile8 | tile8 current favored |
| gdn_out_proj | `1317.763` row_quad_tile8 | `1433.075` row_quad_tile4_shared | `1247.850` row_quad_tile8 | tile8 current favored |
| attn_output | `996.023` row_quad_tile8 | `909.385` row_quad_tile4_shared | `953.096` row_quad_tile8 | tile4 disabled favored |
| attn_qkv_like | `1235.338` row_quad_tile8 | `1429.346` row_quad_tile4_shared | `1212.088` row_quad_tile8 | tile8 current favored |

Full per-batch disabled/default ratios are in the source logs; batch8 is the only row-quad-tile8 decision point.
