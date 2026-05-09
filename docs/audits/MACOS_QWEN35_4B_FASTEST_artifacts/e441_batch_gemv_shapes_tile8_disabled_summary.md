E441 row-quad tile8-disabled shape comparison

| label | batch | default us/policy | disabled us/policy | disabled/default |
| --- | ---: | ---: | ---: | ---: |
| attn_output | 2 | `334.760` row_pair_tile8_shared | `426.433` row_pair_tile8_shared | `1.274x` |
| attn_output | 3 | `607.650` row_pair_tile8_shared | `536.310` row_pair_tile8_shared | `0.883x` |
| attn_output | 4 | `552.200` row_pair_tile8_shared | `630.119` row_pair_tile8_shared | `1.141x` |
| attn_output | 8 | `996.023` row_quad_tile8 | `909.385` row_quad_tile4_shared | `0.913x` |
| attn_qkv_like | 2 | `418.392` row_pair_tile8_shared | `417.133` row_pair_tile8_shared | `0.997x` |
| attn_qkv_like | 3 | `811.804` row_pair_tile8_shared | `873.756` row_pair_tile8_shared | `1.076x` |
| attn_qkv_like | 4 | `960.577` row_pair_tile8_shared | `900.665` row_pair_tile8_shared | `0.938x` |
| attn_qkv_like | 8 | `1235.338` row_quad_tile8 | `1429.346` row_quad_tile4_shared | `1.157x` |
| gdn_out_proj | 2 | `495.569` row_pair_tile8_shared | `533.358` row_pair_tile8_shared | `1.076x` |
| gdn_out_proj | 3 | `903.560` row_pair_tile8_shared | `936.413` row_pair_tile8_shared | `1.036x` |
| gdn_out_proj | 4 | `954.365` row_pair_tile8_shared | `1051.652` row_pair_tile8_shared | `1.102x` |
| gdn_out_proj | 8 | `1317.763` row_quad_tile8 | `1433.075` row_quad_tile4_shared | `1.088x` |
| mlp_down_proj | 2 | `1071.554` row_pair_tile8_shared | `1123.685` row_pair_tile8_shared | `1.049x` |
| mlp_down_proj | 3 | `2024.062` row_pair_tile8_shared | `1990.498` row_pair_tile8_shared | `0.983x` |
| mlp_down_proj | 4 | `2043.938` row_pair_tile8_shared | `2038.460` row_pair_tile8_shared | `0.997x` |
| mlp_down_proj | 8 | `2897.725` row_quad_tile8 | `3108.740` row_quad_tile4_shared | `1.073x` |
