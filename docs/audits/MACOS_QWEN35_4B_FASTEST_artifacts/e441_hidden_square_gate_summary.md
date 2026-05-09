E441 hidden-square row-quad tile8 gate comparison

The candidate gate kept batch8 `row_quad_tile8` for larger projection shapes but
sent the hidden-square attention-output shape `[8,1,2560] x [2560,2560]` back
to `row_quad_tile4_shared`.

| label | candidate default | rollback env | verdict |
| --- | ---: | ---: | --- |
| mlp_down_proj | `2747.446` row_quad_tile8 | `2754.502` row_quad_tile8 | same policy; noise |
| gdn_out_proj | `1325.562` row_quad_tile8 | `1291.787` row_quad_tile8 | same policy; noise |
| attn_output | `969.438` row_quad_tile4_shared | `894.369` row_quad_tile8 | rollback tile8 favored |
| attn_qkv_like | `1275.692` row_quad_tile8 | `1216.490` row_quad_tile8 | same policy; noise |

Earlier global tile8-disable probes made the hidden-square batch8 tile4 path
look faster (`909.385 us`) than the existing tile8 path (`996.023 us` and
`953.096 us`), while the direct implemented-gate A/B reversed that result. The
shape gate was therefore rejected and source was reverted.
