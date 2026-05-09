E444 MLP gate/up serial fast-exp candidate summary

The temporary candidate added a dedicated
`kiln_mlp_gate_up_serial_fast_exp_bf16` kernel selected only for aligned bs=1
serial decode, with `KILN_DISABLE_METAL_MLP_GATE_UP_SERIAL_FAST_EXP=1`
rolling back to the current dedicated serial kernel.

| run | fast-exp b1 | rollback b1 | result |
| --- | ---: | ---: | --- |
| pair 1 | `1684.021 us` | `1679.600 us` | fast/rollback `1.003x` |
| counter pair | `1666.696 us` | `1679.260 us` | fast/rollback `0.993x` |
| longer w10/i50 | `1719.310 us` | `1692.736 us` | fast/rollback `1.016x` |

Batch `1` is the only row that selected the candidate fast-exp serial kernel.
Batch `2/3/4/8` remained on existing kernels and moved only as benchmark noise.
The longer pair favored rollback, so the candidate was rejected and source was
reverted.
