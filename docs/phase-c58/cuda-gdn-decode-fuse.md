# Phase C58 CUDA GDN Decode Fusion Attempt

Date: 2026-04-24
Pod: `yzf1plx0m6z092` (`NVIDIA A40`, `ghcr.io/ericflo/kiln-runpod:latest`)

## Scope

Added a narrow CUDA C-ABI entry point for native-MTP `seq_len == 1` GDN decode that computes gates and advances the recurrent GDN state in one launch for bf16 `dk == dv == 128` tensors. The hook is wired through `BackendRuntime::gdn_decode_gates_recurrent` and the CUDA backend.

## Current Status

The CUDA hook is guarded behind `KILN_ENABLE_FUSED_GDN_DECODE=1` and remains disabled by default. During validation, the state update matched the split CUDA path exactly, but the recurrent output did not:

```text
cuda decode gates+recurrent vs split: max_abs_diff=4.1328125e0 mean_abs_diff=5.352731e-1 state_max=0e0
```

Because output parity is not proven, the production path continues to use the existing split `gdn/gates` + `gdn/recurrent` + `gdn/gated_norm` kernels unless explicitly opted in for debugging.

## Validation Notes

Passing:

```bash
cargo test -p kiln-gdn-kernel --release -- --nocapture
```

Known failing during this attempt:

```bash
cargo test -p kiln-model --release --features cuda gdn -- --nocapture
```

The command included an existing `test_gdn_chunk_body_matches_fallback` failure on this A40 run, and the experimental fused-output parity test failed before it was removed from the default suite.

## Next Step

Do not retry standalone gates/recurrent/gated-norm fusion blindly. First isolate the recurrent output discrepancy with a tiny debug kernel or compare the existing `kiln_gdn_recurrent_forward` output against a C-ABI variant fed the exact `[B, 1, H, D]` layout. If this remains below a 5% decode throughput gain, pivot to `:kiln/gdn/qk_norm` or `:kiln/attn/rope` per the C57 profile guidance.
