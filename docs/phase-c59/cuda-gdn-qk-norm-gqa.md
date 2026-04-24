# Phase C59 — CUDA GDN qk_norm GQA Fast Path

## Goal

Add a minimal CUDA equivalent of the Metal `metal_gdn_qk_norm_gqa_f32_bf16` path for GDN Q/K normalization. The fast path consumes unexpanded GQA Q/K tensors shaped `[batch, seq_len, nk, dk]`, normalizes each source head in F32, applies the Q scale, casts to bf16, and writes expanded `[batch, seq_len, nv, dk]` outputs in one CUDA launch.

## Implementation

- Added `kiln_fused_l2_qk_norm_gqa` in `crates/kiln-rmsnorm-kernel/csrc/fused_l2_qk_norm.cu` with a thin C ABI declaration in `fused_l2_qk_norm.h`.
- Kept the CUDA envelope intentionally narrow: bf16 inputs/outputs, forward-only, contiguous rank-4 tensors, `dk == 128`, and `nv = nk * ratio`.
- Added Rust wrappers `supports_l2_qk_norm_gqa` and `fused_l2_qk_norm_gqa` in `crates/kiln-rmsnorm-kernel/src/lib.rs`.
- Routed the non-Metal CUDA GDN path in `crates/kiln-model/src/forward.rs` through `:kiln/gdn/qk_norm_gqa` before the separate `:kiln/gdn/head_expand` + `:kiln/gdn/qk_norm` fallback.
- Preserved `KILN_DISABLE_FUSED_L2_QK_NORM=1` as the kill switch for both expanded and GQA fused QK normalization.

## Validation

Commands run on RunPod A6000 with `ghcr.io/ericflo/kiln-runpod:latest`:

```bash
cargo test -p kiln-rmsnorm-kernel --release -- --nocapture
cargo test -p kiln-model --release --features cuda gdn_qk_norm -- --nocapture
cargo build --release --features cuda,nvtx --bin kiln-bench
```

Results:

- `cargo test -p kiln-rmsnorm-kernel --release -- --nocapture`: pass, 8 tests.
- `cargo test -p kiln-model --release --features cuda gdn_qk_norm -- --nocapture`: pass, 0 matching tests run after successful `kiln-model` CUDA build (245 lib tests filtered).

## Benchmark

Benchmark command:

```bash
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
```

A6000 single-run results:

| Path | Mean ITL | Decode tok/s | Prefill |
| --- | ---: | ---: | ---: |
| default fused GQA | 22.4 ms | 44.7 tok/s | 351.5 ms |
| `KILN_DISABLE_FUSED_L2_QK_NORM=1` | 22.7 ms | 44.0 tok/s | 350.4 ms |

Decode speedup: `44.7 / 44.0 = 1.015x` (~1.5%), above the 1% floor but small enough to treat as a modest decode-path cleanup, not a new fusion frontier.

## Default Decision

Keep the path default-on behind the existing kill switch. The measured single-run decode gain is ~1.5%, above the 1% floor, and the implementation removes the separate CUDA `:kiln/gdn/head_expand` materialization before QK norm. Do not retry this same fusion shape for larger wins; the next pivot should target a different current top hotspot.
