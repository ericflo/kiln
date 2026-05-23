# Phase 0.5 — DType usage audit

Source of truth: `bench-results/dtype-usage.csv`.
Regenerate: `scripts/audit-dtype-usage.py`.

Why this audit
--------------

Per the issue: every kept dtype in `kiln_tensor::DType` must be justified by either (a) a real Qwen3.5-4B call site or (b) a Phase 8 perf lever. This audit produces the call-site evidence for each candidate dtype so the Phase 1 `DType` enum is sized exactly to what we ship, not a candle-style superset.

## Counts

| dtype | call sites | first seen | top crates |
|---|---:|---|---|
| `FP32 (F32)` | 1116 | `crates/kiln-marlin-gemm/tests/parity.rs:106` | kiln-model=639;kiln-train=179;kiln-vulkan-kernel=104;kiln-rmsnorm-kernel=71;kiln-opd-loss-kernel=36 |
| `BF16` | 1038 | `crates/kiln-marlin-gemm/src/lib.rs:152` | kiln-model=667;kiln-gdn-kernel=134;kiln-rmsnorm-kernel=113;kiln-vulkan-kernel=32;kiln-flash-attn=31 |
| `FP16 (F16)` | 28 | `crates/kiln-marlin-gemm/src/lib.rs:161` | kiln-model=16;kiln-server=7;kiln-marlin-gemm=3;kiln-train=1;kiln-vulkan-kernel=1 |
| `U8` | 9 | `crates/kiln-model/src/decode_buffers.rs:29` | kiln-model=9 |
| `I64` | 1 | `crates/kiln-model/src/cuda_train.rs:316` | kiln-model=1 |
| `U32` | 14 | `crates/kiln-flash-attn/src/lib.rs:286` | kiln-model=6;kiln-flash-attn=3;kiln-train=3;kiln-flce-kernel=1;kiln-vulkan-kernel=1 |
| `FP8 E4M3 (candle)` | 0 | `` |  |
| `FP8 E5M2 (candle)` | 0 | `` |  |
| `FP8 E4M3 (kiln-custom)` | 9 | `crates/kiln-model/src/decode_buffers.rs:21` | kiln-model=9 |
| `FP8 E5M2 (kiln-custom)` | 0 | `` |  |
| `FP8 K/V cache scales` | 24 | `crates/kiln-model/src/kv_cache.rs:35` | kiln-model=24 |
| `INT4 packed (Marlin)` | 112 | `crates/kiln-marlin-gemm/build.rs:19` | kiln-model=80;kiln-marlin-gemm=31;kiln-gdn-kernel=1 |
| `FP4Packed (future)` | 0 | `` |  |

## Proposed `kiln_tensor::DType` for Qwen3.5-4B

Based on the counts above:

```rust
pub enum DType {
    // Activations + accumulators (FP32 is the canonical numerical
    // reference on CPU; BF16 is the production forward + backward
    // dtype for Qwen3.5-4B; FP16 is the candle-Mac path).
    F32,
    BF16,
    F16,
    // Index / mask dtypes (paged-KV slot indices, attention mask).
    U32,
    U8,
    I64,    // tokenizer ids; sampling argmax output
    // FP8 — paged KV cache `new_with_fp8` is the only hot-path
    // user today. Forward-only; backward stays BF16.
    F8E4M3,
    F8E5M2,
    // INT4 packed — Marlin W4A16 forward storage. Backward never
    // dispatches here (anti-pattern: Marlin is forward-only).
    Int4Packed,
    // FP4 packed — Blackwell NVFP4 / MXFP4. Not present today;
    // scaffolded for Phase 8.10.
    Fp4Packed,
}
```

Dtypes deliberately **NOT** in the enum:

- **`I32`** — no call sites in the forward path; Rust's `i32` is
  used for tokenizer scratch but never crosses a tensor boundary.
- **`I8`** — there is no INT8-quantized path in Qwen3.5-4B. (We have
  Marlin W4A16, not W8A8.)
- **`F64`** — no call sites in the hot path. (Some debug receipts
  promote to `f64` Rust-host-side for stable JSON, but not on
  the device.)
- **TF32 / Brain-int / Posit** — out of scope.

## Causal links forward

- **Phase 1 dependency**: this enum lands as part of the `kiln-tensor` scaffold. Each kernel crate declares which `DType`s its `DeviceOp` supports; the dispatch table is sized against this enum.
- **Phase 2.5 dependency**: `Parameter::AmpPolicy` carries a
  `DType` per role (forward / backward / master / accumulation); the
  enum above is the surface that AmpPolicy fields enumerate over.
- **Phase 8.10 hook**: `Fp4Packed` is scaffolded today as a stub variant
  so Blackwell-class hardware (NVFP4 / MXFP6) is a per-DeviceOp
  extension rather than a workspace-wide refactor.
- **Phase 9 enforcement**: re-run this audit as a CI step; an
  enum variant added without a justifying call site fails the gate.
