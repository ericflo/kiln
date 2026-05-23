//! `kiln_tensor::DType` — the full inference + training dtype matrix.
//!
//! The variant list is sized exactly to Qwen3.5-4B per the Phase 0.5 audit
//! ([`bench-results/dtype-usage.md`]): no candle-style superset, no
//! variants without a real call site or a Phase 8 perf lever.
//!
//! [`bench-results/dtype-usage.md`]: ../../../bench-results/dtype-usage.md
//!
//! # Variants
//!
//! - **Numeric**: [`F32`](DType::F32), [`BF16`](DType::BF16),
//!   [`F16`](DType::F16) — activations / accumulators / numerical
//!   reference. F32 has 1,116 call sites in the current repo; BF16 has
//!   1,038; F16 has 28.
//! - **Indices / masks / ids**: [`U32`](DType::U32), [`U8`](DType::U8),
//!   [`I64`](DType::I64).
//! - **FP8**: [`F8E4M3`](DType::F8E4M3), [`F8E5M2`](DType::F8E5M2).
//!   E4M3 is the Qwen3.5-4B paged-KV variant today; E5M2 is reserved
//!   for Phase 8.4 training.
//! - **Packed quantized**: [`Int4Packed`](DType::Int4Packed) — Marlin
//!   W4A16 forward storage; [`Fp4Packed`](DType::Fp4Packed) — Phase
//!   8.10 stub for NVFP4 / MXFP4.
//!
//! # Variants deliberately NOT in the enum
//!
//! - `I32` — no hot-path call sites; Rust-host-side only.
//! - `I8` — Qwen3.5-4B has no W8A8 path.
//! - `F64` — no device-side use; debug receipts promote to `f64`
//!   host-side only.
//! - TF32 / brain-int / posit — out of scope.
//!
//! Adding a variant to this enum without a justifying call site is a
//! Phase 9 audit failure. See the proposal at the bottom of the Phase
//! 0.5 markdown.

use core::fmt;

use crate::{Error, Result};

/// kiln-tensor's dtype enum. Sized exactly to Qwen3.5-4B.
///
/// See the module doc for the per-variant justification and the list
/// of dtypes deliberately excluded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum DType {
    // Numeric.
    /// 32-bit IEEE-754 float. Canonical numerical reference on the CPU
    /// backend; promoted host-side on the bf16 candle CPU path today.
    F32,
    /// `bfloat16` — 1 sign + 8 exponent + 7 mantissa. The production
    /// forward + backward dtype for Qwen3.5-4B.
    BF16,
    /// `half` — 1 sign + 5 exponent + 10 mantissa. Used on the
    /// candle-Mac path and inside kiln-marlin-gemm.
    F16,

    // Indices / masks / ids.
    /// Unsigned 32-bit — paged-KV slot indices, mask bitsets,
    /// per-request flags.
    U32,
    /// Unsigned 8-bit — masks, KV-slot allocation bitmaps.
    U8,
    /// Signed 64-bit — tokenizer ids, sampling argmax output (host-side).
    I64,

    // FP8 — paged-KV + Phase 8.4 forward training.
    /// `fp8_e4m3` (4 exponent + 3 mantissa). Today's `PagedKvCache::new_with_fp8`
    /// uses this; backward stays BF16 (anti-pattern 11's storage-coherence rule).
    F8E4M3,
    /// `fp8_e5m2` (5 exponent + 2 mantissa). Reserved for Phase 8.4
    /// forward training on Hopper / Blackwell.
    F8E5M2,

    // Packed quantized.
    /// 4-bit integer, packed 8 per `u32`. Marlin W4A16 forward storage.
    /// Backward never dispatches through this dtype (Marlin is
    /// forward-only per anti-pattern in Phase 2.5).
    Int4Packed,
    /// 4-bit float, packed 8 per `u32`, with per-block FP8 scales.
    /// **Phase 8.10 stub** — Blackwell NVFP4 / MXFP4 / MXFP6. No call
    /// sites today; the enum variant lets Phase 8 land an `impl
    /// DeviceOp` for it without a workspace-wide refactor.
    Fp4Packed,
}

impl DType {
    /// Size of a single logical element in bytes.
    ///
    /// For packed dtypes ([`Int4Packed`](DType::Int4Packed),
    /// [`Fp4Packed`](DType::Fp4Packed)) this returns the **logical**
    /// per-element size (0 since two elements share one byte); callers
    /// who need the physical packed-buffer footprint use
    /// [`packed_buffer_bytes`](DType::packed_buffer_bytes).
    pub const fn size_in_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::F16 => 2,
            DType::U32 => 4,
            DType::U8 => 1,
            DType::I64 => 8,
            DType::F8E4M3 => 1,
            DType::F8E5M2 => 1,
            // Packed: two 4-bit nibbles per byte. The per-element size
            // is conceptually 0.5; we return 0 and require callers to
            // use packed_buffer_bytes() for layout math.
            DType::Int4Packed => 0,
            DType::Fp4Packed => 0,
        }
    }

    /// Compute the physical byte footprint of a packed buffer with
    /// `n_elements` logical elements.
    ///
    /// For unpacked dtypes this is equivalent to
    /// `n_elements * size_in_bytes()`. For [`Int4Packed`](DType::Int4Packed) /
    /// [`Fp4Packed`](DType::Fp4Packed), this returns
    /// `ceil_div(n_elements, 2)`.
    pub const fn packed_buffer_bytes(self, n_elements: usize) -> usize {
        match self {
            DType::Int4Packed | DType::Fp4Packed => n_elements.div_ceil(2),
            _ => n_elements * self.size_in_bytes(),
        }
    }

    /// Is this dtype a packed quantized type whose per-element size is
    /// less than one byte? Phase 1.4+ storage backends gate their
    /// allocator math on this.
    pub const fn is_packed(self) -> bool {
        matches!(self, DType::Int4Packed | DType::Fp4Packed)
    }

    /// Is this dtype a floating-point type (any precision)?
    pub const fn is_float(self) -> bool {
        matches!(
            self,
            DType::F32 | DType::BF16 | DType::F16 | DType::F8E4M3 | DType::F8E5M2 | DType::Fp4Packed
        )
    }

    /// Is this dtype an integer type (signed or unsigned)?
    pub const fn is_int(self) -> bool {
        matches!(self, DType::U32 | DType::U8 | DType::I64 | DType::Int4Packed)
    }

    /// Conventional short name. Stable across releases — it's how
    /// `bench-results/dtype-usage.csv` rows and parity-tolerance rows
    /// key on dtype.
    pub const fn short_name(self) -> &'static str {
        match self {
            DType::F32 => "f32",
            DType::BF16 => "bf16",
            DType::F16 => "f16",
            DType::U32 => "u32",
            DType::U8 => "u8",
            DType::I64 => "i64",
            DType::F8E4M3 => "f8_e4m3",
            DType::F8E5M2 => "f8_e5m2",
            DType::Int4Packed => "int4_packed",
            DType::Fp4Packed => "fp4_packed",
        }
    }

    /// Parse a [`DType`] from its [`short_name`](DType::short_name).
    ///
    /// Used by env-var-driven configuration and by safetensors-loader
    /// metadata. Returns an [`Error::Msg`] with the unrecognised input
    /// for migration sites that previously matched on
    /// `candle_core::DType` via `parse()`.
    pub fn from_short_name(s: &str) -> Result<Self> {
        match s {
            "f32" | "F32" | "fp32" | "FP32" => Ok(DType::F32),
            "bf16" | "BF16" | "bfloat16" => Ok(DType::BF16),
            "f16" | "F16" | "fp16" | "FP16" | "half" => Ok(DType::F16),
            "u32" | "U32" => Ok(DType::U32),
            "u8" | "U8" => Ok(DType::U8),
            "i64" | "I64" => Ok(DType::I64),
            "f8_e4m3" | "F8E4M3" | "fp8_e4m3" | "FP8E4M3" => Ok(DType::F8E4M3),
            "f8_e5m2" | "F8E5M2" | "fp8_e5m2" | "FP8E5M2" => Ok(DType::F8E5M2),
            "int4_packed" | "INT4_PACKED" | "marlin" | "MARLIN" => Ok(DType::Int4Packed),
            "fp4_packed" | "FP4_PACKED" | "nvfp4" | "NVFP4" | "mxfp4" | "MXFP4" => {
                Ok(DType::Fp4Packed)
            }
            other => Err(Error::Msg(format!(
                "unknown dtype short name {other:?}; expected one of: \
                 f32, bf16, f16, u32, u8, i64, f8_e4m3, f8_e5m2, int4_packed, fp4_packed"
            ))),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.short_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_in_bytes_known_values() {
        assert_eq!(DType::F32.size_in_bytes(), 4);
        assert_eq!(DType::BF16.size_in_bytes(), 2);
        assert_eq!(DType::F16.size_in_bytes(), 2);
        assert_eq!(DType::U32.size_in_bytes(), 4);
        assert_eq!(DType::U8.size_in_bytes(), 1);
        assert_eq!(DType::I64.size_in_bytes(), 8);
        assert_eq!(DType::F8E4M3.size_in_bytes(), 1);
        assert_eq!(DType::F8E5M2.size_in_bytes(), 1);
        assert_eq!(DType::Int4Packed.size_in_bytes(), 0);
        assert_eq!(DType::Fp4Packed.size_in_bytes(), 0);
    }

    #[test]
    fn packed_buffer_bytes_handles_packing() {
        assert_eq!(DType::F32.packed_buffer_bytes(1024), 4096);
        assert_eq!(DType::BF16.packed_buffer_bytes(1024), 2048);
        // Packed: 2 elements per byte, with ceil for odd lengths.
        assert_eq!(DType::Int4Packed.packed_buffer_bytes(8), 4);
        assert_eq!(DType::Int4Packed.packed_buffer_bytes(9), 5);
        assert_eq!(DType::Int4Packed.packed_buffer_bytes(0), 0);
        assert_eq!(DType::Fp4Packed.packed_buffer_bytes(1024), 512);
    }

    #[test]
    fn is_packed_is_float_is_int() {
        for dt in [DType::Int4Packed, DType::Fp4Packed] {
            assert!(dt.is_packed(), "{dt} should be packed");
        }
        for dt in [
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::U32,
            DType::U8,
            DType::I64,
            DType::F8E4M3,
            DType::F8E5M2,
        ] {
            assert!(!dt.is_packed(), "{dt} should not be packed");
        }
        for dt in [
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::Fp4Packed,
        ] {
            assert!(dt.is_float(), "{dt} should be float");
            assert!(!dt.is_int(), "{dt} should not be int");
        }
        for dt in [DType::U32, DType::U8, DType::I64, DType::Int4Packed] {
            assert!(dt.is_int(), "{dt} should be int");
            assert!(!dt.is_float(), "{dt} should not be float");
        }
    }

    #[test]
    fn short_name_roundtrip() {
        for dt in [
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::U32,
            DType::U8,
            DType::I64,
            DType::F8E4M3,
            DType::F8E5M2,
            DType::Int4Packed,
            DType::Fp4Packed,
        ] {
            let s = dt.short_name();
            assert_eq!(DType::from_short_name(s).unwrap(), dt);
        }
    }

    #[test]
    fn from_short_name_accepts_aliases() {
        assert_eq!(DType::from_short_name("FP32").unwrap(), DType::F32);
        assert_eq!(DType::from_short_name("bfloat16").unwrap(), DType::BF16);
        assert_eq!(DType::from_short_name("half").unwrap(), DType::F16);
        assert_eq!(DType::from_short_name("marlin").unwrap(), DType::Int4Packed);
        assert_eq!(DType::from_short_name("NVFP4").unwrap(), DType::Fp4Packed);
    }

    #[test]
    fn from_short_name_rejects_unknown() {
        let err = DType::from_short_name("complex64").unwrap_err();
        assert!(err.to_string().contains("complex64"));
    }

    #[test]
    fn display_uses_short_name() {
        assert_eq!(format!("{}", DType::BF16), "bf16");
        assert_eq!(format!("{}", DType::Int4Packed), "int4_packed");
    }
}
