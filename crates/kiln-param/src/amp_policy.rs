//! `AmpPolicy` — per-Parameter declaration of forward / backward /
//! master / accumulation dtypes.
//!
//! Per the Phase 2.5 issue bullet:
//!
//! > **Explicit AMP / autocast policy on `Parameter`.** Today candle's
//! > autocast handles dtype promotion implicitly — that disappears
//! > with candle. Each Parameter declares `AmpPolicy {
//! > forward_compute_dtype, backward_compute_dtype, master_dtype,
//! > accumulation_dtype }` at construction. Mixed-precision is then a
//! > property of the Parameter, not an implicit behavior of the call
//! > site.
//!
//! # Default for Qwen3.5-4B
//!
//! Per the issue: `{forward=BF16, backward=BF16, master=BF16,
//! accumulation=FP32}`.
//!
//! # FP8 path override
//!
//! Phase 8.4 introduces FP8 forward training:
//! `{forward=FP8E4M3, backward=BF16, master=BF16, accumulation=FP32}`.
//! The override slot is the same struct — `AmpPolicy::fp8_training()`.

use kiln_tensor::DType;

/// Per-Parameter dtype policy.
///
/// `kiln-optim` (Phase 6.5) reads the policy off each Parameter and
/// dispatches the matmul-bwd kernel + accumulator without any global
/// "AMP mode" flag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AmpPolicy {
    /// Dtype the forward pass reads inputs as.
    pub forward_compute_dtype: DType,
    /// Dtype the backward pass reads / writes intermediate grads as.
    pub backward_compute_dtype: DType,
    /// Dtype of the master copy that the optimizer step updates.
    pub master_dtype: DType,
    /// Dtype used by inner accumulators (matmul, reduction, AdamW
    /// running moments).
    pub accumulation_dtype: DType,
}

impl AmpPolicy {
    /// Default for Qwen3.5-4B: BF16 fwd + bwd + master, FP32 accumulation.
    pub const fn qwen3p5_4b_default() -> Self {
        AmpPolicy {
            forward_compute_dtype: DType::BF16,
            backward_compute_dtype: DType::BF16,
            master_dtype: DType::BF16,
            accumulation_dtype: DType::F32,
        }
    }

    /// FP32 everywhere — the numerical-reference policy for parity
    /// tests against the CPU backend.
    pub const fn fp32_reference() -> Self {
        AmpPolicy {
            forward_compute_dtype: DType::F32,
            backward_compute_dtype: DType::F32,
            master_dtype: DType::F32,
            accumulation_dtype: DType::F32,
        }
    }

    /// Phase 8.4 FP8-forward training policy.
    pub const fn fp8_training() -> Self {
        AmpPolicy {
            forward_compute_dtype: DType::F8E4M3,
            backward_compute_dtype: DType::BF16,
            master_dtype: DType::BF16,
            accumulation_dtype: DType::F32,
        }
    }

    /// Marlin-W4A16 training policy (forward through Int4Packed; bwd
    /// through BF16 master; LoRA delta on top per the anti-pattern in
    /// Phase 2.5).
    pub const fn marlin_w4a16_training() -> Self {
        AmpPolicy {
            forward_compute_dtype: DType::Int4Packed,
            backward_compute_dtype: DType::BF16,
            master_dtype: DType::BF16,
            accumulation_dtype: DType::F32,
        }
    }

    /// Sanity check: the four dtypes form a coherent policy.
    /// Returns `Ok` for known-good policies (the four constructors
    /// above plus user-defined coherent combinations); returns
    /// `Err(&'static str)` describing the violation otherwise.
    pub fn validate(self) -> Result<(), &'static str> {
        // Packed dtypes are only allowed in `forward_compute_dtype`.
        if self.master_dtype.is_packed() {
            return Err("master_dtype must not be packed (Int4Packed/Fp4Packed)");
        }
        if self.backward_compute_dtype.is_packed() {
            return Err("backward_compute_dtype must not be packed");
        }
        if self.accumulation_dtype.is_packed() {
            return Err("accumulation_dtype must not be packed");
        }
        // Accumulator must be at least as wide as backward compute
        // (FP32 master + BF16 accumulator would silently drop precision).
        if matches!(self.backward_compute_dtype, DType::F32)
            && !matches!(self.accumulation_dtype, DType::F32)
        {
            return Err("F32 backward requires F32 accumulation_dtype");
        }
        Ok(())
    }
}

impl Default for AmpPolicy {
    fn default() -> Self {
        Self::qwen3p5_4b_default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn qwen_default_is_bf16_with_fp32_accum() {
        let p = AmpPolicy::default();
        assert_eq!(p.forward_compute_dtype, DType::BF16);
        assert_eq!(p.backward_compute_dtype, DType::BF16);
        assert_eq!(p.master_dtype, DType::BF16);
        assert_eq!(p.accumulation_dtype, DType::F32);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn fp32_reference_validates() {
        let p = AmpPolicy::fp32_reference();
        assert_eq!(p.forward_compute_dtype, DType::F32);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn fp8_training_validates() {
        let p = AmpPolicy::fp8_training();
        assert_eq!(p.forward_compute_dtype, DType::F8E4M3);
        assert_eq!(p.backward_compute_dtype, DType::BF16);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn marlin_training_validates() {
        let p = AmpPolicy::marlin_w4a16_training();
        assert_eq!(p.forward_compute_dtype, DType::Int4Packed);
        assert!(p.validate().is_ok());
    }

    #[test]
    fn packed_master_dtype_is_rejected() {
        let p = AmpPolicy {
            forward_compute_dtype: DType::F32,
            backward_compute_dtype: DType::F32,
            master_dtype: DType::Int4Packed,
            accumulation_dtype: DType::F32,
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn bf16_accum_with_f32_bwd_is_rejected() {
        let p = AmpPolicy {
            forward_compute_dtype: DType::F32,
            backward_compute_dtype: DType::F32,
            master_dtype: DType::F32,
            accumulation_dtype: DType::BF16,
        };
        assert!(p.validate().is_err());
    }
}
