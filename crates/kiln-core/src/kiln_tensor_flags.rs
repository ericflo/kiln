//! `KILN_USE_KILN_TENSOR_*` migration flags.
//!
//! Per the Phase 2 issue bullet:
//!
//! > Wire into `swiglu_ffn_impl_no_chunk` behind
//! > `KILN_USE_KILN_TENSOR_MLP_<backend>=1`; candle path stays
//! > default.
//!
//! This module centralizes the per-op migration flags so that:
//! - forward.rs has one import path for "should this op route
//!   through kiln-tensor or stay on candle today?";
//! - the preserve-list audit (Phase 0.7) sees every flag in one
//!   place;
//! - tests can flip the flag without depending on env-var literals
//!   from a dozen call sites.
//!
//! Every flag defaults to **off**. Phase 7 (candle removal) is the
//! point at which the candle paths are deleted and these flags
//! become no-ops; the names are preserved so the migration window
//! has a stable kill-switch story.

use crate::env_flag::env_flag;

/// Per-op + per-backend kiln-tensor migration flags.
///
/// Each variant is the env var name suffix after `KILN_USE_KILN_TENSOR_`.
/// `Mlp(Backend)` → `KILN_USE_KILN_TENSOR_MLP_CUDA` etc.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KilnTensorOp {
    /// MLP gate||up||down. Phase 2 first hot path per backend.
    Mlp(Backend),
    /// Attention QKV projection + RoPE + flash-attn + O. Phase 3.
    Attention(Backend),
    /// Embedding + RMSNorm + residual + final norm + LM head + sampling.
    /// Phase 4.
    Glue(Backend),
    /// MTP + self-speculative drafting. Phase 4.5.
    Mtp(Backend),
    /// Backward pass (matmul / elementwise / norms / softmax). Phase 6b/6c.
    Backward(Backend),
    /// Optimizer step (`kiln-optim::OptimStep`). Phase 6.5.
    Optimizer(Backend),
}

/// Per-backend flag identifier. `Cpu` is the canonical reference
/// path — that flag stays off but exists for completeness and lets
/// the preserve-list audit see every `(op, backend)` cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Backend {
    Cuda,
    Metal,
    Vulkan,
    Cpu,
}

impl Backend {
    const fn name(self) -> &'static str {
        match self {
            Backend::Cuda => "CUDA",
            Backend::Metal => "METAL",
            Backend::Vulkan => "VULKAN",
            Backend::Cpu => "CPU",
        }
    }
}

impl KilnTensorOp {
    /// Full env-var name. E.g. `Mlp(Backend::Cuda)` → `"KILN_USE_KILN_TENSOR_MLP_CUDA"`.
    pub fn env_var(self) -> String {
        let (op, be) = match self {
            KilnTensorOp::Mlp(b) => ("MLP", b),
            KilnTensorOp::Attention(b) => ("ATTN", b),
            KilnTensorOp::Glue(b) => ("GLUE", b),
            KilnTensorOp::Mtp(b) => ("MTP", b),
            KilnTensorOp::Backward(b) => ("BWD", b),
            KilnTensorOp::Optimizer(b) => ("OPTIM", b),
        };
        format!("KILN_USE_KILN_TENSOR_{op}_{}", be.name())
    }

    /// Is this flag set in the current environment?
    ///
    /// All flags default to **off**. Phase 7 removes the candle path
    /// once every flag's tests are green in CI.
    pub fn is_enabled(self) -> bool {
        env_flag(&self.env_var(), false)
    }

    /// Iterator over every `(op, backend)` cell. Used by the preserve-
    /// list audit and the substrate-status dashboard.
    pub fn all() -> impl Iterator<Item = KilnTensorOp> {
        const BACKENDS: [Backend; 4] = [
            Backend::Cuda,
            Backend::Metal,
            Backend::Vulkan,
            Backend::Cpu,
        ];
        BACKENDS.iter().flat_map(|&b| {
            [
                KilnTensorOp::Mlp(b),
                KilnTensorOp::Attention(b),
                KilnTensorOp::Glue(b),
                KilnTensorOp::Mtp(b),
                KilnTensorOp::Backward(b),
                KilnTensorOp::Optimizer(b),
            ]
            .into_iter()
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_var_format_is_stable() {
        assert_eq!(
            KilnTensorOp::Mlp(Backend::Cuda).env_var(),
            "KILN_USE_KILN_TENSOR_MLP_CUDA"
        );
        assert_eq!(
            KilnTensorOp::Attention(Backend::Metal).env_var(),
            "KILN_USE_KILN_TENSOR_ATTN_METAL"
        );
        assert_eq!(
            KilnTensorOp::Glue(Backend::Vulkan).env_var(),
            "KILN_USE_KILN_TENSOR_GLUE_VULKAN"
        );
        assert_eq!(
            KilnTensorOp::Mtp(Backend::Cuda).env_var(),
            "KILN_USE_KILN_TENSOR_MTP_CUDA"
        );
        assert_eq!(
            KilnTensorOp::Backward(Backend::Cuda).env_var(),
            "KILN_USE_KILN_TENSOR_BWD_CUDA"
        );
        assert_eq!(
            KilnTensorOp::Optimizer(Backend::Cuda).env_var(),
            "KILN_USE_KILN_TENSOR_OPTIM_CUDA"
        );
    }

    #[test]
    fn all_iterates_24_cells() {
        let cells: Vec<_> = KilnTensorOp::all().collect();
        assert_eq!(cells.len(), 4 * 6);
    }

    #[test]
    fn flags_default_off() {
        let _g = crate::env_flag::TEST_ENV_LOCK.lock().unwrap();
        // Clear any pre-set kiln-tensor flag from a neighbor test
        // (env mutex protects against interleave).
        for op in KilnTensorOp::all() {
            unsafe {
                std::env::remove_var(op.env_var());
            }
        }
        for op in KilnTensorOp::all() {
            assert!(!op.is_enabled(), "{:?} should default off", op);
        }
    }

    #[test]
    fn flag_can_be_set() {
        let _g = crate::env_flag::TEST_ENV_LOCK.lock().unwrap();
        let op = KilnTensorOp::Mlp(Backend::Cuda);
        unsafe {
            std::env::set_var(op.env_var(), "1");
        }
        assert!(op.is_enabled());
        unsafe {
            std::env::remove_var(op.env_var());
        }
        assert!(!op.is_enabled());
    }

    #[test]
    fn backend_name_is_stable() {
        assert_eq!(Backend::Cuda.name(), "CUDA");
        assert_eq!(Backend::Metal.name(), "METAL");
        assert_eq!(Backend::Vulkan.name(), "VULKAN");
        assert_eq!(Backend::Cpu.name(), "CPU");
    }
}
