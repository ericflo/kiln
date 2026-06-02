//! `BackendMatmul` — the abstract API every per-backend matmul impl
//! satisfies.
//!
//! Per the Phase 2 issue bullet:
//!
//! > BLAS crate: `kiln-blas` (cublasLt) / `kiln-mps`
//! > (MPSMatrixMultiplication + transposed-coop GEMV port) /
//! > `kiln-vulkan-blas` (extend `kiln-vulkan-kernel::vk_ops/matmul*`)
//!
//! `BackendMatmul` is the seam — one trait that the cublasLt
//! handle, the MPS handle, and the Vulkan compute pipeline all
//! implement. Forward.rs reaches for `dyn BackendMatmul` and is
//! free of per-backend conditionals.
//!
//! # API shape
//!
//! - [`MatmulRequest`] — pure-data descriptor: shape + dtype + layout
//!   + alpha/beta + epilogue. CPU-buildable; no GPU types.
//! - [`Epilogue`] — the cublasLt-inspired fused-tail menu, identical
//!   across backends (`Identity`, `Bias`, `Relu`, `Gelu`, `Silu`,
//!   `BiasSilu`, `BiasGelu`).
//! - [`BackendMatmul`] trait — what each backend's `MatmulHandle`
//!   implements once the GPU-specific impl lands.
//! - [`MatmulOutcome`] — return shape for an executed matmul:
//!   `bytes_written` + `algo_blob` to feed back into `AlgoCache`.

use std::fmt;

use crate::{AlgoCacheKey, AlgoCacheValue};

/// Layout of a matmul operand: row-major (`[M, K]` traversed as
/// `row * K + col`) vs col-major (`row + col * M`). Mirrors
/// cublasLt's `CUBLASLT_ORDER_ROW` vs `CUBLASLT_ORDER_COL`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulLayout {
    RowMajor,
    ColMajor,
}

impl MatmulLayout {
    pub const fn name(self) -> &'static str {
        match self {
            MatmulLayout::RowMajor => "row",
            MatmulLayout::ColMajor => "col",
        }
    }
}

impl fmt::Display for MatmulLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Fused tail (cublasLt epilogue, MPS post-op, Vulkan compute
/// epilogue). The menu is shared so backends pick the same fused
/// path for the same logical operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Epilogue {
    /// No fused operation. Plain `C = α A B + β C`.
    Identity,
    /// Add a per-output-column bias vector.
    Bias,
    /// ReLU activation.
    Relu,
    /// GELU activation.
    Gelu,
    /// SiLU / swish activation.
    Silu,
    /// Bias + SiLU. Used by the gate||up MLP step.
    BiasSilu,
    /// Bias + GELU.
    BiasGelu,
}

impl Epilogue {
    /// Stable name suitable for cache keys + receipts.
    pub const fn name(self) -> &'static str {
        match self {
            Epilogue::Identity => "identity",
            Epilogue::Bias => "bias",
            Epilogue::Relu => "relu",
            Epilogue::Gelu => "gelu",
            Epilogue::Silu => "silu",
            Epilogue::BiasSilu => "bias_silu",
            Epilogue::BiasGelu => "bias_gelu",
        }
    }
}

impl fmt::Display for Epilogue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Pure-data matmul descriptor. The shape and dtype an autotune
/// search keys on.
///
/// This is intentionally backend-agnostic — pointer types live in
/// each backend's `MatmulHandle::execute` argument list, never in
/// this descriptor. The descriptor is what the autotune cache reads
/// to compute a cache key.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct MatmulRequest {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    /// `dtype.to_string()` from `kiln_tensor::DType`. Wrapped as
    /// `String` so this crate can stay candle/kiln-tensor-free at
    /// the type level.
    pub dtype: String,
    pub a_layout: MatmulLayout,
    pub b_layout: MatmulLayout,
    pub c_layout: MatmulLayout,
    pub epilogue: Epilogue,
    /// Concurrent-streams hint for the autotune key. Phase 5 hot
    /// path runs QKV on three streams; the optimal algo at
    /// `concurrent_streams=3` is generally SM-light vs the
    /// `concurrent_streams=1` choice.
    pub concurrent_streams: u8,
}

impl MatmulRequest {
    /// Build an `AlgoCacheKey` for this request, suitable to query /
    /// insert into a shared [`crate::AlgoCache`]. Reads the major
    /// version of the kiln-blas crate as the cache namespace.
    pub fn cache_key(&self) -> AlgoCacheKey {
        AlgoCacheKey {
            shape: [self.m, self.n, self.k],
            input_dtype: self.dtype.clone(),
            output_dtype: self.dtype.clone(),
            compute_dtype: "f32".to_string(),
            transpose: [
                matches!(self.a_layout, MatmulLayout::ColMajor),
                matches!(self.b_layout, MatmulLayout::ColMajor),
            ],
            expected_concurrent_streams: self.concurrent_streams,
            kiln_version_major: env!("CARGO_PKG_VERSION_MAJOR").parse().unwrap_or(0),
        }
    }
}

/// What a successful matmul returns. Forwarded into `AlgoCache`
/// after the first execution so subsequent runs of the same shape
/// skip the heuristic search.
#[derive(Debug, Clone)]
pub struct MatmulOutcome {
    /// Bytes written to the output tensor (typically `m * n *
    /// dtype_size`). Used by VRAM accounting.
    pub bytes_written: u64,
    /// Best-effort millisecond timing for the kernel. `None` when
    /// timing was disabled (probe binaries record; production
    /// dispatch does not).
    pub elapsed_ms: Option<f32>,
    /// Opaque per-backend algo blob (cublasLt algo, MPS descriptor,
    /// Vulkan workgroup config). To be saved into `AlgoCache` so the
    /// next call to the same shape picks the same algo without
    /// re-running the heuristic.
    pub algo_blob: AlgoCacheValue,
}

/// What every per-backend `MatmulHandle` exposes.
///
/// Phase 2 ships the abstract trait + the descriptor types here;
/// per-backend impls (kiln-blas's cublasLt path, kiln-mps's MPS
/// path, kiln-vulkan-blas's compute-shader path) plug in behind
/// their respective `--features probe`/`--features cuda`/etc gates.
pub trait BackendMatmul: std::fmt::Debug + Send + Sync {
    /// Stable backend name (`"cublasLt"`, `"mps"`, `"vulkan"`).
    fn backend_name(&self) -> &'static str;

    /// Run the matmul described by `req`. Pointer arguments live in
    /// backend-specific extension traits — this trait covers the
    /// shape/dtype/epilogue selection only, so the abstraction stays
    /// CPU-buildable.
    ///
    /// On success, returns a [`MatmulOutcome`] the caller writes back
    /// into the shared `AlgoCache`.
    fn plan(&self, req: &MatmulRequest) -> MatmulOutcome;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_key_combines_shape_and_layout() {
        let req = MatmulRequest {
            m: 2048,
            n: 18432,
            k: 2560,
            dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::ColMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::BiasSilu,
            concurrent_streams: 1,
        };
        let k = req.cache_key();
        assert_eq!(k.shape, [2048, 18432, 2560]);
        assert_eq!(k.input_dtype, "bf16");
        assert_eq!(k.transpose, [false, true]); // A row → false; B col → true
        assert_eq!(k.expected_concurrent_streams, 1);
    }

    #[test]
    fn epilogue_names_are_stable() {
        assert_eq!(Epilogue::Identity.name(), "identity");
        assert_eq!(Epilogue::Bias.name(), "bias");
        assert_eq!(Epilogue::Relu.name(), "relu");
        assert_eq!(Epilogue::Gelu.name(), "gelu");
        assert_eq!(Epilogue::Silu.name(), "silu");
        assert_eq!(Epilogue::BiasSilu.name(), "bias_silu");
        assert_eq!(Epilogue::BiasGelu.name(), "bias_gelu");
    }

    #[test]
    fn matmul_layout_display() {
        assert_eq!(format!("{}", MatmulLayout::RowMajor), "row");
        assert_eq!(format!("{}", MatmulLayout::ColMajor), "col");
    }

    #[derive(Debug)]
    struct DummyMatmul;
    impl BackendMatmul for DummyMatmul {
        fn backend_name(&self) -> &'static str {
            "dummy"
        }
        fn plan(&self, req: &MatmulRequest) -> MatmulOutcome {
            MatmulOutcome {
                bytes_written: req.m * req.n * 2, // BF16
                elapsed_ms: None,
                algo_blob: AlgoCacheValue {
                    algo_id: -1,
                    workspace_bytes: 0,
                    recorded_ms: 0.0,
                    algo_blob: vec![0, 1, 2, 3],
                },
            }
        }
    }

    #[test]
    fn dispatch_through_trait() {
        let h: Box<dyn BackendMatmul> = Box::new(DummyMatmul);
        let req = MatmulRequest {
            m: 4,
            n: 4,
            k: 4,
            dtype: "f32".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        };
        let outcome = h.plan(&req);
        assert_eq!(outcome.bytes_written, 32);
        assert_eq!(h.backend_name(), "dummy");
    }

    #[test]
    fn concurrent_streams_in_cache_key() {
        let base = MatmulRequest {
            m: 1024,
            n: 1024,
            k: 1024,
            dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        };
        let mut three = base.clone();
        three.concurrent_streams = 3;
        assert_ne!(base.cache_key(), three.cache_key());
    }
}
