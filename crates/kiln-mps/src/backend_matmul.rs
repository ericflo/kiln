//! Metal-side [`BackendMatmul`] adapter.
//!
//! Phase 2.2 of #1082. Ships the backend-agnostic adapter that an
//! eventual `MPSMatmulHandle` (Phase 2.x, feature-gated) plugs into.
//! Today the planner returns the **resolved tile policy** from
//! [`MpsTilePolicy`] — the actual Metal command-queue dispatch
//! lands when the `--features probe` MPS path comes online.
//!
//! # Why a Metal-specific adapter
//!
//! Different backends pick algos along different axes — cublasLt
//! picks an `algo_id` + `workspace_bytes`; MPS picks tile/transpose
//! configuration. The shared [`BackendMatmul`] trait answers
//! "give me a plan for this matmul"; this adapter answers it
//! Metal-side.

use kiln_blas::{
    AlgoCacheValue, BackendMatmul, MatmulOutcome, MatmulRequest,
};

use crate::{MpsTilePolicy, MpsUmaHint};

/// Metal-side adapter. Resolves a [`MpsTilePolicy`] for the
/// requested shape and packs it as the cache `algo_blob`. The
/// actual Metal dispatch lands behind the `probe` feature when the
/// `MPSMatmulHandle` ships.
#[derive(Debug, Default, Clone)]
pub struct MpsBackendMatmul {
    /// UMA hint the dispatch path will honor. On M-series this
    /// picks `MTLStorageModeShared`; on discrete Macs falls back
    /// to `Private`.
    pub uma_hint: MpsUmaHint,
}

impl MpsBackendMatmul {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_uma_hint(uma_hint: MpsUmaHint) -> Self {
        Self { uma_hint }
    }
}

impl BackendMatmul for MpsBackendMatmul {
    fn backend_name(&self) -> &'static str {
        "mps"
    }

    fn plan(&self, req: &MatmulRequest) -> MatmulOutcome {
        // Resolve a tile policy from the request shape. Heuristic:
        // small shapes use the GEMV-friendly tile; medium use the
        // square tile; large use the elongated MLP tile.
        let policy = MpsTilePolicy::recommended_for(req.m, req.n, req.k);
        let bytes_per_element: u64 = match req.dtype.as_str() {
            "f32" => 4,
            "bf16" | "f16" => 2,
            "u8" | "f8_e4m3" | "f8_e5m2" => 1,
            _ => 4,
        };
        let bytes_written = req.m * req.n * bytes_per_element;
        let algo_blob = policy.serialize();
        MatmulOutcome {
            bytes_written,
            elapsed_ms: None,
            algo_blob: AlgoCacheValue {
                algo_id: -1,
                workspace_bytes: 0,
                recorded_ms: 0.0,
                algo_blob,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_blas::{Epilogue, MatmulLayout, MatmulRequest};

    fn req(m: u64, n: u64, k: u64) -> MatmulRequest {
        MatmulRequest {
            m,
            n,
            k,
            dtype: "bf16".to_string(),
            a_layout: MatmulLayout::RowMajor,
            b_layout: MatmulLayout::RowMajor,
            c_layout: MatmulLayout::RowMajor,
            epilogue: Epilogue::Identity,
            concurrent_streams: 1,
        }
    }

    #[test]
    fn backend_name_is_mps() {
        assert_eq!(MpsBackendMatmul::new().backend_name(), "mps");
    }

    #[test]
    fn plan_returns_nonzero_algo_blob() {
        let h = MpsBackendMatmul::new();
        let outcome = h.plan(&req(2048, 18432, 2560));
        assert!(!outcome.algo_blob.algo_blob.is_empty());
        assert_eq!(outcome.bytes_written, 2048 * 18432 * 2);
    }

    #[test]
    fn uma_hint_is_respected() {
        let h = MpsBackendMatmul::with_uma_hint(MpsUmaHint::SharedUma);
        assert!(matches!(h.uma_hint, MpsUmaHint::SharedUma));
    }

    #[test]
    fn plan_picks_per_shape_policy() {
        let h = MpsBackendMatmul::new();
        let small = h.plan(&req(1, 256, 256));
        let large = h.plan(&req(8192, 18432, 2560));
        // Different shapes should resolve to different tile policies
        // → different algo_blob bytes.
        assert_ne!(small.algo_blob.algo_blob, large.algo_blob.algo_blob);
    }
}
