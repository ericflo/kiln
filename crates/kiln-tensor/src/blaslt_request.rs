//! Shared CUDA/ROCm BLASLt request descriptors.
//!
//! `kiln-blas` and `kiln-rocblas` expose separate concrete request types so the
//! CUDA and ROCm crates can stay independently feature-gated. This module owns
//! the pure tensor-side projection that is identical before each backend maps it
//! into its concrete BLAS crate type. Keep this shared request home in
//! `kiln-tensor`, where CUDA and ROCm tensor matmul callers both depend on it,
//! instead of moving it into the model-layer `cuda_rocm_common` helpers.

use crate::{DType, Error, Result};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BlasLtMatmulLayout {
    RowMajor,
    ColMajor,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum BlasLtEpilogue {
    Identity,
    Bias,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct BlasLtMatmulRequest {
    pub(crate) m: u64,
    pub(crate) n: u64,
    pub(crate) k: u64,
    pub(crate) dtype: DType,
    pub(crate) a_layout: BlasLtMatmulLayout,
    pub(crate) b_layout: BlasLtMatmulLayout,
    pub(crate) c_layout: BlasLtMatmulLayout,
    pub(crate) epilogue: BlasLtEpilogue,
    pub(crate) concurrent_streams: u8,
}

impl BlasLtMatmulRequest {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        m: usize,
        n: usize,
        k: usize,
        dtype: DType,
        a_layout: BlasLtMatmulLayout,
        b_layout: BlasLtMatmulLayout,
        c_layout: BlasLtMatmulLayout,
        epilogue: BlasLtEpilogue,
        concurrent_streams: u8,
        op: &str,
    ) -> Result<Self> {
        if concurrent_streams == 0 {
            return Err(Error::Msg(format!(
                "{op}: concurrent_streams must be non-zero"
            )));
        }
        blaslt_dtype_name(dtype, op)?;
        Ok(Self {
            m: m as u64,
            n: n as u64,
            k: k as u64,
            dtype,
            a_layout,
            b_layout,
            c_layout,
            epilogue,
            concurrent_streams,
        })
    }

    pub(crate) fn dtype_name(self) -> &'static str {
        self.dtype.short_name()
    }
}

pub(crate) fn blaslt_dtype_name(dtype: DType, op: &str) -> Result<&'static str> {
    match dtype {
        DType::F32 | DType::BF16 | DType::F16 => Ok(dtype.short_name()),
        other => Err(Error::Msg(format!("{op}: unsupported dtype {other}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blaslt_request_accepts_cuda_rocm_dtype_envelope() {
        for dtype in [DType::F32, DType::BF16, DType::F16] {
            let request = BlasLtMatmulRequest::new(
                2,
                4,
                3,
                dtype,
                BlasLtMatmulLayout::RowMajor,
                BlasLtMatmulLayout::ColMajor,
                BlasLtMatmulLayout::RowMajor,
                BlasLtEpilogue::Bias,
                1,
                "test_blaslt",
            )
            .expect("supported BLASLt dtype should project");
            assert_eq!(request.dtype_name(), dtype.short_name());
        }
    }

    #[test]
    fn blaslt_request_rejects_unsupported_dtype_and_zero_streams() {
        assert!(matches!(
            BlasLtMatmulRequest::new(
                2,
                4,
                3,
                DType::U8,
                BlasLtMatmulLayout::RowMajor,
                BlasLtMatmulLayout::RowMajor,
                BlasLtMatmulLayout::RowMajor,
                BlasLtEpilogue::Identity,
                1,
                "test_blaslt",
            ),
            Err(Error::Msg(_))
        ));
        assert!(matches!(
            BlasLtMatmulRequest::new(
                2,
                4,
                3,
                DType::F32,
                BlasLtMatmulLayout::RowMajor,
                BlasLtMatmulLayout::RowMajor,
                BlasLtMatmulLayout::RowMajor,
                BlasLtEpilogue::Identity,
                0,
                "test_blaslt",
            ),
            Err(Error::Msg(_))
        ));
    }
}
