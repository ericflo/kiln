//! ROCm wrappers for the is_finite_reduce kernel(s) (Phase R.5).
use crate::{DType, Device, Error, Result, RocmStorage, StorageBackend, Tensor};

// The ROCm-side launcher for `csrc/is_finite_reduce.cu`, compiled into
// `libkiln_tensor_rocm_ops.a` by `build.rs::build_rocm()`. Same stable C ABI
// (identical symbol + signature) as the CUDA build's declaration in
// `cuda_storage.rs`, copied verbatim.
unsafe extern "C" {
    fn kiln_is_finite_storage_async(
        x: *const core::ffi::c_void,
        out_flag: *mut core::ffi::c_void,
        n_elements: i64,
        dtype_tag: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn rocm_is_finite_host_scan_threshold() -> Option<usize> {
    let disabled = std::env::var("KILN_ROCM_IS_FINITE_LARGE_HOST_SCAN")
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .is_some_and(|value| matches!(value.as_str(), "0" | "false" | "no" | "off"));
    if disabled {
        return None;
    }

    Some(
        std::env::var("KILN_ROCM_IS_FINITE_HOST_SCAN_ELEMENTS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|&value| value > 0)
            .unwrap_or(16 * 1024 * 1024),
    )
}

/// "Any non-finite?" tensor-wide reduction on a ROCm-resident tensor. Returns
/// `Ok(true)` if every element is finite (no NaN, no `+Inf`, no `-Inf`),
/// `Ok(false)` otherwise. ROCm analog of `cuda_is_finite`, routing through the
/// `kiln_is_finite_storage_async` launcher in `csrc/is_finite_reduce.cu`.
///
/// The kernel is a grid-wide `atomicOr` into a single u32 device buffer (NOT a
/// cross-lane warp-shuffle reduction), so it is wave-size correct on both
/// wave32 and wave64 as-is — no block-reduce fix needed. We issue exactly one
/// 4-byte D2H to read the flag back.
///
/// Supported dtypes: F32, BF16, F16, F8E4M3, F8E5M2. Integer dtypes
/// (U8/U32/I64) and packed dtypes (Int4Packed/Fp4Packed) have no NaN/Inf
/// representation and are vacuously finite (early-return, matching
/// `Tensor::all_finite()` / `cuda_is_finite`).
///
/// Non-contiguous inputs are contiguified via [`crate::rocm_contiguous`] before
/// launch (the kernel walks `[0..n_elements)` directly).
pub fn rocm_is_finite(src: &Tensor) -> Result<bool> {
    let dtype = src.dtype();

    // Integer + packed dtypes have no NaN/Inf — vacuously finite.
    if matches!(dtype, DType::U8 | DType::U32 | DType::I64) {
        return Ok(true);
    }
    if dtype.is_packed() {
        return Ok(true);
    }

    let dtype_tag: i32 = match dtype {
        DType::F32 => 0,
        DType::BF16 => 1,
        DType::F16 => 2,
        DType::F8E4M3 => 3,
        DType::F8E5M2 => 4,
        other => {
            return Err(Error::Msg(format!(
                "rocm_is_finite: unsupported dtype {other}"
            )));
        }
    };

    // The GPU reducer is a diagnostic/anomaly helper. On very large ROCm BF16
    // tensors from long-context training, the reducer path can perturb the
    // following confirmation readback on ROCm 7.2/gfx115x; a direct D2H scan is
    // slower but authoritative and only used when finite checking is enabled.
    if rocm_is_finite_host_scan_threshold()
        .is_some_and(|threshold| src.element_count() >= threshold)
    {
        let host = crate::rocm_to_host_copy(src)?;
        return host.all_finite();
    }

    // Force a contiguous, `start_offset = 0` device buffer. The kernel walks
    // `[0..n_elements)` directly; non-contiguous strided inputs would otherwise
    // need a separate stride-walking kernel.
    let contig = crate::rocm_contiguous(src)?;
    let contig_storage = contig
        .storage()
        .as_any()
        .downcast_ref::<RocmStorage>()
        .ok_or_else(|| {
            Error::Msg("rocm_is_finite: contiguous'd storage must be RocmStorage".to_string())
        })?;
    let ctx = contig_storage.context();
    // RocmStorage's `device` field is private to its module; use the contig
    // tensor's public `.device()` (rocm_contiguous always lands on ROCm).
    let device_index = match contig.device() {
        Device::Rocm(i) => i,
        _ => unreachable!("rocm_is_finite: contig storage device must be Rocm"),
    };

    // 1-element U32 device buffer (4 bytes, zero-init). Kernel atomic-ORs a `1`
    // into it on the first non-finite hit.
    let flag_storage = RocmStorage::zeros_ctx(&ctx, device_index, DType::U32, 1)?;

    let stream_submission = contig_storage.rocm_stream_submission()?;
    let raw_stream = stream_submission.raw_stream();

    // `rocm_contiguous` produces start_offset == 0, so no byte-offset math.
    let (x_base, _) = contig_storage.device_ptr_raw();
    let (flag_base, _) = flag_storage.device_ptr_raw();
    let x_ptr = x_base as *const core::ffi::c_void;
    let flag_ptr = flag_base as *mut core::ffi::c_void;

    let n_elements = src.element_count() as i64;
    let status =
        unsafe { kiln_is_finite_storage_async(x_ptr, flag_ptr, n_elements, dtype_tag, raw_stream) };
    if status != 0 {
        stream_submission.quarantine();
        return Err(Error::Msg(format!(
            "rocm_is_finite: FFI returned status {status}"
        )));
    }
    stream_submission.complete();

    // Read the 4-byte flag back. `memcpy_dtoh` synchronizes against the launch
    // on the same stream.
    let stream = crate::active_rocm_stream(&ctx);
    let flag_host = stream
        .memcpy_dtoh(flag_storage.slice())
        .map_err(|e| Error::Msg(format!("rocm_is_finite: flag D2H failed: {e:?}")))?;
    if flag_host.len() < 4 {
        return Err(Error::Msg(format!(
            "rocm_is_finite: flag D2H returned {} bytes, expected 4",
            flag_host.len()
        )));
    }
    let flag = u32::from_le_bytes([flag_host[0], flag_host[1], flag_host[2], flag_host[3]]);

    // Keep the contiguified input Tensor (and thus its device buffer) alive
    // until *after* the D2H readback. `memcpy_dtoh` synchronizes the stream, so
    // the async kernel launch has fully drained — and read the live input — by
    // the time the flag is back. Without this anchor, NLL could free `contig`
    // right after the pointer was captured, racing the in-flight launch.
    let _input_keepalive = &contig;

    if flag == 0 {
        return Ok(true);
    }

    // Defensive confirmation path: the ROCm reducer is a diagnostic/anomaly
    // substrate, so false positives are worse than a rare D2H confirmation. Large
    // BF16 tensors have produced intermittent device-flag positives while a CPU
    // scan of the same tensor found every element finite. Confirm suspected
    // failures with the canonical CPU stride walker before aborting training.
    let host = crate::rocm_to_host_copy(src)?;
    host.all_finite()
}
