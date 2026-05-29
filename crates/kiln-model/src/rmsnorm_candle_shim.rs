//! Candle-typed glue for the fused RMSNorm / RoPE / sigmoid-mul / LoRA /
//! depthwise-conv1d CUDA kernels — relocated here from
//! `kiln-rmsnorm-kernel` in (#1082) so that the kernel crate can drop
//! its `candle-core` dependency entirely.
//!
//! # Why this module exists
//!
//! `kiln-rmsnorm-kernel` exposes two parallel surfaces over the same
//! `kiln_*` CUDA FFI symbols:
//!
//! * a pure-`kiln_tensor` (kt) surface (`kt_api`, `kt_tape`) used by the
//!   inference + tape-training paths, and
//! * a candle-typed surface (the `*_storage` / `matmul_f32_bf16w` /
//!   `causal_depthwise_conv1d_f32*` wrappers + the `KtForwardOp2`
//!   autograd shim) used ONLY by candle-autograd training call sites in
//!   `kiln-model`.
//!
//! The candle-typed surface was the kernel crate's last remaining use of
//! `candle-core`. Following the proven OPD pattern
//! (`kiln-train::opd_candle_shim`, commit `f214f168`), the candle-typed
//! glue moves UP into the consumer crate (`kiln-model`, which already
//! depends on candle), while the pure-kt building blocks + the raw
//! `extern "C"` FFI declarations stay in the kernel crate.
//!
//! # FFI symbol resolution
//!
//! The `kiln_*` CUDA entry points are compiled into
//! `libkiln_rmsnorm_kernel.a` by the kernel crate's `build.rs` (cc +
//! nvcc) and linked via `cargo:rustc-link-lib=static=kiln_rmsnorm_kernel`.
//! Because `kiln-model` depends on `kiln-rmsnorm-kernel` (and uses its
//! kt surface heavily), that static archive is linked into every
//! `kiln-model` build target. The `extern "C"` block below is a
//! *declaration* (not a definition) of the subset of those symbols this
//! module needs; it resolves to the exact same linker symbols the kernel
//! crate's own kt surface uses — no duplicate-definition issue (extern
//! declarations may appear in multiple crates; only one definition
//! exists, in the cc-compiled archive).
//!
//! # Byte-identical relocation
//!
//! The wrapper bodies below are byte-identical to their former homes in
//! `kiln-rmsnorm-kernel/src/lib.rs` and
//! `kiln-rmsnorm-kernel/src/kt_forward_op.rs`. Only the crate paths
//! changed: `crate::kiln_*` FFI calls now resolve to the locally
//! re-declared `extern "C"` block; `crate::kt_api::*` calls now resolve
//! to the kernel crate's public re-exports (`kiln_rmsnorm_kernel::*`).

use candle_core::{
    backend::BackendStorage, cuda_backend::cudarc::driver::DevicePtr, CudaStorage, DType, Device,
    Layout, Result, Tensor,
};
use half::bf16;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// FFI — the subset of `kiln_*` symbols the candle-typed wrappers below need.
//
// These are *declarations only*; the definitions live in
// `libkiln_rmsnorm_kernel.a` (cc+nvcc, kernel crate build.rs). The kernel
// crate keeps its own copy of these (plus the rest) for its kt surface;
// having matching extern declarations here is standard Rust FFI and
// resolves to the same single linker symbol.
// ---------------------------------------------------------------------------
unsafe extern "C" {
    fn kiln_f32_to_bf16(
        src: *const f32,
        dst: *mut core::ffi::c_void,
        n: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rotary_one(
        x: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        out: *mut core::ffi::c_void,
        batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_rotary_one_bwd(
        grad_y: *const core::ffi::c_void,
        cos: *const f32,
        sin: *const f32,
        grad_x: *mut core::ffi::c_void,
        batch: i32,
        seq_len: i32,
        heads: i32,
        head_dim: i32,
        rotary_dim: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_fused_sigmoid_mul_bf16(
        x: *const core::ffi::c_void,
        gate: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        n: i64,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lora_decode_add_bf16(
        base: *const core::ffi::c_void,
        hidden: *const f32,
        b: *const core::ffi::c_void,
        out: *mut core::ffi::c_void,
        scale: f32,
        rows: i32,
        out_dim: i32,
        rank: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_lora_add_inplace_f32(
        base: *mut f32,
        hidden: *const f32,
        b: *const f32,
        scale: f32,
        rows: i32,
        out_dim: i32,
        rank: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_f32(
        input: *const f32,
        weight: *const f32,
        state: *const f32,
        out: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_inplace_f32(
        input_out: *mut f32,
        weight: *const f32,
        state: *const f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_input_f32(
        grad_out: *const f32,
        weight: *const f32,
        grad_input: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_weight_f32(
        grad_out: *const f32,
        input: *const f32,
        state: *const f32,
        grad_weight: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;

    fn kiln_causal_depthwise_conv1d_bwd_state_f32(
        grad_out: *const f32,
        weight: *const f32,
        grad_state: *mut f32,
        rows: i32,
        channels: i32,
        kernel: i32,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

fn cuda_empty_kernel_outputs_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("KILN_DISABLE_CUDA_EMPTY_KERNEL_OUTPUTS").is_err())
}

/// Whether the fused RMSNorm kernel is available on the given tensor.
///
/// The kernel only supports CUDA + bf16 + contiguous + hidden <= 8192.
pub fn supports(x: &Tensor, weight: &Tensor) -> bool {
    matches!(x.device(), Device::Cuda(_))
        && x.dtype() == DType::BF16
        && weight.dtype() == DType::BF16
        && x.is_contiguous()
        && weight.is_contiguous()
        && x.rank() >= 1
        && x.dims().last().copied().unwrap_or(0) <= 8192
        && weight.dims() == &[x.dims().last().copied().unwrap_or(0)]
}

/// Storage-level single-tensor BF16 RoPE for CUDA custom ops.
pub fn rotary_one_bf16_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    x_cuda: &CudaStorage,
    x_layout: &Layout,
    cos_cuda: &CudaStorage,
    cos_layout: &Layout,
    sin_cuda: &CudaStorage,
    sin_layout: &Layout,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    let x_dims = x_layout.dims();
    let out_dims = out_layout.dims();
    if x_dims.len() != 4 || out_dims != x_dims {
        candle_core::bail!(
            "rotary_one_bf16_storage: x/out must be matching rank-4 tensors, got x={x_dims:?} out={out_dims:?}"
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "rotary_one_bf16_storage: invalid head_dim={head_dim} rotary_dim={rotary_dim}"
        );
    }
    let (batch, seq_len, heads, x_head_dim) = (x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
    if x_head_dim != head_dim {
        candle_core::bail!(
            "rotary_one_bf16_storage: x head dim {x_head_dim} != head_dim {head_dim}"
        );
    }
    if cos_layout.dims() != [seq_len, rotary_dim / 2]
        || sin_layout.dims() != [seq_len, rotary_dim / 2]
    {
        candle_core::bail!(
            "rotary_one_bf16_storage: table shape mismatch cos={:?} sin={:?} expected=[{seq_len}, {}]",
            cos_layout.dims(),
            sin_layout.dims(),
            rotary_dim / 2
        );
    }
    if out_cuda.dtype() != DType::BF16
        || x_cuda.dtype() != DType::BF16
        || cos_cuda.dtype() != DType::F32
        || sin_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "rotary_one_bf16_storage: expected out/x BF16 and cos/sin F32, got out={:?} x={:?} cos={:?} sin={:?}",
            out_cuda.dtype(),
            x_cuda.dtype(),
            cos_cuda.dtype(),
            sin_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous()
        || !x_layout.is_contiguous()
        || !cos_layout.is_contiguous()
        || !sin_layout.is_contiguous()
    {
        candle_core::bail!("rotary_one_bf16_storage: tensors must be contiguous");
    }
    if batch > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || heads > i32::MAX as usize
        || head_dim > i32::MAX as usize
        || rotary_dim > i32::MAX as usize
    {
        candle_core::bail!("rotary_one_bf16_storage: dimensions exceed i32 kernel envelope");
    }

    let stream = x_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(x_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<f32>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<f32>()?
        .slice(sin_layout.start_offset()..);
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);

    let status = unsafe {
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_fused_rotary_one(
            x_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            out_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_rotary_one failed with status {status}");
    }
    Ok(())
}

/// Whether the fused single-tensor BF16 RoPE backward kernel is available.
pub fn supports_rotary_one_bwd_bf16(
    grad_y: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> bool {
    if !matches!(grad_y.device(), Device::Cuda(_))
        || !matches!(cos.device(), Device::Cuda(_))
        || !matches!(sin.device(), Device::Cuda(_))
        || grad_y.dtype() != DType::BF16
        || cos.dtype() != DType::F32
        || sin.dtype() != DType::F32
        || !grad_y.is_contiguous()
        || !cos.is_contiguous()
        || !sin.is_contiguous()
        || grad_y.rank() != 4
        || rotary_dim == 0
        || rotary_dim > head_dim
        || rotary_dim % 2 != 0
    {
        return false;
    }
    let dims = grad_y.dims();
    let batch = dims[0];
    let seq_len = dims[1];
    let heads = dims[2];
    dims[3] == head_dim
        && cos.dims() == [seq_len, rotary_dim / 2]
        && sin.dims() == [seq_len, rotary_dim / 2]
        && batch <= i32::MAX as usize
        && seq_len <= i32::MAX as usize
        && heads <= i32::MAX as usize
        && head_dim <= i32::MAX as usize
        && rotary_dim <= i32::MAX as usize
}

/// Run fused single-tensor BF16 RoPE backward.
pub fn rotary_one_bwd_bf16(
    grad_y: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<Tensor> {
    if !supports_rotary_one_bwd_bf16(grad_y, cos, sin, head_dim, rotary_dim) {
        candle_core::bail!(
            "kiln-rmsnorm-kernel: rotary_one_bwd unsupported shapes grad_y={:?} cos={:?} sin={:?} dtypes=({:?},{:?},{:?}) head_dim={head_dim} rotary_dim={rotary_dim}",
            grad_y.shape(),
            cos.shape(),
            sin.shape(),
            grad_y.dtype(),
            cos.dtype(),
            sin.dtype()
        );
    }

    let grad_x = if cuda_empty_kernel_outputs_enabled() {
        unsafe { Tensor::empty(grad_y.dims(), DType::BF16, grad_y.device())? }
    } else {
        Tensor::zeros(grad_y.dims(), DType::BF16, grad_y.device())?
    };

    {
        let (grad_y_storage, grad_y_layout) = grad_y.storage_and_layout();
        let (cos_storage, cos_layout) = cos.storage_and_layout();
        let (sin_storage, sin_layout) = sin.storage_and_layout();
        let (grad_x_storage, grad_x_layout) = grad_x.storage_and_layout();

        let grad_y_cuda = match &*grad_y_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd grad_y must be on CUDA"),
        };
        let cos_cuda = match &*cos_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd cos must be on CUDA"),
        };
        let sin_cuda = match &*sin_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd sin must be on CUDA"),
        };
        let grad_x_cuda = match &*grad_x_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-rmsnorm-kernel: rotary_one_bwd grad_x must be on CUDA"),
        };

        rotary_one_bwd_bf16_storage(
            grad_x_cuda,
            &grad_x_layout,
            grad_y_cuda,
            &grad_y_layout,
            cos_cuda,
            &cos_layout,
            sin_cuda,
            &sin_layout,
            head_dim,
            rotary_dim,
        )?;
    }
    Ok(grad_x)
}

/// Storage-level single-tensor BF16 RoPE backward for CUDA custom ops.
///
/// Downgraded from `pub` to private in (#1082): zero external callers —
/// the only consumer is [`rotary_one_bwd_bf16`] above, which still calls
/// this helper after extracting the storage and layouts. Shrinks the
/// candle-typed public surface of this crate by one entry.
fn rotary_one_bwd_bf16_storage(
    grad_x_cuda: &CudaStorage,
    grad_x_layout: &Layout,
    grad_y_cuda: &CudaStorage,
    grad_y_layout: &Layout,
    cos_cuda: &CudaStorage,
    cos_layout: &Layout,
    sin_cuda: &CudaStorage,
    sin_layout: &Layout,
    head_dim: usize,
    rotary_dim: usize,
) -> Result<()> {
    let grad_y_dims = grad_y_layout.dims();
    let grad_x_dims = grad_x_layout.dims();
    if grad_y_dims.len() != 4 || grad_x_dims != grad_y_dims {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: grad_y/grad_x must be matching rank-4 tensors, got grad_y={grad_y_dims:?} grad_x={grad_x_dims:?}"
        );
    }
    if rotary_dim == 0 || rotary_dim > head_dim || rotary_dim % 2 != 0 {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: invalid head_dim={head_dim} rotary_dim={rotary_dim}"
        );
    }
    let (batch, seq_len, heads, grad_head_dim) = (
        grad_y_dims[0],
        grad_y_dims[1],
        grad_y_dims[2],
        grad_y_dims[3],
    );
    if grad_head_dim != head_dim {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: grad_y head dim {grad_head_dim} != head_dim {head_dim}"
        );
    }
    if cos_layout.dims() != [seq_len, rotary_dim / 2]
        || sin_layout.dims() != [seq_len, rotary_dim / 2]
    {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: table shape mismatch cos={:?} sin={:?} expected=[{seq_len}, {}]",
            cos_layout.dims(),
            sin_layout.dims(),
            rotary_dim / 2
        );
    }
    if grad_x_cuda.dtype() != DType::BF16
        || grad_y_cuda.dtype() != DType::BF16
        || cos_cuda.dtype() != DType::F32
        || sin_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "rotary_one_bwd_bf16_storage: expected grad_x/grad_y BF16 and cos/sin F32, got grad_x={:?} grad_y={:?} cos={:?} sin={:?}",
            grad_x_cuda.dtype(),
            grad_y_cuda.dtype(),
            cos_cuda.dtype(),
            sin_cuda.dtype()
        );
    }
    if !grad_x_layout.is_contiguous()
        || !grad_y_layout.is_contiguous()
        || !cos_layout.is_contiguous()
        || !sin_layout.is_contiguous()
    {
        candle_core::bail!("rotary_one_bwd_bf16_storage: tensors must be contiguous");
    }
    if batch > i32::MAX as usize
        || seq_len > i32::MAX as usize
        || heads > i32::MAX as usize
        || head_dim > i32::MAX as usize
        || rotary_dim > i32::MAX as usize
    {
        candle_core::bail!("rotary_one_bwd_bf16_storage: dimensions exceed i32 kernel envelope");
    }

    let stream = grad_y_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let grad_y_slice = grad_y_cuda
        .as_cuda_slice::<bf16>()?
        .slice(grad_y_layout.start_offset()..);
    let cos_slice = cos_cuda
        .as_cuda_slice::<f32>()?
        .slice(cos_layout.start_offset()..);
    let sin_slice = sin_cuda
        .as_cuda_slice::<f32>()?
        .slice(sin_layout.start_offset()..);
    let grad_x_slice = grad_x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(grad_x_layout.start_offset()..);

    let status = unsafe {
        let (grad_y_ptr, _grad_y_guard) = grad_y_slice.device_ptr(&stream);
        let (cos_ptr, _cos_guard) = cos_slice.device_ptr(&stream);
        let (sin_ptr, _sin_guard) = sin_slice.device_ptr(&stream);
        let (grad_x_ptr, _grad_x_guard) = grad_x_slice.device_ptr(&stream);
        kiln_fused_rotary_one_bwd(
            grad_y_ptr as *const _,
            cos_ptr as *const f32,
            sin_ptr as *const f32,
            grad_x_ptr as *mut _,
            batch as i32,
            seq_len as i32,
            heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_rotary_one_bwd failed with status {status}");
    }
    Ok(())
}



pub fn supports_sigmoid_mul(x: &Tensor, gate: &Tensor) -> bool {
    matches!(x.device(), Device::Cuda(_))
        && matches!(gate.device(), Device::Cuda(_))
        && x.dtype() == DType::BF16
        && gate.dtype() == DType::BF16
        && x.is_contiguous()
        && gate.is_contiguous()
        && x.dims() == gate.dims()
        && x.elem_count() <= i64::MAX as usize
}

/// Storage-level fused bf16 `x * sigmoid(gate)` for CUDA custom ops.
///
/// Computes `out = x * sigmoid(gate)` for matching contiguous BF16 CUDA
/// tensors. `out` may alias neither input.
pub fn fused_sigmoid_mul_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    x_cuda: &CudaStorage,
    x_layout: &Layout,
    gate_cuda: &CudaStorage,
    gate_layout: &Layout,
) -> Result<()> {
    let out_dims = out_layout.dims();
    let x_dims = x_layout.dims();
    let gate_dims = gate_layout.dims();
    if out_dims != x_dims || out_dims != gate_dims {
        candle_core::bail!(
            "fused_sigmoid_mul_storage: shape mismatch out={out_dims:?} x={x_dims:?} gate={gate_dims:?}"
        );
    }
    if out_cuda.dtype() != DType::BF16
        || x_cuda.dtype() != DType::BF16
        || gate_cuda.dtype() != DType::BF16
    {
        candle_core::bail!(
            "fused_sigmoid_mul_storage: expected BF16 tensors, got out={:?} x={:?} gate={:?}",
            out_cuda.dtype(),
            x_cuda.dtype(),
            gate_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous() || !x_layout.is_contiguous() || !gate_layout.is_contiguous() {
        candle_core::bail!("fused_sigmoid_mul_storage: tensors must be contiguous");
    }
    let elems: usize = out_dims.iter().product();
    if elems > i64::MAX as usize {
        candle_core::bail!("fused_sigmoid_mul_storage: element count exceeds i64");
    }
    if elems == 0 {
        return Ok(());
    }

    let stream = x_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let x_slice = x_cuda
        .as_cuda_slice::<bf16>()?
        .slice(x_layout.start_offset()..);
    let gate_slice = gate_cuda
        .as_cuda_slice::<bf16>()?
        .slice(gate_layout.start_offset()..);
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);
    let status = unsafe {
        let (x_ptr, _x_guard) = x_slice.device_ptr(&stream);
        let (gate_ptr, _gate_guard) = gate_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_fused_sigmoid_mul_bf16(
            x_ptr as *const _,
            gate_ptr as *const _,
            out_ptr as *mut _,
            elems as i64,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_fused_sigmoid_mul_bf16 failed with status {status}");
    }
    Ok(())
}


fn matmul_f32_bf16w_dims(lhs: &Tensor, weight: &Tensor) -> Result<(usize, usize, usize)> {
    let Ok((m, k)) = lhs.dims2() else {
        candle_core::bail!(
            "matmul_f32_bf16w: lhs must be rank-2 [rows,in], got {:?}",
            lhs.shape()
        );
    };
    let Ok((w_k, n)) = weight.dims2() else {
        candle_core::bail!(
            "matmul_f32_bf16w: weight must be rank-2 [in,out], got {:?}",
            weight.shape()
        );
    };
    if k != w_k {
        candle_core::bail!(
            "matmul_f32_bf16w: inner dim mismatch lhs={:?} weight={:?}",
            lhs.shape(),
            weight.shape()
        );
    }
    if m > i32::MAX as usize || k > i32::MAX as usize || n > i32::MAX as usize {
        candle_core::bail!("matmul_f32_bf16w: dimensions exceed i32 kernel envelope");
    }
    Ok((m, k, n))
}

pub fn matmul_f32_bf16w(lhs: &Tensor, weight: &Tensor) -> Result<Tensor> {
    // Inlined predicate (was `pub fn supports_matmul_f32_bf16w`, deleted in (#1082) —
    // zero external callers, only the precondition check inside this fn).
    let supports = matches!(lhs.device(), Device::Cuda(_))
        && matches!(weight.device(), Device::Cuda(_))
        && lhs.dtype() == DType::F32
        && weight.dtype() == DType::BF16
        && matmul_f32_bf16w_dims(lhs, weight).is_ok();
    if !supports {
        candle_core::bail!(
            "matmul_f32_bf16w unsupported lhs={:?} weight={:?} dtypes=({:?},{:?})",
            lhs.shape(),
            weight.shape(),
            lhs.dtype(),
            weight.dtype()
        );
    }
    let (m, _k, n) = matmul_f32_bf16w_dims(lhs, weight)?;
    let lhs = lhs.contiguous()?;
    let weight = weight.contiguous()?;
    if m == 0 {
        return Tensor::zeros((m, n), DType::F32, lhs.device());
    }
    let weight_f32 = weight.to_dtype(DType::F32)?;
    lhs.matmul(&weight_f32)
}

pub fn matmul_f32_bf16w_bwd_lhs(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (m, _n, k) = {
        let Ok((m, out_dim)) = grad_out.dims2() else {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: grad must be rank-2 [rows,out], got {:?}",
                grad_out.shape()
            );
        };
        let Ok((in_dim, weight_out)) = weight.dims2() else {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: weight must be rank-2 [in,out], got {:?}",
                weight.shape()
            );
        };
        if out_dim != weight_out {
            candle_core::bail!(
                "matmul_f32_bf16w_bwd_lhs: out dim mismatch grad={:?} weight={:?}",
                grad_out.shape(),
                weight.shape()
            );
        }
        if m > i32::MAX as usize || in_dim > i32::MAX as usize || out_dim > i32::MAX as usize {
            candle_core::bail!("matmul_f32_bf16w_bwd_lhs: dimensions exceed i32 kernel envelope");
        }
        (m, out_dim, in_dim)
    };
    if !matches!(grad_out.device(), Device::Cuda(_)) || !matches!(weight.device(), Device::Cuda(_))
    {
        candle_core::bail!("matmul_f32_bf16w_bwd_lhs requires CUDA tensors");
    }
    if grad_out.dtype() != DType::F32 || weight.dtype() != DType::BF16 {
        candle_core::bail!(
            "matmul_f32_bf16w_bwd_lhs expects F32 grad and BF16 weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight.dtype()
        );
    }

    let grad_out = grad_out.contiguous()?;
    let weight = weight.contiguous()?;
    if m == 0 {
        return Tensor::zeros((m, k), DType::F32, grad_out.device());
    }
    let weight_t = weight.to_dtype(DType::F32)?.t()?.contiguous()?;
    grad_out.matmul(&weight_t)
}




/// Storage-level fused LoRA add-in-place over F32 base + hidden + B, for
/// CUDA custom ops.
///
/// `base` is mutated in-place. Layouts must describe contiguous rank-2 F32
/// tensors with shapes `base=[rows,out]`, `hidden=[rows,rank]`,
/// `b=[out,rank]`. (The candle-Tensor `lora_add_inplace_f32` wrapper was
/// removed in (#1082); production callers go through `lora_add_inplace_f32_kt`
/// or this storage-level entry from `forward.rs:2861`.)
pub fn lora_add_inplace_f32_storage(
    base_cuda: &CudaStorage,
    base_layout: &Layout,
    hidden_cuda: &CudaStorage,
    hidden_layout: &Layout,
    b_cuda: &CudaStorage,
    b_layout: &Layout,
    scale: f32,
) -> Result<()> {
    let base_dims = base_layout.dims();
    let hidden_dims = hidden_layout.dims();
    let b_dims = b_layout.dims();
    if base_dims.len() != 2 {
        candle_core::bail!("lora_add_inplace_f32_storage: base must be rank-2, got {base_dims:?}");
    }
    if hidden_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: hidden must be rank-2, got {hidden_dims:?}"
        );
    }
    if b_dims.len() != 2 {
        candle_core::bail!("lora_add_inplace_f32_storage: B must be rank-2, got {b_dims:?}");
    }
    let (rows, out_dim) = (base_dims[0], base_dims[1]);
    let (hidden_rows, rank) = (hidden_dims[0], hidden_dims[1]);
    let (b_out, b_rank) = (b_dims[0], b_dims[1]);
    if rows != hidden_rows || out_dim != b_out || rank != b_rank {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: shape mismatch base={base_dims:?} hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    if base_cuda.dtype() != DType::F32
        || hidden_cuda.dtype() != DType::F32
        || b_cuda.dtype() != DType::F32
    {
        candle_core::bail!(
            "lora_add_inplace_f32_storage: expected F32 tensors, got base={:?} hidden={:?} b={:?}",
            base_cuda.dtype(),
            hidden_cuda.dtype(),
            b_cuda.dtype()
        );
    }
    if !base_layout.is_contiguous() || !hidden_layout.is_contiguous() || !b_layout.is_contiguous() {
        candle_core::bail!("lora_add_inplace_f32_storage: tensors must be contiguous");
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        candle_core::bail!("lora_add_inplace_f32_storage: dimensions exceed i32 kernel envelope");
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    let stream = base_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let base_slice = base_cuda
        .as_cuda_slice::<f32>()?
        .slice(base_layout.start_offset()..);
    let hidden_slice = hidden_cuda
        .as_cuda_slice::<f32>()?
        .slice(hidden_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<f32>()?
        .slice(b_layout.start_offset()..);

    let status = unsafe {
        let (base_ptr, _base_guard) = base_slice.device_ptr(&stream);
        let (hidden_ptr, _hidden_guard) = hidden_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);
        kiln_lora_add_inplace_f32(
            base_ptr as *mut f32,
            hidden_ptr as *const f32,
            b_ptr as *const f32,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_lora_add_inplace_f32 failed with status {status}");
    }
    Ok(())
}

/// Storage-level BF16 LoRA add for CUDA custom ops.
///
/// Computes `out = base + scale * hidden @ b.T`, where
/// `base=[rows,out]` and `b=[out,rank]` are BF16, `hidden=[rows,rank]`
/// is F32, and `out=[rows,out]` is BF16.
pub fn lora_add_bf16_storage(
    out_cuda: &CudaStorage,
    out_layout: &Layout,
    base_cuda: &CudaStorage,
    base_layout: &Layout,
    hidden_cuda: &CudaStorage,
    hidden_layout: &Layout,
    b_cuda: &CudaStorage,
    b_layout: &Layout,
    scale: f32,
) -> Result<()> {
    let out_dims = out_layout.dims();
    let base_dims = base_layout.dims();
    let hidden_dims = hidden_layout.dims();
    let b_dims = b_layout.dims();
    if out_dims.len() != 2 || base_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_bf16_storage: out/base must be rank-2, got out={out_dims:?} base={base_dims:?}"
        );
    }
    if hidden_dims.len() != 2 || b_dims.len() != 2 {
        candle_core::bail!(
            "lora_add_bf16_storage: hidden/B must be rank-2, got hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    let (rows, out_dim) = (base_dims[0], base_dims[1]);
    let (hidden_rows, rank) = (hidden_dims[0], hidden_dims[1]);
    let (b_out, b_rank) = (b_dims[0], b_dims[1]);
    if out_dims != base_dims || rows != hidden_rows || out_dim != b_out || rank != b_rank {
        candle_core::bail!(
            "lora_add_bf16_storage: shape mismatch out={out_dims:?} base={base_dims:?} hidden={hidden_dims:?} b={b_dims:?}"
        );
    }
    if out_cuda.dtype() != DType::BF16
        || base_cuda.dtype() != DType::BF16
        || hidden_cuda.dtype() != DType::F32
        || b_cuda.dtype() != DType::BF16
    {
        candle_core::bail!(
            "lora_add_bf16_storage: expected out/base BF16, hidden F32, B BF16; got out={:?} base={:?} hidden={:?} b={:?}",
            out_cuda.dtype(),
            base_cuda.dtype(),
            hidden_cuda.dtype(),
            b_cuda.dtype()
        );
    }
    if !out_layout.is_contiguous()
        || !base_layout.is_contiguous()
        || !hidden_layout.is_contiguous()
        || !b_layout.is_contiguous()
    {
        candle_core::bail!("lora_add_bf16_storage: tensors must be contiguous");
    }
    if rows > i32::MAX as usize || out_dim > i32::MAX as usize || rank > i32::MAX as usize {
        candle_core::bail!("lora_add_bf16_storage: dimensions exceed i32 kernel envelope");
    }
    if rows == 0 || out_dim == 0 || rank == 0 {
        return Ok(());
    }

    let stream = base_cuda.device().cuda_stream();
    let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
    let out_slice = out_cuda
        .as_cuda_slice::<bf16>()?
        .slice(out_layout.start_offset()..);
    let base_slice = base_cuda
        .as_cuda_slice::<bf16>()?
        .slice(base_layout.start_offset()..);
    let hidden_slice = hidden_cuda
        .as_cuda_slice::<f32>()?
        .slice(hidden_layout.start_offset()..);
    let b_slice = b_cuda
        .as_cuda_slice::<bf16>()?
        .slice(b_layout.start_offset()..);

    let status = unsafe {
        let (base_ptr, _base_guard) = base_slice.device_ptr(&stream);
        let (hidden_ptr, _hidden_guard) = hidden_slice.device_ptr(&stream);
        let (b_ptr, _b_guard) = b_slice.device_ptr(&stream);
        let (out_ptr, _out_guard) = out_slice.device_ptr(&stream);
        kiln_lora_decode_add_bf16(
            base_ptr as *const _,
            hidden_ptr as *const f32,
            b_ptr as *const _,
            out_ptr as *mut _,
            scale,
            rows as i32,
            out_dim as i32,
            rank as i32,
            raw_stream,
        )
    };
    if status != 0 {
        candle_core::bail!("kiln_lora_decode_add_bf16 failed with status {status}");
    }
    Ok(())
}

pub fn causal_depthwise_conv1d_f32(
    input: &Tensor,
    weight: &Tensor,
    state: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = input.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!("causal_depthwise_conv1d_f32: weight rank {rank} unsupported"),
    };
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    if input.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
        || state.dtype() != DType::F32
    {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32: expected F32 input/weight/state, got {:?}/{:?}/{:?}",
            input.dtype(),
            weight_flat.dtype(),
            state.dtype()
        );
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!("causal_depthwise_conv1d_f32: dimensions exceed i32 kernel envelope");
    }

    let input = input.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let state = state.contiguous()?;
    let out = unsafe { Tensor::empty((rows, channels), DType::F32, input.device())? };

    {
        let (i_storage, i_layout) = input.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (o_storage, o_layout) = out.storage_and_layout();
        let i_cuda = match &*i_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: input must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: weight must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: state must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32: output must be CUDA"),
        };
        let stream = i_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let i_slice = i_cuda
            .as_cuda_slice::<f32>()?
            .slice(i_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        let status = unsafe {
            let (i_ptr, _i_guard) = i_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_f32(
                i_ptr as *const f32,
                w_ptr as *const f32,
                s_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!("kiln_causal_depthwise_conv1d_f32 failed with status {status}");
        }
    }
    Ok(out)
}

pub fn causal_depthwise_conv1d_f32_inplace(
    input_out: &Tensor,
    weight: &Tensor,
    state: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = input_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_inplace: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32_inplace: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_inplace: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_inplace: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: weight rank {rank} unsupported"
        ),
    };
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    if input_out.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
        || state.dtype() != DType::F32
    {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: expected F32 input/weight/state, got {:?}/{:?}/{:?}",
            input_out.dtype(),
            weight_flat.dtype(),
            state.dtype()
        );
    }
    if !input_out.is_contiguous() {
        candle_core::bail!("causal_depthwise_conv1d_f32_inplace: input must be contiguous");
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_inplace: dimensions exceed i32 kernel envelope"
        );
    }
    if rows == 0 || channels == 0 {
        return Ok(input_out.clone());
    }

    let weight_flat = weight_flat.contiguous()?;
    let state = state.contiguous()?;
    {
        let (io_storage, io_layout) = input_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let io_cuda = match &*io_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: input must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: weight must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_inplace: state must be CUDA"),
        };
        let stream = io_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let io_slice = io_cuda
            .as_cuda_slice::<f32>()?
            .slice(io_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);

        let status = unsafe {
            let (io_ptr, _io_guard) = io_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_inplace_f32(
                io_ptr as *mut f32,
                w_ptr as *const f32,
                s_ptr as *const f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_inplace_f32 failed with status {status}"
            );
        }
    }
    Ok(input_out.clone())
}

pub fn causal_depthwise_conv1d_f32_bwd_input(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_input: empty weight".into())
    })?;
    if kernel <= 1 {
        candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: kernel must be > 1");
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_input: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_input: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32 || weight_flat.dtype() != DType::F32 {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: expected F32 grad/weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight_flat.dtype()
        );
    }
    if rows > i32::MAX as usize || channels > i32::MAX as usize || kernel > i32::MAX as usize {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_input: dimensions exceed i32 kernel envelope"
        );
    }

    let grad_out = grad_out.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let grad_input = unsafe { Tensor::empty((rows, channels), DType::F32, grad_out.device())? };

    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (o_storage, o_layout) = grad_input.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: grad must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: weight must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_input: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);

        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_input_f32(
                g_ptr as *const f32,
                w_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_input_f32 failed with status {status}"
            );
        }
    }
    Ok(grad_input)
}

pub fn causal_depthwise_conv1d_f32_bwd_weight(
    grad_out: &Tensor,
    input: &Tensor,
    state: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    if input.dims() != [rows, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: input shape {:?} != grad {:?}",
            input.shape(),
            grad_out.shape()
        );
    }
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_weight: empty weight".into())
    })?;
    if state.dims() != [kernel - 1, channels] {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: state shape {:?} != [{},{}]",
            state.shape(),
            kernel - 1,
            channels
        );
    }
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_weight: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_weight: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_weight: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32
        || input.dtype() != DType::F32
        || state.dtype() != DType::F32
        || weight_flat.dtype() != DType::F32
    {
        candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: expected F32 tensors");
    }

    let grad_out = grad_out.contiguous()?;
    let input = input.contiguous()?;
    let state = state.contiguous()?;
    let grad_weight = unsafe { Tensor::empty((channels, kernel), DType::F32, grad_out.device())? };
    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (i_storage, i_layout) = input.storage_and_layout();
        let (s_storage, s_layout) = state.storage_and_layout();
        let (o_storage, o_layout) = grad_weight.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: grad must be CUDA"),
        };
        let i_cuda = match &*i_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: input must be CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: state must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_weight: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let i_slice = i_cuda
            .as_cuda_slice::<f32>()?
            .slice(i_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f32>()?
            .slice(s_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);
        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (i_ptr, _i_guard) = i_slice.device_ptr(&stream);
            let (s_ptr, _s_guard) = s_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_weight_f32(
                g_ptr as *const f32,
                i_ptr as *const f32,
                s_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_weight_f32 failed with status {status}"
            );
        }
    }
    if weight.rank() == 3 {
        grad_weight.reshape(weight.dims())
    } else {
        Ok(grad_weight)
    }
}

pub fn causal_depthwise_conv1d_f32_bwd_state(grad_out: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let (rows, channels) = grad_out.dims2()?;
    let kernel = *weight.dims().last().ok_or_else(|| {
        candle_core::Error::Msg("causal_depthwise_conv1d_f32_bwd_state: empty weight".into())
    })?;
    let weight_flat = match weight.rank() {
        2 => {
            let (c, k) = weight.dims2()?;
            if c != channels || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_state: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.clone()
        }
        3 => {
            let (c, one, k) = weight.dims3()?;
            if c != channels || one != 1 || k != kernel {
                candle_core::bail!(
                    "causal_depthwise_conv1d_f32_bwd_state: weight shape {:?} incompatible with channels={channels} kernel={kernel}",
                    weight.shape()
                );
            }
            weight.reshape((channels, kernel))?
        }
        rank => candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_state: weight rank {rank} unsupported"
        ),
    };
    if grad_out.dtype() != DType::F32 || weight_flat.dtype() != DType::F32 {
        candle_core::bail!(
            "causal_depthwise_conv1d_f32_bwd_state: expected F32 grad/weight, got {:?}/{:?}",
            grad_out.dtype(),
            weight_flat.dtype()
        );
    }
    let grad_out = grad_out.contiguous()?;
    let weight_flat = weight_flat.contiguous()?;
    let grad_state =
        unsafe { Tensor::empty((kernel - 1, channels), DType::F32, grad_out.device())? };
    {
        let (g_storage, g_layout) = grad_out.storage_and_layout();
        let (w_storage, w_layout) = weight_flat.storage_and_layout();
        let (o_storage, o_layout) = grad_state.storage_and_layout();
        let g_cuda = match &*g_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: grad must be CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: weight must be CUDA"),
        };
        let o_cuda = match &*o_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("causal_depthwise_conv1d_f32_bwd_state: output must be CUDA"),
        };
        let stream = g_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        let g_slice = g_cuda
            .as_cuda_slice::<f32>()?
            .slice(g_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<f32>()?
            .slice(w_layout.start_offset()..);
        let o_slice = o_cuda
            .as_cuda_slice::<f32>()?
            .slice(o_layout.start_offset()..);
        let status = unsafe {
            let (g_ptr, _g_guard) = g_slice.device_ptr(&stream);
            let (w_ptr, _w_guard) = w_slice.device_ptr(&stream);
            let (o_ptr, _o_guard) = o_slice.device_ptr(&stream);
            kiln_causal_depthwise_conv1d_bwd_state_f32(
                g_ptr as *const f32,
                w_ptr as *const f32,
                o_ptr as *mut f32,
                rows as i32,
                channels as i32,
                kernel as i32,
                raw_stream,
            )
        };
        if status != 0 {
            candle_core::bail!(
                "kiln_causal_depthwise_conv1d_bwd_state_f32 failed with status {status}"
            );
        }
    }
    Ok(grad_state)
}

// ===========================================================================
// kt-forward-op autograd shim — relocated from
// `kiln-rmsnorm-kernel/src/kt_forward_op.rs` in (#1082). Byte-identical
// body; only the call paths changed:
//   crate::kt_api::{fused_rmsnorm_kt, fused_rmsnorm_backward_kt}
//     -> kiln_rmsnorm_kernel::{fused_rmsnorm_kt, fused_rmsnorm_backward_kt}
//   crate::kiln_f32_to_bf16 -> the locally re-declared `kiln_f32_to_bf16`.
// ===========================================================================

/// Returns `true` when `(x, weight)` is in the kt-typed forward+backward
/// envelope: CUDA, both BF16, contiguous, rank >= 1, weight shape matches
/// `x`'s last dim, last dim <= 8192. This is the same envelope as
/// [`supports`] for the CUDA fused forward kernel.
fn shim_envelope_ok(x: &Tensor, weight: &Tensor) -> bool {
    if !matches!(x.device(), Device::Cuda(_)) {
        return false;
    }
    if !matches!(weight.device(), Device::Cuda(_)) {
        return false;
    }
    if x.dtype() != DType::BF16 || weight.dtype() != DType::BF16 {
        return false;
    }
    if !x.is_contiguous() || !weight.is_contiguous() {
        return false;
    }
    if x.rank() < 1 {
        return false;
    }
    let hidden = x.dims().last().copied().unwrap_or(0);
    if hidden == 0 || hidden > 8192 {
        return false;
    }
    if weight.dims() != [hidden] {
        return false;
    }
    true
}

/// kt-shim fused RMSNorm forward+backward with candle-autograd integration.
///
/// Behavioral envelope:
/// - CUDA + BF16 + contiguous + `hidden <= 8192` → routes through
///   [`KtForwardOp2`] over the kt-typed fused forward+backward.
/// - Anything outside the envelope (CPU, non-bf16, etc.) → returns an
///   error. The production caller in `kiln-model::forward::rms_norm`
///   already filters out-of-envelope inputs via `supports(x, weight)`
///   before dispatching here, so this branch is unreachable in practice.
///
/// [`KtForwardOp2`]: kiln_kt_bridge::forward_op::KtForwardOp2
pub fn fused_rmsnorm_via_kt_forward_op(
    x: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    if !shim_envelope_ok(x, weight) {
        candle_core::bail!(
            "fused_rmsnorm_via_kt_forward_op: inputs outside kt-shim envelope \
             (CUDA + BF16 + contiguous + hidden <= 8192 required). \
             Callers must filter via `crate::rmsnorm_candle_shim::supports(x, weight)` first."
        );
    }

    cuda_via_kt_forward_op(x, weight, eps)
}

// ---------------------------------------------------------------------------
// CUDA fast path: KtForwardOp2 over kt-typed forward + backward.
// ---------------------------------------------------------------------------

fn cuda_via_kt_forward_op(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use candle_core::Context;
    use kiln_kt_bridge::forward_op::KtForwardOp2;
    use kiln_kt_bridge::{kt_tensor_from_candle_cuda_borrow, kt_tensor_to_candle_cuda_copy};
    use kiln_rmsnorm_kernel::{fused_rmsnorm_backward_kt, fused_rmsnorm_kt};
    use std::sync::Arc;

    // ----- Force-contiguous inputs ---------------------------------------
    //
    // `apply_op2` passes the input layouts through to the CustomOp's
    // `cuda_fwd` hook; the kt-bridge borrow path requires contiguous
    // storage. The kernel itself also requires contiguous (the FFI
    // assumes row-major linear layout). The OPD shim applies the same
    // defensive `contiguous()`.
    let x_contig = x
        .contiguous()
        .context("force-contiguous x for rmsnorm kt-shim")?;
    let w_contig = weight
        .contiguous()
        .context("force-contiguous weight for rmsnorm kt-shim")?;

    // ----- Forward closure -----------------------------------------------
    //
    // Calls the kt-typed `fused_rmsnorm_kt` directly. Unlike the OPD
    // forward closure (which has an axis-N gather substrate gap and
    // re-runs the candle composite as a leaf op), the rmsnorm
    // kt-typed forward has no substrate gap: it bottoms out in the
    // same `kiln_fused_rmsnorm` FFI symbol that the candle-typed
    // `fused_rmsnorm` calls, just routed through kt borrows instead
    // of candle's storage_and_layout / as_cuda_slice path.
    let forward = move |x_in: &Tensor, w_in: &Tensor| -> Result<Tensor> {
        let x_kt = kt_tensor_from_candle_cuda_borrow(x_in)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: borrow x: {e}")))?;
        let w_kt = kt_tensor_from_candle_cuda_borrow(w_in)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: borrow w: {e}")))?;
        let y_kt = fused_rmsnorm_kt(&x_kt, &w_kt, eps)
            .map_err(|e| candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: kt call: {e}")))?;
        kt_tensor_to_candle_cuda_copy(&y_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim fwd: copy-back y: {e}"))
        })
    };

    // ----- Backward closure ----------------------------------------------
    //
    // The kt path returns `(grad_x: BF16, grad_w_partial: F32 [rows, hidden])`;
    // the kernel writes via atomicAdd into the first `hidden` F32 slots
    // only, so we cast exactly those `hidden` F32 slots to BF16 via a
    // direct `kiln_f32_to_bf16` call (rather than `f32_to_bf16_kt`, which
    // would over-cast `rows*hidden` elements).
    //
    // History: the pre-(#1082) `fused_rmsnorm_backward_via_kt_bridge`
    // helper in `lib.rs` implemented the same dispatch shape for the
    // candle-typed `RmsNormCustomOp::bwd` body. Deleted alongside the
    // CustomOp wrapper; this closure is now the only place the cast is
    // performed.
    //
    // The shim passes us (`arg1=x, arg2=weight, res=y, grad_res=grad_y`);
    // we ignore `res` since the backward doesn't depend on the
    // forward output value (the kernel recomputes `rms_inv` from `x`).
    let backward = move |arg_x: &Tensor,
                         arg_w: &Tensor,
                         _res: &Tensor,
                         grad_res: &Tensor|
          -> Result<(Option<Tensor>, Option<Tensor>)> {
        let x_c = arg_x.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous x: {e}"))
        })?;
        let w_c = arg_w.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous weight: {e}"))
        })?;
        let g_c = grad_res.contiguous().map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: contiguous grad_out: {e}"))
        })?;

        let x_dims = x_c.dims();
        let hidden = *x_dims.last().ok_or_else(|| {
            candle_core::Error::Msg(
                "rmsnorm kt-shim bwd: x must have rank >= 1".to_string(),
            )
        })?;

        let x_kt = kt_tensor_from_candle_cuda_borrow(&x_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow x: {e}"))
        })?;
        let w_kt = kt_tensor_from_candle_cuda_borrow(&w_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow weight: {e}"))
        })?;
        let g_kt = kt_tensor_from_candle_cuda_borrow(&g_c).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: borrow grad_out: {e}"))
        })?;

        let (grad_x_kt, grad_w_partial_kt) =
            fused_rmsnorm_backward_kt(&x_kt, &w_kt, &g_kt, eps).map_err(|e| {
                candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: kt call: {e}"))
            })?;

        // Cast the populated `hidden` prefix of grad_w_partial (F32) to
        // BF16. See `csrc/fused_rmsnorm_bwd.cu` lines 12-19/122-123 for
        // why only the first `hidden` slots of the [rows, hidden] F32
        // partial buffer are populated by the kernel.
        let grad_w_kt = {
            use kiln_tensor::{DType as KtDType, Tensor as KtTensor};
            let partial_ptr = kiln_kt_bridge::cuda_output_device_ptr(&grad_w_partial_kt);
            let partial_st = kiln_kt_bridge::cuda_storage_of_output(&grad_w_partial_kt);
            let raw_stream = partial_st.cuda_stream_raw();
            let dst_kt: KtTensor =
                kiln_kt_bridge::alloc_cuda_tensor(partial_st, KtDType::BF16, vec![hidden])
                    .map_err(|e| {
                        candle_core::Error::Msg(format!(
                            "rmsnorm kt-shim bwd: alloc grad_w BF16: {e}"
                        ))
                    })?;
            let dst_ptr = kiln_kt_bridge::cuda_output_device_ptr(&dst_kt);
            // SAFETY: `partial_ptr` points to a F32 buffer of at least
            // `rows*hidden` elements (kt allocation); we read only the
            // first `hidden`. `dst_ptr` points to a BF16 buffer of
            // exactly `hidden` elements (just allocated above).
            let status = unsafe {
                kiln_f32_to_bf16(
                    partial_ptr as *const f32,
                    dst_ptr as *mut _,
                    hidden as i32,
                    raw_stream,
                )
            };
            if status != 0 {
                return Err(candle_core::Error::Msg(format!(
                    "rmsnorm kt-shim bwd: kiln_f32_to_bf16 failed (status {status})"
                )));
            }
            dst_kt
        };

        let gx = kt_tensor_to_candle_cuda_copy(&grad_x_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: copy-back grad_x: {e}"))
        })?;
        let gw = kt_tensor_to_candle_cuda_copy(&grad_w_kt).map_err(|e| {
            candle_core::Error::Msg(format!("rmsnorm kt-shim bwd: copy-back grad_w: {e}"))
        })?;

        Ok((Some(gx), Some(gw)))
    };

    // ----- Apply ----------------------------------------------------------
    let op = KtForwardOp2::new("kiln-rmsnorm-kt-forward-op", forward, backward);
    x_contig
        .apply_op2_arc(&w_contig, Arc::new(Box::new(op)))
        .context("apply rmsnorm kt-forward-op to (x, weight)")
}
