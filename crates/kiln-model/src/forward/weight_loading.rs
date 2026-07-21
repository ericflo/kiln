use super::*;

/// Convert a `WeightTensor` (raw bytes + shape + dtype) to a kt `Tensor` on `device`.
///
/// CUDA (#1082): kt-native — raw weight bytes upload straight into a kt CUDA
/// tensor via `Tensor::from_raw_bytes_on`, dropping the candle `from_raw_buffer`
/// leaf AND the subsequent device→device bridge copy the old path paid per
/// weight bridge. One host→device H2D, no candle, no extra
/// copy — a load-time win on the dominant loader entry.
///
/// Non-CUDA (Metal / CPU): kt-native raw byte construction. Metal uploads via
/// `Tensor::from_raw_bytes_on`, so the loader no longer creates a CPU/Candle
/// tensor and bridges it after the fact.
#[cfg(feature = "cuda")]
pub(super) fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    Tensor::from_raw_bytes_on(
        *device,
        weight_dtype(w),
        w.as_bytes().to_vec(),
        w.shape.clone(),
    )
    .map_err(|e| anyhow::anyhow!("weight_to_tensor (kt-native CUDA load): {e}"))
}

/// Non-CUDA sibling of [`weight_to_tensor`]: builds the kt tensor directly on
/// the requested device.
#[cfg(not(feature = "cuda"))]
pub(super) fn weight_to_tensor(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    Tensor::from_raw_bytes_on(
        *device,
        weight_dtype(w),
        w.as_bytes().to_vec(),
        w.shape.clone(),
    )
    .map_err(|e| anyhow::anyhow!("weight_to_tensor (kt-native load): {e}"))
}

pub(super) fn weight_dtype(w: &WeightTensor) -> DType {
    match w.dtype {
        TensorDType::F16 => DType::F16,
        TensorDType::BF16 => DType::BF16,
        TensorDType::F32 => DType::F32,
    }
}

pub(super) const TRANSPOSE_ROW_TILE: usize = 32;
pub(super) const TRANSPOSE_COL_TILE: usize = 32;
pub(super) const PARALLEL_TRANSPOSE_MIN_BYTES: usize = 1 << 20;
pub(super) const PARALLEL_TRANSPOSE_ROW_CHUNK: usize = 64;

#[inline(always)]
pub(super) fn copy_transpose_elem_unaligned<T: Copy>(
    data: &[u8],
    out: &mut [u8],
    src: usize,
    dst: usize,
) {
    // Safetensors byte offsets are not guaranteed to satisfy Rust alignment
    // for typed views, so use unaligned loads/stores while still avoiding a
    // tiny `memmove` call per BF16/F32 element.
    unsafe {
        let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
        std::ptr::write_unaligned(out.as_mut_ptr().add(dst).cast::<T>(), value);
    }
}

pub(super) fn transpose_weight_bytes_typed<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    let elem_size = std::mem::size_of::<T>();

    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        copy_transpose_elem_unaligned::<T>(data, out, src, dst);
                    }
                }
            }
        }
    } else {
        transpose_weight_bytes_typed_parallel_rows::<T>(data, out, rows, cols);
    }
}

pub(super) fn transpose_weight_bytes_typed_parallel_rows<T: Copy + Send + Sync>(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
) {
    use rayon::prelude::*;

    let elem_size = std::mem::size_of::<T>();
    let out_col_stride = rows * elem_size;
    let chunks = rows.div_ceil(PARALLEL_TRANSPOSE_ROW_CHUNK);
    let out_addr = out.as_mut_ptr() as usize;

    (0..chunks).into_par_iter().for_each(|chunk_idx| {
        let row0 = chunk_idx * PARALLEL_TRANSPOSE_ROW_CHUNK;
        let row_end = (row0 + PARALLEL_TRANSPOSE_ROW_CHUNK).min(rows);
        let out_ptr = out_addr as *mut u8;

        for row in row0..row_end {
            let mut src = row * cols * elem_size;
            let mut dst = row * elem_size;
            for _ in 0..cols {
                // SAFETY: row chunks are disjoint. For any source element
                // `(row, col)`, the transposed destination is `(col, row)`,
                // so different row chunks write non-overlapping bytes within
                // each output column. `transposed_weight_bytes_2d` validated
                // data/out lengths before dispatching here.
                unsafe {
                    let value = std::ptr::read_unaligned(data.as_ptr().add(src).cast::<T>());
                    std::ptr::write_unaligned(out_ptr.add(dst).cast::<T>(), value);
                }
                src += elem_size;
                dst += out_col_stride;
            }
        }
    });
}

pub(super) fn transpose_weight_bytes_generic(
    data: &[u8],
    out: &mut [u8],
    rows: usize,
    cols: usize,
    elem_size: usize,
) {
    if data.len() < PARALLEL_TRANSPOSE_MIN_BYTES {
        for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
            let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
            for col0 in (0..cols).step_by(TRANSPOSE_COL_TILE) {
                let col_end = (col0 + TRANSPOSE_COL_TILE).min(cols);
                for row in row0..row_end {
                    for col in col0..col_end {
                        let src = (row * cols + col) * elem_size;
                        let dst = (col * rows + row) * elem_size;
                        out[dst..dst + elem_size].copy_from_slice(&data[src..src + elem_size]);
                    }
                }
            }
        }
    } else {
        use rayon::prelude::*;

        let out_col_stride = rows * elem_size;
        let out_block_stride = out_col_stride * TRANSPOSE_COL_TILE;
        out.par_chunks_mut(out_block_stride)
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let col0 = block_idx * TRANSPOSE_COL_TILE;
                let col_end = (col0 + (out_block.len() / out_col_stride)).min(cols);
                for row0 in (0..rows).step_by(TRANSPOSE_ROW_TILE) {
                    let row_end = (row0 + TRANSPOSE_ROW_TILE).min(rows);
                    for col in col0..col_end {
                        let out_col = col - col0;
                        let out_base = out_col * out_col_stride;
                        for row in row0..row_end {
                            let src = (row * cols + col) * elem_size;
                            let dst = out_base + row * elem_size;
                            out_block[dst..dst + elem_size]
                                .copy_from_slice(&data[src..src + elem_size]);
                        }
                    }
                }
            });
    }
}

pub(crate) fn transposed_weight_bytes_2d(w: &WeightTensor) -> Result<(Vec<u8>, [usize; 2])> {
    anyhow::ensure!(
        w.shape.len() == 2,
        "direct transposed weight upload requires a rank-2 tensor, got shape {:?}",
        w.shape
    );
    let rows = w.shape[0];
    let cols = w.shape[1];
    let elem_size = w.dtype.size_bytes();
    let data = w.as_bytes();
    let expected_len = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(elem_size))
        .context("weight tensor byte size overflow")?;
    anyhow::ensure!(
        data.len() == expected_len,
        "weight tensor data length mismatch: got {} bytes, expected {} bytes for shape {:?} and dtype {}",
        data.len(),
        expected_len,
        w.shape,
        w.dtype
    );

    let mut out = vec![0u8; data.len()];
    match elem_size {
        1 => transpose_weight_bytes_typed::<u8>(data, &mut out, rows, cols),
        2 => transpose_weight_bytes_typed::<u16>(data, &mut out, rows, cols),
        4 => transpose_weight_bytes_typed::<u32>(data, &mut out, rows, cols),
        8 => transpose_weight_bytes_typed::<u64>(data, &mut out, rows, cols),
        _ => transpose_weight_bytes_generic(data, &mut out, rows, cols, elem_size),
    }

    Ok((out, [cols, rows]))
}

/// CUDA (#1082): kt-native — the cached transposed weight bytes upload straight
/// into a kt CUDA tensor via `Tensor::from_raw_bytes_on`, dropping the candle
/// `from_raw_buffer` leaf + the device→device bridge copy (same win as
/// [`weight_to_tensor`], on the projection-weight loader entry). Metal/CPU use
/// the same kt-native raw byte path.
#[cfg(feature = "cuda")]
pub(super) fn weight_to_transposed_tensor_2d(
    w: &WeightTensor,
    device: &Device,
    cache_miss_policy: TransposedWeightCacheMissPolicy,
) -> Result<Tensor> {
    let data = transposed_weight_bytes_2d_cached_bytes(w, cache_miss_policy)?;
    Tensor::from_raw_bytes_on(
        *device,
        weight_dtype(w),
        data.as_bytes().to_vec(),
        data.shape().to_vec(),
    )
    .map_err(|e| anyhow::anyhow!("weight_to_transposed_tensor_2d (kt-native CUDA load): {e}"))
}

/// Non-CUDA sibling: upload the cached transposed bytes directly through kt.
#[cfg(not(feature = "cuda"))]
pub(super) fn weight_to_transposed_tensor_2d(
    w: &WeightTensor,
    device: &Device,
    cache_miss_policy: TransposedWeightCacheMissPolicy,
) -> Result<Tensor> {
    let data = transposed_weight_bytes_2d_cached_bytes(w, cache_miss_policy)?;
    Tensor::from_raw_bytes_on(
        *device,
        weight_dtype(w),
        data.as_bytes().to_vec(),
        data.shape().to_vec(),
    )
    .map_err(|e| anyhow::anyhow!("weight_to_transposed_tensor_2d (kt-native load): {e}"))
}

pub(super) fn cached_transpose_for_weight(
    w: &WeightTensor,
    materialized: &Tensor,
    device: &Device,
    cache_miss_policy: TransposedWeightCacheMissPolicy,
) -> Result<Tensor> {
    if ProjectionLoadPolicy::for_model_loader_device(*device)
        .direct_transposed_upload_for_cached_weights
    {
        weight_to_transposed_tensor_2d(w, device, cache_miss_policy)
    } else {
        cached_transpose(materialized)
    }
}

pub(super) fn dropped_weight_stub(w: &WeightTensor, device: &Device) -> Result<Tensor> {
    // kt `zeros` takes `Device` by value (kt Device is Copy) and shape as
    // `Into<Vec<usize>>` (#1082 forward-flip). On Vulkan the stub lands on
    // CPU-host like every other Vulkan weight (`loader_kt_device`).
    Ok(Tensor::zeros(
        vec![1usize],
        weight_dtype(w),
        loader_kt_device(device),
    )?)
}

#[derive(Clone)]
pub(super) struct ProjectionLoadCache {
    policy: ProjectionLoadPolicy,
    pub(super) transposed_cache_miss_policy: TransposedWeightCacheMissPolicy,
    bf16_stub: Option<Tensor>,
    f16_stub: Option<Tensor>,
    f32_stub: Option<Tensor>,
}

impl ProjectionLoadCache {
    pub(super) fn for_base_model_load(device: &Device) -> Result<Self> {
        Self::new(
            device,
            TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
        )
    }

    pub(super) fn for_lazy_mtp_upload(device: &Device) -> Result<Self> {
        Self::new(device, TransposedWeightCacheMissPolicy::ReadOnly)
    }

    fn new(
        device: &Device,
        transposed_cache_miss_policy: TransposedWeightCacheMissPolicy,
    ) -> Result<Self> {
        let policy = ProjectionLoadPolicy::for_model_loader_device(*device);
        if policy.drop_projection_originals || policy.drop_projection_transposes {
            Ok(Self {
                policy,
                transposed_cache_miss_policy,
                bf16_stub: Some(Tensor::zeros(
                    vec![1usize],
                    DType::BF16,
                    loader_kt_device(device),
                )?),
                f16_stub: Some(Tensor::zeros(
                    vec![1usize],
                    DType::F16,
                    loader_kt_device(device),
                )?),
                f32_stub: Some(Tensor::zeros(
                    vec![1usize],
                    DType::F32,
                    loader_kt_device(device),
                )?),
            })
        } else {
            Ok(Self {
                policy,
                transposed_cache_miss_policy,
                bf16_stub: None,
                f16_stub: None,
                f32_stub: None,
            })
        }
    }

    fn stub_for(&self, dtype: DType) -> Option<Tensor> {
        match dtype {
            DType::BF16 => self.bf16_stub.clone(),
            DType::F16 => self.f16_stub.clone(),
            DType::F32 => self.f32_stub.clone(),
            _ => None,
        }
    }

    fn drops_projection_originals(&self) -> bool {
        self.policy.drop_projection_originals
    }

    fn drops_projection_transposes(&self) -> bool {
        self.policy.drop_projection_transposes
    }

    fn parallel_transposed_projection_upload(&self) -> bool {
        self.policy.parallel_transposed_projection_upload_enabled()
    }

    fn synchronizes_after_dropping_originals(&self) -> bool {
        self.drops_projection_originals() && self.policy.synchronize_after_dropping_originals
    }
}

pub(super) fn projection_tensors_for_load(
    w: &WeightTensor,
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<(Tensor, Tensor)> {
    if cache.drops_projection_originals() {
        let transposed =
            weight_to_transposed_tensor_2d(w, device, cache.transposed_cache_miss_policy)?;
        let original_stub = match cache.stub_for(weight_dtype(w)) {
            Some(stub) => stub,
            None => dropped_weight_stub(w, device)?,
        };
        Ok((original_stub, transposed))
    } else if cache.drops_projection_transposes() {
        let materialized = weight_to_tensor(w, device)?;
        let transposed_stub = match cache.stub_for(weight_dtype(w)) {
            Some(stub) => stub,
            None => dropped_weight_stub(w, device)?,
        };
        Ok((materialized, transposed_stub))
    } else {
        let materialized = weight_to_tensor(w, device)?;
        let transposed = cached_transpose(&materialized)?;
        Ok((materialized, transposed))
    }
}

pub(super) fn projection_tensors_for_load_batch(
    weights: &[(&str, &WeightTensor)],
    device: &Device,
    cache: &ProjectionLoadCache,
) -> Result<Vec<(Tensor, Tensor)>> {
    // Backend policy may allow pre-transposing raw weight bytes in parallel and
    // uploading them directly into the target tensors. Every other device falls
    // through to the serial kt-native path below.
    #[cfg(feature = "metal")]
    if cache.parallel_transposed_projection_upload() {
        use rayon::prelude::*;

        let cache_miss_policy = cache.transposed_cache_miss_policy;
        let transposed: Result<Vec<CachedTransposedWeightBytes>> = weights
            .par_iter()
            .map(|(name, w)| {
                transposed_weight_bytes_2d_cached_bytes(w, cache_miss_policy)
                    .with_context(|| format!("{name} transposed projection bytes"))
            })
            .collect();

        let cache = cache.clone();
        let device = device.clone();
        return transposed?
            .into_par_iter()
            .zip(weights.par_iter())
            .map(|(data, (name, w))| {
                let transposed = Tensor::from_raw_bytes_on(
                    device,
                    weight_dtype(w),
                    data.as_bytes().to_vec(),
                    data.shape().to_vec(),
                )
                .with_context(|| format!("{name} transposed projection upload"))?;
                let original_stub = match cache.stub_for(weight_dtype(w)) {
                    Some(stub) => stub,
                    None => dropped_weight_stub(w, &device)
                        .with_context(|| format!("{name} projection stub"))?,
                };
                Ok((original_stub, transposed))
            })
            .collect();
    }

    weights
        .iter()
        .map(|(name, w)| {
            projection_tensors_for_load(w, device, cache)
                .with_context(|| format!("{name} projection tensors"))
        })
        .collect()
}

pub(super) fn aux_tensors_for_load_batch(
    weights: &[(&str, &WeightTensor)],
    device: &Device,
) -> Result<Vec<Tensor>> {
    if !ProjectionLoadPolicy::for_model_loader_device(*device)
        .parallel_auxiliary_weight_upload_enabled()
    {
        return weights
            .iter()
            .map(|(name, w)| weight_to_tensor(w, device).with_context(|| format!("{name} tensor")))
            .collect();
    }

    use rayon::prelude::*;

    let device = device.clone();
    weights
        .par_iter()
        .map(|(name, w)| weight_to_tensor(w, &device).with_context(|| format!("{name} tensor")))
        .collect()
}

/// Cache a transpose for repeated GEMMs.
///
/// Matmuls on the hot path repeatedly consume these tensors, so materialize
/// the transpose once at load time instead of relying on backend-specific
/// strided access behaviour.
pub(super) fn cached_transpose(weight: &Tensor) -> Result<Tensor> {
    Ok(weight.t()?.contiguous()?)
}

pub(super) fn cpu_needs_f32_matmul(lhs: &Tensor, rhs: &Tensor) -> bool {
    matches!(lhs.device(), Device::Cpu) && (lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32)
}

pub(super) fn broadcast_matmul_cpu_compatible(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    // (#1082) GDN-on-Vulkan: several GDN-block intermediates (l2-norm, the conv
    // composite, the chunkwise recurrence) currently land on `Device::Cpu` while
    // the frozen projection weights stay on the accelerator (`loader_kt_device`
    // keeps Vulkan weights CPU-host, but the matmul backend can promote them).
    // The kt `matmul` op requires both operands on the same device, so align the
    // weight to the activation device before the matmul. This is the same class
    // of fix as the rotary `inv_freq.to_device(device)` alignment. No-op on
    // CUDA/Metal and the full-attn path (operands already co-located), so those
    // paths are byte-unchanged.
    let rhs_aligned;
    let rhs = if lhs.device() != rhs.device() {
        rhs_aligned = rhs.to_device(lhs.device())?;
        &rhs_aligned
    } else {
        rhs
    };
    if cpu_needs_f32_matmul(lhs, rhs) {
        let lhs_f32 = lhs.to_dtype(DType::F32)?;
        let rhs_f32 = rhs.to_dtype(DType::F32)?;
        return matmul_no_broadcast_copy(&lhs_f32, &rhs_f32);
    }
    matmul_no_broadcast_copy(lhs, rhs)
}

/// `lhs.broadcast_matmul(rhs)` for the `[B, T, K] @ [K, N] -> [B, T, N]` case
/// that drives every projection in the decoder, without paying for candle's
/// `broadcast_matmul` of materializing the broadcasted RHS via
/// `rhs.broadcast_as(...).contiguous()`. nsys (NVTX `kiln/gdn/in_proj` range)
/// showed that contiguous copy as 78 % of total GPU time at bs=4 on the
/// CUDA + GDN path — the 168 MB weight tensor was being copied across the
/// batch dim before every matmul, dwarfing the matmul itself. Flattening
/// `lhs` to 2D + `matmul(rhs)` + reshape uses the same compute path with no
/// implicit copy.
pub(super) fn matmul_no_broadcast_copy(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let l_dims = lhs.dims();
    let r_dims = rhs.dims();
    if r_dims.len() == 2 && l_dims.len() >= 2 && lhs.is_contiguous() {
        let k = l_dims[l_dims.len() - 1];
        if r_dims[0] == k {
            let out_n = r_dims[1];
            let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
            let lhs2d = lhs.reshape((lead, k))?;

            // Phase 7 opt-in: route the 2D matmul through the kt op
            // contract so the active backend's MatmulOp implementation
            // owns native dispatch. Falls through to the ordinary kt
            // Tensor::matmul path when the gate is off.
            #[cfg(feature = "cuda")]
            if crate::kt_api_policy::experimental_routes_enabled()
                && matches!(lhs2d.dtype(), DType::BF16 | DType::F16 | DType::F32)
                && lhs2d.dtype() == rhs.dtype()
                && lhs2d.is_contiguous()
                && rhs.is_contiguous()
            {
                if let Some(out2d) = try_kt_matmul(&lhs2d, rhs)? {
                    let mut out_shape: Vec<usize> = l_dims[..l_dims.len() - 1].to_vec();
                    out_shape.push(out_n);
                    return Ok(out2d.reshape(out_shape)?);
                }
            }

            let out2d = lhs2d.matmul(rhs)?;
            let mut out_shape: Vec<usize> = l_dims[..l_dims.len() - 1].to_vec();
            out_shape.push(out_n);
            return Ok(out2d.reshape(out_shape)?);
        }
    }
    Ok(lhs.broadcast_matmul(rhs)?)
}

pub(super) fn runtime_matmul_no_broadcast_copy(
    backend: &dyn BackendRuntime,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Option<Tensor>> {
    let l_dims = lhs.dims();
    let r_dims = rhs.dims();
    if r_dims.len() != 2 || l_dims.len() < 2 {
        return Ok(None);
    }
    let k = l_dims[l_dims.len() - 1];
    if r_dims[0] != k {
        return Ok(None);
    }

    let out_n = r_dims[1];
    let lead: usize = l_dims[..l_dims.len() - 1].iter().product();
    let lhs2d = if l_dims.len() == 2 {
        lhs.clone()
    } else if lhs.is_contiguous() {
        lhs.reshape((lead, k))?
    } else {
        return Ok(None);
    };
    let req = MatmulRequest::plain(
        lhs2d.dims().to_vec(),
        rhs.dims().to_vec(),
        lhs2d.dtype(),
        false,
    )
    .with_dtypes(lhs2d.dtype(), rhs.dtype(), lhs2d.dtype());
    let Some(out2d) = LinearBackend::runtime_matmul(backend, &req, &lhs2d, rhs)? else {
        return Ok(None);
    };

    let mut out_shape: Vec<usize> = l_dims[..l_dims.len() - 1].to_vec();
    out_shape.push(out_n);
    Ok(Some(out2d.reshape(out_shape)?))
}

pub(super) fn runtime_matmul_or_broadcast(
    backend: &dyn BackendRuntime,
    lhs: &Tensor,
    rhs: &Tensor,
) -> Result<Tensor> {
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        // This helper is used by MTP's frozen `fc_t` projection. Route before
        // the backend leaf so the concat activation remains connected without
        // recording or materializing a gradient for `fc_t`.
        return require_active_tape_output(
            crate::tape_forward::try_tape_lora_linear_kt(lhs, rhs, None, 0.0)
                .context("MTP frozen fc_t tape matmul")?,
            "MTP frozen fc_t projection",
        );
    }
    if let Some(out) = runtime_matmul_no_broadcast_copy(backend, lhs, rhs)? {
        return Ok(out);
    }
    broadcast_matmul_cpu_compatible(lhs, rhs)
}

/// Phase 7 — kt-API matmul migration helper. Routes a 2D matmul through
/// `kiln_tensor::ops::matmul`, whose `MatmulOp` owns native backend dispatch.
///
/// Returns `Ok(None)` on any incompatibility so the caller falls
/// through to candle's `Tensor::matmul`. NVTX range `kiln/matmul_kt`
/// brackets the migrated call so nsys traces separate the path from
/// the candle baseline.
#[cfg(any(feature = "cuda", feature = "rocm"))]
pub(crate) fn try_kt_matmul(lhs: &Tensor, rhs: &Tensor) -> Result<Option<Tensor>> {
    kiln_nvtx::range!(c"kiln/matmul_kt");

    // When a thread-local `Tape` scope is active,
    // route matmul through the kt op-registry with a `MatmulBackward`
    // recorded onto the tape. The forward output is the same kt
    // matmul kernel; the difference is that the backward node lives
    // on `Tape` instead of leaving the result as a no-autograd
    // candle Tensor. Without a scope this returns `Ok(None)` and decode falls
    // straight through to the kt-native matmul below.
    // #1082 seam flip: kt-native MatmulBackward recorder — no kt->candle->kt.
    // The recorder is device-dispatched. A scope cannot fall through to the
    // forward-only dense matmul on either CUDA or ROCm.
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_matmul_kt(lhs, rhs)
                .context("try_kt_matmul try_tape_matmul_kt")?,
            "matmul",
        )
        .map(Some);
    }

    let out_kt = match kiln_tensor::ops::matmul(lhs, rhs) {
        Ok(t) => t,
        Err(_) => return Ok(None),
    };
    Ok(Some(out_kt))
}

/// Vulkan-routed `[B, T, H] @ [H, D] -> [B, T, D]` matmul with autograd
/// support, falling back to [`broadcast_matmul_cpu_compatible`] when
/// the backend declines.
///
/// Phase 2 sub-step 2: GDN linear-attention layers' in_proj_qkv,
/// in_proj_z, in_proj_a, in_proj_b matmuls were going through
/// `broadcast_matmul_cpu_compatible` directly, bypassing the existing
/// Vulkan routing in `linear_with_lora_t_backend_decode_if`. This
/// helper threads them through `LinearBackend::runtime_linear_prefill_apply`
/// (the autograd-safe `CustomOp1`) under the qualified Vulkan route policy.
/// On Qwen3.5-4B that's 24 GDN layers × 4 in-proj matmuls per layer
/// — the dominant CPU compute in training before this commit.
pub(super) fn gdn_in_proj_matmul(
    backend: &dyn BackendRuntime,
    x: &Tensor,
    weight_t: &Tensor,
) -> Result<Tensor> {
    // (#1443 step 5) GDN `in_proj_a` / `in_proj_b` are FROZEN base projections
    // (no LoRA adapter — they are not in DEFAULT_TARGET_MODULES). On Vulkan the
    // activation `x` is F32 while these base weights are BF16, so the plain
    // equal-dtype kt `matmul` in `broadcast_matmul_cpu_compatible` cannot run.
    // Route through the same `try_tape_lora_linear_kt` recorder the qkv/z/out
    // projections use (with `lora=None`): on Vulkan it dispatches the base
    // matmul through `vk_matmul_bf16w` (F32 act × BF16 weight) and records the
    // dx-only `MatmulBf16wBackward`, so dx flows back to `x`/`normed` while the
    // weight stays frozen BF16. No-op (returns None) off the tape path or on the
    // equal-dtype path, falling through to the existing dispatch below.
    #[cfg(any(
        feature = "cuda",
        feature = "metal",
        feature = "vulkan",
        feature = "rocm"
    ))]
    if crate::tape_forward::tape_scope_active() {
        return require_active_tape_output(
            crate::tape_forward::try_tape_lora_linear_kt(x, weight_t, None, 0.0)
                .context("gdn_in_proj_matmul try_tape_lora_linear_kt")?,
            "GDN input projection",
        );
    }
    if let Some(out) = runtime_matmul_no_broadcast_copy(backend, x, weight_t)? {
        return Ok(out);
    }
    // #1082 item 4: `linear_prefill_apply` is kt-typed — pass kt directly.
    #[cfg(feature = "cuda")]
    {
        if let Some(out) = LinearBackend::runtime_linear_prefill_apply(backend, x, weight_t)? {
            return Ok(out);
        }
    }
    broadcast_matmul_cpu_compatible(x, weight_t)
}

pub(super) fn promote_cpu_activation(t: Tensor) -> Result<Tensor> {
    if matches!(t.device(), Device::Cpu) && t.dtype() != DType::F32 {
        Ok(t.to_dtype(DType::F32)?)
    } else {
        Ok(t)
    }
}

/// Tiny BF16 placeholder that replaces a projection's pre-transposed
/// contiguous copy (`*_proj_t`) once Marlin has absorbed it. Dropping the
/// original `Tensor` field releases the underlying CUDA buffer (the
/// refcounted `Arc<Storage>` hits zero), reclaiming the per-layer BF16
/// residency. The struct layout is preserved so every existing construction
/// site (tests, loaders) continues to compile unchanged.
pub(super) fn dropped_bf16_stub(device: &Device) -> Result<Tensor> {
    // kt `zeros`: shape as `Into<Vec<usize>>`, `Device` by value (#1082).
    // Vulkan stubs are CPU-host (`loader_kt_device`).
    Ok(Tensor::zeros(
        vec![1usize],
        DType::BF16,
        loader_kt_device(device),
    )?)
}

/// Sidecar record: which slot in `layers[layer_idx]` a queued Marlin pack
/// job belongs to. Populated inline with `pack_from_bf16_batch`'s input vec
/// during the per-layer build loop, then replayed after the batch pack
/// finishes so the packed `MarlinPackedProj` lands in the right field.
#[derive(Clone, Copy, Debug)]
pub(super) enum MarlinPackKind {
    QProj,
    GateProj,
    UpProj,
    DownProj,
    GdnOutProj,
}

#[derive(Debug)]
pub(super) struct MarlinPackEntry {
    layer_idx: usize,
    kind: MarlinPackKind,
}

/// Install a successfully packed projection into its target layer slot and
/// drop the corresponding pre-transposed BF16 copy.
pub(super) fn install_marlin_packed(
    layer: &mut GpuLayerWeights,
    kind: MarlinPackKind,
    packed: crate::marlin_proj::MarlinPackedProj,
    device: &Device,
) -> Result<()> {
    match kind {
        MarlinPackKind::QProj => {
            if let GpuAttentionWeights::Full(ref mut full) = layer.attention {
                full.q_proj_marlin = Some(packed);
                full.q_proj_t = dropped_bf16_stub(device)?;
            }
        }
        MarlinPackKind::GateProj => {
            layer.mlp.gate_proj_marlin = Some(packed);
            layer.mlp.gate_proj_t = dropped_bf16_stub(device)?;
        }
        MarlinPackKind::UpProj => {
            layer.mlp.up_proj_marlin = Some(packed);
            layer.mlp.up_proj_t = dropped_bf16_stub(device)?;
        }
        MarlinPackKind::DownProj => {
            layer.mlp.down_proj_marlin = Some(packed);
            layer.mlp.down_proj_t = dropped_bf16_stub(device)?;
        }
        MarlinPackKind::GdnOutProj => {
            if let GpuAttentionWeights::Linear(ref mut lin) = layer.attention {
                lin.out_proj_marlin = Some(packed);
                lin.out_proj_t = dropped_bf16_stub(device)?;
            }
        }
    }
    Ok(())
}

pub(super) fn flush_marlin_pack_inputs(
    marlin_pack_inputs: &mut Vec<(Tensor, i32)>,
    marlin_pack_meta: &mut Vec<MarlinPackEntry>,
    layers: &mut [GpuLayerWeights],
    device: &Device,
    scope: &'static str,
) -> Result<()> {
    if marlin_pack_inputs.is_empty() {
        return Ok(());
    }

    let pack_start = std::time::Instant::now();
    let inputs = std::mem::take(marlin_pack_inputs);
    let metadata = std::mem::take(marlin_pack_meta);
    anyhow::ensure!(
        inputs.len() == metadata.len(),
        "marlin pack input and metadata counts disagree"
    );
    let packed = crate::marlin_proj::pack_from_bf16_batch(&inputs)
        .with_context(|| format!("marlin {scope} batch pack"))?;
    anyhow::ensure!(
        packed.len() == metadata.len(),
        "marlin packed result and metadata counts disagree"
    );
    let pack_elapsed_ms = pack_start.elapsed().as_millis();
    let n_inputs = inputs.len();
    let n_packed = packed.iter().filter(|p| p.is_some()).count();
    tracing::info!(
        scope,
        n_inputs,
        n_packed,
        pack_elapsed_ms = pack_elapsed_ms as u64,
        parallel = true,
        "marlin batch pack complete"
    );
    eprintln!(
        "[kiln] marlin {scope} batch pack: {n_packed}/{n_inputs} projections in {pack_elapsed_ms} ms ({})",
        "parallel"
    );

    for (entry, maybe_packed) in metadata.into_iter().zip(packed.into_iter()) {
        if let Some(packed) = maybe_packed {
            install_marlin_packed(&mut layers[entry.layer_idx], entry.kind, packed, device)
                .with_context(|| {
                    format!(
                        "install marlin {:?} on layer {}",
                        entry.kind, entry.layer_idx
                    )
                })?;
        }
    }
    Ok(())
}

impl GpuWeights {
    pub fn has_mtp(&self) -> bool {
        self.mtp.is_some()
    }

    /// Whether any base-model projection is backed by Marlin-packed weights.
    ///
    /// Marlin inference can apply LoRA deltas over its packed forward path,
    /// but the server training tape is not authoritative for that mixed
    /// representation. Training admission uses this process-lifetime layout
    /// fact to reject before allocating LoRA or optimizer state.
    pub fn has_any_marlin_packed_projection(&self) -> bool {
        self.layers.iter().any(|layer| {
            let attention_is_packed = match &layer.attention {
                GpuAttentionWeights::Full(attention) => attention.q_proj_marlin.is_some(),
                GpuAttentionWeights::Linear(attention) => attention.out_proj_marlin.is_some(),
            };
            attention_is_packed
                || layer.mlp.gate_proj_marlin.is_some()
                || layer.mlp.up_proj_marlin.is_some()
                || layer.mlp.down_proj_marlin.is_some()
        })
    }

    /// Deep-copy every tensor onto `device` — the training-session
    /// residency primitive. On the Vulkan hybrid substrate the serving
    /// weights are kt CPU-storage (with VulkanBuffer caches managed
    /// separately), but TRAINING needs every operand on ONE device or the
    /// tape recorders die on "inputs on different devices". A trainer
    /// uploads a resident copy once per job (≈8 GB BF16 for Qwen3.5-4B —
    /// affordable on unified-memory APUs), trains against it, and drops
    /// it; the serving copy is never touched.
    ///
    /// Same-device moves are cheap (`Tensor::to_device` returns an Arc
    /// clone), so calling this unconditionally is safe.
    pub fn to_device_deep(&self, device: Device) -> Result<GpuWeights> {
        let mv = |t: &Tensor| -> Result<Tensor> {
            t.to_device(device)
                .map_err(|e| anyhow::anyhow!("to_device_deep: {e}"))
        };
        let mv_opt = |t: &Option<Tensor>| -> Result<Option<Tensor>> {
            t.as_ref().map(|t| mv(t)).transpose()
        };
        let mv_w8 =
            |w: &Option<crate::rocm_w8_proj::RocmW8Proj>| -> Result<Option<crate::rocm_w8_proj::RocmW8Proj>> {
                w.as_ref()
                    .map(|w| w.to_device_deep(device))
                    .transpose()
            };
        let mv_marlin = |w: &Option<crate::marlin_proj::MarlinPackedProj>| -> Result<Option<crate::marlin_proj::MarlinPackedProj>> {
            w.as_ref().map(|w| w.to_device_deep(device)).transpose()
        };
        let mv_ffn = |m: &GpuFfnWeights| -> Result<GpuFfnWeights> {
            Ok(GpuFfnWeights {
                gate_proj: mv(&m.gate_proj)?,
                up_proj: mv(&m.up_proj)?,
                down_proj: mv(&m.down_proj)?,
                gate_proj_t: mv(&m.gate_proj_t)?,
                up_proj_t: mv(&m.up_proj_t)?,
                down_proj_t: mv(&m.down_proj_t)?,
                gate_up_proj_t: mv_opt(&m.gate_up_proj_t)?,
                gate_proj_marlin: mv_marlin(&m.gate_proj_marlin)?,
                up_proj_marlin: mv_marlin(&m.up_proj_marlin)?,
                down_proj_marlin: mv_marlin(&m.down_proj_marlin)?,
                gate_up_proj_w8: mv_w8(&m.gate_up_proj_w8)?,
                down_proj_w8: mv_w8(&m.down_proj_w8)?,
            })
        };
        let mv_layer = |l: &GpuLayerWeights| -> Result<GpuLayerWeights> {
            let attention = match &l.attention {
                GpuAttentionWeights::Full(a) => {
                    GpuAttentionWeights::Full(GpuFullAttentionWeights {
                        q_proj: mv(&a.q_proj)?,
                        k_proj: mv(&a.k_proj)?,
                        v_proj: mv(&a.v_proj)?,
                        o_proj: mv(&a.o_proj)?,
                        q_norm: mv(&a.q_norm)?,
                        k_norm: mv(&a.k_norm)?,
                        q_proj_t: mv(&a.q_proj_t)?,
                        k_proj_t: mv(&a.k_proj_t)?,
                        v_proj_t: mv(&a.v_proj_t)?,
                        qkv_proj_t: mv_opt(&a.qkv_proj_t)?,
                        o_proj_t: mv(&a.o_proj_t)?,
                        qkv_proj_w8: mv_w8(&a.qkv_proj_w8)?,
                        o_proj_w8: mv_w8(&a.o_proj_w8)?,
                        q_proj_marlin: mv_marlin(&a.q_proj_marlin)?,
                    })
                }
                GpuAttentionWeights::Linear(a) => {
                    GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                        in_proj_qkv: mv(&a.in_proj_qkv)?,
                        in_proj_z: mv(&a.in_proj_z)?,
                        out_proj: mv(&a.out_proj)?,
                        in_proj_a: mv(&a.in_proj_a)?,
                        in_proj_b: mv(&a.in_proj_b)?,
                        conv1d: mv(&a.conv1d)?,
                        norm: mv(&a.norm)?,
                        a_log: mv(&a.a_log)?,
                        a_log_gates: mv(&a.a_log_gates)?,
                        dt_bias: mv(&a.dt_bias)?,
                        in_proj_qkv_t: mv(&a.in_proj_qkv_t)?,
                        in_proj_z_t: mv(&a.in_proj_z_t)?,
                        in_proj_a_t: mv(&a.in_proj_a_t)?,
                        in_proj_b_t: mv(&a.in_proj_b_t)?,
                        in_proj_ab_t: mv_opt(&a.in_proj_ab_t)?,
                        out_proj_t: mv(&a.out_proj_t)?,
                        out_proj_marlin: mv_marlin(&a.out_proj_marlin)?,
                        in_proj_qkvzab_w8: mv_w8(&a.in_proj_qkvzab_w8)?,
                    })
                }
            };
            Ok(GpuLayerWeights {
                input_layernorm: mv(&l.input_layernorm)?,
                post_attention_layernorm: mv(&l.post_attention_layernorm)?,
                attention,
                mlp: mv_ffn(&l.mlp)?,
            })
        };

        let layers = self
            .layers
            .iter()
            .map(&mv_layer)
            .collect::<Result<Vec<_>>>()?;
        let mtp = match &self.mtp {
            None => None,
            Some(slot) => Some(slot.to_device_deep(device, &mv_layer)?),
        };
        Ok(GpuWeights {
            source_content_sha256: self.source_content_sha256.clone(),
            base_weight_shard_manifest: self.base_weight_shard_manifest.clone(),
            execution_provenance: self.execution_provenance.clone(),
            embed_tokens: mv(&self.embed_tokens)?,
            embed_tokens_t: mv(&self.embed_tokens_t)?,
            layers,
            final_norm: mv(&self.final_norm)?,
            rotary_inv_freq: mv(&self.rotary_inv_freq)?,
            lm_head_w8: mv_w8(&self.lm_head_w8)?,
            mtp,
        })
    }

    pub fn mtp_weights(&self) -> Result<&MtpGpuWeights> {
        let mtp = self.mtp.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "native MTP requested but checkpoint has no mtp.* tensors \
                 (Qwen3.5-4B includes them)"
            )
        })?;
        mtp.get_or_upload()
    }

    pub fn linear_attention_layers_in_prefix(&self, end_layer: usize) -> usize {
        self.layers
            .iter()
            .take(end_layer.min(self.layers.len()))
            .filter(|layer| matches!(layer.attention, GpuAttentionWeights::Linear(_)))
            .count()
    }

    /// kt-typed device accessor for the weight tensors (#1082 Tier 3).
    ///
    /// Returns the `kiln_tensor::Device` corresponding to the candle
    /// device that backs the `embed_tokens` tensor. Callers (in
    /// `kiln-server`) that currently do `weights.embed_tokens.device()`
    /// and feed the resulting candle `&Device` into downstream
    /// candle APIs can use this accessor to surface a kt Device at the
    /// public boundary while the internal storage stays candle-typed.
    ///
    /// Until `GpuWeights` migrates off candle Tensors (the Tier 3
    /// `KtWeights` rewrite this commit is downstream of), this is the
    /// only canonical kt-typed accessor on the struct's surface.
    ///
    /// Always-on (no cuda feature gate): only uses
    /// `kiln_kt_bridge::kt_device_from_candle`, which is a pure
    /// `candle <-> kt` Device enum mapping with no CUDA toolchain
    /// dependency. (#1082)
    pub fn device_kt(&self) -> kiln_tensor::Device {
        // #1082 forward-flip: `embed_tokens` is now a kt tensor, so its
        // `device()` already returns a `kiln_tensor::Device` — identity.
        self.embed_tokens.device()
    }

    /// kt-native view of the token-embedding table (#1082, embedding
    /// region migration).
    ///
    /// Returns `embed_tokens` ([vocab, hidden]) as a contiguous `KtTensor`.
    /// The tensor is already kt-native, so backend eligibility belongs to the
    /// caller's request dispatch rather than this accessor.
    pub fn embed_tokens_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.embed_tokens, "embed_tokens_kt")
    }

    /// kt-native view of the pre-transposed token-embedding table
    /// (#1082, lm_head region migration).
    ///
    /// Returns `embed_tokens_t` ([hidden, vocab], contiguous since load) as a
    /// contiguous `KtTensor`. This is the weight-side kt accessor for the
    /// lm-head region; backend-specific routing is owned by the matmul request
    /// dispatch at the call site.
    pub fn embed_tokens_t_kt(&self) -> Result<KtTensor> {
        kt_contiguous(&self.embed_tokens_t, "embed_tokens_t_kt")
    }

    /// Convert `ModelWeights` (CPU bytes) into candle tensors on the given device.
    ///
    /// `config` is used to precompute the rotary `inv_freq` tensor once so the RoPE
    /// hot path does not re-upload it on every call.
    pub fn from_model_weights(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &Device,
    ) -> Result<Self> {
        Self::from_model_weights_with_upload_policy(
            weights,
            config,
            device,
            &AcceleratorWeightUploadPolicy::unlimited(),
        )
    }

    pub fn from_model_weights_with_upload_policy(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &Device,
        upload_policy: &AcceleratorWeightUploadPolicy,
    ) -> Result<Self> {
        let source_bytes_total = weights.base_model_total_bytes();
        let mut upload_pacer = AcceleratorWeightUploadPacer::new(
            upload_policy,
            source_bytes_total,
            weights.layers.len(),
        )?;
        #[cfg(feature = "vulkan")]
        let _durable_vulkan_allocations = matches!(device, Device::Vulkan(_))
            .then(kiln_vulkan_kernel::buffer_pool::durable_allocation_scope);

        #[cfg(feature = "rocm")]
        let w8a16_enabled = matches!(device, Device::Rocm(_))
            && crate::rocm_policy::current_rocm_kernel_policy().w8_projection;
        let base_source_bytes = weights
            .embedding
            .embed_tokens
            .size_bytes()
            .saturating_add(weights.final_norm.size_bytes());
        upload_pacer.prepare("base_reserved", base_source_bytes, 0)?;
        // On Metal and on Vulkan-active processes, `embed_tokens` itself
        // is never read past `embedding_lookup_from_weights` (which falls
        // back to `embed_tokens_t` whenever the dims don't match the
        // expected `[vocab, hidden]` shape — the stub case). Materializing
        // both copies costs ~1.3 GB of CPU storage on Qwen3.5-4B BF16
        // for nothing, so collapse to a stub on those backends. On a
        // unified-memory APU this is what keeps Phase 0 from sitting on
        // a duplicate embedding table that the Vulkan side has already
        // mirrored to its own buffer cache.
        let projection_load_policy = ProjectionLoadPolicy::for_model_loader_device(*device);
        let (embed_tokens, embed_tokens_t) =
            if projection_load_policy.stub_embedding_table_after_transposed_upload {
                let embed_tokens_t = weight_to_transposed_tensor_2d(
                    &weights.embedding.embed_tokens,
                    device,
                    TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
                )
                .context("embed_tokens transposed upload")?;
                upload_pacer.boundary("base_embedding_transpose_uploaded")?;
                let embed_tokens = dropped_weight_stub(&weights.embedding.embed_tokens, device)
                    .context("embed_tokens stub")?;
                (embed_tokens, embed_tokens_t)
            } else {
                let embed_tokens = weight_to_tensor(&weights.embedding.embed_tokens, device)
                    .context("embed_tokens")?;
                upload_pacer.boundary("base_embedding_uploaded")?;
                let embed_tokens_t = cached_transpose_for_weight(
                    &weights.embedding.embed_tokens,
                    &embed_tokens,
                    device,
                    TransposedWeightCacheMissPolicy::PersistBeforeReadiness,
                )
                .context("embed_tokens cached transpose")?;
                upload_pacer.boundary("base_embedding_transpose_uploaded")?;
                (embed_tokens, embed_tokens_t)
            };
        let lm_head_w8 = {
            #[cfg(feature = "rocm")]
            {
                if w8a16_enabled && projection_load_policy.pack_w8a16_projection_rows {
                    crate::rocm_w8_proj::pack_from_bf16_rows(&embed_tokens)
                        .context("w8 lm_head pack")?
                } else {
                    None
                }
            }
            #[cfg(not(feature = "rocm"))]
            {
                None
            }
        };
        upload_pacer.boundary("base_embedding_pack_complete")?;
        let final_norm = weight_to_tensor(&weights.final_norm, device).context("final_norm")?;
        upload_pacer.boundary("base_final_norm_uploaded")?;
        let rotary_inv_freq =
            compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, device)
                .context("rotary_inv_freq")?;
        upload_pacer.boundary("base_rotary_initialized")?;
        let mut source_bytes_completed = base_source_bytes;
        upload_pacer.checkpoint("base", source_bytes_completed, 0)?;
        // Base weights are uploaded before the server declares readiness, so
        // already-computed cache misses may be persisted synchronously here.
        let projection_load_cache =
            ProjectionLoadCache::for_base_model_load(device).context("projection load cache")?;
        if projection_load_cache.drops_projection_originals() {
            tracing::info!("projection original tensors are dropped after transposed upload");
        } else if projection_load_cache.drops_projection_transposes() {
            tracing::info!(
                "projection transposed tensors are dropped because Vulkan-native training keeps originals"
            );
        }

        // Per-layer `pack_from_bf16` used to run inline during weight load,
        // serializing ~104 calls (8 × q_proj + 96 × MLP gate/up/down) behind
        // a single thread. At ~58s cold load on the Qwen3.5-4B A6000 build
        // this is a significant fraction of server startup. Sidecar the
        // pack inputs here, batch-pack via rayon after the layer loop, and
        // install results into the per-layer slots. A configured upload pace
        // instead flushes each layer before its cooperative checkpoint so a
        // CUDA W4A16 load cannot hide a whole-model upload in finalization.
        let w4a16_enabled = crate::marlin_proj::enabled();
        let mut marlin_pack_inputs: Vec<(Tensor, i32)> = Vec::new();
        let mut marlin_pack_meta: Vec<MarlinPackEntry> = Vec::new();

        let mut layers = Vec::with_capacity(weights.layers.len());
        for (i, lw) in weights.layers.iter().enumerate() {
            let ctx = |name: &str| format!("layer {i} {name}");
            let layer_source_bytes_completed =
                source_bytes_completed.saturating_add(lw.total_bytes());
            upload_pacer.prepare("layer_reserved", layer_source_bytes_completed, i + 1)?;

            let (input_layernorm, post_attention_layernorm, attention) = match &lw.attention {
                crate::weights::AttentionWeights::Full(attn) => {
                    let aux_tensors = aux_tensors_for_load_batch(
                        &[
                            ("input_layernorm", &lw.input_layernorm),
                            ("post_attention_layernorm", &lw.post_attention_layernorm),
                            ("q_norm", &attn.q_norm),
                            ("k_norm", &attn.k_norm),
                        ],
                        device,
                    )
                    .context(ctx("full attention aux tensors"))?;
                    let mut aux_tensors = aux_tensors.into_iter();
                    let input_layernorm =
                        aux_tensors.next().context(ctx("input_layernorm missing"))?;
                    let post_attention_layernorm = aux_tensors
                        .next()
                        .context(ctx("post_attention_layernorm missing"))?;
                    let q_norm = aux_tensors.next().context(ctx("q_norm missing"))?;
                    let k_norm = aux_tensors.next().context(ctx("k_norm missing"))?;

                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("q_proj", &attn.q_proj),
                            ("k_proj", &attn.k_proj),
                            ("v_proj", &attn.v_proj),
                            ("o_proj", &attn.o_proj),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (q_proj, q_proj_t) = attn_proj.next().context(ctx("q_proj missing"))?;
                    let (k_proj, k_proj_t) = attn_proj.next().context(ctx("k_proj missing"))?;
                    let (v_proj, v_proj_t) = attn_proj.next().context(ctx("v_proj missing"))?;
                    let (o_proj, o_proj_t) = attn_proj.next().context(ctx("o_proj missing"))?;
                    let qkv_proj_t = {
                        #[cfg(any(feature = "cuda", feature = "rocm"))]
                        {
                            if projection_load_policy.cache_full_attention_qkv_transpose_concat {
                                Some(
                                    Tensor::cat(&[&q_proj_t, &k_proj_t, &v_proj_t], LAST_DIM)?
                                        .contiguous()
                                        .context(ctx("qkv_proj_t contiguous"))?,
                                )
                            } else {
                                None
                            }
                        }
                        #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                        {
                            None
                        }
                    };
                    // The active Marlin profile queues q_proj for the post-loop
                    // batch pack. The packed weight (and the BF16
                    // drop) are installed after the layer loop via
                    // `install_marlin_packed`, so `q_proj_marlin` starts as
                    // None and `q_proj_t` keeps the BF16 copy until then.
                    if w4a16_enabled {
                        marlin_pack_inputs.push((q_proj_t.clone(), 128));
                        marlin_pack_meta.push(MarlinPackEntry {
                            layer_idx: i,
                            kind: MarlinPackKind::QProj,
                        });
                    }
                    let (qkv_proj_w8, o_proj_w8) = {
                        #[cfg(feature = "rocm")]
                        {
                            if w8a16_enabled
                                && !w4a16_enabled
                                && projection_load_policy.pack_w8a16_projection_rows
                            {
                                let qkv_rows = Tensor::cat(&[&q_proj, &k_proj, &v_proj], 0)?
                                    .contiguous()
                                    .context(ctx("w8 full-attn qkv rows contiguous"))?;
                                (
                                    crate::rocm_w8_proj::pack_from_bf16_rows(&qkv_rows)
                                        .context(ctx("w8 full-attn qkv pack"))?,
                                    crate::rocm_w8_proj::pack_from_bf16_rows(&o_proj)
                                        .context(ctx("w8 full-attn o_proj pack"))?,
                                )
                            } else {
                                (None, None)
                            }
                        }
                        #[cfg(not(feature = "rocm"))]
                        {
                            (None, None)
                        }
                    };
                    (
                        input_layernorm,
                        post_attention_layernorm,
                        GpuAttentionWeights::Full(GpuFullAttentionWeights {
                            q_proj,
                            k_proj,
                            v_proj,
                            o_proj,
                            q_norm,
                            k_norm,
                            q_proj_t,
                            k_proj_t,
                            v_proj_t,
                            qkv_proj_t,
                            o_proj_t,
                            qkv_proj_w8,
                            o_proj_w8,
                            q_proj_marlin: None,
                        }),
                    )
                }
                crate::weights::AttentionWeights::Linear(attn) => {
                    let aux_tensors = aux_tensors_for_load_batch(
                        &[
                            ("input_layernorm", &lw.input_layernorm),
                            ("post_attention_layernorm", &lw.post_attention_layernorm),
                            ("conv1d", &attn.conv1d),
                            ("gdn_norm", &attn.norm),
                            ("a_log", &attn.a_log),
                            ("dt_bias", &attn.dt_bias),
                        ],
                        device,
                    )
                    .context(ctx("linear attention aux tensors"))?;
                    let mut aux_tensors = aux_tensors.into_iter();
                    let input_layernorm =
                        aux_tensors.next().context(ctx("input_layernorm missing"))?;
                    let post_attention_layernorm = aux_tensors
                        .next()
                        .context(ctx("post_attention_layernorm missing"))?;
                    let conv1d = aux_tensors.next().context(ctx("conv1d missing"))?;
                    let norm = aux_tensors.next().context(ctx("gdn_norm missing"))?;
                    let a_log = aux_tensors.next().context(ctx("a_log missing"))?;
                    let dt_bias = aux_tensors.next().context(ctx("dt_bias missing"))?;
                    let a_log_gates = a_log
                        .to_dtype(DType::BF16)
                        .context(ctx("a_log gates bf16 cache"))?;

                    let attn_proj = projection_tensors_for_load_batch(
                        &[
                            ("in_proj_qkv", &attn.in_proj_qkv),
                            ("in_proj_z", &attn.in_proj_z),
                            ("out_proj", &attn.out_proj),
                            ("in_proj_a", &attn.in_proj_a),
                            ("in_proj_b", &attn.in_proj_b),
                        ],
                        device,
                        &projection_load_cache,
                    )
                    .context(ctx("linear attention projection tensors"))?;
                    let mut attn_proj = attn_proj.into_iter();
                    let (in_proj_qkv, in_proj_qkv_t) =
                        attn_proj.next().context(ctx("in_proj_qkv missing"))?;
                    let (in_proj_z, in_proj_z_t) =
                        attn_proj.next().context(ctx("in_proj_z missing"))?;
                    let (out_proj, out_proj_t) =
                        attn_proj.next().context(ctx("out_proj missing"))?;
                    // The expanded Marlin profile also queues the GDN out_proj.
                    // This projection is gated separately
                    // from the rest because it's the last linear in the GDN
                    // block before the residual add, so int4 here is more
                    // quality-sensitive than the in-projections or the MLP.
                    if w4a16_enabled && crate::marlin_proj::gdn_out_proj_enabled() {
                        marlin_pack_inputs.push((out_proj_t.clone(), 128));
                        marlin_pack_meta.push(MarlinPackEntry {
                            layer_idx: i,
                            kind: MarlinPackKind::GdnOutProj,
                        });
                    }
                    let (in_proj_a, in_proj_a_t) =
                        attn_proj.next().context(ctx("in_proj_a missing"))?;
                    let (in_proj_b, in_proj_b_t) =
                        attn_proj.next().context(ctx("in_proj_b missing"))?;
                    let in_proj_ab_t = {
                        #[cfg(any(feature = "cuda", feature = "metal", feature = "rocm"))]
                        {
                            if projection_load_policy.cache_linear_attention_ab_transpose_concat {
                                Some(
                                    Tensor::cat(&[&in_proj_a_t, &in_proj_b_t], LAST_DIM)?
                                        .contiguous()
                                        .context(ctx("in_proj_ab_t contiguous"))?,
                                )
                            } else {
                                None
                            }
                        }
                        #[cfg(not(any(feature = "cuda", feature = "metal", feature = "rocm")))]
                        {
                            None
                        }
                    };
                    let in_proj_qkvzab_w8 = {
                        #[cfg(feature = "rocm")]
                        {
                            if w8a16_enabled && projection_load_policy.pack_w8a16_projection_rows {
                                let rows = Tensor::cat(
                                    &[&in_proj_qkv, &in_proj_z, &in_proj_a, &in_proj_b],
                                    0,
                                )?
                                .contiguous()
                                .context(ctx("w8 gdn in-proj rows contiguous"))?;
                                crate::rocm_w8_proj::pack_from_bf16_rows(&rows)
                                    .context(ctx("w8 gdn in-proj pack"))?
                            } else {
                                None
                            }
                        }
                        #[cfg(not(feature = "rocm"))]
                        {
                            None
                        }
                    };
                    (
                        input_layernorm,
                        post_attention_layernorm,
                        GpuAttentionWeights::Linear(GpuLinearAttentionWeights {
                            in_proj_qkv,
                            in_proj_z,
                            out_proj,
                            in_proj_a,
                            in_proj_b,
                            conv1d,
                            norm,
                            a_log,
                            a_log_gates,
                            dt_bias,
                            in_proj_qkv_t,
                            in_proj_z_t,
                            in_proj_a_t,
                            in_proj_b_t,
                            in_proj_ab_t,
                            out_proj_t,
                            out_proj_marlin: None,
                            in_proj_qkvzab_w8,
                        }),
                    )
                }
            };

            let mlp_proj = projection_tensors_for_load_batch(
                &[
                    ("gate_proj", &lw.mlp.gate_proj),
                    ("up_proj", &lw.mlp.up_proj),
                    ("down_proj", &lw.mlp.down_proj),
                ],
                device,
                &projection_load_cache,
            )
            .context(ctx("mlp projection tensors"))?;
            let mut mlp_proj = mlp_proj.into_iter();
            let (gate_proj, gate_proj_t) = mlp_proj.next().context(ctx("gate_proj missing"))?;
            let (up_proj, up_proj_t) = mlp_proj.next().context(ctx("up_proj missing"))?;
            let (down_proj, down_proj_t) = mlp_proj.next().context(ctx("down_proj missing"))?;
            // The active Marlin profile queues each MLP projection for the
            // post-loop Marlin batch pack. See the q_proj comment above —
            // the `*_proj_marlin` fields start as None, and
            // `install_marlin_packed` drops `*_proj_t` after the batch runs
            // according to the fixed projection-load policy.
            if w4a16_enabled {
                marlin_pack_inputs.push((gate_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::GateProj,
                });
                marlin_pack_inputs.push((up_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::UpProj,
                });
                marlin_pack_inputs.push((down_proj_t.clone(), 128));
                marlin_pack_meta.push(MarlinPackEntry {
                    layer_idx: i,
                    kind: MarlinPackKind::DownProj,
                });
            }
            // Cache gate/up concatenated along the output dim for GPU prefill:
            // one [B*T, hidden] @ [hidden, 2*intermediate] GEMM replaces two
            // [B*T, hidden] @ [hidden, intermediate] matmuls. Skipped when
            // Marlin is going to pack gate/up (the packed path
            // needs the separate projections).
            let gate_up_proj_t = {
                #[cfg(any(feature = "cuda", feature = "rocm"))]
                {
                    if !w4a16_enabled && projection_load_policy.cache_mlp_gate_up_transpose_concat {
                        Some(
                            Tensor::cat(&[&gate_proj_t, &up_proj_t], LAST_DIM)?
                                .contiguous()
                                .context(ctx("gate_up_proj_t contiguous"))?,
                        )
                    } else {
                        None
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "rocm")))]
                {
                    None
                }
            };
            let (gate_up_proj_w8, down_proj_w8) = {
                #[cfg(feature = "rocm")]
                {
                    if w8a16_enabled
                        && !w4a16_enabled
                        && projection_load_policy.pack_w8a16_projection_rows
                    {
                        let gate_up_rows = Tensor::cat(&[&gate_proj, &up_proj], 0)?
                            .contiguous()
                            .context(ctx("w8 gate_up rows contiguous"))?;
                        (
                            crate::rocm_w8_proj::pack_from_bf16_rows(&gate_up_rows)
                                .context(ctx("w8 gate_up pack"))?,
                            crate::rocm_w8_proj::pack_from_bf16_rows(&down_proj)
                                .context(ctx("w8 down pack"))?,
                        )
                    } else {
                        (None, None)
                    }
                }
                #[cfg(not(feature = "rocm"))]
                {
                    (None, None)
                }
            };
            let mlp = GpuFfnWeights {
                gate_proj,
                up_proj,
                down_proj,
                gate_proj_t,
                up_proj_t,
                down_proj_t,
                gate_up_proj_t,
                gate_proj_marlin: None,
                up_proj_marlin: None,
                down_proj_marlin: None,
                gate_up_proj_w8,
                down_proj_w8,
            };

            layers.push(GpuLayerWeights {
                input_layernorm,
                post_attention_layernorm,
                attention,
                mlp,
            });
            if w4a16_enabled && upload_policy.max_bytes_per_second().is_some() {
                flush_marlin_pack_inputs(
                    &mut marlin_pack_inputs,
                    &mut marlin_pack_meta,
                    &mut layers,
                    device,
                    "layer",
                )
                .context(ctx("paced marlin pack"))?;
            }
            source_bytes_completed = layer_source_bytes_completed;
            upload_pacer.checkpoint("layer", source_bytes_completed, i + 1)?;
        }

        // Batch-pack the queued Marlin projections in parallel. On
        // Qwen3.5-4B this is 8 × q_proj + 96 × MLP = 104 projections. The
        // CPU-bound `quantize_and_pack` work now runs across every
        // available worker thread (rayon's default pool) while the
        // GPU↔CPU copies stay sequential inside
        // `pack_from_bf16_batch`; its CPU phase uses the fixed parallel path.
        if w4a16_enabled {
            // (#1082) `pack_from_bf16_batch` is kt-native now: the kt weights
            // go straight in, with no kt-to-candle bridge.
            flush_marlin_pack_inputs(
                &mut marlin_pack_inputs,
                &mut marlin_pack_meta,
                &mut layers,
                device,
                "model",
            )?;
        }

        // Preserve native MTP tensors for explicit model-library and offline
        // qualification callers without uploading them during model load. The
        // serving binary defers MTP CPU loading and rejects enabled speculative
        // methods at startup, so this latent slot is not evidence of a serving
        // route or of already-resident MTP weights.
        let mtp = if let Some(mtp_w) = weights.mtp.as_ref() {
            Some(MtpGpuWeightsSlot::lazy(mtp_w.clone(), device))
        } else {
            weights
                .deferred_mtp
                .as_ref()
                .map(|source| MtpGpuWeightsSlot::lazy_deferred(source.clone(), device))
        };

        if projection_load_cache.synchronizes_after_dropping_originals() {
            // #1082: kt-native Metal queue drain after freeing the projection
            // originals. The backend projection-load policy only enables this
            // for Metal. `wait_until_completed` drains the same MetalCompanion
            // command pool the compute ran on, so no pending GPU write can read
            // a freed buffer. The Vulkan lane never reaches here (Vulkan tensors
            // are CPU-resident), so it is excluded from this arm. (#1082)
            #[cfg(feature = "metal")]
            {
                let idx = if let Device::Metal(i) = device { *i } else { 0 };
                kiln_tensor::primary_metal_companion(idx)
                    .and_then(|c| c.wait_until_completed())
                    .map_err(|e| anyhow::anyhow!("{e}"))
                    .context("synchronize after dropping Metal projection originals")?;
            }
            tracing::info!("Metal projection original buffer cache swept after load");
        }

        upload_pacer.checkpoint("complete", source_bytes_completed, weights.layers.len())?;

        Ok(Self {
            source_content_sha256: weights.source_content_sha256.clone(),
            base_weight_shard_manifest: weights.base_weight_shard_manifest().cloned(),
            execution_provenance: None,
            embed_tokens,
            embed_tokens_t,
            lm_head_w8,
            layers,
            final_norm,
            rotary_inv_freq,
            mtp,
        })
    }

    /// kt-typed parallel entry to [`Self::from_model_weights`] (#1082 Tier 3).
    ///
    /// Takes a `kiln_tensor::Device` instead of candle's `Device`,
    /// bridges at the boundary, and delegates to the existing
    /// candle-typed constructor. The returned `GpuWeights` still holds
    /// candle Tensors internally — the kt typing applies only to the
    /// public surface so kiln-server can call this without importing
    /// candle at the call site.
    ///
    /// Errors when the kt Device has no candle equivalent on this build
    /// (e.g. `Vulkan(_)`; kiln-server's Vulkan path uses a CPU candle
    /// device by convention — pass `kiln_tensor::Device::Cpu` instead).
    ///
    /// Always-on (no cuda feature gate): only uses
    /// `kiln_kt_bridge::candle_device_from_kt`, which is a pure
    /// `candle <-> kt` Device enum mapping with no CUDA toolchain
    /// dependency. (#1082)
    pub fn from_model_weights_kt(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &kiln_tensor::Device,
    ) -> Result<Self> {
        // #1082: post-flip `from_model_weights` already takes a kt `Device`
        // and bridges to candle internally where needed, so this kt entry is
        // now a straight passthrough (kept for the kiln-server call site).
        Self::from_model_weights(weights, config, device)
    }

    pub fn from_model_weights_kt_with_upload_policy(
        weights: &ModelWeights,
        config: &kiln_core::config::ModelConfig,
        device: &kiln_tensor::Device,
        upload_policy: &AcceleratorWeightUploadPolicy,
    ) -> Result<Self> {
        Self::from_model_weights_with_upload_policy(weights, config, device, upload_policy)
    }
}
