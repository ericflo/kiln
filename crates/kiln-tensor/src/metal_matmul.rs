//! Kiln-owned matrix-core Metal GEMM (#1082) — `MatmulOp::metal_fwd`.
//!
//! Replaces the host round-trip that `dispatch2` would otherwise take for
//! matmul on Metal (matmul has no native Metal kernel → it ran the GEMM on
//! the CPU). This is the compute-bound path: prefill QKV/O, MLP gate/up/down
//! at seq>1, the LM head, and any bs>1 decode. It uses Apple's matrix units
//! via `simdgroup_matrix` — kiln owns the MSL ([`gen_steel_msl`]) and
//! dispatches it through objc2-metal via the `MetalCompanion` (NOT candle's
//! `call_mlx_gemm`).
//!
//! The primary kernel is the **kiln steel GEMM** ([`gen_steel_msl`]): a
//! specialized port of MLX's steel_gemm technique. `C[M,N] = A[M,K] @ B[K,N]`,
//! BF16 in/out, **F32 accumulation**, weight pre-transposed to `B=[K,N]`. Its
//! three load-bearing techniques (vs the ~12x-slower naive kernel that remains
//! the arbitrary-K fallback below):
//!
//! 1. **BF16 staging + manual F32 fragment fill** via `thread_elements()` —
//!    the MMA runs on `simdgroup_matrix<float>` (fast), never the M1-emulated
//!    `simdgroup_matrix<bfloat>` (~5x slower).
//! 2. **Direct register→device store** (no F32 `Cs` round-trip → ~5 KiB
//!    threadgroup → high occupancy).
//! 3. **`unroll(full)` on the fragment loops** so `results[]`/`Af[]`/`Bf[]`
//!    stay register-resident — without it, TM*TN≥16 spills to memory and
//!    collapses to ~40 GFLOP/s.
//!
//! On the M1 at the QKV prefill shape: ~1589 GFLOP/s (89% of candle's MLX
//! reference; the naive kernel was ~140). See [`STEEL_CFG`] for tuning.
//!
//! M=1 decode (memory-bound GEMV) is handled correctly here (boundary-tile
//! path) but the matrix units idle, so the model's `metal_transposed_coop_gemv`
//! remains the production decode-projection fast path.

use std::sync::Arc;

use crate::metal_types::{ComputePipeline, buffer_o_kt};
use crate::{DType, Error, MetalStorage, Result};

/// Compile options matching candle's GEMM path: fast math + fast FP
/// functions. The default (`None`) compile path is precise/relaxed; the
/// MLX reference is compiled with these, so the kiln steel kernel must
/// match to be measured on equal footing (and to let the compiler use the
/// cheaper FP intrinsics in the hot MMA loop).
fn fast_compile_options() -> objc2::rc::Retained<objc2_metal::MTLCompileOptions> {
    use objc2_metal::{MTLCompileOptions, MTLMathFloatingPointFunctions, MTLMathMode};
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Fast);
    opts.setMathFloatingPointFunctions(MTLMathFloatingPointFunctions::Fast);
    opts
}

const KILN_GEMM_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 32;    // K-tile (multiple of 8); fewer barriers
constant constexpr uint WM = 2;     // simdgroups down M
constant constexpr uint WN = 2;     // simdgroups across N
constant constexpr uint TM = BM / (8 * WM);  // 4 row-fragments per simd
constant constexpr uint TN = BN / (8 * WN);  // 4 col-fragments per simd

// C[M,N] = A[M,K] @ B[K,N], C[M,N] = A[K,M]^T @ B[K,N] when
// a_transposed != 0, or C[M,N] = A[M,K] @ B[N,K]^T when b_transposed != 0.
// BF16 in/out, f32 accumulate.
// One threadgroup (128 threads = 4 simdgroups) computes a 64x64 C tile.
// Inputs are staged to threadgroup BF16; simdgroup_load up-converts BF16->F32
// fragments (there is no simdgroup_matrix<bfloat> MMA). F32 accumulation.
kernel void kiln_gemm_bf16(
    device const bfloat* A    [[buffer(0)]],   // [batch,M,K] or [batch,K,M] row-major (+offset)
    device const bfloat* B    [[buffer(1)]],   // [batch,K,N] or [batch,N,K] row-major (+offset)
    device bfloat* C          [[buffer(2)]],   // [batch,M,N] row-major
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& a_bs       [[buffer(6)]],   // A per-batch element stride
    constant uint& b_bs       [[buffer(7)]],   // B per-batch element stride
    constant uint& c_bs       [[buffer(8)]],   // C per-batch element stride
    constant uint& a_transposed [[buffer(9)]],
    constant uint& b_transposed [[buffer(10)]],
    uint3 tg   [[threadgroup_position_in_grid]],   // x=n-tile, y=m-tile, z=batch
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint  lane [[thread_index_in_simdgroup]])      // 0..31
{
    threadgroup bfloat As[BM * BK];  // [BM][BK] ld=BK
    threadgroup bfloat Bs[BK * BN];  // [BK][BN] ld=BN
    threadgroup float Cs[BM * BN];   // [BM][BN] ld=BN (store staging)

    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    device const bfloat* Ab = A + tg.z * a_bs;
    device const bfloat* Bb = B + tg.z * b_bs;
    device bfloat* Cb = C + tg.z * c_bs;

    const uint sm = sgid / WN;          // 0..1
    const uint sn = sgid % WN;          // 0..1
    const uint subm = sm * (8 * TM);    // this simd's row base in the tile (0 or 32)
    const uint subn = sn * (8 * TN);    // this simd's col base in the tile (0 or 32)
    const uint tid = sgid * 32 + lane;  // 0..127

    simdgroup_float8x8 acc[TM * TN];
    for (uint i = 0; i < TM * TN; i++) {
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }

    for (uint k0 = 0; k0 < K; k0 += BK) {
        // Cooperative, bounds-checked stage of the A and B K-tiles (BF16).
        for (uint idx = tid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint ar = m0 + r;
            uint ac = k0 + c;
            if (ar < M && ac < K) {
                As[idx] = (a_transposed != 0) ? Ab[ac * M + ar] : Ab[ar * K + ac];
            } else {
                As[idx] = (bfloat)0;
            }
        }
        for (uint idx = tid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint br = k0 + r;
            uint bc = n0 + c;
            if (br < K && bc < N) {
                Bs[idx] = (b_transposed != 0) ? Bb[bc * K + br] : Bb[br * N + bc];
            } else {
                Bs[idx] = (bfloat)0;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {
            // BF16 fragments load directly from BF16 staging (type-matched);
            // the mixed-type MMA up-converts to the F32 accumulator.
            simdgroup_matrix<bfloat, 8, 8> af[TM];
            simdgroup_matrix<bfloat, 8, 8> bf[TN];
            for (uint i = 0; i < TM; i++) {
                simdgroup_load(af[i], As + (subm + i * 8) * BK + kk, BK);
            }
            for (uint j = 0; j < TN; j++) {
                simdgroup_load(bf[j], Bs + kk * BN + (subn + j * 8), BN);
            }
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j++) {
                    simdgroup_multiply_accumulate(acc[i * TN + j], af[i], bf[j], acc[i * TN + j]);
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Each simdgroup stores its 4x4 fragments into disjoint Cs regions (no race).
    for (uint i = 0; i < TM; i++) {
        for (uint j = 0; j < TN; j++) {
            simdgroup_store(acc[i * TN + j], Cs + (subm + i * 8) * BN + (subn + j * 8), BN);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Cooperative, bounds-checked cast F32->BF16 to device C.
    for (uint idx = tid; idx < BM * BN; idx += 128) {
        uint r = idx / BN;
        uint c = idx % BN;
        uint cr = m0 + r;
        uint cc = n0 + c;
        if (cr < M && cc < N) {
            Cb[cr * N + cc] = bfloat(Cs[idx]);
        }
    }
}
"#;

const TILE: usize = 64; // BM == BN (naive fallback)
const TG_THREADS: usize = 128;

/// Production config for the kiln steel GEMM: `BM=BN=64, BK=16, WM=WN=2`
/// (TM*TN=16 accumulators/simdgroup, 128 threads). With the `unroll(full)`
/// pragmas keeping the fragment arrays register-resident, this measured
/// **~1589 GFLOP/s** at the QKV prefill shape — 89% of candle's MLX
/// reference (~1777) and 11x the naive kernel (~140). It beats the smaller
/// 32x32 tile (~1213) because the larger 64x64 tile has higher arithmetic
/// intensity (more reuse per threadgroup-memory load). TM*TN=32 (e.g.
/// WM=1,WN=2 — MLX's own generic pick) still register-spills here (~43),
/// so 64x64 W2x2 is the kiln optimum on the M1.
const STEEL_CFG: GemmCfg = GemmCfg {
    bm: 64,
    bn: 64,
    bk: 16,
    wm: 2,
    wn: 2,
    stg: "bfloat",
};

/// Process-wide compiled kiln steel-GEMM pipeline, one per Metal device.
fn steel_pipeline(metal: &MetalStorage) -> Result<ComputePipeline> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let companion = metal.companion()?;
    let key = companion.device_id();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache
            .lock()
            .map_err(|e| Error::Msg(format!("metal_matmul: steel cache poisoned: {e}")))?;
        if let Some(p) = map.get(&key) {
            return Ok(p.clone());
        }
    }
    let msl = gen_steel_msl(&STEEL_CFG);
    let opts = fast_compile_options();
    let lib = companion
        .device()
        .new_library_with_source(&msl, Some(&opts))
        .map_err(|e| Error::Msg(format!("metal_matmul: compile kiln_gemm library: {e:?}")))?;
    let func = lib
        .get_function("kiln_gemm", None)
        .map_err(|e| Error::Msg(format!("metal_matmul: load kiln_gemm function: {e:?}")))?;
    let pipeline = companion
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| Error::Msg(format!("metal_matmul: build kiln_gemm pipeline: {e:?}")))?;
    let mut map = cache
        .lock()
        .map_err(|e| Error::Msg(format!("metal_matmul: steel cache poisoned: {e}")))?;
    map.insert(key, pipeline.clone());
    Ok(pipeline)
}

/// Process-wide compiled `kiln_gemm_bf16` pipeline, one per Metal device.
fn gemm_pipeline(metal: &MetalStorage) -> Result<ComputePipeline> {
    use std::collections::HashMap;
    use std::sync::{Mutex, OnceLock};
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let companion = metal.companion()?;
    let key = companion.device_id();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    {
        let map = cache
            .lock()
            .map_err(|e| Error::Msg(format!("metal_matmul: pipeline cache poisoned: {e}")))?;
        if let Some(p) = map.get(&key) {
            return Ok(p.clone());
        }
    }
    let lib = companion
        .device()
        .new_library_with_source(KILN_GEMM_MSL, None)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_matmul: compile kiln_gemm_bf16 library: {e:?}"
            ))
        })?;
    let func = lib
        .get_function("kiln_gemm_bf16", None)
        .map_err(|e| Error::Msg(format!("metal_matmul: load kiln_gemm_bf16 function: {e:?}")))?;
    let pipeline = companion
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| {
            Error::Msg(format!(
                "metal_matmul: build kiln_gemm_bf16 pipeline: {e:?}"
            ))
        })?;
    let mut map = cache
        .lock()
        .map_err(|e| Error::Msg(format!("metal_matmul: pipeline cache poisoned: {e}")))?;
    map.insert(key, pipeline.clone());
    Ok(pipeline)
}

#[inline]
fn kt_metal<'a>(t: &'a crate::Tensor, which: &str) -> Result<&'a MetalStorage> {
    t.storage()
        .as_any()
        .downcast_ref::<MetalStorage>()
        .ok_or_else(|| Error::Msg(format!("metal_matmul: {which} must be Metal-backed")))
}

/// BF16 matrix-core GEMM `a [..,M,K] @ b [..,K,N] -> [..,M,N]` on Metal.
///
/// Caller (`MatmulOp::metal_fwd`) has already gated dtype == BF16 and
/// contiguity, and the op's `validate()` checked rank/contraction-dim match.
pub fn metal_matmul(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    let ar = a.rank();
    let m = a_shape[ar - 2];
    let k = a_shape[ar - 1];
    let n = b_shape[b.rank() - 1];
    let batch_dims: Vec<usize> = a_shape[..ar - 2].to_vec();
    let batch: usize = batch_dims.iter().product::<usize>().max(1);

    if k != b_shape[b.rank() - 2] {
        return Err(Error::Msg(format!(
            "metal_matmul: contraction mismatch a.K={k} vs b.K={}",
            b_shape[b.rank() - 2]
        )));
    }
    for &d in &[m, n, k, batch] {
        if d > u32::MAX as usize {
            return Err(Error::Msg(format!("metal_matmul: dim {d} exceeds u32")));
        }
    }

    let a_metal = kt_metal(a, "a")?;
    let b_metal = kt_metal(b, "b")?;
    let companion = a_metal.companion()?;
    let device_index = a_metal.device_index();

    // Output: [batch_dims.., M, N] contiguous BF16.
    let mut out_shape = batch_dims;
    out_shape.push(m);
    out_shape.push(n);
    let out_storage =
        MetalStorage::zeros_kt(companion.device(), device_index, DType::BF16, batch * m * n)?;
    let out = crate::Tensor::from_parts(
        Arc::new(out_storage),
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )?;
    if batch * m * n == 0 {
        return Ok(out);
    }

    // Steel GEMM (the fast path) requires K % BK == 0; otherwise fall back
    // to the naive kernel which handles arbitrary K. Qwen K∈{2560,9216,4096}
    // are all multiples of 16, so production always takes the steel path.
    let use_steel = k % STEEL_CFG.bk == 0;
    let (pipeline, tile_m, tile_n, threads, label) = if use_steel {
        (
            steel_pipeline(a_metal)?,
            STEEL_CFG.bm,
            STEEL_CFG.bn,
            STEEL_CFG.threads(),
            "kiln_gemm",
        )
    } else {
        (
            gemm_pipeline(a_metal)?,
            TILE,
            TILE,
            TG_THREADS,
            "kiln_gemm_bf16",
        )
    };

    let out_metal = kt_metal(&out, "out")?;
    let encoder = companion.command_encoder()?;
    encoder.set_label(label);
    encoder.set_compute_pipeline_state(&pipeline);

    let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
    let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
    let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
    encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
    encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
    encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);
    // Per-batch element strides for contiguous inputs (a:[..,M,K], b:[..,K,N]).
    let a_bs = (m * k) as u32;
    let b_bs = (k * n) as u32;
    let c_bs = (m * n) as u32;
    encoder.set_bytes(3, &m_u);
    encoder.set_bytes(4, &n_u);
    encoder.set_bytes(5, &k_u);
    encoder.set_bytes(6, &a_bs);
    encoder.set_bytes(7, &b_bs);
    encoder.set_bytes(8, &c_bs);
    if !use_steel {
        let a_transposed = 0u32;
        let b_transposed = 0u32;
        encoder.set_bytes(9, &a_transposed);
        encoder.set_bytes(10, &b_transposed);
    }

    let grid = objc2_metal::MTLSize {
        width: n.div_ceil(tile_n),
        height: m.div_ceil(tile_m),
        depth: batch,
    };
    let tg = objc2_metal::MTLSize {
        width: threads,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, tg);
    drop(encoder);

    Ok(out)
}

/// BF16 matrix-core GEMM `a^T @ b` without materialising the transposed left
/// input on Metal.
///
/// `a` is stored as `[..., K, M]`, `b` as `[..., K, N]`, and the output is
/// `[..., M, N]`. This uses the resident arbitrary-K GEMM kernel with a
/// transposed-A tile loader, avoiding Metal's current
/// host-gather/re-upload path for `a.transpose(-2, -1).contiguous()`.
pub fn metal_matmul_lhs_transposed(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    const OP: &str = "metal_matmul_lhs_transposed";
    let ar = a.rank();
    let br = b.rank();
    if ar < 2 || br < 2 {
        return Err(Error::Msg(format!(
            "{OP}: rank must be >= 2, got a={ar} b={br}"
        )));
    }
    if ar != br {
        return Err(Error::Msg(format!("{OP}: rank mismatch a={ar} b={br}")));
    }
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "{OP}: BF16-only kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(format!("{OP}: contiguous inputs required")));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..ar - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let k = a_shape[ar - 2];
    let m = a_shape[ar - 1];
    let k_b = b_shape[br - 2];
    let n = b_shape[br - 1];
    if k != k_b {
        return Err(Error::Msg(format!(
            "{OP}: contraction mismatch a.K={k} vs b.K={k_b}"
        )));
    }
    for &d in &[m, n, k] {
        if d > u32::MAX as usize {
            return Err(Error::Msg(format!("{OP}: dim {d} exceeds u32")));
        }
    }

    let batch_dims: Vec<usize> = a_shape[..ar - 2].to_vec();
    let batch: usize = batch_dims.iter().product::<usize>().max(1);
    if batch > u32::MAX as usize {
        return Err(Error::Msg(format!("{OP}: batch {batch} exceeds u32")));
    }

    let a_metal = kt_metal(a, "a")?;
    let b_metal = kt_metal(b, "b")?;
    let companion = a_metal.companion()?;
    let device_index = a_metal.device_index();

    let mut out_shape = batch_dims;
    out_shape.push(m);
    out_shape.push(n);
    let out_storage =
        MetalStorage::zeros_kt(companion.device(), device_index, DType::BF16, batch * m * n)?;
    let out = crate::Tensor::from_parts(
        Arc::new(out_storage),
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )?;
    if batch * m * n == 0 {
        return Ok(out);
    }

    let pipeline = gemm_pipeline(a_metal)?;
    let out_metal = kt_metal(&out, "out")?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gemm_lhs_t_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
    let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
    let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
    encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
    encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
    encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);
    let a_bs = (k * m) as u32;
    let b_bs = (k * n) as u32;
    let c_bs = (m * n) as u32;
    let a_transposed = 1u32;
    let b_transposed = 0u32;
    encoder.set_bytes(3, &m_u);
    encoder.set_bytes(4, &n_u);
    encoder.set_bytes(5, &k_u);
    encoder.set_bytes(6, &a_bs);
    encoder.set_bytes(7, &b_bs);
    encoder.set_bytes(8, &c_bs);
    encoder.set_bytes(9, &a_transposed);
    encoder.set_bytes(10, &b_transposed);

    let grid = objc2_metal::MTLSize {
        width: n.div_ceil(TILE),
        height: m.div_ceil(TILE),
        depth: batch,
    };
    let tg = objc2_metal::MTLSize {
        width: TG_THREADS,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, tg);
    drop(encoder);

    Ok(out)
}

/// BF16 matrix-core GEMM `a @ b^T` without materialising the transposed right
/// input on Metal.
///
/// `a` is stored as `[..., M, K]`, `b` as `[..., N, K]`, and the output is
/// `[..., M, N]`. This uses the resident arbitrary-K GEMM kernel with a
/// transposed-B tile loader.
pub fn metal_matmul_rhs_transposed(a: &crate::Tensor, b: &crate::Tensor) -> Result<crate::Tensor> {
    const OP: &str = "metal_matmul_rhs_transposed";
    let ar = a.rank();
    let br = b.rank();
    if ar < 2 || br < 2 {
        return Err(Error::Msg(format!(
            "{OP}: rank must be >= 2, got a={ar} b={br}"
        )));
    }
    if ar != br {
        return Err(Error::Msg(format!("{OP}: rank mismatch a={ar} b={br}")));
    }
    if a.dtype() != DType::BF16 || b.dtype() != DType::BF16 {
        return Err(Error::Msg(format!(
            "{OP}: BF16-only kernel (got a={}, b={})",
            a.dtype(),
            b.dtype()
        )));
    }
    if !a.is_contiguous() || !b.is_contiguous() {
        return Err(Error::Msg(format!("{OP}: contiguous inputs required")));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    for axis in 0..ar - 2 {
        if a_shape[axis] != b_shape[axis] {
            return Err(Error::Msg(format!(
                "{OP}: batch axis {axis} mismatch: a={} b={}",
                a_shape[axis], b_shape[axis]
            )));
        }
    }
    let m = a_shape[ar - 2];
    let k = a_shape[ar - 1];
    let n = b_shape[br - 2];
    let k_b = b_shape[br - 1];
    if k != k_b {
        return Err(Error::Msg(format!(
            "{OP}: contraction mismatch a.K={k} vs b.K={k_b}"
        )));
    }
    for &d in &[m, n, k] {
        if d > u32::MAX as usize {
            return Err(Error::Msg(format!("{OP}: dim {d} exceeds u32")));
        }
    }

    let batch_dims: Vec<usize> = a_shape[..ar - 2].to_vec();
    let batch: usize = batch_dims.iter().product::<usize>().max(1);
    if batch > u32::MAX as usize {
        return Err(Error::Msg(format!("{OP}: batch {batch} exceeds u32")));
    }

    let a_metal = kt_metal(a, "a")?;
    let b_metal = kt_metal(b, "b")?;
    let companion = a_metal.companion()?;
    let device_index = a_metal.device_index();

    let mut out_shape = batch_dims;
    out_shape.push(m);
    out_shape.push(n);
    let out_storage =
        MetalStorage::zeros_kt(companion.device(), device_index, DType::BF16, batch * m * n)?;
    let out = crate::Tensor::from_parts(
        Arc::new(out_storage),
        crate::Layout::contiguous(out_shape),
        crate::TensorId::next(),
    )?;
    if batch * m * n == 0 {
        return Ok(out);
    }

    let pipeline = gemm_pipeline(a_metal)?;
    let out_metal = kt_metal(&out, "out")?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gemm_rhs_t_bf16");
    encoder.set_compute_pipeline_state(&pipeline);

    let a_buf = buffer_o_kt(a_metal.buffer().as_ref(), a.layout(), a.dtype());
    let b_buf = buffer_o_kt(b_metal.buffer().as_ref(), b.layout(), b.dtype());
    let out_buf = buffer_o_kt(out_metal.buffer().as_ref(), out.layout(), out.dtype());
    encoder.set_buffer(0, Some(a_buf.buffer), a_buf.offset_in_bytes);
    encoder.set_buffer(1, Some(b_buf.buffer), b_buf.offset_in_bytes);
    encoder.set_buffer(2, Some(out_buf.buffer), out_buf.offset_in_bytes);

    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);
    let a_bs = (m * k) as u32;
    let b_bs = (n * k) as u32;
    let c_bs = (m * n) as u32;
    let a_transposed = 0u32;
    let b_transposed = 1u32;
    encoder.set_bytes(3, &m_u);
    encoder.set_bytes(4, &n_u);
    encoder.set_bytes(5, &k_u);
    encoder.set_bytes(6, &a_bs);
    encoder.set_bytes(7, &b_bs);
    encoder.set_bytes(8, &c_bs);
    encoder.set_bytes(9, &a_transposed);
    encoder.set_bytes(10, &b_transposed);

    let grid = objc2_metal::MTLSize {
        width: n.div_ceil(TILE),
        height: m.div_ceil(TILE),
        depth: batch,
    };
    let tg = objc2_metal::MTLSize {
        width: TG_THREADS,
        height: 1,
        depth: 1,
    };
    encoder.dispatch_thread_groups(grid, tg);
    drop(encoder);

    Ok(out)
}

// ----------------------------------------------------------------------
// GEMM config sweep harness (#1082) — finds the hardware-maxing config.
// ----------------------------------------------------------------------
//
// `metal_matmul`'s kernel above is one point in a large design space
// (tile size, K-tile, staging dtype, threadgroup-memory layout). On the
// M1 the first cut measured ~30 GFLOP/s — a ~70x gap from peak — so the
// config matters enormously. Rather than guess-and-recompile (each cargo
// cycle is ~80s), this harness generates many MSL variants, compiles each
// in-process (~1-2s each), and benchmarks them at a full-tile Qwen shape.
// `#[doc(hidden)]` + driven by `tests/metal_gemm_sweep.rs`.

/// One point in the GEMM tuning space.
#[doc(hidden)]
#[derive(Clone, Copy, Debug)]
pub struct GemmCfg {
    pub bm: usize,
    pub bn: usize,
    pub bk: usize,
    pub wm: usize,
    pub wn: usize,
    /// Threadgroup staging dtype: `"float"` (cast BF16→F32 on load) or
    /// `"bfloat"` (mixed-type MMA into the F32 accumulator).
    pub stg: &'static str,
}

impl GemmCfg {
    fn threads(&self) -> usize {
        self.wm * self.wn * 32
    }
    fn label(&self) -> String {
        format!(
            "BM{} BN{} BK{} W{}x{} {}",
            self.bm, self.bn, self.bk, self.wm, self.wn, self.stg
        )
    }
}

/// Emit a parameterized matrix-core GEMM. `C[M,N]=A[M,K]@B[K,N]`, BF16
/// in/out, F32 accumulate. A single `threadgroup float` pool is carved
/// into the A/B staging tiles (during the K-loop) and reused as the C
/// store-staging (after the K-loop, separated by a barrier) — so the
/// threadgroup-memory footprint is `max(stage_bytes, store_bytes)`, not
/// their sum. That aliasing is the key occupancy lever on the M1's 32 KiB
/// threadgroup budget.
#[doc(hidden)]
pub fn gen_gemm_msl(c: &GemmCfg) -> String {
    let (bm, bn, bk, wm, wn) = (c.bm, c.bn, c.bk, c.wm, c.wn);
    let tm = bm / (8 * wm);
    let tn = bn / (8 * wn);
    let nthreads = c.threads();
    let stg = c.stg;
    let stg_bytes = if stg == "float" { 4 } else { 2 };
    // Shared pool sized to the larger of the two phases (in f32 units).
    let stage_floats = (bm * bk + bk * bn) * stg_bytes / 4;
    let store_floats = bm * bn; // Cs is f32
    let pool_floats = stage_floats.max(store_floats);
    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = {bm};
constant constexpr uint BN = {bn};
constant constexpr uint BK = {bk};
constant constexpr uint WM = {wm};
constant constexpr uint WN = {wn};
constant constexpr uint TM = {tm};
constant constexpr uint TN = {tn};
constant constexpr uint NTHREADS = {nthreads};
constant constexpr uint POOL = {pool_floats};

kernel void kiln_gemm_sweep(
    device const bfloat* A    [[buffer(0)]],
    device const bfloat* B    [[buffer(1)]],
    device bfloat* C          [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& a_bs       [[buffer(6)]],
    constant uint& b_bs       [[buffer(7)]],
    constant uint& c_bs       [[buffer(8)]],
    uint3 tg   [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
    threadgroup float pool[POOL];
    threadgroup {stg}* As = (threadgroup {stg}*)pool;       // [BM][BK] ld=BK
    threadgroup {stg}* Bs = As + (BM * BK);                 // [BK][BN] ld=BN
    threadgroup float* Cs = pool;                           // [BM][BN] ld=BN

    const uint m0 = tg.y * BM;
    const uint n0 = tg.x * BN;
    device const bfloat* Ab = A + tg.z * a_bs;
    device const bfloat* Bb = B + tg.z * b_bs;
    device bfloat* Cb = C + tg.z * c_bs;

    const uint sm = sgid / WN;
    const uint sn = sgid % WN;
    const uint subm = sm * (8 * TM);
    const uint subn = sn * (8 * TN);
    const uint tid = sgid * 32 + lane;

    simdgroup_float8x8 acc[TM * TN];
    for (uint i = 0; i < TM * TN; i++) {{
        acc[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }}

    for (uint k0 = 0; k0 < K; k0 += BK) {{
        for (uint idx = tid; idx < BM * BK; idx += NTHREADS) {{
            uint r = idx / BK;
            uint cc = idx % BK;
            uint ar = m0 + r;
            uint ac = k0 + cc;
            As[idx] = ({stg})((ar < M && ac < K) ? Ab[ar * K + ac] : (bfloat)0);
        }}
        for (uint idx = tid; idx < BK * BN; idx += NTHREADS) {{
            uint r = idx / BN;
            uint cc = idx % BN;
            uint br = k0 + r;
            uint bc = n0 + cc;
            Bs[idx] = ({stg})((br < K && bc < N) ? Bb[br * N + bc] : (bfloat)0);
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint kk = 0; kk < BK; kk += 8) {{
            simdgroup_matrix<{stg}, 8, 8> af[TM];
            simdgroup_matrix<{stg}, 8, 8> bf[TN];
            for (uint i = 0; i < TM; i++) {{
                simdgroup_load(af[i], As + (subm + i * 8) * BK + kk, BK);
            }}
            for (uint j = 0; j < TN; j++) {{
                simdgroup_load(bf[j], Bs + kk * BN + (subn + j * 8), BN);
            }}
            for (uint i = 0; i < TM; i++) {{
                for (uint j = 0; j < TN; j++) {{
                    simdgroup_multiply_accumulate(acc[i * TN + j], af[i], bf[j], acc[i * TN + j]);
                }}
            }}
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }}

    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < TM; i++) {{
        for (uint j = 0; j < TN; j++) {{
            simdgroup_store(acc[i * TN + j], Cs + (subm + i * 8) * BN + (subn + j * 8), BN);
        }}
    }}
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < BM * BN; idx += NTHREADS) {{
        uint r = idx / BN;
        uint cc = idx % BN;
        uint cr = m0 + r;
        uint ccol = n0 + cc;
        if (cr < M && ccol < N) {{
            Cb[cr * N + ccol] = bfloat(Cs[idx]);
        }}
    }}
}}
"#
    )
}

/// Emit the kiln "steel" GEMM — a specialized port of MLX's steel_gemm
/// technique (the SOTA Apple-Silicon matrix-core GEMM). `C[M,N]=A[M,K]@B[K,N]`,
/// BF16 in/out, **F32 accumulate**, A and B both row-major (non-transposed:
/// the weight is pre-transposed to `[K,N]`). The three techniques that
/// close the ~12x gap from the naive kernel above:
///
/// 1. **BF16 threadgroup staging + manual F32 fragment fill.** Stage BF16
///    (half the threadgroup-memory bandwidth) but fill the simdgroup
///    fragments via `thread_elements()` with a per-lane `static_cast<float>`
///    — the MMA then runs on `simdgroup_matrix<float>` (fast), never on the
///    M1-emulated `simdgroup_matrix<bfloat>` (which measured ~5x slower).
/// 2. **Direct register→device store.** Each lane writes its 2 accumulated
///    elements straight to device `D` via `thread_elements()` — no F32 `Cs`
///    threadgroup round-trip. Drops the per-threadgroup footprint to ~5 KiB
///    (vs 16-24 KiB), so far more threadgroups stay resident → latency hides.
/// 3. **+8 bank-conflict padding** on the threadgroup leading dims, and
///    **vectorized loads** (a `VecN` struct copy = one wide transaction).
///
/// Requires `BM*BK % TGP == 0`, `BN*BK % TGP == 0`, `K % BK == 0`
/// (Qwen K∈{2560,9216,4096} are all multiples of 8/16/32). M and N may be
/// unaligned to the tile (per-tile fast/safe loader + store branch).
#[doc(hidden)]
pub fn gen_steel_msl(c: &GemmCfg) -> String {
    let (bm, bn, bk, wm, wn) = (c.bm, c.bn, c.bk, c.wm, c.wn);
    let tgp = wm * wn * 32;
    let pad = 8usize; // 16 bytes / sizeof(bf16)
    let lda = bk + pad; // As is [BM][BK+pad]
    let ldb = bn + pad; // Bs is [BK][BN+pad]
    let tm_stride = 8 * wm;
    let tn_stride = 8 * wn;
    let tm = bm / tm_stride;
    let tn = bn / tn_stride;
    // BlockLoader geometry (MLX formulas), A: [BM rows × BK cols].
    let vec_a = bm * bk / tgp;
    let tcols_a = bk / vec_a;
    let trows_a = tgp / tcols_a;
    let n_rows_a = bm.div_ceil(trows_a);
    // B: [BK rows × BN cols].
    let vec_b = bn * bk / tgp;
    let tcols_b = bn / vec_b;
    let trows_b = tgp / tcols_b;
    let n_rows_b = bk.div_ceil(trows_b);
    let simd_stride_a = tm_stride * lda;
    let simd_stride_b = tn_stride;
    let tile_stride_a = 8usize;
    let tile_stride_b = 8 * ldb;

    format!(
        r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = {bm};
constant constexpr uint BN = {bn};
constant constexpr uint BK = {bk};
constant constexpr uint WM = {wm};
constant constexpr uint WN = {wn};
constant constexpr uint LDA = {lda};
constant constexpr uint LDB = {ldb};
constant constexpr uint TM = {tm};
constant constexpr uint TN = {tn};
constant constexpr uint TM_STRIDE = {tm_stride};
constant constexpr uint TN_STRIDE = {tn_stride};
constant constexpr uint SIMD_STRIDE_A = {simd_stride_a};
constant constexpr uint SIMD_STRIDE_B = {simd_stride_b};
constant constexpr uint TILE_STRIDE_A = {tile_stride_a};
constant constexpr uint TILE_STRIDE_B = {tile_stride_b};

struct VecA {{ bfloat v[{vec_a}]; }};
struct VecB {{ bfloat v[{vec_b}]; }};

kernel void kiln_gemm(
    device const bfloat* A    [[buffer(0)]],
    device const bfloat* B    [[buffer(1)]],
    device bfloat* D          [[buffer(2)]],
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& a_bs       [[buffer(6)]],
    constant uint& b_bs       [[buffer(7)]],
    constant uint& c_bs       [[buffer(8)]],
    uint3 tid  [[threadgroup_position_in_grid]],
    uint  sgid [[simdgroup_index_in_threadgroup]],
    uint  lane [[thread_index_in_simdgroup]])
{{
    threadgroup bfloat As[BM * LDA];
    threadgroup bfloat Bs[BK * LDB];

    const device bfloat* Ab = A + tid.z * a_bs;
    const device bfloat* Bb = B + tid.z * b_bs;
    device bfloat* Db = D + tid.z * c_bs;

    const uint c_row = tid.y * BM;
    const uint c_col = tid.x * BN;
    if (c_row >= M || c_col >= N) return;

    Ab += (size_t)c_row * K;            // A[c_row..][..]
    Bb += c_col;                        // B[..][c_col..]
    Db += (size_t)c_row * N + c_col;    // D[c_row..][c_col..]

    const uint thread_idx = sgid * 32 + lane;

    // BlockLoader thread coordinates.
    const uint bi_a = thread_idx / {tcols_a};
    const uint bj_a = {vec_a} * (thread_idx % {tcols_a});
    const uint bi_b = thread_idx / {tcols_b};
    const uint bj_b = {vec_b} * (thread_idx % {tcols_b});

    // BlockMMA thread -> simdgroup-matrix element map (MLX layout).
    const uint tm = 8 * (sgid / WN);
    const uint tn = 8 * (sgid % WN);
    const uint qid = lane / 4;
    const uint sm = (qid & 4) + (lane / 2) % 4;
    const uint sn = (qid & 2) * 2 + (lane % 2) * 2;
    const uint As_off = sn + (tm + sm) * LDA;
    const uint Bs_off = sm * LDB + (tn + sn);

    simdgroup_matrix<float, 8, 8> results[TM * TN];
    for (uint i = 0; i < TM * TN; i++) {{
        results[i] = make_filled_simdgroup_matrix<float, 8>(0.0f);
    }}

    const uint tgp_bm = min(BM, M - c_row);
    const uint tgp_bn = min(BN, N - c_col);
    const bool a_full = (tgp_bm == BM);
    const bool b_full = (tgp_bn == BN);
    const uint k_iters = K / BK;

    for (uint kit = 0; kit < k_iters; kit++) {{
        const uint k0 = kit * BK;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- load A tile [BM][BK] (rows=M, cols=K) ----
        if (a_full) {{
            #pragma clang loop unroll(full)
            for (uint i = 0; i < {n_rows_a}; i++) {{
                uint row = bi_a + i * {trows_a};
                *((threadgroup VecA*)(As + row * LDA + bj_a)) =
                    *((const device VecA*)(Ab + (size_t)row * K + k0 + bj_a));
            }}
        }} else {{
            #pragma clang loop unroll(full)
            for (uint i = 0; i < {n_rows_a}; i++) {{
                uint row = bi_a + i * {trows_a};
                #pragma clang loop unroll(full)
                for (uint v = 0; v < {vec_a}; v++) {{
                    As[row * LDA + bj_a + v] =
                        (row < tgp_bm) ? Ab[(size_t)row * K + k0 + bj_a + v] : (bfloat)0;
                }}
            }}
        }}

        // ---- load B tile [BK][BN] (rows=K, cols=N) ----
        if (b_full) {{
            #pragma clang loop unroll(full)
            for (uint i = 0; i < {n_rows_b}; i++) {{
                uint row = bi_b + i * {trows_b};
                *((threadgroup VecB*)(Bs + row * LDB + bj_b)) =
                    *((const device VecB*)(Bb + (size_t)(k0 + row) * N + bj_b));
            }}
        }} else {{
            #pragma clang loop unroll(full)
            for (uint i = 0; i < {n_rows_b}; i++) {{
                uint row = bi_b + i * {trows_b};
                #pragma clang loop unroll(full)
                for (uint v = 0; v < {vec_b}; v++) {{
                    Bs[row * LDB + bj_b + v] =
                        ((bj_b + v) < tgp_bn) ? Bb[(size_t)(k0 + row) * N + bj_b + v] : (bfloat)0;
                }}
            }}
        }}

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // ---- MMA: fill float fragments from BF16 staging, accumulate ----
        threadgroup const bfloat* As_p = As + As_off;
        threadgroup const bfloat* Bs_p = Bs + Bs_off;
        simdgroup_matrix<float, 8, 8> Af[TM];
        simdgroup_matrix<float, 8, 8> Bf[TN];
        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < BK; kk += 8) {{
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {{
                Af[i].thread_elements()[0] = (float)As_p[i * SIMD_STRIDE_A + 0];
                Af[i].thread_elements()[1] = (float)As_p[i * SIMD_STRIDE_A + 1];
            }}
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; j++) {{
                Bf[j].thread_elements()[0] = (float)Bs_p[j * SIMD_STRIDE_B + 0];
                Bf[j].thread_elements()[1] = (float)Bs_p[j * SIMD_STRIDE_B + 1];
            }}
            simdgroup_barrier(mem_flags::mem_none);
            #pragma clang loop unroll(full)
            for (uint i = 0; i < TM; i++) {{
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; j++) {{
                    uint js = (i % 2) ? (TN - 1 - j) : j;
                    simdgroup_multiply_accumulate(
                        results[i * TN + js], Af[i], Bf[js], results[i * TN + js]);
                }}
            }}
            As_p += TILE_STRIDE_A;
            Bs_p += TILE_STRIDE_B;
        }}
    }}

    // ---- store accumulators directly to device ----
    device bfloat* Dp = Db + (sm + tm) * N + (tn + sn);
    if (a_full && b_full) {{
        #pragma clang loop unroll(full)
        for (uint i = 0; i < TM; i++) {{
            #pragma clang loop unroll(full)
            for (uint j = 0; j < TN; j++) {{
                thread const auto& acc = results[i * TN + j].thread_elements();
                uint off = i * TM_STRIDE * N + j * TN_STRIDE;
                Dp[off] = (bfloat)acc[0];
                Dp[off + 1] = (bfloat)acc[1];
            }}
        }}
    }} else {{
        int rem_y = (int)tgp_bm - (int)(sm + tm);
        int rem_x = (int)tgp_bn - (int)(tn + sn);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < TM; i++) {{
            if ((int)(i * TM_STRIDE) < rem_y) {{
                #pragma clang loop unroll(full)
                for (uint j = 0; j < TN; j++) {{
                    thread const auto& acc = results[i * TN + j].thread_elements();
                    uint off = i * TM_STRIDE * N + j * TN_STRIDE;
                    if ((int)(j * TN_STRIDE) < rem_x) {{
                        Dp[off] = (bfloat)acc[0];
                    }}
                    if ((int)(j * TN_STRIDE + 1) < rem_x) {{
                        Dp[off + 1] = (bfloat)acc[1];
                    }}
                }}
            }}
        }}
    }}
}}
"#
    )
}

/// Threadgroup-memory pool footprint (bytes) of a config — must stay
/// under the M1's 32 KiB limit or pipeline creation hard-errors.
#[doc(hidden)]
pub fn gemm_pool_bytes(c: &GemmCfg) -> usize {
    let stg_bytes = if c.stg == "float" { 4 } else { 2 };
    let stage = (c.bm * c.bk + c.bk * c.bn) * stg_bytes;
    let store = c.bm * c.bn * 4;
    stage.max(store)
}

/// Compile + benchmark one config at `[M,K]@[K,N]`; returns GFLOP/s.
#[doc(hidden)]
pub fn bench_gemm_cfg(c: &GemmCfg, m: usize, k: usize, n: usize, iters: usize) -> Result<f64> {
    let pool = gemm_pool_bytes(c);
    if pool > 32768 {
        return Err(Error::Msg(format!(
            "{}: threadgroup pool {pool} B exceeds 32 KiB",
            c.label()
        )));
    }
    let companion = crate::primary_metal_companion(0)?;
    let device = companion.device();
    let msl = gen_gemm_msl(c);
    let lib = device
        .new_library_with_source(&msl, None)
        .map_err(|e| Error::Msg(format!("sweep compile {}: {e:?}", c.label())))?;
    let func = lib
        .get_function("kiln_gemm_sweep", None)
        .map_err(|e| Error::Msg(format!("sweep get_function {}: {e:?}", c.label())))?;
    let pipeline: ComputePipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| Error::Msg(format!("sweep pipeline {}: {e:?}", c.label())))?;

    let a = MetalStorage::zeros_kt(device, 0, DType::BF16, m * k)?;
    let b = MetalStorage::zeros_kt(device, 0, DType::BF16, k * n)?;
    let out = MetalStorage::zeros_kt(device, 0, DType::BF16, m * n)?;
    let la = crate::Layout::contiguous(vec![m, k]);
    let lb = crate::Layout::contiguous(vec![k, n]);
    let lc = crate::Layout::contiguous(vec![m, n]);

    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);
    let (a_bs, b_bs, c_bs) = ((m * k) as u32, (k * n) as u32, (m * n) as u32);
    let grid = objc2_metal::MTLSize {
        width: n.div_ceil(c.bn),
        height: m.div_ceil(c.bm),
        depth: 1,
    };
    let tgsz = objc2_metal::MTLSize {
        width: c.threads(),
        height: 1,
        depth: 1,
    };

    let dispatch = || -> Result<()> {
        let encoder = companion.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let ab = buffer_o_kt(a.buffer().as_ref(), &la, DType::BF16);
        let bb = buffer_o_kt(b.buffer().as_ref(), &lb, DType::BF16);
        let cb = buffer_o_kt(out.buffer().as_ref(), &lc, DType::BF16);
        encoder.set_buffer(0, Some(ab.buffer), ab.offset_in_bytes);
        encoder.set_buffer(1, Some(bb.buffer), bb.offset_in_bytes);
        encoder.set_buffer(2, Some(cb.buffer), cb.offset_in_bytes);
        encoder.set_bytes(3, &m_u);
        encoder.set_bytes(4, &n_u);
        encoder.set_bytes(5, &k_u);
        encoder.set_bytes(6, &a_bs);
        encoder.set_bytes(7, &b_bs);
        encoder.set_bytes(8, &c_bs);
        encoder.dispatch_thread_groups(grid, tgsz);
        drop(encoder);
        Ok(())
    };

    // Warmup + sync.
    dispatch()?;
    companion.wait_until_completed()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        dispatch()?;
    }
    companion.wait_until_completed()?;
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    Ok(gflop / secs)
}

/// Validate a steel config's divisibility requirements; `Err` reason if unbuildable.
#[doc(hidden)]
pub fn steel_cfg_valid(c: &GemmCfg) -> Result<()> {
    let tgp = c.wm * c.wn * 32;
    let chk = |cond: bool, msg: &str| -> Result<()> {
        if cond {
            Ok(())
        } else {
            Err(Error::Msg(format!("{}: {msg}", c.label())))
        }
    };
    chk(c.bm % (8 * c.wm) == 0, "BM not mult of 8*WM")?;
    chk(c.bn % (8 * c.wn) == 0, "BN not mult of 8*WN")?;
    chk(c.bk % 8 == 0, "BK not mult of 8")?;
    chk((c.bm * c.bk) % tgp == 0, "BM*BK not mult of TGP")?;
    chk((c.bn * c.bk) % tgp == 0, "BN*BK not mult of TGP")?;
    let vec_a = c.bm * c.bk / tgp;
    let vec_b = c.bn * c.bk / tgp;
    chk(vec_a > 0 && c.bk % vec_a == 0, "BK not divisible by vec_a")?;
    chk(vec_b > 0 && c.bn % vec_b == 0, "BN not divisible by vec_b")?;
    let tcols_a = c.bk / vec_a;
    let tcols_b = c.bn / vec_b;
    chk(tgp % tcols_a == 0, "TGP not divisible by tcols_a")?;
    chk(tgp % tcols_b == 0, "TGP not divisible by tcols_b")?;
    chk(c.bm % (tgp / tcols_a) == 0, "BM not divisible by trows_a")?;
    chk(c.bk % (tgp / tcols_b) == 0, "BK not divisible by trows_b")?;
    // Threadgroup mem: As[BM*(BK+8)] + Bs[BK*(BN+8)] bf16.
    let bytes = (c.bm * (c.bk + 8) + c.bk * (c.bn + 8)) * 2;
    chk(bytes <= 32768, "threadgroup mem exceeds 32 KiB")?;
    Ok(())
}

/// Compile + benchmark the kiln steel GEMM at `[M,K]@[K,N]`; returns GFLOP/s.
#[doc(hidden)]
pub fn bench_steel_cfg(c: &GemmCfg, m: usize, k: usize, n: usize, iters: usize) -> Result<f64> {
    steel_cfg_valid(c)?;
    if k % c.bk != 0 {
        return Err(Error::Msg(format!("{}: K={k} not mult of BK", c.label())));
    }
    let companion = crate::primary_metal_companion(0)?;
    let device = companion.device();
    let msl = gen_steel_msl(c);
    let opts = fast_compile_options();
    let lib = device
        .new_library_with_source(&msl, Some(&opts))
        .map_err(|e| Error::Msg(format!("steel compile {}: {e:?}", c.label())))?;
    let func = lib
        .get_function("kiln_gemm", None)
        .map_err(|e| Error::Msg(format!("steel get_function {}: {e:?}", c.label())))?;
    let pipeline: ComputePipeline = device
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| Error::Msg(format!("steel pipeline {}: {e:?}", c.label())))?;

    let a = MetalStorage::zeros_kt(device, 0, DType::BF16, m * k)?;
    let b = MetalStorage::zeros_kt(device, 0, DType::BF16, k * n)?;
    let out = MetalStorage::zeros_kt(device, 0, DType::BF16, m * n)?;
    let la = crate::Layout::contiguous(vec![m, k]);
    let lb = crate::Layout::contiguous(vec![k, n]);
    let lc = crate::Layout::contiguous(vec![m, n]);
    let (m_u, n_u, k_u) = (m as u32, n as u32, k as u32);
    let (a_bs, b_bs, c_bs) = ((m * k) as u32, (k * n) as u32, (m * n) as u32);
    let grid = objc2_metal::MTLSize {
        width: n.div_ceil(c.bn),
        height: m.div_ceil(c.bm),
        depth: 1,
    };
    let tgsz = objc2_metal::MTLSize {
        width: c.threads(),
        height: 1,
        depth: 1,
    };
    let dispatch = || -> Result<()> {
        let encoder = companion.command_encoder()?;
        encoder.set_compute_pipeline_state(&pipeline);
        let ab = buffer_o_kt(a.buffer().as_ref(), &la, DType::BF16);
        let bb = buffer_o_kt(b.buffer().as_ref(), &lb, DType::BF16);
        let cb = buffer_o_kt(out.buffer().as_ref(), &lc, DType::BF16);
        encoder.set_buffer(0, Some(ab.buffer), ab.offset_in_bytes);
        encoder.set_buffer(1, Some(bb.buffer), bb.offset_in_bytes);
        encoder.set_buffer(2, Some(cb.buffer), cb.offset_in_bytes);
        encoder.set_bytes(3, &m_u);
        encoder.set_bytes(4, &n_u);
        encoder.set_bytes(5, &k_u);
        encoder.set_bytes(6, &a_bs);
        encoder.set_bytes(7, &b_bs);
        encoder.set_bytes(8, &c_bs);
        encoder.dispatch_thread_groups(grid, tgsz);
        drop(encoder);
        Ok(())
    };
    dispatch()?;
    companion.wait_until_completed()?;
    let t = std::time::Instant::now();
    for _ in 0..iters {
        dispatch()?;
    }
    companion.wait_until_completed()?;
    let secs = t.elapsed().as_secs_f64() / iters as f64;
    let gflop = 2.0 * m as f64 * k as f64 * n as f64 / 1e9;
    Ok(gflop / secs)
}

// `bench_mlx_reference` removed (#1082 final step): it was the only
// remaining consumer of candle's `call_mlx_gemm` + the `MetalCompanion`
// `kernels()` cache. The kiln steel-GEMM perf is already validated
// against this reference; the live MLX A/B comparison is expendable now
// that the candle_metal_kernels dependency is being dropped. The rest of
// the sweep harness (`bench_gemm_cfg` / `bench_steel_cfg`, kiln-owned MSL)
// is unchanged.
