//! Kiln-owned matrix-core Metal GEMM (#1082) — `MatmulOp::metal_fwd`.
//!
//! Replaces the host round-trip that `dispatch2` would otherwise take for
//! matmul on Metal (matmul has no native Metal kernel → it ran the GEMM on
//! the CPU). This is the compute-bound path: prefill QKV/O, MLP gate/up/down
//! at seq>1, the LM head, and any bs>1 decode. It uses Apple's matrix units
//! via `simdgroup_float8x8` (NOT candle's `call_mlx_gemm`) — kiln owns the
//! MSL and dispatches it through objc2-metal via the `MetalCompanion`.
//!
//! Design (judge-panel synthesized): `C[M,N] = A[M,K] @ B[K,N]`, BF16 in/out,
//! **F32 accumulation** (the matrix-unit FMA), with the weight pre-transposed
//! to `B=[K,N]` (N-contiguous) so both operands stage row-major-coalesced. A
//! 64×64 output tile per threadgroup (4 simdgroups, each owning a 32×32
//! subtile = 4×4 array of 8×8 F32 accumulators), K-tiled by 8 through
//! threadgroup memory. Inputs are staged to threadgroup **F32** (the
//! cooperative load casts BF16→F32) so the matrix path is unambiguously
//! F32×F32→F32 — there is no `simdgroup_matrix<bfloat>` MMA overload; you
//! load into `simdgroup_float8x8` and the load up-converts.
//!
//! M=1 decode (memory-bound GEMV) is left to the model's
//! `metal_transposed_coop_gemv` fast path; this GEMM handles M=1 correctly
//! too (it just pads to an 8-row tile) but the matrix units idle 7/8 there,
//! so the decode projection path does not route here.

use std::sync::Arc;

use crate::metal_types::{buffer_o_kt, ComputePipeline};
use crate::{DType, Error, MetalStorage, Result};

const KILN_GEMM_MSL: &str = r#"
#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

constant constexpr uint BM = 64;
constant constexpr uint BN = 64;
constant constexpr uint BK = 8;
constant constexpr uint WM = 2;     // simdgroups down M
constant constexpr uint WN = 2;     // simdgroups across N
constant constexpr uint TM = BM / (8 * WM);  // 4 row-fragments per simd
constant constexpr uint TN = BN / (8 * WN);  // 4 col-fragments per simd

// C[M,N] = A[M,K] @ B[K,N], bf16 in/out, f32 accumulate.
// One threadgroup (128 threads = 4 simdgroups) computes a 64x64 C tile.
kernel void kiln_gemm_bf16(
    device const bfloat* A    [[buffer(0)]],   // [batch,M,K] row-major (+offset)
    device const bfloat* B    [[buffer(1)]],   // [batch,K,N] row-major (+offset)
    device bfloat* C          [[buffer(2)]],   // [batch,M,N] row-major
    constant uint& M          [[buffer(3)]],
    constant uint& N          [[buffer(4)]],
    constant uint& K          [[buffer(5)]],
    constant uint& a_bs       [[buffer(6)]],   // A per-batch element stride
    constant uint& b_bs       [[buffer(7)]],   // B per-batch element stride
    constant uint& c_bs       [[buffer(8)]],   // C per-batch element stride
    uint3 tg   [[threadgroup_position_in_grid]],   // x=n-tile, y=m-tile, z=batch
    uint  sgid [[simdgroup_index_in_threadgroup]], // 0..3
    uint  lane [[thread_index_in_simdgroup]])      // 0..31
{
    threadgroup float As[BM * BK];   // [BM][BK] ld=BK
    threadgroup float Bs[BK * BN];   // [BK][BN] ld=BN
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
        // Cooperative, bounds-checked stage of the A and B K-tiles (BF16->F32).
        for (uint idx = tid; idx < BM * BK; idx += 128) {
            uint r = idx / BK;
            uint c = idx % BK;
            uint ar = m0 + r;
            uint ac = k0 + c;
            As[idx] = (ar < M && ac < K) ? float(Ab[ar * K + ac]) : 0.0f;
        }
        for (uint idx = tid; idx < BK * BN; idx += 128) {
            uint r = idx / BN;
            uint c = idx % BN;
            uint br = k0 + r;
            uint bc = n0 + c;
            Bs[idx] = (br < K && bc < N) ? float(Bb[br * N + bc]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        simdgroup_float8x8 af[TM];
        simdgroup_float8x8 bf[TN];
        for (uint i = 0; i < TM; i++) {
            simdgroup_load(af[i], As + (subm + i * 8) * BK, BK);
        }
        for (uint j = 0; j < TN; j++) {
            simdgroup_load(bf[j], Bs + (subn + j * 8), BN);
        }
        for (uint i = 0; i < TM; i++) {
            for (uint j = 0; j < TN; j++) {
                simdgroup_multiply_accumulate(acc[i * TN + j], af[i], bf[j], acc[i * TN + j]);
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

const TILE: usize = 64; // BM == BN
const TG_THREADS: usize = 128;

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
        .map_err(|e| Error::Msg(format!("metal_matmul: compile kiln_gemm_bf16 library: {e:?}")))?;
    let func = lib
        .get_function("kiln_gemm_bf16", None)
        .map_err(|e| Error::Msg(format!("metal_matmul: load kiln_gemm_bf16 function: {e:?}")))?;
    let pipeline = companion
        .device()
        .new_compute_pipeline_state_with_function(&func)
        .map_err(|e| Error::Msg(format!("metal_matmul: build kiln_gemm_bf16 pipeline: {e:?}")))?;
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

    let pipeline = gemm_pipeline(a_metal)?;
    let out_metal = kt_metal(&out, "out")?;
    let encoder = companion.command_encoder()?;
    encoder.set_label("kiln_gemm_bf16");
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
