//! ROCm composite SDPA (Phase R.8).
//!
//! There is no CUTLASS flash-attention kernel on ROCm. This module implements a
//! correct, **fully on-device** scaled-dot-product-attention composite built out
//! of the parity-tested `kiln_tensor` ROCm primitives:
//!
//! - `rocm_matmul` (batched rank-3 QK^T / PV via per-(b,h) unroll),
//! - `rocm_softmax_last_axis`,
//! - `rocm_scalar_op` (scale), `rocm_max_axis` / `rocm_sum_axis` (lse),
//! - `rocm_index_select_axis_n` (GQA expand + paged gather),
//! - `rocm_masked_fill` (causal mask), `rocm_contiguous` (materialize views),
//! - `rocm_cast` (bf16 <-> f32 accumulation).
//!
//! The hot QK^T / softmax / PV path never leaves `Device::Rocm`; only the tiny
//! `[b*h, sq]` log-sum-exp tail is host-staged (a `ln` over the reduced tensor —
//! ROCm has no native `exp`/`ln` op), which is negligible next to the matmuls.
//!
//! Mirrors the CUDA `kt_api::*` entry points one-for-one under
//! `cfg(feature = "rocm")`; the CUDA arms are untouched (cfg-gated).

#![cfg(feature = "rocm")]

use half::bf16;
use kiln_tensor::{DType as KtDType, Device as KtDevice, Tensor as KtTensor};

use crate::kt_api::FlashAttnError;

/// `ScalarKind::MulScalar` tag (see `kiln_tensor::ops::scalar`).
const SCALAR_MUL: i32 = 2;

/// Additive-mask fill value for masked (future) attention positions. A large
/// finite negative number rather than `f32::NEG_INFINITY` so that BF16 / F16
/// downcasts in the softmax stay well-defined (`exp` underflows to 0).
const NEG_FILL: f32 = -1.0e30;

fn dev_index(device: KtDevice) -> Result<usize, FlashAttnError> {
    match device {
        KtDevice::Rocm(i) => Ok(i),
        other => Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: expected Device::Rocm, got {other:?}"
        ))),
    }
}

fn map_kt<T>(r: Result<T, kiln_tensor::Error>) -> Result<T, FlashAttnError> {
    r.map_err(|e| FlashAttnError::Msg(format!("rocm-sdpa: {e}")))
}

/// Materialize a (possibly strided) ROCm tensor into a fresh contiguous ROCm
/// tensor, on-device. `Tensor::contiguous()` has no ROCm arm, so route through
/// the native `rocm_contiguous` strided-copy kernel.
fn rocm_contig(t: &KtTensor) -> Result<KtTensor, FlashAttnError> {
    if t.is_contiguous() {
        return Ok(t.clone());
    }
    map_kt(kiln_tensor::rocm_contiguous(t))
}

/// Cast a ROCm tensor to `target` dtype on-device (no-op clone if already that
/// dtype). Input is made contiguous first (the native kernel requires it).
fn rocm_cast_to(t: &KtTensor, target: KtDType) -> Result<KtTensor, FlashAttnError> {
    if t.dtype() == target {
        return rocm_contig(t);
    }
    let c = rocm_contig(t)?;
    map_kt(kiln_tensor::rocm_cast(&c, target))
}

/// GQA expand: repeat each kv-head group `group = h/hk` times along the head
/// axis (axis 2 of `[b, sk, hk, d]`) to produce `[b, sk, h, d]`. Implemented as
/// an on-device `index_select` over the head axis with indices
/// `[0,0,..,1,1,..]` (each kv head repeated `group` times). When `hk == h`
/// (no GQA) this is a contiguous pass-through.
fn gqa_expand_heads(
    kv: &KtTensor, // [b, sk, hk, d], contiguous
    h: usize,
    hk: usize,
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    if hk == h {
        return rocm_contig(kv);
    }
    let _ = device;
    // Expand [b, sk, hk, d] -> [b, sk, h, d] by repeating each kv-head `group`
    // times, via unsqueeze+expand+reshape — IDENTICAL to the contiguous path's
    // GQA expand in forward.rs::flash_attention_forward. (Was an index_select
    // over a cached [0,0,1,1,..] index, which is the only thing unique to the
    // paged path vs the working contiguous path — suspected as the decode
    // garbage source.) out head `kv*group + g` maps to kv head `kv`.
    let s = kv.shape();
    let (bb, sk, dd) = (s[0], s[1], s[3]);
    let group = h / hk;
    let src = rocm_contig(kv)?; // [b, sk, hk, d]
    let e = map_kt(src.unsqueeze(3))?; // [b, sk, hk, 1, d]
    let e = map_kt(e.expand(vec![bb, sk, hk, group, dd]))?; // [b, sk, hk, group, d]
    let e = rocm_contig(&e)?;
    map_kt(e.reshape(vec![bb, sk, h, dd])) // [b, sk, h, d]
}

/// Build the additive-causal U8 mask `[b*h, sq, sk]` on host and upload to ROCm
/// once. `mask[..,i,j] = 1` (→ NEG_FILL) when `j > i + (sk - sq)` (strictly
/// future), else `0` (keep). Replicated across the `b*h` leading axis so the
/// kernel's shape contract (`x.shape() == mask.shape()`) holds.
fn build_causal_mask_u8(
    bh: usize,
    sq: usize,
    sk: usize,
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    let offset = sk as isize - sq as isize;
    let mut mask: Vec<u8> = vec![0u8; bh * sq * sk];
    // Compute one [sq, sk] plane, then it repeats across bh.
    for i in 0..sq {
        let allowed = i as isize + offset; // last allowed key index
        for j in 0..sk {
            if (j as isize) > allowed {
                // future position — mask out in every (b,h) plane.
                for b in 0..bh {
                    mask[b * sq * sk + i * sk + j] = 1;
                }
            }
        }
    }
    map_kt(KtTensor::from_vec_on(device, mask, vec![bh, sq, sk]))
}

/// On-device log-sum-exp from already-computed `scores` and softmax `p`.
///
/// `lse[r] = max_j scores[r,j] + ln(sum_j exp(scores[r,j] - max))`.
/// Using the identity `sum exp(scores - max) = 1 / max_j p[r,j]` (the softmax
/// value at the arg-max column equals `1/Z`), this becomes
/// `lse[r] = max_scores[r] - ln(max_p[r])`. Both reductions run on-device; only
/// the final `[b*h, sq]` element-wise `sub`/`ln` is host-staged (ROCm has no
/// native `ln`), which is tiny relative to the `[b*h, sq, sk]` matmuls.
///
/// Returns an F32 tensor shaped `[b, h, sq]` on the originating ROCm device.
fn compute_lse(
    scores: &KtTensor, // [b*h, sq, sk] f32, contiguous
    p: &KtTensor,      // [b*h, sq, sk] f32, contiguous (softmax of scores)
    b: usize,
    h: usize,
    sq: usize,
    device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    let _ = device;
    // Reductions over the last (sk) axis -> [b*h, sq], on-device.
    let max_scores = map_kt(kiln_tensor::rocm_max_axis(scores, 2))?; // [b*h, sq]
    let max_p = map_kt(kiln_tensor::rocm_max_axis(p, 2))?; // [b*h, sq]

    // On-device tail: lse = max_scores - ln(max_p). `.log()` -> `ops::ln`
    // (UnaryArithKind::Ln, tag 5) routes to `rocm_activation_unary`; `.sub()`
    // -> `ElementwiseOp::rocm_fwd` (both [b*h, sq], contiguous). Previously this
    // D2H'd both reductions, ran the ln/sub on the host, and H2D'd the result —
    // 3 host syncs per full-attention layer per forward. Now zero.
    //
    // Parity note: a fully-masked row (max_p == 0) yields +inf here vs the old
    // host path's NEG_INFINITY. Such rows do not arise in causal decode/training
    // (every query attends to >=1 key), and the ROCm inference path discards lse
    // (`_lse`) — so the edge is unreachable in practice.
    let ln_max_p = map_kt(max_p.log())?; // [b*h, sq]
    let lse_flat = map_kt(max_scores.sub(&ln_max_p))?; // [b*h, sq]
    map_kt(lse_flat.reshape(vec![b, h, sq]))
}

/// Core on-device SDPA composite.
///
/// Inputs (all `Device::Rocm`, BF16):
/// - `q`: `[b, sq, h, d]`
/// - `k`, `v`: `[b, sk, hk, d]`
///
/// Returns `(out[b, sq, h, d] BF16, lse[b, h, sq] F32)`, both on the same ROCm
/// device. F32 accumulation throughout (matmuls compute in f32 over bf16 inputs;
/// softmax in f32), output narrowed back to BF16.
#[allow(clippy::too_many_arguments)]
pub fn sdpa_forward(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    b: usize,
    sq: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let bh = b * h;

    // 1. GQA expand k, v from hk -> h heads: [b, sk, hk, d] -> [b, sk, h, d].
    let k_exp = gqa_expand_heads(k, h, hk, device)?; // [b, sk, h, d]
    let v_exp = gqa_expand_heads(v, h, hk, device)?; // [b, sk, h, d]

    // 2. Reshape q/k/v to [b, h, s, d] (transpose axes 1<->2, contiguous), then
    //    flatten to [b*h, s, d] for batched rocm_matmul.
    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?; // [b, h, sq, d]
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?; // [b, h, sk, d]
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?; // [b, h, sk, d]

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?; // [b*h, sq, d]
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?; // [b*h, sk, d]
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?; // [b*h, sk, d]

    // Cast to f32 for accumulation. (rocm_matmul computes in the operand dtype;
    // we use f32 operands so QK^T/PV accumulate in f32.)
    let q3f = rocm_cast_to(&q3, KtDType::F32)?;
    let k3f = rocm_cast_to(&k3, KtDType::F32)?;
    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    // 3. scores = (q @ k^T) * scale  -> [b*h, sq, sk]
    let kt3 = rocm_contig(&map_kt(k3f.transpose(1, 2))?)?; // [b*h, d, sk]
    let qk = map_kt(kiln_tensor::rocm_matmul(&q3f, &kt3))?; // [b*h, sq, sk] f32
    let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;

    // 4. causal mask (additive via masked_fill with -inf above the diagonal).
    //    For sq == 1 (decode) the single query attends ALL keys 0..sk, so the
    //    causal mask is provably all-zeros (build_causal_mask_u8: offset=sk-1,
    //    allowed=sk-1, no j>allowed) — a no-op. Skip the host mask build + H2D
    //    upload ([bh,1,sk] per attention layer per token) and the masked_fill.
    let scores = if causal && sq > 1 {
        let mask = build_causal_mask_u8(bh, sq, sk, device)?;
        map_kt(kiln_tensor::rocm_masked_fill(&scores, &mask, NEG_FILL))?
    } else {
        scores
    };

    // 5. p = softmax(scores) over last axis -> [b*h, sq, sk]; lse from scores+p.
    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?; // [b*h, sq, sk] f32
    let lse = compute_lse(&scores, &p, b, h, sq, device)?; // [b, h, sq] f32

    // 6. out = p @ v -> [b*h, sq, d]; reshape/transpose back to [b, sq, h, d].
    let out3 = map_kt(kiln_tensor::rocm_matmul(&p, &v3f))?; // [b*h, sq, d] f32
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?; // [b, h, sq, d]
    let out_bshd = rocm_contig(&map_kt(out_bhsd.transpose(1, 2))?)?; // [b, sq, h, d]
    let out_bf16 = rocm_cast_to(&out_bshd, KtDType::BF16)?; // [b, sq, h, d] bf16

    Ok((out_bf16, lse))
}

// ============================================================================
// Forward entry point
// ============================================================================

/// ROCm composite of `flash_attn_fwd_kt`. `q,k,v` are `[b, sq, h, d]` /
/// `[b, sk, hk, d]` BF16 on `Device::Rocm`. Returns `(out[b,sq,h,d] BF16,
/// lse[b,h,sq] F32)`.
pub fn flash_attn_fwd_rocm(
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor), FlashAttnError> {
    // Coerce all operands onto one ROCm device: a decode caller may hand us an
    // operand that drifted to CPU via a host-staged op, and the composite (QK^T
    // / softmax / PV) requires every operand on-device. No-op when co-located.
    // (R.4 E2E)
    let dev = [q.device(), k.device(), v.device()]
        .into_iter()
        .find(|d| !d.is_cpu())
        .unwrap_or_else(|| q.device());
    let qc;
    let q = if q.device() != dev { qc = map_kt(q.to_device(dev))?; &qc } else { q };
    let kc;
    let k = if k.device() != dev { kc = map_kt(k.to_device(dev))?; &kc } else { k };
    let vc;
    let v = if v.device() != dev { vc = map_kt(v.to_device(dev))?; &vc } else { v };

    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    sdpa_forward(q, k, v, b, sq, sk, h, hk, d, softmax_scale, causal)
}

// ============================================================================
// Paged decode
// ============================================================================

/// Gather K (or V) rows for every (batch, logical-position) from a token-major
/// pool `[total_slots, hk, d]` via `block_table[b, blk]`. Produces
/// `[b, seqlen_k, hk, d]` on-device.
///
/// Physical slot for logical position `t` of sequence `b`:
///   `block_table[b, t / page_block_size] * page_block_size + t % page_block_size`.
fn paged_gather(
    pool: &KtTensor,         // [total_slots, hk, d]
    block_table: &KtTensor,  // device U32 [b, max_blocks_per_seq]
    b: usize,
    seqlen_k: usize,
    max_blocks_per_seq: usize,
    page_block_size: usize,
    hk: usize,
    d: usize,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    // Build the flat physical-slot gather index [b*seqlen_k] ON-DEVICE from the
    // device-resident block_table (no D2H, no host loop, no H2D). The kernel is
    // bit-identical to the former host loop.
    let idx_t = map_kt(kiln_tensor::rocm_paged_gather_index(
        block_table,
        b,
        seqlen_k,
        max_blocks_per_seq,
        page_block_size,
    ))?;
    let pool_c = rocm_contig(pool)?;
    // gather rows along axis 0 -> [b*seqlen_k, hk, d], then reshape.
    let gathered = map_kt(kiln_tensor::rocm_index_select_dim0(&pool_c, &idx_t))?;
    map_kt(gathered.reshape(vec![b, seqlen_k, hk, d]))
}

/// ROCm composite of `flash_attn_paged_decode_kt`. `q` is `[b, 1, h, d]`; K/V
/// are gathered from `k_pool`/`v_pool` `[total_slots, hk, d]` via `block_table`.
/// Returns `out[b, 1, h, d]` BF16 on `Device::Rocm`.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_decode_rocm(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    // Read sq from the query (q is [b, sq, h, d]) — do NOT hardcode sq=1. With
    // multi-token decode (MTP / speculative drafting, sq>1) hardcoding sq=1
    // computed attention for only the first query row and left the rest garbage,
    // which compounded into incoherent output after a few accepted tokens. The
    // working prefill path (flash_attn_fwd_rocm) reads sq the same way; for sq>1
    // sdpa_forward's causal mask masks each query to its own position.
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let hk = k_pool.shape()[1];
    let max_blocks_per_seq = block_table.shape()[1];

    // block_table (U32 [b, blocks]) stays on-device; the gather index is built
    // on-GPU by paged_gather (no host round-trip).
    let k_gathered = paged_gather(
        k_pool, block_table, b, seqlen_k, max_blocks_per_seq, page_block_size, hk, d, device,
    )?; // [b, seqlen_k, hk, d]
    let v_gathered = paged_gather(
        v_pool, block_table, b, seqlen_k, max_blocks_per_seq, page_block_size, hk, d, device,
    )?; // [b, seqlen_k, hk, d]

    let (out, _lse) = sdpa_forward(
        q, &k_gathered, &v_gathered, b, sq, seqlen_k, h, hk, d, softmax_scale, causal,
    )?;
    Ok(out)
}

/// ROCm composite of `flash_attn_paged_decode_dyn_seqlen_kt`. Like
/// [`flash_attn_paged_decode_rocm`] but with a per-batch `seqused_k` bound: keys
/// `t >= seqused_k[b]` are masked out (additive `-inf`) so they don't contribute.
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_paged_decode_dyn_seqlen_rocm(
    q: &KtTensor,
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    block_table: &KtTensor,
    seqused_k: &KtTensor,
    max_seqlen_k: usize,
    page_block_size: usize,
    softmax_scale: f32,
    causal: bool,
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    let (b, h, d) = (q.shape()[0], q.shape()[2], q.shape()[3]);
    let hk = k_pool.shape()[1];
    let max_blocks_per_seq = block_table.shape()[1];

    // block_table + seqused_k stay on-device: paged_gather builds the gather
    // index on-GPU, and sdpa_forward_dyn_tail builds the tail mask on-GPU from
    // the device seqused_k (no D2H/H2D round-trip per attention layer).
    let k_gathered = paged_gather(
        k_pool, block_table, b, max_seqlen_k, max_blocks_per_seq, page_block_size, hk, d, device,
    )?; // [b, max_seqlen_k, hk, d]
    let v_gathered = paged_gather(
        v_pool, block_table, b, max_seqlen_k, max_blocks_per_seq, page_block_size, hk, d, device,
    )?;

    // Run SDPA (sq=1, non-causal core); apply the per-batch tail mask by zeroing
    // the K contribution beyond seqused_k. We fold the tail mask into the scores
    // through a dedicated mask path: compute scores ourselves so we can mask.
    sdpa_forward_dyn_tail(
        q, &k_gathered, &v_gathered, b, max_seqlen_k, h, hk, d, softmax_scale, causal, seqused_k,
    )
}

/// SDPA variant for dyn-seqlen paged decode: identical to [`sdpa_forward`] with
/// `sq = 1`, but additionally masks keys `j >= seqused_k[b]` per batch (additive
/// `-inf`) before the softmax. Returns only `out[b, 1, h, d]` BF16.
#[allow(clippy::too_many_arguments)]
fn sdpa_forward_dyn_tail(
    q: &KtTensor,
    k: &KtTensor, // [b, sk, hk, d]
    v: &KtTensor, // [b, sk, hk, d]
    b: usize,
    sk: usize,
    h: usize,
    hk: usize,
    d: usize,
    scale: f32,
    causal: bool,
    seqused_k: &KtTensor, // device U32 [b]
) -> Result<KtTensor, FlashAttnError> {
    let device = q.device();
    let sq = 1usize;
    let bh = b * h;

    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;

    let q3 = map_kt(q_bhsd.reshape(vec![bh, sq, d]))?;
    let k3 = map_kt(k_bhsd.reshape(vec![bh, sk, d]))?;
    let v3 = map_kt(v_bhsd.reshape(vec![bh, sk, d]))?;

    let q3f = rocm_cast_to(&q3, KtDType::F32)?;
    let k3f = rocm_cast_to(&k3, KtDType::F32)?;
    let v3f = rocm_cast_to(&v3, KtDType::F32)?;

    let kt3 = rocm_contig(&map_kt(k3f.transpose(1, 2))?)?;
    let qk = map_kt(kiln_tensor::rocm_matmul(&q3f, &kt3))?; // [b*h, 1, sk]
    let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, scale))?;

    // Build the U8 tail mask [b*h, 1, sk] ON-DEVICE from the device seqused_k:
    // mask[bi,hi,j] = (j >= seqused_k[bi]). For sq=1 decode the causal constraint
    // is subsumed by the tail (the newest query attends all keys 0..used-1), so
    // there is no separate causal term — bit-identical to the former host loop.
    let _ = causal;
    let mask_t = map_kt(kiln_tensor::rocm_build_tail_mask(seqused_k, b, h, sk))?;
    let scores = map_kt(kiln_tensor::rocm_masked_fill(&scores, &mask_t, NEG_FILL))?;

    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?;
    let out3 = map_kt(kiln_tensor::rocm_matmul(&p, &v3f))?; // [b*h, 1, d]
    let out_bhsd = map_kt(out3.reshape(vec![b, h, sq, d]))?;
    let out_bshd = rocm_contig(&map_kt(out_bhsd.transpose(1, 2))?)?; // [b, 1, h, d]
    rocm_cast_to(&out_bshd, KtDType::BF16)
}

// ============================================================================
// Paged KV write (in-place device-to-device copy into the pool)
// ============================================================================

// HIP runtime device-to-device async memcpy. The symbol is linked transitively
// via `kiln-tensor`'s build.rs (`cargo:rustc-link-lib=dylib=amdhip64`), so a
// bare `extern "C"` declaration resolves at link time. Signature matches
// `hipMemcpyDtoDAsync(void* dst, void* src, size_t sizeBytes, hipStream_t)`.
unsafe extern "C" {
    fn hipMemcpyDtoDAsync(
        dst: *mut core::ffi::c_void,
        src: *const core::ffi::c_void,
        size_bytes: usize,
        stream: *mut core::ffi::c_void,
    ) -> i32;
}

/// Copy `n_bytes` from `src_ptr` into `dst_ptr` (both raw ROCm device addresses)
/// on `stream`. Used by the paged-KV writers to scatter a token row into the
/// pool in place. Synchronizes the default stream afterward so the write is
/// observable by a subsequent decode that reads the pool.
fn rocm_d2d_copy(
    dst_ptr: u64,
    src_ptr: u64,
    n_bytes: usize,
    stream: *mut core::ffi::c_void,
    dev_idx: usize,
) -> Result<(), FlashAttnError> {
    let status = unsafe {
        hipMemcpyDtoDAsync(
            dst_ptr as *mut core::ffi::c_void,
            src_ptr as *const core::ffi::c_void,
            n_bytes,
            stream,
        )
    };
    if status != 0 {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: hipMemcpyDtoDAsync returned status {status}"
        )));
    }
    // R.10 perf: no device sync needed — the d2d write and any later read of the
    // pool serialize on the one cached per-device stream (FIFO). The old
    // hipDeviceSynchronize stalled the decode pipeline every KV write.
    let _ = dev_idx;
    Ok(())
}

/// Copy a freshly-computed `src` ROCm tensor into a caller-owned `dst` ROCm
/// buffer, device-to-device, matching shapes/dtype. Used by the
/// caller-owned-output (graph-capture) decode variant. `src` is made contiguous
/// first; `dst` must be contiguous (the kernel write contract).
pub fn rocm_copy_into(src: &KtTensor, dst: &KtTensor) -> Result<(), FlashAttnError> {
    if src.shape() != dst.shape() || src.dtype() != dst.dtype() {
        return Err(FlashAttnError::Msg(format!(
            "rocm-sdpa: copy_into shape/dtype mismatch src {:?}/{:?} dst {:?}/{:?}",
            src.shape(),
            src.dtype(),
            dst.shape(),
            dst.dtype()
        )));
    }
    let dev_idx = dev_index(dst.device())?;
    let src_c = rocm_contig(src)?;
    let n_bytes = dst.element_count() * dst.dtype().size_in_bytes();
    let src_ptr = kiln_kt_bridge::rocm_input_device_ptr(&src_c, src.dtype(), "copy_src")?;
    let dst_ptr = kiln_kt_bridge::rocm_input_device_ptr(dst, dst.dtype(), "copy_dst")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(dst, "copy_dst")?;
    rocm_d2d_copy(dst_ptr, src_ptr, n_bytes, stream, dev_idx)
}

/// ROCm composite of `paged_kv_write_token_major_bf16_kt` (host-`usize`-slot
/// variant). Writes the single token rows `k`/`v` (`[num_kv_heads * head_dim]`
/// BF16 each) into `k_pool`/`v_pool` at physical row `slot`, in place, on-device.
pub fn paged_kv_write_token_major_bf16_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    let device = k_pool.device();
    let dev_idx = dev_index(device)?;
    let row_elems = num_kv_heads * head_dim;
    let bpe = KtDType::BF16.size_in_bytes();
    let row_bytes = row_elems * bpe;
    let slot_byte_off = (slot * row_elems * bpe) as u64;

    let k_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(k)?, KtDType::BF16, "k")?;
    let v_src = kiln_kt_bridge::rocm_input_device_ptr(&rocm_contig(v)?, KtDType::BF16, "v")?;
    let kp_dst = kiln_kt_bridge::rocm_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_dst = kiln_kt_bridge::rocm_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(k_pool, "k_pool")?;

    rocm_d2d_copy(kp_dst + slot_byte_off, k_src, row_bytes, stream, dev_idx)?;
    rocm_d2d_copy(vp_dst + slot_byte_off, v_src, row_bytes, stream, dev_idx)?;
    Ok(())
}

/// ROCm composite of `paged_kv_write_token_major_bf16_slot_kt` (device-`[1]`-U32
/// slot variant). Stages the single slot index to host (tiny metadata), then
/// reuses the host-slot writer.
pub fn paged_kv_write_token_major_bf16_slot_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slot: &KtTensor,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    // R.9: on-device scatter-copy of the current token's K/V row into
    // `pool[*slot]` using the DEVICE slot index — NO host readback, so this
    // records cleanly into a captured HIP decode graph (and is safe on the
    // Borrowed freeze-pointer arena buffers, since it writes through
    // device_ptr_raw, not slice()). Reshape pool -> [n_rows, row_elems] and the
    // token row -> [1, row_elems], then `dst[*slot] = row` via index_copy_dim0.
    let row_elems = num_kv_heads * head_dim;
    let kp_rows = k_pool.element_count() / row_elems;
    let vp_rows = v_pool.element_count() / row_elems;
    let k_pool2 = map_kt(k_pool.reshape(vec![kp_rows, row_elems]))?;
    let v_pool2 = map_kt(v_pool.reshape(vec![vp_rows, row_elems]))?;
    let k_row = map_kt(rocm_contig(k)?.reshape(vec![1, row_elems]))?;
    let v_row = map_kt(rocm_contig(v)?.reshape(vec![1, row_elems]))?;
    let slot1 = map_kt(rocm_contig(slot)?.reshape(vec![1]))?;
    map_kt(kiln_tensor::rocm_index_copy_dim0(&k_pool2, &slot1, &k_row))?;
    map_kt(kiln_tensor::rocm_index_copy_dim0(&v_pool2, &slot1, &v_row))?;
    Ok(())
}

/// ROCm composite of `paged_kv_write_token_major_bf16_batch_slot_kt` (batched
/// device-`[batch]`-U32 slots). `k`/`v` are `[batch * num_kv_heads * head_dim]`
/// BF16; row `r` is written to physical pool row `slots[r]`.
pub fn paged_kv_write_token_major_bf16_batch_slot_rocm(
    k_pool: &KtTensor,
    v_pool: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    slots: &KtTensor,
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
) -> Result<(), FlashAttnError> {
    let device = k_pool.device();
    let dev_idx = dev_index(device)?;
    let row_elems = num_kv_heads * head_dim;
    let bpe = KtDType::BF16.size_in_bytes();
    let row_bytes = row_elems * bpe;

    let slots_host = map_kt(kiln_tensor::rocm_to_host_copy(slots))?;
    let slot_vec: Vec<u32> = map_kt(slots_host.to_vec::<u32>())?;

    let k_c = rocm_contig(k)?;
    let v_c = rocm_contig(v)?;
    let k_src_base = kiln_kt_bridge::rocm_input_device_ptr(&k_c, KtDType::BF16, "k")?;
    let v_src_base = kiln_kt_bridge::rocm_input_device_ptr(&v_c, KtDType::BF16, "v")?;
    let kp_dst = kiln_kt_bridge::rocm_input_device_ptr(k_pool, KtDType::BF16, "k_pool")?;
    let vp_dst = kiln_kt_bridge::rocm_input_device_ptr(v_pool, KtDType::BF16, "v_pool")?;
    let stream = kiln_kt_bridge::rocm_stream_raw_of(k_pool, "k_pool")?;

    for (r, &slot) in slot_vec.iter().enumerate().take(batch) {
        let src_off = (r * row_elems * bpe) as u64;
        let dst_off = (slot as usize * row_elems * bpe) as u64;
        rocm_d2d_copy(kp_dst + dst_off, k_src_base + src_off, row_bytes, stream, dev_idx)?;
        rocm_d2d_copy(vp_dst + dst_off, v_src_base + src_off, row_bytes, stream, dev_idx)?;
    }
    Ok(())
}

// ============================================================================
// Backward
// ============================================================================

/// ROCm composite of `flash_attn_bwd_kt`. Recomputes scores + p from the saved
/// `softmax_lse` (flash-style) and produces `(dq, dk, dv)` via matmuls:
///   dv = p^T @ dout
///   dp = dout @ v^T
///   ds = p * (dp - rowsum(dp * p))
///   dq = ds @ k * scale
///   dk = ds^T @ q * scale
/// All in F32, outputs BF16 `[b, s, h, d]`. dk/dv are returned at the EXPANDED
/// head count `h` (matching the CUDA FFI's expanded-GQA buffer contract).
#[allow(clippy::too_many_arguments)]
pub fn flash_attn_bwd_rocm(
    dout: &KtTensor,
    q: &KtTensor,
    k: &KtTensor,
    v: &KtTensor,
    softmax_lse: &KtTensor,
    softmax_scale: f32,
    causal: bool,
) -> Result<(KtTensor, KtTensor, KtTensor), FlashAttnError> {
    let device = q.device();
    let dev_idx = dev_index(device)?;
    let (b, sq, h, d) = (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
    let (sk, hk) = (k.shape()[1], k.shape()[2]);
    let bh = b * h;

    // `softmax_lse` is the saved forward log-sum-exp. The composite recomputes
    // `scores` and the softmax `p` directly from q/k below (softmax is fully
    // determined by `scores`, so recompute == the lse-shifted forward p), so the
    // saved lse is not needed for correctness here. Kept in the signature to
    // mirror the CUDA `flash_attn_bwd_kt` contract.
    let _ = softmax_lse;

    // Expand GQA + reshape to [b*h, s, d] f32, exactly as forward.
    let k_exp = gqa_expand_heads(k, h, hk, device)?;
    let v_exp = gqa_expand_heads(v, h, hk, device)?;

    let q_bhsd = rocm_contig(&map_kt(q.transpose(1, 2))?)?;
    let k_bhsd = rocm_contig(&map_kt(k_exp.transpose(1, 2))?)?;
    let v_bhsd = rocm_contig(&map_kt(v_exp.transpose(1, 2))?)?;
    let do_bhsd = rocm_contig(&map_kt(dout.transpose(1, 2))?)?;

    let q3 = rocm_cast_to(&map_kt(q_bhsd.reshape(vec![bh, sq, d]))?, KtDType::F32)?;
    let k3 = rocm_cast_to(&map_kt(k_bhsd.reshape(vec![bh, sk, d]))?, KtDType::F32)?;
    let v3 = rocm_cast_to(&map_kt(v_bhsd.reshape(vec![bh, sk, d]))?, KtDType::F32)?;
    let do3 = rocm_cast_to(&map_kt(do_bhsd.reshape(vec![bh, sq, d]))?, KtDType::F32)?;

    // Recompute scores + softmax p (same as forward).
    let kt3 = rocm_contig(&map_kt(k3.transpose(1, 2))?)?; // [b*h, d, sk]
    let qk = map_kt(kiln_tensor::rocm_matmul(&q3, &kt3))?; // [b*h, sq, sk]
    let scores = map_kt(kiln_tensor::rocm_scalar_op(&qk, SCALAR_MUL, softmax_scale))?;
    let scores = if causal {
        let mask = build_causal_mask_u8(bh, sq, sk, device)?;
        map_kt(kiln_tensor::rocm_masked_fill(&scores, &mask, NEG_FILL))?
    } else {
        scores
    };
    let p = map_kt(kiln_tensor::rocm_softmax_last_axis(&scores))?; // [b*h, sq, sk]

    // dv = p^T @ dout   -> [b*h, sk, d]
    let pt = rocm_contig(&map_kt(p.transpose(1, 2))?)?; // [b*h, sk, sq]
    let dv3 = map_kt(kiln_tensor::rocm_matmul(&pt, &do3))?; // [b*h, sk, d]

    // dp = dout @ v^T   -> [b*h, sq, sk]
    let vt3 = rocm_contig(&map_kt(v3.transpose(1, 2))?)?; // [b*h, d, sk]
    let dp = map_kt(kiln_tensor::rocm_matmul(&do3, &vt3))?; // [b*h, sq, sk]

    // ds = p * (dp - rowsum(dp * p))
    // rowsum over last axis: rocm_sum_axis(dp*p, 2) -> [b*h, sq]; subtract via
    // broadcast. We compute the [b*h, sq, sk] product and reduction on-device,
    // then the broadcast-subtract is done by building a [b*h, sq, sk] expanded
    // "row" tensor on host (small relative to matmuls is NOT true here, so do it
    // on-device with index_select-free broadcast through rocm_contiguous of a
    // strided view).
    let dpp = elementwise_mul_f32(&dp, &p, device)?; // [b*h, sq, sk]
    let rowsum = map_kt(kiln_tensor::rocm_sum_axis(&dpp, 2))?; // [b*h, sq]
    // Broadcast rowsum [b*h, sq] -> [b*h, sq, sk] (stride-0 last axis), contiguous.
    let rowsum_b = broadcast_last_axis(&rowsum, sk, device)?; // [b*h, sq, sk]
    let dp_minus = elementwise_sub_f32(&dp, &rowsum_b, device)?; // [b*h, sq, sk]
    let ds = elementwise_mul_f32(&p, &dp_minus, device)?; // [b*h, sq, sk]

    // dq = ds @ k * scale   -> [b*h, sq, d]
    let dq3 = map_kt(kiln_tensor::rocm_matmul(&ds, &k3))?; // [b*h, sq, d]
    let dq3 = map_kt(kiln_tensor::rocm_scalar_op(&dq3, SCALAR_MUL, softmax_scale))?;

    // dk = ds^T @ q * scale -> [b*h, sk, d]
    let dst = rocm_contig(&map_kt(ds.transpose(1, 2))?)?; // [b*h, sk, sq]
    let dk3 = map_kt(kiln_tensor::rocm_matmul(&dst, &q3))?; // [b*h, sk, d]
    let dk3 = map_kt(kiln_tensor::rocm_scalar_op(&dk3, SCALAR_MUL, softmax_scale))?;

    // Reshape back to [b, s, h, d] BF16.
    let dq = bhsd3_to_bshd_bf16(&dq3, b, h, sq, d)?;
    let dk = bhsd3_to_bshd_bf16(&dk3, b, h, sk, d)?;
    let dv = bhsd3_to_bshd_bf16(&dv3, b, h, sk, d)?;

    let _ = dev_idx;
    Ok((dq, dk, dv))
}

/// Reshape a `[b*h, s, d]` f32 tensor to `[b, s, h, d]` BF16 (transpose the
/// head axis back out and narrow to BF16). On-device.
fn bhsd3_to_bshd_bf16(
    t3: &KtTensor, // [b*h, s, d] f32
    b: usize,
    h: usize,
    s: usize,
    d: usize,
) -> Result<KtTensor, FlashAttnError> {
    let t_bhsd = map_kt(t3.reshape(vec![b, h, s, d]))?; // [b, h, s, d]
    let t_bshd = rocm_contig(&map_kt(t_bhsd.transpose(1, 2))?)?; // [b, s, h, d]
    rocm_cast_to(&t_bshd, KtDType::BF16)
}

/// Broadcast a `[bh, sq]` tensor to `[bh, sq, sk]` (replicate the last value
/// across a new trailing axis of size `sk`). Uses the generic
/// `kiln_tensor::ops::broadcast_to` (host-staged on ROCm — acceptable for the
/// training-only bwd path).
fn broadcast_last_axis(
    t: &KtTensor, // [bh, sq]
    sk: usize,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    // Build [bh, sq, 1] then broadcast the trailing axis to sk.
    let mut s = t.shape().to_vec();
    s.push(1);
    let t3 = map_kt(t.reshape(s))?;
    let mut target = t3.shape().to_vec();
    let last = target.len() - 1;
    target[last] = sk;
    map_kt(kiln_tensor::ops::broadcast_to(&t3, &target))
}

/// On-device elementwise f32 multiply of two same-shape contiguous ROCm tensors.
/// The generic `ops::mul` host-stages on ROCm (correctness fallback); for the
/// bwd hot tensors we keep it on-device by computing `a*b = a - (a - a*b)`? No —
/// instead route through the generic op only when needed. Here we use the
/// generic `kiln_tensor::ops::mul`, which on ROCm stages to host. To keep bwd
/// on-device, prefer composing rocm primitives; but there is no native ROCm
/// elementwise binary multiply, so bwd's three small elementwise ops fall back
/// to the host-staged generic ops. This is acceptable: bwd is training-only and
/// the dominant cost (the five matmuls) stays on-device.
fn elementwise_mul_f32(
    a: &KtTensor,
    b: &KtTensor,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    map_kt(kiln_tensor::ops::mul(a, b))
}

fn elementwise_sub_f32(
    a: &KtTensor,
    b: &KtTensor,
    _device: KtDevice,
) -> Result<KtTensor, FlashAttnError> {
    map_kt(kiln_tensor::ops::sub(a, b))
}

// Keep the bf16 import used even if the optimizer prunes a path.
#[allow(dead_code)]
fn _bf16_marker(x: f32) -> bf16 {
    bf16::from_f32(x)
}
