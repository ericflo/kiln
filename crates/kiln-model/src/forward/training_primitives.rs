use super::*;

pub struct GdnRecurrentBackwardGrads {
    pub dq: Tensor,
    pub dk: Tensor,
    pub dv: Tensor,
    pub dbeta: Tensor,
    pub dg: Tensor,
    pub d_state: Option<Tensor>,
}

/// Activation gradients of [`gated_rms_norm`] (the GDN Step-8
/// `norm(x) * silu(z)` block) when the normalization weight is frozen.
///
/// `dx` and `dz` carry their corresponding input dtype. No weight gradient is
/// represented, allocated, or reduced.
pub struct GdnGatedRmsNormFrozenWeightBackwardGrads {
    pub dx: Tensor,
    pub dz: Tensor,
}

/// Candle-composite analytic backward for [`gated_rms_norm_fallback`] with a
/// frozen normalization weight (`out = rms_norm(x, weight, eps) * silu(z)`).
/// Device-agnostic (runs in candle F32; works on CUDA without a host
/// round-trip), so the kt-tape `crate::tape_forward::GdnGatedRmsNormBackward`
/// op can wrap it the same way `crate::tape_forward::GdnRecurrentBackward`
/// wraps [`gdn_recurrent_backward_no_grad`].
///
/// # Math (per trailing-axis row, `D` = trailing-axis size)
///
/// ```text
/// r       = sqrt(mean_j(x_j^2) + eps)
/// normed  = x * w / r
/// gate    = silu(z) = z * sigmoid(z)
/// out     = normed * gate
///
/// d_normed = dout * gate
/// dz       = dout * normed * silu'(z),  silu'(z) = sig(z) * (1 + z * (1 - sig(z)))
/// S        = Σⱼ d_normed_j * x_j * w_j
/// dx_k     = (d_normed_k * w_k) / r  -  x_k * S / (D * r^3)
/// ```
///
/// The normalization weight is saved only because `dx` depends on it. This
/// route deliberately omits the `dw` reduction and allocation. Outputs are
/// cast back to `x.dtype()` and `z.dtype()` so they match the tape inputs.
pub fn gdn_gated_rms_norm_frozen_weight_backward_no_grad(
    x: &Tensor,
    z: &Tensor,
    weight: &Tensor,
    eps: f64,
    grad_out: &Tensor,
) -> Result<GdnGatedRmsNormFrozenWeightBackwardGrads> {
    let x_dtype = x.dtype();
    let z_dtype = z.dtype();

    let x_f32 = x.to_dtype(DType::F32)?;
    let z_f32 = z.to_dtype(DType::F32)?;
    let w_f32 = weight.to_dtype(DType::F32)?;
    let dout = grad_out.to_dtype(DType::F32)?;

    // r = sqrt(mean(x^2) + eps); inv_r = 1/r (broadcast over the trailing axis).
    let variance = x_f32.sqr()?.mean_keepdim(LAST_DIM)?;
    let inv_r = (variance + eps)?.sqrt()?.recip()?; // [..., 1]
    // normed = x * w / r.
    let normed = x_f32.broadcast_mul(&inv_r)?.broadcast_mul(&w_f32)?;

    // gate = silu(z); silu'(z) = sig(z) + z * sig(z) * (1 - sig(z)).
    let sig = cuda_sigmoid(&z_f32)?;
    let gate = (&z_f32 * &sig)?;
    let one_minus_sig = (1.0 - &sig)?;
    let silu_grad = (&sig + (&z_f32 * (&sig * &one_minus_sig)?)?)?;

    // dz = dout * normed * silu'(z).
    let dz = (&dout * (&normed * &silu_grad)?)?;

    // d_normed = dout * gate — the grad flowing into the rms-norm output.
    let d_normed = (&dout * &gate)?;

    let hidden = *x_f32.dims().last().unwrap() as f64;
    // S = Σⱼ d_normed_j * x_j * w_j  (keepdim over the trailing axis).
    let s = (&d_normed * (&x_f32 * w_f32.broadcast_as(x_f32.shape())?)?)?.sum_keepdim(LAST_DIM)?; // [..., 1]
    // dx_k = (d_normed_k * w_k) / r - x_k * S / (D * r^3).
    let inv_r3 = inv_r.powf(3.0)?;
    let term1 = d_normed.broadcast_mul(&w_f32)?.broadcast_mul(&inv_r)?;
    let term2 = x_f32
        .broadcast_mul(&s.broadcast_mul(&inv_r3)?)?
        .affine(1.0 / hidden, 0.0)?;
    let dx = (&term1 - &term2)?;

    Ok(GdnGatedRmsNormFrozenWeightBackwardGrads {
        dx: dx.to_dtype(x_dtype)?,
        dz: dz.to_dtype(z_dtype)?,
    })
}

/// kt-native analytic backward for the fused "cross-entropy from full
/// logits" tape node (the `crate::tape_forward::CrossEntropyFromLogitsKtBackward` op).
///
/// Computes `dL/d(full logits)` for the next-token-prediction masked
/// cross-entropy loss that `crate::tape_forward::try_tape_cross_entropy_from_logits_kt`
/// (and the candle-authoritative fallback `kiln_train::trainer::cross_entropy_loss`)
/// produces. Device-agnostic (F32, kt in/out — the `_candle` suffix is a misnomer
/// kept for history) so the kt-tape `CrossEntropyFromLogitsKtBackward` op can wrap
/// it exactly the way [`gdn_recurrent_backward_no_grad`] /
/// [`sdpa_fallback_backward_no_grad`] are wrapped. It runs (and is parity-tested)
/// on CPU where candle's own autograd is the oracle — no CUDA needed.
///
/// # Why one fused node
///
/// The forward `cross_entropy_loss` chains FOUR un-taped candle ops
/// (`squeeze(0)` → `narrow(0, 0, T-1)` → `index_select(active)` → `to_dtype(F32)`)
/// before the loss op, so a kt-tape rooted at the loss had a fresh-borrow input
/// island and the chain died one op below the loss (`tape_has_grad=0/50`). This
/// node instead takes the FULL `[1, T, V]` model logits directly and produces
/// the scalar loss, so `dL/d(logits)` reaches the lm_head output once that op is
/// wired (#1082 CP-4 Increment 1).
///
/// # Math (mean reduction over active shifted positions)
///
/// Forward (mirrors `cross_entropy_loss`, lines 5929-5995):
/// ```text
/// lg        = logits.squeeze(0)            [T, V]
/// shift     = lg.narrow(0, 0, T-1)         [T-1, V]   (predict token[i+1] from logit[i])
/// active    = shift.index_select(active_positions, 0)   [A, V]
/// active32  = active.to_dtype(F32)
/// loss      = mean_a( log_sum_exp(active32[a]) - active32[a, label_a] )
/// ```
/// where `active_positions = { i in 0..T-1 : label_mask[i+1] }`, `A = num_active`,
/// and `label_a = input_ids[active_positions[a] + 1]`.
///
/// Backward (`g = dL/dloss`, the seed — typically `1.0`):
/// ```text
/// p_a             = softmax(active32[a])              [A, V]
/// g_active[a]     = (p_a - one_hot(label_a)) * (g / A)   [A, V]   (mean ⇒ the 1/A)
/// grad_shift      = scatter g_active into zeros[T-1, V] at rows active_positions
/// grad_lg         = cat(grad_shift, zeros[1, V], dim=0)  [T, V]   (narrow(0,0,T-1) adjoint)
/// dL/d(logits)    = grad_lg.unsqueeze(0)              [1, T, V]   cast to logits.dtype()
/// ```
/// The trailing zero row is the adjoint of dropping `lg[T-1]` (it never feeds the
/// loss). Rows of `shift` not in `active_positions` get zero gradient (the
/// `index_select` adjoint), which the zeros base already provides.
///
/// The softmax term is scattered back via `index_add` into a `[T-1, V]` zeros
/// tensor along dim 0 (the `index_select` adjoint). The one-hot label term is
/// applied as sparse scalar corrections over flat `[T-1, V]` indices, avoiding
/// an extra dense `[active, vocab]` one-hot allocation.
pub fn cross_entropy_from_logits_grad_candle(
    logits: &Tensor,
    input_ids: &[u32],
    label_mask: &[bool],
    grad_scalar: f64,
) -> Result<Tensor> {
    let logits_dtype = logits.dtype();
    let device = logits.device();
    let seq_len = input_ids.len();

    let dims = logits.dims();
    if dims.len() != 3 || dims[0] != 1 || dims[1] != seq_len {
        anyhow::bail!(
            "cross_entropy_from_logits_grad_candle: logits must be [1, seq_len, vocab], \
             got {dims:?} for seq_len {seq_len}"
        );
    }
    if label_mask.len() != seq_len {
        anyhow::bail!(
            "cross_entropy_from_logits_grad_candle: label_mask length {} != input_ids length {}",
            label_mask.len(),
            seq_len
        );
    }
    anyhow::ensure!(
        seq_len >= 2,
        "cross_entropy_from_logits_grad_candle: requires at least 2 tokens"
    );
    let vocab_size = dims[2];

    // active_positions = { i in 0..T-1 : label_mask[i+1] } (shifted-label mask).
    let active_positions: Vec<u32> = label_mask[1..]
        .iter()
        .enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None })
        .collect();
    anyhow::ensure!(
        !active_positions.is_empty(),
        "cross_entropy_from_logits_grad_candle: no supervised shifted-label positions"
    );
    let num_active = active_positions.len();
    let active_labels: Vec<u32> = active_positions
        .iter()
        .map(|&i| input_ids[i as usize + 1])
        .collect();
    for &label in &active_labels {
        anyhow::ensure!(
            (label as usize) < vocab_size,
            "cross_entropy_from_logits_grad_candle: label {label} >= vocab {vocab_size}"
        );
    }

    // Replicate the forward gather: squeeze(0) -> narrow(0,0,T-1) ->
    // index_select(active) -> to_f32, EXACTLY as `cross_entropy_loss` does.
    let lg = logits.squeeze(0)?; // [T, V]
    let shift = lg.narrow(0, 0, seq_len - 1)?; // [T-1, V]
    let active_indices = Tensor::new(active_positions.as_slice(), device)?;
    let active = shift.index_select(&active_indices, 0)?; // [A, V]
    let active_f32 = active.to_dtype(DType::F32)?;

    // p = softmax(active_f32, last). Numerically-stable max-shift (matches the
    // forward's log_sum_exp; kept inline so this stays a self-contained
    // device-agnostic candle-F32 composite, like the SDPA fallback backward).
    let max_val = active_f32.max_keepdim(LAST_DIM)?; // [A, 1]
    let exp_shifted = active_f32.broadcast_sub(&max_val)?.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(LAST_DIM)?; // [A, 1]
    let p = exp_shifted.broadcast_div(&sum_exp)?; // [A, V]

    // Softmax term: scatter p * (g / A) back to active shifted rows.
    let inv_n = grad_scalar / num_active as f64;
    let g_active_softmax = p.affine(inv_n, 0.0)?; // [A, V]
    let grad_shift_base = Tensor::zeros((seq_len - 1, vocab_size), DType::F32, device)?;
    let grad_shift_softmax = grad_shift_base.index_add(&active_indices, &g_active_softmax, 0)?; // [T-1, V]

    // Sparse one-hot term: subtract g/A at each active (row, label) cell.
    let flat_dim = (seq_len - 1).checked_mul(vocab_size).ok_or_else(|| {
        anyhow::anyhow!("cross_entropy_from_logits_grad_candle: flat grad size overflow")
    })?;
    let mut flat_indices = Vec::with_capacity(num_active);
    for (&row, &label) in active_positions.iter().zip(active_labels.iter()) {
        let flat = (row as usize)
            .checked_mul(vocab_size)
            .and_then(|base| base.checked_add(label as usize))
            .ok_or_else(|| {
                anyhow::anyhow!("cross_entropy_from_logits_grad_candle: flat label index overflow")
            })?;
        flat_indices.push(u32::try_from(flat).with_context(|| {
            format!("cross_entropy_from_logits_grad_candle: flat label index {flat} exceeds u32")
        })?);
    }
    let correction_values =
        Tensor::from_vec_on(device, vec![-(inv_n as f32); num_active], vec![num_active])?;
    let flat_indices_t = Tensor::from_vec_on(device, flat_indices, vec![num_active])?;
    let correction_flat =
        kiln_tensor::ops::scatter_add(&correction_values, 0, &flat_indices_t, flat_dim)?;
    let correction = correction_flat.reshape(vec![seq_len - 1, vocab_size])?;
    let grad_shift = (&grad_shift_softmax + &correction)?;

    // Pad a zero row at the end for the dropped lg[T-1] (the narrow(0,0,T-1)
    // adjoint), -> [T, V].
    let zero_row = Tensor::zeros((1, vocab_size), DType::F32, device)?;
    let grad_lg = Tensor::cat(&[&grad_shift, &zero_row], 0)?; // [T, V]
    let grad_logits = grad_lg.unsqueeze(0)?; // [1, T, V]

    Ok(grad_logits.to_dtype(logits_dtype)?)
}

/// Candle-composite analytic backward for the GDN L2-qk-norm forward
/// `y = l2_normalize(x) * scale` (the Step-4/5 `gdn_qk_norm` op: Q uses
/// `scale = 1/sqrt(dk)`, K uses `scale = 1.0`). Device-agnostic (candle F32),
/// so the kt-tape `crate::tape_forward::GdnL2NormScaleBackward` op can wrap it
/// the same way the gated-rms-norm and recurrence ops wrap their composites.
///
/// # Math (per trailing-axis row, `norm = sqrt(Σⱼ x_j^2 + eps)`)
///
/// Forward (from [`l2_normalize`] + scale): `y_i = scale * x_i / norm`.
/// Same structure as `kiln_autograd::L2NormBackward` (l2norm.rs lines 66-79)
/// with the constant `scale` folded into the upstream grad:
///
/// ```text
/// S    = Σⱼ dy_j * x_j
/// dx_k = scale * ( dy_k / norm  -  x_k * S / norm^3 )
/// ```
///
/// `eps = 1e-6` matches [`l2_normalize`]'s hard-coded epsilon. Output is cast
/// back to `x.dtype()` so it matches the input `Var` layout.
pub fn gdn_l2_norm_scale_backward_no_grad(
    x: &Tensor,
    scale: f64,
    eps: f64,
    grad_out: &Tensor,
) -> Result<Tensor> {
    let x_dtype = x.dtype();
    let x_f32 = x.to_dtype(DType::F32)?;
    let dy = grad_out.to_dtype(DType::F32)?;

    // norm = sqrt(Σⱼ x_j^2 + eps); inv_n = 1/norm (broadcast over trailing axis).
    let sq_sum = x_f32.sqr()?.sum_keepdim(LAST_DIM)?;
    let inv_n = (sq_sum + eps)?.sqrt()?.recip()?; // [..., 1]
    let inv_n3 = inv_n.powf(3.0)?;

    // S = Σⱼ dy_j * x_j.
    let s = (&dy * &x_f32)?.sum_keepdim(LAST_DIM)?; // [..., 1]
    // dx_k = scale * ( dy_k / norm - x_k * S / norm^3 ).
    let term1 = dy.broadcast_mul(&inv_n)?;
    let term2 = x_f32.broadcast_mul(&s.broadcast_mul(&inv_n3)?)?;
    let dx = (&term1 - &term2)?.affine(scale, 0.0)?;

    Ok(dx.to_dtype(x_dtype)?)
}

/// Per-input gradients of the naive scaled-dot-product-attention fallback
/// ([`gqa_attention_core_prefill`]'s non-flash path), returned by
/// [`sdpa_fallback_backward_no_grad`].
///
/// All three grads carry the inputs' dtype. `dq` keeps the query-head count
/// (`[B, nq, T, hd]`); `dk`/`dv` are GQA-collapsed back to the KV-head count
/// (`[B, nkv, T, hd]`) so they match the pre-expand `k`/`v` `Var` layouts.
pub struct SdpaFallbackBackwardGrads {
    pub dq: Tensor,
    pub dk: Tensor,
    pub dv: Tensor,
}

/// Candle-composite analytic backward for the naive SDPA fallback that
/// [`gqa_attention_core_prefill`] runs when flash-attention is unavailable
/// (e.g. `head_dim` ∉ {128, 256}, as on the tiny synthetic test model with
/// `head_dim = 16`). Device-agnostic (runs in candle F32; works on CUDA
/// without a host round-trip), so the kt-tape
/// `crate::tape_forward::SdpaBackward` op can wrap it the same way
/// `crate::tape_forward::GdnRecurrentBackward` wraps
/// [`gdn_recurrent_backward_no_grad`].
///
/// # Inputs
///
/// `q`/`k`/`v` are the **pre-attention, head-FIRST** tensors the fallback
/// consumes: `q = [B, nq, T, hd]`, `k`/`v = [B, nkv, T, hd]` (the
/// `prepared.{q,k,v}.transpose(1,2)` layout — BEFORE the GQA expand). `scale`
/// is the dot-product scale `1/sqrt(head_dim)` (the forward divides scores by
/// `sqrt(head_dim)`; here we pass the reciprocal as a multiplier). `causal`
/// selects the strict-upper-triangular mask the forward applies via
/// [`apply_causal_mask_with_offset`] (offset 0, full prefill).
///
/// # Math (head-FIRST, GQA-expanded; per `[B, nq, T, *]` block)
///
/// Forward:
/// ```text
/// scores = (q @ kᵀ) * scale            [B, nq, T, T]
/// scores[..,i,j] = -inf   if causal && j > i
/// p      = softmax(scores, last)        [B, nq, T, T]
/// out    = p @ v                        [B, nq, T, hd]
/// ```
/// Backward (`g = grad_out`, `[B, nq, T, hd]`):
/// ```text
/// dv_exp  = pᵀ @ g                       [B, nq, T, hd]
/// dp      = g @ vᵀ                       [B, nq, T, T]
/// dscores = p * (dp - Σ_last(p * dp))    (softmax adjoint, masked rows ⇒ 0)
/// dscores[..,i,j] = 0     if causal && j > i
/// dq      = (dscores  @ k) * scale       [B, nq, T, hd]
/// dk_exp  = (dscoresᵀ @ q) * scale       [B, nq, T, hd]
/// ```
/// `dk_exp`/`dv_exp` are then GQA-collapsed from `nq` back to `nkv` by summing
/// each group of `nq/nkv` query heads (mirroring the forward's
/// `unsqueeze(2).expand(...).reshape(...)` broadcast of `k`/`v`).
///
/// The explicit `dscores` re-mask is belt-and-suspenders: `softmax(-inf) ≈ 0`
/// already zeroes the masked column in `p` (and hence its `dscores`
/// contribution), but re-masking guarantees exactness regardless of any
/// finite-`-inf` softmax stabilisation. Outputs are cast back to the inputs'
/// dtypes so they match the `Var` layouts.
pub fn sdpa_fallback_backward_no_grad(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
    grad_out: &Tensor,
) -> Result<SdpaFallbackBackwardGrads> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let v_dtype = v.dtype();

    let (batch, num_heads, q_len, head_dim) = q.dims4()?;
    let num_kv_heads = k.dim(1)?;
    if num_kv_heads == 0 || num_heads % num_kv_heads != 0 {
        anyhow::bail!(
            "sdpa_fallback_backward_no_grad: invalid GQA num_heads={num_heads} \
             num_kv_heads={num_kv_heads}"
        );
    }
    let kv_len = k.dim(2)?;
    anyhow::ensure!(
        v.dims() == [batch, num_kv_heads, kv_len, head_dim].as_slice(),
        "sdpa_fallback_backward_no_grad: v shape {:?} incompatible with q=[{batch},{num_heads},{q_len},{head_dim}] k=[{batch},{num_kv_heads},{kv_len},{head_dim}]",
        v.dims()
    );
    anyhow::ensure!(
        kv_len >= q_len,
        "sdpa_fallback_backward_no_grad: prefix SDPA requires kv_len >= q_len, got q_len={q_len} kv_len={kv_len}"
    );
    let past_len = kv_len - q_len;
    let gqa_ratio = num_heads / num_kv_heads;

    let q_f32 = q.to_dtype(DType::F32)?;
    let k_f32 = k.to_dtype(DType::F32)?;
    let v_f32 = v.to_dtype(DType::F32)?;
    let g_f32 = grad_out.to_dtype(DType::F32)?;

    // GQA-expand k/v from num_kv_heads -> num_heads, exactly mirroring the
    // forward's `unsqueeze(2).expand(...).contiguous().reshape(...)`.
    let expand_heads = |t: &Tensor, last: usize| -> Result<Tensor> {
        if gqa_ratio > 1 {
            Ok(t.unsqueeze(2)?
                .expand([batch, num_kv_heads, gqa_ratio, kv_len, last])?
                .contiguous()?
                .reshape((batch, num_heads, kv_len, last))?)
        } else {
            Ok(t.contiguous()?)
        }
    };
    let k_exp = expand_heads(&k_f32, head_dim)?; // [B, nq, T, hd]
    let v_exp = expand_heads(&v_f32, head_dim)?; // [B, nq, T, hd]
    let q_f32_c = q_f32.contiguous()?;

    // Recompute the forward scores -> probabilities (the backward needs `p`).
    // scores = (q @ kᵀ) * scale; causal-masked; p = softmax(last).
    let scores = kiln_tensor::ops::matmul_rhs_transposed(&q_f32_c, &k_exp)?; // [B, nq, Tq, Tk]
    let scores = scores.affine(scale, 0.0)?;
    let scores = if causal {
        apply_causal_mask_with_offset(&scores, q_len, kv_len, past_len)?
    } else {
        scores
    };
    // Numerically-stable softmax over the last axis (max-shift), matching the
    // forward's `cuda_softmax_last_dim` composite. Kept inline (rather than
    // `candle_nn::ops::softmax_last_dim`) so this stays a self-contained
    // device-agnostic candle-F32 composite.
    let max_val = scores.max_keepdim(LAST_DIM)?; // [B, nq, T, 1]
    let exp_shifted = scores.broadcast_sub(&max_val)?.exp()?;
    let sum_exp = exp_shifted.sum_keepdim(LAST_DIM)?; // [B, nq, T, 1]
    let p = exp_shifted.broadcast_div(&sum_exp)?; // [B, nq, T, T]

    // dv_exp = pᵀ @ g. p is [B, nq, T, T] (T_q x T_k), g is [B, nq, T, hd];
    // pᵀ over the (T_q, T_k) axes gives [B, nq, T_k, T_q] @ [B, nq, T_q, hd].
    let p_c = p.contiguous()?;
    let g_c = g_f32.contiguous()?;
    let dv_exp = kiln_tensor::ops::matmul_lhs_transposed(&p_c, &g_c)?; // [B, nq, Tk, hd]

    // dp = g @ vᵀ. [B, nq, Tq, hd] @ [B, nq, hd, Tk] -> [B, nq, Tq, Tk].
    let dp = kiln_tensor::ops::matmul_rhs_transposed(&g_c, &v_exp)?; // [B, nq, Tq, Tk]

    // Softmax adjoint: dscores = p * (dp - Σ_last(p * dp)).
    let sum_pdp = (&p_c * &dp)?.sum_keepdim(LAST_DIM)?; // [B, nq, Tq, 1]
    let dscores = (&p_c * &dp.broadcast_sub(&sum_pdp)?)?; // [B, nq, Tq, Tk]

    // Re-zero the strictly-future (masked) score positions so no gradient
    // leaks across the causal boundary (exactness; softmax already ≈0 there).
    let dscores = if causal {
        let device = dscores.device();
        // 1.0 where allowed (j <= past_len + i), 0.0 where masked.
        let keep: Vec<f32> = (0..q_len)
            .flat_map(|i| {
                let max_kv = past_len + i + 1;
                (0..kv_len).map(move |j| if j < max_kv { 1.0f32 } else { 0.0f32 })
            })
            .collect();
        let keep = Tensor::new(&keep, device)?.reshape((1, 1, q_len, kv_len))?;
        dscores.broadcast_mul(&keep)?
    } else {
        dscores
    };
    let dscores = dscores.contiguous()?;

    // dq = (dscores @ k) * scale. [B, nq, Tq, Tk] @ [B, nq, Tk, hd].
    let dq = dscores.broadcast_matmul(&k_exp)?.affine(scale, 0.0)?; // [B, nq, Tq, hd]

    // dk_exp = (dscoresᵀ @ q) * scale. dscoresᵀ over (T_q, T_k):
    // [B, nq, Tk, Tq] @ [B, nq, Tq, hd] -> [B, nq, Tk, hd].
    let dk_exp = kiln_tensor::ops::matmul_lhs_transposed(&dscores, &q_f32_c)?.affine(scale, 0.0)?; // [B, nq, Tk, hd]

    // GQA-collapse dk_exp / dv_exp from num_heads back to num_kv_heads by
    // summing each group of `gqa_ratio` query heads (the adjoint of the
    // forward's head broadcast). Reshape [B, nkv, groups, T, hd] then sum the
    // group axis.
    let collapse = |dexp: &Tensor| -> Result<Tensor> {
        if gqa_ratio > 1 {
            let grouped = dexp.reshape((batch, num_kv_heads, gqa_ratio, kv_len, head_dim))?;
            Ok(grouped.sum(2)?) // [B, nkv, T, hd]
        } else {
            Ok(dexp.clone())
        }
    };
    let dk = collapse(&dk_exp)?;
    let dv = collapse(&dv_exp)?;

    Ok(SdpaFallbackBackwardGrads {
        dq: dq.to_dtype(q_dtype)?,
        dk: dk.to_dtype(k_dtype)?,
        dv: dv.to_dtype(v_dtype)?,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn gdn_chunk_prep_f32(
    g: &Tensor,
    v: &Tensor,
    kkt: &Tensor,
    qkt: &Tensor,
    ks_entry: &Tensor,
    q_s: &Tensor,
) -> Result<(Tensor, Tensor, Tensor, Tensor, Tensor, Tensor)> {
    let (batch, heads, chunk, _) = v.dims4()?;
    let device = v.device();
    let g_f32 = g.to_dtype(DType::F32)?;
    let big_g = g_f32.cumsum(LAST_DIM)?;
    let big_g_col = big_g.unsqueeze(3)?;
    let big_g_row = big_g.unsqueeze(2)?;
    let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
    let zero_delta = Tensor::zeros_like(&decay_delta)?;
    let strict_bool = strict_lower_tri_bool(chunk, &device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;
    let causal_bool = causal_lower_tri_bool(chunk, &device)?
        .reshape((1, 1, chunk, chunk))?
        .broadcast_as((batch, heads, chunk, chunk))?;
    // Phase 7 (#1082): route the two `where_cond(...).exp()` steps
    // through `try_kt_exp` when stable KT routes are enabled. The where_cond itself stays
    // candle-side (candle's masked-select plumbing handles the
    // selection), but the resulting tensor's exp goes through the
    // kt-API dispatch instead of the candle `.exp()` composite.
    // Mirrors the `big_g.exp()` and `g_last.exp()` wirings already
    // in this function. Falls through to the candle path when any
    // precondition fails so behavior is identical with the gate
    // off.
    let strict_decay = {
        let masked = strict_bool.where_cond(&decay_delta, &zero_delta)?;
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&masked)? {
                Some(out) => out,
                None => masked.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            masked.exp()?
        }
    };
    let causal_decay = {
        let masked = causal_bool.where_cond(&decay_delta, &zero_delta)?;
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&masked)? {
                Some(out) => out,
                None => masked.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            masked.exp()?
        }
    };
    // Phase 7 (#1082): route `big_g.exp()` through
    // `try_kt_exp` when stable KT routes are enabled. Falls through to the
    // candle composite when any precondition fails so behavior is
    // identical with the gate off.
    let p = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&big_g)? {
                Some(out) => out,
                None => big_g.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            big_g.exp()?
        }
    };
    let p_col = p.unsqueeze(3)?;
    // #1082: U8→F32 cast is unsupported on the kt CUDA path (cast covers
    // float↔float + U32↔I64; the unsupported-CUDA-cast fallback hits the CPU
    // branch on a CUDA tensor). Build the F32 multiplicative masks via
    // `where_cond` (1.0 where set, else 0.0) — the same selection the decay
    // sites above use. This prep runs on CUDA inside the GDN recurrence backward
    // (tape-authoritative); inference uses the fused kernel, so it never hit
    // this cast before.
    let mask_ones = Tensor::ones((batch, heads, chunk, chunk), DType::F32, device)?;
    let mask_zeros = Tensor::zeros((batch, heads, chunk, chunk), DType::F32, device)?;
    let strict_mask = strict_bool.where_cond(&mask_ones, &mask_zeros)?;
    let causal_mask = causal_bool.where_cond(&mask_ones, &mask_zeros)?;

    let v_f32 = v.to_dtype(DType::F32)?;
    let kkt_f32 = kkt.to_dtype(DType::F32)?;
    let qkt_f32 = qkt.to_dtype(DType::F32)?;
    let ks_entry_f32 = ks_entry.to_dtype(DType::F32)?;
    let q_s_f32 = q_s.to_dtype(DType::F32)?;
    let v_prime = (&v_f32 - ks_entry_f32.broadcast_mul(&p_col)?)?;
    let a_strict = kkt_f32
        .broadcast_mul(&strict_decay)?
        .broadcast_mul(&strict_mask)?
        .contiguous()?;
    let b_mask = qkt_f32
        .broadcast_mul(&causal_decay)?
        .broadcast_mul(&causal_mask)?
        .contiguous()?;
    let q_s_scaled = q_s_f32.broadcast_mul(&p_col)?;
    let g_last = big_g.narrow(2, chunk - 1, 1)?;
    // Phase 7 (#1082): route the `g_last.broadcast_sub(&big_g).exp()`
    // step through `try_kt_exp` when stable KT routes are enabled. Mirrors the same gate that
    // already wraps `big_g.exp()` and `g_last.exp()` in this same
    // function. The `broadcast_sub` itself stays candle-side
    // (candle's broadcast plumbing handles the shape), but the
    // resulting tensor goes through the kt-API exp dispatch
    // instead of the candle `.exp()` composite. Falls through to
    // the candle path when any precondition fails so behavior is
    // identical with the gate off.
    let decay_last_col = {
        let g_diff = g_last.broadcast_sub(&big_g)?;
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&g_diff)? {
                Some(out) => out,
                None => g_diff.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            g_diff.exp()?
        }
    };
    // Phase 7 (#1082): same kt-API migration for `g_last.exp()`.
    let p_last_unsqueezed = {
        #[cfg(feature = "cuda")]
        {
            match try_kt_exp(&g_last)? {
                Some(out) => out,
                None => g_last.exp()?,
            }
        }
        #[cfg(not(feature = "cuda"))]
        {
            g_last.exp()?
        }
    };
    let p_last = p_last_unsqueezed.squeeze(2)?;
    Ok((
        a_strict,
        b_mask,
        v_prime,
        q_s_scaled,
        decay_last_col,
        p_last,
    ))
}

pub(super) fn solve_tri_transpose_f32(
    backend: &dyn BackendRuntime,
    a_strict: &Tensor,
    beta: &Tensor,
    dw: &Tensor,
) -> Result<Tensor> {
    let (_, _, chunk, _) = dw.dims4()?;
    if chunk <= 128
        && !any_kt_tensor_tracks_op(&[a_strict, beta, dw])
        && let Some(out) = GdnBackend::runtime_gdn_solve_tri_transpose(backend, a_strict, beta, dw)?
    {
        return Ok(out);
    }
    let mut rows_rev: Vec<Tensor> = Vec::with_capacity(chunk);
    for t in (0..chunk).rev() {
        let dw_t = dw.narrow(2, t, 1)?;
        let dr_t = if rows_rev.is_empty() {
            dw_t
        } else {
            let future_len = chunk - t - 1;
            let mut future_refs: Vec<&Tensor> = Vec::with_capacity(future_len);
            for row in rows_rev.iter().rev() {
                future_refs.push(row);
            }
            let dr_future = Tensor::cat(&future_refs, 2)?;
            let a_col = a_strict.narrow(2, t + 1, future_len)?.narrow(3, t, 1)?;
            let beta_future = beta.narrow(2, t + 1, future_len)?.unsqueeze(3)?;
            let weights = a_col.broadcast_mul(&beta_future)?;
            // Phase 7 (#1082): route the chunkwise-backward sum-over-time
            // reduction through kiln_tensor::cuda_sum_axis when
            // stable KT routes.
            // Falls through to candle on any precondition failure.
            let acc_pre = dr_future.broadcast_mul(&weights)?;
            let acc_sum = {
                #[cfg(feature = "cuda")]
                {
                    if let Some(out) = try_kt_sum_axis(&acc_pre, 2)? {
                        out
                    } else {
                        acc_pre.sum(2)?
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    acc_pre.sum(2)?
                }
            };
            let acc = acc_sum.unsqueeze(2)?;
            (dw_t - acc)?
        };
        rows_rev.push(dr_t);
    }
    rows_rev.reverse();
    let refs: Vec<&Tensor> = rows_rev.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

pub(super) fn reverse_cumsum_time_uses_index_select(device: Device) -> bool {
    match device {
        Device::Cpu => true,
        #[cfg(feature = "cuda")]
        Device::Cuda(_) => true,
        #[cfg(feature = "rocm")]
        Device::Rocm(_) => true,
        _ => false,
    }
}

pub(super) fn reverse_time_indices(chunk: usize, device: Device) -> Result<Tensor> {
    let indices_host: Vec<u32> = (0..chunk as u32).rev().collect();
    Ok(Tensor::from_slice(&indices_host, vec![chunk])?.to_device(device)?)
}

pub(super) fn reverse_cumsum_time_loop(x: &Tensor) -> Result<Tensor> {
    let chunk = x.dim(2)?;
    let mut rows_rev: Vec<Tensor> = Vec::with_capacity(chunk);
    let mut acc: Option<Tensor> = None;
    for t in (0..chunk).rev() {
        let x_t = x.narrow(2, t, 1)?;
        let next = match acc {
            Some(prev) => (&prev + &x_t)?,
            None => x_t,
        };
        rows_rev.push(next.clone());
        acc = Some(next);
    }
    rows_rev.reverse();
    let refs: Vec<&Tensor> = rows_rev.iter().collect();
    Ok(Tensor::cat(&refs, 2)?)
}

pub(super) fn reverse_cumsum_time(x: &Tensor, reverse_indices: Option<&Tensor>) -> Result<Tensor> {
    let Some(indices) = reverse_indices else {
        return reverse_cumsum_time_loop(x);
    };
    let x_c = x.contiguous()?;
    let rev = x_c.index_select(indices, 2)?;
    let rev_cumsum = rev.cumsum(2)?;
    Ok(rev_cumsum.index_select(indices, 2)?)
}

#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_backward_no_grad(
    backend: &dyn BackendRuntime,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    beta: &Tensor,
    g: &Tensor,
    entry_state: &Tensor,
    grad_out: &Tensor,
    grad_exit_state: Option<&Tensor>,
    chunk_size: usize,
) -> Result<GdnRecurrentBackwardGrads> {
    let (batch, heads, seq_len, _dk) = q.dims4()?;
    let full_chunks = seq_len / chunk_size;
    let tail = seq_len - full_chunks * chunk_size;
    let total_chunks = full_chunks + if tail > 0 { 1 } else { 0 };

    let q = q.to_dtype(DType::F32)?;
    let k = k.to_dtype(DType::F32)?;
    let v = v.to_dtype(DType::F32)?;
    let beta = beta.to_dtype(DType::F32)?;
    let g = g.to_dtype(DType::F32)?;
    let grad_out = grad_out.to_dtype(DType::F32)?;
    let mut state = entry_state.to_dtype(DType::F32)?;
    let mut state_snapshots: Vec<Tensor> = Vec::with_capacity(total_chunks);
    let device = q.device();
    let full_chunk_masks = if full_chunks > 0 {
        Some(gdn_backward_chunk_masks(batch, heads, chunk_size, &device)?)
    } else {
        None
    };
    let tail_chunk_masks = if tail > 0 {
        Some(gdn_backward_chunk_masks(batch, heads, tail, &device)?)
    } else {
        None
    };
    let use_fast_reverse_cumsum = reverse_cumsum_time_uses_index_select(device);
    let full_reverse_indices = if use_fast_reverse_cumsum && full_chunks > 0 {
        Some(reverse_time_indices(chunk_size, device)?)
    } else {
        None
    };
    let tail_reverse_indices = if use_fast_reverse_cumsum && tail > 0 {
        Some(reverse_time_indices(tail, device)?)
    } else {
        None
    };

    for ci in 0..total_chunks {
        let chunk = if ci >= full_chunks { tail } else { chunk_size };
        let t_off = ci * chunk_size;
        state_snapshots.push(state.clone());
        let q_c = q.narrow(2, t_off, chunk)?.contiguous()?;
        let k_c = k.narrow(2, t_off, chunk)?.contiguous()?;
        let v_c = v.narrow(2, t_off, chunk)?.contiguous()?;
        let beta_c = beta.narrow(2, t_off, chunk)?.contiguous()?;
        let g_c = g.narrow(2, t_off, chunk)?.contiguous()?;
        let ks_entry = k_c.matmul(&state)?;
        let q_s = q_c.matmul(&state)?;
        let kkt = kiln_tensor::ops::matmul_rhs_transposed(&k_c, &k_c)?;
        let qkt = kiln_tensor::ops::matmul_rhs_transposed(&q_c, &k_c)?;
        let (a_strict, _b_mask, v_prime, _q_s_scaled, decay_last_col, p_last) =
            gdn_chunk_prep_f32(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?;
        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, chunk)?.contiguous()?;
        let state_scaled = state.broadcast_mul(&p_last.unsqueeze(2)?.unsqueeze(3)?)?;
        let w_weighted = w.broadcast_mul(&decay_last_col.unsqueeze(3)?)?;
        let delta_state = kiln_tensor::ops::matmul_lhs_transposed(&k_c, &w_weighted)?;
        state = (state_scaled + delta_state)?;
    }

    let mut dq_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dk_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dv_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dbeta_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut dg_chunks: Vec<Option<Tensor>> = (0..total_chunks).map(|_| None).collect();
    let mut d_s_carry = match grad_exit_state {
        Some(grad) => Some(grad.to_dtype(DType::F32)?),
        None => None,
    };

    for ci in (0..total_chunks).rev() {
        let chunk = if ci >= full_chunks { tail } else { chunk_size };
        let t_off = ci * chunk_size;
        let chunk_masks = if chunk == chunk_size {
            full_chunk_masks
                .as_ref()
                .context("missing GDN full-chunk masks")?
        } else {
            tail_chunk_masks
                .as_ref()
                .context("missing GDN tail-chunk masks")?
        };
        let reverse_indices = if chunk == chunk_size {
            full_reverse_indices.as_ref()
        } else {
            tail_reverse_indices.as_ref()
        };
        let s_in = &state_snapshots[ci];
        let q_c = q.narrow(2, t_off, chunk)?.contiguous()?;
        let k_c = k.narrow(2, t_off, chunk)?.contiguous()?;
        let v_c = v.narrow(2, t_off, chunk)?.contiguous()?;
        let beta_c = beta.narrow(2, t_off, chunk)?.contiguous()?;
        let g_c = g.narrow(2, t_off, chunk)?.contiguous()?;
        let d_out = grad_out.narrow(2, t_off, chunk)?.contiguous()?;
        let ks_entry = k_c.matmul(s_in)?;
        let q_s = q_c.matmul(s_in)?;
        let kkt = kiln_tensor::ops::matmul_rhs_transposed(&k_c, &k_c)?;
        let qkt = kiln_tensor::ops::matmul_rhs_transposed(&q_c, &k_c)?;
        let (a_strict, b_mask, v_prime, q_s_scaled, decay_last_col, p_last) =
            gdn_chunk_prep_f32(&g_c, &v_c, &kkt, &qkt, &ks_entry, &q_s)?;
        let w = compute_w_chunk(backend, &a_strict, &v_prime, &beta_c, chunk)?.contiguous()?;

        let dq_s_scaled = d_out.clone();
        let d_w_scan = kiln_tensor::ops::matmul_lhs_transposed(&b_mask, &d_out)?;
        let d_b_mask = kiln_tensor::ops::matmul_rhs_transposed(&d_out, &w)?;

        let mut d_w_acc = d_w_scan;
        let mut d_decay_last_col_acc =
            Tensor::zeros((batch, heads, chunk), DType::F32, q.device())?;
        let mut d_p_last_acc = Tensor::zeros((batch, heads), DType::F32, q.device())?;
        let mut dk_state_extra: Option<Tensor> = None;
        let mut ds_state_extra: Option<Tensor> = None;

        if let Some(d_s_exit) = d_s_carry.as_ref() {
            let p_last_u = p_last.unsqueeze(2)?.unsqueeze(3)?;
            ds_state_extra = Some(d_s_exit.broadcast_mul(&p_last_u)?);
            // Phase 7 (#1082): wire `try_kt_sum_axis` into the
            // `(s_in * d_s_exit)?.sum(3)?.sum(2)?` reduction chain
            // for `d_p_last_acc`. Both sums collapse a tensor axis
            // (axis 3 then axis 2), and `try_kt_sum_axis` is the
            // axis-removing kt-API analogue of candle's
            // `Tensor::sum(axis)`. Falls through to the candle
            // composite when any precondition fails so behavior is
            // identical with the gate off.
            let prod = (s_in * d_s_exit)?;
            #[cfg(feature = "cuda")]
            let after_sum3 = match try_kt_sum_axis(&prod, 3)? {
                Some(out) => out,
                None => prod.sum(3)?,
            };
            #[cfg(not(feature = "cuda"))]
            let after_sum3 = prod.sum(3)?;
            #[cfg(feature = "cuda")]
            {
                d_p_last_acc = match try_kt_sum_axis(&after_sum3, 2)? {
                    Some(out) => out,
                    None => after_sum3.sum(2)?,
                };
            }
            #[cfg(not(feature = "cuda"))]
            {
                d_p_last_acc = after_sum3.sum(2)?;
            }
            let tmp_dw = k_c.matmul(d_s_exit)?;
            d_w_acc = (&d_w_acc + &tmp_dw.broadcast_mul(&decay_last_col.unsqueeze(3)?)?)?;
            let tmp_dk = kiln_tensor::ops::matmul_rhs_transposed(&w, d_s_exit)?;
            dk_state_extra = Some(tmp_dk.broadcast_mul(&decay_last_col.unsqueeze(3)?)?);
            // Phase 7 (#1082): wire `try_kt_sum_axis` into the
            // `(&k_c * &tmp_dk)?.sum(D::Minus1)?` reduction for
            // `d_decay_last_col_acc`. The operand is a 4D tensor
            // [B, H, C, dim] so `D::Minus1` resolves to axis 3 at
            // runtime; the helper falls through to the candle
            // composite on any precondition failure so behavior is
            // identical with the gate off.
            let prod_kc = (&k_c * &tmp_dk)?;
            #[cfg(feature = "cuda")]
            {
                let axis = prod_kc.rank().saturating_sub(1);
                d_decay_last_col_acc = match try_kt_sum_axis(&prod_kc, axis)? {
                    Some(out) => out,
                    None => prod_kc.sum(LAST_DIM)?,
                };
            }
            #[cfg(not(feature = "cuda"))]
            {
                d_decay_last_col_acc = prod_kc.sum(LAST_DIM)?;
            }
        }
        let dr = solve_tri_transpose_f32(backend, &a_strict, &beta_c, &d_w_acc)?.contiguous()?;
        let a_w = a_strict.matmul(&w)?;
        let pre_beta = (&v_prime - &a_w)?;
        let d_v_prime = dr.broadcast_mul(&beta_c.unsqueeze(3)?)?.contiguous()?;
        // Phase 7 (#1082): wire `try_kt_sum_axis` into the
        // `(&pre_beta * &dr)?.sum(D::Minus1)?` reduction for
        // `d_beta`. The operand is a 4D tensor [B, H, C, dim] so
        // `D::Minus1` resolves to axis 3 at runtime; the helper
        // falls through to the candle composite on any precondition
        // failure so behavior is identical with the gate off.
        let prod_pb = (&pre_beta * &dr)?;
        #[cfg(feature = "cuda")]
        let d_beta = {
            let axis = prod_pb.rank().saturating_sub(1);
            match try_kt_sum_axis(&prod_pb, axis)? {
                Some(out) => out,
                None => prod_pb.sum(LAST_DIM)?,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let d_beta = prod_pb.sum(LAST_DIM)?;
        let dr_w_t = kiln_tensor::ops::matmul_rhs_transposed(&dr, &w)?;
        // #1082: build the F32 multiplicative mask via `where_cond` (1.0 where
        // strict-lower, else 0.0), NOT `.to_dtype(F32)` on the U8 bool mask — the
        // kt U8→F32 cast is unsupported (cast covers float↔float + U32↔I64), and
        // the unsupported-CUDA-cast fallback hits the CPU branch on a CUDA tensor
        // ("CastOp: storage must be CpuStorage on CPU"). This backward only
        // executes now that the tape chain reaches the recurrence.
        let strict_mask = &chunk_masks.strict_mask_f32;
        // Phase 7 (#1082): route the `beta_c.neg()` step through
        // `try_kt_neg` when stable KT routes are enabled. Mirrors the wirings
        // already in this same backward function and in `softplus`
        // / `cuda_sigmoid`. Falls through to the candle composite
        // when any precondition fails so behavior is identical with
        // the gate off.
        let beta_c_neg = {
            #[cfg(feature = "cuda")]
            {
                match try_kt_neg(&beta_c)? {
                    Some(out) => out,
                    None => beta_c.neg()?,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                beta_c.neg()?
            }
        };
        let d_a_strict = dr_w_t
            .broadcast_mul(&beta_c_neg.unsqueeze(3)?)?
            .broadcast_mul(strict_mask)?;

        let big_g = g_c.cumsum(LAST_DIM)?;
        // Phase 7 (#1082): route the `big_g.exp()` step through
        // `try_kt_exp` when stable KT routes are enabled. Mirrors the prefill
        // chunkwise `p = big_g.exp()` wirings (ca8de9eb, 288531c7,
        // 38e4ea3d). Falls through to the candle `.exp()` composite
        // when any precondition fails so behavior is identical with
        // the gate off.
        let p = {
            #[cfg(feature = "cuda")]
            {
                match try_kt_exp(&big_g)? {
                    Some(out) => out,
                    None => big_g.exp()?,
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                big_g.exp()?
            }
        };
        let p_col = p.unsqueeze(3)?;
        let d_v = d_v_prime.clone();
        // Phase 7 (#1082): route the `.neg()` step of the
        // `d_ks_entry` computation through `try_kt_neg` when stable KT routes
        // are enabled. The intermediate `d_v_prime.broadcast_mul(&p_col)`
        // stays candle-side; only the elementwise `.neg()` migrates
        // to a single `cuda_activation_unary` kind 12 (Neg)
        // dispatch. Falls through to candle on any precondition
        // failure.
        let d_ks_entry_pre = d_v_prime.broadcast_mul(&p_col)?;
        let d_ks_entry = {
            #[cfg(feature = "cuda")]
            {
                let negated = match try_kt_neg(&d_ks_entry_pre)? {
                    Some(out) => out,
                    None => d_ks_entry_pre.neg()?,
                };
                negated.contiguous()?
            }
            #[cfg(not(feature = "cuda"))]
            {
                d_ks_entry_pre.neg()?.contiguous()?
            }
        };
        // Phase 7 (#1082): wire `try_kt_sum_axis` into the two
        // `sum(D::Minus1)?` reductions that build `d_g_acc`:
        // first the `(&ks_entry * &d_ks_entry)?.sum(D::Minus1)?`
        // initializer, then the in-place
        // `(&q_s * &dq_s_scaled).broadcast_mul(&p_col).sum(D::Minus1)?`
        // accumulator. Both operands are 4D tensors so `D::Minus1`
        // resolves to axis 3 at runtime; the helper falls through
        // to the candle composite on any precondition failure so
        // behavior is identical with the gate off.
        let g_acc_prod = (&ks_entry * &d_ks_entry)?;
        #[cfg(feature = "cuda")]
        let mut d_g_acc = {
            let axis = g_acc_prod.rank().saturating_sub(1);
            match try_kt_sum_axis(&g_acc_prod, axis)? {
                Some(out) => out,
                None => g_acc_prod.sum(LAST_DIM)?,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let mut d_g_acc = g_acc_prod.sum(LAST_DIM)?;
        let d_q_s = dq_s_scaled.broadcast_mul(&p_col)?.contiguous()?;
        let qss_prod = (&q_s * &dq_s_scaled)?.broadcast_mul(&p_col)?;
        #[cfg(feature = "cuda")]
        let qss_sum = {
            let axis = qss_prod.rank().saturating_sub(1);
            match try_kt_sum_axis(&qss_prod, axis)? {
                Some(out) => out,
                None => qss_prod.sum(LAST_DIM)?,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let qss_sum = qss_prod.sum(LAST_DIM)?;
        d_g_acc = (&d_g_acc + &qss_sum)?;

        let big_g_col = big_g.unsqueeze(3)?;
        let big_g_row = big_g.unsqueeze(2)?;
        let decay_delta = big_g_col.broadcast_sub(&big_g_row)?;
        let zero_delta = Tensor::zeros_like(&decay_delta)?;
        let strict_bool = &chunk_masks.strict_bool;
        let causal_bool = &chunk_masks.causal_bool;
        let strict_mask_f32 = &chunk_masks.strict_mask_f32;
        let causal_mask_f32 = &chunk_masks.causal_mask_f32;
        // Phase 7 (#1082): route both `where_cond(...).exp()` steps
        // through `try_kt_exp` under the same stable KT routes
        // gate. The where_cond stays candle-side; only the
        // elementwise `.exp()` migrates. Mirrors the chunkwise
        // forward `gdn_chunk_prep_f32` strict/causal_decay wirings
        // (commit 38e4ea3d). Falls through to candle on any
        // precondition failure.
        let strict_decay = {
            let masked = strict_bool.where_cond(&decay_delta, &zero_delta)?;
            let exped = {
                #[cfg(feature = "cuda")]
                {
                    match try_kt_exp(&masked)? {
                        Some(out) => out,
                        None => masked.exp()?,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    masked.exp()?
                }
            };
            exped.broadcast_mul(strict_mask_f32)?
        };
        let causal_decay = {
            let masked = causal_bool.where_cond(&decay_delta, &zero_delta)?;
            let exped = {
                #[cfg(feature = "cuda")]
                {
                    match try_kt_exp(&masked)? {
                        Some(out) => out,
                        None => masked.exp()?,
                    }
                }
                #[cfg(not(feature = "cuda"))]
                {
                    masked.exp()?
                }
            };
            exped.broadcast_mul(causal_mask_f32)?
        };
        let d_kkt = d_a_strict.broadcast_mul(&strict_decay)?.contiguous()?;
        let d_qkt = d_b_mask.broadcast_mul(&causal_decay)?.contiguous()?;
        let term_a = d_a_strict
            .broadcast_mul(&strict_decay)?
            .broadcast_mul(&kkt)?;
        let term_b = d_b_mask.broadcast_mul(&causal_decay)?.broadcast_mul(&qkt)?;
        let term = (&term_a + &term_b)?;
        // Phase 7 (#1082): wire `try_kt_sum_axis` into the
        // `term.sum(D::Minus1)?` row-sum reduction. The operand is
        // a 4D tensor [B, H, C, C] so `D::Minus1` resolves to axis
        // 3 at runtime; the helper falls through to the candle
        // composite on any precondition failure so behavior is
        // identical with the gate off. Complements the col-sum
        // (axis 2) which is already wired below.
        #[cfg(feature = "cuda")]
        let row_sum = {
            let axis = term.rank().saturating_sub(1);
            match try_kt_sum_axis(&term, axis)? {
                Some(out) => out,
                None => term.sum(LAST_DIM)?,
            }
        };
        #[cfg(not(feature = "cuda"))]
        let row_sum = term.sum(LAST_DIM)?;
        // Phase 7 (#1082): route the col-sum reduction (axis 2,
        // the strict-mask row dim) through
        // kiln_tensor::cuda_sum_axis when stable KT routes are enabled. Falls
        // through to candle on any precondition failure. The complementary
        // `row_sum` (last-dim non-keepdim) stays on candle since this helper
        // targets `sum(axis)` shapes and
        // candle's `.sum(D::Minus1)` here already feeds the
        // existing fused-decode fast paths upstream.
        let col_sum = {
            #[cfg(feature = "cuda")]
            {
                if let Some(out) = try_kt_sum_axis(&term, 2)? {
                    out
                } else {
                    term.sum(2)?
                }
            }
            #[cfg(not(feature = "cuda"))]
            {
                term.sum(2)?
            }
        };
        d_g_acc = (&d_g_acc + &row_sum)?;
        d_g_acc = (&d_g_acc - &col_sum)?;

        let decay_term = decay_last_col.broadcast_mul(&d_decay_last_col_acc)?;
        let decay_sum = decay_term.sum(LAST_DIM)?.unsqueeze(2)?;
        let last_mask = &chunk_masks.last_mask_f32;
        d_g_acc = (&d_g_acc - &decay_term)?;
        d_g_acc = (&d_g_acc + &decay_sum.broadcast_mul(last_mask)?)?;
        let p_last_term = p
            .narrow(2, chunk - 1, 1)?
            .squeeze(2)?
            .broadcast_mul(&d_p_last_acc)?
            .unsqueeze(2)?
            .broadcast_mul(last_mask)?;
        d_g_acc = (&d_g_acc + &p_last_term)?;
        let d_g = reverse_cumsum_time(&d_g_acc, reverse_indices)?;

        let d_k_from_kkt =
            (&d_kkt.matmul(&k_c)? + &kiln_tensor::ops::matmul_lhs_transposed(&d_kkt, &k_c)?)?;
        let d_k_from_qkt = kiln_tensor::ops::matmul_lhs_transposed(&d_qkt, &q_c)?;
        let d_k_from_ks = kiln_tensor::ops::matmul_rhs_transposed(&d_ks_entry, s_in)?;
        let mut d_k = (&(&d_k_from_kkt + &d_k_from_qkt)? + &d_k_from_ks)?;
        if let Some(extra) = dk_state_extra.as_ref() {
            d_k = (&d_k + extra)?;
        }
        let d_q = (&d_qkt.matmul(&k_c)? + &kiln_tensor::ops::matmul_rhs_transposed(&d_q_s, s_in)?)?;
        let d_s_from_ks = kiln_tensor::ops::matmul_lhs_transposed(&k_c, &d_ks_entry)?;
        let d_s_from_qs = kiln_tensor::ops::matmul_lhs_transposed(&q_c, &d_q_s)?;
        let mut d_s_in = (&d_s_from_ks + &d_s_from_qs)?;
        if let Some(extra) = ds_state_extra.as_ref() {
            d_s_in = (&d_s_in + extra)?;
        }

        dq_chunks[ci] = Some(d_q);
        dk_chunks[ci] = Some(d_k);
        dv_chunks[ci] = Some(d_v);
        dbeta_chunks[ci] = Some(d_beta);
        dg_chunks[ci] = Some(d_g);
        d_s_carry = Some(d_s_in);

        let _ = q_s_scaled;
    }

    // Phase 7 (#1082): when stable KT routes are enabled and every chunk is a
    // contiguous CUDA tensor of a supported dtype, route the
    // time-axis (axis=2) per-gradient concat through
    // `kiln_tensor::cuda_concat(_, 2)`. Mirrors the
    // gdn_chunkwise_recurrence cat_out wiring and the conv1d
    // prefill/decode cat_dim2 wirings. Falls through to the
    // candle composite when any precondition fails so behavior
    // is identical with the gate off. The closure is invoked
    // five times per backward (dq/dk/dv/dbeta/dg) so this
    // covers five fast-path sites in one wire-up.
    let collect = |chunks: &[Option<Tensor>], name: &str| -> Result<Tensor> {
        let mut refs = Vec::with_capacity(chunks.len());
        for (idx, chunk) in chunks.iter().enumerate() {
            refs.push(
                chunk
                    .as_ref()
                    .ok_or_else(|| anyhow::anyhow!("missing {name} chunk {idx}"))?,
            );
        }
        #[cfg(feature = "cuda")]
        {
            if let Some(out) = try_kt_cat_dim2(&refs)? {
                return Ok(out);
            }
        }
        Ok(Tensor::cat(&refs, 2)?)
    };

    let dq = collect(&dq_chunks, "dq")?;
    let dk = collect(&dk_chunks, "dk")?;
    let dv = collect(&dv_chunks, "dv")?;
    let dbeta = collect(&dbeta_chunks, "dbeta")?;
    let dg = collect(&dg_chunks, "dg")?;

    Ok(GdnRecurrentBackwardGrads {
        dq,
        dk,
        dv,
        dbeta,
        dg,
        d_state: d_s_carry,
    })
}
