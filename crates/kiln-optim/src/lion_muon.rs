//! `Lion` + `Muon` — two non-AdamW optimizer variants.
//!
//! Per the Phase 6.5 issue bullet:
//!
//! > `kiln-optim` crate with `OptimStep` trait: AdamW, SGD, Lion, Muon
//!
//! [`Lion`] (Chen et al. 2023, "Symbolic Discovery of Optimization
//! Algorithms") is a compact-state sign-momentum optimizer.
//!
//! [`Muon`] (Bernstein-Newhouse 2024, "Old optimizer, new norm"; Jordan
//! et al. 2024 nanoGPT speedrun) is momentum-orthogonalized SGD: it
//! projects the (Nesterov) momentum onto the closest semi-orthogonal
//! matrix via a Newton-Schulz iteration before each step. This is the
//! production CPU reference — the oracle the per-backend GPU Muon
//! kernels (CUDA / ROCm / Vulkan / Metal) are validated against.

use std::collections::HashMap;

use kiln_param::Parameter;
use kiln_tensor::{CpuStorage, DType, Layout, Storage, Tensor, TensorId};

use crate::{OptimStep, StepError, StochasticRoundingPolicy};

/// Lion (Chen et al. 2023). Compact-state alternative to AdamW —
/// stores only the EMA of grads (no second moment).
///
/// # Algorithm
///
/// ```text
/// c_t = β1 * m_{t-1} + (1 - β1) * g_t            # interim
/// p_t = p_{t-1} - lr * (sign(c_t) + λ * p_{t-1}) # update
/// m_t = β2 * m_{t-1} + (1 - β2) * g_t            # state EMA
/// ```
///
/// `sign(0) = 0`. The sign-based step gives uniform magnitude
/// updates regardless of gradient scale, which is the property that
/// motivates Lion's reduced memory footprint vs AdamW.
#[derive(Debug, Default)]
pub struct Lion {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub weight_decay: f32,
    /// Per-parameter momentum EMA. F32 regardless of master dtype.
    ema: HashMap<TensorId, LionEma>,
}

#[derive(Debug, Clone)]
pub struct LionEma {
    pub m: Vec<f32>,
    pub step: u64,
}

impl Lion {
    pub fn new(lr: f32, beta1: f32, beta2: f32, weight_decay: f32) -> Self {
        Lion {
            lr,
            beta1,
            beta2,
            weight_decay,
            ema: HashMap::new(),
        }
    }

    pub fn default_hp() -> Self {
        // Defaults match the Chen et al. paper's recommended HP for
        // language models.
        Lion::new(
            /*lr=*/ 1e-4, /*β1=*/ 0.9, /*β2=*/ 0.99, /*λ=*/ 0.0,
        )
    }

    pub fn ema_for(&self, id: TensorId) -> Option<&LionEma> {
        self.ema.get(&id)
    }

    pub fn parameter_count(&self) -> usize {
        self.ema.len()
    }
}

impl OptimStep for Lion {
    fn name(&self) -> &'static str {
        "lion"
    }

    fn step(&mut self, param: &mut Parameter, grad: &Tensor) -> Result<(), StepError> {
        let master = param
            .backward_storage()
            .ok_or(StepError::NoBackwardStorage)?
            .clone();
        if grad.shape() != master.shape() {
            return Err(StepError::GradShapeMismatch {
                grad_shape: grad.shape().to_vec(),
                master_shape: master.shape().to_vec(),
            });
        }
        let policy = param.amp_policy();
        if grad.dtype() != policy.backward_compute_dtype {
            return Err(StepError::GradDtypeMismatch {
                grad_dtype: grad.dtype(),
                policy_dtype: policy.backward_compute_dtype,
            });
        }
        if !matches!(policy.master_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Lion: master dtype must be F32/BF16/F16, got {}",
                policy.master_dtype
            ))));
        }

        let n = master.element_count();
        let mut master_f32 = read_master_to_f32(&master)?;
        let grad_f32 = read_master_to_f32(grad)?;

        let entry = self
            .ema
            .entry(param.tensor_id())
            .or_insert_with(|| LionEma {
                m: vec![0.0; n],
                step: 0,
            });
        if entry.m.len() != n {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Lion: ema buffer shape ({}) drifted from master ({})",
                entry.m.len(),
                n
            ))));
        }
        entry.step += 1;

        for i in 0..n {
            let g = grad_f32[i];
            // 1. Interim c_t = β1 * m + (1 - β1) * g
            let c = self.beta1 * entry.m[i] + (1.0 - self.beta1) * g;
            // 2. Update: p -= lr * (sign(c) + λ * p)
            let sign_c = if c > 0.0 {
                1.0
            } else if c < 0.0 {
                -1.0
            } else {
                0.0
            };
            master_f32[i] -= self.lr * (sign_c + self.weight_decay * master_f32[i]);
            // 3. Advance state: m_t = β2 * m + (1 - β2) * g
            entry.m[i] = self.beta2 * entry.m[i] + (1.0 - self.beta2) * g;
        }

        // Write updated master back to the Parameter. Lion keeps
        // round-to-nearest (no stochastic-rounding state); the master
        // write moves back to the param's device.
        let new_master = build_master_tensor(
            policy.master_dtype,
            master.shape(),
            &master_f32,
            StochasticRoundingPolicy::RoundToNearest,
            entry.step,
            master.device(),
        )?;
        param.replace_backward_storage(Some(new_master));
        // #1082 Phase 2.7: end-of-optimizer-step epoch bump (see AdamW).
        param.bump_epoch();
        Ok(())
    }

    fn reset(&mut self) {
        self.ema.clear();
    }
}

/// Read a tensor (any device) into a host `Vec<f32>`, promoting from
/// BF16/F16/F32. Device-resident tensors are D2H-copied first — the
/// on-device GPU kernels handle the resident fast path; this is the
/// portable host reference / fallback.
fn read_master_to_f32(t: &Tensor) -> Result<Vec<f32>, StepError> {
    let host = if matches!(t.device(), kiln_tensor::Device::Cpu) {
        t.clone()
    } else {
        t.to_device(kiln_tensor::Device::Cpu)
            .map_err(StepError::Tensor)?
    };
    let cpu = host
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            StepError::Tensor(kiln_tensor::Error::from_str(
                "lion_muon: tensor storage must be CpuStorage on CPU",
            ))
        })?;
    let bytes = cpu.as_bytes();
    let n = t.element_count();
    let mut out = Vec::with_capacity(n);
    match t.dtype() {
        DType::F32 => {
            for i in 0..n {
                out.push(f32::from_le_bytes(
                    bytes[i * 4..i * 4 + 4].try_into().unwrap(),
                ));
            }
        }
        DType::BF16 => {
            for i in 0..n {
                out.push(
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap()).to_f32(),
                );
            }
        }
        other => {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "lion_muon: unsupported dtype {other}"
            ))));
        }
    }
    Ok(out)
}

/// Stochastically round an `f32` to a `bf16` bit pattern (mirrors
/// `adamw.rs`). Adds a uniform 16-bit value before truncating so the
/// carry into bit 16 fires with probability proportional to the
/// dropped mantissa — unbiased in expectation. NaN passes through
/// round-to-nearest.
#[inline]
fn f32_to_bf16_stochastic_bits(v: f32, r: u16) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return half::bf16::from_f32(v).to_bits();
    }
    let rounded = bits.wrapping_add(r as u32);
    (rounded >> 16) as u16
}

/// Deterministic per-element uniform 16-bit draw from `(seed, step,
/// idx)` via a splitmix64 finalizer (mirrors `adamw.rs`).
#[inline]
fn stochastic_round_rng16(seed: u64, step: u64, idx: usize) -> u16 {
    let mut z = seed
        ^ step.wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ (idx as u64).wrapping_mul(0xd1b5_4a32_d192_ed03);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^= z >> 31;
    (z >> 32) as u16
}

/// Build a master `Tensor` from f32 values, in `dtype`, then move it to
/// `device`. BF16 honors an explicitly supplied stochastic policy, varied by
/// `step`; ordinary product construction uses round-to-nearest.
fn build_master_tensor(
    dtype: DType,
    shape: &[usize],
    values: &[f32],
    rounding: StochasticRoundingPolicy,
    step: u64,
    device: kiln_tensor::Device,
) -> Result<Tensor, StepError> {
    use std::sync::Arc;
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; values.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        DType::BF16 => match rounding {
            StochasticRoundingPolicy::Stochastic { seed } => {
                for (i, &v) in values.iter().enumerate() {
                    let r = stochastic_round_rng16(seed, step, i);
                    let b = f32_to_bf16_stochastic_bits(v, r);
                    bytes[i * 2..i * 2 + 2].copy_from_slice(&b.to_le_bytes());
                }
            }
            StochasticRoundingPolicy::RoundToNearest => {
                for (i, &v) in values.iter().enumerate() {
                    bytes[i * 2..i * 2 + 2].copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
                }
            }
        },
        DType::F16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2].copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        other => {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "lion_muon: unsupported master dtype {other}"
            ))));
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes).map_err(StepError::Tensor)?;
    let storage: Storage = Arc::new(cpu);
    let host = Tensor::from_parts(
        storage,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
    .map_err(StepError::Tensor)?;
    if matches!(device, kiln_tensor::Device::Cpu) {
        Ok(host)
    } else {
        host.to_device(device).map_err(StepError::Tensor)
    }
}

// ----------------------------------------------------------------------
// Muon — momentum-orthogonalized SGD (production CPU reference)
// ----------------------------------------------------------------------

/// Muon (Bernstein-Newhouse 2024 / Jordan et al. 2024). Heavy-ball
/// (optionally Nesterov) momentum-orthogonalized SGD: it projects the
/// momentum matrix onto the nearest semi-orthogonal matrix via a
/// Newton-Schulz quintic iteration before each step, then rescales the
/// update so its per-element RMS is shape-independent.
///
/// # Algorithm (per parameter, master in f32 working precision)
///
/// ```text
/// m_t   = momentum * m_{t-1} + g_t                  # heavy-ball state
/// b_t   = if nesterov { g_t + momentum * m_t } else { m_t }
/// if rank-2 (matrix):
///     O = newton_schulz(b_t, ns_iters)              # ≈ polar factor U Vᵀ
///     O *= sqrt(max(rows, cols))                    # RMS-matching scale
/// else:
///     O = b_t                                       # plain momentum SGD
/// p_t   = p_{t-1} * (1 - lr * weight_decay) - lr * O
/// ```
///
/// The Newton-Schulz uses the paper coefficients `(a, b, c) =
/// (3.4445, -4.7750, 2.0315)` for `ns_iters` (default 5) iterations,
/// computed via the gram-space P-accumulator (see [`newton_schulz`]) so
/// the per-step cost is dominated by two skinny GEMMs over the large
/// matrix dimension — the rest is `k×k` work where `k = min(rows, cols)`
/// (the LoRA rank, ≈16).
///
/// # Non-matrix parameters
///
/// Muon's orthogonalization is defined only for **rank-2** weights.
/// Vectors / scalars fall back to plain SGD-with-(Nesterov-)momentum.
#[derive(Debug)]
pub struct Muon {
    pub lr: f32,
    pub momentum: f32,
    pub nesterov: bool,
    /// Number of Newton-Schulz iterations. Paper uses 5.
    pub ns_iters: u32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
    rounding: StochasticRoundingPolicy,
    /// Per-parameter momentum buffer keyed on TensorId.
    momenta: HashMap<TensorId, MuonState>,
}

impl Default for Muon {
    fn default() -> Self {
        Muon::default_hp()
    }
}

#[derive(Debug, Clone)]
pub struct MuonState {
    pub m: Vec<f32>,
    pub step: u64,
}

impl Muon {
    pub fn new(lr: f32, momentum: f32, nesterov: bool, ns_iters: u32, weight_decay: f32) -> Self {
        Self::new_with_rounding(
            lr,
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
            StochasticRoundingPolicy::RoundToNearest,
        )
    }

    /// Construct Muon with an explicit programmatic rounding policy.
    pub fn new_with_rounding(
        lr: f32,
        momentum: f32,
        nesterov: bool,
        ns_iters: u32,
        weight_decay: f32,
        rounding: StochasticRoundingPolicy,
    ) -> Self {
        Muon {
            lr,
            momentum,
            nesterov,
            ns_iters,
            weight_decay,
            rounding,
            momenta: HashMap::new(),
        }
    }

    /// Recommended Muon defaults: momentum 0.95, Nesterov on, 5
    /// Newton-Schulz iterations, no weight decay, lr 2e-2 (Muon's
    /// orthogonalized + RMS-scaled update tolerates a larger lr than
    /// AdamW's 1e-3).
    pub fn default_hp() -> Self {
        Muon::new(
            /*lr=*/ 2e-2, /*momentum=*/ 0.95, /*nesterov=*/ true,
            /*ns_iters=*/ 5, /*weight_decay=*/ 0.0,
        )
    }

    pub fn momentum_for(&self, id: TensorId) -> Option<&MuonState> {
        self.momenta.get(&id)
    }

    pub fn parameter_count(&self) -> usize {
        self.momenta.len()
    }

    /// Rounding policy captured when this optimizer was constructed.
    pub fn rounding_policy(&self) -> StochasticRoundingPolicy {
        self.rounding
    }

    /// Install validated momentum for a checkpoint-restored parameter.
    ///
    /// The checkpoint owner resolves its stable parameter name to the new
    /// process-local [`TensorId`] before calling this method.
    pub fn restore_momentum(&mut self, id: TensorId, state: MuonState) -> Result<(), StepError> {
        if state.m.iter().any(|value| !value.is_finite()) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(
                "Muon: restored momentum contains non-finite values".to_string(),
            )));
        }
        self.momenta.insert(id, state);
        Ok(())
    }
}

impl OptimStep for Muon {
    fn name(&self) -> &'static str {
        "muon"
    }

    fn step(&mut self, param: &mut Parameter, grad: &Tensor) -> Result<(), StepError> {
        let master = param
            .backward_storage()
            .ok_or(StepError::NoBackwardStorage)?
            .clone();
        if grad.shape() != master.shape() {
            return Err(StepError::GradShapeMismatch {
                grad_shape: grad.shape().to_vec(),
                master_shape: master.shape().to_vec(),
            });
        }
        let policy = param.amp_policy();
        if grad.dtype() != policy.backward_compute_dtype {
            return Err(StepError::GradDtypeMismatch {
                grad_dtype: grad.dtype(),
                policy_dtype: policy.backward_compute_dtype,
            });
        }
        if !matches!(policy.master_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Muon: master dtype must be F32/BF16/F16, got {}",
                policy.master_dtype
            ))));
        }

        let shape = master.shape().to_vec();
        let n = master.element_count();
        let mut master_f32 = read_master_to_f32(&master)?;
        let grad_f32 = read_master_to_f32(grad)?;

        let entry = self
            .momenta
            .entry(param.tensor_id())
            .or_insert_with(|| MuonState {
                m: vec![0.0; n],
                step: 0,
            });
        if entry.m.len() != n {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Muon: momentum buffer shape ({}) drifted from master ({})",
                entry.m.len(),
                n
            ))));
        }
        entry.step += 1;

        // 1. Heavy-ball momentum: m = momentum*m + g (state update).
        for (m, &g) in entry.m.iter_mut().zip(grad_f32.iter()) {
            *m = self.momentum * *m + g;
        }
        // 2. Look-ahead direction b = nesterov ? g + momentum*m : m.
        let mut b = vec![0.0f32; n];
        if self.nesterov {
            for i in 0..n {
                b[i] = grad_f32[i] + self.momentum * entry.m[i];
            }
        } else {
            b.copy_from_slice(&entry.m);
        }

        // 3. Orthogonalize for rank-2 weights; otherwise plain momentum.
        let update = if shape.len() == 2 {
            let (rows, cols) = (shape[0], shape[1]);
            let mut o = newton_schulz(&b, rows, cols, self.ns_iters);
            let scale = (rows.max(cols) as f32).sqrt();
            for v in o.iter_mut() {
                *v *= scale;
            }
            o
        } else {
            b
        };

        // 4. Decoupled weight decay + descent step.
        for i in 0..n {
            master_f32[i] -= self.lr * self.weight_decay * master_f32[i];
            master_f32[i] -= self.lr * update[i];
        }

        let new_master = build_master_tensor(
            policy.master_dtype,
            &shape,
            &master_f32,
            self.rounding,
            entry.step,
            master.device(),
        )?;
        param.replace_backward_storage(Some(new_master));
        // #1082 Phase 2.7: end-of-optimizer-step epoch bump (see AdamW).
        param.bump_epoch();
        Ok(())
    }

    fn reset(&mut self) {
        self.momenta.clear();
    }
}

// ----------------------------------------------------------------------
// Newton-Schulz orthogonalization (gram-space P-accumulator)
// ----------------------------------------------------------------------

/// Newton-Schulz orthogonalization of a row-major `[rows, cols]` matrix
/// `w`. Returns `O ≈ U Vᵀ` (the orthogonal polar factor) flattened
/// row-major, with `‖O‖_F ≈ sqrt(min(rows, cols))`.
///
/// Coefficients `(a, b, c) = (3.4445, -4.7750, 2.0315)`. The iteration
/// `X ← a·X + (b·A + c·A²)·X` (with `A` the gram of the Frobenius-
/// normalized `X`) is reorganized so that each iteration is a `k×k`
/// matrix `M_i = a·I + b·A_i + c·A_i²` (`k = min(rows, cols)`), and the
/// whole product collapses to a single `k×k` accumulator `P`:
/// `O = (P @ W) / ‖W‖_F` when `rows ≤ cols`, or `(W @ P) / ‖W‖_F` when
/// `rows > cols`. Only the initial gram and the final apply touch the
/// large dimension; everything else is `k×k`.
pub fn newton_schulz(w: &[f32], rows: usize, cols: usize, iters: u32) -> Vec<f32> {
    debug_assert_eq!(w.len(), rows * cols);
    let n = rows * cols;
    let frob = w.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if frob == 0.0 {
        return vec![0.0; n];
    }
    let inv_frob = 1.0 / frob;
    // Gram in the smaller dimension. `transpose=false` → rows ≤ cols →
    // gram = W Wᵀ (rows×rows), accumulate P on the left, O = P W.
    // `transpose=true`  → rows >  cols → gram = Wᵀ W (cols×cols),
    // accumulate P on the right, O = W P.
    let transpose = rows > cols;
    let k = rows.min(cols);

    // A0 = gram(W_normalized) = (W Wᵀ or Wᵀ W) * inv_frob².
    let inv_frob2 = inv_frob * inv_frob;
    let mut a = vec![0.0f32; k * k];
    if !transpose {
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for c in 0..cols {
                    s += w[i * cols + c] * w[j * cols + c];
                }
                a[i * k + j] = s * inv_frob2;
            }
        }
    } else {
        for i in 0..k {
            for j in 0..k {
                let mut s = 0.0f32;
                for r in 0..rows {
                    s += w[r * cols + i] * w[r * cols + j];
                }
                a[i * k + j] = s * inv_frob2;
            }
        }
    }

    // P = I_k.
    let mut p = vec![0.0f32; k * k];
    for i in 0..k {
        p[i * k + i] = 1.0;
    }

    let (ca, cb, cc) = (3.4445f32, -4.7750f32, 2.0315f32);
    let mut a2 = vec![0.0f32; k * k];
    let mut m = vec![0.0f32; k * k];
    let mut tmp = vec![0.0f32; k * k];
    for _ in 0..iters {
        // A2 = A @ A.
        matmul_kk(&a, &a, &mut a2, k);
        // M = a*I + b*A + c*A2.
        for i in 0..k {
            for j in 0..k {
                let id = if i == j { 1.0 } else { 0.0 };
                m[i * k + j] = ca * id + cb * a[i * k + j] + cc * a2[i * k + j];
            }
        }
        // Accumulate P: left-multiply (P = M P) when rows ≤ cols,
        // right-multiply (P = P M) when rows > cols.
        if !transpose {
            matmul_kk(&m, &p, &mut tmp, k);
        } else {
            matmul_kk(&p, &m, &mut tmp, k);
        }
        p.copy_from_slice(&tmp);
        // A = M A M  (M symmetric, so M Aᵀ Mᵀ = M A M keeps A symmetric).
        matmul_kk(&m, &a, &mut tmp, k);
        matmul_kk(&tmp, &m, &mut a, k);
    }

    // O = (P @ W) * inv_frob  (rows ≤ cols)  or  (W @ P) * inv_frob.
    let mut o = vec![0.0f32; n];
    if !transpose {
        // O[i,c] = inv_frob * Σ_j P[i,j] W[j,c];  i ∈ [0,rows)=k rows.
        for i in 0..rows {
            for c in 0..cols {
                let mut s = 0.0f32;
                for j in 0..k {
                    s += p[i * k + j] * w[j * cols + c];
                }
                o[i * cols + c] = s * inv_frob;
            }
        }
    } else {
        // O[r,c] = inv_frob * Σ_j W[r,j] P[j,c];  c ∈ [0,cols)=k cols.
        for r in 0..rows {
            for c in 0..cols {
                let mut s = 0.0f32;
                for j in 0..k {
                    s += w[r * cols + j] * p[j * k + c];
                }
                o[r * cols + c] = s * inv_frob;
            }
        }
    }
    o
}

/// `out = a @ b` for row-major `k×k` matrices.
fn matmul_kk(a: &[f32], b: &[f32], out: &mut [f32], k: usize) {
    debug_assert_eq!(a.len(), k * k);
    debug_assert_eq!(b.len(), k * k);
    debug_assert_eq!(out.len(), k * k);
    for i in 0..k {
        for j in 0..k {
            let mut s = 0.0f32;
            for t in 0..k {
                s += a[i * k + t] * b[t * k + j];
            }
            out[i * k + j] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};
    use kiln_tensor::Tensor;

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        Parameter::trainable(
            ForwardStorage::Plain(fwd),
            master,
            AmpPolicy::fp32_reference(),
        )
    }

    #[test]
    fn lion_name_and_construction() {
        let l = Lion::new(1e-4, 0.9, 0.99, 0.01);
        assert_eq!(l.name(), "lion");
        assert_eq!(l.lr, 1e-4);
    }

    #[test]
    fn lion_step_writes_updated_master_with_sign() {
        // Single step: master = [1, 2]; grad = [1, -1]; lr=0.1; β1=0.9.
        // c = 0.9 * 0 + 0.1 * g = [0.1, -0.1]
        // sign(c) = [+1, -1]
        // update: p -= 0.1 * (sign(c) + 0)  → p = [0.9, 2.1]
        let mut l = Lion::new(0.1, 0.9, 0.99, 0.0);
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, -1.0], vec![2]).unwrap();
        l.step(&mut p, &g).unwrap();
        let m = p.backward_storage().unwrap();
        let cpu = m
            .storage()
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .unwrap();
        let values: Vec<f32> = cpu
            .as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((values[0] - 0.9).abs() < 1e-6);
        assert!((values[1] - 2.1).abs() < 1e-6);
    }

    #[test]
    fn lion_ema_state_advances() {
        let mut l = Lion::default_hp();
        let mut p = fresh_param();
        let id = p.tensor_id();
        let g = Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
        assert!(l.ema_for(id).is_none());
        l.step(&mut p, &g).unwrap();
        let m = l.ema_for(id).expect("ema entry");
        assert_eq!(m.step, 1);
        // After step 1 with β2=0.99: m = 0.99*0 + 0.01*g = 0.01*g
        assert!((m.m[0] - 0.001).abs() < 1e-7);
        assert!((m.m[1] - 0.002).abs() < 1e-7);

        l.step(&mut p, &g).unwrap();
        let m2 = l.ema_for(id).unwrap();
        assert_eq!(m2.step, 2);
    }

    #[test]
    fn lion_zero_grad_sign_is_zero() {
        // grad = zeros → c = 0 → sign = 0 → no update from sign term.
        // With weight_decay = 0, master is unchanged.
        let mut l = Lion::new(0.1, 0.9, 0.99, 0.0);
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        l.step(&mut p, &g).unwrap();
        let m = p.backward_storage().unwrap();
        let cpu = m
            .storage()
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .unwrap();
        let values: Vec<f32> = cpu
            .as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(values, vec![1.0, 2.0]);
    }

    #[test]
    fn lion_weight_decay_applies_per_step() {
        // lr=0.1, λ=0.1, grad=zeros. c=0 → sign=0. Update = 0.1*0.1*p = 0.01*p.
        // master = master * (1 - 0.01)
        let mut l = Lion::new(0.1, 0.9, 0.99, 0.1);
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.0f32, 0.0], vec![2]).unwrap();
        l.step(&mut p, &g).unwrap();
        let m = p.backward_storage().unwrap();
        let cpu = m
            .storage()
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .unwrap();
        let values: Vec<f32> = cpu
            .as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert!((values[0] - 0.99).abs() < 1e-6);
        assert!((values[1] - 1.98).abs() < 1e-6);
    }

    #[test]
    fn lion_preserves_tensor_id() {
        let mut l = Lion::default_hp();
        let mut p = fresh_param();
        let id = p.tensor_id();
        let g = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        l.step(&mut p, &g).unwrap();
        assert_eq!(p.tensor_id(), id);
    }

    #[test]
    fn muon_name_and_construction() {
        let m = Muon::new(1e-3, 0.95, true, 5, 0.0);
        assert_eq!(m.name(), "muon");
        assert_eq!(m.momentum, 0.95);
        assert!(m.nesterov);
        assert_eq!(m.ns_iters, 5);
    }

    #[test]
    fn muon_default_hp() {
        let m = Muon::default_hp();
        assert_eq!(m.lr, 2e-2);
        assert_eq!(m.momentum, 0.95);
        assert!(m.nesterov);
        assert_eq!(m.ns_iters, 5);
        assert_eq!(m.weight_decay, 0.0);
    }

    fn fresh_matrix_param() -> Parameter {
        // 2x2 identity-ish matrix.
        let fwd = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        Parameter::trainable(
            ForwardStorage::Plain(fwd),
            master,
            AmpPolicy::fp32_reference(),
        )
    }

    fn read_master(p: &Parameter) -> Vec<f32> {
        let cpu = p
            .backward_storage()
            .unwrap()
            .storage()
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .unwrap();
        cpu.as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn muon_step_rank1_falls_back_to_momentum() {
        // For non-matrix shapes Muon = SGD with (Nesterov) momentum.
        // m_0 = 0; step 1 with grad=ones → m = ones.
        // nesterov b = g + momentum*m = 1 + 0.9*1 = 1.9 → master -= lr*1.9.
        let mut opt = Muon::new(0.1, 0.9, true, 5, 0.0);
        let mut p = fresh_param(); // shape [2]
        let g = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let v = read_master(&p);
        assert!((v[0] - (1.0 - 0.1 * 1.9)).abs() < 1e-6, "got {}", v[0]);
        assert!((v[1] - (2.0 - 0.1 * 1.9)).abs() < 1e-6, "got {}", v[1]);
    }

    #[test]
    fn muon_step_rank1_heavy_ball_when_not_nesterov() {
        // Without Nesterov, b = m = ones → master -= lr*1.
        let mut opt = Muon::new(0.1, 0.9, false, 5, 0.0);
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let v = read_master(&p);
        assert!((v[0] - 0.9).abs() < 1e-6, "got {}", v[0]);
        assert!((v[1] - 1.9).abs() < 1e-6, "got {}", v[1]);
    }

    #[test]
    fn muon_step_rank2_runs_newton_schulz() {
        // For a matrix grad, the orthogonalization changes the update
        // direction. Verify the step runs, produces finite outputs, and
        // advances the step counter.
        let mut opt = Muon::default_hp();
        let mut p = fresh_matrix_param();
        let g = Tensor::from_slice(&[1.0f32, 0.5, 0.5, 1.0], vec![2, 2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let v = read_master(&p);
        assert_eq!(v.len(), 4);
        for x in v {
            assert!(x.is_finite(), "Muon produced non-finite master: {x}");
        }
        let state = opt.momentum_for(p.tensor_id()).unwrap();
        assert_eq!(state.step, 1);
    }

    #[test]
    fn muon_preserves_tensor_id() {
        let mut opt = Muon::default_hp();
        let mut p = fresh_matrix_param();
        let id_before = p.tensor_id();
        let g = Tensor::from_slice(&[0.1f32, 0.1, 0.1, 0.1], vec![2, 2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(p.tensor_id(), id_before);
    }

    #[test]
    fn muon_multi_step_advances_state() {
        let mut opt = Muon::default_hp();
        let mut p = fresh_matrix_param();
        let g = Tensor::from_slice(&[1.0f32, 0.5, 0.5, 1.0], vec![2, 2]).unwrap();
        for _ in 0..3 {
            opt.step(&mut p, &g).unwrap();
        }
        let state = opt.momentum_for(p.tensor_id()).unwrap();
        assert_eq!(state.step, 3);
    }

    // ---- Newton-Schulz P-accumulator correctness ----

    /// Singular values of a row-major `[rows, cols]` matrix via the
    /// eigenvalues of its (small) gram, by symmetric Jacobi.
    fn singular_values(m: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let k = rows.min(cols);
        // gram = the smaller of M Mᵀ / Mᵀ M (k×k, symmetric PSD).
        let mut g = vec![0.0f64; k * k];
        if rows <= cols {
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0f64;
                    for c in 0..cols {
                        s += m[i * cols + c] as f64 * m[j * cols + c] as f64;
                    }
                    g[i * k + j] = s;
                }
            }
        } else {
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0f64;
                    for r in 0..rows {
                        s += m[r * cols + i] as f64 * m[r * cols + j] as f64;
                    }
                    g[i * k + j] = s;
                }
            }
        }
        // Jacobi eigenvalue iteration on the symmetric k×k gram.
        for _ in 0..100 {
            // find largest off-diagonal
            let (mut p, mut q, mut max) = (0, 1, 0.0f64);
            for i in 0..k {
                for j in (i + 1)..k {
                    if g[i * k + j].abs() > max {
                        max = g[i * k + j].abs();
                        p = i;
                        q = j;
                    }
                }
            }
            if max < 1e-12 {
                break;
            }
            let app = g[p * k + p];
            let aqq = g[q * k + q];
            let apq = g[p * k + q];
            let theta = 0.5 * (aqq - app).atan2(2.0 * apq);
            // standard rotation; recompute via theta = 0.5*atan2(2apq, app-aqq)
            let phi = 0.5 * (2.0 * apq).atan2(app - aqq);
            let (c, s) = (phi.cos(), phi.sin());
            let _ = theta;
            let mut gn = g.clone();
            for i in 0..k {
                let gip = g[i * k + p];
                let giq = g[i * k + q];
                gn[i * k + p] = c * gip + s * giq;
                gn[i * k + q] = -s * gip + c * giq;
            }
            g.copy_from_slice(&gn);
            let mut gn2 = g.clone();
            for j in 0..k {
                let gpj = g[p * k + j];
                let gqj = g[q * k + j];
                gn2[p * k + j] = c * gpj + s * gqj;
                gn2[q * k + j] = -s * gpj + c * gqj;
            }
            g.copy_from_slice(&gn2);
        }
        (0..k)
            .map(|i| (g[i * k + i].max(0.0)).sqrt() as f32)
            .collect()
    }

    #[test]
    fn newton_schulz_zero_matrix_is_zero() {
        let z = vec![0.0f32; 9];
        let out = newton_schulz(&z, 3, 3, 5);
        for v in out {
            assert_eq!(v, 0.0);
        }
    }

    /// Naive *direct* Newton-Schulz (the un-factored `X ← a·X + (b·A +
    /// c·A²)·X` iteration), matching the orientation choice of the
    /// production [`newton_schulz`]. The production version factors this
    /// into a `k×k` P-accumulator; this reference recomputes it the slow
    /// way so the two can be cross-checked.
    fn newton_schulz_direct(w: &[f32], rows: usize, cols: usize, iters: u32) -> Vec<f32> {
        let frob = w.iter().map(|&v| v * v).sum::<f32>().sqrt();
        if frob == 0.0 {
            return vec![0.0; rows * cols];
        }
        let transpose = rows > cols; // match newton_schulz orientation
        let k = rows.min(cols);
        let mut x: Vec<f32> = w.iter().map(|&v| v / frob).collect();
        let (ca, cb, cc) = (3.4445f32, -4.7750f32, 2.0315f32);
        for _ in 0..iters {
            // A = gram (k×k): X Xᵀ (rows≤cols) or Xᵀ X (rows>cols).
            let mut a = vec![0.0f32; k * k];
            if !transpose {
                for i in 0..k {
                    for j in 0..k {
                        let mut s = 0.0;
                        for c in 0..cols {
                            s += x[i * cols + c] * x[j * cols + c];
                        }
                        a[i * k + j] = s;
                    }
                }
            } else {
                for i in 0..k {
                    for j in 0..k {
                        let mut s = 0.0;
                        for r in 0..rows {
                            s += x[r * cols + i] * x[r * cols + j];
                        }
                        a[i * k + j] = s;
                    }
                }
            }
            // AA = A@A; Q = b*A + c*AA (k×k).
            let mut q = vec![0.0f32; k * k];
            for i in 0..k {
                for j in 0..k {
                    let mut s = 0.0;
                    for t in 0..k {
                        s += a[i * k + t] * a[t * k + j];
                    }
                    q[i * k + j] = cb * a[i * k + j] + cc * s;
                }
            }
            // X ← a*X + (Q@X if rows≤cols else X@Q).
            let mut xn = vec![0.0f32; rows * cols];
            if !transpose {
                for i in 0..rows {
                    for c in 0..cols {
                        let mut s = 0.0;
                        for t in 0..k {
                            s += q[i * k + t] * x[t * cols + c];
                        }
                        xn[i * cols + c] = ca * x[i * cols + c] + s;
                    }
                }
            } else {
                for r in 0..rows {
                    for c in 0..cols {
                        let mut s = 0.0;
                        for t in 0..k {
                            s += x[r * cols + t] * q[t * k + c];
                        }
                        xn[r * cols + c] = ca * x[r * cols + c] + s;
                    }
                }
            }
            x = xn;
        }
        x
    }

    #[test]
    fn newton_schulz_matches_direct_iteration() {
        // The P-accumulator factoring must reproduce the naive direct
        // iteration bit-closely across square / wide / tall shapes.
        let cases: &[(usize, usize)] = &[(2, 2), (3, 3), (2, 5), (5, 2), (4, 6), (6, 4)];
        for &(rows, cols) in cases {
            let w: Vec<f32> = (0..rows * cols)
                .map(|i| (i as f32 * 0.37).sin() - 0.2 * (i as f32 * 0.11).cos())
                .collect();
            let got = newton_schulz(&w, rows, cols, 5);
            let want = newton_schulz_direct(&w, rows, cols, 5);
            for i in 0..rows * cols {
                assert!(
                    (got[i] - want[i]).abs() < 1e-3,
                    "P-accumulator != direct at {i} for {rows}x{cols}: {} vs {}",
                    got[i],
                    want[i]
                );
            }
        }
    }

    #[test]
    fn newton_schulz_improves_conditioning() {
        // The polar-factor iteration pulls the singular values into a
        // tight unit band (these Keller-Jordan coefficients land them in
        // ~[0.7, 1.0] after 5 iters — NOT exactly 1) and reduces the
        // condition number vs the input.
        let m = vec![0.3f32, 0.0, 0.0, 0.5];
        let in_sv = singular_values(&m, 2, 2);
        let out = newton_schulz(&m, 2, 2, 5);
        let out_sv = singular_values(&out, 2, 2);
        for &s in &out_sv {
            assert!(
                (0.6..=1.1).contains(&s),
                "singular value {s} outside unit band"
            );
        }
        let cond = |sv: &[f32]| {
            sv.iter().cloned().fold(0.0f32, f32::max)
                / sv.iter().cloned().fold(f32::INFINITY, f32::min)
        };
        assert!(
            cond(&out_sv) < cond(&in_sv),
            "conditioning not improved: in {} out {}",
            cond(&in_sv),
            cond(&out_sv)
        );
    }

    #[test]
    fn newton_schulz_short_fat_and_tall_skinny_are_transposes() {
        // NS(Wᵀ) == NS(W)ᵀ : the orthogonalizer commutes with transpose.
        // W is [2,3]; Wt is [3,2].
        let w = vec![0.6f32, 0.1, -0.2, 0.3, 0.5, 0.4];
        let wt = vec![0.6f32, 0.3, 0.1, 0.5, -0.2, 0.4];
        let o = newton_schulz(&w, 2, 3, 5);
        let ot = newton_schulz(&wt, 3, 2, 5);
        // o is [2,3], ot is [3,2]; compare o[i,j] == ot[j,i].
        for i in 0..2 {
            for j in 0..3 {
                assert!(
                    (o[i * 3 + j] - ot[j * 2 + i]).abs() < 1e-4,
                    "transpose mismatch at ({i},{j}): {} vs {}",
                    o[i * 3 + j],
                    ot[j * 2 + i]
                );
            }
        }
    }

    #[test]
    fn muon_rms_scale_normalizes_update_magnitude() {
        // With Nesterov off and momentum 0 the update direction for a
        // rank-2 weight is NS(g)*sqrt(max(rows,cols)). Its RMS per
        // element should be ≈ 1 (the RMS-matching scale), independent of
        // shape. Check that the master moved by ≈ lr in RMS.
        let mut opt = Muon::new(1.0, 0.0, false, 6, 0.0);
        // wide 2x8 matrix of ones-ish.
        let (rows, cols) = (2usize, 8usize);
        let init: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.13).sin()).collect();
        let fwd = Tensor::from_slice(&init, vec![rows, cols]).unwrap();
        let master = Tensor::from_slice(&init, vec![rows, cols]).unwrap();
        let mut p = Parameter::trainable(
            ForwardStorage::Plain(fwd),
            master,
            AmpPolicy::fp32_reference(),
        );
        let g: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.27).cos()).collect();
        let gt = Tensor::from_slice(&g, vec![rows, cols]).unwrap();
        opt.step(&mut p, &gt).unwrap();
        let after = read_master(&p);
        let mut ss = 0.0f32;
        for i in 0..rows * cols {
            let d = (after[i] - init[i]).abs();
            ss += d * d;
        }
        let rms = (ss / (rows * cols) as f32).sqrt();
        // lr=1, update RMS ≈ 1, so move RMS ≈ 1 within tolerance.
        assert!((rms - 1.0).abs() < 0.25, "update RMS {rms} not ≈ 1");
    }

    #[test]
    fn muon_weight_decay_shrinks_master() {
        // grad=zeros so momentum stays 0, b=0, update=0; only decoupled
        // WD acts: master *= (1 - lr*wd).
        let mut opt = Muon::new(0.1, 0.9, true, 5, 0.5);
        let mut p = fresh_matrix_param();
        let g = Tensor::from_slice(&[0.0f32; 4], vec![2, 2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let v = read_master(&p);
        // master was [1,0,0,1]; (1 - 0.1*0.5) = 0.95.
        assert!((v[0] - 0.95).abs() < 1e-6);
        assert!((v[3] - 0.95).abs() < 1e-6);
        assert!(v[1].abs() < 1e-6);
    }

    #[test]
    fn lion_reset_clears_ema() {
        let mut l = Lion::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        l.step(&mut p, &g).unwrap();
        assert_eq!(l.parameter_count(), 1);
        l.reset();
        assert_eq!(l.parameter_count(), 0);
    }

    #[test]
    fn muon_reset_clears_state() {
        let mut opt = Muon::default_hp();
        let mut p = fresh_matrix_param();
        let g = Tensor::from_slice(&[0.1f32; 4], vec![2, 2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.parameter_count(), 1);
        opt.reset();
        assert_eq!(opt.parameter_count(), 0);
    }
}
