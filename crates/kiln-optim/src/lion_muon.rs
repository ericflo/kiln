//! `Lion` + `Muon` — Phase 6.5 issue-menu variants, today as stubs
//! that demonstrate the trait shape generalizes.
//!
//! Per the Phase 6.5 issue bullet:
//!
//! > `kiln-optim` crate with `OptimStep` trait: AdamW, SGD, Lion, Muon
//!
//! `AdamW` (Phase 6.5) and `Sgd` (Phase 6.5.1) are the two concrete
//! impls. `Lion` (Chen et al. 2023, "Symbolic Discovery of Optimization
//! Algorithms") and `Muon` (Bernstein-Newhouse 2024, "Old optimizer,
//! new norm") are scaffolds — they impl `OptimStep` but `step` returns
//! `Err(StepError::Tensor(_))` with a "not yet implemented" message.
//!
//! These stubs exist so:
//!
//! 1. The OptimStep trait API surface is validated against three
//!    + one optimizers (AdamW state-rich, SGD light, Lion compact-state,
//!    Muon momentum-orthogonalized) rather than just two.
//! 2. Downstream callers can write `match opt_kind { OptimKind::Lion => ... }`
//!    today and not have to revisit the dispatch site when Phase 6.5.x
//!    lands the real implementations.

use std::collections::HashMap;

use kiln_param::Parameter;
use kiln_tensor::{CpuStorage, DType, Layout, Storage, Tensor, TensorId};

use crate::{OptimStep, StepError};

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
        Lion::new(/*lr=*/ 1e-4, /*β1=*/ 0.9, /*β2=*/ 0.99, /*λ=*/ 0.0)
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

        // Write updated master back to the Parameter.
        let new_master = build_master_tensor(policy.master_dtype, master.shape(), &master_f32)?;
        param.replace_backward_storage(Some(new_master));
        Ok(())
    }

    fn reset(&mut self) {
        self.ema.clear();
    }
}

fn read_master_to_f32(t: &Tensor) -> Result<Vec<f32>, StepError> {
    let cpu = t
        .storage()
        .as_any()
        .downcast_ref::<CpuStorage>()
        .ok_or_else(|| {
            StepError::Tensor(kiln_tensor::Error::from_str(
                "Lion CPU: tensor storage must be CpuStorage",
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
                    half::bf16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        DType::F16 => {
            for i in 0..n {
                out.push(
                    half::f16::from_le_bytes(bytes[i * 2..i * 2 + 2].try_into().unwrap())
                        .to_f32(),
                );
            }
        }
        other => {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Lion CPU: unsupported dtype {other}"
            ))))
        }
    }
    Ok(out)
}

fn build_master_tensor(
    dtype: DType,
    shape: &[usize],
    values: &[f32],
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
        DType::BF16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::bf16::from_f32(v).to_le_bytes());
            }
        }
        DType::F16 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 2..i * 2 + 2]
                    .copy_from_slice(&half::f16::from_f32(v).to_le_bytes());
            }
        }
        other => {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "Lion CPU: unsupported master dtype {other}"
            ))))
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes).map_err(StepError::Tensor)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(storage, Layout::contiguous(shape.to_vec()), TensorId::next())
        .map_err(StepError::Tensor)
}

/// Muon (Bernstein-Newhouse 2024). Momentum-orthogonalized SGD —
/// projects the momentum onto the closest orthogonal matrix via
/// Newton-Schulz iteration before each step.
///
/// # Algorithm (rank-2 weights)
///
/// ```text
/// m_t = momentum * m_{t-1} + g_t            # heavy-ball momentum
/// U_t = newton_schulz(m_t, iters)           # ≈ polar U factor of m_t
/// p_t = p_{t-1} - lr * U_t
/// ```
///
/// Newton-Schulz iteration (paper coefficients a=3.4445, b=-4.7750,
/// c=2.0315) approximates the polar decomposition for 5 iterations.
///
/// # Non-matrix parameters
///
/// Muon is defined only for **matrix-shaped weights** (rank-2).
/// Bias vectors / embeddings / scalar weights fall back to plain
/// SGD-with-momentum (skip the orthogonalization).
#[derive(Debug, Default)]
pub struct Muon {
    pub lr: f32,
    pub momentum: f32,
    /// Number of Newton-Schulz iterations. Paper uses 5.
    pub ns_iters: u32,
    /// Per-parameter momentum buffer keyed on TensorId.
    momenta: HashMap<TensorId, MuonState>,
}

#[derive(Debug, Clone)]
pub struct MuonState {
    pub m: Vec<f32>,
    pub step: u64,
}

impl Muon {
    pub fn new(lr: f32, momentum: f32, ns_iters: u32) -> Self {
        Muon {
            lr,
            momentum,
            ns_iters,
            momenta: HashMap::new(),
        }
    }

    pub fn default_hp() -> Self {
        Muon::new(/*lr=*/ 2e-2, /*momentum=*/ 0.95, /*ns_iters=*/ 5)
    }

    pub fn momentum_for(&self, id: TensorId) -> Option<&MuonState> {
        self.momenta.get(&id)
    }

    pub fn parameter_count(&self) -> usize {
        self.momenta.len()
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
        // Heavy-ball momentum: m = β m + g.
        for i in 0..n {
            entry.m[i] = self.momentum * entry.m[i] + grad_f32[i];
        }

        // Orthogonalize for rank-2 weights; otherwise plain SGD with
        // the heavy-ball momentum.
        let update = if shape.len() == 2 {
            newton_schulz(&entry.m, shape[0], shape[1], self.ns_iters)
        } else {
            entry.m.clone()
        };

        for i in 0..n {
            master_f32[i] -= self.lr * update[i];
        }

        let new_master = build_master_tensor(policy.master_dtype, &shape, &master_f32)?;
        param.replace_backward_storage(Some(new_master));
        Ok(())
    }

    fn reset(&mut self) {
        self.momenta.clear();
    }
}

/// Newton-Schulz orthogonalization for a row-major `[rows, cols]`
/// matrix flattened into `data`. Paper coefficients
/// `(a, b, c) = (3.4445, -4.7750, 2.0315)`. Returns the orthogonalized
/// matrix flattened back in row-major.
fn newton_schulz(data: &[f32], rows: usize, cols: usize, iters: u32) -> Vec<f32> {
    debug_assert_eq!(data.len(), rows * cols);
    // 1. Frobenius-normalize.
    let frob: f32 = data.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if frob == 0.0 {
        return data.to_vec();
    }
    let mut x: Vec<f32> = data.iter().map(|&v| v / frob).collect();

    let (a, b, c) = (3.4445_f32, -4.7750_f32, 2.0315_f32);
    let mut a_buf = vec![0.0f32; rows.min(cols) * rows.min(cols)];
    let mut aa = a_buf.clone();
    let mut aa_x = vec![0.0f32; rows * cols];

    // Convention: if rows >= cols, use X^T X (cols x cols); else X X^T.
    // This is the paper's recipe — operate on the smaller of the two
    // gram matrices.
    let transpose = rows >= cols;
    let k = if transpose { cols } else { rows };

    for _ in 0..iters {
        // A = X^T X (when transpose=true, shape [cols, cols])
        //     or X X^T (when transpose=false, shape [rows, rows])
        gram(&x, rows, cols, transpose, &mut a_buf);
        // AA = A @ A
        matmul_square(&a_buf, k, &mut aa);
        // Q = b * A + c * AA  (k x k)
        for i in 0..k * k {
            aa[i] = b * a_buf[i] + c * aa[i];
        }
        // X_new = a * X + Q @ X (when transpose=true)  OR  a * X + X @ Q
        // Depending on which side the gram was computed.
        if transpose {
            // Q is cols x cols → multiply on the right: X = a*X + X @ Q
            matmul_rhs(&x, &aa, rows, cols, &mut aa_x);
        } else {
            // Q is rows x rows → multiply on the left: X = a*X + Q @ X
            matmul_lhs(&aa, &x, rows, cols, &mut aa_x);
        }
        for i in 0..rows * cols {
            x[i] = a * x[i] + aa_x[i];
        }
    }
    x
}

/// `A = X^T X` (when transpose) or `X X^T` (otherwise).
fn gram(x: &[f32], rows: usize, cols: usize, transpose: bool, out: &mut [f32]) {
    if transpose {
        // out [cols, cols] = X^T [cols, rows] @ X [rows, cols]
        debug_assert_eq!(out.len(), cols * cols);
        for i in 0..cols {
            for j in 0..cols {
                let mut s = 0.0_f32;
                for r in 0..rows {
                    s += x[r * cols + i] * x[r * cols + j];
                }
                out[i * cols + j] = s;
            }
        }
    } else {
        // out [rows, rows] = X [rows, cols] @ X^T [cols, rows]
        debug_assert_eq!(out.len(), rows * rows);
        for i in 0..rows {
            for j in 0..rows {
                let mut s = 0.0_f32;
                for c in 0..cols {
                    s += x[i * cols + c] * x[j * cols + c];
                }
                out[i * rows + j] = s;
            }
        }
    }
}

fn matmul_square(a: &[f32], n: usize, out: &mut [f32]) {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(out.len(), n * n);
    for i in 0..n {
        for j in 0..n {
            let mut s = 0.0_f32;
            for k in 0..n {
                s += a[i * n + k] * a[k * n + j];
            }
            out[i * n + j] = s;
        }
    }
}

/// `out = x @ q` where x is `[rows, cols]`, q is `[cols, cols]`.
fn matmul_rhs(x: &[f32], q: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0_f32;
            for k in 0..cols {
                s += x[r * cols + k] * q[k * cols + c];
            }
            out[r * cols + c] = s;
        }
    }
}

/// `out = q @ x` where q is `[rows, rows]`, x is `[rows, cols]`.
fn matmul_lhs(q: &[f32], x: &[f32], rows: usize, cols: usize, out: &mut [f32]) {
    debug_assert_eq!(out.len(), rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            let mut s = 0.0_f32;
            for k in 0..rows {
                s += q[r * rows + k] * x[k * cols + c];
            }
            out[r * cols + c] = s;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};
    use kiln_tensor::{DType, Tensor};

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0], vec![2]).unwrap();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, AmpPolicy::fp32_reference())
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
        let m = Muon::new(1e-3, 0.95, 5);
        assert_eq!(m.name(), "muon");
        assert_eq!(m.momentum, 0.95);
        assert_eq!(m.ns_iters, 5);
    }

    fn fresh_matrix_param() -> Parameter {
        // 2x2 identity-ish matrix.
        let fwd = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, AmpPolicy::fp32_reference())
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
    fn muon_step_rank1_falls_back_to_sgd_momentum() {
        // For non-matrix shapes Muon = SGD with heavy-ball momentum.
        // m_0 = 0; step 1 with grad=ones → m = ones → master -= lr * ones.
        let mut opt = Muon::new(0.1, 0.9, 5);
        let mut p = fresh_param(); // shape [2]
        let g = Tensor::from_slice(&[1.0f32, 1.0], vec![2]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let v = read_master(&p);
        assert!((v[0] - 0.9).abs() < 1e-6, "got {}", v[0]);
        assert!((v[1] - 1.9).abs() < 1e-6, "got {}", v[1]);
    }

    #[test]
    fn muon_step_rank2_runs_newton_schulz() {
        // For a matrix grad, the orthogonalization changes the update
        // direction. We just verify the step runs without error,
        // produces finite outputs, and that the step counter advances.
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

    #[test]
    fn newton_schulz_identity_stays_orthogonal_ish() {
        // The identity matrix is already orthogonal; NS should leave
        // it (approximately) at the same shape.
        let i = vec![1.0f32, 0.0, 0.0, 1.0];
        let out = newton_schulz(&i, 2, 2, 5);
        // After NS, ||out||_F ≈ √2 (orthogonal 2x2 has Frobenius norm √2).
        let frob = out.iter().map(|&v| v * v).sum::<f32>().sqrt();
        assert!(
            (frob - 2.0_f32.sqrt()).abs() < 0.1,
            "frob = {frob}, expected ~√2"
        );
    }

    #[test]
    fn newton_schulz_zero_matrix_is_zero() {
        let z = vec![0.0f32; 9];
        let out = newton_schulz(&z, 3, 3, 5);
        for v in out {
            assert_eq!(v, 0.0);
        }
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
