//! `AdamW` — the canonical CPU reference optimizer.
//!
//! Standard AdamW with decoupled weight decay (Loshchilov & Hutter
//! 2017). Runs on `Parameter::backward_storage` (the master copy).
//!
//! # Backend boundary
//!
//! This type is the portable CPU reference and host checkpoint-state path.
//! Production accelerator updates use the backend `OptimizerBackend` fused
//! kernels and the precision contract in `docs/training/NATIVE_SFT_PROFILE.md`.
//!
//! # Determinism
//!
//! CPU arithmetic is deterministic for fixed inputs. Parameters, gradients,
//! and moments are promoted to F32 for the update, then the master parameter is
//! written according to the configured rounding policy. Production fused
//! kernels are qualified independently against the committed source-pinned
//! PyTorch trajectories.

use std::collections::HashMap;

use kiln_param::Parameter;
use kiln_tensor::{CpuStorage, DType, Tensor, TensorId};

use crate::{MomentLocation, OptimStep, StepError, StochasticRoundingPolicy};

/// AdamW hyperparameters. Matches PyTorch's defaults.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdamWHyperparameters {
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    /// Decoupled weight decay coefficient.
    pub weight_decay: f32,
}

impl Default for AdamWHyperparameters {
    /// PyTorch / Adam paper defaults.
    fn default() -> Self {
        AdamWHyperparameters {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
        }
    }
}

/// Per-parameter AdamW running moments. Always FP32 regardless of
/// master dtype.
#[derive(Debug, Clone)]
pub struct AdamWMoments {
    pub m: Vec<f32>, // first-moment estimate
    pub v: Vec<f32>, // second-moment estimate
    pub step: u64,   // step counter for bias correction
    pub location: MomentLocation,
}

/// CPU AdamW reference impl.
///
/// Stores moments keyed on `kiln_tensor::TensorId` per anti-pattern 11
/// (the id is stable across storage-variant transitions).
#[derive(Debug)]
pub struct AdamW {
    hp: AdamWHyperparameters,
    rounding: StochasticRoundingPolicy,
    moments: HashMap<TensorId, AdamWMoments>,
}

impl AdamW {
    /// Construct AdamW with explicit hyperparameters.
    pub fn new(hp: AdamWHyperparameters) -> Self {
        Self::new_with_rounding(hp, StochasticRoundingPolicy::RoundToNearest)
    }

    /// Construct AdamW with an explicit programmatic rounding policy.
    pub fn new_with_rounding(hp: AdamWHyperparameters, rounding: StochasticRoundingPolicy) -> Self {
        AdamW {
            hp,
            rounding,
            moments: HashMap::new(),
        }
    }

    /// Construct AdamW with default hyperparameters.
    pub fn default_hp() -> Self {
        AdamW::new(AdamWHyperparameters::default())
    }

    /// Borrow the moments for a parameter id (None if step has not
    /// been called for this id yet).
    pub fn moments(&self, id: TensorId) -> Option<&AdamWMoments> {
        self.moments.get(&id)
    }

    /// Number of parameters this AdamW instance has stepped at least once.
    pub fn parameter_count(&self) -> usize {
        self.moments.len()
    }

    /// Rounding policy captured when this optimizer was constructed.
    pub fn rounding_policy(&self) -> StochasticRoundingPolicy {
        self.rounding
    }

    /// Install validated moments for a checkpoint-restored parameter.
    ///
    /// Checkpoint formats key state by stable parameter name. After a process
    /// restart the caller resolves that name to the new process-local
    /// [`TensorId`] and installs the exact FP32 moment vectors here.
    pub fn restore_moments(
        &mut self,
        id: TensorId,
        moments: AdamWMoments,
    ) -> Result<(), StepError> {
        if moments.m.len() != moments.v.len() {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW: restored m/v lengths differ ({} != {})",
                moments.m.len(),
                moments.v.len()
            ))));
        }
        if moments
            .m
            .iter()
            .chain(&moments.v)
            .any(|value| !value.is_finite())
        {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(
                "AdamW: restored moments contain non-finite values".to_string(),
            )));
        }
        self.moments.insert(id, moments);
        Ok(())
    }
}

impl Default for AdamW {
    fn default() -> Self {
        AdamW::default_hp()
    }
}

impl OptimStep for AdamW {
    fn name(&self) -> &'static str {
        "adamw"
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

        // Read AMP policy. AdamW master is the `master_dtype` slot.
        let policy = param.amp_policy();
        if grad.dtype() != policy.backward_compute_dtype {
            return Err(StepError::GradDtypeMismatch {
                grad_dtype: grad.dtype(),
                policy_dtype: policy.backward_compute_dtype,
            });
        }
        let master_dtype = policy.master_dtype;
        if !matches!(master_dtype, DType::F32 | DType::BF16 | DType::F16) {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW: master dtype must be F32/BF16/F16, got {master_dtype}"
            ))));
        }

        // Promote grad + master to F32 for the update.
        let n = master.element_count();
        let mut master_f32 = read_to_f32(&master)?;
        let grad_f32 = read_to_f32(grad)?;

        // Get-or-init moments.
        let entry = self
            .moments
            .entry(param.tensor_id())
            .or_insert_with(|| AdamWMoments {
                m: vec![0.0; n],
                v: vec![0.0; n],
                step: 0,
                location: MomentLocation::Device,
            });
        if entry.m.len() != n {
            return Err(StepError::Tensor(kiln_tensor::Error::Msg(format!(
                "AdamW: moments shape ({}) drifted from master ({})",
                entry.m.len(),
                n
            ))));
        }
        entry.step += 1;
        let step_count = entry.step;
        let step = step_count as f32;
        let beta1 = self.hp.beta1;
        let beta2 = self.hp.beta2;
        let bc1 = 1.0 - beta1.powf(step);
        let bc2 = 1.0 - beta2.powf(step);

        // Apply AdamW step elementwise.
        for i in 0..n {
            let g = grad_f32[i];
            entry.m[i] = beta1 * entry.m[i] + (1.0 - beta1) * g;
            entry.v[i] = beta2 * entry.v[i] + (1.0 - beta2) * g * g;
            let m_hat = entry.m[i] / bc1;
            let v_hat = entry.v[i] / bc2;
            let update = self.hp.lr * (m_hat / (v_hat.sqrt() + self.hp.eps));
            // Decoupled weight decay.
            master_f32[i] -= self.hp.lr * self.hp.weight_decay * master_f32[i];
            master_f32[i] -= update;
        }

        // Build a fresh master Tensor and swap it into the
        // Parameter. Preserves `param.tensor_id()` per anti-pattern
        // 11 — `self.moments` keyed on `tensor_id` survives the swap.
        //
        // #1082 box L416: when the master is BF16 and the policy is
        // `Stochastic` (only through an explicit programmatic policy), the f32→bf16
        // master write rounds stochastically — round up/down with
        // probability proportional to the truncated mantissa — so the
        // update is unbiased in expectation under small learning rates
        // (round-to-nearest otherwise drops every sub-ULP update). The
        // per-parameter `step_count` varies the per-element RNG so
        // successive steps don't reuse a rounding pattern; the policy
        // seed keeps a given run reproducible.
        let new_master = build_master_tensor(
            master_dtype,
            master.shape(),
            &master_f32,
            self.rounding,
            step_count,
        )?;
        // `build_master_tensor` produces a CPU tensor; move it back to the
        // master's original device (e.g. Metal) so the next forward reads the
        // updated weights on-device. No-op when the master already lived on CPU.
        let new_master = if matches!(master.device(), kiln_tensor::Device::Cpu) {
            new_master
        } else {
            new_master
                .to_device(master.device())
                .map_err(StepError::Tensor)?
        };
        param.replace_backward_storage(Some(new_master));
        // #1082 Phase 2.7: end-of-optimizer-step epoch bump. A serving
        // thread polling `Parameter::version_handle` observes this as
        // the signal that the master advanced (forward view now stale).
        param.bump_epoch();
        Ok(())
    }

    fn reset(&mut self) {
        self.moments.clear();
    }
}

// ----------------------------------------------------------------------
// CPU helpers
// ----------------------------------------------------------------------

/// Stochastically round an `f32` to a `bf16` bit pattern.
///
/// `bf16` is the top 16 bits of an `f32`; round-to-nearest truncates
/// the low 16 mantissa bits and rounds, which discards every update
/// smaller than half a `bf16` ULP. Stochastic rounding instead adds a
/// uniform 16-bit random value to the full bit pattern before
/// truncating: the carry into bit 16 (a round-*up*) then fires with
/// probability `dropped_bits / 2^16`, i.e. proportional to the
/// truncated mantissa. This rounds the *magnitude* up or down with the
/// correct probability for both signs (the sign bit is untouched and
/// truncation always moves toward zero), so the result is unbiased in
/// expectation. The carry propagates from mantissa into exponent
/// naturally (`wrapping_add`); NaN is passed through round-to-nearest
/// so its payload isn't perturbed.
#[inline]
fn f32_to_bf16_stochastic_bits(v: f32, r: u16) -> u16 {
    let bits = v.to_bits();
    // NaN (exp all ones, non-zero mantissa): don't perturb.
    if (bits & 0x7fff_ffff) > 0x7f80_0000 {
        return half::bf16::from_f32(v).to_bits();
    }
    let rounded = bits.wrapping_add(r as u32);
    (rounded >> 16) as u16
}

/// Deterministic, reproducible per-element uniform 16-bit value from
/// `(seed, step, idx)` via a splitmix64 finalizer. Same `(seed, step,
/// idx)` always yields the same draw, so a run is reproducible at a
/// fixed policy seed; varying `step` decorrelates successive optimizer
/// steps and `idx` decorrelates elements within a step.
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

fn build_master_tensor(
    dtype: DType,
    shape: &[usize],
    values: &[f32],
    rounding: StochasticRoundingPolicy,
    step: u64,
) -> Result<Tensor, StepError> {
    use kiln_tensor::{Layout, Storage, TensorId};
    use std::sync::Arc;
    let per = dtype.size_in_bytes();
    let mut bytes = vec![0u8; values.len() * per];
    match dtype {
        DType::F32 => {
            for (i, &v) in values.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        // #1082 box L416: stochastic rounding applies to the BF16
        // master write (the box is scoped to "BF16 master updates").
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
                "AdamW CPU: unsupported master dtype {other}"
            ))));
        }
    }
    let cpu = CpuStorage::from_bytes(dtype, bytes).map_err(StepError::Tensor)?;
    let storage: Storage = Arc::new(cpu);
    Tensor::from_parts(
        storage,
        Layout::contiguous(shape.to_vec()),
        TensorId::next(),
    )
    .map_err(StepError::Tensor)
}

fn read_to_f32(t: &Tensor) -> Result<Vec<f32>, StepError> {
    // The CPU reference step reads host bytes. Device-resident tensors
    // (CUDA / Metal grads + masters, e.g. the Metal tape-authoritative SFT
    // path) are D2H-copied first — the on-device kernel fast path
    // (`dispatch_adamw_step`) handles the resident case; this host fallback
    // is the portable reference.
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
                "AdamW CPU: tensor storage must be CpuStorage on CPU",
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
                "AdamW CPU: unsupported dtype {other}"
            ))));
        }
    }
    Ok(out)
}

// (Phase 6.5.3) `write_from_f32` replaced by `build_master_tensor` +
// `Parameter::replace_backward_storage`. The CUDA/Metal/Vulkan paths
// will use their own backend-specific `slice_mut()` / `buffer_mut()`
// when those land in Phase 2.x.

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_param::{AmpPolicy, ForwardStorage};
    use kiln_tensor::{DType, Tensor};

    // ---- #1082 box L416: BF16 stochastic rounding ----

    /// The two bf16 grid points bracketing `v` (truncate, and truncate+1).
    fn bf16_neighbors(v: f32) -> (f32, f32) {
        let top = (v.to_bits() >> 16) as u16;
        (
            half::bf16::from_bits(top).to_f32(),
            half::bf16::from_bits(top.wrapping_add(1)).to_f32(),
        )
    }

    #[test]
    fn stochastic_bf16_rounds_only_to_the_two_neighbors() {
        let v = 0.1f32; // not bf16-representable
        let (down, up) = bf16_neighbors(v);
        for r in 0u32..=0xffff {
            let f = half::bf16::from_bits(f32_to_bf16_stochastic_bits(v, r as u16)).to_f32();
            assert!(
                f == down || f == up,
                "stochastic bf16 {f} not in {{{down}, {up}}} at r={r}"
            );
        }
    }

    #[test]
    fn stochastic_bf16_is_unbiased_over_full_rng_range() {
        // Averaging across every 16-bit draw reproduces the f32 value:
        // P(round up) == truncated_mantissa / 2^16, so E[round] == v.
        // 1.0 + 1.0/256.0: exactly half a bf16 ULP above 1.0 (the exact
        // value 1.00390625, written in closed form to keep the mantissa
        // midpoint explicit).
        for &v in &[0.1f32, -0.1, 1.0 + 1.0 / 256.0, -3.7, 1234.567] {
            let mut sum = 0.0f64;
            for r in 0u32..=0xffff {
                sum +=
                    half::bf16::from_bits(f32_to_bf16_stochastic_bits(v, r as u16)).to_f32() as f64;
            }
            let mean = sum / 65536.0;
            let (down, up) = bf16_neighbors(v);
            let ulp = ((up - down).abs() as f64).max(f64::MIN_POSITIVE);
            assert!(
                (mean - v as f64).abs() < ulp * 0.01,
                "stochastic bf16 mean {mean} biased vs {v} (ulp {ulp})"
            );
        }
    }

    #[test]
    fn stochastic_bf16_leaves_exact_values_unchanged() {
        // A bf16-representable value has zero dropped bits, so no draw
        // can carry into the kept mantissa — it must round to itself.
        let v = 1.5f32;
        for r in 0u32..=0xffff {
            let f = half::bf16::from_bits(f32_to_bf16_stochastic_bits(v, r as u16)).to_f32();
            assert_eq!(f, v, "exact value perturbed at r={r}");
        }
    }

    #[test]
    fn stochastic_bf16_passes_through_nan_and_inf() {
        let nan = half::bf16::from_bits(f32_to_bf16_stochastic_bits(f32::NAN, 0x1234)).to_f32();
        assert!(nan.is_nan());
        let inf =
            half::bf16::from_bits(f32_to_bf16_stochastic_bits(f32::INFINITY, 0xffff)).to_f32();
        assert!(inf.is_infinite() && inf > 0.0);
    }

    #[test]
    fn stochastic_round_rng16_is_reproducible_and_decorrelates() {
        assert_eq!(
            stochastic_round_rng16(42, 7, 3),
            stochastic_round_rng16(42, 7, 3)
        );
        let a = stochastic_round_rng16(42, 7, 0);
        let b = stochastic_round_rng16(42, 7, 1);
        let c = stochastic_round_rng16(42, 8, 0);
        assert!(a != b || a != c, "rng16 failed to decorrelate idx/step");
    }

    #[test]
    fn build_master_tensor_bf16_stochastic_branch_is_wired() {
        // v sits exactly half a bf16 ULP above 1.0 (dropped bits =
        // 0x8000), so the stochastic branch rounds up ~50% of the time:
        // over 256 elements it must produce BOTH neighbors, whereas
        // round-to-nearest is constant. P(no variation) = 2^-256.
        let v = 1.0f32 + 1.0f32 / 256.0f32;
        let vals = vec![v; 256];
        let rtn = build_master_tensor(
            DType::BF16,
            &[256],
            &vals,
            StochasticRoundingPolicy::RoundToNearest,
            0,
        )
        .unwrap();
        let sto = build_master_tensor(
            DType::BF16,
            &[256],
            &vals,
            StochasticRoundingPolicy::Stochastic { seed: 1 },
            1,
        )
        .unwrap();
        let rtn_v = read_to_f32(&rtn).unwrap();
        let sto_v = read_to_f32(&sto).unwrap();
        assert!(
            rtn_v.iter().all(|&x| x == rtn_v[0]),
            "round-to-nearest not constant"
        );
        let (down, up) = bf16_neighbors(v);
        assert!(
            sto_v.contains(&down) && sto_v.contains(&up),
            "stochastic branch did not produce both neighbors (down={down} up={up})"
        );
    }

    fn fresh_param() -> Parameter {
        let fwd = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let master = Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let policy = AmpPolicy::fp32_reference();
        Parameter::trainable(ForwardStorage::Plain(fwd), master, policy)
    }

    #[test]
    fn adamw_default_hyperparameters() {
        let hp = AdamWHyperparameters::default();
        assert_eq!(hp.lr, 1e-3);
        assert_eq!(hp.beta1, 0.9);
        assert_eq!(hp.beta2, 0.999);
        assert_eq!(hp.eps, 1e-8);
        assert_eq!(hp.weight_decay, 0.0);
    }

    #[test]
    fn adamw_name_is_stable() {
        let opt = AdamW::default_hp();
        assert_eq!(opt.name(), "adamw");
    }

    #[test]
    fn adamw_step_writes_updated_master() {
        // Single AdamW step on a [1, 2, 3, 4] master with grad=ones.
        // m[i] = 0.1 * 1 = 0.1; v[i] = 0.001 * 1 = 0.001.
        // m_hat = 0.1 / 0.1 = 1; v_hat = 0.001 / 0.001 = 1.
        // update = lr * 1 / (1 + eps) ≈ lr ≈ 1e-3.
        // master_new ≈ master - 1e-3.
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let master = p.backward_storage().expect("master");
        let cpu = master
            .storage()
            .as_any()
            .downcast_ref::<kiln_tensor::CpuStorage>()
            .unwrap();
        let values: Vec<f32> = cpu
            .as_bytes()
            .chunks(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let expected = [1.0f32 - 1e-3, 2.0 - 1e-3, 3.0 - 1e-3, 4.0 - 1e-3];
        for (i, (got, want)) in values.iter().zip(expected.iter()).enumerate() {
            assert!((got - want).abs() < 1e-4, "idx {i}: got {got}, want {want}");
        }
    }

    #[test]
    fn adamw_step_preserves_tensor_id() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let id_before = p.tensor_id();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(p.tensor_id(), id_before);
        // Moments still keyed under the same id.
        assert!(opt.moments(id_before).is_some());
    }

    #[test]
    fn adamw_multi_step_descends_master() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32; 4], vec![4]).unwrap();
        let mut last = [1.0f32, 2.0, 3.0, 4.0];
        for _ in 0..3 {
            opt.step(&mut p, &g).unwrap();
            let cpu = p
                .backward_storage()
                .unwrap()
                .storage()
                .as_any()
                .downcast_ref::<kiln_tensor::CpuStorage>()
                .unwrap();
            let cur: Vec<f32> = cpu
                .as_bytes()
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            for (i, (l, c)) in last.iter().zip(cur.iter()).enumerate() {
                assert!(c < l, "idx {i}: step did not descend ({l} -> {c})");
            }
            last.copy_from_slice(&cur);
        }
    }

    #[test]
    fn adamw_step_initializes_moments() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        let m = opt.moments(p.tensor_id()).expect("moments");
        assert_eq!(m.step, 1);
        assert_eq!(m.m.len(), 4);
        assert_eq!(m.v.len(), 4);
        // After one step: m[i] = (1 - beta1) * grad[i] = 0.1 * grad[i].
        for (i, &expected) in [0.01_f32, 0.02, 0.03, 0.04].iter().enumerate() {
            assert!((m.m[i] - expected).abs() < 1e-6, "m[{i}] = {}", m.m[i]);
        }
        // After one step: v[i] = (1 - beta2) * grad[i]^2 = 0.001 * grad[i]^2.
        // Tolerance is 1e-7 (not 1e-9): the expected values down at 9e-5 lose
        // ~3 ULPs to f32 rounding depending on the multiplication order, and
        // 1e-9 is below that noise floor.
        for (i, &g_i) in [0.1_f32, 0.2, 0.3, 0.4].iter().enumerate() {
            let expected = 0.001 * g_i * g_i;
            assert!((m.v[i] - expected).abs() < 1e-7, "v[{i}] = {}", m.v[i]);
        }
    }

    #[test]
    fn adamw_step_increments_step_counter() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        opt.step(&mut p, &g).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.moments(p.tensor_id()).unwrap().step, 3);
    }

    #[test]
    fn adamw_step_rejects_missing_master() {
        let mut opt = AdamW::default_hp();
        let fwd = Tensor::from_slice(&[1.0f32], vec![1]).unwrap();
        let mut p = Parameter::inference_only(ForwardStorage::Plain(fwd));
        let g = Tensor::from_slice(&[0.0f32], vec![1]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::NoBackwardStorage => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_step_rejects_shape_mismatch() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2], vec![2]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::GradShapeMismatch { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_step_rejects_dtype_mismatch() {
        // Param policy is fp32_reference (BF16-bwd is not the policy);
        // pass a BF16 grad -> mismatch.
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g_bf16: Vec<half::bf16> = (0..4)
            .map(|i| half::bf16::from_f32(i as f32 * 0.1))
            .collect();
        let g = Tensor::from_slice(&g_bf16, vec![4]).unwrap();
        let e = opt.step(&mut p, &g).unwrap_err();
        match e {
            StepError::GradDtypeMismatch { .. } => {}
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[test]
    fn adamw_reset_clears_moments() {
        let mut opt = AdamW::default_hp();
        let mut p = fresh_param();
        let g = Tensor::from_slice(&[0.1f32, 0.2, 0.3, 0.4], vec![4]).unwrap();
        opt.step(&mut p, &g).unwrap();
        assert_eq!(opt.parameter_count(), 1);
        opt.reset();
        assert_eq!(opt.parameter_count(), 0);
    }
}
