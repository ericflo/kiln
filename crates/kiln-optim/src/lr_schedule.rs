//! Learning-rate schedules: `cosine`, `linear`, `linear_warmup_cosine`.
//!
//! Pure functions of `(step, total_steps, warmup_steps)`. Plug into
//! the optimizer by mutating `opt.hp.lr` between steps.

use std::f32::consts::PI;

/// Linear warmup from 0 to `peak_lr` over `warmup` steps, then
/// cosine-decay from `peak_lr` to `min_lr` over the remaining
/// `total - warmup` steps.
pub fn linear_warmup_cosine(step: u64, total: u64, warmup: u64, peak_lr: f32, min_lr: f32) -> f32 {
    if total == 0 {
        return peak_lr;
    }
    let s = step.min(total);
    if s < warmup {
        // Linear warmup.
        let progress = (s as f32) / (warmup.max(1) as f32);
        return peak_lr * progress;
    }
    let post_warmup = (s - warmup) as f32;
    let decay_span = (total - warmup) as f32;
    let progress = if decay_span > 0.0 {
        post_warmup / decay_span
    } else {
        0.0
    };
    let cos = 0.5 * (1.0 + (PI * progress).cos());
    min_lr + (peak_lr - min_lr) * cos
}

/// Cosine decay from `peak_lr` to `min_lr` over `total` steps.
/// No warmup.
pub fn cosine(step: u64, total: u64, peak_lr: f32, min_lr: f32) -> f32 {
    linear_warmup_cosine(step, total, 0, peak_lr, min_lr)
}

/// Linear decay from `peak_lr` to `min_lr` over `total` steps.
pub fn linear(step: u64, total: u64, peak_lr: f32, min_lr: f32) -> f32 {
    if total == 0 {
        return peak_lr;
    }
    let progress = (step.min(total) as f32) / (total as f32);
    peak_lr + (min_lr - peak_lr) * progress
}

/// Constant LR with linear warmup from 0 to `peak_lr`. After
/// `warmup` steps, stays at `peak_lr` forever.
pub fn constant_with_warmup(step: u64, warmup: u64, peak_lr: f32) -> f32 {
    if step < warmup {
        peak_lr * (step as f32) / (warmup.max(1) as f32)
    } else {
        peak_lr
    }
}

/// Step decay: drop LR by `gamma` every `step_size` steps.
/// `lr(t) = base_lr * gamma^floor(t / step_size)`.
pub fn step_decay(step: u64, step_size: u64, base_lr: f32, gamma: f32) -> f32 {
    if step_size == 0 {
        return base_lr;
    }
    let stages = step / step_size;
    base_lr * gamma.powi(stages as i32)
}

/// Exponential decay: `lr(t) = base_lr * gamma^t`. `gamma < 1` typical.
pub fn exponential_decay(step: u64, base_lr: f32, gamma: f32) -> f32 {
    base_lr * gamma.powi(step as i32)
}

/// Inverse square root decay (Vaswani et al. — original transformer):
/// `lr(t) = peak * min(t^(-0.5), t * warmup^(-1.5))`.
pub fn inverse_sqrt(step: u64, warmup: u64, peak_lr: f32) -> f32 {
    let t = step.max(1) as f32;
    let w = warmup.max(1) as f32;
    peak_lr * (1.0_f32 / t.sqrt()).min(t / w.powf(1.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_start_is_peak() {
        let lr = cosine(0, 100, 1.0, 0.0);
        assert!((lr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_end_is_min() {
        let lr = cosine(100, 100, 1.0, 0.0);
        assert!(lr.abs() < 1e-5);
    }

    #[test]
    fn cosine_midpoint_is_halfway() {
        let lr = cosine(50, 100, 1.0, 0.0);
        // At progress 0.5, cos(π*0.5) = 0; lr = 0.5 * 1 = 0.5.
        assert!((lr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn linear_start_is_peak() {
        assert!((linear(0, 100, 1.0, 0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn linear_end_is_min() {
        assert!(linear(100, 100, 1.0, 0.0).abs() < 1e-5);
    }

    #[test]
    fn linear_midpoint() {
        assert!((linear(50, 100, 1.0, 0.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn warmup_starts_at_zero() {
        let lr = linear_warmup_cosine(0, 100, 10, 1.0, 0.0);
        assert!(lr.abs() < 1e-5);
    }

    #[test]
    fn warmup_completes_at_step_n() {
        let lr = linear_warmup_cosine(10, 100, 10, 1.0, 0.0);
        // At end of warmup the cosine progress is 0 → lr = peak.
        assert!((lr - 1.0).abs() < 1e-5);
    }

    #[test]
    fn warmup_midpoint() {
        // Step 5 of 10-step warmup → halfway.
        let lr = linear_warmup_cosine(5, 100, 10, 1.0, 0.0);
        assert!((lr - 0.5).abs() < 1e-5);
    }

    #[test]
    fn step_past_total_clamps_to_min() {
        let lr = linear_warmup_cosine(1000, 100, 10, 1.0, 0.0);
        assert!(lr.abs() < 1e-5);
    }

    #[test]
    fn constant_with_warmup_warmup_phase() {
        assert!((constant_with_warmup(0, 10, 1.0)).abs() < 1e-5);
        assert!((constant_with_warmup(5, 10, 1.0) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn constant_with_warmup_post_warmup() {
        assert!((constant_with_warmup(10, 10, 1.0) - 1.0).abs() < 1e-5);
        assert!((constant_with_warmup(1000, 10, 1.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn step_decay_drops_by_gamma() {
        // step_size=100, gamma=0.5.
        // lr(0..99) = 1.0; lr(100..199) = 0.5; lr(200..299) = 0.25.
        assert!((step_decay(50, 100, 1.0, 0.5) - 1.0).abs() < 1e-5);
        assert!((step_decay(100, 100, 1.0, 0.5) - 0.5).abs() < 1e-5);
        assert!((step_decay(200, 100, 1.0, 0.5) - 0.25).abs() < 1e-5);
    }

    #[test]
    fn exponential_decay_known() {
        // gamma=0.9 → lr(0)=1, lr(1)=0.9, lr(10)=0.9^10 ≈ 0.3487
        assert!((exponential_decay(0, 1.0, 0.9) - 1.0).abs() < 1e-5);
        assert!((exponential_decay(1, 1.0, 0.9) - 0.9).abs() < 1e-5);
        assert!((exponential_decay(10, 1.0, 0.9) - 0.3486784).abs() < 1e-4);
    }

    #[test]
    fn inverse_sqrt_warmup_peak() {
        // At step == warmup, both branches equal peak * (1/√warmup).
        let lr = inverse_sqrt(100, 100, 1.0);
        let expected = 1.0_f32 / 10.0;
        assert!((lr - expected).abs() < 1e-5);
    }
}
