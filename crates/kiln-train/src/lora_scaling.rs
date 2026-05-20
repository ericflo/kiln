//! LoRA rank/alpha scaling guardrails.

use anyhow::{Result, bail};

pub const MAX_LORA_ALPHA_OVER_RANK: f32 = 2.0;
pub const ALLOW_HIGH_LORA_SCALE_FLAG: &str = "--allow-high-lora-scale";

pub fn alpha_over_rank(rank: usize, alpha: f32) -> Result<f32> {
    if rank == 0 {
        bail!("LoRA rank must be greater than zero");
    }
    Ok(alpha / rank as f32)
}

pub fn validate_lora_scaling(
    rank: usize,
    alpha: f32,
    allow_high_lora_scale: bool,
) -> Result<f32> {
    let ratio = alpha_over_rank(rank, alpha)?;
    if ratio > MAX_LORA_ALPHA_OVER_RANK && !allow_high_lora_scale {
        bail!(
            "unsafe LoRA scaling: alpha/rank = {ratio:.3} exceeds the default limit of {MAX_LORA_ALPHA_OVER_RANK:.3}; lower --alpha, raise --rank, or pass {ALLOW_HIGH_LORA_SCALE_FLAG} for a deliberate experiment"
        );
    }
    Ok(ratio)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lora_scaling_accepts_default_limit() -> Result<()> {
        let ratio = validate_lora_scaling(8, 16.0, false)?;
        assert_eq!(ratio, 2.0);
        Ok(())
    }

    #[test]
    fn lora_scaling_rejects_high_ratio_by_default() {
        let err = validate_lora_scaling(4, 12.0, false)
            .unwrap_err()
            .to_string();

        assert!(err.contains("unsafe LoRA scaling"));
        assert!(err.contains("alpha/rank = 3.000"));
        assert!(err.contains(ALLOW_HIGH_LORA_SCALE_FLAG));
    }

    #[test]
    fn lora_scaling_override_allows_high_ratio() -> Result<()> {
        let ratio = validate_lora_scaling(4, 12.0, true)?;
        assert_eq!(ratio, 3.0);
        Ok(())
    }
}
