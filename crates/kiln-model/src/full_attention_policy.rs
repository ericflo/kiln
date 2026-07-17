//! Process-lifetime policy for exact full-attention score geometry.

use anyhow::{Result, bail};
use std::sync::OnceLock;

pub const DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 2048;
pub const MIN_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 64;
pub const MAX_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 2048;

static FULL_ATTENTION_SCORE_BUDGET_MIB: OnceLock<usize> = OnceLock::new();

pub fn validate_full_attention_score_budget_mib(budget_mib: usize) -> Result<()> {
    if !(MIN_FULL_ATTENTION_SCORE_BUDGET_MIB..=MAX_FULL_ATTENTION_SCORE_BUDGET_MIB)
        .contains(&budget_mib)
    {
        bail!(
            "full-attention score budget must be {MIN_FULL_ATTENTION_SCORE_BUDGET_MIB}..={MAX_FULL_ATTENTION_SCORE_BUDGET_MIB} MiB; got {budget_mib}"
        );
    }
    Ok(())
}

/// Install the score ceiling before model execution begins.
///
/// Reinstalling the same value is harmless. A conflicting value fails closed
/// because route geometry must not change after an accelerator starts work.
pub fn install_full_attention_score_budget_mib(budget_mib: usize) -> Result<()> {
    validate_full_attention_score_budget_mib(budget_mib)?;
    if let Some(installed) = FULL_ATTENTION_SCORE_BUDGET_MIB.get() {
        if *installed == budget_mib {
            return Ok(());
        }
        bail!(
            "full-attention score budget is already installed as {installed} MiB; cannot replace it with {budget_mib} MiB"
        );
    }

    #[cfg(any(feature = "cuda", feature = "rocm"))]
    kiln_flash_attn::install_full_attention_score_budget_mib(budget_mib)
        .map_err(anyhow::Error::msg)?;

    match FULL_ATTENTION_SCORE_BUDGET_MIB.set(budget_mib) {
        Ok(()) => Ok(()),
        Err(_) if FULL_ATTENTION_SCORE_BUDGET_MIB.get() == Some(&budget_mib) => Ok(()),
        Err(_) => bail!("full-attention score-budget installation raced with another caller"),
    }
}

pub(crate) fn full_attention_score_budget_mib() -> usize {
    *FULL_ATTENTION_SCORE_BUDGET_MIB.get_or_init(|| DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn supported_budget_range_is_explicit() {
        assert!(validate_full_attention_score_budget_mib(63).is_err());
        assert!(validate_full_attention_score_budget_mib(64).is_ok());
        assert!(validate_full_attention_score_budget_mib(2048).is_ok());
        assert!(validate_full_attention_score_budget_mib(2049).is_err());
    }

    #[test]
    fn first_use_is_the_bounded_default_and_conflicts_fail_closed() {
        assert_eq!(full_attention_score_budget_mib(), 2048);
        install_full_attention_score_budget_mib(2048).unwrap();
        install_full_attention_score_budget_mib(2048).unwrap();
        let detail = install_full_attention_score_budget_mib(1024)
            .unwrap_err()
            .to_string();
        assert!(detail.contains("already installed as 2048 MiB"), "{detail}");
    }
}
