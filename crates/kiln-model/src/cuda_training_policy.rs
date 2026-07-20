use anyhow::Result;
use std::sync::OnceLock;

static CUDA_TRAINING_POLICY: OnceLock<CudaTrainingPolicy> = OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FlashAttentionBackwardMode {
    Fast,
    Deterministic,
}

/// Process-lifetime CUDA training-kernel policy.
///
/// The policy is installed before model construction and read when training
/// tape backward executes. Standalone callers retain the historical fast
/// FlashAttention backward path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaTrainingPolicy {
    pub(crate) flash_attention_backward_mode: FlashAttentionBackwardMode,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl CudaTrainingPolicy {
    pub const fn fast() -> Self {
        Self {
            flash_attention_backward_mode: FlashAttentionBackwardMode::Fast,
        }
    }

    pub const fn deterministic() -> Self {
        Self {
            flash_attention_backward_mode: FlashAttentionBackwardMode::Deterministic,
        }
    }
}

impl Default for CudaTrainingPolicy {
    fn default() -> Self {
        Self::fast()
    }
}

/// Install immutable CUDA training policy. Same-value installation is
/// idempotent; a conflicting value fails before work can begin.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn install_cuda_training_policy(policy: CudaTrainingPolicy) -> Result<()> {
    match CUDA_TRAINING_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) if CUDA_TRAINING_POLICY.get() == Some(&policy) => Ok(()),
        Err(_) => {
            anyhow::bail!("CUDA training policy was already installed with a different value")
        }
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn current_cuda_training_policy() -> CudaTrainingPolicy {
    *CUDA_TRAINING_POLICY.get_or_init(CudaTrainingPolicy::default)
}

#[cfg(test)]
mod tests {
    use super::CudaTrainingPolicy;

    #[test]
    fn profiles_preserve_fast_default_and_explicit_determinism() {
        assert_eq!(
            CudaTrainingPolicy::fast().flash_attention_backward_mode,
            super::FlashAttentionBackwardMode::Fast
        );
        assert_eq!(
            CudaTrainingPolicy::deterministic().flash_attention_backward_mode,
            super::FlashAttentionBackwardMode::Deterministic
        );
        assert_eq!(CudaTrainingPolicy::default(), CudaTrainingPolicy::fast());
    }
}
