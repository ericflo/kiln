use anyhow::Result;
use std::sync::OnceLock;

static CUDA_MARLIN_POLICY: OnceLock<CudaMarlinPolicy> = OnceLock::new();

/// Process-lifetime CUDA Marlin weight-layout policy.
///
/// The server installs one closed projection profile before model weights are
/// uploaded. Embedded callers that do not install a policy retain the
/// historical BF16 layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CudaMarlinPolicy {
    pub(crate) attention_q_and_mlp: bool,
    pub(crate) gdn_out_proj: bool,
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
impl CudaMarlinPolicy {
    /// Preserve the historical unset behavior: do not create Marlin weights.
    pub const fn disabled() -> Self {
        Self {
            attention_q_and_mlp: false,
            gdn_out_proj: false,
        }
    }

    /// Pack full-attention Q and all MLP projections with Marlin W4A16.
    pub const fn attention_mlp() -> Self {
        Self {
            attention_q_and_mlp: true,
            gdn_out_proj: false,
        }
    }

    /// Also pack the more quality-sensitive GDN output projection.
    pub const fn attention_mlp_gdn() -> Self {
        Self {
            attention_q_and_mlp: true,
            gdn_out_proj: true,
        }
    }
}

impl Default for CudaMarlinPolicy {
    fn default() -> Self {
        Self::disabled()
    }
}

/// Install immutable CUDA Marlin policy. Same-value installation is
/// idempotent; a conflicting value fails instead of changing weight layout.
#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub fn install_cuda_marlin_policy(policy: CudaMarlinPolicy) -> Result<()> {
    match CUDA_MARLIN_POLICY.set(policy) {
        Ok(()) => Ok(()),
        Err(policy) if CUDA_MARLIN_POLICY.get() == Some(&policy) => Ok(()),
        Err(_) => anyhow::bail!("CUDA Marlin policy was already installed with a different value"),
    }
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn current_cuda_marlin_policy() -> CudaMarlinPolicy {
    *CUDA_MARLIN_POLICY.get_or_init(CudaMarlinPolicy::default)
}

#[cfg(test)]
mod tests {
    use super::CudaMarlinPolicy;

    #[test]
    fn profiles_are_closed_and_preserve_historical_defaults() {
        assert_eq!(
            CudaMarlinPolicy::disabled(),
            CudaMarlinPolicy {
                attention_q_and_mlp: false,
                gdn_out_proj: false,
            }
        );
        assert_eq!(
            CudaMarlinPolicy::attention_mlp(),
            CudaMarlinPolicy {
                attention_q_and_mlp: true,
                gdn_out_proj: false,
            }
        );
        assert_eq!(
            CudaMarlinPolicy::attention_mlp_gdn(),
            CudaMarlinPolicy {
                attention_q_and_mlp: true,
                gdn_out_proj: true,
            }
        );
    }
}
