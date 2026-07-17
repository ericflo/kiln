//! Process-lifetime policy for kiln-tensor adapter routes.

use anyhow::{Result, bail};
use std::sync::OnceLock;

/// Startup-authoritative kiln-tensor adapter route selection.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum KtApiMode {
    /// Use qualified defaults: stable routes are active and experimental
    /// matmul and paged-KV routes remain inactive.
    #[default]
    Auto,
    /// Activate every adapter route, including experimental routes.
    All,
    /// Disable every adapter route and use the legacy fallbacks.
    Disabled,
}

impl KtApiMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::All => "all",
            Self::Disabled => "disabled",
        }
    }

    #[cfg(any(feature = "cuda", feature = "rocm", test))]
    pub(crate) const fn stable_routes_enabled(self) -> bool {
        !matches!(self, Self::Disabled)
    }

    #[cfg(any(feature = "cuda", feature = "rocm", test))]
    pub(crate) const fn experimental_routes_enabled(self) -> bool {
        matches!(self, Self::All)
    }
}

static KT_API_MODE: OnceLock<KtApiMode> = OnceLock::new();

/// Install the adapter route policy before model execution begins.
///
/// Reinstalling the same value is harmless. A conflicting value fails closed
/// because changing routes after tensors have begun executing would make one
/// process internally inconsistent.
pub fn install_kt_api_mode(mode: KtApiMode) -> Result<()> {
    if let Some(installed) = KT_API_MODE.get() {
        if *installed == mode {
            return Ok(());
        }
        bail!(
            "kiln-tensor API mode is already installed as {}; cannot replace it with {}",
            installed.as_str(),
            mode.as_str()
        );
    }
    KT_API_MODE
        .set(mode)
        .map_err(|_| anyhow::anyhow!("kiln-tensor API mode installation raced with another caller"))
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn stable_routes_enabled() -> bool {
    KT_API_MODE
        .get_or_init(KtApiMode::default)
        .stable_routes_enabled()
}

#[cfg(any(feature = "cuda", feature = "rocm", test))]
pub(crate) fn experimental_routes_enabled() -> bool {
    KT_API_MODE
        .get_or_init(KtApiMode::default)
        .experimental_routes_enabled()
}

#[cfg(test)]
mod tests {
    use super::{
        KtApiMode, experimental_routes_enabled, install_kt_api_mode, stable_routes_enabled,
    };

    #[test]
    fn mode_route_sets_are_explicit() {
        assert!(KtApiMode::Auto.stable_routes_enabled());
        assert!(!KtApiMode::Auto.experimental_routes_enabled());
        assert!(KtApiMode::All.stable_routes_enabled());
        assert!(KtApiMode::All.experimental_routes_enabled());
        assert!(!KtApiMode::Disabled.stable_routes_enabled());
        assert!(!KtApiMode::Disabled.experimental_routes_enabled());
    }

    #[test]
    fn installation_is_idempotent_and_conflicts_fail_closed() {
        assert!(stable_routes_enabled(), "first use must install auto");
        assert!(!experimental_routes_enabled());
        install_kt_api_mode(KtApiMode::Auto).unwrap();
        install_kt_api_mode(KtApiMode::Auto).unwrap();
        let error = install_kt_api_mode(KtApiMode::All).unwrap_err();
        let detail = error.to_string();
        assert!(detail.contains("already installed as auto"), "{detail}");
        assert!(detail.contains("replace it with all"), "{detail}");
    }
}
