//! Process-lifetime score geometry for exact attention routes.

use std::cell::Cell;
use std::sync::OnceLock;

pub const DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 2048;
pub const MIN_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 64;
pub const MAX_FULL_ATTENTION_SCORE_BUDGET_MIB: usize = 2048;
pub(crate) const DEFAULT_MATERIALIZED_SCORE_TILE_MAX_ELEMENTS: usize = 1 << 29;
pub(crate) const DEFAULT_ROCM_ONLINE_QUERY_TILE: usize = 2048;
pub(crate) const DEFAULT_ROCM_ONLINE_KEY_TILE: usize = 4096;
pub(crate) const MAX_ROCM_ONLINE_SCORE_BUDGET_MIB: usize = 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ScoreGeometry {
    pub materialized_budget_mib: usize,
    pub materialized_tile_max_elements: usize,
    pub rocm_online_query_tile: usize,
    pub rocm_online_key_tile: usize,
    pub rocm_online_budget_mib: usize,
}

impl ScoreGeometry {
    const fn from_budget_mib(budget_mib: usize) -> Self {
        Self {
            materialized_budget_mib: budget_mib,
            materialized_tile_max_elements: DEFAULT_MATERIALIZED_SCORE_TILE_MAX_ELEMENTS,
            rocm_online_query_tile: DEFAULT_ROCM_ONLINE_QUERY_TILE,
            rocm_online_key_tile: DEFAULT_ROCM_ONLINE_KEY_TILE,
            rocm_online_budget_mib: if budget_mib < MAX_ROCM_ONLINE_SCORE_BUDGET_MIB {
                budget_mib
            } else {
                MAX_ROCM_ONLINE_SCORE_BUDGET_MIB
            },
        }
    }
}

impl Default for ScoreGeometry {
    fn default() -> Self {
        Self::from_budget_mib(DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB)
    }
}

static SCORE_GEOMETRY: OnceLock<ScoreGeometry> = OnceLock::new();

pub fn validate_full_attention_score_budget_mib(budget_mib: usize) -> Result<(), String> {
    if !(MIN_FULL_ATTENTION_SCORE_BUDGET_MIB..=MAX_FULL_ATTENTION_SCORE_BUDGET_MIB)
        .contains(&budget_mib)
    {
        return Err(format!(
            "full-attention score budget must be {MIN_FULL_ATTENTION_SCORE_BUDGET_MIB}..={MAX_FULL_ATTENTION_SCORE_BUDGET_MIB} MiB; got {budget_mib}"
        ));
    }
    Ok(())
}

/// Install score geometry before accelerator execution begins.
pub fn install_full_attention_score_budget_mib(budget_mib: usize) -> Result<(), String> {
    validate_full_attention_score_budget_mib(budget_mib)?;
    let requested = ScoreGeometry::from_budget_mib(budget_mib);
    if let Some(installed) = SCORE_GEOMETRY.get() {
        if *installed == requested {
            return Ok(());
        }
        return Err(format!(
            "full-attention score budget is already installed as {} MiB; cannot replace it with {budget_mib} MiB",
            installed.materialized_budget_mib
        ));
    }
    SCORE_GEOMETRY.set(requested).map_err(|_| {
        "full-attention score-budget installation raced with another caller".to_owned()
    })
}

pub(crate) fn score_geometry() -> ScoreGeometry {
    SCORE_GEOMETRY
        .get_or_init(ScoreGeometry::default)
        .to_owned()
}

thread_local! {
    static TEST_SCORE_GEOMETRY: Cell<Option<ScoreGeometry>> = const { Cell::new(None) };
}

pub(crate) fn effective_score_geometry() -> ScoreGeometry {
    TEST_SCORE_GEOMETRY
        .with(Cell::get)
        .unwrap_or_else(score_geometry)
}

/// Scoped exact-attention geometry override for real-device tests.
///
/// This avoids process-global environment mutation while retaining forced
/// multi-tile parity coverage. Production code must use the installed policy.
#[doc(hidden)]
pub fn with_test_score_geometry<T>(
    materialized_budget_mib: usize,
    rocm_online_query_tile: usize,
    rocm_online_key_tile: usize,
    rocm_online_budget_mib: usize,
    f: impl FnOnce() -> T,
) -> T {
    assert!(materialized_budget_mib > 0);
    assert!(rocm_online_query_tile > 0);
    assert!(rocm_online_key_tile > 0);
    assert!(rocm_online_budget_mib > 0);
    let geometry = ScoreGeometry {
        materialized_budget_mib,
        materialized_tile_max_elements: DEFAULT_MATERIALIZED_SCORE_TILE_MAX_ELEMENTS,
        rocm_online_query_tile,
        rocm_online_key_tile,
        rocm_online_budget_mib,
    };
    let previous = TEST_SCORE_GEOMETRY.with(|cell| cell.replace(Some(geometry)));
    struct Guard(Option<ScoreGeometry>);
    impl Drop for Guard {
        fn drop(&mut self) {
            TEST_SCORE_GEOMETRY.with(|cell| cell.set(self.0));
        }
    }
    let _guard = Guard(previous);
    f()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn installed_budget_derives_bounded_route_geometry() {
        let policy = ScoreGeometry::from_budget_mib(DEFAULT_FULL_ATTENTION_SCORE_BUDGET_MIB);
        assert_eq!(policy.materialized_budget_mib, 2048);
        assert_eq!(policy.rocm_online_budget_mib, 1024);
        assert_eq!(policy.rocm_online_query_tile, 2048);
        assert_eq!(policy.rocm_online_key_tile, 4096);
        assert_eq!(policy.materialized_tile_max_elements, 1 << 29);

        let reduced = ScoreGeometry::from_budget_mib(512);
        assert_eq!(reduced.materialized_budget_mib, 512);
        assert_eq!(reduced.rocm_online_budget_mib, 512);
    }

    #[test]
    fn validation_rejects_unbounded_geometry() {
        assert!(validate_full_attention_score_budget_mib(63).is_err());
        assert!(validate_full_attention_score_budget_mib(64).is_ok());
        assert!(validate_full_attention_score_budget_mib(2048).is_ok());
        assert!(validate_full_attention_score_budget_mib(2049).is_err());
    }
}
