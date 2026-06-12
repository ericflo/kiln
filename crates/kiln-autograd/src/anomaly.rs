//! `KILN_DETECT_ANOMALY` — NaN/Inf trap surface for the autograd tape.
//!
//! Scaffold ship for #1082's "every backward op asserts `is_finite()` on
//! its output and panics with the op's tape position on first violation"
//! deliverable. This module ships the **env-var probe**; the per-op
//! finite checks are a follow-up (see "Follow-up wiring" below).
//!
//! # Design
//!
//! `kiln_autograd::Tape::backward` walks tape nodes in reverse topo
//! order (see `tape.rs`). When [`anomaly_detection_enabled`] is `true`,
//! each [`BackwardOp::apply`] return is scanned for non-finite elements
//! before the gradient is propagated. The first violation triggers
//! [`anomaly_panic`] with:
//!
//! - the offending op's `name()` (matches NVTX range + parity-tolerance.csv key),
//! - the op's tape position (`node_index`), so the user can `grep` it
//!   directly in a debug-mode `Tape::nodes()` dump,
//! - the input shape and a short summary of which element(s) violated
//!   (the implementation prints the first NaN index; an `Inf` is
//!   reported as `+Inf` / `-Inf`).
//!
//! Cost: ~5% per step when enabled, per the #1082 bullet. The flag is
//! **off by default** in release and CI. Training parity tests in CI
//! enable it via `KILN_DETECT_ANOMALY=1` so corruption surfaces at the
//! producing kernel, not 100 steps later when loss diverges. Composes
//! with the in-place tape-version invariant (anti-pattern 16) — both
//! are tape-position-aware safety nets.
//!
//! # Env var contract
//!
//! `KILN_DETECT_ANOMALY=1` (or any of `1`, `true`, `TRUE`, `yes`, `on`)
//! enables the trap. Anything else (unset, `0`, `false`, the empty
//! string, `off`) leaves it off. The probe is read once per
//! `Tape::backward` call via [`anomaly_detection_enabled`] and not
//! cached, so unit tests can toggle the flag between calls.
//!
//! # Tape wiring
//!
//! The actual finite-checks per op live in `Tape::backward` and must:
//!
//! 1. Read [`anomaly_detection_enabled`] **once** at the top of
//!    `Tape::backward` to amortise the env-var lookup across the walk.
//! 2. After each `op.apply(&grad_output)` returns its `Vec<Option<Tensor>>`,
//!    iterate the `Some(t)` gradients and run `is_finite(t)` against
//!    the underlying storage on whatever backend the tensor lives on.
//!    CPU storage uses an in-place stride walker. CUDA storage today
//!    uses a D2H bridge (`cuda_to_host_copy` → CPU walker) so the
//!    check is fully wired end-to-end for both CPU and CUDA training
//!    paths under `KILN_DETECT_ANOMALY=1`. The per-backend
//!    `is_finite_storage` reduction kernel (CUDA/Metal/Vulkan) is
//!    planned to replace the bridge so anomaly detection on GPU
//!    stops paying the D2H copy per node visit.
//! 3. On first violation, call [`anomaly_panic`] with the op's `name()`
//!    and the tape position. Panicking is intentional — the issue body
//!    says "panics with the op's tape position on first violation."
//! 4. Compose with the existing anti-pattern-16 version check: the
//!    version drift check runs first (already in `tape.rs`), then
//!    `op.apply` runs, then the returned per-input gradients are
//!    finite-checked before propagation. This ordering catches
//!    stale-tape bugs separately from kernel-NaN bugs.
//!
//! # Why a scaffold first
//!
//! Splitting the env-var surface from the per-op finite checks kept
//! the original diff reviewable. Follow-up PRs can still land the
//! `is_finite_storage` shim on each backend independently without
//! changing the env-var contract again.

use std::env;

/// Env var name the autograd tape reads to decide whether to scan
/// backward-op outputs for NaN/Inf.
///
/// Exposed as a constant so the eventual `Tape::backward` wiring and
/// the test suite agree on the spelling without copying string
/// literals.
pub const ENV_DETECT_ANOMALY: &str = "KILN_DETECT_ANOMALY";

/// Returns `true` iff `KILN_DETECT_ANOMALY` is set to one of the
/// accepted truthy values: `1`, `true`, `TRUE`, `yes`, `on` (case
/// preserved as-shown, plus case-insensitive variants). Any other
/// value — including unset, the empty string, `0`, `false`, and
/// `off` — returns `false`.
///
/// The probe is intentionally cheap: a single `env::var` lookup plus
/// an ASCII-lowercase comparison. Callers that walk many tape nodes
/// should call this **once per backward pass** at the top of
/// `Tape::backward` and reuse the boolean — there is no internal
/// caching because unit tests need to toggle the flag between calls.
pub fn anomaly_detection_enabled() -> bool {
    match env::var(ENV_DETECT_ANOMALY) {
        Ok(v) => is_truthy(&v),
        Err(_) => false,
    }
}

/// Internal: shared truthy-string parser. Pulled out so the unit test
/// can hit it without `unsafe { env::set_var(...) }`.
fn is_truthy(v: &str) -> bool {
    let trimmed = v.trim();
    if trimmed.is_empty() {
        return false;
    }
    matches!(
        trimmed.to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

/// Panic with a structured anomaly-detection message. Called by
/// `Tape::backward` when a backward-op output contains a NaN or Inf.
///
/// The message is parsed by the training-parity tests' `should_panic`
/// directives; keep the `kiln_autograd: anomaly detected` prefix
/// stable across refactors.
///
/// `node_index` is the offset into `Tape::nodes()` so the user can
/// inspect the upstream graph. `op_name` matches the offending op's
/// [`crate::BackwardOp::name`] return.
#[track_caller]
pub fn anomaly_panic(node_index: usize, op_name: &str, detail: &str) -> ! {
    panic!(
        "kiln_autograd: anomaly detected at tape position {node_index} \
         (op `{op_name}`): {detail}. \
         Set KILN_DETECT_ANOMALY=0 to disable this trap."
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truthy_values_enable() {
        for v in ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON"] {
            assert!(is_truthy(v), "expected {v:?} to be truthy");
        }
    }

    #[test]
    fn falsy_values_disable() {
        for v in [
            "", "0", "false", "FALSE", "False", "no", "off", "OFF", "random",
        ] {
            assert!(!is_truthy(v), "expected {v:?} to be falsy");
        }
    }

    #[test]
    fn whitespace_is_trimmed() {
        assert!(is_truthy("  1  "));
        assert!(is_truthy("\ttrue\n"));
        assert!(!is_truthy("   "));
    }

    #[test]
    #[should_panic(expected = "kiln_autograd: anomaly detected at tape position 7")]
    fn anomaly_panic_contains_position_and_op_name() {
        anomaly_panic(7, "test/some_op", "grad[3] is NaN");
    }

    #[test]
    #[should_panic(expected = "op `test/some_op`")]
    fn anomaly_panic_contains_op_name_in_backticks() {
        anomaly_panic(0, "test/some_op", "grad[0] is +Inf");
    }
}
