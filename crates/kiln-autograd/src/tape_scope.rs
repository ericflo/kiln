//! Thread-local `Tape` scope — Phase 6a/CP-4 ergonomics shim (#1082).
//!
//! # Why this module exists
//!
//! [`crate::Tape`] is explicit-pass by design: every recording site
//! threads `&mut Tape` through its signature. That works fine when the
//! caller already has a tape (e.g. unit tests, `kiln-train::tape_step`),
//! but it does NOT scale when the recording site is buried inside a
//! candle-typed forward function whose 20+ transitive callers have no
//! `&mut Tape` in scope.
//!
//! The wave-12 (#1082) audit
//! ([`docs/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`])
//! documented why a per-call-site `&mut Tape` flip of
//! `kiln_model::forward::rms_norm` could not land: 20+ candle-typed
//! call sites would all need to thread the parameter, with no caller
//! able to construct one. The wave-12 fix was a thread-local `Tape`
//! handle so the parameter doesn't have to be plumbed — `with_thread_local_tape(...)`
//! installs one for the scope of a forward call, and recording sites
//! pick it up via `with_active_tape(...)`.
//!
//! That implementation originally lived in
//! `kiln-model::tape_forward::ACTIVE_TAPE`. Wave-13 (#1082) promotes
//! the same machinery into `kiln-autograd` so the OPD and FLCE kernel
//! crates (and their `kiln-train` callers) can route through the same
//! thread-local handle without taking a `kiln-model` dependency.
//!
//! `kiln-model::tape_forward` re-exports these symbols for back-compat,
//! and continues to host the candle-typed `try_tape_rms_norm_cuda`
//! adapter (it depends on `kiln-rmsnorm-kernel` + `kiln-kt-bridge`
//! which the autograd crate does not — and should not — pull in).
//!
//! # Tape-scope contract
//!
//! 1. [`with_thread_local_tape`] installs a fresh `Tape` on the current
//!    thread for the duration of `f`. Returns `(f_result, finalised_tape)`.
//! 2. [`with_active_tape`] runs `f` with `&mut Tape` if a scope is
//!    active; returns `None` otherwise. The recording site then
//!    decides whether to fall back to a non-tape path (e.g. the
//!    candle-autograd `CustomOp` shim).
//! 3. Nesting `with_thread_local_tape` panics — gradient accumulation
//!    semantics across nested scopes are undefined until CP-4 figures
//!    that out.
//!
//! # Production-safety
//!
//! Off by default. With no scope open, `with_active_tape` returns
//! `None` and recording sites do not record. Crates calling
//! `tape_forward_enabled()` must additionally check the
//! `KILN_USE_TAPE_FORWARD` env var (or an equivalent opt-in) before
//! attempting a tape route — see the rmsnorm adapter at
//! `kiln-model::tape_forward::try_tape_rms_norm_cuda` for the
//! canonical pattern.
//!
//! # See also
//!
//! * [`crate::Tape`] — the recording surface itself.
//! * `kiln-model::tape_forward` — re-exports + the rmsnorm production
//!   adapter wired into `kiln-model::forward::rms_norm`.

use std::cell::RefCell;
use std::sync::OnceLock;

use crate::Tape;

/// `KILN_USE_TAPE_FORWARD` env var — opt-in only. Cached after first read.
///
/// Returns `true` only if the env var is set and not one of the
/// disable values (`0`, `false`, `no`, empty). Matches the convention
/// used by `KILN_VULKAN_RMSNORM` and friends in `kiln-model::forward`.
///
/// Lifted verbatim from `kiln-model::tape_forward` so kernel crates
/// can gate their own tape-routing adapters on the same env without
/// taking a `kiln-model` dependency.
pub fn tape_forward_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("KILN_USE_TAPE_FORWARD")
            .map(|v| {
                let v = v.trim().to_lowercase();
                !(v.is_empty() || v == "0" || v == "false" || v == "no")
            })
            .unwrap_or(false)
    })
}

thread_local! {
    /// Thread-local active `Tape`. Wrapped in `RefCell` so the
    /// `record(...)` paths can take a mutable borrow; we hand out a
    /// `&mut Tape` only for the duration of a single
    /// `with_active_tape` callback so concurrent recordings on the
    /// same tape from a single forward call don't overlap.
    ///
    /// `None` outside a `with_thread_local_tape` scope. Tape-aware
    /// primitives must check this and fall back to the
    /// non-tape-tracking path when no scope is active.
    static ACTIVE_TAPE: RefCell<Option<Tape>> = const { RefCell::new(None) };
}

/// Run `f` with a freshly-allocated `Tape` installed as the active
/// thread-local tape. Returns `(result_of_f, finalised_tape)` so the
/// caller can subsequently drive `Tape::backward` against it.
///
/// Panics if a tape is already installed on this thread — nesting is
/// not supported until the CP-4 work figures out gradient-accumulation
/// semantics across nested scopes. The panic is preferable to silent
/// shadowing because shadowing would route some ops onto a parent
/// tape and others onto the child.
pub fn with_thread_local_tape<R>(f: impl FnOnce() -> R) -> (R, Tape) {
    ACTIVE_TAPE.with(|cell| {
        assert!(
            cell.borrow().is_none(),
            "kiln-autograd::tape_scope: nested tape scopes are not supported \
             (a Tape is already active on this thread)"
        );
        *cell.borrow_mut() = Some(Tape::new());
    });

    let result = f();

    let tape = ACTIVE_TAPE.with(|cell| {
        cell.borrow_mut()
            .take()
            .expect("kiln-autograd::tape_scope: active tape vanished mid-scope")
    });
    (result, tape)
}

/// Run `f` with mutable access to the active thread-local tape. If no
/// tape is currently installed, returns `None` without invoking `f` —
/// callers must treat that as "no tape recording requested" and use
/// the non-tape path instead.
///
/// This is the canonical hook for kt-tape adapters living outside
/// `kiln-model`: the `kt-tape forward + record` happens inside the
/// closure, then the caller copies the kt result back into whatever
/// container the production caller expects (e.g. a candle Tensor).
pub fn with_active_tape<R>(f: impl FnOnce(&mut Tape) -> R) -> Option<R> {
    ACTIVE_TAPE.with(|cell| {
        let mut borrow = cell.borrow_mut();
        borrow.as_mut().map(f)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tape_forward_enabled_caches_first_read() {
        // The cache is process-wide; we can't reliably toggle the env
        // inside a single test process. This test just confirms the
        // function returns a stable bool across calls.
        let a = tape_forward_enabled();
        let b = tape_forward_enabled();
        assert_eq!(a, b);
    }

    #[test]
    fn with_thread_local_tape_round_trips() {
        let (result, tape) = with_thread_local_tape(|| 42);
        assert_eq!(result, 42);
        // Newly-allocated tape with no records.
        assert!(tape.is_empty());
    }

    #[test]
    fn with_active_tape_returns_none_outside_scope() {
        let out: Option<i32> = with_active_tape(|_| 7);
        assert!(out.is_none());
    }

    #[test]
    fn with_active_tape_returns_some_inside_scope() {
        let ((), _tape) = with_thread_local_tape(|| {
            let out = with_active_tape(|_tape| 7);
            assert_eq!(out, Some(7));
        });
    }

    #[test]
    #[should_panic(expected = "nested tape scopes")]
    fn nested_scopes_panic() {
        let _ = with_thread_local_tape(|| {
            let _ = with_thread_local_tape(|| 0);
        });
    }
}
