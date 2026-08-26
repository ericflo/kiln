//! Thread-local [`Tape`] scope and recording authority.
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
//! (`docs/archive/candle-removal/rmsnorm-kt-tape-production-caller-stop-2026-05-28.md`)
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
//! and hosts the kt-native per-op tape recorders (which depend on
//! `kiln-rmsnorm-kernel` + `kiln-kt-bridge` / `kiln-tensor` that the
//! autograd crate does not — and should not — pull in).
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
//! # Production contract
//!
//! The active thread-local scope is the sole authority for tape routing and
//! recording. [`tape_scope_active`] is false outside
//! [`with_thread_local_tape`], so inference keeps its normal fast paths. It is
//! true for the complete scoped closure and cannot be disabled by process
//! environment, so training cannot silently sever its gradient graph.
//!
//! # See also
//!
//! * [`crate::Tape`] — the recording surface itself.
//! * `kiln-model::tape_forward` — re-exports + the rmsnorm production
//!   adapter wired into `kiln-model::forward::rms_norm`.

use std::cell::RefCell;

use crate::{Tape, TapeOptions};

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

/// Returns whether the current thread is inside a tape-recording scope.
///
/// Production tape-aware operations must use this predicate, or
/// [`with_active_tape`] directly, as their routing authority. A failed borrow
/// means the tape is already mutably borrowed by a recording callback, which
/// also proves that a scope is active.
pub fn tape_scope_active() -> bool {
    ACTIVE_TAPE.with(|cell| {
        cell.try_borrow()
            .map(|active| active.is_some())
            .unwrap_or(true)
    })
}

struct TapeScopeGuard;

impl Drop for TapeScopeGuard {
    fn drop(&mut self) {
        ACTIVE_TAPE.with(|cell| {
            cell.borrow_mut().take();
        });
    }
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
    with_thread_local_tape_options(TapeOptions::default(), f)
}

/// Run `f` with a fresh thread-local tape carrying `options`.
///
/// The options are captured before `f` starts and cannot change during the
/// scope. This is the request-safe entry point for training APIs that expose
/// tape diagnostics in typed job configuration.
pub fn with_thread_local_tape_options<R>(options: TapeOptions, f: impl FnOnce() -> R) -> (R, Tape) {
    ACTIVE_TAPE.with(|cell| {
        assert!(
            cell.borrow().is_none(),
            "kiln-autograd::tape_scope: nested tape scopes are not supported \
             (a Tape is already active on this thread)"
        );
        *cell.borrow_mut() = Some(Tape::with_options(options));
    });
    let guard = TapeScopeGuard;

    let result = f();

    let tape = ACTIVE_TAPE.with(|cell| {
        cell.borrow_mut()
            .take()
            .expect("kiln-autograd::tape_scope: active tape vanished mid-scope")
    });
    drop(guard);
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
    use std::sync::{Arc, Barrier};

    #[test]
    fn tape_scope_authority_tracks_scope() {
        assert!(!tape_scope_active());
        let (result, tape) = with_thread_local_tape(|| 42);
        assert_eq!(result, 42);
        assert!(tape.is_empty());
        assert!(!tape_scope_active());
    }

    #[test]
    fn tape_scope_captures_explicit_options() {
        let options = TapeOptions {
            detect_anomaly: true,
        };
        let (_, tape) = with_thread_local_tape_options(options, || ());
        assert_eq!(tape.options(), options);
    }

    #[test]
    fn concurrent_tape_scopes_keep_options_request_local() {
        let barrier = Arc::new(Barrier::new(2));
        let spawn = |options| {
            let barrier = Arc::clone(&barrier);
            std::thread::spawn(move || {
                let (_, tape) = with_thread_local_tape_options(options, || {
                    barrier.wait();
                    assert!(tape_scope_active());
                });
                tape.options()
            })
        };

        let disabled = spawn(TapeOptions::default());
        let enabled = spawn(TapeOptions {
            detect_anomaly: true,
        });

        assert_eq!(disabled.join().unwrap(), TapeOptions::default());
        assert!(enabled.join().unwrap().detect_anomaly);
    }

    #[test]
    fn with_active_tape_returns_none_outside_scope() {
        let out: Option<i32> = with_active_tape(|_| 7);
        assert!(out.is_none());
    }

    #[test]
    fn with_active_tape_returns_some_inside_scope() {
        let ((), _tape) = with_thread_local_tape(|| {
            assert!(tape_scope_active());
            let out = with_active_tape(|_tape| 7);
            assert_eq!(out, Some(7));
        });
    }

    #[test]
    fn panicking_scope_is_cleaned_up() {
        let result = std::panic::catch_unwind(|| {
            let _ = with_thread_local_tape::<()>(|| panic!("test panic"));
        });
        assert!(result.is_err());
        assert!(!tape_scope_active());

        let (_, tape) = with_thread_local_tape(|| ());
        assert!(tape.is_empty());
    }

    #[test]
    #[should_panic(expected = "nested tape scopes")]
    fn nested_scopes_panic() {
        let _ = with_thread_local_tape(|| {
            let _ = with_thread_local_tape(|| 0);
        });
    }
}
