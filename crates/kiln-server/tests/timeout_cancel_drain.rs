//! Regression test for issue #664 — `tokio::time::timeout` cancels the
//! outer future, but `tokio::task::spawn_blocking` does not honor that
//! cancellation; the inner closure keeps running on the blocking thread
//! pool and continues to hold any locks it acquired (`runner.read()`,
//! `prefix_cache.lock()`, ...). Subsequent requests then race against the
//! still-held state and 5xx with `prefill f...`.
//!
//! The fix in `kiln-server/src/api/completions.rs` is a cooperative
//! cancellation handle (`kiln_model::CancelHandle`) that the inner decode
//! loops poll between tokens, plus a drain step on the timeout branch:
//! signal cancel, then `.await` the join handle so the closure releases
//! its locks before we return 408.
//!
//! These tests pin the mechanism end-to-end without standing up a real
//! `ModelRunner` (which would need GPU weights). They model the
//! lock-contention scenario directly with `RwLock<()>` (mirroring
//! `runner.read()`) and `Mutex<()>` (mirroring `prefix_cache.lock()`).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;

use kiln_model::CancelHandle;

/// Without the drain step, a follow-up request that needs a write-lock on
/// the same `RwLock` deadlocks for as long as the orphaned closure runs.
/// With cancel + drain, the read-lock releases promptly and the next
/// request acquires its write-lock without contention.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cancel_drain_releases_runner_read_lock_for_next_request() {
    let runner_lock: Arc<RwLock<()>> = Arc::new(RwLock::new(()));
    let prefix_cache_lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    let cancel = CancelHandle::new();

    let runner_inner = runner_lock.clone();
    let prefix_inner = prefix_cache_lock.clone();
    let cancel_inner = cancel.clone();
    let polls = Arc::new(AtomicUsize::new(0));
    let polls_inner = polls.clone();

    // Stand-in for the spawn_blocking generation closure: acquires the
    // same locks the real path acquires (`runner.read()` then
    // `prefix_cache.lock()`), then loops doing CPU work while polling
    // the cancel flag the way the decode loop does.
    let generation = tokio::task::spawn_blocking(move || {
        let _runner_guard = runner_inner.read().unwrap();
        let _pc_guard = prefix_inner.lock().unwrap();
        for _ in 0..10_000 {
            polls_inner.fetch_add(1, Ordering::SeqCst);
            if cancel_inner.is_cancelled() {
                return Err("cancelled");
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        Ok(())
    });

    tokio::pin!(generation);
    let timed_out = tokio::time::timeout(Duration::from_millis(50), &mut generation)
        .await
        .is_err();
    assert!(timed_out, "generation should outlast the 50ms timeout");

    // Drain pattern from completions.rs: signal cancel + await join.
    cancel.cancel();
    let join_result = generation.await;
    assert!(
        join_result.is_ok(),
        "join handle should resolve cleanly after cancel"
    );
    assert_eq!(
        join_result.unwrap(),
        Err("cancelled"),
        "closure should observe the cancel and return early"
    );

    // The follow-up request now needs a write-lock on the same RwLock
    // (e.g. an adapter swap) and a fresh acquisition of the prefix cache
    // mutex. Both must succeed promptly because the drained closure
    // released them.
    let runner_for_next = runner_lock.clone();
    let prefix_for_next = prefix_cache_lock.clone();
    let next = tokio::time::timeout(Duration::from_millis(500), async move {
        tokio::task::spawn_blocking(move || {
            let _w = runner_for_next.write().unwrap();
            let _p = prefix_for_next.lock().unwrap();
        })
        .await
    })
    .await;

    assert!(
        next.is_ok(),
        "follow-up request deadlocked — locks were not released"
    );
    next.unwrap().expect("follow-up join should succeed");
}

/// Without the drain step, the lock stays held after the timeout fires —
/// pin the broken behavior so a future refactor can't silently regress it.
/// We only assert the *un*-drained case to keep the test deterministic
/// (the orphaned thread sleeps in 2ms increments, well past the 50ms
/// timeout we use here).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_drain_means_lock_still_held_after_timeout() {
    let lock: Arc<Mutex<()>> = Arc::new(Mutex::new(()));
    let lock_inner = lock.clone();

    let _generation = tokio::task::spawn_blocking(move || {
        let _g = lock_inner.lock().unwrap();
        std::thread::sleep(Duration::from_millis(500));
    });

    // Give the spawn_blocking closure a moment to actually take the lock.
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Outer-future timeout fires; we drop the JoinHandle without awaiting.
    let lock_for_probe = lock.clone();
    let probe = tokio::time::timeout(
        Duration::from_millis(50),
        tokio::task::spawn_blocking(move || {
            let _g = lock_for_probe.lock().unwrap();
        }),
    )
    .await;

    assert!(
        probe.is_err(),
        "without drain, the next lock acquisition must time out — \
         this proves the cascade scenario from #664 actually requires \
         the drain step"
    );
}

/// Confirms the inner closure can observe a cancel signal that was
/// delivered while it was already running (the realistic scenario:
/// timeout fires mid-generation).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn inner_closure_observes_cancel_set_after_start() {
    let cancel = CancelHandle::new();
    let cancel_inner = cancel.clone();

    let generation = tokio::task::spawn_blocking(move || {
        for step in 0..1_000 {
            if cancel_inner.is_cancelled() {
                return step;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        -1
    });

    tokio::time::sleep(Duration::from_millis(30)).await;
    cancel.cancel();

    let observed_step = generation.await.unwrap();
    assert!(
        observed_step >= 0,
        "closure should have returned via cancel branch"
    );
    assert!(
        observed_step < 1_000,
        "closure should not have run to completion (got step {observed_step})"
    );
}
