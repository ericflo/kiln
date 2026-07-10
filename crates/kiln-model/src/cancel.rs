use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// Cooperative cancellation shared by a request owner and model execution.
///
/// Tiled prefill polls the handle between tile submissions and decode polls it
/// at token boundaries. Cancellation therefore does not preempt an in-flight
/// backend operation. Streaming callers must signal cancellation and continue
/// observing the stream's explicit settlement acknowledgement before releasing
/// request lifetimes or treating GPU-owned state as reusable.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
    prefill_tokens_completed: Arc<Mutex<u64>>,
    prefill_progress_gauge: Option<Arc<AtomicU64>>,
}

impl CancelHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prefill_progress_gauge(prefill_progress_gauge: Arc<AtomicU64>) -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            prefill_tokens_completed: Arc::new(Mutex::new(0)),
            prefill_progress_gauge: Some(prefill_progress_gauge),
        }
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
    }

    pub fn report_prefill_tokens_completed(&self, completed: u64) {
        if self.is_cancelled() {
            return;
        }

        // The progress value and its external gauge must change under one
        // lock. Otherwise cancellation could clear the value between the two
        // atomic gauge operations and briefly wrap the unsigned gauge.
        let mut current = self
            .prefill_tokens_completed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if self.is_cancelled() || completed <= *current {
            return;
        }
        let previous = *current;
        *current = completed;
        if let Some(gauge) = &self.prefill_progress_gauge {
            gauge.fetch_add(completed - previous, Ordering::SeqCst);
        }

        // Cancellation may race the update after the check above. Remove the
        // contribution while still holding the progress lock; a concurrent
        // owner-side clear will then observe zero and remain idempotent.
        if self.is_cancelled() {
            *current = 0;
            if let Some(gauge) = &self.prefill_progress_gauge {
                gauge.fetch_sub(completed, Ordering::SeqCst);
            }
        }
    }

    pub fn prefill_tokens_completed(&self) -> u64 {
        *self
            .prefill_tokens_completed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    pub fn clear_prefill_progress(&self) {
        let mut current = self
            .prefill_tokens_completed
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::mem::replace(&mut *current, 0);
        if previous > 0 {
            if let Some(gauge) = &self.prefill_progress_gauge {
                gauge.fetch_sub(previous, Ordering::SeqCst);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_starts_uncancelled() {
        let h = CancelHandle::new();
        assert!(!h.is_cancelled());
    }

    #[test]
    fn cancel_sets_flag() {
        let h = CancelHandle::new();
        h.cancel();
        assert!(h.is_cancelled());
    }

    #[test]
    fn clones_share_state() {
        let h = CancelHandle::new();
        let h2 = h.clone();
        h.cancel();
        assert!(h2.is_cancelled());
    }

    #[test]
    fn prefill_progress_updates_shared_gauge() {
        let gauge = Arc::new(AtomicU64::new(0));
        let h = CancelHandle::with_prefill_progress_gauge(gauge.clone());
        h.report_prefill_tokens_completed(128);
        assert_eq!(h.prefill_tokens_completed(), 128);
        assert_eq!(gauge.load(Ordering::SeqCst), 128);

        let h2 = h.clone();
        h2.report_prefill_tokens_completed(256);
        assert_eq!(h.prefill_tokens_completed(), 256);
        assert_eq!(gauge.load(Ordering::SeqCst), 256);

        h.report_prefill_tokens_completed(64);
        assert_eq!(h.prefill_tokens_completed(), 256);
        assert_eq!(gauge.load(Ordering::SeqCst), 256);

        h.clear_prefill_progress();
        assert_eq!(h.prefill_tokens_completed(), 0);
        assert_eq!(gauge.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn cancelled_handle_rejects_and_clears_racing_prefill_progress() {
        for _ in 0..128 {
            let gauge = Arc::new(AtomicU64::new(0));
            let handle = CancelHandle::with_prefill_progress_gauge(Arc::clone(&gauge));
            let reporter = handle.clone();
            let canceller = handle.clone();
            let barrier = Arc::new(std::sync::Barrier::new(3));
            let report_barrier = Arc::clone(&barrier);
            let cancel_barrier = Arc::clone(&barrier);

            let report_thread = std::thread::spawn(move || {
                report_barrier.wait();
                reporter.report_prefill_tokens_completed(128);
            });
            let cancel_thread = std::thread::spawn(move || {
                cancel_barrier.wait();
                canceller.cancel();
                canceller.clear_prefill_progress();
            });
            barrier.wait();
            report_thread.join().expect("progress reporter");
            cancel_thread.join().expect("progress canceller");

            assert_eq!(handle.prefill_tokens_completed(), 0);
            assert_eq!(gauge.load(Ordering::SeqCst), 0);
            handle.report_prefill_tokens_completed(256);
            assert_eq!(handle.prefill_tokens_completed(), 0);
            assert_eq!(gauge.load(Ordering::SeqCst), 0);
        }
    }
}
