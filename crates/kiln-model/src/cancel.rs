use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

// `kiln-server` wraps blocking generation in `tokio::time::timeout`, but
// `tokio::task::spawn_blocking` does not honor outer-future cancellation —
// the inner closure keeps running on the blocking thread pool, holding any
// locks it acquired (`runner.read()`, `prefix_cache.lock()`, ...). That
// causes a cascade of 5xx on subsequent requests racing the still-held
// state. The server signals this handle on timeout, the decode loop polls
// `is_cancelled()` between tokens, and the closure returns early with an
// error so the locks release cleanly.
#[derive(Debug, Clone, Default)]
pub struct CancelHandle {
    flag: Arc<AtomicBool>,
    prefill_tokens_completed: Arc<AtomicU64>,
    prefill_progress_gauge: Option<Arc<AtomicU64>>,
}

impl CancelHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_prefill_progress_gauge(prefill_progress_gauge: Arc<AtomicU64>) -> Self {
        Self {
            flag: Arc::new(AtomicBool::new(false)),
            prefill_tokens_completed: Arc::new(AtomicU64::new(0)),
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
        let previous = self
            .prefill_tokens_completed
            .swap(completed, Ordering::SeqCst);
        if let Some(gauge) = &self.prefill_progress_gauge {
            if completed >= previous {
                gauge.fetch_add(completed - previous, Ordering::SeqCst);
            } else {
                gauge.fetch_sub(previous - completed, Ordering::SeqCst);
            }
        }
    }

    pub fn prefill_tokens_completed(&self) -> u64 {
        self.prefill_tokens_completed.load(Ordering::SeqCst)
    }

    pub fn clear_prefill_progress(&self) {
        let previous = self.prefill_tokens_completed.swap(0, Ordering::SeqCst);
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

        h.clear_prefill_progress();
        assert_eq!(h.prefill_tokens_completed(), 0);
        assert_eq!(gauge.load(Ordering::SeqCst), 0);
    }
}
