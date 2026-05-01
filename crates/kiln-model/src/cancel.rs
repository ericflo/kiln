use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

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
}

impl CancelHandle {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::SeqCst)
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
}
