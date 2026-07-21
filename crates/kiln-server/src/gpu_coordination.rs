//! Shared inference/training GPU ownership and causal writer attribution.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kiln_model::BackendHealthHandle;
use tokio::sync::{OwnedRwLockReadGuard, OwnedRwLockWriteGuard, RwLock};

use crate::latency_observability::{
    BackendPhaseDurations, BlockingBackendPhase, BlockingBackendPhaseTracker,
};

/// Coordination lock for GPU memory sharing between inference and training.
///
/// Inference acquires a read lock (multiple concurrent inference requests are
/// permitted). Training and memory mutations acquire a write lock so their
/// accelerator ownership cannot overlap inference.
pub type GpuCoordinationLock = Arc<RwLock<()>>;

#[cfg(test)]
pub(crate) fn read_guard(gpu_lock: &GpuCoordinationLock) -> OwnedRwLockReadGuard<()> {
    futures::executor::block_on(gpu_lock.clone().read_owned())
}

/// Acquire shared inference ownership and attribute only the exclusive writer
/// intervals that actually overlapped this read wait. The uncontended path
/// avoids touching the interval tracker.
pub(crate) fn read_guard_with_phases(
    gpu_lock: &GpuCoordinationLock,
    phase_tracker: &BlockingBackendPhaseTracker,
) -> (OwnedRwLockReadGuard<()>, BackendPhaseDurations) {
    let wait_started = Instant::now();
    let (guard, blocker_phases) = match gpu_lock.clone().try_read_owned() {
        Ok(guard) => (guard, BackendPhaseDurations::default()),
        Err(_) => {
            let before = phase_tracker.snapshot();
            let guard = futures::executor::block_on(gpu_lock.clone().read_owned());
            (guard, phase_tracker.observed_since(before))
        }
    };
    let mut phases = blocker_phases;
    phases.observe_gpu_lock_wait(wait_started.elapsed());
    (guard, phases)
}

impl kiln_train::trainer::GpuStepWriterObserver for BlockingBackendPhaseTracker {
    fn writer_acquired(self: Arc<Self>) -> Box<dyn Send> {
        Box::new(self.begin(BlockingBackendPhase::Training))
    }
}

#[cfg(test)]
pub(crate) fn write_guard(gpu_lock: &GpuCoordinationLock) -> OwnedRwLockWriteGuard<()> {
    futures::executor::block_on(gpu_lock.clone().write_owned())
}

const HEALTH_POLL: Duration = Duration::from_millis(5);

/// Wait for exclusive GPU ownership without entering an uninterruptible wait
/// behind an inference owner whose completion state has been quarantined.
pub(crate) fn write_guard_while_healthy(
    gpu_lock: &GpuCoordinationLock,
    backend_health: &BackendHealthHandle,
) -> anyhow::Result<OwnedRwLockWriteGuard<()>> {
    loop {
        backend_health.ensure_healthy()?;
        if let Ok(guard) = gpu_lock.clone().try_write_owned() {
            backend_health.ensure_healthy()?;
            return Ok(guard);
        }
        std::thread::sleep(HEALTH_POLL);
    }
}

/// Async counterpart of [`write_guard_while_healthy`].
pub(crate) async fn write_guard_while_healthy_async(
    gpu_lock: &GpuCoordinationLock,
    backend_health: &BackendHealthHandle,
) -> anyhow::Result<OwnedRwLockWriteGuard<()>> {
    loop {
        backend_health.ensure_healthy()?;
        if let Ok(guard) = gpu_lock.clone().try_write_owned() {
            backend_health.ensure_healthy()?;
            return Ok(guard);
        }
        tokio::time::sleep(HEALTH_POLL).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn read_owner_moves_and_excludes_writer_until_drop() {
        let gpu_lock: GpuCoordinationLock = Arc::new(RwLock::new(()));
        let read_owner = read_guard(&gpu_lock);
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            ready_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            drop(read_owner);
        });
        ready_rx.recv().unwrap();
        assert!(gpu_lock.try_write().is_err());
        release_tx.send(()).unwrap();
        reader.join().unwrap();
        assert!(gpu_lock.try_write().is_ok());
    }

    #[test]
    fn reader_attributes_only_overlapping_writer_phase() {
        let gpu_lock: GpuCoordinationLock = Arc::new(RwLock::new(()));
        let phase_tracker = Arc::new(BlockingBackendPhaseTracker::default());
        let writer = gpu_lock.clone().try_write_owned().unwrap();
        let training_phase = phase_tracker.begin(BlockingBackendPhase::Training);
        let reader_lock = gpu_lock.clone();
        let reader_phases = phase_tracker.clone();
        let (started_tx, started_rx) = std::sync::mpsc::channel();
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let reader = std::thread::spawn(move || {
            started_tx.send(()).unwrap();
            let (guard, phases) = read_guard_with_phases(&reader_lock, &reader_phases);
            result_tx.send(phases).unwrap();
            drop(guard);
        });

        started_rx.recv().unwrap();
        std::thread::sleep(Duration::from_millis(10));
        drop(training_phase);
        drop(writer);
        let phases = result_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        reader.join().unwrap();

        assert!(phases.gpu_lock_wait >= Some(Duration::from_millis(5)));
        assert!(phases.training >= Some(Duration::from_millis(5)));
        assert_eq!(phases.resize, None);
        assert_eq!(phases.trim, None);
        assert_eq!(phases.adapter, None);
    }

    #[test]
    fn health_checked_writer_rejects_without_waiting_for_retained_reader() {
        let gpu_lock: GpuCoordinationLock = Arc::new(RwLock::new(()));
        let retained_reader = read_guard(&gpu_lock);
        let backend_health = BackendHealthHandle::default();
        let worker_lock = gpu_lock.clone();
        let worker_health = backend_health.clone();
        let (result_tx, result_rx) = std::sync::mpsc::channel();

        std::thread::spawn(move || {
            result_tx
                .send(write_guard_while_healthy(&worker_lock, &worker_health).map(drop))
                .unwrap();
        });

        assert!(result_rx.recv_timeout(Duration::from_millis(25)).is_err());
        backend_health.quarantine("injected unknown inference completion");
        let error = result_rx
            .recv_timeout(Duration::from_millis(250))
            .expect("quarantine must interrupt the writer wait")
            .expect_err("quarantined writer must reject");
        assert!(error.to_string().contains("requires restart"));
        assert!(gpu_lock.try_write().is_err());
        std::mem::forget(retained_reader);
    }

    #[tokio::test]
    async fn async_health_checked_writer_rejects_retained_reader() {
        let gpu_lock: GpuCoordinationLock = Arc::new(RwLock::new(()));
        let retained_reader = gpu_lock.clone().read_owned().await;
        let backend_health = BackendHealthHandle::default();
        let worker_lock = gpu_lock.clone();
        let worker_health = backend_health.clone();
        let writer = tokio::spawn(async move {
            write_guard_while_healthy_async(&worker_lock, &worker_health)
                .await
                .map(drop)
        });

        tokio::time::sleep(Duration::from_millis(25)).await;
        assert!(!writer.is_finished());
        backend_health.quarantine("injected async unknown inference completion");
        let error = tokio::time::timeout(Duration::from_millis(250), writer)
            .await
            .expect("quarantine must interrupt the async writer wait")
            .unwrap()
            .expect_err("quarantined async writer must reject");
        assert!(error.to_string().contains("requires restart"));
        assert!(gpu_lock.try_write().is_err());
        std::mem::forget(retained_reader);
    }
}
