//! Cooperative pacing and cancellation for startup checkpoint reads.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Serialize;

pub const CHECKPOINT_READ_CANCELLATION_POLL_MILLISECONDS: u64 = 25;
pub const CHECKPOINT_READ_PHASES: [&str; 3] = [
    "snapshot_copy",
    "initial_content_verification",
    "post_upload_content_verification",
];

const CANCELLATION_POLL: Duration =
    Duration::from_millis(CHECKPOINT_READ_CANCELLATION_POLL_MILLISECONDS);
const PROGRESS_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CheckpointReadPhaseReport {
    pub stage: &'static str,
    pub logical_bytes_completed: u64,
    pub logical_bytes_total: u64,
    pub rate_limited_bytes_completed: u64,
    pub elapsed_milliseconds: u64,
    pub paced_milliseconds: u64,
    pub complete: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize)]
pub struct CheckpointReadReport {
    pub configured_bytes_per_second: Option<u64>,
    pub snapshot_copy: CheckpointReadPhaseReport,
    pub initial_content_verification: CheckpointReadPhaseReport,
    pub post_upload_content_verification: CheckpointReadPhaseReport,
    pub complete: bool,
}

#[derive(Clone, Debug)]
pub struct CheckpointReadPolicy {
    max_bytes_per_second: Option<u64>,
    cancellation: Arc<AtomicBool>,
    report: Arc<Mutex<CheckpointReadReport>>,
}

impl CheckpointReadPolicy {
    pub fn paced(max_bytes_per_second: u64, cancellation: Arc<AtomicBool>) -> Result<Self> {
        anyhow::ensure!(
            max_bytes_per_second > 0,
            "checkpoint-read rate must be nonzero"
        );
        Ok(Self::new(Some(max_bytes_per_second), cancellation))
    }

    pub fn unlimited() -> Self {
        Self::cancellable(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancellable(cancellation: Arc<AtomicBool>) -> Self {
        Self::new(None, cancellation)
    }

    fn new(max_bytes_per_second: Option<u64>, cancellation: Arc<AtomicBool>) -> Self {
        Self {
            max_bytes_per_second,
            cancellation,
            report: Arc::new(Mutex::new(CheckpointReadReport {
                configured_bytes_per_second: max_bytes_per_second,
                ..CheckpointReadReport::default()
            })),
        }
    }

    pub fn max_bytes_per_second(&self) -> Option<u64> {
        self.max_bytes_per_second
    }

    pub fn report(&self) -> CheckpointReadReport {
        self.report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn ensure_active(&self) -> Result<()> {
        if self.cancellation.load(Ordering::Acquire) {
            return Err(CheckpointReadCancelled.into());
        }
        Ok(())
    }

    pub(crate) fn phase(
        &self,
        phase: CheckpointReadPhase,
        logical_bytes_total: u64,
    ) -> Result<CheckpointReadPacer<'_>> {
        CheckpointReadPacer::new(self, phase, logical_bytes_total)
    }
}

#[derive(Debug, thiserror::Error)]
#[error("checkpoint read cancelled during shutdown")]
pub struct CheckpointReadCancelled;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CheckpointReadPhase {
    SnapshotCopy,
    InitialContentVerification,
    PostUploadContentVerification,
}

impl CheckpointReadPhase {
    fn name(self) -> &'static str {
        match self {
            Self::SnapshotCopy => CHECKPOINT_READ_PHASES[0],
            Self::InitialContentVerification => CHECKPOINT_READ_PHASES[1],
            Self::PostUploadContentVerification => CHECKPOINT_READ_PHASES[2],
        }
    }
}

pub(crate) struct CheckpointReadPacer<'a> {
    policy: &'a CheckpointReadPolicy,
    phase: CheckpointReadPhase,
    started: Instant,
    logical_bytes_total: u64,
    paced: Duration,
    last_logged_bytes: u64,
}

impl<'a> CheckpointReadPacer<'a> {
    fn new(
        policy: &'a CheckpointReadPolicy,
        phase: CheckpointReadPhase,
        logical_bytes_total: u64,
    ) -> Result<Self> {
        policy.ensure_active()?;
        let pacer = Self {
            policy,
            phase,
            started: Instant::now(),
            logical_bytes_total,
            paced: Duration::ZERO,
            last_logged_bytes: 0,
        };
        pacer.publish(0, 0, false);
        Ok(pacer)
    }

    pub(crate) fn checkpoint(
        &mut self,
        logical_bytes_completed: u64,
        rate_limited_bytes_completed: u64,
    ) -> Result<()> {
        self.update(logical_bytes_completed, rate_limited_bytes_completed, false)
    }

    pub(crate) fn complete(
        &mut self,
        logical_bytes_completed: u64,
        rate_limited_bytes_completed: u64,
    ) -> Result<()> {
        anyhow::ensure!(
            logical_bytes_completed == self.logical_bytes_total,
            "checkpoint-read phase {} completed {logical_bytes_completed} logical bytes, expected {}",
            self.phase.name(),
            self.logical_bytes_total
        );
        self.update(logical_bytes_completed, rate_limited_bytes_completed, true)
    }

    fn update(
        &mut self,
        logical_bytes_completed: u64,
        rate_limited_bytes_completed: u64,
        complete: bool,
    ) -> Result<()> {
        anyhow::ensure!(
            logical_bytes_completed <= self.logical_bytes_total,
            "checkpoint-read phase {} exceeded its logical byte total",
            self.phase.name()
        );
        anyhow::ensure!(
            rate_limited_bytes_completed <= logical_bytes_completed,
            "checkpoint-read phase {} rate-limited bytes exceeded logical progress",
            self.phase.name()
        );
        self.publish(logical_bytes_completed, rate_limited_bytes_completed, false);
        self.policy.ensure_active()?;
        if let Some(rate) = self.policy.max_bytes_per_second() {
            let target = target_elapsed(rate_limited_bytes_completed, rate);
            loop {
                self.policy.ensure_active()?;
                let elapsed = self.started.elapsed();
                if elapsed >= target {
                    break;
                }
                let sleep_for = target.saturating_sub(elapsed).min(CANCELLATION_POLL);
                let sleep_started = Instant::now();
                std::thread::sleep(sleep_for);
                self.paced = self.paced.saturating_add(sleep_started.elapsed());
                self.publish(logical_bytes_completed, rate_limited_bytes_completed, false);
            }
        }
        self.policy.ensure_active()?;
        self.publish(
            logical_bytes_completed,
            rate_limited_bytes_completed,
            complete,
        );

        let should_log = complete
            || logical_bytes_completed.saturating_sub(self.last_logged_bytes) >= PROGRESS_BYTES;
        if should_log {
            self.last_logged_bytes = logical_bytes_completed;
            tracing::info!(
                phase = self.phase.name(),
                logical_bytes_completed,
                logical_bytes_total = self.logical_bytes_total,
                rate_limited_bytes_completed,
                configured_mib_per_second = ?self
                    .policy
                    .max_bytes_per_second()
                    .map(|rate| rate / (1024 * 1024)),
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                paced_ms = self.paced.as_millis() as u64,
                complete,
                "checkpoint read progress"
            );
        }
        Ok(())
    }

    fn publish(
        &self,
        logical_bytes_completed: u64,
        rate_limited_bytes_completed: u64,
        complete: bool,
    ) {
        let phase_report = CheckpointReadPhaseReport {
            stage: self.phase.name(),
            logical_bytes_completed,
            logical_bytes_total: self.logical_bytes_total,
            rate_limited_bytes_completed,
            elapsed_milliseconds: self.started.elapsed().as_millis() as u64,
            paced_milliseconds: self.paced.as_millis() as u64,
            complete,
        };
        let mut report = self
            .policy
            .report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match self.phase {
            CheckpointReadPhase::SnapshotCopy => report.snapshot_copy = phase_report,
            CheckpointReadPhase::InitialContentVerification => {
                report.initial_content_verification = phase_report
            }
            CheckpointReadPhase::PostUploadContentVerification => {
                report.post_upload_content_verification = phase_report
            }
        }
        report.complete = report.snapshot_copy.complete
            && report.initial_content_verification.complete
            && report.post_upload_content_verification.complete;
    }
}

fn target_elapsed(bytes: u64, bytes_per_second: u64) -> Duration {
    let nanos = (bytes as u128)
        .saturating_mul(1_000_000_000)
        .div_ceil(bytes_per_second as u128);
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cloned_logical_bytes_do_not_consume_read_rate() -> Result<()> {
        let policy = CheckpointReadPolicy::paced(1, Arc::new(AtomicBool::new(false)))?;
        let mut pacer = policy.phase(CheckpointReadPhase::SnapshotCopy, 1024)?;
        pacer.complete(1024, 0)?;
        let report = policy.report().snapshot_copy;
        assert_eq!(report.logical_bytes_completed, 1024);
        assert_eq!(report.rate_limited_bytes_completed, 0);
        assert_eq!(report.paced_milliseconds, 0);
        assert!(report.complete);
        Ok(())
    }

    #[test]
    fn cancellation_interrupts_a_pacing_wait() -> Result<()> {
        let cancellation = Arc::new(AtomicBool::new(false));
        let policy = CheckpointReadPolicy::paced(1, Arc::clone(&cancellation))?;
        let mut pacer = policy.phase(CheckpointReadPhase::InitialContentVerification, 1)?;
        let cancel_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            cancellation.store(true, Ordering::Release);
        });
        let started = Instant::now();
        let error = pacer.complete(1, 1).unwrap_err();
        cancel_thread.join().unwrap();
        assert!(error.downcast_ref::<CheckpointReadCancelled>().is_some());
        assert!(started.elapsed() < Duration::from_millis(500));
        assert!(!policy.report().initial_content_verification.complete);
        Ok(())
    }
}
