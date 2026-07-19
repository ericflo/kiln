//! Cooperative pacing and cancellation for base-model accelerator upload.

use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Serialize;

pub const ACCELERATOR_WEIGHT_UPLOAD_CANCELLATION_POLL_MILLISECONDS: u64 = 25;
pub const ACCELERATOR_WEIGHT_UPLOAD_CANCELLATION_BOUNDARY: &str =
    "reserve_before_base_and_each_layer; base_upload_then_transpose_then_pack_then_final";

const CANCELLATION_POLL: Duration =
    Duration::from_millis(ACCELERATOR_WEIGHT_UPLOAD_CANCELLATION_POLL_MILLISECONDS);
const PROGRESS_LAYER_INTERVAL: usize = 4;

/// Final or in-progress source-byte accounting for one base-model upload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct AcceleratorWeightUploadReport {
    pub stage: &'static str,
    pub configured_bytes_per_second: Option<u64>,
    pub source_bytes_completed: u64,
    pub source_bytes_total: u64,
    pub source_bytes_reserved: u64,
    pub completed_layers: usize,
    pub total_layers: usize,
    pub reserved_layers: usize,
    pub elapsed_milliseconds: u64,
    pub paced_milliseconds: u64,
    pub complete: bool,
}

/// Immutable process-startup policy for accelerator weight upload.
#[derive(Clone, Debug)]
pub struct AcceleratorWeightUploadPolicy {
    max_bytes_per_second: Option<u64>,
    cancellation: Arc<AtomicBool>,
    report: Arc<Mutex<AcceleratorWeightUploadReport>>,
}

impl AcceleratorWeightUploadPolicy {
    pub fn paced(max_bytes_per_second: u64, cancellation: Arc<AtomicBool>) -> Result<Self> {
        anyhow::ensure!(
            max_bytes_per_second > 0,
            "weight-upload rate must be nonzero"
        );
        Ok(Self {
            max_bytes_per_second: Some(max_bytes_per_second),
            cancellation,
            report: Arc::new(Mutex::new(AcceleratorWeightUploadReport {
                stage: "policy_resolved",
                configured_bytes_per_second: Some(max_bytes_per_second),
                source_bytes_completed: 0,
                source_bytes_total: 0,
                source_bytes_reserved: 0,
                completed_layers: 0,
                total_layers: 0,
                reserved_layers: 0,
                elapsed_milliseconds: 0,
                paced_milliseconds: 0,
                complete: false,
            })),
        })
    }

    pub fn unlimited() -> Self {
        Self::cancellable(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancellable(cancellation: Arc<AtomicBool>) -> Self {
        Self {
            max_bytes_per_second: None,
            cancellation,
            report: Arc::new(Mutex::new(AcceleratorWeightUploadReport {
                stage: "policy_resolved",
                configured_bytes_per_second: None,
                source_bytes_completed: 0,
                source_bytes_total: 0,
                source_bytes_reserved: 0,
                completed_layers: 0,
                total_layers: 0,
                reserved_layers: 0,
                elapsed_milliseconds: 0,
                paced_milliseconds: 0,
                complete: false,
            })),
        }
    }

    pub fn max_bytes_per_second(&self) -> Option<u64> {
        self.max_bytes_per_second
    }

    pub fn report(&self) -> AcceleratorWeightUploadReport {
        self.report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    pub fn ensure_active(&self) -> Result<()> {
        if self.cancellation.load(Ordering::Acquire) {
            return Err(AcceleratorWeightUploadCancelled.into());
        }
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
#[error("accelerator weight upload cancelled during shutdown")]
pub struct AcceleratorWeightUploadCancelled;

pub(crate) struct AcceleratorWeightUploadPacer<'a> {
    policy: &'a AcceleratorWeightUploadPolicy,
    started: Instant,
    source_bytes_total: usize,
    total_layers: usize,
    paced: Duration,
    source_bytes_completed: usize,
    completed_layers: usize,
    source_bytes_reserved: usize,
    reserved_layers: usize,
}

impl<'a> AcceleratorWeightUploadPacer<'a> {
    pub(crate) fn new(
        policy: &'a AcceleratorWeightUploadPolicy,
        source_bytes_total: usize,
        total_layers: usize,
    ) -> Result<Self> {
        policy.ensure_active()?;
        anyhow::ensure!(
            source_bytes_total > 0,
            "weight upload requires source bytes"
        );
        let pacer = Self {
            policy,
            started: Instant::now(),
            source_bytes_total,
            total_layers,
            paced: Duration::ZERO,
            source_bytes_completed: 0,
            completed_layers: 0,
            source_bytes_reserved: 0,
            reserved_layers: 0,
        };
        pacer.publish("started", false);
        Ok(pacer)
    }

    pub(crate) fn prepare(
        &mut self,
        stage: &'static str,
        source_bytes_reserved: usize,
        reserved_layers: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            source_bytes_reserved >= self.source_bytes_completed
                && source_bytes_reserved <= self.source_bytes_total,
            "weight-upload reservation is outside remaining source bytes"
        );
        anyhow::ensure!(
            reserved_layers >= self.completed_layers && reserved_layers <= self.total_layers,
            "weight-upload layer reservation is outside remaining layers"
        );
        self.source_bytes_reserved = source_bytes_reserved;
        self.reserved_layers = reserved_layers;
        self.publish(stage, false);
        self.wait_for_target(source_bytes_reserved, stage)?;
        self.publish(stage, false);
        tracing::info!(
            stage,
            source_bytes_completed = self.source_bytes_completed,
            source_bytes_reserved,
            source_bytes_total = self.source_bytes_total,
            completed_layers = self.completed_layers,
            reserved_layers,
            total_layers = self.total_layers,
            configured_mib_per_second = ?self
                .policy
                .max_bytes_per_second()
                .map(|rate| rate / (1024 * 1024)),
            elapsed_ms = self.started.elapsed().as_millis() as u64,
            paced_ms = self.paced.as_millis() as u64,
            "accelerator weight upload reservation ready"
        );
        Ok(())
    }

    pub(crate) fn boundary(&mut self, stage: &'static str) -> Result<()> {
        self.publish(stage, false);
        self.policy.ensure_active()?;
        tracing::info!(
            stage,
            source_bytes_completed = self.source_bytes_completed,
            source_bytes_reserved = self.source_bytes_reserved,
            completed_layers = self.completed_layers,
            reserved_layers = self.reserved_layers,
            elapsed_ms = self.started.elapsed().as_millis() as u64,
            paced_ms = self.paced.as_millis() as u64,
            "accelerator weight upload cooperative boundary"
        );
        Ok(())
    }

    pub(crate) fn checkpoint(
        &mut self,
        stage: &'static str,
        source_bytes_completed: usize,
        completed_layers: usize,
    ) -> Result<()> {
        anyhow::ensure!(
            source_bytes_completed <= self.source_bytes_total,
            "weight-upload progress exceeds source total"
        );
        anyhow::ensure!(
            completed_layers <= self.total_layers,
            "weight-upload layer progress exceeds layer total"
        );
        if stage == "complete" {
            anyhow::ensure!(
                source_bytes_completed == self.source_bytes_total
                    && completed_layers == self.total_layers,
                "completed weight-upload progress does not match its declared totals"
            );
        }
        self.source_bytes_completed = source_bytes_completed;
        self.completed_layers = completed_layers;
        self.source_bytes_reserved = source_bytes_completed;
        self.reserved_layers = completed_layers;
        self.publish(stage, false);
        self.policy.ensure_active()?;
        self.wait_for_target(source_bytes_completed, stage)?;
        self.policy.ensure_active()?;
        self.publish(stage, stage == "complete");

        if completed_layers == 0
            || completed_layers == self.total_layers
            || completed_layers % PROGRESS_LAYER_INTERVAL == 0
            || stage == "complete"
        {
            tracing::info!(
                stage,
                source_bytes_completed,
                source_bytes_total = self.source_bytes_total,
                completed_layers,
                total_layers = self.total_layers,
                configured_mib_per_second = ?self
                    .policy
                    .max_bytes_per_second()
                    .map(|rate| rate / (1024 * 1024)),
                elapsed_ms = self.started.elapsed().as_millis() as u64,
                paced_ms = self.paced.as_millis() as u64,
                "accelerator weight upload progress"
            );
        }
        Ok(())
    }

    fn wait_for_target(&mut self, source_bytes: usize, stage: &'static str) -> Result<()> {
        if let Some(rate) = self.policy.max_bytes_per_second() {
            let target = target_elapsed(source_bytes, rate);
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
                self.publish(stage, false);
            }
        }
        Ok(())
    }

    fn publish(&self, stage: &'static str, complete: bool) {
        let mut report = self
            .policy
            .report
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        *report = AcceleratorWeightUploadReport {
            stage,
            configured_bytes_per_second: self.policy.max_bytes_per_second(),
            source_bytes_completed: u64::try_from(self.source_bytes_completed).unwrap_or(u64::MAX),
            source_bytes_total: u64::try_from(self.source_bytes_total).unwrap_or(u64::MAX),
            source_bytes_reserved: u64::try_from(self.source_bytes_reserved).unwrap_or(u64::MAX),
            completed_layers: self.completed_layers,
            total_layers: self.total_layers,
            reserved_layers: self.reserved_layers,
            elapsed_milliseconds: self.started.elapsed().as_millis() as u64,
            paced_milliseconds: self.paced.as_millis() as u64,
            complete,
        };
    }
}

fn target_elapsed(source_bytes: usize, bytes_per_second: u64) -> Duration {
    let nanos = (source_bytes as u128)
        .saturating_mul(1_000_000_000)
        .div_ceil(bytes_per_second as u128);
    Duration::from_nanos(u64::try_from(nanos).unwrap_or(u64::MAX))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_bytes_convert_to_exact_elapsed_budget() {
        assert_eq!(
            target_elapsed(256 * 1024 * 1024, 512 * 1024 * 1024),
            Duration::from_millis(500)
        );
    }

    #[test]
    fn cancelled_policy_rejects_upload_before_work() -> Result<()> {
        let cancellation = Arc::new(AtomicBool::new(true));
        let policy = AcceleratorWeightUploadPolicy::paced(1024, cancellation)?;
        let error = AcceleratorWeightUploadPacer::new(&policy, 1024, 1)
            .err()
            .expect("cancelled policy must reject upload");
        assert!(
            error
                .downcast_ref::<AcceleratorWeightUploadCancelled>()
                .is_some()
        );
        Ok(())
    }

    #[test]
    fn progress_rejects_impossible_accounting() -> Result<()> {
        let policy = AcceleratorWeightUploadPolicy::unlimited();
        let mut pacer = AcceleratorWeightUploadPacer::new(&policy, 1024, 1)?;
        let error = pacer.checkpoint("layer", 1025, 1).unwrap_err();
        assert!(error.to_string().contains("exceeds source total"));
        Ok(())
    }

    #[test]
    fn pacing_wait_observes_cancellation() -> Result<()> {
        let cancellation = Arc::new(AtomicBool::new(false));
        let policy = AcceleratorWeightUploadPolicy::paced(1, Arc::clone(&cancellation))?;
        let mut pacer = AcceleratorWeightUploadPacer::new(&policy, 1, 1)?;
        let cancel_thread = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(10));
            cancellation.store(true, Ordering::Release);
        });
        let started = Instant::now();
        let error = pacer.prepare("base_reserved", 1, 1).unwrap_err();
        cancel_thread.join().unwrap();
        assert!(
            error
                .downcast_ref::<AcceleratorWeightUploadCancelled>()
                .is_some()
        );
        assert!(started.elapsed() < Duration::from_millis(500));
        let report = policy.report();
        assert_eq!(report.source_bytes_completed, 0);
        assert_eq!(report.source_bytes_reserved, 1);
        assert_eq!(report.reserved_layers, 1);
        assert!(!report.complete);
        Ok(())
    }

    #[test]
    fn completed_report_matches_declared_totals() -> Result<()> {
        let policy = AcceleratorWeightUploadPolicy::unlimited();
        let mut pacer = AcceleratorWeightUploadPacer::new(&policy, 1024, 2)?;
        pacer.prepare("base_reserved", 24, 0)?;
        pacer.boundary("base_embedding_uploaded")?;
        pacer.checkpoint("base", 24, 0)?;
        pacer.prepare("layer_reserved", 512, 1)?;
        pacer.checkpoint("layer", 512, 1)?;
        pacer.prepare("layer_reserved", 1024, 2)?;
        pacer.checkpoint("layer", 1024, 2)?;
        pacer.checkpoint("complete", 1024, 2)?;
        let report = policy.report();
        assert_eq!(report.stage, "complete");
        assert_eq!(report.source_bytes_completed, 1024);
        assert_eq!(report.source_bytes_total, 1024);
        assert_eq!(report.source_bytes_reserved, 1024);
        assert_eq!(report.completed_layers, 2);
        assert_eq!(report.reserved_layers, 2);
        assert_eq!(report.total_layers, 2);
        assert!(report.complete);
        Ok(())
    }
}
