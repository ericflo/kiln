//! Live decode performance ring buffer.
//!
//! Records request-local inter-token gaps. A single global token timestamp ring
//! would treat two concurrent requests' adjacent emissions as one request's
//! ITL and produce deceptively small values at batch size greater than one.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;

use crate::latency_observability::{
    ABSOLUTE_STALL_THRESHOLD, LatencyStallReasonCounts, TokenGapObservation, duration_ms,
    percentile,
};

/// Window over which live decode stats are aggregated.
const WINDOW: Duration = Duration::from_secs(60);

/// Snapshot of recent decode performance.
///
/// All latency fields are in milliseconds. `sample_count` is the number of
/// inter-token gaps used to compute the snapshot — at least 2 token samples
/// are needed to derive a single inter-token gap, so values < 2 sample_count
/// indicate insufficient data and the latency fields will be 0.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct DecodeStatsSnapshot {
    pub tok_per_sec: f64,
    pub p50_itl_ms: f64,
    pub p99_itl_ms: f64,
    pub p999_itl_ms: f64,
    pub mean_itl_ms: f64,
    pub max_itl_ms: f64,
    pub stall_threshold_ms: f64,
    pub stall_count: u64,
    pub unexplained_stall_count: u64,
    pub stall_reasons: LatencyStallReasonCounts,
    pub sample_count: usize,
    pub window_secs: f64,
}

impl DecodeStatsSnapshot {
    fn empty() -> Self {
        Self {
            tok_per_sec: 0.0,
            p50_itl_ms: 0.0,
            p99_itl_ms: 0.0,
            p999_itl_ms: 0.0,
            mean_itl_ms: 0.0,
            max_itl_ms: 0.0,
            stall_threshold_ms: duration_ms(ABSOLUTE_STALL_THRESHOLD),
            stall_count: 0,
            unexplained_stall_count: 0,
            stall_reasons: LatencyStallReasonCounts::default(),
            sample_count: 0,
            window_secs: WINDOW.as_secs_f64(),
        }
    }
}

/// Bounded ring buffer of token-emit timestamps.
///
/// Old samples (older than `WINDOW`) are evicted on `record_token` and
/// `snapshot`. The ring is also capped at `capacity` to bound memory under
/// pathological burstiness.
#[derive(Debug, Clone, Copy)]
struct TimedGap {
    observed_at: Instant,
    observation: TokenGapObservation,
}

pub struct DecodeStatsRing {
    samples: VecDeque<TimedGap>,
    capacity: usize,
}

impl DecodeStatsRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record one request-local gap at the time it reached the stream handler.
    pub fn record_gap(&mut self, observed_at: Instant, observation: TokenGapObservation) {
        self.evict_old(observed_at);
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(TimedGap {
            observed_at,
            observation,
        });
    }

    /// Compute a snapshot of inter-token latency and tok/s over the rolling
    /// window ending at `now`.
    pub fn snapshot(&self, now: Instant) -> DecodeStatsSnapshot {
        if self.samples.is_empty() {
            return DecodeStatsSnapshot::empty();
        }

        let cutoff = now.checked_sub(WINDOW).unwrap_or(now);
        let recent: Vec<TokenGapObservation> = self
            .samples
            .iter()
            .filter(|sample| sample.observed_at >= cutoff)
            .map(|sample| sample.observation)
            .collect();

        if recent.is_empty() {
            return DecodeStatsSnapshot::empty();
        }

        let mut deltas_ms: Vec<f64> = recent
            .iter()
            .map(|sample| duration_ms(sample.gap))
            .collect();
        deltas_ms.sort_by(f64::total_cmp);

        let p50_itl_ms = percentile(&deltas_ms, 0.50).unwrap_or(0.0);
        let p99_itl_ms = percentile(&deltas_ms, 0.99).unwrap_or(0.0);
        let p999_itl_ms = percentile(&deltas_ms, 0.999).unwrap_or(0.0);
        let mean_itl_ms = deltas_ms.iter().sum::<f64>() / deltas_ms.len() as f64;
        let max_itl_ms = deltas_ms.last().copied().unwrap_or(0.0);
        let stall_threshold_ms = duration_ms(ABSOLUTE_STALL_THRESHOLD).max(p50_itl_ms * 5.0);
        let mut stall_reasons = LatencyStallReasonCounts::default();
        for sample in &recent {
            if duration_ms(sample.gap) >= stall_threshold_ms {
                stall_reasons.increment(sample.reason);
            }
        }
        let stall_count = stall_reasons.total();

        // One request-local gap represents one subsequently emitted token.
        // Summing gaps avoids counting cross-request adjacency as throughput.
        let span_secs = recent
            .iter()
            .map(|sample| sample.gap.as_secs_f64())
            .sum::<f64>();
        let tok_per_sec = if span_secs > 0.0 {
            recent.len() as f64 / span_secs
        } else {
            0.0
        };

        DecodeStatsSnapshot {
            tok_per_sec,
            p50_itl_ms,
            p99_itl_ms,
            p999_itl_ms,
            mean_itl_ms,
            max_itl_ms,
            stall_threshold_ms,
            stall_count,
            unexplained_stall_count: stall_reasons.unexplained,
            stall_reasons,
            sample_count: deltas_ms.len(),
            window_secs: WINDOW.as_secs_f64(),
        }
    }

    fn evict_old(&mut self, now: Instant) {
        let cutoff = now.checked_sub(WINDOW).unwrap_or(now);
        while let Some(front) = self.samples.front() {
            if front.observed_at < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::latency_observability::LatencyStallReason;

    fn gap(milliseconds: u64, reason: LatencyStallReason) -> TokenGapObservation {
        TokenGapObservation {
            gap: Duration::from_millis(milliseconds),
            reason,
            attributed_duration: Duration::from_millis(milliseconds),
        }
    }

    #[test]
    fn empty_returns_zeros() {
        let ring = DecodeStatsRing::new(16);
        let snap = ring.snapshot(Instant::now());
        assert_eq!(snap.sample_count, 0);
        assert_eq!(snap.tok_per_sec, 0.0);
        assert_eq!(snap.p50_itl_ms, 0.0);
        assert_eq!(snap.p99_itl_ms, 0.0);
        assert_eq!(snap.p999_itl_ms, 0.0);
        assert_eq!(snap.mean_itl_ms, 0.0);
        assert_eq!(snap.stall_count, 0);
        assert!((snap.window_secs - 60.0).abs() < 1e-9);
    }

    #[test]
    fn one_request_local_gap_is_a_complete_sample() {
        let mut ring = DecodeStatsRing::new(16);
        let t0 = Instant::now();
        ring.record_gap(t0, gap(20, LatencyStallReason::ActorDecode));
        let snap = ring.snapshot(t0);
        assert_eq!(snap.sample_count, 1);
        assert!((snap.p50_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.p99_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.p999_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.mean_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.tok_per_sec - 50.0).abs() < 0.01);
    }

    #[test]
    fn percentile_correctness_using_known_distribution() {
        let mut ring = DecodeStatsRing::new(256);
        let t0 = Instant::now();
        // One hundred request-local gaps of 10ms each.
        for i in 0..100 {
            ring.record_gap(
                t0 + Duration::from_millis(10 * i),
                gap(10, LatencyStallReason::ActorDecode),
            );
        }
        let snap = ring.snapshot(t0 + Duration::from_millis(10 * 99));
        assert_eq!(snap.sample_count, 100);
        assert!(
            (snap.p50_itl_ms - 10.0).abs() < 0.5,
            "p50 should be ~10ms, got {}",
            snap.p50_itl_ms
        );
        assert!(
            (snap.p99_itl_ms - 10.0).abs() < 0.5,
            "p99 should be ~10ms, got {}",
            snap.p99_itl_ms
        );

        // Five one-second gaps move the tail and cross the dynamic stall gate.
        let mut next = t0 + Duration::from_millis(10 * 99);
        for _ in 0..5 {
            next += Duration::from_millis(100);
            ring.record_gap(next, gap(1_000, LatencyStallReason::ActorPrefill));
        }
        let snap2 = ring.snapshot(next);
        assert!(
            snap2.p99_itl_ms > 50.0,
            "p99 should jump after 5x 100ms outliers, got {}",
            snap2.p99_itl_ms
        );
        // p50 should still be ~10ms (5 outliers in 100+ samples don't move
        // the median).
        assert!(
            (snap2.p50_itl_ms - 10.0).abs() < 1.0,
            "p50 should still be ~10ms, got {}",
            snap2.p50_itl_ms
        );
        assert_eq!(snap2.stall_count, 5);
        assert_eq!(snap2.stall_reasons.actor_prefill, 5);
        assert_eq!(snap2.unexplained_stall_count, 0);
    }

    #[test]
    fn capacity_eviction() {
        let mut ring = DecodeStatsRing::new(4);
        let t0 = Instant::now();
        for i in 0..10 {
            ring.record_gap(
                t0 + Duration::from_millis(i),
                gap(1, LatencyStallReason::ActorDecode),
            );
        }
        assert_eq!(ring.samples.len(), 4);
        let snap = ring.snapshot(t0 + Duration::from_millis(9));
        assert_eq!(snap.sample_count, 4);
    }

    #[test]
    fn evicts_samples_older_than_window() {
        let mut ring = DecodeStatsRing::new(256);
        let t0 = Instant::now();
        // Drop three samples at t0, then jump 120s and add three more.
        for i in 0..3 {
            ring.record_gap(
                t0 + Duration::from_millis(i),
                gap(1, LatencyStallReason::ActorDecode),
            );
        }
        let later = t0 + Duration::from_secs(120);
        for i in 0..3 {
            ring.record_gap(
                later + Duration::from_millis(i),
                gap(1, LatencyStallReason::ActorDecode),
            );
        }
        let snap = ring.snapshot(later + Duration::from_millis(2));
        assert_eq!(snap.sample_count, 3);
    }

    #[test]
    fn snapshot_does_not_mutate() {
        let mut ring = DecodeStatsRing::new(16);
        let t0 = Instant::now();
        ring.record_gap(t0, gap(5, LatencyStallReason::ActorDecode));
        ring.record_gap(
            t0 + Duration::from_millis(5),
            gap(5, LatencyStallReason::ActorDecode),
        );
        let _ = ring.snapshot(t0 + Duration::from_millis(5));
        assert_eq!(ring.samples.len(), 2);
    }
}
