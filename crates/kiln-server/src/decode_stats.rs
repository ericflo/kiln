//! Live decode performance ring buffer.
//!
//! Records the wall-clock instant at which each streaming completion token was
//! emitted. The most recent samples (within a fixed time window) are used to
//! compute live tokens-per-second and inter-token latency percentiles for
//! display on the /ui dashboard.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;

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
    pub mean_itl_ms: f64,
    pub sample_count: usize,
    pub window_secs: f64,
}

impl DecodeStatsSnapshot {
    fn empty() -> Self {
        Self {
            tok_per_sec: 0.0,
            p50_itl_ms: 0.0,
            p99_itl_ms: 0.0,
            mean_itl_ms: 0.0,
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
pub struct DecodeStatsRing {
    samples: VecDeque<Instant>,
    capacity: usize,
}

impl DecodeStatsRing {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record a token emit at `now`. Evicts the oldest sample if over capacity
    /// and any samples older than the rolling window.
    pub fn record_token(&mut self, now: Instant) {
        self.evict_old(now);
        if self.samples.len() >= self.capacity {
            self.samples.pop_front();
        }
        self.samples.push_back(now);
    }

    /// Compute a snapshot of inter-token latency and tok/s over the rolling
    /// window ending at `now`.
    pub fn snapshot(&self, now: Instant) -> DecodeStatsSnapshot {
        if self.samples.len() < 2 {
            return DecodeStatsSnapshot::empty();
        }

        let cutoff = now.checked_sub(WINDOW).unwrap_or(now);
        let recent: Vec<Instant> = self
            .samples
            .iter()
            .copied()
            .filter(|t| *t >= cutoff)
            .collect();

        if recent.len() < 2 {
            return DecodeStatsSnapshot::empty();
        }

        let mut deltas_ms: Vec<f64> = recent
            .windows(2)
            .map(|w| (w[1] - w[0]).as_secs_f64() * 1000.0)
            .collect();
        deltas_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let p50_itl_ms = percentile(&deltas_ms, 0.50);
        let p99_itl_ms = percentile(&deltas_ms, 0.99);
        let mean_itl_ms = deltas_ms.iter().sum::<f64>() / deltas_ms.len() as f64;

        // tok/s is computed over the actual span covered by the recent samples,
        // not the full window. This avoids understating throughput when the
        // server has only been generating for a fraction of the window.
        let span_secs = (recent[recent.len() - 1] - recent[0]).as_secs_f64();
        let tok_per_sec = if span_secs > 0.0 {
            (recent.len() - 1) as f64 / span_secs
        } else {
            0.0
        };

        DecodeStatsSnapshot {
            tok_per_sec,
            p50_itl_ms,
            p99_itl_ms,
            mean_itl_ms,
            sample_count: deltas_ms.len(),
            window_secs: WINDOW.as_secs_f64(),
        }
    }

    fn evict_old(&mut self, now: Instant) {
        let cutoff = now.checked_sub(WINDOW).unwrap_or(now);
        while let Some(front) = self.samples.front() {
            if *front < cutoff {
                self.samples.pop_front();
            } else {
                break;
            }
        }
    }
}

/// Linear-interpolation percentile of a pre-sorted slice.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_zeros() {
        let ring = DecodeStatsRing::new(16);
        let snap = ring.snapshot(Instant::now());
        assert_eq!(snap.sample_count, 0);
        assert_eq!(snap.tok_per_sec, 0.0);
        assert_eq!(snap.p50_itl_ms, 0.0);
        assert_eq!(snap.p99_itl_ms, 0.0);
        assert_eq!(snap.mean_itl_ms, 0.0);
        assert!((snap.window_secs - 60.0).abs() < 1e-9);
    }

    #[test]
    fn single_sample_returns_zeros() {
        let mut ring = DecodeStatsRing::new(16);
        let t0 = Instant::now();
        ring.record_token(t0);
        let snap = ring.snapshot(t0);
        assert_eq!(snap.sample_count, 0);
        assert_eq!(snap.tok_per_sec, 0.0);
    }

    #[test]
    fn two_samples_compute_itl() {
        let mut ring = DecodeStatsRing::new(16);
        let t0 = Instant::now();
        ring.record_token(t0);
        ring.record_token(t0 + Duration::from_millis(20));
        let snap = ring.snapshot(t0 + Duration::from_millis(20));
        assert_eq!(snap.sample_count, 1);
        assert!((snap.p50_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.p99_itl_ms - 20.0).abs() < 0.01);
        assert!((snap.mean_itl_ms - 20.0).abs() < 0.01);
        // 1 inter-token gap over a 20ms span = 50 tok/s
        assert!((snap.tok_per_sec - 50.0).abs() < 0.01);
    }

    #[test]
    fn percentile_correctness_using_known_distribution() {
        let mut ring = DecodeStatsRing::new(256);
        let t0 = Instant::now();
        // 100 samples 10ms apart (99 inter-token gaps of 10ms each).
        for i in 0..100 {
            ring.record_token(t0 + Duration::from_millis(10 * i));
        }
        let snap = ring.snapshot(t0 + Duration::from_millis(10 * 99));
        assert_eq!(snap.sample_count, 99);
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

        // Now inject 5 outliers spaced 100ms apart and verify p99 jumps.
        // With 99 baseline gaps of 10ms + 5 outlier gaps of 100ms (104 total),
        // p99 lands inside the outlier tail and the value should be well above
        // the baseline 10ms.
        let mut next = t0 + Duration::from_millis(10 * 99);
        for _ in 0..5 {
            next += Duration::from_millis(100);
            ring.record_token(next);
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
    }

    #[test]
    fn capacity_eviction() {
        let mut ring = DecodeStatsRing::new(4);
        let t0 = Instant::now();
        for i in 0..10 {
            // Spaced 1ms apart so all stay within the 60s window.
            ring.record_token(t0 + Duration::from_millis(i));
        }
        // Capacity is 4, so only the last 4 timestamps should be retained,
        // giving 3 inter-token gaps.
        assert_eq!(ring.samples.len(), 4);
        let snap = ring.snapshot(t0 + Duration::from_millis(9));
        assert_eq!(snap.sample_count, 3);
    }

    #[test]
    fn evicts_samples_older_than_window() {
        let mut ring = DecodeStatsRing::new(256);
        let t0 = Instant::now();
        // Drop 3 samples at t0, then jump 120s forward and drop 3 more.
        for i in 0..3 {
            ring.record_token(t0 + Duration::from_millis(i));
        }
        let later = t0 + Duration::from_secs(120);
        for i in 0..3 {
            ring.record_token(later + Duration::from_millis(i));
        }
        let snap = ring.snapshot(later + Duration::from_millis(2));
        // Only the 3 recent samples should count → 2 inter-token gaps.
        assert_eq!(snap.sample_count, 2);
    }

    #[test]
    fn snapshot_does_not_mutate() {
        let mut ring = DecodeStatsRing::new(16);
        let t0 = Instant::now();
        ring.record_token(t0);
        ring.record_token(t0 + Duration::from_millis(5));
        let _ = ring.snapshot(t0 + Duration::from_millis(5));
        assert_eq!(ring.samples.len(), 2);
    }
}
