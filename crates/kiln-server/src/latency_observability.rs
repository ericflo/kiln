//! Bounded, request-aware latency attribution for inference streams.
//!
//! The batching actor is single-threaded, so actor phase durations accumulated
//! between two tokens are causal blocking candidates for every request that
//! was active during those phases. Candidate durations may overlap (response
//! delivery can proceed while the actor runs another forward), so they are not
//! presented as an additive wall-clock decomposition.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use serde::Serialize;

pub const ABSOLUTE_STALL_THRESHOLD: Duration = Duration::from_millis(250);
const RELATIVE_STALL_MULTIPLIER: f64 = 5.0;
const MAX_RETAINED_REQUEST_GAPS: usize = 8_192;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendPhaseDurations {
    pub sampling: Option<Duration>,
    pub readback: Option<Duration>,
    pub gpu_lock_wait: Option<Duration>,
    pub graph_capture: Option<Duration>,
    pub graph_replay: Option<Duration>,
    pub synchronization: Option<Duration>,
    pub resize: Option<Duration>,
    pub trim: Option<Duration>,
    pub adapter: Option<Duration>,
    pub training: Option<Duration>,
}

impl BackendPhaseDurations {
    fn add_observation(total: &mut Option<Duration>, duration: Duration) {
        *total = Some(total.unwrap_or(Duration::ZERO).saturating_add(duration));
    }

    pub fn observe_sampling(&mut self, duration: Duration) {
        Self::add_observation(&mut self.sampling, duration);
    }

    pub fn observe_readback(&mut self, duration: Duration) {
        Self::add_observation(&mut self.readback, duration);
    }

    pub fn observe_gpu_lock_wait(&mut self, duration: Duration) {
        Self::add_observation(&mut self.gpu_lock_wait, duration);
    }

    pub fn observe_graph_capture(&mut self, duration: Duration) {
        Self::add_observation(&mut self.graph_capture, duration);
    }

    pub fn observe_graph_replay(&mut self, duration: Duration) {
        Self::add_observation(&mut self.graph_replay, duration);
    }

    pub fn observe_synchronization(&mut self, duration: Duration) {
        Self::add_observation(&mut self.synchronization, duration);
    }

    pub fn observe_resize(&mut self, duration: Duration) {
        Self::add_observation(&mut self.resize, duration);
    }

    pub fn observe_trim(&mut self, duration: Duration) {
        Self::add_observation(&mut self.trim, duration);
    }

    pub fn observe_adapter(&mut self, duration: Duration) {
        Self::add_observation(&mut self.adapter, duration);
    }

    pub fn observe_training(&mut self, duration: Duration) {
        Self::add_observation(&mut self.training, duration);
    }

    fn add_to(self, totals: &mut Self) {
        for (total, observation) in [
            (&mut totals.sampling, self.sampling),
            (&mut totals.readback, self.readback),
            (&mut totals.gpu_lock_wait, self.gpu_lock_wait),
            (&mut totals.graph_capture, self.graph_capture),
            (&mut totals.graph_replay, self.graph_replay),
            (&mut totals.synchronization, self.synchronization),
            (&mut totals.resize, self.resize),
            (&mut totals.trim, self.trim),
            (&mut totals.adapter, self.adapter),
            (&mut totals.training, self.training),
        ] {
            if let Some(duration) = observation {
                Self::add_observation(total, duration);
            }
        }
    }

    fn max_covered_duration(self) -> Duration {
        [
            self.sampling,
            self.readback,
            self.gpu_lock_wait,
            self.graph_capture,
            self.graph_replay,
            self.synchronization,
            self.resize,
            self.trim,
            self.adapter,
            self.training,
        ]
        .into_iter()
        .flatten()
        .max()
        .unwrap_or(Duration::ZERO)
    }

    fn actor_decode_residual(self, actor_decode: Duration) -> Duration {
        // Backend phases are narrower candidates contained by the broad actor
        // decode wall interval. Subtract only the largest observed candidate:
        // the detailed phases may overlap, so summing them would manufacture
        // coverage and could erase genuine actor-side decode work.
        actor_decode.saturating_sub(self.max_covered_duration())
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct TokenPhaseDurations {
    pub actor_queue: Duration,
    pub actor_admission: Duration,
    pub actor_prefill: Duration,
    pub actor_decode: Duration,
    pub response_delivery: Duration,
    pub backend: BackendPhaseDurations,
    pub unexplained: Duration,
}

impl TokenPhaseDurations {
    pub fn add_actor_queue(&mut self, duration: Duration) {
        self.actor_queue = self.actor_queue.saturating_add(duration);
    }

    pub fn add_actor_admission(&mut self, duration: Duration) {
        self.actor_admission = self.actor_admission.saturating_add(duration);
    }

    pub fn add_actor_prefill(&mut self, duration: Duration) {
        self.actor_prefill = self.actor_prefill.saturating_add(duration);
    }

    pub fn add_actor_decode(&mut self, duration: Duration) {
        self.actor_decode = self.actor_decode.saturating_add(duration);
    }

    pub fn add_response_delivery(&mut self, duration: Duration) {
        self.response_delivery = self.response_delivery.saturating_add(duration);
    }

    pub fn add_backend(&mut self, phases: BackendPhaseDurations) {
        phases.add_to(&mut self.backend);
    }

    pub fn account_unexplained_wall_time(&mut self, wall_time: Duration) {
        let actor_serial = self
            .actor_queue
            .saturating_add(self.actor_admission)
            .saturating_add(self.actor_prefill)
            .saturating_add(self.actor_decode);
        // Delivery can overlap actor work for other rows. Without an interval
        // trace, max() is the conservative lower bound on the covered union;
        // summing would double-count overlap and hide genuinely unexplained
        // wall time.
        let accounted = actor_serial
            .max(self.response_delivery)
            .max(self.backend.max_covered_duration());
        self.unexplained = wall_time.saturating_sub(accounted);
    }

    fn dominant(self) -> (LatencyStallReason, Duration) {
        [
            (LatencyStallReason::ActorQueue, self.actor_queue),
            (LatencyStallReason::ActorAdmission, self.actor_admission),
            (LatencyStallReason::ActorPrefill, self.actor_prefill),
            (
                LatencyStallReason::ActorDecode,
                self.backend.actor_decode_residual(self.actor_decode),
            ),
            (LatencyStallReason::ResponseDelivery, self.response_delivery),
            (
                LatencyStallReason::Sampling,
                self.backend.sampling.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Readback,
                self.backend.readback.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::GpuLockWait,
                self.backend.gpu_lock_wait.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::GraphCapture,
                self.backend.graph_capture.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::GraphReplay,
                self.backend.graph_replay.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Synchronization,
                self.backend.synchronization.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Resize,
                self.backend.resize.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Trim,
                self.backend.trim.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Adapter,
                self.backend.adapter.unwrap_or(Duration::ZERO),
            ),
            (
                LatencyStallReason::Training,
                self.backend.training.unwrap_or(Duration::ZERO),
            ),
            (LatencyStallReason::Unexplained, self.unexplained),
        ]
        .into_iter()
        .max_by_key(|(_, duration)| *duration)
        .unwrap_or((LatencyStallReason::Unexplained, Duration::ZERO))
    }

    fn add_to(self, totals: &mut Self) {
        totals.actor_queue = totals.actor_queue.saturating_add(self.actor_queue);
        totals.actor_admission = totals.actor_admission.saturating_add(self.actor_admission);
        totals.actor_prefill = totals.actor_prefill.saturating_add(self.actor_prefill);
        totals.actor_decode = totals.actor_decode.saturating_add(self.actor_decode);
        self.backend.add_to(&mut totals.backend);
        totals.unexplained = totals.unexplained.saturating_add(self.unexplained);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EngineTokenTiming {
    pub ready_at: Instant,
    pub producer_delivered_at: Option<Instant>,
    pub phases_since_previous_token: TokenPhaseDurations,
}

impl EngineTokenTiming {
    pub fn ready(ready_at: Instant, phases_since_previous_token: TokenPhaseDurations) -> Self {
        Self {
            ready_at,
            producer_delivered_at: None,
            phases_since_previous_token,
        }
    }

    pub fn mark_producer_delivered(&mut self, delivered_at: Instant) {
        self.producer_delivered_at = Some(delivered_at);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LatencyStallReason {
    ActorQueue,
    ActorAdmission,
    ActorPrefill,
    ActorDecode,
    ResponseDelivery,
    HandlerQueue,
    ClientDelivery,
    Sampling,
    Readback,
    GpuLockWait,
    GraphCapture,
    GraphReplay,
    Synchronization,
    Resize,
    Trim,
    Adapter,
    Training,
    Unexplained,
}

impl LatencyStallReason {
    pub const ALL: [Self; 18] = [
        Self::ActorQueue,
        Self::ActorAdmission,
        Self::ActorPrefill,
        Self::ActorDecode,
        Self::ResponseDelivery,
        Self::HandlerQueue,
        Self::ClientDelivery,
        Self::Sampling,
        Self::Readback,
        Self::GpuLockWait,
        Self::GraphCapture,
        Self::GraphReplay,
        Self::Synchronization,
        Self::Resize,
        Self::Trim,
        Self::Adapter,
        Self::Training,
        Self::Unexplained,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ActorQueue => "actor_queue",
            Self::ActorAdmission => "actor_admission",
            Self::ActorPrefill => "actor_prefill",
            Self::ActorDecode => "actor_decode",
            Self::ResponseDelivery => "response_delivery",
            Self::HandlerQueue => "handler_queue",
            Self::ClientDelivery => "client_delivery",
            Self::Sampling => "sampling",
            Self::Readback => "readback",
            Self::GpuLockWait => "gpu_lock_wait",
            Self::GraphCapture => "graph_capture",
            Self::GraphReplay => "graph_replay",
            Self::Synchronization => "synchronization",
            Self::Resize => "resize",
            Self::Trim => "trim",
            Self::Adapter => "adapter",
            Self::Training => "training",
            Self::Unexplained => "unexplained",
        }
    }

    pub const fn index(self) -> usize {
        match self {
            Self::ActorQueue => 0,
            Self::ActorAdmission => 1,
            Self::ActorPrefill => 2,
            Self::ActorDecode => 3,
            Self::ResponseDelivery => 4,
            Self::HandlerQueue => 5,
            Self::ClientDelivery => 6,
            Self::Sampling => 7,
            Self::Readback => 8,
            Self::GpuLockWait => 9,
            Self::GraphCapture => 10,
            Self::GraphReplay => 11,
            Self::Synchronization => 12,
            Self::Resize => 13,
            Self::Trim => 14,
            Self::Adapter => 15,
            Self::Training => 16,
            Self::Unexplained => 17,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenGapObservation {
    pub gap: Duration,
    pub reason: LatencyStallReason,
    pub attributed_duration: Duration,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize)]
pub struct LatencyPhaseTimings {
    pub actor_queue_ms: Option<f64>,
    pub actor_admission_ms: Option<f64>,
    pub tokenization_ms: Option<f64>,
    pub prefill_ms: Option<f64>,
    pub decode_ms: Option<f64>,
    pub sampling_ms: Option<f64>,
    pub readback_ms: Option<f64>,
    pub response_delivery_ms: Option<f64>,
    pub handler_queue_ms: Option<f64>,
    pub client_delivery_ms: Option<f64>,
    pub gpu_lock_wait_ms: Option<f64>,
    pub graph_capture_ms: Option<f64>,
    pub graph_replay_ms: Option<f64>,
    pub synchronization_ms: Option<f64>,
    pub resize_ms: Option<f64>,
    pub trim_ms: Option<f64>,
    pub adapter_ms: Option<f64>,
    pub training_ms: Option<f64>,
    pub unexplained_ms: Option<f64>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
pub struct LatencyStallReasonCounts {
    pub actor_queue: u64,
    pub actor_admission: u64,
    pub actor_prefill: u64,
    pub actor_decode: u64,
    pub response_delivery: u64,
    pub handler_queue: u64,
    pub client_delivery: u64,
    pub sampling: u64,
    pub readback: u64,
    pub gpu_lock_wait: u64,
    pub graph_capture: u64,
    pub graph_replay: u64,
    pub synchronization: u64,
    pub resize: u64,
    pub trim: u64,
    pub adapter: u64,
    pub training: u64,
    pub unexplained: u64,
}

impl LatencyStallReasonCounts {
    pub(crate) fn increment(&mut self, reason: LatencyStallReason) {
        let value = match reason {
            LatencyStallReason::ActorQueue => &mut self.actor_queue,
            LatencyStallReason::ActorAdmission => &mut self.actor_admission,
            LatencyStallReason::ActorPrefill => &mut self.actor_prefill,
            LatencyStallReason::ActorDecode => &mut self.actor_decode,
            LatencyStallReason::ResponseDelivery => &mut self.response_delivery,
            LatencyStallReason::HandlerQueue => &mut self.handler_queue,
            LatencyStallReason::ClientDelivery => &mut self.client_delivery,
            LatencyStallReason::Sampling => &mut self.sampling,
            LatencyStallReason::Readback => &mut self.readback,
            LatencyStallReason::GpuLockWait => &mut self.gpu_lock_wait,
            LatencyStallReason::GraphCapture => &mut self.graph_capture,
            LatencyStallReason::GraphReplay => &mut self.graph_replay,
            LatencyStallReason::Synchronization => &mut self.synchronization,
            LatencyStallReason::Resize => &mut self.resize,
            LatencyStallReason::Trim => &mut self.trim,
            LatencyStallReason::Adapter => &mut self.adapter,
            LatencyStallReason::Training => &mut self.training,
            LatencyStallReason::Unexplained => &mut self.unexplained,
        };
        *value = value.saturating_add(1);
    }

    pub const fn total(self) -> u64 {
        self.actor_queue
            .saturating_add(self.actor_admission)
            .saturating_add(self.actor_prefill)
            .saturating_add(self.actor_decode)
            .saturating_add(self.response_delivery)
            .saturating_add(self.handler_queue)
            .saturating_add(self.client_delivery)
            .saturating_add(self.sampling)
            .saturating_add(self.readback)
            .saturating_add(self.gpu_lock_wait)
            .saturating_add(self.graph_capture)
            .saturating_add(self.graph_replay)
            .saturating_add(self.synchronization)
            .saturating_add(self.resize)
            .saturating_add(self.trim)
            .saturating_add(self.adapter)
            .saturating_add(self.training)
            .saturating_add(self.unexplained)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestLatencyDiagnostics {
    pub emitted_tokens: u64,
    pub gap_samples: u64,
    pub retained_gap_samples: usize,
    pub gap_samples_truncated: bool,
    pub ttft_ms: Option<f64>,
    pub itl_ms_p50: Option<f64>,
    pub itl_ms_p99: Option<f64>,
    pub itl_ms_p999: Option<f64>,
    pub max_itl_ms: Option<f64>,
    pub stall_threshold_ms: Option<f64>,
    pub stall_count: u64,
    pub unexplained_stall_count: u64,
    pub stall_reasons: LatencyStallReasonCounts,
    pub phases: LatencyPhaseTimings,
}

#[derive(Debug, Clone, Copy)]
struct RetainedGap {
    gap: Duration,
    reason: LatencyStallReason,
}

pub struct RequestLatencyTracker {
    request_started_at: Instant,
    tokenization: Option<Duration>,
    actor_phases_observed: bool,
    first_token_ready_at: Option<Instant>,
    previous_token_ready_at: Option<Instant>,
    emitted_tokens: u64,
    gap_samples: u64,
    retained_gaps: VecDeque<RetainedGap>,
    gap_samples_truncated: bool,
    phase_totals: TokenPhaseDurations,
    handler_queue_total: Duration,
    client_delivery_total: Duration,
    client_delivery_observed: bool,
}

impl RequestLatencyTracker {
    pub fn new(request_started_at: Instant, tokenization: Option<Duration>) -> Self {
        Self::with_actor_phase_coverage(request_started_at, tokenization, true)
    }

    pub fn direct(request_started_at: Instant, tokenization: Option<Duration>) -> Self {
        Self::with_actor_phase_coverage(request_started_at, tokenization, false)
    }

    fn with_actor_phase_coverage(
        request_started_at: Instant,
        tokenization: Option<Duration>,
        actor_phases_observed: bool,
    ) -> Self {
        Self {
            request_started_at,
            tokenization,
            actor_phases_observed,
            first_token_ready_at: None,
            previous_token_ready_at: None,
            emitted_tokens: 0,
            gap_samples: 0,
            retained_gaps: VecDeque::with_capacity(MAX_RETAINED_REQUEST_GAPS),
            gap_samples_truncated: false,
            phase_totals: TokenPhaseDurations::default(),
            handler_queue_total: Duration::ZERO,
            client_delivery_total: Duration::ZERO,
            client_delivery_observed: false,
        }
    }

    pub fn record_token(
        &mut self,
        timing: EngineTokenTiming,
        handler_received_at: Instant,
    ) -> Option<TokenGapObservation> {
        self.emitted_tokens = self.emitted_tokens.saturating_add(1);
        self.first_token_ready_at.get_or_insert(timing.ready_at);
        timing
            .phases_since_previous_token
            .add_to(&mut self.phase_totals);

        let producer_delivered_at = timing.producer_delivered_at.unwrap_or(timing.ready_at);
        self.phase_totals.response_delivery = self
            .phase_totals
            .response_delivery
            .saturating_add(producer_delivered_at.saturating_duration_since(timing.ready_at));
        self.handler_queue_total = self
            .handler_queue_total
            .saturating_add(handler_received_at.saturating_duration_since(producer_delivered_at));

        let observation = self.previous_token_ready_at.map(|previous| {
            let gap = timing.ready_at.saturating_duration_since(previous);
            let (mut reason, attributed_duration) = timing.phases_since_previous_token.dominant();
            let minimum_explanation = ABSOLUTE_STALL_THRESHOLD.min(gap / 2);
            if attributed_duration < minimum_explanation {
                reason = LatencyStallReason::Unexplained;
            }
            TokenGapObservation {
                gap,
                reason,
                attributed_duration,
            }
        });
        self.previous_token_ready_at = Some(timing.ready_at);

        if let Some(observation) = observation {
            self.gap_samples = self.gap_samples.saturating_add(1);
            if self.retained_gaps.len() == MAX_RETAINED_REQUEST_GAPS {
                self.retained_gaps.pop_front();
                self.gap_samples_truncated = true;
            }
            self.retained_gaps.push_back(RetainedGap {
                gap: observation.gap,
                reason: observation.reason,
            });
        }
        observation
    }

    pub fn record_client_delivery(
        &mut self,
        handler_received_at: Instant,
        body_enqueued_at: Instant,
    ) {
        self.client_delivery_observed = true;
        self.client_delivery_total = self
            .client_delivery_total
            .saturating_add(body_enqueued_at.saturating_duration_since(handler_received_at));
    }

    pub fn diagnostics(&self) -> RequestLatencyDiagnostics {
        let mut sorted_ms: Vec<f64> = self
            .retained_gaps
            .iter()
            .map(|sample| duration_ms(sample.gap))
            .collect();
        sorted_ms.sort_by(f64::total_cmp);
        let p50 = percentile(&sorted_ms, 0.50);
        let stall_threshold_ms = p50
            .map(|p50| duration_ms(ABSOLUTE_STALL_THRESHOLD).max(p50 * RELATIVE_STALL_MULTIPLIER));
        let mut stall_reasons = LatencyStallReasonCounts::default();
        let mut stall_count = 0_u64;
        if let Some(threshold) = stall_threshold_ms {
            for sample in &self.retained_gaps {
                if duration_ms(sample.gap) >= threshold {
                    stall_count = stall_count.saturating_add(1);
                    stall_reasons.increment(sample.reason);
                }
            }
        }

        let token_phase_observed = self.emitted_tokens > 0;
        RequestLatencyDiagnostics {
            emitted_tokens: self.emitted_tokens,
            gap_samples: self.gap_samples,
            retained_gap_samples: self.retained_gaps.len(),
            gap_samples_truncated: self.gap_samples_truncated,
            ttft_ms: self
                .first_token_ready_at
                .map(|ready| duration_ms(ready.saturating_duration_since(self.request_started_at))),
            itl_ms_p50: p50,
            itl_ms_p99: percentile(&sorted_ms, 0.99),
            itl_ms_p999: percentile(&sorted_ms, 0.999),
            max_itl_ms: sorted_ms.last().copied(),
            stall_threshold_ms,
            stall_count,
            unexplained_stall_count: stall_reasons.unexplained,
            stall_reasons,
            phases: LatencyPhaseTimings {
                actor_queue_ms: (token_phase_observed && self.actor_phases_observed)
                    .then(|| duration_ms(self.phase_totals.actor_queue)),
                actor_admission_ms: (token_phase_observed && self.actor_phases_observed)
                    .then(|| duration_ms(self.phase_totals.actor_admission)),
                tokenization_ms: self.tokenization.map(duration_ms),
                prefill_ms: (token_phase_observed && self.actor_phases_observed)
                    .then(|| duration_ms(self.phase_totals.actor_prefill)),
                decode_ms: (token_phase_observed && self.actor_phases_observed)
                    .then(|| duration_ms(self.phase_totals.actor_decode)),
                sampling_ms: self.phase_totals.backend.sampling.map(duration_ms),
                readback_ms: self.phase_totals.backend.readback.map(duration_ms),
                response_delivery_ms: token_phase_observed
                    .then(|| duration_ms(self.phase_totals.response_delivery)),
                handler_queue_ms: token_phase_observed
                    .then(|| duration_ms(self.handler_queue_total)),
                client_delivery_ms: self
                    .client_delivery_observed
                    .then(|| duration_ms(self.client_delivery_total)),
                gpu_lock_wait_ms: self.phase_totals.backend.gpu_lock_wait.map(duration_ms),
                graph_capture_ms: self.phase_totals.backend.graph_capture.map(duration_ms),
                graph_replay_ms: self.phase_totals.backend.graph_replay.map(duration_ms),
                synchronization_ms: self.phase_totals.backend.synchronization.map(duration_ms),
                resize_ms: self.phase_totals.backend.resize.map(duration_ms),
                trim_ms: self.phase_totals.backend.trim.map(duration_ms),
                adapter_ms: self.phase_totals.backend.adapter.map(duration_ms),
                training_ms: self.phase_totals.backend.training.map(duration_ms),
                unexplained_ms: token_phase_observed
                    .then(|| duration_ms(self.phase_totals.unexplained)),
            },
        }
    }
}

pub fn duration_ms(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

pub fn percentile(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    if sorted.len() == 1 {
        return Some(sorted[0]);
    }
    let rank = p * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        Some(sorted[lo])
    } else {
        let fraction = rank - lo as f64;
        Some(sorted[lo] * (1.0 - fraction) + sorted[hi] * fraction)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_tracker_attributes_stalls_and_keeps_missing_subphases_null() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(start, Some(Duration::from_millis(3)));
        let first = EngineTokenTiming::ready(
            start + Duration::from_millis(20),
            TokenPhaseDurations {
                actor_queue: Duration::from_millis(5),
                actor_prefill: Duration::from_millis(12),
                unexplained: Duration::from_millis(3),
                ..TokenPhaseDurations::default()
            },
        );
        tracker.record_token(first, start + Duration::from_millis(21));

        for offset_ms in [30, 40] {
            let ready_at = start + Duration::from_millis(offset_ms);
            tracker.record_token(
                EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default()),
                ready_at,
            );
        }

        let stalled_ready = start + Duration::from_millis(340);
        let mut stalled = EngineTokenTiming::ready(
            stalled_ready,
            TokenPhaseDurations {
                actor_prefill: Duration::from_millis(280),
                actor_decode: Duration::from_millis(20),
                ..TokenPhaseDurations::default()
            },
        );
        stalled.mark_producer_delivered(stalled_ready + Duration::from_millis(2));
        let gap = tracker
            .record_token(stalled, stalled_ready + Duration::from_millis(4))
            .unwrap();
        tracker.record_client_delivery(
            stalled_ready + Duration::from_millis(4),
            stalled_ready + Duration::from_millis(7),
        );

        assert_eq!(gap.reason, LatencyStallReason::ActorPrefill);
        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.emitted_tokens, 4);
        assert_eq!(diagnostics.gap_samples, 3);
        assert_eq!(diagnostics.ttft_ms, Some(20.0));
        assert_eq!(diagnostics.max_itl_ms, Some(300.0));
        assert_eq!(diagnostics.stall_threshold_ms, Some(250.0));
        assert_eq!(diagnostics.stall_count, 1);
        assert_eq!(diagnostics.stall_reasons.actor_prefill, 1);
        assert_eq!(diagnostics.unexplained_stall_count, 0);
        assert_eq!(diagnostics.phases.tokenization_ms, Some(3.0));
        assert_eq!(diagnostics.phases.client_delivery_ms, Some(3.0));
        assert_eq!(diagnostics.phases.actor_admission_ms, Some(0.0));
        assert!(diagnostics.phases.sampling_ms.is_none());
        assert!(diagnostics.phases.readback_ms.is_none());
    }

    #[test]
    fn weak_phase_overlap_is_not_allowed_to_explain_a_large_gap() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(start, None);
        tracker.record_token(
            EngineTokenTiming::ready(start, TokenPhaseDurations::default()),
            start,
        );
        for offset_ms in [10, 20] {
            let ready_at = start + Duration::from_millis(offset_ms);
            tracker.record_token(
                EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default()),
                ready_at,
            );
        }
        let observation = tracker
            .record_token(
                EngineTokenTiming::ready(
                    start + Duration::from_secs(1),
                    TokenPhaseDurations {
                        actor_decode: Duration::from_millis(10),
                        ..TokenPhaseDurations::default()
                    },
                ),
                start + Duration::from_secs(1),
            )
            .unwrap();
        assert_eq!(observation.reason, LatencyStallReason::Unexplained);
        assert_eq!(tracker.diagnostics().unexplained_stall_count, 1);
    }

    #[test]
    fn overlapping_delivery_and_actor_work_do_not_hide_unexplained_time() {
        let mut phases = TokenPhaseDurations {
            actor_decode: Duration::from_millis(60),
            response_delivery: Duration::from_millis(80),
            ..TokenPhaseDurations::default()
        };
        phases.account_unexplained_wall_time(Duration::from_millis(100));
        assert_eq!(phases.unexplained, Duration::from_millis(20));
    }

    #[test]
    fn backend_subphase_can_explain_a_stall_inside_broad_actor_decode_wall_time() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(start, None);
        for offset_ms in [0, 10, 20] {
            let ready_at = start + Duration::from_millis(offset_ms);
            tracker.record_token(
                EngineTokenTiming::ready(ready_at, TokenPhaseDurations::default()),
                ready_at,
            );
        }

        let mut backend = BackendPhaseDurations::default();
        backend.observe_synchronization(Duration::from_millis(260));
        let ready_at = start + Duration::from_millis(320);
        let observation = tracker
            .record_token(
                EngineTokenTiming::ready(
                    ready_at,
                    TokenPhaseDurations {
                        actor_decode: Duration::from_millis(290),
                        backend,
                        ..TokenPhaseDurations::default()
                    },
                ),
                ready_at,
            )
            .unwrap();

        assert_eq!(observation.reason, LatencyStallReason::Synchronization);
        assert_eq!(observation.attributed_duration, Duration::from_millis(260));
        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.stall_reasons.synchronization, 1);
        assert_eq!(diagnostics.phases.decode_ms, Some(290.0));
        assert_eq!(diagnostics.phases.synchronization_ms, Some(260.0));
        assert_eq!(diagnostics.phases.sampling_ms, None);
    }

    #[test]
    fn zero_duration_measured_phases_remain_distinct_from_unsupported_phases() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(start, Some(Duration::ZERO));
        tracker.record_token(
            EngineTokenTiming::ready(start, TokenPhaseDurations::default()),
            start,
        );
        tracker.record_client_delivery(start, start);
        let phases = tracker.diagnostics().phases;
        assert_eq!(phases.actor_queue_ms, Some(0.0));
        assert_eq!(phases.tokenization_ms, Some(0.0));
        assert_eq!(phases.client_delivery_ms, Some(0.0));
        assert_eq!(phases.unexplained_ms, Some(0.0));
        assert_eq!(phases.sampling_ms, None);
    }

    #[test]
    fn direct_tracker_keeps_actor_phases_null_and_transport_phases_measured() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::direct(start, Some(Duration::from_millis(3)));
        let first_ready = start + Duration::from_millis(20);
        let mut first = EngineTokenTiming::ready(first_ready, TokenPhaseDurations::default());
        first.mark_producer_delivered(first_ready + Duration::from_millis(2));
        tracker.record_token(first, first_ready + Duration::from_millis(4));
        tracker.record_client_delivery(
            first_ready + Duration::from_millis(4),
            first_ready + Duration::from_millis(7),
        );

        let second_ready = start + Duration::from_millis(30);
        let mut phases = TokenPhaseDurations::default();
        phases.account_unexplained_wall_time(Duration::from_millis(10));
        let mut second = EngineTokenTiming::ready(second_ready, phases);
        second.mark_producer_delivered(second_ready + Duration::from_millis(1));
        let gap = tracker
            .record_token(second, second_ready + Duration::from_millis(2))
            .unwrap();
        tracker.record_client_delivery(
            second_ready + Duration::from_millis(2),
            second_ready + Duration::from_millis(3),
        );

        assert_eq!(gap.reason, LatencyStallReason::Unexplained);
        assert_eq!(gap.attributed_duration, Duration::from_millis(10));
        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.ttft_ms, Some(20.0));
        assert_eq!(diagnostics.phases.tokenization_ms, Some(3.0));
        assert_eq!(diagnostics.phases.response_delivery_ms, Some(3.0));
        assert_eq!(diagnostics.phases.handler_queue_ms, Some(3.0));
        assert_eq!(diagnostics.phases.client_delivery_ms, Some(4.0));
        assert_eq!(diagnostics.phases.unexplained_ms, Some(10.0));
        assert!(diagnostics.phases.actor_queue_ms.is_none());
        assert!(diagnostics.phases.actor_admission_ms.is_none());
        assert!(diagnostics.phases.prefill_ms.is_none());
        assert!(diagnostics.phases.decode_ms.is_none());
    }

    #[test]
    fn retained_gap_memory_is_bounded_and_disclosed() {
        let start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(start, None);
        tracker.record_token(
            EngineTokenTiming::ready(start, TokenPhaseDurations::default()),
            start,
        );
        for index in 1..=(MAX_RETAINED_REQUEST_GAPS + 5) {
            let ready = start + Duration::from_millis(index as u64);
            tracker.record_token(
                EngineTokenTiming::ready(
                    ready,
                    TokenPhaseDurations {
                        actor_decode: Duration::from_millis(1),
                        ..TokenPhaseDurations::default()
                    },
                ),
                ready,
            );
        }
        let diagnostics = tracker.diagnostics();
        assert_eq!(diagnostics.retained_gap_samples, MAX_RETAINED_REQUEST_GAPS);
        assert_eq!(
            diagnostics.gap_samples,
            (MAX_RETAINED_REQUEST_GAPS + 5) as u64
        );
        assert!(diagnostics.gap_samples_truncated);
    }

    #[test]
    fn latency_measurement_hot_path_is_bounded() {
        use std::hint::black_box;
        use std::mem::size_of;
        use std::sync::Mutex;

        use crate::decode_stats::DecodeStatsRing;
        use crate::metrics::Metrics;

        const TOKENS: usize = 20_000;
        const MAX_NANOSECONDS_PER_TOKEN: u128 = 100_000;
        const MAX_TRACKER_SAMPLE_BYTES: usize = 256 * 1024;

        let retained_bytes = MAX_RETAINED_REQUEST_GAPS * size_of::<RetainedGap>();
        assert!(
            retained_bytes <= MAX_TRACKER_SAMPLE_BYTES,
            "request latency samples require {retained_bytes} bytes"
        );

        let request_start = Instant::now();
        let mut tracker = RequestLatencyTracker::new(request_start, None);
        let metrics = Metrics::new();
        let rolling = Mutex::new(DecodeStatsRing::new(MAX_RETAINED_REQUEST_GAPS));
        let measured_at = Instant::now();
        for index in 0..TOKENS {
            let ready = request_start + Duration::from_millis(index as u64);
            let observation = tracker.record_token(
                EngineTokenTiming::ready(
                    ready,
                    TokenPhaseDurations {
                        actor_decode: Duration::from_millis(1),
                        ..TokenPhaseDurations::default()
                    },
                ),
                ready,
            );
            if let Some(observation) = observation {
                metrics.observe_token_gap(observation);
                rolling.lock().unwrap().record_gap(ready, observation);
            }
            black_box(observation);
        }
        black_box(tracker.diagnostics());
        black_box(
            rolling
                .lock()
                .unwrap()
                .snapshot(request_start + Duration::from_millis(TOKENS as u64)),
        );
        let elapsed = measured_at.elapsed();
        let nanoseconds_per_token = elapsed.as_nanos() / TOKENS as u128;
        eprintln!(
            "latency observability overhead: {nanoseconds_per_token} ns/token; tracker samples: {retained_bytes} bytes"
        );
        assert!(
            nanoseconds_per_token <= MAX_NANOSECONDS_PER_TOKEN,
            "latency observability cost {nanoseconds_per_token} ns/token exceeds {MAX_NANOSECONDS_PER_TOKEN} ns/token"
        );
    }
}
