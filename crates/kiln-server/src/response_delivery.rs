//! Fair, non-blocking delivery of batching-engine response events.
//!
//! The worker drives a deterministic state machine with a monotonic [`Instant`].
//! No lane is ever awaited or serviced with a blocking send.

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::io;
use std::sync::mpsc as std_mpsc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use kiln_core::token::TokenId;
use tokio::sync::mpsc;
use uuid::Uuid;

use crate::batching_engine::{BatchedGenerationOutput, EngineEvent};
use crate::latency_observability::EngineTokenTiming;

/// Identifies one incarnation of a request's response lane.
///
/// The generation prevents delayed commands for a previous incarnation from
/// reaching a newly registered lane that happens to reuse the same request ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct DeliveryKey {
    pub(crate) request_id: Uuid,
    pub(crate) generation: u64,
}

impl DeliveryKey {
    pub(crate) fn new(request_id: Uuid, generation: u64) -> Self {
        Self {
            request_id,
            generation,
        }
    }
}

/// One ordered unit of work for a response lane.
///
/// A terminal batch may include the final token so cancellation, completion,
/// or a full channel cannot reorder the terminal event ahead of that token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DeliveryBatch {
    Token {
        token: TokenId,
        timing: EngineTokenTiming,
        sequence: u64,
    },
    Terminal {
        preceding_token: Option<(TokenId, EngineTokenTiming)>,
        terminal: DeliveryTerminal,
        sequence: u64,
    },
}

impl DeliveryBatch {
    fn sequence(&self) -> u64 {
        match self {
            Self::Token { sequence, .. } | Self::Terminal { sequence, .. } => *sequence,
        }
    }

    fn is_terminal(&self) -> bool {
        matches!(self, Self::Terminal { .. })
    }

    fn into_events(self) -> VecDeque<EngineEvent> {
        let mut events = VecDeque::with_capacity(2);
        match self {
            Self::Token { token, timing, .. } => {
                events.push_back(EngineEvent::Token { token, timing })
            }
            Self::Terminal {
                preceding_token,
                terminal,
                ..
            } => {
                if let Some((token, timing)) = preceding_token {
                    events.push_back(EngineEvent::Token { token, timing });
                }
                events.push_back(terminal.into_event());
            }
        }
        events
    }
}

#[derive(Debug, Clone, PartialEq)]
pub(crate) enum DeliveryTerminal {
    Done(BatchedGenerationOutput),
    Error(String),
}

impl DeliveryTerminal {
    fn into_event(self) -> EngineEvent {
        match self {
            Self::Done(output) => EngineEvent::Done { output },
            Self::Error(error) => EngineEvent::Error(error),
        }
    }
}

pub(crate) enum DeliveryCommand {
    Register {
        key: DeliveryKey,
        response_tx: mpsc::Sender<EngineEvent>,
    },
    Deliver {
        key: DeliveryKey,
        batch: DeliveryBatch,
    },
    DeliverMany {
        deliveries: Vec<(DeliveryKey, DeliveryBatch)>,
    },
    Barrier {
        reply: std_mpsc::Sender<()>,
    },
    /// Terminate a lane without overtaking an already accepted token.
    Terminate {
        key: DeliveryKey,
        error: String,
    },
    /// Best-effort error notification followed by immediate sender teardown.
    Shutdown {
        error: String,
    },
}

impl fmt::Debug for DeliveryCommand {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Register { key, response_tx } => f
                .debug_struct("Register")
                .field("key", key)
                .field("channel_capacity", &response_tx.max_capacity())
                .finish(),
            Self::Deliver { key, batch } => f
                .debug_struct("Deliver")
                .field("key", key)
                .field("batch", batch)
                .finish(),
            Self::DeliverMany { deliveries } => f
                .debug_struct("DeliverMany")
                .field("deliveries", deliveries)
                .finish(),
            Self::Barrier { .. } => f.debug_struct("Barrier").finish(),
            Self::Terminate { key, error } => f
                .debug_struct("Terminate")
                .field("key", key)
                .field("error", error)
                .finish(),
            Self::Shutdown { error } => f.debug_struct("Shutdown").field("error", error).finish(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeliveryResult {
    BackpressureStarted {
        key: DeliveryKey,
        sequence: u64,
        capacity: usize,
    },
    Delivered {
        key: DeliveryKey,
        sequence: u64,
        terminal: bool,
        waited: Duration,
    },
    Closed {
        key: DeliveryKey,
        sequence: u64,
        waited: Duration,
        backpressured: bool,
    },
    TimedOut {
        key: DeliveryKey,
        sequence: u64,
        waited: Duration,
    },
    ProtocolError(DeliveryProtocolError),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeliveryProtocolError {
    Shutdown,
    AlreadyRegistered(DeliveryKey),
    UnknownKey(DeliveryKey),
    LaneBusy(DeliveryKey),
    UnexpectedSequence {
        key: DeliveryKey,
        expected: u64,
        actual: u64,
    },
    SequenceExhausted(DeliveryKey),
}

impl fmt::Display for DeliveryProtocolError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Shutdown => write!(f, "response delivery is shut down"),
            Self::AlreadyRegistered(key) => write!(f, "delivery lane already registered: {key:?}"),
            Self::UnknownKey(key) => write!(f, "delivery lane is not registered: {key:?}"),
            Self::LaneBusy(key) => write!(f, "delivery lane already has a pending batch: {key:?}"),
            Self::UnexpectedSequence {
                key,
                expected,
                actual,
            } => write!(
                f,
                "delivery sequence mismatch for {key:?}: expected {expected}, got {actual}"
            ),
            Self::SequenceExhausted(key) => {
                write!(f, "delivery sequence exhausted for {key:?}")
            }
        }
    }
}

impl std::error::Error for DeliveryProtocolError {}

/// A non-blocking destination for worker results.
///
/// Implementations used by the batching actor must not retain that actor
/// strongly. A weak wake handle plus a separately owned result queue avoids a
/// worker/actor lifecycle cycle while preserving every accepted result.
pub(crate) trait DeliveryResultSink: Send + 'static {
    fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError>;

    /// Notify the sink owner once after one or more results have been queued.
    fn notify(&mut self) -> Result<(), DeliveryResultNotifyError> {
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeliveryResultNotifyError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DeliveryBarrierError;

impl fmt::Display for DeliveryBarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "response delivery worker stopped or did not acknowledge its barrier within {} ms",
            DELIVERY_BARRIER_TIMEOUT.as_millis()
        )
    }
}

impl std::error::Error for DeliveryBarrierError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum DeliveryResultSinkError {
    Full(DeliveryResult),
    Closed(DeliveryResult),
}

impl DeliveryResultSink for std_mpsc::Sender<DeliveryResult> {
    fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError> {
        self.send(result)
            .map_err(|error| DeliveryResultSinkError::Closed(error.0))
    }
}

impl DeliveryResultSink for std_mpsc::SyncSender<DeliveryResult> {
    fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError> {
        std_mpsc::SyncSender::try_send(self, result).map_err(|error| match error {
            std_mpsc::TrySendError::Full(result) => DeliveryResultSinkError::Full(result),
            std_mpsc::TrySendError::Disconnected(result) => DeliveryResultSinkError::Closed(result),
        })
    }
}

struct PendingBatch {
    sequence: u64,
    terminal: bool,
    events: VecDeque<EngineEvent>,
    backpressure_started_at: Option<Instant>,
    accumulated_wait: Duration,
}

impl PendingBatch {
    fn new(batch: DeliveryBatch) -> Self {
        let sequence = batch.sequence();
        let terminal = batch.is_terminal();
        Self {
            sequence,
            terminal,
            events: batch.into_events(),
            backpressure_started_at: None,
            accumulated_wait: Duration::ZERO,
        }
    }

    fn terminate(&mut self, error: String) {
        self.terminal = true;
        match self.events.len() {
            // A token batch has not been attempted, or its prior attempt was
            // full. Keep that token first and append the terminal error.
            1 if matches!(self.events.front(), Some(EngineEvent::Token { .. })) => {
                self.events.push_back(EngineEvent::Error(error));
            }
            // A terminal batch's preceding token was already delivered, or it
            // never had one. Replace its not-yet-delivered terminal event.
            1 => {
                self.events.pop_front();
                self.events.push_back(EngineEvent::Error(error));
            }
            // Preserve a terminal batch's pending token while replacing Done
            // (or an earlier Error) with the cancellation reason.
            2 => {
                self.events.pop_back();
                self.events.push_back(EngineEvent::Error(error));
            }
            _ => unreachable!("a delivery batch contains one or two events"),
        }
    }

    fn waited(&self, now: Instant) -> Duration {
        self.accumulated_wait
            .saturating_add(self.continuous_wait(now))
    }

    fn continuous_wait(&self, now: Instant) -> Duration {
        self.backpressure_started_at
            .map_or(Duration::ZERO, |started| {
                now.saturating_duration_since(started)
            })
    }

    fn was_backpressured(&self) -> bool {
        !self.accumulated_wait.is_zero() || self.backpressure_started_at.is_some()
    }

    fn record_progress(&mut self, now: Instant) {
        if let Some(started) = self.backpressure_started_at.take() {
            self.accumulated_wait = self
                .accumulated_wait
                .saturating_add(now.saturating_duration_since(started));
        }
    }
}

struct DeliveryLane {
    response_tx: mpsc::Sender<EngineEvent>,
    next_sequence: u64,
    pending: Option<PendingBatch>,
}

/// Pure state for a fair, non-blocking response-delivery worker.
pub(crate) struct DeliveryState {
    stall_grace: Duration,
    lanes: HashMap<DeliveryKey, DeliveryLane>,
    newly_ready_lanes: VecDeque<DeliveryKey>,
    cadence_blocked_lanes: VecDeque<DeliveryKey>,
    shutdown: bool,
}

impl DeliveryState {
    pub(crate) fn new(stall_grace: Duration) -> Self {
        Self {
            stall_grace,
            lanes: HashMap::new(),
            newly_ready_lanes: VecDeque::new(),
            cadence_blocked_lanes: VecDeque::new(),
            shutdown: false,
        }
    }

    pub(crate) fn handle(&mut self, command: DeliveryCommand) -> Result<(), DeliveryProtocolError> {
        if self.shutdown {
            return Err(DeliveryProtocolError::Shutdown);
        }

        match command {
            DeliveryCommand::Register { key, response_tx } => {
                if self.lanes.contains_key(&key) {
                    return Err(DeliveryProtocolError::AlreadyRegistered(key));
                }
                self.lanes.insert(
                    key,
                    DeliveryLane {
                        response_tx,
                        next_sequence: 0,
                        pending: None,
                    },
                );
            }
            DeliveryCommand::Deliver { key, batch } => {
                self.enqueue_deliveries(vec![(key, batch)])?;
            }
            DeliveryCommand::DeliverMany { deliveries } => {
                self.enqueue_deliveries(deliveries)?;
            }
            DeliveryCommand::Terminate { key, error } => {
                let Some(lane) = self.lanes.get_mut(&key) else {
                    // Cancellation can race a Closed/TimedOut result. The
                    // generation key makes a missing lane unambiguously stale.
                    return Ok(());
                };
                if let Some(pending) = &mut lane.pending {
                    pending.terminate(error);
                } else {
                    let sequence = lane.next_sequence;
                    lane.next_sequence = lane
                        .next_sequence
                        .checked_add(1)
                        .ok_or(DeliveryProtocolError::SequenceExhausted(key))?;
                    lane.pending = Some(PendingBatch::new(DeliveryBatch::Terminal {
                        preceding_token: None,
                        terminal: DeliveryTerminal::Error(error),
                        sequence,
                    }));
                    self.newly_ready_lanes.push_back(key);
                }
            }
            DeliveryCommand::Barrier { .. } => {
                unreachable!("delivery barriers are handled by the worker")
            }
            DeliveryCommand::Shutdown { error } => {
                // Shutdown is deliberately best-effort: full and closed lanes
                // are dropped immediately rather than extending process exit.
                for lane in self.lanes.values_mut() {
                    if let Some(pending) = &mut lane.pending {
                        pending.terminate(error.clone());
                        while let Some(mut event) = pending.events.pop_front() {
                            if let EngineEvent::Token { timing, .. } = &mut event {
                                timing.mark_actor_delivered(Instant::now());
                            }
                            if lane.response_tx.try_send(event).is_err() {
                                break;
                            }
                        }
                    } else {
                        let _ = lane.response_tx.try_send(EngineEvent::Error(error.clone()));
                    }
                }
                self.lanes.clear();
                self.newly_ready_lanes.clear();
                self.cadence_blocked_lanes.clear();
                self.shutdown = true;
            }
        }
        Ok(())
    }

    /// Gives every pending lane one bounded, non-blocking cadence attempt.
    ///
    /// `now` is supplied by the worker, making grace-window behavior exact and
    /// deterministic in tests. A batch contains at most a token and terminal
    /// event; both are attempted in order until the lane reports full or closed.
    pub(crate) fn poll(&mut self, now: Instant) -> Vec<DeliveryResult> {
        if self.shutdown || !self.has_pending() {
            return Vec::new();
        }

        // Snapshot old blocked work before ready attempts append newly blocked
        // lanes. Each lane is therefore attempted at most once in this pass.
        let blocked_at_start = self.cadence_blocked_lanes.len();
        let mut results = Vec::new();
        for _ in 0..blocked_at_start {
            let Some(key) = self.cadence_blocked_lanes.pop_front() else {
                break;
            };
            self.poll_key(key, now, &mut results);
        }
        self.poll_ready_into(now, &mut results);
        results
    }

    /// Attempts only work newly accepted since the previous worker turn.
    fn poll_ready(&mut self, now: Instant) -> Vec<DeliveryResult> {
        if self.shutdown || self.newly_ready_lanes.is_empty() {
            return Vec::new();
        }
        let mut results = Vec::new();
        self.poll_ready_into(now, &mut results);
        results
    }

    #[cfg(test)]
    fn lane_count(&self) -> usize {
        self.lanes.len()
    }

    fn has_pending(&self) -> bool {
        !self.newly_ready_lanes.is_empty() || !self.cadence_blocked_lanes.is_empty()
    }

    fn enqueue_deliveries(
        &mut self,
        deliveries: Vec<(DeliveryKey, DeliveryBatch)>,
    ) -> Result<(), DeliveryProtocolError> {
        // Validate the entire actor batch before transferring ownership of any
        // row. A malformed item cannot strand earlier rows as worker-owned.
        let mut seen = HashSet::with_capacity(deliveries.len());
        for (key, batch) in &deliveries {
            if !seen.insert(*key) {
                return Err(DeliveryProtocolError::LaneBusy(*key));
            }
            let lane = self
                .lanes
                .get(key)
                .ok_or(DeliveryProtocolError::UnknownKey(*key))?;
            if lane.pending.is_some() {
                return Err(DeliveryProtocolError::LaneBusy(*key));
            }
            let actual = batch.sequence();
            if actual != lane.next_sequence {
                return Err(DeliveryProtocolError::UnexpectedSequence {
                    key: *key,
                    expected: lane.next_sequence,
                    actual,
                });
            }
            lane.next_sequence
                .checked_add(1)
                .ok_or(DeliveryProtocolError::SequenceExhausted(*key))?;
        }

        for (key, batch) in deliveries {
            let lane = self
                .lanes
                .get_mut(&key)
                .expect("validated delivery lane remains registered");
            lane.next_sequence += 1;
            lane.pending = Some(PendingBatch::new(batch));
            self.newly_ready_lanes.push_back(key);
        }
        Ok(())
    }

    fn poll_ready_into(&mut self, now: Instant, results: &mut Vec<DeliveryResult>) {
        let ready_at_start = self.newly_ready_lanes.len();
        for _ in 0..ready_at_start {
            let Some(key) = self.newly_ready_lanes.pop_front() else {
                break;
            };
            self.poll_key(key, now, results);
        }
    }

    fn poll_key(&mut self, key: DeliveryKey, now: Instant, results: &mut Vec<DeliveryResult>) {
        let Some(lane) = self.lanes.get_mut(&key) else {
            return;
        };
        let disposition = poll_lane(key, lane, now, self.stall_grace, results);
        match disposition {
            LaneDisposition::Keep => {
                if lane.pending.is_some() {
                    self.cadence_blocked_lanes.push_back(key);
                }
            }
            LaneDisposition::Remove => {
                self.lanes.remove(&key);
            }
        }
    }

    #[cfg(test)]
    fn lane_mut(&mut self, key: DeliveryKey) -> Result<&mut DeliveryLane, DeliveryProtocolError> {
        self.lanes
            .get_mut(&key)
            .ok_or(DeliveryProtocolError::UnknownKey(key))
    }
}

enum LaneDisposition {
    Keep,
    Remove,
}

fn poll_lane(
    key: DeliveryKey,
    lane: &mut DeliveryLane,
    now: Instant,
    stall_grace: Duration,
    results: &mut Vec<DeliveryResult>,
) -> LaneDisposition {
    let Some(pending) = &mut lane.pending else {
        return LaneDisposition::Keep;
    };
    loop {
        let mut event = pending
            .events
            .pop_front()
            .expect("a pending delivery batch contains an event");

        if let EngineEvent::Token { timing, .. } = &mut event {
            timing.mark_actor_delivered(now);
        }

        match lane.response_tx.try_send(event) {
            Ok(()) => {
                pending.record_progress(now);
                if !pending.events.is_empty() {
                    continue;
                }

                let terminal = pending.terminal;
                results.push(DeliveryResult::Delivered {
                    key,
                    sequence: pending.sequence,
                    terminal,
                    waited: pending.waited(now),
                });
                lane.pending = None;
                return if terminal {
                    LaneDisposition::Remove
                } else {
                    LaneDisposition::Keep
                };
            }
            Err(mpsc::error::TrySendError::Closed(_)) => {
                results.push(DeliveryResult::Closed {
                    key,
                    sequence: pending.sequence,
                    waited: pending.waited(now),
                    backpressured: pending.was_backpressured(),
                });
                return LaneDisposition::Remove;
            }
            Err(mpsc::error::TrySendError::Full(event)) => {
                pending.events.push_front(event);
                if pending.backpressure_started_at.is_none() {
                    pending.backpressure_started_at = Some(now);
                    results.push(DeliveryResult::BackpressureStarted {
                        key,
                        sequence: pending.sequence,
                        capacity: lane.response_tx.max_capacity(),
                    });
                }

                if pending.continuous_wait(now) >= stall_grace {
                    results.push(DeliveryResult::TimedOut {
                        key,
                        sequence: pending.sequence,
                        waited: pending.waited(now),
                    });
                    return LaneDisposition::Remove;
                }
                return LaneDisposition::Keep;
            }
        }
    }
}

const WORKER_DISCONNECTED_ERROR: &str = "response delivery command channel closed";
const RESULT_SINK_CLOSED_ERROR: &str = "response delivery result sink closed";
const WORKER_DROP_ERROR: &str = "response delivery worker dropped";
const DELIVERY_BARRIER_TIMEOUT: Duration = Duration::from_secs(5);

/// Owning handle for the response-delivery thread.
///
/// Commands use an unbounded standard channel: submission never waits for the
/// worker's poll cadence, and a disconnected worker returns the original
/// command to its caller. Delivery and result publication remain non-blocking.
pub(crate) struct DeliveryWorker {
    command_tx: Option<std_mpsc::Sender<DeliveryCommand>>,
    join: Option<JoinHandle<()>>,
}

impl DeliveryWorker {
    pub(crate) fn start<S>(
        stall_grace: Duration,
        poll_cadence: Duration,
        sink: S,
    ) -> io::Result<Self>
    where
        S: DeliveryResultSink,
    {
        if poll_cadence.is_zero() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "response delivery poll cadence must be greater than zero",
            ));
        }

        let (command_tx, command_rx) = std_mpsc::channel();
        let join = thread::Builder::new()
            .name("kiln-response-delivery".into())
            .spawn(move || {
                run_delivery_worker(
                    DeliveryState::new(stall_grace),
                    poll_cadence,
                    sink,
                    command_rx,
                );
            })?;
        Ok(Self {
            command_tx: Some(command_tx),
            join: Some(join),
        })
    }

    /// Queue a command, returning it intact if the worker has stopped.
    pub(crate) fn command(&self, command: DeliveryCommand) -> Result<(), DeliveryCommand> {
        let Some(command_tx) = &self.command_tx else {
            return Err(command);
        };
        command_tx.send(command).map_err(|error| error.0)
    }

    /// Wait until every result produced before this FIFO barrier has been
    /// accepted and published as one or more sink cohorts.
    pub(crate) fn barrier(&self) -> Result<(), DeliveryBarrierError> {
        let (reply, reply_rx) = std_mpsc::channel();
        self.command(DeliveryCommand::Barrier { reply })
            .map_err(|_| DeliveryBarrierError)?;
        reply_rx
            .recv_timeout(DELIVERY_BARRIER_TIMEOUT)
            .map_err(|_| DeliveryBarrierError)
    }

    /// Request ordered best-effort lane shutdown and join the worker thread.
    pub(crate) fn shutdown(&mut self, error: String) -> thread::Result<()> {
        let _ = self.command(DeliveryCommand::Shutdown { error });
        self.join()
    }

    /// Close command input and join, even if an explicit shutdown was omitted.
    pub(crate) fn join(&mut self) -> thread::Result<()> {
        self.command_tx.take();
        match self.join.take() {
            Some(join) => join.join(),
            None => Ok(()),
        }
    }
}

impl Drop for DeliveryWorker {
    fn drop(&mut self) {
        if self.join.is_none() {
            return;
        }
        if let Some(command_tx) = self.command_tx.take() {
            let _ = command_tx.send(DeliveryCommand::Shutdown {
                error: WORKER_DROP_ERROR.into(),
            });
        }
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
    }
}

fn run_delivery_worker<S>(
    mut state: DeliveryState,
    poll_cadence: Duration,
    mut sink: S,
    command_rx: std_mpsc::Receiver<DeliveryCommand>,
) where
    S: DeliveryResultSink,
{
    let mut pending_results = VecDeque::new();
    let mut pending_barriers = VecDeque::new();
    let mut results_enqueued = 0_u128;
    let mut results_published = 0_u128;
    let mut next_poll = Instant::now();

    loop {
        let flush = flush_delivery_results(&mut sink, &mut pending_results);
        if flush.is_closed() {
            let _ = state.handle(DeliveryCommand::Shutdown {
                error: RESULT_SINK_CLOSED_ERROR.into(),
            });
            return;
        }
        results_published += flush.accepted as u128;
        acknowledge_delivery_barriers(&mut pending_barriers, results_published);

        let now = Instant::now();
        if now >= next_poll {
            enqueue_delivery_results(&mut pending_results, &mut results_enqueued, state.poll(now));
            next_poll = now.checked_add(poll_cadence).unwrap_or(now);
            let flush = flush_delivery_results(&mut sink, &mut pending_results);
            if flush.is_closed() {
                let _ = state.handle(DeliveryCommand::Shutdown {
                    error: RESULT_SINK_CLOSED_ERROR.into(),
                });
                return;
            }
            results_published += flush.accepted as u128;
            acknowledge_delivery_barriers(&mut pending_barriers, results_published);
        }

        let received = if !state.has_pending() && pending_results.is_empty() {
            match command_rx.recv() {
                Ok(command) => WorkerReceive::Command(command),
                Err(_) => WorkerReceive::Disconnected,
            }
        } else {
            let wait = next_poll.saturating_duration_since(Instant::now());
            match command_rx.recv_timeout(wait) {
                Ok(command) => WorkerReceive::Command(command),
                Err(std_mpsc::RecvTimeoutError::Timeout) => WorkerReceive::Timeout,
                Err(std_mpsc::RecvTimeoutError::Disconnected) => WorkerReceive::Disconnected,
            }
        };
        match received {
            WorkerReceive::Command(command) => {
                if let DeliveryCommand::Barrier { reply } = command {
                    pending_barriers.push_back(PendingDeliveryBarrier {
                        result_target: results_enqueued,
                        reply,
                    });
                    acknowledge_delivery_barriers(&mut pending_barriers, results_published);
                    continue;
                }
                let shutdown = matches!(&command, DeliveryCommand::Shutdown { .. });
                let poll_immediately = matches!(
                    &command,
                    DeliveryCommand::Deliver { .. }
                        | DeliveryCommand::DeliverMany { .. }
                        | DeliveryCommand::Terminate { .. }
                );
                if let Err(error) = state.handle(command) {
                    enqueue_delivery_results(
                        &mut pending_results,
                        &mut results_enqueued,
                        [DeliveryResult::ProtocolError(error)],
                    );
                }
                if shutdown {
                    return;
                }
                if poll_immediately {
                    enqueue_delivery_results(
                        &mut pending_results,
                        &mut results_enqueued,
                        state.poll_ready(Instant::now()),
                    );
                }
            }
            WorkerReceive::Timeout => {}
            WorkerReceive::Disconnected => {
                let _ = state.handle(DeliveryCommand::Shutdown {
                    error: WORKER_DISCONNECTED_ERROR.into(),
                });
                return;
            }
        }
    }
}

struct PendingDeliveryBarrier {
    result_target: u128,
    reply: std_mpsc::Sender<()>,
}

fn enqueue_delivery_results<I>(
    pending_results: &mut VecDeque<DeliveryResult>,
    results_enqueued: &mut u128,
    results: I,
) where
    I: IntoIterator<Item = DeliveryResult>,
{
    for result in results {
        pending_results.push_back(result);
        *results_enqueued += 1;
    }
}

fn acknowledge_delivery_barriers(
    pending_barriers: &mut VecDeque<PendingDeliveryBarrier>,
    results_published: u128,
) {
    while pending_barriers
        .front()
        .is_some_and(|barrier| barrier.result_target <= results_published)
    {
        let barrier = pending_barriers
            .pop_front()
            .expect("front barrier remains present");
        let _ = barrier.reply.send(());
    }
}

enum WorkerReceive {
    Command(DeliveryCommand),
    Timeout,
    Disconnected,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SinkStatus {
    Open,
    Closed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlushOutcome {
    status: SinkStatus,
    accepted: usize,
}

impl FlushOutcome {
    fn is_closed(self) -> bool {
        self.status.is_closed()
    }
}

impl SinkStatus {
    fn is_closed(self) -> bool {
        matches!(self, Self::Closed)
    }
}

fn flush_delivery_results<S>(
    sink: &mut S,
    pending_results: &mut VecDeque<DeliveryResult>,
) -> FlushOutcome
where
    S: DeliveryResultSink,
{
    let mut accepted = 0;
    while let Some(result) = pending_results.pop_front() {
        match sink.try_send(result) {
            Ok(()) => accepted += 1,
            Err(DeliveryResultSinkError::Full(result)) => {
                pending_results.push_front(result);
                if accepted > 0 && sink.notify().is_err() {
                    return FlushOutcome {
                        status: SinkStatus::Closed,
                        accepted,
                    };
                }
                return FlushOutcome {
                    status: SinkStatus::Open,
                    accepted,
                };
            }
            Err(DeliveryResultSinkError::Closed(_)) => {
                if accepted > 0 {
                    let _ = sink.notify();
                }
                return FlushOutcome {
                    status: SinkStatus::Closed,
                    accepted,
                };
            }
        }
    }
    if accepted > 0 && sink.notify().is_err() {
        return FlushOutcome {
            status: SinkStatus::Closed,
            accepted,
        };
    }
    FlushOutcome {
        status: SinkStatus::Open,
        accepted,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc as std_mpsc;
    use std::sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::{Duration, Instant};

    use kiln_model::FinishReason;
    use tokio::sync::mpsc;
    use uuid::Uuid;

    use super::*;

    const GRACE: Duration = Duration::from_millis(50);

    fn key(request: u128, generation: u64) -> DeliveryKey {
        DeliveryKey::new(Uuid::from_u128(request), generation)
    }

    fn token(token: TokenId, ready_at: Instant, sequence: u64) -> DeliveryBatch {
        DeliveryBatch::Token {
            token,
            timing: EngineTokenTiming::ready(ready_at, Default::default()),
            sequence,
        }
    }

    fn timing(ready_at: Instant) -> EngineTokenTiming {
        EngineTokenTiming::ready(ready_at, Default::default())
    }

    fn assert_token_event(event: EngineEvent, expected_token: TokenId, expected_ready_at: Instant) {
        match event {
            EngineEvent::Token { token, timing } => {
                assert_eq!(token, expected_token);
                assert_eq!(timing.ready_at, expected_ready_at);
                assert!(
                    timing
                        .actor_delivered_at
                        .is_some_and(|at| at >= expected_ready_at)
                );
            }
            other => panic!("expected token event, got {other:?}"),
        }
    }

    fn output(token: TokenId) -> BatchedGenerationOutput {
        BatchedGenerationOutput {
            text: token.to_string(),
            token_ids: vec![token],
            finish_reason: FinishReason::MaxTokens,
            completion_tokens: 1,
            action_tokens: None,
            prefill_duration: Duration::from_millis(2),
            decode_duration: Duration::from_millis(3),
            actor_queue_duration: Duration::from_millis(4),
            actor_admission_duration: Duration::from_millis(1),
            actor_prefill_wall_duration: Some(Duration::from_millis(5)),
        }
    }

    fn register(
        state: &mut DeliveryState,
        key: DeliveryKey,
        capacity: usize,
    ) -> mpsc::Receiver<EngineEvent> {
        let (response_tx, response_rx) = mpsc::channel(capacity);
        state
            .handle(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        response_rx
    }

    fn occupy_lane(state: &DeliveryState, key: DeliveryKey) {
        state
            .lanes
            .get(&key)
            .expect("registered delivery lane")
            .response_tx
            .try_send(EngineEvent::Error("occupied".into()))
            .unwrap();
    }

    fn recv_engine_event(
        response_rx: &mut mpsc::Receiver<EngineEvent>,
        timeout: Duration,
    ) -> EngineEvent {
        let deadline = Instant::now() + timeout;
        loop {
            match response_rx.try_recv() {
                Ok(event) => return event,
                Err(mpsc::error::TryRecvError::Empty) if Instant::now() < deadline => {
                    thread::sleep(Duration::from_millis(1));
                }
                Err(error) => panic!("response event was not delivered: {error:?}"),
            }
        }
    }

    struct NotifyingChannelSink {
        result_tx: std_mpsc::Sender<DeliveryResult>,
        notifications: Arc<AtomicUsize>,
    }

    impl DeliveryResultSink for NotifyingChannelSink {
        fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError> {
            self.result_tx
                .send(result)
                .map_err(|error| DeliveryResultSinkError::Closed(error.0))
        }

        fn notify(&mut self) -> Result<(), DeliveryResultNotifyError> {
            self.notifications.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
    }

    struct BoundedRecordingSink {
        capacity: usize,
        accepted: Vec<DeliveryResult>,
        notified_lengths: Vec<usize>,
        notification_fails: bool,
    }

    impl DeliveryResultSink for BoundedRecordingSink {
        fn try_send(&mut self, result: DeliveryResult) -> Result<(), DeliveryResultSinkError> {
            if self.accepted.len() >= self.capacity {
                return Err(DeliveryResultSinkError::Full(result));
            }
            self.accepted.push(result);
            Ok(())
        }

        fn notify(&mut self) -> Result<(), DeliveryResultNotifyError> {
            self.notified_lengths.push(self.accepted.len());
            if self.notification_fails {
                Err(DeliveryResultNotifyError)
            } else {
                Ok(())
            }
        }
    }

    #[test]
    fn worker_thread_delivers_and_shuts_down_cleanly() {
        let ready_at = Instant::now();
        let (result_tx, result_rx) = std_mpsc::channel();
        let mut worker = DeliveryWorker::start(GRACE, Duration::from_millis(1), result_tx).unwrap();
        let key = key(11, 0);
        let (response_tx, mut response_rx) = mpsc::channel(1);

        worker
            .command(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key,
                batch: token(31, ready_at, 0),
            })
            .unwrap();

        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: false,
                waited: Duration::ZERO,
            }
        );
        assert_token_event(
            recv_engine_event(&mut response_rx, Duration::from_secs(1)),
            31,
            ready_at,
        );

        worker.shutdown("test complete".into()).unwrap();
        let unsent = worker
            .command(DeliveryCommand::Shutdown {
                error: "late".into(),
            })
            .expect_err("joined worker returns the unsent command");
        assert!(matches!(unsent, DeliveryCommand::Shutdown { .. }));
    }

    #[test]
    fn worker_deliver_many_is_immediate_and_notifies_once_for_all_results() {
        let ready_at = Instant::now();
        let (result_tx, result_rx) = std_mpsc::channel();
        let notifications = Arc::new(AtomicUsize::new(0));
        let mut worker = DeliveryWorker::start(
            Duration::from_secs(1),
            Duration::from_secs(60),
            NotifyingChannelSink {
                result_tx,
                notifications: notifications.clone(),
            },
        )
        .unwrap();
        let key_a = key(1, 0);
        let key_b = key(2, 0);
        let (response_tx_a, mut response_rx_a) = mpsc::channel(1);
        let (response_tx_b, mut response_rx_b) = mpsc::channel(1);
        worker
            .command(DeliveryCommand::Register {
                key: key_a,
                response_tx: response_tx_a,
            })
            .unwrap();
        worker
            .command(DeliveryCommand::Register {
                key: key_b,
                response_tx: response_tx_b,
            })
            .unwrap();
        worker
            .command(DeliveryCommand::DeliverMany {
                deliveries: vec![
                    (key_a, token(10, ready_at, 0)),
                    (key_b, token(20, ready_at, 0)),
                ],
            })
            .unwrap();

        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            DeliveryResult::Delivered {
                key: key_a,
                sequence: 0,
                terminal: false,
                waited: Duration::ZERO,
            }
        );
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            DeliveryResult::Delivered {
                key: key_b,
                sequence: 0,
                terminal: false,
                waited: Duration::ZERO,
            }
        );
        let notify_deadline = Instant::now() + Duration::from_secs(1);
        while notifications.load(Ordering::SeqCst) == 0 && Instant::now() < notify_deadline {
            thread::yield_now();
        }
        assert_eq!(notifications.load(Ordering::SeqCst), 1);
        assert_token_event(
            recv_engine_event(&mut response_rx_a, Duration::from_secs(1)),
            10,
            ready_at,
        );
        assert_token_event(
            recv_engine_event(&mut response_rx_b, Duration::from_secs(1)),
            20,
            ready_at,
        );
        worker.shutdown("test complete".into()).unwrap();
    }

    #[test]
    fn worker_barrier_waits_for_prior_result_publication() {
        let ready_at = Instant::now();
        let key = key(7, 0);
        let occupied = DeliveryResult::BackpressureStarted {
            key,
            sequence: 99,
            capacity: 1,
        };
        let (result_tx, result_rx) = std_mpsc::sync_channel(1);
        result_tx.try_send(occupied.clone()).unwrap();
        let mut worker = DeliveryWorker::start(GRACE, Duration::from_millis(1), result_tx).unwrap();
        let (response_tx, mut response_rx) = mpsc::channel(1);
        worker
            .command(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key,
                batch: token(77, ready_at, 0),
            })
            .unwrap();
        assert_token_event(
            recv_engine_event(&mut response_rx, Duration::from_secs(1)),
            77,
            ready_at,
        );

        thread::scope(|scope| {
            let (barrier_done_tx, barrier_done_rx) = std_mpsc::channel();
            let worker_ref = &worker;
            scope.spawn(move || {
                barrier_done_tx.send(worker_ref.barrier()).unwrap();
            });
            assert!(matches!(
                barrier_done_rx.recv_timeout(Duration::from_millis(20)),
                Err(std_mpsc::RecvTimeoutError::Timeout)
            ));

            assert_eq!(result_rx.recv().unwrap(), occupied);
            assert_eq!(
                barrier_done_rx
                    .recv_timeout(Duration::from_secs(1))
                    .unwrap(),
                Ok(())
            );
            assert_eq!(
                result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
                DeliveryResult::Delivered {
                    key,
                    sequence: 0,
                    terminal: false,
                    waited: Duration::ZERO,
                }
            );
        });

        worker.shutdown("test complete".into()).unwrap();
        assert_eq!(worker.barrier(), Err(DeliveryBarrierError));
    }

    #[test]
    fn worker_barrier_is_not_acknowledged_after_notification_failure() {
        let ready_at = Instant::now();
        let mut worker = DeliveryWorker::start(
            GRACE,
            Duration::from_millis(1),
            BoundedRecordingSink {
                capacity: 1,
                accepted: Vec::new(),
                notified_lengths: Vec::new(),
                notification_fails: true,
            },
        )
        .unwrap();
        let key = key(8, 0);
        let (response_tx, _response_rx) = mpsc::channel(1);
        worker
            .command(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key,
                batch: token(88, ready_at, 0),
            })
            .unwrap();

        assert_eq!(worker.barrier(), Err(DeliveryBarrierError));
        worker.join().unwrap();
    }

    #[test]
    fn worker_thread_services_a_ready_lane_while_a_peer_is_full() {
        let ready_at = Instant::now();
        let (result_tx, result_rx) = std_mpsc::channel();
        let mut worker =
            DeliveryWorker::start(Duration::from_secs(1), Duration::from_millis(1), result_tx)
                .unwrap();
        let key_a = key(1, 0);
        let key_b = key(2, 0);
        let (response_tx_a, mut response_rx_a) = mpsc::channel(1);
        let (response_tx_b, mut response_rx_b) = mpsc::channel(1);
        response_tx_a
            .try_send(EngineEvent::Error("occupied".into()))
            .unwrap();

        worker
            .command(DeliveryCommand::Register {
                key: key_a,
                response_tx: response_tx_a,
            })
            .unwrap();
        worker
            .command(DeliveryCommand::Register {
                key: key_b,
                response_tx: response_tx_b,
            })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key: key_a,
                batch: token(10, ready_at, 0),
            })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key: key_b,
                batch: token(20, ready_at, 0),
            })
            .unwrap();

        let first = result_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let second = result_rx.recv_timeout(Duration::from_secs(1)).unwrap();
        let results = [first, second];
        assert!(results.contains(&DeliveryResult::BackpressureStarted {
            key: key_a,
            sequence: 0,
            capacity: 1,
        }));
        assert!(results.contains(&DeliveryResult::Delivered {
            key: key_b,
            sequence: 0,
            terminal: false,
            waited: Duration::ZERO,
        }));
        assert_token_event(
            recv_engine_event(&mut response_rx_b, Duration::from_secs(1)),
            20,
            ready_at,
        );
        assert_eq!(
            response_rx_a.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );
        worker.shutdown("test complete".into()).unwrap();
    }

    #[test]
    fn worker_retries_a_full_result_sink_without_losing_the_result() {
        let ready_at = Instant::now();
        let key = key(9, 0);
        let occupied = DeliveryResult::BackpressureStarted {
            key,
            sequence: 99,
            capacity: 1024,
        };
        let (result_tx, result_rx) = std_mpsc::sync_channel(1);
        result_tx.try_send(occupied.clone()).unwrap();
        let mut worker = DeliveryWorker::start(GRACE, Duration::from_millis(1), result_tx).unwrap();
        let (response_tx, mut response_rx) = mpsc::channel(1);
        worker
            .command(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key,
                batch: token(77, ready_at, 0),
            })
            .unwrap();

        // The event proves the worker completed the lane attempt while its
        // corresponding result could not enter the full sink.
        assert_token_event(
            recv_engine_event(&mut response_rx, Duration::from_secs(1)),
            77,
            ready_at,
        );
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            occupied
        );
        assert_eq!(
            result_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
            DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: false,
                waited: Duration::ZERO,
            }
        );
        worker.shutdown("test complete".into()).unwrap();
    }

    #[test]
    fn partial_result_flush_notifies_once_before_full_and_once_after_retry() {
        let first = DeliveryResult::Delivered {
            key: key(1, 0),
            sequence: 0,
            terminal: false,
            waited: Duration::ZERO,
        };
        let second = DeliveryResult::Delivered {
            key: key(2, 0),
            sequence: 0,
            terminal: false,
            waited: Duration::ZERO,
        };
        let mut pending = VecDeque::from(vec![first.clone(), second.clone()]);
        let mut sink = BoundedRecordingSink {
            capacity: 1,
            accepted: Vec::new(),
            notified_lengths: Vec::new(),
            notification_fails: false,
        };

        assert_eq!(
            flush_delivery_results(&mut sink, &mut pending).status,
            SinkStatus::Open
        );
        assert_eq!(sink.accepted, vec![first.clone()]);
        assert_eq!(sink.notified_lengths, vec![1]);
        assert_eq!(pending, VecDeque::from(vec![second.clone()]));

        sink.capacity = 2;
        assert_eq!(
            flush_delivery_results(&mut sink, &mut pending).status,
            SinkStatus::Open
        );
        assert_eq!(sink.accepted, vec![first, second]);
        assert_eq!(sink.notified_lengths, vec![1, 2]);
        assert!(pending.is_empty());
    }

    #[test]
    fn result_flush_closes_when_cohort_notification_fails() {
        let result = DeliveryResult::Delivered {
            key: key(1, 0),
            sequence: 0,
            terminal: false,
            waited: Duration::ZERO,
        };
        let mut pending = VecDeque::from(vec![result.clone()]);
        let mut sink = BoundedRecordingSink {
            capacity: 1,
            accepted: Vec::new(),
            notified_lengths: Vec::new(),
            notification_fails: true,
        };

        assert_eq!(
            flush_delivery_results(&mut sink, &mut pending).status,
            SinkStatus::Closed
        );
        assert_eq!(sink.accepted, vec![result]);
        assert_eq!(sink.notified_lengths, vec![1]);
        assert!(pending.is_empty());
    }

    #[test]
    fn worker_terminates_registered_lanes_when_the_result_sink_closes() {
        let ready_at = Instant::now();
        let (result_tx, result_rx) = std_mpsc::sync_channel(1);
        drop(result_rx);
        let mut worker = DeliveryWorker::start(GRACE, Duration::from_millis(1), result_tx).unwrap();
        let key = key(9, 0);
        let (response_tx, mut response_rx) = mpsc::channel(2);
        worker
            .command(DeliveryCommand::Register { key, response_tx })
            .unwrap();
        worker
            .command(DeliveryCommand::Deliver {
                key,
                batch: token(88, ready_at, 0),
            })
            .unwrap();

        assert_token_event(
            recv_engine_event(&mut response_rx, Duration::from_secs(1)),
            88,
            ready_at,
        );
        assert_eq!(
            recv_engine_event(&mut response_rx, Duration::from_secs(1)),
            EngineEvent::Error(RESULT_SINK_CLOSED_ERROR.into())
        );
        worker.join().unwrap();
    }

    #[test]
    fn worker_drop_wakes_and_joins_even_with_a_long_poll_cadence() {
        let (result_tx, result_rx) = std_mpsc::channel();
        let worker = DeliveryWorker::start(GRACE, Duration::from_secs(60), result_tx).unwrap();

        let started = Instant::now();
        drop(worker);
        assert!(started.elapsed() < Duration::from_secs(1));
        assert!(matches!(
            result_rx.recv_timeout(Duration::from_millis(100)),
            Err(std_mpsc::RecvTimeoutError::Disconnected)
        ));
    }

    #[test]
    fn full_lane_does_not_block_ready_peer() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key_a = key(1, 0);
        let key_b = key(2, 0);
        let mut rx_a = register(&mut state, key_a, 1);
        let mut rx_b = register(&mut state, key_b, 1);

        // Occupy A's only slot without involving the state machine.
        occupy_lane(&state, key_a);
        state
            .handle(DeliveryCommand::Deliver {
                key: key_a,
                batch: token(10, now, 0),
            })
            .unwrap();
        state
            .handle(DeliveryCommand::Deliver {
                key: key_b,
                batch: token(20, now, 0),
            })
            .unwrap();

        let results = state.poll(now);
        assert_eq!(
            results,
            vec![
                DeliveryResult::BackpressureStarted {
                    key: key_a,
                    sequence: 0,
                    capacity: 1,
                },
                DeliveryResult::Delivered {
                    key: key_b,
                    sequence: 0,
                    terminal: false,
                    waited: Duration::ZERO,
                },
            ]
        );
        assert_eq!(
            rx_a.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );
        assert_token_event(rx_b.try_recv().unwrap(), 20, now);
    }

    #[test]
    fn immediate_ready_work_does_not_repoll_an_old_blocked_lane() {
        let started = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let blocked_key = key(1, 0);
        let ready_key = key(2, 0);
        let _blocked_rx = register(&mut state, blocked_key, 1);
        let mut ready_rx = register(&mut state, ready_key, 1);
        occupy_lane(&state, blocked_key);
        state
            .handle(DeliveryCommand::Deliver {
                key: blocked_key,
                batch: token(10, started, 0),
            })
            .unwrap();
        assert_eq!(
            state.poll_ready(started),
            vec![DeliveryResult::BackpressureStarted {
                key: blocked_key,
                sequence: 0,
                capacity: 1,
            }]
        );

        state
            .handle(DeliveryCommand::DeliverMany {
                deliveries: vec![(ready_key, token(20, started, 0))],
            })
            .unwrap();
        assert_eq!(
            state.poll_ready(started + GRACE),
            vec![DeliveryResult::Delivered {
                key: ready_key,
                sequence: 0,
                terminal: false,
                waited: Duration::ZERO,
            }],
            "immediate work must not advance the cadence-blocked queue"
        );
        assert_token_event(ready_rx.try_recv().unwrap(), 20, started);

        assert_eq!(
            state.poll(started + GRACE),
            vec![DeliveryResult::TimedOut {
                key: blocked_key,
                sequence: 0,
                waited: GRACE,
            }]
        );
    }

    #[test]
    fn deliver_many_validation_is_all_or_nothing() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key_a = key(1, 0);
        let key_b = key(2, 0);
        let _rx_a = register(&mut state, key_a, 1);
        let _rx_b = register(&mut state, key_b, 1);

        assert_eq!(
            state.handle(DeliveryCommand::DeliverMany {
                deliveries: vec![(key_a, token(10, now, 0)), (key_b, token(20, now, 1))],
            }),
            Err(DeliveryProtocolError::UnexpectedSequence {
                key: key_b,
                expected: 0,
                actual: 1,
            })
        );
        assert!(state.newly_ready_lanes.is_empty());
        for key in [key_a, key_b] {
            let lane = state.lanes.get(&key).unwrap();
            assert_eq!(lane.next_sequence, 0);
            assert!(lane.pending.is_none());
        }

        state
            .handle(DeliveryCommand::Deliver {
                key: key_a,
                batch: token(10, now, 0),
            })
            .unwrap();
    }

    #[test]
    fn terminate_is_idempotent_after_lane_retirement() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let retired_key = key(1, 0);
        let response_rx = register(&mut state, retired_key, 1);
        drop(response_rx);
        state
            .handle(DeliveryCommand::Deliver {
                key: retired_key,
                batch: token(10, now, 0),
            })
            .unwrap();
        assert_eq!(
            state.poll_ready(now),
            vec![DeliveryResult::Closed {
                key: retired_key,
                sequence: 0,
                waited: Duration::ZERO,
                backpressured: false,
            }]
        );

        for _ in 0..2 {
            state
                .handle(DeliveryCommand::Terminate {
                    key: retired_key,
                    error: "late cancellation".into(),
                })
                .unwrap();
        }
        assert_eq!(
            state.handle(DeliveryCommand::Deliver {
                key: retired_key,
                batch: token(11, now, 1),
            }),
            Err(DeliveryProtocolError::UnknownKey(retired_key))
        );
        assert!(!state.has_pending());
    }

    #[test]
    fn terminal_batch_preserves_final_token_then_done_with_one_slot() {
        let ready_at = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 1);
        let expected_output = output(42);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: DeliveryBatch::Terminal {
                    preceding_token: Some((42, timing(ready_at))),
                    terminal: DeliveryTerminal::Done(expected_output.clone()),
                    sequence: 0,
                },
            })
            .unwrap();

        assert_eq!(
            state.poll(ready_at),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        assert_token_event(rx.try_recv().unwrap(), 42, ready_at);

        assert_eq!(
            state.poll(ready_at + Duration::from_millis(1)),
            vec![DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: true,
                waited: Duration::from_millis(1),
            }]
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Done {
                output: expected_output,
            }
        );
        assert_eq!(state.lane_count(), 0);
    }

    #[test]
    fn terminal_batch_delivers_token_and_done_in_one_poll_when_capacity_allows() {
        let ready_at = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 2);
        let expected_output = output(42);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: DeliveryBatch::Terminal {
                    preceding_token: Some((42, timing(ready_at))),
                    terminal: DeliveryTerminal::Done(expected_output.clone()),
                    sequence: 0,
                },
            })
            .unwrap();

        assert_eq!(
            state.poll(ready_at),
            vec![DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: true,
                waited: Duration::ZERO,
            }]
        );
        assert_token_event(rx.try_recv().unwrap(), 42, ready_at);
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Done {
                output: expected_output,
            }
        );
        assert_eq!(state.lane_count(), 0);
        assert!(!state.has_pending());
    }

    #[test]
    fn token_progress_resets_stall_grace_and_preserves_accumulated_wait() {
        let started = Instant::now();
        let token_progress_at = started + Duration::from_millis(40);
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 1);
        let expected_output = output(42);
        occupy_lane(&state, key);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: DeliveryBatch::Terminal {
                    preceding_token: Some((42, timing(started))),
                    terminal: DeliveryTerminal::Done(expected_output.clone()),
                    sequence: 0,
                },
            })
            .unwrap();

        assert_eq!(
            state.poll(started),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );

        // The token advances after 40ms, then Done encounters the now-full
        // one-slot channel. That progress starts a fresh continuous grace
        // episode while retaining the first episode for reported wait time.
        assert_eq!(
            state.poll(token_progress_at),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        assert!(
            state
                .poll(token_progress_at + Duration::from_millis(20))
                .is_empty(),
            "20ms of continuous terminal backpressure must not exhaust a 50ms grace"
        );
        assert_token_event(rx.try_recv().unwrap(), 42, started);

        assert_eq!(
            state.poll(token_progress_at + Duration::from_millis(25)),
            vec![DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: true,
                waited: Duration::from_millis(65),
            }]
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Done {
                output: expected_output,
            }
        );
    }

    #[test]
    fn full_lane_times_out_at_actual_elapsed_grace() {
        let started = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let _rx = register(&mut state, key, 1);
        occupy_lane(&state, key);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: token(10, started, 0),
            })
            .unwrap();

        assert_eq!(
            state.poll(started),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        assert!(
            state
                .poll(started + GRACE - Duration::from_nanos(1))
                .is_empty()
        );
        assert_eq!(
            state.poll(started + GRACE),
            vec![DeliveryResult::TimedOut {
                key,
                sequence: 0,
                waited: GRACE,
            }]
        );
        assert_eq!(state.lane_count(), 0);
    }

    #[test]
    fn closed_lane_reports_elapsed_backpressure_and_is_removed() {
        let started = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let rx = register(&mut state, key, 1);
        occupy_lane(&state, key);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: token(10, started, 0),
            })
            .unwrap();
        assert_eq!(
            state.poll(started),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        drop(rx);

        let waited = Duration::from_millis(17);
        assert_eq!(
            state.poll(started + waited),
            vec![DeliveryResult::Closed {
                key,
                sequence: 0,
                waited,
                backpressured: true,
            }]
        );
        assert_eq!(state.lane_count(), 0);
    }

    #[test]
    fn generations_with_the_same_request_id_have_isolated_lanes() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let old = key(7, 1);
        let new = key(7, 2);
        let mut old_rx = register(&mut state, old, 1);
        let mut new_rx = register(&mut state, new, 1);
        state
            .handle(DeliveryCommand::Deliver {
                key: old,
                batch: token(11, now, 0),
            })
            .unwrap();
        state
            .handle(DeliveryCommand::Deliver {
                key: new,
                batch: token(22, now, 0),
            })
            .unwrap();

        let results = state.poll(now);
        assert_eq!(results.len(), 2);
        assert_token_event(old_rx.try_recv().unwrap(), 11, now);
        assert_token_event(new_rx.try_recv().unwrap(), 22, now);
    }

    #[test]
    fn token_ready_at_survives_backpressure_retries() {
        let ready_at = Instant::now();
        let first_poll = ready_at + Duration::from_millis(9);
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 1);
        occupy_lane(&state, key);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: token(99, ready_at, 0),
            })
            .unwrap();
        state.poll(first_poll);
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );

        let delivered_at = first_poll + Duration::from_millis(12);
        assert_eq!(
            state.poll(delivered_at),
            vec![DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: false,
                waited: Duration::from_millis(12),
            }]
        );
        assert_token_event(rx.try_recv().unwrap(), 99, ready_at);
    }

    #[test]
    fn terminate_keeps_an_accepted_token_ahead_of_error() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 1);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: token(5, now, 0),
            })
            .unwrap();
        state
            .handle(DeliveryCommand::Terminate {
                key,
                error: "cancelled".into(),
            })
            .unwrap();

        assert_eq!(
            state.poll(now),
            vec![DeliveryResult::BackpressureStarted {
                key,
                sequence: 0,
                capacity: 1,
            }]
        );
        assert_token_event(rx.try_recv().unwrap(), 5, now);
        assert_eq!(
            state.poll(now),
            vec![DeliveryResult::Delivered {
                key,
                sequence: 0,
                terminal: true,
                waited: Duration::ZERO,
            }]
        );
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Error("cancelled".into())
        );
    }

    #[test]
    fn shutdown_and_drop_never_wait_for_full_lanes() {
        let mut state = DeliveryState::new(GRACE);
        let active_key = key(1, 0);
        let mut rx = register(&mut state, active_key, 1);
        occupy_lane(&state, active_key);

        state
            .handle(DeliveryCommand::Shutdown {
                error: "server stopping".into(),
            })
            .unwrap();
        assert_eq!(state.lane_count(), 0);
        assert!(matches!(
            state.handle(DeliveryCommand::Terminate {
                key: active_key,
                error: "late".into(),
            }),
            Err(DeliveryProtocolError::Shutdown)
        ));
        assert_eq!(
            rx.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));

        // Dropping a separate state with a full channel likewise only drops
        // senders; there is no background wait or terminal-send loop.
        let mut dropped = DeliveryState::new(GRACE);
        let dropped_key = key(2, 0);
        let mut dropped_rx = register(&mut dropped, dropped_key, 1);
        occupy_lane(&dropped, dropped_key);
        drop(dropped);
        assert_eq!(
            dropped_rx.try_recv().unwrap(),
            EngineEvent::Error("occupied".into())
        );
        assert!(matches!(
            dropped_rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn shutdown_never_overtakes_a_pending_token_on_a_one_slot_lane() {
        let ready_at = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let key = key(1, 0);
        let mut rx = register(&mut state, key, 1);
        state
            .handle(DeliveryCommand::Deliver {
                key,
                batch: token(17, ready_at, 0),
            })
            .unwrap();

        state
            .handle(DeliveryCommand::Shutdown {
                error: "server stopping".into(),
            })
            .unwrap();

        // Shutdown gets exactly one slot: the accepted token wins it, and the
        // best-effort error is dropped instead of overtaking or waiting.
        assert_token_event(rx.try_recv().unwrap(), 17, ready_at);
        assert!(matches!(
            rx.try_recv(),
            Err(mpsc::error::TryRecvError::Disconnected)
        ));
    }

    #[test]
    fn rejects_busy_and_out_of_order_batches() {
        let now = Instant::now();
        let mut state = DeliveryState::new(GRACE);
        let active_key = key(1, 0);
        let mut rx = register(&mut state, active_key, 2);

        assert_eq!(
            state.handle(DeliveryCommand::Deliver {
                key: active_key,
                batch: token(1, now, 1),
            }),
            Err(DeliveryProtocolError::UnexpectedSequence {
                key: active_key,
                expected: 0,
                actual: 1,
            })
        );
        state
            .handle(DeliveryCommand::Deliver {
                key: active_key,
                batch: token(1, now, 0),
            })
            .unwrap();
        assert_eq!(
            state.handle(DeliveryCommand::Deliver {
                key: active_key,
                batch: token(2, now, 1),
            }),
            Err(DeliveryProtocolError::LaneBusy(active_key))
        );
        state.poll(now);
        rx.try_recv().unwrap();
        state
            .handle(DeliveryCommand::Deliver {
                key: active_key,
                batch: token(2, now, 1),
            })
            .unwrap();

        let exhausted_key = key(2, 0);
        let _rx = register(&mut state, exhausted_key, 1);
        state.lane_mut(exhausted_key).unwrap().next_sequence = u64::MAX;
        assert_eq!(
            state.handle(DeliveryCommand::Deliver {
                key: exhausted_key,
                batch: token(3, now, u64::MAX),
            }),
            Err(DeliveryProtocolError::SequenceExhausted(exhausted_key))
        );
    }
}
