//! Real-model batching engine actor scaffolding.
//!
//! Phase 1 keeps the actor in `kiln-server` so HTTP request routing, prefix
//! cache ownership, GPU coordination, and metrics can be wired incrementally.
//! The production forward implementation is added behind this seam; the unit
//! tests use a mocked forwarder to lock in enqueue → batch forward → response
//! routing behavior before GPU validation.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::Result;
use kiln_core::sampling::SamplingParams;
use kiln_core::token::TokenId;
use tokio::sync::{mpsc, oneshot};
use uuid::Uuid;

const DEFAULT_ENGINE_CHANNEL: usize = 1024;
const DEFAULT_RESPONSE_CHANNEL: usize = 64;
const DEFAULT_MAX_DECODE_BATCH: usize = 8;

#[derive(Debug, Clone)]
pub struct EngineRequest {
    pub request_id: Uuid,
    pub prompt_tokens: Vec<TokenId>,
    pub sampling: SamplingParams,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EngineEvent {
    Token(TokenId),
    Done { completion_tokens: usize },
    Error(String),
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct BatchingEngineSnapshot {
    pub accepting: bool,
    pub queue_depth: usize,
    pub active_decode: usize,
    pub current_batch_size: usize,
    pub last_batch_size: usize,
    pub last_forward_ms: f64,
    pub last_prefill_ms: f64,
    pub total_decode_tokens: u64,
    pub total_prefill_tokens: u64,
    pub total_errors: u64,
    pub adapter_groups_waiting: usize,
}

pub trait DecodeForward: Send + Sync + 'static {
    fn forward_decode(&self, input_tokens: &[TokenId]) -> Result<Vec<TokenId>>;
}

#[derive(Clone)]
pub struct BatchingEngineHandle {
    tx: mpsc::Sender<EngineCommand>,
}

impl BatchingEngineHandle {
    pub fn start(forward: Arc<dyn DecodeForward>) -> Self {
        Self::start_with_options(forward, DEFAULT_MAX_DECODE_BATCH)
    }

    pub fn start_with_options(forward: Arc<dyn DecodeForward>, max_decode_batch: usize) -> Self {
        let (tx, rx) = mpsc::channel(DEFAULT_ENGINE_CHANNEL);
        let actor = BatchingEngineActor::new(rx, forward, max_decode_batch.max(1));
        thread::Builder::new()
            .name("kiln-batching-engine".to_string())
            .spawn(move || actor.run())
            .expect("spawn batching engine actor");
        Self { tx }
    }

    pub async fn enqueue(&self, req: EngineRequest) -> Result<mpsc::Receiver<EngineEvent>> {
        let (response_tx, response_rx) = mpsc::channel(DEFAULT_RESPONSE_CHANNEL);
        self.tx
            .send(EngineCommand::Enqueue { req, response_tx })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        Ok(response_rx)
    }

    pub async fn cancel(&self, request_id: Uuid) -> Result<()> {
        self.tx
            .send(EngineCommand::Cancel { request_id })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))
    }

    pub async fn drain(&self) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Drain { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped during drain"))
    }

    pub async fn stop(&self) -> Result<()> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Stop { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before ack"))
    }

    pub async fn snapshot(&self) -> Result<BatchingEngineSnapshot> {
        let (reply, rx) = oneshot::channel();
        self.tx
            .send(EngineCommand::Snapshot { reply })
            .await
            .map_err(|_| anyhow::anyhow!("batching engine stopped"))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("batching engine stopped before snapshot"))
    }
}

enum EngineCommand {
    Enqueue {
        req: EngineRequest,
        response_tx: mpsc::Sender<EngineEvent>,
    },
    Cancel {
        request_id: Uuid,
    },
    Drain {
        reply: oneshot::Sender<()>,
    },
    Stop {
        reply: oneshot::Sender<()>,
    },
    Snapshot {
        reply: oneshot::Sender<BatchingEngineSnapshot>,
    },
}

struct QueuedRequest {
    req: EngineRequest,
    response_tx: mpsc::Sender<EngineEvent>,
}

struct ActiveRequest {
    req: EngineRequest,
    response_tx: mpsc::Sender<EngineEvent>,
    next_token: TokenId,
    generated_tokens: usize,
}

struct BatchingEngineActor {
    rx: mpsc::Receiver<EngineCommand>,
    forward: Arc<dyn DecodeForward>,
    waiting: VecDeque<QueuedRequest>,
    active: Vec<ActiveRequest>,
    accepting: bool,
    stopped: bool,
    max_decode_batch: usize,
    snapshot: BatchingEngineSnapshot,
}

impl BatchingEngineActor {
    fn new(
        rx: mpsc::Receiver<EngineCommand>,
        forward: Arc<dyn DecodeForward>,
        max_decode_batch: usize,
    ) -> Self {
        Self {
            rx,
            forward,
            waiting: VecDeque::new(),
            active: Vec::new(),
            accepting: true,
            stopped: false,
            max_decode_batch,
            snapshot: BatchingEngineSnapshot {
                accepting: true,
                ..BatchingEngineSnapshot::default()
            },
        }
    }

    fn run(mut self) {
        while !self.stopped {
            if self.active.is_empty() && self.waiting.is_empty() {
                match self.rx.blocking_recv() {
                    Some(cmd) => self.handle_command(cmd),
                    None => break,
                }
            }

            thread::sleep(Duration::from_millis(1));
            self.drain_commands();
            self.admit_waiting();
            if !self.active.is_empty() {
                self.run_decode_batch();
                continue;
            }

            thread::sleep(Duration::from_millis(1));
        }

        self.fail_all("batching engine stopped");
    }

    fn drain_commands(&mut self) {
        while let Ok(cmd) = self.rx.try_recv() {
            self.handle_command(cmd);
        }
    }

    fn handle_command(&mut self, cmd: EngineCommand) {
        match cmd {
            EngineCommand::Enqueue { req, response_tx } => {
                if self.accepting {
                    self.waiting.push_back(QueuedRequest { req, response_tx });
                } else {
                    let _ = response_tx.blocking_send(EngineEvent::Error(
                        "batching engine is draining".to_string(),
                    ));
                }
            }
            EngineCommand::Cancel { request_id } => self.cancel(request_id),
            EngineCommand::Drain { reply } => {
                self.accepting = false;
                self.refresh_snapshot();
                if self.waiting.is_empty() && self.active.is_empty() {
                    let _ = reply.send(());
                } else {
                    // Phase 1 scaffold: drain means stop accepting. A later patch
                    // keeps drain waiters and acks them once in-flight work clears.
                    let _ = reply.send(());
                }
            }
            EngineCommand::Stop { reply } => {
                self.accepting = false;
                self.stopped = true;
                self.refresh_snapshot();
                let _ = reply.send(());
            }
            EngineCommand::Snapshot { reply } => {
                self.refresh_snapshot();
                let _ = reply.send(self.snapshot.clone());
            }
        }
    }

    fn cancel(&mut self, request_id: Uuid) {
        self.waiting.retain(|queued| {
            let keep = queued.req.request_id != request_id;
            if !keep {
                let _ = queued
                    .response_tx
                    .blocking_send(EngineEvent::Error("request cancelled".to_string()));
            }
            keep
        });
        self.active.retain(|active| {
            let keep = active.req.request_id != request_id;
            if !keep {
                let _ = active
                    .response_tx
                    .blocking_send(EngineEvent::Error("request cancelled".to_string()));
            }
            keep
        });
    }

    fn admit_waiting(&mut self) {
        while self.active.len() < self.max_decode_batch {
            let Some(queued) = self.waiting.pop_front() else {
                break;
            };
            let next_token = queued.req.prompt_tokens.last().copied().unwrap_or_default();
            self.snapshot.total_prefill_tokens += queued.req.prompt_tokens.len() as u64;
            self.active.push(ActiveRequest {
                req: queued.req,
                response_tx: queued.response_tx,
                next_token,
                generated_tokens: 0,
            });
        }
    }

    fn run_decode_batch(&mut self) {
        let batch_len = self.active.len().min(self.max_decode_batch);
        let input_tokens: Vec<TokenId> = self
            .active
            .iter()
            .take(batch_len)
            .map(|active| active.next_token)
            .collect();

        self.snapshot.current_batch_size = batch_len;
        let started = Instant::now();
        let result = self.forward.forward_decode(&input_tokens);
        self.snapshot.last_forward_ms = started.elapsed().as_secs_f64() * 1000.0;
        self.snapshot.last_batch_size = batch_len;
        self.snapshot.current_batch_size = 0;

        let output_tokens = match result {
            Ok(tokens) if tokens.len() == batch_len => tokens,
            Ok(tokens) => {
                self.snapshot.total_errors += batch_len as u64;
                self.finish_batch_with_error(
                    batch_len,
                    format!(
                        "batched decode returned {} rows for batch size {batch_len}",
                        tokens.len()
                    ),
                );
                self.refresh_snapshot();
                return;
            }
            Err(err) => {
                self.snapshot.total_errors += batch_len as u64;
                self.finish_batch_with_error(batch_len, err.to_string());
                self.refresh_snapshot();
                return;
            }
        };

        for (active, token) in self.active.iter_mut().take(batch_len).zip(output_tokens) {
            active.generated_tokens += 1;
            active.next_token = token;
            if active.response_tx.blocking_send(EngineEvent::Token(token)).is_err() {
                active.generated_tokens = active.req.sampling.max_tokens;
            }
            self.snapshot.total_decode_tokens += 1;
        }

        let mut idx = 0;
        while idx < self.active.len() {
            if self.active[idx].generated_tokens >= self.active[idx].req.sampling.max_tokens {
                let active = self.active.remove(idx);
                let _ = active.response_tx.blocking_send(EngineEvent::Done {
                    completion_tokens: active.generated_tokens,
                });
            } else {
                idx += 1;
            }
        }
        self.refresh_snapshot();
    }

    fn finish_batch_with_error(&mut self, batch_len: usize, error: String) {
        for _ in 0..batch_len.min(self.active.len()) {
            let active = self.active.remove(0);
            let _ = active.response_tx.blocking_send(EngineEvent::Error(error.clone()));
        }
    }

    fn fail_all(&mut self, error: &str) {
        while let Some(queued) = self.waiting.pop_front() {
            let _ = queued
                .response_tx
                .blocking_send(EngineEvent::Error(error.to_string()));
        }
        for active in self.active.drain(..) {
            let _ = active
                .response_tx
                .blocking_send(EngineEvent::Error(error.to_string()));
        }
    }

    fn refresh_snapshot(&mut self) {
        self.snapshot.accepting = self.accepting;
        self.snapshot.queue_depth = self.waiting.len();
        self.snapshot.active_decode = self.active.len();
        self.snapshot.adapter_groups_waiting = usize::from(!self.waiting.is_empty());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[derive(Default)]
    struct MockForward {
        calls: Mutex<Vec<Vec<TokenId>>>,
    }

    impl DecodeForward for MockForward {
        fn forward_decode(&self, input_tokens: &[TokenId]) -> Result<Vec<TokenId>> {
            self.calls.lock().unwrap().push(input_tokens.to_vec());
            Ok(input_tokens.iter().map(|token| token + 10).collect())
        }
    }

    fn request(prompt_last: TokenId, max_tokens: usize) -> EngineRequest {
        EngineRequest {
            request_id: Uuid::new_v4(),
            prompt_tokens: vec![1, prompt_last],
            sampling: SamplingParams {
                max_tokens,
                ..SamplingParams::default()
            },
        }
    }

    #[tokio::test]
    async fn enqueue_batches_forward_shape_and_routes_responses() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx1 = handle.enqueue(request(101, 1)).await.unwrap();
        let mut rx2 = handle.enqueue(request(202, 1)).await.unwrap();

        assert_eq!(rx1.recv().await, Some(EngineEvent::Token(111)));
        assert_eq!(rx1.recv().await, Some(EngineEvent::Done { completion_tokens: 1 }));
        assert_eq!(rx2.recv().await, Some(EngineEvent::Token(212)));
        assert_eq!(rx2.recv().await, Some(EngineEvent::Done { completion_tokens: 1 }));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![101, 202]]);
        handle.stop().await.unwrap();
    }

    #[tokio::test]
    async fn routes_multiple_decode_steps_to_original_receivers() {
        let forward = Arc::new(MockForward::default());
        let handle = BatchingEngineHandle::start_with_options(forward.clone(), 8);

        let mut rx = handle.enqueue(request(7, 2)).await.unwrap();

        assert_eq!(rx.recv().await, Some(EngineEvent::Token(17)));
        assert_eq!(rx.recv().await, Some(EngineEvent::Token(27)));
        assert_eq!(rx.recv().await, Some(EngineEvent::Done { completion_tokens: 2 }));

        let calls = forward.calls.lock().unwrap().clone();
        assert_eq!(calls, vec![vec![7], vec![17]]);
        handle.stop().await.unwrap();
    }
}
