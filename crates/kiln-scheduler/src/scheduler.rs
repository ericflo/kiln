use kiln_core::block::BlockManager;
use kiln_core::request::{Request, RequestId, RequestState};
use kiln_core::token::TokenId;
use std::collections::VecDeque;

/// Configuration for the scheduler.
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Maximum total tokens in a single forward pass (decode + prefill combined).
    /// This is the "token budget" per iteration. Sarathi-style chunked prefill
    /// ensures prefills never starve decode requests.
    pub max_batch_tokens: usize,

    /// Maximum number of concurrent sequences.
    pub max_batch_size: usize,

    /// KV cache block size (tokens per block).
    pub block_size: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
        }
    }
}

/// A request selected for the current iteration, with the tokens it should process.
#[derive(Debug)]
pub struct ScheduledRequest {
    pub request_id: RequestId,
    /// Number of tokens this request contributes to the current forward pass.
    /// - Decode: always 1
    /// - Prefill: chunk_size (may be less than full prompt if budget-constrained)
    pub num_tokens: usize,
    /// Whether this is a prefill or decode step.
    pub is_prefill: bool,
}

/// Output of a scheduling step — tells the model runner what to execute.
#[derive(Debug)]
pub struct SchedulerOutput {
    pub scheduled: Vec<ScheduledRequest>,
    pub total_tokens: usize,
    pub num_prefill_tokens: usize,
    pub num_decode_tokens: usize,
    pub completed_ids: Vec<RequestId>,
}

/// Iteration-level continuous batching scheduler.
///
/// Implements the Sarathi-style chunked prefill algorithm:
/// 1. Pack all active decode requests (1 token each) — these have priority
/// 2. Continue partial prefills with remaining budget
/// 3. Start new prefills with any remaining budget
///
/// This ensures decode requests are never stalled by long prefills.
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: Vec<Request>,
    block_manager: BlockManager,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, num_blocks: usize) -> Self {
        let block_manager = BlockManager::new(num_blocks, config.block_size);
        Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            block_manager,
        }
    }

    /// Add a new request to the waiting queue.
    pub fn add_request(&mut self, request: Request) {
        tracing::debug!(id = %request.id, prompt_len = request.prompt_tokens.len(), "request queued");
        self.waiting.push_back(request);
    }

    /// Cancel a request by ID. Returns true if found and cancelled.
    pub fn cancel_request(&mut self, id: &RequestId) -> bool {
        // Check waiting queue
        if let Some(pos) = self.waiting.iter().position(|r| r.id == *id) {
            self.waiting.remove(pos);
            tracing::debug!(%id, "cancelled waiting request");
            return true;
        }
        // Check running requests
        if let Some(req) = self.running.iter_mut().find(|r| r.id == *id) {
            req.state = RequestState::Cancelled;
            tracing::debug!(%id, "cancelled running request");
            return true;
        }
        false
    }

    /// Run one scheduling step. Returns what the model should execute this iteration.
    pub fn step(&mut self) -> SchedulerOutput {
        let mut completed_ids = Vec::new();

        // 1. Remove completed/cancelled requests, free their blocks
        self.running.retain_mut(|req| {
            let should_remove = matches!(
                req.state,
                RequestState::Complete | RequestState::Cancelled
            );
            if should_remove {
                self.block_manager.free_all(&req.block_ids);
                completed_ids.push(req.id);
                tracing::debug!(id = %req.id, "freed {} blocks", req.block_ids.len());
            }
            !should_remove
        });

        let mut scheduled = Vec::new();
        let mut budget = self.config.max_batch_tokens;
        let mut num_prefill_tokens = 0;
        let mut num_decode_tokens = 0;

        // 2. Pack all active decode requests (1 token each) — highest priority
        for req in &self.running {
            if budget == 0 || scheduled.len() >= self.config.max_batch_size {
                break;
            }
            if req.state == RequestState::Decoding {
                // Ensure this request has a block for its next token
                let needed_blocks = req.blocks_needed(self.config.block_size);
                if needed_blocks > req.block_ids.len() && !self.block_manager.can_allocate(1) {
                    // Can't allocate — skip this request (preemption could go here)
                    continue;
                }
                scheduled.push(ScheduledRequest {
                    request_id: req.id,
                    num_tokens: 1,
                    is_prefill: false,
                });
                budget -= 1;
                num_decode_tokens += 1;
            }
        }

        // 3. Continue partial prefills
        for req in &self.running {
            if budget == 0 || scheduled.len() >= self.config.max_batch_size {
                break;
            }
            if let RequestState::Prefilling { .. } = req.state {
                let remaining = req.remaining_prefill();
                if remaining > 0 {
                    let chunk = remaining.min(budget);
                    scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        num_tokens: chunk,
                        is_prefill: true,
                    });
                    budget -= chunk;
                    num_prefill_tokens += chunk;
                }
            }
        }

        // 4. Promote waiting requests and start new prefills
        let mut promoted = Vec::new();
        while budget > 0 && scheduled.len() < self.config.max_batch_size {
            let Some(mut req) = self.waiting.pop_front() else {
                break;
            };

            // Allocate blocks for the full prompt
            let blocks_needed = req.blocks_needed(self.config.block_size);
            if !self.block_manager.can_allocate(blocks_needed) {
                // Can't fit this request — put it back and stop promoting
                self.waiting.push_front(req);
                break;
            }

            let blocks = self.block_manager.allocate(blocks_needed).unwrap();
            req.block_ids = blocks;
            req.state = RequestState::Prefilling {
                tokens_processed: 0,
            };

            let chunk = req.prompt_tokens.len().min(budget);
            scheduled.push(ScheduledRequest {
                request_id: req.id,
                num_tokens: chunk,
                is_prefill: true,
            });
            budget -= chunk;
            num_prefill_tokens += chunk;

            tracing::debug!(
                id = %req.id,
                prompt_len = req.prompt_tokens.len(),
                chunk_size = chunk,
                blocks = req.block_ids.len(),
                "promoted request"
            );
            promoted.push(req);
        }

        self.running.extend(promoted);

        let total_tokens = num_prefill_tokens + num_decode_tokens;

        SchedulerOutput {
            scheduled,
            total_tokens,
            num_prefill_tokens,
            num_decode_tokens,
            completed_ids,
        }
    }

    /// Update a request's state after the model produces output.
    /// Called by the engine after each forward pass.
    pub fn update_request(
        &mut self,
        id: &RequestId,
        new_token: Option<TokenId>,
        finished: bool,
        prefill_tokens_processed: Option<usize>,
    ) {
        let Some(req) = self.running.iter_mut().find(|r| r.id == *id) else {
            return;
        };

        if let Some(processed) = prefill_tokens_processed {
            if processed >= req.prompt_tokens.len() {
                // Prefill complete, transition to decode
                req.state = RequestState::Decoding;
            } else {
                req.state = RequestState::Prefilling {
                    tokens_processed: processed,
                };
            }
        }

        if let Some(token) = new_token {
            req.output_tokens.push(token);

            // Allocate a new block if needed
            let needed = req.blocks_needed(self.config.block_size);
            if needed > req.block_ids.len() {
                if let Ok(block) = self.block_manager.allocate_one() {
                    req.block_ids.push(block);
                }
            }
        }

        if finished {
            req.state = RequestState::Complete;
        }
    }

    /// Get a reference to a running request by ID.
    pub fn get_request(&self, id: &RequestId) -> Option<&Request> {
        self.running
            .iter()
            .find(|r| r.id == *id)
            .or_else(|| self.waiting.iter().find(|r| r.id == *id))
    }

    pub fn num_waiting(&self) -> usize {
        self.waiting.len()
    }

    pub fn num_running(&self) -> usize {
        self.running.len()
    }

    pub fn block_manager(&self) -> &BlockManager {
        &self.block_manager
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::sampling::SamplingParams;

    fn make_request(prompt_len: usize) -> Request {
        Request::new(vec![1; prompt_len], SamplingParams::greedy(), None)
    }

    #[test]
    fn basic_scheduling() {
        let config = SchedulerConfig {
            max_batch_tokens: 100,
            max_batch_size: 8,
            block_size: 16,
        };
        let mut sched = Scheduler::new(config, 100);

        // Add a request with 50 tokens
        let req = make_request(50);
        let id = req.id;
        sched.add_request(req);

        // First step: should schedule the prefill
        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].num_tokens, 50);
        assert!(output.scheduled[0].is_prefill);
        assert_eq!(output.num_prefill_tokens, 50);

        // Mark prefill complete, transition to decode
        sched.update_request(&id, None, false, Some(50));

        // Next step: should schedule 1 decode token
        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].num_tokens, 1);
        assert!(!output.scheduled[0].is_prefill);
    }

    #[test]
    fn chunked_prefill() {
        let config = SchedulerConfig {
            max_batch_tokens: 30,
            max_batch_size: 8,
            block_size: 16,
        };
        let mut sched = Scheduler::new(config, 100);

        // 50-token prompt, but budget is only 30
        let req = make_request(50);
        let id = req.id;
        sched.add_request(req);

        // First step: chunks to 30 tokens
        let output = sched.step();
        assert_eq!(output.scheduled[0].num_tokens, 30);

        // Update: 30 processed, 20 remaining
        sched.update_request(&id, None, false, Some(30));

        // Second step: remaining 20
        let output = sched.step();
        assert_eq!(output.scheduled[0].num_tokens, 20);
    }

    #[test]
    fn decode_priority_over_prefill() {
        let config = SchedulerConfig {
            max_batch_tokens: 10,
            max_batch_size: 8,
            block_size: 16,
        };
        let mut sched = Scheduler::new(config, 100);

        // Add first request, prefill it, start decoding
        let req1 = make_request(5);
        let id1 = req1.id;
        sched.add_request(req1);
        sched.step(); // prefills req1
        sched.update_request(&id1, None, false, Some(5)); // now decoding

        // Add a second request with a long prompt
        let req2 = make_request(20);
        sched.add_request(req2);

        // Step: decode (1 token) should come first, then prefill fills remaining budget
        let output = sched.step();
        assert_eq!(output.scheduled.len(), 2);
        assert!(!output.scheduled[0].is_prefill); // decode first
        assert!(output.scheduled[1].is_prefill); // prefill second
        assert_eq!(output.scheduled[0].num_tokens, 1);
        assert_eq!(output.scheduled[1].num_tokens, 9); // 10 - 1 = 9 remaining budget
    }

    #[test]
    fn request_completion() {
        let config = SchedulerConfig::default();
        let mut sched = Scheduler::new(config, 100);

        let req = make_request(5);
        let id = req.id;
        sched.add_request(req);

        sched.step();
        sched.update_request(&id, None, false, Some(5));
        sched.update_request(&id, Some(42), true, None);

        // Next step should show it as completed
        let output = sched.step();
        assert!(output.completed_ids.contains(&id));
        assert_eq!(sched.num_running(), 0);
    }
}
