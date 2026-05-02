use kiln_core::block::BlockManager;
use kiln_core::prefix_cache::PrefixCache;
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

    /// Enable prefix caching for shared prompt prefixes.
    pub prefix_cache_enabled: bool,

    /// Maximum blocks the prefix cache may retain (0 = unlimited up to num_blocks / 4).
    pub prefix_cache_max_blocks: Option<usize>,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_tokens: 8192,
            max_batch_size: 64,
            block_size: 16,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: None,
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

/// Prefix-cache effectiveness counters and gauges exposed by the scheduler.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefixCacheStats {
    pub lookup_hits: u64,
    pub lookup_misses: u64,
    pub hit_tokens: u64,
    pub hit_blocks: u64,
    pub cached_blocks: usize,
    pub max_blocks: usize,
    pub cached_entries: usize,
    pub max_entries: usize,
    pub cached_state_bytes: u64,
    pub max_state_bytes: u64,
}

/// Iteration-level continuous batching scheduler.
///
/// Implements the Sarathi-style chunked prefill algorithm:
/// 1. Pack all active decode requests (1 token each) — these have priority
/// 2. Continue partial prefills with remaining budget
/// 3. Start new prefills with any remaining budget
///
/// This ensures decode requests are never stalled by long prefills.
///
/// When prefix caching is enabled, new requests check the prefix cache before
/// allocating blocks. If a prefix is found, the cached KV blocks are reused and
/// only the non-cached suffix is scheduled for prefill.
pub struct Scheduler {
    config: SchedulerConfig,
    waiting: VecDeque<Request>,
    running: Vec<Request>,
    block_manager: BlockManager,
    prefix_cache: Option<PrefixCache>,
    prefix_cache_stats: PrefixCacheStats,
    /// Tracks which blocks per request came from prefix cache (should not be freed).
    cached_block_counts: std::collections::HashMap<RequestId, usize>,
}

impl Scheduler {
    pub fn new(config: SchedulerConfig, num_blocks: usize) -> Self {
        let block_manager = BlockManager::new(num_blocks, config.block_size);
        let prefix_cache_max_blocks = if config.prefix_cache_enabled {
            config.prefix_cache_max_blocks.unwrap_or(num_blocks / 4)
        } else {
            0
        };
        let prefix_cache = if config.prefix_cache_enabled {
            Some(PrefixCache::new(config.block_size, prefix_cache_max_blocks))
        } else {
            None
        };
        Self {
            config,
            waiting: VecDeque::new(),
            running: Vec::new(),
            block_manager,
            prefix_cache,
            prefix_cache_stats: PrefixCacheStats {
                max_blocks: prefix_cache_max_blocks,
                ..Default::default()
            },
            cached_block_counts: std::collections::HashMap::new(),
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
                let cached_count = self.cached_block_counts.remove(&req.id).unwrap_or(0);
                // Free only the non-cached blocks (suffix blocks allocated by us)
                if cached_count < req.block_ids.len() {
                    self.block_manager.free_all(&req.block_ids[cached_count..]);
                }
                // Release the cached blocks back to the prefix cache
                if cached_count > 0 {
                    if let Some(ref mut pc) = self.prefix_cache {
                        pc.release_blocks(&req.block_ids[..cached_count]);
                    }
                }
                completed_ids.push(req.id);
                tracing::debug!(
                    id = %req.id,
                    total_blocks = req.block_ids.len(),
                    cached = cached_count,
                    freed = req.block_ids.len() - cached_count,
                    "freed blocks"
                );
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

            // Check prefix cache first
            let mut cached_blocks = 0usize;
            let mut tokens_already_cached = 0usize;

            if let Some(ref mut pc) = self.prefix_cache {
                if let Some((cached_tokens, cached_block_ids)) = pc.lookup(&req.prompt_tokens) {
                    cached_blocks = cached_block_ids.len();
                    tokens_already_cached = cached_tokens;
                    req.block_ids = cached_block_ids;
                    self.prefix_cache_stats.lookup_hits += 1;
                    self.prefix_cache_stats.hit_tokens += cached_tokens as u64;
                    self.prefix_cache_stats.hit_blocks += cached_blocks as u64;
                    tracing::debug!(
                        id = %req.id,
                        cached_tokens,
                        cached_blocks,
                        "prefix cache hit"
                    );
                } else {
                    self.prefix_cache_stats.lookup_misses += 1;
                }
            }

            // Allocate remaining blocks for the non-cached suffix
            let total_blocks_needed = req.blocks_needed(self.config.block_size);
            let extra_blocks_needed = total_blocks_needed.saturating_sub(cached_blocks);

            if extra_blocks_needed > 0 {
                if !self.block_manager.can_allocate(extra_blocks_needed) {
                    // Can't fit this request — release cached blocks and put back
                    if cached_blocks > 0 {
                        if let Some(ref mut pc) = self.prefix_cache {
                            pc.release_blocks(&req.block_ids[..cached_blocks]);
                        }
                        req.block_ids.clear();
                    }
                    self.waiting.push_front(req);
                    break;
                }
                let new_blocks = self.block_manager.allocate(extra_blocks_needed).unwrap();
                req.block_ids.extend(new_blocks);
            }

            // Track cached block count for this request
            if cached_blocks > 0 {
                self.cached_block_counts.insert(req.id, cached_blocks);
            }

            // Start prefill from the non-cached position
            req.state = RequestState::Prefilling {
                tokens_processed: tokens_already_cached,
            };

            let remaining_prefill = req.prompt_tokens.len() - tokens_already_cached;
            let chunk = remaining_prefill.min(budget);

            if chunk > 0 {
                scheduled.push(ScheduledRequest {
                    request_id: req.id,
                    num_tokens: chunk,
                    is_prefill: true,
                });
                budget -= chunk;
                num_prefill_tokens += chunk;
            } else {
                // Entire prompt was cached — go straight to decode
                req.state = RequestState::Decoding;
                // Schedule a decode token if budget allows
                if budget > 0 {
                    scheduled.push(ScheduledRequest {
                        request_id: req.id,
                        num_tokens: 1,
                        is_prefill: false,
                    });
                    budget -= 1;
                    num_decode_tokens += 1;
                }
            }

            tracing::debug!(
                id = %req.id,
                prompt_len = req.prompt_tokens.len(),
                cached_tokens = tokens_already_cached,
                chunk_size = if chunk > 0 { chunk } else { 1 },
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

                // Register this prefix in the cache.
                // Mark the prefix blocks as "cached" so they won't be freed
                // when this request completes — the prefix cache now owns them.
                if let Some(ref mut pc) = self.prefix_cache {
                    let num_prefix_blocks = req.prompt_tokens.len() / self.config.block_size;
                    if num_prefix_blocks > 0 && req.block_ids.len() >= num_prefix_blocks {
                        pc.register(&req.prompt_tokens, &req.block_ids);
                        // Record that these blocks are cache-owned, so free()
                        // skips them on request completion. Only update if not
                        // already tracked (from a cache hit).
                        self.cached_block_counts
                            .entry(req.id)
                            .or_insert(num_prefix_blocks);
                    }
                }
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

    pub fn prefix_cache(&self) -> Option<&PrefixCache> {
        self.prefix_cache.as_ref()
    }

    pub fn prefix_cache_stats(&self) -> PrefixCacheStats {
        let mut stats = self.prefix_cache_stats;
        if let Some(prefix_cache) = self.prefix_cache.as_ref() {
            stats.cached_blocks = prefix_cache.total_cached_blocks();
        }
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kiln_core::sampling::SamplingParams;

    fn make_request(prompt_len: usize) -> Request {
        Request::new(vec![1; prompt_len], SamplingParams::greedy(), None)
    }

    fn make_request_with_tokens(tokens: Vec<TokenId>) -> Request {
        Request::new(tokens, SamplingParams::greedy(), None)
    }

    #[test]
    fn basic_scheduling() {
        let config = SchedulerConfig {
            max_batch_tokens: 100,
            max_batch_size: 8,
            block_size: 16,
            prefix_cache_enabled: false,
            ..Default::default()
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
            prefix_cache_enabled: false,
            ..Default::default()
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
            prefix_cache_enabled: false,
            ..Default::default()
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
        let config = SchedulerConfig {
            prefix_cache_enabled: false,
            ..Default::default()
        };
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

    // --- Prefix caching tests ---

    #[test]
    fn prefix_cache_hit_skips_prefill() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: Some(100),
        };
        let mut sched = Scheduler::new(config, 200);

        // First request: system prompt (16 tokens = 4 blocks)
        let system_prompt: Vec<TokenId> = (0..16).collect();
        let mut tokens1 = system_prompt.clone();
        tokens1.extend_from_slice(&[100, 101, 102, 103]); // user message
        let req1 = make_request_with_tokens(tokens1.clone());
        let id1 = req1.id;
        sched.add_request(req1);

        // Schedule and complete prefill for req1
        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].num_tokens, 20); // full 20 tokens
        assert!(output.scheduled[0].is_prefill);

        // Complete prefill — this registers the prefix in the cache
        sched.update_request(&id1, None, false, Some(20));
        sched.update_request(&id1, Some(42), true, None);
        sched.step(); // clear completed

        // Verify prefix cache has an entry
        assert!(sched.prefix_cache().is_some());
        let pc = sched.prefix_cache().unwrap();
        assert!(pc.num_cached_entries() > 0);

        // Second request: same system prompt, different user message
        let mut tokens2 = system_prompt;
        tokens2.extend_from_slice(&[200, 201, 202, 203]); // different user message
        let req2 = make_request_with_tokens(tokens2);
        let id2 = req2.id;
        sched.add_request(req2);

        // Schedule req2 — should skip the cached prefix (16 tokens) and only prefill 4
        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].request_id, id2);
        assert!(output.scheduled[0].is_prefill);
        assert_eq!(output.scheduled[0].num_tokens, 4); // only the suffix!
        assert_eq!(output.num_prefill_tokens, 4);
    }

    #[test]
    fn prefix_cache_full_hit_goes_to_decode() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: Some(100),
        };
        let mut sched = Scheduler::new(config, 200);

        // Prompt exactly block-aligned (8 tokens = 2 blocks)
        let tokens: Vec<TokenId> = (0..8).collect();
        let req1 = make_request_with_tokens(tokens.clone());
        let id1 = req1.id;
        sched.add_request(req1);

        sched.step();
        sched.update_request(&id1, None, false, Some(8));
        sched.update_request(&id1, Some(42), true, None);
        sched.step(); // clear

        // Same exact prompt — entire prompt is cached
        let req2 = make_request_with_tokens(tokens);
        let id2 = req2.id;
        sched.add_request(req2);

        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert_eq!(output.scheduled[0].request_id, id2);
        // Should go straight to decode since all tokens are cached
        assert!(!output.scheduled[0].is_prefill);
        assert_eq!(output.scheduled[0].num_tokens, 1);
        assert_eq!(output.num_decode_tokens, 1);
        assert_eq!(output.num_prefill_tokens, 0);
    }

    #[test]
    fn prefix_cache_no_hit_different_tokens() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: Some(100),
        };
        let mut sched = Scheduler::new(config, 200);

        // First request
        let tokens1: Vec<TokenId> = (0..8).collect();
        let req1 = make_request_with_tokens(tokens1);
        let id1 = req1.id;
        sched.add_request(req1);
        sched.step();
        sched.update_request(&id1, None, false, Some(8));
        sched.update_request(&id1, Some(42), true, None);
        sched.step();

        // Second request — completely different tokens, no cache hit
        let tokens2: Vec<TokenId> = (100..108).collect();
        let req2 = make_request_with_tokens(tokens2);
        sched.add_request(req2);

        let output = sched.step();
        assert_eq!(output.scheduled.len(), 1);
        assert!(output.scheduled[0].is_prefill);
        assert_eq!(output.scheduled[0].num_tokens, 8); // full prefill

        let stats = sched.prefix_cache_stats();
        assert_eq!(stats.lookup_hits, 0);
        assert_eq!(stats.lookup_misses, 2);
        assert_eq!(stats.hit_tokens, 0);
        assert_eq!(stats.hit_blocks, 0);
        assert_eq!(stats.cached_blocks, 2);
        assert_eq!(stats.max_blocks, 100);
    }

    #[test]
    fn prefix_cache_stats_track_hits() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: Some(100),
        };
        let mut sched = Scheduler::new(config, 200);

        let tokens: Vec<TokenId> = (0..8).collect();
        let req1 = make_request_with_tokens(tokens.clone());
        let id1 = req1.id;
        sched.add_request(req1);
        sched.step();
        sched.update_request(&id1, None, false, Some(8));
        sched.update_request(&id1, Some(42), true, None);
        sched.step();

        let req2 = make_request_with_tokens(tokens);
        sched.add_request(req2);
        sched.step();

        let stats = sched.prefix_cache_stats();
        assert_eq!(stats.lookup_hits, 1);
        assert_eq!(stats.lookup_misses, 1);
        assert_eq!(stats.hit_tokens, 8);
        assert_eq!(stats.hit_blocks, 2);
        assert_eq!(stats.cached_blocks, 2);
        assert_eq!(stats.max_blocks, 100);
    }

    #[test]
    fn prefix_cache_blocks_freed_correctly() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: true,
            prefix_cache_max_blocks: Some(100),
        };
        let num_blocks = 50;
        let mut sched = Scheduler::new(config, num_blocks);

        let initial_free = sched.block_manager().num_free();

        // First request: 8 tokens = 2 blocks
        let tokens: Vec<TokenId> = (0..8).collect();
        let req1 = make_request_with_tokens(tokens.clone());
        let id1 = req1.id;
        sched.add_request(req1);
        sched.step();
        sched.update_request(&id1, None, false, Some(8));
        sched.update_request(&id1, Some(42), true, None);
        sched.step(); // completes req1, prefix cached

        // After first request completes: the 2 cached blocks are still held
        // by the prefix cache, so free blocks = initial - 2
        let free_after_cache = sched.block_manager().num_free();
        assert_eq!(free_after_cache, initial_free - 2);

        // Second request with same prefix — reuses cached blocks
        let req2 = make_request_with_tokens(tokens);
        let id2 = req2.id;
        sched.add_request(req2);
        sched.step();
        // req2 goes straight to decode, no extra blocks allocated (prompt was 8 = 2 blocks, all cached)
        sched.update_request(&id2, Some(99), true, None);
        sched.step(); // completes req2

        // After second request: cached blocks still held by cache
        let free_after_second = sched.block_manager().num_free();
        assert_eq!(free_after_second, initial_free - 2);
    }

    #[test]
    fn prefix_cache_disabled() {
        let config = SchedulerConfig {
            max_batch_tokens: 200,
            max_batch_size: 8,
            block_size: 4,
            prefix_cache_enabled: false,
            ..Default::default()
        };
        let mut sched = Scheduler::new(config, 200);
        assert!(sched.prefix_cache().is_none());

        // First request
        let tokens: Vec<TokenId> = (0..8).collect();
        let req1 = make_request_with_tokens(tokens.clone());
        let id1 = req1.id;
        sched.add_request(req1);
        sched.step();
        sched.update_request(&id1, None, false, Some(8));
        sched.update_request(&id1, Some(42), true, None);
        sched.step();

        // Second request with same tokens — no cache, full prefill
        let req2 = make_request_with_tokens(tokens);
        sched.add_request(req2);
        let output = sched.step();
        assert_eq!(output.scheduled[0].num_tokens, 8);
        assert!(output.scheduled[0].is_prefill);
    }
}
