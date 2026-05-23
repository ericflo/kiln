//! `ReplayBuffer` — off-policy rollout buffer.
//!
//! Per the Phase 2.7 issue bullet:
//!
//! > **Replay / experience buffer as a first-class abstraction.**
//! > On-policy GRPO above generates rollouts at the policy's current
//! > epoch and discards them after the step. Off-policy methods
//! > (PPO with stale rollouts, DPO with offline preference pairs,
//! > KTO with binary feedback, ORPO with reference-free preferences,
//! > IPO, SimPO) want a *buffer* of rollouts keyed by
//! > `Parameter::snapshot(epoch)` — the same primitive Phase 2.7
//! > already exposes for eval-while-training. Type:
//! > `ReplayBuffer { entries: Vec<{prompt, completion, reward?,
//! > ref_logprobs?, policy_epoch}>, max_age_epochs, sampling_strategy }`.
//!
//! # Phase 1.47 scope
//!
//! Data structures only — no actual rollout / inference path. Phase
//! 2.7's `Parameter::snapshot(epoch)` lands separately; this
//! module's `policy_epoch: u64` is the placeholder the snapshot will
//! key on.
//!
//! # Sampling strategies
//!
//! - `Uniform` — random uniform draw across entries
//! - `RecentBias { decay }` — geometric decay; recent entries
//!   weighted higher
//! - `Sequential` — FIFO order
//!
//! The sampler returns indices, not entries themselves; callers
//! borrow + read after.

use std::collections::VecDeque;

/// One rollout entry in a [`ReplayBuffer`].
#[derive(Debug, Clone)]
pub struct ReplayEntry {
    /// Prompt that produced this rollout. Stored as token ids
    /// (the loader is responsible for tokenization).
    pub prompt_ids: Vec<u32>,
    /// Completion tokens produced by the policy.
    pub completion_ids: Vec<u32>,
    /// Scalar reward for this rollout. `None` for DPO-style preference
    /// pairs where the reward is implicit in pair ordering.
    pub reward: Option<f32>,
    /// Reference-model logprobs of the completion. **Computed once at
    /// insertion** so the reference forward doesn't need to re-run
    /// during DPO / IPO / SimPO training. `None` if the algorithm
    /// doesn't need them (PPO with stale rollouts only needs the
    /// `reward`).
    pub ref_logprobs: Option<Vec<f32>>,
    /// The `Parameter::version` counter at the time this rollout was
    /// generated. Used by the `max_age_epochs` policy to evict stale
    /// rollouts.
    pub policy_epoch: u64,
}

/// Sampling strategy for [`ReplayBuffer::sample_indices`].
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SamplingStrategy {
    /// Uniformly at random across all entries.
    Uniform { seed: u64 },
    /// Recent entries weighted higher by `decay^age`. `decay = 0.99`
    /// means an entry 100 steps old has ~37% the weight of the
    /// newest entry.
    RecentBias { decay: f32, seed: u64 },
    /// FIFO — sample in insertion order. Useful for "step through
    /// every rollout exactly once" curricula.
    Sequential,
}

/// Off-policy rollout buffer.
///
/// Stored as a `VecDeque` so eviction at both ends is O(1).
#[derive(Debug)]
pub struct ReplayBuffer {
    entries: VecDeque<ReplayEntry>,
    /// Maximum age (in policy-epoch deltas) before an entry is
    /// considered stale. Entries with
    /// `current_epoch - entry.policy_epoch > max_age_epochs` are
    /// evicted on the next `insert` or `evict_stale` call.
    max_age_epochs: u64,
    /// Maximum size cap. When exceeded on `insert`, oldest entries
    /// are dropped first. `usize::MAX` disables the cap.
    capacity: usize,
    /// Sequential strategy needs a cursor.
    sequential_cursor: usize,
}

impl ReplayBuffer {
    /// New empty buffer with the given staleness policy + capacity.
    pub fn new(max_age_epochs: u64, capacity: usize) -> Self {
        ReplayBuffer {
            entries: VecDeque::new(),
            max_age_epochs,
            capacity,
            sequential_cursor: 0,
        }
    }

    /// Number of entries currently buffered.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True iff the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Borrow the entries (in insertion order, oldest first).
    pub fn entries(&self) -> impl Iterator<Item = &ReplayEntry> {
        self.entries.iter()
    }

    /// Insert a rollout. Evicts stale entries first (entries whose
    /// `policy_epoch` is more than `max_age_epochs` behind `entry.policy_epoch`),
    /// then enforces the `capacity` cap by dropping the oldest entry.
    pub fn insert(&mut self, entry: ReplayEntry) {
        let current_epoch = entry.policy_epoch;
        self.evict_stale(current_epoch);
        if self.entries.len() >= self.capacity && !self.entries.is_empty() {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    /// Evict every entry whose `policy_epoch` is more than
    /// `max_age_epochs` behind `current_epoch`. Idempotent.
    pub fn evict_stale(&mut self, current_epoch: u64) {
        while let Some(front) = self.entries.front() {
            // Use saturating_sub to avoid u64 underflow when
            // current_epoch < front.policy_epoch (shouldn't happen
            // in practice but harmless to handle).
            let age = current_epoch.saturating_sub(front.policy_epoch);
            if age > self.max_age_epochs {
                self.entries.pop_front();
            } else {
                break;
            }
        }
    }

    /// Draw `n` sample indices from the buffer per the strategy.
    /// Indices are in `0..self.len()`. Returns `None` if the buffer
    /// is empty.
    pub fn sample_indices(&mut self, n: usize, strategy: SamplingStrategy) -> Option<Vec<usize>> {
        let len = self.entries.len();
        if len == 0 {
            return None;
        }
        let mut out = Vec::with_capacity(n);
        match strategy {
            SamplingStrategy::Uniform { seed } => {
                let mut rng = SplitMix64 { state: seed };
                for _ in 0..n {
                    let idx = (rng.next_u64() as usize) % len;
                    out.push(idx);
                }
            }
            SamplingStrategy::RecentBias { decay, seed } => {
                // Weights: decay^age_from_newest. Cumulative-sum +
                // uniform-sample binary search.
                let mut weights = Vec::with_capacity(len);
                let mut cum = 0.0_f32;
                for i in 0..len {
                    let age = (len - 1 - i) as f32;
                    cum += decay.powf(age);
                    weights.push(cum);
                }
                let total = *weights.last().unwrap_or(&0.0);
                let mut rng = SplitMix64 { state: seed };
                for _ in 0..n {
                    let r = (rng.next_u64() as f32 / u64::MAX as f32) * total;
                    let idx = weights.partition_point(|&w| w < r);
                    out.push(idx.min(len - 1));
                }
            }
            SamplingStrategy::Sequential => {
                for _ in 0..n {
                    let idx = self.sequential_cursor % len;
                    self.sequential_cursor = self.sequential_cursor.wrapping_add(1);
                    out.push(idx);
                }
            }
        }
        Some(out)
    }
}

/// Tiny stdlib-only seeded PRNG (splitmix64). Used by the sampler;
/// avoids pulling in `rand` as a dep. Deterministic per seed.
#[derive(Debug)]
struct SplitMix64 {
    state: u64,
}
impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(prompt: u32, completion: u32, reward: Option<f32>, epoch: u64) -> ReplayEntry {
        ReplayEntry {
            prompt_ids: vec![prompt],
            completion_ids: vec![completion],
            reward,
            ref_logprobs: None,
            policy_epoch: epoch,
        }
    }

    #[test]
    fn new_buffer_is_empty() {
        let b = ReplayBuffer::new(100, 1000);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
    }

    #[test]
    fn insert_grows_buffer() {
        let mut b = ReplayBuffer::new(100, 1000);
        b.insert(entry(1, 10, Some(0.5), 1));
        b.insert(entry(2, 20, Some(0.6), 2));
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn capacity_evicts_oldest_first() {
        let mut b = ReplayBuffer::new(100, 2);
        b.insert(entry(1, 10, None, 1));
        b.insert(entry(2, 20, None, 2));
        b.insert(entry(3, 30, None, 3));
        // capacity=2 → oldest entry (epoch=1) evicted
        assert_eq!(b.len(), 2);
        let ids: Vec<u32> = b.entries().map(|e| e.prompt_ids[0]).collect();
        assert_eq!(ids, vec![2, 3]);
    }

    #[test]
    fn stale_entries_evicted_on_insert() {
        // max_age_epochs=5; insert at epochs 1, 2, 3, then at epoch 10.
        // Entries 1, 2, 3 should be evicted (10 - epoch > 5).
        let mut b = ReplayBuffer::new(5, 1000);
        b.insert(entry(1, 10, None, 1));
        b.insert(entry(2, 20, None, 2));
        b.insert(entry(3, 30, None, 3));
        b.insert(entry(99, 990, None, 10));
        assert_eq!(b.len(), 1);
        assert_eq!(b.entries().next().unwrap().prompt_ids[0], 99);
    }

    #[test]
    fn evict_stale_idempotent() {
        let mut b = ReplayBuffer::new(5, 1000);
        b.insert(entry(1, 10, None, 1));
        b.insert(entry(2, 20, None, 2));
        b.evict_stale(10);
        let len_after_first = b.len();
        b.evict_stale(10);
        assert_eq!(b.len(), len_after_first);
        assert_eq!(b.len(), 0); // Both 1 and 2 are stale (10 - 1 = 9 > 5)
    }

    #[test]
    fn sample_uniform_returns_indices_in_range() {
        let mut b = ReplayBuffer::new(100, 1000);
        for i in 1..=5 {
            b.insert(entry(i as u32, (i * 10) as u32, None, i));
        }
        let idxs = b
            .sample_indices(20, SamplingStrategy::Uniform { seed: 42 })
            .unwrap();
        assert_eq!(idxs.len(), 20);
        for i in idxs {
            assert!(i < 5);
        }
    }

    #[test]
    fn sample_sequential_advances_cursor() {
        let mut b = ReplayBuffer::new(100, 1000);
        for i in 1..=3 {
            b.insert(entry(i as u32, (i * 10) as u32, None, i));
        }
        let idxs = b.sample_indices(6, SamplingStrategy::Sequential).unwrap();
        // Sequential cycles through indices.
        assert_eq!(idxs, vec![0, 1, 2, 0, 1, 2]);
    }

    #[test]
    fn sample_recent_bias_favors_recent_entries() {
        let mut b = ReplayBuffer::new(100, 1000);
        for i in 1..=10 {
            b.insert(entry(i as u32, (i * 10) as u32, None, i));
        }
        // With decay=0.5, the last index has ~512x the weight of the first.
        // Sample 200; expect index 9 to dominate.
        let idxs = b
            .sample_indices(
                200,
                SamplingStrategy::RecentBias {
                    decay: 0.5,
                    seed: 7,
                },
            )
            .unwrap();
        let mut counts = vec![0_usize; 10];
        for i in idxs {
            counts[i] += 1;
        }
        // The newest entry (index 9) should appear more than the
        // oldest (index 0). Loose bound to avoid PRNG flake.
        assert!(
            counts[9] > counts[0],
            "expected recent-bias to favor index 9; counts = {counts:?}"
        );
    }

    #[test]
    fn sample_on_empty_returns_none() {
        let mut b = ReplayBuffer::new(100, 1000);
        assert!(b
            .sample_indices(1, SamplingStrategy::Uniform { seed: 0 })
            .is_none());
    }

    #[test]
    fn entry_with_ref_logprobs() {
        // Verify the ref_logprobs slot is usable (DPO / IPO path).
        let e = ReplayEntry {
            prompt_ids: vec![1, 2],
            completion_ids: vec![3, 4],
            reward: None,
            ref_logprobs: Some(vec![-1.5, -2.3]),
            policy_epoch: 7,
        };
        let mut b = ReplayBuffer::new(100, 1000);
        b.insert(e);
        let stored = b.entries().next().unwrap();
        assert_eq!(stored.ref_logprobs.as_ref().unwrap().len(), 2);
        assert!(stored.reward.is_none());
    }
}
