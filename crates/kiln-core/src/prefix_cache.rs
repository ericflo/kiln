use std::collections::HashMap;

use crate::token::TokenId;

/// A prefix cache that maps token block sequences to physical KV cache block IDs.
///
/// When multiple requests share a common prefix (e.g. the same system prompt),
/// the second request can reuse the already-computed KV cache blocks instead of
/// recomputing them. Blocks are reference-counted so they aren't freed while
/// still in use, and evicted via LRU when memory pressure requires it.
///
/// Uses hash chaining: each block's key is derived from the tokens in that block
/// AND the previous block's hash. This means any prefix that matches block-by-block
/// from the start will find the longest cached run, even if different suffixes
/// were registered from different prompts.
#[derive(Debug)]
pub struct PrefixCache {
    /// Maximum number of physical blocks the prefix cache may hold.
    max_blocks: usize,
    /// Total physical blocks currently held by cached entries.
    total_cached_blocks: usize,
    /// Monotonically increasing counter used for LRU ordering.
    clock: u64,
    /// Block size in tokens — must match the block manager's block_size.
    block_size: usize,
    /// Hash chain key → cached block entry.
    /// Each entry represents one block at a specific position in a specific prefix.
    entries: HashMap<u64, BlockCacheEntry>,
    /// Reference counts for individual physical block IDs.
    /// A block with refcount > 0 must not be evicted.
    refcounts: HashMap<u32, usize>,
}

#[derive(Debug, Clone)]
struct BlockCacheEntry {
    /// Physical block ID for this cached block.
    block_id: u32,
    /// The hash chain value at this position (used as the key).
    chain_hash: u64,
    /// Last-access timestamp for LRU eviction.
    last_used: u64,
}

impl PrefixCache {
    pub fn new(block_size: usize, max_blocks: usize) -> Self {
        Self {
            max_blocks,
            total_cached_blocks: 0,
            clock: 0,
            block_size,
            entries: HashMap::new(),
            refcounts: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_cached_entries(&self) -> usize {
        self.entries.len()
    }

    pub fn total_cached_blocks(&self) -> usize {
        self.total_cached_blocks
    }

    /// Look up how many leading blocks of `tokens` are already cached.
    ///
    /// Returns `(num_cached_tokens, Vec<block_id>)` for the longest cached prefix.
    /// The returned block IDs have their refcount incremented — the caller must
    /// call `release_blocks` when the request finishes.
    pub fn lookup(&mut self, tokens: &[TokenId]) -> Option<(usize, Vec<u32>)> {
        if tokens.is_empty() {
            return None;
        }

        let num_full_blocks = tokens.len() / self.block_size;
        if num_full_blocks == 0 {
            return None;
        }

        let mut cached_blocks = Vec::new();
        let mut prev_hash: u64 = 0;

        for block_idx in 0..num_full_blocks {
            let start = block_idx * self.block_size;
            let end = start + self.block_size;
            let block_tokens = &tokens[start..end];
            let chain_hash = Self::chain_hash(prev_hash, block_tokens);

            if let Some(entry) = self.entries.get(&chain_hash) {
                cached_blocks.push(entry.block_id);
                prev_hash = chain_hash;
            } else {
                break; // Chain broken — no more cached blocks
            }
        }

        if cached_blocks.is_empty() {
            return None;
        }

        let cached_tokens = cached_blocks.len() * self.block_size;

        // Bump LRU timestamps for all matched entries
        self.clock += 1;
        let ts = self.clock;
        let mut ph: u64 = 0;
        for block_idx in 0..cached_blocks.len() {
            let start = block_idx * self.block_size;
            let end = start + self.block_size;
            let block_tokens = &tokens[start..end];
            let ch = Self::chain_hash(ph, block_tokens);
            if let Some(entry) = self.entries.get_mut(&ch) {
                entry.last_used = ts;
            }
            ph = ch;
        }

        // Increment refcounts
        for &bid in &cached_blocks {
            *self.refcounts.entry(bid).or_insert(0) += 1;
        }

        Some((cached_tokens, cached_blocks))
    }

    /// Register a completed prefix so future requests can reuse it.
    ///
    /// `tokens` is the full prompt, `block_ids` are the physical blocks that
    /// hold its KV cache. Registers each block-aligned block in the hash chain.
    pub fn register(&mut self, tokens: &[TokenId], block_ids: &[u32]) {
        let num_full_blocks = tokens.len() / self.block_size;
        if num_full_blocks == 0 || block_ids.len() < num_full_blocks {
            return;
        }

        self.clock += 1;
        let ts = self.clock;
        let mut prev_hash: u64 = 0;

        for block_idx in 0..num_full_blocks {
            let start = block_idx * self.block_size;
            let end = start + self.block_size;
            let block_tokens = &tokens[start..end];
            let chain_hash = Self::chain_hash(prev_hash, block_tokens);

            if self.entries.contains_key(&chain_hash) {
                // Already cached — just bump LRU
                if let Some(entry) = self.entries.get_mut(&chain_hash) {
                    entry.last_used = ts;
                }
            } else {
                // Evict if needed to make room for one block
                while self.total_cached_blocks >= self.max_blocks {
                    if !self.evict_one() {
                        // Can't evict anything — stop caching
                        prev_hash = chain_hash;
                        continue;
                    }
                }

                // Increment refcount for this block
                *self.refcounts.entry(block_ids[block_idx]).or_insert(0) += 1;
                self.total_cached_blocks += 1;

                self.entries.insert(
                    chain_hash,
                    BlockCacheEntry {
                        block_id: block_ids[block_idx],
                        chain_hash,
                        last_used: ts,
                    },
                );
            }

            prev_hash = chain_hash;
        }
    }

    /// Decrement refcounts for blocks that were obtained via `lookup`.
    /// Call this when a request that used cached prefix blocks finishes.
    pub fn release_blocks(&mut self, block_ids: &[u32]) {
        for &bid in block_ids {
            if let Some(rc) = self.refcounts.get_mut(&bid) {
                *rc = rc.saturating_sub(1);
            }
        }
    }

    /// Returns the set of physical block IDs that are held by the prefix cache.
    /// The block manager must not free these blocks.
    pub fn held_block_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.entries.values().map(|e| e.block_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Check if a physical block ID is currently held by the prefix cache
    /// (with refcount > 0 from active requests or from the cache entry itself).
    pub fn is_block_held(&self, block_id: u32) -> bool {
        self.refcounts.get(&block_id).copied().unwrap_or(0) > 0
    }

    /// Evict the least-recently-used block entry whose block has refcount ≤ 1
    /// (i.e. only the cache itself holds it, no active requests).
    /// Returns true if an entry was evicted.
    fn evict_one(&mut self) -> bool {
        let mut best_hash: Option<u64> = None;
        let mut best_ts = u64::MAX;

        for (&hash, entry) in &self.entries {
            if entry.last_used < best_ts {
                let rc = self.refcounts.get(&entry.block_id).copied().unwrap_or(0);
                if rc <= 1 {
                    best_hash = Some(hash);
                    best_ts = entry.last_used;
                }
            }
        }

        if let Some(hash) = best_hash {
            if let Some(entry) = self.entries.remove(&hash) {
                self.total_cached_blocks -= 1;
                if let Some(rc) = self.refcounts.get_mut(&entry.block_id) {
                    *rc = rc.saturating_sub(1);
                    if *rc == 0 {
                        self.refcounts.remove(&entry.block_id);
                    }
                }
                return true;
            }
        }
        false
    }

    /// Compute a chained hash for a block of tokens, incorporating the previous block's hash.
    /// This ensures that the same token block at different positions (after different prefixes)
    /// gets different hashes.
    fn chain_hash(prev_hash: u64, block_tokens: &[TokenId]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        prev_hash.hash(&mut hasher);
        block_tokens.hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_lookup() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        // Register a 12-token prefix (3 full blocks)
        let tokens: Vec<TokenId> = (0..12).collect();
        let block_ids = vec![10, 20, 30];
        cache.register(&tokens, &block_ids);

        assert_eq!(cache.num_cached_entries(), 3); // one per block
        assert_eq!(cache.total_cached_blocks(), 3);

        // Look up the same tokens — should find all 3 blocks
        let result = cache.lookup(&tokens);
        assert!(result.is_some());
        let (cached_tokens, found_blocks) = result.unwrap();
        assert_eq!(cached_tokens, 12);
        assert_eq!(found_blocks, vec![10, 20, 30]);
    }

    #[test]
    fn test_partial_prefix_match() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        // Register a 12-token prefix
        let prefix: Vec<TokenId> = (0..12).collect();
        cache.register(&prefix, &[10, 20, 30]);

        // Look up a longer sequence that shares the same 12-token prefix
        // but has different suffix tokens
        let mut longer: Vec<TokenId> = (0..12).collect();
        longer.extend_from_slice(&[99, 98, 97, 96, 95, 94, 93, 92]);
        let result = cache.lookup(&longer);
        assert!(result.is_some());
        let (cached_tokens, found_blocks) = result.unwrap();
        assert_eq!(cached_tokens, 12);
        assert_eq!(found_blocks, vec![10, 20, 30]);
    }

    #[test]
    fn test_shared_prefix_different_suffix() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        // Register a 20-token prompt (5 blocks)
        let tokens1: Vec<TokenId> = (0..20).collect();
        cache.register(&tokens1, &[10, 20, 30, 40, 50]);

        // Look up a prompt that shares the first 16 tokens but differs in the last block
        let mut tokens2: Vec<TokenId> = (0..16).collect();
        tokens2.extend_from_slice(&[200, 201, 202, 203]);

        let result = cache.lookup(&tokens2);
        assert!(result.is_some());
        let (cached_tokens, found_blocks) = result.unwrap();
        // Should match first 4 blocks (16 tokens), not 5
        assert_eq!(cached_tokens, 16);
        assert_eq!(found_blocks, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_no_match_different_tokens() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        let tokens_a: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens_a, &[10, 20]);

        // Different tokens — no match
        let tokens_b: Vec<TokenId> = (100..108).collect();
        let result = cache.lookup(&tokens_b);
        assert!(result.is_none());
    }

    #[test]
    fn test_too_short_for_cache() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        // Only 3 tokens — less than 1 block, should not cache
        let tokens: Vec<TokenId> = vec![1, 2, 3];
        cache.register(&tokens, &[10]);
        assert_eq!(cache.num_cached_entries(), 0);

        let result = cache.lookup(&tokens);
        assert!(result.is_none());
    }

    #[test]
    fn test_lru_eviction() {
        let block_size = 4;
        // max_blocks = 4, so can hold 4 block entries total
        let mut cache = PrefixCache::new(block_size, 4);

        // Register prefix A (2 blocks)
        let tokens_a: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens_a, &[10, 20]);
        assert_eq!(cache.total_cached_blocks(), 2);

        // Register prefix B (2 blocks)
        let tokens_b: Vec<TokenId> = (100..108).collect();
        cache.register(&tokens_b, &[30, 40]);
        assert_eq!(cache.total_cached_blocks(), 4);

        // Register prefix C (2 blocks) — should evict A's blocks (oldest)
        let tokens_c: Vec<TokenId> = (200..208).collect();
        cache.register(&tokens_c, &[50, 60]);
        assert_eq!(cache.total_cached_blocks(), 4);

        // A should be gone (its blocks were evicted)
        assert!(cache.lookup(&tokens_a).is_none());
        // B and C should still be there
        assert!(cache.lookup(&tokens_b).is_some());
        assert!(cache.lookup(&tokens_c).is_some());
    }

    #[test]
    fn test_refcount_prevents_eviction() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 4);

        // Register prefix A and keep a reference via lookup
        let tokens_a: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens_a, &[10, 20]);
        let _lookup_a = cache.lookup(&tokens_a); // refcount now 2 (cache + lookup)

        // Register prefix B
        let tokens_b: Vec<TokenId> = (100..108).collect();
        cache.register(&tokens_b, &[30, 40]);

        // Try to register prefix C — A can't be evicted (refcount > 1), B can
        let tokens_c: Vec<TokenId> = (200..208).collect();
        cache.register(&tokens_c, &[50, 60]);

        // A should still be there (protected by refcount)
        assert!(cache.lookup(&tokens_a).is_some());
        // B should be evicted
        assert!(cache.lookup(&tokens_b).is_none());
        // C should be there
        assert!(cache.lookup(&tokens_c).is_some());
    }

    #[test]
    fn test_release_blocks() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        let tokens: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens, &[10, 20]);

        // Lookup increases refcount
        let (_, blocks) = cache.lookup(&tokens).unwrap();
        assert!(cache.is_block_held(10));
        assert!(cache.is_block_held(20));

        // Release the lookup reference
        cache.release_blocks(&blocks);

        // Cache still holds them (refcount == 1 from cache entry)
        assert!(cache.is_block_held(10));
        assert!(cache.is_block_held(20));
    }

    #[test]
    fn test_duplicate_register_is_noop() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        let tokens: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens, &[10, 20]);
        assert_eq!(cache.num_cached_entries(), 2); // 2 block entries

        // Register again — should not create duplicates
        cache.register(&tokens, &[10, 20]);
        assert_eq!(cache.num_cached_entries(), 2);
        assert_eq!(cache.total_cached_blocks(), 2);
    }

    #[test]
    fn test_held_block_ids() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        let tokens_a: Vec<TokenId> = (0..8).collect();
        cache.register(&tokens_a, &[10, 20]);

        let tokens_b: Vec<TokenId> = (100..108).collect();
        cache.register(&tokens_b, &[30, 40]);

        let held = cache.held_block_ids();
        assert_eq!(held, vec![10, 20, 30, 40]);
    }
}
