use std::collections::HashMap;

use crate::token::TokenId;

/// A prefix cache that maps token block sequences to physical KV cache block IDs.
///
/// When multiple requests share a common prefix (e.g. the same system prompt),
/// the second request can reuse the already-computed KV cache blocks instead of
/// recomputing them. Blocks are reference-counted so they aren't freed while
/// still in use, and evicted via LRU when memory pressure requires it.
///
/// Uses a radix/trie layout over block-token hashes. Each edge is one full token
/// block, each non-root node owns the physical KV cache block for that prefix
/// position, and siblings share their common prefix nodes.
#[derive(Debug)]
pub struct PrefixCache {
    /// Maximum number of physical blocks the prefix cache may hold.
    max_blocks: usize,
    /// Total physical blocks currently held by cached trie nodes.
    total_cached_blocks: usize,
    /// Monotonically increasing counter used for LRU ordering.
    clock: u64,
    /// Block size in tokens — must match the block manager's block_size.
    block_size: usize,
    /// Radix/trie nodes. Node 0 is always the root and does not own a block.
    nodes: Vec<RadixNode>,
    /// Reference counts for individual physical block IDs.
    /// A block with refcount > 0 must not be freed by the block manager.
    refcounts: HashMap<u32, usize>,
}

#[derive(Debug, Clone)]
struct RadixNode {
    /// Physical block ID for this cached block. The root has no block.
    block_id: Option<u32>,
    /// Parent node ID. The root has no parent.
    parent: Option<usize>,
    /// Edge key from the parent to this node.
    edge_hash: u64,
    /// Child edge key → node ID.
    children: HashMap<u64, usize>,
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
            nodes: vec![RadixNode::root()],
            refcounts: HashMap::new(),
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_cached_entries(&self) -> usize {
        self.total_cached_blocks
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

        let mut node_id = 0;
        let mut matched_nodes = Vec::new();
        let mut cached_blocks = Vec::new();

        for block_idx in 0..num_full_blocks {
            let block_hash = self.block_hash_at(tokens, block_idx);
            let Some(&child_id) = self.nodes[node_id].children.get(&block_hash) else {
                break;
            };
            let Some(block_id) = self.nodes[child_id].block_id else {
                break;
            };

            matched_nodes.push(child_id);
            cached_blocks.push(block_id);
            node_id = child_id;
        }

        if cached_blocks.is_empty() {
            return None;
        }

        self.clock += 1;
        let ts = self.clock;
        for &matched_node in &matched_nodes {
            self.nodes[matched_node].last_used = ts;
        }

        for &block_id in &cached_blocks {
            *self.refcounts.entry(block_id).or_insert(0) += 1;
        }

        Some((cached_blocks.len() * self.block_size, cached_blocks))
    }

    /// Register a completed prefix so future requests can reuse it.
    ///
    /// `tokens` is the full prompt, `block_ids` are the physical blocks that
    /// hold its KV cache. Registers each block-aligned block in the radix tree.
    pub fn register(&mut self, tokens: &[TokenId], block_ids: &[u32]) {
        let num_full_blocks = tokens.len() / self.block_size;
        if self.max_blocks == 0 || num_full_blocks == 0 || block_ids.len() < num_full_blocks {
            return;
        }

        self.clock += 1;
        let ts = self.clock;
        let mut node_id = 0;

        for (block_idx, &block_id) in block_ids.iter().take(num_full_blocks).enumerate() {
            let block_hash = self.block_hash_at(tokens, block_idx);

            if let Some(&child_id) = self.nodes[node_id].children.get(&block_hash) {
                self.nodes[child_id].last_used = ts;
                node_id = child_id;
                continue;
            }

            while self.total_cached_blocks >= self.max_blocks {
                if !self.evict_one() {
                    return;
                }
            }

            let child_id = self.nodes.len();
            self.nodes.push(RadixNode {
                block_id: Some(block_id),
                parent: Some(node_id),
                edge_hash: block_hash,
                children: HashMap::new(),
                last_used: ts,
            });
            self.nodes[node_id].children.insert(block_hash, child_id);
            *self.refcounts.entry(block_id).or_insert(0) += 1;
            self.total_cached_blocks += 1;
            node_id = child_id;
        }
    }

    /// Decrement refcounts for blocks that were obtained via `lookup`.
    /// Call this when a request that used cached prefix blocks finishes.
    pub fn release_blocks(&mut self, block_ids: &[u32]) {
        for &block_id in block_ids {
            if let Some(refcount) = self.refcounts.get_mut(&block_id) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    self.refcounts.remove(&block_id);
                }
            }
        }
    }

    /// Returns the set of physical block IDs that are held by the prefix cache.
    /// The block manager must not free these blocks.
    pub fn held_block_ids(&self) -> Vec<u32> {
        let mut ids: Vec<u32> = self.nodes.iter().filter_map(|node| node.block_id).collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    /// Check if a physical block ID is currently held by the prefix cache
    /// (with refcount > 0 from active requests or from the cache entry itself).
    pub fn is_block_held(&self, block_id: u32) -> bool {
        self.refcounts.get(&block_id).copied().unwrap_or(0) > 0
    }

    /// Evict the least-recently-used leaf node whose block has refcount ≤ 1
    /// (i.e. only the cache itself holds it, no active requests).
    /// Returns true if an entry was evicted.
    fn evict_one(&mut self) -> bool {
        let mut best_node: Option<usize> = None;
        let mut best_ts = u64::MAX;

        for (node_id, node) in self.nodes.iter().enumerate().skip(1) {
            let Some(block_id) = node.block_id else {
                continue;
            };
            if !node.children.is_empty() || node.last_used >= best_ts {
                continue;
            }
            let refcount = self.refcounts.get(&block_id).copied().unwrap_or(0);
            if refcount <= 1 {
                best_node = Some(node_id);
                best_ts = node.last_used;
            }
        }

        let Some(node_id) = best_node else {
            return false;
        };
        self.remove_leaf(node_id);
        true
    }

    fn remove_leaf(&mut self, node_id: usize) {
        debug_assert!(node_id != 0);
        debug_assert!(self.nodes[node_id].children.is_empty());

        let parent_id = self.nodes[node_id]
            .parent
            .expect("non-root radix node must have a parent");
        let edge_hash = self.nodes[node_id].edge_hash;
        self.nodes[parent_id].children.remove(&edge_hash);

        if let Some(block_id) = self.nodes[node_id].block_id.take() {
            self.total_cached_blocks -= 1;
            if let Some(refcount) = self.refcounts.get_mut(&block_id) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    self.refcounts.remove(&block_id);
                }
            }
        }
    }

    fn block_hash_at(&self, tokens: &[TokenId], block_idx: usize) -> u64 {
        let start = block_idx * self.block_size;
        let end = start + self.block_size;
        Self::block_hash(&tokens[start..end])
    }

    /// Compute the edge hash for one full block of tokens.
    fn block_hash(block_tokens: &[TokenId]) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        block_tokens.hash(&mut hasher);
        hasher.finish()
    }
}

impl RadixNode {
    fn root() -> Self {
        Self {
            block_id: None,
            parent: None,
            edge_hash: 0,
            children: HashMap::new(),
            last_used: 0,
        }
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

        assert_eq!(cache.num_cached_entries(), 3);
        assert_eq!(cache.total_cached_blocks(), 3);

        // Look up the same prefix
        let result = cache.lookup(&tokens);
        assert!(result.is_some());
        let (cached_tokens, found_blocks) = result.unwrap();
        assert_eq!(cached_tokens, 12);
        assert_eq!(found_blocks, block_ids);
    }

    #[test]
    fn test_partial_prefix_hit() {
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
    fn test_sibling_prefixes_share_common_parent() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 100);

        let mut tokens_a: Vec<TokenId> = (0..8).collect();
        tokens_a.extend_from_slice(&[10, 11, 12, 13]);
        cache.register(&tokens_a, &[100, 200, 300]);

        let mut tokens_b: Vec<TokenId> = (0..8).collect();
        tokens_b.extend_from_slice(&[20, 21, 22, 23]);
        cache.register(&tokens_b, &[100, 200, 400]);

        assert_eq!(cache.num_cached_entries(), 4);
        assert_eq!(cache.lookup(&tokens_a).unwrap(), (12, vec![100, 200, 300]));
        assert_eq!(cache.lookup(&tokens_b).unwrap(), (12, vec![100, 200, 400]));
        assert_eq!(cache.held_block_ids(), vec![100, 200, 300, 400]);
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

        // Register prefix C (2 blocks) — should evict A's leaf, then A's parent
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
    fn test_lru_eviction_keeps_shared_internal_prefix_until_leaf() {
        let block_size = 4;
        let mut cache = PrefixCache::new(block_size, 4);

        let mut tokens_a: Vec<TokenId> = (0..8).collect();
        tokens_a.extend_from_slice(&[10, 11, 12, 13]);
        let mut tokens_b: Vec<TokenId> = (0..8).collect();
        tokens_b.extend_from_slice(&[20, 21, 22, 23]);

        cache.register(&tokens_a, &[1, 2, 3]);
        cache.register(&tokens_b, &[1, 2, 4]);
        assert_eq!(cache.total_cached_blocks(), 4);

        let tokens_c: Vec<TokenId> = (100..108).collect();
        cache.register(&tokens_c, &[5, 6]);

        assert_eq!(cache.total_cached_blocks(), 4);
        assert_eq!(cache.lookup(&tokens_a).unwrap(), (8, vec![1, 2]));
        assert_eq!(cache.lookup(&tokens_b).unwrap(), (8, vec![1, 2]));
        assert_eq!(cache.lookup(&tokens_c).unwrap(), (8, vec![5, 6]));
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
