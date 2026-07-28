use super::RealPrefixCache;

impl RealPrefixCache {
    pub(super) fn release_entry_blocks(&mut self, block_ids: &[u32]) -> Vec<u32> {
        let mut freed = Vec::new();
        for &block_id in block_ids {
            if let Some(refcount) = self.block_refcounts.get_mut(&block_id) {
                *refcount = refcount.saturating_sub(1);
                if *refcount == 0 {
                    self.block_refcounts.remove(&block_id);
                    freed.push(block_id);
                }
            }
        }
        freed
    }

    /// Evict unleased entries in LRU order until at least `min_blocks` physical
    /// blocks are no longer cache-owned, or no safe eviction remains.
    ///
    /// Live decode takes priority over speculative prefix reuse. Entries with
    /// active leases remain pinned, and shared blocks are returned only after
    /// their final cache refcount reaches zero.
    pub(crate) fn evict_unleased_lru_blocks(&mut self, min_blocks: usize) -> Vec<u32> {
        let mut released = Vec::new();
        while released.len() < min_blocks {
            let Some(evict_idx) = self
                .entries
                .iter()
                .enumerate()
                .filter(|(_, entry)| entry.active_uses == 0 && !entry.retired)
                .min_by_key(|(_, entry)| (entry.last_used, entry.id))
                .map(|(idx, _)| idx)
            else {
                break;
            };
            let evicted = self.entries.remove(evict_idx);
            released.extend(self.release_entry_blocks(&evicted.block_ids));
        }
        released
    }
}
