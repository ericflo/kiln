use std::collections::VecDeque;
use thiserror::Error;

#[derive(Debug, Error)]
pub enum BlockError {
    #[error("out of memory: no free blocks available (need {needed}, have {available})")]
    OutOfMemory { needed: usize, available: usize },
}

/// Manages physical KV cache blocks using a simple free list.
///
/// Each block holds `block_size` tokens worth of KV cache data.
/// Blocks are identified by integer IDs that index into the pre-allocated
/// GPU KV cache tensor: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
#[derive(Debug)]
pub struct BlockManager {
    block_size: usize,
    num_blocks: usize,
    free_blocks: VecDeque<u32>,
    #[cfg(debug_assertions)]
    allocated: Vec<bool>,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks = (0..num_blocks as u32).collect();
        Self {
            block_size,
            num_blocks,
            free_blocks,
            #[cfg(debug_assertions)]
            allocated: vec![false; num_blocks],
        }
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn num_blocks(&self) -> usize {
        self.num_blocks
    }

    pub fn num_free(&self) -> usize {
        self.free_blocks.len()
    }

    pub fn num_used(&self) -> usize {
        self.num_blocks - self.free_blocks.len()
    }

    /// Allocate a single block. Returns the physical block ID.
    pub fn allocate_one(&mut self) -> Result<u32, BlockError> {
        let block_id = self
            .free_blocks
            .pop_front()
            .ok_or(BlockError::OutOfMemory {
                needed: 1,
                available: 0,
            })?;
        self.mark_allocated(block_id);
        Ok(block_id)
    }

    /// Allocate `n` contiguous-in-ID blocks. Returns the block IDs.
    /// (They need not be contiguous in the GPU tensor — that's handled by the block table.)
    pub fn allocate(&mut self, n: usize) -> Result<Vec<u32>, BlockError> {
        if self.free_blocks.len() < n {
            return Err(BlockError::OutOfMemory {
                needed: n,
                available: self.free_blocks.len(),
            });
        }
        let blocks: Vec<u32> = (0..n)
            .map(|_| self.free_blocks.pop_front().unwrap())
            .collect();
        for &block_id in &blocks {
            self.mark_allocated(block_id);
        }
        Ok(blocks)
    }

    /// Free a single block, returning it to the free list.
    pub fn free_one(&mut self, block_id: u32) {
        self.mark_freed(block_id);
        self.free_blocks.push_back(block_id);
    }

    /// Free multiple blocks.
    pub fn free_all(&mut self, block_ids: &[u32]) {
        for &id in block_ids {
            self.free_one(id);
        }
    }

    /// Can we allocate `n` blocks right now?
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_blocks.len() >= n
    }

    fn mark_allocated(&mut self, block_id: u32) {
        debug_assert!(
            (block_id as usize) < self.num_blocks,
            "allocated block id {block_id} is outside block manager range 0..{}",
            self.num_blocks
        );
        #[cfg(debug_assertions)]
        {
            let slot = &mut self.allocated[block_id as usize];
            debug_assert!(!*slot, "block {block_id} allocated while already live");
            *slot = true;
        }
    }

    fn mark_freed(&mut self, block_id: u32) {
        debug_assert!(
            (block_id as usize) < self.num_blocks,
            "freed block id {block_id} is outside block manager range 0..{}",
            self.num_blocks
        );
        #[cfg(debug_assertions)]
        {
            let slot = &mut self.allocated[block_id as usize];
            debug_assert!(*slot, "block {block_id} freed while not allocated");
            *slot = false;
        }
    }
}

/// Per-request mapping from logical block index to physical block ID.
/// This is the "page table" for one request's KV cache.
#[derive(Debug, Clone, Default)]
pub struct BlockTable {
    /// Physical block IDs in logical order.
    pub blocks: Vec<u32>,
}

impl BlockTable {
    pub fn new() -> Self {
        Self { blocks: Vec::new() }
    }

    /// Given a token position, return (physical_block_id, offset_within_block).
    pub fn lookup(&self, token_pos: usize, block_size: usize) -> Option<(u32, usize)> {
        let logical_block = token_pos / block_size;
        let offset = token_pos % block_size;
        self.blocks.get(logical_block).map(|&phys| (phys, offset))
    }

    /// Compute the physical slot index for a token position.
    /// slot = physical_block_id * block_size + offset
    pub fn slot_for(&self, token_pos: usize, block_size: usize) -> Option<usize> {
        self.lookup(token_pos, block_size)
            .map(|(block_id, offset)| block_id as usize * block_size + offset)
    }

    /// Number of token slots currently allocated.
    pub fn capacity(&self, block_size: usize) -> usize {
        self.blocks.len() * block_size
    }

    pub fn push(&mut self, block_id: u32) {
        self.blocks.push(block_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_manager_allocate_and_free() {
        let mut bm = BlockManager::new(10, 16);
        assert_eq!(bm.num_free(), 10);

        let blocks = bm.allocate(3).unwrap();
        assert_eq!(blocks.len(), 3);
        assert_eq!(bm.num_free(), 7);

        bm.free_all(&blocks);
        assert_eq!(bm.num_free(), 10);
    }

    #[test]
    #[should_panic(expected = "freed while not allocated")]
    fn block_manager_debug_asserts_double_free() {
        let mut bm = BlockManager::new(2, 16);
        let blocks = bm.allocate(1).unwrap();
        bm.free_all(&blocks);
        bm.free_all(&blocks);
    }

    #[test]
    fn block_manager_oom() {
        let mut bm = BlockManager::new(2, 16);
        let result = bm.allocate(3);
        assert!(result.is_err());
    }

    #[test]
    fn block_table_lookup() {
        let mut bt = BlockTable::new();
        bt.push(5);
        bt.push(12);
        bt.push(3);

        let block_size = 16;
        // Token 0 → block 5, offset 0
        assert_eq!(bt.lookup(0, block_size), Some((5, 0)));
        // Token 15 → block 5, offset 15
        assert_eq!(bt.lookup(15, block_size), Some((5, 15)));
        // Token 16 → block 12, offset 0
        assert_eq!(bt.lookup(16, block_size), Some((12, 0)));
        // Token 40 → block 3, offset 8
        assert_eq!(bt.lookup(40, block_size), Some((3, 8)));
        // Token 48 → out of range
        assert_eq!(bt.lookup(48, block_size), None);
    }

    #[test]
    fn block_table_slot_mapping() {
        let mut bt = BlockTable::new();
        bt.push(5);
        bt.push(12);

        let block_size = 16;
        // Token 0 → slot 5*16 + 0 = 80
        assert_eq!(bt.slot_for(0, block_size), Some(80));
        // Token 17 → slot 12*16 + 1 = 193
        assert_eq!(bt.slot_for(17, block_size), Some(193));
    }
}
