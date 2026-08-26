use std::collections::VecDeque;
use thiserror::Error;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockState {
    Free,
    InUse,
    Retired,
}

#[derive(Debug, Error)]
pub enum BlockError {
    #[error("out of memory: no free blocks available (need {needed}, have {available})")]
    OutOfMemory { needed: usize, available: usize },
}

/// Manages physical KV cache blocks using a locality-preserving free list.
///
/// Each block holds `block_size` tokens worth of KV cache data.
/// Blocks are identified by integer IDs that index into the pre-allocated
/// GPU KV cache tensor: `[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]`
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BlockManager {
    block_size: usize,
    num_blocks: usize,
    free_blocks: VecDeque<u32>,
    /// O(1) ownership checks for allocation/free and allocation-free lookup of
    /// the live high-water mark. Keeping this alongside the free/retired queues
    /// avoids rebuilding pool-sized hash sets while decode is paused for cache
    /// cleanup or a physical resize.
    block_states: Vec<BlockState>,
    /// Blocks taken OUT of circulation by a dynamic shrink (the memory governor
    /// lowering the inference KV footprint so a coexisting job / training run can
    /// use that VRAM). They are neither free nor in-use — `set_target_usable`
    /// restores them when memory pressure eases. Empty in steady state.
    retired_blocks: VecDeque<u32>,
    /// The dynamic logical capacity: at most this many blocks may be in
    /// circulation (free + in-use). `<= num_blocks`; defaults to `num_blocks`.
    target_usable: usize,
}

impl BlockManager {
    pub fn new(num_blocks: usize, block_size: usize) -> Self {
        let free_blocks = (0..num_blocks as u32).collect();
        Self {
            block_size,
            num_blocks,
            free_blocks,
            block_states: vec![BlockState::Free; num_blocks],
            retired_blocks: VecDeque::new(),
            target_usable: num_blocks,
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
        self.num_blocks - self.free_blocks.len() - self.retired_blocks.len()
    }

    /// Blocks currently retired (out of circulation) by a dynamic shrink.
    pub fn num_retired(&self) -> usize {
        self.retired_blocks.len()
    }

    /// Current dynamic capacity (free + in-use ceiling). `<= num_blocks`.
    pub fn target_usable(&self) -> usize {
        self.target_usable
    }

    /// Dynamically resize the inference KV footprint to `target` usable blocks
    /// (clamped to `[blocks_in_use, num_blocks]` — we never retire a block a live
    /// request still holds). SHRINK retires free blocks immediately; any excess
    /// in-use blocks retire automatically as their request frees them. GROW
    /// returns retired blocks to the free list. Returns the achieved
    /// `target_usable` (may exceed `target` if in-use blocks block a full shrink).
    ///
    /// This is the actuator the memory governor drives: under pressure it shrinks
    /// so kiln hands VRAM back; when pressure eases it grows again. Safe to call
    /// concurrently with serving — in-flight requests are never disrupted.
    pub fn set_target_usable(&mut self, target: usize) -> usize {
        let in_use = self.num_used();
        // Store the REQUESTED target (only bounded by num_blocks) so a shrink
        // below the current in-use count keeps retiring blocks as requests drain,
        // via free_one/free_all — rather than being lost to an up-front clamp.
        let target = target.min(self.num_blocks);
        self.target_usable = target;
        let in_circulation = self.num_blocks - self.retired_blocks.len();
        if in_circulation > target {
            // Shrink: retire free blocks (in-use ones retire on free in free_one).
            // Retire the HIGHEST free block IDs first so the pool's tail clears,
            // making a subsequent physical truncate (#26 `physical_truncate`)
            // possible — the dead region we hand back to the OS must be the tail
            // `[new_num_blocks, num_blocks)`, never a hole in the middle.
            let to_retire = (in_circulation - target).min(self.free_blocks.len());
            if to_retire > 0 {
                let mut ids: Vec<u32> = self.free_blocks.iter().copied().collect();
                ids.sort_unstable();
                let keep = ids.len() - to_retire;
                for &id in &ids[keep..] {
                    self.transition(id, BlockState::Free, BlockState::Retired);
                    self.retired_blocks.push_back(id);
                }
                // Keep the free list low-ID-biased so allocation reuses low
                // blocks first, which also helps keep the tail clear.
                self.free_blocks = ids[..keep].iter().copied().collect();
            }
        } else if in_circulation < target {
            // Grow: bring retired blocks back into circulation, lowest IDs first
            // so the usable range extends contiguously up from the live region.
            let to_restore = target - in_circulation;
            if to_restore > 0 && !self.retired_blocks.is_empty() {
                let mut ids: Vec<u32> = self.retired_blocks.iter().copied().collect();
                ids.sort_unstable();
                let restore = to_restore.min(ids.len());
                for &id in &ids[..restore] {
                    self.transition(id, BlockState::Retired, BlockState::Free);
                    self.free_blocks.push_back(id);
                }
                self.retired_blocks = ids[restore..].iter().copied().collect();
            }
        }
        // After a shrink that couldn't fully complete (in-use > target), the
        // effective ceiling is what's actually in circulation.
        (self.num_blocks - self.retired_blocks.len()).max(in_use)
    }

    /// Highest physical block ID currently held by a live request (in-use),
    /// or `None` if nothing is in use. In-use blocks are those neither free nor
    /// retired. This is the high-water mark that bounds how far the physical
    /// pool may be truncated — the tensor tail above this ID holds no live KV.
    pub fn highest_in_use_block(&self) -> Option<u32> {
        self.block_states
            .iter()
            .rposition(|state| *state == BlockState::InUse)
            .map(|id| id as u32)
    }

    /// Smallest `num_blocks` the physical pool could shrink to RIGHT NOW without
    /// dropping any live KV: `highest_in_use_block + 1` (0 if nothing is in use).
    /// A physical truncate to any value `>=` this floor is sound; below it would
    /// orphan a live request's blocks. Distinct from `target_usable` (logical).
    pub fn physical_floor(&self) -> usize {
        self.highest_in_use_block().map_or(0, |id| id as usize + 1)
    }

    /// Physically shrink the manager's block space to `new_num_blocks`, dropping
    /// the tail `[new_num_blocks, num_blocks)` from the free/retired lists. The
    /// CALLER must realloc the backing KV pool tensor to match in lockstep
    /// (`PagedKvCacheKt::physical_resize_to`, the kt-side twin in kiln-model).
    ///
    /// Sound only when the tail holds no live request — fails with `OutOfMemory`
    /// (repurposed: "tail still in use") if `physical_floor() > new_num_blocks`.
    /// Drive a logical [`Self::set_target_usable`] shrink first and let it drain, then
    /// truncate once `physical_floor()` has dropped to the target.
    pub fn physical_truncate(&mut self, new_num_blocks: usize) -> Result<(), BlockError> {
        if new_num_blocks >= self.num_blocks {
            return Ok(());
        }
        let floor = self.physical_floor();
        if floor > new_num_blocks {
            return Err(BlockError::OutOfMemory {
                needed: floor,
                available: new_num_blocks,
            });
        }
        let cap = new_num_blocks as u32;
        self.free_blocks.retain(|&id| id < cap);
        self.retired_blocks.retain(|&id| id < cap);
        self.block_states.truncate(new_num_blocks);
        self.num_blocks = new_num_blocks;
        self.target_usable = self.target_usable.min(new_num_blocks);
        Ok(())
    }

    /// Physically grow the block space to `new_num_blocks`, adding the new tail
    /// `[num_blocks, new_num_blocks)` as free (or retired, if above the current
    /// `target_usable` ceiling). The CALLER must realloc the KV pool tensor to
    /// match in lockstep. No-op if `new_num_blocks <= num_blocks`.
    pub fn physical_grow(&mut self, new_num_blocks: usize) {
        if new_num_blocks <= self.num_blocks {
            return;
        }
        for id in self.num_blocks as u32..new_num_blocks as u32 {
            if (id as usize) < self.target_usable {
                self.free_blocks.push_back(id);
                self.block_states.push(BlockState::Free);
            } else {
                self.retired_blocks.push_back(id);
                self.block_states.push(BlockState::Retired);
            }
        }
        self.num_blocks = new_num_blocks;
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
        self.transition(block_id, BlockState::Free, BlockState::InUse);
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
            self.transition(block_id, BlockState::Free, BlockState::InUse);
        }
        Ok(blocks)
    }

    /// Free a single block. Normally returns it to the free list, but if a
    /// pending dynamic shrink left circulation above `target_usable`, the block
    /// is RETIRED instead (this is how an in-use-blocked shrink completes as
    /// requests drain).
    pub fn free_one(&mut self, block_id: u32) {
        if self.num_blocks - self.retired_blocks.len() > self.target_usable {
            self.transition(block_id, BlockState::InUse, BlockState::Retired);
            self.retired_blocks.push_back(block_id);
        } else {
            self.transition(block_id, BlockState::InUse, BlockState::Free);
            // Prefer an already-touched block over walking untouched capacity.
            // This is especially important for host-resident Vulkan KV pools:
            // first-touching a FIFO of sparse zero pages grows RSS and can incur
            // transparent-huge-page faults long after model startup.
            self.free_blocks.push_front(block_id);
        }
    }

    /// Free multiple blocks.
    pub fn free_all(&mut self, block_ids: &[u32]) {
        // #673: A double-free corrupts the free list and lets concurrent
        // requests get the same physical block twice. The prefix-cache layer
        // is the canonical source of these IDs and should never hand us a
        // duplicate or a block that is still on the free list.
        debug_assert!(
            {
                let mut seen = std::collections::HashSet::with_capacity(block_ids.len());
                block_ids.iter().all(|id| seen.insert(*id))
            },
            "BlockManager::free_all called with duplicate block IDs: {block_ids:?}",
        );
        // Reverse + push_front preserves the request's logical block order while
        // placing the whole just-freed run ahead of untouched capacity. A retry
        // of the same shape therefore gets the same physical run immediately.
        for &id in block_ids.iter().rev() {
            // Retire instead of free while a pending shrink keeps circulation
            // above the target (see free_one).
            if self.num_blocks - self.retired_blocks.len() > self.target_usable {
                self.transition(id, BlockState::InUse, BlockState::Retired);
                self.retired_blocks.push_back(id);
            } else {
                self.transition(id, BlockState::InUse, BlockState::Free);
                self.free_blocks.push_front(id);
            }
        }
    }

    /// Can we allocate `n` blocks right now?
    pub fn can_allocate(&self, n: usize) -> bool {
        self.free_blocks.len() >= n
    }

    fn transition(&mut self, block_id: u32, expected: BlockState, next: BlockState) {
        let state = self
            .block_states
            .get_mut(block_id as usize)
            .unwrap_or_else(|| panic!("BlockManager block ID {block_id} is out of range"));
        assert_eq!(
            *state, expected,
            "BlockManager invalid block transition for ID {block_id}: expected {expected:?}, found {state:?}, next {next:?}"
        );
        *state = next;
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

/// Return physical block IDs referenced by the supplied tables, preserving
/// first-seen order and dropping duplicates.
pub fn unique_physical_blocks(block_tables: &[&BlockTable]) -> Vec<u32> {
    let total_blocks = block_tables.iter().map(|table| table.blocks.len()).sum();
    let mut out = Vec::with_capacity(total_blocks);
    let mut seen = std::collections::HashSet::with_capacity(total_blocks);
    for table in block_tables {
        for &block_id in &table.blocks {
            if seen.insert(block_id) {
                out.push(block_id);
            }
        }
    }
    out
}

/// Return the physical start slot when `[start_pos .. start_pos+len]`
/// resolves to one contiguous slot run in the shared paged KV pool, else
/// `None`.
///
/// Relocated here from `kiln_model::paged_kv_cache` during the #1082
/// candle-drop: it is pure `BlockTable` bookkeeping with no tensor/device
/// dependency, so it belongs next to `BlockTable` and stays available on
/// non-CUDA builds (the candle `paged_kv_cache.rs` that previously hosted
/// it was deleted, and its kt replacement is CUDA-only).
pub fn contiguous_slot_run_start(
    block_table: &BlockTable,
    block_size: usize,
    start_pos: usize,
    len: usize,
) -> Option<usize> {
    if len == 0 {
        return None;
    }

    let start_slot = block_table.slot_for(start_pos, block_size)?;
    if len == 1 {
        return Some(start_slot);
    }

    let start_block = start_pos / block_size;
    let end_pos = start_pos + len - 1;
    let end_block = end_pos / block_size;

    if start_block == end_block {
        return Some(start_slot);
    }

    let first_phys_block = *block_table.blocks.get(start_block)? as usize;
    for logical_block in (start_block + 1)..=end_block {
        let expected_phys_block = first_phys_block + (logical_block - start_block);
        let phys_block = *block_table.blocks.get(logical_block)? as usize;
        if phys_block != expected_phys_block {
            return None;
        }
    }

    Some(start_slot)
}

/// Return one physical start slot per batch row when every logical window
/// is a contiguous run in the shared paged KV pool, else `None`.
///
/// Relocated here from `kiln_model::paged_kv_cache` during the #1082
/// candle-drop (see [`contiguous_slot_run_start`]).
pub fn contiguous_slot_run_starts(
    block_tables: &[&BlockTable],
    block_size: usize,
    start_positions: &[usize],
    len: usize,
) -> Option<Vec<usize>> {
    if len == 0 || block_tables.len() != start_positions.len() {
        return None;
    }

    let mut starts = Vec::with_capacity(block_tables.len());
    for (block_table, &start_pos) in block_tables.iter().zip(start_positions) {
        starts.push(contiguous_slot_run_start(
            block_table,
            block_size,
            start_pos,
            len,
        )?);
    }
    Some(starts)
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
    fn block_manager_reuses_freed_run_before_untouched_capacity() {
        let mut bm = BlockManager::new(10, 16);
        let first = bm.allocate(3).unwrap();
        assert_eq!(first, vec![0, 1, 2]);

        bm.free_all(&first);

        assert_eq!(bm.allocate(3).unwrap(), first);
    }

    #[test]
    fn block_manager_reuses_one_freed_block_immediately() {
        let mut bm = BlockManager::new(10, 16);
        let first = bm.allocate_one().unwrap();
        bm.free_one(first);

        assert_eq!(bm.allocate_one().unwrap(), first);
    }

    #[test]
    #[should_panic(expected = "invalid block transition")]
    fn block_manager_rejects_double_free_without_scanning_the_free_pool() {
        let mut bm = BlockManager::new(1_000_000, 16);
        let block = bm.allocate_one().unwrap();
        bm.free_one(block);
        bm.free_one(block);
    }

    #[test]
    fn dynamic_shrink_retires_free_blocks_and_grow_restores() {
        let mut bm = BlockManager::new(10, 16);
        // Shrink to 6 usable: 4 free blocks retired, only 6 allocatable.
        let achieved = bm.set_target_usable(6);
        assert_eq!(achieved, 6);
        assert_eq!(bm.num_retired(), 4);
        assert_eq!(bm.num_free(), 6);
        assert!(!bm.can_allocate(7));
        assert!(bm.can_allocate(6));
        // Grow back to 10: retired blocks return to free.
        let achieved = bm.set_target_usable(10);
        assert_eq!(achieved, 10);
        assert_eq!(bm.num_retired(), 0);
        assert_eq!(bm.num_free(), 10);
    }

    #[test]
    fn dynamic_shrink_blocked_by_in_use_completes_as_requests_drain() {
        let mut bm = BlockManager::new(10, 16);
        // 8 blocks in use, 2 free. Ask to shrink to 4 — we can only retire the
        // 2 free now; the in-use blocks retire as they free (never yanked).
        let held = bm.allocate(8).unwrap();
        let achieved = bm.set_target_usable(4);
        assert_eq!(achieved, 8, "can't go below in-use count yet");
        assert_eq!(bm.num_retired(), 2); // the 2 free blocks retired immediately
        assert_eq!(bm.num_free(), 0);
        // Drain 4 of the held blocks -> they retire (not freed) until target met.
        bm.free_all(&held[0..4]);
        assert_eq!(bm.num_retired(), 6); // 2 + 4 retired -> circulation now 4
        assert_eq!(bm.num_used(), 4);
        assert_eq!(bm.num_free(), 0);
        // Freeing the rest now returns to the free list (target reached).
        bm.free_all(&held[4..8]);
        assert_eq!(bm.num_retired(), 6);
        assert_eq!(bm.num_free(), 4);
        assert_eq!(bm.num_used(), 0);
    }

    #[test]
    fn shrink_retires_highest_ids_so_tail_clears() {
        let mut bm = BlockManager::new(10, 16);
        // All 10 free. Shrink to 6 must retire the HIGHEST 4 ids (6,7,8,9) so
        // the pool tail clears and a physical truncate to 6 becomes sound.
        bm.set_target_usable(6);
        assert_eq!(bm.num_retired(), 4);
        // Nothing in use, so the physical floor is 0 and we can truncate to 6.
        assert_eq!(bm.physical_floor(), 0);
        bm.physical_truncate(6).expect("tail is clear");
        assert_eq!(bm.num_blocks(), 6);
        assert_eq!(bm.num_retired(), 0); // retired ids 6..10 dropped with the tail
        assert_eq!(bm.num_free(), 6);
        // Every surviving id is < 6 — no orphaned high block.
        for _ in 0..6 {
            assert!(bm.allocate_one().unwrap() < 6);
        }
    }

    #[test]
    fn physical_truncate_blocked_by_live_tail_then_succeeds_after_drain() {
        let mut bm = BlockManager::new(8, 16);
        // Hold ids 0..6 (front-pop), leaving 6,7 free.
        let held = bm.allocate(6).unwrap();
        // Logical shrink to 4: retire the highest free ids (6,7).
        bm.set_target_usable(4);
        assert_eq!(bm.num_retired(), 2);
        // A live block still sits at id 5 → floor is 6, can't truncate to 4 yet.
        assert_eq!(bm.physical_floor(), 6);
        assert!(bm.physical_truncate(4).is_err());
        // Drain the high holders (ids 4,5) — they retire (target not yet met).
        bm.free_all(&held[4..6]);
        assert_eq!(bm.physical_floor(), 4); // ids 0..4 still live
        bm.physical_truncate(4).expect("tail clear after drain");
        assert_eq!(bm.num_blocks(), 4);
        assert_eq!(bm.num_used(), 4);
    }

    #[test]
    fn physical_grow_restores_capacity_and_round_trips() {
        let mut bm = BlockManager::new(6, 16);
        bm.set_target_usable(4);
        bm.physical_truncate(4).unwrap();
        assert_eq!(bm.num_blocks(), 4);
        // Grow back to 10: add physical capacity FIRST (set_target_usable is
        // clamped to num_blocks), then raise the logical ceiling to make the new
        // blocks usable. The new ids 4..10 land retired (above the old ceiling),
        // then the ceiling-raise un-retires them into the free list.
        bm.physical_grow(10);
        assert_eq!(bm.num_blocks(), 10);
        bm.set_target_usable(10);
        assert_eq!(bm.num_retired(), 0);
        assert!(bm.can_allocate(10));
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

    #[test]
    fn unique_physical_blocks_preserves_first_seen_order() {
        let mut first = BlockTable::new();
        first.push(5);
        first.push(9);
        first.push(5);

        let mut second = BlockTable::new();
        second.push(9);
        second.push(2);
        second.push(7);

        let empty = BlockTable::new();
        let block_tables = [&first, &empty, &second];
        assert_eq!(unique_physical_blocks(&block_tables), vec![5, 9, 2, 7]);
    }

    // Relocated from kiln_model::paged_kv_cache during the #1082 candle-drop.
    #[test]
    fn test_contiguous_slot_run_start_detection() {
        let mut contiguous = BlockTable::new();
        contiguous.push(5);
        contiguous.push(6);
        contiguous.push(7);

        assert_eq!(contiguous_slot_run_start(&contiguous, 4, 0, 6), Some(20));
        assert_eq!(contiguous_slot_run_start(&contiguous, 4, 2, 6), Some(22));
        assert_eq!(contiguous_slot_run_start(&contiguous, 4, 4, 4), Some(24));

        let mut non_contiguous = BlockTable::new();
        non_contiguous.push(5);
        non_contiguous.push(7);
        non_contiguous.push(8);

        assert_eq!(
            contiguous_slot_run_start(&non_contiguous, 4, 0, 4),
            Some(20)
        );
        assert_eq!(contiguous_slot_run_start(&non_contiguous, 4, 0, 6), None);
        assert_eq!(contiguous_slot_run_start(&non_contiguous, 4, 2, 6), None);
    }

    // Relocated from kiln_model::paged_kv_cache during the #1082 candle-drop.
    #[test]
    fn test_contiguous_slot_run_starts_detection() {
        let mut first = BlockTable::new();
        first.push(5);
        first.push(6);
        first.push(7);

        let mut second = BlockTable::new();
        second.push(11);
        second.push(12);
        second.push(13);

        let block_tables = [&first, &second];
        assert_eq!(
            contiguous_slot_run_starts(&block_tables, 4, &[0, 2], 6),
            Some(vec![20, 46])
        );

        let mut non_contiguous = BlockTable::new();
        non_contiguous.push(20);
        non_contiguous.push(22);

        let block_tables = [&first, &non_contiguous];
        assert_eq!(
            contiguous_slot_run_starts(&block_tables, 4, &[0, 0], 6),
            None
        );
        assert_eq!(
            contiguous_slot_run_starts(&block_tables, 4, &[0, 0], 0),
            None
        );
        assert_eq!(contiguous_slot_run_starts(&block_tables, 4, &[0], 6), None);
    }
}
