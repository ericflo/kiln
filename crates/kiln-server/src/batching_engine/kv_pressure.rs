use anyhow::Result;
use kiln_core::block::BlockError;

use super::{DecodeSlot, RealDecodeForward, blocks_needed_for_tokens};

impl RealDecodeForward {
    /// Grow each ready row independently. A capacity shortage starves only the
    /// affected rows, allowing peers with sufficient KV to keep decoding.
    pub(super) fn grow_for_decode_per_slot(
        &self,
        slots: &mut [&mut DecodeSlot],
    ) -> Result<Vec<usize>> {
        let block_size = self.block_manager_guard()?.block_size();
        let mut starved = Vec::new();
        for (idx, slot) in slots.iter_mut().enumerate() {
            let DecodeSlot::Real {
                state,
                first_token_pending: false,
                ..
            } = &mut **slot
            else {
                continue;
            };
            let required_blocks =
                blocks_needed_for_tokens(state.seq_len.saturating_add(1), block_size);
            let missing = required_blocks.saturating_sub(state.block_table.blocks.len());
            if missing == 0 {
                continue;
            }
            let initial_allocation = {
                let mut block_manager = self.block_manager_guard()?;
                block_manager.allocate(missing)
            };
            let allocated = match initial_allocation {
                Ok(new_blocks) => Some(new_blocks),
                Err(BlockError::OutOfMemory { available, .. }) => {
                    let shortfall = missing.saturating_sub(available);
                    let released = self
                        .prefix_cache_guard()?
                        .evict_unleased_lru_blocks(shortfall);
                    if released.is_empty() {
                        None
                    } else {
                        let released_count = released.len();
                        let retry = {
                            let mut block_manager = self.block_manager_guard()?;
                            block_manager.free_all(&released);
                            block_manager.allocate(missing)
                        };
                        tracing::warn!(
                            requested_blocks = missing,
                            reclaimed_prefix_blocks = released_count,
                            "reclaimed unleased prefix-cache blocks for live decode growth"
                        );
                        retry.ok()
                    }
                }
            };
            match allocated {
                Some(new_blocks) => {
                    state.block_table.blocks.extend(new_blocks.iter().copied());
                    state.allocated_blocks.extend(new_blocks.iter().copied());
                }
                None => starved.push(idx),
            }
        }
        Ok(starved)
    }
}
