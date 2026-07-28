use super::*;

#[test]
fn decode_pressure_evicts_lru_entries_without_revoking_live_leases() -> anyhow::Result<()> {
    let config = tiny_linear_config();
    let device = cpu_device!();
    let mut cache = RealPrefixCache::new(true, 4, 4, 1024, 49);
    cache.register(
        None,
        PagedPrefixRegistration {
            prompt_tokens: vec![1, 2, 3, 4],
            block_ids: vec![10],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: None,
        },
    );
    cache.register(
        test_adapter("evictable"),
        PagedPrefixRegistration {
            prompt_tokens: vec![5, 6, 7, 8, 9, 10, 11, 12],
            block_ids: vec![20, 21],
            linear_state: LinearAttentionState::new(&config, &device)?,
            next_token: None,
        },
    );
    let pinned = cache
        .lookup(&None, &[1, 2, 3, 4, 99], &SamplingParams::greedy())
        .hit?
        .expect("first entry must have a live lease");

    assert_eq!(cache.evict_unleased_lru_blocks(2), vec![20, 21]);
    assert_eq!(cache.stats().cached_blocks, 1);
    assert_eq!(cache.stats().active_leases, 1);
    assert_eq!(cache.block_refcounts, HashMap::from([(10, 1)]));

    cache.release_hit(pinned.entry_id);
    assert_eq!(cache.evict_unleased_lru_blocks(1), vec![10]);
    assert_eq!(cache.stats().cached_blocks, 0);
    assert!(cache.entries.is_empty());
    Ok(())
}
