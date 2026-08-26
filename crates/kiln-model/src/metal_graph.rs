//! Metal ICB graph runner seam for single-token paged decode.
//!
//! This mirrors the CUDA/ROCm graph-runner lifecycle: environment-gated
//! enablement, graph invalidation on adapter/weight-pointer changes, and
//! eager fallback while capture is unavailable for a shape. The actual
//! replay object is `kiln_graph_metal::MetalCapturedGraph`, which owns an
//! `MTLIndirectCommandBuffer` under the `metal` feature.

#[cfg(feature = "metal")]
use anyhow::Context;
use anyhow::Result;
#[cfg(feature = "metal")]
use std::collections::{HashMap, hash_map::DefaultHasher};
#[cfg(feature = "metal")]
use std::hash::{Hash, Hasher};

use kiln_core::block::BlockTable;
use kiln_core::config::ModelConfig;
use kiln_core::token::TokenId;

use crate::PagedKvCacheKt;
use crate::backend::BackendRuntime;
#[cfg(feature = "metal")]
use crate::backend::SamplingBackend;
#[cfg(feature = "metal")]
use crate::forward::MetalPagedDecodeIcbInputs;
use crate::forward::{GpuWeights, LinearAttentionState, model_forward_paged_next_token_greedy};
// Consumed only by the `metal`-gated impl block below (stable-decode steps +
// weight fingerprint); keep gated so feature-less builds don't warn.
#[cfg(feature = "metal")]
use crate::forward::{
    GpuAttentionWeights, model_forward_paged_decode_contiguous_batch_greedy_with_stable_buffers,
    model_forward_paged_decode_contiguous_batch_hidden_with_stable_buffers, rms_norm,
};
use crate::lora_loader::LoraWeights;

#[cfg(feature = "metal")]
pub(crate) fn replay_paged_decode_icb_graph_through_replay_plan(
    graph: &crate::backend::metal::MetalPagedDecodeIcbGraph,
    max_seqlen_k: u32,
    softmax_scale: f32,
) -> Result<()> {
    let mut plan = graph.replay_plan(max_seqlen_k, softmax_scale);
    let replay_key = kiln_graph::ReplayPlan::key(&plan);
    let replay_inputs = kiln_graph::ReplayInputs::new(&replay_key, graph.replay_resources());
    kiln_graph::ReplayPlan::replay(&mut plan, replay_inputs)
        .map(|_| ())
        .map_err(|e| anyhow::anyhow!("{e}"))
}

#[cfg(feature = "metal")]
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
struct MetalGraphKey {
    stable_metadata: bool,
    batch_size: usize,
    seq_len: usize,
    block_table: Vec<u32>,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
}

#[cfg(feature = "metal")]
impl MetalGraphKey {
    fn new(block_table: &BlockTable, paged_cache: &PagedKvCacheKt, seq_len: usize) -> Self {
        Self::new_batch(&[block_table], paged_cache, &[seq_len])
    }

    fn new_batch(
        block_tables: &[&BlockTable],
        paged_cache: &PagedKvCacheKt,
        seq_lens: &[usize],
    ) -> Self {
        let stable_metadata = true;
        let batch_size = seq_lens.len();
        let max_seq_len = seq_lens.iter().copied().max().unwrap_or(0);
        let attention_len = max_seq_len + 1;
        let kblock_n = crate::generate::FA2_KBLOCK_N;
        let max_seqlen_k = attention_len.div_ceil(kblock_n) * kblock_n;
        let pages_per_chunk = kblock_n / paged_cache.block_size();
        let max_blocks_per_seq = (max_seqlen_k / kblock_n) * pages_per_chunk;
        Self {
            stable_metadata,
            batch_size,
            seq_len: if stable_metadata { 0 } else { max_seq_len },
            block_table: if stable_metadata {
                Vec::new()
            } else {
                block_tables
                    .iter()
                    .flat_map(|bt| bt.blocks.iter().copied())
                    .collect()
            },
            max_seqlen_k,
            max_blocks_per_seq,
        }
    }
}

/// Runs Metal decode through pre-recorded ICBs when available, otherwise
/// preserves the eager Metal path. Capture wiring lands after the substrate
/// can pre-encode every graph-stable decode dispatch.
#[derive(Debug)]
pub struct MetalGraphRunner {
    enabled: bool,
    adapter_generation: u64,
    stable_path_warned: bool,
    sampled_stable_path_warned: bool,
    #[cfg(feature = "metal")]
    stable_buffers: HashMap<MetalGraphKey, MetalStableDecodeBuffers>,
    #[cfg(feature = "metal")]
    last_decode_seq_len: Option<usize>,
    #[cfg(feature = "metal")]
    last_decode_block0: Option<u32>,
    #[cfg(feature = "metal")]
    last_weight_fingerprint: Option<u64>,
}

#[cfg(feature = "metal")]
#[derive(Debug)]
struct MetalStableDecodeBuffers {
    adapter_gen: u64,
    token_ids: kiln_tensor::Tensor,
    positions: kiln_tensor::Tensor,
    block_table: kiln_tensor::Tensor,
    seqused_k: kiln_tensor::Tensor,
    kv_slot: kiln_tensor::Tensor,
    rotary_cos: kiln_tensor::Tensor,
    rotary_sin: kiln_tensor::Tensor,
    stable_q: Vec<kiln_tensor::Tensor>,
    stable_k: Vec<kiln_tensor::Tensor>,
    stable_v: Vec<kiln_tensor::Tensor>,
    attn_out: Vec<kiln_tensor::Tensor>,
    softmax_lse: Vec<kiln_tensor::Tensor>,
    icb_graphs: Vec<Option<crate::backend::metal::MetalPagedDecodeIcbGraph>>,
    max_seqlen_k: usize,
}

impl MetalGraphRunner {
    pub fn new(device: &kiln_tensor::Device, enabled: bool) -> Self {
        let is_metal = matches!(device, kiln_tensor::Device::Metal(_));
        let actually_enabled = enabled && is_metal;
        if actually_enabled {
            tracing::info!("Metal ICB graphs enabled for decode");
        }
        Self {
            enabled: actually_enabled,
            adapter_generation: 0,
            stable_path_warned: false,
            sampled_stable_path_warned: false,
            #[cfg(feature = "metal")]
            stable_buffers: HashMap::new(),
            #[cfg(feature = "metal")]
            last_decode_seq_len: None,
            #[cfg(feature = "metal")]
            last_decode_block0: None,
            #[cfg(feature = "metal")]
            last_weight_fingerprint: None,
        }
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    pub fn adapter_generation(&self) -> u64 {
        self.adapter_generation
    }

    pub fn invalidate(&mut self) {
        self.adapter_generation += 1;
        self.stable_path_warned = false;
        self.sampled_stable_path_warned = false;
        #[cfg(feature = "metal")]
        {
            self.stable_buffers.clear();
            self.last_decode_seq_len = None;
            self.last_decode_block0 = None;
            self.last_weight_fingerprint = None;
        }
    }

    #[cfg(all(test, feature = "metal"))]
    pub(crate) fn stable_buffer_count(&self) -> usize {
        self.stable_buffers.len()
    }

    #[cfg(all(test, feature = "metal"))]
    pub(crate) fn captured_graph_count(&self) -> usize {
        self.stable_buffers
            .values()
            .map(|buffers| {
                buffers
                    .icb_graphs
                    .iter()
                    .filter(|graph| graph.is_some())
                    .count()
            })
            .sum()
    }

    #[cfg(all(test, feature = "metal"))]
    pub(crate) fn captured_graph_replay_count_sum(&self) -> u64 {
        self.stable_buffers
            .values()
            .flat_map(|buffers| buffers.icb_graphs.iter())
            .filter_map(|graph| graph.as_ref())
            .map(|graph| graph.replay_count())
            .sum()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_greedy(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<TokenId> {
        if !self.enabled {
            return Self::eager_greedy(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            );
        }

        #[cfg(feature = "metal")]
        {
            self.invalidate_if_weights_changed(weights);

            if let Some(token) = self.try_decode_step_paged_greedy_stable(
                backend,
                token_id,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                linear_state,
                lora,
            )? {
                return Ok(token);
            }
        }

        if !self.stable_path_warned {
            tracing::warn!(
                "Metal ICB graph runner enabled, but stable decode path is unavailable for this shape; using eager Metal decode"
            );
            self.stable_path_warned = true;
        }

        Self::eager_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            linear_state,
            lora,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_greedy_batch(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_state: Option<&mut LinearAttentionState>,
        lora: Option<&LoraWeights>,
    ) -> Result<Option<Vec<TokenId>>> {
        if !self.enabled {
            return Ok(None);
        }

        #[cfg(feature = "metal")]
        {
            self.invalidate_if_weights_changed(weights);
            return self.try_decode_step_paged_greedy_batch_stable(
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                linear_state,
                lora,
            );
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                linear_state,
                lora,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn decode_step_paged_sample_batch(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_state: Option<&mut LinearAttentionState>,
        lora: Option<&LoraWeights>,
        history_rows: &[u32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalties: &[f32],
        presence_penalties: &[f32],
        frequency_penalties: &[f32],
        temperatures: &[f32],
        top_k: &[u32],
        top_p: &[f32],
        min_p: &[f32],
        seeds: &[u64],
    ) -> Result<Option<Vec<TokenId>>> {
        if !self.enabled {
            return Ok(None);
        }

        #[cfg(feature = "metal")]
        {
            self.invalidate_if_weights_changed(weights);
            return self.try_decode_step_paged_sample_batch_stable(
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                linear_state,
                lora,
                history_rows,
                history_indices,
                history_counts,
                repetition_penalties,
                presence_penalties,
                frequency_penalties,
                temperatures,
                top_k,
                top_p,
                min_p,
                seeds,
            );
        }

        #[cfg(not(feature = "metal"))]
        {
            let _ = (
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                linear_state,
                lora,
                history_rows,
                history_indices,
                history_counts,
                repetition_penalties,
                presence_penalties,
                frequency_penalties,
                temperatures,
                top_k,
                top_p,
                min_p,
                seeds,
            );
            Ok(None)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn eager_greedy(
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<TokenId> {
        model_forward_paged_next_token_greedy(
            backend,
            token_id,
            weights,
            config,
            paged_cache,
            block_table,
            seq_len,
            Some(linear_state),
            lora,
            None,
        )
    }
}

#[cfg(feature = "metal")]
impl MetalGraphRunner {
    fn invalidate_if_weights_changed(&mut self, weights: &GpuWeights) {
        let fingerprint = metal_graph_weight_fingerprint(weights);
        if self
            .last_weight_fingerprint
            .is_some_and(|previous| previous != fingerprint)
        {
            tracing::debug!(
                "Metal graph: weight tensor identity changed - evicting stable decode buffers"
            );
            self.invalidate();
        }
        self.last_weight_fingerprint = Some(fingerprint);
    }

    #[allow(clippy::too_many_arguments)]
    fn try_decode_step_paged_greedy_stable(
        &mut self,
        backend: &dyn BackendRuntime,
        token_id: u32,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        linear_state: &mut LinearAttentionState,
        lora: Option<&LoraWeights>,
    ) -> Result<Option<TokenId>> {
        if !matches!(weights.embed_tokens.device(), kiln_tensor::Device::Metal(_)) {
            return Ok(None);
        }
        let block0 = block_table.blocks.first().copied();
        let continues = block0.is_some()
            && self.last_decode_seq_len == Some(seq_len.wrapping_sub(1))
            && self.last_decode_block0 == block0;
        if !continues && !self.stable_buffers.is_empty() {
            tracing::debug!(
                seq_len,
                "Metal graph: request boundary - evicting stable bs=1 decode buffers"
            );
            self.stable_buffers.clear();
        }
        self.last_decode_seq_len = Some(seq_len);
        self.last_decode_block0 = block0;

        let key = MetalGraphKey::new(block_table, paged_cache, seq_len);
        if key.max_blocks_per_seq == 0 {
            return Ok(None);
        }

        if !self.stable_buffers.contains_key(&key) {
            let buffers = MetalStableDecodeBuffers::new(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                &key,
            )?;
            self.stable_buffers.insert(key.clone(), buffers);
        }

        let buffers = self
            .stable_buffers
            .get_mut(&key)
            .context("Metal stable decode buffers vanished after allocation")?;
        if buffers.adapter_gen != self.adapter_generation {
            *buffers = MetalStableDecodeBuffers::new(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_table,
                seq_len,
                &key,
            )?;
        }
        buffers.refresh(token_id, config, paged_cache, block_table, seq_len)?;

        let block_tables = [block_table];
        let start_positions = [seq_len];
        let metal_icb_inputs = MetalPagedDecodeIcbInputs {
            q: buffers.stable_q.as_slice(),
            k: buffers.stable_k.as_slice(),
            v: buffers.stable_v.as_slice(),
            graphs: buffers.icb_graphs.as_mut_slice(),
        };
        match model_forward_paged_decode_contiguous_batch_greedy_with_stable_buffers(
            backend,
            &[token_id],
            weights,
            config,
            paged_cache,
            &block_tables,
            &start_positions,
            Some(linear_state),
            lora,
            &buffers.positions,
            &buffers.token_ids,
            &buffers.block_table,
            &buffers.seqused_k,
            &buffers.attn_out,
            &buffers.softmax_lse,
            &buffers.rotary_cos,
            &buffers.rotary_sin,
            &buffers.kv_slot,
            Some(metal_icb_inputs),
        ) {
            Ok(tokens) => {
                let token = tokens
                    .first()
                    .copied()
                    .context("Metal stable greedy decode returned no token")?;
                if !self.stable_path_warned {
                    tracing::info!(
                        max_seqlen_k = buffers.max_seqlen_k,
                        "Metal graph runner using graph-stable bs=1 decode buffers"
                    );
                    self.stable_path_warned = true;
                }
                Ok(Some(token))
            }
            Err(err) => {
                tracing::warn!(
                    error = %err,
                    "Metal graph-stable decode path failed; falling back to eager Metal decode"
                );
                self.stable_buffers.remove(&key);
                Ok(None)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_decode_step_paged_greedy_batch_stable(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_state: Option<&mut LinearAttentionState>,
        lora: Option<&LoraWeights>,
    ) -> Result<Option<Vec<TokenId>>> {
        if !matches!(weights.embed_tokens.device(), kiln_tensor::Device::Metal(_)) {
            return Ok(None);
        }
        let batch = token_ids.len();
        if batch == 0 || block_tables.len() != batch || seq_lens.len() != batch {
            return Ok(None);
        }

        let key = MetalGraphKey::new_batch(block_tables, paged_cache, seq_lens);
        if key.max_blocks_per_seq == 0 {
            return Ok(None);
        }

        if !self.stable_buffers.contains_key(&key) {
            let buffers = MetalStableDecodeBuffers::new_batch(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                &key,
            )?;
            self.stable_buffers.insert(key.clone(), buffers);
        }

        let buffers = self
            .stable_buffers
            .get_mut(&key)
            .context("Metal stable batched decode buffers vanished after allocation")?;
        if buffers.adapter_gen != self.adapter_generation {
            *buffers = MetalStableDecodeBuffers::new_batch(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                &key,
            )?;
        }
        buffers.refresh_batch(token_ids, config, paged_cache, block_tables, seq_lens)?;

        if batch == 1
            && lora.is_none()
            && SamplingBackend::runtime_supports_linear_decode_sample(backend, 1)
        {
            let metal_icb_inputs = MetalPagedDecodeIcbInputs {
                q: buffers.stable_q.as_slice(),
                k: buffers.stable_k.as_slice(),
                v: buffers.stable_v.as_slice(),
                graphs: buffers.icb_graphs.as_mut_slice(),
            };
            let sample_result = (|| -> Result<Vec<TokenId>> {
                let hidden =
                    model_forward_paged_decode_contiguous_batch_hidden_with_stable_buffers(
                        backend,
                        token_ids,
                        weights,
                        config,
                        paged_cache,
                        block_tables,
                        seq_lens,
                        linear_state,
                        lora,
                        &buffers.positions,
                        &buffers.token_ids,
                        &buffers.block_table,
                        &buffers.seqused_k,
                        &buffers.attn_out,
                        &buffers.softmax_lse,
                        &buffers.rotary_cos,
                        &buffers.rotary_sin,
                        &buffers.kv_slot,
                        Some(metal_icb_inputs),
                    )?;
                let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
                let token = SamplingBackend::runtime_linear_decode_sample(
                    backend,
                    &normed,
                    &weights.embed_tokens_t,
                    &[],
                    &[],
                    1.0,
                    0.0,
                    0.0,
                    1.0,
                    1,
                    1.0,
                    0.0,
                    0,
                )?
                .context("Metal graph greedy sampler tail declined top-k=1")?;
                Ok(vec![token])
            })();
            match sample_result {
                Ok(tokens) => {
                    if !self.stable_path_warned {
                        tracing::info!(
                            batch,
                            max_seqlen_k = buffers.max_seqlen_k,
                            "Metal graph runner using graph-stable batched decode buffers"
                        );
                        self.stable_path_warned = true;
                    }
                    return Ok(Some(tokens));
                }
                Err(err) => {
                    tracing::warn!(
                        batch,
                        error = %err,
                        "Metal graph-stable greedy sampler tail failed; falling back to eager Metal decode"
                    );
                    self.stable_buffers.remove(&key);
                    return Ok(None);
                }
            }
        }

        let metal_icb_inputs = MetalPagedDecodeIcbInputs {
            q: buffers.stable_q.as_slice(),
            k: buffers.stable_k.as_slice(),
            v: buffers.stable_v.as_slice(),
            graphs: buffers.icb_graphs.as_mut_slice(),
        };
        match model_forward_paged_decode_contiguous_batch_greedy_with_stable_buffers(
            backend,
            token_ids,
            weights,
            config,
            paged_cache,
            block_tables,
            seq_lens,
            linear_state,
            lora,
            &buffers.positions,
            &buffers.token_ids,
            &buffers.block_table,
            &buffers.seqused_k,
            &buffers.attn_out,
            &buffers.softmax_lse,
            &buffers.rotary_cos,
            &buffers.rotary_sin,
            &buffers.kv_slot,
            Some(metal_icb_inputs),
        ) {
            Ok(tokens) => {
                if !self.stable_path_warned {
                    tracing::info!(
                        batch,
                        max_seqlen_k = buffers.max_seqlen_k,
                        "Metal graph runner using graph-stable batched decode buffers"
                    );
                    self.stable_path_warned = true;
                }
                Ok(Some(tokens))
            }
            Err(err) => {
                tracing::warn!(
                    batch,
                    error = %err,
                    "Metal graph-stable batched decode path failed; falling back to eager Metal decode"
                );
                self.stable_buffers.remove(&key);
                Ok(None)
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn try_decode_step_paged_sample_batch_stable(
        &mut self,
        backend: &dyn BackendRuntime,
        token_ids: &[u32],
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        linear_state: Option<&mut LinearAttentionState>,
        lora: Option<&LoraWeights>,
        history_rows: &[u32],
        history_indices: &[u32],
        history_counts: &[u32],
        repetition_penalties: &[f32],
        presence_penalties: &[f32],
        frequency_penalties: &[f32],
        temperatures: &[f32],
        top_k: &[u32],
        top_p: &[f32],
        min_p: &[f32],
        seeds: &[u64],
    ) -> Result<Option<Vec<TokenId>>> {
        if !matches!(weights.embed_tokens.device(), kiln_tensor::Device::Metal(_)) {
            return Ok(None);
        }
        let batch = token_ids.len();
        if batch == 0
            || block_tables.len() != batch
            || seq_lens.len() != batch
            || repetition_penalties.len() != batch
            || presence_penalties.len() != batch
            || frequency_penalties.len() != batch
            || temperatures.len() != batch
            || top_k.len() != batch
            || top_p.len() != batch
            || min_p.len() != batch
            || seeds.len() != batch
            || history_rows.len() != history_indices.len()
            || history_rows.len() != history_counts.len()
        {
            return Ok(None);
        }
        if lora.is_some()
            || !SamplingBackend::runtime_supports_linear_decode_sample_batch(
                backend,
                top_k,
                temperatures,
            )
        {
            return Ok(None);
        }

        let key = MetalGraphKey::new_batch(block_tables, paged_cache, seq_lens);
        if key.max_blocks_per_seq == 0 {
            return Ok(None);
        }

        if !self.stable_buffers.contains_key(&key) {
            let buffers = MetalStableDecodeBuffers::new_batch(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                &key,
            )?;
            self.stable_buffers.insert(key.clone(), buffers);
        }

        let buffers = self
            .stable_buffers
            .get_mut(&key)
            .context("Metal stable sampled decode buffers vanished after allocation")?;
        if buffers.adapter_gen != self.adapter_generation {
            *buffers = MetalStableDecodeBuffers::new_batch(
                self.adapter_generation,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                &key,
            )?;
        }
        buffers.refresh_batch(token_ids, config, paged_cache, block_tables, seq_lens)?;

        let metal_icb_inputs = MetalPagedDecodeIcbInputs {
            q: buffers.stable_q.as_slice(),
            k: buffers.stable_k.as_slice(),
            v: buffers.stable_v.as_slice(),
            graphs: buffers.icb_graphs.as_mut_slice(),
        };
        let sample_result = (|| -> Result<Option<Vec<TokenId>>> {
            let hidden = model_forward_paged_decode_contiguous_batch_hidden_with_stable_buffers(
                backend,
                token_ids,
                weights,
                config,
                paged_cache,
                block_tables,
                seq_lens,
                linear_state,
                lora,
                &buffers.positions,
                &buffers.token_ids,
                &buffers.block_table,
                &buffers.seqused_k,
                &buffers.attn_out,
                &buffers.softmax_lse,
                &buffers.rotary_cos,
                &buffers.rotary_sin,
                &buffers.kv_slot,
                Some(metal_icb_inputs),
            )?;
            let normed = rms_norm(&hidden, &weights.final_norm, config.rms_norm_eps)?;
            let tokens = SamplingBackend::runtime_linear_decode_sample_batch(
                backend,
                &normed,
                &weights.embed_tokens_t,
                history_rows,
                history_indices,
                history_counts,
                repetition_penalties,
                presence_penalties,
                frequency_penalties,
                temperatures,
                top_k,
                top_p,
                min_p,
                seeds,
            )?;
            Ok(tokens)
        })();

        match sample_result {
            Ok(Some(tokens)) => {
                if !self.sampled_stable_path_warned {
                    tracing::info!(
                        batch,
                        max_seqlen_k = buffers.max_seqlen_k,
                        "Metal graph runner using graph-stable sampled decode buffers"
                    );
                    self.sampled_stable_path_warned = true;
                }
                Ok(Some(tokens))
            }
            Ok(None) => Ok(None),
            Err(err) => {
                tracing::warn!(
                    batch,
                    error = %err,
                    "Metal graph-stable sampled decode path failed; falling back to eager Metal decode"
                );
                self.stable_buffers.remove(&key);
                Ok(None)
            }
        }
    }
}

#[cfg(feature = "metal")]
fn metal_graph_weight_fingerprint(weights: &GpuWeights) -> u64 {
    let mut hasher = DefaultHasher::new();
    hash_tensor_id(&mut hasher, &weights.embed_tokens);
    hash_tensor_id(&mut hasher, &weights.embed_tokens_t);
    hash_tensor_id(&mut hasher, &weights.final_norm);
    hash_tensor_id(&mut hasher, &weights.rotary_inv_freq);
    for layer in &weights.layers {
        hash_tensor_id(&mut hasher, &layer.input_layernorm);
        hash_tensor_id(&mut hasher, &layer.post_attention_layernorm);
        match &layer.attention {
            GpuAttentionWeights::Full(attn) => {
                0u8.hash(&mut hasher);
                hash_tensor_id(&mut hasher, &attn.q_proj);
                hash_tensor_id(&mut hasher, &attn.k_proj);
                hash_tensor_id(&mut hasher, &attn.v_proj);
                hash_tensor_id(&mut hasher, &attn.o_proj);
                hash_tensor_id(&mut hasher, &attn.q_norm);
                hash_tensor_id(&mut hasher, &attn.k_norm);
                hash_tensor_id(&mut hasher, &attn.q_proj_t);
                hash_tensor_id(&mut hasher, &attn.k_proj_t);
                hash_tensor_id(&mut hasher, &attn.v_proj_t);
                hash_optional_tensor_id(&mut hasher, attn.qkv_proj_t.as_ref());
                hash_tensor_id(&mut hasher, &attn.o_proj_t);
            }
            GpuAttentionWeights::Linear(attn) => {
                1u8.hash(&mut hasher);
                hash_tensor_id(&mut hasher, &attn.in_proj_qkv);
                hash_tensor_id(&mut hasher, &attn.in_proj_z);
                hash_tensor_id(&mut hasher, &attn.out_proj);
                hash_tensor_id(&mut hasher, &attn.in_proj_a);
                hash_tensor_id(&mut hasher, &attn.in_proj_b);
                hash_tensor_id(&mut hasher, &attn.conv1d);
                hash_tensor_id(&mut hasher, &attn.norm);
                hash_tensor_id(&mut hasher, &attn.a_log);
                hash_tensor_id(&mut hasher, &attn.a_log_gates);
                hash_tensor_id(&mut hasher, &attn.dt_bias);
                hash_tensor_id(&mut hasher, &attn.in_proj_qkv_t);
                hash_tensor_id(&mut hasher, &attn.in_proj_z_t);
                hash_tensor_id(&mut hasher, &attn.in_proj_a_t);
                hash_tensor_id(&mut hasher, &attn.in_proj_b_t);
                hash_optional_tensor_id(&mut hasher, attn.in_proj_ab_t.as_ref());
                hash_tensor_id(&mut hasher, &attn.out_proj_t);
            }
        }
        hash_tensor_id(&mut hasher, &layer.mlp.gate_proj);
        hash_tensor_id(&mut hasher, &layer.mlp.up_proj);
        hash_tensor_id(&mut hasher, &layer.mlp.down_proj);
        hash_tensor_id(&mut hasher, &layer.mlp.gate_proj_t);
        hash_tensor_id(&mut hasher, &layer.mlp.up_proj_t);
        hash_tensor_id(&mut hasher, &layer.mlp.down_proj_t);
        hash_optional_tensor_id(&mut hasher, layer.mlp.gate_up_proj_t.as_ref());
    }
    hasher.finish()
}

#[cfg(feature = "metal")]
fn hash_tensor_id(hasher: &mut DefaultHasher, tensor: &kiln_tensor::Tensor) {
    tensor.id().hash(hasher);
}

#[cfg(feature = "metal")]
fn hash_optional_tensor_id(hasher: &mut DefaultHasher, tensor: Option<&kiln_tensor::Tensor>) {
    match tensor {
        Some(tensor) => {
            true.hash(hasher);
            hash_tensor_id(hasher, tensor);
        }
        None => false.hash(hasher),
    }
}

#[cfg(feature = "metal")]
impl MetalStableDecodeBuffers {
    fn new(
        adapter_gen: u64,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
        key: &MetalGraphKey,
    ) -> Result<Self> {
        Self::new_batch(
            adapter_gen,
            weights,
            config,
            paged_cache,
            &[block_table],
            &[seq_len],
            key,
        )
    }

    fn new_batch(
        adapter_gen: u64,
        weights: &GpuWeights,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
        key: &MetalGraphKey,
    ) -> Result<Self> {
        let device = weights.embed_tokens.device();
        let batch = key.batch_size;
        anyhow::ensure!(
            batch > 0 && block_tables.len() == batch && seq_lens.len() == batch,
            "Metal stable decode buffers batch metadata mismatch"
        );
        let full_layers = config.num_full_attention_layers;
        let mut stable_q = Vec::with_capacity(full_layers);
        let mut stable_k = Vec::with_capacity(full_layers);
        let mut stable_v = Vec::with_capacity(full_layers);
        let mut attn_out = Vec::with_capacity(full_layers);
        let mut softmax_lse = Vec::with_capacity(full_layers);
        for _ in 0..full_layers {
            stable_q.push(kiln_tensor::Tensor::zeros_on(
                device.clone(),
                vec![batch, 1, config.num_attention_heads, config.head_dim],
                kiln_tensor::DType::BF16,
            )?);
            stable_k.push(kiln_tensor::Tensor::zeros_on(
                device.clone(),
                vec![batch, 1, config.num_kv_heads, config.head_dim],
                kiln_tensor::DType::BF16,
            )?);
            stable_v.push(kiln_tensor::Tensor::zeros_on(
                device.clone(),
                vec![batch, 1, config.num_kv_heads, config.head_dim],
                kiln_tensor::DType::BF16,
            )?);
            attn_out.push(kiln_tensor::Tensor::zeros_on(
                device.clone(),
                vec![batch, 1, config.num_attention_heads, config.head_dim],
                kiln_tensor::DType::BF16,
            )?);
            softmax_lse.push(kiln_tensor::Tensor::zeros_on(
                device.clone(),
                vec![batch, config.num_attention_heads, 1],
                kiln_tensor::DType::F32,
            )?);
        }

        let mut buffers = Self {
            adapter_gen,
            token_ids: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0u32; batch],
                vec![batch],
            )?,
            positions: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0.0f32; batch],
                vec![batch],
            )?,
            block_table: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0u32; batch * key.max_blocks_per_seq],
                vec![batch, key.max_blocks_per_seq],
            )?,
            seqused_k: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0u32; batch],
                vec![batch],
            )?,
            kv_slot: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0u32; batch],
                vec![batch],
            )?,
            rotary_cos: kiln_tensor::Tensor::from_vec_on(
                device.clone(),
                vec![0.0f32; batch * (config.rotary_dim() / 2)],
                vec![batch, config.rotary_dim() / 2],
            )?,
            rotary_sin: kiln_tensor::Tensor::from_vec_on(
                device,
                vec![0.0f32; batch * (config.rotary_dim() / 2)],
                vec![batch, config.rotary_dim() / 2],
            )?,
            stable_q,
            stable_k,
            stable_v,
            attn_out,
            softmax_lse,
            icb_graphs: (0..full_layers).map(|_| None).collect(),
            max_seqlen_k: key.max_seqlen_k,
        };
        buffers.refresh_batch(
            &vec![0u32; batch],
            config,
            paged_cache,
            block_tables,
            seq_lens,
        )?;
        Ok(buffers)
    }

    fn refresh(
        &mut self,
        token_id: u32,
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_table: &BlockTable,
        seq_len: usize,
    ) -> Result<()> {
        self.refresh_batch(&[token_id], config, paged_cache, &[block_table], &[seq_len])
    }

    fn refresh_batch(
        &mut self,
        token_ids: &[u32],
        config: &ModelConfig,
        paged_cache: &PagedKvCacheKt,
        block_tables: &[&BlockTable],
        seq_lens: &[usize],
    ) -> Result<()> {
        let batch = token_ids.len();
        anyhow::ensure!(
            batch > 0 && block_tables.len() == batch && seq_lens.len() == batch,
            "Metal stable decode refresh batch metadata mismatch"
        );
        kiln_tensor::metal_write_host_in_place(&self.token_ids, token_ids)
            .context("update Metal graph token buffer")?;
        let positions: Vec<f32> = seq_lens.iter().map(|&seq_len| seq_len as f32).collect();
        kiln_tensor::metal_write_host_in_place(&self.positions, positions.as_slice())
            .context("update Metal graph position buffer")?;
        let padded = padded_block_table_batch(
            block_tables,
            paged_cache,
            self.max_seqlen_k,
            self.block_table.dims()[1],
        )?;
        kiln_tensor::metal_write_host_in_place(&self.block_table, padded.as_slice())
            .context("update Metal graph block table buffer")?;
        let seqused_k: Result<Vec<u32>> = seq_lens
            .iter()
            .map(|&seq_len| {
                u32::try_from(seq_len + 1).context("Metal graph seqused_k exceeds u32 range")
            })
            .collect();
        let seqused_k = seqused_k?;
        kiln_tensor::metal_write_host_in_place(&self.seqused_k, seqused_k.as_slice())
            .context("update Metal graph seqused_k buffer")?;
        let slots: Result<Vec<u32>> = block_tables
            .iter()
            .zip(seq_lens.iter().copied())
            .map(|(block_table, seq_len)| {
                let slot = block_table
                    .slot_for(seq_len, paged_cache.block_size())
                    .with_context(|| {
                        format!("no Metal graph KV slot for decode position {seq_len}")
                    })?;
                u32::try_from(slot).context("Metal graph KV slot exceeds u32 range")
            })
            .collect();
        let slots = slots?;
        kiln_tensor::metal_write_host_in_place(&self.kv_slot, slots.as_slice())
            .context("update Metal graph KV slot buffer")?;
        // #34 BUG2 FIX: compute the rotary tables on the GPU via eager's exact
        // path (`forward::rotary_tables_from_tensor`), one position per batch row,
        // not host CPU cos/sin. CPU cos != GPU cos perturbs only the RoPE
        // full-attention layers on replay. Same root cause + fix as CUDA/ROCm.
        let dev = self.rotary_cos.device();
        let inv_freq =
            crate::forward::compute_rotary_inv_freq(config.rotary_dim(), config.rope_theta, &dev)?;
        let pos_f32: Vec<f32> = seq_lens.iter().map(|&p| p as f32).collect();
        let n = pos_f32.len();
        let pos = kiln_tensor::Tensor::from_vec_on(dev, pos_f32, vec![n])?;
        let (cos, sin) = crate::forward::rotary_tables_from_tensor(&pos, &inv_freq)?;
        let cos = cos
            .to_dtype(self.rotary_cos.dtype())?
            .reshape(self.rotary_cos.dims().to_vec())?;
        let sin = sin
            .to_dtype(self.rotary_sin.dtype())?
            .reshape(self.rotary_sin.dims().to_vec())?;
        self.rotary_cos
            .slice_set(&cos, 0, 0)
            .context("update Metal graph rotary cos buffer (gpu)")?;
        self.rotary_sin
            .slice_set(&sin, 0, 0)
            .context("update Metal graph rotary sin buffer (gpu)")?;
        Ok(())
    }
}

#[cfg(feature = "metal")]
fn padded_block_table(
    block_table: &BlockTable,
    paged_cache: &PagedKvCacheKt,
    max_seqlen_k: usize,
) -> Result<Vec<u32>> {
    padded_block_table_batch(
        &[block_table],
        paged_cache,
        max_seqlen_k,
        (max_seqlen_k / paged_cache.block_size()).max(1),
    )
}

#[cfg(feature = "metal")]
fn padded_block_table_batch(
    block_tables: &[&BlockTable],
    paged_cache: &PagedKvCacheKt,
    max_seqlen_k: usize,
    max_blocks_per_seq: usize,
) -> Result<Vec<u32>> {
    let page_block_size = paged_cache.block_size();
    anyhow::ensure!(
        page_block_size > 0,
        "Metal graph padded block table requires non-zero page block size"
    );
    anyhow::ensure!(
        max_blocks_per_seq >= (max_seqlen_k / page_block_size).max(1),
        "Metal graph padded block table width is smaller than max_seqlen bucket"
    );
    let mut padded = Vec::with_capacity(block_tables.len() * max_blocks_per_seq);
    for block_table in block_tables {
        let take = max_blocks_per_seq.min(block_table.blocks.len());
        let row_start = padded.len();
        padded.extend_from_slice(&block_table.blocks[..take]);
        let pad_block = *padded
            .last()
            .context("Metal graph padded block table requires at least one block")?;
        while padded.len() < row_start + max_blocks_per_seq {
            padded.push(pad_block);
        }
    }
    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_device_never_enables() {
        let r = MetalGraphRunner::new(&kiln_tensor::Device::Cpu, true);
        assert!(!r.is_enabled());
    }

    #[test]
    fn invalidate_advances_generation() {
        let mut r = MetalGraphRunner::new(&kiln_tensor::Device::Cpu, false);
        assert_eq!(r.adapter_generation(), 0);
        r.invalidate();
        assert_eq!(r.adapter_generation(), 1);
    }
}
