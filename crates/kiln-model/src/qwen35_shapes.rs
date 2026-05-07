//! Compile-baked Qwen3.5-4B shape constants for decode-specialized code.
//!
//! Kiln intentionally targets one model architecture. Keeping these values as
//! Rust constants gives the raw decode executor and CUDA-call wrappers stable
//! dimensions for validation and later ptxas constant folding. `ModelConfig`
//! remains the public configuration surface for loading, prefill, and training;
//! production decode code should prefer this module when it needs canonical
//! Qwen3.5-4B dimensions.

use anyhow::{Result, ensure};

pub const HIDDEN: usize = 2560;
pub const NUM_HEADS: usize = 16;
pub const NUM_KV_HEADS: usize = 4;
pub const HEAD_DIM: usize = 256;
pub const NUM_LAYERS: usize = 32;
pub const NUM_GDN_LAYERS: usize = 24;
pub const NUM_FULL_ATTN_LAYERS: usize = 8;
pub const FULL_ATTN_INTERVAL: usize = 4;
pub const MLP_HIDDEN: usize = 9216;
pub const VOCAB: usize = 248_320;
pub const MAX_POSITION_EMBEDDINGS: usize = 262_144;
pub const RMS_NORM_EPS: f64 = 1e-6;
pub const PARTIAL_ROTARY: f64 = 0.25;
pub const ROTARY_DIM: usize = 64;
pub const ROPE_THETA: f64 = 10_000_000.0;
pub const ATTN_OUTPUT_GATE: bool = true;

pub const GDN_NUM_KEY_HEADS: usize = 16;
pub const GDN_KEY_HEAD_DIM: usize = 128;
pub const GDN_NUM_VALUE_HEADS: usize = 32;
pub const GDN_VALUE_HEAD_DIM: usize = 128;
pub const GDN_CONV_KERNEL_DIM: usize = 4;

pub const FULL_Q_WIDTH: usize = NUM_HEADS * HEAD_DIM;
pub const FULL_KV_WIDTH: usize = NUM_KV_HEADS * HEAD_DIM;
pub const FULL_Q_PROJ_OUT: usize = FULL_Q_WIDTH * 2;
pub const GDN_QK_WIDTH: usize = GDN_NUM_KEY_HEADS * GDN_KEY_HEAD_DIM;
pub const GDN_V_WIDTH: usize = GDN_NUM_VALUE_HEADS * GDN_VALUE_HEAD_DIM;
pub const GDN_QKV_PROJ_OUT: usize = GDN_QK_WIDTH * 2 + GDN_V_WIDTH;
pub const GDN_Z_PROJ_OUT: usize = GDN_V_WIDTH;
pub const GDN_AB_PROJ_OUT: usize = GDN_NUM_VALUE_HEADS;
pub const GDN_STATE_ELEMENTS_PER_STREAM: usize =
    GDN_NUM_VALUE_HEADS * GDN_KEY_HEAD_DIM * GDN_VALUE_HEAD_DIM;
pub const KV_ELEMENTS_PER_TOKEN_PER_FULL_LAYER: usize = NUM_KV_HEADS * HEAD_DIM * 2;

pub fn is_full_attention_layer(layer_idx: usize) -> bool {
    layer_idx % FULL_ATTN_INTERVAL == 0
}

pub fn full_attention_layer_index(layer_idx: usize) -> Option<usize> {
    if layer_idx < NUM_LAYERS && is_full_attention_layer(layer_idx) {
        Some(layer_idx / FULL_ATTN_INTERVAL)
    } else {
        None
    }
}

pub fn gdn_layer_index(layer_idx: usize) -> Option<usize> {
    if layer_idx >= NUM_LAYERS || is_full_attention_layer(layer_idx) {
        return None;
    }
    Some(layer_idx - (layer_idx / FULL_ATTN_INTERVAL) - 1)
}

pub fn assert_matches_config(config: &kiln_core::config::ModelConfig) -> Result<()> {
    ensure!(config.hidden_size == HIDDEN, "Qwen3.5 hidden_size drift");
    ensure!(config.num_layers == NUM_LAYERS, "Qwen3.5 num_layers drift");
    ensure!(
        config.num_attention_heads == NUM_HEADS,
        "Qwen3.5 num_attention_heads drift"
    );
    ensure!(
        config.num_kv_heads == NUM_KV_HEADS,
        "Qwen3.5 num_kv_heads drift"
    );
    ensure!(config.head_dim == HEAD_DIM, "Qwen3.5 head_dim drift");
    ensure!(
        config.intermediate_size == MLP_HIDDEN,
        "Qwen3.5 intermediate_size drift"
    );
    ensure!(config.vocab_size == VOCAB, "Qwen3.5 vocab_size drift");
    ensure!(
        config.max_position_embeddings == MAX_POSITION_EMBEDDINGS,
        "Qwen3.5 max_position_embeddings drift"
    );
    ensure!(
        config.num_full_attention_layers == NUM_FULL_ATTN_LAYERS,
        "Qwen3.5 full-attention count drift"
    );
    ensure!(
        config.full_attention_interval == FULL_ATTN_INTERVAL,
        "Qwen3.5 full-attention interval drift"
    );
    ensure!(
        config.attn_output_gate == ATTN_OUTPUT_GATE,
        "Qwen3.5 output gate drift"
    );
    ensure!(
        config.linear_num_key_heads == GDN_NUM_KEY_HEADS,
        "Qwen3.5 GDN key heads drift"
    );
    ensure!(
        config.linear_key_head_dim == GDN_KEY_HEAD_DIM,
        "Qwen3.5 GDN key dim drift"
    );
    ensure!(
        config.linear_num_value_heads == GDN_NUM_VALUE_HEADS,
        "Qwen3.5 GDN value heads drift"
    );
    ensure!(
        config.linear_value_head_dim == GDN_VALUE_HEAD_DIM,
        "Qwen3.5 GDN value dim drift"
    );
    ensure!(
        config.linear_conv_kernel_dim == GDN_CONV_KERNEL_DIM,
        "Qwen3.5 GDN conv drift"
    );
    ensure!(
        (config.partial_rotary_factor - PARTIAL_ROTARY).abs() < f64::EPSILON,
        "Qwen3.5 partial rotary drift"
    );
    ensure!(
        (config.rope_theta - ROPE_THETA).abs() < f64::EPSILON,
        "Qwen3.5 rope theta drift"
    );
    ensure!(
        config.rotary_dim() == ROTARY_DIM,
        "Qwen3.5 rotary dim drift"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_match_model_config_source_of_truth() {
        let config = kiln_core::config::ModelConfig::qwen3_5_4b();
        assert_matches_config(&config).unwrap();
    }

    #[test]
    fn attention_layer_indexing_matches_24_8_split() {
        let full: Vec<_> = (0..NUM_LAYERS)
            .filter(|&idx| is_full_attention_layer(idx))
            .collect();
        assert_eq!(full, vec![0, 4, 8, 12, 16, 20, 24, 28]);
        assert_eq!(full.len(), NUM_FULL_ATTN_LAYERS);
        assert_eq!(
            (0..NUM_LAYERS).filter_map(gdn_layer_index).count(),
            NUM_GDN_LAYERS
        );
        assert_eq!(full_attention_layer_index(20), Some(5));
        assert_eq!(gdn_layer_index(1), Some(0));
        assert_eq!(gdn_layer_index(31), Some(23));
    }
}
