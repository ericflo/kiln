//! Metal shared-library and compute-pipeline cache helpers.
//!
//! This module owns Metal shader compilation and per-device pipeline state
//! caches. Operation modules request named pipelines through these helpers;
//! command encoding stays with the operation-family modules.

use anyhow::Result;

use super::metal_config::MetalTransposedCoopGemvTile;
use super::metal_core::MetalPipelineHost;
use super::metal_msl::*;
use kiln_tensor::metal_types::{ComputePipeline, Library};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

pub(super) fn metal_shared_library(device: &dyn MetalPipelineHost) -> Result<Library> {
    static LIBRARIES: OnceLock<Mutex<HashMap<u64, Library>>> = OnceLock::new();
    let cache = LIBRARIES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal shared library cache poisoned"))?;
    if let Some(library) = cache.get(&device.pipeline_cache_key()) {
        return Ok(library.clone());
    }

    let shared_source = [
        METAL_RMSNORM_KERNEL,
        METAL_ROTARY_QK_KERNEL,
        METAL_GDN_QK_NORM_KERNEL,
        METAL_GDN_DECODE_QKV_CONV_NORM_KERNEL,
        METAL_GDN_GATES_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_KERNEL,
        METAL_GDN_DECODE_GATES_RECURRENT_RMSNORM_KERNEL,
        METAL_GATED_RMSNORM_KERNEL,
        METAL_GDN_RECURRENT_KERNEL,
        METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_KERNEL,
        METAL_GDN_RECURRENT_PREFILL_HEAD_LAST_DECAY_KERNEL,
        METAL_GDN_FULL_CHUNK_FORWARD_KERNEL,
        METAL_CONV1D_PREFILL_KERNEL,
        METAL_CONV1D_UPDATE_KERNEL,
        METAL_LM_HEAD_KERNEL,
        METAL_MLP_GATE_UP_KERNEL,
        METAL_ATTN_GATE_SIGMOID_MUL_KERNEL,
        METAL_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_FUSED_QKV_TRANSPOSED_COOP_GEMV_KERNEL,
        METAL_LORA_DELTA_DECODE_KERNEL,
        METAL_GDN_IN_PROJ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_KERNEL,
        METAL_PAGED_KV_HEAD_MAJOR_READ_APPEND_TOKEN_MAJOR_KERNEL,
        METAL_PAGED_ATTN_DECODE_CONTIGUOUS_KERNEL,
        METAL_PAGED_KV_WRITE_TOKEN_MAJOR_KERNEL,
    ]
    .join("");
    let library = device
        .pipeline_raw_device()
        .new_library_with_source(&shared_source, None)
        .map_err(|e| anyhow::anyhow!("compile metal shared library: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), library.clone());
    Ok(library)
}

pub(super) fn metal_rms_norm_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rmsnorm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_rotary_qk_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal rotary qk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_rotary_qk_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal rotary qk function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal rotary qk pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_qk_norm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_qk_norm_gqa_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn qk norm gqa pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_qk_norm_gqa_f32_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn qk norm gqa function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn qk norm gqa pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_decode_qkv_conv_norm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn decode qkv conv/norm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_qkv_conv_norm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode qkv conv/norm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode qkv conv/norm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_argmax_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_chunks_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_argmax_reduce_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax reduce pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_reduce_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax reduce function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax reduce pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_argmax_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head argmax batch pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_chunks_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax batch function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax batch pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_argmax_reduce_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal lm head argmax reduce batch pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_argmax_reduce_batch_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head argmax reduce batch function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head argmax reduce batch pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_sample_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head sample pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_sample_topk_chunks_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head sample function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head sample pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lm_head_sample_reduce_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal lm head sample reduce pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lm_head_sample_reduce_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal lm head sample reduce function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal lm head sample reduce pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_mlp_gate_up_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp gate/up pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_gate_up_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp gate/up function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp gate/up pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_mlp_gate_up_serial_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp gate/up serial pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_gate_up_serial_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp gate/up serial function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp gate/up serial pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_mlp_silu_mul_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal mlp silu*mul pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_mlp_silu_mul_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal mlp silu*mul function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal mlp silu*mul pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_attn_gate_sigmoid_mul_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal attn gate sigmoid/mul pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_attn_gate_sigmoid_mul_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal attn gate sigmoid/mul function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal attn gate sigmoid/mul pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_transposed_coop_gemv_pipeline(
    device: &dyn MetalPipelineHost,
    tile: MetalTransposedCoopGemvTile,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<
        Mutex<HashMap<(u64, MetalTransposedCoopGemvTile), ComputePipeline>>,
    > = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal transposed coop GEMV pipeline cache poisoned"))?;
    let key = (device.pipeline_cache_key(), tile);
    if let Some(pipeline) = cache.get(&key) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(tile.function_name(), None)
        .map_err(|e| anyhow::anyhow!("load metal transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(key, pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_transposed_coop_gemv_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal batch transposed coop GEMV cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_transposed_coop_gemv8_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal batch transposed coop GEMV function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal batch transposed coop GEMV pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_transposed_coop_gemv_batch_row_triple_tile8_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal batch transposed coop GEMV row-triple tile8 cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_transposed_coop_gemv8_batch_row_triple_tile8_bf16",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!(
                "load metal batch transposed coop GEMV row-triple tile8 function: {e:?}"
            )
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!(
                "build metal batch transposed coop GEMV row-triple tile8 pipeline: {e:?}"
            )
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_transposed_coop_gemv_batch_row_quad_tile8_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal batch transposed coop GEMV row-quad tile8 cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_transposed_coop_gemv8_batch_row_quad_tile8_bf16", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal batch transposed coop GEMV row-quad tile8 function: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal batch transposed coop GEMV row-quad tile8 pipeline: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_fused_qkv_transposed_coop_gemv_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal fused QKV projection pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_fused_qkv_transposed_coop_gemv8_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal fused QKV projection function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal fused QKV projection pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lora_hidden_decode_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal LoRA hidden decode pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lora_hidden_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal LoRA hidden decode function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal LoRA hidden decode pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_lora_add_decode_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal LoRA add decode pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_lora_add_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal LoRA add decode function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal LoRA add decode pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_in_proj_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn in-proj pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_in_proj_decode_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn in-proj function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn in-proj pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_head_major_read_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv read pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_head_major_read_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv read function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_head_major_read_append_token_major_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv read+append pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_kv_head_major_read_append_token_major_bf16",
            None,
        )
        .map_err(|e| anyhow::anyhow!("load metal paged kv read+append function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv read+append pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_attn_decode_contiguous_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal contiguous paged decode attention cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_bf16_d256", None)
        .map_err(|e| anyhow::anyhow!("load metal contiguous paged decode attention: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal contiguous paged decode attention: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_attn_decode_contiguous_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal contiguous paged batch decode pipeline poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_attn_decode_contiguous_batch_bf16_d256", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal contiguous paged batch decode attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal contiguous paged batch decode attention: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal dyn-seqlen paged batch decode pipeline poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!("load metal dyn-seqlen paged batch decode attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal dyn-seqlen paged batch decode attention: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_attn_decode_contiguous_batch_dyn_seqlen_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal dyn-seqlen paged batch decode ICB pipeline poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function(
            "kiln_paged_attn_decode_contiguous_batch_dyn_seqlen_bf16_d256",
            None,
        )
        .map_err(|e| {
            anyhow::anyhow!("load metal dyn-seqlen paged batch decode ICB attention: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal dyn-seqlen paged batch decode ICB attention: {e:?}")
        })?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal dyn-seqlen paged batch decode ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_write_token_major_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv write function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv write pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_write_token_major_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv write ICB pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv write ICB function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv write ICB pipeline: {e:?}"))?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal paged kv write ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_write_token_major_batch_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv batch write pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv batch write function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv batch write pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_paged_kv_write_token_major_batch_pipeline_indirect(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal paged kv batch write ICB pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_paged_kv_write_token_major_batch_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal paged kv batch write ICB function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function_for_indirect_commands(&function)
        .map_err(|e| anyhow::anyhow!("build metal paged kv batch write ICB pipeline: {e:?}"))?;
    anyhow::ensure!(
        pipeline.supports_indirect_command_buffers(),
        "metal paged kv batch write ICB pipeline did not enable indirect-command support"
    );
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_gates_pipeline(device: &dyn MetalPipelineHost) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_gates_decay_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates decay pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_decay_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates decay function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates decay pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_gates_decay_ab_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn_gates decay A/B pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_gates_decay_ab_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn_gates decay A/B function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn_gates decay A/B pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_decode_gates_recurrent_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn decode gates+recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn decode gates+recurrent function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn decode gates+recurrent pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_decode_gates_recurrent_rmsnorm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn decode gates+recurrent+rmsnorm pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_decode_gates_recurrent_rmsnorm_bf16", None)
        .map_err(|e| {
            anyhow::anyhow!("load metal gdn decode gates+recurrent+rmsnorm function: {e:?}")
        })?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| {
            anyhow::anyhow!("build metal gdn decode gates+recurrent+rmsnorm pipeline: {e:?}")
        })?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gated_rms_norm_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gated rmsnorm pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gated_rmsnorm_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gated rmsnorm function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gated rmsnorm pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_recurrent_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_recurrent_prefill_head_last_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn recurrent prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_prefill_head_last_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent prefill function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent prefill pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_forward_substitution_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn forward-substitution pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_forward_substitution_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn forward-substitution function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn forward-substitution pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_forward_substitution_f32_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn forward-substitution f32 pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_forward_substitution_f32", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn forward-substitution f32 function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn forward-substitution f32 pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_chunk_prep_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn chunk-prep pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_chunk_prep_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn chunk-prep function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn chunk-prep pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_recurrent_prefill_head_last_decay_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache.lock().map_err(|_| {
        anyhow::anyhow!("metal gdn recurrent prefill decay pipeline cache poisoned")
    })?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_recurrent_prefill_head_last_decay_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn recurrent prefill decay function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn recurrent prefill decay pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_full_chunk_forward_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn full-chunk pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_full_chunk_forward_bf16", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn full-chunk function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn full-chunk pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_conv1d_prefill_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal conv1d prefill pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_causal_conv1d_prefill_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d prefill function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d prefill pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_gdn_prefill_qkv_conv_split_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal gdn prefill qkv conv-split pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_gdn_prefill_qkv_conv_split_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal gdn prefill qkv conv-split function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal gdn prefill qkv conv-split pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}

pub(super) fn metal_conv1d_update_pipeline(
    device: &dyn MetalPipelineHost,
) -> Result<ComputePipeline> {
    static PIPELINES: OnceLock<Mutex<HashMap<u64, ComputePipeline>>> = OnceLock::new();
    let cache = PIPELINES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut cache = cache
        .lock()
        .map_err(|_| anyhow::anyhow!("metal conv1d update pipeline cache poisoned"))?;
    if let Some(pipeline) = cache.get(&device.pipeline_cache_key()) {
        return Ok(pipeline.clone());
    }

    let library = metal_shared_library(device)?;
    let function = library
        .get_function("kiln_causal_conv1d_update_bf16_f32_k4", None)
        .map_err(|e| anyhow::anyhow!("load metal conv1d update function: {e:?}"))?;
    let pipeline = device
        .pipeline_raw_device()
        .new_compute_pipeline_state_with_function(&function)
        .map_err(|e| anyhow::anyhow!("build metal conv1d update pipeline: {e:?}"))?;
    cache.insert(device.pipeline_cache_key(), pipeline.clone());
    Ok(pipeline)
}
