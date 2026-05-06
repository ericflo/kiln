//! Explicit decode weight registry for Qwen3.5-4B.
//!
//! This removes the hot-path dependency on stringly tensor lookups. Weight
//! loading can continue to use Candle/safetensors names, but decode code indexes
//! a compile-checked `(layer_idx, layer_kind, projection)` key and receives the
//! already-uploaded BF16 transpose or W4A16 Marlin buffers it needs for kernel
//! dispatch.

use anyhow::{Context, Result, bail, ensure};
use candle_core::{DType, Tensor};
use std::collections::BTreeMap;

use crate::forward::{GpuAttentionWeights, GpuWeights};
use crate::qwen35_shapes as shapes;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum RegistryLayer {
    Embedding,
    FinalNorm,
    FullAttention { layer_idx: usize },
    LinearAttention { layer_idx: usize },
    Mlp { layer_idx: usize },
}

impl RegistryLayer {
    pub fn validate(self) -> Result<()> {
        match self {
            Self::Embedding | Self::FinalNorm => Ok(()),
            Self::FullAttention { layer_idx } => {
                ensure!(layer_idx < shapes::NUM_LAYERS, "full-attention layer index {layer_idx} out of range");
                ensure!(shapes::is_full_attention_layer(layer_idx), "layer {layer_idx} is not a Qwen3.5 full-attention layer");
                Ok(())
            }
            Self::LinearAttention { layer_idx } => {
                ensure!(layer_idx < shapes::NUM_LAYERS, "linear-attention layer index {layer_idx} out of range");
                ensure!(!shapes::is_full_attention_layer(layer_idx), "layer {layer_idx} is not a Qwen3.5 GDN layer");
                Ok(())
            }
            Self::Mlp { layer_idx } => {
                ensure!(layer_idx < shapes::NUM_LAYERS, "MLP layer index {layer_idx} out of range");
                Ok(())
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProjectionKind {
    EmbedTokens,
    LmHeadTied,
    FinalNorm,
    InputLayerNorm,
    PostAttentionLayerNorm,
    FullQ,
    FullK,
    FullV,
    FullO,
    FullQNorm,
    FullKNorm,
    GdnInProjQkv,
    GdnInProjZ,
    GdnOutProj,
    GdnInProjA,
    GdnInProjB,
    GdnConv1d,
    GdnNorm,
    GdnALog,
    GdnALogGates,
    GdnDtBias,
    MlpGate,
    MlpUp,
    MlpDown,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct PackedWeightKey {
    pub layer: RegistryLayer,
    pub projection: ProjectionKind,
}

impl PackedWeightKey {
    pub fn new(layer: RegistryLayer, projection: ProjectionKind) -> Result<Self> {
        layer.validate()?;
        Ok(Self { layer, projection })
    }
}

#[derive(Clone, Debug)]
pub enum PackedWeightStorage {
    Bf16 { tensor: Tensor, dims: Vec<usize>, transposed: bool },
    F32 { tensor: Tensor, dims: Vec<usize> },
    MarlinW4A16 { packed: crate::marlin_proj::MarlinPackedProj, k: usize, n: usize },
}

impl PackedWeightStorage {
    pub fn dims(&self) -> &[usize] {
        match self {
            Self::Bf16 { dims, .. } | Self::F32 { dims, .. } => dims,
            Self::MarlinW4A16 { .. } => &[],
        }
    }

    pub fn dtype(&self) -> Option<DType> {
        match self {
            Self::Bf16 { tensor, .. } | Self::F32 { tensor, .. } => Some(tensor.dtype()),
            Self::MarlinW4A16 { .. } => None,
        }
    }

    pub fn is_transposed_bf16(&self) -> bool {
        matches!(self, Self::Bf16 { transposed: true, .. })
    }

    #[cfg(feature = "cuda")]
    pub fn with_bf16_device_ptr<R>(&self, f: impl FnOnce(*const core::ffi::c_void) -> R) -> Result<R> {
        use half::bf16;

        let tensor = match self {
            Self::Bf16 { tensor, .. } => tensor,
            Self::F32 { .. } => bail!("requested BF16 pointer from FP32 registry weight"),
            Self::MarlinW4A16 { .. } => bail!("requested BF16 pointer from Marlin registry weight"),
        };
        ensure!(tensor.dtype() == DType::BF16, "registry BF16 pointer requires BF16 tensor, got {:?}", tensor.dtype());
        ensure!(tensor.is_contiguous(), "registry BF16 pointer requires contiguous tensor");
        let (storage, layout) = tensor.storage_and_layout();
        let cuda = match &*storage {
            candle_core::Storage::Cuda(cuda) => cuda,
            _ => bail!("registry BF16 pointer requires CUDA storage"),
        };
        let stream = cuda.device().cuda_stream();
        let slice = cuda.as_cuda_slice::<bf16>()?.slice(layout.start_offset()..);
        unsafe {
            let (ptr, _guard) = slice.device_ptr(&stream);
            Ok(f(ptr as *const core::ffi::c_void))
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct GpuPackedWeightRegistry {
    entries: BTreeMap<PackedWeightKey, PackedWeightStorage>,
}

impl GpuPackedWeightRegistry {
    pub fn new() -> Self {
        Self { entries: BTreeMap::new() }
    }

    pub fn insert(&mut self, key: PackedWeightKey, storage: PackedWeightStorage) -> Result<()> {
        key.layer.validate()?;
        if self.entries.insert(key, storage).is_some() {
            bail!("duplicate packed-weight registry key: {key:?}");
        }
        Ok(())
    }

    pub fn get(&self, key: PackedWeightKey) -> Option<&PackedWeightStorage> {
        self.entries.get(&key)
    }

    pub fn require(&self, key: PackedWeightKey) -> Result<&PackedWeightStorage> {
        self.get(key).with_context(|| format!("missing packed-weight registry key: {key:?}"))
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (&PackedWeightKey, &PackedWeightStorage)> {
        self.entries.iter()
    }

    pub fn from_gpu_weights(weights: &GpuWeights) -> Result<Self> {
        let mut registry = Self::new();
        registry.insert_bf16(RegistryLayer::Embedding, ProjectionKind::EmbedTokens, &weights.embed_tokens, false)?;
        registry.insert_bf16(RegistryLayer::Embedding, ProjectionKind::LmHeadTied, &weights.embed_tokens_t, true)?;
        registry.insert_bf16(RegistryLayer::FinalNorm, ProjectionKind::FinalNorm, &weights.final_norm, false)?;

        for (layer_idx, layer) in weights.layers.iter().enumerate() {
            registry.insert_bf16(RegistryLayer::Mlp { layer_idx }, ProjectionKind::InputLayerNorm, &layer.input_layernorm, false)?;
            registry.insert_bf16(RegistryLayer::Mlp { layer_idx }, ProjectionKind::PostAttentionLayerNorm, &layer.post_attention_layernorm, false)?;
            match &layer.attention {
                GpuAttentionWeights::Full(full) => {
                    let layer_key = RegistryLayer::FullAttention { layer_idx };
                    registry.insert_projection_or_marlin(layer_key, ProjectionKind::FullQ, &full.q_proj_t, full.q_proj_marlin.clone())?;
                    registry.insert_bf16(layer_key, ProjectionKind::FullK, &full.k_proj_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::FullV, &full.v_proj_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::FullO, &full.o_proj_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::FullQNorm, &full.q_norm, false)?;
                    registry.insert_bf16(layer_key, ProjectionKind::FullKNorm, &full.k_norm, false)?;
                }
                GpuAttentionWeights::Linear(linear) => {
                    let layer_key = RegistryLayer::LinearAttention { layer_idx };
                    registry.insert_bf16(layer_key, ProjectionKind::GdnInProjQkv, &linear.in_proj_qkv_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnInProjZ, &linear.in_proj_z_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnOutProj, &linear.out_proj_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnInProjA, &linear.in_proj_a_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnInProjB, &linear.in_proj_b_t, true)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnConv1d, &linear.conv1d, false)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnNorm, &linear.norm, false)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnALog, &linear.a_log, false)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnALogGates, &linear.a_log_gates, false)?;
                    registry.insert_bf16(layer_key, ProjectionKind::GdnDtBias, &linear.dt_bias, false)?;
                }
            }

            let mlp_key = RegistryLayer::Mlp { layer_idx };
            registry.insert_projection_or_marlin(mlp_key, ProjectionKind::MlpGate, &layer.mlp.gate_proj_t, layer.mlp.gate_proj_marlin.clone())?;
            registry.insert_projection_or_marlin(mlp_key, ProjectionKind::MlpUp, &layer.mlp.up_proj_t, layer.mlp.up_proj_marlin.clone())?;
            registry.insert_projection_or_marlin(mlp_key, ProjectionKind::MlpDown, &layer.mlp.down_proj_t, layer.mlp.down_proj_marlin.clone())?;
        }
        Ok(registry)
    }

    fn insert_bf16(&mut self, layer: RegistryLayer, projection: ProjectionKind, tensor: &Tensor, transposed: bool) -> Result<()> {
        let key = PackedWeightKey::new(layer, projection)?;
        self.insert(key, PackedWeightStorage::Bf16 { tensor: tensor.clone(), dims: tensor.dims().to_vec(), transposed })
    }

    fn insert_projection_or_marlin(
        &mut self,
        layer: RegistryLayer,
        projection: ProjectionKind,
        bf16_t: &Tensor,
        marlin: Option<crate::marlin_proj::MarlinPackedProj>,
    ) -> Result<()> {
        let key = PackedWeightKey::new(layer, projection)?;
        let storage = if let Some(packed) = marlin {
            PackedWeightStorage::MarlinW4A16 { k: packed.k, n: packed.n, packed }
        } else {
            PackedWeightStorage::Bf16 { tensor: bf16_t.clone(), dims: bf16_t.dims().to_vec(), transposed: true }
        };
        self.insert(key, storage)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};

    #[test]
    fn validates_layer_kind_against_qwen35_pattern() {
        assert!(RegistryLayer::FullAttention { layer_idx: 0 }.validate().is_ok());
        assert!(RegistryLayer::FullAttention { layer_idx: 1 }.validate().is_err());
        assert!(RegistryLayer::LinearAttention { layer_idx: 1 }.validate().is_ok());
        assert!(RegistryLayer::LinearAttention { layer_idx: 4 }.validate().is_err());
        assert!(RegistryLayer::Mlp { layer_idx: shapes::NUM_LAYERS }.validate().is_err());
    }

    #[test]
    fn registry_rejects_duplicate_keys() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((2, 2), DType::BF16, &device).unwrap();
        let key = PackedWeightKey::new(RegistryLayer::Mlp { layer_idx: 0 }, ProjectionKind::MlpGate).unwrap();
        let storage = PackedWeightStorage::Bf16 { tensor, dims: vec![2, 2], transposed: true };
        let mut registry = GpuPackedWeightRegistry::new();
        registry.insert(key, storage.clone()).unwrap();
        assert!(registry.insert(key, storage).is_err());
    }

    #[test]
    fn registry_returns_compile_checked_keys() {
        let device = Device::Cpu;
        let tensor = Tensor::zeros((shapes::HIDDEN, shapes::MLP_HIDDEN), DType::BF16, &device).unwrap();
        let key = PackedWeightKey::new(RegistryLayer::Mlp { layer_idx: 7 }, ProjectionKind::MlpUp).unwrap();
        let mut registry = GpuPackedWeightRegistry::new();
        registry.insert(key, PackedWeightStorage::Bf16 { tensor, dims: vec![shapes::HIDDEN, shapes::MLP_HIDDEN], transposed: true }).unwrap();
        let storage = registry.require(key).unwrap();
        assert_eq!(storage.dims(), &[shapes::HIDDEN, shapes::MLP_HIDDEN]);
        assert!(storage.is_transposed_bf16());
    }
}
