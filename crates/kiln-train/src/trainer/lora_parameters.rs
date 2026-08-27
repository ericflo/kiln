use super::*;

/// Which linear projections to train LoRA on.
pub(super) const DEFAULT_TARGET_MODULES: &[&str] = crate::adapter_shape::TRAINABLE_TARGET_MODULES;
pub(super) const ADAPTER_SMOKE_TEST_PROMPTS: &[&str] = &[
    "In one short sentence, name a primary color:",
    "Complete this sentence with a brief answer: The capital of France is",
    "Return a compact JSON tool call for a weather lookup in Paris:",
];
pub(super) const ADAPTER_SMOKE_TEST_MAX_NEW_TOKENS: usize = 4;

pub(super) struct AdapterSmokeGeneration {
    pub(super) output: String,
    pub(super) output_tokens: usize,
    pub(super) elapsed_ms: u64,
}

/// Trainable LoRA parameters as kt-native `kiln_param::Parameter`s (#1082).
///
/// Each `Parameter` holds a BF16 master kt tensor and a stable kt
/// `TensorId`. The trainer threads each param's primary kt tensor into
/// the kt tape forward (via [`Self::as_lora_weights`] →
/// `LoraProjectionWeights`); the tape backward then yields a grad keyed
/// by `Parameter::tensor_id()`, which the kt optimizer
/// (`kiln_optim::AdamW`) consumes directly. NO candle autograd `Var`,
/// NO `loss.backward()`.
pub struct TrainableLoraParams {
    /// Per-layer, per-module (A, B) parameter pairs.
    /// Indexed as: `layers[layer_idx].module_name` -> (Param_A, Param_B)
    pub layers: Vec<TrainableLoraLayerParams>,
    /// LoRA pairs for the native MTP draft block (MTP training plan
    /// PR-B). `None` unless the post-SFT MTP alignment phase initialized
    /// them. Deliberately EXCLUDED from `all_params`/`all_params_mut` —
    /// the main training phase must not see parameters that its forward
    /// graph never touches; the alignment phase drives these through
    /// [`Self::mtp_params`]/[`Self::mtp_params_mut`].
    pub mtp: Option<TrainableLoraLayerParams>,
    pub rank: usize,
    pub alpha: f32,
    pub scale: f32,
}

/// Trainable LoRA A/B pairs for one transformer layer.
#[derive(Default)]
pub struct TrainableLoraLayerParams {
    pub q_proj: Option<(Parameter, Parameter)>,
    pub k_proj: Option<(Parameter, Parameter)>,
    pub v_proj: Option<(Parameter, Parameter)>,
    pub o_proj: Option<(Parameter, Parameter)>,
    pub in_proj_qkv: Option<(Parameter, Parameter)>,
    pub in_proj_z: Option<(Parameter, Parameter)>,
    pub gdn_out_proj: Option<(Parameter, Parameter)>,
    pub gate_proj: Option<(Parameter, Parameter)>,
    pub up_proj: Option<(Parameter, Parameter)>,
    pub down_proj: Option<(Parameter, Parameter)>,
}

pub(super) struct LoraParamRef<'a> {
    pub(super) layer_idx: usize,
    pub(super) module: &'static str,
    pub(super) matrix: &'static str,
    pub(super) param: &'a Parameter,
}

pub(super) fn push_lora_param_pair<'a>(
    params: &mut Vec<LoraParamRef<'a>>,
    layer_idx: usize,
    module: &'static str,
    pair: &'a Option<(Parameter, Parameter)>,
) {
    if let Some((a, b)) = pair {
        params.push(LoraParamRef {
            layer_idx,
            module,
            matrix: "A",
            param: a,
        });
        params.push(LoraParamRef {
            layer_idx,
            module,
            matrix: "B",
            param: b,
        });
    }
}

impl TrainableLoraParams {
    /// Initialize fresh LoRA parameters with Kaiming-uniform A and zero B.
    ///
    /// This matches the standard LoRA initialization:
    /// - A: Kaiming uniform (so the product A*B starts near zero)
    /// - B: zeros (so initial LoRA contribution is zero)
    ///
    /// Equivalent to `initialize_seeded(.., None)` — the device-global RNG
    /// drives A initialization. Tests, benches, and any caller that does not
    /// need byte-for-byte reproducibility should use this entry point.
    pub fn initialize(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
    ) -> Result<Self> {
        Self::initialize_seeded(config, weights, rank, alpha, device, None)
    }

    /// Like [`Self::initialize`], but uses a deterministic RNG seeded with `seed`
    /// to draw A. Used by the SFT/GRPO training loops so an adapter
    /// initialized with the same seed against the same base weights produces
    /// byte-identical LoRA-A tensors on every run, even on the CPU device,
    /// where kt `Device` has no `set_seed` (candle's was a no-op there anyway).
    ///
    /// `seed: None` falls back to the device-global RNG (preserves the
    /// pre-replay behavior).
    pub fn initialize_seeded(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<Self> {
        Self::initialize_seeded_with_precision_policy(
            config,
            weights,
            rank,
            alpha,
            device,
            seed,
            training_precision_policy_for_device(device),
        )
    }

    pub fn initialize_seeded_with_precision_policy(
        config: &ModelConfig,
        weights: &GpuWeights,
        rank: usize,
        alpha: f32,
        device: &Device,
        seed: Option<u64>,
        precision_policy: TrainingPrecisionPolicy,
    ) -> Result<Self> {
        // (#1082) kt `Device` has no `set_seed` (candle's was a no-op on CPU
        // anyway); the seeded `StdRng` below is what actually delivers
        // byte-for-byte determinism for LoRA-A. LoRA-B is plain zeros and
        // never touches a device RNG, so nothing else here needs seeding.
        let mut rng = seed.map(StdRng::seed_from_u64);

        let scale = alpha / rank as f32;
        let num_layers = config.num_layers;
        let hidden = config.hidden_size;
        let intermediate = config.intermediate_size;

        // (#1082) LoRA-param dtype follows the backend-owned training precision
        // policy. CUDA/ROCm/Metal track the base dtype, while Vulkan keeps LoRA
        // parameters F32 to match its F32 activation policy.
        //
        // (#1443 step 2) On Vulkan the ACTIVATION dtype is F32 regardless of the
        // base WEIGHT dtype — the mixed-precision design keeps base projection
        // weights BF16 (the VRAM win) but runs F32 activations through the
        // F32-only Vulkan rmsnorm/softmax kernels, and the embedding output is
        // cast BF16→F32 at the head of the forward. So on Vulkan LoRA A/B are
        // ALWAYS F32 (matching the F32 activations / LoRA delta path) even on a
        // BF16 base; otherwise a BF16 LoRA on a BF16 base would mismatch the F32
        // `x2d` in `try_tape_lora_linear_kt`'s LoRA branch and decline. CUDA/Metal
        // keep `embed_tokens.dtype()` (BF16 activations end-to-end) unchanged.
        let lora_dtype =
            precision_policy.lora_parameter_dtype_for_base_weight(weights.embed_tokens.dtype());

        // Kaiming uniform bound: sqrt(1 / in_features) for A
        let bound_hidden = (1.0 / hidden as f64).sqrt();
        let bound_intermediate = (1.0 / intermediate as f64).sqrt();

        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            let mut layer_params = TrainableLoraLayerParams::default();

            // Determine actual dimensions from the weight tensors
            let layer_weights = &weights.layers[layer_idx];

            for &module in DEFAULT_TARGET_MODULES {
                let (in_features, out_features, bound) = match module {
                    "q_proj" | "k_proj" | "v_proj" | "o_proj" => {
                        // Read the transposed weight's shape. Post-Phase 4.x
                        // residency, this tensor is a `broadcast_as` view
                        // that preserves the original `[hidden, out_dim]`
                        // dims while sharing 2 bytes of storage — so
                        // `.dims()` still returns the right shape and
                        // we don't have to mirror Qwen3.5-specific quirks
                        // (e.g. attn_output_gate doubling q_proj out_dim)
                        // here.
                        let w_t = match &layer_weights.attention {
                            kiln_model::forward::GpuAttentionWeights::Full(full) => match module {
                                "q_proj" => &full.q_proj_t,
                                "k_proj" => &full.k_proj_t,
                                "v_proj" => &full.v_proj_t,
                                "o_proj" => &full.o_proj_t,
                                _ => unreachable!(),
                            },
                            // Linear attention layers don't have q/k/v/o_proj
                            kiln_model::forward::GpuAttentionWeights::Linear(_) => {
                                continue;
                            }
                        };
                        let dims = w_t.dims();
                        anyhow::ensure!(
                            dims.len() == 2,
                            "expected rank-2 {module}_t for layer {layer_idx}, got {:?}",
                            dims
                        );
                        // Transposed weight is [in_features, out_features].
                        (dims[0], dims[1], bound_hidden)
                    }
                    "in_proj_qkv" | "in_proj_z" | "out_proj" => {
                        let w_t = match &layer_weights.attention {
                            kiln_model::forward::GpuAttentionWeights::Linear(linear) => {
                                match module {
                                    "in_proj_qkv" => &linear.in_proj_qkv_t,
                                    "in_proj_z" => &linear.in_proj_z_t,
                                    "out_proj" => &linear.out_proj_t,
                                    _ => unreachable!(),
                                }
                            }
                            // Full-attention layers use o_proj; these names are
                            // reserved for GDN/LinearAttention PEFT adapters.
                            kiln_model::forward::GpuAttentionWeights::Full(_) => {
                                continue;
                            }
                        };
                        let dims = w_t.dims();
                        anyhow::ensure!(
                            dims.len() == 2,
                            "expected rank-2 {module}_t for layer {layer_idx}, got {:?}",
                            dims
                        );
                        let in_features = dims[0];
                        let out_features = dims[1];
                        let bound = (1.0 / in_features as f64).sqrt();
                        (in_features, out_features, bound)
                    }
                    "gate_proj" => (hidden, intermediate, bound_hidden),
                    "up_proj" => (hidden, intermediate, bound_hidden),
                    "down_proj" => (intermediate, hidden, bound_intermediate),
                    _ => continue,
                };

                // A: [rank, in_features] — Kaiming uniform
                // Phase 10: BF16 storage + FP32-accumulate via tensor cores (audit
                // docs/audits/PHASE10_LORA_PRECISION_STUDY.md §5). (#1082) The
                // dtype now follows the base (`lora_dtype`): BF16 base ⇒ BF16
                // (unchanged); F32 base (Vulkan-only) ⇒ F32 so the tape recorder
                // matches the F32 activations.
                let a =
                    kaiming_uniform_a(rng.as_mut(), bound, (rank, in_features), lora_dtype, device)
                        .with_context(|| format!("init LoRA A for layer {layer_idx} {module}"))?;

                // B: [out_features, rank] — zeros
                let b = lora_param_zeros((out_features, rank), lora_dtype, device)
                    .with_context(|| format!("init LoRA B for layer {layer_idx} {module}"))?;

                match module {
                    "q_proj" => layer_params.q_proj = Some((a, b)),
                    "k_proj" => layer_params.k_proj = Some((a, b)),
                    "v_proj" => layer_params.v_proj = Some((a, b)),
                    "o_proj" => layer_params.o_proj = Some((a, b)),
                    "in_proj_qkv" => layer_params.in_proj_qkv = Some((a, b)),
                    "in_proj_z" => layer_params.in_proj_z = Some((a, b)),
                    "out_proj" => layer_params.gdn_out_proj = Some((a, b)),
                    "gate_proj" => layer_params.gate_proj = Some((a, b)),
                    "up_proj" => layer_params.up_proj = Some((a, b)),
                    "down_proj" => layer_params.down_proj = Some((a, b)),
                    _ => {}
                }
            }

            layers.push(layer_params);
        }

        Ok(Self {
            layers,
            mtp: None,
            rank,
            alpha,
            scale,
        })
    }

    /// Initialize LoRA A/B pairs for the native MTP draft block's seven
    /// modules (q/k/v/o + gate/up/down), shaped from the checkpoint's
    /// actual `mtp.*` tensors. Returns `Ok(false)` (no-op) when the
    /// checkpoint ships no MTP tensors. Same Kaiming-A / zero-B init and
    /// precision policy as the main layers. (MTP training plan PR-B.)
    pub fn initialize_mtp_seeded(
        &mut self,
        weights: &GpuWeights,
        device: &Device,
        seed: Option<u64>,
    ) -> Result<bool> {
        if weights.mtp.is_none() {
            return Ok(false);
        }
        let mtp = weights
            .mtp_weights()
            .context("initialize_mtp_seeded: materializing mtp.* tensors")?;
        let full = match &mtp.layer.attention {
            kiln_model::forward::GpuAttentionWeights::Full(full) => full,
            kiln_model::forward::GpuAttentionWeights::Linear(_) => {
                anyhow::bail!(
                    "initialize_mtp_seeded: MTP layer is linear-attention — the loader \
                     guarantees full attention; checkpoint is malformed"
                )
            }
        };
        let mut rng = seed.map(StdRng::seed_from_u64);
        let lora_dtype = training_precision_policy_for_device(device)
            .lora_parameter_dtype_for_base_weight(weights.embed_tokens.dtype());
        let rank = self.rank;

        let mut pairs = TrainableLoraLayerParams::default();
        let mut make_pair = |w_t: &kiln_tensor::Tensor,
                             module: &str|
         -> Result<(Parameter, Parameter)> {
            let dims = w_t.dims();
            anyhow::ensure!(
                dims.len() == 2,
                "initialize_mtp_seeded: expected rank-2 {module}_t, got {dims:?}"
            );
            let (in_features, out_features) = (dims[0], dims[1]);
            let bound = (1.0 / in_features as f64).sqrt();
            let a = kaiming_uniform_a(rng.as_mut(), bound, (rank, in_features), lora_dtype, device)
                .with_context(|| format!("init MTP LoRA A for {module}"))?;
            let b = lora_param_zeros((out_features, rank), lora_dtype, device)
                .with_context(|| format!("init MTP LoRA B for {module}"))?;
            Ok((a, b))
        };
        pairs.q_proj = Some(make_pair(&full.q_proj_t, "q_proj")?);
        pairs.k_proj = Some(make_pair(&full.k_proj_t, "k_proj")?);
        pairs.v_proj = Some(make_pair(&full.v_proj_t, "v_proj")?);
        pairs.o_proj = Some(make_pair(&full.o_proj_t, "o_proj")?);
        pairs.gate_proj = Some(make_pair(&mtp.layer.mlp.gate_proj_t, "gate_proj")?);
        pairs.up_proj = Some(make_pair(&mtp.layer.mlp.up_proj_t, "up_proj")?);
        pairs.down_proj = Some(make_pair(&mtp.layer.mlp.down_proj_t, "down_proj")?);
        self.mtp = Some(pairs);
        Ok(true)
    }

    /// MTP draft-block params for the alignment phase's grad lookup +
    /// optimizer (empty when [`Self::mtp`] is `None`).
    pub fn mtp_params(&self) -> Vec<&Parameter> {
        let mut out = Vec::new();
        if let Some(mtp) = &self.mtp {
            for pair in [
                &mtp.q_proj,
                &mtp.k_proj,
                &mtp.v_proj,
                &mtp.o_proj,
                &mtp.gate_proj,
                &mtp.up_proj,
                &mtp.down_proj,
            ]
            .into_iter()
            .flatten()
            {
                out.push(&pair.0);
                out.push(&pair.1);
            }
        }
        out
    }

    /// Mutable variant for the alignment phase's optimizer step.
    pub fn mtp_params_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out: Vec<&mut Parameter> = Vec::new();
        if let Some(mtp) = self.mtp.as_mut() {
            for pair in [
                &mut mtp.q_proj,
                &mut mtp.k_proj,
                &mut mtp.v_proj,
                &mut mtp.o_proj,
                &mut mtp.gate_proj,
                &mut mtp.up_proj,
                &mut mtp.down_proj,
            ] {
                if let Some((a, b)) = pair.as_mut() {
                    out.push(a);
                    out.push(b);
                }
            }
        }
        out
    }

    /// Phase 4.1: register every LoRA `Var` (A and B for all modules
    /// across all layers) in the backend's resident activation
    /// registry. After this call, the trainer's training-time forward
    /// path dispatches the LoRA delta on-device via
    /// `lora_delta_resident` (which wraps the dispatch in
    /// `VulkanLoraOp` — a CustomOp3 with analytic backward), and the
    /// trainer's `apply_sgd_update` prefers the on-device
    /// `dispatch_sgd_step` path that writes to the registry buffer
    /// in-place.
    ///
    /// Caller invokes this once after [`Self::initialize_seeded`], typically
    /// from `sft_train` / `grpo_train`. Test code that doesn't
    /// exercise the registry path skips this call — the trainer's
    /// existing fall-through logic handles the not-resident case
    /// transparently.
    ///
    /// Memory cost: one DMA upload per Var (~16 MB total for
    /// Qwen3.5-4B at rank=8 when GDN targets are present). On
    /// non-resident-supporting backends (CPU/Metal/CUDA today) the
    /// hook is a no-op.
    ///
    /// Lifecycle: each optimizer update keeps the registry buffer
    /// in sync with the kt param master (or vice versa, depending
    /// on whether the on-device or CPU optimizer path fired; see
    /// [`Self::sync_to_master`]). The
    /// matching [`Self::evict_from_backend`] runs at training
    /// completion to release registry entries before the trainer
    /// returns.
    pub fn register_with_backend(&self, backend: &dyn BackendRuntime) -> Result<()> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(());
        }
        for param in self.all_params() {
            ResidencyBackend::runtime_register_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            )?;
        }
        for param in self.mtp_params() {
            ResidencyBackend::runtime_register_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            )?;
        }
        Ok(())
    }

    /// Inverse of [`Self::register_with_backend`]: evict every LoRA param
    /// from the resident activation registry. Caller invokes this
    /// after the training loop completes (or per-step if Phase 4.1
    /// step 2 makes the registry the data-of-record and the trainer
    /// re-registers per step).
    pub fn evict_from_backend(&self, backend: &dyn BackendRuntime) {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return;
        }
        for param in self.all_params() {
            ResidencyBackend::runtime_evict_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            );
        }
        for param in self.mtp_params() {
            ResidencyBackend::runtime_evict_resident_activation(
                backend,
                param.forward_storage().primary_tensor(),
            );
        }
    }

    /// Pull every LoRA param's current value from the registry buffer
    /// back into its kt master storage.
    ///
    /// The on-device SGD and AdamW dispatch paths leave the kt master
    /// stale (the registry buffer is the source of truth between
    /// training steps). Callers that need the current master —
    /// `save_peft`, checkpoint writes — invoke this first. The refresh
    /// swaps the param's forward + backward storage to the resolved kt
    /// tensor while preserving `Parameter::tensor_id()` (anti-pattern 11).
    ///
    /// No-op on backends without resident-activation support. Returns
    /// the number of params synced for telemetry.
    pub fn sync_to_master(&mut self, backend: &dyn BackendRuntime) -> Result<usize> {
        if !ResidencyBackend::runtime_supports_resident_activation(backend) {
            return Ok(0);
        }
        let mut synced = 0;
        fn sync_one(
            backend: &dyn BackendRuntime,
            param: &mut Parameter,
            synced: &mut usize,
        ) -> Result<()> {
            let primary = param.forward_storage().primary_tensor().clone();
            if !ResidencyBackend::runtime_has_resident_activation(backend, &primary) {
                return Ok(());
            }
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            if let Some(resolved) = ResidencyBackend::runtime_resolve_resident_activation(
                backend, &primary, &dims, dtype,
            )? {
                let resolved = if resolved.device() == primary.device() {
                    resolved
                } else {
                    resolved
                        .to_device(primary.device())
                        .context("realign resolved LoRA parameter to its owner device")?
                };
                param
                    .replace_plain_trainable_tensor(resolved)
                    .map_err(|error| anyhow::anyhow!("sync LoRA parameter identity: {error}"))?;
                *synced += 1;
            }
            Ok(())
        }
        for param in self.all_params_mut() {
            sync_one(backend, param, &mut synced)?;
        }
        // The MTP draft-block pairs (alignment phase) sync too — save_peft
        // serializes them under the mtp.* keys.
        for param in self.mtp_params_mut() {
            sync_one(backend, param, &mut synced)?;
        }
        Ok(synced)
    }

    /// (#1082) Allocate AdamW optimizer state.
    ///
    /// CORRECTNESS (C1, candle-drop): the on-device CUDA AdamW kernel
    /// (`dispatch_adamw_step`) reads/writes the first/second-moment buffers
    /// **in place**. It therefore needs two *real* per-parameter device
    /// tensors `m`/`v` — distinct from the param. The candle-drop interim
    /// passed `&primary` twice in place of `m`/`v`, which aliased the moments
    /// onto the param (corrupting the weight and keeping NO Adam state). This
    /// restores the pre-flip design (`feaf2e99`'s `AdamWMoments{m,v}`) in
    /// kt-native form: a zero-init device `m`/`v` per LoRA param, matching the
    /// param master's shape/dtype/device, keyed by `Parameter::tensor_id()`.
    ///
    /// The CPU `kiln_optim::AdamW` instance is retained as the genuine
    /// non-resident host fallback (it owns its own host-side moments + grad
    /// dtype checks); the on-device path never touches it.
    ///
    /// `lr`/`beta1`/`beta2`/`eps`/`weight_decay` come from the trainer config
    /// (constant lr across steps — no scheduler). `device` is the param
    /// device; the moments are allocated on the *param's* device
    /// (`primary_tensor().device()`) so the CUDA gate
    /// (`cuda_optimizer_tensors_supported_for_kt`) sees four same-device
    /// same-dtype contiguous tensors.
    pub fn allocate_adamw_state(
        &self,
        lr: f64,
        beta1: f32,
        beta2: f32,
        eps: f32,
        weight_decay: f32,
        _device: &Device,
    ) -> Result<OptimizerState> {
        let hp = KtAdamWHyperparameters {
            lr: lr as f32,
            beta1,
            beta2,
            eps,
            weight_decay,
        };
        let mut moments: HashMap<KtTensorId, KtAdamWMoments> = HashMap::new();
        for param in self.all_params() {
            let primary = param.forward_storage().primary_tensor();
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            // Allocate on the param's own device (CUDA → on-device zeros via
            // `Tensor::zeros_on` → `cuda_zeros_ctx`, NOT zeros_cpu) so the
            // dispatch gate's same-device/same-dtype/contiguous checks pass
            // and the kernel updates m/v in VRAM without a host round-trip.
            let m = KtTensor::zeros_on(primary.device(), dims.clone(), dtype)
                .with_context(|| "allocating AdamW first-moment tensor")?;
            let v = KtTensor::zeros_on(primary.device(), dims, dtype)
                .with_context(|| "allocating AdamW second-moment tensor")?;
            moments.insert(param.tensor_id(), KtAdamWMoments { m, v });
        }
        Ok(OptimizerState::AdamW {
            adamw: KtAdamW::new(hp),
            moments,
            host_authoritative: HashSet::new(),
            step: 0,
        })
    }

    /// (#1082) Allocate kt-native Muon optimizer state: one zero-init
    /// device momentum tensor per LoRA `Parameter` (same shape+dtype as
    /// the param master, on the param's own device so the on-device
    /// Newton-Schulz kernel's same-device/same-dtype/contiguous gate
    /// passes), plus the CPU reference `kiln_optim::Muon` host fallback.
    pub fn allocate_muon_state(
        &self,
        lr: f64,
        momentum: f32,
        nesterov: bool,
        ns_iters: u32,
        weight_decay: f32,
        _device: &Device,
    ) -> Result<OptimizerState> {
        let mut momenta: HashMap<KtTensorId, KtMuonMomentum> = HashMap::new();
        for param in self.all_params() {
            let primary = param.forward_storage().primary_tensor();
            let dims: Vec<usize> = primary.dims().to_vec();
            let dtype = primary.dtype();
            let m = KtTensor::zeros_on(primary.device(), dims, dtype)
                .with_context(|| "allocating Muon momentum tensor")?;
            momenta.insert(param.tensor_id(), KtMuonMomentum { m });
        }
        Ok(OptimizerState::Muon {
            muon: KtMuon::new(lr as f32, momentum, nesterov, ns_iters, weight_decay),
            momenta,
            host_authoritative: HashSet::new(),
            step: 0,
        })
    }
}
pub(super) fn checkpoint_parameter_key(layer_idx: usize, module: &str, matrix: &str) -> String {
    let sub = if matches!(
        module,
        "q_proj" | "k_proj" | "v_proj" | "o_proj" | "in_proj_qkv" | "in_proj_z" | "out_proj"
    ) {
        "self_attn"
    } else {
        "mlp"
    };
    format!("base_model.model.model.layers.{layer_idx}.{sub}.{module}.lora_{matrix}.weight")
}

impl TrainableLoraParams {
    /// Convert trainable params to a `LoraWeights` for use with the forward pass.
    ///
    /// The returned `LoraWeights` holds tensors that are backed by our Vars,
    /// so autograd tracks all operations through them.
    pub fn as_lora_weights(&self) -> LoraWeights {
        let layers: Vec<LoraLayerWeights> = self
            .layers
            .iter()
            .map(|lp| {
                // (#1082) `LoraProjectionWeights.a/.b` are kt `Tensor` now;
                // thread each param's primary kt tensor (the BF16 LoRA
                // master) straight in. The tape forward records ops over
                // these kt tensors, so the backward grad keys on
                // `Parameter::tensor_id()` == `a/.b.id()`.
                let make_proj =
                    |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                        pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                            a: a.forward_storage().primary_tensor().clone(),
                            b: b.forward_storage().primary_tensor().clone(),
                        })
                    };
                LoraLayerWeights {
                    q_proj: make_proj(&lp.q_proj),
                    k_proj: make_proj(&lp.k_proj),
                    v_proj: make_proj(&lp.v_proj),
                    o_proj: make_proj(&lp.o_proj),
                    in_proj_qkv: make_proj(&lp.in_proj_qkv),
                    in_proj_z: make_proj(&lp.in_proj_z),
                    gdn_out_proj: make_proj(&lp.gdn_out_proj),
                    gate_proj: make_proj(&lp.gate_proj),
                    up_proj: make_proj(&lp.up_proj),
                    down_proj: make_proj(&lp.down_proj),
                }
            })
            .collect();

        let make_proj_view =
            |pair: &Option<(Parameter, Parameter)>| -> Option<LoraProjectionWeights> {
                pair.as_ref().map(|(a, b)| LoraProjectionWeights {
                    a: a.forward_storage().primary_tensor().clone(),
                    b: b.forward_storage().primary_tensor().clone(),
                })
            };
        let mtp = self.mtp.as_ref().map(|mp| LoraLayerWeights {
            q_proj: make_proj_view(&mp.q_proj),
            k_proj: make_proj_view(&mp.k_proj),
            v_proj: make_proj_view(&mp.v_proj),
            o_proj: make_proj_view(&mp.o_proj),
            gate_proj: make_proj_view(&mp.gate_proj),
            up_proj: make_proj_view(&mp.up_proj),
            down_proj: make_proj_view(&mp.down_proj),
            ..Default::default()
        });

        LoraWeights {
            layers,
            mtp,
            rank: self.rank,
            alpha: self.alpha,
            scale: self.scale,
            source_identity: None,
        }
    }

    /// Collect all LoRA `Parameter` references for grad lookup + updates.
    pub fn all_params(&self) -> Vec<&Parameter> {
        self.all_params_with_modules()
            .into_iter()
            .map(|entry| entry.param)
            .collect()
    }

    pub(super) fn all_params_with_modules(&self) -> Vec<LoraParamRef<'_>> {
        let mut params = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            push_lora_param_pair(&mut params, layer_idx, "q_proj", &layer.q_proj);
            push_lora_param_pair(&mut params, layer_idx, "k_proj", &layer.k_proj);
            push_lora_param_pair(&mut params, layer_idx, "v_proj", &layer.v_proj);
            push_lora_param_pair(&mut params, layer_idx, "o_proj", &layer.o_proj);
            push_lora_param_pair(&mut params, layer_idx, "in_proj_qkv", &layer.in_proj_qkv);
            push_lora_param_pair(&mut params, layer_idx, "in_proj_z", &layer.in_proj_z);
            push_lora_param_pair(&mut params, layer_idx, "out_proj", &layer.gdn_out_proj);
            push_lora_param_pair(&mut params, layer_idx, "gate_proj", &layer.gate_proj);
            push_lora_param_pair(&mut params, layer_idx, "up_proj", &layer.up_proj);
            push_lora_param_pair(&mut params, layer_idx, "down_proj", &layer.down_proj);
        }
        params
    }

    /// Mutable variant — the optimizer step + `sync_to_master` mutate
    /// each `Parameter`'s storage in place (preserving `tensor_id`).
    /// Same traversal order as [`Self::all_params`].
    pub fn all_params_mut(&mut self) -> Vec<&mut Parameter> {
        let mut out: Vec<&mut Parameter> = Vec::new();
        for layer in &mut self.layers {
            for pair in [
                &mut layer.q_proj,
                &mut layer.k_proj,
                &mut layer.v_proj,
                &mut layer.o_proj,
                &mut layer.in_proj_qkv,
                &mut layer.in_proj_z,
                &mut layer.gdn_out_proj,
                &mut layer.gate_proj,
                &mut layer.up_proj,
                &mut layer.down_proj,
            ] {
                if let Some((a, b)) = pair.as_mut() {
                    out.push(a);
                    out.push(b);
                }
            }
        }
        out
    }

    /// Stable PEFT-compatible names for the main-loop trainable parameters.
    /// Tensor IDs are process-local and must never appear in durable optimizer
    /// state, so checkpoint save/restore joins state through this ordering.
    pub(super) fn checkpoint_param_keys(&self) -> Vec<String> {
        self.all_params_with_modules()
            .into_iter()
            .map(|entry| checkpoint_parameter_key(entry.layer_idx, entry.module, entry.matrix))
            .collect()
    }

    pub(super) fn checkpoint_params(&self) -> Vec<(String, &Parameter)> {
        self.checkpoint_param_keys()
            .into_iter()
            .zip(self.all_params())
            .collect()
    }

    pub(super) fn checkpoint_params_mut(&mut self) -> Vec<(String, &mut Parameter)> {
        let keys = self.checkpoint_param_keys();
        keys.into_iter().zip(self.all_params_mut()).collect()
    }

    /// Capture exact main-loop adapter parameters into CPU storage without
    /// PEFT receipts/config. The enclosing checkpoint writer owns atomicity
    /// and checksums.
    pub(crate) fn capture_checkpoint_parameters(&self) -> Result<CheckpointTensorSnapshot> {
        let mut owned = Vec::with_capacity(self.all_params().len());
        for (key, param) in self.checkpoint_params() {
            let tensor = param
                .forward_storage()
                .primary_tensor()
                .to_device(kiln_tensor::Device::Cpu)
                .and_then(|tensor| tensor.contiguous())
                .map_err(|error| {
                    anyhow::anyhow!("checkpoint adapter parameter {key}: to CPU: {error}")
                })?;
            owned.push((key, tensor));
        }
        CheckpointTensorSnapshot::new(owned, "adapter parameter")
    }

    /// Save adapter parameters directly. Production loop checkpointing uses
    /// a coordinated CPU snapshot and publishes it after releasing the GPU
    /// lock; this wrapper remains useful to codecs and focused tests.
    pub fn save_checkpoint_parameters(&self, path: &Path) -> Result<()> {
        self.capture_checkpoint_parameters()?.save(path)
    }

    /// Restore exact main-loop adapter parameters by stable name. Missing,
    /// extra, shape-drifted, or dtype-drifted tensors fail before mutation.
    pub fn load_checkpoint_parameters(&mut self, path: &Path) -> Result<()> {
        let mut loaded = kiln_tensor::safetensors::load_cpu(path)
            .map_err(|error| anyhow::anyhow!("load checkpoint adapter parameters: {error}"))?;
        let expected: BTreeSet<_> = self.checkpoint_param_keys().into_iter().collect();
        let actual: BTreeSet<_> = loaded.keys().cloned().collect();
        anyhow::ensure!(
            actual == expected,
            "checkpoint adapter parameter set mismatch: expected {expected:?}, found {actual:?}"
        );

        // Validate the entire file before replacing the first live parameter.
        for (key, param) in self.checkpoint_params() {
            let tensor = loaded
                .get(&key)
                .with_context(|| format!("checkpoint adapter parameter {key} missing"))?;
            let current = param.forward_storage().primary_tensor();
            anyhow::ensure!(
                tensor.dims() == current.dims(),
                "checkpoint adapter parameter {key} shape mismatch: expected {:?}, found {:?}",
                current.dims(),
                tensor.dims()
            );
            anyhow::ensure!(
                tensor.dtype() == current.dtype(),
                "checkpoint adapter parameter {key} dtype mismatch: expected {}, found {}",
                current.dtype(),
                tensor.dtype()
            );
            checkpoint_ensure_finite_tensor(tensor, &key)?;
        }

        for (key, param) in self.checkpoint_params_mut() {
            let current_device = param.forward_storage().primary_tensor().device();
            let tensor = loaded
                .remove(&key)
                .expect("validated checkpoint parameter must exist")
                .to_device(current_device)
                .map_err(|error| {
                    anyhow::anyhow!("checkpoint adapter parameter {key}: to device: {error}")
                })?;
            param
                .replace_plain_trainable_tensor(tensor)
                .map_err(|error| {
                    anyhow::anyhow!("restore checkpoint adapter parameter {key}: {error}")
                })?;
        }
        Ok(())
    }

    /// Load a previously-saved PEFT adapter into the existing LoRA
    /// `Parameter`s, replacing the seeded-init values.
    ///
    /// Reads `<adapter_dir>/adapter_model.safetensors` and installs each
    /// tensor into the matching LoRA `Parameter`. The adapter's rank,
    /// alpha, and target_modules must match this `TrainableLoraParams`
    /// instance — those are passed at `initialize_seeded` time and not
    /// reconfigurable here.
    ///
    /// Used by Phase 3 verifier-free chaining: take a strong Phase 2
    /// adapter, run `--no-policy-loss` from those weights, save a new
    /// adapter that's a verifier-free continuation. Without this, the
    /// `--base-adapter` CLI flag is effectively a lineage label.
    ///
    /// Returns the number of tensors loaded. Training entry points call
    /// `validate_base_adapter_compatibility` before this method so missing,
    /// extra, rank-mismatched, or shape-mismatched tensors fail before
    /// optimizer setup instead of leaving seeded-init gaps.
    // (#1082) `&mut self` now: loading replaces each `Parameter`'s
    // forward + backward storage (preserving `tensor_id`) rather than
    // calling the (deleted) candle `Var::set`. Safetensors load is
    // kt-native (`kiln_tensor::safetensors::load_cpu`); each loaded kt
    // tensor is moved to the training device and installed directly.
    pub fn load_from_safetensors(&mut self, adapter_dir: &Path, device: &Device) -> Result<usize> {
        let st_path = adapter_dir.join("adapter_model.safetensors");
        // (#1082) kt-native safetensors load — `kiln_tensor::safetensors::load_cpu`
        // returns CPU kt tensors; each is moved to the training device and
        // installed directly. No candle: was a candle `safetensors::load` + a
        // per-tensor candle->kt borrow.
        let tensors = kiln_tensor::safetensors::load_cpu(&st_path)
            .with_context(|| format!("loading adapter safetensors from {}", st_path.display()))?;

        let install = |param: &mut Parameter, t: &KtTensor, key: &str| -> Result<()> {
            let kt = t
                .to_device(*device)
                .map_err(|e| anyhow::anyhow!("load adapter {key}: to device: {e}"))?;
            param
                .replace_plain_trainable_tensor(kt)
                .map_err(|error| anyhow::anyhow!("load PEFT adapter parameter {key}: {error}"))?;
            Ok(())
        };

        let mut loaded = 0usize;
        for (layer_idx, layer) in self.layers.iter_mut().enumerate() {
            let mut load_proj = |name: &str,
                                 pair: &mut Option<(Parameter, Parameter)>,
                                 is_attn: bool|
             -> Result<()> {
                if let Some((a, b)) = pair.as_mut() {
                    let sub = if is_attn { "self_attn" } else { "mlp" };
                    let prefix = format!("base_model.model.model.layers.{layer_idx}.{sub}.{name}");
                    let a_key = format!("{prefix}.lora_A.weight");
                    let b_key = format!("{prefix}.lora_B.weight");
                    if let Some(a_t) = tensors.get(&a_key) {
                        install(a, a_t, &a_key)?;
                        loaded += 1;
                    }
                    if let Some(b_t) = tensors.get(&b_key) {
                        install(b, b_t, &b_key)?;
                        loaded += 1;
                    }
                }
                Ok(())
            };

            load_proj("q_proj", &mut layer.q_proj, true)?;
            load_proj("k_proj", &mut layer.k_proj, true)?;
            load_proj("v_proj", &mut layer.v_proj, true)?;
            load_proj("o_proj", &mut layer.o_proj, true)?;
            load_proj("in_proj_qkv", &mut layer.in_proj_qkv, true)?;
            load_proj("in_proj_z", &mut layer.in_proj_z, true)?;
            load_proj("out_proj", &mut layer.gdn_out_proj, true)?;
            load_proj("gate_proj", &mut layer.gate_proj, false)?;
            load_proj("up_proj", &mut layer.up_proj, false)?;
            load_proj("down_proj", &mut layer.down_proj, false)?;
        }

        tracing::info!(
            path = %adapter_dir.display(),
            num_tensors = loaded,
            "loaded base adapter into TrainableLoraParams"
        );
        Ok(loaded)
    }

    /// Save the trained adapter in PEFT-compatible format.
    ///
    /// Creates `adapter_config.json` and `adapter_model.safetensors` that can
    /// be loaded by the existing `LoraWeights::load()` method.
    pub fn save_peft(&self, output_dir: &Path, _num_layers: usize) -> Result<PathBuf> {
        std::fs::create_dir_all(output_dir)
            .with_context(|| format!("creating adapter dir: {}", output_dir.display()))?;

        // Write adapter_config.json
        let config = serde_json::json!({
            "r": self.rank,
            "lora_alpha": self.alpha,
            "target_modules": crate::adapter_shape::TRAINABLE_TARGET_MODULES,
            "task_type": "CAUSAL_LM",
            "bias": "none",
            "peft_type": "LORA",
        });
        let config_path = output_dir.join("adapter_config.json");
        std::fs::write(&config_path, serde_json::to_string_pretty(&config)?)?;

        // Collect all LoRA tensors for safetensors serialization.
        // (#1082) kt-native: read each `Parameter`'s primary kt tensor, move it
        // to CPU (contiguous) for the writer, and serialize via
        // `kiln_tensor::safetensors::save_cpu`. No candle: was a per-tensor
        // kt->candle copy + a candle writer.
        let mut owned: Vec<(String, KtTensor)> = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let mut save_proj =
                |name: &str, pair: &Option<(Parameter, Parameter)>, is_attn: bool| -> Result<()> {
                    if let Some((a, b)) = pair {
                        let sub = if is_attn { "self_attn" } else { "mlp" };
                        let prefix =
                            format!("base_model.model.model.layers.{layer_idx}.{sub}.{name}");
                        let to_cpu = |kt: &KtTensor, key: &str| -> Result<KtTensor> {
                            kt.to_device(kiln_tensor::Device::Cpu)
                                .and_then(|t| t.contiguous())
                                .map_err(|e| anyhow::anyhow!("save adapter {key}: to cpu: {e}"))
                        };
                        let a_key = format!("{prefix}.lora_A.weight");
                        let b_key = format!("{prefix}.lora_B.weight");
                        let a_cpu = to_cpu(a.forward_storage().primary_tensor(), &a_key)?;
                        let b_cpu = to_cpu(b.forward_storage().primary_tensor(), &b_key)?;
                        owned.push((a_key, a_cpu));
                        owned.push((b_key, b_cpu));
                    }
                    Ok(())
                };

            save_proj("q_proj", &layer.q_proj, true)?;
            save_proj("k_proj", &layer.k_proj, true)?;
            save_proj("v_proj", &layer.v_proj, true)?;
            save_proj("o_proj", &layer.o_proj, true)?;
            save_proj("in_proj_qkv", &layer.in_proj_qkv, true)?;
            save_proj("in_proj_z", &layer.in_proj_z, true)?;
            save_proj("out_proj", &layer.gdn_out_proj, true)?;
            save_proj("gate_proj", &layer.gate_proj, false)?;
            save_proj("up_proj", &layer.up_proj, false)?;
            save_proj("down_proj", &layer.down_proj, false)?;
        }

        // MTP draft-block LoRA (MTP training plan PR-B). Keyed under
        // `...mtp.layers.0...` — the loader parses these into
        // `LoraWeights.mtp` (and `parse_peft_key.is_mtp` keeps them from
        // aliasing main layer 0).
        if let Some(mtp) = &self.mtp {
            let mut save_mtp =
                |name: &str, pair: &Option<(Parameter, Parameter)>, is_attn: bool| -> Result<()> {
                    if let Some((a, b)) = pair {
                        let sub = if is_attn { "self_attn" } else { "mlp" };
                        let prefix = format!("base_model.model.model.mtp.layers.0.{sub}.{name}");
                        let to_cpu = |kt: &KtTensor, key: &str| -> Result<KtTensor> {
                            kt.to_device(kiln_tensor::Device::Cpu)
                                .and_then(|t| t.contiguous())
                                .map_err(|e| anyhow::anyhow!("save adapter {key}: to cpu: {e}"))
                        };
                        let a_key = format!("{prefix}.lora_A.weight");
                        let b_key = format!("{prefix}.lora_B.weight");
                        let a_cpu = to_cpu(a.forward_storage().primary_tensor(), &a_key)?;
                        let b_cpu = to_cpu(b.forward_storage().primary_tensor(), &b_key)?;
                        owned.push((a_key, a_cpu));
                        owned.push((b_key, b_cpu));
                    }
                    Ok(())
                };
            save_mtp("q_proj", &mtp.q_proj, true)?;
            save_mtp("k_proj", &mtp.k_proj, true)?;
            save_mtp("v_proj", &mtp.v_proj, true)?;
            save_mtp("o_proj", &mtp.o_proj, true)?;
            save_mtp("gate_proj", &mtp.gate_proj, false)?;
            save_mtp("up_proj", &mtp.up_proj, false)?;
            save_mtp("down_proj", &mtp.down_proj, false)?;
        }

        let st_path = output_dir.join("adapter_model.safetensors");
        let save_map: std::collections::HashMap<&str, &KtTensor> =
            owned.iter().map(|(k, v)| (k.as_str(), v)).collect();
        kiln_tensor::safetensors::save_cpu(&save_map, &st_path)
            .with_context(|| format!("saving safetensors to {}", st_path.display()))?;
        let adapter_name = output_dir
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("adapter");
        crate::adapter_output::write_adapter_output_receipt(output_dir, adapter_name, None)
            .with_context(|| {
                format!("writing adapter output receipt to {}", output_dir.display())
            })?;

        tracing::info!(
            path = %output_dir.display(),
            num_tensors = owned.len(),
            "saved PEFT adapter"
        );

        Ok(output_dir.to_path_buf())
    }
}
