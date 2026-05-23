# Vulkan-Native Training (`vk-native`)

A GPU-resident training stack for Kiln. Replaces candle's `Tensor` and
autograd with a native `VkTensor` type and eager autograd tape so
every forward intermediate, every gradient, and every optimizer state
lives in Vulkan device memory.

## Status (2026-05-11)

End-to-end vk-native training is **wired together for hybrid
Qwen3.5-4B** (24 GDN + 8 FullAttn layers). Subject to runtime
validation on a real model + adapter run.

**Ready to use** (gated behind `KILN_VK_NATIVE_TRAINING=1`):
- `vk_native_sft_train` (multi-epoch loop, LoRA init, AdamW, adapter
  save in PEFT format) — wired into `kiln-server`'s training queue.
- `vk_native_grpo_train` / `vk_native_grpo_train_jsonl` (GRPO loss
  with on-the-fly reference-logprob recompute via frozen-base forward,
  PPO-style ratio clipping, checkpoint cadence). Independently
  toggleable via `KILN_VK_NATIVE_GRPO`. **Issue #1076**: workload-shape
  auto-tunes between non-recompute (`vk_grpo_train_step_with_state` —
  full activation tape, fastest on comfortable VRAM) and recompute
  (`vk_recompute_grpo_train_step_with_state` — layerwise reverse-
  recompute, memory-saving). Pin recompute with `KILN_VK_RECOMPUTE_GRPO=1`;
  pin non-recompute with `KILN_NO_GRAD_CHECKPOINT=1`. ECHO env-CE
  forces recompute (only path with the ECHO term).
- `vk_native_opd_train` (off-policy distillation, reverse-KL against
  teacher top-K logprobs, `top_k ∈ {16, 32}` via the fused
  `vk_opd_top_k_reverse_kl_loss` kernel). Independently toggleable via
  `KILN_VK_NATIVE_OPD`. V1 envelope: `training_mode = off_policy`,
  `objective = reverse_kl`, `loss = teacher_top_k`; on-policy student
  rollouts, cross-entropy, full-vocab KL, ECHO env-CE, and continual
  `base_adapter` resume all stay on the candle path. **Issue #1076**:
  workload-shape auto-tunes between non-recompute
  (`vk_opd_train_step_with_state`) and recompute
  (`vk_recompute_opd_train_step_with_state`), with hybrid GDN models
  now able to use the non-recompute path when VRAM is comfortable.
  Pin recompute with `KILN_VK_RECOMPUTE_OPD=1`; pin non-recompute with
  `KILN_NO_GRAD_CHECKPOINT=1`.
- FullAttn layer with Qwen3.5 specifics: per-head Q/K-norm,
  `attn_output_gate` (q_proj fused with [Q, gate], sigmoid·attn),
  RoPE precomputed from `rotary_inv_freq` + applied between QK-norm
  and SDPA.
- GDN layer (chunkwise forward + backward, conv1d, gates,
  gated_rms_norm, full state plumbing).
- `VkModelWeights::from_gpu_weights`: candle `GpuWeights` →
  vk-native, dispatching FullAttention vs LinearAttention.
- Gradient checkpointing for FullAttn-only models
  (`vk_checkpointed_train_step` — segmented forward + per-segment
  backward via the scalar trick `seg_loss = sum(seg_out · upstream)`).
  Auto-enabled via `KILN_GRAD_CHECKPOINT_SEGMENTS` (default 4).
- 27 parity / smoke tests across 4 suites: 11 smoke (incl. full-
  pipeline adapter-save test), 4 GDN-foundation, 6 GDN-backward,
  6 GDN-chunkwise.

**Known gaps** (Phase 7+ follow-ups):
- Gradient checkpointing for hybrid (GDN) models needs per-segment
  recurrent-state snapshots — the current path bypasses checkpointing
  when GDN layers are present. For Qwen3.5-4B, set
  `KILN_NO_GRAD_CHECKPOINT=1` and accept the single-step memory
  budget for v1 validation.
- Per-LoRA-param gradient parity vs the candle reference at T=64 has
  not been measured. The math is parity-tested per-shader / per-op,
  but the end-to-end Qwen3.5-4B numerical bar is the Phase 7 gate.

**GPU residency status**: The vk-native training step is now FULLY
GPU-resident. Every per-chunk and per-layer operation in both forward
and backward runs as a Vulkan dispatch. CPU touchpoints in a step
are: (1) input_ids upload at the start, (2) loss scalar readback for
logging, (3) adapter safetensors save at the end of the run.
Everything else — embedding, RMSNorm, projections (Q/K/V/O +
gate/up/down), SDPA, RoPE, FLCE loss, GDN chunkwise forward
(chunk_prep + solve_tri_v2 + matmul + state_update), GDN backward
(solve_tri_transpose + chunk_scan_bwd + state_exit_bwd +
gated_rms_norm_bwd + chunk_prep_bwd + reverse_cumsum + gates_bwd +
conv1d_bwd), AdamW step — all GPU.

See `docs/vk_native_gdn.md` for the GDN math derivation.

On unified-memory hardware (AMD Strix Halo APUs, Intel iGPUs, etc.)
the bytes never physically move — they just stop being accounted as
anon-rss by the kernel, eliminating the OOM ceiling that candle's
CPU autograd hits on long sequences. On discrete GPUs the same path
keeps activations off the CPU entirely.

## Why

The candle-based training path holds every `CustomOp::cpu_fwd` output
as a `Tensor` backed by `CpuStorage`. At Qwen3.5-4B with T=918 and
LoRA rank=16, that means ~10 GB of F32 forward intermediates plus a
similar amount of gradient buffers — peak ~23 GB anon-rss on a 30 GB
UMA box. Stubbing transposed weight caches frees ~7 GB of model
residency but does nothing about the autograd intermediates, so the
ceiling remains. The vk-native path eliminates those CPU buffers
entirely.

Beyond the immediate Strix Halo unblock, this means **Kiln can
train on the long tail of GPUs that Vulkan reaches** — AMD (RADV /
AMDVLK), Intel (ANV), Nvidia (vulkan driver), Apple (MoltenVK), Adreno,
Mali, plus integrated graphics on commodity laptops — without a
CUDA/ROCm/Metal dependency in the training path.

## Architecture

### `VkTensor` — the GPU-resident tensor type

```rust
pub struct VkTensor(Arc<VkTensorInner>);

struct VkTensorInner {
    storage: Arc<VulkanBuffer>,        // GPU memory (device-local)
    shape: Vec<usize>,                  // C-contiguous
    dtype: VkDType,                     // F32 or Bf16Packed
    device: Arc<VulkanDevice>,
    grad_fn: Option<Arc<dyn VkBackwardOp>>,
    requires_grad: bool,
    op_id: u64,                         // monotonic, for topo ordering
    param_id: Option<TensorId>,         // set on parameter leaves
}
```

Cloning is an `Arc::clone` of the inner shell — cheap. The buffer is
refcount-owned; when the last reference drops, the
`VulkanBuffer::Drop` frees device memory.

Three constructors:
- `from_buffer(...)`: leaf, no autograd link
- `parameter(..., TensorId)`: leaf with `requires_grad=true` and a
  `param_id` that the autograd uses to key gradients
- `from_op(..., grad_fn)`: output of a forward op with a backward closure

### Autograd

Eager tape, PyTorch-style. Each forward op produces a `VkTensor` whose
`grad_fn` is an `Arc<dyn VkBackwardOp>` holding `Arc` references to
its inputs. `vk_backward(loss)` walks the graph in topological
reverse order, accumulating gradients per `op_id` and emitting a
`VkGradStore` keyed by `TensorId` for every reachable parameter leaf:

```rust
pub fn vk_backward(loss: &VkTensor) -> Result<VkGradStore>
```

Multi-use accumulation uses `vk_add_no_grad`. Buffer lifetime is
refcount-driven — as backward visits each op and drops its reference,
intermediate buffers are freed immediately, mirroring PyTorch's
"free-as-you-go" backward.

### Op surface

All implemented as `vk_<op>` functions in `kiln-vulkan-kernel/src/vk_ops/`.
Each module exposes:
- a forward `vk_<op>` that attaches an autograd `<Op>Backward`
- a `vk_<op>_no_grad` for autograd-internal use (e.g., gradient accumulation)

| Op | Shaders | Backward |
|----|---------|----------|
| elementwise add/sub/mul/div | `vk_elementwise_binary_f32` | analytic |
| sum / mean reduction | `vk_reduce_sum_f32` + `vk_broadcast_scalar_f32` | scalar→broadcast |
| F32 ↔ BF16 cast | `vk_cast_{f32_to_bf16,bf16_to_f32}` | identity (precision drop) |
| reshape (metadata) / transpose 2D | `vk_transpose_2d_f32` | inverse transpose |
| matmul 2D | `vk_matmul_f32` | two matmuls (dA, dB) |
| matmul batched 3D | `vk_matmul_batched_f32` + `vk_transpose_3d_f32` | per-batch dA, dB |
| RMSNorm | reuses `qwen_rmsnorm_{forward,backward}` | analytic |
| softmax (last dim) | `vk_softmax_lastdim_{f32,bwd_f32}` | analytic |
| SiLU | `vk_silu_{f32,bwd_f32}` | analytic |
| RoPE | `vk_rope_{f32,bwd_f32}` | inverse rotation |
| permute rh↔hr | `vk_permute_{rh_to_hr,hr_to_rh}_f32` | inverse permute |
| repeat KV heads (GQA) | `vk_repeat_kv_heads_f32` | `vk_sum_kv_groups_f32` |
| causal mask + scale | `vk_causal_mask_add_f32`, `vk_scale_inplace_f32` | in-place, no-grad |
| embedding lookup | `vk_embedding_lookup_{f32,bf16w_f32}` | scatter-add (frozen by default) |
| FLCE (fused linear + xent) | five `vk_flce_*` shaders | softmax−one_hot, per chunk |
| OPD top-K reverse-KL | `vk_opd_topk_kl_{fwd,bwd}_{f32,bf16w}` + `vk_opd_topk_metrics_{f32,bf16w}` | analytic d_hidden, recomputes p_hat/log_p_hat in shared mem (mirrors CUDA) |
| SDPA prefill (causal, GQA) | composition of the above | composition |

Roughly **25 new `.comp` shaders** plus a handful of helper kernels;
all under 100 LOC each, all SPIR-V-embedded via `build.rs`.

### Composed building blocks (no new shaders)

| Block | Composition |
|-------|-------------|
| Linear + LoRA | `vk_matmul` for base + `transpose + matmul + matmul + scale + add` for LoRA delta |
| SwiGLU MLP | `vk_matmul` (gate) + `vk_matmul` (up) + `vk_silu` + `vk_mul` + `vk_matmul` (down) |
| GQA attention block | `permute_rh_to_hr` × 3 + `repeat_kv_heads` × 2 + `transpose_batched` + `matmul_batched` + `scale_inplace` + `causal_mask_inplace` + `softmax_lastdim` + `matmul_batched` + `permute_hr_to_rh` |
| Transformer layer | `rmsnorm` + qkv LoRA-linears + `sdpa` + o LoRA-linear + residual `add` + `rmsnorm` + SwiGLU MLP + residual `add` |
| Full model forward | `embedding_lookup` + N × transformer layer + `rmsnorm` + `flce_loss` |

### Boundary with candle

Three boundary points only:

1. **Parameter init / upload**: candle `Var` → `VkTensor::from_candle` once
   at training start. The Var's `TensorId` keys the parameter VkTensor.
2. **Optimizer step**: `dispatch_adamw_step_f32` called directly with
   `VulkanBuffer` handles. No candle Tensor round-trip.
3. **Adapter save**: `VkTensor::to_candle()` reads each LoRA param back
   to CPU (one ~30 KB readback per parameter at rank=16) before
   `candle_core::safetensors::save`.

Inference is unchanged.

## Training step

```rust
use kiln_model::vk_forward::{vk_model_forward_loss, VkLoraLayer, VkModelWeights};
use kiln_train::vk_train::{allocate_adamw_state, vk_train_step, VkAdamWConfig};

// One-time setup
let weights: VkModelWeights = ...;            // upload base weights once
let lora: Vec<VkLoraLayer> = ...;             // init random A, zero B
let mut adamw = allocate_adamw_state(&dev, &lora)?;
let cfg = VkAdamWConfig { lr: 5e-5, ..Default::default() };

// Loop
for (step, input_ids) in batches.enumerate() {
    let loss = vk_train_step(&weights, &lora, &input_ids, &mut adamw, &cfg, step as u32 + 1)?;
    info!("step={step} loss={loss}");
}

save_vk_lora_adapter(&lora, rank, alpha, &output_path)?;
```

## What's proven

End-to-end vk-native training of a synthetic transformer (1 layer,
hidden=32, heads_q=2/heads_kv=1, head_dim=16, intermediate=64,
vocab=32, rank=4, 7 LoRA pairs per layer = 14 trainable tensors)
**loss strictly decreases** over 10 AdamW steps on real Vulkan
hardware:

```
losses: 3.572 → 3.572 → 3.448 → 3.277 → 3.093 → 2.911
      → 2.715 → 2.551 → 2.432 → 2.335 → 2.250
```

Step 0 and 1 are identical because B is zero-initialized (so LoRA
delta = 0 → no contribution to forward). After step 1's update, B is
nonzero and the loss begins decreasing monotonically.

## Test coverage

81 vk-native parity tests on real GPU hardware:

```
tests/vk_tensor_parity.rs       15 tests   (Phase A: VkTensor + autograd)
tests/vk_matmul_parity.rs        4 tests   (Phase B: matmul + LoRA composition)
tests/vk_rmsnorm_parity.rs       2 tests   (Phase B: RMSNorm)
tests/vk_softmax_parity.rs       2 tests   (Phase C: softmax)
tests/vk_attention_parity.rs     4 tests   (Phase C: SDPA + permute + GQA repeat)
tests/vk_flce_parity.rs          1 test    (Phase E: FLCE forward + backward)
tests/vk_opd_parity.rs           10 tests  (Phase F: OPD top-K reverse-KL fwd/bwd + metrics)
src/vk_tensor.rs (unit)          3 tests   (roundtrip, detach)
+ pre-existing tests             43 tests  (gdn_parity, device, etc.)

crates/kiln-train/tests/vk_train_smoke.rs:
- minimal LoRA grad exists
- FLCE through matmul → param
- FLCE through RMSNorm + matmul → param
- FLCE through full SDPA chain → params
- full transformer layer → flce → grads (14 params)
- end-to-end multi-step training (loss decreases)
```

Each parity test compares vk-native output against an analytical CPU
reference and asserts max-abs-diff under tolerance (1e-5 for F32
forward, 1e-4 for backward through compositions).

## What's next (Phase G+)

The Phase E end-to-end demonstration runs on a synthetic small model.
To productionize the path for Qwen3.5-4B real-weights SFT, the
remaining work is:

- **VkModelWeights from real safetensors**: upload Qwen3.5's loaded
  `GpuWeights` into `VkModelWeights` lazily. Re-use existing
  BF16-packed buffers via `kernels::buffer_to_tensor` rather than
  re-uploading.
- **`KILN_VK_NATIVE_TRAINING=1` server route**: in `kiln-server`,
  route SFT jobs to `vk_train_step` when the flag is set.
- **Slice op with autograd**: replace the `to_vec_f32` readback in
  `vk_model_forward_loss` (used to drop the last position before FLCE)
  with a proper `vk_narrow` op.
- **BF16 elementwise + accumulate**: optional, lets LoRA params stay
  BF16 throughout (matches existing AdamW-bf16 dispatch).
- **Memory pool / arena allocator**: reuse same-shape buffers
  between forward and backward to cut allocator overhead.
- **Flash-attention with backward**: replace the explicit
  `Q@K.T → softmax → @V` chain with a fused single-kernel
  forward+backward. Big win at large T.
- **Mask + scale gradient correctness**: current `vk_causal_mask_inplace`
  and `vk_scale_inplace` are no-grad in-place mutations. Their
  effect on gradient is partially absorbed by the surrounding matmul
  backward, but a clean fix is to wrap them as autograd ops too. The
  Phase F smoke test still trains correctly (loss decreases monotonically),
  but the gradient is approximate near these ops.
- **Gradient checkpointing on VkTensor**: re-implement the existing
  `checkpointed_forward_backward` pattern in the VkTensor world, with
  boundary states as detached VkTensors (no CPU round-trip).

## File map

```
crates/kiln-vulkan-kernel/
├── csrc/shaders/
│   ├── vk_elementwise_binary_f32.comp
│   ├── vk_fill_f32.comp
│   ├── vk_reduce_sum_f32.comp
│   ├── vk_broadcast_scalar_f32.comp
│   ├── vk_cast_{f32_to_bf16,bf16_to_f32}.comp
│   ├── vk_transpose_2d_f32.comp
│   ├── vk_matmul_f32.comp
│   ├── vk_softmax_lastdim_{f32,bwd_f32}.comp
│   ├── vk_silu_{f32,bwd_f32}.comp
│   ├── vk_rope_{f32,bwd_f32}.comp
│   ├── vk_causal_mask_add_f32.comp
│   ├── vk_scale_inplace_f32.comp
│   ├── vk_matmul_batched_f32.comp
│   ├── vk_transpose_3d_f32.comp
│   ├── vk_permute_{rh_to_hr,hr_to_rh}_f32.comp
│   ├── vk_repeat_kv_heads_f32.comp
│   ├── vk_sum_kv_groups_f32.comp
│   ├── vk_embedding_lookup_{f32,bf16w_f32}.comp
│   └── vk_flce_{chunk_stats,gather_correct,log_sum_exp_combine,per_token_loss,grad_chunk}_f32.comp
├── src/
│   ├── vk_tensor.rs               // VkTensor type + candle adapters
│   ├── vk_autograd.rs             // VkBackwardOp trait, VkGradStore, vk_backward
│   └── vk_ops/
│       ├── elementwise.rs
│       ├── reduce.rs
│       ├── cast.rs
│       ├── shape.rs
│       ├── matmul.rs
│       ├── matmul_batched.rs
│       ├── rmsnorm.rs
│       ├── softmax.rs
│       ├── silu.rs
│       ├── rope.rs
│       ├── mask.rs                 // mask + scale + autograd-aware vk_scale
│       ├── permute.rs
│       ├── mlp.rs                  // SwiGLU composition
│       ├── attention.rs            // SDPA composition
│       ├── embedding.rs
│       └── flce.rs

crates/kiln-model/src/
└── vk_forward.rs                  // VkLoraPair, VkLoraLayer, VkModelWeights,
                                    // vk_transformer_layer, vk_model_forward_loss

crates/kiln-train/src/
└── vk_train.rs                    // VkAdamWState, vk_train_step, save_vk_lora_adapter
```

## Literature

Every concept here is standard:

- **Eager autograd tape**: PyTorch since v1.0; Karpathy's `micrograd`
  is the 200-line version.
- **PyTorch-style `Function.backward` per op**: the structure of
  `torch.autograd.Function` subclasses.
- **Memory-efficient autograd via Arc/refcount**: PyTorch's caching
  allocator does the heavyweight version; we ride on Rust's `Drop`.
- **Mixed-precision BF16 with F32 accumulators**: standard since
  AMP/Apex (2018).
- **Fused linear cross-entropy**: Liger Kernel, axolotl's FLCE, etc.;
  our chunked online-LSE design matches the kiln-flce-kernel reference.
- **GQA with KV head broadcast**: introduced with PaLM/LLaMA-2,
  standard in modern LMs.
- **Causal SDPA prefill**: foundational; the explicit
  `Q@K.T → softmax → @V` chain is the canonical reference, fused
  flash-attention being the production optimization.

The Vulkan substrate (GLSL compute shaders → SPIR-V via `glslc`) is
the one mildly novel choice — most projects pick CUDA/ROCm/Metal.
The kernels themselves are vanilla compute primitives that compile
the same way on every modern GPU; Vulkan's portability is the win.
