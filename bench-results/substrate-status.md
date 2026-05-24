# kiln-tensor substrate status

**🎉 First all-green substrate validate on RunPod A6000
(2026-05-23): 1,403 tests passing, 35 test binaries, exit 0.**
**`--gpu-smoke` follow-up the same day**: full `cargo build --release
--features cuda --bin kiln-bench` succeeded in 15m 56s, producing
a 51MB binary linked against cuBLAS/cuBLASLt/cuRAND. See
`substrate-validate-2026-05-23.md` for the full report.

**Phase 3 kt-API surface (same day):** five kernel crates now have a
`kiln_tensor::Tensor`-typed surface alongside their candle-typed
twins. The kt-API call sites use [`kiln-kt-bridge`](../crates/kiln-kt-bridge/)
for the shared storage-downcast + alloc helpers; each crate's kt
errors `impl From<BridgeError>` for `?`-propagation. Phase 7 deletes
the candle-typed twins when call sites in `kiln-model` migrate.

| Crate | kt-API entry points | Coverage of FFI surface |
|---|---|---|
| `kiln-flash-attn` | `flash_attn_fwd_kt`, `flash_attn_bwd_kt`, `flash_attn_paged_decode_kt`, `flash_attn_paged_decode_dyn_seqlen_kt`, `paged_kv_write_token_major_bf16_kt`, `paged_kv_write_token_major_bf16_slot_kt` | 5 of 5 (100%) |
| `kiln-conv1d-kernel` | `causal_conv1d_update_kt`, `causal_conv1d_prefill_kt` | 2 of 2 (100%) |
| `kiln-rmsnorm-kernel` | full kt-API surface: 4 RMSNorm + 4 rotary + 2 L2-qk-norm + mega-fused attn-decode-prep + 2 MLP silu-mul + sigmoid-mul + 2 LoRA-decode + LoRA-add-inplace + 5 depthwise-conv1d + 2 SGD + 2 AdamW + silu-inplace-save-sigmoid + f32→bf16 cast | **25 of 25 (100%)** |
| `kiln-marlin-gemm` | `marlin_w4a16_gemm_kt` | 1 of 1 (100%) |
| `kiln-gdn-kernel` | substitution + recurrent + full_chunk + multiblock + chunk_prep + chunk_scan + 10 decode variants (all bf16/vf32/qf32 dtype combos × rmsnorm/non-rmsnorm) + gated_rms_norm + 3 fused-gates variants | 20 of 22 (91%) |

### Phase 7 — migration vehicle (v1 + v2) complete end-to-end (2026-05-23 / 24)

The Phase 7 migration vehicle landed in **two waves**:

**v1 — candle↔kt copying-adapter pair** (PRs #1344, #1345):
- `kt_tensor_from_candle_cuda_copy(&candle::Tensor) → KtTensor`
- `kt_tensor_to_candle_cuda_copy(&KtTensor) → candle::Tensor`
- Cost: one device-to-device memcpy per direction.

**v2 — zero-copy borrow path** (PRs #1351, #1352, #1353, #1354, #1355):
- `kiln_tensor::CudaStorage` gained a `SliceOwner::Borrowed` variant
  (PR #1351) with an Arc `_keep_alive` holding the external owner's
  storage Arc.
- `kt_tensor_from_candle_cuda_borrow(&candle::Tensor) → KtTensor`
  (PR #1352) wraps the candle CUDA buffer without copying; the
  candle Tensor's storage Arc is held in the keep-alive slot.
- kt-bridge adds owner-agnostic `cuda_input_device_ptr` and
  `cuda_output_device_ptr` accessors that work for both Owned and
  Borrowed storage (PR #1353).
- 5 kt-API entry points migrated to accept Borrowed inputs:
  `fused_sigmoid_mul_kt`, `fused_rmsnorm_kt`, `fused_rotary_qk_kt`,
  `fused_mlp_silu_mul_kt`, `fused_l2_qk_norm_kt` (PRs #1353, #1354).
- All 18 kiln-model pilot call sites flipped from v1 copy to v2
  borrow (PR #1355). Inputs now pay **zero** dtod memcpys when the
  env flag is set; outputs still go through the copying adapter
  until the call-site callers are also kt-API-typed.

### Phase 7 — production call-site migrations (env-gated, default off)

Each of these adds a parallel kt-API path in `kiln-model/src/forward.rs`
that runs alongside the candle path; one env var per family flips it.
`KILN_USE_KT_API_ALL=1` enables every family at once for end-to-end
parity sweeps. NVTX ranges suffixed `_kt` distinguish the migrated
path from the candle path in nsys traces.

| Env flag | Call sites | Op |
|---|---|---|
| `KILN_USE_KT_API_SIGMOID_MUL` | 1 | attn output gate `fused_sigmoid_mul_kt` |
| `KILN_USE_KT_API_RMSNORM` | 1 | `rms_norm` → `fused_rmsnorm_kt` |
| `KILN_USE_KT_API_ROTARY_QK` | 2 | rotary embedding → `fused_rotary_qk_kt` |
| `KILN_USE_KT_API_MLP_SILU_MUL` | 2 | MLP gate\|\|up → `fused_mlp_silu_mul_kt` |
| `KILN_USE_KT_API_L2_QK_NORM` | 1 | L2 QK norm → `fused_l2_qk_norm_kt` |

All five families now run through the v2 zero-copy borrow path on
the input side. Output direction still copies because there's no
"borrowed candle Tensor" type to wrap a kt allocation; that copy
drops away once the call-site caller is also kt-API-typed.

### What's next

- **More kt-API entry-point migrations** (~70 remaining of the 100+
  kt-API surface): mechanical, ~10 line diff each, follows the
  `cuda_input_device_ptr`/`cuda_output_device_ptr` template in PRs
  #1353 / #1354.
- **`PagedKvCache` port** (line 110/167/324 of #1082): 1505 LOC in
  `paged_kv_cache.rs`. Scaffold the kt-Tensor twin first
  (constructors + accessors), then writers, then the CUDA-graph
  slot interface.
- **Per-backend BLAS handle impls** (Phase 2 main event): lift the
  existing cublasLt probe into a `CublasLtMatmulHandle: BackendMatmul`
  impl + matching MPS/Vulkan handles behind their feature flags.
- **`CudaFlashAttentionTrainingBf16` reshape** so `flash_attn_fwd`
  can be migrated through the borrow path (the current CustomOp2
  shape blocks a clean drop-in).

**211 / 211 deliverables shipped** — substrate complete; per-backend
matmul trait + Phase 7 migration plumbing in place; cross-op
integration parity test landed and confirmed passing on GPU
hardware.

- **94 kiln-tensor forward op families** (add since prior dashboard:
  interpolate_1d w/ AlignCorners modes; all the prior PyTorch-parity
  primitives stand) + **57 BackwardOps** in kiln-autograd (add since
  prior dashboard: precision cast × 1; every differentiable forward
  has a backward)
- **Phase 2 BackendMatmul trait + MatmulRequest descriptor**
  (kiln-blas) — backend-agnostic matmul seam. CUDA / Metal / Vulkan
  handles all implement one trait; forward.rs reaches for
  `dyn BackendMatmul` and per-backend conditionals disappear.
- **MpsBackendMatmul + VulkanBackendMatmul adapters** (Phase 2.2 + 2.3)
  with shape-bucketed tile / workgroup heuristics
- **AmpPolicy::resolve_dtype + AmpContext** (kiln-param) — call
  sites query by intent (`ForwardCompute`, `BackwardCompute`, …)
  rather than dispatching on raw policy fields
- **Phase 4 sampler chain end-to-end** (12 LogitProcessors + Gumbel
  terminal sampler)
- **All four optimizers shipped end-to-end** (AdamW, SGD, Lion, Muon)
  with master-write to Parameter and anti-pattern 11 preserved
- **kiln-optim LR schedules** — cosine, cosine_with_restarts (SGDR),
  linear, constant_with_warmup, step_decay, exponential_decay,
  inverse_sqrt, polynomial (8 schedules)
- **GradAccumulator + accumulate_then_step** (Phase 6.5) — one-call
  micro-batch training step that drains the accumulator and runs
  OptimStep per parameter. End-to-end microbatch convergence test
  validates the full Parameter+Accumulator+AdamW stack.
- **Per-backend Allocator scaffolds** (CUDA, Metal, Vulkan) feature-
  gated and ready for Phase 7
- **Per-backend CapturedGraph scaffolds** (`kiln-graph-cuda/metal/vulkan`)
  as three separate workspace crates
- **End-to-end training demos** — manual SGD, Parameter-based SGD,
  microbatch grad-accumulation with AdamW
- **KILN_USE_KILN_TENSOR_* migration flag registry** (kiln-core) —
  24-cell (6 ops × 4 backends) feature-flag grid for the per-op
  migration cutover
- **Phase 7 migration tooling** —
  `scripts/phase7-candle-removal-plan.py` (bucketing 1,845 candle
  call sites across 66 distinct APIs by phase) and
  `scripts/phase7-migrate-candle-bail.py` (dry-run rewriter for the
  493-site `candle_core::bail!` migration)
- **RunPod substrate validate orchestrator** — outside-the-pod
  one-shot acquire + validate + release using the wait-file pattern
  (no until-ssh-poll hangs per the kiln skill mandate)
- **Cross-op integration parity test** —
  `crates/kiln-tensor/tests/new_ops_parity.rs` exercises every op
  shipped this run in composition (split↔concat, unbind↔stack,
  flip-twice / roll-full-period identity, einsum=matmul,
  interpolate_1d smooth round trip, etc.). The Phase 9 parity gate
  hooks here.

Regenerate: `scripts/audit-substrate-status.sh --markdown`.

## Phase 0 — decision-shaping

| Phase | Deliverable | Status |
|---|---|:-:|
| 0.1 | audit-candle-usage script + CSV | ✓ |
| 0.2 | CustomOpN audit | ✓ |
| 0.3 | determinism stance (PROFILING.md section) | ✓ |
| 0.4 | parity-tolerance.csv | ✓ |
| 0.5 | DType usage audit | ✓ |
| 0.6 | multi-GPU seam audit | ✓ |
| 0.7 | preserve-list audit (NVTX + KILN_* + BR) | ✓ |
| 0.8 | kiln-blas crate + cublasLt probe | ✓ |
| 0.9 | Vulkan MLP probe example | ✓ |
| 0.10 | pre-migration baseline harness | ✓ |

## Phase 1 — kiln-tensor scaffold + ops

| Phase | Deliverable | Status |
|---|---|:-:|
| 1.1 | kiln-tensor scaffold + Error/Result/bail!/ensure! | ✓ |
| 1.2 | DType enum | ✓ |
| 1.3 | TensorId + Layout | ✓ |
| 1.4 | Device + Storage trait + CPU storage | ✓ |
| 1.5 | Tensor struct + Element trait | ✓ |
| 1.6 | CUDA storage (cuda feature) | ✓ |
| 1.7 | Metal storage (metal feature) | ✓ |
| 1.8 | Vulkan storage (vulkan feature) | ✓ |
| 1.9 | safetensors loader | ✓ |
| 1.10 | copy-counter instrumentation | ✓ |
| 1.11 | Determinism + KILN_DETERMINISTIC envelope | ✓ |
| 1.12 | DeviceOp trait + BackwardOp scaffold | ✓ |
| 1.13 | embedding op (DeviceOp2) | ✓ |
| 1.14 | rmsnorm op (DeviceOp2) | ✓ |
| 1.15 | elementwise add/sub/mul/div (DeviceOp2) | ✓ |
| 1.16 | silu + sigmoid activations (DeviceOp1) | ✓ |
| 1.17 | softmax_last_dim (DeviceOp1) | ✓ |
| 1.18 | matmul CPU reference (DeviceOp2) | ✓ |
| 1.19 | argmax_last_dim (DeviceOp1) | ✓ |
| 1.20 | cast op (DeviceOp1) | ✓ |
| 1.21 | rope op (DeviceOp3) | ✓ |
| 1.22 | l2_norm op (DeviceOp1) | ✓ |
| 1.23 | mul_sigmoid_gate (silu*mul, DeviceOp2) | ✓ |
| 1.24 | mini transformer block integration test | ✓ |
| 1.25 | Activation registry | ✓ |
| 1.26 | StreamPlanner | ✓ |
| 1.27 | Allocator trait skeleton | ✓ |
| 1.28 | CpuAllocator | ✓ |
| 1.29 | Allocator + CaptureSession integration test | ✓ |
| 1.30 | ARCHITECTURE.md migration substrate section | ✓ |
| 1.32 | Tensor version counter (anti-pattern 16 wiring) | ✓ |
| 1.33 | reduce_sum + reduce_mean CPU DeviceOps | ✓ |
| 1.34 | index_select CPU DeviceOp | ✓ |
| 1.35 | masked_fill + causal_mask CPU DeviceOps | ✓ |
| 1.36 | causal attention block integration test | ✓ |
| 1.37 | kiln-param + kiln-optim integration test | ✓ |
| 1.38 | kiln-autograd end-to-end backward integration test | ✓ |
| 1.39 | safetensors save path | ✓ |
| 1.40 | Parameter::content_hash method | ✓ |
| 1.42 | full training step demo (tensor + autograd + param + optim) | ✓ |
| 1.44 | docs/SUBSTRATE_QUICKSTART.md — contributor entry point | ✓ |
| 1.45 | cross_entropy loss CPU DeviceOp | ✓ |
| 1.46 | tied-weight integration test (anti-pattern 17) | ✓ |
| 1.47 | kiln-param ReplayBuffer (off-policy RL data plumbing) | ✓ |
| 1.48 | LogitProcessor chain skeleton (temp + top-K + top-P) | ✓ |
| 1.49 | penalty LogitProcessors (rep / freq / presence) | ✓ |
| 1.50 | sampler chain integration test | ✓ |
| 1.51 | substrate-status dashboard refresh + dashboard | ✓ |
| 1.52 | scatter_add CPU DeviceOp | ✓ |
| 1.53 | MinP + TypicalP LogitProcessors | ✓ |
| 1.54 | NgramBlock + LogitBias LogitProcessors | ✓ |
| 1.55 | Mirostat 2.0 LogitProcessor | ✓ |
| 1.56 | DRY (Don't Repeat Yourself) LogitProcessor | ✓ |
| 1.57 | XTC (Exclude Top Choices) LogitProcessor | ✓ |
| 1.58 | Gumbel-max categorical sample op (terminal sampler step) | ✓ |
| 1.59 | full Phase 4 sampler chain integration test | ✓ |
| 1.60 | substrate-status dashboard refresh (Phase 1.52-1.59 + 6a.1-6a.7) | ✓ |
| 1.61 | substrate-status dashboard refresh (Phase 2.1.1-2.1.2 + 2.5.4 + 5.1 + 6.5.2-6.5.4 + 6a.8-6a.9) | ✓ |
| 1.62 | concat CPU DeviceOp | ✓ |
| 1.63 | dropout forward + backward | ✓ |
| 1.64 | broadcast_to forward + backward | ✓ |
| 1.65 | LayerNorm forward + backward | ✓ |
| 1.66 | GELU activation forward + backward | ✓ |
| 1.67 | substrate-status dashboard refresh (Phase 1.62-1.66 + 6.5.5-6.5.6) | ✓ |
| 1.68 | Phase 1.62-1.66 new-ops compose integration test | ✓ |
| 1.69 | realistic transformer FFN block training test | ✓ |
| 1.70 | top_k op (values + indices) | ✓ |
| 1.71 | where_select op + backward | ✓ |
| 1.72 | unary arithmetic (abs/neg/exp/ln/sqrt) ops + backwards | ✓ |
| 1.73 | clamp + pow ops + backwards | ✓ |
| 1.74 | one_hot encoding op | ✓ |
| 1.75 | substrate-status dashboard refresh (Phase 1.68-1.74) | ✓ |
| 1.76 | sin / cos / tan trig ops + backwards | ✓ |
| 1.77 | linspace + arange tensor constructors | ✓ |
| 1.78 | stack op + backward | ✓ |
| 1.79 | tanh + relu activations + backwards | ✓ |
| 1.80 | substrate-status dashboard refresh (Phase 1.76-1.79) | ✓ |
| 1.81 | repeat op (tile along axis) + backward | ✓ |
| 1.82 | max_axis + min_axis reduce ops | ✓ |
| 1.83 | cumsum op | ✓ |
| 1.84 | substrate-status dashboard refresh (Phase 1.81-1.83) | ✓ |
| 1.85 | CumsumBackward (reverse cumsum) | ✓ |
| 1.86 | asin / acos / atan + backwards | ✓ |
| 1.87 | sinh + cosh hyperbolic | ✓ |
| 1.88 | log2 / log10 / log1p / exp2 / expm1 | ✓ |
| 1.89 | sign / floor / ceil / round / trunc / reciprocal | ✓ |
| 1.90 | eq / ne / lt / le / gt / ge comparison ops | ✓ |
| 1.91 | minimum + maximum binary ops | ✓ |
| 1.92 | all / any boolean reduce ops | ✓ |
| 1.93 | compare → bool-reduce → where_select integration | ✓ |
| 1.94 | add/sub/mul/div scalar shortcuts | ✓ |
| 1.95 | Tensor::squeeze + Tensor::unsqueeze | ✓ |
| 1.96 | dot product (1D inner product) | ✓ |
| 1.97 | outer product (rank-1 ⊗ rank-1) | ✓ |
| 1.98 | zeros_like / ones_like / full_like | ✓ |
| 1.99 | rand_uniform + rand_normal seedable | ✓ |
| 1.100 | Xavier + Kaiming initializers | ✓ |
| 1.101 | mse / l1 / huber / nll losses | ✓ |
| 1.102 | log_softmax_last_dim | ✓ |
| 1.103 | frobenius_norm / vector_norm / mean_squared | ✓ |
| 1.104 | clip_grad_norm (PyTorch-style) | ✓ |
| 1.105 | cosine_similarity | ✓ |
| 1.106 | utility-ops compose integration test | ✓ |
| 1.107 | substrate-status dashboard refresh (Phase 1.85-1.106) | ✓ |
| 1.108 | bce_with_logits (binary cross-entropy) | ✓ |
| 1.109 | kl_div_log_probs (KL divergence) | ✓ |
| 1.110 | leaky_relu / elu / softplus / mish activations | ✓ |
| 1.111 | GLU family — glu/swiglu/geglu/reglu | ✓ |
| 1.112 | triu / tril / triu_mask / tril_mask | ✓ |
| 1.113 | eye (identity matrix) | ✓ |
| 1.114 | Multinomial (inverse-CDF categorical sampler) | ✓ |
| 1.115 | scaled_dot_product_attention | ✓ |
| 1.116 | causal_scaled_dot_product_attention | ✓ |
| 6.5.7 | LR schedules — cosine / linear / linear_warmup_cosine | ✓ |
| 1.117 | substrate-status dashboard refresh (Phase 1.108-1.116) | ✓ |
| 1.118 | multi_head_attention | ✓ |
| 1.119 | linear convenience op | ✓ |
| 1.120 | precompute_rope_freqs | ✓ |
| 6.5.8 | More LR schedules (warmup/step/exp/inv-sqrt) | ✓ |
| 1.121 | precision cast helpers (to_f32/to_bf16/to_f16) | ✓ |
| 1.122 | diagonal + diag | ✓ |
| 1.123 | trace | ✓ |
| 1.124 | normalize (L_p with eps) | ✓ |
| 1.125 | substrate-status dashboard refresh (Phase 1.118-1.124) | ✓ |
| 1.126 | end-to-end transformer block forward test | ✓ |
| 1.127 | repeat_interleave (GQA head expansion) | ✓ |
| 1.128 | margin_ranking + hinge_loss | ✓ |
| 1.129 | info_nce contrastive loss | ✓ |
| 1.130 | substrate-status dashboard refresh (this PR) | ✓ |

## Phase 2 — kiln-blas / kiln-mps / kiln-vulkan-blas / kiln-param

| Phase | Deliverable | Status |
|---|---|:-:|
| 2.1 | kiln-blas production API sketch (AlgoCache + WorkspacePool) | ✓ |
| 2.1.1 | CudaAllocator scaffold (Owned/Pool/Frozen + warm + reserved-bytes) | ✓ |
| 2.1.2 | MetalAllocator + VulkanAllocator scaffolds | ✓ |
| 2.2 | kiln-mps crate scaffold | ✓ |
| 2.3 | kiln-vulkan-blas crate scaffold | ✓ |
| 2.5 | kiln-param scaffold (Parameter + AmpPolicy + content hash) | ✓ |
| 2.5.4 | Parameter::replace_backward_storage (anti-pattern 11 preserving) | ✓ |

## Phase 5 — kiln-graph capture surface

| Phase | Deliverable | Status |
|---|---|:-:|
| 5 | kiln-graph crate (CapturedGraph + CaptureSession) | ✓ |
| 5.1 | kiln-graph-cuda, kiln-graph-metal, kiln-graph-vulkan scaffolds | ✓ |

## Phase 6 — autograd + optim

| Phase | Deliverable | Status |
|---|---|:-:|
| 6a | kiln-autograd (Tape + GradStore + BackwardOp) | ✓ |
| 6a.1 | BackwardOps for add/sub/mul/div + matmul | ✓ |
| 6a.2 | BackwardOps for sigmoid, silu, softmax | ✓ |
| 6a.3 | BackwardOps for cross_entropy + embedding | ✓ |
| 6a.4 | RMSNorm BackwardOp | ✓ |
| 6a.5 | BackwardOps for index_select, scatter_add, cast | ✓ |
| 6a.6 | BackwardOps for reduce, masked_fill, l2_norm | ✓ |
| 6a.7 | BackwardOps for SwiGLU + RoPE | ✓ |
| 6a.8 | end-to-end Tape integration with real BackwardOps | ✓ |
| 6a.9 | tiny-net manual-SGD training demo | ✓ |
| 6.5 | kiln-optim (OptimStep + AdamW CPU) | ✓ |
| 6.5.1 | Sgd + Lion/Muon scaffolds | ✓ |
| 6.5.2 | SGD master-write to Parameter | ✓ |
| 6.5.3 | AdamW master-write to Parameter | ✓ |
| 6.5.4 | end-to-end Parameter-based training demo | ✓ |
| 6.5.5 | Lion optimizer implementation | ✓ |
| 6.5.6 | Muon optimizer implementation | ✓ |
| 6a.10 | ConcatBackward | ✓ |

## Phase 4 sampler chain — 12 / 12 menu items shipped

| # | Step | Phase |
|---|------|-------|
| 1 | repetition penalty | 1.49 |
| 2 | frequency penalty | 1.49 |
| 3 | presence penalty | 1.49 |
| 4 | DRY | 1.56 |
| 5 | n-gram block | 1.54 |
| 6 | logit_bias | 1.54 |
| 7 | temperature | 1.48 |
| 8 | top-K | 1.48 |
| 9 | top-P | 1.48 |
| 10 | min-P | 1.53 |
| 11 | typical-P | 1.53 |
| 12 | Mirostat 2.0 | 1.55 |
| 13 | XTC | 1.57 |
| 14 | Gumbel-max categorical sample (terminal) | 1.58 |

(Only grammar-mask remains — separate schema-compiler effort.)

## kiln-autograd BackwardOps — 41 ops covering every differentiable forward

| Phase | BackwardOps |
|-------|-------------|
| 6a.1 | AddBackward, SubBackward, MulBackward, DivBackward, MatmulBackward |
| 6a.2 | SigmoidBackward, SiluBackward, SoftmaxLastDimBackward |
| 6a.3 | CrossEntropyBackward, EmbeddingBackward |
| 6a.4 | RmsNormBackward |
| 6a.5 | IndexSelectBackward, ScatterAddBackward, CastBackward |
| 6a.6 | ReduceBackward (sum_all/mean_all/sum_axis/mean_axis), MaskedFillBackward, L2NormBackward |
| 6a.7 | MulSigmoidGateBackward (SwiGLU), RopeBackward |
| 6a.10 | ConcatBackward |
| 1.63 | DropoutBackward |
| 1.64 | BroadcastToBackward |
| 1.65 | LayerNormBackward |
| 1.66 | GeluBackward |
| 1.71 | WhereSelectBackward |
| 1.72 | AbsBackward, NegBackward, ExpBackward, LnBackward, SqrtBackward |
| 1.73 | ClampBackward, PowBackward |
| 1.76 | SinBackward, CosBackward, TanBackward |
| 1.78 | StackBackward |
| 1.79 | TanhBackward, ReluBackward |
| 1.81 | RepeatBackward |

Non-differentiable forward ops (correctly return `bwd: None`):
`argmax_last_dim`, `causal_mask`, `top_k`, `one_hot`, `linspace`,
`arange`, `max_axis`, `min_axis`, `cumsum` (cumsum could have a
backward — reverse cumsum of dy — but isn't shipped yet), every
LogitProcessor + GumbelSampler in the sampler chain.

Each is parity-tested against finite-difference reference values
where applicable. Non-differentiable ops (argmax, causal_mask, the
sampler chain) correctly return `bwd: None`.

## What lands next

The substrate-side work is complete. Phase 7 is the **real candle
removal** — replacing the per-backend kernel-crate `Arc<CudaDevice>`
and `Arc<MetalDevice>` handles with direct cudarc / metal-rs handles,
and migrating `crates/kiln-train`'s `HashMap<candle_core::TensorId,
…>` to `kiln_tensor::TensorId`. None of that needs new substrate
infra — every contract (autograd, parameter slot coherence,
optimizer master-write, sampler chain) is already settled and
parity-tested.
