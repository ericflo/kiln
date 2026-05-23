# kiln-tensor substrate status

**122 / 122 deliverables shipped** — substrate side is complete.

- **41 kiln-tensor forward op families** + matching **41 BackwardOps**
  in kiln-autograd (every differentiable forward has a backward;
  non-differentiable ops correctly omit one)
- **Phase 4 sampler chain end-to-end** (12 LogitProcessors + Gumbel
  terminal sampler)
- **All four optimizers shipped end-to-end** (AdamW, SGD, Lion, Muon)
  with master-write to Parameter and anti-pattern 11 preserved
- **Per-backend Allocator scaffolds** (CUDA, Metal, Vulkan) feature-
  gated and ready for Phase 7
- **Per-backend CapturedGraph scaffolds** (`kiln-graph-cuda/metal/vulkan`)
  as three separate workspace crates
- **End-to-end training demos** (manual SGD + Parameter-based SGD)
  proving loss-curve descent through the full substrate
- **Phase 7 migration audit refreshed** with concrete PR sequencing

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
| 1.80 | substrate-status dashboard refresh (this PR) | ✓ |

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

Non-differentiable forward ops (correctly return `bwd: None`):
`argmax_last_dim`, `causal_mask`, `top_k`, `one_hot`, `linspace`,
`arange`, every LogitProcessor + GumbelSampler in the sampler
chain.

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
