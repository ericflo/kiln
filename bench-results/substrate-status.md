# kiln-tensor substrate status

**89 / 89 deliverables shipped** — Phase 4 sampler chain end-to-end
complete; **kiln-autograd BackwardOp coverage at 22 / 22 differentiable
forward ops**.

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
| 1.60 | substrate-status dashboard refresh (this PR) | ✓ |

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

## Phase 6a backward ops — 22 / 22 differentiable forward ops covered

| Phase | BackwardOps |
|-------|-------------|
| 6a.1 | AddBackward, SubBackward, MulBackward, DivBackward, MatmulBackward |
| 6a.2 | SigmoidBackward, SiluBackward, SoftmaxLastDimBackward |
| 6a.3 | CrossEntropyBackward, EmbeddingBackward |
| 6a.4 | RmsNormBackward |
| 6a.5 | IndexSelectBackward, ScatterAddBackward, CastBackward |
| 6a.6 | ReduceBackward (sum_all/mean_all/sum_axis/mean_axis), MaskedFillBackward, L2NormBackward |
| 6a.7 | MulSigmoidGateBackward (SwiGLU), RopeBackward |

Each is parity-tested against finite-difference reference values
where applicable. Non-differentiable ops (argmax, causal_mask, the
sampler chain) correctly return `bwd: None`.

## Phase 2 / 2.5 / 5 / 6a / 6.5 — new crates

| Phase | Deliverable | Status |
|---|---|:-:|
| 2.1 | kiln-blas production API sketch (AlgoCache + WorkspacePool) | ✓ |
| 2.2 | kiln-mps crate scaffold | ✓ |
| 2.3 | kiln-vulkan-blas crate scaffold | ✓ |
| 2.5 | kiln-param scaffold (Parameter + AmpPolicy + content hash) | ✓ |
| 5 | kiln-graph crate (CapturedGraph + CaptureSession) | ✓ |
| 6a | kiln-autograd (Tape + GradStore + BackwardOp) | ✓ |
| 6.5 | kiln-optim (OptimStep + AdamW CPU) | ✓ |
| 6.5.1 | Sgd + Lion/Muon scaffolds | ✓ |
