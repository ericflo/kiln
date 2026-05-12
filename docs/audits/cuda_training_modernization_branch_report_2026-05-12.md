# CUDA Training Modernization Branch Report

**Date:** 2026-05-12
**Branch:** `main` (CUDA modernization merged from `cuda-training-modernization`)
**Reference report:** `docs/audits/vulkan_training_branch_report_2026-05-11.md`
**Baseline:** `ab0963af` - `Synchronize Metal GDN streaming tiles`
**Status:** Mainline CUDA training modernization ledger. CUDA now tracks the latest Vulkan/Metal
training process changes on `main`; remaining work is still incremental and validation-gated.

## Branch Progress

| Commit | Slice | Evidence |
| --- | --- | --- |
| `c782a854` | Baseline CUDA modernization ledger | This report, derived from the Vulkan branch completion criteria. |
| `6ad88446` | Backend training capability telemetry | `BackendRuntime::training_capabilities`, CUDA/Vulkan profiles, `ModelRunner` startup log. |
| `0b1b8d2f` | CUDA TensorId residency lifecycle registry | CUDA implements register/has/update/evict metadata hooks; no side-buffer ownership claimed. |
| `c67de64f` | Registered CUDA LoRA delta path | `CudaBackend::lora_delta_resident` engages for registered CUDA A/B and delegates to candle CUDA autograd. |
| `6dba1644` | CUDA training projection hook | `CudaBackend::linear_prefill_apply` routes compatible CUDA projection matmuls through the backend seam using candle CUDA autograd. |
| `89926670` | CUDA runtime hook parity tests | Adds A6000-runnable tests for the projection hook and registered LoRA delta hook. |
| `4aee22f9` | CUDA offset training matmul hook | Adds `linear_prefill_apply_offset` for chunked CUDA matmuls, tested but not yet wired into FLCE auto-routing. |
| `b53602df` | CUDA FLCE backend provider gate | Wires CUDA FLCE backend chunk matmul behind `KILN_CUDA_FLCE=1`; default CUDA SFT remains on the existing candle CUDA Phase B chunked loss path until benchmarks justify auto-on. |
| `86291129` | CUDA optimizer dispatch explicit decline | Adds CUDA SGD/AdamW dispatch hooks that log first use and return `false` until CUDA owns a real in-place optimizer update. |
| `6a0824bd` | CUDA attention fast-path training guard | CUDA FlashAttention prefill/paged decode decline autograd-tracked tensors until a CUDA attention op with backward is wired. |
| `4d60b0a9` | CUDA A6000 preflight rejection coverage | Adds a discrete CUDA/A6000 long-context rejection test for the shared fit-before-run estimator. |
| `e1b45394` | CUDA resident optimizer kernels | Adds CUDA SGD and AdamW in-place kernels for registered contiguous CUDA F32/BF16 tensors. |
| `51e1d727` | CUDA optimizer support check | Tightens optimizer dispatch support detection so CUDA only claims supported registered device tensors. |
| `d57e5eec` | CUDA optimizer validation ledger | Records A6000 release-mode validation for resident optimizer kernels and updates the CUDA baseline. |
| `8f7041fa` | CUDA trainer optimizer dispatch proof | Adds backend dispatch counters and a CUDA trainer test proving `optimizer_step_from_map` reaches resident SGD/AdamW kernels. |
| `a60ed1e5` | CUDA trainer dispatch validation ledger | Records A6000 release-mode validation for trainer-level optimizer dispatch. |
| `0db3490b` | CUDA optimizer adapter-save proof | Extends the CUDA trainer test to save PEFT safetensors after resident optimizer dispatch and compare saved weights to updated CUDA Vars. |
| `accff468` | CUDA optimizer adapter-save validation ledger | Records A6000 release-mode validation for CUDA optimizer adapter-save behavior. |
| `8ea44f01` | CUDA training projection routing proof | Adds CUDA linear/offset dispatch counters and a trainer test proving projection matmuls plus FLCE chunk matmuls reach backend hooks. |
| `bee4af4f` | CUDA projection routing validation ledger | Records A6000 release-mode validation for CUDA projection/FLCE trainer routing. |
| `39ef69c1` | CUDA attention training fallback proof | Counts tracked FlashAttention declines and extends the CUDA trainer routing test to prove differentiable candle-CUDA attention fallback. |
| `e4fdb98e` | CUDA attention fallback validation ledger | Records A6000 release-mode validation for tracked FlashAttention decline plus candle-CUDA training fallback. |
| `84f29526` | Real Qwen3.5-4B CUDA SFT smoke ledger | Records A6000 release-mode `kiln-bench` validation on downloaded Qwen3.5-4B weights. |
| `1142aabf` | CUDA Qwen SFT smoke script | Adds `scripts/cuda_qwen_sft_smoke.sh` so the real Qwen3.5-4B CUDA one-step SFT validation is repeatable. |
| `d12fcafd` | CUDA Qwen smoke script validation ledger | Records A6000 release-mode validation of the repeatable real Qwen3.5-4B SFT smoke script. |
| `de34a2ee` | CUDA training tensor shell | Adds a CUDA-only `CudaTrainTensor` boundary that rejects CPU tensors and delegates resident SGD/AdamW updates to CUDA kernels. |
| `d9aa352a` | CUDA train tensor metadata | Adds op IDs, parameter `TensorId`, `requires_grad`, and detach metadata to the CUDA training tensor boundary. |
| `d57d499d` | CUDA training autograd scaffold | Adds a CUDA backward-op trait, reverse-topology traversal, per-`TensorId` grad store, and CUDA tensor gradient accumulation for synthetic training graphs. |
| `23efaecb` | CUDA training add backward op | Adds a CUDA tensor add op with a backward rule proving gradients route to both parameters and accumulate through shared inputs. |
| `0edef988` | CUDA training mul backward op | Adds a CUDA tensor multiply op with product-rule backward coverage using saved input values. |
| `3e0e32be` | CUDA training sum backward op | Adds scalar `cuda_sum_all` reduction with broadcast-gradient backward coverage, including `sum(x * x)` graph validation. |
| `1ce54012` | CUDA training matmul backward op | Adds 2D CUDA matmul with backward coverage for `sum(lhs @ rhs)` gradients to both operands. |
| `7a440ddf` | CUDA native SGD step helper | Connects `CudaGradStore` to resident SGD kernels and proves a tiny native CUDA loss decreases after one update. |
| `cca827fe` | CUDA native AdamW step helper | Adds CUDA AdamW config/state helpers, connects `CudaGradStore` to resident AdamW kernels, and proves a tiny native CUDA loss decreases. |
| `aeb64efe` | CUDA native linear train step | Adds a `kiln-train` CUDA helper that runs a minimal linear sum-square AdamW step through native CUDA tensor/autograd primitives. |
| `7399abcb` | CUDA training arena accounting | Adds a conservative `CudaTrainArena` that owns step-lifetime CUDA tensor handles and tracks approximate allocation bytes by dtype/shape. |
| `f4eff9b0` | CUDA linear train arena bridge | Adds an arena-backed CUDA linear AdamW train-step variant and proves the train-crate bridge accounts forward/loss tensors. |
| `68b798a3` | CUDA training tensor save boundary | Adds a minimal safetensors save helper for named CUDA training tensors and proves saved weights reflect post-AdamW CUDA state. |
| `86ba8916` | CUDA train sub and mean ops | Adds native CUDA training subtraction and all-element mean ops with backward coverage, extending the synthetic autograd surface toward Vulkan parity. |
| `299f0c1c` | CUDA train reshape and transpose ops | Adds CUDA training reshape and 2D transpose ops with backward coverage for shape/view movement used by transformer compositions. |
| `1ef4791e` | CUDA train scale and div ops | Adds CUDA scalar scale and elementwise division ops with analytic backward coverage for attention/loss composition parity. |
| `59eb74c3` | CUDA train dtype cast op | Adds CUDA F32/BF16 dtype conversion with passthrough backward coverage for mixed-precision native graph composition. |
| `cf53a978` | CUDA train batched matmul op | Adds exact-batch CUDA batched matmul with backward coverage, extending the native tape toward SDPA and GDN compositions. |
| `3436153b` | CUDA train softmax op | Adds F32 CUDA softmax-last-dim with analytic backward coverage for native SDPA composition. |
| `3ccfaa28` | CUDA train SiLU op | Adds F32 CUDA SiLU with analytic backward coverage for native SwiGLU/GDN gate composition. |
| `76196d4a` | CUDA train sigmoid op | Adds F32 CUDA sigmoid with analytic backward coverage for attention gate and GDN beta composition. |
| `a225eb36` | CUDA train repeat-KV heads op | Adds F32 rank-3 repeat-KV-heads with backward group summation for native GQA/SDPA composition. |
| `fe357495` | CUDA train narrow last-dim op | Adds F32 last-dimension narrow with backward zero-padding scatter for native routing and chunk slicing. |
| `0eee38b1` | CUDA train row index-select op | Adds F32 rank-2 row gather with backward scatter-add, including duplicate-index accumulation coverage. |
| `65491a70` | CUDA train attention permute ops | Adds F32 rank-3 `rows,heads -> heads,rows` and inverse permutes with inverse-backward coverage for native SDPA composition. |
| `9d2e945d` | CUDA train last-two transpose op | Adds rank-N last-two-dimension transpose with inverse-backward coverage for native `Q @ K.T` SDPA composition. |
| `55527848` | CUDA train unmasked SDPA composition | Adds an F32 native CUDA train-shell SDPA composition from permute, repeat-KV, last-two transpose, batched matmul, scale, softmax, and inverse permute ops, with backward smoke coverage to Q/K/V. |
| `b90cbd88` | CUDA train causal mask op | Adds F32 rank-3 additive causal masking with constant-mask backward passthrough coverage for native SDPA prefill composition. |
| `306be815` | CUDA train causal SDPA composition | Wires the causal mask into a native F32 CUDA train-shell SDPA prefill helper and proves masked first-row output plus backward gradients to Q/K/V. |
| `d6247dff` | CUDA train RMSNorm op | Adds F32 Qwen RMSNorm with analytic `dX` backward and frozen-base-weight coverage, matching the Vulkan training contract. |
| `a70121be` | CUDA train RoPE op | Adds F32 rank-3 RoPE with inverse-rotation backward coverage and passthrough tail dimensions for native attention composition. |
| `8ff4817a` | CUDA train embedding lookup op | Exposes rank-2 embedding lookup over CUDA row gather with scatter-add backward coverage for duplicate token IDs. |
| `0b28edd7` | CUDA train SwiGLU MLP composition | Adds a native CUDA train-shell SwiGLU MLP helper composed from matmul, SiLU, multiply, and matmul, with gradient smoke coverage to input and all three weights. |
| `e87f35ce` | CUDA train FullAttention layer composition | Adds a native CUDA train-shell FullAttention layer helper composed from RMSNorm, q/k/v projections, QK norm, RoPE, causal SDPA, output gate, output projection, residuals, post-attention RMSNorm, and SwiGLU MLP. |
| `c8dd47d3` | CUDA train FullAttention AdamW bridge | Adds a `kiln-train` one-step AdamW helper that runs loss/backward through the native CUDA FullAttention layer and updates supplied CUDA parameters. |
| `df549c7a` | CUDA train tiny FullAttention model bridge | Adds a tiny native CUDA model step bridge that runs embedding lookup, FullAttention, final RMSNorm, LM head projection, loss/backward, and AdamW against supplied CUDA parameters. |
| `3edaef53` | CUDA LoRA adapter save bridge | Adds CUDA-native LoRA pair/layer holders, LoRA AdamW state allocation, and PEFT-keyed safetensors adapter save for CUDA training tensors. |
| `0ab549f1` | CUDA LoRA adapter directory save | Adds the CUDA-native adapter directory final-save contract: `adapter_model.safetensors` plus PEFT `adapter_config.json`. |
| `7748c248` | CUDA native model weight import | Adds a typed CUDA-native model weight view imported from `GpuWeights`, including FullAttention and LinearAttention layer weight holders. |
| `7f525f29` | CUDA LoRA linear train bridge | Adds a native CUDA LoRA linear helper for `input @ base + scale * input @ A.T @ B.T` and proves AdamW updates the LoRA pair. |
| `e4c5c410` | CUDA FullAttention LoRA train bridge | Adds a native CUDA FullAttention-layer helper that applies trainable LoRA deltas across attention and MLP projections over frozen base weights. |
| `63ef6edd` | CUDA FullAttention LoRA model step | Adds a model-level native CUDA LoRA AdamW step over `CudaModelWeights` for FullAttention-only layer stacks. |
| `a97fd20c` | CUDA GDN LoRA adapter save slots | Extends CUDA LoRA allocation and safetensors/config save plumbing to include GDN `in_proj_qkv`, `in_proj_z`, and `gdn_out_proj` slots. |
| `09b6a414` | CUDA model LoRA initialization | Adds deterministic CUDA LoRA initialization from `CudaModelWeights`, populating FullAttention and GDN layer slots. |
| `e22a1f6a` | CUDA FullAttention LoRA token train loop | Adds a multi-epoch token-sequence training loop over the native CUDA FullAttention-only LoRA model step and resident AdamW state. |
| `3606aac6` | CUDA token train adapter save wrapper | Adds a wrapper that initializes CUDA LoRA, trains token sequences, and saves a PEFT adapter directory. |
| `985484e1` | CUDA native SFT train entrypoint | Adds a `cuda_native_sft_train` entry point matching the Vulkan trainer shape for CUDA FullAttention-only native SFT. |
| `62d38ee0` | CUDA native SFT server route | Adds `KILN_CUDA_NATIVE_TRAINING=1` routing in the training queue to call `cuda_native_sft_train`, plus the matching native-GRPO rejection. |
| `535561c1` | CUDA native SFT unsupported-layer preflight | Rejects empty or LinearAttention/GDN CUDA-native SFT models before LoRA/AdamW allocation. |
| `0cbd130d` | CUDA GDN training state foundation | Adds zeroed CUDA LinearAttention recurrent/conv state holders and a GDN-layer counter for future native GDN threading. |
| `9c823d10` | CUDA GDN LoRA input projections | Adds native CUDA RMSNorm plus LoRA-enabled GDN q/k/v/z input projections with gradient coverage for the trainable LoRA slots. |
| `bf10bf5b` | CUDA train last-dim bias op | Adds autograd-safe CUDA broadcast bias add with leading-dimension gradient reduction, needed for native GDN gate bias composition. |
| `8f2f269a` | CUDA train exp and softplus ops | Adds autograd-safe CUDA exp and softplus primitives needed for native GDN gate composition. |
| `c7a42fd3` | CUDA train last-dim multiply op | Adds autograd-safe CUDA broadcast multiply with leading-dimension gradient reduction, needed for native GDN decay scaling. |
| `476c7d8a` | CUDA train last-dim broadcast multiply | Adds autograd-safe `[..., D] * [..., 1]` multiply with last-dimension gradient reduction for per-token GDN gate scaling. |
| `77472a62` | CUDA GDN gate composition | Adds native CUDA beta/g gate composition from GDN a/b projections, dt bias, softplus, and decay scaling. |
| `955bdad4` | CUDA train row narrow op | Adds autograd-safe CUDA row slicing with backward zero-padding, a building block for causal conv/chunk windows. |
| `cad62355` | CUDA train row concat op | Adds autograd-safe CUDA row concatenation with backward gradient splitting, needed for causal conv padding/window composition. |
| `1dc880ad` | CUDA train zero-state causal conv1d | Adds a native CUDA training-shell causal depthwise conv1d composition for fresh-state prefill windows. |
| `40b7f943` | CUDA train zero-state conv1d state update | Adds CUDA-native next-state construction for fresh-state causal conv prefill. |
| `d0797943` | CUDA train stateful causal conv1d | Generalizes CUDA train causal depthwise conv1d to consume and return explicit conv state. |
| `4c951645` | CUDA GDN conv qkv bridge | Adds a GDN train bridge for LoRA qkv projection through stateful causal conv, SiLU, q/k/v split, and next conv state. |
| `a3c71a00` | CUDA GDN output projection bridge | Adds GDN gated RMSNorm plus LoRA output projection composition with gradient coverage for the reserved `gdn_out_proj` slot. |
| `dc3b95a3` | CUDA GDN single-token recurrence | Adds an autograd-safe single-token GDN recurrence composition that returns both recurrent output and next state. |
| `d0b0ff30` | CUDA GDN sequence recurrence | Threads the CUDA single-token GDN recurrence across a one-head sequence, preserving output rows and final recurrent state. |
| `593231ba` | CUDA GDN multi-head recurrence | Threads one-head CUDA GDN recurrence across flattened head blocks, preserving per-head output blocks and recurrent state blocks. |
| `157364a7` | CUDA GDN LoRA layer bridge | Composes GDN q/k/v conv, beta/g gates, multi-head recurrence, per-head gated RMSNorm, LoRA output projection, residual add, and next recurrent/conv states. |
| `dec251d3` | CUDA GDN model-step state threading | Adds a mixed FullAttention/GDN CUDA LoRA model-step helper that threads `CudaLinearAttentionState` through GDN layers before loss/backward/AdamW. |
| `a04fe559` | CUDA GDN-state token train loop | Adds a token-sequence training loop over the mixed CUDA LoRA model step, creating fresh GDN state per example. |
| `1252b056` | CUDA GDN-state adapter train wrapper | Adds an adapter-producing wrapper around the mixed GDN-state token train loop and validates adapter file creation. |
| `223230dd` | CUDA native SFT GDN-state routing | Routes `cuda_native_sft_train` through the mixed GDN-state model step when imported CUDA weights contain LinearAttention/GDN layers. |
| `15ba059c` | Mainline CUDA modernization merge | Merges the CUDA modernization stack onto `main` at `ab0963af`, preserving current exact SFT labeling, long-context checkpoint/preflight, Metal FLCE, and Metal GDN tile synchronization work. |
| `7b4b2b78` | Stable CUDA native Qwen smoke | Adds stable CUDA train softplus, native-CUDA bench/smoke routing, and GDN q/k L2 normalization before recurrence; fixes the real Qwen native CUDA SFT NaN. |
| `858f83ee` | Latest main remerge | Merges the newest Metal/Candle cache work from `github/main` (`ffef3bfd`, `c168b367`, `bd57e81a`) after CUDA stabilization and revalidates CUDA release tests/smoke. |
| This commit | CUDA native masked CE SFT objective | Adds a CUDA shifted-label cross-entropy autograd seed and threads tokenizer `label_mask` through the CUDA-native SFT route; validated on A6000. |

Local validation so far:

- `cargo test -p kiln-model backend::tests::portable_training_capabilities_are_conservative --lib --quiet` passed after each code slice.
- `cargo test -p kiln-server preflight_rejects_long_context_on_a6000_cuda_budget --lib --quiet` passed for CUDA/discrete fit-gate coverage.
- `cargo test -p kiln-train flce_provider --lib --quiet` passed for the CUDA opt-in gate and Vulkan auto-heuristic tests.
- `cargo test -p kiln-train flce_auto --lib --quiet` passed for the FLCE active-token floor tests.
- `git diff --check` passed before each code commit.
- `cargo fmt --all --check` is blocked in this workspace because the active musl Rust toolchain lacks `rustfmt`.
- Local CUDA-feature tests are blocked in this workspace because `cudarc` cannot find `nvcc`.
- RunPod A6000 validation on pod `pdauarcdn9k62l`, release mode with `KILN_CUDA_ARCHS=86`, passed:
  - `cargo test --release -p kiln-model --features cuda cuda_training_capabilities_do_not_overclaim_native_training --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_resident_activation_registry_lifecycle --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_linear_prefill_apply_matches_candle_cuda_matmul --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_linear_prefill_apply_offset_matches_candle_cuda_chunk --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_registered_lora_delta_matches_candle_cuda_reference --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_checkpointed_loss_matches_standard --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_flce_parity_vs_naive_loss --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda flce_provider --lib --quiet`
  - `KILN_CUDA_FLCE=1 cargo test --release -p kiln-train --features cuda test_flce_parity_vs_naive_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_optimizer_dispatch_hooks_decline_until_owned_kernel_exists --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_flash_attention_declines_tracked_training_tensors --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_sgd_step_resident_round_trip_f32 --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_adamw_step_resident_round_trip_f32 --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_sgd_and_adamw_resident_round_trip_bf16 --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda test_checkpointed_loss_matches_standard --lib --quiet` re-run after optimizer kernel wiring
  - `cargo test --release -p kiln-train --features cuda cuda_optimizer_step_from_map_engages_backend_kernels --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_optimizer_step_from_map_engages_backend_kernels --lib --quiet` re-run after adding adapter-save safetensors comparison
  - `cargo test --release -p kiln-train --features cuda cuda_training_forward_uses_projection_and_flce_backend_hooks --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_training_forward_uses_projection_and_flce_backend_hooks --lib --quiet` re-run after adding tracked FlashAttention-decline assertion
  - `KILN_SPEC_METHOD=off KILN_USE_FLCE=1 cargo run --release --features cuda --bin kiln-bench -- --model-path /workspace/qwen3.5-4b --prompt-tokens 8 --max-output-tokens 1 --training-steps 1 --paged --quiet` passed after downloading `Qwen/Qwen3.5-4B` with `hf download`; CUDA backend loaded the real model, completed one SFT step with `loss=1.598035`, `2.85s/step`, and `18952 MB` peak VRAM.
  - `scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --skip-build` passed on the same A6000/model checkout with `loss=1.598035`, `2.65s/step`, and `18952 MB` peak VRAM.
  - `cargo test --release -p kiln-model --features cuda cuda_train_tensor --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_backward --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA autograd scaffolding
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA add backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA multiply backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA sum backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA matmul backward coverage
  - `cargo test --release -p kiln-model --features cuda cuda_native_sgd_step_decreases_sum_square_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding native SGD step coverage
  - `cargo test --release -p kiln-model --features cuda cuda_native_adamw_step_decreases_sum_square_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding native AdamW step coverage
  - `cargo test --release -p kiln-train --features cuda cuda_linear_adamw_train_step_decreases_loss --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train_arena --lib --quiet`
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA training arena accounting
  - `cargo test --release -p kiln-train --features cuda cuda_linear_adamw_train_step --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_linear_weight_save_reflects_updated_cuda_tensor --lib --quiet`
  - `cargo test --release -p kiln-train --features cuda cuda_linear_adamw_train_step --lib --quiet` re-run after adding the minimal CUDA safetensors save-boundary proof
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA subtraction and mean-all backward coverage; 19 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA reshape and transpose backward coverage; 21 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA scale and division backward coverage; 23 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA dtype-cast backward coverage; 24 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA batched matmul backward coverage; 25 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA softmax-last-dim backward coverage; 26 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA SiLU backward coverage; 27 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA sigmoid backward coverage; 28 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA repeat-KV-heads backward coverage; 29 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA narrow-last-dim backward coverage; 30 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA row index-select backward coverage; 31 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA attention permute backward coverage; 33 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA last-two transpose backward coverage; 34 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA unmasked SDPA composition backward smoke coverage; 35 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA causal-mask backward coverage; 36 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA causal SDPA prefill composition coverage; 37 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA RMSNorm analytic backward coverage; 38 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA RoPE inverse-backward coverage; 39 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA embedding lookup backward coverage; 40 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA SwiGLU MLP composition coverage; 41 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-model --features cuda cuda_train --lib --quiet` re-run after adding CUDA FullAttention layer composition coverage; 42 CUDA train-shell tests passed.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_adamw_train_step_updates_projection_weight --lib --quiet` passed after adding the train-crate CUDA FullAttention AdamW bridge.
  - `cargo test --release -p kiln-train --features cuda cuda_tiny_full_attention_model_adamw_step_updates_lm_head --lib --quiet` passed after adding the tiny native CUDA FullAttention model bridge.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_adapter_save_uses_peft_keys --lib --quiet` passed after adding the CUDA-native LoRA adapter save bridge.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_adapter_save_uses_peft_keys --lib --quiet` re-run after adding the CUDA-native adapter directory/config save contract.
  - `cargo test --release -p kiln-model --features cuda cuda_model_weights_from_gpu_weights_imports_full_attention_layer --lib --quiet` passed after adding CUDA-native model weight import.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_linear_adamw_updates_lora_pair --lib --quiet` passed after adding the CUDA-native LoRA linear train bridge.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_lora_layer_adamw_updates_lora_pair --lib --quiet` passed after adding the CUDA-native FullAttention LoRA train bridge.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_lora_model_step_updates_lora_pair --lib --quiet` passed after adding the CUDA-native FullAttention LoRA model step.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_adapter_save_includes_gdn_slots --lib --quiet` passed after adding CUDA GDN LoRA adapter save slots.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` passed after adding CUDA model LoRA initialization.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_lora_model_step_updates_lora_pair --lib --quiet` re-run after adding the CUDA-native FullAttention LoRA token-sequence train loop.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_lora_model_step_updates_lora_pair --lib --quiet` re-run after adding the CUDA token-train adapter save wrapper.
  - `cargo test --release -p kiln-train --features cuda cuda_full_attention_lora_model_step_updates_lora_pair --lib --quiet` re-run after adding the CUDA-native SFT train entry point.
  - `cargo test --release -p kiln-server --features cuda native_training_env_enabled --lib --quiet` passed after adding `KILN_CUDA_NATIVE_TRAINING=1` server routing; compile coverage verifies the CUDA-native trainer call is linked from `kiln-server`.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding CUDA-native SFT unsupported-layer preflight rejection.
  - `cargo test --release -p kiln-model --features cuda cuda_linear_attention_state_zeros_allocates_layer_state --lib --quiet` passed after adding CUDA GDN recurrent/conv state foundations.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding CUDA GDN LoRA input projection coverage.
  - `cargo test --release -p kiln-model --features cuda cuda_add_last_dim_bias_backward_reduces_leading_dims --lib --quiet` passed after adding CUDA train last-dim broadcast bias.
  - `cargo test --release -p kiln-model --features cuda cuda_exp_and_softplus_backward_match_cpu_reference --lib --quiet` passed after adding CUDA train exp and softplus primitives.
  - `cargo test --release -p kiln-model --features cuda cuda_mul_last_dim_weight_backward_reduces_leading_dims --lib --quiet` passed after adding CUDA train last-dim broadcast multiply.
  - `cargo test --release -p kiln-model --features cuda cuda_mul_last_dim_broadcast_backward_reduces_last_dim --lib --quiet` passed after adding CUDA train per-token last-dim broadcast multiply.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding native CUDA GDN beta/g gate composition.
  - `cargo test --release -p kiln-model --features cuda cuda_narrow_rows_backward_pads_zero_gradients --lib --quiet` passed after adding CUDA train row slicing.
  - `cargo test --release -p kiln-model --features cuda cuda_cat_rows_backward_splits_gradients --lib --quiet` passed after adding CUDA train row concatenation.
  - `cargo test --release -p kiln-model --features cuda cuda_causal_depthwise_conv1d_prefill_zero_state_matches_reference --lib --quiet` passed after adding CUDA train zero-state causal depthwise conv1d.
  - `cargo test --release -p kiln-model --features cuda cuda_causal_depthwise_conv1d_next_state_zero_state_tracks_tail --lib --quiet` passed after adding CUDA train zero-state conv next-state construction.
  - `cargo test --release -p kiln-model --features cuda cuda_causal_depthwise_conv1d_prefill_with_state_threads_state --lib --quiet` passed after adding stateful CUDA train causal depthwise conv1d.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding CUDA GDN conv qkv bridge coverage.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding CUDA GDN gated RMSNorm/output projection bridge coverage.
  - `cargo test --release -p kiln-model --features cuda cuda_gdn_single_token_recurrence_matches_reference --lib --quiet` passed after adding CUDA GDN single-token recurrence.
  - `cargo test --release -p kiln-model --features cuda cuda_gdn_single_head_sequence_recurrence_threads_state --lib --quiet` passed after adding CUDA GDN one-head sequence recurrence.
  - `cargo test --release -p kiln-model --features cuda cuda_gdn_multi_head_sequence_recurrence_threads_head_blocks --lib --quiet` passed after adding CUDA GDN flattened multi-head recurrence.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after adding CUDA GDN LoRA layer bridge coverage.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_model_step_with_gdn_state_threads_gdn_state --lib --quiet` passed after adding mixed CUDA LoRA model-step GDN state threading.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_model_step_with_gdn_state_threads_gdn_state --lib --quiet` re-run after adding the mixed CUDA GDN-state token-sequence train loop.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_model_step_with_gdn_state_threads_gdn_state --lib --quiet` re-run after adding the mixed CUDA GDN-state adapter train/save wrapper.
  - `cargo test --release -p kiln-train --features cuda cuda_init_lora_layers_populates_full_attention_and_gdn_slots --lib --quiet` re-run after routing CUDA-native SFT through the mixed GDN-state step for LinearAttention/GDN layers.
  - `cargo test --release -p kiln-server --features cuda native_training_env_enabled --lib --quiet` re-run after routing CUDA-native SFT through the mixed GDN-state step; compile coverage verifies the server route links the updated CUDA-native trainer.
  - `cargo test --release -p kiln-model --features cuda cuda_exp_and_softplus_backward_match_cpu_reference --lib --quiet` re-run after replacing CUDA train softplus with the stable `max(x,0)+log(1+exp(-abs(x)))` form.
  - `cargo test --release -p kiln-train --features cuda flce_provider --lib --quiet` passed after merging the CUDA FLCE opt-in provider tests with current main's Vulkan/Metal FLCE changes.
  - `cargo test --release -p kiln-train --features cuda cuda_lora_model_step_with_gdn_state_threads_gdn_state --lib --quiet` re-run after adding CUDA-native GDN q/k L2 normalization before recurrence.
  - `cargo test --release -p kiln-server --features cuda native_training_env_enabled --lib --quiet` re-run after mainline merge and native CUDA bench/smoke routing.
  - `KILN_CUDA_ARCHS=86 scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --native-cuda` passed on A6000 after q/k normalization: native CUDA SFT produced `loss=69478864.000000`, `53.26s/step`, `41384 MB` peak VRAM, and non-null JSON `"training": {"num_steps": 1, ...}`.
  - After merging latest `github/main` (`858f83ee`), the same focused release tests passed again and `KILN_CUDA_ARCHS=86 scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --native-cuda` passed again with `loss=69478864.000000`, `53.08s/step`, `41384 MB` peak VRAM, and non-null training JSON.
  - After fast-forwarding to current `github/main` (`16389b1c`), `cargo test --release -p kiln-model --features cuda cuda_shifted_cross_entropy_loss_matches_cpu_reference --lib --quiet` passed for the CUDA shifted-label CE value/backward parity test, `cargo test --release -p kiln-train --features cuda cuda_lora_model_step_with_gdn_state_threads_gdn_state --lib --quiet` passed for the mixed GDN-state model step, and `KILN_CUDA_ARCHS=86 scripts/cuda_qwen_sft_smoke.sh --model-path /workspace/qwen3.5-4b --native-cuda` passed with masked CE SFT loss `7.271030`, `52.81s/step`, `41672 MB` peak VRAM, and non-null training JSON.
  - Debug-mode CUDA test was intentionally rejected after `nvcc -G` hit exit 137 in `kiln-flash-attn`; release mode is the required kiln CUDA path.

## Executive Summary

The Vulkan training modernization branch established the target shape for backend training:

1. reject jobs that cannot fit before training starts;
2. route training math through autograd-safe GPU paths, not decode-only fast paths;
3. chunk vocab, row, batch, and sequence work before dispatch;
4. keep trainable and checkpoint state backend-resident between steps;
5. read back trainable state only at adapter-save/checkpoint boundaries;
6. eventually replace candle's host-storage training contract with a native tensor/autograd stack.

CUDA starts from a different place than Vulkan. The current CUDA backend already benefits from
candle CUDA tensors, FlashAttention, GDN kernels, fused RMSNorm, FLCE Phase B, BF16 LoRA storage,
and several inference/decode fusions. This branch has added CUDA training capability telemetry, a
lightweight TensorId residency registry, autograd-safe candle-CUDA LoRA/projection hooks, explicit
attention training declines, and resident in-place SGD/AdamW kernels for registered CUDA tensors.
It also now has an initial CUDA-only training tensor boundary over candle CUDA storage, including
parameter metadata, detach semantics, a backward-op trait, reverse-topology traversal, and a
per-`TensorId` gradient store for a future CUDA autograd graph. CUDA add, subtract, multiply,
division, scalar scale, F32/BF16 dtype casts, sum and mean reductions, reshape, 2D transpose,
last-two transpose, 2D matmul, exact-batch batched matmul, F32 softmax-last-dim, F32 sigmoid, F32 SiLU, rank-3
repeat-KV-heads, last-dim narrow, rank-2 row index-select, and rank-3 attention permute ops prove
that the tape can propagate and accumulate gradients through real CUDA tensor ops, including product
and quotient rules, mixed-precision passthrough casts, scalar-loss reductions, shape/view movement,
projection/attention-shaped matmul, softmax Jacobian-vector products, sigmoid gates, SwiGLU
activation derivatives, GQA head-repeat gradient summation, slice-gradient scattering, row gather
scatter-add, inverse attention-layout permutes, inverse last-two transposes for `Q @ K.T`
gradients, Qwen RMSNorm analytic `dX` with frozen base weights, RoPE inverse-rotation backward, and embedding lookup scatter-add. Initial unmasked and causal SDPA helpers, a SwiGLU MLP helper, and a synthetic FullAttention layer helper now compose the native train-shell ops
end-to-end and prove loss/backward reaches attention and MLP weights. A tiny native CUDA model step now composes embedding lookup, FullAttention, final RMSNorm, LM head projection, scalar loss/backward, and AdamW. Native SGD and AdamW helpers now apply those gradients through resident optimizer kernels and
prove tiny loss decreases, and `kiln-train` has a minimal CUDA-native linear AdamW train-step
bridge. A conservative CUDA training arena now owns step-lifetime tensor handles and tracks
approximate allocation bytes, and the train-crate linear, synthetic FullAttention, and tiny model bridges can run against caller-owned arenas.
The train-crate bridge also has a minimal safetensors save boundary, a PEFT-keyed CUDA LoRA
adapter save bridge, and a CUDA-native final adapter directory save helper proving post-update
CUDA trainable tensors are read back only at save time. `crates/kiln-model/src/cuda_train.rs`
now also has a typed `CudaModelWeights::from_gpu_weights` import boundary for frozen CUDA model
weights, covering FullAttention and LinearAttention layer holders as a prerequisite for a real
native CUDA SFT route. The train-crate bridge can now apply trainable CUDA LoRA deltas over frozen
base projections with `input @ base + scale * input @ A.T @ B.T` and update those LoRA tensors
through resident AdamW. The same LoRA projection path now composes through a native CUDA
FullAttention layer, covering attention projections, output projection, and MLP projections over
frozen base weights. A model-level FullAttention-only LoRA AdamW helper now threads
`CudaModelWeights`, token embedding lookup, per-layer FullAttention LoRA, final RMSNorm, LM head,
loss/backward, and resident AdamW.
CUDA LoRA allocation and adapter save plumbing now includes the reserved GDN LoRA slots
(`in_proj_qkv`, `in_proj_z`, and `gdn_out_proj`), but the native recurrent GDN forward/backward
route is still missing.
CUDA can now deterministically initialize LoRA layers from `CudaModelWeights`, including the
FullAttention q/k/v/o and MLP slots plus the reserved GDN projection slots.
CUDA also has a small token-sequence training loop over the FullAttention-only LoRA model step,
threading resident AdamW state across examples and epochs.
That token loop now has an adapter-producing wrapper that initializes LoRA from `CudaModelWeights`,
trains resident CUDA tensors, and writes the PEFT adapter directory at the save boundary.
`cuda_native_sft_train` now mirrors the Vulkan trainer entry-point shape: import `GpuWeights` into
`CudaModelWeights`, tokenize SFT examples, preserve the tokenizer's assistant-response
`label_mask`, initialize LoRA, keep AdamW state resident, report progress, checkpoint adapters, and
save the final adapter.
The GDN train-crate bridge now covers a first full-layer composition: LoRA q/k/v/z projections,
stateful causal conv q/k/v, beta/g gates, flattened multi-head recurrence, per-head gated RMSNorm,
LoRA output projection, residual add, and explicit next recurrent/conv states. A mixed
FullAttention/GDN CUDA LoRA model-step helper now threads `CudaLinearAttentionState` through GDN
layers before final RMSNorm, LM head, loss/backward, and AdamW. A token-sequence train loop now
creates fresh CUDA GDN state per example over that mixed model step, and an adapter-producing wrapper
now saves the resulting LoRA adapter directory. `cuda_native_sft_train` now selects the mixed
GDN-state model step whenever imported CUDA weights contain LinearAttention/GDN layers. Native CUDA
model-step helpers now use shifted next-token cross-entropy over the supervised label mask when
called from SFT, while retaining the synthetic squared-logits loss for older token-sequence smoke
helpers. The CE path is still a full-logits correctness slice, not the Vulkan-equivalent chunked
FLCE route.
This is still not a full CUDA native SFT route equivalent to Vulkan's native stack, and CUDA still
does **not** have native Qwen forward/backward ops or a custom pooled allocator equivalent to the
Vulkan native training path.

The CUDA port should therefore not copy Vulkan's buffer-upload mechanics blindly. CUDA candle tensors
already live on the device, so the first useful parity target is to make CUDA training decisions
explicit, testable, and observable:

- decline inference-only kernels for tracked tensors unless they preserve autograd;
- expose CUDA training capabilities separately from decode capabilities;
- add device-resident optimizer/LoRA hooks only when they truly avoid host synchronization;
- preserve the existing long-context safety gates that Phase 10 proved necessary.

## Current CUDA Training Baseline

| Area | Current evidence | Status |
| --- | --- | --- |
| Fit gate | `crates/kiln-server/src/training_preflight.rs` has a shared working-set estimator, treats CUDA/discrete devices as `WeightResidency::SingleCopy`, accepts small A6000 payloads, and rejects 64k-token Qwen3.5-4B SFT on the A6000 budget. | Present, shared with Vulkan. |
| FLCE | `crates/kiln-flce-kernel` implements Phase B chunked-vocab CustomOp; `PHASE10_CLOSURE.md` records T=8192 A40 closure and A6000 prediction. | Present. Must remain default for SFT. |
| RMSNorm training | `forward.rs::rms_norm` routes autograd tensors to `fused_rmsnorm_with_autograd` only behind the 47 GiB gate; small GPUs fall back. | Present with safety gate. |
| Attention training | CUDA FlashAttention prefill/paged-decode hooks decline tracked tensors; the trainer routing test proves tracked FlashAttention is offered, declined, and followed by differentiable candle-CUDA attention fallback. | Honest GPU fallback present; native CUDA attention backward op still missing. |
| LoRA precision | `compute_lora_delta` casts A/B to `x.dtype()`; LoRA Vars initialize as BF16. `PHASE10_LORA_PRECISION_STUDY.md` closed performance as null but accepted parity/safety. | Present. |
| CUDA decode LoRA | `CudaBackend::lora_decode_add` declines tracked tensors and only runs the forward-only fused add for inference. | Correct for safety, not a training acceleration. |
| Resident activation registry | CUDA implements `register`, `has`, `update`, and `evict` TensorId metadata hooks while keeping `resolve` conservative unless a caller already owns the tensor. | Present as lifecycle/telemetry registry; no false side-buffer ownership claimed. |
| Device optimizer dispatch | CUDA implements resident in-place SGD and AdamW kernels for registered contiguous CUDA F32/BF16 tensors, with first-use telemetry, dispatch counters, and fallback declines for unsupported tensors. | Kernel path, trainer-level engagement, saved adapter contents, and one-step real Qwen3.5-4B SFT smoke proven. |
| Autograd-safe projection backend op | `CudaBackend::linear_prefill_apply` and `linear_prefill_apply_offset` route compatible CUDA matmuls through candle CUDA autograd and expose dispatch counters. | Present for direct parity tests, trainer-level projection/FLCE routing, and one-step real-model smoke. |
| Native CUDA training stack | `crates/kiln-model/src/cuda_train.rs` provides an initial CUDA-only tensor shell over candle CUDA storage with op IDs, parameter `TensorId`, `requires_grad`, detach semantics, a backward-op trait, reverse-topology traversal, per-parameter grad storage, CUDA add/sub/mul/div/scale/cast/sum/mean/reshape/transpose/last-two-transpose/matmul/batched-matmul/softmax/shifted-cross-entropy/sigmoid/SiLU/repeat-KV/narrow/index-select/embedding-lookup/attention-permute/causal-mask/RMSNorm/RoPE backward ops, last-dim broadcast ops, causal conv state helpers, single-token/one-head/multi-head GDN recurrence, unmasked and causal SDPA plus SwiGLU MLP and synthetic FullAttention layer composition helpers, resident SGD/AdamW optimizer delegation, tiny optimizer loss-decrease proofs, conservative arena allocation accounting, and typed `CudaModelWeights` import from existing CUDA `GpuWeights`. `crates/kiln-train/src/cuda_train.rs` adds CUDA LoRA pair/layer holders, deterministic LoRA initialization from `CudaModelWeights`, LoRA AdamW state allocation for FullAttention and reserved GDN slots, native CUDA LoRA linear, FullAttention-layer, GDN-layer, FullAttention-only and mixed FullAttention/GDN model-step helpers, FullAttention-only and mixed GDN-state token-sequence train loops, FullAttention-only and mixed GDN-state adapter-producing token-train wrappers, a CUDA-native SFT entry point that selects the mixed GDN-state path for hybrid models, plus minimal linear and synthetic FullAttention AdamW bridges with caller-owned arena support, PEFT-keyed LoRA safetensors, and final adapter directory/config save helpers. The SFT entrypoint imports existing CUDA `GpuWeights`, tokenizes examples, trains resident CUDA LoRA tensors with masked shifted-label CE, reports progress, checkpoints, saves at the adapter boundary, and is routable from `KILN_CUDA_NATIVE_TRAINING=1` for SFT. There is still no chunked native CUDA FLCE-equivalent loss or custom pooled training allocator. | Initial tensor/autograd/optimizer/arena/model-import/save boundary, masked CE SFT objective slice, train-crate SFT entrypoint, server SFT routing, GDN layer bridge, mixed GDN state-threading model step, mixed GDN-state token train loop, mixed adapter save wrapper, hybrid SFT routing, and one-step real-model validation present; chunked FLCE and allocator parity still missing. |

## Phase Plan

### Phase C0: Capability and Telemetry Baseline

Goal: make the CUDA training surface explicit before changing math.

Tasks:

- add CUDA training capability logging at startup or first backend use;
- expose which training paths are CUDA-native, candle-CUDA, or declined;
- add tests that CUDA inference-only hooks decline tracked tensors;
- document the exact completion criteria copied from the Vulkan report.

Acceptance:

- CPU-host tests can verify default trait declines without requiring a GPU;
- CUDA-feature builds compile with the new capability surface;
- operator logs can distinguish "safe candle CUDA training path" from "backend-owned CUDA training
  kernel engaged."

### Phase C1: CUDA Resident Registry Semantics

Goal: implement the `BackendRuntime` resident hooks for CUDA without adding false host-copy claims.

Design constraint: candle CUDA tensors are already device-resident. A CUDA "registry" should key
logical training tensors and provide lifecycle/telemetry first; it should only add side buffers when
we need pointer stability, stale-host semantics, or custom kernels.

Tasks:

- implement `supports_resident_activation`, `register_resident_activation`, `has`, `evict`, and
  `update` for CUDA as a lightweight TensorId registry;
- keep `resolve_resident_activation` conservative until there is a real custom buffer source;
- add lifecycle tests that run without a CUDA device if possible, and CUDA-feature tests otherwise;
- wire first-dispatch logs for LoRA and optimizer hooks.

Acceptance:

- register/has/evict/re-register semantics match Vulkan's caller contract;
- no training path silently claims stale candle storage unless CUDA owns an alternate buffer;
- checkpointed training tests still pass with CUDA hooks compiled in.

### Phase C2: Autograd-Safe CUDA LoRA Delta Path

Goal: make CUDA LoRA training engagement explicit and preserve gradients.

Current fallthrough already uses candle CUDA `broadcast_matmul`, which is autograd-safe. The first
CUDA implementation should wrap or report this path rather than replacing it with a leaf tensor.

Tasks:

- implement a CUDA `lora_delta_resident` path only if it returns a tensor connected to `x`, `A`, and
  `B`;
- if the implementation delegates to candle CUDA matmul, label it honestly as candle-CUDA resident;
- add gradient parity and nonzero-LoRA-gradient tests;
- preserve `lora_decode_add` as inference-only.

Acceptance:

- loss backward produces gradients for `A` and `B`;
- forced CUDA resident path matches `compute_lora_delta`;
- tracked tensors never route through `kiln_rmsnorm_kernel::lora_decode_add`.

### Phase C3: Device Optimizer Dispatch

Goal: update trainable state without requiring a host readback between steps.

Tasks:

- add custom CUDA SGD and AdamW kernels or a candle-CUDA in-place equivalent that updates the actual
  canonical tensor storage;
- only return `true` from `dispatch_sgd_step` / `dispatch_adamw_step` when the backend really owns
  the update;
- add F32/BF16 parity, shape-mismatch error, missing-residency decline, and lazy-sync tests.

Acceptance:

- CUDA optimizer step matches CPU reference for F32 and BF16;
- stale-host/current-after-sync behavior is explicit if a side buffer is introduced;
- adapter save sees current LoRA weights.

### Phase C4: Projection and LM-Head Training Dispatch

Goal: add the CUDA equivalent of Vulkan's autograd-safe projection and offset matmul hooks.

Tasks:

- implement `linear_prefill_apply` for training tensors with backward;
- implement `linear_prefill_apply_offset` for FLCE chunks without re-uploading or slicing into
  dangerous submits;
- add output-dim and row chunking ceilings;
- prove q/k/v/o, GDN in-proj, gate/up/down, and lm_head route correctly.

Acceptance:

- forward/backward parity for projection shapes;
- long-shape dispatch is chunked, not monolithic;
- FLCE never materializes `[tokens, vocab]` logits.

### Phase C5: Native CUDA Training Stack

Goal: converge on the same end state as `vk_native_sft_train`, but for CUDA.

Tasks:

- add a CUDA-native tensor/autograd module only after C1-C4 establish the contracts;
- wire Qwen3.5 FullAttention and GDN forward/backward;
- add checkpointed SFT, AdamW, adapter save, and real-model server routing;
- add an arena/pool to avoid allocation churn.

Acceptance:

- one-layer native CUDA smoke loss decreases;
- hybrid Qwen3.5-4B SFT succeeds on a real model;
- planned hot-path readbacks are limited to scalar loss logging and adapter/checkpoint save.

## Commit Discipline

This branch is expected to be long-lived and multi-commit. Each commit should include:

- one phase-sized behavior change or one audit/runbook update;
- a test or compile gate that actually covers the changed behavior;
- a push to the configured remote before starting the next phase.

Current configured `origin` in this workspace is `/data/repo-cache/ericflo/kiln.git`, not a GitHub
HTTPS/SSH remote. Push evidence should therefore record the exact remote used until a GitHub remote
is configured.

## Completion Criteria

CUDA should not be called equivalent to the Vulkan training modernization until the branch proves:

- impossible training jobs are rejected before allocation;
- default SFT uses FLCE or equivalent chunked loss;
- training projections and LoRA deltas are autograd-safe;
- RMSNorm and attention training paths are on GPU for production shapes;
- checkpoint boundaries and trainable LoRA state have explicit CUDA residency semantics;
- SGD and AdamW update CUDA-resident state in place or honestly decline;
- adapter save is the only required trainable-state readback;
- a hybrid Qwen3.5-4B SFT job succeeds on real CUDA hardware;
- operator logs prove the CUDA training paths that engaged;
- parity tests cover forward, backward, fallback, shape errors, BF16/F32, and end-to-end loss
  decrease.
