# WSL CUDA Qwen3.5-4B Shortlog

Date: 2026-05-09

- Rejected CUDA graph stable paged metadata after a `libcuda.so.1.1` SIGSEGV
  and WSL `dxgkio_make_resident: -12` memory pressure.
- Rejected `KILN_CUDA_GRAPH_CACHE_MAX=128` for 16 GiB GPUs: warmed latency can
  improve, but cold-fill and memory risk are not shippable.
- Accepted inference-only fused RMSNorm dispatch:
  - untracked inference tensors use `kiln_rmsnorm_kernel::fused_rmsnorm`;
  - tracked/autograd tensors keep the existing 47 GiB training gate;
  - RMSNorm kill switches still force fallback.
- Patched default server, no force override, warmed 64-token latency:
  `2.7625, 2.7347, 2.7066, 2.7377, 2.7352` seconds.
- Recovered fallback baseline warmed latency:
  `3.296, 3.296, 3.284, 3.273, 3.371` seconds.
- Net warmed gain: 17.2% lower request latency, 20.8% higher completion
  throughput.
- Training smoke completed on the same 4090 Laptop GPU:
  `state=completed`, `final_loss=4.8016462326049805`.
- Training log showed `fused_path="OFF"` for autograd on this 16 GiB GPU, so
  the small-GPU training OOM guard remains active.
- Validation:
  `cargo fmt --all --check`;
  `cargo build --release --features cuda --bin kiln`;
  `cargo test --release -p kiln-model --features cuda rms_norm --lib`.
- Follow-up accepted: rate-limit CUDA graph cache-full warnings to one WARN per
  graph runner, DEBUG afterward, with adapter invalidation resetting the flag.
- Six-request default-log probe warning count dropped from 369 to 1.
- Warmed default-log latency moved from 2.7687s to 2.7435s for 64 tokens
  (+0.9% completion throughput); quiet-log control was 2.7251s.
- Follow-up validation:
  `cargo fmt --all --check`;
  `cargo build --release --features cuda --bin kiln`;
  `cargo test --release -p kiln-model --features cuda cuda_graph --lib`.
- Rejected current CUDA continuous-batch knobs:
  - `KILN_BATCHING_ENGINE=1` c=2 non-streaming fell to 19.2046s wall,
    6.67 aggregate tok/s, with about 13.0 GiB residency;
  - streaming `KILN_DECODE_BATCH_WAIT_US=1000` coalesced to batch 2 but fell to
    19.3178s wall, 6.63 aggregate tok/s;
  - disabling the live streaming batcher was also worse for unique c=2 prompts
    (7.4672s wall, 17.14 aggregate tok/s).
- Rejected CUDA native-MTP auto-default:
  - warmed latency improved from explicit opt-out 2.8463s to 2.5971s
    (24.64 tok/s), but output parity failed on a deterministic 32-token greedy
    prompt;
  - `KILN_MTP_ARGMAX_FP32=1` stayed non-parity and measured 2.6259s.
- No code shipped for those rejected paths; native MTP remains opt-in only.
- Correction after stream-parity isolation: the bad `Thinking Process!!!!!!!!`
  output was CUDA graph replay on the server graph-capturable stream, not MTP.
  Native MTP matched eager/no-graph tokens for the parity prompt.
- The apparent recent OOM was WSL/CUDA residency pressure and a CUDA allocation
  OOM when projection originals were retained, not a Linux OOM-killer event.
  Keep `KILN_DROP_PROJECTION_ORIGINALS=1` for Qwen3.5-4B on 16 GiB VRAM.
- Accepted safety fix: CUDA graphs are now opt-in by default; enable only with
  explicit `KILN_CUDA_GRAPHS=true`. Default eager/interleaved decode preserved
  token parity. Training code and FLCE/checkpointing guards were not changed.
- Validation:
  `cargo fmt`;
  `cargo test -p kiln-server config::tests::test_defaults --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln`;
  default API parity probe without `KILN_CUDA_GRAPHS`;
  `cargo test -p kiln-train test_flce_parity_vs_naive_loss --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`.
- Post-push live SFT smoke on pushed HEAD completed:
  one rank-1 example, `epochs=1`, `seed=1234`, final loss
  `0.9267818927764893`; logs confirmed 32 checkpoint segments and
  `fused_path="OFF"` for the small-GPU RMSNorm training gate.
- Accepted CUDA BF16 inference GDN state:
  `ModelRunner` and inference benches now create recurrent state in model dtype
  on CUDA/Metal inference while leaving the training/test constructor
  unchanged. Rollback: `KILN_DISABLE_CUDA_BF16_INFERENCE_STATE=1`.
- Same-binary WSL CUDA A/B, 64-token greedy completions:
  rollback warmed `2.7920s` (22.92 tok/s) vs default `2.6879s`
  (23.81 tok/s), a 3.7% latency win. The 32-token parity prompt kept the
  expected `Thinking Process:\n\n1.  **Analyze the Request:**` prefix.
- BF16-state validation:
  `cargo fmt`;
  `cargo test -p kiln-model linear_attention_state --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln`.
- Live training + LoRA inference smoke on the candidate binary completed:
  one rank-1 SFT example auto-loaded `bf16-state-smoke`, final loss
  `1.0715702772140503`, `/health` reported the active adapter, and an
  adapter-backed chat request completed.
- Accepted CUDA live decode-batcher guard:
  CUDA now defaults the live streaming decode batcher to `max_batch=1` unless
  `KILN_DECODE_BATCH_MAX` is explicitly set. This prevents
  `KILN_DECODE_BATCH_WAIT_US` from accidentally coalescing rows onto the known
  slow CUDA batch-2 GDN path while preserving forced A/B testing.
- Current WSL CUDA measurements:
  wait-zero c=2 streaming 64-token rounds averaged `5.4148s`, metrics
  `submitted=393 batches=393 rows=393 max_observed_batch=1`;
  pre-guard wait-100 completed rounds averaged `17.9588s` and a short profile
  confirmed `max_observed_batch=2`;
  candidate wait-100 rounds averaged `5.3672s`, metrics
  `submitted=393 batches=393 rows=393 max_observed_batch=1`, a 3.35x wall-time
  recovery versus pre-guard wait-100.
- Guard validation:
  `cargo fmt`;
  `cargo test -p kiln-model decode_batcher_default_ --lib --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  default parity prompt on the candidate;
  live rank-1 SFT smoke auto-loaded `cuda-batcher-guard-smoke`, final loss
  `0.9538126587867737`, and adapter-backed chat completed.
- Accepted CUDA GDN decode unexpanded Q/K:
  the CUDA fused decode gates+recurrent path now keeps Q/K at native GQA head
  count for `seq_len == 1`, expanding only if the fused backend declines and
  the split fallback runs. Rollback:
  `KILN_DISABLE_GDN_DECODE_UNEXPANDED_QK=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `41.064`, `40.990`, `41.042` ms averaged
  `41.032ms` / `24.371 tok/s`; default unexpanded-Q/K runs `40.236`,
  `40.418`, `40.495` ms averaged `40.383ms` / `24.763 tok/s`, a 1.6%
  decode ITL win.
- Unexpanded-Q/K validation:
  `cargo fmt --check`;
  `cargo test -p kiln-gdn-kernel test_cuda_decode_gates_recurrent --quiet`;
  `cargo test -p kiln-model test_causal_conv1d_update_matches_fallback --lib --quiet`
  compiled the model crate with zero matching tests;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `cuda-unexpanded-qk-smoke`, final loss
  `1.338575839996338`, `/health` reported the active adapter, and
  adapter-backed chat completed.
- Accepted CUDA fused RoPE Q/K:
  CUDA now fuses RoPE over contiguous bf16 Q/K tensors using precomputed f32
  cos/sin tables for table-backed and tensor-backed model forward paths.
  Unsupported shapes fall back to the existing Candle path. Rollback:
  `KILN_DISABLE_FUSED_CUDA_ROTARY_QK=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `40.443`, `40.420`, `40.621` ms averaged
  `40.495ms` / `24.695 tok/s`; default fused-RoPE runs `37.838`,
  `38.006`, `38.111` ms averaged `37.985ms` / `26.326 tok/s`, a 6.2%
  decode ITL win.
- Fused-RoPE validation:
  `cargo fmt --check`;
  `cargo test -p kiln-rmsnorm-kernel rotary_qk_parity_qwen_shape --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `cuda-rotary-qk-smoke`, final loss
  `1.5020499229431152`, `/health` reported the active adapter, and
  adapter-backed chat completed.
- Accepted CUDA fused MLP SiLU multiply:
  CUDA now fuses bf16 `silu(gate) * up` for matching contiguous inference
  tensors in the SwiGLU MLP middle. The model route is forward-only and gated
  off for autograd-tracked tensors, so training keeps the existing Candle path.
  Rollback: `KILN_DISABLE_FUSED_CUDA_MLP_SILU_MUL=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `37.928`, `37.956`, `38.354` ms averaged
  `38.080ms` / `26.261 tok/s`; default fused-MLP runs `37.030`,
  `36.920`, `36.982` ms averaged `36.977ms` / `27.044 tok/s`, a 2.9%
  decode ITL win.
- Fused-MLP validation:
  `cargo fmt --check`;
  `cargo test -p kiln-rmsnorm-kernel mlp_silu_mul_parity_qwen_shape --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `cuda-mlp-silu-mul-smoke`, final loss
  `1.603104591369629`, `/health` reported the active adapter, and
  adapter-backed chat completed.
- Accepted CUDA folded GDN QK norm + decode recurrent:
  CUDA decode now accepts raw F32/bf16 unexpanded GDN Q/K, performs the same
  L2-normalize + bf16 epilogue inside the fused gates+recurrent kernel, and
  skips the separate qk_norm launch/intermediate tensors. Gated RMSNorm stays
  separate. Rollback:
  `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `37.062`, `37.194`, `36.972` ms averaged
  `37.076ms` / `26.972 tok/s`; default folded-QK/recurrent runs `33.430`,
  `33.378`, `33.468` ms averaged `33.425ms` / `29.918 tok/s`, a 9.8%
  decode ITL win.
- Folded-QK/recurrent validation:
  `cargo fmt --check`;
  `cargo test -p kiln-gdn-kernel test_cuda_decode --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `cuda-qk-norm-recurrent-smoke`, final
  loss `1.5285838842391968`, `/health` reported the active adapter, and
  adapter-backed chat completed.
- Accepted CUDA fused attention output gate:
  CUDA now fuses bf16 `x * sigmoid(gate)` for matching contiguous inference
  tensors after attention. The model route is forward-only and gated off for
  autograd-tracked tensors, so training keeps the existing Candle path.
  Rollback: `KILN_DISABLE_FUSED_CUDA_ATTN_SIGMOID_MUL=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `33.144`, `34.127`, `33.457` ms averaged
  `33.576ms` / `29.783 tok/s`; default fused-attn-gate runs `33.064`,
  `33.057`, `32.802` ms averaged `32.974ms` / `30.327 tok/s`, a 1.8%
  decode ITL win.
- Fused-attn-gate validation:
  `cargo fmt --check`;
  `cargo test -p kiln-rmsnorm-kernel sigmoid_mul_parity_qwen_attn_gate_shape --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `attn-sigmoid-mul-smoke-r1`, final loss
  `1.5107368230819702`, `/health` reported the active adapter, and
  adapter-backed chat completed with and without thinking enabled.
- Accepted training-safety follow-up: forward-only GDN backend fast paths now
  decline Candle autograd-tracked tensors, so CUDA/Metal/Vulkan inference keeps
  the fast kernels while training uses differentiable Candle ops through GDN QK
  norm, gated RMSNorm, causal conv, gate/decay, and recurrent/chunk paths.
- This follow-up makes no throughput claim. Release CUDA latency sanity check
  stayed in range: mean ITL runs `33.356`, `33.696`, `33.287` ms averaged
  `33.446ms` / `29.899 tok/s`.
- GDN autograd guard validation:
  `cargo fmt --check`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `gdn-autograd-guard-smoke-r1`, final loss
  `1.5467936992645264`, `/health` reported the active adapter, no-thinking
  chat returned `kiln violet`, and thinking chat emitted reasoning content.
- Accepted CUDA folded GDN QK norm + recurrent + gated RMSNorm:
  CUDA decode now accepts raw F32/bf16 unexpanded GDN Q/K plus V, computes
  Q/K normalization, gates, recurrent update, and gated RMSNorm in one
  single-token kernel. The model route is still forward-only and only reached
  through the existing `gdn_forward_only_fastpaths` guard, so training keeps
  the differentiable Candle path. Rollback:
  `KILN_DISABLE_CUDA_GDN_DECODE_QK_NORM_RECURRENT_RMSNORM=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `33.4`, `33.3`, `33.2` ms averaged `33.3ms` /
  `30.03 tok/s`; default fused-QK/recurrent/RMSNorm runs `26.0`, `26.1`,
  `26.2` ms averaged `26.1ms` / `38.31 tok/s`, a 21.6% decode ITL win.
- Short profile confirmed decode selects `qk_norm_gates_recur_gated_norm`;
  the split `qk_norm_gates_recur` stage disappeared and `gated_norm` fell to
  reshape-level timing.
- Fused-QK/recurrent/RMSNorm validation:
  `cargo fmt`;
  `cargo test -p kiln-gdn-kernel test_cuda_decode_qk_norm_gates_recurrent_rmsnorm_matches_split_path -- --nocapture`;
  `cargo test -p kiln-gdn-kernel test_cuda_decode_qk_norm_gates_recurrent_matches_split_path --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Post-rebase live rank-1 SFT smoke on the rebuilt 0.2.14 binary used
  `KILN_DROP_PROJECTION_ORIGINALS=1` on this 16 GiB GPU, auto-loaded
  `gdn-qk-recur-rmsnorm-smoke-r3`, final loss `1.6381032466888428`,
  `/health` reported the active adapter, no-thinking chat returned
  `kiln copper`, and thinking chat emitted reasoning content.
- Accepted CUDA direct paged decode routing:
  single-request CUDA decode now bypasses the contiguous-slot prefill fallback
  and reaches the native paged FlashAttention decode kernel, avoiding GQA K/V
  expansion on full-attention decode. Metal keeps its contiguous specialized
  path. Rollback: `KILN_DISABLE_CUDA_DIRECT_PAGED_DECODE=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `27.112`, `27.283`, `27.155` ms averaged
  `27.183ms` / `36.787 tok/s`; default direct-paged runs `26.774`,
  `26.718`, `26.730` ms averaged `26.740ms` / `37.397 tok/s`, a 1.6%
  decode ITL win.
- Route profile confirmed CUDA full-attention decode logs
  `decode_attn_paged` instead of `decode_attn_fallback`; candidate profile
  measured `26.7ms` mean ITL / `37.5 tok/s`.
- Direct-paged validation:
  `cargo fmt --check`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `cuda-direct-paged-smoke-r1`, final loss
  `1.590428352355957`, `/health` reported the active adapter, no-thinking
  chat returned `kiln silver`, and thinking chat emitted reasoning content.
- Accepted CUDA fused decode QKV prep:
  single-token paged CUDA decode now fuses Q/gate split, Q/K RMSNorm, and
  RoPE into one forward-only kernel for untracked inference tensors with
  RoPE tables. Training and debug tap captures keep the existing Candle path.
  Rollback: `KILN_DISABLE_CUDA_ATTN_DECODE_QKV_PREP=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `26.692654`, `26.683835`, `26.654270` ms averaged
  `26.676920ms` / `37.485602 tok/s`; default fused-QKV-prep runs
  `25.850827`, `25.849904`, `25.991987` ms averaged `25.897573ms` /
  `38.613914 tok/s`, a 2.9% decode ITL win.
- Route profile confirmed decode logs `qkv_split_qk_norm_rope` instead of
  separate decode `qkv_split`, `qk_norm`, and `rope` rows; the fused stage
  totaled `16.803ms` across `512` layer-token rows.
- Fused-QKV-prep validation:
  `cargo fmt --check`;
  `cargo test -p kiln-rmsnorm-kernel attn_decode_qkv_prep_parity_qwen_shape --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`.
- Live rank-1 SFT smoke auto-loaded `attn-qkv-prep-smoke-r1`, final loss
  `1.6304359436035156`, `/health` reported the active adapter, no-thinking
  chat returned non-empty content, and thinking chat emitted reasoning
  content.
- Accepted CUDA token-major KV write fast path:
  normal single-request bf16 CUDA decode now writes token-major paged K/V with
  one CUDA kernel using a host slot value, replacing the two generic Candle
  `slice_set` writes. CUDA graph replay keeps its existing device-slot variant.
  Rollback: `KILN_DISABLE_CUDA_PAGED_KV_WRITE_TOKEN_MAJOR=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `26.279893`, `26.081714`, `25.957450` ms averaged
  `26.106352ms` / `38.305843 tok/s`; default CUDA KV-write runs
  `26.656393`, `25.908104`, `25.270606` ms averaged `25.945034ms` /
  `38.561360 tok/s`, a 0.6% decode ITL win.
- Targeted profile confirmed decode `kv_write` dropped from `9.159ms` total to
  `5.398ms` total across `512` full-attention layer-token rows, a 41.1%
  reduction in that stage.
- KV-write validation:
  `cargo fmt --check`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo test -p kiln-model write_token_major_native --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live rank-1 SFT smoke auto-loaded `cuda-kvwrite-smoke-r1`, final loss
  `1.3608112335205078`, `/health` reported the active adapter, no-thinking
  chat returned `kiln amber`, and thinking chat emitted reasoning content.
- Accepted CUDA default projection-original drop:
  the training-capable CUDA server load no longer requires manual
  `KILN_DROP_PROJECTION_ORIGINALS=1`. The old no-env default failed during
  model load on this 16 GiB WSL RTX 4090 Laptop at layer 30 MLP `up_proj` with
  `CUDA_ERROR_OUT_OF_MEMORY`; kernel logs showed no Linux OOM-killer event.
- CUDA now drops large original projection tensors by default and keeps the
  hot-path transposed caches that inference and SFT already use for shape
  discovery and forward passes. Rollback/debug override:
  `KILN_KEEP_PROJECTION_ORIGINALS=1`.
- Candidate no-env server load reported `post_load_used_vram_gb=10.032775168`,
  `kv_cache_gb=0.268435456`, `training_budget_gb=6.870269952`, and reached
  `Ready on http://127.0.0.1:8420`.
- No-env paged latency bench stayed in the manual-drop envelope:
  previous manual drop `25.867788ms` / `38.658118 tok/s` / `9497 MB`;
  default CUDA drop `25.721224ms` / `38.878399 tok/s` / `9497 MB`.
- Projection-drop validation:
  `cargo fmt --check`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`;
  `cargo test -p kiln-train test_lora_initialize_uses_transposed_projection_shapes --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`.
- Live no-env rank-1 SFT smoke auto-loaded `cuda-default-drop-smoke`, final
  loss `2.7255642414093018`, `/health` reported the active adapter, and
  adapter-backed chat returned `Kiln target model`.
- Accepted CUDA LoRA decode delta/add fast path:
  single-token BF16 CUDA decode now computes LoRA `x @ A.t()` into F32 scratch
  and adds `scale * hidden @ B.t()` to the base projection with custom kernels.
  The backend declines any autograd-tracked tensor, so SFT keeps the
  differentiable Candle path. Rollback:
  `KILN_DISABLE_CUDA_LORA_DECODE_ADD=1`.
- Recent memory-pressure check showed no Linux OOM-killer entry; kernel logs
  showed WSL/DXG `dxgkio_make_resident: -12`, consistent with CUDA residency
  allocation pressure. No `kiln serve` process was still running, and GPU
  memory was idle at `71 MiB` used of `16376 MiB`.
- Same-binary live adapter-backed chat A/B:
  rollback warmed runs averaged `0.729129s` / `27.43 tok/s`; default fused
  LoRA decode add measured `0.675411s` / `29.61 tok/s`, a 7.4% wall-time win
  and 8.0% throughput win.
- LoRA decode-add validation:
  `cargo fmt --check`;
  `cargo test -p kiln-rmsnorm-kernel lora_decode_add_parity_cuda --quiet`;
  `cargo test -p kiln-model test_backend_linear_decode_adds_lora_delta --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live rank-1 SFT smoke auto-loaded `cuda-lora-add-smoke`, final loss
  `1.3705849647521973`, `/health` reported the active adapter, and
  adapter-backed chat returned `kiln lora`.
- Accepted CUDA GDN A/B projection combine:
  CUDA model load now caches the combined `[hidden, 2 * nv]` GDN A/B transpose,
  and untracked single-token CUDA inference uses one A/B matmul plus two views
  instead of two tiny 32-column matmuls per GDN layer. Training stays on the
  previous separate path because the route requires `!x.track_op()`. Rollback:
  `KILN_DISABLE_CUDA_GDN_AB_IN_PROJ=1`.
- Same-binary WSL CUDA paged latency A/B:
  rollback mean ITL runs `26.627227`, `26.017787`, `26.481529` ms averaged
  `26.375514ms` / `37.917653 tok/s`; default combined A/B runs `26.224632`,
  `26.096284`, `26.034409` ms averaged `26.118442ms` / `38.287475 tok/s`,
  a 1.0% decode ITL win.
- Targeted profile confirmed GDN `in_proj` dropped from `246.444ms` total to
  `204.744ms` total across `1632` profiled rows, a 16.9% reduction in that
  stage. Post-build candidate sanity measured `26.062018ms` / `38.370014 tok/s`.
- GDN A/B combine validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded `cuda-gdn-ab-smoke`, final loss
  `1.1042242050170898`, `/health` reported `training_budget_gb=6.870269952`
  and the active adapter, and adapter-backed chat returned `kiln gdn ab`.
- Accepted CUDA forced batch-2 GDN row-loop guard:
  when CUDA GDN decode is explicitly forced above batch 1, the live scheduler
  now keeps the admitted batch but executes each row through the known-good
  single-row paged greedy path instead of the slow true batched GDN path.
  Full-attention-only CUDA models keep true batching, and training is untouched
  because this route is inference-only. Rollback:
  `KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP=1`.
- Same-binary WSL CUDA live streaming A/B, four unique concurrent requests,
  `max_tokens=32`, `KILN_DECODE_BATCH_MAX=2`,
  `KILN_DECODE_BATCH_WAIT_US=50000`, `KILN_DECODE_BATCH_MIXED_SEQ=1`:
  rollback true batched GDN took `18.627071s`; default row-loop took
  `3.832016s`, with `124` rows, `max_observed_batch=2`, and zero failed jobs
  in both runs. That is a 79.4% wall-time reduction, or 4.86x faster, for the
  forced batch-2 workload.
- Profile confirmation emitted
  `stage=cuda_gdn_row_loop_forward batch=2` for three profiled candidate
  batches. Validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded `cuda-rowloop-smoke`, final loss
  `1.6998780965805054`, and the server log reported `SFT training complete`
  with no `CUDA_ERROR_OUT_OF_MEMORY`.
- Accepted CUDA full-attention Q/K/V projection combine:
  CUDA model load now caches an optional `[hidden, q_raw + k + v]` transpose
  for full-attention layers, and untracked single-token BF16 CUDA decode uses
  one combined Q/K/V matmul plus views instead of three projection matmuls.
  Training remains on the separate differentiable path because the route
  requires `!x.track_op()` and no LoRA/Marlin/MTP debug path. Rollback:
  `KILN_DISABLE_CUDA_FULL_ATTN_QKV_IN_PROJ=1`.
- Same-binary profiled WSL CUDA paged latency A/B:
  rollback `qkv_proj` totaled `45.349ms` across `544` calls
  (`0.083362ms` avg), while default combined Q/K/V totaled `24.835ms`
  (`0.045653ms` avg), a 45.2% reduction in the target stage. Both reported
  `9913 MB` model VRAM.
- Unprofiled repeated runs on `prompt_tokens=64`, `max_output_tokens=33`:
  rollback mean ITL `26.4`, `27.3`, `27.4` ms averaged `27.033ms`;
  default combined Q/K/V `26.6`, `26.5`, `26.4` ms averaged `26.500ms`,
  a 2.0% decode ITL win.
- Q/K/V combine validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-fullattn-qkv-smoke-011048`, final loss `1.6919400691986084`,
  and the server log reported `SFT training complete` with no
  `CUDA_ERROR_OUT_OF_MEMORY`. A post-smoke kernel scan showed no new Linux
  OOM-killer or WSL/DXG residency error, and GPU memory returned to `71 MiB`.
- Accepted CUDA empty output allocations for fully overwritten custom kernels:
  RMSNorm, L2 QK norm, GQA L2 QK norm, rotary Q/K, and decode QKV prep now use
  `Tensor::empty` for CUDA outputs when enabled, avoiding the extra zero-fill
  before kernels that overwrite every element. Rollback:
  `KILN_DISABLE_CUDA_EMPTY_KERNEL_OUTPUTS=1`.
- Token parity on `prompt_tokens=64`, `max_output_tokens=33` matched rollback
  for the first 32 generated token ids.
- Same-binary WSL CUDA paged latency A/B on `prompt_tokens=64`,
  `max_output_tokens=33`: rollback mean ITL runs `26.548371`, `26.752817`,
  `26.917567` ms averaged `26.739585ms` / `37.398932 tok/s`; default empty
  output runs `26.367942`, `26.222591`, `26.306814` ms averaged
  `26.299116ms` / `38.024288 tok/s`, a 1.6% decode ITL win.
- Longer decode repeat on `prompt_tokens=64`, `max_output_tokens=129`:
  rollback averaged `23.893290ms` / `41.854684 tok/s`; default empty outputs
  averaged `23.520915ms` / `42.515354 tok/s`, a 1.6% decode ITL win.
- Empty-output validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-rmsnorm-kernel --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-empty-output-smoke-0905`, final loss `1.2110248804092407`, and
  `/health` reported `training_budget_gb=6.434062336` with the adapter active.
  The server log reported `SFT training complete` and `auto-loaded trained
  adapter`; a post-smoke kernel scan showed no new Linux OOM-killer or WSL/DXG
  residency error, and GPU memory returned to `71 MiB`.
- Accepted direct `ModelRunner::new` CUDA graph default alignment:
  `ModelRunner::new` now constructs runners with CUDA graphs disabled, matching
  the production server default. Explicit graph experiments remain available
  through `ModelRunner::new_with_options(..., true)`.
- Same-binary WSL CUDA `kiln-bench` direct runner sweep on `prompt_tokens=64`,
  `max_output_tokens=33`: graph-enabled default averaged `29.6374`, `29.7530`,
  `29.8214`, `29.7961 tok/s` for batch `1/4/8/16` and `33.0421ms` latency
  ITL; no-graph default averaged `29.7338`, `29.9410`, `29.9987`,
  `29.7895 tok/s` and `33.0020ms` latency ITL. This is a small win for batch
  `1/4/8`, neutral for batch `16`, and aligns direct callers with production.
- ModelRunner default validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-model cuda_graph --lib --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-modelrunner-nograph-smoke-040042`, final loss `3.018186330795288`,
  and `/health` reported `training_budget_gb=6.434062336` with the adapter
  active. The adapter was then unloaded and deleted; a post-smoke kernel scan
  showed no Linux OOM-killer event, the server log had no
  `CUDA_ERROR_OUT_OF_MEMORY`, and GPU memory returned to `71 MiB`.
- Accepted mixed-sequence CUDA GDN batcher row-loop gate:
  forced mixed-sequence live CUDA GDN batches can now enter the existing
  `cuda_gdn_row_loop_forward` route when all rows are greedy, the paged cache
  is non-FP8, the backend is CUDA, and
  `KILN_DISABLE_CUDA_GDN_BATCHED_DECODE_ROW_LOOP` is unset. Full-attention-only
  and non-CUDA mixed-sequence behavior remains gated on uniform positions.
- Same-binary WSL CUDA live A/B with four unique concurrent streaming requests,
  `max_tokens=32`, `KILN_DECODE_BATCH_MAX=2`,
  `KILN_DECODE_BATCH_WAIT_US=50000`,
  `KILN_DECODE_BATCH_MIXED_SEQ=1`, and stage profiling:
  rollback old gate took `18.563151s` with 124 rows, 62 batches,
  `max_observed_batch=2`, and 62 `batched_forward batch=2` profile entries;
  default mixed-seq row-loop gate took `3.599959s` with the same rows/batches,
  `max_observed_batch=2`, and 62 `cuda_gdn_row_loop_forward batch=2` profile
  entries. Both reported zero failed decode-batcher jobs; wall time improved
  80.6%, or 5.16x.
- Mixed-seq row-loop gate validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_mixed_seq_lens_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-mixed-gate-smoke-042353`, final loss `1.7761503458023071`, then
  unload/delete returned `{"status":"unloaded"}` and `{"status":"deleted"}`.
  A post-smoke kernel scan showed no Linux OOM-killer event, the server log had
  no `CUDA_ERROR_OUT_OF_MEMORY`, and GPU memory returned to `71 MiB`.
- Accepted CUDA GDN prefill fused gates:
  `CudaBackend::gdn_gates` now allows the existing arbitrary-row CUDA gates
  kernel on forward-only prefill, while `KILN_DISABLE_CUDA_GDN_PREFILL_GATES=1`
  restores the old prefill Candle gates path. Training remains on the autograd
  path because backend gates are only called when `x.track_op()` is false.
- Same-binary WSL CUDA paged latency A/B: p64/o33 prefill improved from
  `73.307ms` rollback to `69.000ms` default (5.9%), with mean ITL essentially
  neutral (`26.217ms` rollback, `26.246ms` default). p512/o33 prefill improved
  from `336.569ms` rollback to `331.758ms` default (1.4%), with decode neutral
  to favorable (`26.418ms` rollback, `26.088ms` default).
- Stage profile confirmed the target moved: p64/o17 GDN prefill gates dropped
  from `16.140ms` total / `0.3362ms` avg to `3.727ms` total / `0.0776ms` avg
  across 48 calls.
- Prefill gates validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-gdn-kernel gates --release --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_mixed_seq_lens_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-prefill-gates-smoke-045502`, final loss `1.6567338705062866`, then
  unload/delete returned `{"status":"unloaded"}` and `{"status":"deleted"}`.
  A post-smoke kernel scan showed no Linux OOM-killer event, the server log had
  no `CUDA_ERROR_OUT_OF_MEMORY`, and GPU memory returned to `71 MiB`.
- Accepted short CUDA GDN prefill A/B projection:
  CUDA `gdn_gates` now accepts row-strided A/B views, allowing short
  forward-only prefill tiles (`seq_len <= 128`) to reuse the existing combined
  A/B projection without the previous gates-materialization blow-up.
  `KILN_DISABLE_CUDA_GDN_PREFILL_AB_IN_PROJ=1` restores the old separate
  prefill A/B matmuls; the broader `KILN_DISABLE_CUDA_GDN_AB_IN_PROJ=1`
  still disables the combined path entirely.
- Same-binary WSL CUDA paged latency A/B:
  p64/o33 improved from `69.723ms` prefill rollback to `68.621ms` default
  (1.6%), with mean ITL neutral-to-better (`26.220ms` to `26.145ms`).
  p128/o33 (`actual prompt=115`) improved from `96.921ms` to `95.463ms`
  (1.5%). p512/o33 (`actual prompt=494`) is thresholded to the old path; the
  two-pair spot check stayed noise-level favorable after unrestricted
  `seq_len >= 1` testing showed a longer-prompt regression.
- Stage profile confirmed the scoped target moved: p64/o17 GDN prefill
  `in_proj` dropped from `143.778ms` total / `2.9954ms` avg to `125.799ms`
  total / `2.6208ms` avg across 48 calls. Strided gates increased from
  `1.824ms` to `4.299ms`, but avoided the prior copy blow-up and left a net
  short-prefill win.
- Short prefill A/B validation:
  `cargo fmt --check`;
  `git diff --check`;
  `cargo test -p kiln-gdn-kernel gates --release --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_max_batch_backend_policy --lib --quiet`;
  `cargo test -p kiln-model test_decode_batcher_default_mixed_seq_lens_backend_policy --lib --quiet`;
  `cargo test -p kiln-train test_checkpointed_loss_matches_standard --quiet`;
  `cargo build --quiet --release --features cuda --bin kiln --bin kiln-bench`
  with `CARGO_BUILD_JOBS=1 KILN_CUDA_ARCHS=89`.
- Live no-env rank-1 SFT smoke auto-loaded
  `cuda-prefill-ab-smoke-0510`, final loss `1.665740728378296`, then
  unload/delete returned `{"status":"unloaded"}` and `{"status":"deleted"}`.
  A post-smoke kernel scan showed no Linux OOM-killer event, the server log had
  no `CUDA_ERROR_OUT_OF_MEMORY`, and GPU memory returned to `71 MiB`.
