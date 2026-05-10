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
