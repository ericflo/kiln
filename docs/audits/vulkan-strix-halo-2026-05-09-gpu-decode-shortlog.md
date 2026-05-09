# Vulkan Strix Halo GPU Decode Shortlog, 2026-05-09

Target: Qwen3.5-4B on AMD Radeon 8060S Graphics (RADV_STRIX_HALO), Linux Vulkan backend.

This file is the compact durable index for the 2026-05-09 Vulkan decode work.
The detailed log is
`docs/audits/vulkan-strix-halo-2026-05-09-gpu-decode-log.md`.

Future Vulkan optimization work should update this file and the detailed log
before or with each accepted change and each measured rejection.

| ID | Change / Experiment | Evidence | Verdict |
| --- | --- | --- | --- |
| A001 | Fix Vulkan correctness and GPU-routed decode before perf work. | Server Vulkan smoke produced coherent text; paged bench completed with token IDs `[2838,6587,310,5227,1024,75119,220]`; corrected baseline around `648.1ms` prefill and `277.2ms` mean ITL. | Keep, commit `717a7de` after rebase. |
| A002 | Route LoRA base projections through backend decode, then add LoRA delta. | `test_backend_linear_decode_adds_lora_delta` passed; synthetic rank-8 projection showed `4.084ms` backend vs `58.747ms` fallback, `14.385x`; no-LoRA tokens stayed coherent. | Keep, commit `3066ed4` after rebase. |
| A003 | Reject simple single-token linear retile and transfer experiments. | Host-visible single-submit regressed to `~278.9ms`; 32x8 was noise at `~276.5ms`; 64x4 regressed to `~285.0ms`; fused GDN/conv envs regressed or failed to win. | Rejected; do not repeat without new evidence. |
| A004 | Route GDN `out_proj` through backend decode. | Same token IDs as corrected baseline; serial mean ITL improved to `179.2ms`; GDN `out_proj` profile dropped from `337.330ms total / 4.685ms mean` to `51.707ms total / 0.718ms mean`; warmed 4-prompt greedy batch improved from `~4.77s` to `~4.03s`. | Keep, commit `6ae731c` after rebase. |
| A005 | Audit generic continuous sampled batch path with `KILN_BATCHING_ENGINE=1`. | Sampled actor batch returned HTTP 500: Vulkan declined batched contiguous paged attention at full-attention layer 3. | Fixed by A006; this was a correctness/availability issue. |
| A006 | Add rowwise full-attention fallback for generic continuous batch when backend declines batched paged attention. | Same sampled actor batch now returns HTTP 200; rebuilt server rerun was `time_total=9.525668s`, 72 prompt tokens, 48 completion tokens, all finish reasons `length`. Serial bench stayed coherent at token IDs `[2838,6587,310,5227,1024,75119,220]`, `174.7ms` mean ITL. | Keep, commit `b6e699f`; correctness/availability fix, not a throughput win. |
| A007 | Recheck existing Vulkan feature gates. | Tokens stayed coherent. Fused conv1d regressed to `197.3ms`; disabling MLP decode regressed to `209.8ms`; GDN fused did not prove a win; disabling full-attn QKV was noisy and not better than default on rerun. | Reject toggles; keep defaults. |
| A008 | Route full-attention `o_proj` through backend decode. | Full-attn `o_proj` profile dropped from `68.313ms total / 4.270ms mean` to `5.413ms total / 0.338ms mean`. Unprofiled serial bench stayed coherent with token IDs `[2838,6587,310,5227,1024,75119,220]`, `480.8ms` prefill, `135.1ms` mean ITL, `7.4 tok/s`. | Keep; serial throughput win. |
| A009 | Trial Vulkan MLP decode single-submit command recording. | Focused MLP parity passed, but default single-submit regressed serial bench to `159.6ms` mean ITL; disabling it in the same binary returned to `135.8ms` with the same token IDs. | Rejected and removed before commit. |
| A010 | Add full-attention QKV single-submit dispatch. | Default runs preserved token IDs `[2838,6587,310,5227,1024,75119,220]` and measured `132.3ms` and `133.1ms` mean ITL. Rollback first measured `134.2ms` with same IDs; a second rollback was noisy and diverged IDs. Profiled QKV bucket moved slightly from `59.768ms` to `59.019ms`. | Keep as a small Vulkan-only win. |
| A011 | Add Vulkan dyn-seqlen paged decode batch attention for sampled/non-uniform continuous batches. | New parity test passed; full Vulkan kernel suite passed `20` tests. Primary sampled actor batch improved from A006 `9.525668s` to `7.754065s` for `48` completion tokens. Rollback env `KILN_DISABLE_VULKAN_PAGED_ATTN_DECODE_BATCH=1` took `8.571721s` on the same `/no_think` sampled shape. Serial tokens stayed `[2838,6587,310,5227,1024,75119,220]`, `130.0ms` mean ITL. | Keep; batch throughput/availability win, but K/V compaction/upload remains the next target. |

Validation snapshot after A011:

- `rustfmt --edition 2024 --check crates/kiln-vulkan-kernel/src/kernels.rs crates/kiln-vulkan-kernel/tests/gdn_parity.rs crates/kiln-model/src/backend/vulkan.rs`
- `git diff --check`
- `cargo check -p kiln-model`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-model test_backend_linear_decode_adds_lora_delta --lib -- --nocapture`
- `cargo test -p kiln-model test_gdn_chunkwise_masks_decay_before_exp --lib -- --nocapture`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `KILN_BATCHING_ENGINE=1 KILN_MODEL_PATH=Qwen3.5-4B ./target/release/kiln serve` sampled batch smoke, plus rollback env smoke.
- `KILN_BENCH_LOG_ITL=1 KILN_BENCH_LOG_TOKENS=1 ./target/release/kiln-bench --model-path Qwen3.5-4B --latency-only --paged --prompt-tokens 8 --max-output-tokens 6 --skip-training`

CUDA and Metal checks were attempted but are environment-blocked on this Linux
host before project typecheck: CUDA lacks `nvcc`; Metal `objc2` requires an
Apple target.
