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
| A012 | Trial MLP gate/up shader retiles (`32x8`, `128x2`). | Current profile showed `mlp:fused total=200.336ms / mean=2.087ms`. Both retiles passed focused MLP parity and preserved token IDs, but `32x8` regressed to `134.1ms` mean ITL and `128x2` regressed to `134.6ms`, worse than the `130-132ms` accepted anchors. | Rejected; restored original `64x4` geometry. |
| A013 | Trial generic Vulkan fused GDN batch hook. | Wiring `backend.gdn_decode_gates_recurrent_rmsnorm` passed `cargo check` and serial stayed coherent at `131.3ms` mean ITL, but the primary sampled actor batch regressed to `14.470374s`; rollback env `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED_BATCH=1` restored `7.668599s`. | Rejected and removed; existing fused GDN batch kernel loses in the current data-movement shape. |
| A014 | Trial sampled contiguous-batch logits route. | New route passed `cargo check` and release build, and serial stayed coherent at `133.6ms` mean ITL, but primary sampled actor batch regressed to `8.430564s` / `8.311119s`; rollback env `KILN_DISABLE_PAGED_CONTIG_BATCH_LOGITS=1` restored `7.757198s`. | Rejected and removed; do not broadly route sampled batches through contiguous full logits without reducing state/logits overhead first. |
| A015 | Use online softmax in Vulkan dyn-seqlen paged decode batch attention. | Focused paged-attn parity passed; full Vulkan kernel parity passed `20` tests; `cargo check -p kiln-model --features vulkan` and release build passed. Serial stayed coherent at `131.8ms` mean ITL. Primary sampled batch stayed HTTP 200 at `7.743591s`. Longer sampled A/B improved from old-shader rebuild `27.612193s` to online rebuild `27.327221s` for `411` prompt and `64` completion tokens. | Keep; small longer-context continuous-batch win, Vulkan-only. |
| A016 | Trial direct Rust K/V compaction for dyn-seqlen batch attention. | Compact-slice dispatcher parity passed; `cargo check` and release build passed. Unthresholded direct compaction improved longer sampled batch from rollback `27.058533s` to `26.873467s`, but regressed the primary short sampled batch to `7.795046s` / `7.807411s` while rollback was `7.670023s`. Thresholding at `max_seqlen_k >= 64` avoided most short regression (`7.727216s`) but gave no meaningful longer win (`27.040179s`). | Rejected and removed; next K/V work should target residency/upload strategy, not another host compaction path. |
| A017 | Trial `exp2(delta * LOG2_E)` in Vulkan online softmax. | Focused paged-attn parity passed, full Vulkan kernel parity passed `20` tests, `cargo check` and release build passed. Exp2 sampled smokes returned HTTP 200, but measured `7.689383s` short and `27.282147s` longer, slower than fresh non-exp2 rollback evidence from A016 (`7.670023s` short, `27.058533s` longer). | Rejected and removed; keep the current `exp` online softmax. |
| A018 | Trial lowering Vulkan MLP row-pair threshold from `batch >= 8` to `batch >= 4`. | Focused MLP batched parity and gate/up parity passed; `cargo check` and release build passed. Primary sampled batch returned HTTP 200 but measured `7.884895s` and `7.811283s`, slower than the accepted `~7.67-7.74s` envelope. Env-disabled run was `8.035747s` but is not a clean old-threshold rollback because it also disables existing prefill row-pair use. | Rejected and removed; keep row-pair MLP gated at `batch >= 8`. |
| A019 | Trial Vulkan two-stage LoRA delta decode (`x @ A^T @ B^T`). | A temporary full-shape rank-8 synthetic PEFT adapter exercised all trained LoRA modules. The new two-shader delta path passed focused parity and `cargo check -p kiln-model --features vulkan`, and release build passed, but active-adapter sampled batches measured `14.988255s` / `15.065539s` for `94` prompt and `48` completion tokens. Rollback env `KILN_DISABLE_VULKAN_LORA_DELTA_DECODE=1` in the same binary measured `12.825559s` / `12.724448s` for `98` prompt and `48` completion tokens. | Rejected and removed; two extra Vulkan dispatches per LoRA projection lose without deeper LoRA fusion/residency. |
| A020 | Trial Vulkan batched GDN input-projection `32x8` tile in place of the current `80x3` tile. | Focused batched GDN in-proj parity passed; `cargo check -p kiln-model --features vulkan` and release build passed. Candidate sampled batches returned HTTP 200 at `9.170221s` / `9.119555s` for `106` prompt and `48` completion tokens. Same-branch rollback to `80x3` returned `8.824926s` for the same token shape, plus `8.742473s` / `8.506257s` on adjacent `102` prompt-token runs. | Rejected and removed; keep the existing `80x3` geometry until a broader GDN residency/fusion change. |
| A021 | Trial Metal-inspired Vulkan MLP gate/up two-column single-token shader. | Focused MLP parity passed (`3` tests); `cargo check -p kiln-model --features vulkan` and release build passed. Candidate stayed coherent but regressed serial paged bench to `141.6ms` mean ITL. Candidate profile showed `mlp:fused total=218.030ms / mean=2.271ms`, worse than pre-trial `~202.545ms / 2.110ms`. Same-branch rollback rebuilt and returned to coherent tokens with `129.1ms` mean ITL. | Rejected and removed; Metal's two-column gate/up shape does not transfer to Vulkan F32 on Strix Halo. |
| A022 | Trial direct generation-loop Vulkan GDN recurrent resident-state scope. | Temporary RAII scope in `generate.rs` passed rustfmt, `cargo check -p kiln-model --features vulkan`, and release build. Candidate direct `/v1/chat/completions` stayed HTTP 200 but measured `4.975489s` / `4.920384s` for `30` prompt and `24` completion tokens. Same-binary rollback with `KILN_DISABLE_VULKAN_GDN_RECURRENT_RESIDENT_STATE=1` measured `4.796903s` for the same shape. Longer 48-token rollback measured `8.307010s`, candidate `8.477510s`. | Rejected and removed; real API resident-state wins need request/batcher-owned residency, not a thread-local direct-loop scope. |
| A023 | Store Vulkan linear-decode weights as packed BF16 buffers and expand in shader. | New packed-BF16 parity tests passed; full Vulkan parity passed `24` tests; `cargo check -p kiln-model --features vulkan`, release build, and `git diff --check` passed. Serial stayed coherent with token IDs `[2838,6587,310,5227,1024,75119,220]`; candidate mean ITL was `121.4ms` / `114.0ms` versus same-binary rollback `132.5ms`. Sampled continuous batch improved `9.049283s` rollback to `8.394559s`; greedy continuous batch improved `8.940242s` rollback to `8.284110s`. Synthetic rank-8 LoRA smoke returned HTTP 200. | Keep; measurable Vulkan-only serial and continuous-batch win, rollback via `KILN_DISABLE_VULKAN_BF16_PACKED_LINEAR_WEIGHTS=1`. |
| A024 | Store Vulkan GDN `in_proj` weights as packed BF16 buffers and expand in shader. | Focused GDN in-proj parity passed for F32 and packed-BF16 single/batched variants; full Vulkan parity passed `26` tests; `cargo check -p kiln-model --features vulkan`, `cargo check -p kiln-model`, release build, and `git diff --check` passed. Serial stayed coherent with token IDs `[2838,6587,310,5227,1024,75119,220]`; candidate mean ITL was `115.5ms` / `118.2ms` versus same-binary GDN rollback `121.4ms`. Sampled continuous batch improved `8.525863s` rollback to `7.663571s`. | Keep; measurable Vulkan-only serial and sampled continuous-batch win, rollback via `KILN_DISABLE_VULKAN_BF16_PACKED_GDN_IN_PROJ_WEIGHTS=1`. |
| A025 | Expose explicit `chat_template_kwargs` so callers can set Qwen `enable_thinking=false` without prompt-text hacks. | Full JSON showed empty `content` was coherent Qwen reasoning routed to `reasoning_content`, not Vulkan garbage. Added shared tokenizer/server kwargs plumbing and cache-key coverage. `cargo check -p kiln-server`, focused core/server tests, release Vulkan build, and `git diff --check` passed. Rebuilt Vulkan direct chat with `{"chat_template_kwargs":{"enable_thinking":false}}` returned `content == "blue"` / no reasoning in 3 completion tokens; literal `/no_think` prompt text stayed on default reasoning path; unique batch prompt returned two `text == "blue"` completions. | Keep; correctness/API fix, no CUDA/Metal/Vulkan kernel changes. |
| A026 | Store Vulkan full-attention QKV BF16 weights as packed buffers and expand in shader. | Added packed-BF16 QKV shader, dispatcher, backend gate, and parity test. `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, full Vulkan parity (`27 passed`), release Vulkan build, and `git diff --check` passed. Main-branch serial stayed coherent with token IDs `[2838,6587,310,5227,1024,75119,220]`; candidate measured `398.8ms` prefill / `111.5ms` mean ITL versus same-binary rollback `621.6ms` prefill / `118.4ms` mean ITL. Direct, explicit batch, and concurrent live-batcher smokes with `enable_thinking=false` all returned non-empty visible text. | Keep; measurable Vulkan-only serial win, rollback via `KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS=1`. |
| A027 | Store single-row Vulkan MLP gate/up/down weights as packed BF16 buffers and expand in shader. | Added packed-BF16 gate/up shader, fused MLP dispatcher path, backend row-count gate, and parity test. `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, focused MLP parity, full Vulkan parity (`28 passed`), release Vulkan build, and `git diff --check` passed. Main-branch serial stayed coherent with token IDs `[2838,6587,310,5227,1024,75119,220]`; candidate measured `102.1ms` / `95.0ms` mean ITL versus same-binary rollback `112.4ms` / `113.6ms`. F32 MLP weights remain prewarmed for prefill/batched paths after packed-only prewarm regressed cold prefill to `3298.7ms`. Direct, explicit batch, and concurrent live-batcher smokes with `enable_thinking=false` all returned non-empty visible text. | Keep; measurable Vulkan-only serial win, rollback via `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`. |
| A028 | Extend packed-BF16 Vulkan MLP decode to small multi-row batches below the row-pair threshold. | Added packed-BF16 batched gate/up shader and reused `linear_decode_batched_bf16w` for down projection when flattened row count is `2..7`; row-pair batches stay on F32. Focused MLP parity covers batch `1` and `3`; full Vulkan parity passed (`28 passed`); release Vulkan build and `git diff --check` passed. Serial stayed coherent at `98.3ms` mean ITL. First four-prompt A/B was mixed (`4.390s` candidate vs `4.330s` rollback explicit batch, `9.493s` candidate vs `9.818s` rollback concurrent chat); second fresh-prompt A/B favored candidate on both (`4.320s` vs `4.544s`, `10.092s` vs `10.378s`). | Keep; small-batch Vulkan win with row-pair prefill left untouched, rollback via `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`. |
| A029 | Trial packed-BF16 Vulkan MLP row-pair shaders for `row_count >= 8`. | Temporary row-pair gate/up and down BF16 shaders passed `cargo check`, focused parity including batch `9`, release Vulkan build, and `git diff --check`. Candidate stayed coherent and measured `384.1ms` prefill / `96.5ms` mean ITL, then `390.2ms` / `98.0ms` after adding a row-pair rollback env. Same-binary row-pair rollback with `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_ROWPAIR_WEIGHTS=1` produced the same tokens at `383.4ms` prefill / `96.5ms` mean ITL. | Rejected and removed; direct row-pair BF16 mirroring did not beat the current F32 row-pair path. |
| A030 | Enable mixed-sequence live decode batching by default for Vulkan. | Pre-change streaming A/B showed mixed-seq grouping reduces default-wait live-batcher fragmentation (`12.921s`, `38` batches, max batch `3`) versus disabled (`13.470s`, `69` one-row batches), with a larger wait-50ms win (`12.444s` vs `17.034s`). Post-change no-env Vulkan server logged `mixed_seq_lens=true` and returned four correct visible streams in `13.365s`, `68` rows, `38` batches, max batch `3`. Focused policy test, `cargo check -p kiln-model --features vulkan`, `cargo check -p kiln-server --features vulkan`, and release Vulkan build passed. | Keep; backend-aware default only, CPU/CUDA remain off by default, Metal default preserved. Rollback via `KILN_DECODE_BATCH_MIXED_SEQ=0`. |
| A031 | Trial Vulkan actor mixed-sequence greedy rows through the contiguous-batch argmax path. | Temporary Vulkan-only gate relaxation passed `cargo check -p kiln-model --features vulkan` and release Vulkan build. Current-main actor baseline was `14.299s` for four varied-length greedy non-streaming requests. Candidate returned correct visible text but regressed to `19.584s`; same-binary rollback with `KILN_DISABLE_VULKAN_ACTOR_MIXED_SEQ_GREEDY_BATCH=1` and the exact candidate prompts returned `13.871s`. | Rejected and removed; admission-gate-only routing to dyn-seqlen actor argmax is slower than the existing hidden/logits fallback. |
| A032 | Pair adjacent QKV and Z columns in batched Vulkan GDN input projection. | Added F32 and packed-BF16 paired batched shaders inspired by recent Metal column-pairing work. Focused GDN in-proj parity, full Vulkan GDN parity (`28 passed`), `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, `cargo check -p kiln-server --features vulkan`, release Vulkan build, and `git diff --check` passed. Four varied-length live streams returned HTTP 200 with non-empty visible text and empty reasoning text. Endpoint A/B improved candidate `13.076s` vs rollback `15.680s` with identical `68` rows / `38` batches / max batch `3`; second pair was candidate `13.735s` vs rollback `14.000s`. | Keep; Vulkan-only batched GDN in-proj win, batch `1` unchanged, CUDA/Metal sources untouched. Rollback via `KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_PAIR_QKV_Z=1`. |
| A033 | Profile current Vulkan live path after A032. | Profiled four varied prompt-length streams with `max_tokens=3` and thinking disabled. All streams returned HTTP 200 with visible text and empty reasoning text. Filtered live totals, excluding prewarm rows: MLP `5203.672 ms`, GDN `11274.449 ms`, full-attention `1931.328 ms`. Top decode-only `seq_len=1` stages were `mlp:fused` `198.206 ms`, `gdn:gates` `178.687 ms`, `gdn:recurrent` `120.919 ms`, and `gdn:in_proj` `117.978 ms`. | Keep as target-selection evidence; no source change. |
| A034 | Trial flattened batched Vulkan full-attention QKV fusion for prefill rows. | Temporary F32 and packed-BF16 batched QKV shaders passed focused parity and full Vulkan parity (`30 passed`), plus `cargo check -p kiln-model --features vulkan`, `cargo check -p kiln-server --features vulkan`, release Vulkan build, and `git diff --check`. Candidate returned correct visible text but regressed to `9.856s`; same-binary rollback was `5.490s` with the same `8` jobs / `4` batches / `8` rows / max batch `3`. | Rejected and removed; do not retry by only flattening rows into a fused Q/K/V dispatch. |
| A035 | Trial push-constant `seq_lens` for Vulkan dyn-seqlen paged attention. | Temporary shader/dispatcher path passed focused paged-attn parity for candidate and rollback, `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, and release Vulkan build. Greedy streaming actor smoke was inconclusive (`20.086s` candidate vs `20.857s` rollback) with correct visible text, but the more relevant sampled actor batch regressed: rollback `16.944s`, candidate `17.467s`, same usage and outputs. | Rejected and removed; row-length push constants are not enough. Next dyn-seqlen work needs K/V residency or compact-window upload/materialization changes. |
| A036 | Trial Metal-inspired packed-BF16 Vulkan generic linear row-pair for small live batches. | Temporary row-pair shader/dispatcher path passed focused BF16 generic linear parity for candidate and rollback, focused packed-BF16 MLP parity, `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, and release Vulkan build. No-env live greedy decode-batcher smoke with thinking disabled returned coherent visible text, but regressed: rollback `18.438s`, candidate `19.136s`, both `23` jobs / `14` batches / `23` rows / max batch `3`. | Rejected and removed; Metal's row-pair shape does not directly transfer to Vulkan packed-BF16 generic linear decode. Future work needs RADV/Strix-Halo-specific tiling evidence first. |
| A037 | Trial Vulkan-only default `5ms` live decode-batcher rendezvous wait. | Env-only sweep suggested `5ms` could help (`16.916s`, `8` batches) versus `0us` (`19.276s`, `14` batches), and a single-stream env probe showed no penalty. Temporary code passed `cargo test -p kiln-model decode_batcher_default --lib`, `cargo check -p kiln-model --features vulkan`, and release Vulkan build. The decisive same-binary no-env candidate regressed badly: rollback `15.159s`, candidate `21.191s`, with coherent visible text and empty reasoning in both. | Rejected and removed; do not set a static Vulkan default wait from env sweeps alone. Future wait work needs adaptive policy and stronger repeated A/B. |
| A038 | Add packed-BF16 Vulkan GDN in-proj row-pair shader for `batch >= 4`. | Inspired by Metal E369 but gated away from batch `2/3`. Focused GDN in-proj parity passed for existing batch-3 and new batch-5 row-pair paths, rollback parity passed with `KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR=1`, `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, and release Vulkan build passed. Two same-binary endpoint pairs with wait `5000us` and max batch `4` improved candidate `20.143s` vs rollback `21.973s`, then candidate `20.294s` vs rollback `21.480s`, with identical counters and coherent visible text. | Keep; Vulkan-only larger-batch GDN in-proj win. Batch `1/2/3`, CPU, CUDA, Metal, and F32 Vulkan path stay unchanged. Rollback via `KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR=1`. |
| A039 | Profile Vulkan after A038 GDN in-proj row-pair. | Four concurrent streaming requests with wait `5000us`, `max_tokens=3`, and `chat_template_kwargs: {"enable_thinking": false}` returned HTTP 200 with visible text (`"6 7"`, `"11 "`, `"16 "`, `"13"`) and empty reasoning text. Wall time was `17.514s`, counters were `8` jobs / `3` batches / `8` rows / max batch `4`. Profile totals were MLP `12790.460ms`, GDN `10374.416ms`, full-attn `1828.641ms`; live `seq_len=1` top stage was `mlp:fused` at `2635.685ms`. | Keep as target-selection evidence; no source change. Next Vulkan work should split or optimize packed-BF16 MLP decode before smaller GDN residuals. |
| A040 | Trial packed-BF16 Vulkan MLP gate/up and down row-pair for live batches `4..7`. | Temporary row-pair gate/up and down shaders passed focused candidate and rollback BF16 MLP parity, full Vulkan parity (`29 passed`), `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, and release Vulkan build. Endpoint A/B with wait `5000us` and max batch `4` was unstable: candidate won pair 1 (`17.081s` vs rollback `17.957s`) but lost pair 2 (`20.896s` vs rollback `19.352s`) with identical counters and coherent visible text. | Rejected and removed; direct live-batch MLP row-pair did not produce a reliable gain. |
| A041 | Trial packed-BF16 Vulkan MLP gate/up-only row-pair for live batches `4..7`. | Temporary gate/up-only row-pair path passed focused candidate and rollback BF16 MLP parity and release Vulkan build. Endpoint A/B with wait `5000us` and max batch `4` consistently favored rollback: pair 1 rollback `16.244s` vs candidate `21.055s`; pair 2 candidate `18.424s` vs rollback `15.632s`, with identical counters and coherent visible text. | Rejected and removed; do not retry MLP live-batch row-pair by direct shader mirroring. Kept the added batch-5 BF16 MLP parity coverage. |
| A042 | Trial packed-BF16 Vulkan MLP branchless SiLU shader variants. | Temporary single-row and batched BF16 gate/up shaders passed focused candidate and rollback MLP parity, full Vulkan parity (`29 passed`), `cargo check -p kiln-model --features vulkan`, and release Vulkan build. Endpoint A/B with wait `5000us` favored rollback: pair 1 rollback `16.782s` vs candidate `18.933s`; pair 2 candidate `16.454s` vs rollback `16.142s` even though rollback generated more rows (`23` vs `17`). | Rejected and removed; keep the existing stable sigmoid MLP shaders. |
| A043 | Trial Metal-inspired F32 Vulkan MLP gate/up row-quad for full batch `8`. | Temporary row-quad gate/up shader passed focused candidate and rollback batched MLP parity, full Vulkan parity (`29 passed`), `cargo check -p kiln-model --features vulkan`, and release Vulkan build. Eight-request endpoint A/B with wait `5000us` exercised max batch `8` with identical `56` jobs / `8` batches / `56` rows: candidate lost pair 1 (`25.939s` vs rollback `22.018s`) and won pair 2 (`24.860s` vs rollback `27.180s`). | Rejected and removed; not a reliable n8 endpoint win, so Metal row-quad does not directly transfer to Vulkan F32 MLP gate/up. |
| A044 | Temporarily split Vulkan MLP inner-stage timings. | Temporary `KILN_PROFILE_VULKAN_MLP_KERNEL_STAGES=1` instrumentation timed extract, alloc, upload, gate/up, down, readback, and tensor creation inside `dispatch_mlp_decode_cached_impl`; release Vulkan build passed. Four streaming requests with wait `5000us` returned coherent visible text and empty reasoning. Live packed-BF16 inner totals were gate/up `83.117ms`, down `58.250ms`, readback `37.642ms`, upload `34.392ms`, with tiny alloc/extract/tensor times. The outer `mlp:fused seq_len=1` profile bucket was still much larger at `4875.221ms`. | Keep as target-selection evidence; no source change retained. Next MLP work should target residency/transfer or backend boundary overhead before another direct shader-shape trial. |
| A045-A048 | Fix Vulkan prewarm readiness and decode-weight warmup. | A045 backend-boundary timers showed first live packed-BF16 MLP was paying lazy weight-cache upload, not shader math: live BF16 `kernel_dispatch` was `172.441ms` but batch-1 BF16 gate/up/down cache misses cost about `2.454s`; request wall was `17.144901s`. A046 wired backend decode-weight prewarm but showed health still reported ready before prewarm completed (`17.314253s`). A047 made Vulkan readiness wait for background prewarm and dropped the same measured request shape to `3.039910s`. A048 removed temporary instrumentation; no-profile final was `3.086735s`, `7` tokens, `7` jobs, `3` batches, `7` rows, max batch `4`, with texts `6`, `11`, `16`, `13` and empty reasoning. | Accepted. Server background prewarm now calls backend decode-weight prewarm, and readiness treats Vulkan as needing inference prewarm even though Candle's model device is CPU. |
| A049 | Profile Vulkan after prewarm readiness and weight warmup. | Rebuilt release Vulkan binary and profiled four concurrent streams after `/health` reported `inference_prewarm_complete=true`, with wait `5000us`, all built-in layer/stage profile flags, `max_tokens=3`, and `chat_template_kwargs: {"enable_thinking": false}`. Wall was `4.082501s`; counters were `7` jobs / `3` worker batches / `7` rows / max batch `4`; outputs were correct texts `6`, `11`, `16`, `13` with empty reasoning. Corrected post-prewarm live `seq_len=1` totals: `mlp:fused` `129.645ms`, `gdn:recurrent` `110.270ms`, `gdn:in_proj` `101.692ms`, `gdn:gates` `80.466ms`, `gdn:gated_norm` `51.386ms`. | Keep as target-selection evidence; no source change. Supersedes A039 for warmed-serving decisions. |
| A050 | Trial wiring Vulkan fused GDN gates+recurrent+RMSNorm into the generic forward hook. | Temporary support hook and rollback env `KILN_DISABLE_VULKAN_GDN_DECODE_FUSED=1` were added, then removed. Focused Vulkan fused GDN parity passed (`2` tests), release Vulkan build passed, and both endpoint arms returned correct texts `6`, `11`, `16`, `13` with empty reasoning. Same-binary no-profile A/B favored rollback: candidate `4.950954s`, `7` jobs / `3` batches / `7` rows / max batch `3`; rollback `4.055286s`, same jobs/batches/rows and max batch `4`. | Rejected and removed; do not reuse the fused hook without changing its residency/data movement behavior. |
| A051 | Fix Vulkan cached f32 uploads for non-F32 tensors. | `upload_tensor_f32_buffer` claimed to upload f32 values but previously uploaded raw tensor bytes, which is unsafe when cached f32 shader bindings receive BF16 tensors such as GDN gate aux data. The helper now converts non-F32 tensors to F32 before extracting bytes. Added GDN cached-gates parity coverage for BF16 `a_log` / `dt_bias` aux tensors. Validation passed: `cargo fmt --check`, `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, `cargo check -p kiln-server --features vulkan`, focused Vulkan GDN gate parity, release Vulkan build, and a no-profile prewarmed endpoint run with correct texts `6`, `11`, `16`, `13`, empty reasoning, `7` jobs / `3` batches / `7` rows / max batch `3`, wall `3.990150s`. | Accepted. Vulkan correctness fix; no CUDA/Metal source changes. Future tuning can rely on cached f32 uploads matching the helper contract. |
| A052 | Trial full-size resident Vulkan paged-KV mirror for dyn-seqlen attention. | Temporary hook, resident K/V buffers, partial row uploads, and a block-table-addressed resident attention shader passed focused Vulkan paged-attn parity and Vulkan/model/server checks plus release build. Endpoint candidate never reached `inference_prewarm_complete`: server logged normal health responses and Vulkan decode weight prewarm, but no `background inference prewarm complete`; the stopped run logged `90` health responses, `prewarm_weight_log_seen=true`, `prewarm_complete_log_seen=false`. Current Qwen3.5-4B sizing reports `kv_cache_gb=65.7457152`, making full-size resident mirroring the wrong default shape. | Rejected and removed. Do not retry whole-pool resident mirroring; next K/V work needs sparse/active-slot or compact-window residency. |
| A053 | Probe post-PR1004 current Vulkan output with ambiguous prompts. | Rebuilt release Vulkan binary and ran four post-prewarm profiled streams. Server log showed Vulkan initialization and decode-weight prewarm, and responses were HTTP 200 with non-empty visible text and empty reasoning, but the `"number after N"` prompt wording produced `"5"`, `"1"`, `"5"`, `"13"` against an arithmetic expected-string checker. | Inconclusive harness/prompt issue, not treated as backend failure; superseded by A054. |
| A054 | Confirm post-PR1004 Vulkan correctness/profile with unambiguous arithmetic prompts. | Post-prewarm profiled streams returned correct visible texts `"6"`, `"11"`, `"16"`, `"13"` with empty reasoning. Server log showed `Vulkan device initialized`, `Vulkan decode weight cache prewarmed`, and `background inference prewarm complete`; wall was `3.258736s`, counters were `7` jobs / `3` batches / `7` rows / max batch `3`. Live `seq_len=1` top stages included `mlp:fused` `127.327ms`, `gdn:recurrent` `111.416ms`, `gdn:in_proj` `95.951ms`, and `gdn:gates` `59.722ms`. | Keep as post-merge correctness/profile evidence; no source change. |
| A055 | Batch Vulkan GDN gates upload/readback transfer submissions. | Added Vulkan-only helpers so `dispatch_gdn_gates_cached` uploads `a`/`b` in one transfer submission and reads `beta`/`g` in one readback submission, with rollback env `KILN_DISABLE_VULKAN_GDN_GATES_BATCHED_TRANSFERS=1`. Validation passed: `cargo fmt --check`, `cargo check -p kiln-vulkan-kernel`, `cargo check -p kiln-model --features vulkan`, focused GDN gates parity, full Vulkan parity (`29 passed`), and release Vulkan build. Profiled same-binary A/B returned correct texts on both arms and improved live `gdn:gates seq_len=1` from rollback `72.540ms` to candidate `60.203ms`; no-profile pairs were mixed (`+0.104s`, then `-0.016s`) with correct outputs. | Accepted as a targeted Vulkan gates overhead reduction; no CUDA/Metal source changes. Rollback via `KILN_DISABLE_VULKAN_GDN_GATES_BATCHED_TRANSFERS=1`. |

Additional validation after A044:

- Temporary instrumentation build passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Profiled Vulkan server run with four concurrent streams,
  `KILN_DECODE_BATCH_WAIT_US=5000`,
  `KILN_PROFILE_VULKAN_MLP_KERNEL_STAGES=1`,
  `KILN_PROFILE_MLP_STAGES=1`, `max_tokens=3`, and
  `chat_template_kwargs: {"enable_thinking": false}`.
- All four streams returned HTTP 200 with non-empty visible content and empty
  reasoning text.
- Temporary instrumentation was removed after profiling.

Additional validation after A048:

- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- A045, A046, and A047 temporary backend-profile runs retained as artifacts;
  temporary source instrumentation was removed before the accepted A048 build.
- A048 no-profile Vulkan server run waited for
  `inference_prewarm_complete=true`, confirmed the server log contained both
  `Vulkan decode weight cache prewarmed` and `background inference prewarm
  complete` before `A048_MEASURE_START`, and returned four HTTP 200 streaming
  responses with non-empty visible content and empty reasoning.

Additional validation after rejected A050:

- `cargo fmt --check`.
- `cargo check -p kiln-model --features vulkan`.
- `cargo test -p kiln-vulkan-kernel gdn_decode_gates_recurrent_rmsnorm -- --nocapture`.
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Same-binary candidate/rollback endpoint A/B with prewarm readiness confirmed
  and `chat_template_kwargs: {"enable_thinking": false}`.
- Candidate source changes were removed after the rejected A/B.

Additional validation after rejected A043:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused batched MLP parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  mlp_decode_batched_matches_cpu_reference -- --nocapture`.
- Temporary candidate passed full Vulkan kernel parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback n8 server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after the paired endpoint A/B was not a reliable
  win.

Additional validation after rejected A042:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused packed-BF16 MLP parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed full Vulkan kernel parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after rollback beat candidate.

Additional validation after rejected A041:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused packed-BF16 MLP parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after rollback beat candidate.

Additional validation after rejected A040:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused packed-BF16 MLP parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Temporary candidate passed full Vulkan kernel parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after the paired endpoint A/B was not a reliable
  win.

Additional validation after A039:

- Profiled Vulkan server run with four concurrent streams,
  `KILN_DECODE_BATCH_WAIT_US=5000`, `KILN_PROFILE_FULL_ATTN_STAGES=1`,
  `KILN_PROFILE_GDN_STAGES=1`, `KILN_PROFILE_MLP_STAGES=1`, `max_tokens=3`,
  and `chat_template_kwargs: {"enable_thinking": false}`.
- All four streams returned HTTP 200 with non-empty visible content and empty
  reasoning text.

Additional validation after A038:

- `rustfmt --edition 2024 crates/kiln-vulkan-kernel/build.rs
  crates/kiln-vulkan-kernel/src/pipeline.rs
  crates/kiln-vulkan-kernel/src/kernels.rs
  crates/kiln-vulkan-kernel/tests/gdn_parity.rs`
- `cargo check -p kiln-vulkan-kernel`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity
  gdn_in_proj_decode_batched_bf16_packed_weights -- --nocapture`
- `KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_ROW_PAIR=1 cargo test
  -p kiln-vulkan-kernel --test gdn_parity
  gdn_in_proj_decode_batched_bf16_packed_weights_row_pair_matches_cpu_reference
  -- --nocapture`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo check -p kiln-model --features vulkan`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- Two same-binary endpoint A/B pairs returned HTTP 200 with non-empty visible
  content and empty reasoning text.

Additional validation after rejected A037:

- Temporary candidate passed
  `cargo test -p kiln-model decode_batcher_default --lib`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Env-only wait sweep and same-binary rollback/candidate server runs returned
  HTTP 200 with non-empty visible content and empty reasoning text.
- Source change was removed after same-binary rollback beat candidate.

Additional validation after rejected A036:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused BF16 generic linear parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  linear_decode_batched_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Temporary candidate passed focused packed-BF16 MLP parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after rollback beat candidate.

Additional validation after rejected A035:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate and rollback passed focused paged-attn parity:
  `cargo test -p kiln-vulkan-kernel --test gdn_parity
  paged_attn_decode_batch_matches_cpu_reference -- --nocapture`.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after the sampled actor rollback beat candidate.

Additional validation after rejected A034:

- Temporary candidate passed `cargo check -p kiln-vulkan-kernel`.
- Temporary candidate passed
  `cargo test -p kiln-vulkan-kernel full_attn_qkv_decode --test gdn_parity -- --nocapture`
  with `4` tests.
- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
  with `30` tests.
- Temporary candidate passed `cargo check -p kiln-server --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- Candidate and rollback server runs returned HTTP 200 with non-empty visible
  content and empty reasoning text.
- Source change was removed after rollback beat candidate.

Additional validation after A033:

- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- Profiled no-env Vulkan server run with four concurrent different-length
  prompts, `stream=true`, and
  `chat_template_kwargs: {"enable_thinking": false}`.

Additional validation after A032:

- `rustfmt --edition 2024 crates/kiln-vulkan-kernel/build.rs crates/kiln-vulkan-kernel/src/pipeline.rs crates/kiln-vulkan-kernel/src/kernels.rs`
- `cargo check -p kiln-vulkan-kernel`
- `cargo test -p kiln-vulkan-kernel gdn_in_proj_decode_batched --test gdn_parity -- --nocapture`
- `cargo check -p kiln-model --features vulkan`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo check -p kiln-server --features vulkan`
- Two no-env Vulkan live streaming endpoint A/B pairs against
  `KILN_DISABLE_VULKAN_GDN_IN_PROJ_BATCH_PAIR_QKV_Z=1`.

Additional validation after rejected A031:

- Temporary candidate passed `cargo check -p kiln-model --features vulkan`.
- Temporary candidate passed
  `cargo build --release --features vulkan --bin kiln --bin kiln-bench`.
- `KILN_BATCHING_ENGINE=1` actor baseline, candidate, and same-binary rollback
  all returned HTTP 200 with non-empty visible content and empty reasoning text.
- Source change was removed after rollback beat candidate.

Additional validation after A030:

- `rustfmt --edition 2024 crates/kiln-model/src/generate.rs crates/kiln-server/src/state.rs`
- `cargo test -p kiln-model test_decode_batcher_default_mixed_seq_lens_backend_policy --lib`
- `cargo check -p kiln-model --features vulkan`
- `cargo check -p kiln-server --features vulkan`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- No-env Vulkan streaming smoke with four concurrent different-length prompts,
  `stream=true`, and `chat_template_kwargs: {"enable_thinking": false}`.

Additional validation after A025:

- `rustfmt --edition 2024 crates/kiln-core/src/tokenizer.rs crates/kiln-server/src/api/completions.rs`
- `git diff --check`
- `cargo check -p kiln-server`
- `cargo test -p kiln-core qwen35_4b_chat_template_can_disable_thinking`
- `cargo test -p kiln-server chat_template_kwargs`
- `cargo test -p kiln-server qwen35_no_think_text_is_not_a_control_flag`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `KILN_BATCHING_ENGINE=1 KILN_MODEL_PATH=Qwen3.5-4B ./target/release/kiln serve` direct and batch smokes for default thinking, explicit `enable_thinking=false`, and literal `/no_think` prompt text.

CUDA and Metal checks were attempted but are environment-blocked on this Linux
host before project typecheck: CUDA lacks `nvcc`; Metal `objc2` requires an
Apple target.

Additional validation after A026:

- `rustfmt --edition 2024 crates/kiln-vulkan-kernel/build.rs crates/kiln-vulkan-kernel/src/pipeline.rs crates/kiln-vulkan-kernel/src/kernels.rs crates/kiln-vulkan-kernel/tests/gdn_parity.rs crates/kiln-model/src/backend/vulkan.rs`
- `cargo check -p kiln-vulkan-kernel`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `git diff --check`
- Serial paged A/B on `main` with rollback env
  `KILN_DISABLE_VULKAN_BF16_PACKED_FULL_ATTN_QKV_WEIGHTS=1`
- `KILN_BATCHING_ENGINE=1 KILN_MODEL_PATH=Qwen3.5-4B ./target/release/kiln serve`
  direct, explicit batch, and concurrent live-batcher smokes with
  `chat_template_kwargs: {"enable_thinking": false}`

Additional validation after A028:

- `rustfmt --edition 2024 crates/kiln-vulkan-kernel/build.rs crates/kiln-vulkan-kernel/src/pipeline.rs crates/kiln-vulkan-kernel/src/kernels.rs crates/kiln-vulkan-kernel/tests/gdn_parity.rs crates/kiln-model/src/backend/vulkan.rs`
- `cargo check -p kiln-vulkan-kernel`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `git diff --check`
- Serial paged smoke on `main`
- Two direct server A/B pairs against rollback env
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1` for explicit batch
  and four concurrent live-batcher requests

Additional validation after rejected A029:

- Temporary row-pair BF16 MLP shaders passed focused parity including batch `9`.
- Temporary row-pair BF16 MLP code passed `cargo check -p kiln-vulkan-kernel`,
  `cargo check -p kiln-model --features vulkan`, release Vulkan build, and
  `git diff --check`.
- Same-binary serial A/B favored rollback, so the row-pair code was removed.

Additional validation after A027:

- `rustfmt --edition 2024 crates/kiln-vulkan-kernel/build.rs crates/kiln-vulkan-kernel/src/pipeline.rs crates/kiln-vulkan-kernel/src/kernels.rs crates/kiln-vulkan-kernel/tests/gdn_parity.rs crates/kiln-model/src/backend/vulkan.rs`
- `cargo check -p kiln-vulkan-kernel`
- `cargo check -p kiln-model --features vulkan`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity mlp_decode_bf16_packed_weights_match_cpu_reference -- --nocapture`
- `cargo test -p kiln-vulkan-kernel --test gdn_parity -- --nocapture`
- `cargo build --release --features vulkan --bin kiln --bin kiln-bench`
- `git diff --check`
- Serial paged A/B on `main` with rollback env
  `KILN_DISABLE_VULKAN_BF16_PACKED_MLP_DECODE_WEIGHTS=1`
- `KILN_BATCHING_ENGINE=1 KILN_MODEL_PATH=Qwen3.5-4B ./target/release/kiln serve`
  direct, explicit batch, and concurrent live-batcher smokes with
  `chat_template_kwargs: {"enable_thinking": false}`
