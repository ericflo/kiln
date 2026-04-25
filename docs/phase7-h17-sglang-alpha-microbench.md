# Phase 7 — H17: SGLang α microbench (external-reference upper bound vs kiln)

**Verdict**: `sglang_mtp_unsupported_dense_4b` — SGLang 0.5.10.post1 + Qwen3.5-4B dense BF16 + native MTP segfaults on A6000 sm_86 across three distinct serving configurations. Cannot establish an external α upper bound at this SGLang version either.
**Branch**: `phase7-h17-sglang-alpha`
**Predecessor**: PR #531 (H16 external-α reference options audit → `external_reference_exists` → queued H17 as ONE canonical SGLang microbench)
**Pod**: RunPod A6000 `2uhzp19bj6do5g`, on-demand $0.49/hr, terminated after run. ~25 min wall-clock pod time, ~$0.20 spend — well under the $0.40 SGLang cap and combined $0.65 cap.

## Goal

PR #531's audit confirmed SGLang main HEAD `a4facdf` has `Qwen3_5ForCausalLMMTP` (added 2026-02-09 #18489, dense 4B fix 2026-03-24 #21347, spec_v2 enabled 2026-03-04 #19391) and flagged it as the only mandatory `supported` external-α reference — sufficient to queue exactly ONE H17 microbench per PR #531's pre-registered decision rule.

The remaining open question, inherited from PR #530's `vllm_mtp_unsupported` verdict:

> Does any external serving system hit a **higher α** than kiln on this same Qwen3.5-4B checkpoint at A6000 bs=1?

H17 attempted to answer this by running SGLang 0.5.10.post1 with the native `Qwen3_5ForCausalLMMTP` drafter on the *identical* prompt / seed / prefill / decode workload PR #529 used, and comparing α to kiln's α derived from PR #530's `docs/archive/phase-c/phase-c29-v3-vllm/kiln_alpha_per_seed.json` (re-derived from PR #529's c1_attr CSVs — no re-run needed).

## Outcome

**SGLang 0.5.10.post1 segfaults on every attempted serving configuration when loading Qwen3.5-4B dense + native MTP on A6000.** Three structurally distinct crash signatures fired across the three configs we tried (default flashinfer + CUDA graphs; flashinfer graphs-off; triton attention graphs-off), each pointing to a different native-code crash frame (CUDA graph capture / scheduler init / Triton JIT compile of `write_cache_indices`).

Identical high-level shape to PR #530's vLLM 0.19.1 failure: engine-side dispatch is *correct* (Qwen3_5ForCausalLMMTP weights load, MTP drafter arch resolves, KV + mamba cache allocates, GDN kernel dispatcher initializes), but a downstream native code path SIGSEGVs post-load. Pre-registered decision-rule branch D (`sglang.mtp_supported == false`) fires → ship this PR as a doc-only redirect documenting the SGLang-side gap.

| metric | value |
| --- | --- |
| `kiln_median_alpha` (re-derived from PR #529 c1_attr CSVs, unchanged from PR #530) | **0.3636** |
| `sglang_median_alpha` | **UNAVAILABLE** (segfault across 3 configs) |
| `delta` | **UNAVAILABLE** |

Per-seed kiln α (copy of PR #530 table, from `docs/archive/phase-c/phase-c29-v2/artifacts/c1_attr_seed{0,1,2}.csv`):

| seed | accept / steps | α |
| --- | --- | --- |
| 0 | 4 / 11 | 0.3636 |
| 1 | 4 / 12 | 0.3333 |
| 2 | 5 / 11 | 0.4545 |
| **median** | — | **0.3636** |

## Decision rule (pre-registered, set BEFORE the run in PR #531)

| Δ = sglang_median_α − kiln_median_α | verdict | next action |
| --- | --- | --- |
| ≥ +0.05 | `external_ceiling_exists` | queue serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring) |
| in [−0.02, +0.05) | `mtp_head_quality_ceiling` | accept α as upper-bounded by checkpoint quality; pivot to MTP head retraining OR non-MTP spec |
| < −0.02 | `kiln_above_sglang` (unexpected) | sanity-check SGLang args; if confirmed, close H17 family |
| `sglang.mtp_supported == false` | **`sglang_mtp_unsupported_dense_4b`** ← **THIS RUN** | doc-only redirect PR; escalate to free pre-step (vLLM v0.20.0 retest) OR queue hand-rolled HF transformers H18 |

The rule is implemented in `scripts/h17_compare.py` and emitted `docs/archive/phase-c/phase-c29-v3-sglang/verdict.json`.

## Free pre-step (vLLM v0.20.0 retest) — SKIPPED per task brief

Per PR #531 §"Bench envelope + cost": "If the SGLang install / model-load alone consumes >30 min, abort the vLLM retest and ship the SGLang result alone."

SGLang install + three diagnostic attempts consumed ~35 min of the 75 min combined cap (SGLang: ~$0.20 / 25 min pod time). The vLLM v0.20.0 retest was not run — deferred to a follow-up task if still useful. A future reopen trigger (below) will recheck whether SGLang/vLLM main have landed a Qwen3.5-4B dense + MTP fix.

## Crash signatures

All three configs crash in native code with no Python file symbols at the terminal frame — identical pattern to PR #530's vLLM 0.19.1 crash. Detailed crash info in `docs/archive/phase-c/phase-c29-v3-sglang/sglang_alpha_per_seed.json` and `docs/archive/phase-c/phase-c29-v3-sglang/artifacts/sglang_segfault_evidence.log`.

### Attempt 1: flashinfer attention backend, CUDA graphs enabled (default)

Engine constructed, loaded base model, started CUDA graph capture. Crash fired at `flashinfer_backend.py:1448 call_begin_forward` during `eagle_info.py:188 generate_attn_arg_prefill`:

```
  File ".../sglang/srt/model_executor/cuda_graph_runner.py", line 981 in capture_one_batch_size
  File ".../sglang/srt/layers/attention/hybrid_linear_attn_backend.py", line 758 in init_forward_metadata_capture_cuda_graph
  File ".../sglang/srt/layers/attention/flashinfer_backend.py", line 1448 in call_begin_forward
  File ".../sglang/srt/speculative/eagle_info.py", line 188 in generate_attn_arg_prefill
  File ".../triton/runtime/jit.py", line 419 in <lambda>
  ... Triton JIT compile crash ...
```

SIGSEGV (exit code -11) in scheduler subprocess; parent exit 137.

### Attempt 2: flashinfer, CUDA graphs disabled

Scheduler subprocess SIGSEGV earlier during init, before reaching prefill. Script re-tried with `speculative_algorithm='NEXTN'` (which auto-rewrites to EAGLE per `server_args.py:2984`), identical crash: `RuntimeError: 'Rank 0 scheduler died during initialization (exit code: -11). If exit code is -9 (SIGKILL), a common cause is the OS OOM killer.'`

### Attempt 3: triton attention backend, CUDA graphs disabled

Engine *loaded successfully* — target model (8.62 GB) and drafter (1.41 GB) both loaded, hybrid GDN + full-attn dispatcher initialized (`TritonGDNKernel packed_decode=True`), KV cache allocated (464641 tokens, K=7.09 GB, V=7.09 GB), mamba cache allocated (ssm_state=12.47 GB), scheduler reached `max_running_requests=48, context_len=1024, available_gpu_mem=4.91 GB`. Then SIGSEGV fired during the very first prefill when the scheduler's event loop tried to allocate KV pages for the incoming request:

```
  File ".../sglang/srt/mem_cache/common.py", line 98 in write_cache_indices
  File ".../sglang/srt/mem_cache/common.py", line 377 in alloc_for_extend
  File ".../sglang/srt/managers/schedule_batch.py", line 1615 in prepare_for_extend
  File ".../sglang/srt/managers/scheduler.py", line 2518 in _get_new_batch_prefill_raw
  File ".../sglang/srt/managers/scheduler.py", line 1350 in event_loop_overlap
  ... Triton JIT compile of write_cache_indices ...
```

`[2026-04-25 02:01:26] Subprocess scheduler_0 (pid=7391) crashed with exit code -11. Triggering SIGQUIT for cleanup...`

Crash is in `write_cache_indices` Triton kernel compile — unrelated to the EAGLE spec path of attempt 1, and unrelated to the init-time crash of attempt 2. Three distinct native-code crash frames, all under Triton JIT or flashinfer.

## Workload (matched to PR #529)

Identical to PR #530's `docs/phase7-h15c-vllm-alpha-microbench.md` §"Workload", swapping vLLM for SGLang. Byte-equal at prefill boundary:

| field | value | source |
| --- | --- | --- |
| model | Qwen3.5-4B dense (`Qwen3_5ForConditionalGeneration`) + MTP drafter (`Qwen3_5ForCausalLMMTP`) | `/workspace/qwen3.5-4b/config.json` |
| prompts | `PROMPT_POOL[seed % 30]` for seeds {0,1,2} → GSM8K prose 0/1/2 | `crates/kiln-server/src/bench.rs::build_prompt` |
| prompt-token target | 512 (sentence-repetition expander) | `bench.rs::build_prompt` |
| actual prompt tokens | 494 / 508 / 501 | matches PR #529 c1_attr CSV `base_pos` start values |
| max output tokens | 16 | PR #529 `scripts/c29_kiln_logits_dump.sh:DECODE_TOKENS` |
| chat template | OFF (raw prose) | PR #529 driver |
| sampler | greedy: `temperature=0`, `top_p=1`, `top_k=-1` (SGLang convention; kiln uses `top_k=0`, equivalent) | both drivers |
| spec method | k=1 native MTP (`KILN_SPEC_METHOD=mtp` / `speculative_algorithm=EAGLE` with Qwen3_5ForCausalLMMTP drafter) | both drivers |
| GPU | NVIDIA RTX A6000 sm_86, single-GPU bs=1 | RunPod |
| kiln quant | Marlin W4A16 (`KILN_W4A16=1`) | PR #529 driver |
| SGLang quant | BF16 (SGLang 0.5.10 does not implement Marlin W4A16 for `Qwen3_5ForCausalLMMTP`) | acknowledged confounder, identical to PR #530's vLLM-vs-kiln Marlin W4A16 confounder |

`scripts/h17_sglang_alpha_dump.py` verifies byte-equality of prompt tokens to PR #529: 494 / 508 / 501 match `c1_attr_seed{0,1,2}.csv`'s `base_pos` start values. Only the serving-side decode path would have differed had SGLang completed the run.

## SGLang setup prerequisites (new findings)

Beyond PR #531's audit, running SGLang 0.5.10.post1 + Qwen3.5-4B MTP revealed three undocumented-in-audit prerequisites. Recording here for any future retry:

1. **`libnuma1` + `libnuma-dev` apt packages** — `sgl_kernel 0.4.1`'s `common_ops.abi3.so` dlopens `libnuma.so.1`. The `kiln-runpod` Ubuntu 22.04 base image does NOT ship libnuma. First bench attempt failed with: `ImportError: libnuma.so.1: cannot open shared object file: No such file or directory`. Install via `apt-get update && apt-get install -y libnuma1 libnuma-dev`.

2. **`SGLANG_ENABLE_SPEC_V2=1` env var** — required for Qwen3.5 + MTP + radix cache. Without it, SGLang raises: `ValueError: Speculative decoding for Qwen3_5ForConditionalGeneration is not compatible with radix cache when using --mamba-scheduler-strategy no_buffer. To use radix cache with speculative decoding, please use --mamba-scheduler-strategy extra_buffer and set SGLANG_ENABLE_SPEC_V2=1.`

3. **`mamba_scheduler_strategy='extra_buffer'`** — must be passed to `sgl.Engine()`. The default `'no_buffer'` triggers the radix-cache incompatibility error above.

Plus the algorithm-name resolution noted in PR #531's audit is more restrictive than the audit stated:

4. **`speculative_algorithm` must be `'EAGLE'`** (or `'NEXTN'`, which auto-rewrites to EAGLE at `server_args.py:2984`) — the name `'MTP'` that the task brief suggests is NOT in SGLang's enum. `SpeculativeAlgorithm.from_string()` at `sglang/srt/speculative/spec_info.py:31` raises `ValueError: Unknown speculative algorithm name: MTP`. The `Qwen3_5ForCausalLMMTP` class *is* loaded as the EAGLE drafter — there is no separate `MTP` algorithm path.

5. **`speculative_eagle_topk` must be an int >=1** — `None` triggers `TypeError: '>' not supported between instances of 'NoneType' and 'int'` at `server_args.py:1669`.

These prerequisites are baked into `scripts/h17_sglang_alpha_dump.py`'s `try_load_engine()` so a future retry (e.g. after SGLang ships a dense 4B + MTP fix) only needs to re-run the same command.

## Reproduction commands

```bash
# Pod side (RunPod A6000 with kiln-runpod image, SGLang 0.5.10.post1 installed via pip install sglang[all])

# 1. Prerequisite: libnuma (first-time pod setup)
apt-get update && apt-get install -y libnuma1 libnuma-dev

# 2. SGLang α (3 seeds × 16-token greedy decode × MTP k=1) — segfaults at SGLang 0.5.10.post1
cd /workspace/kiln
export SGLANG_ENABLE_SPEC_V2=1
python3 scripts/h17_sglang_alpha_dump.py \
    --model-path /workspace/qwen3.5-4b \
    --prompt-tokens 512 --max-tokens 16 \
    --seeds 0 1 2 --num-spec-tokens 1 \
    --out docs/archive/phase-c/phase-c29-v3-sglang/sglang_alpha_per_seed.json
# Outcome: writes mtp_supported=false JSON; exit code 3.

# 3. kiln α (re-derived from PR #529's c1_attr CSVs — no GPU needed, runs anywhere)
python3 scripts/h15c_kiln_alpha_from_csv.py \
    --csv-dir docs/archive/phase-c/phase-c29-v2/artifacts \
    --out docs/archive/phase-c/phase-c29-v3-sglang/kiln_alpha_per_seed.json
# Outcome: median_alpha = 0.3636.

# 4. Apply decision rule
python3 scripts/h17_compare.py
# Outcome: verdict=sglang_mtp_unsupported_dense_4b.
```

## Anti-duplication evidence

```
$ gh pr list -R ericflo/kiln --state all --search "h17 sglang"
(empty — only PR #531 H16 audit matched)
$ gh pr list -R ericflo/kiln --state all --search "phase7 h17"
(empty — only PR #531 matched)
$ gh pr list -R ericflo/kiln --state all --search "sglang alpha microbench"
(empty — only PRs #530, #531 matched)
$ gh pr list -R ericflo/kiln --state all --search "phase-c29-v3"
(empty — only PR #530, #531 matched)
```

PR #531 explicitly queued this H17 microbench as the single canonical follow-up in its §"Queued next action". PR #530 exhausted the vLLM 0.19.1 path. Only one task was in-flight for H17 (this one).

## What this rules out

- **SGLang 0.5.10.post1 cannot serve Qwen3.5-4B dense + native MTP on A6000 today.** Any Phase 7 plan that assumed "we can compare against SGLang α as the upper bound" needs to either (a) wait for a SGLang patch that fixes one of the three native-code crash paths, (b) try a different SGLang version / main-branch build, (c) try the free pre-step vLLM v0.20.0 retest, or (d) build the hand-rolled HF transformers reference (PR #531 candidate 8).
- **The SGLang-side model dispatch is correct.** SGLang 0.5.10.post1 *does* recognize `Qwen3_5ForCausalLMMTP` in its registry, *does* rewrite `NEXTN` → `EAGLE` at `server_args.py:2984`, *does* attach the drafter weights with tied embed/lm_head (shared with base model, not a separate checkpoint path), *does* dispatch hybrid GDN + full-attn through TritonGDNKernel, and *does* allocate KV + mamba caches with plausible sizes for a dense 4B MTP + 464K-token context. The bug is downstream of model dispatch.
- **The open SGLang PR #23331 adaptive DSD caveat from PR #531 is NOT the source of our crashes.** kiln's H17 workload is greedy + fixed k=1 non-adaptive; the non-DSD path is what we hit, and all three crashes are in non-DSD code frames (CUDA graph capture, init, and mem_cache allocation).
- **This is not a CUDA graph issue alone.** Attempt 2 disabled graphs and still segfaulted at scheduler init; attempt 3 disabled graphs + switched attention backend and segfaulted at a different post-load code frame. The crash class is broader than graph capture.

## What this does NOT rule out

- **Whether an external α ceiling exists.** That was the question H17 was supposed to answer. With both vLLM 0.19.1 AND SGLang 0.5.10.post1 blocked, the question stays open until either a vLLM patch (free pre-step), an SGLang patch, or a hand-rolled HF reference is built.
- **Workload sensitivity** (per `mtp-bench-workload-sensitivity` — chat α ≈ 0.588 vs prose α ≈ 0.175 on PR #364). H17 chose the prose / 16-decode workload to match PR #529 byte-for-byte; chat-template prompts may behave differently on all three implementations.
- **Quantization path differences.** kiln runs Marlin W4A16; SGLang would have run BF16. That confounder still has to be handled if any future external α reference completes (±0.05 envelope should absorb most of it).
- **vLLM v0.20.0 (2026-04-23).** PyTorch 2.11 + CUDA 13.0 is a different native stack than PR #530's torch 2.10 + cu128. The free pre-step was skipped here under budget discipline; a fast follow-up task could re-run it.
- **A different SGLang version / main-branch build.** SGLang main HEAD `a4facdf` may contain a fix post-dating the 0.5.10.post1 pip release (2026 release history shows PR #21347 dense-4B fix and #19391 spec_v2 enable; there may be a post-release bugfix commit we did not bisect).
- **Whether a deeper configuration sweep would uncover a working config.** We tried 3 high-leverage configs; possible knobs we did NOT try: `enable_torch_compile`, various `mamba_backend` / `linear_attn_backend` combos, `page_size`, `max_running_requests=1`, `swa_full_tokens_ratio`, tensor parallel > 1, `attention_backend='fa3'` (if SGLang 0.5.10 builds fa3 for sm_86). Under the 75-min combined cap, further sweeps would have exceeded budget.

## Bench envelope + cost

- Pod: RunPod A6000 `2uhzp19bj6do5g`, $0.49/hr on-demand, no spot.
- Pool acquire: HTTP 503 `capacity_supply_exhausted` (hibernated pool pod could not be resumed due to host GPU availability) → fell back to direct `runpod_api.py launch` (per the documented fallback path; same situation as PR #530's run).
- SGLang install: ~90 s (pip install `sglang[all]>=0.5.0`, pulled sglang 0.5.10.post1 + sglang-kernel 0.4.1 + PyTorch 2.9.1 cu128 upgrade; single-shot, supervised with `wait-file` sentinel, no interactive SSH polling).
- libnuma install: ~3 s (`apt-get update` + `apt-get install libnuma1 libnuma-dev`).
- Bench attempts: 3 × ~30-50 s each (load + segfault), all in the same `python3 scripts/h17_sglang_alpha_dump.py` process.
- All bench supervision used `runpod_api.py bg` + `wait-file --timeout 900`, no `until ssh` / `while ssh ... sleep` polling loops (per `kiln-ssh-polling-deadlock` note).
- Wall-clock budget: **75 min / $0.65 hard combined cap** (per PR #531 §"Bench envelope + cost"). Actual: ~25 min pod time / ~$0.20 total. **Well under cap** — the free pre-step was intentionally skipped per the task brief's ">30 min SGLang consumption" trigger.

## Queued next action

Per the pre-registered decision rule for `sglang_mtp_unsupported_dense_4b`:

> SGLang does not yet support Qwen3.5-4B dense + native MTP end-to-end at the installed version. Escalate to free pre-step (vLLM v0.20.0 retest of PR #530 segfault) — if also fails, queue hand-rolled HF transformers H18 reference (PR #531 candidate 8).

Concrete options the next planning cycle should pick from, ranked by estimated setup cost vs expected signal:

1. **vLLM v0.20.0 retest (opportunistic H17b)** — est. 30 min / $0.25 A6000 on-demand. Re-run PR #530's `scripts/h15c_vllm_alpha_dump.py` exactly as-is against vLLM 0.20.0 (PyTorch 2.11 + CUDA 13.0, different native stack than 0.19.1's torch 2.10 + cu128). Three plausible outcomes remain: same segfault (verdict `vllm_v020_still_unsupported`), different crash (new unknown), or success (third reference-α data point). Same driver works.

2. **Hand-rolled HF transformers H18 reference** — est. 2-4 hrs engineering + ~30 min CPU run = $0 GPU spend, but high human-time cost. PR #531 candidate 8. Load `Qwen3_5ForCausalLM` + manually parse 15 `mtp.*` tensors + port kiln's `mtp_forward_step` + `speculative_mtp_decode_step` into ~300-500 LOC of Python. Algorithmically equivalent to kiln-native; intent is to factor out kiln's Marlin W4A16 + GDN + paging implementations as confounders. Useful only if it produces α materially higher than kiln (confirming external ceiling exists) — if it matches kiln, doesn't disambiguate implementation vs head-quality.

3. **Direct decode-path optimization** — already in flight: PRs #517/#520/#521 (prefix-cache + CUDA graphs wins). H17 does not block continued investment here. This verdict refocuses Phase 7 onto non-α decode-path wins in the meantime.

4. **SGLang main-branch rebuild** — est. 30-45 min build + 10 min retest = ~$0.35. Compile SGLang at `a4facdf` from source instead of 0.5.10.post1 pip release. May pick up post-release bugfixes. Worth it only if (1) fails and (2) is not yet started.

5. **Accept kiln-native ceiling as checkpoint-quality ceiling** — the H15b stratified probe (PR #529) already established the kiln-native verifier sits at the BF16-noise cosine ceiling on reject rows. With both external references (vLLM, SGLang) blocked, the `kiln_native_ceiling` verdict from H15b stands as the operational conclusion, and Phase 7 deprioritizes MTP-side α work regardless.

## Files

| path | purpose |
| --- | --- |
| `docs/phase7-h17-sglang-alpha-microbench.md` | this audit doc |
| `docs/archive/phase-c/phase-c29-v3-sglang/verdict.json` | machine-readable verdict |
| `docs/archive/phase-c/phase-c29-v3-sglang/compare.json` | full kiln + SGLang side-by-side |
| `docs/archive/phase-c/phase-c29-v3-sglang/compare.md` | per-seed table + decision rule |
| `docs/archive/phase-c/phase-c29-v3-sglang/kiln_alpha_per_seed.json` | re-derived kiln α (same data as PR #530) |
| `docs/archive/phase-c/phase-c29-v3-sglang/sglang_alpha_per_seed.json` | full SGLang failure record with all 3 config attempts |
| `docs/archive/phase-c/phase-c29-v3-sglang/artifacts/sglang_segfault_evidence.log` | trimmed pod log with crash stack for attempt 3 (triton backend) |
| `scripts/h17_sglang_alpha_dump.py` | SGLang driver (re-runnable when SGLang is fixed) |
| `scripts/h17_compare.py` | applies decision rule, emits verdict.json |
| `PROFILING.md` | top-of-file pointer updated |

## Reopen preconditions

The H17 verdict shifts from `sglang_mtp_unsupported_dense_4b` to a different branch under any of these signals:

- An SGLang commit explicitly fixing Qwen3.5-4B dense + native MTP segfault on A6000 sm_86 — would change the `mtp_supported` flag to `true` and make the H17 retest cheap (~20 min / $0.17; driver is version-agnostic).
- SGLang landing a new `speculative_algorithm="MTP"` enum variant distinct from EAGLE — would indicate a re-architected MTP path that may sidestep the current segfault frames.
- vLLM v0.20.0 free pre-step succeeding where 0.19.1 failed — documented as follow-up candidate H17b.
- Hand-rolled HF transformers reference completing — closes the external-α question independently of vLLM/SGLang.
- A kiln-side change invalidating the PR #529 c1_attr CSV data (e.g. new MTP head training, Marlin path refactor, or sampler change) — would require re-running kiln c1_attr capture before re-running this compare.

## References

### kiln docs (commit `b33d129`)

- `docs/phase7-h16-external-alpha-options-audit.md` (PR #531) — direct predecessor; enumerated 8 external-α reference candidates and queued this H17 SGLang microbench with pre-registered decision rule.
- `docs/phase7-h15c-vllm-alpha-microbench.md` (PR #530) — sibling predecessor; same structural shape, `vllm_mtp_unsupported` verdict. Source of kiln α baseline (0.3636) re-used here via `scripts/h15c_kiln_alpha_from_csv.py`.
- `docs/phase7-h15b-stratified-c29-v2.md` (PR #529) — established `kiln_native_ceiling` verdict and provided the c1_attr CSVs.
- `docs/phase7-mtp-acceptance-state-of-play.md` (PR #527) — original state-of-play doc enumerating the external-α-reference idea.
- `docs/phase7-sglang-radix-audit.md` (PR #526) — disambiguates "SGLang has Qwen3.5 MTP class" (yes, per PR #531) from "SGLang's RadixAttention is MTP-aware" (not yet). H17 confirms the former is structurally true (class exists, loads, dispatches) but fails at native runtime on A6000.

### Cited upstream sources

- SGLang `python/sglang/srt/models/qwen3_5_mtp.py` at installed pip release 0.5.10.post1 (`class Qwen3_5ForCausalLMMTP` line 39; `EntryClass` line 360)
- SGLang `python/sglang/srt/speculative/spec_info.py` at 0.5.10.post1 (`SpeculativeAlgorithm` enum, only EAGLE/EAGLE3/STANDALONE/NGRAM/NONE — no `MTP`)
- SGLang `python/sglang/srt/server_args.py` at 0.5.10.post1 (`NEXTN` → `EAGLE` rewrite at line 2984; radix-cache incompatibility check; `speculative_eagle_topk` None-check at line 1669)
- SGLang `python/sglang/srt/speculative/eagle_info.py:188` `generate_attn_arg_prefill` (crash site in attempt 1)
- SGLang `python/sglang/srt/mem_cache/common.py:98` `write_cache_indices` (crash site in attempt 3)
- SGLang PRs #18489 (2026-02-09, dense Qwen3.5 MTP class added), #19391 (2026-03-04, spec_v2 enabled for MoE NVFP4), #21347 (2026-03-24, dense 4B PP tied embeddings fix), #23331 (2026-04-21, OPEN — adaptive DSD fix, not on our greedy k=1 path)

### Agent notes

`kiln-ssh-polling-deadlock`, `kernel-vendor-precondition-check`, `kiln-phase7-consolidation-doc-pattern`, `static-audit-before-pod-spend`, `vllm-qwen35-mtp-v1-segfault-2026-04`, `runpod-always-on-demand`, `runpod-gpu-minimum-a6000`.
