# Phase 7 — H15c: vLLM α microbench (external-reference upper bound vs kiln)

**Verdict**: `vllm_mtp_unsupported` — vLLM 0.19.1 + Qwen3.5-4B + native MTP segfaults during V1 engine init on A6000. Cannot establish an external α upper bound at this vLLM version.
**Branch**: `phase7-h15c-vllm-alpha-microbench`
**Predecessor**: PR #529 (H15b stratified C29 v2 reject-row probe → `kiln_native_ceiling`)
**Pod**: RunPod A6000 `wl4oh30qxnzyqz`, on-demand $0.49/hr, terminated after run

## Goal

PR #529's stratified H15b probe established that kiln's MTP head logits sit at
the BF16-noise cosine ceiling (`reject_cos_sim_median = 0.999978`) on the C29 v2
reject sub-population. Verifier numerical drift on reject rows is RULED OUT as
the α-gap source. The remaining open question:

> Does any external serving system hit a **higher α** on this same Qwen3.5-4B
> checkpoint at A6000 bs=1?

H15c attempted to answer this by running vLLM 0.19.1 with the native
`qwen3_5_mtp` head on the *identical* prompt / seed / prefill / decode workload
PR #529 used, and comparing α to kiln's α derived from the same-run
`c1_attr_seed{0,1,2}.csv`.

## Outcome

**vLLM 0.19.1 segfaults during V1 engine init on this checkpoint, on every MTP
method alias.** The crash fires after the drafter MTP weights successfully
load and after `eagle.py:1377/1433` confirms tied embed/lm_head sharing — i.e.
vLLM has the right model dispatch, the right MTP arch (`Qwen3_5MTP` per
`vllm/model_executor/models/registry.py`), and the right speculative method
plumbing (`qwen3_5_mtp` → `mtp` rewrite in `vllm/config/speculative.py:368`).
The crash is in a *post-load native code path* (autograd / CUDA extension
boundary, no Python file symbols in the backtrace).

The pre-registered decision rule maps this to the `vllm_mtp_unsupported`
branch — ship this PR as a doc-only redirect documenting the vLLM-side gap and
queue the next H from PR #527 §"Queued next action".

| metric | value |
| --- | --- |
| `kiln_median_alpha` (re-derived from PR #529 c1_attr CSVs) | **0.3636** |
| `vllm_median_alpha` | **UNAVAILABLE** (segfault) |
| `delta` | **UNAVAILABLE** |

Per-seed kiln α (from `docs/phase-c29-v2/artifacts/c1_attr_seed{0,1,2}.csv`,
all 11/12/11 rows per seed, not only the H15b-probed subset):

| seed | accept / steps | α |
| --- | --- | --- |
| 0 | 4 / 11 | 0.3636 |
| 1 | 4 / 12 | 0.3333 |
| 2 | 5 / 11 | 0.4545 |
| **median** | — | **0.3636** |

The 7-accepts / 22-rows hint in the task brief (≈ 0.318) refers to the union
of accept-labeled + reject-labeled splice-dump rows in PR #529's H15b probe
(POSITIONS=0..3 × MAX_STEPS=2 = 22 rows total, of which 7 were accepted). The
H15c kiln α uses *every* `speculative_mtp_decode_step` call, not only the
splice-dumped ones — that's the strict kiln α at this workload.

## Decision rule (pre-registered, set BEFORE the run)

| Δ = vllm_median_α − kiln_median_α | verdict | next action |
| --- | --- | --- |
| ≥ +0.05 | `external_ceiling_exists` | queue serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring) |
| in [−0.02, +0.05) | `mtp_head_quality_ceiling` | accept α as upper-bounded by checkpoint quality; deprioritize MTP-side α work |
| < −0.02 | `kiln_above_vllm` (unexpected) | sanity-check vLLM args; if confirmed, document and queue next H |
| `vllm.mtp_supported == false` | **`vllm_mtp_unsupported`** ← **THIS RUN** | doc-only redirect PR; queue next H from PR #527 |

The rule is implemented in `scripts/h15c_compare.py` and emitted
`docs/phase-c29-v3-vllm/verdict.json`.

## Crash signature

All three method aliases (`qwen3_5_mtp`, `mtp`, `qwen3_next_mtp` — the latter
two get rewritten to internal `qwen3_5_mtp` dispatch) reach the same point:

1. Tokenizer + multimodal config loaded successfully (`Resolved architecture: Qwen3_5ForConditionalGeneration`)
2. `attention block size to 528 tokens` calibrated against mamba page size — vLLM understands the hybrid GDN + GQA layout
3. `running in text-only mode` confirms `limit_mm_per_prompt={image:0,video:0}` took effect
4. EngineCore worker boots fine, target model loads (~8 GiB, ~3 s)
5. Drafter MTP weights load (`Detected MTP model. Sharing target model embedding/lm_head weights with the draft model.`)
6. `gpu_model_runner.py:4820 Model loading took 8.21 GiB memory and 3.5 seconds`
7. **`!!!!!!! Segfault encountered !!!!!!!`** with a 70-frame native backtrace through `_PyEval_EvalFrameDefault` / `_PyFunction_Vectorcall` / `PyObject_Call` / `THPFunction_apply` (compiled-extension crash, no Python file symbols).

`RuntimeError: Engine core initialization failed. See root cause above.
Failed core proc(s): {}` is the wrapper exception the parent process raises
after the EngineCore worker dies.

Workarounds that all *did* take effect but did NOT prevent the crash:

- `enforce_eager=True` — disables `torch.compile` and CUDA graph capture (vLLM logs `Cudagraph is disabled under eager mode`)
- `limit_mm_per_prompt={"image": 0, "video": 0}` + `skip_mm_profiling=True` — skips the multimodal dummy profile_run (vLLM logs `running in text-only mode`)
- `max_model_len=1024` + `gpu_memory_utilization=0.85` — no OOM signal in the log

The crash is therefore NOT in CUDA graph capture, NOT in MM dummy profiling,
and NOT a clean OOM. It is a native-code crash inside the vLLM V1 engine
post-load setup that's specific to the (Qwen3_5 + MTP) combination at vLLM
0.19.1 + torch 2.10.0 + CUDA 12.8 on sm_86. See
`docs/phase-c29-v3-vllm/artifacts/vllm_segfault_evidence.log` for the trimmed
log capture (3 attempts × ~13 lines each, EngineCore PIDs 4722 / 5122 / 5502).

## Workload (matched to PR #529)

| field | value | source |
| --- | --- | --- |
| model | Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`) | `/workspace/qwen3.5-4b/config.json` |
| prompts | `PROMPT_POOL[seed % 30]` for seeds {0,1,2} → GSM8K prose 0/1/2 | `crates/kiln-server/src/bench.rs:376` |
| prompt-token target | 512 (sentence-repetition expander) | `bench.rs::build_prompt` |
| actual prompt tokens | 494 / 508 / 501 | matches PR #529 c1_attr CSV `base_pos` start values |
| max output tokens | 16 | PR #529 `scripts/c29_kiln_logits_dump.sh:DECODE_TOKENS` |
| chat template | OFF (raw prose) | PR #529 driver |
| sampler | greedy: `temperature=0`, `top_p=1`, `top_k=0` (kiln) / `top_k=-1` (vLLM) | both drivers |
| spec method | k=1 native MTP (`KILN_SPEC_METHOD=mtp` / `qwen3_5_mtp`) | both drivers |
| GPU | NVIDIA RTX A6000 sm_86, single-GPU bs=1 | RunPod |
| kiln quant | Marlin W4A16 (`KILN_W4A16=1`) | PR #529 driver |
| vLLM quant | BF16 (vLLM does not implement Marlin W4A16 for `Qwen3_5MTP`) | acknowledged confounder |

The h15c driver verified that the prompt-builder reproduces PR #529's encoded
prompts byte-for-byte: prompt token counts 494 / 508 / 501 exactly match the
`base_pos` start values in `c1_attr_seed{0,1,2}.csv` (494, 508, 501). This is
how we know the workload is identical at the prefill boundary — only the
serving-side decode path would have differed had vLLM completed the run.

## Reproduction commands

```bash
# Pod side (RunPod A6000 with kiln-runpod image, vLLM 0.19.1 installed)
cd /workspace/kiln

# 1. vLLM α (3 seeds × 16-token greedy decode × MTP k=1) — segfaults at vLLM 0.19.1
python3 scripts/h15c_vllm_alpha_dump.py \
    --model-path /workspace/qwen3.5-4b \
    --prompt-tokens 512 --max-tokens 16 \
    --seeds 0 1 2 --num-spec-tokens 1 \
    --out docs/phase-c29-v3-vllm/vllm_alpha_per_seed.json
# Outcome: writes mtp_supported=false JSON; exit code 3.

# 2. kiln α (re-derived from PR #529's c1_attr CSVs — no GPU needed, runs anywhere)
python3 scripts/h15c_kiln_alpha_from_csv.py \
    --csv-dir docs/phase-c29-v2/artifacts \
    --out docs/phase-c29-v3-vllm/kiln_alpha_per_seed.json
# Outcome: median_alpha = 0.3636.

# 3. Apply decision rule
python3 scripts/h15c_compare.py
# Outcome: verdict=vllm_mtp_unsupported.
```

## Anti-duplication evidence

```
$ gh pr list -R ericflo/kiln --state all --search "vllm alpha microbench"
(empty)
$ gh pr list -R ericflo/kiln --state all --search "h15c vllm"
(empty)
$ gh pr list -R ericflo/kiln --state all --search "H15c"
(empty)
$ gh pr list -R ericflo/kiln --state all --search "qwen3_next_mtp vllm"
527  phase7: MTP acceptance-rate state-of-play audit (doc-only)         MERGED
245  mtp: native MTP speculative decode step + h_prev wiring (WIP)      MERGED
```

PR #527 explicitly queued this H15c microbench in its "Recommended next H"
section, and PR #529 re-confirmed it as the next action under its
`kiln_native_ceiling` verdict. PR #525 (vLLM Triton GDN audit, doc-only) is
scope-distinct (kernel audit, not α microbench).

## What this rules out

- **vLLM 0.19.1 cannot serve Qwen3.5-4B + native MTP on A6000 today.** Any
  Phase 7 plan that assumes "we can compare against vLLM α as the upper bound"
  needs to either (a) wait for a vLLM patch, (b) try a different vLLM version
  / nightly, (c) try a different external serving system (SGLang has no MTP
  for Qwen3.5 either as of PR #526), or (d) accept that the upper bound has to
  come from a hand-rolled HF transformers reference implementation.
- **The vLLM-side method dispatch is correct.** vLLM 0.19.1 *does* recognize
  `Qwen3_5MTP` in its registry, *does* rewrite `qwen3_5_mtp` → `mtp` per
  `vllm/config/speculative.py:323-330`, and *does* attach the drafter weights
  with tied embed/lm_head. The bug is downstream of dispatch.

## What this does NOT rule out

- **Whether an external α ceiling exists.** That was the question H15c was
  supposed to answer. With vLLM blocked, the question stays open until either
  vLLM is unblocked or another reference is built.
- **Workload sensitivity** (per `mtp-bench-workload-sensitivity` — chat α ≈
  0.588 vs prose α ≈ 0.175 on PR #364). H15c chose the prose / 16-decode
  workload to match PR #529 exactly; chat-template prompts may behave
  differently.
- **Quantization path differences.** kiln runs Marlin W4A16; the eventual
  reference will likely run BF16. That confounder still has to be handled.

## Bench envelope + cost

- Pod: RunPod A6000 (`wl4oh30qxnzyqz`), $0.49/hr on-demand, no spot.
- Pool acquire: HTTP 503 `capacity_supply_exhausted` on hibernated pods →
  fell back to direct `runpod_api.py launch` (per the documented fallback path
  for "pool at cap or unavailable").
- vLLM 0.19.1 install: ~1.5 min (background, `wait-file` supervised, single
  PyTorch downgrade 2.4.1 → 2.10.0 + cu128 stack).
- Bench attempts: 3 × ~30 s each (load + segfault), all in one process run.
- All bench supervision used `runpod_api.py bg` + `wait-file --timeout 1500`,
  no SSH polling loops (per `kiln-ssh-polling-deadlock` note).
- Wall-clock budget: 90 min / $40 hard cap (per project §"Wall-clock budget").
  Estimated total spend: ~$0.40 — well under cap.

## Queued next action

Per the pre-registered decision rule for `vllm_mtp_unsupported`:

> Ship this as a doc-only redirect PR documenting the gap, queue the next H
> from PR #527 §"Queued next action".

Concrete options the next planning cycle should pick from (PR #527 enumerated
several non-α optimization paths; this verdict refocuses Phase 7 onto those):

1. **Direct decode-path optimization** — already in flight: PR #521
   (prefix-cache + CUDA graphs wins) and PR #526 (SGLang RadixAttention
   audit). H15c does not block continued investment here.
2. **HF transformers reference α** — build a hand-rolled HF reference with
   Qwen3.5's pretrained MTP head and run the same 3-seed prose workload.
   Higher engineering cost than reusing vLLM, but would close the
   external-upper-bound question without depending on a vLLM patch.
3. **Retry vLLM at a different version / nightly** — when a vLLM patch
   exposes a working Qwen3.5-MTP path, re-run this PR's `scripts/h15c_*`
   exactly as-is. The driver and compare are version-agnostic; only the
   `vllm_alpha_per_seed.json` and `compare.json/md` artifacts re-generate.
4. **Switch to SGLang or another OSS server** — SGLang doesn't yet support
   Qwen3.5 MTP either (per PR #526), so this option waits on the SGLang
   side too.

## Files

| path | purpose |
| --- | --- |
| `docs/phase7-h15c-vllm-alpha-microbench.md` | this audit doc |
| `docs/phase-c29-v3-vllm/verdict.json` | machine-readable verdict |
| `docs/phase-c29-v3-vllm/compare.json` | full kiln + vllm side-by-side |
| `docs/phase-c29-v3-vllm/compare.md` | per-seed table + decision rule |
| `docs/phase-c29-v3-vllm/kiln_alpha_per_seed.json` | re-derived kiln α |
| `docs/phase-c29-v3-vllm/vllm_alpha_per_seed.json` | trimmed vLLM failure record |
| `docs/phase-c29-v3-vllm/artifacts/vllm_segfault_evidence.log` | trimmed pod log of all 3 segfaults |
| `scripts/h15c_kiln_alpha_from_csv.py` | derives kiln α from PR #529 CSVs |
| `scripts/h15c_vllm_alpha_dump.py` | vLLM driver (re-runnable when vLLM is fixed) |
| `scripts/h15c_compare.py` | applies decision rule, emits verdict.json |
