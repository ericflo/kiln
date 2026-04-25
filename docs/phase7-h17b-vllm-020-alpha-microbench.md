# H17b — vLLM v0.20.0 α microbench retest (free pre-step)

**Verdict: `vllm_020_mtp_unsupported_dense_4b`.**

vLLM v0.20.0 (the "free pre-step" branch from PR #531's pre-registered
decision rule, fired by PR #532's `sglang_mtp_unsupported_dense_4b`)
also cannot serve Qwen3.5-4B dense BF16 + native MTP on A6000 sm_86 —
this run, however, fails for a distinct **runtime-stack** reason rather
than another segfault inside dispatched MTP code. The vLLM 0.20.0
prerelease wheels (`vllm-0.20.0+cu129`, the only x86_64 wheel published
as of 2026-04-25 for the v0.20.0 release tag of 2026-04-23 21:02 UTC)
pull `torch 2.11.0+cu130` as a transitive dependency, which requires
NVIDIA driver ≥575 to load CUDA 12.9 runtime libraries and ≥580 for
CUDA 13.0. RunPod's A6000 stock image ships driver 550.127.08
(CUDA 12.4 max), so `torch.cuda.is_available()` returns `False` with
warning *"NVIDIA driver on your system is too old (found version
12040)"*, and vLLM's V1 EngineCore subprocess crashes at first GPU
init with `RuntimeError: Cannot re-initialize CUDA in forked
subprocess`.

This was reproduced for all three method aliases (`qwen3_5_mtp` /
`mtp` / `qwen3_next_mtp`); each takes ~1-3 s wall-clock to crash and
each terminates inside the same CUDA-init code path before any
MTP-specific dispatch can run.

Kiln median α (unchanged from PR #530 / PR #532, re-derived from
PR #529 c1_attr CSVs): **0.3636**. vLLM 0.20.0 median α: **UNAVAILABLE**.
Δ: UNAVAILABLE.

Per the pre-registered decision rule (set BEFORE the run, see
`scripts/h17b_compare.py`), `mtp_supported == false` maps to the
`vllm_020_mtp_unsupported_dense_4b` branch — ship as a doc-only
redirect PR, queue **hand-rolled HF transformers H18 reference** (PR
#531 candidate 8) as the next external-α reference path. The
`kiln_native_ceiling` verdict from H15b stands until H18 lands.

## Pre-registered decision rule

| Δ = vllm_020_α − kiln_α | verdict | next action |
| --- | --- | --- |
| ≥ +0.05 | `external_ceiling_exists` | queue serving-difference bisect |
| in [−0.02, +0.05) | `mtp_head_quality_ceiling` | accept α as upper-bounded by checkpoint quality; deprioritize MTP-side α work |
| < −0.02 | `kiln_above_vllm` (unexpected) | sanity-check vLLM args |
| `vllm_020.mtp_supported == false` | **`vllm_020_mtp_unsupported_dense_4b`** ← THIS RUN | doc-only redirect; queue H18 hand-rolled HF transformers reference |

This is the same decision rule as PR #530 (vLLM 0.19.1) and PR #531/#532
(SGLang main HEAD), with the load-failure branch renamed to a
v0.20-specific verdict label so the verdict.json is unambiguous.

## Workload (matched to PR #530 / PR #532 / PR #529 byte-for-byte)

- Model: Qwen3.5-4B (`Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`),
  pulled fresh from `Qwen/Qwen3.5-4B` on HF (BF16, 9.32 GB, 2 safetensors shards)
- Prompts: `PROMPT_POOL[seed % 30]` for seeds {1,2,3} → GSM8K prose 1/2/0
  (rotated +1 vs PR #530's seeds {0,1,2}, per H17b task spec, to avoid trivial
  cache-hit confounders if the same vLLM build runs against the same pod state)
- Prefill 512 tokens, decode 16 tokens, greedy (T=0, top_p=1, top_k=-1)
- Spec: k=1 native MTP, no chat template
- GPU: NVIDIA RTX A6000 sm_86, single-GPU bs=1, driver 550.127.08, CUDA 12.4
- vLLM: **0.20.0+cu129** (from `https://wheels.vllm.ai/0.20.0/cu129/`,
  the prerelease index for tag commit `101584af0a8ddf67165fc89cae77ed560fb0096b`).
  PyPI does not yet publish 0.20.0 (latest there is 0.19.1 — same as PR #530).
- vLLM speculative config kwargs: `enforce_eager=True`,
  `limit_mm_per_prompt={image:0,video:0}`, `skip_mm_profiling=True` —
  identical to PR #530's matched workload, all confirmed in the engine
  config dump (`running in text-only mode`).
- kiln quant: Marlin W4A16 (PR #529 c1_attr CSVs); vLLM quant: BF16 (`quantization=None`).
  Same confounder as PR #530 — vLLM has no Marlin path for `Qwen3_5MTP`.
  H17b is establishing an *upper bound*, so BF16 is at worst pessimistic for the
  upper-bound question.

## Crash signature (3 attempts, identical terminal frame)

All three method aliases reach the same point and die identically:

1. Tokenizer + multimodal config loaded (`Resolved architecture:
   Qwen3_5ForConditionalGeneration`)
2. `attention block size to 528 tokens` calibrated against mamba page size
3. `running in text-only mode` confirms MM workaround took effect
4. EngineCore worker boots
5. **Crash at `torch.accelerator.set_device_index(self.device)`** →
   `torch._C._accelerator_setDeviceIndex(device_index)` →
   `torch/cuda/__init__.py:466 _lazy_init` →
   `RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use
   CUDA with multiprocessing, you must use the 'spawn' start method`
6. EngineCore process `_bootstrap` re-raises the RuntimeError →
   parent vLLM raises `RuntimeError: Engine core initialization failed.
   See root cause above. Failed core proc(s): {'EngineCore': 1}`

Three attempts (`qwen3_5_mtp` → `mtp` → `qwen3_next_mtp`) in one Python
driver process. Each crash is ~1.5-3s wall-clock from
`speculative_config={...}` kwargs supplied to `LLM(...)` until terminal
exception. Wall-clock for the entire 3-attempt loop: ~33 s (including
tokenizer load + 3 × engine init).

Trimmed evidence: `docs/phase-c29-v3-vllm-020/artifacts/vllm_020_failure_evidence.log`
(the full driver log retained at `artifacts/driver_full.log`, 276 lines,
89 KB; the trimmed evidence is 105 lines, 65 KB — covers header
context + first attempt's full traceback + the multi-attempt summary
tail. The remaining two attempts are byte-identical except for
EngineCore PID, so they're elided rather than triplicated.)

## Why the failure is different from PR #530's vLLM 0.19.1 segfault

PR #530's verdict (`vllm_mtp_unsupported`) was for vLLM 0.19.1 which
**loaded the model and drafter weights successfully** (~8.21 GiB load,
3.5 s) and segfaulted in the autograd / spec-decode code path during
runtime profiling — a downstream-of-dispatch crash with a 70-frame
native backtrace (`THPFunction_apply` / `_PyEval_EvalFrameDefault`,
no Python file symbols).

H17b's vLLM 0.20.0 fails **upstream of model load** — at the very
first CUDA device-set call, before any MTP-specific code runs. The
underlying root cause is **the version-stack delta itself**: vLLM
0.20.0 ships with PyTorch 2.11 + CUDA 13.0 default (as documented in
the v0.20.0 release notes' "Highlights" section), and RunPod's stock
A6000 image driver 550.x doesn't support the CUDA 12.9 runtime that
the cu129 wheel needs.

This is qualitatively different from "vLLM 0.20.0 has a bug in its
MTP code that segfaults" — the H17b run cannot make any statement
about whether 0.20.0's MTP path actually works correctly, because it
never reaches MTP code at all. What this run **does** prove:

- vLLM 0.20.0 prerelease wheels are not yet usable on RunPod's
  A6000 image without a driver upgrade
- A driver upgrade itself is out of scope for this task budget
  ($0.40 / 45 min)
- The natural follow-up is **H18 hand-rolled HF transformers
  reference** (PR #531 candidate 8), which is `feasible` and runs on
  any driver/CUDA combo since it doesn't depend on vLLM's runtime
  stack

## Newly documented vLLM 0.20.0 runtime stack delta

Beyond PR #531's static audit (which only inspected the v0.20.0 source
tree and noted "PyTorch 2.11 + CUDA 13.0 default change vs PR #530's
torch 2.10 + cu128"), this run documents the concrete RunPod-side
implications:

| stack layer | PR #530 (vLLM 0.19.1) | H17b (vLLM 0.20.0+cu129) |
| --- | --- | --- |
| torch wheel | 2.10.0+cu128 | **2.11.0+cu130** (transitive dep of vllm 0.20.0+cu129) |
| nvidia-cuda-runtime | 12.8.x | **13.0.96** |
| nvidia-cublas | 12.x | **13.1.0.3** |
| nvidia-cudnn | 9.x cu12 | **9.19.0.56 cu13** |
| Driver requirement | ≥555 (CUDA 12.5) | **≥580 (CUDA 13.0)** |
| Stock RunPod kiln-runpod-image driver | 550.127.08 (CUDA 12.4) | 550.127.08 (CUDA 12.4) — unchanged |
| Net | torch.cuda.is_available() = True | **torch.cuda.is_available() = False** |

The cu129 wheel is pulled because that's the closest CUDA-version-tagged
vLLM 0.20.0 wheel published. There is also a `cu130` directory but no
wheel listed there as of 2026-04-25 02:28 UTC. PyPI does not publish
vLLM 0.20.0 at all yet (still 0.19.1 latest).

If RunPod's kiln-runpod base image is later updated to driver 580+
(CUDA 13.0+) AND vLLM publishes a stable v0.20.x release, this exact
H17b script becomes a one-command rerun.

## Anti-duplication evidence

```
$ gh pr list -R ericflo/kiln --state all --search "h17b"
532  phase 7: H17 SGLang α microbench against kiln on Qwen3.5-4B + native MTP   MERGED  2026-04-25
   (no H17b match)

$ gh pr list -R ericflo/kiln --state all --search "vllm 0.20"
   (only PR #532 by fuzzy match — no H17b/v0.20-specific PR exists)

$ gh pr list -R ericflo/kiln --state all --search "vllm v0.20"
   (empty)

$ gh pr list -R ericflo/kiln --state all --search "phase7 h17b"
   (empty)

$ ce tasks-list --project-id faf9b108c04f7f73a9a39fca --status running
   (this task + the project planning task — no concurrent H17b/H18 work)
```

PR #532 explicitly queued this H17b as the canonical free pre-step
("vLLM v0.20.0 retest — est. 30 min / $0.25 A6000 on-demand"). PR #531
classified candidate 3 (vLLM v0.20.0 release) as `unknown` and predicted
exactly this kind of stack-delta uncertainty.

## What this rules out / does not rule out

**Rules out:**

- vLLM 0.20.0 default wheels (`+cu129`) are usable on RunPod's stock
  A6000 image as of 2026-04-25.
- A free pre-step "the v0.20 stack sidesteps the v0.19.1 MTP segfault"
  outcome — we cannot verify, because v0.20.0 doesn't reach MTP code.

**Does NOT rule out:**

- Whether vLLM 0.20.0's MTP code path actually works. To prove that
  either way, we'd need a pod with driver ≥580 (CUDA 13.0+) and a
  rerun of `scripts/h17b_vllm_020_alpha_dump.py` exactly as-is.
- Whether a future stable v0.20.x release (with `+cu124` or `+cu128`
  wheels) would behave differently. The current prerelease is
  cu129/cu130-only.
- Whether a CPU-only fallback in vLLM 0.20 would let us measure α
  without the GPU init crash. (This isn't worth pursuing — α
  measurements on CPU would not be representative of A6000 behavior
  even if we could measure them.)

## Bench envelope + cost

- Pod: RunPod A6000 `mfk88l8i8tab02`, $0.49/hr on-demand, leased from
  pool (`ce kiln-pod-acquire`). No fallback to direct
  `runpod_api.py launch` needed — the pool path succeeded on first try
  this time (PR #530 + PR #532 both fell back to direct launch after
  HTTP 503 `capacity_supply_exhausted`).
- Bench supervision: `runpod_api.py bg` + `wait-file --timeout`
  exclusively — NO `until ssh` / `while ssh ... sleep` polling loops
  (per `kiln-ssh-polling-deadlock` note, $99.76 incident 2026-04-20).
- Pool lease released at idle_warm (success path) so future tasks can
  reuse it without paying re-warm cost.
- **Actual: ~13 min pod time / ~$0.10** — well under $0.40 H17b cap.
  Free pre-step assumption (~30 min / $0.25) was conservative; actual
  spend is ~40% of that.

## Reproduction

```bash
cd /workspace/kiln  # on a RunPod A6000 with kiln-runpod-image baked

# 1. Install vLLM 0.20.0 prerelease wheel (cu129 variant)
pip install --extra-index-url https://wheels.vllm.ai/0.20.0/cu129/ vllm==0.20.0

# 2. Sanity-check torch/CUDA compatibility against pod driver
python3 -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
# Expected on driver <575: torch 2.11.0+cu130  False
# Expected on driver ≥580: torch 2.11.0+cu130  True

# 3. vLLM α microbench (will fail on driver <575 with the documented signature)
python3 scripts/h17b_vllm_020_alpha_dump.py \
    --model-path /workspace/qwen3.5-4b \
    --prompt-tokens 512 --max-tokens 16 \
    --seeds 1 2 3 --num-spec-tokens 1

# 4. Re-derive kiln α (no GPU needed; reuses PR #529 CSVs)
python3 scripts/h15c_kiln_alpha_from_csv.py \
    --out docs/phase-c29-v3-vllm-020/kiln_alpha_per_seed.json

# 5. Apply decision rule
python3 scripts/h17b_compare.py
```

## Files

| path | purpose |
| --- | --- |
| `docs/phase7-h17b-vllm-020-alpha-microbench.md` | this audit doc |
| `docs/phase-c29-v3-vllm-020/verdict.json` | machine-readable verdict |
| `docs/phase-c29-v3-vllm-020/compare.json` | full kiln + vLLM 0.20.0 side-by-side |
| `docs/phase-c29-v3-vllm-020/compare.md` | per-seed table + decision rule |
| `docs/phase-c29-v3-vllm-020/kiln_alpha_per_seed.json` | re-derived kiln α (0.3636) |
| `docs/phase-c29-v3-vllm-020/vllm_020_alpha_per_seed.json` | full vLLM 0.20.0 failure record |
| `docs/phase-c29-v3-vllm-020/artifacts/vllm_020_failure_evidence.log` | trimmed driver log (105 lines) |
| `docs/phase-c29-v3-vllm-020/artifacts/driver_full.log` | full driver log (276 lines, retained for completeness) |
| `scripts/h17b_vllm_020_alpha_dump.py` | re-runnable v0.20-specific driver |
| `scripts/h17b_compare.py` | applies decision rule, emits verdict.json |
| `PROFILING.md` | top-of-file pointer entry mirroring PR #525-#532 |

## Reopen preconditions

H17b verdict shifts from `vllm_020_mtp_unsupported_dense_4b` to a different
branch under any of:

- vLLM publishing a `+cu124` or `+cu128` 0.20.x stable wheel that the
  current RunPod stock driver (550.x) can load — makes the H17b retest
  cheap (~10 min / $0.10; driver script is unchanged).
- RunPod's `kiln-runpod` base image being upgraded to driver ≥580 →
  CUDA 13.0 capable → cu129/cu130 wheels load → rerun
  `scripts/h17b_vllm_020_alpha_dump.py` as-is.
- Anyone successfully reproducing PR #530's 0.19.1 segfault on vLLM
  0.20.x in any other GPU/driver combo, definitively settling whether
  the segfault is fixed in v0.20 — the H17b decision tree currently
  carries this as an open question.
- Hand-rolled HF transformers H18 reference completing — closes the
  external-α question independently and renders both
  `vllm_020_mtp_unsupported_dense_4b` and SGLang's
  `sglang_mtp_unsupported_dense_4b` moot for upper-bound purposes.
- Kiln-side change invalidating PR #529's c1_attr CSV data — would
  require re-running kiln capture before re-running this compare.

## Next action (queued, not included in this PR)

Per the pre-registered decision rule, the next planning cycle should
queue **exactly one** of:

1. **Hand-rolled HF transformers H18 reference** (PR #531 candidate 8) —
   est. 2-4 hrs engineering + ~30 min CPU/GPU run = **$0 GPU spend**,
   high human-time cost. **This is the canonical next step** — both
   PR #530 (vLLM 0.19.1 unsupported), PR #532 (SGLang unsupported),
   and now H17b (vLLM v0.20.0 unsupported on current driver) point to
   it as the only remaining viable external-α reference.
2. **Accept kiln-native ceiling as checkpoint-quality ceiling** — with
   all three external-OSS-server references now blocked, the H15b
   `kiln_native_ceiling` verdict stands; Phase 7 can deprioritize
   external-α work and refocus on non-α decode-path wins (PR
   #517/#520/#521 prefix-cache + CUDA graphs already in flight).
3. **Wait for RunPod driver upgrade or vLLM stable +cu124/+cu128** —
   both eventually likely, neither has an ETA. Not a recommended
   primary path; treat as opportunistic if either lands.

The H17b task spec and the PR #532 PR body are both consistent: option
(1) is the canonical next step.
