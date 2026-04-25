# Phase 7 — H16: External-α reference options audit (doc-only)

**Verdict**: `external_reference_exists` — SGLang main supports `Qwen3_5ForCausalLMMTP` for dense Qwen3.5-4B with spec_v2 enabled. Queue exactly ONE H17 SGLang α microbench task.
**Branch**: `phase7-h16-external-alpha-options`
**Predecessor**: PR #530 (H15c vLLM α microbench → `vllm_mtp_unsupported`)
**Pod**: none — doc-only, $0 GPU spend. Auditor: Cloud Eric task `9305f01ec18b1bf9b85ed0b7`.

## Inspection commits / versions

| repo | ref | SHA / tag | inspection date |
| --- | --- | --- | --- |
| ericflo/kiln | main | `045bc9c` | 2026-04-25 |
| vllm-project/vllm | main | `bc2ae5a` | 2026-04-25 |
| vllm-project/vllm | release | `v0.20.0` (2026-04-23) | 2026-04-25 |
| vllm-project/vllm | release | `v0.19.1` (2026-04-18) | reference (PR #530) |
| sgl-project/sglang | main | `a4facdf` | 2026-04-25 |
| sgl-project/sglang | (PR #526 audit ref) | `76da28f6` | reference (PR #526) |
| huggingface/transformers | main | `c472755` | 2026-04-25 |
| SafeAILab/EAGLE | main | `cb7e0841` (2026-02-20) | 2026-04-25 |
| FasterDecoding/Medusa | main | `e2a5d20` (2024-04-18) | 2026-04-25 |
| NVIDIA/TensorRT-LLM | main | `1989520` (2026-04-24) | 2026-04-25 |

## Decision rule (pre-registered, set BEFORE inspection)

| Number of mandatory candidates with `supported` verdict | Action |
| --- | --- |
| ≥ 1 | Queue exactly ONE H17 microbench task naming the chosen winner (cheapest setup time, highest fidelity to vLLM-segfault-replacement role). |
| 0, but ≥ 1 candidate is `unknown` and the unknown resolves with ≤30 min A6000 / ≤$0.25 | Queue ONE bounded "verify candidate X actually works on this checkpoint" task with a $0.25 hard budget cap. |
| 0 supported AND no resolvable unknowns | Verdict `no_external_alpha_reference_available_2026_04`. Refocus Phase 7 to non-α decode-path wins. Reopen precondition: any of (vLLM main commit referencing `qwen3.*MTP\|mtp.*qwen3` after this PR's merge date, SGLang adding Qwen3.5 spec-decode, an Eagle/Medusa qwen3 config landing). |

**This run hit branch 1** (1 supported candidate: SGLang main).

## Mandatory candidates table

| # | name | repo + commit | verdict | evidence | setup cost | confounders |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | HF transformers Qwen3.5 MTP path | `huggingface/transformers@c472755` | `unsupported` | `src/transformers/models/qwen3_5/modeling_qwen3_5.py` lines 69-1803 declares 23 classes (`Qwen3_5VisionRotaryEmbedding`…`Qwen3_5ForConditionalGeneration`); none is an MTP / NextN class. `src/transformers/generation/candidate_generator.py:78,341,898,1017,1172` declares 5 candidate-generator types (`AssistedCandidateGenerator`, `AssistedCandidateGeneratorDifferentTokenizers`, `UniversalSpeculativeDecodingGenerator`, `PromptLookupCandidateGenerator`, `EarlyExitCandidateGenerator`); none can drive Qwen3.5's pretrained `mtp.*` head as the drafter — they all use a separate draft model with full forward pass. | 0 (static read) | n/a — feature absent |
| 2 | vLLM main HEAD (post-0.19.1) | `vllm-project/vllm@bc2ae5a` | `unknown` | `vllm/model_executor/models/qwen3_5_mtp.py` (size 17,099) declares `Qwen3_5MultiTokenPredictor` and is reachable via the registry that PR #530 confirmed routes correctly. PR #530's segfault was in a *downstream native code path* with no Python file symbols (autograd/CUDA extension). Between v0.19.1 (2026-04-18) and main, vLLM has merged: Eagle prefill full-CUDA-graph (#37588), fused probabilistic rejection-sample kernels (#38496), `[Spec Decode] Move SpecDecodeBaseProposer out of eagle.py` (#40732, 2026-04-23), GDN kernel fusion (#37813), GPU↔CPU syncs eliminated in prefill+spec-decode (#38047, #38361), TMA aligned with upstream FLA (#38981). The `qwen3_5_mtp.py` *file itself* changed by only +9/0 lines (NVFP4 fc workaround, PR #38650 — does not apply to BF16 checkpoint). PyTorch 2.11 + CUDA 13.0 are the default in v0.20.0+, vs torch 2.10 + CUDA 12.8 in PR #530's run. | ~30 min A6000 install + ~10 min retest = ~$0.25 (within bounded-verify budget) | None of the merged spec-decode / GDN / autograd commits are documented as a Qwen3.5-MTP segfault fix; this is "the surface where the bug lived has changed substantially" rather than "PR XX explicitly fixes the crash." |
| 3 | vLLM v0.20.0 prerelease/release | `vllm-project/vllm@v0.20.0` (2026-04-23) | `unknown` | Same as (2). v0.20.0 is the first tagged release containing all the kernel + autograd-path changes since v0.19.1. Release notes do not specifically call out a Qwen3.5-MTP segfault fix, but enumerate 576 commits / 300+ files between v0.19.1 and v0.20.0 with notable spec_decode + GDN + mamba updates. | identical install cost to (2) | identical to (2) |
| 4 | SGLang main (post-PR #526 audit) | `sgl-project/sglang@a4facdf` | **`supported`** (with caveat) | `python/sglang/srt/models/qwen3_5_mtp.py` (size 13,573) declares `class Qwen3_5ForCausalLMMTP(nn.Module)` (line 39, exported as `EntryClass = [Qwen3_5ForCausalLMMTP]` line 360). File added 2026-02-09 (`model: support Qwen3.5 (#18489)`) and refactored 2026-02-12 (`[Qwen3_5] Refactor Qwen3_5ForCausalLMMTP class implementation (#18538)`). Spec_v2 enabled 2026-03-04 (`[Qwen3.5] Enable MTP spec_v2 and add test for nvidia/Qwen3.5-397B-A17B-NVFP4 (#19391)`). Dense 4B path landed 2026-03-24 (`[Bugfix] Fix PP tied embeddings weight loading for qwen3.5 4B dense model (#21347)`). The same file *was* present at PR #526's audit ref `76da28f6` (size 13,573, blob `3fa89fcd`) — PR #526's "no Qwen3.5 MTP" claim was about the cross-ref question of whether *RadixAttention* covered MTP, not whether SGLang has the model class. **The class supports both MoE (NVFP4 397B) and dense (4B) checkpoints**: load_weights() at line 170 branches on `getattr(self.config, "num_experts", None)` and skips MoE expert mapping when None. | ~20 min A6000 install (`pip install sglang[srt]` against torch 2.x) + ~10 min retest = ~$0.20 | Caveat 1: open PR `#23331` (2026-04-21, OPEN) titled "[BugFix] Resolve adaptive speculative decoding conflicts for Qwen3.5 (hybrid GDN)" — the **adaptive** DSD path has a known bug. kiln's H17 workload is greedy + fixed k=1, not adaptive; the non-DSD path should be unaffected. Caveat 2: SGLang's MTP testing has been against the MoE NVFP4 checkpoint, not the dense 4B BF16 checkpoint we want; dense 4B + MTP is structurally compatible per #21347 but unverified end-to-end. |
| 5 | Eagle (SafeAILab/EAGLE) native Qwen3.5-MTP support | `SafeAILab/EAGLE@cb7e0841` (2026-02-20) | `unsupported` | `eagle/model/` directory at HEAD has `modeling_llama_kv.py`, `modeling_mixtral_kv.py`, `modeling_qwen2_kv.py`, `modeling_qwen3_kv.py` — no `modeling_qwen3_5_kv.py` and no MTP-head loader. Eagle uses *its own draft-head architecture* trained per base model (per README "EAGLE … extrapolating the second-top-layer contextual feature vectors") — even if a Qwen3.5 modeling file landed, it would not load Qwen3.5's pretrained `mtp.*` tensors. Eagle's published Qwen3.5 support timeline shows latest README support entry "2024.8.8: We now support Qwen-2." | n/a | Even with engineering, Eagle would produce a *different* α (its own draft head's α), not Qwen3.5's pretrained MTP head's α. |
| 6 | Medusa / Hydra / SpecBench Qwen3.5 support | `FasterDecoding/Medusa@e2a5d20` (2024-04-18) | `unsupported` | Medusa main HEAD dates to 2024-04-18 (stale ~2 years). No Qwen3.5 support. Architecture mismatch identical to Eagle: Medusa heads are model-specific and trained per base, not loaded from Qwen3.5's pretrained MTP. | n/a | Same as (5) — would produce a different α. |
| 7 | NVIDIA TensorRT-LLM Qwen3.5-MTP support | `NVIDIA/TensorRT-LLM@1989520` (2026-04-24) | `unsupported` | `tensorrt_llm/models/qwen/model.py` (size 22,989) declares `QWenDecoderLayer` (line 42), `QWenModel` (170), `QWenForCausalLM` (234) — Qwen-1/2 era only. `tensorrt_llm/models/` has no `qwen3_5/` directory; the directory listing (`baichuan, bert, …, qwen, …, eagle, medusa, redrafter, …`) shows generic spec-decode infrastructure (eagle/medusa/redrafter dirs) but no Qwen3.5 path through any of them. Search for `qwen3_5` / `Qwen3.5` / `MultiTokenPredictor` in TRT-LLM returned no matches before rate limit; the `qwen/` path strongly suggests TRT-LLM has not added Qwen3.5 yet. | n/a — feature absent | n/a |
| 8 | Hand-rolled HF transformers reference α | `huggingface/transformers@c472755` + new code | `feasible` (not "supported" in the strict sense) | HF transformers `Qwen3_5ForCausalLM` (line 1690 in `modeling_qwen3_5.py`) provides the dense base model. Building an MTP reference would require: (a) load `Qwen3_5ForCausalLM`, (b) manually parse `mtp.*` tensors from the safetensors index (15 prefixed tensors per `kiln-mtp-audit-vllm-config-cross-ref` agent note), (c) port `crates/kiln-model/src/forward.rs::mtp_forward_step` and `crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step` into ~300-500 LOC of Python, (d) wire greedy verify-target loop. Algorithmically equivalent to kiln-native; intent is to factor out kiln's Marlin W4A16 + GDN + paging implementations as confounders. | ~2-4 hours engineering + ~30 min CPU run = $0 GPU spend, but high human-time cost | The reference is "kiln's reading of the canonical contract executed in pure Python" — if it produces α matching kiln's α, that doesn't tell us whether kiln+the reference both share an implementation flaw or whether α=0.689 is the head's intrinsic ceiling. Useful only if it produces α materially higher than kiln. |

**Mandatory candidate count: 1 supported, 2 unknown, 4 unsupported, 1 feasible-but-not-strictly-supported.**

## Optional candidates (not in the mandatory list, found during inspection)

| name | verdict | evidence |
| --- | --- | --- |
| LMStudio / llama.cpp draft model | `unsupported` | llama.cpp's `--model-draft` path needs a separate draft model in GGUF; Qwen3.5's `mtp.*` head has not been packaged as a GGUF draft model and the architecture (single MTP layer attending to base-model post-final-norm h_prev) is not what llama.cpp's draft-model path expects. |
| Mistral.rs (`EricLBuehler/mistral.rs`) | `unsupported` (no `main` branch found via gh API) | gh api 404 on `mistral.rs/contents/...?ref=main`. Repository may use a non-default branch name, but absent positive evidence of Qwen3.5-MTP support, classify as unsupported for this audit. Search for `MTP` in the repo returned no results before rate limit. |
| ExLlamaV2 / candle examples / mlx-lm | `unsupported` (no inspection) | None of these have published Qwen3.5-MTP support. Inspection skipped under the wall-clock budget. |
| Tinker / NVIDIA Triton Inference Server speculative path | `unsupported` (TRT-LLM is the relevant upstream; covered in row 7) | Triton Inference Server delegates spec decoding to TRT-LLM as the executor backend; with TRT-LLM lacking Qwen3.5, this path inherits the same gap. |

## Decision-rule application

Mandatory `supported` count: **1** (SGLang main, with the open-PR-#23331 caveat noted).
Mandatory `unknown` count: **2** (vLLM main, vLLM v0.20.0 — same code-state, count as one resolvable unknown).
Mandatory `unsupported` count: **4** (transformers, Eagle, Medusa, TRT-LLM).
Mandatory `feasible` count: **1** (hand-rolled HF reference).

→ **Branch 1 of decision rule fires: queue exactly ONE H17 microbench task.**

## Queued next action

**H17 — SGLang main α microbench against PR #529's prefill-byte-equal workload.**

### Scope

One A6000 pool lease, ≤45 min wall-clock, ≤$0.40 hard cost cap. Re-uses PR #530's `scripts/h15c_*` infrastructure with a new `scripts/h17_sglang_alpha_dump.py` driver.

### Workload (matched to PR #529 byte-for-byte at prefill boundary)

Identical to PR #530's matched-workload table, swapping vLLM for SGLang:

| field | value | source |
| --- | --- | --- |
| model | Qwen3.5-4B (dense, BF16, `Qwen3_5ForCausalLMMTP` arch) | `/workspace/qwen3.5-4b/config.json` |
| prompts | `PROMPT_POOL[seed % 30]` for seeds {0,1,2} → GSM8K prose 0/1/2 | `crates/kiln-server/src/bench.rs::build_prompt` |
| prompt-token target | 512 (sentence-repetition expander) | same as PR #530 |
| max output tokens | 16 | same as PR #530 |
| chat template | OFF (raw prose) | same as PR #530 |
| sampler | greedy: `temperature=0`, `top_p=1`, `top_k=0` (kiln) / `top_k=-1` (SGLang) | both drivers |
| spec method | k=1 native MTP (`KILN_SPEC_METHOD=mtp` / SGLang `--speculative-algorithm MTP --speculative-num-draft-tokens 1`) | both drivers |
| GPU | NVIDIA RTX A6000 sm_86, single-GPU bs=1 | RunPod |
| kiln quant | Marlin W4A16 (`KILN_W4A16=1`) | inherited from PR #529 |
| SGLang quant | BF16 (matches dense 4B checkpoint default) | acknowledged confounder, identical to vLLM-vs-kiln Marlin W4A16 confounder in PR #530 |

### Pre-registered decision rule for H17

| Δ = sglang_median_α − kiln_median_α | verdict | next action |
| --- | --- | --- |
| ≥ +0.05 | `external_ceiling_exists` | queue serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring); precedent for the bisect lattice exists in C-series |
| in [−0.02, +0.05) | `mtp_head_quality_ceiling` | accept α as upper-bounded by checkpoint quality; deprioritize MTP-side α work; refocus Phase 7 to decode-path wins |
| < −0.02 | `kiln_above_sglang` (unexpected) | sanity-check SGLang args; if confirmed, document and close project goal as kiln-native ceiling |
| `sglang.dense_4b_mtp_failed_to_load` | `sglang_mtp_unsupported_dense_4b` | doc-only redirect PR matching PR #530's shape; queue free pre-step (vLLM v0.20.0 retest) AND/OR escalate to hand-rolled HF reference |

### Required files in H17 PR

- `docs/phase7-h17-sglang-alpha-microbench.md` (full audit doc, mirroring PR #530's `docs/phase7-h15c-vllm-alpha-microbench.md` shape)
- `scripts/h17_sglang_alpha_dump.py` (SGLang driver — re-runnable, version-agnostic)
- `scripts/h17_compare.py` (apply decision rule, emit verdict.json)
- `docs/phase-c29-v3-sglang/{verdict,compare,kiln_alpha_per_seed,sglang_alpha_per_seed}.json`
- `docs/phase-c29-v3-sglang/compare.md`
- PROFILING.md top-of-file pointer entry

### Free pre-step (do BEFORE H17 GPU spend)

While the H17 lease is being acquired, the bench supervisor SHOULD also run a $0.25 bounded **vLLM v0.20.0 retest** of PR #530's exact driver (`scripts/h15c_vllm_alpha_dump.py`). Hard cap: 30 min wall-clock, no extension. Three plausible outcomes:

1. **vLLM v0.20.0 also segfaults** — `vllm_mtp_unsupported_v0_20_0` verdict; no further vLLM-track work; document the second segfault snapshot in the H17 PR appendix.
2. **vLLM v0.20.0 runs successfully** — third reference-α data point; verifies SGLang H17 result via cross-check; close H16 with stronger evidence.
3. **vLLM v0.20.0 segfaults at a different boundary** — log the new crash signature; classify as a separate unknown for a future H.

Either way, the SGLang H17 microbench is the canonical H17 deliverable; the vLLM retest is an opportunistic confirmation, not a substitute.

### Wall-clock budget

- H17 SGLang microbench: 45 min / $0.40
- Free pre-step (vLLM v0.20.0 retest, opportunistic): 30 min / $0.25
- Combined hard cap if both run: 75 min / $0.65

If the SGLang install / model-load alone consumes >30 min, abort the vLLM retest and ship the SGLang result alone. Per project §"Wall-clock budget (hard cap)" the executor MUST stop at the cap regardless of perceived completion proximity.

## What this PR does NOT queue

Per `kiln-phase-handoff-claims-need-reverification`, `kernel-vendor-precondition-check`, and PR #527's "What NOT to queue" list:

- ❌ Re-running PR #530's vLLM 0.19.1 driver. Already shipped, already `vllm_mtp_unsupported`. The free pre-step uses v0.20.0, not 0.19.1.
- ❌ Re-test of any C-series ruled-out hypothesis (C12 fp32 head, C20 dual-norm, C19 fc_norm, C24 RoPE A/B, C23 sampler drift, C27 H8 sequencing).
- ❌ Eagle / Medusa retraining for Qwen3.5. Different draft-head architecture; would not produce a Qwen3.5-MTP α reference.
- ❌ TRT-LLM path. No Qwen3.5 model in TRT-LLM as of `1989520`.
- ❌ Hand-rolled HF reference as the *primary* H17. Higher engineering cost than SGLang. Reserved as fallback if SGLang fails to load dense 4B + MTP and vLLM v0.20.0 also segfaults.
- ❌ MTP head fine-tune. Out of scope per C33 / H12 N/A.

## Anti-duplication evidence

```
$ gh pr list -R ericflo/kiln --state all --search "h16 external alpha"
(empty)

$ gh pr list -R ericflo/kiln --state all --search "external alpha reference"
530  phase 7: H15c vLLM α microbench — vllm_mtp_unsupported  MERGED  2026-04-25

$ gh pr list -R ericflo/kiln --state all --search "hf transformers mtp qwen"
425  phase6: emit full replay context for native mtp dumps   CLOSED  2026-04-23  (unrelated — replay context, not external reference)

$ gh pr list -R ericflo/kiln --state all --search "eagle medusa qwen3"
(empty)

$ gh pr list -R ericflo/kiln --state all --search "phase7 h16"
(empty)

$ gh pr list -R ericflo/kiln --state all --search "options audit"
530  phase 7: H15c vLLM α microbench — vllm_mtp_unsupported  MERGED  (same as above)

$ gh pr list -R ericflo/kiln --state all --search "phase7 audit"
226  phase7: GDN prefill memory audit (Phase 7 opener)
525  phase7: audit vLLM Triton GDN kernel against kiln-gdn-kernel
526  phase 7: audit kiln radix prefix cache vs SGLang RadixAttention
527  phase7: MTP acceptance-rate state-of-play audit (doc-only)
528  phase7: H15a Marlin pack determinism correlation (doc-only)
529  phase 7: H15b stratified C29 v2 reject-row probe — kiln_native_ceiling
530  phase 7: H15c vLLM α microbench — vllm_mtp_unsupported
(none cover external-α reference options enumeration)
```

The closest existing neighbors are #527 (MTP state-of-play, listed H15a/b as queued, predates H15c+H16), #530 (H15c, exhausted vLLM 0.19.1 path, queued H16 in §"Queued next action"), #526 (SGLang RadixAttention audit, scope-distinct — kernel cache, not α reference). No prior PR enumerates external-α reference candidates.

## What this rules out

- **Eagle, Medusa, TRT-LLM are dead ends for Qwen3.5-MTP α.** None has Qwen3.5 model coverage; even if added, Eagle and Medusa use their own draft heads, not Qwen3.5's pretrained `mtp.*` head. TRT-LLM lacks the model entirely.
- **HF transformers does not natively expose an MTP draft path.** The 5 candidate-generator types in `transformers/generation/candidate_generator.py` all assume a separate draft model. A Qwen3.5-MTP HF reference must be hand-rolled (option 8) — that is feasible but higher engineering cost than reusing SGLang.
- **PR #527 / PR #530's "SGLang has no MTP for Qwen3.5" cross-reference is mis-stated** (see Caveat under candidate 4). SGLang main has had `Qwen3_5ForCausalLMMTP` since 2026-02-09 (PR #18489) and dense 4B fix since 2026-03-24 (PR #21347). PR #526's accurate claim was that *SGLang's RadixAttention prefix cache* doesn't currently model MTP-aware reuse — a different question. Future planning cycles should treat SGLang as MTP-supported for Qwen3.5 going forward.

## What this does NOT rule out

- **SGLang dense Qwen3.5-4B + native MTP may fail at runtime** despite the static evidence above. The class is structurally compatible per the dense-path fix in PR #21347 and the load_weights() branching at line 170, but no end-to-end SGLang dense 4B + MTP run is in the SGLang test suite. The H17 retest is the empirical answer.
- **Even if SGLang loads, its α may match kiln's** (decision rule branch B → `mtp_head_quality_ceiling`). This is the most likely outcome given the H15b verdict that kiln's MTP head logits are at the BF16-noise cosine ceiling on reject rows. If H17 lands in branch B, the Phase 7 decode-track is not blocked — kiln's MTP is shippable from the head-quality side and the focus shifts to non-α decode-path wins (already in flight: PRs #517/#520/#521 prefix-cache, #525/#526 audits).
- **vLLM main's segfault may persist on dense 4B + MTP**, even with the kernel + spec-decode + autograd improvements in v0.20.0. The free pre-step would tell us; without it, vLLM remains unknown.

## Bench envelope + cost (for this PR)

- Pod: none. $0 GPU spend.
- Wall-clock spent: ~40 min Claude time (under 45 min cap).
- Inspection actions: only `gh api` + `gh pr list` + `gh search code`. No `git clone` of upstream repos. No SSH. No installs.
- Confidence floor: each `supported` / `unsupported` / `unknown` verdict cites file path + commit SHA / release version inspected. Search rate-limit hit twice during TRT-LLM and Mistral.rs inspection — both downgraded to `unsupported` based on directory listings + absent match evidence rather than positive search confirmation.

## Files

| path | purpose |
| --- | --- |
| `docs/phase7-h16-external-alpha-options-audit.md` | this audit doc |
| `PROFILING.md` (top-of-file pointer) | one-paragraph link entry matching #525/#526/#527/#530 shape |

No production source files (`crates/**/*.rs`, scripts run on a pod) are touched in this PR.

## Reopen preconditions

The H16 verdict shifts from `external_reference_exists` to a different branch under any of these signals:

- A vLLM commit on main referencing `qwen3.*MTP|mtp.*qwen3` and explicitly fixing the PR #530 segfault — would change candidate (2)/(3) from `unknown` to `supported`, possibly elevating vLLM above SGLang as the H17 winner.
- A SGLang main commit reverting Qwen3_5ForCausalLMMTP support or breaking the dense 4B path — would invalidate this PR's `supported` verdict on candidate (4).
- An Eagle or Medusa upstream config landing for Qwen3.5 with a mechanism to load Qwen3.5's pretrained `mtp.*` head (vs training their own head) — would change candidates (5)/(6) from `unsupported` to a third reference path.
- TRT-LLM adding `tensorrt_llm/models/qwen3_5/` — would change candidate (7) from `unsupported` to `unknown`/`supported`.
- The H17 SGLang microbench fails to load dense 4B + MTP at runtime (verdict branch D) — would invalidate the strict reading of the SGLang `supported` verdict and require the free-pre-step or hand-rolled-HF fallback.

## References

### kiln docs (commit `045bc9c`)

- `docs/phase7-h15c-vllm-alpha-microbench.md` — direct predecessor; defined the H16 free-text preconditions in §"Queued next action" item (a)/(b).
- `docs/phase7-h15b-stratified-c29-v2.md` — established `kiln_native_ceiling` verdict; framed the open question for H15c and H16.
- `docs/phase7-mtp-acceptance-state-of-play.md` (PR #527) — listed the 4 options in §"Fallback if H15b is also null" that this audit operationalizes.
- `docs/phase7-sglang-radix-audit.md` (PR #526) — disambiguates the SGLang "has Qwen3.5 MTP class" finding from the orthogonal "RadixAttention is MTP-aware" question. PR #526 audited the latter; H16 audits the former.

### Cited upstream sources

- vLLM `vllm/model_executor/models/qwen3_5_mtp.py` (main `bc2ae5a`, v0.19.1, v0.20.0)
- vLLM `vllm/model_executor/models/qwen3_next_mtp.py` (main `bc2ae5a`)
- vLLM `vllm/v1/spec_decode/eagle.py`, `vllm/v1/spec_decode/__init__.py`, `vllm/v1/spec_decode/extract_hidden_states.py`
- vLLM `vllm/config/speculative.py` (recent commit history `626daa2` 2026-04-24, `d0009dd` 2026-04-23, `27c0ca5` 2026-04-15)
- SGLang `python/sglang/srt/models/qwen3_5_mtp.py` (main `a4facdf`, blob `3fa89fcd`)
- SGLang `python/sglang/srt/models/qwen3_5.py` (main `a4facdf`, size 72,113)
- SGLang `python/sglang/srt/speculative/` (eagle_worker_v2, dflash_worker, multi_layer_eagle_worker, etc.)
- SGLang PRs #18489, #18538, #19391, #19767, #21347, #23331
- HF transformers `src/transformers/models/qwen3_5/modeling_qwen3_5.py` (main `c472755`, blob `2c4eba9`)
- HF transformers `src/transformers/generation/candidate_generator.py` (main `c472755`)
- SafeAILab/EAGLE `eagle/model/` directory listing (main `cb7e0841`, README support entry "2024.8.8: Qwen-2")
- FasterDecoding/Medusa root listing (main `e2a5d20`, 2024-04-18)
- NVIDIA/TensorRT-LLM `tensorrt_llm/models/` directory listing + `tensorrt_llm/models/qwen/model.py` (main `1989520`)

### Agent notes (verified to exist via `ce notes-search --q <topic>`)

`vllm-qwen35-mtp-v1-segfault-2026-04`, `kiln-mtp-reference-source-of-truth`, `kiln-speculative-decoding-design`, `kiln-mtp-audit-vllm-config-cross-ref`, `kernel-vendor-precondition-check`, `kiln-phase7-consolidation-doc-pattern`, `static-audit-before-pod-spend`, `kiln-mtp-phase-c-structural-audit-pattern`.
