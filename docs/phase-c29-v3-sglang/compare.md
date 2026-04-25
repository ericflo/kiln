# H17 — SGLang α microbench vs kiln (external-reference upper bound)

**Verdict:** `sglang_mtp_unsupported_dense_4b`

- kiln median α: **0.36363636363636365**
- SGLang median α: **UNAVAILABLE** (SGLang 0.5.10.post1 + Qwen3.5-4B dense BF16 + native MTP segfaults on A6000 sm_86 across three distinct serving configurations. All three attempts produced Segmentation fault / exit code -11 in native-extension code with no Python file symbols at the crash point. This mirrors the vLLM 0.19.1 failure class observed in PR #530 but manifests in different upstream code paths.)

## Decision rule (pre-registered, from PR #531)

| verdict | bound | next action |
|---|---|---|
| `external_ceiling_exists` | `delta >= +0.05` | SGLang MTP heads beat kiln MTP heads at same workload. Queue an H18 serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring) to find what kiln diverges on. Since kiln is Marlin W4A16 and SGLang is BF16 this confounder must be isolated first (re-run kiln in BF16 or port SGLang's quant path). |
| `mtp_head_quality_ceiling` | `-0.02 <= delta < +0.05` | Both implementations hit the pretrained MTP head's quality ceiling. No external α headroom available from SGLang either. Close the external-α reference family. Pivot Phase 7 to: (a) MTP head retraining (H18 candidate), or (b) non-MTP speculative decoding (Eagle-style draft model trained from scratch on kiln logs). |
| `kiln_above_sglang` | `delta < -0.02` | Unexpected — kiln α exceeds SGLang at same workload. Sanity-check SGLang args (spec_algorithm, num_draft_tokens, sampler, dtype). If confirmed, no external reference path can help raise α. Close H17 family and document kiln-native ceiling. |
| `sglang_mtp_unsupported_dense_4b` | `sglang_alpha_per_seed.json.mtp_supported == false` | SGLang does not yet support Qwen3.5-4B dense + native MTP end-to-end at the installed version. Escalate to free pre-step (vLLM v0.20.0 retest of PR #530 segfault) — if also fails, queue hand-rolled HF transformers H18 reference (PR #531 candidate 8). |

## Per-seed table

| seed | kiln α | notes |
|---|---|---|
| 0 | 0.3636 | SGLang unavailable |
| 1 | 0.3333 | SGLang unavailable |
| 2 | 0.4545 | SGLang unavailable |

## Next action

SGLang does not yet support Qwen3.5-4B dense + native MTP end-to-end at the installed version. Escalate to free pre-step (vLLM v0.20.0 retest of PR #530 segfault) — if also fails, queue hand-rolled HF transformers H18 reference (PR #531 candidate 8).
