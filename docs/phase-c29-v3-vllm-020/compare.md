# H17b — vLLM v0.20.0 α microbench vs kiln (free pre-step retest)

**Verdict:** `vllm_020_mtp_unsupported_dense_4b`

**vLLM version:** `0.20.0`

- kiln median α: **0.36363636363636365**
- vLLM v0.20.0 median α: **UNAVAILABLE** (vLLM v0.20 failed to load Qwen3.5-4B with MTP speculative config: vLLM failed to load with any MTP speculative config; tried:
  kwargs={'speculative_config': {'method': 'qwen3_5_mtp', 'num_speculative_tokens': 1, 'target_model_config': ModelConfig(model='/workspace/qwen3.5-4b', model_weights='', run)

## Decision rule (pre-registered, unchanged from PR #530/#531/#532)

| verdict | bound | next action |
|---|---|---|
| `external_ceiling_exists` | `delta >= +0.05` | External system hits materially higher α. Queue a serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring) to find what kiln diverges on. |
| `mtp_head_quality_ceiling` | `-0.02 <= delta < +0.05` | α is upper-bounded by checkpoint quality. Deprioritize MTP-side α work and redirect Phase 7 to decode-path optimizations that do not depend on raising α (e.g. post-#521 prefix-cache + CUDA graphs wins, SGLang RadixAttention port from #526). |
| `kiln_above_vllm` | `delta < -0.02` | Unexpected — kiln α exceeds vLLM at same workload. Sanity-check vLLM args (spec method, num_speculative_tokens, sampler). If confirmed, document and queue next H. |
| `vllm_020_mtp_unsupported_dense_4b` | `vllm_020_alpha_per_seed.json.mtp_supported == false` | vLLM v0.20.0 also fails to serve Qwen3.5-4B dense + native MTP on A6000 (mirrors PR #530's PR-#530 0.19.1 segfault + PR #532's SGLang segfault). Both external-OSS-server candidates from PR #531 are now blocked. Queue the hand-rolled HF transformers H18 reference (PR #531 candidate 8) as the next external-α reference path; kiln-native-ceiling verdict from H15b stands until H18 lands. |

## Per-seed table

| seed | kiln α | notes |
|---|---|---|
| 0 | 0.3636 | vLLM v0.20.0 unavailable |
| 1 | 0.3333 | vLLM v0.20.0 unavailable |
| 2 | 0.4545 | vLLM v0.20.0 unavailable |

## Next action

vLLM v0.20.0 also fails to serve Qwen3.5-4B dense + native MTP on A6000 (mirrors PR #530's PR-#530 0.19.1 segfault + PR #532's SGLang segfault). Both external-OSS-server candidates from PR #531 are now blocked. Queue the hand-rolled HF transformers H18 reference (PR #531 candidate 8) as the next external-α reference path; kiln-native-ceiling verdict from H15b stands until H18 lands.
