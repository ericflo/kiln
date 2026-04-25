# H15c — vLLM α microbench vs kiln (external-reference upper bound)

**Verdict:** `vllm_mtp_unsupported`

- kiln median α: **0.36363636363636365**
- vLLM median α: **UNAVAILABLE** (vLLM 0.19.1 segfaults during V1 engine init when loading Qwen3.5-4B with native MTP speculative decoding on A6000, post drafter weight load. All three method aliases (qwen3_5_mtp -> deprecated to mtp / mtp / qwen3_next_mtp -> deprecated to mtp) reach the same crash point and produce identical native-extension backtraces with no Python file symbols. enforce_eager=True (CUDA graphs off), limit_mm_per_prompt={image:0,video:0}, and skip_mm_profiling=True all took effect (vLLM logs 'running in text-only mode' and 'Cudagraph is disabled under eager mode') yet the segfault still fires. See artifacts/vllm_segfault_evidence.log for the trimmed log evidence and reproduction context.)

## Decision rule (pre-registered)

| verdict | bound | next action |
|---|---|---|
| `external_ceiling_exists` | `delta >= +0.05` | External system hits materially higher α. Queue a serving-difference bisect (sampler / KV layout / quantization path / RoPE / MTP-head wiring) to find what kiln diverges on. |
| `mtp_head_quality_ceiling` | `-0.02 <= delta < +0.05` | α is upper-bounded by checkpoint quality. Deprioritize MTP-side α work and redirect Phase 7 to decode-path optimizations that do not depend on raising α (e.g. post-#521 prefix-cache + CUDA graphs wins, SGLang RadixAttention port from #526). |
| `kiln_above_vllm` | `delta < -0.02` | Unexpected — kiln α exceeds vLLM at same workload. Sanity-check vLLM args (spec method, num_speculative_tokens, sampler). If confirmed, document and queue next H. |
| `vllm_mtp_unsupported` | `vllm_alpha_per_seed.json.mtp_supported == false` | vLLM does not yet support Qwen3.5-4B native MTP at the installed version. Ship this as a doc-only redirect PR documenting the gap, queue the next H from PR #527 §'Queued next action'. |

## Per-seed table

| seed | kiln α | notes |
|---|---|---|
| 0 | 0.3636 | vLLM unavailable |
| 1 | 0.3333 | vLLM unavailable |
| 2 | 0.4545 | vLLM unavailable |

## Next action

vLLM does not yet support Qwen3.5-4B native MTP at the installed version. Ship this as a doc-only redirect PR documenting the gap, queue the next H from PR #527 §'Queued next action'.
