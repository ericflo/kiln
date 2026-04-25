# H18 — hand-rolled HF transformers MTP α reference vs kiln

**Verdict:** `kiln_above_hf`

**transformers:** `5.7.0.dev0` **torch:** `2.4.1+cu124`

- kiln median α: **0.3636**
- HF transformers median α: **0.2500**
- Δ (hf − kiln): **-0.1136**

## Decision rule (pre-registered, derived from PR #531/#533)

| verdict | bound | next action |
|---|---|---|
| `external_ceiling_exists_hf` | `delta >= +0.05` | HF transformers reference α exceeds kiln α by ≥ 0.05. Kiln has α-quality work to do — queue an α-improvement task to bisect the kiln-vs-HF MTP-head implementation difference (likely candidates from prior phases: Marlin W4A16 quantization drift, fc.weight transpose convention, or pre_fc_norm swap). The kiln-native-ceiling verdict from H15b is *displaced* by this finding. |
| `mtp_head_quality_ceiling` | `-0.02 <= delta < +0.05` | Both implementations land within ±0.05 of each other. The pretrained MTP head's checkpoint quality is the dominant ceiling. Accept and refocus Phase 7 on non-α decode-path wins (post-PR #521 prefix-cache + CUDA graph work, SGLang RadixAttention port from PR #526). The H15b `kiln_native_ceiling` verdict stands as the operative checkpoint-quality ceiling. |
| `kiln_above_hf` | `delta < -0.02` | Kiln α exceeds the HF transformers reference by > 0.02 — unexpected for an MTP head that should be implementation-deterministic at greedy decode. Sanity-check the HF reference (MTP head loader prefix, pre_fc_norm swap convention, RoPE position threading). If confirmed, the H15b kiln-native-ceiling verdict stands as a safe lower bound and the H17/H18 family closes. |
| `hf_mtp_load_failure` | `hf_alpha_per_seed.json.mtp_supported == false` | Hand-rolled HF transformers reference also failed to load or drive the pretrained MTP head end-to-end. With vLLM 0.19.1 + v0.20.0 (PR #530, #533), SGLang main HEAD (PR #532), AND HF transformers all blocked, no external α reference is available for Qwen3.5-4B + native MTP on A6000 sm_86. Close the external-α reference family with the `kiln_native_ceiling` verdict from H15b as the operative checkpoint-quality ceiling. Phase 7 next steps: refocus on non-α decode-path wins (prefix-cache + CUDA graphs, RadixAttention port). |

## Per-seed table

| seed | kiln α | HF α | kiln accept/steps | HF accept/steps |
|---|---|---|---|---|
| 0 | 0.3636 | 0.3636 | 4/11 | 4/11 |
| 1 | 0.3333 | 0.2308 | 4/12 | 3/13 |
| 2 | 0.4545 | 0.2500 | 5/11 | 3/12 |

## Next action

Kiln α exceeds the HF transformers reference by > 0.02 — unexpected for an MTP head that should be implementation-deterministic at greedy decode. Sanity-check the HF reference (MTP head loader prefix, pre_fc_norm swap convention, RoPE position threading). If confirmed, the H15b kiln-native-ceiling verdict stands as a safe lower bound and the H17/H18 family closes.
