# Phase 7 Audit: MTP Acceptance-Rate State of Play

Date: 2026-04-24
kiln main at audit time: `2f38eb6` (post-PR #526)
vLLM reference at audit time: `vllm/model_executor/models/qwen3_next_mtp.py` snapshot 2026-04-21 (per phase-c17/c25/c26 cites)
Scope: doc-only static audit consolidating Phase B / C work, no pod spend, no SSH, no new Rust code.

## Summary

Native MTP speculative decoding for Qwen3.5-4B is the third and last open project goal in `Project: Kiln`. The other two ("audit kiln-gdn-kernel against vLLM," "audit kiln radix prefix cache vs SGLang RadixAttention") closed via doc-only PRs #525 and #526. The MTP α gap is the live blocker for closing the decode-speed gap to vLLM/SGLang.

After **34 MTP-named phase-C PRs** (#311 through #382, with C40a–C40f and C57+ refresh attempts) and the recent post-#481/#500/#502 GDN-kernel work, current state on a stable A6000 humaneval+chat-template anchor is:

- **Median α = 0.689 (N=20, C40f anchor, PR #379)**, 95% CI `[0.652, 0.723]`, with 6/20 seeds ≥ paper-floor 0.72.
- **Median decode = 38.25 tok/s (C40f anchor)** with chat-template + W4A16 + FP32 argmax. Post-#500 native-MTP decode anchor (C60) re-measured at **23.5 tok/s / α=0.716** on a single seed; C57 single-seed decode at 26.5 tok/s / α=0.764. The anchor is variance-bound, not contract-bound.
- **C36 identity-bias re-measure (post-#417, 3 seeds, default flags) showed a hard seed split:** seed 0 reached α=0.7397, seed 1 α=0.3093, seed 2 α=0.3956 — with 80.85% of all reject rows showing `draft_top1 == last_token` (identity bias). One seed cleared the floor; two stayed trapped in the low-α regime on the same standard workload.

Every named hypothesis on the inference-side pipeline (H1–H14) has been ruled out, and the prompt-distribution / chat-template / dtype residuals (C34/C35) are quantified and partially absorbed into the anchor. Five non-trivial unknowns remain — three related to seed-conditioned regime split (C36, C38–C45 layer-1 row-broadcast bisect that landed in C47-classified-out-of-scope), one related to MTP fine-tune absence (C33/H12 N/A), and one related to verifier α-conditional-on-match diagnostics that were never re-run on the post-C18 build.

**Recommendation: do NOT queue a fresh GPU-spend bisection task in this PR.** The next planning cycle should pick the one falsifiable next H from §4 below. If §4 indicates "fresh decode-step trace first," it should queue a $0 doc-only re-profile that anchors on the post-#502 MTP path, not another single-hypothesis A/B.

## Verdict

**Hold the project goal open. Document state. Recommend a single next falsifiable H plus a fallback "re-anchor first" path.**

- The α gap is **real and persistent**: 0.689 median is below the 0.72 paper floor and far enough from 1.0 that ~31% of draft tokens are rejected at the verifier; net decode is *slower than plain decode* at this α (post-#481 single-seed: 41.2 tok/s MTP-on at α=0.730 vs ~49.76 tok/s post-#166 plain-decode baseline).
- The α gap is **not** in the inference-side accept/reject pipeline, lm_head trio, h_prev frame, MTP-block sequencing, or weight loading — all conclusively ruled out via static + empirical evidence (see §1).
- The α gap **is plausibly** in (a) the seed-conditioned identity-bias regime split first surfaced in C36 and never explained, (b) the unmeasured class of "verifier-side numerical drift on Class B reject rows" that C29 only checked on accepted tokens, or (c) the workload distribution still drifting from the MTP head's training distribution despite the chat-template fix. Section §4 scores these.
- A "fresh decode-step per-step trace + identity-bias breakdown on the post-#500 build" is the cheapest doc-only next step that would either confirm one of (a)/(b)/(c) or establish that identity-bias has been rendered moot by the post-#481 / post-#500 GDN refactors.

Reopen precondition for the optimization track: a single seed on the C40f-style harness anchor that produces α ≥ 0.78 and decode ≥ +15% over plain-decode baseline. This is the speed-evidence threshold that would mean MTP is shippable enough to start ratcheting α upward via head fine-tuning instead of root-causing the head.

## Preconditions verified

| Precondition | Status | Evidence |
| --- | --- | --- |
| Latest MTP-named PR before this audit | #382 (`Reconcile C40 MTP docs after C40f`), merged 2026-04-22 | `gh pr list -R ericflo/kiln --state all --search "mtp in:title" --limit 100`. Subsequent MTP-touching PRs were profile-refresh (#417, #420, #426, #454, #462, #464, #465, #468, #470, #480, #491, #493, #495) and Metal-routing (#400, #402, #437) — none added a new H or moved α. |
| C40f N=20 anchor still canonical | Yes — `docs/phase-c40f/summary.json` is the most recent N≥20 α distribution. | C50/C51/C57/C60 are single-seed or 3-seed refreshes that anchor against C40f's median. |
| Post-#500 / post-#502 GDN refactors do not silently fix α | Yes — C60 single seed: α=0.716. C57 single seed: α=0.764. Within C40f distribution. No code path was changed that would alter MTP head output between #382 and #502. | `git log --oneline 2f38eb6..382 -- crates/kiln-model/src/speculative.rs crates/kiln-model/src/forward.rs | grep mtp` shows only the conv1d prefill fix (#481) and Metal-routing changes (none of which touch CUDA MTP draft-head output). |
| No open or pending PR already covering this state-of-play | Verified | `gh pr list -R ericflo/kiln --state all --search "mtp acceptance state-of-play" --limit 5` returned `[]`; same for `"mtp audit consolidated"` and `"mtp" --state open`. |
| C29 top-K Jaccard verdict still cited as "head logits clean" | Yes, with caveat | C29 measured cos_sim 0.999978 / top-1 match 100% / Jaccard@5 median 1.0 across 49 paired dumps — but only on accepted-token positions (h_main was reference-echoed). C36 demonstrated this does not contradict 80.85% reject-row identity bias because the C29 dumps were not stratified by accept/reject. |
| C16 / C30 accept-reject + KV rollback static audits | Yes — both REJECTED on direct source reading. The accept/reject machinery is not the bug. | `docs/phase-c16/c16-mtp-accept-reject-audit.md`, `docs/phase-c30/c30-accept-reject-kv-rollback.md`. |

## Sources

### kiln source (commit `2f38eb6`)

- `crates/kiln-model/src/speculative.rs::speculative_mtp_decode_step` — k=1 verify loop. Greedy draft + greedy verify pos-0 + greedy bonus pos-1. `_params: &SamplingParams` and `_rng: &mut StdRng` are underscore-prefixed proving non-consumption (C23/C34 evidence).
- `crates/kiln-model/src/forward.rs::model_forward_paged_with_last_hidden`, `model_forward_paged_inner`, `LmHeadMode::FullWithLastHidden`, `mtp_forward_step` — base + MTP forward chain. `h_prev` is now post-final-norm (C18 fix).
- `crates/kiln-model/src/weights.rs::MtpWeights`, `crates/kiln-model/src/loader.rs::load_mtp_weights` / `load_mtp_if_present` / `detect_mtp_prefix` / `extract_tensor` — 15 MTP-prefixed tensors loaded with byte-equal shapes/dtypes vs HF safetensors header (C26 audit).
- `crates/kiln-server/src/bench.rs::bench_latency_paged_mtp` — drives the C40f-style MTP latency anchor with `--paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --chat-template --latency-only --prompt-subset humaneval --temperature 0.0` and `KILN_SPEC_METHOD=mtp KILN_BENCH_FORCE_MTP=1 KILN_MTP_ARGMAX_FP32=1 KILN_W4A16=1 KILN_CUDA_GRAPHS=true`.
- `PROFILING.md` §"Native MTP speculative decoding — preflight 2026-04-20" (PR #240) — original GREEN preflight that quoted Qwen3.5/DeepSeek-V3 paper-floor α ≈ 0.72–0.85 → math ceiling 1.53×–1.57× over the 49.76 tok/s decode baseline.
- `PROFILING.md` §"Phase 6 post-#481 current-main profile refresh" — single-seed A40 fallback with native MTP at 41.2 tok/s / α=0.730. NOT a ship verdict.

### vLLM reference (per C17/C25/C26/C28/C30 citations)

- `vllm/model_executor/models/qwen3_next_mtp.py::Qwen3NextMultiTokenPredictor.forward` — post-final-norm `last_hidden_state` contract (C17 verdict, C18 fix).
- `vllm/v1/sample/rejection_sampler.py::rejection_sample` `all_greedy` branch — greedy draft / greedy verify / integer-compare accept (C23/C34 contract source).
- `vllm/v1/spec_decode/eagle.py::_greedy_sample` — shared draft+verifier sampling primitive.

### Phase B / C run anchors (in chronological order, ALL on A6000 unless noted)

- C5 (PR #316, 2026-04-21): α median 0.124, decode 32.6 tok/s vs Off 43.5 tok/s (-25.1%). Class B head-error dominant at 87.6%.
- C18 (PR #341, 2026-04-22): h_prev post-final-norm fix. α 0.000 → 0.153 (4.7×). Below 0.5 floor.
- C29 (PR #355, 2026-04-22): top-K Jaccard clean on 49 paired dumps. cos_sim 0.999978, Jaccard@5 median 1.0. **Only checked accepted-token positions.**
- C35 (PR #364, 2026-04-22): chat-template raises α from 0.176 to 0.588 (prose) / 0.608 (chat × FP32). Workload sensitivity is the dominant factor; dtype is noise-level (≤0.02).
- C37 (PR #369, 2026-04-22): N=10 variance re-anchor at C35 Cell D. Median α 0.6076 (matched C35), N=10 demonstrated sub-3-seed sampling was structurally too thin.
- C39–C45 / C47 (PRs #371–#445, 2026-04-22 to 2026-04-23): row-by-row bisect of seed-1 upstream `h_main` divergence. Found a layer-1 input-layernorm broadcast-output tolerance artifact at the C45 boundary. C47 stop-condition: do not change production inference math at the C45 broadcast boundary unless a fresh current-main reproducer again fails the current mask. C48 verdict per #459: leave alone, harness comparability is the real story.
- C40f (PR #379, 2026-04-22): **N=20 paper-floor sweep at C40f-style anchor.** Median α 0.689, 95% CI [0.652, 0.723], 6/20 seeds ≥ 0.72. Median decode 38.25 tok/s. **This remains the canonical N≥20 α distribution as of `2f38eb6`.**
- C36 (PR #420, 2026-04-23): post-#417 identity-bias re-measure. 3 seeds, standard workload. Seed 0 α=0.7397 / 10.96% identity bias; seed 1 α=0.3093 / 59.79% identity bias; seed 2 α=0.3956 / 52.75% identity bias. **80.85% of reject rows are identity-biased; 0% of accept rows are.** Seed split is unexplained.
- C50/C51/C52/C57/C60 (PRs #442/#465/#468/#495/#502, 2026-04-24): post-#466/#476/#481/#498/#500/#502 single-seed-or-3-seed C40f-style anchor refreshes. α stable in [0.588, 0.789] band; decode 23.5–26.5 tok/s in profiled runs (Nsight overhead included), 36.5–46.4 tok/s in unprofiled bench. None claim α moved on the median.

## 1. H-hypothesis inventory

Each row covers one named hypothesis attempt. "Verdict" is the doc's own word. "Doc-only?" indicates whether the PR spent pod time. "Α moved?" reports whether any reported α median changed by >0.02.

| Phase / PR | Hypothesis | Doc-only? | Verdict | Α moved? | Evidence file |
| --- | --- | :---: | --- | :---: | --- |
| B Tier 2 #260 | KILN_MTP_DEBUG instrumentation | No | DELIVERED — produced the α=0.154 / 46% identity-bias trace that opened the entire C-hypothesis tree | n/a | logs only |
| B12 #309 | Layer-31 drift signal | No (bench) | RULED OUT — layer-31 drift is benign bf16 accumulation | No | `docs/MTP_PHASE_B12.md` |
| C1 #311 | Per-step accept/reject CSV attribution | No (bench) | DELIVERED — α=15.11%, Class A=0, Class B=87.6%, head bug not accept-path bug | n/a | `docs/phase-c1/` |
| C2 #313 | RoPE position threading bisect | No (bench) | CONFIRMED — RoPE divergence at mtp_pos≥1 | n/a (pre-fix) | `docs/phase-c2/` |
| C3 #314 | RoPE fix (apply C2 verdict) | No (bench) | LANDED — pre-RoPE cos_sim 1.000 at all positions | No (α stayed at 0.124) | `docs/phase-c3/` |
| C5 #316 | Bench α + decode tok/s post-C3 | No (bench) | MISS on both axes (α 0.124 / decode -25.1%) | No | `docs/phase-c5/` |
| C6/C7/C8 #317/#319/#320 | Pre-RoPE input bisect → SDPA-internal bisect → kv_len single-site fix | Mixed | DELIVERED — kv_len mismatch fixed at single site | No (α-impact deferred to later C-row bench) | `docs/phase-c8/` |
| C9 #322 | fc_output drift | No (bench) | NULL — bf16 noise, not α signal | No | `docs/phase-c9/` |
| C10 #325 | fp32 HF reference comparator (WIP) | Partial | EXECUTION BLOCKED — comparator scaffold only | n/a | `docs/phase-c10/` |
| C11 #326 | Marlin W4A16 per-channel scale drift (H?) | No | NULL on static audit | No | `docs/phase-c11/` |
| C12 #329 | fp32 draft head kill switch (H?) | No (bench) | PRIMARY-NEGATIVE — α 0.0325 vs 0.0325, costs 8.3% tok/s | No | `docs/phase-c12/` |
| C13 #331 | MTP weight-loading + pre-projection splice | Partial | DELIVERED audit + dump tap | No | `docs/phase-c13/` |
| C14 #334 | Post-MTP-transformer-block splice dump | Partial | DELIVERED dump | No | `docs/phase-c14/` |
| C15 #338 | h_main drift across decode steps | No (bench) | DELIVERED — found 2.0–2.4× kiln/HF magnitude ratio on h_main, smoking gun for missing RMSNorm | n/a | `docs/phase-c15/` |
| C16 #339 | Accept/Reject plumbing audit (4 sub-H) | Yes | ALL FOUR REJECTED on static audit | No | `docs/phase-c16/` |
| C17 #340 | h_prev reference-frame audit | Yes | CONFIRMED — kiln passed pre-final-norm `h_prev` instead of post-final-norm | n/a (pre-fix) | `docs/phase-c17/` |
| C18 #341 | h_prev post-norm fix | No (bench) | LANDED + α 0.000 → 0.153 (4.7×). Below 0.5 floor; residual gap to C19+. | **+0.153** | `docs/phase-c18/` |
| C19 #343 | mtp.fc_norm reuse (H1 of C18 handoff) | Yes | DISPROVEN — Qwen3.5-4B has no `mtp.fc_norm.weight`; it has `pre_fc_norm_{embedding,hidden}` and kiln already loads + applies both | No | `docs/phase-c19/` |
| C20 #344 | MTP-block dual-norm wiring (H2) | Yes | DISPROVEN — kiln binds the four MTP-specific RMSNorms in canonical order | No | `docs/phase-c20/` |
| C21 #346 | Rotary mtp_pos offset (H3 static) | Yes | INCONCLUSIVE — vLLM has two conventions, kiln matches DFlash one | No | `docs/phase-c21/` |
| C22 #347 | fc + residual parameterization (H4) | Yes | DISPROVEN — line-for-line match to vLLM | No | `docs/phase-c22/` |
| C23 #348 | Draft-side sampler / temp / penalty (H5) | Yes | DISPROVEN — `_params` underscore-prefixed proves non-consumption | No | `docs/phase-c23/` |
| C24 #350 | Rotary mtp_pos hardware A/B (H3) | No (bench) | REJECTED — Δα +0.010 inside ±0.05 null band; both conventions essentially identical | No | `docs/phase-c24/` |
| C25 #351 | Verifier vs draft logit path (H6) | Yes | NULL — both paths tied to embed_tokens_t, both feed post-final-norm | No | `docs/phase-c25/` |
| C26 #352 | MTP weight-reload sanity (H7) | Yes | NULL — 15 tensors land with expected names/shapes/dtypes, escape Marlin packing cleanly | No | `docs/phase-c26/` |
| C27 #353 | MTP block sequencing + paged-KV carryover (H8) | Yes | RULED OUT — 5/5 sub-properties consistent | No | `docs/phase-c27/` |
| C28 #354 | h_prev post-norm contract (H10) | Yes | RULED OUT — kiln matches the 4-step vLLM contract bit-for-bit structurally | No | `docs/phase-c28/` |
| C29 #355 | Empirical MTP logits compare (H9) | No (bench) | TOP-K JACCARD CLEAN on 49 paired dumps. **Caveat: only on accepted-token positions.** | No | `docs/phase-c29/` |
| C30 #356 | Accept/Reject + KV rollback (H11) | Yes | REFUTED on static audit (3 sub-H) | No | `docs/phase-c30/` |
| C31 #358 | Class B head-trio bisect | Yes | REFUTED — head trio is the most exhaustively audited surface | No | `docs/phase-c31/` |
| C32 #359 | h_main fp32 parity (final Class B) | Yes | REFUTED via prior-evidence consolidation (C15+C18+C29 transitively) | No | `docs/phase-c32/` |
| C33 #360 | MTP head fine-tune budget (H12) | Yes | N/A — kiln has no MTP fine-tune path; weights only ever come from the published checkpoint | No | `docs/phase-c33/` |
| C34 #362 | Sampler contract parity vs vLLM (H13 static) | Yes | INCONCLUSIVE — kiln matches the rejection_sampler all_greedy branch functionally; two non-zero deltas (chat-template, fp32 argmax) flagged for C35 to A/B | No | `docs/phase-c34/` |
| C35 #364 | H13 4-cell A/B (prose/chat × BF16/FP32) | No (bench) | REFUTED on dtype residual; CONFIRMED-PARTIAL on prompt-workload residual. **Chat-templated α 0.6076 vs prose 0.176; +234% absolute. One seed hit 0.764.** | **+0.43** (anchor moved to chat-template) | `docs/phase-c35/` |
| C36 #368 / #420 | H14a decode-length sweep + post-#417 identity-bias re-measure | No (bench) | H14a NULL (decode-length not the issue); identity-bias dominant on reject rows (80.85%); seed split unexplained | No | `docs/phase-c36/` |
| C37 #369 | N=10 variance re-anchor at C35 Cell D | No (bench) | Median α 0.6076 (matched C35); 3-seed sampling was too thin | No | `docs/phase-c37/` |
| C38 #370 | Expand prompt pool 8→30 | No (bench) | Confirmed seed-1 stays in low-α regime even with broader pool | No | `docs/phase-c38/` |
| C39 #371 | HumanEval-only N=20 domain-focused α | No (bench) | Domain matters but does not close the gap | No | `docs/phase-c39/` |
| C40+C40a–C40f #372–#382 | Per-domain α floors synthesis + N=20 distribution | No (bench) | **C40f N=20: median α 0.689, 95% CI [0.652, 0.723], 6/20 seeds ≥ 0.72.** Anchor for everything since. | No | `docs/phase-c40f/summary.json` |
| C41–C45 #427–#445 | Layer-1 sub-op bisect (input layernorm) | No (bench) | DELIVERED — localized seed-1 upstream `h_main` divergence to a layer-1 input_norm broadcast tolerance boundary | No | `docs/phase-c4{1..5}/` |
| C47 #459 | Tolerance-artifact classification at C45 boundary | Yes | OUT-OF-SCOPE for production-math change. Stop condition: do not change inference math unless a fresh current-main reproducer again fails the current mask. | No | logged in PROFILING.md C45 §"2026-04-24 C47" |
| C48–C50 #462–#465 | Forced-MTP A6000 benchmark refresh + harness-parity A/B | No (bench) | C48 regression was a harness/workload comparability artifact (C49/C50 verdict). C40f-style anchor restored. | No | `docs/phase-c{48,49,50}/` |
| C51–C57 #468–#495 | Post-#466/#476/#481/#498 native-MTP profile refreshes | No (bench) | DELIVERED decode hotspot tables; α stayed in [0.588, 0.789] band. | No | `docs/phase-c{51,52,53,54,57}/` |
| C58–C60 #498–#502 | CUDA GDN decode fusion + post-#500 / post-#502 audits | No (bench) | LANDED kernels (gates+gated_norm+qk_norm GQA fast paths). α did not move materially. | No | `docs/phase-c{58,59,60}/` |
| C62 #506-class | GDN prefill memory preflight | No (bench) | Activation-bound at ~33 GiB peak at 64k; streaming prefill path validated | No | `docs/phase-c62/` |
| C63 #511 | CUDA streaming prefill default ≥65k | No (bench) | LANDED (not an MTP-axis change) | No | `docs/phase-c63/` |
| C64 #524 | Post-#521 fused-kernel kill-switch bisection (perf, not α) | No (bench) | NULL — no single fused kernel default path explains the 7.9% post-#521 decode regression vs post-#166 baseline | No | `docs/phase-c64/post523-killswitch-bisection.md` |

**Net inventory: 34 MTP-named PRs, 1 fix landed that moved α (C18: +0.153), 1 anchor change that re-classified the workload baseline (C35: +0.43 absolute via chat-template flag), and ~20 doc-only static audits that ruled out individual hypotheses without α movement.** No PR after #382 has moved the α median by >0.02.

## 2. Current α gap

| Metric | Current value | Target | Gap | Source |
| --- | ---: | ---: | ---: | --- |
| Median α (C40f anchor, N=20, A6000, chat-template + W4A16 + FP32 argmax + humaneval) | **0.689** | **0.720** (Qwen3.5 paper floor); **0.80–0.85** stretch (DeepSeek-V3 / Qwen3-Next published k=1 MTP rates) | **−0.031** to floor; **−0.111 to −0.161** to stretch | `docs/phase-c40f/summary.json` |
| 95% CI of median α | `[0.652, 0.723]` | floor inside interval would clear | floor 0.72 is at the edge of the upper CI | same |
| Seeds ≥ 0.72 (out of N=20) | **6 / 20 = 30%** | "most seeds" (>50%) for shipping confidence | -20 percentage points | same |
| Median decode tok/s (C40f anchor) | **38.25** | **49.76 × 1.10 = 54.7** to clear +10% MTP-on vs plain-decode floor on the same hardware | **−16.5 tok/s** (-30%) | C40f vs PR #166 baseline |
| MTP single-seed decode (post-#481, A40 fallback, single seed-1) | 41.2 tok/s at α=0.730 | match plain-decode 49.76 tok/s baseline | -8.6 tok/s | PROFILING.md §"Phase 6 post-#481 current-main profile refresh" |
| Post-#500 / C60 single seed | 23.5 tok/s at α=0.716 (Nsight overhead included) / unprofiled 36.5 tok/s | n/a (profile-only) | n/a | `docs/phase-c60/summary.json` |
| Identity-bias rate on reject rows (C36 post-#417, 3 seeds, default workload) | **80.85%** | "expected baseline" unknown — not measured on a healthy MTP impl. Hypothesis: <40% on a clean run. | unknown — diagnostic gap | `docs/phase-c36/c36-identity-bias.md` |
| Identity-bias rate on accept rows (C36) | **0%** | 0% trivially | n/a (consistent) | same |
| Seed split (C36, default workload, 3 seeds) | seed 0 α=0.74; seeds 1+2 α=0.31, 0.40 | uniform >0.6 | bimodal regime split | same |

**Note on the seed split:** the C36 default-workload run is a different harness shape than the C40f anchor (C40f forces chat-template, FP32 argmax, humaneval subset, temp=0). Some of the seed split disappears under C40f flags but does not vanish: the C40f N=20 distribution stdev is 0.052 with min=0.628 and max=0.789, so the worst seed under C40f is still 0.628 (well below floor) and the best is 0.789 (above floor). The split is harness-attenuated, not eliminated.

**Gap framing for §4 below:** the "30% of seeds clear floor" data point is the most actionable signal in the entire C-series. It says the head IS capable of paper-floor performance on this build and this workload, but is not consistently capable. Whatever causes the regime split is therefore the single largest remaining lever.

## 3. Cross-referenced agent notes

These notes are load-bearing for any next planning cycle on this axis. Each one is verified to exist via `ce notes-search --q <topic>` at audit time.

- `kiln-speculative-decoding-design` — original PR #240 preflight: math ceiling 1.53–1.57× over 49.76 tok/s baseline at α 0.80–0.85, k=1. **Skip-layer self-spec is fallback only on Qwen3.5-4B.**
- `kiln-spec-decode-gdn-rollback` — primary implementation risk for kiln spec-decode is GDN recurrent-state rollback on rejected tokens; vLLM's qwen3_next_mtp does not handle this. **Closed by C58 GDN snapshot/restore landed in PR #498.**
- `kiln-mtp-alpha-regression-qwen35-4b` — PR #257 first observation of α=0.411 on k=1 MTP. Investigation order recommended: (1) head-input tensor ordering, (2) fc/pre_fc_norm parity, (3) sampler parity. All three (1) (2) (3) have since been ruled out.
- `kiln-mtp-c5-bench-results` — α median 0.124 / decode -25.1% post-C3 RoPE fix. Class B = 87.6%. **Aligns with current state-of-play: head predicts wrong top-1, not a verifier bug.**
- `kiln-mtp-c12-fp32-head-negative` — PRIMARY-NEGATIVE: fp32 upcast costs 8.3% tok/s with zero α benefit. **Do not retry precision-axis hypotheses.**
- `mtp-bench-workload-sensitivity` — chat-template raises α 3-4× over prose. Always default the reference α measurement to chat-templated prompts. **Already absorbed into the C40f anchor.**
- `mtp-identity-bias-diagnostic` — identity-bias signature (`draft_top1 == last_token`) at 46% in Phase B trace. C36 re-measure showed 80.85% on reject rows. **Recommended next bisect targets in note: (1) mtp.fc embed-half dominance via L2 of the two 2560-dim halves, (2) h_prev source mis-selection, (3) RoPE off-by-one.** (3) was ruled out by C24; (1) and (2) are open.
- `mtp-acceptance-attribution-pattern` — KILN_C1_ATTR_PATH per-step CSV pattern. **Reusable for the recommended next H.** Cost: ~60 min A6000 / ~$0.20 to produce one fresh seed sweep.
- `mtp-tap-level-vs-end-to-end-metrics` — tap-level fixes can validate without moving α; always report both. **Direct caution against single-A/B α conclusions.**
- `kiln-mtp-spec-scaffolding-vs-full-impl` — historical lesson on shipping scaffolding instead of full impl. **Not directly relevant — current state is overshooting on diagnostics, not under-building.**
- `kiln-mtp-reference-source-of-truth` — vLLM has two MTP position conventions (`mtp`/`draft_model`/`DFlash` use query_pos=N; `eagle`/`eagle3` use position=N-1). **Already applied in C24 hardware A/B — both conventions test the same on kiln's anchor.**
- `kiln-mtp-rope-position-threading` — pattern documented at PR #314 / C3.
- `kiln-mtp-sdpa-kv-len-divergence` — pattern documented at PR #319 / C8.
- `kiln-mtp-verify-gdn-state-gap` — pattern documented at PR #257 / C58.
- `kiln-mtp-harness-parity-c49` — C49 verdict that C48 regression was harness comparability, not model math. **Anchored everything since on C40f-style flags.**
- `kiln-mtp-single-prompt-ab-misleading` — N≥8 multi-prompt paired-difference required before canonical rewires. **Anchored C37/C40 sweep design.**
- `kiln-phase-handoff-claims-need-reverification` — direct caution against repeating ruled-out hypotheses. **Applied below in §4.**
- `kiln-mtp-audit-vllm-config-cross-ref` — vLLM + Qwen3.5-4B config.json cross-ref pattern.

## 4. Remaining-unknowns scoring

Each remaining hypothesis is scored on three axes: probability mass (how much of the residual α gap could it explain), falsifiability cost, and kill-switch availability (can we A/B it without committing inference-math changes). Probability is **subjective best-estimate based on the evidence trail above** — it is NOT a measured probability.

| H | Hypothesis | Probability mass | Falsifiability cost | Kill-switch available? | Score | Notes |
| --- | --- | :---: | --- | :---: | :---: | --- |
| H15a | **C36 seed-split is a state carryover bug across the Marlin batch pack / model-load boundary** (cold-load drift). 6 of 20 seeds clearing floor would correspond to roughly 1-in-3 model loads catching the "good" Marlin pack state. | 25% | $0 doc-only re-read of the Marlin pack code path + correlate C40f bench-seed JSON's `model_load_secs` vs α. If model_load_secs distribution correlates with α, that's a smoking gun. | Yes — `KILN_MARLIN_DETERMINISTIC` does not exist; could ship one + rerun C40f. | **High** | The C40f summary already has `model_load_secs` per row; the correlation analysis is purely Python on the existing artifact. Cost: 0 GPU-min. |
| H15b | **Verifier-side numerical drift on Class B reject rows.** C29's top-K Jaccard verdict only measured accepted-token positions (h_main was reference-echoed). Repeat C29 stratified by accept/reject; the reject side may have cos_sim < 0.99 and explain why specifically these tokens get rejected. | 25% | ~30 min A6000 / ~$0.25 — re-run the C29 dump pipeline with KILN_C1_ATTR_PATH set, then post-process to stratify Jaccard by accepted/rejected. | No new env var needed — C29 instrumentation already exists; only the post-processing is new. | **High** | Directly addresses C29's caveat. Inverts the current verdict from "head logits clean" to "head logits clean ON ACCEPT, unknown on REJECT." Cheap and produces falsifiable numbers. |
| H15c | **`mtp.fc` embed-half dominates h_prev-half on identity-biased reject rows.** Per `mtp-identity-bias-diagnostic` note: split the 2×2560 fc input concat into [embed‖h_prev] and L2-norm each half on identity-biased vs non-identity-biased reject rows. If embed-half norm is 5–10× h_prev-half on identity rows and ~equal on non-identity, the head is dominated by the input embedding and re-emits the last accepted token. | 20% | ~45 min A6000 / ~$0.40 — add a one-line tap inside `mtp_forward_step` that captures the two halves' L2 alongside the existing C1 attribution row. | Tap behind a new env var; default off. | **Medium** | This is the original B-trace recommendation (note dated PR #261) that was deferred when C19/C20 ruled out the obvious mtp.fc parity bugs. C19 confirmed weights match; this hypothesis is about *runtime* tensor magnitudes, not weight loading. |
| H15d | **MTP head was trained on a slightly different tokenizer chat-template surface than what kiln's `--chat-template` flag emits.** Even the C35 chat-template anchor may be feeding tokens the head has never seen during training. | 15% | ~30 min CPU — diff kiln's `tokenizer.apply_chat_template` output against the canonical Qwen3.5-4B chat-template fixture (look for `<|im_start|>` / `<|im_end|>` placement, system prompt presence, BOS/EOS handling). $0 pod spend. | Yes — kiln already supports `--chat-template` flag; could add `--chat-template-style=qwen` to switch. | **Medium** | Static doc-only diff is feasible. Risk: even a perfect chat-template match would only close the ~0.11 gap to 0.80 if H15d is the dominant residual. |
| H15e | **The α gap simply reflects an unfine-tuned MTP head ceiling and there is no inference-side bug left to find.** Qwen3.5-4B's published MTP head was trained against a draft-and-verify rejection sampler that may have learned to over-predict in a way that maximizes the HF reference-impl's token distribution. Kiln's verifier is bit-equal but the head's training distribution may not match kiln's full-precision base-model output. C33 already documented kiln has no MTP fine-tune path. | 15% | $0 — it's a logical claim. Falsifying it requires either implementing MTP head fine-tune (out of scope) or finding evidence in vLLM/SGLang that *they* hit α=0.80+ on the same checkpoint without fine-tune. | n/a | **Low-medium** | Closing path: build a vLLM α microbench on Qwen3.5-4B + chat-template + humaneval and compare to kiln's 0.689. If vLLM also lands at 0.69, this is the answer. ~60 min CPU + ~30 min vLLM install. |

### Recommended next H (single, with concrete bench plan)

**H15b — Verifier-side numerical drift on Class B reject rows (re-run C29 stratified by accept/reject).**

Rationale:

1. Highest probability mass tied with H15a (25%), but H15b is fully self-contained: it produces hard numbers (cos_sim, Jaccard@K, max_abs_delta) for reject rows that have never been measured. H15a is a correlation analysis on existing artifacts and should be done in parallel for $0 (see "free pre-step" below).
2. Directly closes the explicit C29 caveat documented in §1: top-K Jaccard was measured only on accepted positions because the HF reference dump echoes kiln's h_main. Stratifying the same dump by accept/reject does NOT require a new reference build — the existing `scripts/mtp_h_main_reference_dump.py` already produces paired tensors.
3. Either result is informative:
   - **If reject-row cos_sim ≥ 0.999:** the head's logits at the reject position match the verifier — the reject is a tie-break or a vocab-distribution drift the verifier sees correctly. This pivots toward H15e (head ceiling) and lets us close the project goal with a "kiln-native ceiling" verdict.
   - **If reject-row cos_sim < 0.99:** there IS verifier-side drift, and the next bisect can target the specific layer where it appears. This re-opens the inference-path investigation with new evidence.
4. Bench cost is low: 1 A6000 pool lease, ~30 min wall-clock, $0.25 estimated. The fitting next-PR shape is ONE bench task with a tight remaining-work precondition (current C29 cos_sim numbers are still in-tree as the prior baseline).

#### Concrete bench plan for the next planning cycle to queue

**Branch:** `phase7-mtp-verifier-reject-stratified-c29-v2`

**Files / taps to touch:**

- `scripts/mtp_c1_summarize.py` — extend the C1 attribution summarizer to take an additional `--stratified-jaccard-pkl <path>` argument that reads the C29-style pickle of `(kiln_logits, ref_logits, accept_flag)` triples and emits Jaccard@5/@10/@20 + cos_sim + max_abs_delta tables stratified by `accept_flag`.
- `scripts/mtp_compare.py` (existing) — add a `--c29-v2` mode that emits the pickle alongside the existing CSV.
- `crates/kiln-model/src/c1_attr.rs` (existing) — already emits `accepted` per row; verify the column survives a C29-style replay.

**Bench commands (one A6000 pool lease, no harness changes from C40f):**

```bash
# 3 seeds, C40f-style anchor, with both C1 attribution and MTP head logit dump
ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000'
# ... usual setup ...
for seed in 0 1 2; do
  KILN_SPEC_METHOD=mtp \
  KILN_BENCH_FORCE_MTP=1 \
  KILN_MTP_ARGMAX_FP32=1 \
  KILN_W4A16=1 \
  KILN_CUDA_GRAPHS=true \
  KILN_C1_ATTR_PATH=docs/phase7-mtp-c29-v2/c1_seed${seed}.csv \
  KILN_MTP_DUMP_PATH='docs/phase7-mtp-c29-v2/seed-'${seed}'/step-{step}.safetensors' \
  KILN_MTP_DUMP_HEAD_LOGITS=1 \
  ./target/release/kiln-bench --model-path /workspace/qwen3.5-4b \
    --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --seed $seed \
    --chat-template --latency-only --prompt-subset humaneval --temperature 0.0
done

python3 scripts/mtp_h_main_reference_dump.py \
  --checkpoint /workspace/qwen3.5-4b \
  --kiln-dump 'docs/phase7-mtp-c29-v2/seed-{seed}/step-{step}.safetensors' \
  --out docs/phase7-mtp-c29-v2/ref.safetensors \
  --c29-v2 --seeds 0,1,2

python3 scripts/mtp_compare.py --c29-v2 \
  --kiln 'docs/phase7-mtp-c29-v2/seed-{seed}/step-{step}.safetensors' \
  --ref docs/phase7-mtp-c29-v2/ref.safetensors \
  --c1-csv 'docs/phase7-mtp-c29-v2/c1_seed{seed}.csv' \
  > docs/phase7-mtp-c29-v2/stratified-jaccard.txt
```

**Decision rule (in the next bench's task brief):**

- If reject-row cos_sim median ≥ 0.999 AND Jaccard@5 median ≥ 0.95 → close project goal as kiln-native ceiling, escalate to H15e (compare against vLLM).
- If reject-row cos_sim median ≥ 0.99 AND < 0.999 → ship the result, queue layer-by-layer h_main bisect on reject rows only.
- If reject-row cos_sim median < 0.99 → smoking gun; queue a focused per-layer bisect.

**Free pre-step (do BEFORE GPU spend):**

Before queueing the bench task, do a $0 doc-only correlation analysis on `docs/phase-c40f/summary.json`'s `rows[].model_load_secs` vs `rows[].acceptance_rate` (H15a). This is 5 minutes of Python — Spearman + scatter plot. If the correlation is significant (|ρ| > 0.5), the next H becomes "MTP cold-load determinism" instead of H15b, and the bench plan changes accordingly.

### What NOT to queue

Per `kiln-phase-handoff-claims-need-reverification` and the kernel-vendor-precondition-check pattern, the next planning cycle MUST NOT queue any of:

- ❌ Re-test of C23 sampler drift (H5) — DISPROVEN; `_params` is underscore-prefixed in source.
- ❌ Re-test of C27 H8 positional / MTP-block sequencing — RULED OUT 5/5 sub-properties.
- ❌ Re-test of C12 fp32 draft head — PRIMARY-NEGATIVE; costs 8.3% tok/s for 0 α benefit.
- ❌ Re-test of C20 MTP-block dual-norm wiring — DISPROVEN.
- ❌ Re-test of C19 mtp.fc_norm reuse — DISPROVEN at structural level (Qwen3.5-4B has no `mtp.fc_norm.weight`).
- ❌ Re-test of C24 RoPE abs_pos hardware A/B — REJECTED with hardware data.
- ❌ Vendoring a vLLM `qwen3_next_mtp` port — kiln's MTP IS already implemented and structurally matches the contract; the gap is α, not feature presence.
- ❌ "MTP head fine-tune" plan — requires implementing kiln-train integration that does not exist (C33 / H12 N/A); out of scope until α is shippable.

### Fallback if H15b is also null

If H15b returns "reject-row cos_sim ≥ 0.999," the recommended fallback is a **doc-only vLLM α microbench comparison**:

1. Set up vLLM 0.6.x with Qwen3.5-4B locally (~30 min CPU).
2. Run vLLM with the same chat-template + humaneval prompt + greedy + k=1 MTP for 20 seeds.
3. If vLLM α median is also in the [0.65, 0.70] band, accept α=0.689 as the kiln-native ceiling for this checkpoint and pivot the project goal to "speed-only" (FlashInfer / FP8 KV / prefix cache for the verifier path).
4. If vLLM α median ≥ 0.78 on the same prompt, the gap is verifier-side and a fresh per-layer bisect on reject rows is justified.

## 5. Anti-duplication evidence

Verified at audit time:

| Search | Result |
| --- | --- |
| `gh pr list -R ericflo/kiln --state all --search "mtp acceptance state-of-play" --limit 5` | `[]` (empty) |
| `gh pr list -R ericflo/kiln --state all --search "mtp audit consolidated" --limit 5` | `[]` (empty) |
| `gh pr list -R ericflo/kiln --state open --search "mtp" --limit 20` | `[]` (no open MTP PRs) |
| `gh pr list -R ericflo/kiln --state open --limit 10` | `[]` (no open PRs at all at audit start) |
| Recent MTP-named PRs | last MTP-named was #382 merged 2026-04-22; nothing since has the H-hypothesis or state-of-play shape |
| Existing phase7 docs | `docs/phase7-prefix-cache-reuse-ab.md` (PR #517), `docs/phase7-sglang-radix-audit.md` (PR #526) — neither covers MTP |

No prior PR or open task overlaps the consolidated audit shape proposed by this doc. The doc-only redirect pattern (precedent: PR #525 vLLM GDN audit, PR #526 SGLang radix audit) is the right shape for this PR.

## 6. References

### Phase B / C primary docs (in order of citation)

- `docs/MTP_PHASE_B12.md` — Phase B12 verdict (layer-31 drift benign)
- `docs/phase-c1/c1-mtp-acceptance-attribution.md` — C1 verdict (Class B 87.6% head bug)
- `docs/phase-c5/c5-bench-report.md` — C5 ship-floor MISS verdict
- `docs/phase-c8/` through `docs/phase-c14/` — pre-C18 splice / SDPA / weight-loading bisect chain
- `docs/phase-c15/c15-h-main-drift-verdict.md` — 2.0–2.4× kiln/HF magnitude smoking gun
- `docs/phase-c16/c16-mtp-accept-reject-audit.md` — H1–H4 all REJECTED
- `docs/phase-c17/c17-h-prev-reference-frame-audit.md` — h_prev pre-norm vs post-norm verdict
- `docs/phase-c18/c18-h-prev-post-norm-fix.md` — α 0.000 → 0.153 (4.7×)
- `docs/phase-c{19..32}/` — H1–H11 + Class B head trio + h_main fp32 parity static audits
- `docs/phase-c33/c33-mtp-finetune-static-audit.md` — H12 N/A (no fine-tune path)
- `docs/phase-c34/c34-sampler-parity-audit.md` — H13 INCONCLUSIVE on static
- `docs/phase-c35/` — H13 4-cell A/B; chat-template +234% α
- `docs/phase-c36/c36-identity-bias.md` — post-#417 identity-bias re-measure (80.85% / 0%, seed split)
- `docs/phase-c36/c36-h14a-decode-length-sweep.md` — H14a NULL
- `docs/phase-c37/` — N=10 variance re-anchor at C35 Cell D
- `docs/phase-c38/c38-post422-seed1-upstream-bisect.md` — seed1 upstream split
- `docs/phase-c{39..47}/` — layer-1 input-layernorm row-broadcast bisect → C47 OUT-OF-SCOPE classification
- `docs/phase-c40f/summary.json` — **canonical N=20 α distribution anchor**
- `docs/phase-c{48,49,50}/` — harness-parity verdict
- `docs/phase-c{51,52,53,54,57}/` — post-#466/#476/#481 profile refreshes
- `docs/phase-c{58,59,60}/` — CUDA GDN decode fusion audits
- `docs/phase-c64/post523-killswitch-bisection.md` — perf-axis NULL kill-switch bisection

### PROFILING.md anchors

- §"Native MTP speculative decoding — preflight 2026-04-20" (PR #240)
- §"Phase 6 post-#481 current-main profile refresh" — single-seed A40 native-MTP at α=0.730 / 41.2 tok/s
- §"Phase 6 post-#415 current-main native MTP A/B refresh" — single-seed A6000 native MTP at α=0.245 / 27.67 tok/s (default-flags / pre-C40f-anchor)
- §"Phase 6 C50 C40f-style native MTP decode profile" — restored anchor on current main
- §"Post-PR-#500 current-main profile refresh" — α=0.716 single-seed at C60

### Cited vLLM source (per C17/C25/C28 access dates)

- `vllm/model_executor/models/qwen3_next_mtp.py::Qwen3NextMultiTokenPredictor`
- `vllm/v1/sample/rejection_sampler.py::rejection_sample` `all_greedy` branch
- `vllm/v1/spec_decode/eagle.py::_greedy_sample`

### Agent notes (verified to exist via `ce notes-search --q <topic>`)

`kiln-speculative-decoding-design`, `kiln-spec-decode-gdn-rollback`, `kiln-mtp-alpha-regression-qwen35-4b`, `kiln-mtp-c5-bench-results`, `kiln-mtp-c12-fp32-head-negative`, `mtp-bench-workload-sensitivity`, `mtp-identity-bias-diagnostic`, `mtp-acceptance-attribution-pattern`, `mtp-tap-level-vs-end-to-end-metrics`, `kiln-mtp-spec-scaffolding-vs-full-impl`, `kiln-mtp-reference-source-of-truth`, `kiln-mtp-rope-position-threading`, `kiln-mtp-sdpa-kv-len-divergence`, `kiln-mtp-verify-gdn-state-gap`, `kiln-mtp-harness-parity-c49`, `kiln-mtp-single-prompt-ab-misleading`, `kiln-phase-handoff-claims-need-reverification`, `kiln-mtp-audit-vllm-config-cross-ref`, `kiln-mtp-dump-parent-dirs`, `kiln-mtp-dump-generation-gotchas`, `mtp-multi-position-bisect-pattern`, `mtp-dump-compare-first-divergence-gotcha`, `kiln-mtp-b10-layer-0-divergence`, `kiln-mtp-ref-dump-tap-exclusivity`.
