# Phase C40 — Per-domain MTP α floors (C38 + C39 synthesis)

## TL;DR

Kiln MTP α at the C35 Cell D anchor is **heterogeneous across token
distributions**, not a single flat deficit. The Qwen3.5 paper floor of
α ≥ 0.72 is reached on math-prose prompts, straddled on English-prose
prompts, and **structurally missed on Python-code prompts** — with the
HumanEval 95% CI sitting entirely below the floor after C39 doubled the
code-prompt sample size without moving the center.

- **GSM8K (math prose):** median α **0.7887**, CI [0.7639, 0.8551] —
  **clears** the paper floor. 9/10 seeds above 0.72.
- **C4 (English prose):** median α **0.7162**, CI [0.6885, 0.7823] —
  **straddles** the floor. 4/10 seeds above 0.72.
- **HumanEval (Python code):** median α **0.6933** (N=20) with CI
  **[0.6602, 0.7162]** — **fails** the floor. Entire CI below 0.72.

The C38 N=30 aggregate CI of [0.6996, 0.7760] that straddled 0.72 is
now understood as the blend of these three regimes. HumanEval drags the
aggregate; GSM8K lifts it; C4 noise does the rest.

## Per-domain floors table

Anchor: C35 Cell D — `KILN_W4A16=1 KILN_MTP_ARGMAX_FP32=1
KILN_SPEC_ENABLED=1 KILN_SPEC_METHOD=mtp KILN_CUDA_GRAPHS=true
--paged --chat-template --prompt-tokens 512 --max-output-tokens 128`.

| Domain | Source phase | N | pool | median α | 95% CI | vs 0.72 | Verdict |
| --- | --- | --- | --- | ---: | --- | --- | --- |
| **GSM8K** (math prose) | C38 | 10 | 10-GSM8K slice | 0.7887 | [0.7639, 0.8551] | CI fully above | **clears** |
| **C4** (English prose) | C38 | 10 | 10-C4 slice | 0.7162 | [0.6885, 0.7823] | straddles | **marginal** |
| **HumanEval** (Python code) | C39 | 20 | 10-HumanEval × 2 | **0.6933** | **[0.6602, 0.7162]** | CI fully below | **fails** |
| HumanEval (prior) | C38 slice | 10 | 10-HumanEval slice | 0.6888 | [0.6494, 0.7162] | CI fully below | fails |
| Aggregate (C38 N=30) | C38 | 30 | 30-domain balanced | 0.7162 | [0.6996, 0.7760] | straddles | **ambiguous** |

Spread between best and worst domain median: **≈ 0.095** (GSM8K 0.789 →
HumanEval 0.693). Domain ordering is stable across phases: GSM8K > C4
> HumanEval.

## Methodology

All α measurements below share the **C35 Cell D** anchor established in
C35 (PR #364) and re-validated in C37 (PR #369):

| Knob | Value |
| --- | --- |
| Quantization | `KILN_W4A16=1` (Marlin W4A16) |
| MTP argmax dtype | `KILN_MTP_ARGMAX_FP32=1` (FP32) |
| Speculative method | `KILN_SPEC_METHOD=mtp`, `KILN_SPEC_ENABLED=1` |
| CUDA graphs | `KILN_CUDA_GRAPHS=true` |
| KV path | `--paged` |
| Prompt framing | `--chat-template` (Qwen ChatML) |
| Prompt tokens | 512 |
| Decode tokens | 128 |
| Sampling | `temperature=0` (greedy) |
| Hardware | RTX A6000 (pool lease) |

Phase-specific methodology:

- **C37 (PR #369):** N=10 at Cell D on the 8-prose prompt pool used
  through Phases B2–C37. Found N=10 α CI was a prompt-variance CI rather
  than a seed-variance CI (seeds 8+ wrapped back to seed 0's prompt).
- **C38 (PR #370):** pool expanded to 30 domain-balanced entries
  (0–9 GSM8K, 10–19 HumanEval, 20–29 C4). N=30 aggregate plus per-domain
  N=10 slices. Only the `PROMPT_POOL` content changed; call site still
  uses `seed % PROMPT_POOL.len()`.
- **C39 (PR #371):** added `--prompt-subset {all,gsm8k,humaneval,c4}`
  CLI flag to `kiln-bench` so a single domain can be swept without
  prose dilution. N=20 on HumanEval (seeds 0..19 over the 10 code
  prompts, each prompt visited twice, keeping seed-to-seed variance a
  pure RNG effect rather than a prompt re-hit artifact).

## Interpretation

**The paper-floor shortfall is a code-distribution effect, not seed
variance.** Three independent pieces of evidence support this:

1. **Sample-size stability.** The C39 N=20 HumanEval median (0.6933)
   is statistically indistinguishable from the C38 N=10 HumanEval slice
   median (0.6888). Doubling the code-prompt sample size tightened the
   CI from [0.6494, 0.7162] to [0.6602, 0.7162] without shifting the
   center — the expected behavior when the underlying population mean
   is genuinely below the target, and the opposite of what regression
   to a higher mean would look like.
2. **Stable domain ordering.** Across C38 slices and the C39 re-bench,
   ordering is GSM8K > C4 > HumanEval with ≈ 0.07–0.09 α spread
   between the best and worst domains. The anchor does not move between
   these measurements — the same W4A16 + FP32-argmax + chat-template +
   paged-decode stack hits 0.789 on math prose and drops to 0.69 on
   Python code.
3. **Not a numerical or kernel issue.** The knobs that control the
   numerical path (FP32 argmax, W4A16, CUDA graphs, paged KV) are held
   constant across all three domain slices. Only the prompt
   distribution changes. This localizes the deficit to the MTP head's
   ability to predict the token distribution being decoded, not to
   anything in the verifier, mask, or sampling path.

**Qwen3.5 paper context.** The published 0.72 floor is an aggregate
number over a heavily natural-language-weighted eval. Applying it as a
single threshold to a code-heavy workload overstates the expected α on
that workload — the empirical distribution from C38/C39 suggests the
paper's aggregate is the blend of a higher prose number and a lower
code number, consistent with what we measure.

## Implication for gating

A single flat α ≥ 0.72 gate is the wrong acceptance criterion for kiln
MTP in code-heavy workloads. After the full C40 sequence, the right
read is:

- **Per-domain floors**, derived from the observed distribution and
  the paper floor:
  - GSM8K-class (math prose) ≥ 0.78
  - C4-class (English prose) ≥ 0.70
  - HumanEval-class (Python code) ≥ 0.65
- **Aggregate floor** conditional on workload mix — e.g. 70% prose /
  30% code would weight toward 0.74, while a pure-code workload cannot
  be held to the paper 0.72.

**Important:** these numbers are floors on what kiln **currently**
delivers. They are not a claim that 0.72 is unreachable on code; they
are the empirical baseline against which any future intervention should
be compared. C40's completed follow-ups narrowed the interpretation:

## Post-C40f synthesis

1. **C40a falsified the easiest framing explanation.** Removing
   `--chat-template` on HumanEval did not recover the gap; it collapsed
   alpha instead, so chat templating is load-bearing for code-domain MTP
   on this bench surface rather than a hindrance.
2. **C40b falsified the easy sampling explanation.** Switching to
   `temperature=0.1` on the same HumanEval anchor lowered the median to
   `0.6558` with CI `[0.6452, 0.6933]`, so greedy decoding is not the
   simple bottleneck and this queue item should not remain presented as
   an obvious pending fix.
3. **C40e established a real paired seed-0 BF16 win.** On the same
   binary, same pod, same flags, flipping only `KILN_W4A16` moved
   HumanEval alpha from `0.6282` to `0.7887`.
4. **C40f showed that C40e result does not generalize distributionally.**
   The BF16 HumanEval N=20 sweep on current main landed at median
   `0.6888`, bootstrap 95% CI `[0.6517, 0.7232]`, which does not
   materially beat the already-landed C39 W4A16 distribution
   (median `0.6933`, CI `[0.6602, 0.7162]`).

## Recommendation

Do **not** treat BF16 as the resolved HumanEval load-mode winner from
the current evidence. The durable conclusion after C40f is narrower:

- the HumanEval gap is **not** explained by "chat-template hurts code";
- BF16 can produce a strong paired single-seed win;
- that win is **not** enough to replace the C39/C40 distributional read.

So this document should retire the old C40a/b/c "open queue" framing
and keep the gate discussion anchored on workload-aware distributions,
not on an unresolved assumption that one single-knob follow-up must
still close the HumanEval gap.

## Data provenance

| Phase | PR | Branch | Commit data path |
| --- | --- | --- | --- |
| C35 | #364 | `ce/mtp-c35-h13-residual-ab` | `docs/archive/phase-c/phase-c35/` |
| C36 | #368 | `ce/mtp-c36-h14a-decode-length-sweep` | `docs/archive/phase-c/phase-c36/` |
| C37 | #369 | `ce/mtp-c37-variance-reanchor` | `docs/archive/phase-c/phase-c37/` |
| C38 | #370 | `ce/mtp-c38-expanded-prompt-pool` | `crates/kiln-server/src/bench.rs` (30-prompt pool) |
| C39 | #371 | `ce/mtp-c39-humaneval-domain` | `PROFILING-MTP-C39.md`, pod path `/workspace/c39-results/seed-{0..19}.json` |
| **C40** | **this PR** | `ce/mtp-c40-per-domain-floors-docs` | docs-only synthesis |

Seeds: C37 used 0..9; C38 used 0..29 (pool-aligned); C39 used 0..19
(HumanEval subset).

Bootstrap CIs throughout: 10,000 resamples on the per-seed median,
rng=12345.

## What this PR does not do

- No bench runs. No pod acquisition. No code, config, or kernel change.
- No benchmark reruns in this document. The conclusions above are a
  synthesis of the already-landed C39/C40a/C40e/C40f evidence.
- No new agent note; existing notes
  (`kiln-mtp-humaneval-code-domain-deficit`,
  `mtp-alpha-variance-is-domain-heterogeneity`,
  `mtp-bench-workload-sensitivity`) already capture the durable
  learnings this synthesis draws on.

## Cross-references

- C35 anchor: PR #364 — `docs/archive/phase-c/phase-c35/c35-h13-residual-ab.md`
- C36 decode-length sweep: PR #368 — `docs/archive/phase-c/phase-c36/c36-h14a-decode-length-sweep.md`
- C37 variance re-anchor: PR #369 — `docs/archive/phase-c/phase-c37/c37-variance-reanchor.md`
- C38 expanded prompt pool: PR #370 — `crates/kiln-server/src/bench.rs`
- C39 HumanEval-only: PR #371 — `PROFILING-MTP-C39.md`
