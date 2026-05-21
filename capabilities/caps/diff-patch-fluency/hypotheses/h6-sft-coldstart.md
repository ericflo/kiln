# Hypothesis: h6-sft-coldstart

## Family
H6 — off-policy SFT cold-start. Bypass OPD's high-baseline rollout-
overlap problem by directly SFT-training on 27B teacher rollouts.

## Claim
SFT on 24 teacher-generated diffs lifts cap #5 composite above baseline
(0.85). The 27B is better at diffs than the 4B; teaching the 4B to
imitate teacher outputs should propagate that lift.

## Mechanism
Generate teacher diffs via vLLM with enable_thinking=False (the 27B
is a thinking model; we want bare answers, not reasoning preambles).
For each of 24 training prompts, the teacher emits a diff. Use those
as SFT (prompt, target) pairs. Train the 4B's LoRA on cross-entropy
over the assistant tokens. Standard SFT — no rollouts, no teacher
logprobs query per step. Bypasses the OPD overlap trap entirely.

## Configuration
- Teacher: vllm/qwen3.6-27b-awq with enable_thinking=False
- Training data: 24 (prompt, teacher_diff) pairs
- Trainer: kiln cuda_sft_file --trainer generic
- 2 epochs (initial) then 4 epochs (extended)
- LoRA rank: kiln default
- checkpoint_interval: 25

## What's held constant from previous iters
Same 24 training prompts as iter 1 and iter 2.

## Falsification plan
- If composite < baseline (0.85) after 2 epochs: try 4 epochs.
- If composite < baseline after 4 epochs: training set is too small;
  the SFT ceiling on N=24 is below baseline. Document and retire.
- If composite > baseline: ship the adapter; iter 5 = OPD refinement
  on top.

## Inspection plan
Compare baseline vs SFT responses on 4 eval prompts. Look for:
- Are the diffs strictly formatted? (expected yes — SFT learns format)
- Do they apply? (the test)
- Do they capture intent?

---

## Verdict
✗ Falsified. SFT cold-start regressed below baseline at every
checkpoint, including 4-epoch full training:

| Adapter | composite | strict | applies | intent | minimal |
|---------|-----------|--------|---------|--------|---------|
| baseline       | **0.850** | 1.00 | 0.867 | 0.767 | 0.867 |
| 2ep ckpt-25    | 0.640 | 1.00 | 0.600 | 0.600 | 0.600 |
| 2ep final      | 0.790 | 1.00 | 0.767 | 0.767 | 0.767 |
| 4ep ckpt-24    | 0.760 | 1.00 | 0.733 | 0.733 | 0.733 |
| 4ep ckpt-48    | 0.790 | 1.00 | 0.767 | 0.767 | 0.767 |
| 4ep ckpt-72    | 0.790 | 1.00 | 0.767 | 0.767 | 0.767 |
| 4ep final      | 0.788 | 1.00 | 0.767 | 0.767 | 0.756 |

Training plateaued at 0.79 across 4 epochs. More epochs don't lift
past this ceiling — the model has memorized the 24 teacher diffs
and is now stuck on whatever generalization that memorization
permits. SFT loss went from 5 (cold) to 0.005 (final), confirming
near-perfect memorization. The eval-vs-train gap of 0.79 - 0.85 =
-6pp is the surface-form-transfer cost.

Inspection: SFT outputs are clean bare diffs (strict_format = 1.0
across all checkpoints) but make small structural errors — wrong
`@@` line counts (e.g. `-1,5 +1,5` when file has 4 lines), incorrect
content swaps (delete A, B; add A, B with same lines back). These
are the teacher's own occasional errors AMPLIFIED by SFT-memorization
on the 24 examples + small corpus.

The deeper finding: **for high-baseline capabilities, small SFT
corpora hurt generalization.** The 4B's baseline diff capability is
~85% across all eval prompts. SFT on 24 specific (prompt, target)
pairs deepens the model's response to those specific surface forms,
at the cost of generalization to held-out forms. The eval drops
from 0.85 → 0.79 even with perfect format adherence.

What would unstick this: a teacher SFT corpus of 100+ diverse prompts.
At that data scale, the model would learn the GENERAL pattern of
"emit correct diff for any edit request" rather than memorizing 24
specific cases. Out of scope for this session — requires corpus
design + curation investment.

**Final disposition for cap #5: ship the base model.** The 4B at
composite 0.85 IS the cap #5 adapter. No LoRA from OPD (regressed)
or H6 SFT (also regressed). Both attempts produced rich findings
documented in the skill but no improvement over baseline.

Three additive findings from cap #5:
1. Eval-design failure goes both directions (cap #1 + cap #5)
2. OPD has a high-baseline failure mode (iter 1 + iter 2)
3. SFT cold-start needs sufficient data to generalize (iter 3 H6)
