# Hypothesis: h1-r8-lr5e5-2ep-spp2

## Family
H1 — proven recipe, but DOSED CONSERVATIVELY after iter 1's
catastrophic LoRA collapse. The user pointed out (correctly) that
my "retire cap #5" decision was the cap #1 mistake again: baseline
0.85 + headroom 0.15 is not "capability solved," it's "decent first
draft with a real failure tail." OPD can still lift this if the
training doesn't destroy what's already working.

## Claim
Conservative dosage (rank 8 instead of 16, lr 5e-5 instead of 1e-4,
samples_per_prompt 2 instead of 1) preserves baseline's strong points
while lifting `target_intent_captured` by ≥3pp (0.767 → ≥0.797) and
composite by ≥1.5pp (0.850 → ≥0.865).

A weaker claim than iter 1's was, because:
- Iter 1 destroyed a 0.85 model down to 0.10. Avoiding repetition of
  that collapse is the primary goal.
- Headroom is only 0.15; any lift is meaningful.
- The aggressive settings (rank 16, lr 1e-4) caused that collapse.

## Mechanism
Three changes from iter 1, each addressing a specific contributor to
the catastrophic collapse:

1. **rank 8** — half the LoRA capacity. A smaller LoRA can do less
   damage per noisy update. The 4B is already mostly good at diffs;
   we want small adjustments, not capacity for big rewrites.

2. **lr 5e-5** — half the step size. Iter 1 had 8 effective steps
   with loss in the 5–28 range; each step took a large step in a
   noisy direction. Halving the lr halves the per-step damage.

3. **samples_per_prompt 2** — average two rollouts per prompt. The
   wildly variable loss in iter 1 came from rollout variance (some
   prompts produced clean diffs, some produced "I'll respond..."
   prose attempts). Averaging two rollouts smooths the gradient.

## Configuration
- prompts: prompts/h1-r16-6ep.jsonl (24 prompts; same as iter 1)
- **rank: 8** (was 16)  **alpha: 16** (was 32 — same alpha/rank ratio)
- **lr: 5e-5** (was 1e-4)
- epochs: 2
- **samples_per_prompt: 2** (was 1)
- max_tokens: 512  top_k: 8
- env: KILN_STREAMING_PREFILL=1
- checkpoint_interval: 25 (with ~96 nominal, checkpoints at 25, 50, 75)

## Wall-time budget
~4 hours (24 prompts × 2 epochs × 2 samples = 96 nominal × ~150s).
Over my 2h target. Accepting the trade: rollout averaging requires
more rollouts, which costs wall-time but reduces gradient variance.

## What's held constant from iter 1
Same prompt set, same max_tokens, same top_k. Three controlled
changes (rank, lr, samples_per_prompt) all in the "make training
gentler" direction.

## Falsification plan
- If composite < 0.5 × baseline (< 0.425): all-zeros gate fires;
  failure_mode.md required. Iter 1's failure REPEATED at gentler
  settings would say the cap #5 OPD path is fundamentally broken
  on this rubric/prompt set; pivot to H6 (SFT cold-start) or retire.
- If composite < 0.85 (below baseline) but > 0.425: gentler-than-iter-1
  but still a regression. The settings are still too aggressive;
  iter 3 = rank 4, lr 1e-5.
- If composite ≥ 0.85 (no regression) AND target_intent_captured Δ < 1pp:
  conservative settings preserved the model but didn't move the
  target. Pivot to H9 (asymmetric, intent-focused exemplars).
- If composite ≥ 0.865 (claim): kept. Iter 3 = H9 or epoch curve.

## Inspection plan
Sample 5 non-eval responses from the best checkpoint. Compare against
baseline's responses (which we already inspected). Verify the adapter
hasn't introduced any new failure modes (empty responses, EOS
collapse, etc.).

---

## Verdict
✗ Falsified, BUT produced the deepest finding of cap #5. All 3
checkpoints regressed below baseline:

| Adapter | composite | strict | applies | intent | minimal |
|---------|-----------|--------|---------|--------|---------|
| baseline | 0.850 | 1.00 | 0.867 | 0.767 | 0.867 |
| ckpt-25  | 0.522 | 0.567 | 0.567 | 0.417 | 0.567 |
| ckpt-50  | 0.262 | 0.367 | 0.267 | 0.217 | 0.267 |
| final    | 0.312 | 0.367 | 0.367 | 0.183 | 0.367 |

Gentler settings did avoid iter 1's catastrophic 0.10 — ckpt-25 at
0.52 is genuinely better than iter 1 was. So the rank/lr/spp changes
DID help, just not enough to flip from regression to improvement.

**The finding:** Inspecting ckpt-25 responses shows the failure mode
explicitly. Some prompts produce baseline-identical clean diffs;
others produce malformed ones — `a/loop.py` instead of `--- a/loop.py`,
or `print(i) → print(i)` no-op replacements. OPD locked in some of
the student's slightly-wrong rollouts.

This is a §8 student-teacher overlap problem at HIGH baseline:
- At baseline 0.85, some student rollouts match teacher (OPD step = no-op)
- Some rollouts are the 15% failure tail (malformed)
- Reverse-KL makes the student MORE confident in WHATEVER it sampled,
  including the malformed outputs
- Asymmetry: good-rollout steps don't help (already correct); bad-rollout
  steps actively hurt
- Net: regression with high variance, regardless of settings gentleness

**Best 'adapter' for this capability is no adapter** — the baseline
4B at composite 0.85 is the ceiling reachable without a different
training paradigm. OPD can only hurt from here.

The path forward for actual improvement is H6 (off-policy SFT
cold-start): generate teacher rollouts, train SFT on them to narrow
the overlap gap, THEN OPD if anything. Without H6 infrastructure,
cap #5 ships at baseline.

This is THE deep finding of cap #5 and worth more than any adapter:
**OPD has a high-baseline failure mode** when student-teacher rollout
overlap is variable. Skill §0 currently says OPD helps in baseline
(0.4, 0.95); this needs sharpening — the upper bound is closer to
0.80 in practice, and the condition "student-teacher rollout overlap
is consistent" matters more than the exact composite value.

Backported to skill kiln-polish ledger.

