# Hypothesis: numeric-drill-control

(Control / triangulation for iter 1 per §3 Phase 6 step 2 and §15 surprising
patterns: "first kept result is the loudest" — confirm prose was doing
causal work before basing future iters on it.)

## Claim

32 worked **numerical** examples covering the same 7 math domains as
iter 1, with explicit symbolic working in the assistant turn and a
**numerical** final answer, will **not** match iter 1's +0.31 delta.
If iter 2 lifts comparably or more than iter 1, the prose route was a
confound — what lifted the score was just SFT on anything math-shaped,
not prose specifically.

## Mechanism

This is a control ablation. The dataset shape is the exact OPPOSITE of
iter 1: numeric where iter 1 was prose, symbolic where iter 1 was
verbal, with-final-answer where iter 1 had no final answer. If the
score lift is driven by general "the model gets math SFT", both should
lift equally. If the lift is driven by prose-routing specifically,
this should lift less.

The control deliberately uses **the eval's likely surface form** (rows
of numerical problems with terminal numerical answers). That is a
firewall principle violation in spirit but is required for a clean
control — we are testing whether matching the eval's surface form is
what lifts the score, vs. whether teaching the *concept frame* is what
lifts the score.

## Dataset shape

- Size: 32 examples (half iter 1, since this is a control)
- Modality of supervision: numeric + brief symbolic working. Assistant
  turns contain numbers, equations, and a final answer.
- Distribution by math domain (matching iter 1's split, rounded):
  -  5 arithmetic
  -  5 algebra
  -  5 geometry
  -  4 trig
  -  4 systems-of-equations
  -  5 calculus
  -  4 ODE / exponential
- Surface form held OUT: nothing — this control intentionally matches
  the eval's likely surface form.
- System prompt: none.

## Construction recipe

Hand-drafted. Each example is short:

  USER:  "[Numeric math problem in 1-2 sentences]"

  ASSISTANT: "[1-3 sentences of brief working ending with the answer
              stated as a number / expression.]"

## Risk

The main risk is **confounded interpretation**: if iter 2 happens to
match iter 1 closely (within MAD), I cannot distinguish "both routes
work equally" from "prose was uniquely the key and the surface-form
drill happened to also help via different mechanism". A second control
ablation would be needed to disentangle further.

A secondary risk is **answer-form contamination**: by training on
numerical surface form, the adapter will produce numerical surface
form on eval items. If the eval is biased toward numerical exact-match
scoring (likely), this adapter will be "form-friendly" with the eval —
which would inflate its score and bias against the prose hypothesis.

## Falsification plan (committed BEFORE seeing the score)

Let iter 1's adapted score be S1 = 0.903.

- If iter 2 (numeric-control) S2 ≥ S1 - 0.03: prose was not uniquely
  doing the work. SFT on any math-shaped data lifts equally or more.
  Next iter pivots to investigating which subset of math domains drives
  the lift. The prose-route hypothesis is downgraded but not discarded.
- If iter 2 S2 in [S1 - 0.10, S1 - 0.03): prose contributed somewhat
  but not dominantly. Both routes work; prose has a small edge.
  Next iter is a hybrid (`mixed-prose-numeric`) testing additivity.
- If iter 2 S2 in [baseline + 0.05, S1 - 0.10): prose specifically was
  doing significant causal work. Numeric SFT helps but less. Confidence
  in the prose route increases. Next iter is `prose-approach-broad-
  paraphrased` to refine.
- If iter 2 S2 ≤ baseline + 0.05: numeric drill *doesn't help*. Either
  the small dataset (32 vs 64) was undersized, or numeric form clobbers
  some other competence. Re-run with 64 numeric examples
  (`numeric-drill-control-64`) before drawing conclusions.
