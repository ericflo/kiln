# Hypothesis: h9-asym-ckpt

## Family
H9 — same asymmetric teacher conditioning as iter 2, now with kiln's
new periodic checkpointing wired into the loop. Tests whether iter 2's
mechanism-confirmed approach (skip rate 93.6%→~0%, clean loss
convergence) produces a high-quality adapter when allowed to complete
OR when interrupted with a recoverable checkpoint.

## Claim
Asymmetric teacher prompts produce an adapter that:
- Composite ≥ 0.92 (≥ +2.6pp vs iter 1's 0.8939)
- parses ≥ 0.97 (recovers from iter 1's 0.933)
- type_correctness ≥ 0.97 (recovers from iter 1's 0.933)
- required_fields ≥ 0.87 (holds iter 1's 0.867; ideally improves)
- no_extra_fields ≥ 0.87 (holds iter 1's 0.869)

Total composite uplift target: ≥ +2.6pp on top of iter 1's +4.30pp.

If wall-time runs out, the most recent --checkpoint-interval 25 save
is eval'd as the de facto iter 3 adapter.

## Mechanism
Same as iter 2 (H9). The pristine tool-call exemplars in
`teacher_extra_messages` make the teacher distribution
near-deterministic on tool-call SHAPE at every rollout position.
Reverse-KL pulls the student toward that sharp distribution, recovering
the shape-confusion failure mode iter 1 exhibited.

## Configuration
- prompts: prompts/h9-asym.jsonl (same as iter 2; 26 prompts with
  teacher_extra_messages)
- rank: 16  alpha: 32  lr: 1e-4
- epochs: 6  samples_per_prompt: 1  max_tokens: 384  top_k: 8
- env: KILN_STREAMING_PREFILL=1
- **NEW: --checkpoint-interval 25** — every 25 effective steps,
  the LoRA is saved as a complete PEFT adapter at
  /tmp/opd-toolcall-h9-asym-ckpt/toolcall-h9-asym-ckpt-checkpoint-<step>/.

## What's held constant from iter 2
EVERYTHING. Same prompts, same hyperparams, same asymmetric mechanism.
Only difference: the trainer now writes intermediate checkpoints.

## Falsification plan
- If final-or-latest-checkpoint composite < 0.89 (worse than iter 1):
  H9 mechanism didn't translate to adapter quality. Either too many
  steps (over-trained on small prompt set) or the asymmetric prefix
  caused a regression. Pivot to H9-short (2 epochs).
- If parses or type_correctness DROP from iter 1: asymmetric prefix
  over-narrowed the teacher distribution. Pivot to fewer exemplars (1
  per tool shape instead of 3).
- If composite ≥ 0.92 (claim): keep. Iter 4 = triangulation on a
  different capability or H4 (rank capacity) on this one.

## Inspection plan
After training (or kill), sample 5 non-eval responses including the
prompts that produced shape-confused outputs in iter 1 (list_files-*).
Verify call-shape (not result-shape), required fields, correct types,
no extras.

---

## Verdict
✗ Falsified on adapter quality, BUT confirmed mechanism + uncovered a
sharper failure mode. All 7 checkpoints evaluated — every single one
underperformed iter 1's symmetric adapter (composite 0.8939).

Eval table (composite | parses | required | type | no_extra):

| Adapter | comp | parses | req | type | no_extra |
|---------|------|--------|-----|------|----------|
| baseline | 0.851 | 1.00 | 0.700 | 0.989 | 0.871 |
| **iter 1 sym** (best) | **0.894** | 0.933 | 0.867 | 0.933 | 0.869 |
| ckpt-25 (ep 1) | 0.446 | 0.517 | 0.367 | 0.533 | 0.436 |
| ckpt-50 (ep 2) | 0.733 | 0.367 | 0.833 | 0.722 | 0.733 |
| ckpt-75 (ep 3) | 0.792 | 0.417 | 0.833 | 0.833 | 0.833 |
| ckpt-100 (ep 4) | 0.665 | 0.350 | 0.700 | 0.700 | 0.700 |
| ckpt-125 (ep 5) | 0.214 | 0.100 | 0.267 | 0.200 | 0.189 |
| ckpt-150 (ep 6) | 0.538 | 0.317 | 0.533 | 0.633 | 0.517 |
| final 156 | 0.343 | 0.183 | 0.367 | 0.350 | 0.367 |

**The failure mode: lost EOS behavior.** Inspection of ckpt-75 (the
best-of-iter-3) shows correct JSON tool calls followed by streams of
`.` and `\n` until max_tokens. The final adapter (step 156) produces
pathological `{"..}{"..}` repetition. The student learned to emit
JSON-shaped tokens correctly but lost the signal that says "stop
after the closing brace."

This is the OPD paper literature's "thinking-pattern mismatch" /
"flawed prefix trap": the asymmetric teacher's distribution — sharply
focused on the 3 pristine exemplars' shape at every rollout position —
weights JSON tokens so heavily that `<|im_end|>` never gets enough
mass for reverse-KL to teach the student to emit it. Over many
effective steps, the student internalises "always emit JSON tokens"
and forgets stopping.

Iter 1's symmetric adapter ESCAPED this because its 10 effective steps
(93.6% skip rate) stopped before the collapse. Iter 3's 156 effective
steps (asymmetric = 0% skip rate, 15× the gradient updates) crashed
through it.

§16 loss-chasing strongly confirmed: loss descended to 0.07 by step
155 while eval composite at the same checkpoint was 0.34 — among the
worst across all 7 checkpoints. The lowest-loss adapter is the WORST
adapter.

Sub-score regressions vs iter 1 (best ckpt-75):
- parses: 0.933 → 0.417 (-51.6pp, MAJOR collapse — the failure mode)
- required_fields: 0.867 → 0.833 (-3.4pp minor)
- type_correctness: 0.933 → 0.833 (-10.0pp major)
- no_extra_fields: 0.869 → 0.833 (-3.6pp minor)

**Best-so-far for this capability stays at iter 1's
`toolcall-h1-r16-6ep`** (composite 0.8939, the symmetric adapter).

The H9 mechanism IS real — skip rate eliminated, loss converges
cleanly — but at 6-epoch dosage on this prompt set + this teacher
prefix design, it over-shoots into EOS collapse. Future iters need:
1. Fewer effective steps (2 epochs not 6), OR
2. Looser teacher prefix (1 exemplar not 3), OR
3. Anti-shortcut sub-score for "produces EOS" / "single JSON object only" —
   the rubric's `parses` uses best-effort extraction which gives partial
   credit to "JSON then garbage." A strict-only-JSON sub-score would
   have made the collapse more visible.

The H9-vs-iter-1 result is, in retrospect, a Goodhart story too:
asymmetric mechanism removed the skip-rate "safety brake" that iter 1
had, exposed the model to 15× more gradient updates, and the resulting
over-training collapsed the EOS behavior the symmetric adapter
preserved by happy accident.

Next: option A — H9-short (2 epochs, ~52 effective steps) with same
3-exemplar prefix; ckpt-75 of THIS iter is roughly the equivalent
training-amount and scored 0.7917. If H9-short produces ~0.79, the
mechanism's ceiling on this capability is below iter 1; retire H9 here.
Option B — retire OPD #4 with iter 1 as the kept result, move to
OPD #5 with the rubric-anti-shortcut lesson applied upfront.

