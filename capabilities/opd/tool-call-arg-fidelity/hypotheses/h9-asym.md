# Hypothesis: h9-asym

## Family
H9 — Teacher-side conditioning (asymmetric prompts). First real user of
`OpdPrompt.teacher_extra_messages` shipped in this session.

## Claim
Adding 3 pristine tool-call exemplars to `teacher_extra_messages` per
prompt — visible only to the teacher — will:
1. Recover iter 1's regressions on saturated sub-scores: `parses` ≥
   0.98 (was 0.933) and `type_correctness` ≥ 0.98 (was 0.933).
2. Hold or improve `required_fields` ≥ 0.87 (was 0.867).
3. Composite ≥ 0.92 (was 0.8939) — a +2.6pp improvement on top of
   iter 1's +4.3pp, captured because we recover the saturated
   sub-scores without giving up the target.
4. (Soft probe) Reduce skip rate below iter 1's 93.6% — sharper teacher
   distributions on tool-call SHAPE tokens may help the kernel's
   alignment.

## Mechanism
The teacher's distribution at each rollout position is now
`P_T(· | exemplars, prompt, rollout_<i)`. The exemplars are 3 perfect
tool calls for the same tool shape. They:
- Push teacher mass onto `{` opening, JSON-key tokens (the right keys),
  string-value tokens, `}` closing — the SHAPE.
- Push teacher mass off "I'll call file_read like so:" and similar
  prose openers (which the 4B sometimes produces).
- Don't directly teach the SPECIFIC user request's content (the
  exemplars are different specific cases), so the gradient flows on
  shape not memorisation.

Iter 1's adapter learned "include more keys" but occasionally
overshot into emitting tool-RESULT-shaped JSON (e.g. `{"files":[...]}`
for a list_files call). With pristine call-shaped exemplars on the
teacher side, the teacher's distribution will NOT favor `"files"` as
a key at position 1 of a list_files response — it'll favor
`"directory"`. The student gets pulled away from result-shape and
toward call-shape.

## Configuration
- prompts: prompts/h9-asym.jsonl (26 prompts, each with
  `teacher_extra_messages` carrying 3 exemplars for its tool shape)
- rank: 16  alpha: 32  lr: 1e-4
- epochs: 6  samples_per_prompt: 1  max_tokens: 384  top_k: 8
- env: KILN_STREAMING_PREFILL=1
- avg teacher prefix length: ~540 chars (~150 tokens)

## What's held constant from previous iteration
EVERYTHING except `teacher_extra_messages` (was empty, now 3 exemplars).
Same prompts, rank, lr, epochs, spp, top_k, max_tokens. Single-knob
change to isolate the effect of asymmetric teacher conditioning.

## Falsification plan
- If parses Δ ≤ 0 AND type_correctness Δ ≤ 0: asymmetric prompts did
  NOT help the regression. Two interpretations:
  - Mechanism wrong: 3 exemplars aren't enough; or exemplars don't
    cover the failure modes the adapter overshot into. Pivot to H9 v2
    with 5 exemplars and ANTI-pattern call-outs explicitly listing the
    bad shapes ({"files":...}, "I'll call ...").
  - Implementation wrong: my plumbing has a position-alignment bug.
    Pivot to debugging (sample 5 responses, log raw token sequences).
- If required_fields drops by >2pp: exemplars are over-narrowing the
  teacher's distribution and the model is regressing on key recall.
  Pivot to fewer exemplars (1-2) or omit the "Use these to inform..."
  instruction.
- If composite < 0.5 × baseline (< 0.425): all-zeros gate fires; treat
  as broken adapter, write failure_mode.md.
- If skip rate falls below 80%: independent finding — H9 helps the
  structural floor too.

## Inspection plan
Sample 5 non-eval responses, including the prompts that produced
shape-confused outputs in iter 1 (list_files-*, the one that emitted
`{"files":[...]}`). Check that:
- JSON parses
- Required fields all present
- Types correct
- No result-shape leakage

---

## Verdict
? Inconclusive — training killed at step 61/156 (~39% through, ~2h45
elapsed) without an eval-able adapter (OPD's trainer only saves at run
completion; no intermediate checkpoint). BUT this iter produced THE
major mechanism finding of the session despite producing no LoRA:

- **Skip rate dropped from 93.6% (iter 1) to ~0%** with asymmetric prompts
  active. The first 26 nominal steps in epoch 1 produced 26 effective
  steps. iter 1 hit only 10 effective in 156 nominal across all 6 epochs.
- **Loss landscape qualitatively different.** Iter 1 loss spiked to 30+;
  iter 2 had one early 29.5 spike then settled into a steady 0.12–1.9
  range by step 30, descending to 0.13 by step 55. The sharper teacher
  distribution gives the student a cleaner gradient.
- **Per-step time tripled** (~150s vs iter 1's 24s). Longer teacher
  token sequence (~268-token prefix + prompt + rollout) plus a teacher
  query on every step (vs every ~6th step in iter 1). Net: similar
  elapsed-time-per-effective-step, but iter 2 was on track to produce
  ~150 effective steps vs iter 1's 10 — much more total signal.

Confirms the H9 hypothesis at the **mechanism** level (asymmetric teacher
conditioning resolves the structural-output skip-rate floor) even though
no adapter survives. The skill's first real user of the new
`OpdPrompt.teacher_extra_messages` feature validated the feature's
intent. Adapter quality remains uneval'd.

Sub-scores: N/A.

Best-so-far for this capability stays at iter 1: composite 0.8939 (the
toolcall-h1-r16-6ep adapter).

Next: iter 3 should be sized to FIT in a wall-time budget that lets the
run complete. Options:
- **H9 short (2 epochs, asymmetric)** — ~52 nominal × 150s ≈ 2h
- **H9 small-LR (2 epochs, lr 5e-5, asymmetric)** — even more
  conservative on LoRA capacity

Killing at 61/156 is the symptom; the cure is sizing the experiment to
fit a known wall-time budget, not running to nominal convergence.

