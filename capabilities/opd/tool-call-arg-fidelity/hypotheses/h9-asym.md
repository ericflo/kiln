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
(filled after eval)
