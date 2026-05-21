# Hypothesis: prose-approach-broad-paraphrased

(F-family per §4 of SKILL.md. Triggered by iter-4 falsification plan:
mild regression in mixed-bigger → retire that direction; build F on iter 1.)

## Claim

64 word-situation prose examples in five distinct voice styles (formal
academic / casual conversational / terse-bulleted / narrative-storytelling
/ novice-asking-expert), covering the same 7 math domains as iter 1
but with deliberately different *situations* and structurally varied
voices, will outperform iter 1 (0.903) — testing whether F-family
framing diversity adds lift on top of T-family content.

We compare both to iter 1 (prose alone, 64 ex, single voice) and to
iter 3 (prose+numeric, 96 ex, mixed). The interesting question is
whether voice diversity alone matches the form-anchor effect that the
numeric portion of iter 3 provided.

## Mechanism

Iter 1 (prose-approach-broad) was written largely in a single
explanatory voice — third-person, expository, semi-formal. The F-family
hypothesis (§4) says: 30 paraphrases of one rule beat 1 statement of
30 rules, but ALSO that voice diversity broadens the basin into which
the model can deposit a frame. If iter 1's lift came from frame-routing
specifically (and not from the particular voice), then voice diversity
should preserve the routing lift while also producing a more robust
output style — possibly reducing the anchor regression seen in iter 1
without needing numeric form-anchors.

If iter 5 > iter 1 with anchor preserved, voice diversity is enough.
If iter 5 ≈ iter 1 with anchor still ~0.94, voice diversity helps
neither math nor anchor.
If iter 5 < iter 1, voice diversity actively hurts (style chaos may
confuse the model's output policy).

## Dataset shape

- Size: 64 examples
- Modality: pure prose, no numerics, no equations (same as iter 1)
- Voice styles distributed roughly evenly:
  - formal academic (~13)
  - casual conversational (~13)
  - terse / bulleted (~13)
  - narrative / metaphorical (~13)
  - novice-explainer / self-teaching (~12)
- Distribution by domain: same 7-domain split as iter 1
- Situations: deliberately DIFFERENT from iter 1's (pizza chef vs baker,
  cabbie vs taxi driver, etc.) so the model isn't seeing the same
  situation twice in any sense
- System prompt: none

## Construction recipe

Hand-drafted directly into `datasets/prose-approach-broad-paraphrased.jsonl`.
Each example chooses one voice style and one situation within one of
the 7 domains, then writes 1-4 sentences of conceptual prose in that
voice. Assistant turn contains no numbers and no equations.

## Risk

Main risk: **voice noise**. Five wildly different voices could confuse
the model's output style preference rather than broaden it. If the
model learns "respond in many voices" rather than "recognise concepts
across many framings", the eval score might not move or might drop.

Secondary risk: **lost content depth**. Some paraphrased examples are
notably shorter (1-2 sentences) than iter 1's typical 4-5 sentences.
If iter 1's lift came partly from extended exposition, the terse
examples here may dilute it.

## Falsification plan (committed BEFORE seeing the score)

- S5 ≥ 0.93 AND ANCHOR5 ≥ 0.98: voice diversity alone matches the
  full iter-3 effect (math lift + no anchor regression). Strong result.
  Next iter is `mixed-paraphrased-numeric` to test if combining
  F-diversity and numeric anchors stacks further.
- S5 ≥ 0.93 AND ANCHOR5 < 0.98: voice diversity helps math but not
  anchor. Next iter is `prose-paraphrased-rank2` to chase the anchor
  fix at lower rank.
- S5 in [0.88, 0.93): voice diversity gives partial lift. Iter 3
  (with numeric anchors) remains the winner. Next iter pivots to
  trying a different T-shape: `prose-mistake-named`.
- S5 in [0.80, 0.88): voice diversity reverses some of iter-1's gain.
  Voice chaos hypothesis confirmed. Retire F-family; next iter is
  `prose-mistake-named`.
- S5 < 0.80: severe regression. Voice diversity is actively bad. Note
  in dead ends; next iter pivots to a completely different family
  (M-family: `algorithm-prose-by-domain`).
