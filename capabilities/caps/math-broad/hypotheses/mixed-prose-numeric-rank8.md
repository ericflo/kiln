# Hypothesis: mixed-prose-numeric-rank8

(H-family per §4. Same iter-3 dataset, rank 8 instead of 4.)

## Claim
Raising LoRA rank from 4 to 8 on iter 3's winning dataset will lift
the math score by at least +0.01. Tests whether iter-3 was bandwidth-
limited by rank 4. If yes, higher rank should unlock more lift.

## Mechanism
More rank = more parameters in the adapter = more capacity to learn
patterns. If iter 3 was bottlenecked at rank 4, rank 8 should help.
§15 caveat: "Bigger data wins less than you expect at rank 4" implies
rank 4 was saturated wrt DATA — but with the same data, rank 8 might
extract more from the 96 examples.

## Dataset shape
Same as iter 3. 96 examples.

## Risk
Higher rank can overfit on 96 examples, possibly hurting anchor.

## Falsification plan
- S ≥ 0.96 AND anchor ≥ 0.97: rank 8 is a win.
- S in [0.93, 0.96): mild lift but not transformative.
- S in [0.90, 0.93): no lift; rank 4 was correctly tuned.
- S < 0.90: rank 8 overfits; rank 4 is the right ceiling.
- Anchor < 0.95: stylistic clobber from over-capacity.
