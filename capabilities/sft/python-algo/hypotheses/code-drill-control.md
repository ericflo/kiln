# Hypothesis: code-drill-control

(Iter-4 pivot per the prose-route's failures. The math-broad triangulation
control here. Direct Python supervision.)

## Claim

32 worked Python algorithmic examples — complete function solutions
across DP, graph, recursion, search, data structure, strings, greedy —
will outperform iter-1's prose attempt (0.7603) and likely beat
baseline (0.8068). Expected S in [0.83, 0.90].

If S ≥ baseline + 1×variance (~0.83), direct code SFT works for this
capability (in contrast to math-broad where it was BEATEN by meta-routing).
This would invert the math-broad result and indicate that for
thinking-disabled code-output evals, the form must be the supervision.

If S < baseline, even direct code drill hurts on a 4B model that already
has decent baseline Python ability — meaning we need a higher-quality
or differently-curated set of examples.

## Mechanism

The prose iterations failed because they trained the model to produce
non-code answers. Direct code training trains the model to produce code,
which matches the eval's expectation. The hypothesis is that the code
examples expose the model to a diverse set of algorithm-shape→Python-shape
mappings, which is what's actually needed.

## Dataset shape

- Size: 32 worked Python algorithmic examples
- Modality: full Python functions in fenced code blocks; no prose
  explanation in the assistant turn beyond the code itself
- Distribution by algorithm domain (matches iter 1's planning):
  - 6 DP (stairs, coin-change, LIS, edit-distance, knapsack, rob)
  - 5 graph (BFS, mirror, components, grid path, topological sort)
  - 5 recursion (in-order, depth, flatten, permutations, subsets)
  - 5 search (binary search, rotated min, quickselect, lower-bound, merge)
  - 4 data structures (parens, moving average, two-sum, dup check)
  - 4 strings (palindrome, longest distinct, anagrams, find-occurrences)
  - 3 greedy (meeting rooms, non-overlapping, jump)
- Surface form HOLDS: matches eval's expected output form.
- System prompt: none.

## Falsification plan (committed before seeing the score)

- S ≥ 0.86: code drill works strongly. Direct supervision is the
  winning route for code-output capabilities. Iter 5 scales up or
  adds anchors.
- S in [0.82, 0.86): code drill matches baseline + small lift. Real
  but modest. Iter 5 mixes with prose or scales up.
- S in [0.78, 0.82): code drill neutral. Surprising — implies the
  base model already has these algorithms and SFT can't add. Iter 5
  pivots to harder problems.
- S in [0.72, 0.78): code drill regresses. Indicates the specific
  examples don't generalise to the eval's surface form. Iter 5 tries
  a different curriculum.
- S < 0.72: code drill actively hurts. Disturbing. Iter 5 reverts to
  config-only changes (rank, lr).
