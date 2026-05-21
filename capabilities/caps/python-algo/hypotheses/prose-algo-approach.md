# Hypothesis: prose-algo-approach

(T-family per §4 — iter 1 must be T-family. Pure prose teaching algorithmic
problem-solving approach, no code, tests the §4 asymmetry at a baseline
where the eval has thinking disabled.)

## Claim

32 word-situation / prose examples in which the user asks how to
approach an algorithmic problem (DP, graph, recursion, search, etc.)
and the assistant explains the *conceptual strategy* in 3-5 sentences
of prose (no code, no Python, no equations), will lift the python-algo
oracle accuracy above baseline (0.807). Expected outcome bands:

- If prose alone lifts even with `enable_thinking=False` eval, the
  prose route works for code-output capabilities too.
- If prose alone tanks (model produces prose at eval time instead of
  Python), the prose route is incompatible with code-output evals
  and iter 2 must combine with code-anchors.

## Mechanism

Same as math-broad iter 1: a 4B model already has Python *primitives*
(syntax, common idioms, basic algorithms). What it lacks is reliable
*routing* — when faced with a problem description, deciding which
algorithm shape applies. Prose supervision describes the *recognition*
step (this is a DP problem because…, this is a BFS because…) without
exercising the surface form (code). The hypothesis is that improved
recognition translates to better algorithm choice in the eval.

The major risk specific to this task is **answer-form drift**: with
`enable_thinking=False` the eval expects Python directly in the
response. If our training shifts the model toward prose-as-answer
style, the model may produce prose at eval and score zero on
items whose extractor needs a Python function. Per §11 the iter-2
fix is to add 10-20% code-form anchor examples.

## Dataset shape

- Size: 32 examples
- Modality: pure prose. Assistant turns contain zero Python, zero
  pseudo-code, zero variable names. Just conceptual prose.
- Distribution by algorithm domain (rough):
  - 6 dynamic programming
  - 5 graph traversal (BFS / DFS)
  - 5 recursion / divide-and-conquer
  - 5 search (binary, linear, two-pointer)
  - 4 data-structure (stacks, queues, hashmaps)
  - 4 string algorithms
  - 3 greedy
- Surface form held OUT: no code blocks, no `def`, no `for`, no
  variable names, no function signatures.
- System prompt: none.

## Construction recipe

Hand-drafted. Each example:

  USER:   "[Real-world scenario or algorithmic problem in everyday
            language] How would you think about this kind of problem?"

  ASSISTANT: "[3-5 sentences. Identifies the algorithm family
              (DP / graph / recursion / etc.). Describes the strategy
              in prose. Names the key recurrence / invariant /
              data structure. No code.]"

Variety from: different real-world framings (e.g. inventory planning
vs. word-puzzle solving), different protagonists (engineer vs.
analyst vs. designer), different abstraction levels.

## Risk

**Major**: answer-form drift. With `enable_thinking=False` the model
produces output directly. Pure-prose supervision may train it to
respond with prose paragraphs at eval, scoring zero on code-extraction.

**Minor**: 32 examples is small. If signal is borderline, scale to 64.

## Falsification plan (committed BEFORE seeing the score)

Baseline = 0.8068. Variance ≈ 0.025. So meaningful Δ threshold is ~0.04.

- S ≥ 0.86: prose alone meaningfully lifts. Next iter is mixed-prose-code
  (32 prose + 8 code anchors) testing additivity.
- S in [0.82, 0.86): mild lift but within ~2× variance. Next iter is
  shuffle confirm OR direct code-anchor combo.
- S in [0.78, 0.82): essentially flat. Either prose route doesn't transfer
  to code-output evals, or 32 examples too few. Next iter is mixed (anchors).
- S in [0.65, 0.78): meaningful regression. Likely answer-form drift.
  Next iter is `prose-algo-approach-anchored` (same prose + 8 short
  code-anchor examples) per §11.
- S < 0.65: severe form-drift regression. Retire pure prose for this
  capability; next iter pivots to meta-question variant (which
  worked in math-broad and may bypass the form-drift problem).
- Anchor regression > 0.03: flag in notes.
