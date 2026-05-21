# Hypothesis: meta-what-kind-of-algo

(Meta-routing pivot per iter-2 falsification plan. Math-broad's surprise
winner — meta-questions transferred much better than prose-explanation
per §15. Test whether the same pattern holds for code-output evals.)

## Claim

32 meta-question examples in which the user describes a situation and
asks "what kind of algorithmic problem is this?" with the assistant
classifying the problem type in prose (no code) will outperform
iter 1's pure prose. Expected S in [0.78, 0.86].

If S ≥ 0.85 (above-baseline by >1×variance), meta-routing transfers
to code-output evals despite enable_thinking=False. Big finding.

If S < 0.74, meta also fails for code-output capabilities; iter 4
pivots to mixed-meta-code (meta + code anchors).

## Mechanism

Meta-questions train problem *recognition*: identify the algorithm
family from a problem description. This is the gatekeeping operation
that precedes coding. Even though the eval expects code, if the model
learns better recognition, it routes to better algorithms — and the
code-generation primitives that already exist in Qwen3.5-4B handle
the rest.

The math-broad data: meta-alone (rank 4, 32 examples) gave 0.919
vs prose-alone (rank 4, 64 examples) at 0.903. Meta was MORE efficient
per example AND beat prose. If that pattern holds, this iter should
land above iter 1's 0.7603 — possibly substantially.

Risk that may NOT transfer: math-broad eval allowed thinking; python-algo
eval has thinking disabled. Meta-routing teaches a *recognition* step
that ideally precedes code generation, but with thinking off there's
no place for the recognition to be explicit in the response. The
hypothesis is that the recognition becomes IMPLICIT in the generated
code, not absent.

## Dataset shape

- Size: 32 examples
- Modality: prose-only assistant turns naming the algorithm family
  and brief mechanism, no code
- Distribution: same 7 algorithm-family split as iter 1
- Surface form held OUT: no Python, no code blocks, no def

## Falsification plan

- S ≥ 0.85: meta beats baseline cleanly. Next iter mixes meta + code
  anchors (iter 4) to test additivity.
- S in [0.81, 0.85): meta neutral; not regressing but not lifting.
  Next iter mixes with code anchors.
- S in [0.74, 0.81): meta partially regresses (slight answer-form pull).
  Same answer-form drift as iter 1 but milder. Iter 4 with anchors.
- S < 0.74: meta-alone fails for this capability. Big departure from
  math-broad. Iter 4 pivots to code-only training (numeric-drill
  equivalent — full worked examples).
