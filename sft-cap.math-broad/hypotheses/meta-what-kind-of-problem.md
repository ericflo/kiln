# Hypothesis: meta-what-kind-of-problem

(T-family variant per §4 and §15 surprising patterns: "Meta-questions
transfer surprisingly well.")

## Claim

32 examples where the user asks "what kind of math problem is this?"
given a real-world situation, and the assistant classifies the problem
type and briefly sketches the approach, will lift the math score to
within ±0.03 of iter 3 (0.935) — i.e., matching the mixed-prose-numeric
winner — without requiring numeric anchors.

If S ≥ 0.93, meta-questions transfer at least as well as iter 3's
hybrid recipe. This would be a clean, smaller-recipe win.

## Mechanism

Asking "what kind of problem is this" forces the model into a
classification task: it must identify the math domain and the relevant
solution shape, but not produce numerical work. This is meta-routing:
the model practices the *first step* of every math problem (recognise
the type) without practising any subsequent step.

Per skill §15 ("Meta-questions transfer surprisingly well"), this
should outperform direct prose explanations. The intuition is that
the model is being trained on the *gatekeeping* operation rather than
the downstream procedural work.

## Dataset shape

- Size: 32 examples
- Modality: prose. Each assistant turn names the problem type
  (e.g., "linear equation in one variable", "first-order ODE",
  "triangulation", "geometric scaling") and briefly explains the
  routing rationale.
- Distribution by domain: ~4-5 examples per domain across the 7 domains.
- Surface form held OUT: assistant turns contain no numbers, no
  equations, no worked steps.
- System prompt: none.

## Construction recipe

Hand-drafted. Each example uses an iter-1 situation but the assistant
turn structure is:
  "This is fundamentally a [problem type]. [Brief mechanism note].
   [Optional frame label or comparison]."

## Risk

Main risk: the model learns to classify problems but not to *solve*
them. At eval time, it might say "this is a quadratic" instead of
giving the answer. Manifests as a regression like iter 5's voice
paraphrasing problem.

Secondary risk: 32 examples is small. If signal is borderline,
scale to 64 in iter 12.

## Falsification plan (committed BEFORE seeing the score)

- S ≥ 0.94: meta-routing matches or exceeds iter 3. Next iter is
  `mixed-meta-numeric` (32 meta + 32 numeric) to test if anchor + meta
  stacks above iter 3.
- S in [0.90, 0.94): meta is comparable to iter 1's prose lift but
  doesn't beat iter 3. Next iter is `mixed-meta-numeric` anyway, to
  test additivity with proven anchors.
- S in [0.83, 0.90): partial lift; meta-routing helps but less than
  iter 1's constructive prose. Retire as primary route.
- S < 0.83: regression. Meta-routing trains classification at the
  expense of solving. Note in dead ends.
