# Ideas backlog

Promising hypothesis families we haven't tried this round. Append as
bullets; prune when tried; the skill mines this on resume.

- `prose-approach-broad-paraphrased` (T+F): same recipe as iter 1 but
  vastly more diverse framings — formal academic, casual conversation,
  terse technical, narrative storytelling. Tests whether *more framing
  diversity* on the same content lifts further.
- `prose-approach-broad-anchored` (T+anchor, §11): iter 1 dataset + 12
  short-answer anchor examples. Run if iter 1 regresses (answer-form drift).
- `prose-mistake-named` (T variant): assistant turns *name the mistake*
  a novice would make rather than describing the right approach. Tests
  the negative-contrast supervision route.
- `meta-what-kind-of-problem` (T variant): user asks "what kind of math
  problem is this?" given a real-world scene; assistant classifies
  ("This is fundamentally a rate problem because...") without solving.
- `numeric-drill-control` (control, M-family): 32 worked numerical
  examples on the same math domains. Use as triangulation against a
  prose winner to confirm prose was doing causal work, not just SFT.
- `mixed-prose-numeric` (M+T): half prose-approach, half numeric worked
  examples. Tests whether the routes are additive or competitive.
- `terse-form-anchors-only` (S, §11): pure short-answer dataset to
  isolate whether form discipline alone helps without content lift.
- `algorithm-prose-by-domain` (T at finer grain): 32-example datasets
  that drill ONE domain (e.g., only calculus) with prose. Tests whether
  narrow prose transfers to narrow eval items, or whether breadth in
  prose was necessary for broad lift.
- `enable_thinking_off` (S, kiln-side): set Qwen3 thinking off during
  inference. Tests the user's "fewer thinking tokens" preference
  directly. Note: this is an INFERENCE-time change, not a training
  intervention; run as a configuration ablation, not a dataset one.
- `terse-thinking-distillation` (T+S): train on examples whose
  reasoning_content channel contains very short, structured thinking
  ("Identify: <concept>. Recall: <formula>. Compute: <one step>.
  Answer."). Tests whether brevity in thinking transfers.
