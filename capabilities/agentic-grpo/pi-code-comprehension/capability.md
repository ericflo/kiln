# pi-code-comprehension — mental model from reading

**Status:** Scaffold. **Rank 3/10**. Every safe edit downstream depends
on the agent having a correct model of the code it's about to change.

**Goal.** Given a function or small module, the agent — using only its
tool surface (`Read`, `grep`) — produces a structured summary of:
inputs, returns, mutations (filesystem / global / argument-mutating),
called helpers, callers, and the function's invariants. The summary
must cite line numbers grounding each field.

## Why this capability

Two clouderic patterns:

1. **Edit-without-comprehension.** The saved note
   `verify-architecture-claims-in-source` is exactly this — "always
   verify architecture claims by reading the specific source file, not
   inferring from adjacent code." Multiple PRs landed broken because
   the agent inferred from the *call site* rather than reading the
   *definition*.
2. **Drive-by edits to long functions.** Agents that don't internalize
   a function's invariants (e.g. "this function assumes the lock is
   already held") edit around the invariant, ship green CI, and break
   things subtly.

This cap is the agentic complement to OPD's
`faithful-code-summarization` — that one trains generation given a
single file in context; this one trains *agentic* comprehension where
the model has to *fetch* the relevant context across files.

## Task shape

Each task is `(repo_snapshot, target_symbol, gold_summary)`:

- **Repo snapshot** — small-to-mid repo.
- **Target symbol** — `(file, function_name)` or `(file, class_name)`.
- **Gold summary** — JSON with fields:
  ```json
  {
    "inputs":   [{"name": "...", "type": "...", "source_line": N}],
    "returns":  [{"type": "...", "source_line": N}],
    "mutates":  ["filesystem:writes to /tmp", "global:STATE"],
    "calls":    [{"name": "...", "file": "...", "line": N}],
    "called_by":[{"file": "...", "line": N}],
    "invariants":["...", "..."],
    "side_effects": ["...", "..."]
  }
  ```

Example prompt:

```
Produce a structured summary of the function `apply_chat_template` in
`crates/kiln-tokenizer/src/lib.rs`. Output JSON with the schema:
{inputs, returns, mutates, calls, called_by, invariants}. Cite line
numbers in each field.
```

Correct trajectory:

1. `Read` the file (paginated if needed).
2. `grep -rn "apply_chat_template"` to find call sites.
3. (Optional) read one or two callers to confirm conventions.
4. Emit the JSON summary.

## Rubric design (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor | Mean F1 across the structured fields (`inputs`, `returns`, `mutates`, `calls`, `called_by`, `invariants`). Field-specific tokenizers; line numbers allowed ±N off-by. | Empty JSON → 0. Free-form prose → JSON parse failure → 0. |
| `grounding` | 0.20 | Per-field accuracy of cited line numbers: gold has the line, predicted line is within ±2. | Citing line 1 for every field. |
| `cross_file_caller_recall` | 0.15 | Recall on the `called_by` set. Specifically: did the agent find at least one caller in a different file? | Listing only intra-file callers. |
| `invariant_coverage` | 0.10 | Fraction of gold invariants the summary names (semantic-match, with embedding similarity ≥ 0.7 to a gold paraphrase). | One generic invariant per task. |
| `format_compliance` | 0.05 | JSON parses; schema valid. | Malformed JSON → 0. |

**Composite** = `outcome × (0.20·grounding + 0.15·caller_recall + 0.10·invariants + 0.05·format + 0.50·base)`

## ECHO recipe

**Moderate fit.** The forward pass is read-heavy; env tokens are file
contents, which are extremely predictable from the path. ECHO loss on
the read response biases the model toward "predict what the file
says," which is exactly the comprehension skill.

But the *output* (the JSON summary) is action-side. So ECHO helps the
*read*; GRPO scores the *summary*. This is exactly the standard
multi-turn shape — ECHO defaults apply.

**Hypothesis worth testing:** raise lambda to `0.075` early. For tasks
this read-heavy, predicting the file content is a major fraction of
the learning signal.

## Hypotheses

- **H_paginated_reads** — for long files (>200 lines), force the agent
  to use paginated reads with `offset/limit`. Tests whether the model
  reads the *right* slice or asks for the whole thing.
- **H_invariant_inference** — provide a subset of tasks where the
  function's docstring is removed; the agent must infer invariants
  from the body. Predicts: dramatic drop without docstring; ECHO closes
  some of the gap.
- **H_callgraph_recall** — vary the caller diversity per task; check if
  agents trained here generalize to repos with many call sites.

## Adversarial design (§0)

**Q: bluff the JSON without reading.** Mitigation: `grounding` is line-cited;
ungrounded fields score 0.

**Q: copy the docstring as "invariants".** Mitigation: gold invariants
include *implicit* ones the docstring doesn't state (e.g. "assumes lock
held," "must run after `init()`"). Pure-docstring copies miss those.

**Q: read everything, emit no structure.** Mitigation: `format_compliance`
gates parse; outcome F1 requires schema fields populated.

## Headroom (estimated)

- Baseline composite ~0.40–0.55. The 4B base can produce structured
  JSON reasonably but loses on `grounding` (it doesn't cite lines) and
  `cross_file_caller_recall` (it doesn't grep).
- Target sub-score: `grounding` (highest movable mass — the model can
  emit citations, it just doesn't by default).

## Files to create

- [ ] `rubric.py` — composite. The `invariant_coverage` term needs a
  small embedding model for semantic match; use BGE-small or hash if
  embedding cost is too high.
- [ ] `task_scaffold.py` — gold summaries can be bootstrapped by
  running a strong teacher (the vLLM teacher used in OPD caps) over
  hand-picked symbols and human-editing the result. ~50 hand-curated
  golds; the rest synth-and-filter.
- [ ] `rollout.py`, `build_corpus.py`, `capability.oracle.sh`,
  `run_iter.sh`.
- [ ] `calibration/{good,bad}.jsonl` — good: read-then-grep-then-summary
  trajectories. Bad: emit JSON without reading.

## Next steps for the agent picking this up

1. Read the shared README and `capabilities/opd/faithful-code-summarization/capability.md`
   — the OPD sibling is the closest reference.
2. Pick ~30 target symbols from kiln itself, biased toward
   non-trivial: training functions, tokenizer methods, scheduler
   methods. Hand-write golds; this anchors the corpus quality.
3. Decide on the embedding choice for `invariant_coverage` upfront —
   running BGE-small adds ~10ms per task; hash-fallback is free but
   noisier.
4. Build calibration first; verify that "JSON without reads" is
   clearly worse than "read-then-summary" in your rubric.
5. Iter 0 baseline, then iter 1 with ECHO defaults.

## References

- `capabilities/opd/faithful-code-summarization/` — OPD sibling.
- `capabilities/opd/code-symbol-extraction/` — closely related.
- `crates/kiln-tokenizer/src/lib.rs::apply_chat_template` — a good
  candidate target symbol (cross-file callers; non-trivial invariants).
