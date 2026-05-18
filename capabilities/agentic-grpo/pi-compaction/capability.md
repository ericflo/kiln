# Capability: pi-compaction

## Description

When a pi (earendil-works/pi) coding-agent session grows past its context
budget, pi triggers **compaction**: it serializes the messages-to-discard
into text, asks the model to produce a structured summary in pi's EXACT
format, and replaces those messages with that summary so the session can
continue.

The model that produces the compaction summary is the same one that drives
the pi session (kiln-served `qwen-3.5-4b-kiln` in our case). A great
compaction summary lets a fresh agent resume the work seamlessly. A poor
one drops critical state — file paths, error messages, the user's actual
request — and the agent restarts from scratch or, worse, hallucinates
state it then acts on.

**The target capability is a single-turn text-completion task** in the
sense that it's one model call (no tools, no multi-turn). But the
*subject matter* is multi-turn pi sessions: the model has to understand
what just happened across many turns and tool calls, and compress it
losslessly into a fixed-format summary.

This makes it **agentic-input single-turn-output GRPO** — a hybrid where
the rollout has no tool loop but the data distribution is the
hardest-case "context-collapse moment" of a real agent session.

## Why this matters in practice

In production:
- Long pi sessions hit context exhaustion regularly (the default
  `keepRecentTokens` is 20K, `reserveTokens` is 16K).
- Every compaction is a chance to lose state. A bad summary means an
  agent that was 80% through a task essentially restarts.
- Most modern LMs do compaction well in isolation but pi specifically
  enforces a rigid markdown format with specific section names — and
  small models often deviate from this format under pressure.

A 4B that can do this reliably is the lowest-bandwidth deployment story
for pi.

## Base model

Qwen3.5-4B served by `kiln serve` on port 8420. Adapter loaded via
`POST /v1/adapters/load`.

## Rollout source

**Single-turn text completion.** Each rollout is one chat-completion
request to kiln:

- `system`: pi's compaction system prompt (verbatim from
  `packages/coding-agent/src/core/compaction/utils.ts`):

  > You are a context summarization assistant. Your task is to read a
  > conversation between a user and an AI coding assistant, then produce
  > a structured summary following the exact format specified.
  >
  > Do NOT continue the conversation. Do NOT respond to any questions in
  > the conversation. ONLY output the structured summary.

- `user`: pi's full compaction user message:

  ```
  <conversation>
  {serialized conversation text via pi's serializeConversation()}
  </conversation>

  {SUMMARIZATION_PROMPT — pi's verbatim section template}
  ```

The conversation source comes from real Claude / pi production sessions
that exceeded ~40K input tokens. They are extracted from
`/data/apps/trajectory-trainer/trajectories.db` and re-serialized using
pi's serializer so the rollout input matches the production input exactly.

## v1 rubric

See `rubric.py` for the implementation. The rubric is intentionally
multi-component and **anti-shortcut by construction** — each
"reward presence" sub-score has a paired "punish shortcut" sub-score.

### Composite

```
composite = (
    0.10 * format_score             # rigid pi-format compliance
  + 0.30 * content_score             # preserves goals/files/errors/identifiers
  + 0.20 * faithfulness_score        # no hallucination (reverse-direction)
  + 0.10 * compression_score         # right size (not too long, not too short)
  + 0.10 * continuability_score      # next-steps section is concrete
  + 0.20 * outcome_score             # gate: format + content non-zero
)
```

`outcome_score` is the hard gate: if the output is not pi-formatted OR
the content_score is 0, composite collapses to ~0. This prevents the
adapter from learning to produce gibberish that happens to satisfy the
length or compression sub-scores.

### Format compliance (10%)

Pi's compaction format is **rigid**. The summary must have, in order:

1. `## Goal`
2. `## Constraints & Preferences`
3. `## Progress` with `### Done`, `### In Progress`, `### Blocked`
4. `## Key Decisions`
5. `## Next Steps`
6. `## Critical Context`

Sub-scores (each 0 or 1, averaged):

- `format.has_goal` — `## Goal` heading present
- `format.has_constraints` — `## Constraints & Preferences` present
- `format.has_progress` — `## Progress` + at least one of Done/InProgress/Blocked subsections
- `format.has_key_decisions` — `## Key Decisions` present
- `format.has_next_steps` — `## Next Steps` present
- `format.has_critical_context` — `## Critical Context` present
- `format.order_correct` — sections appear in pi's specified order
- `format.no_continuation` — does NOT continue the conversation (no
  "Sure, I can help with..." / "Here's what I'd do next..." etc.)

### Content (30%)

The summary must preserve the load-bearing facts from the source. These
facts are extracted programmatically from the source conversation.

- `content.first_user_goal_recall` — does the Goal section contain
  tokens from the FIRST user message? (a fuzzy substring / n-gram
  recall metric)
- `content.file_paths_recall` — fraction of distinct file paths
  mentioned in source tool calls that appear in the summary
- `content.error_recall` — fraction of error/exception lines from tool
  results that are referenced in the summary (by error type, key
  substring)
- `content.identifier_recall` — fraction of distinct function / class /
  module names from source that appear in the summary
- `content.read_file_block_correctness` — fraction of files-read-only
  that appear in the trailing `<read-files>` block
- `content.modified_file_block_correctness` — fraction of files-modified
  that appear in the trailing `<modified-files>` block

### Faithfulness (20%) — reverse direction

For every concrete claim in the summary, check the source supports it:

- `faithfulness.file_paths_in_source` — fraction of file paths IN the
  summary that ALSO appear in the source
- `faithfulness.identifiers_in_source` — fraction of identifiers IN the
  summary that ALSO appear in the source
- `faithfulness.no_invented_errors` — penalize if the Blocked section
  cites errors not present in the source
- `faithfulness.semantic_overlap` — fastText / character-n-gram overlap
  between summary content (excluding section headers) and source

### Compression (10%)

- `compression.is_smaller` — 1 if summary < source, 0 otherwise
- `compression.ratio_band` — piecewise linear:
  - score = 1.0 when summary is 5%–25% of source length
  - score = linear ramp from 0 to 1 as length grows 25% → 50%
  - score = linear ramp from 1 to 0 as length shrinks 5% → 0%

### Continuability (10%)

- `continuability.next_steps_present` — ≥1 item in the Next Steps list
- `continuability.next_steps_concrete` — each item references a file
  path, identifier, or error from the source
- `continuability.next_steps_count_in_band` — 1 if 2-5 next-step items,
  else penalize

### Outcome gate (20%)

`outcome = format_score >= 0.5 AND content_score >= 0.2`. If false,
composite is set to `0.1 * (format_score + content_score + faithfulness_score) / 3` — a small floor.

## Adversarial design audit

Cheapest paths to 1.0 without doing the capability, and the rubric's
counter for each:

| Shortcut | Counter |
|----------|---------|
| Output the format template with `(none)` everywhere | `content.first_user_goal_recall` requires source words; all `*_recall` sub-scores zero |
| Copy the source verbatim | `compression.is_smaller` = 0; `compression.ratio_band` = 0 |
| Output the right format but with hallucinated content | `faithfulness.file_paths_in_source` < 1; `faithfulness.identifiers_in_source` < 1 |
| Skip the format and just dump source-quoted sentences | `format.*` all zero |
| Continue the conversation instead of summarizing | `format.no_continuation` = 0; `outcome` gate kills composite |
| Generate file-list-only output | format checks fail; content.* checks fail except file-block correctness |
| Add lots of extra unrelated content | `compression.ratio_band` punishes; `faithfulness.identifiers_in_source` punishes |

## Eval set vs. training set

- **Training tasks** (`datasets/train.tasks.jsonl`): ~40 real long
  conversations sampled from `trajectories.db`, length-stratified
  (30K-50K, 50K-80K, 80K-120K tokens). Pi-serialized via
  `task_scaffold.py` so the rollout input matches what pi would send
  the model in production. **No truncation** — pi compaction is
  defined as "summarize this long conversation"; truncating the input
  defeats the task.

- **Eval tasks** (`datasets/eval.tasks.jsonl`): ~24 held-out
  conversations from the trajectories `val` and `test` splits — no
  overlap with training tasks. Same length stratification, same
  no-truncation rule.

Each task carries a precomputed `ground_truth` payload with the
file paths, identifiers, error messages, and first-user-goal tokens
extracted by `extract_ground_truth.py` (run once per task).

### Note on iter 1's truncation experiment

In iter 1 we briefly explored truncating source_text to make GRPO
training tractable (the full 50K-token serialized inputs proved too
slow for `cuda_grpo_ablation` within reasonable pod TTLs).
**That was the wrong direction** — truncating the input defines a
different task ("micro-compaction") that wouldn't generalize to pi's
real use case (compact a full long session). The training-too-slow
finding is kept as `kiln-polish.jsonl#training-too-slow-on-long-context-grpo`
so it's tractable to fix kiln-train rather than rewrite the cap.

`truncate_corpus.py` is kept in the tree as a diagnostic tool, not as
part of the canonical training recipe.

## Calibration

`calibration/good.jsonl` and `calibration/bad.jsonl` are hand-crafted
reference summaries on the same 5–6 prompts:

- `good`: real Claude-Opus-produced compaction summaries (or
  hand-written gold summaries) that score ≥ 0.85 composite.
- `bad`: copy-the-source, template-with-(none), wrong-format-prose,
  hallucinated-content variants, each designed to score ≤ 0.20.

`rubric_sanity.py` runs the rubric on both and reports composite
ranges. Iter 0 will not begin until calibration passes.

## Hypothesis log

| Iter | Slug | Family | Composite | Δ | Status | Notes |
|------|------|--------|-----------|---|--------|-------|

(populated below as iterations land)

## Kiln-polish prerequisites

See `kiln-polish.jsonl`. v0 sidesteps the multi-turn token-masking gap
because compaction is single-turn output. Kiln server's adapter-load /
serve path is exercised the same way as the pi-doctest cap.
