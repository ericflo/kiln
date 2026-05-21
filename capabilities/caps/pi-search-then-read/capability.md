# Capability: pi-search-then-read

**Status:** New in round 2. Scaffold.

## Description

Before reading a large file, the agent should use `grep` / `find` /
search to locate the relevant byte range. The 4B today often:

- Opens whole files (8K-30K LoC) when the task references a specific
  function or class.
- Re-reads the same file multiple times to relocate the same symbol.
- Uses `cat` (or `Read` with no offset) instead of targeted lookup.

The capable behavior: `grep -n target_symbol file.py` → see the
line number → read a 30-line window around it. Saves 100× the context
budget, leaves more room for actual reasoning.

This cap composes with `pi-code-search` (which is about *which file*
to look in) — pi-search-then-read is about *which window of that file*.

## Base model

Qwen3.5-4B (kiln serve on `http://localhost:8420`).

## Rollout source

Pi sessions. Multi-turn (max 6 turns).

## Task shape

Each task is a `(workspace, query, target, gold_window)`:

- **Workspace** — repo with at least one large file (≥2000 LoC) in
  the relevant module set.
- **Query** — natural-language question that requires reading
  *specific* lines of the large file (e.g. "What's the return type of
  `parse_config` in `lib/config.py`?").
- **Target** — ground-truth file path and symbol.
- **Gold window** — ground-truth byte range that contains the answer
  (e.g. lines 4567-4585 of `lib/config.py`).

The eval set spans a mix of file sizes (500 LoC, 2000 LoC, 8000 LoC)
to test scaling.

## Rubric (v0)

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor (multiplicative) | Final assistant turn correctly answers the query (exact-match or LLM-judge equivalence to gold answer). | — |
| `format_compliance` | multiplicative gate | Final turn cites the file:line of the answer. Forces the agent to ground its answer. | — |
| `search_efficiency` | 0.35 | **TARGET.** `1 - (bytes_read / file_size)`. 1.0 when bytes_read ≤ 30 lines of a 30K-LoC file; decays toward 0 as bytes_read approaches file size. Per-file averaged. | Targeted re-reads of the same window still count once; the bytes are the same. |
| `search_before_read` | 0.25 | 1.0 iff a `grep` / `find` / `rg` call referencing the query symbol appeared before any `read` of the large file. | Random `grep` against unrelated terms — rubric requires the grep pattern overlap the query. |
| `no_redundant_reads` | 0.20 | 1.0 - (duplicate_read_signatures / total_reads). | — |
| `precision_of_first_read` | 0.10 | If the first `read` after search lands in the gold window, score 1; otherwise score (lines_overlap / 30). | Reading the whole file in one go → low score per line. |
| `base` | 0.10 | — | — |

**Composite = `outcome × format_compliance × (0.35·search_efficiency + 0.25·search_before_read + 0.20·no_redundant_reads + 0.10·precision_of_first_read + 0.10)`**

## Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

A1: Always grep, even when the file is small enough to read whole.

   Mitigation: penalty is only applied when search-efficiency was
   *needed*. For files < 200 LoC, the rubric awards full
   search_efficiency regardless of whether grep was used (small files
   are fine to read whole).

A2: grep for any random keyword — wins `search_before_read`.

   Mitigation: the grep pattern must overlap the query terms.
   Implemented as: split query into nouns/identifiers; the grep
   command must contain ≥1 of them.

A3: Read all small files in the repo to make the *aggregate*
   bytes_read low.

   Mitigation: `search_efficiency` is per-file averaged, not
   aggregate. Reading 20 small files at 100% each scores 0 on each.

A4: Read just the first 30 lines of every file, declare the answer
   from imports. Sometimes lucky on type questions.

   Mitigation: outcome verifies the answer; cheap-strategy outcome=0
   on most tasks.

## Headroom estimate

- Baseline composite: **~0.40** (the 4B reads whole files routinely).
- Headroom: ~0.60.
- Target sub-score: `search_efficiency` (huge movable mass).

## Hypotheses

- **H1: default GRPO recipe.** Hypothesis: composite +0.15.
- **H2: ECHO-heavier (λ=0.075).** Search results are env tokens; more
  env attention should help.
- **H3: tasks scaled across file sizes** (small, medium, large) — make
  sure recipe doesn't overfit to one regime.
- **H4: chain from pi-code-search best** — code-search trained the
  "which file" decision; this trains "which window."

## Composition with other caps

- **Upstream:** `pi-code-search` (composes naturally).
- **Downstream:** All other caps benefit from search-first habit
  (reduces context burn, more room for reasoning).
- **Integration test:** included in
  `integration/cross-cap-coherence/`.

## Round 2 standard workflow

```bash
python3 build_corpus.py             # Mine large-file repos from OSS
./capability.oracle.sh              # Baseline (expect ~0.40)
./run_iter.sh h1-default-recipe     # H1
```

See `../../LAYOUT.md` and `../README.md` for shared infrastructure.
