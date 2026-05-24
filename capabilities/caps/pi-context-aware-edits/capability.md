# Capability: pi-context-aware-edits

**Status:** New in round 2. Scaffold.

## Description

Before editing a file, the agent must **read enough surrounding context
to make the edit stylistically and structurally consistent**: imports,
neighbor functions, naming conventions, type-annotation style, comment
style, error-handling pattern.

This is *distinct* from `pi-precondition-check` (which is about
staleness — does the claim still hold). This cap is about *idiom
consistency* — the edit fits the file's existing style.

Concrete failure modes the 4B exhibits:

- Adds `import os` at the bottom of a file that imports at the top
  alphabetically.
- Uses `snake_case` in a `camelCase` codebase (or vice versa).
- Uses `type: ignore` when the codebase uses `# pyright: ignore`.
- Emits `print(...)` in a logging-everywhere codebase.
- Adds `try/except Exception` in a typed codebase that uses Result
  patterns.
- Introduces a new dependency for a function the codebase already has
  3 times.

These look like minor style issues but they're the #1 reviewer-rejection
reason in clouderic PRs. Training context-awareness is a high-leverage
behavior cap.

## Base model

Qwen3.5-4B (kiln serve on `http://localhost:8420`).

## Rollout source

Pi sessions. Multi-turn. Max turns 6 (read → understand → edit →
verify, with headroom).

## Task shape

Each task is a `(workspace, edit_request, conventions, gold_edit)`
tuple:

- **Workspace** — a small Python/Rust/Go/JS repo with consistent
  conventions throughout.
- **Edit request** — natural-language description of the change
  (e.g. "Add a function `parse_date(s: str) -> datetime` to
  `lib/dates.py`").
- **Conventions** — ground truth: the file's existing import style
  (top alphabetical / per-section), naming case, type-annotation
  conventions, error-handling pattern. Used by the rubric.
- **Gold edit** — at least one valid edit that respects all
  conventions. Multiple gold edits per task to handle the "many
  right answers" problem.

Six convention categories tracked:

1. **Import style** — top vs. inline, alphabetical vs. grouped
2. **Naming case** — snake_case / camelCase / PascalCase
3. **Type annotations** — present / absent / partial / strict
4. **Error handling** — try/except / Result / panic / unwrap
5. **Logging style** — print / logging.X / structured log
6. **Comment style** — docstrings / inline / minimal

## Rubric (v0)

Multi-component with multiplicative format gate (round-2 default).

| Sub-score | Weight | What it measures | Cannot be cheated by |
|-----------|--------|-------------------|----------------------|
| `outcome` | hard floor (multiplicative) | The edit was made and the file still parses (Python: `compile()`; Rust: `cargo check`; etc.). | Empty edit (file unchanged → outcome 0). Broken syntax → outcome 0. |
| `format_compliance` | multiplicative gate | Final assistant turn names the file modified and the convention it followed. | Boilerplate without convention name. |
| `convention_consistency` | 0.40 | Per-category score across the 6 convention categories. For each, the rubric inspects the edit and compares to the file's existing pattern. Average of 6 binary checks (was the import style preserved? was the naming case preserved? etc.). | Lucky guessing — score is averaged across 6 checks; can't be max'd without actually reading context. |
| `read_before_edit` | 0.20 | Session shows ≥1 `read` operation on the target file (or a `grep` against the same file) before the `write`/`edit` operation. | Reading a tangential file. Rubric verifies the read covered the byte range adjacent to the edit. |
| `no_redundant_imports` | 0.15 | If the function the agent adds already exists in the file or a sibling module, score 0. | — |
| `no_style_drift` | 0.10 | Adjacent functions in the same file are stylistically uniform; the new addition matches that uniformity. | — |
| `base` | 0.15 | Floor for any complete-and-parseable edit. | — |

**Composite = `outcome × format_compliance × (0.40·convention_consistency + 0.20·read_before_edit + 0.15·no_redundant_imports + 0.10·no_style_drift + 0.15)`**

## Adversarial design (§0)

**Q: What's the cheapest way to score 1.0 without doing the capability?**

A1: Always read the file before editing — wins `read_before_edit`
   even when ignoring the contents.

   Mitigation: `convention_consistency` only scores when the edit
   actually follows the file's conventions. Reading without
   attending is recoverable by ECHO env-CE — and the convention
   sub-score zeros if convention isn't followed.

A2: Copy-paste an existing function in the file and modify slightly.
   Trivially follows all conventions because it IS the convention.

   Mitigation: `outcome` requires the function to actually do what
   the task asks. Copy-paste with the wrong logic → outcome 0.

A3: Use the same import line every time — wins import-style category
   when the codebase happens to use that style.

   Mitigation: tasks span workspaces with *different* conventions, so
   a constant strategy averages to ~50% across the eval set.

A4: Skip the actual edit, claim the function already exists.

   Mitigation: outcome verifies the workspace was modified
   appropriately.

## Headroom estimate

- Repaired local baseline (`baseline-0`, 2026-05-23, thinking on,
  Pi output cap 1024): **0.4800 ± 0.4869** over 12 tasks × 3
  rollouts.
- Headroom is dominated by `format_compliance` (0.6528; ~52% of
  weighted headroom) and `outcome` (0.6944; ~45%).
- `convention_consistency` is already high (0.9583), so the original
  scaffold assumption that convention preservation is the biggest
  movable mass is stale for the repaired oracle.
- Efficiency baseline: 5.00 tool calls/rollout, 1488.5 thinking chars,
  302.7 thinking chars/tool call.

## Hypotheses

- **H1: default GRPO recipe with ECHO λ=0.05.**
  - Result: rejected. Full rank-16, max-groups=4 rank-8, and
    max-groups=1 rank-4 arms OOMed on RTX 4090 during checkpointed
    GRPO backward. A two-completion rank-4 micro arm trained and
    verified, but regressed blind eval composite from 0.4800 to
    0.4528. Thinking efficiency improved slightly (302.7 → 291.5
    chars/tool), but format, outcome, convention, and read-before-edit
    all dropped.
  - Gradient-checkpointing follow-up: the earlier OOMs used the default
    8 checkpoint segments. Explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=32`
    fit a max-groups=4 rank-8 arm on RTX 4090 (peak 18,839 MiB, 3 kept
    groups, 12 completions), but blind eval collapsed to 0.1625
    (`delta=-0.3175`). This rejects scaling the same sparse GRPO signal:
    it preserved conventions but damaged `format_compliance` and
    `outcome`.
- **H2: short positive rollout SFT bootstrap.**
  - Result: rejected. Built a 14-example train-only SFT dataset from
    the shortest high-reward H1 rollout completion per group
    (`reward >= 0.95`, 1341-2005 chars after sandbox path
    normalization). Default checkpointing OOMed in fused linear
    cross-entropy; explicit `KILN_GRAD_CHECKPOINT_SEGMENTS=32` trained
    and verified a rank-4/alpha-8 adapter in 159s. Blind eval collapsed
    composite from 0.4800 to 0.2208 (`delta=-0.2592`), mostly through
    `outcome` dropping to 0.3333. This rejects naive SFT on flattened Pi
    tool-action transcripts: high-reward rollout text still carries
    brittle execution artifacts, and copying it hurts edit completion.
- **H3: prompt-conditioned format/outcome ablation** — test whether a
  stricter runtime contract can improve the dominant format/outcome
  headroom before training another adapter.
  - Result: rejected. A no-adapter strict workflow prompt reduced tool
    calls (5.00 to 4.03) and thinking chars/tool (302.7 to 284.2), but
    blind eval composite fell to 0.2778 (`delta=-0.2022`). It hurt
    `format_compliance` (0.4861) and did not recover `outcome`
    (0.6111). This rejects prompt distillation for this cap: stricter
    wording buys efficiency by prematurely narrowing the workflow.
- **H4: idealized train-only SFT trajectories** — synthesize compact
  read/edit/verify/final traces from train tasks instead of copying noisy
  rollout transcripts.
  - Result: rejected. A 12-example verifier-backed ideal-trace SFT arm
    trained and verified locally (`rank=4`, `alpha=8`, `lr=5e-6`, peak
    VRAM 17,026 MiB). It improved `outcome` from 0.6944 to 0.7222 and
    drove `read_before_edit` to 1.0000, but `format_compliance` fell to
    0.5000, so gated composite regressed to 0.3597 (`delta=-0.1203`).
    Ideal traces are useful for edit completion, but not sufficient for
    the final-response contract.
- **H5: pairwise final-format GRPO on H4.** Use synthetic positive/negative
  pairs with identical ideal tool traces and only the final sentence
  changed, chained on H4, to isolate the `format_compliance` gate without
  relearning edit completion.
  - Result: rejected. A 12-group/24-completion synthetic GRPO arm trained
    and verified locally on top of H4 (`rank=4`, `alpha=8`, `lr=5e-6`,
    `KILN_GRAD_CHECKPOINT_SEGMENTS=32`, peak VRAM 17,301 MiB). It recovered
    `format_compliance` above baseline (0.6806 vs. 0.6528) while preserving
    `convention_consistency` (0.9583) and `read_before_edit` (1.0000), but
    `outcome` fell to 0.6389 and composite landed at 0.4667
    (`delta=-0.0133`). This rejects isolated final-format GRPO as a
    promotion path: it moved the requested gate but did not preserve edit
    completion.
- **H6: mixed-language corpus** (Python + Rust + Go + JS) — does
  context-awareness generalize across language conventions, or is it
  language-specific?
- **H7: OPD chained on best kept adapter** — distill teacher's stronger
  convention-preservation onto the GRPO-trained adapter.
- **H8: stratified by convention category** — equal task counts per
  category.

## Composition with other caps

- **Upstream:** `pi-precondition-check` (read-before-mutation is a
  shared discipline; both caps train it on different surfaces).
- **Downstream:** `pi-source-mod-workflow` (PR-quality edits need
  context-awareness).
- **Integration test:** included in `integration/cross-cap-coherence/`.

## Round 2 standard workflow

```bash
python3 build_corpus.py              # Synthetic corpus + a few real OSS slices
./capability.oracle.sh               # Baseline
./run_iter.sh h1-default-recipe      # H1
```

See `../../LAYOUT.md` for layout and `../README.md` for shared defaults.
