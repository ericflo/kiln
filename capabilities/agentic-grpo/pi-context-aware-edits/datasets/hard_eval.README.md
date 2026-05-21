# hard_eval.tasks.jsonl — pi-context-aware-edits hard-eval pool

Tasks the base 4B is expected to violate conventions on.

## How to build it

Brand-new cap (no round-1 archive). Build by:

1. Run base on standard eval; mark tasks with
   `convention_consistency < 0.5`.
2. Hand-construct adversarial cases:
   - **Mixed-style workspace** — half files snake, half camel; agent
     must read the *specific target file* not the directory average.
   - **Subtle conventions** — e.g. type annotation style is "all
     keyword args annotated, positional unannotated" (real OSS convention).
   - **Multi-language workspace** — Python + Rust + Go in one repo;
     edit the .rs file with Rust conventions, not Python ones.
   - **Recently-refactored conventions** — old conventions visible in
     recent commits but obsoleted; the agent must read the latest
     file state, not infer from one stale neighbor.

## Expected hard-eval lift

| profile | base | trained | lift |
|---------|------|---------|------|
| py_strict_typed_snake | ~0.50 | ~0.80 | +0.30 |
| py_camel_loose | ~0.40 | ~0.75 | +0.35 |
| rust_snake_result | ~0.45 | ~0.75 | +0.30 |
| go_camel_pascal | ~0.40 | ~0.70 | +0.30 |
| mixed_language | ~0.30 | ~0.65 | +0.35 |

File is gitignored.
