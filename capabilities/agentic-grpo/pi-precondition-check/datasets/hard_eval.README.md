# hard_eval.tasks.jsonl — pi-precondition-check hard-eval pool

Adversarial pre-condition tasks where the base 4B is most likely to
mutate without verifying.

## How to build it

Brand-new cap (no round-1 archive). Build by:

1. Run base on standard eval; mark tasks where the model:
   - Mutated without any read preceding the edit, OR
   - Emitted `precondition_failed` on a holds_true task.
2. Hand-construct adversarial cases:
   - **Plausible-but-stale claims** — claim references a file that
     exists with a sibling symbol; tempting to mutate the sibling.
   - **Multi-file claims** — claim references a file but the actual
     definition lives in a parent module via import; agent must read
     the import chain.
   - **Recent-rename traps** — git log shows a recent rename matching
     the claim; the rename has *already shipped*; agent must verify
     the current file, not infer from history.
   - **Symbol-with-similar-name** — the claim's symbol shares prefix
     with a still-present symbol; agent reading the file might
     conflate them.

## Expected hard-eval lift

| claim type | base | trained | lift |
|------------|------|---------|------|
| holds-true (simple) | ~0.50 | ~0.85 | +0.35 |
| stale (simple) | ~0.30 | ~0.70 | +0.40 |
| holds-true (adversarial) | ~0.30 | ~0.70 | +0.40 |
| stale (adversarial) | ~0.15 | ~0.60 | +0.45 |

File is gitignored.
