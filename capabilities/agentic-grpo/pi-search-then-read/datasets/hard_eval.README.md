# hard_eval.tasks.jsonl — pi-search-then-read hard-eval pool

Tasks where the base 4B is expected to read whole files unnecessarily
or fail to cite file:line.

## How to build it

Brand-new cap. Build by:

1. Run base on standard eval; mark tasks with
   `search_efficiency < 0.5` (reading too much).
2. Hand-construct adversarial cases:
   - **5K+ line file** where the target is at line 4500 — reading whole
     is wasteful even by lazy standards.
   - **Multiple matches** for the same symbol name across the file
     (overloaded function name); first match isn't the answer.
   - **No grep matches** at all (the symbol is dynamically generated);
     the agent should report "not found" with citation rather than
     reading whole file.
   - **Cross-file** where the answer is in a different file; the agent
     must `grep -r` not just `grep`.

## Expected hard-eval lift

| tier | base | trained | lift |
|------|------|---------|------|
| small (200 lines) | ~0.70 | ~0.85 | +0.15 (less headroom; small files are easier) |
| medium (800 lines) | ~0.40 | ~0.75 | +0.35 |
| large (2000+ lines) | ~0.25 | ~0.70 | +0.45 |

The largest lift is on large files — exactly where the cap matters most.

File is gitignored.
