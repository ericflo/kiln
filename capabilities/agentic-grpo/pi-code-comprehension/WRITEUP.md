# pi-code-comprehension — final writeup (DRAFT, in-progress)

**Status:** drafting; iter results being filled in as they land. This
section will be finalised after the 50-iter loop closes (or runs out
of session-wall-clock budget).

## What we trained

A capability called `pi-code-comprehension`. The task: given a target
symbol in a small Python code snapshot, an agent (kiln-served Qwen3.5-4B
driven by the [pi coding agent](https://github.com/earendil-works/pi))
reads + greps + emits a STRUCTURED JSON SUMMARY with seven fields:
`inputs`, `returns`, `mutates`, `calls`, `called_by`, `invariants`,
`side_effects`. Each gets `source_line` cites where applicable.

The GRPO reward (composite) is a 5-component combination:

```
composite = outcome × (
    0.20·grounding
  + 0.15·cross_file_caller_recall
  + 0.10·invariant_coverage
  + 0.05·format_compliance
  + 0.50)
```

Where `outcome` = weighted mean F1 across the 7 structured fields. The
multiplicative outcome term means an empty / unparseable answer scores
0 regardless of the agentic process.

## Baseline (iter 0)

12-task held-out eval, Qwen3.5-4B base model (no adapter):

| sub-score | value |
|-----------|-------|
| **composite** | **0.611** |
| outcome | 0.678 |
| grounding | 0.750 |
| cross_file_caller_recall | 0.833 |
| invariant_coverage | 0.236 |
| format_compliance | 0.833 |
| mean wall_clock_s / rollout | 74.4 |

11/12 rollouts produced parseable JSON.

## Best adapter so far

**Iter 1 — `h1-default-recipe` — composite 0.707 (+0.096 over base).**

Recipe:
- 16 train tasks × 4 generations per task = 64 rollouts
- Strong-signal filter at var > 0.001 → 11/16 groups kept
- GRPO + ECHO Phase 1 defaults: lr 1e-5, rank 16/alpha 32, ECHO λ=0.05,
  Dr.GRPO advantage, token-level loss aggregation
- 1.1M training-step pass at 12.1 GB VRAM peak
- Trained in ~13 minutes on RTX 6000 Ada (Ampere arch); A6000 confirmed
  to work after pod-reaping incident

Eval delta vs base:

| sub-score | base | iter 1 | Δ |
|-----------|------|--------|---|
| **composite** | **0.611** | **0.707** | **+0.096** |
| outcome | 0.678 | 0.788 | +0.110 |
| grounding | 0.750 | 0.785 | +0.035 |
| cross_file_caller_recall | 0.833 | 1.000 | **+0.167 (saturated)** |
| invariant_coverage | 0.236 | 0.375 | +0.139 |
| format_compliance | 0.833 | 1.000 | **+0.167 (saturated)** |
| mean wall_clock_s | 74.4 | 18.5 | **−75%** |

12/12 rollouts produced parseable JSON. Adapter became *decisive*: 4×
faster per rollout, no failed parses, saturated cross-file recall and
format compliance.

Adapter location:
- on pod: `/tmp/iter1-adapter/pi-cc-iter1/` (will be lost on pod
  termination)
- **B2**:
  `b2://clouderic/kiln/pi-code-comprehension/20260519/iter-1-train/adapter`
  (gzipped tar, 47 MB)

## Lessons from negative results

### Iter 2 — overtraining (`h1-more-tasks-24`, 24 train tasks, **0.589**, -0.118 vs iter 1)

Same recipe as iter 1 but with 24 train tasks instead of 16. Filter
kept 18 groups (vs iter 1's 11), training ran 1.15M steps at lr=1e-5,
finished in ~70 min. Eval composite dropped to 0.589:

| sub-score | iter 1 | iter 2 | Δ |
|-----------|--------|--------|---|
| composite | 0.707 | 0.589 | -0.118 |
| cross_file_caller_recall | 1.000 | 0.750 | -0.250 (DESATURATED) |
| format_compliance | 1.000 | 0.750 | -0.250 (DESATURATED) |
| wall_clock_s / rollout | 18.5 | 86.7 | +4.7× |

The adapter overshot the iter 1 sweet spot. Hypothesis: at lr=1e-5,
each extra group adds ~64K tokens × 8 update steps; iter 2's 18 groups
× lr=1e-5 = 3.3× iter 1's effective update budget. Future runs at
larger data need proportional lr cuts, OR truncation at iter-1-equivalent
step count.

### Iter 3+ — (to be filled)

## How to use the kept adapter

On a kiln-serving A6000:

```bash
# Restore the adapter from B2
mkdir -p /workspace/qwen3.5-4b/adapters/pi-cc-iter1
aws --endpoint-url=https://s3.us-west-002.backblazeb2.com s3 cp \
  s3://clouderic/kiln/pi-code-comprehension/20260519/iter-1-train/adapter \
  /tmp/pi-cc-iter1.tgz
tar xzf /tmp/pi-cc-iter1.tgz -C /workspace/qwen3.5-4b/adapters/pi-cc-iter1
# (The tar has a leading 'pi-cc-iter1/' directory — flatten if needed.)

# Load via kiln HTTP
curl -X POST http://localhost:8420/v1/adapters/load \
  -H 'Content-Type: application/json' \
  -d '{"name":"pi-cc-iter1"}'

# Use via pi as usual — kiln serves it as the active adapter
pi -p "Summarise the foo function in lib/foo.py — emit a JSON with
       inputs/returns/mutates/calls/called_by/invariants/side_effects."
```

Expected behaviour on a fresh code snapshot:
- Reads the target file
- `grep -rn` for callers
- Optionally reads 1-2 callers
- Emits `<answer>{...}</answer>` JSON in plain text
- Whole session ~15-30s on A6000

## Reproducing iter 1 from scratch

```bash
cd capabilities/agentic-grpo/pi-code-comprehension
python3 build_corpus.py --n-eval 12 --max 200
# On a pod with kiln + pi installed:
python3 drive.py --pod <pod_id> --start-iter 1 --stop-iter 1
```

Drive's per-iter recipe is in `recipes.json`. Iter 1's recipe:

```json
{"iter": 1, "slug": "h1-default-recipe", "family": "H1",
 "num_train": 16, "num_gens": 4, "lr": "1e-5", "filter_var": "0.02",
 "rank": 16, "alpha": 32, "echo_lambda": null,
 "rollout_concurrency": 1, "kind": "train"}
```

The kept adapter weights are at the B2 location above.

## Open questions for future work

1. **Why didn't more train data help?** At lr=1e-5, iter 2 overshoot.
   Either lr-anneal proportional to data, or cap step count.
2. **Warm-start dynamics are tricky.** Iter 3 onwards used iter 1 as
   warm-start, but the iter 1 adapter is so confident that its rollouts
   have near-zero per-group reward variance — GRPO has no advantage
   signal. Either inject sampling diversity (temperature > 0.8) or roll
   from base each iter (less efficient but consistent signal).
3. **Invariant coverage is the biggest movable headroom (0.375
   remaining).** Future capability work: hand-curate a gold corpus
   with rich implicit invariants. The auto-generated AST-derived gold
   only catches `assert` / `if ...: raise` patterns; many real-world
   invariants are commented or implied.
4. **Saturation on cross-file recall + format compliance after iter 1.**
   With both at 1.0, further iters need a harder corpus (deeper
   call-graphs, more file types) to keep these moving.

## Code shipped

All files under `capabilities/agentic-grpo/pi-code-comprehension/`:

| File | Purpose |
|------|---------|
| `capability.md` | The contract: task shape, rubric, adversarial review |
| `rubric.py` | The composite reward — F1 outcome × inner sum |
| `rubric_sanity.py` | 10-case calibration battery (passes: perfect=1.0, no_answer=0.0) |
| `task_scaffold.py` | Workdir init + pi-prompt template |
| `build_corpus.py` | AST-driven gold extraction + cross-file synthetic callers |
| `rollout.py` | Pi-runner; ECHO-compatible trajectory shape |
| `drive.py` | End-to-end Python iter driver (replaces fragile bash) |
| `recipes.json` | Per-iter hyperparameter recipes |
| `run_iter.sh` / `record_iter.py` | Per-iter shell + result-logger |
| `backup_to_b2.py` | Adapter + rollouts upload to B2 |
| `capability.oracle.sh` | Blind eval interface |
| `capability.jsonl` | Append-only iter log |
| `failures.jsonl` | Sidecar log for iter exceptions |
| `seed_repos/` | Hand-crafted small Python repos for corpus seeding |
| `datasets/` | Generated train + eval task JSONL (200 tasks total) |

## Audit log

(Same content as `IN_PROGRESS.md`; final writeup will fold them
together.)
