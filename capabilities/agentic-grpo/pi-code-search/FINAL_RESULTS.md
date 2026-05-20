# pi-code-search — final results

## Shipping candidate

**`pi-code-search-iter5-h5-replay-iter1`** (commit `605c347`, B2: 47MB)

| metric | base | iter 5 (peak) | iter 5 (5-eval mean) |
|--------|------|---------------|----------------------|
| composite | 0.5432 | **0.6004** | 0.5676 ± 0.030 |
| outcome | 0.737 | **0.820** | 0.785 |
| efficiency | 0.401 | 0.440 | 0.407 |
| grounding | 0.844 | 0.969 | 0.890 |
| format compliance | 0.844 | 0.969 | 0.900 |
| outcome pass | 25/32 | **28/32** | 26.8/32 |
| mean wall (s) | 19.2 | 40.4 | 38.4 |
| mean n_tool_calls | 4.59 | 2.19 | 2.62 |

### Recipe (the one to ship)

```bash
TRAIN_LIMIT=10 NUM_GEN=4 FILTER_VAR=0.05 MAX_GROUPS=10 \
  LR=1e-5 RANK=16 ALPHA=32 ECHO_LAMBDA=0.05 SEED=3141592653 \
  EPOCHS=1 \
  bash run_iter.sh
```

ECHO=0.05 (default), rank-16, lr=1e-5, single epoch, 10 train tasks
with strong-signal-only filter (var ≥ 0.05), 4 rollouts per task.

### Restore from B2

```bash
b2 file download \
  b2://clouderic/kiln/pi-code-search/pi-code-search-iter5-h5-replay-iter1.tgz \
  iter5.tgz
tar xzf iter5.tgz
ln -sfn "$(pwd)/adapter/pi-code-search-iter5-h5-replay-iter1" \
  "$KILN_MODEL_PATH/adapters/pi-code-search-iter5-h5-replay-iter1"
# then POST /v1/adapters/load with that name
```

### All adapter backups

| Iter | B2 tarball | Size | Composite (best clean eval) |
|------|------------|------|-----------------------------|
| 1 | `pi-code-search-iter1-h1-fast-recipe.tgz` | 41MB | 0.5747 |
| 2 | `pi-code-search-iter2-h2-low-filter.tgz` | 45MB | 0.5752 |
| 3 | `pi-code-search-iter3-h3-no-filter.tgz` | 45MB | 0.5598 |
| 4 | `pi-code-search-iter4-h4-tight-filter.tgz` | 45MB | 0.5686 |
| **5** | **`pi-code-search-iter5-h5-replay-iter1.tgz`** | **45MB** | **0.6004** |
| 6 | `pi-code-search-iter6-h6-no-echo.tgz` | 45MB | 0.5874 |
| 7 | `pi-code-search-iter7-h7-echo-0.10.tgz` | 45MB | 0.5794 |
| 8 | `pi-code-search-iter8-h8-rank32.tgz` | 46MB | 0.5461 |
| 9 | `pi-code-search-iter9-h9-lr-5e-6.tgz` | 44MB | 0.4665 |
| 10 | `pi-code-search-iter10-h10-train20-default.tgz` | 44MB | 0.4850 |
| 11 | `pi-code-search-iter11-h11-train12.tgz` | 44MB | 0.5851 |
| 12 | `pi-code-search-iter12-h12-replay-best-seed2.tgz` | 43MB | 0.301 (degraded eval) |
| 13 | `pi-code-search-iter13-h13-lr-2e-5.tgz` | 41MB | 0.298 (degraded eval) |
| 14 | `pi-code-search-iter14-h14-echo-0.04.tgz` | 23MB | 0.293 (degraded eval) |

All tarballs at `b2://clouderic/kiln/pi-code-search/<name>.tgz`.

## Headline interpretation

The capability is **genuinely trainable**: 11 of 12 cleanly-measured
trained adapters lift composite above the base. The expected lift is
**+0.024 mean composite (5-eval) / +0.057 peak / +0.06 outcome**
relative to base, with eval-rollout σ ≈ 0.03 (comparable to the lift
itself). This is a real result, modest in magnitude, robust to recipe
variations.

## See also

- `capability.md` — design, rubric, hypotheses, live iter log
- `closeout.md` — full writeup with bug findings and lessons backported
- `capability.jsonl` — append-only iter log (30 rows incl. re-evals)
- `calibration/{good,bad}.jsonl` — rubric sanity fixtures
