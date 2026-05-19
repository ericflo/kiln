# pi-faithful-completion — final closeout (50-iter loop)

**Status:** _filled by the loop driver after iter 50_
**Driver:** autonomous 50-iter `/goal` loop, 2026-05-19→…
**Capability:** pi-faithful-completion — terminal-state discipline for the
Pi coding agent: emit the required OUTPUT FORMAT, never ask the user,
never soft-punt, honestly declare failure.

## Result (TO BE FILLED)

- **Best adapter:** `pi-faithful-<slug>`  (iter `<N>`, family `<F>`)
- **Best composite (eval, n=57):** `<X.YY>` ± `<noise>`
- **Baseline composite:** 0.7237
- **Delta vs baseline:** `<+X.YY>` ( `<percent>`% relative)
- **B2 location:** `b2://clouderic/capabilities/pi-faithful-completion/adapters/pi-faithful-<best-slug>.tar.gz`

## TL;DR

_3-4 sentence summary of what worked, what didn't, why._

## Sub-score deltas (best vs baseline)

| Sub-score | Baseline | Best | Δ |
| --- | --- | --- | --- |
| outcome.value_correct | 0.7193 | _TBD_ | _TBD_ |
| honesty.score        | 0.7719 | _TBD_ | _TBD_ |
| format_strict        | 0.9824 | _TBD_ | _TBD_ |
| no_question          | 1.0000 | _TBD_ | _TBD_ |
| no_soft_punt         | 1.0000 | _TBD_ | _TBD_ |
| terseness            | 0.9807 | _TBD_ | _TBD_ |

## Top-5 best iters

| Rank | Iter | Slug | Family | Composite | Notes |
| --- | --- | --- | --- | --- | --- |
| 1 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 2 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 3 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 4 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |
| 5 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | _TBD_ |

## What worked

_Filled from capability.jsonl analysis: family-level mean deltas, the
hypothesis families that yielded the most uplift, etc._

## What didn't

_The null/negative families, what we learned about the task shape from
them, and which proposals didn't pan out._

## Surprises

_Things we didn't predict — interactions between knobs, a sub-score
moving where we didn't aim at it, etc._

## Reproduction recipe

```
# Acquire pod
ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000' --task-id <id>

# Bootstrap kiln + model weights
bash deploy/runpod/kiln-setup.sh --clone

# Sync this capability dir to pod
python3 build_corpus.py
python3 rubric_sanity.py    # must PASS

# Reproduce best iter
cd capabilities/agentic-grpo/pi-faithful-completion
bash run_iter.sh --iter <N> --slug <best-slug> \
   <whatever the hypothesis row says...>
```

## Files

- `capability.md` — design + rubric + adversarial review
- `capability.jsonl` — full per-iter log (50 rows)
- `hypotheses.json` — pre-registered hypothesis plan
- `rubric.py`, `task_scaffold.py`, `build_corpus.py` — core scoring + corpus
- `rollout.py`, `run_iter.sh`, `drive_iter.py`, `drive_iters.sh` — iter loop
- `backup_to_b2.py`, `log_iter.py` — bookkeeping
- `eval-summaries/`, `train-summaries/` — per-iter summary JSONs
- `closeout.md` — this file
