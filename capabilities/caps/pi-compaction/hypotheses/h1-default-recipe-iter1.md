# Hypothesis H1 — Phase 1 GRPO recipe, iter 1

**Family:** H1 (default kiln GRPO recipe, lr 1e-5, rank 16, 1 epoch)
**Target sub-score:** `composite` (multi-component rubric v1.2)
**Pre-registered before knowing iter 0 final**

## Claim

Training Qwen3.5-4B against the pi-compaction v1.2 rubric using
kiln's Phase 1 GRPO defaults (DrGRPO, TokenLevel IS, K1 KL,
dynamic_sampling, Clip-Higher 0.20/0.28, lr=1e-5, rank=16, alpha=32,
seed=3141592653, 1 epoch) on **12 training tasks × 4 generations**
will lift composite by ≥+0.05 over the iter 0 baseline.

## Mechanism

Iter 0 baseline shows the base model has bimodal behaviour: ~40% of
eval tasks score ~0.77-0.80 (passes format + content + faith gates);
~60% score the failed-gate floor of 0.07.

The variance is concentrated in *format compliance*. When the base
model happens to emit Markdown headings, the gate passes and content
scoring takes over. GRPO should be able to bias the policy toward the
format-passing region — the gradient signal is strong (variance ≈
0.4 across the 4 generations within most groups).

## Recipe

- **Train tasks:** 12 tasks from `datasets/train.tasks.jsonl`,
  filtered to `len(source_text) < 100_000` so rollout time per task
  is bounded at ~60s.
- **Generations:** 4 per task = 48 rollouts total
- **Concurrency:** 4 (H100 should handle it; ~3 batches of 16 rollouts
  in flight at any time)
- **Strong-signal filter:** keep groups with var > 0.05
- **GRPO step:** Phase 1 defaults via `cuda_grpo_ablation`

## Falsification threshold

- composite uplift ≥ +0.05 → kept, plan iter 2
- composite uplift < +0.02 → falsified, switch hypothesis family
- composite regression < −0.05 → adapter discarded, re-baseline

## Wall-clock budget

- Rollouts: 12 tasks × ~50s × 4 gens / 4 conc = ~10 min
- Strong-signal filter: <1 min
- GRPO step: ~3 min
- Eval (23 tasks × 1 gen / 2 conc): ~10 min
- Total: ~24 min

## Results

[Filled in after run]
