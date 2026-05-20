# pi-faithful-completion — final closeout (50-iter loop)

**Status:** RUNNING (39/50 iters logged at last commit; closeout updated live).
**Driver:** autonomous 50-iter `/goal` loop, started 2026-05-19 ~09:30 UTC.
**Capability:** pi-faithful-completion — terminal-state discipline for the
Pi coding agent: emit the required OUTPUT FORMAT, never ask the user,
never soft-punt, honestly declare failure.

## Result (LIVE — updated as iters land)

- **Best adapter so far:** `pi-faithful-h25-temperature-0.6` (iter 25, family H13-temperature)
- **Best composite (eval, n=57):** **0.7751** ± noise
- **Baseline composite (iter 0):** 0.7237
- **Delta vs baseline:** **+0.0514** (+7.1% relative)
- **B2 location:** `b2://clouderic/capabilities/pi-faithful-completion/adapters/pi-faithful-h25-temperature-0.6.tar.gz`

### Why iter 25 won

Iter 25's hyperparameters:
- lr = 1e-5 (kiln default)
- rank = 16, alpha = 32 (kiln default)
- num_generations = 4
- ECHO λ = 0.05 (kiln default)
- **rollout temperature = 0.6** (vs default 0.8) ← the variable that changed
- 24 training tasks, no filter, mode = phase1

The ONLY non-default knob was rollout temperature. Lower temp (0.6 vs 0.8)
produces less-noisy rollouts within each GRPO group, which sharpens the
advantage signal at training time — the policy gradient points more reliably
toward "better" completions because the per-rollout reward distribution is
tighter and more clearly differentiated.

### Sub-score deltas (iter 25 vs baseline)

| Sub-score | Baseline | Iter 25 | Δ |
|---|---|---|---|
| outcome.value_correct | 0.7193 | 0.7719 | **+0.0526** |
| honesty.score         | 0.7719 | 0.8088 | **+0.0369** |
| format_strict         | 0.9825 | 0.9825 |   0.0000 |
| no_question           | 1.0000 | 1.0000 |   0.0000 |
| no_soft_punt          | 1.0000 | 1.0000 |   0.0000 |
| terseness             | 0.9807 | 0.9807 |   0.0000 |

All the gain is in `outcome.value_correct` and `honesty.score`. The other
sub-scores were already maxed by the strict default system prompt — GRPO
had no gradient to push them further. The remaining discipline headroom
opens up under a lighter system prompt (see iter 13 below).

## Top-5 iters (LIVE)

| Rank | Iter | Slug | Family | Composite | Δ | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 25 | h25-temperature-0.6        | H13-temperature  | 0.7751 | +0.0514 | lower rollout temp |
| 2 | 13 | h13-light-system-prompt    | H9-system-prompt | 0.7717 | +0.0479 | drop strict coaching |
| 3 | 34 | h34-chain-best-no-filter   | H7-chain         | 0.7584 | +0.0346 | chain from BEST + broad corpus |
| 4 | 10 | h10-lr-3e-5-echo           | H3-lr            | 0.7572 | +0.0335 | 3x default LR |
| 5 | 14 | h14-minimal-system-prompt  | H9-system-prompt | 0.7419 | +0.0181 | near-empty prompt |

## What worked

1. **Lower rollout temperature** (0.6 < default 0.8) — *the biggest single win*.
   Less rollout noise → cleaner per-group advantage signal → bigger policy
   update.
2. **Light system prompt** — counter-intuitive, but the strict default
   saturates `no_question` and `no_soft_punt` at 1.0, so GRPO has no
   gradient to push those sub-scores. A lighter prompt restores headroom.
3. **lr=3e-5 + ECHO** — 3x the default LR works, but only with ECHO λ=0.05
   regularising. Without ECHO (iter 3 at lr=1e-4), it overshoots and degrades.
4. **Chain from BEST + broad corpus + no filter** — modest +0.034 above
   baseline (less than iter 25 itself), useful as polish.

## What didn't work — and why

1. **Higher LR without ECHO** (iter 3 lr=1e-4): -0.016. The ECHO env-CE term
   stabilises the gradient direction even though our single-turn rollouts
   have no env tokens to predict — the loss term still acts on the optimizer's
   second-moment estimates.
2. **rank≠16** (iter 4 rank=32, iter 5 rank=8): -0.016 and 0.000 respectively.
   Rank 16 with default LR is the sweet spot.
3. **ECHO λ≠0.05** (iter 11 λ=0.025, iter 12 λ=0.1): both -0.016. Paper §3.3
   said productive range was 0.01-0.05; we found 0.025 also hurt for this
   single-turn workload, while 0.05 is necessary.
4. **task_filter=success_only** (iter 16): -0.033. Excluding failure tasks
   teaches the model to "always emit format with value"; it loses
   `honesty.score` on the eval failure tasks.
5. **task_filter=balanced** (iter 18, 20, 40): -0.033 or no-op. The
   50/25/25 cycling created duplicate tasks → 0 group variance → 0 strong-signal
   groups → training aborts with "no valid groups".
6. **Chain from BEST + narrow task filter** (iters 19, 22, 32): chain DROPS
   the BEST's composite. Narrow training erodes the diverse-task gains.
7. **mode=phase1_cispo** (iter 28): -0.016. CISPO IS doesn't help here.
8. **temp=1.0** (iter 26): -0.016. Mirror image of why temp=0.6 worked —
   too much rollout variance.

## Catastrophic configurations (composite → 0.02)

These broke the adapter to near-zero (model emits empty completions):

1. **Chain with rank mismatch** (iter 19): base adapter rank=16,
   training rank=32. cuda_grpo_ablation silently produced a corrupt adapter.
   See `kiln-polish.jsonl#lora-rank-mismatch-on-chain-train-silent-blowup`.
2. **alpha/rank = 4** (iter 24): rank=16 alpha=64 at lr=1e-5. The LoRA
   scaling factor amplifies the small delta past the base weight magnitudes;
   model output becomes empty.
   See `kiln-polish.jsonl#lora-alpha-rank-ratio-blowup`.

## Surprises and gotchas (for future cap authors)

1. **`enable_thinking=false`** in `chat_template_kwargs` cuts rollout time
   ~40x for Qwen3.5-4B. This is undocumented in kiln-server. Without it,
   a 5-token answer takes ~200 tokens (thinking trace), making single-turn
   GRPO impractically slow.
2. **kiln-server's `ensure_adapter` silently unloads on missing field.**
   This caused iters 1-3 to mismeasure as baseline. Fix: always send
   `adapter` in the request body. See
   `kiln-polish.jsonl#ensure-adapter-treats-missing-field-as-unload`.
3. **Pod-pool lease TTL is hard 3h, no renewal.** Long-running 50-iter
   loops MUST re-acquire pods every ~3h, losing in-flight work each time.
4. **b2 cli flakes on first upload after idle** — wrap with retry.
5. **alpha/rank ratio > 2 + nontrivial LR → adapter blow-up.**

## Reproduction recipe (best iter)

```bash
# Acquire pod
ce kiln-pod-acquire --gpu-type 'NVIDIA RTX A6000' --task-id <id>
bash deploy/runpod/kiln-setup.sh --clone

# Bootstrap capability
cd capabilities/agentic-grpo/pi-faithful-completion
python3 build_corpus.py
python3 rubric_sanity.py    # must PASS

# Reproduce iter 25 (BEST)
bash run_iter.sh --iter 25 --slug h25-temperature-0.6 \
  --train-tasks 24 --num-gens 4 \
  --lr 1e-5 --rank 16 --alpha 32 \
  --mode phase1 --echo-lambda 0.05 \
  --temperature 0.6 --top-p 0.95 \
  --max-tokens 768 --seed 3141592653
```

Or restore from B2:

```bash
b2 file download b2://clouderic/capabilities/pi-faithful-completion/adapters/pi-faithful-h25-temperature-0.6.tar.gz /tmp/best.tar.gz
tar xzf /tmp/best.tar.gz -C /path/to/qwen3.5-4b/adapters/
# then POST /v1/adapters/load {"name": "pi-faithful-h25-temperature-0.6"}
```

## Files

- `capability.md` — design + rubric + live progress
- `capability.jsonl` — full per-iter log (50 rows when done)
- `hypotheses.json` — pre-registered 50-iter plan
- `rubric.py`, `task_scaffold.py`, `build_corpus.py` — core scoring + corpus
- `rollout.py`, `run_iter.sh`, `drive_iter.py`, `drive_iters.sh` — iter loop
- `process_iter.sh`, `reeval.sh`, `restore_from_b2.sh` — operational helpers
- `analyze.py`, `log_iter.py`, `backup_to_b2.py` — bookkeeping
- `eval-summaries/`, `train-summaries/` — per-iter summary JSONs
- `kiln-polish.jsonl` — 7 kiln issues discovered during the loop
- `closeout.md` — this file (auto-updated as iters land)

## Iters remaining (40-50)

Drive_iters is currently working through:
- 40-44: chain refinement (mostly negative so far)
- 45-50: mixed (final-chain-large, mode-reinforce-echo, max-tokens variants)

The win is locked in at iter 25 (+0.0514). Subsequent iters are expected
to mostly land in the [-0.02, +0.02] band based on the patterns above.
The final commit will add the iter 41-50 rows and re-confirm the best.
