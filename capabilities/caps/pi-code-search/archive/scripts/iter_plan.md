# pi-code-search — iteration plan (50 iters)

Compact map of hypothesis families across the loop. Update as we go.

## Iter 0–2: baselines and infra

| Iter | Slug                     | Goal |
|------|--------------------------|------|
| 0    | baseline-base            | Eval base model on eval set; establish composite floor. |
| 1    | h1-default-recipe        | Vanilla ECHO+GRPO with 25 strong train tasks × 4 gens, lr=1e-5. |
| 2    | h1b-replay-baseline      | Repeat base eval with a different shuffle seed — measure eval variance. |

## Iter 3–10: training-data quality

| Iter | Slug | Hypothesis |
|------|------|------------|
| 3    | h2-strong-signal         | Filter to groups with reward var ≥ 0.04 (`FILTER_VAR=0.04`). |
| 4    | h3-drop-easy             | Drop top 20% by mean reward (`FILTER_HEAD_FRAC=0.20`). |
| 5    | h4-defines-only          | Train only `define:` tasks (refs are harder, dilute signal). |
| 6    | h5-refs-only             | Train only `refs:` tasks (more headroom). |
| 7    | h6-balanced              | 50/50 define+refs; balanced corpus. |
| 8    | h7-large-corpus          | 60 train tasks × 4 gens (more groups). |
| 9    | h8-small-corpus          | 12 train tasks × 8 gens (more reps per task). |
| 10   | h9-2-epoch               | 2 epochs over filtered set (probably regresses). |

## Iter 11–20: training hyperparams

| Iter | Slug | Hypothesis |
|------|------|------------|
| 11   | h10-lr-3e-5              | lr=3e-5 (more aggressive). |
| 12   | h11-lr-5e-6              | lr=5e-6 (more conservative). |
| 13   | h12-rank-32              | LoRA rank=32 (double capacity). |
| 14   | h13-rank-8               | LoRA rank=8 (half capacity). |
| 15   | h14-kl-0p05              | KL coeff = 0.05 (lighter regularization). |
| 16   | h15-kl-0p2               | KL coeff = 0.20 (heavier regularization). |
| 17   | h16-echo-0p075           | ECHO lambda = 0.075 (paper upper band). |
| 18   | h17-echo-0p10            | ECHO lambda = 0.10 (paper edge). |
| 19   | h18-no-echo              | Disable ECHO (control: does ECHO help here?). |
| 20   | h19-clip-0p1             | Clip epsilon = 0.10 (tighter trust region). |

## Iter 21–30: rubric variants

| Iter | Slug | Hypothesis |
|------|------|------------|
| 21   | h20-rubric-grounding-stronger | Grounding floor 0.2 instead of 0.40 (stricter). |
| 22   | h21-rubric-efficiency-weight  | Eff weight 0.70 (less tool_choice). |
| 23   | h22-rubric-no-tool-choice     | Drop tool_choice (only eff + grd + fmt). |
| 24   | h23-line-tol-0                | Line tolerance 0 (exact match required). |
| 25   | h24-line-tol-5                | Line tolerance ±5 (more lenient). |
| 26   | h25-target-bytes-tight        | target_bytes ÷ 2 (tighter efficiency budget). |
| 27   | h26-target-bytes-loose        | target_bytes × 3 (looser budget). |
| 28   | h27-no-format-component       | Drop format sub-score. |
| 29   | h28-no-loop-penalty           | Add no-loop sub-score (penalize repeat calls). |
| 30   | h29-anti-thrash-bonus         | Bonus when n_tool_calls ≤ 3. |

## Iter 31–40: corpus / task design

| Iter | Slug | Hypothesis |
|------|------|------------|
| 31   | h30-symbol-difficulty | Stratify corpus by symbol length (short = harder). |
| 32   | h31-rust-only         | Restrict to .rs files. |
| 33   | h32-mixed-langs       | Add Python files (.py) to corpus. |
| 34   | h33-long-symbols      | Train only on symbols ≥ 20 chars. |
| 35   | h34-short-symbols     | Train only on symbols ≤ 10 chars. |
| 36   | h35-corpus-2x         | Build a fresh, bigger corpus (200 train tasks). |
| 37   | h36-shuffle-seed-1    | Same iter 5 recipe, different shuffle seed. |
| 38   | h37-shuffle-seed-2    | Yet another shuffle seed. |
| 39   | h38-recipe-replay-best | Replay the best iter so far on fresh rollouts. |
| 40   | h39-multi-seed-mean   | Train 3 adapters at best recipe, average eval. |

## Iter 41–50: composition + closure

| Iter | Slug | Hypothesis |
|------|------|------------|
| 41   | h40-best-plus-2epoch  | Best recipe + 2 epochs. |
| 42   | h41-best-plus-rank-32 | Best recipe + rank 32. |
| 43   | h42-best-plus-no-echo | Best recipe + no ECHO (ablation). |
| 44   | h43-best-plus-strict-grounding | Best + 0.20 grounding floor. |
| 45   | h44-curriculum-easy-then-hard | First epoch easy tasks, second hard. |
| 46   | h45-replay-best-recipe-seed1 | 2nd-seed verify best. |
| 47   | h46-replay-best-recipe-seed2 | 3rd-seed verify best. |
| 48   | h47-best-plus-format-strict | Strict format check (require code block). |
| 49   | h48-shipping-candidate | Lock the recipe; this is what we ship. |
| 50   | h49-eval-double-n | 2× eval generations for final variance bound. |

## Stop conditions

- ≥ +0.10 composite lift vs baseline, verified across ≥ 2 seeds → consider closing early.
- ≤ 5 consecutive iters with no movement past prior best → consider closing.

## Note on adapter naming

Each adapter is `pi-code-search-iter<N>-<slug>`. Best so far: tracked in
`capability.jsonl` last row's `composite` (compared against all prior).
