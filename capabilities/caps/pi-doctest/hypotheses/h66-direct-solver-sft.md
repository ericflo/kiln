# H66 Direct-Solver Ideal SFT

## Hypothesis

H59-H62 showed that tiny SFT corpora are unstable, but H60's synthetic ideal
data at least avoided immediate zero-rollout and latency collapse. H66 expands
that idea: generate a broader train-only ideal workflow corpus by asking the
base model for direct function bodies, validating those bodies with doctest,
and rendering the successful implementations as concise read/edit/doctest/DONE
tool traces.

The falsifiable prediction was that broader, validated teacher-style ideal
data would be safer than hand-authored four-example SFT or base-success-copy
SFT.

## Data

- Source tasks: `capabilities/caps/pi-doctest/datasets/train.tasks.jsonl`
  only.
- Direct-solver generation: local base model, non-reasoning direct body
  requests via `chat_template_kwargs.enable_thinking=false`, then independent
  doctest validation.
- Output: `/tmp/pi-doctest-h66-direct-solver-sft/sft.direct-solver-ideal.g8.jsonl`.
- Examples: 8.
- Task IDs: `task_0024` through `task_0031`.
- Selected body lengths: 43, 123, 121, 308, 36, 28, 143, 93 chars.
- Rendered trace: read `solution.py`, edit the `raise NotImplementedError`
  body, run `python3 -m doctest -v solution.py`, then `DONE`.

The rendered corpus had 15,336 total chars and 4,901 assistant-action chars.
Tokenized training examples ranged from 487 to 615 tokens. The SFT receipt
reported 1,445 action tokens and 3,001 context tokens.

## Training

Adapter: `pi-doctest-h66-direct-solver-sft-g8-r4a4lr5e8`.

Training used `cuda_sft_file --trainer native`, rank 4 / alpha 4, lr `5e-8`,
1 epoch, and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

It completed successfully in 164.923s observed with peak observed VRAM 15986
MiB. Adapter verify passed with 400 nonzero LoRA tensors and a LoRA update
proxy of 0.011464.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.990625, no zero rollouts, mean wall-clock 26.05s.
- H66: composite 0.7125, one zero rollout, mean wall-clock 62.74s.
- Delta: -0.278125 composite, slower by 36.69s.

No wider promotion check was run.

## Verdict

Rejected at smoke. The useful result is systems-side: direct-solver validated
SFT data is cheap to create and train locally. The capability result is
negative: even broader ideal SFT with a lower learning rate still pushes the
thinking-enabled agent into slower and less reliable behavior.

This strongly suggests that SFT on idealized tool traces is mismatched to the
current pi-doctest failure mode. Next experiments should avoid further
positive-only ideal SFT on base unless it is part of a different chain with a
confirmed stabilizing base. More promising routes are prompt/harness-level
controls, stronger external teacher data, or reward learning that explicitly
preserves successful base behavior.

No eval task contents or per-example eval transcripts were inspected.
