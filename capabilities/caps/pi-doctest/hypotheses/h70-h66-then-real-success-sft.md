# H70 H66 Then Real-Success SFT

## Hypothesis

H66 showed that train-only validated ideal SFT was harmful when served as the
live thinking-enabled agent: it regressed score, introduced a zero rollout, and
more than doubled latency. H70 tested a small version of the
`pi-faithful-completion` oscillation lesson: after an idealized SFT step,
continue with a different distribution that anchors real successful behavior.

The falsifiable prediction was that continuing from H66 with train-only concise
base-success traces would preserve the useful ideal-workflow prior while
counteracting H66's mismatch to live agent rollouts.

## Training

Base adapter:
`pi-doctest-h66-direct-solver-sft-g8-r4a4lr5e8`.

Output adapter:
`pi-doctest-h70-h66-then-real-success-sft-r4a4lr5e8`.

Dataset:
`/tmp/pi-doctest-h59-concise-success-sft/sft.concise-success.g4.jsonl`.

The dataset contained four train-only concise successful traces from H59. The
rendered line lengths were 2148, 2386, 2626, and 2343 chars, and the trainer
tokenized them to 508, 676, 613, and 664 tokens.

Training used `cuda_sft_file` with the generic trainer because the native SFT
trainer does not yet support `--base-adapter`. Hyperparameters were rank 4 /
alpha 4 / lr `5e-8`, 1 epoch, seed `3235536621`, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`. The receipt reported 578 action tokens and
1883 context tokens. Observed elapsed time was 105.118s, with peak observed
VRAM 15985 MiB.

## Verification

Adapter verify passed against the running server:

- Rank 4 / alpha 4.
- 400 LoRA tensors, 200 matched projection pairs.
- LoRA update proxy 0.016049.
- Safetensors hash:
  `sha256:4cc718be261a35f59a242e5263a5e0f9a41eff44612d59943725296c3235d205`.
- Server load succeeded for
  `pi-doctest-h70-h66-then-real-success-sft-r4a4lr5e8`.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.953125, no zero rollouts, mean wall-clock 24.98s.
- H70: composite 0.934375, no zero rollouts, mean wall-clock 46.32s.
- Delta: -0.01875 composite, slower by 21.34s.

No wider promotion check was run.

## Verdict

Rejected at smoke. The chained real-success anchor recovered most of H66's
severe composite loss and avoided H66's zero rollout, but it still trailed the
paired base draw and almost doubled mean wall-clock. This suggests that H66's
ideal-SFT direction is not a useful first stage for pi-doctest mini-oscillation:
later real-success SFT can soften the damage, but does not turn the chain into a
kept adapter.

Future SFT work should not continue from H66 unless there is a confirmed
stabilizing stage first. The more promising direction is still qualitatively new
data or reward structure that preserves base reliability directly.

No eval task contents or per-example eval transcripts were inspected.
