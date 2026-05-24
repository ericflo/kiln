# H73 H33 Then Mixed Suffix

## Hypothesis

H33 was the strongest earlier reliability signal: it passed a `LIMIT=8` gate
with outcome 1.0 and no zero rollouts, but it was slower and used more tool
calls. H58 later showed that mixed suffix data could produce a strong smoke win
but failed a wider gate from base. H73 tested the composition lesson from
`pi-faithful-completion`: use H33 as a reliability base, then apply a low-dose
mixed suffix stage to recover efficiency without losing outcome reliability.

The falsifiable prediction was that H33 would stabilize the H58-style suffix
update, preserving the no-zero reliability while improving or at least not
worsening latency.

## Data

Base adapter:
`pi-doctest-h33-hardneg-g2-noecho-r4a8`.

Suffix-stage dataset:
`/tmp/pi-doctest-h58-mixed-reliability-efficiency/grpo-train.mixed-reliability-efficiency.g2x3.jsonl`.

The suffix data used six one-action groups from two train-only H54 success
anchors:

- post-read: correct `edit` action versus premature `DONE`, reward 1.0 vs 0.60;
- post-edit: doctest `bash` action versus premature `DONE`, reward 1.0 vs 0.60;
- post-pass: concise `DONE` versus verbose `DONE`, reward 1.0 vs 0.85.

Dry-run shape was 6 groups, 12 completions, reward stdev 0.178924, 300 action
tokens, 0 env tokens, and 5422 context tokens. Dataset hash:
`sha256:5b260627f62a1aa5c27cab21921678411c998a479823ffae939662c3b16b4542`.

## Training

Training used `cuda_grpo_ablation --mode phase1`, `--base-adapter` pointing to
the H33 artifact, rank 4 / alpha 8 / lr `5e-8`, no ECHO, seed `3141592653`,
and `KILN_GRAD_CHECKPOINT_SEGMENTS=24`. It completed in 344.653s observed with
peak observed VRAM 15961 MiB. The receipt reported 340.461s wall-clock.

Output adapter:
`pi-doctest-h73-h33-then-mixed-suffix-r4a8lr5e8`.

Safetensors hash:
`sha256:e41ed9a35333d3fd8a71050e25a8e3779cf642f69ad7dd2ea39c8f38df0ec09f`.

## Verification

Adapter verify passed against the running server:

- Rank 4 / alpha 8.
- 400 LoRA tensors, 200 matched projection pairs.
- LoRA update proxy 0.256005.
- Server load succeeded for
  `pi-doctest-h73-h33-then-mixed-suffix-r4a8lr5e8`.

## Blind Eval

Smoke, `LIMIT=4 SEEDS=1`:

- Base: composite 0.8875, no zero rollouts, mean wall-clock 46.50s.
- H73: composite 0.75, one zero rollout, mean wall-clock 49.04s.
- Delta: -0.1375 composite, slower by 2.53s.

No wider promotion check was run.

## Verdict

Rejected at smoke. H33's reliability direction did not stabilize the H58 mixed
suffix update. The chain reintroduced a zero rollout and lost composite while
also running slightly slower. This falsifies the simple composition hypothesis:
an outcome-reliability base is not enough to make narrow suffix-stage
efficiency/reliability data safe.

Future composition attempts need a qualitatively stronger behavior anchor or a
different representation than single-action suffix ranking. Chaining rejected
or caveated policy adapters continues to amplify live search drift.

No eval task contents or per-example eval transcripts were inspected.
