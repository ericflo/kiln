# PB-H1: Verifier-Backed Ideal Trace SFT

## Hypothesis

The fresh postbreach baseline is borderline-low because the model often either
does not complete the edit correctly (`outcome=0.5000`) or completes the edit
without the required final response (`format_compliance=0.5625`). A gentle SFT
bootstrap on train-only ideal traces should lift both axes by teaching the
full read/edit/verify/final rhythm on the fresh mixed-language distribution.

## Recipe

- Data: `datasets/sft.pb-h1-ideal-traces.jsonl`, generated from
  `datasets/train.tasks.jsonl` only.
- Prepared data: 32 verified examples, 8 per profile, SHA256
  `0516987eddd2b103857e29e948293f6693f464e8ae96e962215c302e5a6b692e`.
- Trace shape: read target file, full-file edit, run verifier, final sentence
  naming the modified file and preserved conventions.
- Trainer: `cuda_sft_file`, generic trainer, `rank=4`, `alpha=8`,
  `lr=5e-6`, 1 epoch, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Verify adapter after training and run full blind 3-seed eval.

## Falsification

Reject if any of:

- composite lift < +0.05 versus postbreach baseline;
- `outcome <= 0.50`;
- `format_compliance <= 0.5625`;
- `convention_consistency < 0.90`;
- thinking chars/tool call > 25% above baseline (`>386`);
- adapter verification fails.

## Result

Rejected.

Training:

- adapter: `pi-context-aware-edits-postbreach-pb-h1-ideal-sft-r4a8`;
- 32 examples trained, 1 epoch;
- `rank=4`, `alpha=8`, `lr=5e-6`, seed `3235536621`;
- `KILN_GRAD_CHECKPOINT_SEGMENTS=32`;
- token counts: 1530 action tokens, 4644 context tokens;
- peak observed VRAM: 17,399 MiB;
- adapter verification: OK, LoRA update proxy `3.679127`.

Blind 3-seed eval:

- composite: 0.1954 (`delta=-0.1042` vs. baseline 0.2996);
- stdev: 0.3868;
- `outcome`: 0.5417, only a small lift over baseline 0.5000;
- `format_compliance`: 0.4583, below baseline 0.5625;
- `convention_consistency`: 0.9306, below baseline 0.9736;
- `read_before_edit`: 0.9583, below baseline 1.0000;
- mean tool calls: 5.85, above baseline 5.46;
- thinking chars/tool call: 317.3, above baseline 308.8;
- mean wall-clock: 46.0s, above baseline 20.8s.

This falsifies the hypothesis. Ideal traces taught a bit of edit completion,
but they damaged the final-response contract and made the agent less efficient.
This repeats the prebreach H4 pattern and argues against more single-
distribution SFT. No eval task contents or per-example eval transcripts were
inspected.
