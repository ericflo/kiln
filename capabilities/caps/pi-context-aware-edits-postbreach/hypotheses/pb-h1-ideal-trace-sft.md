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

Pending.
