# H4: Idealized Train-Only Trace SFT

## Hypothesis

H2 failed because it copied sampled Pi action text, including brittle execution
artifacts, into the supervised target. H3 showed that strict prompting can make
the workflow cheaper but also hurts `format_compliance` and `outcome`.

For this cap, the train tasks are small deterministic edit templates. Instead
of copying rollouts, synthesize compact ideal action traces from train tasks:
read the target file, edit the full file to a verifier-passing state, run the
provided verifier, and emit the required one-sentence final response. This
should improve `outcome` and `format_compliance` while keeping the assistant
span shorter and cleaner than H2.

## Recipe

- Source: `datasets/train.tasks.jsonl` only.
- No eval tasks or eval transcripts.
- Build 12 SFT examples: four per train profile, with short wording variants
  but deterministic tool-call structure.
- Assistant target format:
  `read` tool call -> `edit` tool call -> `bash` verifier call -> final
  sentence.
- Completion length target: under 2600 chars.
- Trainer: `cuda_sft_file`, generic SFT, `rank=4`, `alpha=8`, `lr=5e-6`,
  one epoch, `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.

The lower LR versus H2 (`5e-6` instead of `1e-5`) is intentional: the train
corpus has only three unique prompt families, so the overfit risk is higher
than the data volume suggests.

## Falsification

Reject or do not promote if the 3-seed blind eval shows any of:

- composite lift < +0.05 versus `baseline-0`;
- `outcome` or `format_compliance` fails to improve versus baseline;
- `convention_consistency` drops below 0.90;
- mean thinking chars/tool call increases by more than 25% over baseline;
- adapter verification fails.

## Result

Pending.
