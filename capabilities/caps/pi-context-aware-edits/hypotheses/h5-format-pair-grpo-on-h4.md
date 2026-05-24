# H5: Pairwise Final-Format GRPO on H4

## Hypothesis

H4 moved the load-bearing edit-completion axis in the right direction:
`outcome` improved from 0.6944 to 0.7222 and `read_before_edit` reached
1.0000. It still regressed composite because `format_compliance` fell to
0.5000.

The next adapter should not relearn the whole workflow. Build synthetic
pairwise GRPO groups from H4's ideal traces where the positive and negative
completion have identical read/edit/verify action text, and differ only in
the final assistant sentence:

- positive: names the relative file and says style/conventions were preserved;
- negative: generic final response such as `Done.`

With equal prefixes and rewards 1/0, group-relative advantages should mostly
cancel on the shared action tokens and concentrate gradient on the final
sentence. Chain this on top of H4 so its outcome lift is the starting point.

## Recipe

- Source: `datasets/sft.h4-ideal-traces.jsonl` only.
- Build 12 GRPO groups, one per H4 ideal trace, each with two completions.
- Rewards: positive final format = 1.0, generic final = 0.0.
- Trainer: `cuda_grpo_ablation`, `--base-adapter` H4, `rank=4`,
  `alpha=8`, `lr=5e-6`, `--no-echo`, `--filter-var-min=0.05`,
  `KILN_GRAD_CHECKPOINT_SEGMENTS=32`.
- Run `--dry-run` before GPU training.

## Falsification

Reject or do not promote if the 3-seed blind eval shows any of:

- composite lift < +0.05 versus `baseline-0`;
- `format_compliance` fails to recover above baseline;
- H4's `outcome` lift disappears (`outcome < 0.70`);
- `convention_consistency` drops below 0.90;
- mean thinking chars/tool call increases by more than 25% over baseline;
- adapter verification fails.

## Result

Pending.
