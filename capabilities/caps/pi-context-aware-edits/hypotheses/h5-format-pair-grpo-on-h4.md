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

Rejected.

Dry-run accepted all 12 groups and 24 completions:

- rewards: mean 0.5, stdev 0.5, no degenerate groups;
- token plan: 876 action tokens, 0 env tokens, 3624 context tokens;
- base adapter: `pi-context-aware-edits-h4-ideal-trace-sft-r4a8-lr5e6`.

Training completed locally with CUDA 13.2 and explicit checkpointing:

- adapter: `pi-context-aware-edits-h5-format-pair-grpo-on-h4-r4a8`;
- `rank=4`, `alpha=8`, `lr=5e-6`, seed `3141592653`;
- `KILN_GRAD_CHECKPOINT_SEGMENTS=32`;
- 12 groups / 24 completions trained;
- peak observed VRAM: 17,301 MiB;
- adapter smoke test passed;
- installed under `/home/ericflo/.cache/kiln/adapters`;
- verification status: OK, LoRA update proxy `1.938045`.

Blind 3-seed eval over 12 tasks / 36 rollouts:

- composite: 0.4667 (`delta=-0.0133` vs. baseline 0.4800);
- stdev: 0.4936;
- `format_compliance`: 0.6806, recovered above baseline 0.6528;
- `outcome`: 0.6389, below baseline 0.6944 and below H4 0.7222;
- `convention_consistency`: 0.9583;
- `read_before_edit`: 1.0000;
- mean tool calls: 5.03;
- thinking chars/tool call: 301.0.

This falsifies the hypothesis. The isolated final-sentence signal moved the
format gate in the intended direction, but the H4 outcome lift disappeared and
the composite did not meet the `+0.05` promotion gate. No eval task contents or
per-example eval transcripts were inspected.
