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

Status: rejected.

- Dataset: 12 verifier-backed examples, four per train profile. Completion
  lengths 1415-2025 chars. Dataset SHA:
  `sha256:49150da7b194b58b993f5369fa1d903e444708b7d20e9217c22245124a8ff085`.
- Training succeeded locally in 134s wall clock with observed peak VRAM
  17,026 MiB. Receipt: 12 examples trained, 570 action tokens, 1812 context
  tokens, rank 4, alpha 8, `lr=5e-6`, one epoch.
- Adapter verification passed with nonzero LoRA tensors and offline update
  proxy `lora_update_l2_upper_bound=1.447610`.
- Blind 12-task x 3-seed eval scored 0.3597 versus the 0.4800 baseline
  (`delta=-0.1203`, stdev 0.4785).
- Sub-score movement was informative: `outcome` improved from 0.6944 to
  0.7222 and `read_before_edit` improved to 1.0000, but
  `format_compliance` fell from 0.6528 to 0.5000. `convention_consistency`
  stayed at 0.9583.
- Thinking efficiency stayed essentially baseline: 4.86 tool calls,
  1477.2 thinking chars, 301.3 thinking chars/tool.

Conclusion: ideal traces carry useful edit-completion signal, unlike H2's
sampled rollout transcripts, but final-response format remains the dominant
failure. The next iteration should isolate the final-response contract from
tool-action imitation, for example a tiny final-turn-only or final-bookend SFT
stage chained after H4, or a GRPO arm that rewards only format on top of the
H4 completion lift.
