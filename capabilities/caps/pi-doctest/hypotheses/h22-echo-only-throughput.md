# H22: ECHO-only throughput probe

## Hypothesis

H19 and H20 showed that tiny SFT pressure can create false-positive smoke
lifts, and H21 rejected prompt-only variants. H22 tests a qualitatively
different signal: verifier-free agentic training with policy loss disabled,
so only ECHO env-token prediction is trained.

This should be safer than policy-gradient GRPO on the current data because
the selected train groups are reward-saturated. The intended effect is to
improve the model's conditioning on tool outputs, especially the passing
doctest output, without directly pushing on action-token style.

## Data

Dataset: `/tmp/pi-doctest-h17-action650-success-anchors/grpo-train.jsonl`.

Dry-run command used `cuda_grpo_ablation` with:

- `--no-policy-loss`
- `--echo-lambda 0.05`
- rank 4 / alpha 8 / lr `5e-6`
- no reward variance filter

Dry-run passed:

- 2 groups.
- 7 completions.
- 2584 action tokens.
- 2201 env tokens after warning filtering.
- 2002 context tokens.
- `no_policy_loss=true`.

The dry-run emitted the expected saturation warning: 2/2 groups were all-pass
or all-fail enough that policy-gradient GRPO would be risky. That warning
supports the ECHO-only choice.

## Training Attempt

The offline CUDA example was built from a clean detached worktree because the
main working tree's unrelated `Cargo.lock` change currently makes cargo reject
the lockfile as containing a duplicate `kiln-graph` package.

Before training, the kiln server was stopped to free VRAM, then restarted after
the attempt. The server returned healthy with `default_thinking_enabled=true`
and no active adapter.

Training command used the same data and recipe as the dry-run, plus
`--adapter-smoke-test` and install into `Qwen3.5-4B/adapters`.

Result: aborted by the 900s wrapper without a completed training step and
without an adapter artifact. The log reached model load and the startup
saturation warning, then produced no progress line. GPU utilization was near
100% and memory near the 16GB limit throughout.

After that, an explicit gradient-checkpointed retry was run because the first
offline command did not force the server's `KILN_GRAD_CHECKPOINT_SEGMENTS=24`
setting into the standalone example process.

Retry command added:

- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`
- `KILN_PROFILE_CHECKPOINT_SEGMENTS=1`
- `RUST_LOG=info`
- `--max-groups 1`

The retry proved that checkpointing was active:

`streamed GRPO gradient checkpointing enabled num_segments=24`

It got substantially farther than the first attempt, but still timed out at
900s without an adapter artifact. The first completion's backward pass took
270214 ms, the second took 177960 ms, and the run timed out after the third
completion's policy forward completed and its backward pass began. The
per-segment logs identified the slow path: lower/mid checkpoint segments
around layers 4-14, especially segments spanning layers 12-14, 8-10, and 4-6,
can take tens to hundreds of seconds.

## Verdict

Reject H22 as a local training route in this form. ECHO-only is methodologically
better matched to saturated agentic data than more SFT, but the local backward
cost is still too high on this laptop even with explicit 24-segment gradient
checkpointing on one H17 group.

Future ECHO-only attempts need a smaller synthetic or mined environment-only
dataset, likely one group or one completion at a time, before trying a full
adapter. Another option is to make the trainer support an env-only microbatch
path, or a checkpointed GRPO path that can skip policy/reference work when
`--no-policy-loss` is set, so ECHO can be trained without carrying all
action-token sequence cost.

No eval task contents or per-example eval transcripts were inspected.
