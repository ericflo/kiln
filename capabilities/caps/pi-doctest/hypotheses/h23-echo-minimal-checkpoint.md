# H23: Minimal checkpointed ECHO probe

## Hypothesis

H22 showed that ECHO-only training is methodologically reasonable for saturated
agentic data, but the local CUDA GRPO path is too slow on the full H17 train
batch even with 24 checkpoint segments. H23 tests whether the smallest viable
ECHO microbatch can produce an adapter locally.

Because `--no-policy-loss` removes the policy-gradient term, the hoped-for
minimum was one group with one high-quality completion. If the trainer accepts
that shape, ECHO could be trained one environment trace at a time without
paying for unnecessary preference-group size.

## Data

Source dataset:
`/tmp/pi-doctest-h17-action650-success-anchors/grpo-train.jsonl`.

The first probe selected the highest-reward short completion from the H17 data:

- source group index: 0
- source completion index: 1
- reward: 0.9871964285714285
- completion text chars: 987
- output: `/tmp/pi-doctest-h23-echo-one-completion/grpo-train.one.jsonl`

Dry-run rejected it before training with `failure_reason=zero_groups`. The
dynamic GRPO filter still requires a non-degenerate reward group before ECHO
can run, even when `--no-policy-loss` is set.

The second probe selected the smallest accepted group shape: one group with two
completions from the same source task, chosen for reward spread:

- source group index: 1
- source completion indices: 0 and 2
- rewards: 0.953625 and 0.9814821428571429
- mean reward: 0.9675535714285715
- reward range: 0.027857142857142914
- completion text chars: 3542
- output: `/tmp/pi-doctest-h23-echo-two-completion/grpo-train.two.jsonl`

Dry-run command used:

- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`
- `--no-policy-loss`
- `--echo-lambda 0.05`
- rank 4 / alpha 8 / lr `5e-6`

Dry-run passed:

- 1 valid group.
- 2 valid completions.
- 896 action tokens.
- 759 env tokens after warning filtering.
- 598 context tokens.
- reward stdev 0.013928571428571401.

## Training Attempt

The kiln server was stopped to free VRAM and restarted immediately after the
attempt. The restarted server returned healthy with
`default_thinking_enabled=true`, no active adapter, and
`KILN_GRAD_CHECKPOINT_SEGMENTS=24`.

Training used the two-completion dry-run data with:

- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`
- `KILN_PROFILE_CHECKPOINT_SEGMENTS=1`
- `RUST_LOG=info`
- `--no-policy-loss`
- `--echo-lambda 0.05`
- rank 4 / alpha 8 / lr `5e-6`
- 900s timeout wrapper

Checkpointing was confirmed in the streamed GRPO path:

`streamed GRPO gradient checkpointing enabled num_segments=24`

The run made real progress but timed out without producing an adapter artifact.
The first completion's backward pass completed in 196290 ms. The second
completion's policy forward alone then took 425734 ms; the run timed out after
the second backward pass started.

VRAM stayed near the local 16GB limit while GPU utilization remained near 100%,
so checkpointing solved the memory pressure but turned the current ECHO path
into a recompute-heavy wall-clock bottleneck.

## Verdict

Reject H23 as a local training route. Gradient checkpointing is active and
necessary, but it is not sufficient: the current `--no-policy-loss` path still
pays policy/reference forward and policy backward costs instead of training a
cheap env-token-only CE microbatch.

The next useful iteration should be a trainer change, not another data-size
tweak: when `--no-policy-loss` is set and only ECHO loss is active, skip
reference-policy work and avoid action-token policy loss plumbing. If that is
too invasive, add a separate env-only ECHO trainer path for trajectory
environment tokens.

No eval task contents or per-example eval transcripts were inspected.
