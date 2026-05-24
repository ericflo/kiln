# H25: Baseline-mode two-completion GRPO+ECHO

## Hypothesis

H24 proved that baseline-mode per-sample training is a viable local throughput
primitive, but its pure env-only ECHO signal worsened action efficiency. H25
turns policy loss back on for the smallest reward-spread group from H23 so the
update can directly prefer the better action trajectory while still avoiding
Phase 1's token-level group accumulation.

This tests whether old baseline GRPO, despite being less modern than Phase 1,
is the right local compromise for policy-side discipline on this 16GB laptop.

## Data

Dataset:
`/tmp/pi-doctest-h23-echo-two-completion/grpo-train.two.jsonl`.

Selected completions:

- source group index: 1
- completion indices: 0 and 2
- rewards: 0.953625 and 0.9814821428571429
- reward stdev: 0.013928571428571401
- action tokens: 896
- env tokens: 759
- context tokens: 598

Dry-run command used:

- `--mode baseline`
- policy loss enabled
- ECHO lambda `0.05`
- rank 4 / alpha 8 / lr `5e-6`
- `KILN_GRAD_CHECKPOINT_SEGMENTS=24`

Dry-run passed:

- 1 valid group.
- 2 valid completions.
- 896 action tokens.
- 759 env tokens after warning filtering.
- reward variance bucket: low.

The saturated all-pass warning still applies; this is a risky policy-gradient
signal, but it is the smallest available reward-spread group and tests whether
per-sample baseline GRPO can train where Phase 1 timed out.

## Training Attempt

The kiln server was stopped to free VRAM and restarted immediately after the
timeout. The restarted server returned healthy with
`default_thinking_enabled=true` and no active adapter.

Training command used:

- `--mode baseline`
- policy loss enabled
- `--echo-lambda 0.05`
- rank 4 / alpha 8 / lr `5e-6`
- 24 checkpoint segments
- 900s timeout wrapper

Result: timeout before adapter save.

The run made it through the first completion and performed an optimizer step,
but was too slow to finish the second completion:

- first completion sequence length: 1243
- first completion action tokens: 571
- first completion backward: 644457 ms
- slowest segment: layers 0-2 at 342198 ms
- other slow lower/mid segments included layers 2-4 at 77019 ms, 4-6 at
  33993 ms, 6-8 at 35625 ms, 8-10 at 31778 ms, and 12-14 at 36378 ms
- second completion sequence length: 1010
- timeout occurred after second policy forward started
- peak observed VRAM: about 15991 MiB

No adapter artifact was produced.

## Verdict

Reject H25 as a local training route at this sequence length.

Baseline/per-sample mode fixed H23's group-accumulation failure mode, but
policy-on training still makes lower-layer checkpoint reverse passes too slow
on 1000+ token trajectories. The next policy-side experiment must enforce a
much stricter sequence length/action-token cap before training. A plausible
target is sub-800 sequence length and sub-300 action tokens per completion,
using either mined concise successes or synthetic concise action traces.

No eval task contents or per-example eval transcripts were inspected.
