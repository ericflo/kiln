# H15: local H12 server replay

## Hypothesis

Replaying the round-1 H12 strong-signal recipe on this WSL RTX 4090 Mobile setup, using thinking-on Kiln serving, the server-native `/v1/train/grpo` path, and the archived 12 train-task strong-signal subset, will reproduce the stable pi-doctest uplift family while reducing unnecessary tool/thinking overhead: composite near 0.89, lower rollout variance than base, and no worse than one zero-outcome rollout on the 24-task blind eval.

## Falsification Plan

If the trained adapter's 3-generation blind eval is within baseline noise or below baseline mean, the local server-native GRPO path or current base/runtime changed the recipe enough that H12 no longer transfers. If composite improves but mean wall-clock, zero-outcome count, or mean thinking chars per successful rollout worsens materially, treat the adapter as an ablation rather than a kept stage.

## Expected Magnitude

Expected composite lift is +0.03 to +0.06 against the local 3-generation thinking-on baseline, with lower `composite_stdev`, `outcome >= baseline`, and no material increase in `mean_tool_calls` or `mean_thinking_chars_per_tool_call`.

## Recipe

- Rollout data: `datasets/train.tasks.jsonl` filtered by `datasets/h12_strong_signal.task_ids`
- Generations: 4 per task
- Method: agentic-GRPO with ECHO
- Training: server-native `/v1/train/grpo`
- Hyperparameters: rank 16, alpha 32, lr 1e-5, reward filter variance min 0.05, one pass
- Server policy: `KILN_DEFAULT_THINKING_ENABLED=true`; rollout.py leaves Pi `--thinking` unset so Kiln controls the binary thinking mode. Pi uses the `qwen-3.5-4b-kiln-pi1024` alias so each turn has a 1024-token ceiling instead of the default 32768-token provider ceiling.
- Eval: patched `./capability.oracle.sh` with `SEEDS=3` against the blind eval pool; v1 composite unchanged, with aggregate thinking-length diagnostics tracked out-of-band.

## Lessons Applied

- From `pi-doctest` round 1: strong-signal filtering was robust; more groups or extra epochs overtrained.
- From `pi-code-comprehension`: ECHO helps agentic caps, but chaining SFT/self-distillation on agentic traces can destabilize sub-scores.
- From `pi-faithful-completion`: successful chains often alternate complementary data distributions gently; if H15 is positive, the next experiment should be an oscillating chain that alternates H12 strong-signal GRPO with a light SFT/GRPO regularizer on efficient successful train rollouts, stopping before over-chain.
