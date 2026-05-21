# Kiln CLI reference

The 11 kiln CLIs the round-3 layout depends on. All shipped as part of the
40-issue round-2 backlog (see
[`capabilities/KILN_IMPROVEMENT_ISSUES.md`](../../../../capabilities/KILN_IMPROVEMENT_ISSUES.md)).
Do not re-implement any of these in cap scripts.

## Serving

### `kiln serve --eval-mode` (issue 15)

Canonical eval-time server config:
- thinking disabled by default for tool agents (issue 17)
- adapter changes flagged as warnings (issue 16 stress-test)
- deterministic decode where possible
- `/v1/health` exposes p50/p95/p99 latency + last adapter (issue 13)

```bash
kiln serve --eval-mode \
  --model-path /workspace/Qwen3.5-4B \
  --adapter-dir /workspace/adapters &
```

## Adapter management

### `kiln adapter verify <name>` (issue 4)

Proves an adapter is loadable + behavioral. Exits non-zero on failure.
What it checks:
- Adapter directory layout: `adapter_config.json` + `adapter_model.safetensors`
- Config + safetensors tensor names consistent
- Loads through `/v1/adapters/load`
- Registry shows it as active after load
- Behavioral: logit-delta nonzero on a fixed prompt with adapter vs without

```bash
kiln adapter verify <cap>-stage-N-<slug> \
  --adapter-dir /workspace/adapters \
  --url http://localhost:8420
```

### `kiln adapter restore <manifest>` (issue 36)

Re-materializes an adapter from `adapter_manifest.json`. Verifies hashes.
Use as the canonical "restore from B2" path; don't write custom backup scripts.

```bash
kiln adapter restore /path/to/adapter_manifest.json --target /workspace/adapters/
```

### `kiln adapter list / load / unload`

Pre-existing. The issue-1 chat completion semantics make `load`/`unload`
the *only* state mutators; chat requests no longer silently change default.

## Trajectory inspection

### `kiln trajectory inspect <jsonl>` (issue 10)

Mask + token-count diagnostic. Outputs action_mask, env_mask, context_mask
counts and per-segment summaries. Use BEFORE every agentic stage's training
to confirm trajectories have non-empty action AND env masks.

```bash
kiln trajectory inspect /tmp/<cap>-iter-<N>/grpo-train.jsonl --json
```

JSON output is the round-3 canonical schema; pi-trajectory.py outputs the
same shape.

## Eval

### `kiln eval-adapter --adapter ... --seeds N` (issue 33)

The standard multi-seed paired eval driver. Defaults to `--seeds 3`. Writes
`eval_summary.json` with mean composite, stdev, per-sub-score breakdown,
sigma_warning, verdict.

```bash
kiln eval-adapter \
  --url http://localhost:8420 \
  --adapter "<cap>-stage-N-<slug>" \
  --adapter-dir /workspace/adapters \
  --tasks datasets/eval.tasks.jsonl \
  --seeds 3 \
  --scorer ./rubric.py \
  --output /tmp/<cap>-eval-stage-N.json \
  --thinking off
```

Paired mode means base and adapter are run against the same prompts on the
same seeds, so paired-eval delta is the trustworthy comparison.

### `kiln rollout --adapter ... --tasks ...` (issue 34)

Direct HTTP rollout, alternative to pi for single-turn tasks. Use when the
task can be expressed as `{prompt} → {completion}` without tool calls.

## Trainers

### `cuda_grpo_ablation` (multiple issues)

Single-turn and agentic GRPO. Key flags:

- `--dry-run` (issue 9) — pre-GPU validation. Run BEFORE every training step.
- `--filter-var-min 0.05` (issue 22) — strong-signal filter.
- `--on-empty-filter {fail,train-all,skip}` (issue 22) — empty-filter behavior.
- `--install-adapter-dir <dir> --install-adapter-name <name>` (issue 5) —
  atomic install into registry. No cap-side path arithmetic.
- `--adapter-smoke-test` (issue 19) — post-train sanity check.
- `--base-adapter <name>` (issue 6) — chain training from a prior adapter.
  Validates rank + target-module compatibility BEFORE optimizer setup.
- `--echo-lambda <f> --no-policy-loss --echo-env-mask-mode <mode>` — ECHO controls.

### `cuda_opd_remote` (issue 37)

OPD trainer. Requires live teacher server.

```bash
cuda_opd_remote \
  --prompts datasets/opd.prompts.jsonl \
  --model /workspace/Qwen3.5-4B \
  --teacher-url http://localhost:8002 \
  --teacher-name qwen3.6-27b-awq \
  --base-adapter <prev-stage-adapter> \   # optional, for chaining
  --output /tmp/<cap>-iter-<N>/adapter \
  --adapter <cap>-stage-N-<slug> \
  --rank 16 --alpha 32 --lr 1e-4 --epochs 6 --samples-per-prompt 2 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <cap>-stage-N-<slug>
```

Receipt carries `n_prompts`, `effective_steps` (true step count after EOS-
skip filtering), `teacher_calls_made`, `skip_rate`.

### `cuda_sft_file` (pre-existing)

SFT trainer.

```bash
cuda_sft_file \
  --data datasets/sft.train.jsonl \
  --model /workspace/Qwen3.5-4B \
  --base-adapter <prev-stage-adapter> \   # optional, for chaining
  --output /tmp/<cap>-iter-<N>/adapter \
  --adapter <cap>-stage-N-<slug> \
  --rank 4 --alpha 8 --lr 1e-4 --epochs 1 --dataset-cap 128 \
  --adapter-smoke-test \
  --install-adapter-dir /workspace/adapters \
  --install-adapter-name <cap>-stage-N-<slug>
```

## Receipts and manifests

### `train_receipt.json` (issue 8)

Trainer-owned. Written next to every trained adapter. Schema documented in
`docs/TRAIN_RECEIPT_SCHEMA.md`. Cap scripts read this directly; no
log-grepping.

Key fields by section:
- **Identity:** kiln_commit, model_config_hash, tokenizer_config_hash,
  training_data_path + sha256
- **Hyperparameters:** rank, alpha, lr, epochs, seed, mode, kl_coeff,
  clip_epsilon, dynamic_sampling
- **ECHO:** echo_enabled, echo_lambda, echo_env_mask_mode, echo_warning_filter,
  no_policy_loss
- **Data stats:** groups_seen, groups_filtered, groups_trained, filter_var_min,
  reward_{mean,stdev,min,max}, group_variance_histogram, action/env/context_token_count
- **Diagnostics:** echo_metrics, warning_filter_masked_bytes,
  lora_delta_norm_summary, grad_norm_min_mean_max, wall_clock_s, peak_vram_mib
- **Failure:** status, failure_reason (issue 24 standard reasons)
- **Outputs:** output_adapter_path + sha256, adapter_manifest_path

### `adapter_manifest.json` (issue 36)

Per-adapter manifest. Captures everything needed to restore. Schema:
- model_id, model_sha256
- base_adapter_path (if chained), base_adapter_sha256
- training_data_path, training_data_sha256
- hyperparameters, seed
- safetensors path + sha256
- training_receipt path

`kiln adapter restore <manifest>` round-trips it.

## Phase 1 GRPO defaults (DrGrpo + TokenLevel + dynamic_sampling)

These defaults mitigate the published GRPO failure modes:
- DAPO §2 length drift: per-token (not per-sample) loss normalization
- Magistral mode collapse: KL coeff 0.1 anchor + clip 0.20 keeps exploration
- Cui et al. entropy collapse: TokenLevel limits per-token KL contribution

Override only with evidence; the defaults are not arbitrary.

## Issue numbering map (frequent references)

| Issue | What it enables |
|---|---|
| 1 | Chat adapter semantics (omitted = default, null = base, named = override) |
| 2 | Adapter load validation (rejects nested-output mistake) |
| 4 | `kiln adapter verify` (loadable + behavioral) |
| 5 | `--install-adapter-dir / --install-adapter-name` (atomic install) |
| 6 | `--base-adapter` chains with rank validation |
| 7 | Unsafe LoRA scale (alpha/rank > 2) requires override |
| 8 | `train_receipt.json` schema |
| 9 | `cuda_*_ablation --dry-run` |
| 10 | `kiln trajectory inspect` |
| 11 | `kiln_train::pi_trajectory` (Rust normalizer) |
| 13 | `/v1/health` perf counters |
| 15 | `kiln serve --eval-mode` |
| 16 | Adapter load/unload stress test |
| 17 | First-class thinking mode config |
| 19 | `--adapter-smoke-test` |
| 21 | Saturated-reward diagnostics |
| 22 | `--filter-var-min` (strong-signal filter) |
| 24 | Standard failure reasons |
| 27 | Byte-identical adapter diagnostic (long-context) |
| 28 | Testable warning-prefix masking |
| 33 | `kiln eval-adapter --seeds N` |
| 34 | `kiln rollout` (HTTP) |
| 36 | `adapter_manifest.json` + `kiln adapter restore` |
| 37 | `cuda_opd_remote` |
| 38 | Reward-saturation-aware training recommendation |
| 40 | Regression test backlog |
