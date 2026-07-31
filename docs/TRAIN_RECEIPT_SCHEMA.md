# Training receipts

`train_receipt.json` is Kiln's machine-readable account of one SFT, GRPO, or
OPD attempt. Read it to answer:

- what training mode and effective configuration were used;
- which model, tokenizer, data, base adapter, and output adapter were observed;
- which tokens, rewards, timings, losses, and norms were measured;
- which execution and precision evidence was available; and
- whether the attempt succeeded, failed, or only performed GRPO dry-run
  validation.

Do not scrape trainer logs for these facts. Also do not treat the receipt as a
checkpoint, a quality certificate, or proof of reproducibility.

## Find and summarize a receipt

Kiln writes the file beneath the adapter output directory:

```text
<adapter-root>/<adapter-name>/train_receipt.json
```

Inspect the decision-bearing fields first:

```bash
receipt_path=./adapters/my-lora/train_receipt.json

jq '{
  schema_version,
  receipt_type,
  adapter_name,
  produced_at,
  status,
  failure_reason,
  failure_message,
  mode: .hyperparameters.mode,
  data_source: .training_data.source,
  output: .adapters.output,
  runtime,
  adapter_smoke_test
}' "$receipt_path"
```

For a failed attempt, keep both `failure_reason` and `failure_message`.
For a successful receipt, check `hyperparameters.mode` and
`training_data.source` before assuming that optimizer work occurred. A
successful GRPO dry run uses `jsonl_grpo_groups_dry_run`; it proves validation
completed and may have no output adapter weights.

## Envelope and versioning

The current envelope is:

| Field | Contract |
| --- | --- |
| `schema_version` | Integer `1`. |
| `receipt_type` | Exact string `kiln_train_receipt`. |
| `adapter_name` | Non-empty output adapter name. |
| `produced_at` | RFC 3339 timestamp. |
| `status` | `success` or `failed`. |
| `failure_reason` | Stable failure category for failed attempts; null for success. |
| `failure_message` | Detailed diagnostic when available; null for success. |

Kiln rejects unsupported versions, a different type tag, an empty adapter
name, a malformed timestamp, a failed receipt without a reason, and a
successful receipt that carries failure fields.

Additive optional fields may be introduced within version 1. Consumers should
ignore fields they do not recognize. A breaking rename, type change, or
semantic change requires a new schema version.

## Top-level field map

The receipt groups identity, effective inputs, measurements, and diagnostics:

| Field | What it records | Important limit |
| --- | --- | --- |
| `kiln` | Package version plus best-effort Git commit, dirty state, and source path. | A checkout found at runtime is not proof that it built the executable. |
| `model` | Optional model path, model-config hash, and optional complete base-weight shard manifest. | A path and config hash do not identify weight bytes; use the shard manifest. |
| `tokenizer` | Legacy combined config hash plus separate tokenizer, serving-template, and training-template hashes. | Optional fields reflect what the run could capture. |
| `adapters` | Base and output paths, model-file byte counts, and model-file SHA-256 values. | Paths are informational and can change after publication. |
| `training_data` | Source kind, optional path, and source-specific content digest. | A path alone does not bind mutable bytes. Interpret the digest by source kind. |
| `hyperparameters` | Mode, rank, alpha, effective alpha/rank, learning rate, epochs, resolved seed, and shuffle selector. | Reconstructing shuffled order also requires identical data and ordering code. |
| `grpo` | GRPO policy, clipping, sampling, KL, behavior-source, and optional policy-audit fields; null otherwise. | Importance ratios and KL use separate denominators. |
| `opd` | OPD mode, objective, loss granularity, teacher identity, top-K, losses, token counts, and ECHO combination; omitted otherwise. | Teacher identity must be interpreted with its own content-revision contract. |
| `echo` | Whether the environment-CE term actually fired, its settings, observed CE values, and any drop reason. | `enabled` describes observed contribution, not merely requested configuration. |
| `no_policy_loss` | Whether verifier-free environment-only GRPO was requested. | Read failure fields as well; unsupported loss compositions fail rather than silently changing the objective. |
| `data` | Read, filtered, and trained counts plus optional reward-filter and SFT-ingestion evidence. | Counts describe admission and execution, not data quality. |
| `rewards` | Count, mean, standard deviation, range, group diagnostics, and variance histogram. | Summary statistics do not preserve individual rewards. |
| `token_counts` | Action, environment, warning-filter, and context token totals. | Counts are aggregates, not a token transcript. |
| `phase_timings` | Cumulative measured work by phase. | Zeros can mean non-applicable or not instrumented; they are not proof of zero cost. |
| `runtime` | Wall time, optional peak VRAM, SFT loss route, execution provenance, and concrete training precision. | Some early failures, dry runs, legacy receipts, and synthetic callers lack optional runtime evidence. |
| `config_hashes` | Model, tokenizer, template, and effective trainer-config digests. | `kiln.env_config_hash` is a legacy duplicate of the effective config hash, not a hash of the process environment. |
| `lora_delta_norms` | Per-module A/B norms and an upper bound on the LoRA delta norm. | The upper bound is not the exact merged-weight delta. |
| `lora_grad_norms` | Sample count and min/mean/max gradient norms by module. | Empty means unavailable, not necessarily zero gradients. |
| `adapter_smoke_test` | Finite-logit, output, latency, and per-prompt canary diagnostics when run. | A smoke pass is not a task-quality evaluation. |
| `config` | Full serialized effective trainer configuration. | This can be large; use `config_hashes.effective_config_hash` for comparison. |

All receipt-owned SHA-256 digests use
`sha256:<64 lowercase hexadecimal characters>`. Do not apply that rule to
ordinary strings such as a Git commit, source path, teacher alias, or failure
message.

## Status and failure semantics

`status: "success"` means the selected operation reached its successful
receipt path. For normal training that follows adapter serialization; for a
GRPO dry run it means schema, reward, tokenization, mask, provenance, and
filter checks completed without model forward or backward work.

`status: "failed"` carries a stable `failure_reason` category and usually a
more specific `failure_message`. Current categories include:

- `data_schema_error`
- `adapter_load_failed`
- `zero_groups`
- `zero_action_tokens`
- `zero_env_tokens`
- `cancelled`
- `unsupported_loss_config`
- `nan_loss`
- `oom`
- `shape_mismatch`
- `unsafe_lora_scale`
- `base_adapter_missing`
- `training_error`

Kiln classifies these categories from the diagnostic message. Preserve the
message for diagnosis and do not treat the category as a lossless error code.

SFT and GRPO failure-path receipt writes are best effort so the original
training failure remains primary. OPD and GRPO dry-run receipt-write failures
are returned to the caller. An absent receipt can therefore mean a legacy
adapter, an explicitly synthetic path, or a failed write; it does not establish
that training never ran.

Receipt JSON is written directly, not through a crash-atomic
temporary-file-and-rename protocol. A process interruption can leave a missing
or malformed file.

## Identity fields

### Kiln, model, and tokenizer

`kiln.git_commit`, `git_dirty`, and `git_source` are best-effort source
navigation. `kiln.package_version` is always the compiled package version.
`kiln.env_config_hash` retains an old field name but receives the same value as
`config_hashes.effective_config_hash`; it does not identify `KILN_*`
environment variables.

`model.base_weight_shard_manifest` is the authoritative base-weight content
identity when present. `model.path` and `model.config_hash` are not substitutes.
See [Base-weight identity](BASE_WEIGHT_PROVENANCE.md).

The tokenizer fields separate:

- the legacy combined `config_hash`;
- `tokenizer_config_hash`;
- `chat_template_hash` for serving; and
- `training_chat_template_hash` for the effective training template.

On write and read, Kiln rejects malformed training-template digests and any
disagreement among the tokenizer, `config_hashes`, and execution-provenance
copies that are present.

### Adapter and data content

`adapters.base` and `adapters.output` describe only
`adapter_model.safetensors`: path, byte count, and SHA-256. The separate
`adapter_manifest.json` binds the model file, adapter configuration, receipt,
and available provenance as one portable file set.

Training-data digest semantics depend on the source:

- SFT uses the transport-independent ordered kept-corpus digest from
  `data.sft_ingestion.kept_corpus_sha256`.
- Streamed GRPO uses the pinned JSONL source bytes and checks that the source
  remains unchanged across its validation and training passes.
- Inline GRPO and OPD use Kiln's serialized effective in-memory input.
- Pre-scored OPD JSONL records the exact loaded source-byte digest.

A digest match establishes equality under that source's encoding contract. It
does not imply that two different source encodings are semantically
equivalent.

## Mode-specific evidence

### SFT admission

SFT receipts carry `data.sft_ingestion` with contract
`kiln.sft-ingestion.v1`. Kiln validates the nested counts, kept and rejected
row identities, rejection order, aggregate digest, and these cross-field
bindings:

- mode is `sft`;
- source and source locator match `training_data`;
- the kept-corpus digest equals `training_data.sha256`;
- read and rejected counts match `data`; and
- the effective invalid-row policy matches `config`.

`config.training_profile` identifies the fixed native SFT contract described
in [Native SFT profile](NATIVE_SFT_PROFILE.md).

### GRPO policy audit

When present, `grpo.policy_audit` keeps importance sampling and KL evidence
separate:

- `importance_sampling` compares the current policy with the configured
  behavior policy and records ratio scope, extrema, and clipping counts.
- `kl_reference` compares the current policy with the independent frozen KL
  reference and records pre- and post-mask estimator means.
- `recorded_provenance` summarizes sampled versus forced action tokens and
  content-addresses the distinct recorded rollout behavior sources.

Do not interpret the KL reference as the behavior-policy denominator or vice
versa.

### OPD teacher identity

`opd.teacher_content_revision` identifies the numerical logit source. A live
model-backed source can carry the complete canonical `teacher_identity`;
materialized or composite sources instead bind their scored rows or algorithm
contract. `training_data.sha256` independently identifies pre-scored input
bytes when applicable.

See [OPD teacher JSONL](OPD_TEACHER_JSONL.md) and
[Immutable vLLM teacher identity](VLLM_TEACHER_IDENTITY.md).

### ECHO evidence

`echo.enabled` and `opd.echo_combined` mean the environment-CE term actually
contributed during the run. If ECHO was configured but no eligible environment
tokens remained, the receipt keeps `enabled: false` and explains why in
`dropped_reason`.

Token counts expose both sides of warning-prefix filtering:

| Field | Meaning |
| --- | --- |
| `env_tokens_before_warning_filter` | Observation tokens before the filter. |
| `warning_tokens_filtered` | Tokens excluded by `warning_prefix_len`. |
| `env_tokens_after_warning_filter` | Active environment-CE tokens after filtering. |
| `env_tokens` | Backward-compatible alias of the post-filter active count. |

`action_tokens` counts assistant or policy targets.
`context_tokens` counts remaining non-action, non-active-environment input
tokens. Repeated-epoch SFT totals are multiplied by epoch count; GRPO totals
describe groups that remain after admission and dynamic filtering.

## Timings and runtime

`phase_timings` contains cumulative milliseconds:

- `tokenize_ms`
- `mask_build_ms`
- `reference_forward_ms`
- `policy_forward_ms`
- `backward_ms`
- `optimizer_ms`
- `gpu_writer_wait_ms`
- `gpu_writer_held_ms`
- `gpu_writer_acquisitions`

The last three fields describe server-coordinated GRPO GPU phases. Direct
trainer calls without coordination leave them at zero. Writer-held time
includes mandatory device settlement before each yield; it excludes disk
encoding and checkpoint publication.

`runtime.execution_provenance` is the complete startup-owned execution
envelope. `runtime.training_precision` separately records parameter,
optimizer-state, activation, gradient, and stochastic-rounding behavior
observed after trainer setup. `runtime.sft_loss_route` identifies the
backend-owned SFT loss route. Kiln validates each present typed record on both
write and read.

See [Execution identity and provenance](EXECUTION_PROVENANCE.md).

## Integrity and interpretation

`train_receipt.json` has no signature or self-digest. For a successfully
published native adapter, `adapter_manifest.json` records the receipt filename
and `receipt_hash`; offline restore verifies that hash before copying the file.
Compare the manifest when you need receipt-byte integrity.

Kiln's receipt reader validates:

- the envelope version, type, timestamp, adapter name, and status consistency;
- SFT ingestion and its cross-field bindings;
- agreement among training-template identities;
- the complete base-weight shard manifest;
- execution provenance; and
- concrete training precision.

It does not re-run training, re-hash every external path, recompute adapter
norms, or authenticate the producer. A receipt can support an audit claim only
to the extent that its relevant hashes are independently anchored and its
measurement procedure is trusted.

Use an immutable `.kiln-checkpoint` for exact supported continuation and an
evaluation result for outcome evidence. Use the receipt to understand the
attempt that produced—or failed to produce—the adapter.
