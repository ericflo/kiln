# Training Receipt Schema

`train_receipt.json` is written next to every adapter produced by kiln-owned
SFT, GRPO, CUDA-native SFT, Vulkan-native SFT/GRPO, and OPD training paths.
GRPO dry-run validation also writes the same receipt at the intended adapter
path before any model forward/backward occurs. It is the stable
machine-readable training audit artifact. Scripts should read this file instead
of scraping trainer stdout.

## Location

`<adapter_dir>/<adapter_name>/train_receipt.json`

## Versioning

- `schema_version`: `1`
- `receipt_type`: `"kiln_train_receipt"`
- Additive nullable fields may be added without changing `schema_version`.
- Breaking field renames, type changes, or semantic changes require a new
  `schema_version`.

## Top-Level Fields

Required top-level fields:

- `schema_version`: integer schema version.
- `receipt_type`: constant string, `"kiln_train_receipt"`.
- `adapter_name`: output adapter directory name.
- `produced_at`: UTC RFC3339 timestamp for receipt creation.
- `status`: `"success"` or `"failed"`.
- `failure_reason`: string when `status == "failed"`, otherwise `null`.
- `kiln`: source revision and package metadata.
- `model`: model path, when discoverable, and model config hash.
- `tokenizer`: tokenizer plus chat-template config hash.
- `adapters`: base and output adapter paths, hashes, and byte sizes.
- `training_data`: source type, path when applicable, and SHA-256.
- `hyperparameters`: mode, rank, alpha, learning rate, epochs, and seed.
- `grpo`: GRPO-specific settings, or `null` for non-GRPO runs.
- `opd`: OPD-specific settings, or `null` for non-OPD runs. Identity-aware
  runs include `teacher_id`, `teacher_content_revision`, and the complete
  canonical `teacher_identity` used for scoring.
- `echo`: ECHO settings in effect for the run.
- `no_policy_loss`: verifier-free env-only GRPO flag.
- `data`: examples/groups/completions read, filtered, and trained. SFT receipts
  additionally contain the validated `sft_ingestion` object with the explicit
  invalid-row policy and stable kept/rejected row hashes.
- `rewards`: reward count, mean, stdev, and group-variance histogram.
- `token_counts`: action, env, and context token counts.
- `phase_timings`: aggregate phase timings in milliseconds for tokenization,
  mask construction, reference forward, policy forward, backward, and optimizer
  work. Non-applicable phases are `0.0`.
- `runtime`: wall-clock milliseconds, nullable peak VRAM, optional complete
  `execution_provenance`, and optional concrete `training_precision`.
- `lora_delta_norms`: per-module LoRA A/B norm and delta upper-bound summary.
- `config`: full serialized effective trainer config.

Hashes are lowercase hex SHA-256 strings prefixed with `sha256:`.

For SFT, `training_data.sha256` is the transport-independent ordered
`data.sft_ingestion.kept_corpus_sha256`. The nested object uses schema
`kiln.sft-ingestion.v1`; receipt reads validate its counts, hashes, rejected-row
ordering, and aggregate. See
[SFT Ingestion, Invalid Rows, and Row Identity](sft-ingestion.md).

New successful model-backed runs record a validated
`kiln.execution-provenance.v1` object at `runtime.execution_provenance`. It
binds the backend/device, bounded driver/runtime evidence, exact executable and
optional source revision, model/tokenizer/template identity, resolved precision
policy, compiled kernels, and effective server configuration/environment.
`runtime.training_precision` separately records the actual parameter,
optimizer-state, activation, and gradient dtypes plus stochastic-rounding
policy. Early failures before optimizer setup, dry runs, synthetic callers, and
legacy receipts may omit one or both fields. Receipt reads reject an internally
tampered execution record or malformed concrete precision contract. See
[Execution Provenance](EXECUTION_PROVENANCE.md).

For OPD, `teacher_content_revision` identifies the exact numeric logit source.
A live model-backed source uses the complete canonical teacher identity;
materialized or composite fixtures instead hash their exact scored rows or
algorithm contract. The nested `teacher_identity`, when present, independently
binds base or adapter content, numeric tokenizer vocabulary/config,
implementation/runtime contract, protocol, and scoring bounds. Its canonical
digest is the `identity_revision` returned by `GET /v1/teachers`. Pre-scored
JSONL also records the SHA-256 of the exact loaded source bytes under
`training_data`.

## Failed Receipts

Known validation failures write the same schema with:

```json
{
  "status": "failed",
  "failure_reason": "..."
}
```

Adapter output hashes and LoRA norm summaries may be `null` or empty when the
failure occurs before adapter weights are written. The intended adapter
directory is still created so automation can find the failure receipt at the
normal path.

## Token Counts

- `action_tokens`: assistant/policy target tokens.
- `env_tokens`: environment/tool-result target tokens supervised by ECHO.
- `context_tokens`: non-action, non-env input tokens seen by the trainer.

For repeated-epoch SFT paths, counts are multiplied by epoch count. For GRPO
paths, counts reflect trained groups after dynamic sampling and tokenization
filters.

## Phase Timings

`phase_timings` records cumulative phase durations:

- `tokenize_ms`
- `mask_build_ms`
- `reference_forward_ms`
- `policy_forward_ms`
- `backward_ms`
- `optimizer_ms`
- `gpu_writer_wait_ms`: cumulative time waiting for in-flight inference readers
  before bounded server-coordinated GRPO phases.
- `gpu_writer_held_ms`: cumulative time those phases held exclusive GPU
  ownership, including the mandatory device settlement before each yield.
- `gpu_writer_acquisitions`: number of separately scheduled GRPO GPU phases.

GRPO and GRPO dry-run paths populate tokenization and mask timings. GRPO
training paths additionally populate reference, policy, backward, and optimizer
timings. Server-submitted GRPO additionally populates the three GPU-writer
fields; direct library calls without coordination leave them at zero. Other
training paths may leave fields at `0.0` until they wire the same
instrumentation. Disk encoding and checkpoint publication are deliberately not
included in writer-held time.
