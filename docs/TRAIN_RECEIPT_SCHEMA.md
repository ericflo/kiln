# Training Receipt Schema

`train_receipt.json` is written next to every adapter produced by kiln-owned
SFT, GRPO, CUDA-native SFT, Vulkan-native SFT/GRPO, and OPD training paths.
It is the stable machine-readable training audit artifact. Scripts should read
this file instead of scraping trainer stdout.

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
- `echo`: ECHO settings in effect for the run.
- `no_policy_loss`: verifier-free env-only GRPO flag.
- `data`: examples/groups/completions read, filtered, and trained.
- `rewards`: reward count, mean, stdev, and group-variance histogram.
- `token_counts`: action, env, and context token counts.
- `runtime`: wall-clock milliseconds and nullable peak VRAM.
- `lora_delta_norms`: per-module LoRA A/B norm and delta upper-bound summary.
- `config`: full serialized effective trainer config.

Hashes are lowercase hex SHA-256 strings prefixed with `sha256:`.

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
