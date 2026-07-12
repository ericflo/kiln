# Base-Weight Provenance

Kiln assigns one canonical identity to the complete set of safetensors shards
behind the resident base model. The loader creates this manifest while hashing
the immutable startup snapshot. It retains the result in memory and carries it
through GPU transfer, training, and evaluation; status and artifact writes do
not re-read weight files during inference.

Production model startup requires a valid manifest. `None` is reserved for
legacy archives and explicitly synthetic or mock model paths.

## Manifest schema

The current type is `kiln.base-weight-shards.v1` with schema version `1`:

```json
{
  "schema_version": 1,
  "manifest_type": "kiln.base-weight-shards.v1",
  "aggregate_algorithm": "kiln.base-model-content.v1",
  "aggregate_sha256": "sha256:<64 lowercase hex digits>",
  "total_size_bytes": 123456,
  "shards": [
    {
      "filename": "model-00001-of-00002.safetensors",
      "size_bytes": 61728,
      "sha256": "sha256:<64 lowercase hex digits>"
    }
  ]
}
```

Each shard digest covers every byte in the named file. Shards that contribute
no tensor requested by the current model implementation are still included.
The strict validator requires at least one and at most 4,096 shards, non-zero
sizes, unique portable `.safetensors` filenames sorted lexicographically, valid
lowercase prefixed SHA-256 values, an exact byte total, and no unknown fields.

## Content identity

`kiln.base-model-content.v1` deliberately separates byte identity from audit
filenames. To compute `aggregate_sha256`:

1. Decode every shard SHA-256 and pair it with its unsigned 64-bit byte size.
2. Sort records by digest bytes and then size. Preserve duplicate records.
3. Hash the ASCII domain `kiln.base-model-content.v1` followed by a NUL byte,
   the record count as little-endian `u64`, then for each record its size as
   little-endian `u64` followed by the 32 digest bytes.
4. Encode the result as lowercase `sha256:<hex>`.

The aggregate changes when any shard byte, size, or multiplicity changes. It is
independent of absolute path, filename, safetensors-index order, and directory
enumeration order. Filenames remain in the manifest so an operator can locate
and audit a shard, but renaming identical shard bytes does not make an exact
resume incompatible.

## Runtime and artifact surfaces

- `/health` and `/v1/health` expose the bounded `base_weight_identity` summary:
  type, aggregate algorithm/digest, shard count, and total bytes.
- Gated `/v1/debug/model-state` exposes `base_weight_shard_manifest` in full.
  Enable it only through the existing eval/debug endpoint policy.
- Exact SFT, GRPO, and OPD checkpoints store the full manifest in
  `auxiliary_state.base_weight_shard_manifest` and retain the aggregate in
  `base_model_weights_sha256` as a consistency check.
- Completed `train_receipt.json` files store it under
  `model.base_weight_shard_manifest`; `adapter_manifest.json` copies it to the
  top level. Both readers reject internally inconsistent manifests.
- Eval admission snapshots the resident manifest. Job list/detail responses,
  terminal archives, raw result JSON, CLI output, and dashboard drill-ins retain
  it. Dashboard outcome JSONL exports include it on each self-contained row.

The browser and CLI show the aggregate, count, and byte total for routine use;
raw JSON carries the complete shard list.

## Exact resume and compatibility

Exact resume validates both the checkpoint and current manifests before any GPU
ownership. Their validated content aggregates must match. A byte, size, or
multiplicity change fails closed; a path or filename-only change is accepted.

Legacy exact checkpoints that contain only `base_model_weights_sha256` cannot
prove their constituent shard artifacts and are rejected with a migration
error. They remain ordinary files and may still be used as documented
serving-only or weights-only inputs where exact optimizer continuation is not
claimed. Legacy train/eval archives may deserialize without the optional full
manifest, but all newly admitted production jobs include one.

The manifest proves loaded base-weight bytes, not the entire replay envelope.
Tokenizer/template identity, adapter revision, data, seed, executable/source,
backend, device, driver/runtime, precision, and kernels remain separate
provenance inputs.
