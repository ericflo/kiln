# Base-weight identity and compatibility

Kiln gives the complete set of safetensors shards behind a resident model one
canonical content identity. The model loader builds the identity while hashing
its startup snapshot, retains it in memory, and carries it through GPU
transfer, training, evaluation, checkpoints, and receipts.

Use the aggregate to answer “are these the same base-weight bytes?” Do not use
a model name, directory name, Hub ID, or filename list as a substitute.

Production model startup requires a valid shard manifest. A missing manifest is
reserved for legacy archives and explicitly synthetic or mock model paths.

## Read the identity

For a quick resident-model check, inspect either health route:

```bash
curl -fsS http://localhost:8420/health \
  | jq '.base_weight_identity'
```

`/health` and `/v1/health` return a bounded summary:

```json
{
  "manifest_type": "kiln.base-weight-shards.v1",
  "aggregate_algorithm": "kiln.base-model-content.v1",
  "aggregate_sha256": "sha256:<64 lowercase hex>",
  "shard_count": 2,
  "total_size_bytes": 9319828096
}
```

The aggregate is enough for a content-equivalence comparison. Use the gated
`/v1/debug/model-state` endpoint or a receipt/checkpoint that already carries
the complete manifest when you need per-shard audit detail.

## Manifest schema

The current contract is `kiln.base-weight-shards.v1`, schema version 1:

```json
{
  "schema_version": 1,
  "manifest_type": "kiln.base-weight-shards.v1",
  "aggregate_algorithm": "kiln.base-model-content.v1",
  "aggregate_sha256": "sha256:<64 lowercase hex>",
  "total_size_bytes": 9319828096,
  "shards": [
    {
      "filename": "model-00001-of-00002.safetensors",
      "size_bytes": 5329398688,
      "sha256": "sha256:<64 lowercase hex>"
    },
    {
      "filename": "model-00002-of-00002.safetensors",
      "size_bytes": 3990429408,
      "sha256": "sha256:<64 lowercase hex>"
    }
  ]
}
```

Each shard digest covers every byte in the named file. A shard remains part of
the identity even when the current model implementation does not request a
tensor from it.

The strict validator requires:

- schema version 1, the exact manifest type, and the exact aggregate algorithm;
- between 1 and 4,096 shards;
- a nonzero byte size and lowercase `sha256:` digest for every shard;
- unique portable `.safetensors` filenames, no more than 255 bytes each,
  sorted lexicographically;
- an exact, non-overflowing total of all shard sizes;
- an aggregate recomputed from the shard records; and
- no unknown fields.

## How the aggregate is computed

`kiln.base-model-content.v1` separates content identity from audit filenames:

1. Decode every shard SHA-256 and pair it with the shard's unsigned 64-bit
   byte size.
2. Sort those records by digest bytes and then by size. Preserve repeated
   content records from differently named shards.
3. Hash the ASCII domain `kiln.base-model-content.v1`, a NUL byte, the record
   count as little-endian `u64`, and then each little-endian size followed by
   its raw 32-byte digest.
4. Encode the result as lowercase `sha256:<hex>`.

The aggregate changes if any shard byte, size, or multiplicity changes. It is
independent of absolute paths, filenames, safetensors-index order, and
directory enumeration order.

Filenames stay in the full manifest for diagnosis. Renaming the same shard
bytes changes the audit manifest but not the content aggregate.

## Where the identity appears

| Surface | Representation | Reader job |
| --- | --- | --- |
| `/health`, `/v1/health` | Aggregate, algorithm, shard count, total bytes | Compare resident base content quickly. |
| Gated `/v1/debug/model-state` | Complete shard manifest | Diagnose an exact shard or filename difference. |
| Exact SFT, GRPO, and OPD checkpoints | Full manifest plus aggregate consistency field | Reject incompatible resume before GPU ownership. |
| `train_receipt.json` | `model.base_weight_shard_manifest` | Audit the base used for a completed run. |
| `adapter_manifest.json` | Top-level copy | Carry base provenance with the serving adapter. |
| Eval jobs, terminal archives, and raw result JSON | Full manifest | Bind eval outcomes to exact base bytes. |
| Dashboard outcome JSONL | Full manifest on each self-contained row | Preserve identity outside the dashboard. |

Routine browser and CLI views show the bounded summary. Raw JSON retains the
complete shard list where the contract calls for it.

## Decide what “compatible” means

| Comparison | Same aggregate required? | Other identity required? |
| --- | :---: | --- |
| “These model directories contain the same base weights.” | yes | No; paths and filenames can differ. |
| “This adapter was trained from these base bytes.” | compare the adapter manifest or receipt aggregate | Also inspect model config, tokenizer/template, and adapter lineage. |
| “This eval result used the resident base.” | yes | Also inspect execution, tokenizer/template, request, and adapter identities. |
| “This checkpoint can resume exactly.” | yes | Yes: exact checkpoint, execution, data, optimizer, scheduler, RNG, adapter, tokenizer, and configuration contracts must all pass. |

The shard manifest proves only the loaded base-weight bytes. It does not bind
model configuration, tokenizer, chat template, adapter revision, training
data, seed, executable, backend, driver, precision, or kernels.

## Exact-resume behavior

Exact resume validates both the checkpoint's manifest and the current
resident manifest before GPU ownership. Their validated content aggregates
must match:

- a byte, size, or multiplicity change fails closed;
- a path or filename-only change passes the base-weight comparison; and
- a matching aggregate can still fail resume at another identity boundary.

Legacy checkpoints that recorded only `base_model_weights_sha256` cannot prove
their constituent shards and are rejected for exact resume. They remain
archival checkpoint files; they do not become PEFT serving adapters or valid
weights-only restore inputs.

Legacy train and eval archives may deserialize without the optional full
manifest. Every newly admitted production job records one.

See [execution provenance](EXECUTION_PROVENANCE.md) for process, backend,
device, runtime, precision, kernel, and effective-configuration identity; see
[native training checkpoints](../training/training-checkpoints.md) for the rest of the
exact-resume envelope.
