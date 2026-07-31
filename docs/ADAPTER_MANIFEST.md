# Adapter manifests and restore

Kiln writes `adapter_manifest.json` beside a successfully completed adapter.
The manifest binds the serving files to their SHA-256 digests and carries
training provenance copied from `train_receipt.json`.

Use it to answer three questions:

1. Which config, weights, and optional receipt belong to this adapter?
2. Did those files change after the manifest was written?
3. Which recorded model, data, runtime, and precision produced them?

The manifest is not an exact-training checkpoint, a signature, or proof that
the adapter is compatible with the model currently loaded by a server.

## Files that travel together

A complete native adapter directory normally contains:

```text
support-bot-v3/
├── adapter_config.json
├── adapter_model.safetensors
├── train_receipt.json
└── adapter_manifest.json
```

`adapter_config.json` and `adapter_model.safetensors` are the serving
artifacts. `train_receipt.json` describes the completed training run.
`adapter_manifest.json` binds their filenames and digests so they can be moved
and restored as one set.

Kiln writes a manifest only when:

- the training receipt reports success; and
- both canonical serving files exist.

A failed or incomplete run does not receive a completed-adapter manifest.

## Normative schema

The generated [Artifact Lifecycle API
Schema](../contracts/kiln-artifacts-v1.schema.json) is the field-level
authority for `AdapterManifest`, adapter-list responses, and adapter mutation
requests. This guide explains lifecycle and interpretation.

The current manifest contract is:

```text
schema_version: 1
manifest_type: kiln_adapter_manifest
```

### Content-binding fields

| Field | Required | Meaning |
| --- | --- | --- |
| `adapter_name` | yes | Name recorded by the completed training run. |
| `safetensors_hash` | yes | SHA-256 of `adapter_model.safetensors`. |
| `config_hash` | yes | SHA-256 of `adapter_config.json`. |
| `receipt_hash` | no | SHA-256 of `train_receipt.json` when that file was present. |
| `files.adapter_model` | yes | Source filename for adapter weights. Native Kiln output uses `adapter_model.safetensors`. |
| `files.adapter_config` | yes | Source filename for the PEFT configuration. Native Kiln output uses `adapter_config.json`. |
| `files.train_receipt` | no | Source receipt filename when one is bound. |

### Lineage fields

| Field | Required | Meaning |
| --- | --- | --- |
| `parent_adapter` | no | Base adapter name, or the recorded base-adapter path when training continued from an adapter. |
| `model_config_hash` | no | Model-configuration digest copied from the receipt. |
| `training_chat_template_hash` | no | Effective training-template digest. This is the template used to build training labels, not necessarily the inference template. |
| `base_weight_shard_manifest` | no | Validated `kiln.base-weight-shards.v1` identity for every base-weight shard. |
| `execution_provenance` | no | Validated `kiln.execution-provenance.v1` record for process, backend, build, model/tokenizer, kernels, and effective configuration. |
| `training_precision` | no | Parameter, optimizer-state, activation, and gradient dtypes plus stochastic-rounding policy. |
| `kiln_commit` | no | Git commit reported by training. |
| `training_data_hash` | no | Exact training-data digest when the source supplied one. |
| `training_data_source` | no | Receipt label for the training-data route. |
| `training_data_path` | no | Recorded source path when known. This is provenance, not a portable restore location. |
| `openenv_training_data` | no | Validated `kiln.openenv-training-data.v1` environment, schema, ordered group-plan, seed, step, and termination identity copied from an all-OpenEnv GRPO receipt. |

Legacy manifests can omit the newer lineage fields. Missing provenance means
“not recorded,” not “compatible.”

## Compact example

The generated schema contains the complete nested objects. A typical manifest
has this top-level shape:

```json
{
  "schema_version": 1,
  "manifest_type": "kiln_adapter_manifest",
  "adapter_name": "support-bot-v3",
  "safetensors_hash": "sha256:<64 lowercase hex>",
  "config_hash": "sha256:<64 lowercase hex>",
  "receipt_hash": "sha256:<64 lowercase hex>",
  "parent_adapter": "support-bot-v2",
  "model_config_hash": "sha256:<64 lowercase hex>",
  "training_chat_template_hash": "sha256:<64 lowercase hex>",
  "base_weight_shard_manifest": {
    "schema_version": 1,
    "manifest_type": "kiln.base-weight-shards.v1",
    "aggregate_algorithm": "kiln.base-model-content.v1",
    "aggregate_sha256": "sha256:<64 lowercase hex>",
    "total_size_bytes": 9319828096,
    "shards": []
  },
  "execution_provenance": {
    "schema_version": 1,
    "provenance_type": "kiln.execution-provenance.v1",
    "provenance_sha256": "sha256:<64 lowercase hex>"
  },
  "training_precision": {
    "parameter_dtype": "bf16",
    "optimizer_state_dtype": "f32",
    "activation_dtype": "f32",
    "gradient_dtype": "f32",
    "stochastic_rounding": {"mode": "round_to_nearest"}
  },
  "kiln_commit": "<git commit>",
  "training_data_hash": "sha256:<64 lowercase hex>",
  "training_data_source": "jsonl_grpo_groups",
  "training_data_path": "/data/groups.jsonl",
  "files": {
    "adapter_model": "adapter_model.safetensors",
    "adapter_config": "adapter_config.json",
    "train_receipt": "train_receipt.json"
  }
}
```

The shortened nested objects and placeholder hashes above illustrate the
shape; they are not valid production values.

## What each check establishes

| Action | What it checks | What it does not check |
| --- | --- | --- |
| `GET /v1/adapters` | Reads the manifest when present and reports parsed metadata or `adapter_manifest_error`. | It does not re-hash the adapter files merely to list the registry. |
| `kiln adapters restore` | Copies the named config, weights, optional receipt, and manifest; then verifies the copied file hashes. | It does not connect to a server or compare recorded base weights and runtime with a resident model. |
| `kiln adapters verify NAME_OR_PATH` | Checks local layout, parses PEFT config, reads safetensors, summarizes LoRA tensors, and reports file hashes and a delta proxy. | Offline verification cannot prove that a particular running model can load or use the adapter. |
| `kiln adapters verify NAME --url URL` | Adds server discovery, load, and base-versus-adapter behavior checks. | A single prompt is a smoke check, not a quality evaluation. |

The server can discover and load a legacy adapter that has the two serving
files but no manifest. A malformed manifest is surfaced separately in
`adapter_manifest_error`; do not mistake absence of a parse error for content
verification.

## Restore into a registry

Keep the manifest beside every file named by its `files` object, then run:

```bash
kiln adapters restore \
  ./runs/support-bot-v3/adapter_manifest.json \
  --adapter-dir /models/teacher-model/adapters
```

By default, the target directory name comes from `adapter_name`. Use
`--name NEW_NAME` to choose another path-safe registry name.

Restore performs this sequence:

1. Read the manifest and validate any embedded base-weight, execution,
   training-template, and precision records.
2. Copy the named config and weights into a temporary directory using the
   canonical serving filenames.
3. Copy the receipt when `files.train_receipt` is present.
4. Copy `adapter_manifest.json`.
5. Hash the copied config, weights, and bound receipt and compare them with the
   manifest.
6. Validate the resulting adapter layout and move it into the registry.

The command prints a JSON receipt containing the final adapter path, copied
filenames, and verified hashes.

An existing target is rejected by default. `--overwrite` is destructive: the
current implementation removes the existing target before publishing the
restored directory. Back up an adapter you may need, and prefer a new
`--name`, before using this flag.

## Check compatibility before activation

Restore proves that the moved files match one another. It does not prove that
the destination server has the recorded base weights, tokenizer, template,
backend, precision policy, or kernel contract.

Before activation:

1. Compare `base_weight_shard_manifest` with the resident model's content
   identity.
2. Review `execution_provenance`, `training_precision`, and
   `training_chat_template_hash` for the compatibility boundary that matters
   to the workload.
3. Run offline verification.
4. Run server-backed verification against the intended Kiln instance.
5. Use a task-specific eval before promotion.

The manifest can support that audit because it preserves exact recorded
inputs. It cannot replace the audit.

See [base-weight provenance](BASE_WEIGHT_PROVENANCE.md) for content
equivalence and exact-resume boundaries, [execution
provenance](EXECUTION_PROVENANCE.md) for the runtime envelope, and [training
receipt schema](TRAIN_RECEIPT_SCHEMA.md) for the source record from which a
native manifest is built.
