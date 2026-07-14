# Adapter Manifest

Kiln writes `adapter_manifest.json` next to completed adapter artifacts. The
manifest is the adapter's storage-neutral provenance record: it identifies the
adapter, pins the files needed to serve it, and records the training/source
hashes needed to audit or restore it.

## Schema

The generated [Artifact Lifecycle API Schema](../contracts/kiln-artifacts-v1.schema.json)
is the field-level authority for `AdapterManifest` and every adapter API
request and response that contains it. This document describes storage and
audit semantics.

Current schema version: `1`.

Required fields:

- `schema_version`: integer schema version, currently `1`.
- `manifest_type`: string, `kiln_adapter_manifest`.
- `adapter_name`: adapter name produced by training.
- `safetensors_hash`: SHA-256 for `adapter_model.safetensors`.
- `config_hash`: SHA-256 for `adapter_config.json`.
- `files.adapter_model`: adapter weights filename.
- `files.adapter_config`: adapter config filename.

Optional provenance fields:

- `receipt_hash`: SHA-256 for `train_receipt.json` when present.
- `files.train_receipt`: train receipt filename when present.
- `parent_adapter`: base/parent adapter name or path when training continued
  from an adapter.
- `model_config_hash`: model config hash from `train_receipt.json`.
- `training_chat_template_hash`: effective SFT template SHA-256 from
  `train_receipt.json`; for the qualified Qwen3.5 path this identifies TRL's
  prefix-preserving assistant-mask template, not the inference template.
- `base_weight_shard_manifest`: strict `kiln.base-weight-shards.v1` identity
  copied from `train_receipt.json`, including every shard SHA-256 and byte size.
- `execution_provenance`: strict `kiln.execution-provenance.v1` process,
  backend, runtime, build, tokenizer/template, kernel, and configuration
  identity copied from `train_receipt.json`.
- `training_precision`: concrete parameter, optimizer-state, activation, and
  gradient dtypes plus stochastic-rounding policy copied from the receipt.
- `kiln_commit`: kiln git commit recorded by training.
- `training_data_hash`: training data hash from `train_receipt.json`.
- `training_data_source`: training data source label.
- `training_data_path`: training data path when known.

Example:

```json
{
  "schema_version": 1,
  "manifest_type": "kiln_adapter_manifest",
  "adapter_name": "support-bot-v3",
  "safetensors_hash": "sha256:...",
  "config_hash": "sha256:...",
  "receipt_hash": "sha256:...",
  "parent_adapter": "support-bot-v2",
  "model_config_hash": "sha256:...",
  "training_chat_template_hash": "sha256:...",
  "base_weight_shard_manifest": {
    "schema_version": 1,
    "manifest_type": "kiln.base-weight-shards.v1",
    "aggregate_algorithm": "kiln.base-model-content.v1",
    "aggregate_sha256": "sha256:...",
    "total_size_bytes": 123456,
    "shards": [
      {
        "filename": "model.safetensors",
        "size_bytes": 123456,
        "sha256": "sha256:..."
      }
    ]
  },
  "execution_provenance": {
    "schema_version": 1,
    "provenance_type": "kiln.execution-provenance.v1",
    "backend": {
      "name": "rocm",
      "device": "rocm:0",
      "numerical_runtime_sha256": "sha256:..."
    },
    "build": {
      "package_version": "0.4.1",
      "target": "linux-x86_64",
      "executable_sha256": "sha256:..."
    },
    "model": {
      "model_config_sha256": "sha256:...",
      "tokenizer_vocab_sha256": "sha256:...",
      "tokenizer_config_sha256": "sha256:...",
      "chat_template_sha256": "sha256:...",
      "training_chat_template_sha256": "sha256:..."
    },
    "precision": {
      "inference_dtype": "bf16",
      "training_policy": "rocm_native_float"
    },
    "kernels": {
      "contract_type": "kiln.kernel-contract.v1",
      "versions": {"kiln-model": "0.4.1"},
      "compiled_features": ["rocm"],
      "contract_sha256": "sha256:..."
    },
    "configuration": {
      "effective_server_config_sha256": "sha256:...",
      "effective_environment_sha256": "sha256:..."
    },
    "provenance_sha256": "sha256:..."
  },
  "training_precision": {
    "parameter_dtype": "bf16",
    "optimizer_state_dtype": "f32",
    "activation_dtype": "f32",
    "gradient_dtype": "f32",
    "stochastic_rounding": {"mode": "round_to_nearest"}
  },
  "kiln_commit": "abc123",
  "training_data_hash": "sha256:...",
  "training_data_source": "jsonl_grpo_groups",
  "training_data_path": "/data/groups.jsonl",
  "files": {
    "adapter_model": "adapter_model.safetensors",
    "adapter_config": "adapter_config.json",
    "train_receipt": "train_receipt.json"
  }
}
```

## Restore

To restore an adapter, keep `adapter_manifest.json` beside the files named in
`files`, then run:

```bash
kiln adapters restore ./adapter_manifest.json --adapter-dir /models/Qwen3.5-4B/adapters
```

By default the restored adapter name is `adapter_name` from the manifest. Use
`--name <adapter>` to restore under a different registry name. Existing adapter
paths are not replaced unless `--overwrite` is passed.

The restore command copies `adapter_config.json`, `adapter_model.safetensors`,
`train_receipt.json` when listed, and `adapter_manifest.json`, then verifies the
copied config, safetensors, and receipt hashes before reporting success.
Manifest reads validate the complete base-weight shard identity, execution
record, effective training-template digest, and concrete precision contract
when present. A direct training-template digest must agree with the copy inside
the execution record. Legacy adapter manifests may omit these optional fields.
See
[Base-Weight Provenance](BASE_WEIGHT_PROVENANCE.md) for content-equivalence and
exact-resume semantics, and [Execution Provenance](EXECUTION_PROVENANCE.md) for
the process/runtime envelope.
