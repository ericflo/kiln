# Adapter Manifest

Kiln writes `adapter_manifest.json` next to completed adapter artifacts. The
manifest is the adapter's storage-neutral provenance record: it identifies the
adapter, pins the files needed to serve it, and records the training/source
hashes needed to audit or restore it.

## Schema

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
