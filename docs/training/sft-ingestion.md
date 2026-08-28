# SFT Ingestion, Invalid Rows, and Row Identity

Kiln reduces every SFT input source to one versioned admission contract before
queue publication or GPU ownership. Inline `examples`, server-local
`dataset_path` JSONL, uploaded named datasets, `corrections:active`, recipe
steps, and direct Rust training all pass through the same structural checks,
chat template, tokenizer, and assistant-only label validation. JSONL sources
also share one row parser.

The admitted rows are consumed by the fixed
[`native_online_lora_v1` update contract](NATIVE_SFT_PROFILE.md).

## Invalid-row policy

`config.invalid_row_policy` accepts exactly two values:

- `"fail"` is the default. The first invalid non-blank row rejects the entire
  submission. No training job is queued and no rows are silently discarded.
- `"skip"` keeps valid rows, rejects invalid rows, and queues the job only when
  at least one row remains. The submission response reports both counts and the
  final train receipt records ordered hashes for every kept and rejected row.

The CLI exposes the same choice:

```bash
kiln train sft \
  --file /data/corrections.jsonl \
  --adapter support-bot \
  --invalid-row-policy skip
```

The API form is:

```json
{
  "dataset_path": "/data/corrections.jsonl",
  "config": {
    "training_profile": "native_online_lora_v1",
    "output_name": "support-bot",
    "invalid_row_policy": "skip"
  }
}
```

Omitting `invalid_row_policy` is identical to specifying `"fail"`. Unknown
values are request-schema errors; there is no truthy or best-effort coercion.

## What makes a row invalid

Admission rejects a row when any of these checks fails:

1. A JSONL row is not valid JSON for an `SftExample`.
2. `messages` is absent or empty.
3. A message role is not `system`, `user`, `assistant`, or `tool`.
4. The effective training template or tokenizer rejects the conversation.
5. Tokenization is empty, shorter than two tokens, or produces no supervised
   assistant token after the causal next-token shift.

Blank JSONL lines are transport whitespace and do not count as rows. Invalid
JSON syntax for the complete HTTP body is an HTTP transport/schema failure, not
an element of the inline `examples` array, so it cannot be skipped as a row.
Once an inline row has deserialized, the same structural and tokenization
policy applies as every other source.

See [SFT Tokenization and Assistant-Only Loss](sft-tokenization.md) for the
normative render and label-mask contract used by the last two checks.

## Stable row identity

The receipt schema is `kiln.sft-ingestion.v1`. A successfully parsed row is
serialized as canonical JSON with recursively sorted object keys, prefixed by
the `kiln.sft-parsed-row.v1` domain, and hashed with SHA-256. Consequently JSON
whitespace, object-key order, source path, dataset name, and transport do not
change a parsed row's identity. Ordered hash lists preserve corpus order and
duplicate multiplicity.

A malformed JSONL row has no parsed representation. Kiln trims surrounding
ASCII whitespace, hashes the remaining bytes under the separate
`kiln.sft-raw-row.v1` domain, and retains only that hash plus the bounded reason
`invalid_json`. Rejected content and parser error text are not copied into the
train receipt.

`kept_corpus_sha256` is an ordered aggregate of `kept_row_hashes`. It is the
SFT `training_data.sha256` and exact-checkpoint data identity, so the same kept
rows have the same training-data identity through every transport. The full
ingestion receipt is separately bound into exact-checkpoint auxiliary state;
changing rejected rows or the policy invalidates exact resume even when the
kept corpus is unchanged.

## Receipt fields

Completed runs and failed runs that reached receipt creation place the
following object at `train_receipt.json -> data.sft_ingestion`:

```json
{
  "schema": "kiln.sft-ingestion.v1",
  "source": "dataset_path",
  "source_locator": "/data/corrections.jsonl",
  "invalid_row_policy": "skip",
  "rows_read": 3,
  "rows_kept": 2,
  "rows_rejected": 1,
  "kept_row_hashes": ["sha256:...", "sha256:..."],
  "rejected_rows": [
    {
      "row_index": 2,
      "row_sha256": "sha256:...",
      "reason": "empty_messages"
    }
  ],
  "kept_corpus_sha256": "sha256:..."
}
```

`row_index` is one-based among non-blank corpus rows. `source` is one of
`inline`, `dataset_path`, `named_dataset`, `corrections`, `recipe`, or
`rust_api`; it describes provenance and never contributes to row identity.

Registered named datasets additionally select a persisted `dataset_split`
(default `train`) and expose the full dataset, split-manifest, and admitted
corpus identities through the public training job. See
[Dataset Splits and Train/Eval Separation](../contracts/DATASET_SPLITS.md) for group-aware
assignment, held-out synthesis, and post-training contamination admission.

The adjacent data counters use the same admission result:

- `examples_read == rows_read`
- `examples_filtered == rows_rejected`
- `examples_trained == rows_kept * epochs` after a complete SFT run

Receipt reads validate counts, hash syntax, rejected-row ordering, and the
kept-corpus aggregate. A malformed or internally inconsistent ingestion object
causes receipt verification to fail.

## Queue consistency

Inline, named-dataset, corrections, and recipe inputs are materialized as
validated examples before queue publication. A local `dataset_path` job keeps
only the path and submit-time ingestion manifest in the queue to avoid pinning
the complete corpus while it waits. The worker reparses and retokenizes the
file, requires the complete manifest to match, and fails the job before GPU
ownership if the selected or rejected rows changed.

For `corrections:active` with `skip`, only kept correction IDs are marked
`trained_into` after successful completion. Rejected corrections remain active
and repairable.

The `SftRequest.ingestion` field is server-owned. HTTP deserialization ignores
caller-supplied values, so clients cannot forge kept/rejected evidence. Direct
Rust callers should use the ordinary `sft_train*` entry points, which construct
the manifest themselves.
