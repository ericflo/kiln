# HF/TRL Interoperability Contract

Kiln's native trainer is the bounded `native_online_lora_v1` single-GPU LoRA
profile described in [NATIVE_SFT_PROFILE.md](NATIVE_SFT_PROFILE.md). General,
distributed, and broadly configurable training belongs in Hugging Face
Transformers, TRL, and PEFT. The interoperability route moves data and
adapters between those systems without discarding the identities needed to
audit the handoff.

The version-1 manifest, validation library, atomic SFT bundle writer, pinned
SFT reference runner, and public SFT export/download API are implemented. The
CLI transport, validated import API, GRPO exporter, and GRPO runner are not yet
available; until import ships, Kiln does not claim that manually assembled
PEFT output has passed this contract.

## Bundle Model

One handoff directory contains two independently self-verifying documents:

- `kiln_hf_export.json` (`kiln.hf-trl-export.v1`) is written by Kiln before
  external training.
- `kiln_hf_result.json` (`kiln.hf-trl-result.v1`) is written after external
  training and refers to the exact export digest.

The export binds all of the following:

- the served model ID and complete base-weight shard manifest;
- the validated execution provenance of the Kiln process that made the
  export, cross-checked against the exported model/tokenizer/template files;
- exact model configuration, tokenizer, and tokenizer-vocabulary identity;
- inference, native-training, and TRL-training template bytes;
- exact exported dataset bytes, ordered corpus identity, and row count;
- the complete SFT ingestion receipt, including ordered kept and rejected row
  hashes, or the exact rollout-provenance schema for GRPO;
- an optional evaluation split manifest and optional input PEFT adapter; and
- Kiln's exact reference script and pinned Python environment lock.

The result binds the export digest, task, package versions, trainer kind,
effective configuration, preserved executed-script bytes, and exact PEFT
configuration and safetensors output. A custom script is permitted and remains
distinguishable from Kiln's reference script by content hash.

Every artifact identity contains a normalized relative path, nonzero byte
length, and a complete lowercase `sha256:<64 hex>` digest. Bundle paths must
not be absolute, contain `..`, contain backslashes, or traverse symlinks.
Readers reject unsupported schema versions, unknown or duplicate fields,
missing files, non-regular files, and byte or length mismatches.

SFT exports declare `assistant_only_generation_spans`: labels must come from
the TRL template's generation blocks. The compact selection fields in the
export manifest are cross-checked against the complete `sft_ingestion.json`
receipt, including its source, row counts, invalid-row policy, ordered corpus
digest, and internally validated per-row identities.

## Template Identities

The three template artifacts are intentionally distinct:

- `chat_template.jinja` is the inference template loaded by Kiln.
- `kiln_training_chat_template.jinja` is the exact effective template identity
  retained in native checkpoints, receipts, and adapters.
- `training_chat_template.jinja` is the template supplied to HF/TRL.

For the source-pinned Qwen3.5 route, the last file retains TRL's
`{% generation %}` blocks for assistant-only labels. Kiln's native minijinja
renderer cannot parse that extension, so its effective identity uses the same
template with those blocks converted through Kiln's internal span path. The
two hashes must not be substituted for one another.

## Stable Digests

Manifest self-digests exclude only their own digest field. Kiln recursively
sorts JSON object keys, emits compact UTF-8 JSON, hashes the resulting bytes
with SHA-256, and prefixes the lowercase digest with `sha256:`. Arrays retain
their order.

Effective trainer settings use tagged values: `boolean`, `integer`,
`unsigned`, `decimal`, `text`, and `text_list`. Decimal values are finite
numbers carried as exact strings. This avoids cross-language digest drift from
valid but different Rust and Python floating-point JSON spellings.

The result's executed script is preserved as `executed_train.py`. Import can
therefore report whether it is byte-identical to the exported `train.py`
without preventing deliberate custom training code.

## Atomic SFT Bundles

Library-created export directories end in `.kiln-hf` and are immutable. The
writer revalidates every prepared example against the server-owned ingestion
receipt and tokenizer before writing. It then snapshots only the declared
model, tokenizer, template, dataset, receipt, script, environment, split, and
optional adapter artifacts into a new sibling staging directory.

Every file is created without overwrite and synced before its identity enters
the manifest. The complete file set, manifest self-digest, cross-record
identities, and artifact hashes are validated in staging. Publication is one
directory rename followed by a parent-directory sync; in-process failure
removes staging, and an existing final target is never replaced. Optional
input-adapter files reject symlinks and are copied while the caller holds its
adapter revision stable.

## Public SFT Export API

Create one server-owned immutable export with `POST
/v1/train/hf/sft/exports`:

```bash
curl -s http://localhost:8420/v1/train/hf/sft/exports \
  -H 'content-type: application/json' \
  -d '{
    "name": "support_run_01",
    "dataset_path": "/data/support-sft.jsonl",
    "invalid_row_policy": "fail",
    "input_adapter": "support-base",
    "split_manifest": {"schema": "example.split.v1", "train": [0, 1]}
  }'

curl -s http://localhost:8420/v1/train/hf/exports/support_run_01/download \
  -o support_run_01.kiln-hf.tar.gz
```

The request accepts exactly one of inline `examples`, a server-local
`dataset_path`, or an uploaded eval-dataset `dataset` name. The special
`corrections:active` name snapshots currently trainable corrections without
marking them trained. `invalid_row_policy` has the same `fail`/`skip` behavior
and canonical row identities as native SFT. `input_adapter` and the
object-valued `split_manifest` are optional.

Export names are 1 to 128 ASCII bytes, start with an alphanumeric character,
and contain only alphanumerics, `-`, or `_`. They are immutable identities:
creation never replaces an existing name. Inline request bodies are limited
to 256 MiB; use `dataset_path` for larger local corpora. The private registry
holds at most 256 published exports. Dataset admission and hashing run on a
blocking worker rather than an HTTP reactor thread.

The remaining management routes are:

- `GET /v1/train/hf/exports` lists validated manifest summaries without
  rehashing every potentially large artifact.
- `GET /v1/train/hf/exports/{name}` revalidates the exact file set and all
  bytes before returning the complete manifest.
- `GET /v1/train/hf/exports/{name}/download` performs the same full validation
  and streams a gzip-compressed tar whose top directory is
  `{name}.kiln-hf`; it is never buffered as one archive in memory.
- `DELETE /v1/train/hf/exports/{name}` durably renames and removes an export.
  Download first when the bundle must be retained. Explicit deletion validates
  the registry target shape rather than requiring intact artifact bytes, so an
  operator can remove a damaged export that cannot be listed or downloaded.

The registry is private (`0700` on Unix). Export creation, download, and
deletion share a dedicated lock; an optional adapter snapshot additionally
holds the adapter revision barrier for the exact copy. A download retains its
registry owner until the stream ends, so deletion cannot race the archive.
Crash-left staging and deletion directories are removed on the next creation,
but unexpected symlinks or special entries fail closed. Download responses use
`Cache-Control: private, no-store` because a deliberately deleted name may be
reused for different bytes.

## Pinned SFT Runner

`scripts/hf_trl/train_sft.py` is the exact standalone runner embedded in every
public SFT export. `scripts/hf_trl/requirements-sft.lock` pins the public package
versions it accepts:

| Package | Version |
| --- | --- |
| PyTorch | `2.13.0` |
| Transformers | `5.13.1` |
| TRL | `1.8.0` |
| PEFT | `0.19.1` |
| Datasets | `5.0.0` |
| Accelerate | `1.14.0` |
| Tokenizers | `0.22.2` |
| Safetensors | `0.8.0` |
| Jinja2 | `3.1.6` |

Install the PyTorch build from the official index appropriate to CUDA, ROCm,
CPU, or MPS while retaining the exact public version, then install the
remaining pinned requirements. The script rejects a missing package, version
drift, or a PyTorch version other than the allowed platform-suffixed pin.

Run the exported copy, not the repository source:

```bash
python ./my-run.kiln-hf/train.py ./my-run.kiln-hf \
  --base-model /absolute/path/to/the/hf-model
```

`--base-model` is deliberately local and explicit. Before importing PyTorch,
the runner verifies the export self-digest, closed schema, exact recursive
file set, every artifact size and hash, complete SFT selection receipt,
generation-span template, environment lock, local tokenizer bytes, and every
base-weight shard byte. `--verify-only` performs this dependency-free
preflight.

The default route uses TRL `SFTTrainer`, assistant-only generation masks,
PEFT LoRA, no packing or dataset shuffle, AdamW, and a materialized seed. It
derives `max_length` from the admitted rows and refuses an explicit value that
would truncate any row. Input adapters are resumed as trainable PEFT adapters;
new-adapter LoRA shape flags are rejected in that mode rather than silently
ignored. A modified runner requires `--allow-custom-script` and remains
distinguishable through `executed_train.py`.

The pinned v1 runner is deliberately a single-process correctness reference;
it rejects `WORLD_SIZE != 1` instead of allowing multiple ranks to race result
publication. The bundle format itself supports custom distributed
Transformers/TRL/PEFT training. Run that through an explicit modified script
with `--allow-custom-script` so the result records the exact code that ran.

Training uses a sibling temporary work directory. On rank zero, the runner
publishes `executed_train.py`, `adapter_config.json`, and
`adapter_model.safetensors`, then creates the self-verifying
`kiln_hf_result.json` last. Import therefore cannot mistake an interrupted
publication for a complete result. In-process failures clean partial result
files; a later invocation can recover a result carrying the explicit
incomplete sentinel, while unattributed output files fail closed.

## Validation Boundary

This contract proves byte and identity continuity across an explicitly
provided handoff. It does not prove that an external process executed the
reported script, that package version strings are trustworthy, or that the
trainer produced a correct optimization result. The pinned runner and oracle
fixtures provide narrower behavioral evidence; real cross-stack round-trip
tests remain required before the route is declared complete.

Import must validate both self-digests, the result-to-export link, every
referenced file, trainer/task consistency, and current resident
base/tokenizer/template identities before publishing an adapter. Validation is
performed on a Kiln-owned, quiescent bundle directory; concurrently mutating a
bundle during validation is outside the contract.
