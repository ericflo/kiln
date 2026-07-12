# HF/TRL Interoperability Contract

Kiln's native trainer is the bounded `native_online_lora_v1` single-GPU LoRA
profile described in [NATIVE_SFT_PROFILE.md](NATIVE_SFT_PROFILE.md). General,
distributed, and broadly configurable training belongs in Hugging Face
Transformers, TRL, and PEFT. The interoperability route moves data and
adapters between those systems without discarding the identities needed to
audit the handoff.

The version-1 manifests, validation library, atomic bundle/envelope writers,
pinned SFT reference runner, public SFT export/download CLI and API, and
resident-validated PEFT import API are implemented. The first-party import
upload CLI, GRPO exporter, and GRPO runner are not yet available. A PEFT
directory installed through the generic adapter upload route has not passed
this contract.

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
  Download first when the bundle must be retained. Creation, detail, and
  download return the quoted `export_sha256` as `ETag`. Send that value in
  `If-Match` to delete only those exact bytes; a name reused for a different
  identity returns HTTP 412. An explicit unconditioned operator deletion
  validates the registry target shape rather than requiring intact artifact
  bytes, so a damaged export that cannot be listed or downloaded remains
  removable.

The registry is private (`0700` on Unix). Export creation, download, and
deletion share a dedicated lock; an optional adapter snapshot additionally
holds the adapter revision barrier for the exact copy. A download retains its
registry owner until the stream ends, so deletion cannot race the archive.
Crash-left staging and deletion directories are removed on the next creation,
but unexpected symlinks or special entries fail closed. Download responses use
`Cache-Control: private, no-store` because a deliberately deleted name may be
reused for different bytes.

## Verified CLI Handoff

The CLI performs creation, download, local verification, atomic publication,
and server cleanup as one fail-closed workflow:

```bash
mkdir -p ./handoffs
kiln train hf export-sft \
  --file /data/support-sft.jsonl \
  --name support_run_01 \
  --output ./handoffs/support_run_01.tar.gz \
  --invalid-row-policy fail \
  --input-adapter support-base \
  --split-manifest ./support-split.json
```

`--file` is a path in the running server's filesystem. It intentionally has
the same server-local meaning as native JSONL training and is not uploaded from
the CLI host. Use `--dataset corrections:active` to snapshot active
corrections without marking them trained, or pass another named server
dataset. Exactly one of `--file` and `--dataset` is required. The optional
split file is read by the CLI and must contain a JSON object.

The default local output is `{name}.kiln-hf.tar.gz`. Before creating anything
on the server, the CLI rejects an existing output, a missing output directory,
an invalid name or policy, and an invalid split manifest. It uses an HTTP
client with redirects disabled and computes the compressed archive SHA-256
while streaming into a private sibling temporary file. It does not trust the
response's `download_url`; it constructs the expected name-bound endpoint
itself. Creation and download must both return a strong `ETag` equal to the
manifest identity. A response that remains idle for 120 seconds or a compressed
archive larger than 64 GiB fails instead of consuming unbounded time or disk.

Before local publication, the CLI extracts into a private temporary directory
and requires no more than 4096 entries and 64 GiB of declared expanded bytes.
Every tar entry must be a regular file or directory under the single exact
`{name}.kiln-hf` root; absolute paths, parent traversal, links, special files,
duplicate paths, extra roots, invalid gzip trailers, and empty archives fail.
The pristine bundle verifier then checks the exact recursive file set,
manifest self-digest, every artifact hash and size, and equality with the
creation receipt's `export_sha256`.

Only a fully verified archive is published with a no-clobber atomic rename
followed by a parent-directory sync. Failure leaves no partial local output
and retains the named server export for an explicit retry. After successful
local publication the CLI deletes the server copy by default with
`If-Match: "{export_sha256}"`, so concurrent name reuse cannot delete a
replacement export. Cleanup failure does not discard or mislabel the verified
local artifact; it prints the exact retry command. Use `--keep-server-copy` to
retain it deliberately.

```bash
kiln train hf list
kiln train hf list --json
kiln train hf delete \
  --name support_run_01 \
  --export-sha256 sha256:<64-hex-from-create-or-list>
```

Omitting `--export-sha256` is an intentional unconditioned operator deletion.
Use it only after inspecting the current name, or when a damaged manifest
prevents identity-conditional cleanup.

After extraction, run the copy embedded in the bundle:

```bash
tar -xzf support_run_01.kiln-hf.tar.gz
python ./support_run_01.kiln-hf/train.py ./support_run_01.kiln-hf \
  --base-model /absolute/path/to/the/hf-model
```

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
ignored. New adapters target exactly Kiln's supported `q_proj`, `k_proj`,
`v_proj`, `o_proj`, `in_proj_qkv`, `in_proj_z`, `out_proj`, `gate_proj`,
`up_proj`, and `down_proj` modules by default. `--target-modules` accepts a
comma-separated subset and rejects unknown or duplicate modules; PEFT's
`all-linear` selector is intentionally rejected because it also trains
projections Kiln cannot apply. A modified runner requires
`--allow-custom-script` and remains distinguishable through
`executed_train.py`.

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

## Completed Bundle And Import Envelope

A completed training directory is an exact closed superset of the pristine
export. It contains every original export artifact plus exactly these four
root files:

- `executed_train.py`
- `adapter_config.json`
- `adapter_model.safetensors`
- `kiln_hf_result.json`

`verify_hf_trl_completed_bundle` validates both self-digests, the
result-to-export link, the complete original export, every result byte, and the
exact recursive file set. A partial result, stale sentinel, extra log/checkpoint
file, symlink, or changed source artifact is not a completed bundle.

The import transport uses a derived `.kiln-hf-import` envelope rather than
uploading the training corpus. Its exact ten files are both manifests, the
executed script, PEFT configuration and weights, and the exported model
configuration, tokenizer, inference template, native training template, and
TRL training template. Dataset, ingestion receipt, split, input-adapter,
environment-lock, and reference-runner bytes are intentionally excluded. The
export manifest still binds their identities, while the receiving server gets
the exact model-side bytes needed for resident identity comparison.

`write_hf_trl_import_envelope` first verifies the complete source, copies only
that allowlist into a private sibling staging directory, revalidates the
envelope, fsyncs it, and publishes with a kernel-enforced no-clobber atomic
rename on Linux and Apple platforms. Unsupported platforms fail closed rather
than falling back to an existence-check race.

## Validated PEFT Import API

`POST /v1/train/hf/peft/imports/{name}` installs one completed external result.
The request body is a gzip-compressed tar with `Content-Type:
application/gzip`, absent or `identity` content encoding, and the single exact
root `{name}.kiln-hf-import`. Materialize that directory with
`write_hf_trl_import_envelope`, naming its target for the desired adapter, then
archive and upload it:

```bash
tar -C ./handoffs -czf support-v2.kiln-hf-import.tar.gz \
  support-v2.kiln-hf-import
curl --fail-with-body -X POST \
  http://localhost:8420/v1/train/hf/peft/imports/support-v2 \
  -H 'content-type: application/gzip' \
  --data-binary @support-v2.kiln-hf-import.tar.gz
```

The server streams the compressed body into a private Kiln-owned staging file,
then performs extraction and hashing on a blocking worker. It admits at most 2
GiB compressed, 4 GiB expanded, and 32 tar entries, with a 60-second body-idle
timeout. Manifests are limited to 8 MiB, the executed script to 16 MiB, PEFT
configuration to 1 MiB, other identity artifacts to 512 MiB each, and the
safetensors header to 16 MiB. Paths must be ASCII and unique even under
case-folding. Traversal, aliases, links, special entries, nonzero directory
payloads, nonzero or excessive tar tail data, trailing bytes, another gzip
member, malformed framing headers, and unlisted files fail before publication.

After the envelope verifies, the server recomputes the current served model
ID, complete base-shard manifest, model configuration, tokenizer vocabulary
and configuration, and inference/native/TRL template hashes. Every value must
equal the export. It then validates the PEFT configuration and safetensors
metadata without copying the complete weight file into heap memory: rank,
alpha, task type, target-module set, A/B pairing, floating dtype, layer range,
and every projection dimension must be loadable by the resident model. A
self-consistent result for a different model therefore returns HTTP 409; a
self-consistent but structurally unloadable adapter returns HTTP 400.

Publication shares the adapter mutation barrier and optional disk quota with
the rest of the registry. Quota measurement fails closed on unreadable,
symlinked, or special registry entries. The target name is checked twice and
the final move uses a kernel no-replace operation, so neither an API race nor
an external empty-directory race can replace an existing name. All six final
files are synced before publication: PEFT configuration and weights, executed
script, both source manifests, and `kiln_hf_import.json`
(`kiln.hf-trl-import.v1`). The receipt binds the requested name, task, export
and result digests, current resident identity, reference-script match, and its
own digest. Success returns HTTP 201, the receipt digest as a strong `ETag`,
and the adapter content revision. The first-party command that derives,
archives, streams, and reports this API transaction remains the next CLI
checkpoint.

## Validation Boundary

This contract proves byte and identity continuity across an explicitly
provided handoff. It does not prove that an external process executed the
reported script, that package version strings are trustworthy, or that the
trainer produced a correct optimization result. The pinned runner and oracle
fixtures provide narrower behavioral evidence; real cross-stack round-trip
tests remain required before the route is declared complete.

Import validates both self-digests, the result-to-export link, every transported
file, trainer/task consistency, current resident base/tokenizer/template
identities, and PEFT tensor compatibility before publishing an adapter.
Validation is performed on a private Kiln-owned staging directory. The API
does not attest that an external trainer actually executed the claimed code;
the preserved bytes and reference-script flag make that distinction auditable.
