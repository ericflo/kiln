# HF/TRL export and PEFT import

Use this workflow when Kiln's fixed
[`native_online_lora_v1`](NATIVE_SFT_PROFILE.md) profile does not fit your
training run. Kiln exports an immutable SFT or recorded-GRPO bundle, an
external Hugging Face Transformers/TRL/PEFT process trains the adapter, and
Kiln verifies the completed result against the running server before importing
it.

This contract preserves the identity of the data, model, tokenizer, templates,
script, settings, and adapter files across that handoff. It does **not** prove
that external training was correct or that a custom script executed as
reported. A bare PEFT directory installed through the generic adapter upload
route has not passed these checks.

The generated [Artifact lifecycle API
schema](../../contracts/kiln-artifacts-v1.schema.json) is the source of truth for
HTTP fields and validation constraints. This guide explains the workflow and
its trust boundary.

## End-to-end workflow

1. Export and download a verified bundle with `kiln train hf export-sft` or
   `kiln train hf export-grpo`.
2. Extract the archive and run the `train.py` copy inside it. The pinned
   reference runner supports SFT and recorded GRPO; a modified script must use
   `--allow-custom-script`.
3. Import the completed directory with `kiln train hf import-peft`. Kiln
   verifies the result and resident model identity before publishing the new
   adapter.

```bash
mkdir -p ./handoffs
kiln train hf export-sft \
  --file /data/support-sft.jsonl \
  --name support_run_01 \
  --output ./handoffs/support_run_01.kiln-hf.tar.gz

tar -xzf ./handoffs/support_run_01.kiln-hf.tar.gz -C ./handoffs
python ./handoffs/support_run_01.kiln-hf/train.py \
  ./handoffs/support_run_01.kiln-hf \
  --base-model /absolute/path/to/the/hf-model

kiln train hf import-peft \
  --bundle ./handoffs/support_run_01.kiln-hf \
  --name support-v2
```

The `--file` value is read by the running Kiln server. It is not uploaded from
the CLI machine.

## What crosses the boundary

| Direction | Included | Deliberately excluded |
| --- | --- | --- |
| Kiln to external trainer | Dataset, model and weight-shard identity, tokenizer, three templates, optional split and input adapter, reference runner, environment lock, and export manifest | Server configuration and unrelated adapter-registry contents |
| External trainer to Kiln | Export and result manifests, executed script, PEFT configuration and weights, model configuration, tokenizer, and three templates | Training corpus, ingestion receipt, split data, input-adapter bytes, environment lock, and reference-runner bytes |

The import envelope excludes the corpus to avoid uploading training data back
to the server. Its export manifest still binds the excluded files by digest.

## Bundle model

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

### Recorded-GRPO corpus requirements

A GRPO manifest is not accepted merely because it names
`kiln.rollout-provenance.v1`. Its `train.jsonl` must contain one canonical
compact JSON group per LF-terminated line, with no blank rows, aliases,
unknown or duplicate fields, alternate whitespace, or missing final newline.
The verifier streams at most 64 GiB, permits at most 10 million groups and 256
MiB per row, requires 2 to 1024 completions per group, and requires one
uniform group width so the pinned TRL runner cannot drop or reshape examples.
The separate
`kiln.hf-trl-grpo-corpus.v1` digest length-frames every canonical row in order;
the dataset file identity also binds the exact newline bytes.

Every completion must carry validated exact rollout provenance and a finite
reward. The verifier recomputes prompt and scored-payload identities, requires
the complete served-model and base-shard identity, binds either the base model
or the exact exported input-adapter content revision, and rejects mixed
behavior-policy identities. It reconstructs the exported tokenizer and
inference template, then replays each group through Kiln's production
tokenization and action/environment-mask path. Tokenizer bytes, vocabulary,
template invocation, prompt boundary, complete token sequence, sampled versus
forced action positions, and sampled behavior log-probabilities must agree.

The synchronous `write_hf_trl_grpo_bundle` library API accepts either borrowed
`GrpoGroup` values or an existing canonical JSONL file. In-memory groups are
serialized canonically. File sources must already use the exact compact,
final-LF representation; links, special files, path replacement while opening,
and noncanonical-but-parseable JSON fail closed instead of being silently
normalized. Both source forms produce the same corpus and export identity for
the same groups.

The writer snapshots the same exact model, tokenizer, three templates, source
execution provenance, optional split, optional input adapter, reference script,
and environment lock as the SFT writer. It writes create-new files into a
private sibling staging directory, fsyncs them, runs the complete recursive
file-set and deep corpus verifier, and publishes with one kernel-enforced
no-replace rename. It syncs the parent and verifies the published directory
again. Failure cleans staging, existing targets are never replaced, and later
source mutation cannot change the published bundle.

The same writer backs `POST /v1/train/hf/grpo/exports`. The API accepts
exactly one provenance-complete inline `groups` array or server-local canonical
`dataset_path`, plus the same optional split manifest and revision-stable input
adapter as SFT. Both forms enter the existing immutable registry and therefore
share its verification, download, conditional deletion, capacity, locking, and
crash-recovery behavior. The task-aware pinned runner and first-party
`export-grpo` CLI support recorded GRPO. The production-model round-trip
workload described below is the local hardware gate for proving that the
complete route works in one captured environment.

## Template identities

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

## Stable digests

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

## Atomic SFT bundle creation

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

## HTTP export API

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
`dataset_path`, or a registered eval-dataset `dataset` name. The special
`corrections:active` name snapshots currently trainable corrections without
marking them trained. `invalid_row_policy` has the same `fail`/`skip` behavior
and canonical row identities as native SFT. `input_adapter` and the
object-valued `split_manifest` are optional.

Create a recorded-rollout GRPO export with `POST
/v1/train/hf/grpo/exports`:

```bash
curl -s http://localhost:8420/v1/train/hf/grpo/exports \
  -H 'content-type: application/json' \
  -d '{
    "name": "support_grpo_01",
    "dataset_path": "/data/support-rollouts.jsonl",
    "split_manifest": {"schema": "example.split.v1", "train": [0, 1]}
  }'
```

The request accepts exactly one non-empty inline `groups` array or
`dataset_path`. The path is read by the server and must already contain the
canonical compact, final-LF JSONL emitted by Kiln's rollout tooling. Inline
groups are canonically serialized by the server. In both forms, every
completion must carry exact recorded rollout provenance matching the resident
model, base shards, tokenizer, inference template, and optional input adapter;
invalid or mixed behavior-policy data cannot publish an export.
Admission failures return `hf_trl_invalid_request` before registry publication;
filesystem, locking, or post-admission verification failures remain distinct
`hf_trl_export_failed` server errors.

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

## CLI export and download

The SFT and recorded-GRPO commands create the server export, download it,
verify it locally, publish it without replacing an existing file, and clean up
the server copy:

```bash
mkdir -p ./handoffs
kiln train hf export-sft \
  --file /data/support-sft.jsonl \
  --name support_run_01 \
  --output ./handoffs/support_run_01.kiln-hf.tar.gz \
  --invalid-row-policy fail \
  --input-adapter support-base \
  --split-manifest ./support-split.json

kiln train hf export-grpo \
  --file /data/support-recorded-rollouts.jsonl \
  --name support_grpo_01 \
  --output ./handoffs/support_grpo_01.kiln-hf.tar.gz \
  --split-manifest ./support-grpo-split.json
```

For `export-sft`, choose exactly one of `--file` and `--dataset`. `--file` is a
path in the running server's filesystem, with the same server-local meaning as
native JSONL training; it is not uploaded from the CLI host. Use `--dataset
corrections:active` to snapshot active corrections without marking them
trained, or pass another named server dataset. The optional split file is read
by the CLI and must contain a JSON object.

`export-grpo --file` has the same server-local path semantics and requires the
canonical, final-LF, provenance-complete JSONL contract described above. It
does not accept a named SFT dataset or an invalid-row policy; malformed rows
fail the whole export before publication.

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

After extraction, run the script embedded in the bundle:

```bash
tar -xzf ./handoffs/support_run_01.kiln-hf.tar.gz -C ./handoffs
python ./handoffs/support_run_01.kiln-hf/train.py \
  ./handoffs/support_run_01.kiln-hf \
  --base-model /absolute/path/to/the/hf-model
```

## Pinned reference runner

`scripts/hf_trl/train_sft.py` is the standalone runner embedded in every SFT
export and exposed by the GRPO bundle writer. Despite its historical filename,
it reads the task from the self-verifying manifest and supports both SFT and
provenance-complete recorded GRPO.
`scripts/hf_trl/requirements-sft.lock` pins the versions it accepts:

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
file set, every artifact size and hash, environment lock, local tokenizer
semantics, and every base-weight shard byte. Tokenizer comparison normalizes
only Tokenizers' equivalent legacy string-pair and current two-token-array BPE
merge framing; any vocabulary, merge, normalizer, pre-tokenizer,
post-processor, decoder, or added-token drift still fails. SFT additionally
verifies its complete selection receipt and generation-span template. GRPO
streams and reconstructs the typed canonical corpus, then checks every prompt,
scored payload,
tokenizer, template, behavior policy, adapter revision, sampling control,
thinking budget, token boundary, sampled/forced action, behavior log-probability,
row count, and ordered corpus digest. It enforces the same 64 GiB dataset,
256 MiB row, ten-million-group, and 2-to-1,024-completion bounds as the Rust
verifier. `--verify-only` performs this dependency-free preflight.

The default route uses TRL `SFTTrainer`, assistant-only generation masks,
PEFT LoRA, no packing or dataset shuffle, AdamW, and an explicit effective
seed. It
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

For GRPO, the runner does not ask Transformers or vLLM to generate replacement
samples. It copies the verified JSONL into a private work snapshot, rechecks
that snapshot at every epoch boundary, and supplies TRL the exact recorded
prompt and suffix token IDs. Recorded sampled-token log-probabilities become
TRL's `old_per_token_logps`; they alone define the PPO importance ratio.
Forced thinking-budget close tokens and trajectory environment tokens remain
in sequence context but receive `env_mask=0`, so they cannot become policy
targets. Recorded rewards are forwarded unchanged to the reward function.
The independently frozen initial adapter, or the base model for a new LoRA,
supplies KL-reference probabilities. It is never substituted with recorded
behavior probabilities.

TRL 1.8.0 has one run-global `num_generations` and drops an incomplete unique
prompt batch. The correctness runner therefore requires one uniform completion
count across groups and requires `--batch-size` to divide the group count. It
fixes `steps_per_generation=1`, `num_iterations=1`, disables dataset shuffle,
vLLM generation, truncation masking, and KL bias correction, and fails instead
of omitting corpus rows. GRPO defaults are DAPO token-level aggregation,
unscaled group-relative rewards, token-level recorded-policy ratios, symmetric
epsilon `0.2`, and beta `0.1`. TRL's independent KL term is the K3 estimator;
the exact effective configuration records that fact. `--loss-type`,
`--scale-rewards`, `--importance-sampling-level`, `--epsilon-high`, and
`--beta` expose the bounded alternatives supported by this pinned route.
`--max-length` is SFT-only because GRPO sequence boundaries are provenance,
not a retokenization choice.

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

## Completed bundle and import envelope

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

## CLI adapter import

Use the completed extracted directory produced by the embedded runner, not the
downloaded tarball and not a bare PEFT directory:

```bash
kiln train hf import-peft \
  --bundle ./support_run_01.kiln-hf \
  --name support-v2 \
  --url http://localhost:8420
```

Adapter names are portable archive roots: 1 to 128 ASCII bytes, beginning with
an alphanumeric character, followed only by alphanumerics, `-`, `_`, or `.`,
with no `..`. Before making a request, the CLI fully verifies the completed
bundle, materializes the exact ten-file envelope in a private temporary
directory, revalidates every copied byte, and applies the server's 4 GiB
expanded and per-file limits. A malformed, partial, changed, linked, or
over-limit local result therefore never contacts the server.

The request body is generated on a blocking worker as deterministic USTAR plus
gzip and fed directly to reqwest through a four-chunk bounded channel. Each
chunk is at most 256 KiB and compressed output is capped at 2 GiB, so neither
the adapter nor archive is buffered in memory. USTAR prefix fields carry the
bounded root without GNU/PAX extension entries that the strict server would
reject. Redirects are disabled and response reads have a 120-second idle
timeout. Private envelope staging is removed on success and every error; the
completed `--bundle` directory is never modified or removed.

Before upload, exact resident equivalence lets the CLI derive the receipt the
server must publish, including `import_sha256`, adapter content revision,
reference-script match, installed byte count, and six-file count. Success is
accepted only as HTTP 201 with JSON and one strong `ETag` matching those local
values plus the task and export/result identities. A normal structured API
rejection means no adapter was published. A connection failure after sending
the request has an unknown outcome; the CLI says so and directs the operator
to inspect `kiln adapters list` before retrying. If an invalid success response
arrives, it similarly warns that the named adapter may already be installed.
An existing name returns the server's `adapter_already_exists` error and is
never replaced.

## HTTP adapter import

`POST /v1/train/hf/peft/imports/{name}` is the lower-level transport used by
the verified CLI and installs one completed external result.
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
archives, streams, predicts, and verifies this API transaction is
`kiln train hf import-peft`.

## Hardware qualification

The committed `hf-trl-production-roundtrip-v1` workload is an end-to-end
correctness gate for one explicit backend, model, binary, and host. It runs
with external network access disabled and loopback enabled. For both SFT and
recorded GRPO, it:

1. starts the selected production Kiln backend against the real model;
2. generates two exact scored GRPO completions and exports both task bundles;
3. drains and stops Kiln before the external process acquires the accelerator;
4. verifies each bundle and runs one BF16 rank-1 `q_proj` update through the
   bundle's pinned HF/TRL/PEFT script;
5. restarts the same Kiln binary and requires identical execution provenance;
6. imports both completed bundles through `kiln train hf import-peft`;
7. requires nonzero adapter tensors, a measurable LoRA delta, successful
   server load, hash-bound paired base/adapter evals, a healthy unquarantined
   backend, zero HTTP errors, a drained scheduler, and zero live KV blocks; and
8. stops the server cleanly so its private model snapshot must be removed.

The external environment must already contain the exact versions from
`scripts/hf_trl/requirements-sft.lock`, with PyTorch installed from the index
for the accelerator platform. Build `target/release/kiln` for the selected
backend, commit the workload and source first, then run it from a clean tree.
Use a stable identifier for the machine that produced the receipt:

```bash
python3 scripts/qualification/run.py \
  qualification/workloads/hf-trl-production-roundtrip-v1.json \
  --variant rocm \
  --host-id local-accelerator-host \
  --model /absolute/path/to/Qwen3.5-4B \
  --model-id Qwen/Qwen3.5-4B \
  --var trainer_python=/absolute/path/to/pinned-venv/bin/python
```

Use `--variant vulkan` with a Vulkan-built server binary. The workload
currently defines ROCm and Vulkan variants but no cross-backend comparison
policy. The repository's committed receipts for this workload were captured on
one Strix Halo host; they qualify only the source, binary, backend, model, and
host recorded in each receipt. They do not establish performance or
correctness for every ROCm or Vulkan device.

CUDA and Metal need explicit workload variants and their own passed receipts
before they can claim this gate. A compile check, a receipt from another
device, or a manually assembled transcript is not a substitute.

## What the contract does not prove

The bundle contract proves byte and identity continuity across an explicitly
provided handoff. By itself it does not prove that an external process executed
the reported script, that package version strings are trustworthy, or that the
trainer produced a correct optimization result. The pinned scalar/tensor
oracles cover the update equations; the local production-model workload adds
process, package-lock, accelerator, import, and inference evidence bound to a
specific source tree, binary, backend, model, and host. Neither claim extends
to a backend that lacks its own passed receipt.

Import validates both self-digests, the result-to-export link, every transported
file, trainer/task consistency, current resident base/tokenizer/template
identities, and PEFT tensor compatibility before publishing an adapter.
Validation is performed on a private Kiln-owned staging directory. The API
does not attest that an external trainer actually executed the claimed code;
the preserved bytes and reference-script flag make that distinction auditable.
