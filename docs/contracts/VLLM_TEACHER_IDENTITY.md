# Immutable vLLM teachers

Use Kiln's launcher when a remote vLLM deployment will supply numeric
prompt-logprobs for OPD. The launcher stages an immutable model snapshot,
binds the runtime and selected accelerator into a `TeacherIdentityV1`, disables
mutable model surfaces, and supervises the complete vLLM process group.

There are no machine-specific serving defaults in this contract. The launcher
records the accelerator and every accepted inference option that the operator
selected. Device-specific choices belong in per-machine qualification
artifacts, not in the general runtime path.

## What the identity proves

| The identity binds | The identity does not prove |
| --- | --- |
| Exact base-weight, tokenizer, optional static-adapter, and model-tree content | Numerical parity with Kiln, Transformers, or another engine |
| Python implementation and the installed vLLM, torch, Transformers, and tokenizers content | Deterministic output from every kernel or backend |
| Selected accelerator, driver/runtime facts, visible devices, and inference-affecting options | Throughput, latency, memory safety, or fitness for a workload |
| Context, top-K, response-budget, raw-logprob, and model-ID limits | The identity of a network peer |

A stock vLLM fingerprint is not enough: it does not prove all of these inputs.
Kiln therefore rejects stock `vllm-*` fingerprints and accepts only the
canonical `kiln-teacher-v1` fingerprint carried by every scoring response.

## Before you launch

You need:

- vLLM 0.20.1rc0 or newer, plus PyTorch, Transformers, and tokenizers in the
  Python environment used to run the launcher. This is a code-level minimum,
  not a qualification result; every real launch probes and hashes the installed
  runtime again.
- A local Hugging Face model directory with safetensors base weights. The
  launcher rejects Hub IDs, revisions, `trust_remote_code`, alternate
  tokenizers, and alternate base-model inputs.
- A fully materialized model directory. Symlinks and special files anywhere
  below the model or adapter root are rejected, so do not pass a symlink-based
  Hugging Face cache checkout directly.
- Private snapshot and runtime-cache parents with enough free bytes and inodes
  for a full copy plus bounded metadata headroom. Capacity checks assume copy
  fallback even when reflinks are available.

The final snapshot and cache parents must be owned by the launcher UID and have
mode `0700`. Model, adapter, snapshot, and cache roots must be distinct and
must not nest. Their ancestry must be owned by the launcher UID or root and
must not permit unsafe renames; a sticky shared directory such as `/tmp` is
allowed. The launcher also recognizes the kernel's overflow UID only for the
single-entry rootless user-namespace mapping it validates explicitly. This
exception applies to ancestry, never to the private final parent.

For a static adapter, provide one local PEFT directory containing
`adapter_config.json` and exactly one of `adapter_model.safetensors` or
`adapter_model.bin`. Prefer safetensors. A pickle-based `.bin` file is
executable input and must come from a trusted source.

## Launch a base teacher

Additional vLLM arguments follow one `--` delimiter. Each option must use one
unambiguous `--key=value` token, except for the launcher's small set of vetted
valueless switches.

This device-neutral example lets vLLM choose its supported attention backend
and dtype:

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/teacher-model \
  --snapshot-root=/var/tmp/kiln-teacher-snapshots \
  --cache-root=/var/tmp/kiln-vllm-runtime-caches \
  --served-model-id=teacher-large \
  --max-top-k=32 \
  --max-model-len=32768 \
  --max-prompt-logprob-candidates=1000000 \
  --max-provenance-read-mib-per-second=256 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --api-key=replace-with-a-random-secret
```

Select an explicit dtype, attention backend, memory fraction, or scheduler only
after qualifying it on the intended device. Accepted values become part of the
teacher identity; the launcher does not infer a machine model and inject
hardware-specific tuning.

Immediately before spawn, the launcher prints the fingerprint and private
snapshot path. It invokes the same Python interpreter without a shell, uses `/`
as the working directory, and points vLLM only at the staged tree:

```text
python3 -m vllm.entrypoints.cli.main serve \
  /var/tmp/kiln-teacher-snapshots/ready-.../model \
  --served-model-name=teacher-large \
  --max-model-len=32768 \
  --max-logprobs=32 \
  --logprobs-mode=raw_logprobs \
  --generation-config=vllm \
  --load-format=safetensors \
  --fingerprint-mode=custom \
  --fingerprint-value=kiln-teacher-v1.<base64url-json>.<sha256>
```

The base loader is always safetensors. Caller-supplied `--load-format` is
rejected, and alternate `.bin` files cannot win loader selection. Every extra
file in the model tree still participates in the snapshot digest.

`--max-provenance-read-mib-per-second` accepts `1..=16384`. One limiter spans
snapshot copying and verification, source hashing, model and optional-adapter
fingerprinting, and installed-runtime hashing. It does not reset between files
or phases. The pre-spawn child recheck uses the same numeric ceiling in its own
process. Omit the option for the historical unlimited startup mode; it never
changes timed request behavior or semantic identity.

## Launch a static-adapter teacher

Add one immutable adapter path and give the deployment one public model ID:

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/teacher-model \
  --adapter-path=/adapters/domain-v7 \
  --snapshot-root=/var/tmp/kiln-teacher-snapshots \
  --cache-root=/var/tmp/kiln-vllm-runtime-caches \
  --served-model-id=teacher-domain-v7 \
  --max-top-k=32 \
  --max-model-len=32768 \
  --max-prompt-logprob-candidates=1000000 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --api-key=replace-with-a-random-secret
```

The launcher adds exactly one staged static module:

```text
--enable-lora --max-loras=1 --max-cpu-loras=1 \
--max-lora-rank=<config-derived-cap> \
--lora-modules={"name":"teacher-domain-v7","path":".../ready-.../adapter","base_model_name":"kiln-base-..."}
```

Runtime adapter load/unload, resolver plugins, prompt adapters, multiple
LoRAs, and caller-supplied adapter options are rejected.

vLLM still exposes its internal base alias when serving a LoRA, and its custom
fingerprint is process-wide. A base response can therefore carry the adapter
deployment's fingerprint without carrying adapter logits. Kiln closes that
bypass by requiring both the request model and response `model` to equal the
identity's `served_model_id`. Any non-Kiln client must enforce the same check.
If an endpoint must expose only the adapter name, put it behind an
authenticated proxy that rejects the internal base ID.

## How snapshot isolation works

The staged snapshot is the only tree vLLM can load:

1. The launcher validates limits, environment variables, and extra vLLM
   options before copying model data.
2. It inventories the complete model and optional adapter trees and enforces
   aggregate file, directory, byte, path, depth, free-space, inode, and
   manifest bounds.
3. It creates a private `.building-*` directory beneath an already-open
   snapshot-root descriptor. It reflinks each regular file where supported or
   copies it otherwise, then hashes source and destination for the exact
   initially observed length.
4. It records every directory and each file's path, byte length, and SHA-256
   in a canonical manifest. Files become `0400`, directories become `0500`,
   and the launcher re-enumerates and re-hashes the complete tree.
5. It atomically renames the directory to `ready-*`, synchronizes the parent,
   verifies the tree again, and computes all identities from staged paths.
6. It verifies the snapshot immediately before spawning vLLM.
7. On exit, it removes the snapshot through the anchored parent descriptor
   without following links or crossing filesystems.

Growth, truncation, replacement, links, special files, and mutation during
copy or verification fail the launch. Source files may change after a
successful copy without changing the running teacher. Reflinks remain
copy-on-write.

The default snapshot parent is `~/.cache/kiln/teacher-snapshots`. Set
`--snapshot-root=/dedicated/path` or `KILN_VLLM_SNAPSHOT_ROOT` to override it.
A same-filesystem parent improves the chance of reflink success.

## How runtime-cache isolation works

Every real launch receives a fresh, empty cache. `--cache-root` names only its
private parent and defaults to `~/.cache/kiln/vllm-runtime-caches`. The
launcher creates a unique mode-`0700` `cache-<pid>-<nonce>` child and supplies
that path to vLLM as `VLLM_CACHE_ROOT`. An ambient `VLLM_CACHE_ROOT` is
rejected.

The nonce path is excluded from the inference digest because the child is
empty at spawn, is never reused, and is output-only. The private-cache policy
itself is hashed. A launch therefore cannot inherit compiled code, model
metadata, or autotuning state from another user, runtime, model, profile, or
benchmark arm.

After the process group exits, the launcher removes the cache through its
anchored parent descriptor. An unrecoverable kill can leave an ignored
`cache-*` child, but no later launch consumes it. Cold compilation can require
substantial disk and startup time; those costs are qualification isolation,
not measured request throughput.

## Process-group ownership

The default `--process-group-mode=detached` creates a launcher-owned vLLM
session. The launcher forwards `SIGINT`, `SIGTERM`, `SIGHUP`, and `SIGQUIT`,
drains descendants, and then removes the snapshot and runtime cache.

An owned local serving benchmark uses
`--process-group-mode=inherited`. This Linux-only mode is accepted only when
the launcher is already the leader of an isolated process group. The benchmark
driver creates that group before snapshot or model work, then signals the
complete group during shutdown. The launcher drains peers without signaling
itself and retains cleanup ownership.

## Response-allocation limit

`max_prompt_logprob_candidates` is the aggregate candidate ceiling for one
response, not the top-K width of one position. A row may contain up to
`min(max_top_k + 1, vocab_size)` candidates because vLLM can return the sampled
token in addition to the requested top-K.

The ceiling must fit at least one maximum-width row. It cannot exceed either
1,000,000 candidates or
`max_model_len × min(max_top_k + 1, vocab_size)`. When omitted, the launcher
uses the smaller of those two maxima. The value is identity-bound, and Kiln
rejects oversized responses before allocating the complete candidate payload.

## Identity contract

The compact identity uses this normative field order:

```text
schema, protocol, served_model_id, base_model_sha256,
tokenizer_vocab_sha256, tokenizer_config_sha256, adapter, vocab_size,
max_top_k, max_model_len, max_prompt_logprob_candidates, logprobs_mode,
implementation, inference_config_sha256
```

Its constants are:

```text
schema:         kiln.teacher-identity.v1
protocol:       vllm.prompt-logprobs.numeric-token-ids.causal.v1
logprobs_mode:  raw_logprobs
fingerprint:    kiln-teacher-v1.<base64url without padding>.<sha256 of JSON>
```

All SHA fields contain exactly 64 lowercase hexadecimal characters without a
`sha256:` prefix.

### Model and tokenizer identity

`base_model_sha256` matches Kiln's Rust loader. For each safetensors shard,
Kiln records the raw-byte digest and length, sorts by digest and then length,
and hashes the domain `kiln.base-model-content.v1\0`, the little-endian `u64`
record count, and each little-endian length plus raw 32-byte digest.

`tokenizer_vocab_sha256` hashes the complete Transformers `get_vocab()` map.
Pairs are sorted by `(u32 token ID, raw UTF-8 token bytes)` and encoded under
`kiln.tokenizer-vocab.v1\0`. The pair count and
`get_vocab_size(with_added_tokens=true)` must agree. Every token ID must fit
the model's vocabulary width from `config.json`; `config.json.vocab_size` and
`text_config.vocab_size` must agree when both exist. The model width may exceed
the tokenizer entry count because some models reserve padded rows.

`tokenizer_config_sha256` hashes the exact UTF-8 fast-tokenizer backend JSON
from `backend_tokenizer.to_str()`, matching Rust
`Tokenizer::to_string(false)`.

For a static adapter, `weights_sha256` is a domain-separated digest over the
selected filename, byte length, and raw weight digest. `config_sha256` is the
raw `adapter_config.json` digest. The adapter rank and every `rank_pattern`
value select the smallest supported vLLM rank cap, which is included in the
inference digest.

### Runtime and inference identity

`inference_config_sha256` binds:

- the full staged-tree manifest digest and exact `config.json`;
- context, top-K, aggregate response budget, raw-logprob mode, safetensors
  format, adapter mode, and every accepted non-transport vLLM option;
- the Python implementation and version, resolved interpreter, and the
  versions and installed content of vLLM, torch, Transformers, and tokenizers;
- accelerator type, driver/runtime identity, visible device names,
  architectures, and total memory;
- inference-affecting vLLM, CUDA, ROCm/HIP, RCCL/NCCL, torch, Triton, Hugging
  Face, and native-library environment values; and
- deterministic policy inputs such as seed, eager mode, `PYTHONHASHSEED`, TF32
  overrides, cuBLAS workspace policy, and cuDNN policy.

Installed-content hashing covers the actual import trees, distribution
metadata and recorded files, Python and native-extension files, and
editable-install source. A fresh child re-hashes the contract under the exact
launch environment immediately before spawn.

These are determinism inputs; the identity does not promise numerical
determinism. Host, port, API key, TLS, and CORS are transport settings and are
excluded because they do not alter logits. The API key is never embedded in
the identity and is redacted from dry-run output.

## Accepted options and environment

The launcher owns all model, tokenizer, weight-format, generation-config,
logprob, fingerprint, middleware, LoRA, and adapter arguments. It accepts
transport options and a versioned allowlist of reviewed scalar inference
options. Unknown options fail closed. File, path, directory, and template
inputs are rejected because hashing a path string would not bind its content.

`--attention-backend` has a closed value set because vLLM can interpret the
value as an importable class. The only reviewed explicit value is
`TRITON_ATTN`; omit the option for vLLM's automatic supported selection.
Qualifying another backend requires adding its exact name to the launcher,
testing the rejection boundary, and retaining correctness and performance
evidence for the intended machine.

`--language-model-only` is a reviewed, identity-bound valueless switch for
hybrid text/multimodal architectures. It sets every multimodal input limit to
zero. Use it only for a qualified text-only deployment; it does not synthesize
missing image or audio processors.

The launcher rejects `PYTHONPATH`, `PYTHONHOME`, Python safe/user-site
overrides, `LD_PRELOAD`, and `LD_LIBRARY_PATH` for real launches. It also
rejects file-, path-, plugin-, executable-, home-, root-, and cache-naming
environment inputs in the vLLM, CUDA, ROCm, torch, Triton, and Hugging Face
namespaces, including logging configurations and
`HIPBLASLT_TUNING_OVERRIDE_FILE`. Device visibility variables such as
`ROCR_VISIBLE_DEVICES` are accepted but identity-bound.

Runtime hashing is bounded at 250,000 files, 100,000 directories, 64 GiB of
logical content, 128 directory levels, 4,096 bytes per canonical label, and
64 MiB of aggregate labels. Missing distribution files, ambiguous namespace
roots, unreadable or special files, path changes, and content mutation abort
the launch. The child receives `PYTHONDONTWRITEBYTECODE=1`, so importing the
launcher cannot create bytecode inside an identity-bound package tree.

`VLLM_ALLOW_RUNTIME_LORA_UPDATING` is forced to `0`. `VLLM_PLUGINS`, resolver
configuration, and skipped model-name validation fail before spawn.

## Register the teacher with Kiln

The generated [Artifact Lifecycle API
Schema](../../contracts/kiln-artifacts-v1.schema.json) is authoritative for
teacher request and response fields.

A credential-free teacher must use an exact loopback URL, and Kiln itself must
also be bound to loopback. For an off-host teacher—or a network-bound Kiln
server—configure an exact canonical HTTPS origin and a server-owned secret
environment variable:

```toml
# kiln.toml
[teachers.credentials.primary-vllm]
origin = "https://teacher.example.com:8443"
api_key_env = "KILN_VLLM_API_KEY"
```

```bash
export KILN_VLLM_API_KEY='replace-with-the-teacher-secret'
kiln serve --config kiln.toml

curl -fsS -X POST http://localhost:8420/v1/teachers \
  -H 'content-type: application/json' \
  -d '{
    "alias": "teacher-large@vllm",
    "kind": "remote",
    "provider": "vllm",
    "model_id": "teacher-large",
    "url": "https://teacher.example.com:8443",
    "credential_id": "primary-vllm"
  }'
```

The TOML stores only the secret variable's name and authorized origin. Kiln
validates that variable at startup, checks it again when resolving a credential
handle, and reads the bearer value immediately before an outbound request. The
secret is never stored in application state, serialized, logged, cached, or
written to a receipt. Missing, non-Unicode, and whitespace-only values fail
closed, and authenticated error responses are redacted.

Registration cannot supply identity, tokenizer hash, vocabulary size, top-K,
full-vocabulary support, or a secret environment-variable name. Kiln sends one
top-1 scoring probe and one probe at the deployment's advertised maximum K.
Both must carry the same canonical fingerprint and valid numeric vocabulary.
Kiln also requires the teacher's numeric token IDs and vocabulary width to
match the loaded student.

Only after both probes pass does Kiln persist and publish the alias. The
response includes status, usability, identity revision, bounds, and the exact
`off_policy_manifest`. Aliases are immutable; delete and re-register to change
a deployment. Each queued job pins the complete teacher spec and repeats both
probes before GPU ownership or a cache lookup. Every later scoring response
must carry the same fingerprint.

Older registries can contain `api_key_env`. Kiln removes that legacy field
without trusting it and atomically rewrites the registry. The migrated alias
remains unusable until it is deleted and registered again with `credential_id`
and a fresh probe.

## Inspect an identity without launching vLLM

Tests can supply a strict precomputed input without importing vLLM,
Transformers, or torch and without reading model weights. This mode can never
launch a server.

```json
{
  "schema": "kiln.vllm-teacher-input.v2",
  "base_model_sha256": "<64 lowercase hex>",
  "snapshot_content_sha256": "<64 lowercase hex>",
  "model_config_sha256": "<64 lowercase hex>",
  "tokenizer_vocab_sha256": "<64 lowercase hex>",
  "tokenizer_config_sha256": "<64 lowercase hex>",
  "adapter": null,
  "adapter_max_rank": null,
  "vocab_size": 152064,
  "implementation": "vllm:0.25.0",
  "runtime_versions": {
    "python": "3.12.7",
    "python_implementation": "CPython",
    "vllm": "0.25.0",
    "torch": "2.9.1+cu129",
    "transformers": "5.0.0",
    "tokenizers": "0.22.2"
  },
  "runtime_content_sha256": "<64 lowercase hex>",
  "accelerator": {
    "type": "cuda",
    "driver": "nvidia:580.65;cuda-runtime:12.9",
    "devices": [
      {
        "index": 0,
        "name": "example accelerator",
        "architecture": "example-arch",
        "total_memory_bytes": 25757220864
      }
    ]
  }
}
```

Emit the identity:

```bash
python3 scripts/vllm_teacher.py \
  --identity-input=/tmp/teacher-input.json \
  --served-model-id=teacher-large \
  --max-top-k=32 \
  --max-model-len=32768 \
  --manifest-only
```

Add `--model-path` and `--dry-run` to emit a redacted command preview. Dry-run
uses source paths and labels them as such; it does not allocate a snapshot. A
precomputed input manifest is forbidden for a real launch.

## Qualify every intended machine

Run portable launcher tests on each checkout:

```bash
python3 -m unittest scripts/qualification/tests/test_vllm_teacher.py -v
```

Then qualify each intended CUDA, ROCm, or other accelerator host:

1. Record package versions, runtime-content digest, driver, visible devices,
   model snapshot digest, identity, and redacted argv.
2. Send a non-streaming numeric-token `/v1/completions` request with
   `prompt_logprobs`, `max_tokens: 1`, and the exact model ID. Confirm that its
   fingerprint equals the launcher output.
3. Exercise Kiln's remote-teacher smoke at first, interior, and final causal
   positions and at one-row and aggregate response-budget boundaries.
4. Restart unchanged and confirm identity stability. Mutate one bound model,
   tokenizer, package, extension, runtime, accelerator, or adapter input at a
   time and require an identity change or launch failure.
5. Confirm that runtime LoRA endpoints, resolver plugins, unknown options,
   unbound file options, and shadow-code environment variables fail closed.
6. Interrupt the launcher and verify that its process group exits and its
   snapshot and cache disappear. Exercise child-start failure and stale-root
   recovery.
7. Separately measure numerical parity, latency, memory, and throughput at the
   intended batch sizes. Identity qualification alone proves none of them.

Do not copy a runtime manifest to another host and call it qualified. Recreate
the isolated runtime, capture two byte-identical manifests from the exact
launch argv, and retain that host's receipts. Any bound package, native
library, interpreter, model, tokenizer, accelerator, or option change requires
a new manifest.

## Historical hardware evidence

The repository keeps device-specific qualification artifacts as evidence.
They do not configure the general launcher and do not establish portable
defaults.

### CUDA bootstrap fixture

[The RTX 4090 bootstrap launch
document](../../qualification/server-launch/vllm-cuda-rtx4090-serving-bootstrap-v1.json)
is an explicit historical fixture. It selects a copied interpreter, private
snapshot and cache parents, inherited benchmark process-group ownership, BF16,
a 32,768-token context and batch-token bound, 64 sequence slots, 75% device
memory utilization, FCFS, seed zero, prefix caching, and a text-only surface.
It deliberately omits the ROCm-qualified `TRITON_ATTN` choice.

The JSON document is not a runtime manifest and does not pin an installed vLLM
wheel. A CUDA qualification run must install a reviewed compatible version in
its ignored venv, run
`scripts/qualification/capture_vllm_runtime_manifest.py`, require two
byte-identical strict-valid results, and publish the machine's own manifest
without overwrite. The benchmark binds that manifest as `--runtime-artifact`
and verifies the selected accelerator before startup.

### Retained Strix Halo ROCm sequence

These artifacts describe one host and one historical investigation:

- [Runtime
  manifest](../../qualification/runtime/vllm/rocm/strix-halo/vllm-rocm723-qwen35-4b-triton-text-v2.json)
- [Private-cache launch
  document](../../qualification/server-launch/vllm-rocm-strix-halo-triton-text-private-cache-v3.json)
- [First failed v2 startup
  receipt](../../benchmarks/receipts/rocm/strix-halo/20260718t213402-rocm-strix-halo-vllm-triton-text-v2-smoke.json)
- [Successful private-cache v3 smoke
  receipt](../../benchmarks/receipts/rocm/strix-halo/20260718t220209-rocm-strix-halo-vllm-triton-text-private-cache-v3-smoke.json)

The manifest binds the exact vLLM, PyTorch, HIP, Transformers, tokenizers,
installed-runtime, gfx1151, Qwen3.5-4B, and inference-option identities recorded
in the artifact. The v1 launch selected the wrong multimodal surface and failed
before weight allocation. The v2 launch selected the correct text surface but
inherited a root-owned global compilation cache and failed during its first
profiling forward. The v3 launch used a new private cache, cold-compiled,
became ready, served one strict warmup and one one-token measured request, and
drained without leaving a listener, process, snapshot, or cache child.

That v3 smoke proves startup and cleanup only. It is neither performance
evidence nor a claim about another ROCm device.

The subsequent exact-source greedy pair is retained as:

- [Kiln
  receipt](../../benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.kiln.json)
- [vLLM
  receipt](../../benchmarks/receipts/rocm/strix-halo/20260718t223203-rocm-strix-halo-greedy-short-c1-32-sourcepair-v1.vllm.json)

On that one host and workload, vLLM measured 18.04 output tokens/s at
concurrency 8 and 51.59 at concurrency 32; Kiln measured 8.39 and 7.12. This is
evidence for preferring vLLM for that captured high-concurrency ROCm workload,
not a general backend claim.

The pair failed exact-output comparison at every concurrency. The retained
single-request divergence receipts localize the first token mismatch:

- [Kiln
  divergence](../../benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.kiln.json)
- [vLLM
  divergence](../../benchmarks/receipts/rocm/strix-halo/20260718t232632-rocm-strix-halo-greedy-c1-divergence-v1.vllm.json)

Both outputs contain 64 visible tokens and agree through generated token index
2 (`To establish a`). At index 3, Kiln emits token `25045` (` baseline`) and
vLLM emits `15787` (` foundation`). This localizes the disagreement but does
not identify the correct implementation. An independent eager
Transformers/PyTorch next-token oracle is required before attributing the
fault.

## Recovery and residual trust

Normal exits, forwarded signals, spawn failures, and Python exceptions remove
the snapshot and cache. `SIGKILL`, host reset, kernel panic, or power loss can
leave `.building-*`, `ready-*`, or `cache-*` entries. After confirming that no
launcher or vLLM process owns the dedicated parents, remove only the stale
children. Never run broad recursive cleanup against a shared path.

Mode bits are an operational guard, not cryptographic immutability. The
launcher trusts the process UID, root, kernel, and snapshot filesystem. The
same UID or root can alter staged files between final verification and loader
read, or change an interpreter/package after child revalidation and before
import. Use a separately owned read-only mount, fs-verity or equivalent
verification, and a constrained service account when a hostile local operator
is in scope.

The identity hashes the interpreter and the vLLM, torch, Transformers, and
tokenizers import/distribution content. It does not recursively hash every
transitive distribution, system library, driver library, JIT tool, or compiler
outside those trees. Keep the machine image immutable and record the selected
attention/kernel backend in the machine qualification receipt.

The fingerprint does not authenticate the server. Use verified TLS for
off-host teachers, pin the expected identity in Kiln, and keep certificate
verification enabled. Bind plain HTTP to loopback only. An API key authorizes
the client; it does not authenticate the peer.

Once registration succeeds, use the returned `off_policy_manifest` to build an
identity-bound corpus with the [off-policy OPD teacher JSONL
guide](../training/OPD_TEACHER_JSONL.md).
