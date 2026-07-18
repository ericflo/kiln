# Immutable vLLM teachers

Kiln accepts remote prompt logprobs only when each response carries an
authoritative `TeacherIdentityV1`. Launch vLLM 0.20.1rc0 or newer through
[`scripts/vllm_teacher.py`](../scripts/vllm_teacher.py). The launcher stages a
private content-verified snapshot, fingerprints only that snapshot, binds the
runtime and accelerator into the identity, disables mutable model surfaces,
and supervises vLLM until its entire process group exits.

A stock vLLM fingerprint is not sufficient. Its default fingerprint does not
prove the exact weight bytes, tokenizer ID map, static adapter, runtime
packages, accelerator, or response limits Kiln relies on. A served model name
is also only an alias. Kiln therefore rejects stock `vllm-*` fingerprints for
remote teachers.

## Requirements

- vLLM 0.20.1rc0 or newer, PyTorch, Transformers, and tokenizers installed in the
  Python environment used to run the launcher.
- This minimum is the upstream release candidate in which custom OpenAI
  response fingerprints shipped. Every real launch also probes the installed
  runtime in a fresh child, so satisfying the version floor alone is not
  sufficient.
- A local Hugging Face model directory with safetensors base weights. Hub IDs,
  revisions, `trust_remote_code`, alternate tokenizers, and alternate base
  model inputs are not accepted.
- A materialized model directory containing only intended deployment files.
  Symlinks and special files anywhere below the model or adapter root are
  rejected. In particular, do not pass a symlink-based Hugging Face cache
  checkout directly.
- A dedicated snapshot root owned by the launcher UID with mode `0700`. It may
  not be a filesystem root, an ancestor or descendant of the model, or an
  ancestor or descendant of the adapter. Existing directory permissions are
  validated and never changed by the launcher. Its ancestry must not be
  group/world writable unless the writable ancestor has sticky-directory
  rename protection, as `/tmp` normally does, and every component must be owned
  by the launcher UID or root.
- Enough free bytes and inodes for a full copy plus bounded metadata headroom,
  even when the filesystem supports reflinks.

For a static adapter, provide one local PEFT directory containing
`adapter_config.json` and exactly one of `adapter_model.safetensors` or
`adapter_model.bin`. Prefer safetensors. A pickle-based `.bin` is executable
input and must come from a trusted source.

## Snapshot lifecycle

The fingerprint-to-load race is closed by making the staged snapshot the only
model tree vLLM can load:

1. Validate requested limits, environment variables, and extra vLLM options
   before copying model data.
2. Inventory the complete model and optional adapter trees. Enforce aggregate
   file, directory, byte, path, depth, manifest, free-space, and inode bounds.
3. Create a private `.building-*` directory under an open snapshot-root file
   descriptor. Reflink each regular file where supported, otherwise copy it.
   Hash source and destination for the exact initially observed length and
   reject growth, truncation, replacement, links, or special files.
4. Record every directory and every file `(path, byte length, SHA-256)` in a
   canonical manifest. Make files `0400`, directories `0500`, then re-enumerate
   and re-hash the complete tree.
5. Atomically rename the directory to `ready-*`, synchronize the snapshot root,
   and verify it again. Only then compute the tokenizer, adapter, model, and
   inference identities from the staged paths.
6. Verify the snapshot immediately before spawning vLLM. The vLLM argv points
   only at `ready-*/model` and, when present, `ready-*/adapter`.
7. Start vLLM without a shell. Standalone mode creates and supervises a new
   session. Externally owned mode keeps vLLM in the launcher's already isolated
   process group. Forward termination signals, terminate descendants left after
   the leader exits, then remove the snapshot through the anchored root
   descriptor without following links or crossing filesystems.

Source files may change after a successful copy without affecting the running
teacher. Mutation during copy or verification fails the launch. Reflinks are
copy-on-write snapshots; source writes do not change the staged extents.

The default snapshot root is
`~/.cache/kiln/teacher-snapshots`. Override it with
`--snapshot-root=/dedicated/path` or `KILN_VLLM_SNAPSHOT_ROOT`. A same-filesystem
root gives reflinks the best chance to succeed, but capacity checks always
assume copy fallback.

## Runtime cache lifecycle

Every real launch also gets a fresh, empty vLLM runtime cache. The typed
`--cache-root` option names only a private parent; it defaults to
`~/.cache/kiln/vllm-runtime-caches`. The launcher creates a unique mode-0700
`cache-<pid>-<nonce>` child, verifies the parent and child through open
directory descriptors, and derives `VLLM_CACHE_ROOT` for the supervised vLLM
process. An ambient `VLLM_CACHE_ROOT` is rejected with a pointer to the typed
option. Model, adapter, snapshot, and cache roots must be separate and
non-nested.

The generated cache path is a nonce and is excluded from the inference digest.
That is valid because the child is empty at spawn, is used only as output, is
never reused, and the private-cache policy itself is part of the hashed launcher
runtime. Cache contents therefore cannot carry compiled code, model metadata,
or autotuning state from another user, vLLM build, model, profile, or benchmark
arm. A serving profile still warms the running server before measurement;
server startup and compilation are not silently borrowed from a prior profile.

After vLLM and its descendants exit, the launcher recursively removes the cache
through its anchored parent descriptor without following links or crossing a
filesystem boundary. Normal child failure follows the same cleanup path. An
unrecoverable kill can leave an ignored `cache-*` directory, but no later launch
will consume it; inspect it only after confirming that no owning process remains.
Cold compilation can use substantial temporary disk and startup time, so place
`--cache-root` on a filesystem with appropriate headroom. Those costs are
deliberate qualification isolation, not measured request throughput.

## Base teacher

Additional vLLM arguments follow `--` and use one `--key=value` token each,
except for the launcher's small set of vetted valueless switches.

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/Qwen3.5-4B \
  --snapshot-root=/var/tmp/kiln-teacher-snapshots \
  --cache-root=/var/tmp/kiln-vllm-runtime-caches \
  --served-model-id=qwen35-4b-teacher \
  --max-top-k=20 \
  --max-model-len=32768 \
  --max-prompt-logprob-candidates=500000 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --dtype=bfloat16 \
  --attention-backend=TRITON_ATTN \
  --language-model-only \
  --api-key=replace-with-a-random-secret
```

The generated command uses the same Python interpreter, a fixed `/` working
directory, no shell, and a staged path:

```text
python3 -m vllm.entrypoints.cli.main serve \
  /var/tmp/kiln-teacher-snapshots/ready-.../model \
  --served-model-name=qwen35-4b-teacher \
  --max-model-len=32768 \
  --max-logprobs=20 \
  --logprobs-mode=raw_logprobs \
  --generation-config=vllm \
  --load-format=safetensors \
  --fingerprint-mode=custom \
  --fingerprint-value=kiln-teacher-v1.<base64url-json>.<sha256>
```

The base loader is always forced to safetensors. Caller-supplied `--load-format`
is rejected. Alternate `.bin` files cannot win loader selection, and every
extra file is still included in the full snapshot digest.

The launcher prints the fingerprint and snapshot path immediately before
spawn. Kiln must receive that exact fingerprint in every scoring response.

## Process-group ownership

Standalone launches use the default `--process-group-mode=detached`: the
launcher creates a new vLLM session, forwards `SIGINT`, `SIGTERM`, `SIGHUP`, and
`SIGQUIT`, drains descendants, and removes the immutable snapshot.

An owned local serving benchmark must instead pass
`--process-group-mode=inherited`. This mode is accepted only on Linux when the
launcher PID is already the leader of an isolated process group. The benchmark
driver creates that group before any snapshot or model work, so the launcher,
vLLM, and ordinary worker descendants receive thermal `SIGSTOP`/`SIGCONT` and
shutdown signals as one unit. The launcher drains every peer without signaling
itself and still owns snapshot cleanup. A direct signal delivered only to the
launcher is forwarded to the child; the external supervisor should signal the
complete group.

Do not use detached mode behind a process-group thermal guard. The guarded
launcher would stop while the detached inference child continued running. The
serving benchmark independently rejects that shape because readiness must come
from a listener held by the launched process group.

## Static adapter teacher

Add one adapter path. The API model ID is also the only static adapter name:

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/Qwen3.5-4B \
  --adapter-path=/adapters/math-v7 \
  --snapshot-root=/var/tmp/kiln-teacher-snapshots \
  --served-model-id=qwen35-4b-math-v7 \
  --max-top-k=20 \
  --max-model-len=32768 \
  --max-prompt-logprob-candidates=500000 \
  -- \
  --host=127.0.0.1 \
  --port=8000 \
  --dtype=bfloat16
```

The launcher adds exactly one static module using its staged adapter path:

```text
--enable-lora --max-loras=1 --max-cpu-loras=1 \
--max-lora-rank=<config-derived-cap> \
--lora-modules={"name":"qwen35-4b-math-v7","path":".../ready-.../adapter","base_model_name":"kiln-base-..."}
```

Runtime adapter load/unload, resolver plugins, prompt adapters, multiple LoRAs,
and caller-supplied adapter options are rejected. Configure Kiln only with the
adapter model ID, never vLLM's internal base alias.

vLLM necessarily lists and accepts its internal base alias when serving a
LoRA. Its custom fingerprint is process-wide, so the base response carries the
same fingerprint even though it does not carry adapter logits. The fingerprint
alone is therefore insufficient for an adapter request. Kiln requires the
configured request model and response `model` to equal the identity's
`served_model_id`, which rejects this bypass. Any non-Kiln client must enforce
the same check. If the endpoint itself must expose only the adapter name, put it
behind an authenticated proxy that rejects the internal base ID.

## Response budget

`max_prompt_logprob_candidates` is an aggregate response-allocation ceiling,
not a per-position top-K. One position can contain up to
`min(max_top_k + 1, vocab_size)` candidates because the sampled token may be in
addition to the requested top-K.

The configured ceiling must fit at least one maximum-width row and may not
exceed either `1,000,000` candidates or
`max_model_len * min(max_top_k + 1, vocab_size)`. If omitted, the launcher uses
the smaller of `1,000,000` and that theoretical response maximum. The value is
part of the canonical teacher identity and Kiln rejects oversized responses
before allocating their full candidate payload. The explicit one-million cap
also prevents an identity from advertising multi-gigabyte responses that the
client's bounded HTTP transport could never accept.

## Identity contract

The compact JSON field order is normative:

```text
schema, protocol, served_model_id, base_model_sha256,
tokenizer_vocab_sha256, tokenizer_config_sha256, adapter, vocab_size,
max_top_k, max_model_len, max_prompt_logprob_candidates, logprobs_mode,
implementation, inference_config_sha256
```

The constants are:

```text
schema:         kiln.teacher-identity.v1
protocol:       vllm.prompt-logprobs.numeric-token-ids.causal.v1
logprobs_mode:  raw_logprobs
fingerprint:    kiln-teacher-v1.<base64url without padding>.<sha256 of JSON>
```

All SHA fields are exactly 64 lowercase hexadecimal characters without a
`sha256:` prefix.

`base_model_sha256` matches Kiln's Rust loader. For each safetensors shard,
hash the raw bytes and record `(digest, byte_length)`. Sort by digest and then
length. Hash the domain `kiln.base-model-content.v1\0`, a little-endian `u64`
record count, then each little-endian `u64` length and raw 32-byte digest.

`tokenizer_vocab_sha256` hashes the complete Transformers `get_vocab()` map.
Pairs are sorted by `(u32 ID, raw UTF-8 token bytes)` and encoded under the
`kiln.tokenizer-vocab.v1\0` domain. The map pair count, backend tokenizer
`get_vocab_size(with_added_tokens=true)` must agree. The model's embedding and
logit width comes from `config.json.vocab_size` or
`config.json.text_config.vocab_size`; both must agree when present. That width
may exceed the tokenizer entry count because some models reserve padded rows,
but the entry count and every assigned token ID must fit within it. The
canonical teacher identity records the model width, not the tokenizer entry
count.

`tokenizer_config_sha256` hashes the exact UTF-8 fast-tokenizer backend JSON
returned by `backend_tokenizer.to_str()`, corresponding to Rust
`Tokenizer::to_string(false)`.

For a static adapter, `weights_sha256` is a domain-separated digest over the
selected filename, byte length, and raw weight digest. `config_sha256` is the
raw `adapter_config.json` digest. Adapter rank and every `rank_pattern` value
select the smallest supported vLLM rank cap, which is bound into the inference
digest.

`inference_config_sha256` binds:

- the canonical full-tree snapshot manifest digest and exact `config.json`;
- top-K, context, aggregate response budget, raw-logprob mode, safetensors
  format, adapter mode, and every vetted non-transport vLLM option;
- exact Python implementation/version, the resolved interpreter executable,
  and the vLLM, torch, Transformers, and tokenizers package versions and
  installed content. The content digest covers each actual import tree,
  distribution metadata and recorded files, Python/native-extension files,
  and editable-install source. A fresh child re-hashes this contract under the
  exact launch environment immediately before vLLM is spawned;
- accelerator type, driver/runtime identity, visible device names,
  architectures, and total memory;
- inference-affecting vLLM, CUDA, ROCm/HIP, RCCL/NCCL, Torch, Triton, HF, and
  native-library environment values; and
- deterministic policy inputs including `--seed`, eager mode,
  `PYTHONHASHSEED`, TF32 overrides, cuBLAS workspace policy, and cuDNN policy.

This records determinism inputs; it does not claim that a given vLLM/kernel
combination is numerically deterministic.

Transport settings such as host, port, API key, TLS, and CORS are excluded
because they do not alter logits. The API key is never embedded in the
identity and is redacted from dry-run output.

## Fail-closed options and environment

The launcher owns all model, tokenizer, weight-format, generation-config,
logprob, fingerprint, middleware, LoRA, and adapter arguments. Transport
options and a versioned allowlist of reviewed scalar inference options are
accepted. Unknown vLLM options fail closed. Options containing file, path,
directory, or template inputs are rejected because a path string does not bind
the referenced bytes. Add and test a new option in the launcher before relying
on it.

`--attention-backend` is additionally constrained by value because vLLM can
interpret its value as an importable class path. The only currently reviewed
explicit value is `TRITON_ATTN`; omit the option to retain vLLM's automatic
selection. Other built-in names and custom class paths fail before any model
snapshot or server process is created. Qualifying another backend requires
adding its exact name to the closed launcher set, testing the rejection
boundary, and capturing backend-specific correctness and performance evidence.

`--language-model-only` is a reviewed, identity-bound valueless switch for
hybrid text/multimodal architectures. It tells vLLM to set every multimodal
input limit to zero. Use it only when the qualified model surface is text-only;
it does not synthesize missing image/audio processors and it deliberately makes
multimodal requests unsupported.

`PYTHONPATH`, `PYTHONHOME`, Python safe/user-site overrides, `LD_PRELOAD`, and
`LD_LIBRARY_PATH` are rejected for real launches so the child cannot resolve
shadow Python or native code under an unchanged package-version identity.
Environment names that identify files, paths, plugin modules, executables,
homes, roots, or cache directories are also rejected across the vLLM, CUDA,
ROCm, Torch, Triton, and HF namespaces. This includes vLLM shared-library and
logging-config paths plus `HIPBLASLT_TUNING_OVERRIDE_FILE`. Device-visibility
variables including `ROCR_VISIBLE_DEVICES` are allowed but identity-bound.
`VLLM_CACHE_ROOT` is the one derived cache environment value: callers configure
`--cache-root`, and the launcher supplies a fresh child path rather than
accepting process-global state.

Runtime hashing is intentionally bounded and fail-closed: at most 250,000
files, 100,000 directories, 64 GiB of logical file content, 128 directory
levels, 4,096 bytes per canonical label, and 64 MiB of aggregate label data.
Missing files named by distribution metadata, ambiguous namespace roots,
unreadable files, special files, path changes, and content mutation during a
hash abort launch.
The launcher forces `PYTHONDONTWRITEBYTECODE=1` in the child so its own import
does not mutate an identity-bound package tree by creating bytecode caches.

`VLLM_ALLOW_RUNTIME_LORA_UPDATING` is forced to `0`. `VLLM_PLUGINS`, LoRA
resolver configuration, and skipped model-name validation fail before spawn.

## Inspect without vLLM

Tests and tooling can provide a strict input manifest without importing vLLM,
Transformers, torch, or touching model weights. This mode can never launch a
server.

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
  "vocab_size": 248320,
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
        "name": "NVIDIA GeForce RTX 4090",
        "architecture": "sm_89",
        "total_memory_bytes": 25757220864
      }
    ]
  }
}
```

Emit only the identity:

```bash
python3 scripts/vllm_teacher.py \
  --identity-input=/tmp/teacher-input.json \
  --served-model-id=qwen35-4b-teacher \
  --max-top-k=20 \
  --max-model-len=32768 \
  --manifest-only
```

Add `--model-path` and `--dry-run` to emit a redacted command preview. Dry-run
uses source paths only and clearly labels that fact; it does not allocate a
snapshot. A precomputed manifest is forbidden for a real launch.

## Crash recovery and residual trust

Normal exits, forwarded signals, spawn failures, and Python exceptions remove
the snapshot. `SIGKILL`, host reset, kernel panic, or power loss cannot run
cleanup and may leave `.building-*` or `ready-*` directories. After confirming
that no launcher or vLLM process references that dedicated root, remove stale
entries manually. Never run generic recursive cleanup against a shared path.

Mode bits are an operational guard, not cryptographic immutability. The
launcher assumes the process UID, root, kernel, and snapshot filesystem are
trusted. The same UID or root can chmod or replace staged files between the
last verification and a loader read, and can replace an interpreter or package
file after the fresh-child runtime revalidation but before vLLM imports it.
Protect against a hostile local operator with a separately owned read-only
mount, filesystem verification such as fs-verity, and a constrained service
account. Hardware faults and malicious kernel/filesystem behavior are outside
this contract.

The identity hashes the interpreter plus the vLLM, torch, Transformers, and
tokenizers import/distribution content. It does not recursively hash every
transitive distribution, system shared library, driver library, or JIT tool
binary outside those trees. Auto-selected Triton, flash-attention, FlashInfer,
xFormers, rocBLAS, hipBLASLt, compilers, and system-library bytes remain part of
the trusted machine image unless their files reside in one of the four bound
package trees. Keep that image immutable, record the detected attention/kernel
backend in the machine qualification receipt, and expect an identity change
only when one of the explicitly bound inputs changes.

The fingerprint also does not authenticate a network peer. Use verified TLS,
pin the expected teacher identity in Kiln, and keep certificate verification
enabled. Bind plain HTTP to loopback only. An API key authorizes a client but
does not authenticate the server.

## Register with Kiln

The generated [Artifact Lifecycle API Schema](../contracts/kiln-artifacts-v1.schema.json)
is authoritative for `RegisterTeacherRequest`, `TeacherEntry`,
`TeachersListResponse`, deletion responses, capabilities, and the complete
teacher identity object. The rules below explain registration and trust
semantics rather than defining a second copy of those fields.

Credential-free registration is limited to an exact loopback URL. For an
off-host teacher, configure an exact canonical HTTPS origin and a server-owned
secret environment variable before Kiln starts:

```toml
# kiln.toml
[teachers.credentials.primary-vllm]
origin = "https://teacher.example.com:8443"
api_key_env = "KILN_VLLM_API_KEY"
```

```bash
export KILN_VLLM_API_KEY='replace-with-the-teacher-secret'
kiln serve --config kiln.toml

curl -X POST http://localhost:8420/v1/teachers \
  -H 'content-type: application/json' \
  -d '{
    "alias": "qwen35@vllm",
    "kind": "remote",
    "provider": "vllm",
    "model_id": "qwen35-4b-teacher",
    "url": "https://teacher.example.com:8443",
    "credential_id": "primary-vllm"
  }'
```

The request cannot set identity, tokenizer hash, vocabulary size, top-K,
full-vocabulary support, or a secret environment-variable name. Kiln sends two
operational probes, verifies the complete numeric vocabulary against its
student, persists the returned canonical identity before publishing the alias,
and returns `status`, `usable`, `identity_revision`, bounds, and an exact
`off_policy_manifest`. Aliases are immutable; delete and re-register to change
a deployment. Every queued job pins the complete spec and repeats the two
probes before GPU ownership or a cache lookup. Every scoring response must
continue carrying the same fingerprint.

Registries created by the older API may contain `api_key_env`. On first load,
Kiln removes that field without trusting it and atomically rewrites the file.
The migrated remote alias remains unusable until it is deleted and registered
again with `credential_id` and a fresh identity probe.

## Per-machine qualification

GPU CI is not required for this contract. Run portable tests on every checkout:

```bash
python3 -m unittest scripts/qualification/tests/test_vllm_teacher.py -v
```

Then qualify each ROCm, CUDA, and other intended machine locally:

1. Record package versions, runtime-content digest, driver, visible devices,
   model snapshot digest, identity, and redacted argv.
2. Launch a base teacher and send a non-streaming `/v1/completions` request
   with numeric prompt IDs, `prompt_logprobs`, `max_tokens: 1`, and the exact
   model ID. Confirm the response fingerprint exactly matches launcher output.
3. Run Kiln's remote-teacher smoke over first, interior, and final causal
   positions and at both one-row and aggregate candidate-budget boundaries.
4. Restart unchanged and confirm identity stability. Change a copied shard,
   auxiliary file, tokenizer backend, editable package source file, native
   extension, runtime version, accelerator selection, and adapter input one at
   a time; confirm the identity changes or launch fails. Also mutate a bound
   package after identity construction and confirm the final child
   revalidation refuses to spawn vLLM.
5. Confirm runtime LoRA endpoints, resolver plugins, unknown vLLM options,
   unbound file options, and shadow-code environment variables fail closed.
6. Interrupt the launcher and confirm the process group exits and its snapshot
   disappears. Exercise a child-start failure and stale-snapshot recovery.
7. Measure numerical parity, latency pauses, VRAM behavior, and throughput at
   intended batch sizes. Identity qualification proves provenance and limits;
   it does not prove numerical parity or performance.

### Retained Strix Halo ROCm baseline

The repository retains the current machine-specific inputs for the local
Strix Halo comparison:

- `qualification/runtime/vllm/rocm/strix-halo/vllm-rocm723-qwen35-4b-triton-text-v2.json`
  is the launcher-produced runtime manifest;
- `qualification/server-launch/vllm-rocm-strix-halo-triton-text-v2.json` is the
  atomic owned-launch document; and
- `qualification/host-policies/strix-halo-serving-benchmark-fast-guard-v1.json`
  is the thermal policy used by the first guarded startup and comparison runs.

The manifest binds vLLM
`0.23.1rc1.dev1261+gc71a583aa.rocm723`, PyTorch
`2.11.0+gitd0c8b1f`, HIP `7.2.53211`, Transformers `5.14.1`, tokenizers
`0.22.2`, the complete installed runtime content, the gfx1151 accelerator,
the local Qwen3.5-4B model/tokenizer content, and every inference-affecting
argument. The launch selects BF16 `TRITON_ATTN`, a 32,768-token context and
batch-token limit, 64 sequence slots, 40 percent device-memory utilization,
fixed seed zero, FCFS scheduling, prefix caching, and 16-token cache blocks.
It also selects `--language-model-only`, matching Kiln's qualified text surface
and setting every vLLM multimodal input limit to zero. The retained v1 artifacts
omit that switch and are rejected counterexample inputs: vLLM resolved the
hybrid architecture, requested an image processor absent from the text-serving
checkpoint, and exited before weight allocation.
This is a qualification input, not a portable claim that another ROCm host has
the same runtime and not evidence that vLLM has passed startup or performance.

The ignored local environment is rooted at
`.qualification/vllm-rocm-venv`. Its `bin/python-kiln` must be a regular copied
interpreter inside the venv, not its ordinary symlink: the serving driver
canonicalizes `command[0]`, and resolving a venv symlink to the base interpreter
would bypass venv package discovery. The snapshot and log roots are private
mode-0700 directories beneath `.qualification`. On this host the wheel does
not require the shell's `ROCM_PATH`, and the local model does not require an HF
credential. Manifest generation and the benchmark driver therefore run with
those ambient inputs removed and bytecode writes disabled:

```bash
env -u ROCM_PATH -u HF_TOKEN PYTHONDONTWRITEBYTECODE=1 \
  python3 scripts/bench-concurrent-batch.py ...
```

Before the v2 startup, the root-owned model-info cache left by another runtime
was quarantined intact and replaced by an empty user-owned mode-0700
`~/.cache/vllm/modelinfos`. vLLM keys those records by the model implementation
module hash, but an unreadable or externally seeded cache is not acceptable
qualification state. Cache initialization and any resulting files must be
reported with the startup evidence; do not silently repair permissions during
a measured lifecycle.

The v2 startup then loaded both weight shards and 7.99 GiB of model state before
failing in its first profiling forward: a separate 7.0 GiB root-owned global
Torch compile cache prevented creation of the requested AOT cache key. The
driver-v6 failure receipt is retained under
`benchmarks/receipts/rocm/strix-halo/20260718t213402-rocm-strix-halo-vllm-triton-text-v2-smoke.json`.
That foreign tree was moved intact to
`~/.cache/vllm/torch_compile_cache.root-owned-20260718`; it was not deleted or
permission-mutated. The typed private-cache lifecycle above removes global
compile and metadata caches from subsequent launch inputs. A later v3 manifest
and guarded receipt, not the repaired host-global path, are required to accept
startup.

Do not copy this manifest to another machine and call it qualified. Recreate
the isolated runtime there, emit a new manifest through the exact launch argv,
require two byte-identical captures, and retain that platform's own manifest
and receipts. Any package, native-library, interpreter, model, tokenizer,
accelerator, or inference-option change must produce a new manifest and
invalidate comparison against the old fingerprint.
