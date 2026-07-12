# Execution Provenance

Kiln constructs one immutable execution identity after the model, tokenizer,
backend, and numerical runtime have initialized. The record binds the process
that will perform inference, training, and evaluation to its effective model,
build, accelerator, precision, kernel, and configuration envelope. It is
created once at production startup and retained with the resident weights;
request handling never re-probes the driver or re-hashes the executable.

Real backends require a present, internally valid record before `/health` or
`/v1/health` reports ready. Mock and explicitly synthetic paths may omit it.

## Schema

The current type is `kiln.execution-provenance.v1` with schema version `1`:

```json
{
  "schema_version": 1,
  "provenance_type": "kiln.execution-provenance.v1",
  "backend": {
    "name": "rocm",
    "device": "rocm:0",
    "numerical_runtime_sha256": "sha256:<64 lowercase hex digits>"
  },
  "build": {
    "package_version": "0.4.1",
    "target": "linux-x86_64",
    "executable_sha256": "sha256:<64 lowercase hex digits>",
    "git_commit": "<optional source revision>",
    "source_tree_sha256": "sha256:<optional 64 lowercase hex digits>",
    "source_dirty": false
  },
  "model": {
    "model_config_sha256": "sha256:<64 lowercase hex digits>",
    "tokenizer_vocab_sha256": "sha256:<64 lowercase hex digits>",
    "tokenizer_config_sha256": "sha256:<64 lowercase hex digits>",
    "chat_template_sha256": "sha256:<optional 64 lowercase hex digits>"
  },
  "precision": {
    "inference_dtype": "bf16",
    "training_policy": "rocm_native_bf16"
  },
  "kernels": {
    "contract_type": "kiln.kernel-contract.v1",
    "versions": {
      "kiln-model": "0.4.1"
    },
    "compiled_features": ["rocm", "rocm-archs=gfx1151"],
    "contract_sha256": "sha256:<64 lowercase hex digits>"
  },
  "configuration": {
    "effective_server_config_sha256": "sha256:<64 lowercase hex digits>",
    "effective_environment_sha256": "sha256:<64 lowercase hex digits>"
  },
  "provenance_sha256": "sha256:<64 lowercase hex digits>"
}
```

The validator rejects unknown fields, unsupported versions or type tags,
unbounded or control-bearing text, malformed hashes, unsorted or duplicate
compiled features, kernel-contract drift, and any top-level digest mismatch.
The top-level digest covers every field except itself. The kernel-contract
digest covers its type, version map, and ordered compiled-feature list.

## Evidence sources

- `executable_sha256` hashes the exact running executable inode. On Linux this
  is `/proc/self/exe`, so replacing the path after launch cannot change the
  identity.
- `numerical_runtime_sha256` binds the selected device, OS and architecture,
  CPU identity and ISA, kernel and libc evidence, loaded numerical libraries,
  and a bounded accelerator probe. CUDA uses `nvidia-smi`; ROCm uses
  `rocminfo` and `rocm-smi`; Vulkan uses `vulkaninfo --summary`; Metal uses
  `system_profiler`. Missing, failed, truncated, and timed-out probes remain
  distinct inputs. Probe output is never published directly.
- Model and tokenizer fields are computed from the resident model
  configuration and tokenizer/template objects used by the runner.
- Precision records the loaded inference dtype and the backend's resolved
  training-precision policy.
- Kernel identity records the compiled backend features and the versions of
  Kiln's numerical kernel crates. The executable digest remains the
  authoritative identity for their exact compiled bytes.
- The configuration digest covers the fully resolved `KilnConfig`. The
  environment digest covers sorted `KILN_*` names and effective values.
  Secret-bearing name segments such as `KEY`, `TOKEN`, `SECRET`, `PASSWORD`,
  `AUTH`, or `CREDENTIAL` contribute only a `<redacted-present>` marker. Raw
  environment values are not returned by health or debug APIs.

Git metadata is supplemental. `KILN_COMMIT` overrides automatic `git
rev-parse HEAD` detection; otherwise Kiln also records whether that checkout
was dirty when it can inspect the source tree. A build system may supply a
strict prefixed digest through `KILN_SOURCE_TREE_HASH`. These fields help map a
binary back to source, but `executable_sha256` is authoritative because a
checkout observed at runtime need not be the checkout that produced the
binary.

## Runtime surfaces

- `/health` and `/v1/health` expose `execution_identity`, a bounded summary
  containing the overall, executable, runtime, kernel, server-config, and
  environment digests plus backend, device, and precision names.
- Health includes an `execution_provenance_valid` check. A real backend with a
  missing or invalid record returns `503 degraded`; mock mode may omit it.
- Gated `/v1/debug/model-state` exposes the complete typed record. It is
  available only under the existing eval/debug endpoint policy and returns
  `500` rather than publishing an internally inconsistent record.
- Startup logs only the bounded overall, executable, runtime, and kernel
  digests together with backend and device names.

The record proves integrity and equivalence of the declared execution
envelope. It does not by itself prove that two executions produced the same
outputs, that a driver is correct, or that a source digest built the running
binary. Replay claims require an explicit replay procedure and matching input,
seed, model/adapter, data, and generation or training state in addition to this
record.
