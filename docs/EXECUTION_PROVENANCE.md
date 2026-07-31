# Execution identity and provenance

Use execution provenance to answer a bounded question: **what software and
numerical environment did this Kiln process declare at startup?**

Kiln constructs one self-verifying record after the model, tokenizer, backend,
and numerical runtime are ready. It retains that record for the life of the
resident model and copies it into health responses, debug output, checkpoints,
training receipts, adapter manifests, and evaluation artifacts.

The record is not a unique process identifier. It contains no PID or startup
timestamp. Two processes with the same captured inputs can have the same
provenance digest, which is useful for comparison but does not prove that they
will produce identical outputs.

Production model backends must have a present, valid record before
`/health` or `/v1/health` reports ready. Mock and explicitly synthetic paths
may omit it.

## Check the running identity

Inspect the bounded health summary first:

```bash
curl -sS http://127.0.0.1:8420/health |
  jq '{
    status,
    execution_identity,
    execution_provenance_check:
      ([.checks[] | select(.name == "execution_provenance_valid")][0])
  }'
```

For a production backend, expect HTTP `200`, `status: "ok"`, a non-null
`execution_identity`, and a passing `execution_provenance_valid` check.
A missing or internally inconsistent record makes readiness fail closed with
HTTP `503` and `status: "degraded"`.

Use the health fields according to the comparison you need:

| Comparison | Field to compare | What a match establishes |
| --- | --- | --- |
| Complete declared envelope | `provenance_sha256` | Every field in the typed record matches. |
| Running program bytes | `executable_sha256` | The exact executable bytes match. |
| Captured driver and host evidence | `numerical_runtime_sha256` | The bounded startup probe inputs match. |
| Compiled kernel contract | `kernel_contract_sha256` | The declared kernel versions and compiled feature list match. |
| Resolved server configuration | `effective_server_config_sha256` | The serialized effective `KilnConfig` matches. |
| Set `KILN_*` environment | `effective_environment_sha256` | The captured, redacted environment map matches. |

Do not use `provenance_sha256` as a substitute for base-weight, adapter,
request, dataset, seed, or output identity. Those identities live in their own
manifests and artifacts.

## Record shape

The current contract is `kiln.execution-provenance.v1`, schema version `1`.
This device-neutral example shows the field structure; angle-bracketed values
stand for the values captured by the running process:

```json
{
  "schema_version": 1,
  "provenance_type": "kiln.execution-provenance.v1",
  "backend": {
    "name": "<selected backend>",
    "device": "<selected device>",
    "numerical_runtime_sha256": "sha256:<64 lowercase hex digits>"
  },
  "build": {
    "package_version": "<Kiln package version>",
    "target": "<operating system>-<architecture>",
    "executable_sha256": "sha256:<64 lowercase hex digits>",
    "git_commit": "<optional source revision>",
    "source_tree_sha256": "sha256:<optional 64 lowercase hex digits>",
    "source_dirty": false
  },
  "model": {
    "model_config_sha256": "sha256:<64 lowercase hex digits>",
    "tokenizer_vocab_sha256": "sha256:<64 lowercase hex digits>",
    "tokenizer_config_sha256": "sha256:<64 lowercase hex digits>",
    "chat_template_sha256": "sha256:<optional 64 lowercase hex digits>",
    "training_chat_template_sha256": "sha256:<optional 64 lowercase hex digits>"
  },
  "precision": {
    "inference_dtype": "<loaded inference dtype>",
    "training_policy": "<resolved backend training policy>"
  },
  "kernels": {
    "contract_type": "kiln.kernel-contract.v1",
    "versions": {
      "<kernel crate>": "<package version>"
    },
    "compiled_features": ["<compiled feature>"],
    "contract_sha256": "sha256:<64 lowercase hex digits>"
  },
  "configuration": {
    "effective_server_config_sha256": "sha256:<64 lowercase hex digits>",
    "effective_environment_sha256": "sha256:<64 lowercase hex digits>"
  },
  "provenance_sha256": "sha256:<64 lowercase hex digits>"
}
```

Optional build and template fields are omitted when Kiln cannot capture them.
The validator rejects unknown fields, unsupported version or type tags,
empty or control-bearing text, malformed hashes, oversized collections,
unsorted or duplicate compiled features, a changed kernel contract, and a
changed top-level digest.

The top-level digest covers every field except itself. The kernel-contract
digest covers the contract type, version map, and ordered compiled-feature
list. These are integrity checks, not signatures: they detect inconsistency
inside the record but do not authenticate who produced it.

## What each identity covers

### Executable and source

`build.executable_sha256` hashes the executable inode that the process is
running. On Linux, Kiln opens `/proc/self/exe`, so replacing the executable's
path after launch does not change the captured bytes.

Git fields are supplemental:

- `KILN_COMMIT` supplies `git_commit` directly and suppresses runtime dirty-tree
  detection.
- Without that override, Kiln tries `git rev-parse HEAD` and
  `git status --porcelain` in `KILN_REPO_ROOT` or the source root compiled into
  the binary.
- `KILN_SOURCE_TREE_HASH`, when set, must already be a
  `sha256:<64 lowercase hex>` digest.

A checkout found at runtime is not proof that it built the executable.
Use `executable_sha256` for exact binary identity and treat source metadata as
navigation back to a candidate source tree.

### Backend and numerical runtime

`backend.name` and `backend.device` identify the backend and device selected by
the runner. Kiln computes `numerical_runtime_sha256` once at startup from the
device, operating system, architecture, CPU and ISA evidence, relevant host
runtime files or commands, loaded numerical libraries, and an
accelerator-specific probe:

| Device family | Accelerator probe |
| --- | --- |
| CUDA | `nvidia-smi` |
| ROCm | `rocminfo` and `rocm-smi` |
| Vulkan | `vulkaninfo --summary` |
| Metal | `system_profiler` |
| CPU | Host runtime evidence without an accelerator command |

Probe commands have a five-second deadline and bounded output capture.
Success, nonzero exit, timeout, spawn failure, read failure, and truncation
produce distinct digest inputs. Raw probe output is not published through the
provenance APIs.

The digest says that the captured evidence matches. It does not certify driver
correctness, enumerate every runtime setting, or guarantee equivalent numerical
results.

### Model and tokenizer

The model block identifies the resident model configuration, tokenizer
vocabulary, tokenizer configuration, inference chat template, and effective
training chat template. The inference and training templates are separate
because training may use a prefix-preserving template that differs from the
serving template.

These hashes do **not** identify the model-weight shard bytes. Compare the
[base-weight manifest](BASE_WEIGHT_PROVENANCE.md) for that.

### Precision and kernels

`inference_dtype` records the model's loaded inference dtype.
`training_policy` records the backend's resolved training-precision policy.
A training receipt can additionally carry `runtime.training_precision`, which
records concrete parameter, optimizer-state, activation, gradient, and
stochastic-rounding behavior observed after trainer setup.

The kernel contract lists Kiln's numerical kernel crates, their package
versions, and compiled backend features. It intentionally describes the
compiled software contract, not a preferred GPU model or a machine-specific
runtime default. The executable digest remains the authoritative identity for
the exact compiled bytes.

### Effective configuration and environment

`effective_server_config_sha256` covers the fully resolved `KilnConfig`,
including defaults and overrides. Health response metadata also exposes this
value as `config_hashes.effective_config_hash`.

`effective_environment_sha256` covers the sorted names and values of `KILN_*`
variables that are set in the process environment. Variables that are absent
are not entries; resolved defaults are already covered by the server-config
digest. If a name contains a sensitive segment such as `KEY`, `TOKEN`,
`SECRET`, `PASSWORD`, `PRIVATE`, `AUTH`, or `CREDENTIAL`, Kiln hashes
`<redacted-present>` instead of the value. Consequently, the digest records
that such a variable was present but does not distinguish two different secret
values.

Neither the environment map nor raw secret values are returned by health or
debug APIs.

## Where the full record appears

The health routes expose only `execution_identity`, the bounded summary shown
above. To inspect the complete record, enable either
`server.debug_model_state=true` or `server.eval_mode=true`, then request:

```bash
curl -sS http://127.0.0.1:8420/v1/debug/model-state |
  jq '.model.execution_provenance'
```

The gated endpoint returns HTTP `500` instead of publishing an internally
inconsistent record. Startup logs contain only the overall, executable,
runtime, and kernel digests plus backend and device names.

Kiln also preserves the complete record in these artifacts:

| Surface | Location or behavior |
| --- | --- |
| Exact SFT, GRPO, and OPD checkpoints | `auxiliary_state.execution_provenance`; exact resume validates the record and requires its overall digest to match the current process before GPU ownership. |
| Successful model-backed training receipts | `runtime.execution_provenance`; concrete trainer precision remains a separate sibling field. |
| Adapter manifests | Top-level `execution_provenance`, copied with training-template and precision evidence. |
| Eval job responses and raw JSON | Full typed record copied at admission. |
| Terminal eval archives | Validated on save and restart load; a tampered archive is rejected. |
| Downloaded eval outcome JSONL | Repeated on every self-contained row. |

Legacy receipts, adapter manifests, eval archives, and synthetic or mock
results may omit this optional field. Legacy exact checkpoints without the
complete record are not exact-resumable.

## What the record cannot prove

Matching provenance establishes equality of the declared and hashed execution
envelope. By itself it does not prove:

- that two runs received the same request, seed, adapter, weights, or data;
- that two runs produced the same output;
- that a driver or kernel is correct;
- that the optional source-tree digest built the running executable;
- that the record came from a trusted producer; or
- that an old environment can still be recreated.

A replay or reproduction claim needs the relevant request lineage,
base-weight and adapter manifests, data and seed identity, generation or
training state, output evidence, and an explicit comparison procedure in
addition to execution provenance.
