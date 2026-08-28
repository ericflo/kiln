# Local Hardware Qualification

Kiln qualification answers a bounded question: **did this declared workload
pass on this backend, device, source tree, model, and configuration?**

It does not turn the measured machine into a product requirement. Device names,
UUIDs, memory sizes, architecture labels, and observed performance belong in
evidence. Portable runtime policy must continue to discover and use the
capabilities exposed by each supported backend.

## Choose the Right Workflow

| Goal | Start here | Result |
| --- | --- | --- |
| Check whether a backend is visible | `environment-v1.json` | Environment and device evidence |
| Check bounded tensor and matmul correctness | `correctness-core-v1.json` | Backend correctness receipt |
| Investigate a known lifecycle or serving path | The matching named workload | Targeted correctness receipt |
| Establish capacity, throughput, latency, or endurance | A serving or endurance workload | Workload-specific measurement receipt |
| Compare two equivalent runs | `compare_receipts.py` with the exact workload manifest | Policy-bound comparison report |

Qualification is not a single “supported” bit. A core-correctness pass does not
establish serving capacity, and a short serving pass does not establish
endurance.

## Before You Run

You need:

- a clean Git worktree;
- the workload committed at `HEAD`;
- Rust and the locked repository dependencies;
- a conformant loader, driver, and device for the selected backend;
- any Python environment named by the workload’s oracle or reference-runtime
  manifest; and
- enough uninterrupted time for the workload’s declared deadlines.

Backend prerequisites are capability-based:

- **CUDA:** an NVIDIA driver, CUDA toolkit, and visible CUDA device;
- **ROCm:** ROCm userspace and a visible supported AMD device;
- **Vulkan:** a conformant Vulkan loader and driver exposing the operations the
  selected workload requires; and
- **Metal:** Apple platform and Metal tooling.

The generic runner does not require a named laptop, GPU model, or temperature
sensor. A workload may require its selected backend and specific capabilities.
A device-specific regression fixture must say so explicitly and remains
evidence for that fixture—not a portable default.

## Run a Workload

First inspect and validate the committed contract:

```bash
python3 scripts/qualification/workload.py \
  qualification/workloads/correctness-core-v1.json

jq '.variants[] | {id, backend, cases: [.cases[].id]}' \
  qualification/workloads/correctness-core-v1.json
```

Then run one exact variant. This example exercises the Vulkan core-correctness
contract without naming a device model:

```bash
python3 scripts/qualification/run.py \
  qualification/workloads/correctness-core-v1.json \
  --variant vulkan \
  --host-id my-vulkan-host \
  --output .qualification/receipts/vulkan-core-local.json
```

Use `--var NAME=VALUE` for declared workload variables. Workloads that require
a model also use `--model <local-model-directory>` and may use
`--model-id <label>`.

```bash
python3 scripts/qualification/run.py --help
```

The runner refuses to:

- execute an uncommitted or modified workload;
- start from a dirty worktree;
- invoke case commands through a shell;
- reuse an existing run directory; or
- overwrite an existing receipt.

Raw case output and normalized evidence are written below
`.qualification/runs/`. Put exploratory receipts below
`.qualification/receipts/`; only intentionally reviewed evidence belongs in
the checked-in receipt tree.

## What a Workload Contract Closes

A workload under `qualification/workloads/` declares:

- purpose, deterministic seed, variables, and repetition count;
- backend family and whether a device is required;
- variant configuration and skip policy;
- ordered argv, working directory, environment, and finite deadlines;
- output assertions and structured-result ownership;
- declared metrics and their definitions; and
- the allowed receipt pairs and comparison rules.

JSON Schema checks the wire shape. `scripts/qualification/workload.py` adds
semantic checks for placeholder delivery, canonical ordering, bounded
execution, metric ownership, variant compatibility, and cross-backend
comparison policy.

See the [qualification workload contract](../../qualification/schema/workload-v1.schema.json)
for the field-by-field reference.

## What the Runner Owns

Before execution, the runner binds:

- the Git commit and source-tree digest;
- the exact committed workload bytes;
- resolved variables and selected variant;
- the filtered case environment;
- backend and platform probe results; and
- model, tokenizer, and weight-file identities when a model is present.

During execution, cases run serially in declared order with bounded output,
structured-result, process, and time budgets. Command-produced case evidence
must match the selected case ID, declared metrics, and effective
configuration. The runner replaces command-reported duration with its own
monotonic measurement and can downgrade command-reported status.

After execution, the runner:

- aggregates repetitions;
- applies required-result and infrastructure failures;
- fingerprints the model again and rejects identity drift;
- checks that the source tree did not change;
- records cleanup and containment evidence; and
- writes a new receipt without replacing an existing one.

The [case-result contract](../../qualification/schema/case-result-v1.schema.json)
documents the command-to-runner boundary. A command case-result file is not the
qualification verdict.

## Platform and Process Boundaries

Native Linux uses the containment mechanism selected by the runner. WSL2 may
use a systemd user scope through `scripts/qualification/wsl_scope_exec.py`.
The default WSL2 scope has no fixed CPU quota or machine-sized memory ceiling;
it does have a bounded PID count, private result snapshot, wall-clock
deadlines, and explicit cleanup.

On macOS, workloads that forbid network access use a fail-closed
`sandbox-exec` profile. The profile permits loopback and denies external
inbound and outbound traffic. Each case receives a new session and process
group, bounded descendant settlement, and termination cleanup.

Platform probes record unavailable capabilities instead of inventing values.
Temperature observations, when available, are read-only evidence. They do not
pace execution, choose a route, define a device allowlist, or set product
memory policy.

Bounded build wrappers provide offline dependency use, filtered environments,
finite runtime, and cleanup:

```bash
scripts/cargo-bounded.sh build --release --locked --offline
scripts/qualification/cargo-test-bounded.sh test --locked --offline
```

Normal builds use the target selected by the existing toolchain and backend.
The wrapper does not pin one ROCm architecture or a machine-sized minimum
available-memory value unless an explicit regression fixture supplies that
bound.

## Serving Qualification

Serving workloads bind the source, binary, configuration, model, tokenizer,
runtime, prompt, generation settings, and concurrency envelope. Their drivers
own one server process group, bound readiness and request time, reject
listener or process residue, and retain strict result and receipt documents.

Use the [Serving Benchmark Protocol](../serving/SERVING_BENCHMARK_PROTOCOL.md) for the
measurement and publication contract. A host-specific serving receipt may
record device IDs, memory capacity, and architecture names; none of those
values becomes a portable server default.

## Validate a Receipt

Basic validation checks the closed record, cross-field rules, finite values,
timestamp relationships, verdict consistency, and platform evidence:

```bash
python3 scripts/qualification/receipt.py \
  .qualification/receipts/vulkan-core-local.json
```

Use stricter checks when the required evidence is available:

```bash
python3 scripts/qualification/receipt.py \
  --require-current-source \
  --require-local-artifacts \
  --require-known-commit \
  .qualification/receipts/vulkan-core-local.json
```

Those flags mean different things:

| Flag | Additional proof | Important limit |
| --- | --- | --- |
| `--require-current-source` | Recomputes the source-tree digest in this checkout | Historical receipts normally fail after source changes |
| `--require-local-artifacts` | Recomputes size and SHA-256 for `local_ignored` files under `.qualification` | External artifacts are not downloaded or checked |
| `--require-known-commit` | Confirms the recorded commit exists in the local Git object database | It does not authenticate who produced the commit |

The receipt itself has no signature or self-digest. Artifact hashes detect
content changes only when a consumer possesses the referenced bytes and
actually recomputes them. Establish provenance and custody separately before
treating a receipt as trusted evidence.

See the [qualification receipt contract](../../qualification/schema/receipt-v1.schema.json)
for every field and trust boundary.

## Compare Equivalent Receipts

Use the exact workload manifest named by both receipts:

```bash
python3 scripts/qualification/compare_receipts.py \
  path/to/baseline.json \
  path/to/candidate.json \
  --workload-manifest qualification/workloads/<workload>.json \
  --json
```

The comparison checks source and workload compatibility, selected variants,
allowed effective-configuration differences, metric definitions, required
measurements, operators, and tolerances.

Do not compare:

- different prompts, output lengths, sampling settings, or repetition counts;
- different model or tokenizer identities;
- results outside a declared variant or backend pair;
- throughput from failed rows;
- native Linux and WSL2 as though the platform boundary were identical; or
- two receipts merely because their metric names happen to match.

The workload’s comparison policy—not a case command’s advisory tolerance—is
the authority.

## Read the Verdict

A passing receipt establishes only the declared workload on the recorded
source, backend, device, model, configuration, and environment. It does not
establish:

- correctness outside the executed cases;
- performance on another device;
- high-concurrency behavior from a single-request run;
- endurance beyond the measured interval;
- native Linux behavior from a WSL2 run;
- capacity for a different prompt or output shape; or
- product-wide support for every operation the backend can expose.

A failed receipt is still useful counterevidence when it localizes a
correctness, capacity, lifecycle, or cleanup failure. Keep the failed verdict
visible, and never average failed rows into a performance claim.

## Retained Evidence Examples

The repository retains concrete machine-specific receipts because evidence
must identify the machine that produced it. These July 28–29, 2026 examples
are historical snapshots, not current defaults or a device allowlist.

| Evidence area | Retained examples | Scope |
| --- | --- | --- |
| Platform boundary | [CUDA/WSL2 environment](../../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t050414137676z-cuda-rtx4090-laptop-wsl2-local-environment-v1-df3e8fee15-v1.json), [Metal/macOS environment](../../qualification/receipts/metal/macbook-air-m1/20260728t223911446266z-metal-macbook-air-m1-local-environment-v1-99474c38c1-v1.json) | Probe and containment evidence only |
| Core correctness | [CUDA](../../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t051305956568z-cuda-rtx4090-laptop-wsl2-cuda-metal-core-correctn-9f21d75c94-v1.json), [Metal](../../qualification/receipts/metal/macbook-air-m1/20260728t225405419496z-metal-macbook-air-m1-cuda-metal-core-correctn-d119e83143-v1.json), [ROCm](../../qualification/receipts/rocm/strix-halo/20260728t222535119053z-rocm-strix-halo-core-correctness-v1-9faecc7321-v1.json), [Vulkan](../../qualification/receipts/vulkan/strix-halo/20260728t222644757744z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json) | Declared tensor, matmul, graph, or training subset only |
| Memory lifecycle | [CUDA](../../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t060537096336z-cuda-rtx4090-laptop-wsl2-cuda-memory-lifecycle-v1-61a2e68c95-v1.json), [Metal](../../qualification/receipts/metal/macbook-air-m1/20260728t230542216939z-metal-macbook-air-m1-metal-memory-lifecycle-v-dfc8d17c13-v1.json) | Controlled admission, allocation-failure, reclaim, and cleanup cases |
| Serving capacity | [CUDA passing c1–c4](../../benchmarks/receipts/cuda/rtx4090-laptop-wsl2/20260728t090043z-cuda-wsl2-qwen35-4b-greedy-short-c1-4-qualified-v1.kiln.json), [CUDA first-nonfit search](../../benchmarks/receipts/cuda/rtx4090-laptop-wsl2/20260728t084724z-cuda-wsl2-qwen35-4b-greedy-short-c1-16-capacity-v1.kiln.json), [Metal c19/c20 boundary](../../benchmarks/receipts/metal/macbook-air-m1/20260729t025314z-metal-macbook-air-m1-qwen35-4b-greedy-short-c19-64-capacity-boundary-search-v1.kiln.json) | Fixed model, prompt, output, configuration, timeout, and memory gate |
| Endurance | [CUDA](../../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t113040389600z-cuda-rtx4090-laptop-wsl2-serving-cuda-endurance-v-0d78751328-v1.json), [Metal passing](../../qualification/receipts/metal/macbook-air-m1/20260729t125321404525z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json), [Metal failed counterevidence](../../qualification/receipts/metal/macbook-air-m1/20260729t043210429221z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json) | Declared eight-hour mixed-load envelope only |
| Targeted serving closure | [ROCm](../../qualification/receipts/rocm/strix-halo/20260729t023946445804z-rocm-strix-halo-serving-backend-regressi-b18fa140d9-v1.json), [Vulkan](../../qualification/receipts/vulkan/strix-halo/20260729t013047275616z-vulkan-strix-halo-serving-backend-regressi-2a1ed8c677-v1.json) | Two-request KV-growth correctness fixture, not performance evidence |
| Hosted build checkpoint | [GitHub Actions run 30498143581](https://github.com/ericflo/kiln/actions/runs/30498143581) | Compile and test evidence; hosted CUDA and ROCm lanes did not execute hardware |

For current performance claims and comparable benchmark rows, use
[Benchmarks](../public/BENCHMARKS.md). The evidence table above exists to show how
scope is bounded, not to summarize current speed.

## Numerical Oracles

Hugging Face next-token and ROCm attribution drivers use a bounded process
runner with an explicit start gate, finite timeout, new process group,
`SIGTERM` followed by bounded `SIGKILL`, and closed cleanup evidence.

The Vulkan full-model oracle compares all-vocabulary logits, argmax, top-10
overlap, maximum and mean absolute error, and cosine similarity. The wrapper
does not change numerical tolerances.

Validate specialized oracle results with:

```bash
python3 scripts/qualification/check_oracle_results.py \
  /absolute/path/to/result.json
```

Version 1 oracle results that contain machine-temperature policy are
intentionally unsupported.

## Resumable GDN Prefill Residency Telemetry

ROCm and Vulkan resumable-prefill cases can retain resident and nonresident
forward counts, recurrent-state continuity, prompt-chunk boundaries, and
allocator activity. Those fields can show that the requested resident route
executed and that chunked prefill matched the declared monolithic reference.
They prove only the route and sequence named by the workload.

## Batched Recurrent-State Cache Telemetry

Batched hybrid-model cases can retain active and idle slot counts,
recurrent-state bytes, admissions, releases, and route failures. A passing
case requires closed ownership accounting and exact state continuity across
the declared request sequence; it does not establish behavior outside that
sequence.

## Failure Triage

When a run fails, preserve the first failure and inspect evidence in this order:

1. **Preflight:** clean worktree, committed workload, selected variant, and
   resolved variables.
2. **Environment:** backend visibility, selected device, required tools, and
   unsupported probe entries.
3. **Execution:** exit code, timeout, stdout/stderr truncation, and process
   cleanup.
4. **Structured result:** case ID, effective configuration, declared metrics,
   finite values, and canonical ordering.
5. **Correctness gates:** output assertions, numerical policy, and required
   case status.
6. **Final integrity:** source drift, model drift, artifact hashes, listener or
   descendant residue, and receipt validation.

Do not “fix” a missing capability by copying another machine’s identifier into
the workload. Either make the implementation discover and use the available
capability, or declare a narrowly scoped regression fixture.

## Publication Checklist

Before publishing or promoting a result:

1. Confirm the worktree was clean and the source commit was pushed.
2. Validate the exact committed workload.
3. Confirm the intended backend and physical device executed every required
   case.
4. Confirm model, tokenizer, source, configuration, runtime, and artifact
   identities.
5. Confirm numerical and output correctness before quoting performance.
6. Confirm bounded shutdown, listener cleanup, process cleanup, and device
   cleanup.
7. Compare only rows admitted by the workload’s comparison policy.
8. Publish failed boundary rows as counterevidence rather than hiding them.
9. State the exact workload, backend, device, platform, and date.
10. Keep every machine-specific fact out of portable product defaults.

Run the focused tooling suite before changing qualification contracts:

```bash
python3 -m unittest discover \
  -s scripts/qualification/tests \
  -p 'test_*.py'
```
