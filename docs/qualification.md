# Local Hardware Qualification

Kiln qualification proves a declared contract on the machine that ran it. A
receipt records evidence; it does not create product defaults for that machine.

## Principles

- Product behavior stays portable across supported backends.
- Hardware qualification is additive: a CUDA, ROCm, Vulkan, or Metal case may
  require that backend, but not an unrelated laptop model or temperature
  sensor.
- Current evidence is source-bound, artifact-bound, schema-validated, and
  fail-closed.
- Historical receipts remain historical. They are not copied into current
  workload admission or runtime policy.
- Ordinary wall-clock timing controls startup, execution, and shutdown.

## Prerequisites

Install the toolchain required by the backend under test:

- Rust and the repository lockfile dependencies;
- NVIDIA driver and CUDA toolkit for CUDA;
- ROCm userspace and a supported AMD device for ROCm;
- a conformant Vulkan loader/driver for Vulkan;
- Apple platform tooling for Metal; and
- Python environments named by oracle or reference-runtime manifests.

The generic local runner reports missing tools and devices. It does not pretend
that a skipped hardware case passed.

## Workloads

Workload contracts live under `qualification/workloads/`. Validate them before
execution:

```bash
python3 scripts/qualification/run.py \
  qualification/workloads/<workload>.json \
  --variant <variant-id> \
  --host-id <host-id> \
  --output .qualification/receipts/<receipt>.json
```

Use `python3 scripts/qualification/run.py --help` for current variable and
platform options.

Each contract closes:

- case order, seed, repetitions, and parallelism;
- backend/device requirement and skip policy;
- command, environment, working directory, and timeout;
- output assertions;
- declared result metrics; and
- comparison policy.

## Platform Boundary

Native Linux runs use the local process boundary selected by the runner. WSL2
runs may use a systemd user scope through
`scripts/qualification/wsl_scope_exec.py`.

The default WSL2 scope has:

- no CPU quota;
- no fixed memory ceiling;
- a bounded PID count;
- a private result snapshot;
- explicit process cleanup; and
- ordinary wall-clock deadlines.

The boundary records what the platform actually supports. It does not infer
Windows driver identity from a Linux package, fabricate unavailable NVML data,
fabricate unavailable temperature data, or silently disable a required workload
assertion.

## Environment Receipt

An environment receipt identifies the evidence host without turning that host
into a product requirement. Depending on the backend, it may record:

- OS, kernel, architecture, and WSL identity;
- compiler and Rust toolchain;
- accelerator API, driver, toolkit, and device inventory;
- Python/runtime manifest identity;
- systemd/cgroup capability;
- point-in-time host and selected-device temperature observability;
- model and tokenizer content hashes; and
- source commit and tree.

WSL2 records host temperature from readable Linux hwmon inputs or the Windows
formatted thermal provider, and records the selected CUDA device temperature
through NVML. These are typed read-only observations: unavailable sources are
reported explicitly, and readings do not pace workloads, define operating
limits, or select product behavior. The outer runner requires host
observability; the contained case records Windows telemetry as unavailable
because Landlock intentionally blocks Windows execution.

The current retained laptop boundary evidence is
[`20260728t050414137676z-cuda-rtx4090-laptop-wsl2-local-environment-v1-df3e8fee15-v1.json`](../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t050414137676z-cuda-rtx4090-laptop-wsl2-local-environment-v1-df3e8fee15-v1.json).
It passed from clean pushed source `6699d9775e3a`, recorded all declared outer
WSL2 capabilities as available, and passed the contained environment case. It
is evidence for that laptop under WSL2 only.

## Build Boundary

Bounded build wrappers provide deterministic environment filtering, offline
dependency use, finite runtime, and cleanup:

```bash
scripts/cargo-bounded.sh build --release --locked --offline
scripts/qualification/cargo-test-bounded.sh test --locked --offline
```

The wrapper does not select a single ROCm architecture unless a caller is
explicitly building a hardware regression fixture. Normal ROCm builds use the
toolchain/device target selected by the existing build system.

The qualification launcher also does not pin a machine-sized minimum available
memory value. `cargo-bounded.sh` derives its admission floor and host reserve
from the current host, while keeping one build job and aggregate cgroup
accounting. A caller may still declare an explicit bound when a committed
workload needs one.

## Serving Qualification

The serving protocol is defined in
[Serving Benchmark Protocol](SERVING_BENCHMARK_PROTOCOL.md). Current serving
drivers:

- bind source, binary, configuration, model, tokenizer, and runtime;
- own one server process group;
- bound readiness, requests, and shutdown;
- reject listener/process residue;
- use wall-clock timing; and
- retain strict result and receipt documents.

Specific benchmark receipts may name the host that produced them. Do not reuse
those host IDs, UUIDs, memory sizes, or architecture names as current workload
requirements.

## Numerical Oracles

The Hugging Face next-token, ROCm path-attribution, and layer-attribution
drivers use a small process runner with:

- an explicit start gate;
- a finite worker timeout;
- a new process group;
- `SIGTERM` followed by bounded `SIGKILL` cleanup; and
- closed process-containment evidence.

They do not apply temperature thresholds, fixed host-memory ceilings, swap
policy, CPU quotas, or a hardcoded ROCm architecture.

Current result schemas:

- `qualification/schema/rocm-hf-next-token-oracle-v2.schema.json`
- `qualification/schema/rocm-hf-path-attribution-v2.schema.json`
- `qualification/schema/rocm-hf-layer-attribution-v2.schema.json`

The Vulkan full-model oracle compares all vocabulary logits, argmax, top-10
overlap, maximum error, mean error, and cosine similarity. The process wrapper
does not change numerical tolerances.

## Resumable GDN Prefill Residency Telemetry

ROCm and Vulkan resumable-prefill cases retain resident/nonresident forward
counts, recurrent-state continuity, prompt-chunk boundaries, and allocator
activity. Use those fields to prove that a requested resident route actually
executed and that chunked prefill matches the monolithic reference.

## Batched Recurrent-State Cache Telemetry

Batched hybrid-model cases retain active/idle slot counts, recurrent-state
bytes, admissions, releases, and route failures. A passing case requires closed
ownership accounting and exact state continuity across the declared request
sequence.

## Validation

Run focused qualification tooling tests:

```bash
python3 -m unittest discover \
  -s scripts/qualification/tests \
  -p 'test_*.py'
```

Validate current specialized oracle results with:

```bash
python3 scripts/qualification/check_oracle_results.py \
  /absolute/path/to/result.json
```

Version 1 oracle results containing machine temperature policy are intentionally
unsupported by the current dispatcher.

## Receipt Interpretation

A passing receipt establishes only its declared workload, source, artifacts,
backend, and host. It does not establish:

- correctness outside the tested cases;
- performance on a different device;
- native Linux behavior from a WSL2 run;
- a desktop result from a laptop result;
- high-concurrency parity from a c1 run; or
- endurance beyond the measured duration.

A failing receipt may be retained when it localizes a correctness or lifecycle
defect. Do not average failed rows into performance claims.

## Publication Checklist

Before publishing or promoting a result:

1. Confirm the worktree was clean and pushed.
2. Validate the workload and result schemas.
3. Confirm model, tokenizer, binary, configuration, and runtime hashes.
4. Confirm the required device really executed the case.
5. Confirm correctness and numerical tolerances.
6. Confirm shutdown, listener cleanup, and process cleanup.
7. Compare only equivalent workload rows.
8. State the exact backend and hardware scope.
9. Keep machine-specific evidence out of portable defaults.
