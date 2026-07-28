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

## Current CUDA Core Evidence

The current retained laptop CUDA core receipt is
[`20260728t051305956568z-cuda-rtx4090-laptop-wsl2-cuda-metal-core-correctn-9f21d75c94-v1.json`](../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t051305956568z-cuda-rtx4090-laptop-wsl2-cuda-metal-core-correctn-9f21d75c94-v1.json).
It passed from clean source `b81c6787d0a4` and required the selected CUDA
device, tensor and matmul parity, CUDA graph replay against eager execution,
one complete CUDA LoRA SFT step, and a twenty-step BF16 AdamW trajectory
against the pinned PyTorch oracle. Each required case exited zero with no
output-assertion failure; every owned WSL2 scope was removed without a cgroup
memory event.

This is evidence for the declared core subset on the measured RTX 4090 Laptop
under WSL2. The separate memory-lifecycle receipt below supplies memory
evidence, and the separate serving-capacity receipts supply serving and
concurrency evidence. This core receipt is not soak, native Linux, desktop RTX
4090, or Metal evidence.

## Current CUDA Memory Lifecycle Evidence

The current retained laptop CUDA memory-lifecycle receipt is
[`20260728t060537096336z-cuda-rtx4090-laptop-wsl2-cuda-memory-lifecycle-v1-61a2e68c95-v1.json`](../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t060537096336z-cuda-rtx4090-laptop-wsl2-cuda-memory-lifecycle-v1-61a2e68c95-v1.json).
It passed from clean pushed source `4b68c4493972`. Its five required cases
selected the declared laptop GPU, reclaimed two GiB held by the CUDA pool,
rejected a request one block above the live admission ceiling before
allocation, recovered from a controlled injected allocator error by retrying
a smaller real CUDA cache, and preserved a marker through a
4,000-to-500-to-4,000 block physical KV resize. Every case exited zero with no
output-assertion failure; every owned WSL2 scope reported zero cgroup memory
events, was removed, and left no CUDA or qualification process.

The controlled error proves the server allocation-retry path and its real CUDA
fallback allocation. It does not claim a physical device OOM, full-model
memory pressure, soak, native Linux, desktop RTX 4090, or Metal evidence. The
serving and concurrency evidence is retained separately below.

## Current CUDA Serving Capacity Evidence

The passing laptop CUDA serving receipt is
[`20260728t090043z-cuda-wsl2-qwen35-4b-greedy-short-c1-4-qualified-v1.kiln.json`](../benchmarks/receipts/cuda/rtx4090-laptop-wsl2/20260728t090043z-cuda-wsl2-qwen35-4b-greedy-short-c1-4-qualified-v1.kiln.json).
It ran from clean pushed source `a7156931b130` with a source-built CUDA binary,
fixed model and server configuration, temperature zero, seed 17, and exact
64-token outputs. Under the independent 15 GiB whole-device limit, c1 through
c4 passed at 22.92, 22.98, 26.82, and 27.39 aggregate output tokens per second.

The companion
[`20260728t084724z-cuda-wsl2-qwen35-4b-greedy-short-c1-16-capacity-v1.kiln.json`](../benchmarks/receipts/cuda/rtx4090-laptop-wsl2/20260728t084724z-cuda-wsl2-qwen35-4b-greedy-short-c1-16-capacity-v1.kiln.json)
is retained capacity counterevidence rather than a passing receipt. It repeated
the c1-through-c4 passes and found c5 to be the first non-fitting concurrency:
peak device memory was 16,303,263,744 bytes against the 16,106,127,360-byte
limit. All 136 measured requests still succeeded with the exact required
64-token output; only the absolute-memory gate failed from c5 onward. Both
runs preserved their source, artifact, configuration, model, and runtime
identities and passed final process, port, and GPU cleanup.

The c4 ceiling is specific to this fixed workload, configuration, memory gate,
and measured RTX 4090 Laptop under WSL2. These receipts are not an SLO claim
and do not themselves provide endurance, native Linux, desktop RTX 4090, or
Metal evidence. The separate receipt below supplies the declared endurance
evidence.

## Current CUDA Endurance Evidence

The retained laptop CUDA endurance receipt is
[`20260728t113040389600z-cuda-rtx4090-laptop-wsl2-serving-cuda-endurance-v-0d78751328-v1.json`](../qualification/receipts/cuda/rtx4090-laptop-wsl2/20260728t113040389600z-cuda-rtx4090-laptop-wsl2-serving-cuda-endurance-v-0d78751328-v1.json).
It passed from clean pushed source `fe3f1a694704`. The contained case measured
28,812.55 seconds using ordinary monotonic time over the fixed c1/c4 and
16/32/64/96-word prompt envelope. All 6,980 measured requests completed the
exact 32-token response oracle with zero failures, and all 698 scheduled
cancellations were confirmed. Aggregate output throughput was 7.752 tokens per
second.

The gate established a 17,171,480,576-byte whole-envelope GPU high-water
baseline before measurement. The measured peak matched that baseline, the
final value was 15,365,832,704 bytes, and post-baseline GPU growth was zero.
RSS grew 8,650,752 bytes. The fixed 303-block KV capacity ended idle with zero
unaccounted blocks. There were zero device faults, unexplained ITL outliers,
host-memory guard trips, request-worker residues, forced or nonzero shutdowns,
snapshot residues, cgroup memory events, surviving scopes, or CUDA processes.

CUDA graphs, allocator reclaim, and prefix caching were disabled for this
workload; the prefix cache remained quarantined for CUDA prefill semantics.
The receipt preserves the exact source, workload, effective configuration,
model hashes, platform boundary, and hashed local logs. This is evidence only
for the declared eight-hour mixed-load workload on the measured RTX 4090
Laptop under WSL2. It is not native Linux, desktop RTX 4090, Metal, an SLO, or
a broader workload claim.

## Build Boundary

Bounded build wrappers provide deterministic environment filtering, offline
dependency use, finite runtime, and cleanup:

```bash
scripts/cargo-bounded.sh build --release --locked --offline
scripts/qualification/cargo-test-bounded.sh test --locked --offline
```

The qualification test launcher uses a private transient systemd service on
native Linux and the runner-owned delegated cgroup inside the declared WSL2
scope. It fails closed instead of treating the WSL2-only boundary as portable
to native ROCm or Vulkan hosts.

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
