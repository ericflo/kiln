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

On macOS, workloads that forbid network access run under a fail-closed
`sandbox-exec` profile that permits loopback and denies external inbound and
outbound traffic. Each case owns a new session and process group; the runner
requires descendant settlement and applies bounded termination and kill
cleanup. The contained environment case independently verifies the network
denial, loopback path, session, and process-group identity.

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

## Current Metal Platform Evidence

The current retained MacBook Air boundary receipt is
[`20260728t223911446266z-metal-macbook-air-m1-local-environment-v1-99474c38c1-v1.json`](../qualification/receipts/metal/macbook-air-m1/20260728t223911446266z-metal-macbook-air-m1-local-environment-v1-99474c38c1-v1.json).
It passed from clean pushed source `c8d9f5856ce9`. The selected Metal device,
`system_profiler` hardware view, `sysctl` chip identity, and Metal runtime
agreed on Apple M1. The measured host reported a MacBook Air model, eight GPU
cores, and 17,179,869,184 unified-memory bytes; the workload does not use the
core count or memory total as a product default or assume them for another
MacBook Air configuration.

The receipt binds macOS 26.6 build 25G72, Xcode 26.6 build 17F113, the macOS
26.5 SDK build 25F70, Command Line Tools, the installed Metal toolchain, and
the live Metal runtime. Its compiler probe produced nonempty AIR and metallib
artifacts. The workspace probe identified writable APFS and passed full file
sync, directory sync, atomic replacement, hardlink, relative-symlink, and
case-insensitive lookup checks.

The outer runner and contained case both required
`macos-sandbox-loopback-only-v1`. The contained case connected over loopback,
received a permission denial for external networking, proved that its PID,
session, and process-group IDs matched, and exited without descendant residue.
At capture, `memory_pressure` reported 71% free memory, swap total and use were
zero, and the receipt retained page, compressor, and paging counters. `pmset`
reported no recorded thermal, performance, or CPU-power warning. macOS exposed
no supported unprivileged host or selected-GPU temperature reading, so both
typed observations are explicitly unavailable. These observations are
read-only and do not pace execution, select routes, or set memory policy.

This receipt proves only the platform boundary on the measured M1 MacBook Air.
By itself it is not Metal tensor, training, serving, capacity, performance, or
endurance evidence; those Phase 7.2 receipts are retained separately.

## Current Metal Core Evidence

The current retained MacBook Air Metal core receipt is
[`20260728t225405419496z-metal-macbook-air-m1-cuda-metal-core-correctn-d119e83143-v1.json`](../qualification/receipts/metal/macbook-air-m1/20260728t225405419496z-metal-macbook-air-m1-cuda-metal-core-correctn-d119e83143-v1.json).
It passed from clean pushed source `94d03199f334`. The five required cases
selected Apple M1, compared contiguous, strided-view, and BF16 tensor round
trips with exact CPU values, and compared F32, batched, left-transposed, and
right-transposed matrix-core matmuls with CPU oracles.

The graph case required real single-row and two-row Metal ICB replay across two
sequence buckets, with exact token and KV-cache parity against eager execution.
The training case completed one Metal LoRA SFT forward/backward/AdamW step and
a twenty-step BF16 AdamW trajectory against the pinned PyTorch oracle. Every
case exited zero with no output-assertion failure, and the receipt passed
independent current-source and local-artifact validation.

This is bounded correctness evidence for the declared subset on the measured
M1 MacBook Air. It is not memory-lifecycle, serving, capacity, performance,
endurance, or other-Apple-Silicon evidence.

## Current Metal Unified-Memory Lifecycle Evidence

The retained MacBook Air Metal memory-lifecycle receipt is
[`20260728t230542216939z-metal-macbook-air-m1-metal-memory-lifecycle-v-dfc8d17c13-v1.json`](../qualification/receipts/metal/macbook-air-m1/20260728t230542216939z-metal-macbook-air-m1-metal-memory-lifecycle-v-dfc8d17c13-v1.json).
It passed from clean pushed source `ee39b573566d`. The admission case used the
live Apple unified-memory snapshot and a controlled floor that left a 64 MiB
budget. The exact server paged-KV gate rejected a 10,241-block request before
Metal `currentAllocatedSize` changed.

The allocation-failure case injected the first server auto-sizer allocation
error, retried at the declared lower fraction, allocated a real Metal paged-KV
cache, released it near the pre-test baseline, and successfully allocated
again. The reclaim case made a separate 268,435,456-byte Metal allocation,
observed it through `currentAllocatedSize`, dropped its sole storage owner,
observed all 268,435,456 bytes reclaimed, and successfully allocated again.

All four required cases passed in runner-owned macOS sessions/process groups
with bounded descendant settlement and cleanup. The receipt passed independent
current-source and local-artifact validation. This is controlled UMA
admission, injected-failure recovery, ownership-release reclaim, and process
cleanup evidence. It is not a physical host OOM, full-model pressure,
serving-capacity, performance, endurance, or other-Apple-Silicon claim.

## Current Metal Serving Capacity Evidence

The fixed greedy-short passing receipts are
[`c1-c12`](../benchmarks/receipts/metal/macbook-air-m1/20260729t005123z-metal-macbook-air-m1-qwen35-4b-greedy-short-c1-12-capacity-v1.kiln.json),
[`c13`](../benchmarks/receipts/metal/macbook-air-m1/20260729t013755z-metal-macbook-air-m1-qwen35-4b-greedy-short-c13-capacity-v1.kiln.json),
[`c14`](../benchmarks/receipts/metal/macbook-air-m1/20260729t015029z-metal-macbook-air-m1-qwen35-4b-greedy-short-c14-capacity-v1.kiln.json),
[`c15`](../benchmarks/receipts/metal/macbook-air-m1/20260729t020539z-metal-macbook-air-m1-qwen35-4b-greedy-short-c15-capacity-v1.kiln.json),
[`c16`](../benchmarks/receipts/metal/macbook-air-m1/20260729t021732z-metal-macbook-air-m1-qwen35-4b-greedy-short-c16-capacity-v1.kiln.json),
[`c17`](../benchmarks/receipts/metal/macbook-air-m1/20260729t023011z-metal-macbook-air-m1-qwen35-4b-greedy-short-c17-capacity-first-nonfit-v1.kiln.json),
and
[`c18`](../benchmarks/receipts/metal/macbook-air-m1/20260729t023743z-metal-macbook-air-m1-qwen35-4b-greedy-short-c18-capacity-v1.kiln.json).
The final
[`c19/c20 boundary receipt`](../benchmarks/receipts/metal/macbook-air-m1/20260729t025314z-metal-macbook-air-m1-qwen35-4b-greedy-short-c19-64-capacity-boundary-search-v1.kiln.json)
is retained counterevidence with a failed overall verdict. The c17 run
identifier was named prospectively: its signed verdict, all 17 requests,
memory gate, and every other gate passed.

All rows used the same model-visible prompt set and workload semantics:
temperature zero, seed 17, one repeat, 158 prompt tokens, an exact 64-token
completion, a 600-second request timeout, and a 16,106,127,360-byte whole-host
Apple unified-memory limit sampled every 250 ms. Clean pushed sources
`5187097af964`, `c4f2769f17a5`, and `ed63928a8f6a` supplied the c1-c18 passes.
The final adjacent c19/c20 discriminator used clean pushed source
`74a62a614545` and one exact runtime artifact.

All 190 measured requests from c1 through c19 completed successfully with the
required exact output. Aggregate output throughput ranged from 0.544908 to
4.517342 tokens per second, and the largest observed passing whole-host peak
was 15,118,284,882 bytes at c9, below the limit. c19 passed 19/19 at 3.086786
tokens per second with a 14,602,888,807-byte peak. c20 reached 600.024 seconds
with 12/20 successes and a 14,431,090,115-byte peak. The eight incomplete
streams lacked terminal usage records, failing request-success,
positive-usage, fixed-output, and uniform-prompt-accounting gates rather than
the memory gate. The stop control retained c19 and c20, recorded c21-c64 as an
unexecuted suffix, and issued no higher load.

Every receipt preserved its repository, model, artifact, and execution
identity and completed owned-server finalization. The c19/c20 server returned
zero after non-forced shutdown and left no listener or process group. SLO
goodput and tail latency are retained as performance measurements but are not
capacity verdict gates. The c19 ceiling applies only to this fixed workload,
configuration, timeout, and measured M1 MacBook Air. It is not an endurance,
other-workload, or other-Apple-Silicon claim.

## Current Metal Endurance Evidence

The retained passing MacBook Air Metal endurance receipt is
[`20260729t125321404525z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json`](../qualification/receipts/metal/macbook-air-m1/20260729t125321404525z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json).
It passed from clean pushed source `fa3ab5a9dab7` and passed independent
current-source, local-artifact, and known-commit validation. One source-built
server first completed four stable stabilization cycles and then ran for
28,811.809 ordinary monotonic seconds. Across 739 alternating c1/c4 waves it
completed 1,846 exact 32-token responses over fixed 16/32/64/96-word prompt
cohorts, confirmed 184 cancellations, and produced 2.050270 output tokens per
second.

Request, zero-token, non-finite-response, synchronization-failure,
device-fault, batching-error, unexplained-ITL-outlier, and worker-residue
counts were zero. All 1,291 ITL outliers were attributed to retained runtime
events. The receipt retains 902 slow Metal external-yield synchronizations as
diagnostics; none failed. The fixed 389-block KV pool ended unchanged with
zero used or unaccounted blocks. Shutdown was non-forced and zero, with no
snapshot or descendant residue.

The whole-host `memory_pressure` baseline was 13,056,700,580 bytes, the peak
was 14,774,687,499, and the end value was 12,884,901,888. The
1,717,986,919-byte peak increase stayed below the raw 2 GiB growth target plus
one explicit 171,798,692-byte counter-resolution step, an effective
2,319,282,340-byte gate. The peak also stayed below the independent
16,106,127,360-byte absolute limit, and final whole-host growth was zero.
Process RSS started at 2,665,070,592 bytes, peaked at 2,666,692,608, ended at
2,644,131,840, and had zero final growth.

The earlier
[`20260729t043210429221z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json`](../qualification/receipts/metal/macbook-air-m1/20260729t043210429221z-metal-macbook-air-m1-serving-metal-endurance--267c4e3d84-v1.json)
is retained as failed counterevidence. It completed 28,811.213 monotonic
seconds with zero correctness or cleanup failure, but its
2,233,382,994-byte peak increase crossed the raw 2 GiB gate by the
`memory_pressure` counter's next one-percentage-point quantum. The passing
source checkpoint makes that measurement tolerance explicit while preserving
the raw growth target and 15 GiB absolute ceiling. This evidence applies only
to the declared workload and measured M1 MacBook Air. It is not an SLO or
evidence for another Apple Silicon device.

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

## Current Strix Halo Core Evidence

The current ROCm and Vulkan core receipts are
[`20260728t222535119053z-rocm-strix-halo-core-correctness-v1-9faecc7321-v1.json`](../qualification/receipts/rocm/strix-halo/20260728t222535119053z-rocm-strix-halo-core-correctness-v1-9faecc7321-v1.json)
and
[`20260728t222644757744z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json`](../qualification/receipts/vulkan/strix-halo/20260728t222644757744z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-v1.json).
Both passed from the same clean pushed source `8e419e43051b`. ROCm selected
the gfx1151 AMD Radeon 8060S through ROCm 7.2.4; Vulkan selected the same
physical GPU through the RADV Strix Halo driver.

Each backend required its device probe, real-device tensor transfer and
parity, and deterministic dense matmul parity against the CPU oracle. All six
required cases exited zero with no output-assertion failure. This is the
bounded core-correctness subset on this Strix Halo host. It is not serving,
KV-pressure, prefix-cache, endurance, capacity, performance, or evidence for
another ROCm or Vulkan device.

## Targeted ROCm/Vulkan Serving Closure

The correctness-only
[`serving-backend-regression-closure-v1.json`](../qualification/workloads/serving-backend-regression-closure-v1.json)
contract is the bounded current-source serving check for the final
cross-backend closure. It builds one source-bound server per backend with a
fixed eight-block KV pool. Its 253-token prime prompt emits 16 tokens, crosses
token 256, and retains four prefix blocks. Its 255-token pressure prompt
initially owns the other four blocks and emits exactly 256 tokens across four
decode-growth block boundaries without changing workload semantics.

On ROCm, a pass requires structured evidence that live decode growth reclaimed
unleased prefix-cache blocks. On Vulkan, where cross-request prefix reuse
remains correctness-quarantined, a pass requires the same exact output and KV
growth with zero prefix-cache activity or retained state. Both variants require
zero request, device-fault, resize, synchronization-failure, and unaccounted-KV
evidence; drained leases and pending releases; non-forced zero shutdown; and no
private snapshot residue. This is a two-request correctness check, not an
endurance, thermal, capacity, or performance campaign. ROCm retains its
600-second request and 1,800-second overall deadlines. Vulkan uses finite
2,100-second request, 2,400-second overall, and 2,700-second outer-case
deadlines: the correctness-quarantined current-source path completed the prime
request exactly but emitted only 86 of the required 256 pressure tokens before
the former 600-second request deadline. The longer Vulkan window changes no
prompt, token, KV, output, or cleanup gate and is not a throughput threshold.

The retained ROCm result is
[`20260729t023946445804z-rocm-strix-halo-serving-backend-regressi-b18fa140d9-v1.json`](../qualification/receipts/rocm/strix-halo/20260729t023946445804z-rocm-strix-halo-serving-backend-regressi-b18fa140d9-v1.json).
It passed from clean pushed source `29ace2467bea`. Both requests terminated by
length with exact 253/255 prompt and 16/256 completion-token counts. The
pressure request crossed four decode-growth block boundaries, and its first
growth reclaimed all four unleased prime-cache blocks through one structured
event. Final state retained three valid cached blocks while active leases,
pending releases, and unaccounted blocks were zero. This refresh postdates the
shared LM-head routing repair used by Vulkan, proving that change did not
regress the ROCm closure path.

The retained Vulkan result is
[`20260729t013047275616z-vulkan-strix-halo-serving-backend-regressi-2a1ed8c677-v1.json`](../qualification/receipts/vulkan/strix-halo/20260729t013047275616z-vulkan-strix-halo-serving-backend-regressi-2a1ed8c677-v1.json).
It passed from clean pushed source `69246007ac48` with the same exact request
geometry and four pressure decode-growth blocks. The Vulkan correctness
quarantine remained effective: prefix-cache enablement, lookup activity,
cached blocks, leases, pending releases, reclaim events, and retained state
bytes were all zero.

Both receipts recorded zero request, exact-output, device-fault, batching,
synchronization-failure, resize, policy-attestation, unaccounted-KV,
forced-shutdown, nonzero-shutdown, early-exit, and snapshot-residue evidence.
Each passed independent local-artifact, known-commit, and current-source
validation when produced. Later receipt-only Metal serving checkpoints through
`aaca10d4e` did not change runtime or qualification tooling and therefore do
not invalidate these backend-targeted results. Those Metal receipts pass
through c18. Although the c17 run identifier contains `first-nonfit`, its
signed verdict, all requests, memory gate, and every other gate passed, as did
c18; it does not establish a non-fitting boundary. Checkpoint `74a62a614`
subsequently made ascending capacity sweeps stop after the first actually
failed row and retain the unexecuted suffix as structured failure. That
campaign driver is separate from this bounded Strix closure runner. Its
source-bound Metal sweep subsequently passed c19 and retained c20 as the first
non-fitting row.

## Hosted Cross-Backend Checkpoint

GitHub Actions
[`30416196655`](https://github.com/ericflo/kiln/actions/runs/30416196655)
passed `backend_build=all` from exact pushed source `56000a370874`. The
separate Metal, CUDA, ROCm, and Vulkan jobs all passed, along with Linux
default features, formatting, dependency policy, and portable receipt
validation. This is an interim hosted compile/test checkpoint: Phase 7.2 was
still active at that source, so final cross-backend closure still requires the
same matrix from the eventual integrated source.

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
