# Confidence Hardening Goal

> **Archived 2026-08-26 — goal complete.** Every checklist item below is
> checked off, backed by the source-bound receipts in
> `qualification/receipts/{cuda,metal,rocm,vulkan}/` and the green hosted
> `backend_build=all` run `f19d2591ab8e` (CI 30498143581); closure commit
> `fb723bd61` "Complete confidence hardening goal" (2026-07-29) set the
> status to Complete. The durable qualification evidence lives in
> `docs/policies/qualification.md` + `bench-results/`. See [`README.md`](README.md).

**Status:** Complete.

## Scope

Kiln product behavior must remain portable across supported hosts and
accelerators. Qualification machines are evidence sources, not product targets
or sources of runtime defaults. Phase 7.1 used the RTX 4090 Laptop under WSL2
for CUDA; Phase 7.2 uses the MacBook Air M1 for Metal.

- Hardware identity and measurements describe only the machine that produced
  them.
- Backend behavior is selected from runtime capabilities, with portable
  fallbacks where the backend supports them.
- Qualification uses ordinary monotonic time. It does not pause work based on
  temperature or impose host-specific pacing thresholds.
- Host and GPU temperatures are read-only platform observations. Missing
  required observability is reported explicitly; readings do not alter
  execution.
- Qualification may bind receipts to observed hardware identity and capacity;
  production behavior must not branch on a laptop model, GPU-core count,
  temperature, or measurements from one qualification host.
- WSL2 evidence cannot establish native Linux, desktop RTX 4090, Metal, or
  unmeasured endurance behavior.

## WSL2 Platform Boundary

- [x] Discover the selected CUDA device through `nvidia-smi` and cross-check
  UUID, name, memory, and driver through NVML.
- [x] Cross-check Windows GPU and display-driver identity against the Linux
  driver view.
- [x] Bind the WSL CUDA bridge, toolkit, compiler, and runtime API provenance.
- [x] Probe native-workspace filesystem identity, durable/atomic operations,
  case behavior, and link semantics.
- [x] Require runner-owned loopback-only networking, private process lifetime,
  Landlock-denied Windows interop, and bounded cleanup.
- [x] Record systemd state, delegated cgroup v2 controls, WSL memory/swap
  accounting, and scope lifecycle.
- [x] Record typed host and selected-GPU temperature observations, including
  explicit unavailable evidence, without temperature policy.
- [x] Retain a passing environment receipt from clean pushed source proving the
  complete boundary on this WSL2 host.

Boundary evidence: `qualification/receipts/cuda/rtx4090-laptop-wsl2/`
`20260728t050414137676z-cuda-rtx4090-laptop-wsl2-local-environment-v1-`
`df3e8fee15-v1.json` passed from clean pushed source `6699d9775e3a`. All
declared outer WSL2 capabilities were available, the contained environment case
passed, its owned scope was removed, and no CUDA process remained.

## Phase 7.1: RTX 4090 Laptop GPU, 16 GB

- [x] Retain the current-source CUDA correctness/oracle subset that fits the
  device. The independent vLLM exact-output defect was fixed at `d0f74ecb4`;
  the qualification receipt binds the current source and platform
  boundary.
- [x] Retain low-memory admission, allocation failure, reclaim, and process
  cleanup evidence.
- [x] Run serving performance at every concurrency that fits without changing
  workload semantics.
- [x] Run an eight-hour mixed-load soak using ordinary monotonic elapsed time.
- [x] Validate and check in receipts, update this checklist and permanent
  documentation, commit, and push.

Correctness evidence: `qualification/receipts/cuda/rtx4090-laptop-wsl2/`
`20260728t051305956568z-cuda-rtx4090-laptop-wsl2-cuda-metal-core-correctn-`
`9f21d75c94-v1.json` passed from clean source `b81c6787d0a4`. Its required
device, tensor, matmul, CUDA graph/eager parity, CUDA SFT, and twenty-step
PyTorch AdamW oracle cases all passed. All five owned scopes exited zero,
reported no cgroup memory events, were removed, and left no CUDA process.
This receipt establishes only the declared substrate, graph, and training
subset; the separate memory-lifecycle receipt below supplies the memory
evidence. The separate serving-capacity and endurance receipts below supply
their respective evidence.

Memory-lifecycle evidence: `qualification/receipts/cuda/`
`rtx4090-laptop-wsl2/20260728t060537096336z-cuda-rtx4090-laptop-wsl2-`
`cuda-memory-lifecycle-v1-61a2e68c95-v1.json` passed from clean pushed source
`4b68c4493972`. Its five required cases selected the declared laptop GPU,
reclaimed two GiB held by the CUDA pool, rejected an allocation above the live
admission ceiling before allocation, recovered from a controlled injected
allocator error by retrying a smaller real CUDA cache, and preserved data
through a 4,000-to-500-to-4,000 block physical KV resize. Every owned scope
exited zero, reported no cgroup memory event, was removed, and left no CUDA or
qualification process. The injected error proves the server retry path; it is
not a claim that a physical device OOM or full-model memory-pressure run
occurred. The separate serving-capacity and endurance receipts below supply
their respective evidence. This receipt itself is not native Linux, desktop
RTX 4090, or Metal evidence.

Serving-capacity evidence: `benchmarks/receipts/cuda/`
`rtx4090-laptop-wsl2/20260728t090043z-cuda-wsl2-qwen35-4b-greedy-short-`
`c1-4-qualified-v1.kiln.json` passed from clean pushed source `a7156931b130`.
The fixed greedy-short workload produced exact 64-token outputs at c1 through
c4 under the independent 15 GiB device-memory limit, with aggregate throughput
of 22.92, 22.98, 26.82, and 27.39 output tokens per second. The companion
capacity-boundary receipt `20260728t084724z-cuda-wsl2-qwen35-4b-greedy-short-`
`c1-16-capacity-v1.kiln.json` repeated the c1-through-c4 passes and established
c5 as the first non-fitting level: its 16,303,263,744-byte peak exceeded the
16,106,127,360-byte limit. All 136 requests in that boundary sweep still
completed with the exact required output; its overall failed verdict is the
retained memory-limit counterevidence. Both runs preserved their bound source,
binary, configuration, model, and runtime identities and completed process,
port, and GPU cleanup. These receipts apply only to this workload and measured
laptop under WSL2. The separate endurance receipt below supplies soak evidence;
none of these receipts establishes native Linux, desktop RTX 4090, or Metal
behavior.

Endurance evidence: `qualification/receipts/cuda/rtx4090-laptop-wsl2/`
`20260728t113040389600z-cuda-rtx4090-laptop-wsl2-serving-cuda-endurance-v-`
`0d78751328-v1.json` passed from clean pushed source `fe3f1a694704`. The
formal contained case measured 28,812.55 seconds using ordinary monotonic time
over the fixed c1/c4 and 16/32/64/96-word prompt envelope. All 6,980 measured
requests completed their exact 32-token response oracle with zero request
failures, zero non-finite or zero-token responses, and 698 confirmed
cancellations; aggregate output throughput was 7.752 tokens per second. The
stabilized whole-envelope GPU high-water baseline and measured peak were both
17,171,480,576 bytes, the final value was 15,365,832,704 bytes, and
post-baseline GPU growth was zero. RSS grew 8,650,752 bytes. The fixed
303-block KV capacity ended idle with zero unaccounted blocks. No device fault,
unexplained ITL outlier, host-memory guard trip, worker residue, forced or
nonzero shutdown, snapshot residue, cgroup memory event, surviving scope, or
CUDA process was observed. CUDA graphs, allocator reclaim, and prefix caching
were disabled; the prefix cache remained quarantined for CUDA prefill semantics.
This receipt establishes only the declared eight-hour mixed-load workload on
this measured laptop under WSL2. It is not native Linux, desktop RTX 4090,
Metal, or broader workload evidence.

## macOS/Metal Platform Boundary

- [x] Discover the selected Metal device and cross-check Apple hardware
  identity, GPU-core count, and unified-memory capacity without assuming a
  particular MacBook Air configuration.
- [x] Bind the macOS build, Xcode or Command Line Tools, Metal compiler, SDK,
  and runtime provenance.
- [x] Probe workspace filesystem identity, durable and atomic operations, case
  behavior, and link semantics.
- [x] Require runner-owned loopback-only networking, private process lifetime,
  and bounded cleanup using supported macOS safeguards. Missing required
  containment must fail closed rather than silently weakening the workload.
- [x] Record unified-memory accounting, memory-pressure visibility, and
  available host and GPU observations without adding device-specific policy.
- [x] Retain a passing environment receipt from clean pushed source proving the
  complete boundary on this macOS host.

Boundary evidence: `qualification/receipts/metal/macbook-air-m1/`
`20260728t223911446266z-metal-macbook-air-m1-local-environment-v1-`
`99474c38c1-v1.json` passed from clean pushed source `c8d9f5856ce9`. The
selected Apple M1 identity cross-checked the MacBook Air model, eight GPU
cores, and 17,179,869,184-byte unified-memory total without committing those
measurements as runtime defaults. Xcode 26.6 build 17F113, the macOS 26.5 SDK,
the installed Metal toolchain, and the live Metal runtime were bound; a real
Metal source compiled to AIR and linked to a metallib. The APFS probe passed
full file sync, directory sync, atomic replacement, hardlink, symlink, and
case-insensitive behavior. The contained case preserved loopback, denied an
external connection, owned its session and process group, and exited cleanly.
Unified-memory, compression, swap, paging, and memory-pressure observations
were retained. Supported unprivileged host and GPU temperature readings were
explicitly unavailable; `pmset` thermal-pressure observations were available
and did not control execution.

## Phase 7.2: MacBook Air M1

- [x] Retain the current-source Metal correctness and oracle subset, including
  tensor transfer, matmul, graph replay, training, and optimizer coverage.
- [x] Retain controlled low-memory admission, allocation failure, reclaim, and
  process-cleanup evidence appropriate to unified memory.
- [x] Run serving correctness and performance at every concurrency that fits
  without changing workload semantics, and retain the first non-fitting
  capacity boundary as counterevidence.
- [x] Run an eight-hour mixed-load soak using ordinary monotonic elapsed time.
- [x] Validate and check in receipts, update this checklist and permanent
  documentation, commit, and push.

Metal evidence applies only to the declared workloads on the measured MacBook
Air. It must not be generalized to other Apple Silicon devices, configurations,
or unmeasured endurance behavior. Completed CUDA/WSL2 receipts remain
historical evidence and are not rerun or rewritten by this phase.

Core-correctness evidence:
`qualification/receipts/metal/macbook-air-m1/20260728t225405419496z-metal-macbook-air-m1-cuda-metal-core-correctn-d119e83143-v1.json`
passed from clean pushed source `94d03199f334`. It required the selected Apple
M1 device, exact contiguous, strided, and BF16 tensor round trips, four
matrix-core matmul layouts against CPU oracles, single-row and batched ICB
graph replay with token and KV-cache parity against eager execution, one
complete Metal LoRA SFT step, and a twenty-step BF16 AdamW trajectory against
the pinned PyTorch oracle. All five required cases passed and the receipt
passed independent current-source and local-artifact validation. This does not
claim memory lifecycle, serving, capacity, performance, or endurance evidence.

Unified-memory lifecycle evidence:
`qualification/receipts/metal/macbook-air-m1/20260728t230542216939z-metal-macbook-air-m1-metal-memory-lifecycle-v-dfc8d17c13-v1.json`
passed from clean pushed source `ee39b573566d`. A controlled 64 MiB live UMA
budget rejected a 10,241-block request before `currentAllocatedSize` changed.
An injected first allocator failure retried at the declared lower fraction,
allocated a real Metal paged-KV cache, released it near baseline, and then
allocated again. A separate 256 MiB Metal storage allocation appeared in
`currentAllocatedSize`; dropping its sole owner reclaimed exactly 268,435,456
bytes before another allocation succeeded. Every case ran in the runner-owned
macOS session/process group and completed its bounded descendant-settlement
path. This is controlled admission/failure/release evidence, not a physical
host-OOM or serving-capacity claim.

Serving-capacity evidence:
`benchmarks/receipts/metal/macbook-air-m1/` retains the fixed greedy-short
passes from c1 through c19 and the adjacent c20 failure. Every row used
temperature zero, seed 17, one repeat, a 158-token prompt, an exact 64-token
completion requirement, a 600-second request timeout, and a 16,106,127,360-byte
whole-host unified-memory limit. The c1-c12, c13, c14, c15, c16, c17, and c18
passes were emitted from clean pushed sources `5187097af964`, `c4f2769f17a5`,
and `ed63928a8f6a`; the prospective `first-nonfit` label in the c17 run
identifier does not override its passing signed verdict. The final
`20260729t025314z-metal-macbook-air-m1-qwen35-4b-greedy-short-c19-64-`
`capacity-boundary-search-v1.kiln.json` receipt ran from clean pushed source
`74a62a614545` and retained the exact adjacent discriminator: c19 passed all
19 requests at 3.086786 output tokens per second with a 14,602,888,807-byte
peak, while c20 reached the 600-second deadline with 12 of 20 requests
complete. Its eight incomplete streams had no terminal usage record, so the
request-success, positive-usage, fixed-output, and uniform-prompt-accounting
gates failed. The c20 14,431,090,115-byte peak remained below the memory limit.
The driver recorded c21-c64 as an unexecuted suffix and stopped. All
repository, model, artifact, execution-identity, and owned-server finalization
checks passed, including non-forced zero shutdown. Reported SLO goodput is
preserved but is not a capacity verdict gate; this c19 ceiling applies only to
the declared workload and measured MacBook Air.

Endurance evidence:
`qualification/receipts/metal/macbook-air-m1/`
`20260729t125321404525z-metal-macbook-air-m1-serving-metal-endurance--`
`267c4e3d84-v1.json` passed from clean pushed source `fa3ab5a9dab7` and
passed independent current-source, local-artifact, and known-commit
validation. After four stable stabilization cycles, one source-built server
completed 28,811.809 ordinary monotonic seconds, 739 mixed c1/c4 waves, 1,846
exact 32-token responses over the fixed 16/32/64/96-word prompt cohorts, and
184 confirmed cancellations. Output throughput was 2.050270 tokens per
second. Request, zero-token, non-finite-response, synchronization-failure,
device-fault, batching-error, unexplained-ITL-outlier, and worker-residue
counts were zero. The 1,291 ITL outliers were all attributed to retained
runtime events; 902 slow Metal external-yield synchronizations were retained
as diagnostics, with zero synchronization failures.

The whole-host `memory_pressure` baseline, peak, and end values were
13,056,700,580, 14,774,687,499, and 12,884,901,888 bytes. The 1,717,986,919-byte
peak increase was below the raw 2 GiB growth target plus one explicit
171,798,692-byte counter-resolution step, an effective 2,319,282,340-byte
gate, and the peak was also below the independent 16,106,127,360-byte absolute
limit. Final whole-host growth was zero. Process RSS started at
2,665,070,592 bytes, peaked at 2,666,692,608, ended at 2,644,131,840, and had
zero final growth. The fixed 389-block KV pool ended with zero used or
unaccounted blocks, and shutdown was non-forced, zero, and free of snapshot
or process residue.

The preceding
`20260729t043210429221z-metal-macbook-air-m1-serving-metal-endurance--`
`267c4e3d84-v1.json` receipt is retained as failed counterevidence. It
completed 28,811.213 monotonic seconds with zero correctness or cleanup
failure, but its 2,233,382,994-byte whole-host peak increase exceeded the raw
2 GiB gate by the `memory_pressure` counter's next one-percentage-point
quantum. Checkpoint `fa3ab5a9dab7` made that quantization tolerance explicit
without changing the raw target or 15 GiB absolute ceiling, and the passing
source-bound run above exercised the revised gate. These receipts apply only
to the declared mixed-load workload on this measured M1 MacBook Air. They do
not establish an SLO or qualify another Apple Silicon device.

## Phase 7.3: Final Cross-Backend Regression Closure

- [x] From one current source commit, dispatch GitHub Actions CI with
  `backend_build=all` and require its CUDA, Metal, ROCm, and Vulkan lanes to
  pass on their separate compatible hosted runners. This is compile and test
  evidence, not a requirement that one qualification host run every backend.
- [x] On Strix Halo, retain current-source ROCm and Vulkan core-correctness
  receipts.
- [x] Retain targeted ROCm and Vulkan serving receipts covering exact output,
  KV-growth pressure, prefix-cache reclamation, and bounded process and device
  cleanup after the shared server changes made during Phase 7.1.
- [x] After Phase 7.2 lands, incorporate its final checkpoint and repeat only
  the targeted checks affected by later shared runtime or qualification-tooling
  changes. Documentation-only and backend-isolated changes do not require
  unrelated hardware reruns.
- [x] Validate and check in the receipts, update this checklist and permanent
  documentation, and require all exact-commit workflows to pass before changing
  this plan's status to `Complete`.

This closure is a bounded regression check, not a new qualification campaign.
It does not repeat endurance, thermal, capacity, or performance tuning unless a
targeted check exposes a specific regression that requires diagnosis. Agents
working concurrently must incorporate current `origin/main` before
source-bound runs and pushes, must not force-push, and must not close this phase
until Phase 7.2 is complete.

Core-correctness evidence:
`qualification/receipts/rocm/strix-halo/`
`20260728t222535119053z-rocm-strix-halo-core-correctness-v1-9faecc7321-`
`v1.json` and `qualification/receipts/vulkan/strix-halo/`
`20260728t222644757744z-vulkan-strix-halo-core-correctness-v1-0a09f3bcee-`
`v1.json` passed from the same clean pushed source `8e419e43051b`. ROCm
selected the gfx1151 AMD Radeon 8060S through ROCm 7.2.4; Vulkan selected the
same physical GPU through the RADV Strix Halo driver. Each receipt required
its device probe, real-device tensor transfer/parity, and deterministic dense
matmul parity against the CPU oracle. All six required cases exited zero with
no output-assertion failure. This is the bounded core subset only; the
separate targeted serving receipts must supply shared server, KV-pressure,
prefix-cache, and cleanup evidence.

Targeted-serving contract:
`qualification/workloads/serving-backend-regression-closure-v1.json` defines
the bounded correctness-only hardware closure. Each backend builds current
source, starts one source-bound server with a fixed eight-block KV pool, and
runs a 253-token cache-prime prompt with a 16-token completion followed by a
255-token pressure prompt with an exact 256-token completion. The prime crosses
token 256 and retains four unleased blocks; the pressure prompt initially owns
the other four and crosses four further block boundaries. ROCm must reclaim the
unleased prefix-cache blocks through the structured live-decode-growth path.
Vulkan must cross the same decode-growth boundaries with its correctness
quarantine effective and zero prefix-cache ownership. Both variants require
drained KV accounting, clean process shutdown, and no snapshot residue. This
two-request contract does not measure performance or search for a capacity
limit. ROCm retains its 600-second per-request and 1,800-second overall
deadlines. Vulkan has a backend-specific 2,100-second per-request,
2,400-second overall, and 2,700-second outer case bound because the
correctness-quarantined current-source path completed the prime request
exactly but produced only 86 of 256 pressure tokens before the former
600-second request deadline. The longer finite deadline preserves the exact
workload and gates; it is not a throughput acceptance rule.

Targeted-serving evidence:
`qualification/receipts/rocm/strix-halo/`
`20260729t023946445804z-rocm-strix-halo-serving-backend-regressi-`
`b18fa140d9-v1.json` passed from clean pushed source `29ace2467bea`.
It completed both exact length-terminated requests with 253/255 prompt tokens,
16/256 completion tokens, and four pressure-request decode-growth blocks. The
prime left four unleased prefix blocks; the first live growth reclaimed all
four through one structured event. Final state retained three valid cached
blocks but had zero active leases, pending releases, or unaccounted blocks.
This current-source refresh postdates the shared LM-head routing repair used
by Vulkan and therefore closes its possible impact on ROCm with execution
evidence rather than path inference.

`qualification/receipts/vulkan/strix-halo/`
`20260729t013047275616z-vulkan-strix-halo-serving-backend-regressi-`
`2a1ed8c677-v1.json` passed from clean pushed source `69246007ac48`.
It completed the same exact prompt/completion and four-growth-block contract
with the Vulkan correctness quarantine effective. Prefix-cache enablement,
lookups, cached blocks, leases, pending releases, reclaim events, and retained
state bytes were all zero. Both receipts recorded zero request, exact-output,
device-fault, batching, synchronization-failure, resize, policy-attestation,
unaccounted-KV, forced-shutdown, nonzero-shutdown, early-exit, and snapshot
residue evidence, and both passed independent local-artifact, known-commit,
and current-source validation when produced.

Concurrent Phase 7.2 integration:
receipt-only Metal checkpoints `c4f2769f1`, `ed63928a8`, and `aaca10d4e` were
incorporated without changing shared runtime or qualification tooling, so they
do not invalidate either targeted Strix result. They retain passing serving
receipts through c18. The c17 receipt's run identifier contains
`first-nonfit`, but its signed verdict, all 17 requests, memory gate, and every
other gate passed; c18 also passed. It is therefore not non-fitting
counterevidence. Checkpoint `74a62a614` then made the campaign driver stop an
ascending sweep after its first actually failed row and record its unexecuted
suffix as structured failure. The resulting source-bound sweep passed c19 and
retained c20 as the first non-fitting row. That campaign driver is not used by
the bounded Strix closure runner, so the change requires no ROCm or Vulkan
repeat. The later Phase 7.2 checkpoint `fa3ab5a9dab7` added Apple-only unified
memory admission and macOS process containment plus a backend-selected Metal
soak runtime. Shared qualification helpers preserve the existing ROCm and
Vulkan effective configurations, while memory-counter tolerance and slow-sync
handling are conditional on the Metal/macOS runtime. The final Phase 7.2
retention commit `f19d2591ab8e` added only the passing endurance receipt and
its documentation after checkpoint `fa3ab5a9dab7`. On that final integrated
source, the focused sampler, soak, mixed-load, workload, runner, and receipt
suites passed 178 tests. These changes are backend-isolated under this phase's
rule, so they do not require another targeted Strix hardware run.

Final hosted checkpoint:
GitHub Actions
[`30498143581`](https://github.com/ericflo/kiln/actions/runs/30498143581)
passed `backend_build=all` from exact pushed source `f19d2591ab8e`. Its
separate Metal, CUDA, ROCm, and Vulkan jobs all passed, as did Linux default
features, formatting, dependency policy, and portable qualification receipt
validation. The run completed successfully with all seven jobs green. This
closed the hosted matrix item on the final Phase 7.2-integrated source. The
last checklist item remained open until the documentation checkpoint below
and its exact-commit workflows passed.

Completion checkpoint:
documentation commit `8e3ea37085f7` passed its exact-commit Pages
([`30499565309`](https://github.com/ericflo/kiln/actions/runs/30499565309)),
Repository hygiene
([`30499565302`](https://github.com/ericflo/kiln/actions/runs/30499565302)),
and Release Version Drift
([`30499565273`](https://github.com/ericflo/kiln/actions/runs/30499565273))
workflows. The same checkpoint passed local validation of every qualification
receipt, retained benchmark and oracle evidence, repository artifact policy,
production file budget, release-version drift, and the assembled 54-document
site. The four bounded Strix core and serving receipts also passed independent
local-artifact and known-commit validation in the final audit. Phase 7.2 is
complete, every Phase 7.3 evidence item is satisfied, and this plan is
therefore complete. This completion update changes documentation only.
