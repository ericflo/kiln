# Confidence Hardening Goal

**Status:** Active.

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
- [ ] Run serving correctness and performance at every concurrency that fits
  without changing workload semantics, and retain the first non-fitting
  capacity boundary as counterevidence.
- [ ] Run an eight-hour mixed-load soak using ordinary monotonic elapsed time.
- [ ] Validate and check in receipts, update this checklist and permanent
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

## Phase 7.3: Final Cross-Backend Regression Closure

- [ ] From one current source commit, dispatch GitHub Actions CI with
  `backend_build=all` and require its CUDA, Metal, ROCm, and Vulkan lanes to
  pass on their separate compatible hosted runners. This is compile and test
  evidence, not a requirement that one qualification host run every backend.
- [x] On Strix Halo, retain current-source ROCm and Vulkan core-correctness
  receipts.
- [x] Retain targeted ROCm and Vulkan serving receipts covering exact output,
  KV-growth pressure, prefix-cache reclamation, and bounded process and device
  cleanup after the shared server changes made during Phase 7.1.
- [ ] After Phase 7.2 lands, incorporate its final checkpoint and repeat only
  the targeted checks affected by later shared runtime or qualification-tooling
  changes. Documentation-only and backend-isolated changes do not require
  unrelated hardware reruns.
- [ ] Validate and check in the receipts, update this checklist and permanent
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
the pending correctness-only hardware closure. Each backend builds current
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
suffix as structured failure. That driver is not used by the bounded Strix
closure runner, so the change requires no ROCm or Vulkan repeat. Phase 7.2's
required first non-fitting capacity boundary, soak, and final documentation
items remain open.

Interim hosted checkpoint:
GitHub Actions
[`30416196655`](https://github.com/ericflo/kiln/actions/runs/30416196655)
passed `backend_build=all` from exact pushed source `56000a370874`. Its
separate Metal, CUDA, ROCm, and Vulkan jobs all passed, as did Linux default
features, formatting, dependency policy, and portable receipt validation.
This records a green hosted baseline without closing the first checklist item:
Phase 7.2 is still active, so the same matrix remains required from the final
integrated source.
