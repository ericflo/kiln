# Confidence Hardening Goal

**Status:** Active.

## Scope

Kiln product behavior must remain portable across supported hosts and
accelerators. The RTX 4090 Laptop under WSL2 is the current CUDA validation
host, not a product target or a source of runtime defaults.

- Hardware identity and measurements describe only the machine that produced
  them.
- Backend behavior is selected from runtime capabilities, with portable
  fallbacks where the backend supports them.
- Qualification uses ordinary monotonic time. It does not pause work based on
  temperature or impose host-specific pacing thresholds.
- Host and GPU temperatures are read-only platform observations. Missing
  required observability is reported explicitly; readings do not alter
  execution.
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

## Next Backend

Metal qualification follows this CUDA/WSL2 checkpoint. No Metal claim is made
by the work above.
