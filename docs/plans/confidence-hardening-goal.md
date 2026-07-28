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
- [ ] Retain a passing environment receipt from clean pushed source proving the
  complete boundary on this WSL2 host.

## Phase 7.1: RTX 4090 Laptop GPU, 16 GB

- [ ] Retain the current-source CUDA correctness/oracle subset that fits the
  device. The independent vLLM exact-output defect was fixed at `d0f74ecb4`;
  the qualification receipt still must bind the current source and platform
  boundary.
- [ ] Retain low-memory admission, allocation failure, reclaim, and process
  cleanup evidence.
- [ ] Run serving performance at every concurrency that fits without changing
  workload semantics.
- [ ] Run an eight-hour mixed-load soak using ordinary monotonic elapsed time.
- [ ] Validate and check in receipts, update this checklist and permanent
  documentation, commit, and push.

## Next Backend

Metal qualification follows this CUDA/WSL2 checkpoint. No Metal claim is made
by the work above.
