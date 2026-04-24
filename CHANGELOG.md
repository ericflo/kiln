# Kiln Server Changelog

## kiln-v0.2.1 — 2026-04-24

Server re-cut to include CI fixes that were missing from kiln-v0.2.0. No
user-facing behavior changes from v0.2.0 in the core server; this cut also
picks up phase-6 Metal and CUDA kernel work landed on main between v0.2.0 and
v0.2.1.

### CI fixes shipped for the full platform matrix
- Bump Jimver/cuda-toolkit from v0.2.19 to v0.2.35 to handle NVIDIA's renamed
  installer URLs (#469)
- Install MSVC dev env on Windows before CUDA build; fixes M_LOG2E undefined
  in flash_api_c.cu under MSVC (#472)
- Force static MSVC CRT on Windows CUDA build; fixes CRT mismatch between
  esaxx-rs and kiln-marlin-gemm (#477)

### Phase 6 CUDA decode work
- Fuse CUDA GDN gated RMSNorm (#466)
- Add CUDA conv1d prefill fast path and fix conv1d prefill launch bounds
  (#481)
- Document post-466 and post-468 MTP decode profiles and post-476 MTP profile
  failure (#468, #480)
- Audit GDN conv decode hotspot and refresh post-#481 current-main profile
  (#473, #483)

### Phase 6 Metal decode work
- Fuse Metal LM-head argmax for greedy decode and reduce Metal LM-head argmax
  on GPU (#471)
- Speed up Metal decode GEMV and route GDN out-proj through Metal decode GEMV
- Fuse Metal full-attention QKV projections and fuse Metal GDN decode QKV
  conv norm
- Persist transposed weight cache asynchronously

### Infrastructure
- Cap cargo and nvcc parallelism and add OOM postmortem helper for RunPod
  builds (#474)

## kiln-v0.2.0 — 2026-04-24

Coordinated release aligned with desktop-v0.2.0. Headline work is the Metal
decode path for Apple Silicon: a new fused GDN kernel family, MTP speculative
decoding improvements, and a batch of macOS startup and prefill reductions.

### Metal GDN kernel fusion
- Fuse Metal GDN decode input projections
- Fuse Metal GDN chunk prep
- Fuse Metal RoPE for prefill (#418)
- Fuse Metal GQA qk norm expansion (#393)
- Default Metal MLP gate-up fusion (#447)
- Add Metal full-chunk GDN prefill (#394), head-last layout (#395)
- Speed up Metal GDN recurrent prefill (#398) and avoid zeroing recurrent outputs (#419)
- Use direct GDN chunk slices on Metal (#391); read full chunks from strided views (#455)
- Use unexpanded GDN QK for Metal decode (#456)
- Use head-major KV read for Metal decode (#452) and head-major SDPA for paged decode (#342)
- Use uninitialized Metal outputs for full-write kernels (#449)
- Parallelize Metal conv1d prefill

### MTP (multi-token prediction) decode
- Route MTP prefill through Metal streaming (#400)
- Speed up macOS default MTP decode
- Mirror desktop speculative routing in bench (#404)
- Align skip-layer bench draft state (#410)
- Defer MTP upload and trim draft state (#408)
- Avoid native MTP during Metal prewarm (#402)
- Guard non-streaming MTP final window (#454)
- Raise Metal skip-layer crossover to 4096 (#442)
- Route long macOS decode through paged skip-layer

### macOS startup and prefill
- Reduce macOS startup and KV prefill overhead
- Speed up macOS startup and skip-layer prefill
- Improve macOS startup and short-prompt routing (#440) and speculative routing (#437)
- Defer Metal precompile until background prewarm (#453); precompile Metal kernels during startup
- Move tokenizer warmup after listen (#460)
- Prewarm macOS speculative path (#386) and make prewarm opportunistic
- Tune macOS Metal hot paths (#385) and streaming prefill defaults (#377)
- Enable tiled prefill by default on Metal (#367)
- Route Metal prefix attention through head-major SDPA (#366)
- Optimize Metal paged KV prefill reads (#416)
- Speed up Metal LM head decode
- Gate Metal readiness prewarm (#332)
- Harden Metal prewarm and KV auto-sizing
- Drop redundant Metal embedding upload (#335)
- Batch Metal auxiliary weight uploads (#443); stream transposed weight cache reads (#445); mmap transposed weight cache hits (#461)

### Server runtime
- Keep default GPU sampling on device (#336); speed up default sampling and fix speculative KV advance (#328)
- Avoid zero-filling server KV pools (#337)
- Hoist paged decode debug gates (#333)

### Profiling and phase 6 kernel work
- Extensive phase 6 decode profiling work (C35–C50) and MTP α-stability re-benches documented in PROFILING.md and PROFILING-MTP-*.md

## kiln-v0.1.2 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.1 — 2026-04-20
- See the GitHub release for details.

## kiln-v0.1.0 — 2026-04-18
- Initial public release.
