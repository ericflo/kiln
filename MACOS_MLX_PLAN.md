# macOS Support via candle-metal + MLX

Tracking document for porting Kiln to macOS / Apple Silicon. Replaces the "CUDA-only" caveats in README and BENCHMARKS once Phase 7 ships.

## Strategic frame

**One source tree, two backends — landed in two waves.**

1. **Wave 1 — candle-metal.** Our `kiln-model` forward pass already uses `candle_core::Tensor` throughout. Candle ships a Metal backend. Swap device, port the two custom CUDA kernels (`kiln-flash-attn`, `kiln-gdn-kernel`) to Metal equivalents, and we have a working macOS build with a fraction of the refactor cost. Expected perf: 30-50 % of MLX peak, but ships in weeks not months.

2. **Wave 2 — mlx-rs.** Once the backend trait is in place, add an MLX implementation as a second peer backend. Peak Apple Silicon performance (MLX's `fast::scaled_dot_product_attention`, native MX INT4 quantization, Apple's aggressive kernel tuning). Users who care pick their platform's best backend.

**Not a fork.** Build-time selection via Cargo features:

| Flag | Backend | Platform |
|---|---|---|
| `--features cuda` | candle CUDA + kiln-flash-attn + kiln-gdn-kernel | Linux / Windows + NVIDIA |
| `--features metal` | candle Metal + Metal kernel ports | macOS + Apple Silicon |
| `--features mlx` (Wave 2) | mlx-rs native | macOS + Apple Silicon |
| no feature | CPU-only mock mode | Any (for tests) |

Scheduler, block manager, prefix cache, request state, training queue, HTTP API, tokenizer, CLI, and config stay unchanged — they're already backend-agnostic.

## Phases

### Phase 0 — Feasibility & baselines (days)

- [x] Create this document, branch, draft PR.
- [ ] Get `cargo build` (no CUDA feature) passing on macOS.
- [ ] Mock-mode server launches on macOS; existing integration tests pass.
- [ ] Investigate current state of `candle-metal` for the ops we use (matmul, softmax, rms_norm surrogate, gather/scatter). Document any gaps.
- [ ] Tiny standalone spike: load Qwen3.5-4B safetensors via candle on Metal, run one forward pass on a 32-token prompt, confirm numerics reasonable vs a CUDA reference output. No kiln integration yet.

### Phase 1 — Backend trait refactor (CUDA-only, no behavior change)

Push all backend-specific operations in `kiln-model` behind a `BackendRuntime` trait. The candle-core tensor type stays universal — we only abstract the **non-candle ops**: FlashAttention forward, FlashAttention paged decode, FlashAttention backward, GDN recurrent step, GDN forward substitution, CUDA graph capture/replay, NVTX ranges, device detection.

Deliverables:
- New `kiln-model/src/backend/mod.rs` with the trait definition.
- `kiln-model/src/backend/cuda.rs` implementing the trait against the existing CUDA crates.
- All call sites in `forward.rs`, `generate.rs`, `kv_cache.rs`, `paged_kv_cache.rs`, `cuda_graph.rs`, `speculative.rs`, `kiln-train/src/trainer.rs` route through the trait.
- Zero regression on `kiln-bench --paged` (same tok/s within noise).
- Existing real-model integration test passes unchanged.

### Phase 2 — Metal custom kernels

Port the two vendored CUDA kernels to Metal. Land behind `--features metal`.

- `kiln-flash-attn` → new Metal shader implementing BF16 + hdim {128, 256} + causal. Start with non-paged forward; paged decode next; backward (needed for training) last.
- `kiln-gdn-kernel` → Metal shader for the fused recurrent step (decay + delta + state update + output) and chunkwise forward substitution. Fall back to candle ops outside the supported envelope (same policy as the CUDA path).

Either use Metal Shading Language via the `metal` crate + MSL shaders, or use candle's `Metal` device + `MetalKernels` where we can compose out of existing primitives. Prefer composition first, custom kernel only when profiling says so.

Deliverables:
- Metal implementations of `BackendRuntime`.
- `kiln-model/src/backend/metal.rs`.
- Parity tests: same prompt → same generated tokens on CUDA and Metal (within sampling tolerance).
- Single-stream decode target: ≥ 40 tok/s on M3 Max for Qwen3.5-4B BF16.

### Phase 3 — macOS paged KV cache + LoRA

- Paged scatter/gather via candle Metal ops (or a Metal kernel if the primitives path is too slow).
- LoRA delta application on Metal (same `linear_with_lora_t` pattern).
- Two-phase `ensure_adapter` pattern verified under unified memory (likely simpler, no HtoD).
- Chunked prefill + continuous batching working on macOS.

### Phase 4 — Unified memory & VRAM budgeting

- `kiln-core/src/vram.rs` grows an Apple Silicon path: unified memory size via `sysctl hw.memsize`, plus MLX/Metal active-allocation query for the runtime budget.
- `GpuMemoryBudget` logic in `kiln-server/src/state.rs` extended — unified memory means the "model weights + KV cache + training" split still applies but the host/device boundary is gone.
- Default block counts and gradient checkpoint segments tuned for M-series chips (8 / 16 / 32 / 64 / 128 GB unified memory tiers).

### Phase 5 — Training on macOS

- SFT end-to-end on Metal backend.
- GRPO end-to-end on Metal backend.
- Gradient checkpointing validated — segment boundaries and recomputation unchanged, just the underlying kernels swap.
- Expected throughput: 1/3 to 1/2 of CUDA on equivalent price point, but on-laptop training is a compelling story on its own.

### Phase 6 — Quantization

- Candle Metal supports INT8 / INT4 weight loading; wire that in.
- FP16 KV cache native on Metal (halves memory vs BF16 CUDA baseline anyway).
- FP8 E4M3 KV cache: defer until users hit context-length walls. No native Metal FP8; would emulate via uint8 bit-pack (current approach).
- GPTQ INT4 loader: existing path dequantizes to BF16 on CPU. Works unchanged. Future: direct GPU-native INT4 Metal kernel.

### Phase 7 — Desktop app for macOS

The Tauri workspace under `desktop/` is currently configured Windows + Linux only. Add macOS.

- `tauri.conf.json`: add macOS target, bundle identifier, category.
- Menu-bar integration via `tray-icon` crate (replaces/augments the system-tray path for Win/Linux).
- `.dmg` installer via `cargo tauri build --target universal-apple-darwin` — but **arm64 only** since Metal/MLX require Apple Silicon (document this, reject x86_64 macs at install time or pre-flight).
- Apple Developer code signing + notarization for Gatekeeper.
- Update `desktop/README.md` with macOS build & dev instructions.
- Add macOS installer link to the main README.md download table.
- GitHub Actions: add `macos-14` to the `desktop-build.yml` matrix.

### Phase 8 — Wave 2: mlx-rs backend (follow-up)

Peak Apple Silicon perf via Apple's MLX framework. Lives alongside
`MetalBackend`: `--features metal` alone uses candle-metal (no-Xcode
default); `--features mlx` layers MLX's fused attention primitive on
top, keeping candle-metal for everything else.

**Build prerequisites for `--features mlx`:**

1. Full Xcode (not just Command Line Tools).
2. Accepted license: `sudo xcodebuild -license accept`.
3. On-demand Metal Toolchain component (Xcode 16+ split this out):
   `xcodebuild -downloadComponent MetalToolchain` (~700 MB download).
4. CMake ≥ 3.25 (mlx-sys's vendored MLX source requires it).

`--features metal` uses candle-metal's runtime MSL JIT and works with
CLT alone — that stays the default Apple Silicon path.

**Testing:** MLX and candle-metal both grab the default Metal device;
concurrent tests can SIGSEGV. Run with `--test-threads=1` when
exercising `--features mlx`. CI already does this.

**Checklist:**

- [x] Add `MlxBackend` in `kiln-model/src/backend/mlx.rs` implementing
  `BackendRuntime` via `mlx-rs`.
- [x] `mlx_rs::fast::scaled_dot_product_attention` for both prefill and
  paged decode, with stride-aware candle↔mlx tensor conversion
  (MLX's fused SDPA returns non-contiguous arrays that `Array::as_slice`
  doesn't handle correctly without the fix).
- [x] Parity: MLX SDPA matches a naive F32 reference at machine
  precision (0.00000 diff) for both causal and non-causal.
- [x] Keep candle-metal as a selectable backend.
- [ ] Port weights loading to MLX arrays for zero-copy — currently
  tensors round-trip through host memory at the candle↔mlx boundary.
- [ ] Use MLX's native MX INT4 format for weights (convert from GPTQ
  at load).
- [ ] `mx.compile` on the decode step as MLX's analog to CUDA graphs.
- [ ] Benchmark head-to-head vs candle-metal; document in BENCHMARKS.md.

### Phase 9 — CI, benchmarks, docs

- GitHub Actions `macos-14` runners build + test (mock mode + small real-model integration test).
- `kiln-bench` extended to report which backend it's running.
- Head-to-head: Kiln-metal vs Kiln-mlx vs llama.cpp-metal vs MLX-LM on M3 Max. New BENCHMARKS.md section.
- README.md: remove "CUDA-only" caveats, add macOS quickstart.
- QUICKSTART.md: `brew install` path for dependencies, `cargo build --features metal` flow.
- ARCHITECTURE.md: backend trait section documenting the abstraction.

## Hard problems (flagged upfront)

1. **CUDA graphs have no Metal analog.** Metal command buffers can be reused but don't quite match CUDA-graph semantics. MLX's `mx.compile` (Wave 2) may be a better fit. For Metal in Wave 1, measure before optimizing — decode-step kernel-launch overhead may not dominate on M-series the way it does on CUDA.

2. **Pre-transposed weights (PR #128).** +4.9 GB VRAM was acceptable on CUDA; on unified memory it's different accounting. Measure whether MLX / candle-metal matmul benefits from pre-transposed A. If not, skip the optimization on the Metal path — unified memory is precious.

3. **FlashAttention on Metal.** Hugging Face's `candle-flash-attn` is CUDA-only. Options: use candle's native `scaled_dot_product_attention` on Metal (exists but less optimized), or vendor a Metal FlashAttention port (more work, better perf). Likely ship with candle SDPA first, profile, decide.

4. **GDN recurrent kernel on Metal.** Our CUDA version fuses decay + delta + state update + output. Composing these out of candle primitives will work but won't fuse — likely a bottleneck on the 24 GDN layers of every forward pass. Will need a real Metal kernel here before the perf numbers get respectable.

5. **Unified memory lock discipline.** `GpuCoordinationLock` still works semantically (inference reads, training writes), but there's no HtoD/DtoH cost to amortize. Simpler, but the cost model for "how long does training block inference?" shifts — may need to revisit the gradient checkpointing segment sizing.

6. **candle-metal maturity.** Some ops in candle-metal fall back to CPU or aren't implemented. Phase 0 needs to enumerate the gap concretely per-op. Most risky ops: any custom reshape/transpose patterns, gather/scatter for paged KV, FP8 emulation. We'll likely contribute upstream patches to `candle` or vendor small pieces.

7. **mlx-rs maturity (Wave 2 concern).** The crate is less mature than candle. Its autodiff may not handle the full forward pass shape without modification. Scope Wave 2 work cautiously.

## What stays untouched

- `kiln-core` (block manager, prefix cache, request lifecycle, config, tokenizer).
- `kiln-scheduler` (chunked prefill priority order, prefix-cache integration).
- `kiln-server/src/api/*` (HTTP routing, SSE streaming, two-phase ensure_adapter).
- `kiln-server/src/training_queue.rs` (FIFO worker).
- `kiln-train/src/trainer.rs` structure (SFT / GRPO loops, gradient checkpointing segmentation — only the forward/backward ops swap).
- TOML config schema, CLI subcommands, Prometheus metrics, error types.
- Request state machine, sampling parameters.

The whole upper half of the stack is backend-agnostic — the point of Phase 1 is to make the rest of it backend-agnostic too.

## Non-goals

- **x86_64 macOS support.** MLX and effective Metal perf need Apple Silicon. Don't ship an x86 Mac binary — it's strictly worse than running Linux in a VM.
- **Rosetta translation.** Same reason.
- **Removing the CUDA path.** Wave 1's success criterion is "no regression on CUDA." Keep both code paths first-class.
- **Abstracting over tensor type.** We keep `candle_core::Tensor` universal across CUDA + Metal. MLX in Wave 2 gets its own tensor type inside its backend impl but surfaces candle tensors at the trait boundary where practical, or converts at the seam.
- **Multi-tenancy improvements bundled with this.** Stay scoped: macOS port only.

## Success criteria

To call this "done" and merge:

1. `kiln serve` runs on M-series Mac serving Qwen3.5-4B via the Metal backend, same HTTP API, same config schema. No regression on the CUDA path.
2. Single-stream decode ≥ 40 tok/s on M3 Max for Qwen3.5-4B BF16.
3. SFT + GRPO training both work end-to-end on Metal; loss curves match CUDA reference within numerical tolerance.
4. Signed + notarized `.dmg` for Kiln Desktop.
5. `cargo test --workspace` passes on `macos-14` GitHub Actions runners.
6. BENCHMARKS.md extended with macOS numbers vs llama.cpp-metal.
7. All README "CUDA-only" caveats removed; QUICKSTART has a macOS path.

Wave 2 (mlx-rs) lands in a separate PR after this one merges.

## Working log

- **2026-04-18** — Branch created, plan written. Starting Phase 0 exploration.
