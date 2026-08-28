# Concurrent batched decode bench — 2026-05-26 (#1082)

Production headline measurement for the #1082 DoD bullet
"`decode bs=64 ≥ vLLM at same shape on L40S`". Captured via
`scripts/bench-concurrent-batch.py` against a live `kiln serve` HTTP
server on RunPod A6000.

This doc is the canonical record of what was measured — replaces
`kiln-bench`-based numbers that turned out to measure sequential
single-prompt decode at bs=1, not concurrent batched decode (the
distinction surfaced during the Phase 5 default-on regression
investigation, see `cuda-graph-status.md`).

## Bench config

- Model: `Qwen3.5-4B` W4A16 paged decode
- GPU: NVIDIA RTX A6000 (Ampere, 48 GB)
- Build: `cargo build --release --features cuda --bin kiln`
  - `KILN_CUDA_ARCHS=86`
  - `SCCACHE_RECACHE=1` + `cargo clean -p kiln-gdn-kernel` first
    (sccache CUDA dlink stale-cache workaround, see
    `feedback-sccache-cuda-dlink-stale` memory note)
- Serve: `./target/release/kiln --model-path /workspace/Qwen3.5-4B
  --port 8420`
  - `KILN_W4A16=1`
  - `KILN_CUDA_GRAPHS=true` (bs=1 graph capture; default-on, healthy)
- Bench: `python3 scripts/bench-concurrent-batch.py
  --sizes 1,2,4,8,16,32,64 --max-tokens 128 --mode concurrent
  --warmup`

## Headline (HEAD `2d9d4fc4` — GDN-decode contiguity fix shipped)

Eager batched path (`KILN_CUDA_GRAPHS_BATCHED=0` —
production default after the Phase 5 revert at `909e2e61`):

| concurrency | tok/s | per-request OK |
|---|---|---|
|  1 |  84.19 | 1/1   |
|  2 | 144.05 | 2/2   |
|  4 | 264.26 | 4/4   |
|  8 | 449.36 | 8/8   |
| 16 | 475.47 | 16/16 |
| 32 | 482.85 | 32/32 |
| 64 | **498.34** | 64/64 |

**Scales 5.9× from bs=1 → bs=64.** Throughput climbs steeply up to
bs=8 (5.3× from bs=1), then plateaus at ~480-498 tok/s as compute
becomes the bottleneck.

## Phase 5 captured-graph path (BROKEN, opt-in only)

`KILN_CUDA_GRAPHS_BATCHED=1 KILN_CUDA_GRAPHS_BATCHED_KV_FUSED=1`:

| concurrency | tok/s | per-request OK |
|---|---|---|
|  1 |  83.28 | 1/1   |
|  2 |   0    | 0/2 (HTTP 500) |
|  4 |   0    | 0/4 (HTTP 500) |
|  8 |   0    | 0/8 (HTTP 500) |
| 16 |   0    | 0/16 (HTTP 500) |
| 32 |   0    | 0/32 (HTTP 500) |
| 64 |   0    | 0/64 (HTTP 500) |

Root cause not fully diagnosed yet (see `cuda-graph-status.md`
"Headline (2026-05-26 — REVERTED)" section for the analysis trail).
The captured-graph path failure does NOT affect production after
the `909e2e61` revert.

## Bug fix landed in same session

Commit `2d9d4fc4` fixed a separate non-contig view bug in three
`gdn_decode_*` kt paths (`cuda.rs:~1130`, `cuda.rs:~1325`,
`cuda.rs:~1495`). Before the fix, every concurrent request ≥2
returned HTTP 500 with `"gdn_decode_rmsnorm a → kt failed: tensor
must be contiguous"` because `a`/`b` arrive as
`ab.narrow(2, .., nv)` strided views in the batched code path. The
fix mirrors the existing `.contiguous()` pattern in the sibling
`gdn_gates` kt path (`cuda.rs:~1574`).

After the fix, both code paths (captured-graph and eager-batched)
get past the contig hazard — eager-batched is healthy, captured-
graph fails on the next downstream issue (the swallowed
graph-capture inner error documented in `cuda-graph-status.md`).

## Comparison vs vLLM L40S DoD target

The original `#1082` DoD bullet expects `decode bs=64 ≥ vLLM at
same shape on L40S`. vLLM published baseline: **1907 tok/s on L40S
at bs=64**. L40S is ≈1.5× faster than A6000 for this shape, so the
back-calculated A6000 target is **~1271 tok/s**.

Today's A6000 measurement: **498 tok/s @ bs=64**. That's ~39% of
the back-calculated A6000 target.

Gap drivers (rough split, to be refined by profiling):
- Phase 5 captured-graph optimization OFF (capture-bug-blocked) —
  expected to recover meaningful tokens-per-second when the
  swallowed-inner-error follow-up lands.
- Throughput plateau at bs=8 suggests a per-step or per-token
  serialization point — candidate sites: KV cache writer
  contention, scheduler step granularity, GPU SM occupancy at
  small per-row work. To be re-profiled.
- Cross-GPU normalization: L40S→A6000 conversion assumes ~1.5×
  proportional; measuring on L40S directly would tighten the
  comparison.

## Reproducibility notes

The pod-time variance on serve-build + bench is high enough that
single-run numbers should not be over-interpreted. Re-running the
bench with three repeats and taking median is the right next step
when the captured-graph bug fix lands.
