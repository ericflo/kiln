# Phase 6 Post-#502 GDN Full-Chunk Audit

Date: 2026-04-24

## Scope

Audit whether the freshly merged post-PR-#500 profile in PR #502 leaves one
bounded CUDA improvement to port into Kiln's vendored GDN full-chunk prefill
kernel.

The requested target was intentionally narrow:

- bf16 activations
- F32 accumulators only where the existing kernel already requires them
- Qwen3.5-4B envelope (`chunk_size=64`, `dk<=128`, `dv<=128`)
- forward path only
- no Triton runtime
- no broad re-vendor or call-envelope rewrite

## Remaining-Work Preflight

Result: stop with a documentation-only PR.

- `gh pr list -R ericflo/kiln --limit 10` returned no open PRs, so no open PR
  was already editing `crates/kiln-gdn-kernel/csrc/gdn_full_chunk_forward.cu`
  or the same GDN full-chunk/recurrent path.
- `PROFILING.md` on `origin/main` includes the post-PR-#500 / PR #502 section
  at Kiln commit `76ea9523912312d3ea45f14ab2b6d9646605a48e`. It shows prompt-heavy
  prefill's top GPU-kernel bucket is `gdn_full_chunk_forward_kernel` at
  **27.9%**, followed by `ucopy_bf16` at **22.3%**.
- The current CUDA file still exists and remains the full-chunk prefill path,
  but the upstream audit below found no bounded remaining port inside this
  task's envelope.

## Upstream Sources Audited

- `fla-org/flash-linear-attention` commit
  `101240f396a6b53e452defb371e3d6e98211535a`
- `fla/ops/gated_delta_rule/chunk.py`
- `fla/ops/gated_delta_rule/chunk_fwd.py`
  (`chunk_gated_delta_rule_fwd_intra`,
  `chunk_gated_delta_rule_fwd_kkt_solve_kernel`)
- `fla/ops/gated_delta_rule/wy_fast.py` (`recompute_w_u_fwd`)
- `vllm-project/vllm` commit
  `9f771b3ab92d26a7d91a8255572c5d8d2b3ad601`
- `vllm/model_executor/layers/fla/ops/chunk.py`
- `vllm/model_executor/layers/fla/ops/chunk_delta_h.py`
  (`chunk_gated_delta_rule_fwd_kernel_h_blockdim64`)
- `vllm/model_executor/layers/fla/ops/fused_recurrent.py`
  (`fused_recurrent_gated_delta_rule_fwd_kernel`,
  `fused_recurrent_gated_delta_rule_packed_decode_kernel`)

## Audit Result

No minimal code change is appropriate after PR #502.

The newer upstream implementations still win through a structural ownership
model that Kiln's current full-chunk kernel deliberately does not use:

- FLA/vLLM split the chunk path into WY-style intermediate preparation plus a
  tiled state/update kernel.
- vLLM's `chunk_gated_delta_rule_fwd_kernel_h_blockdim64` owns FP32 recurrent
  state tiles in registers (`[BV, 64]` slices over K/V) and updates them with
  Triton dot products after loading key/value tiles once per chunk.
- Kiln's `gdn_full_chunk_forward_kernel` already consumes the cuBLAS-produced
  `kkt`, `qkt`, `ks_entry`, and `q_s` tensors, then performs scalar value-lane
  forward substitution, output accumulation, and BF16 state epilogue inside one
  fixed C ABI.

Porting the remaining upstream advantage would require changing the kernel's
state ownership and likely moving or replacing the existing cuBLAS-side chunk
matmul boundary. That is a full re-vendor/rewrite, not a bounded patch to
`gdn_full_chunk_forward.cu`.

## Previously Consumed Bounded Ideas

The apparent small slices from the upstream design have already been tried or
invalidated in current `PROFILING.md` history:

- post-#397: removing recurrent-state BF16 scalar round trips failed parity.
- post-#399: hoisting decay weighting into shared `W` rows passed parity but
  slowed prompt-heavy prefill.
- post-#401: staging shared `k_t` rows passed parity but slowed prompt-heavy
  prefill.
- post-#403: triangular front-half packing was a warm-pod artifact and did not
  produce a durable win.
- post-#406 / #411: tiled recurrent-state epilogue passed parity but was
  **1.8% slower** than warmed `main`.
- post-#442 / #446: vLLM full-chunk audit reached the same structural-only
  conclusion and left no bounded micro-port.

PR #502 refreshed the profile numbers, but it did not change this code frontier:
`gdn_full_chunk_forward_kernel` is still the top prompt-heavy kernel bucket, yet
upstream's remaining improvement is larger than this task's safe edit envelope.

## Validation

No RunPod was launched and no CUDA build was run because the remaining-work
precondition failed before editing source code. Local validation was limited to
`git diff --check` for the documentation change.

## Next Step

Do not queue another minimal GDN full-chunk micro-port against this kernel. The
next Phase 6 task should either:

1. declare and scope the larger structural re-vendor of the GDN chunk/state path,
   including a new call envelope and benchmark gate; or
2. re-profile current `main` and choose a different hotspot with an unconsumed,
   bounded implementation path.
