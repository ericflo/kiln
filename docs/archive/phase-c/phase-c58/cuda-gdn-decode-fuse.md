# Phase C58 CUDA GDN Decode Fusion

Date: 2026-04-24
Pod: `mfk88l8i8tab02` (`NVIDIA RTX A6000`, `ghcr.io/ericflo/kiln-runpod:latest`)

## Scope

PR #498 adds a narrow opt-in CUDA C-ABI entry point for native-MTP `seq_len == 1` GDN decode that computes gates and advances the recurrent state in one launch for bf16 `dk == dv == 128` tensors. The hook is wired through `BackendRuntime::gdn_decode_gates_recurrent` and the CUDA backend.

The production path remains disabled by default and is only selected when `KILN_ENABLE_FUSED_GDN_DECODE=1` is set.

## Parity Resolution

The reported output mismatch was caused by the parity probe using `Tensor::clone()` for mutable recurrent state. Candle tensor clones share storage, so the split path advanced the same state storage before the fused path ran. The fused kernel was then compared from an already-mutated input state, which produced a recurrent output mismatch while making the final state appear identical.

The restored GPU parity test now deep-copies state with `Tensor::copy()` before each arm and proves exact parity against the existing split CUDA path:

```text
cuda decode gates+recurrent vs split: out max=0e0 mean=0e0, state max=0e0 mean=0e0
```

The fused hook only covers gates + recurrent state/output. Gated RMSNorm remains on the existing standalone kernel, per the earlier combined-RMSNorm fusion risk.

## Validation

RunPod A6000 validation passed:

```bash
cargo test -p kiln-gdn-kernel --release -- --nocapture
KILN_ENABLE_FUSED_GDN_DECODE=1 cargo test -p kiln-model --release --features cuda gdn_decode -- --nocapture
cargo build --release --features cuda,nvtx --bin kiln-bench
```

## Benchmark Result

Focused A6000 decode benchmark command:

```bash
./target/release/kiln-bench --model-path /workspace/qwen3.5-4b --paged --prompt-tokens 512 --max-output-tokens 128 --skip-training --prompt-subset humaneval --chat-template --latency-only --temperature 0.0 --seed 1
```

Results:

| Path | Mean ITL | Decode tok/s |
| --- | ---: | ---: |
| Default | 22.493 ms | 44.458 |
| `KILN_ENABLE_FUSED_GDN_DECODE=1` | 22.501 ms | 44.442 |

The measured decode delta is effectively parity and below the 5% enable-by-default floor. Leave `KILN_ENABLE_FUSED_GDN_DECODE` opt-in and pivot to `:kiln/gdn/qk_norm` or `:kiln/attn/rope` rather than retrying this fusion shape.
