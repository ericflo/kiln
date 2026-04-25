# Benchmarks

Inference performance for **kiln** on Qwen3.5-4B against external references on
the same single-GPU class. Headline numbers track current `main`; older runs
are kept as historical context where the methodology was the same.

## Setup

| Component | Value |
|---|---|
| GPU | NVIDIA RTX A6000 (48 GB, compute capability 8.6) |
| Driver | 550.127.08 |
| CUDA | 12.4 |
| Host | RunPod on-demand pod (`ghcr.io/ericflo/kiln-runpod:latest` image) |
| Model | Qwen3.5-4B (~4.2B params, 32 layers, hybrid GDN + GQA) |
| Weights | bf16 base, W4A16 Marlin-packed MLP projections at runtime |
| Prompt length | 512 tokens (humaneval subset, ~494 after tokenization) |
| Output length | 128 tokens |
| kiln commit | `c5cf77d` (post-PR #535, captured by PR #536) |
| kiln features | `--features cuda`, `KILN_CUDA_ARCHS=86` |
| kiln env | `KILN_W4A16=1`, `KILN_CUDA_GRAPHS=true` |
| kiln binary | `kiln-bench` |

The current single-stream protocol is `--paged --prompt-tokens 512
--max-output-tokens 128 --skip-training --prompt-subset humaneval
--chat-template --latency-only --temperature 0.0 --seed N` with three fresh
processes. Median-of-3 governs.

## Results

### Single-stream throughput (kiln, post-PR #536, A6000)

512 prompt → 128 generated, MTP-Off, three seeds, median-of-3:

| Metric | Value |
|---|---:|
| Decode (tok/s, median) | **44.75** |
| Decode (tok/s, range) | 44.23 – 45.01 (Δ = 1.8 %) |
| Prefill latency (ms, median) | 355.4 |
| Prefill (tok/s) | ~1390 |
| Mean ITL (ms) | 22.35 |
| P50 ITL (ms) | 22.33 |
| P99 ITL (ms) | 27.33 |
| Peak VRAM | ~10 GB |

Source: `docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv` (PR #536), with the
hotspot mix and post-#534 comparison in `docs/archive/phase-c/phase-c65/post-534-profile.md`
(PR #535).

### Multi-engine comparison (Qwen3.5-4B + native MTP, A6000 sm_86)

Phase 7 audited the obvious external references against kiln on the same GPU
class. As of this refresh, no other Qwen3.5-4B + native-MTP serving stack runs
end-to-end on the kiln-runpod stock image, so a side-by-side decode tok/s
table is not currently producible. The audit results are summarized below.

| Engine | Status on A6000 sm_86 / driver 550.x | Decode (tok/s) | TTFT | Peak VRAM | Source |
|---|---|---:|---:|---:|---|
| **kiln (post-#536)** | runs | **44.75** (median) | 355 ms | ~10 GB | this doc |
| **llama.cpp `9d34231` (current main, 512 → 128)¹** | runs | **68.99** (median) | ~85 ms | ~8.8 GB | `bench-results/llama-bench-a6000-post536.json` |
| **llama.cpp `9d34231` (current main, 512 → 256)¹** | runs | **69.23** (median) | ~85 ms | ~8.8 GB | `bench-results/llama-bench-a6000-post536.json` |
| **vLLM v0.19.1** | unsupported² | — | — | — | PR #530 |
| **vLLM v0.20.0** | unsupported³ | — | — | — | PR #533 |
| **SGLang 0.5.10.post1** | unsupported⁴ | — | — | — | PR #532 |
| **HF transformers reference** | runs (α-only)⁵ | — | — | — | PR #534 |

¹ Current-main llama.cpp A6000 re-bench captured on the same kiln-runpod
A6000 / driver 550.127.08 base image as the kiln row above. llama-bench
build commit `9d34231`, Qwen3.5-4B converted to bf16 GGUF via
`convert_hf_to_gguf.py --outtype bf16`. Flags: `n_batch=2048`,
`n_ubatch=512`, `flash_attn=false` (`-fa 0`), `n_gpu_layers=99` (`-ngl 99`),
K/V cache in f16. Three repeats per shape, median-of-3 reported. Decode
tok/s is essentially flat across decode lengths on this stack (Δ = 0.35 %
between tg128 and tg256), so the headline 512 → 128 number is what the
multi-engine apples-to-apples comparison anchors on. Prefill (pp512):
5989 tok/s (median). The pre-existing RTX 6000 Ada (sm_89) row on kiln
commit `f3d5089` is now historical (see [section
below](#historical-llamacpp-head-to-head-on-rtx-6000-ada-kiln-f3d5089)).

² vLLM 0.19.1 loads the model + drafter weights but segfaults inside
spec-decode native code during runtime profiling on Qwen3.5-4B + native MTP.
70-frame native backtrace, no Python file symbol at the terminal frame.
Verdict: `vllm_mtp_unsupported`. See PR #530.

³ vLLM v0.20.0+cu129 transitively requires `torch 2.11.0+cu130`, which itself
requires NVIDIA driver ≥580 (CUDA 13.0). The kiln-runpod stock image ships
driver 550.127.08, so `torch.cuda.is_available()` returns `False` and vLLM's
V1 `EngineCore` worker crashes during CUDA init before any MTP code runs.
Verdict: `vllm_020_mtp_unsupported_dense_4b`. See PR #533.

⁴ SGLang 0.5.10.post1 + Qwen3.5-4B dense BF16 + native MTP segfaults across
three distinct serving configurations (flashinfer + CUDA graphs, flashinfer
graphs-off, triton attention graphs-off). Engine-side dispatch is correct
(`Qwen3_5ForCausalLMMTP` weights load, hybrid GDN dispatcher initializes); the
crash is downstream of model dispatch. Verdict:
`sglang_mtp_unsupported_dense_4b`. See PR #532.

⁵ The hand-rolled HuggingFace `transformers` reference loads and runs
end-to-end but is α-only (acceptance-rate microbench), not a decode tok/s
serving comparison. It is included here because it is the only external stack
that produces a usable Qwen3.5-4B + MTP signal on this GPU; with seed 0
producing a bit-for-bit accept/reject trace match against kiln's c1_attr CSVs
and median α 0.2500 vs kiln 0.3636, the verdict was `kiln_above_hf`. See
PR #534.

The headline takeaway is structural rather than numeric: as of this refresh,
**kiln is the only stack producing end-to-end Qwen3.5-4B + native MTP decode
numbers on a stock A6000 / driver 550.x base image.** Comparison tables here
will fill in as upstream stacks regain support.

### Native MTP self-spec — α below break-even at bs=1

PR #536 ran a three-seed MTP-On vs MTP-Off A/B against post-#535 main:

| Arm | Decode (tok/s, median) | α (median) | P99 ITL (ms) |
|---|---:|---:|---:|
| MTP-Off | **44.75** | — | 27.33 |
| MTP-On (forced) | 43.09 | 0.6842 | 56.78 (~2.08×) |

Median Δ = −3.7 %, verdict: `mtp_no_decode_win`. α has improved 5.5× since
the last bs=1 measurement (PR #316: α=0.124, MTP-On −25.1 % slower) but is
still below the bs=1 break-even floor of α≈0.72; one seed (α=0.778) cleared
the floor at +8.5 %, the other two (α=0.620, α=0.684) lost 9.2 % and 4.3 %
respectively. P99 ITL roughly doubles when MTP is on because rejected-draft
steps add a heavy tail. `KILN_SPEC_METHOD=mtp` therefore stays opt-in and
gated on prompt length; see PR #536 and `docs/archive/phase-c/phase-c66/` for reopen
triggers.

This is the operative reason kiln's headline decode tok/s is what it is: the
quantized + GDN-fused base path is already running close to its bs=1 ceiling
on A6000, and the most obvious next-step lever (native MTP self-spec) does
not yet pay back the verifier cost at current α.

### kiln steady-state — refresh pending

The previous version of this doc included a 1 / 4 / 8 / 16 sequential-runs
table (RTX 6000 Ada, kiln commit `f3d5089`, 512 → 256) showing throughput
flat across runs. Those numbers are pre-Phase-6 and no longer represent
current main; PRs #535 and #536 did not re-run the same configuration on
A6000. A refresh of the sequential-runs steady-state table is on the
to-measure list and will follow a future profiling run.

### Historical: llama.cpp head-to-head on RTX 6000 Ada (kiln `f3d5089`)

Kept for reference only. This was the original end-to-end head-to-head
captured on RTX 6000 Ada (sm_89). The multi-engine row anchors on the
A6000 sm_86 re-bench above; this section exists so the original numbers
remain auditable. Both sides are out of date — kiln-side decode is
~10 tok/s on `f3d5089` versus ~45 tok/s on post-#536, and llama.cpp on
A6000 sm_86 today is 68.99 tok/s at 512 → 128 / 69.23 at 512 → 256
versus the 88.43 quoted here at 512 → 256 on RTX 6000 Ada sm_89.

| Engine | Prefill (tok/s) | Decode (tok/s) | TTFT | Peak VRAM |
|---|---:|---:|---:|---:|
| llama.cpp `408225b` (RTX 6000 Ada, 512 → 256) | **8224.45** | **88.43** | ~62 ms | **8820 MB** |
| kiln `f3d5089` (pre-Phase-6, RTX 6000 Ada, 512 → 256) | 36.31 | 10.89 | 13934 ms | 10278 MB |

Raw JSON: `bench-results/kiln-bench.json` (kiln side, stale) and
`bench-results/llama-bench.json` (llama.cpp side, RTX 6000 Ada — superseded
in the multi-engine table by `bench-results/llama-bench-a6000-post536.json`).

## Interpretation

- **Decode is now ~5× faster on current main than on the original `f3d5089`
  baseline** (10.89 → 44.75 tok/s on the 512 → 128 humaneval shape with
  W4A16 + CUDA graphs). The Phase 6 fused kernels (RMSNorm, GDN gates, GDN
  qk-norm, paged decode) and the W4A16 Marlin MLP wire-in (PR #152) are
  carrying most of the win.
- **The remaining gap to llama.cpp at the same 512 → 128 shape on the same
  A6000 sm_86 / driver 550.x base image is 1.54×** (kiln 44.75 vs llama.cpp
  68.99, both median-of-3, current main on both sides). At 512 → 256 the
  llama.cpp side is essentially flat (69.23 tok/s), so widening the decode
  window does not change the gap — this is not a TTFT-vs-decode artifact.
  The original doc estimated ~2× by comparing the kiln post-#536 A6000
  number against the historical RTX 6000 Ada (sm_89) llama.cpp row at
  512 → 256; the apples-to-apples A6000 sm_86 re-bench in this refresh
  closes that measurement gap and lands the actual delta closer to ~1.5×.
- **vLLM and SGLang comparisons are blocked engine-side**, not because kiln
  is faster. Whenever a vLLM stable wheel ships `+cu124` / `+cu128` (or the
  RunPod kiln-runpod base image upgrades to driver ≥580), the H17b /
  H17 drivers in `scripts/h17b_vllm_020_alpha_dump.py` and
  `scripts/h17_sglang_alpha_dump.py` are re-runnable in 10–30 minutes for
  ~$0.10–$0.20 per engine. Wire those numbers into the multi-engine table at
  that point.
- **VRAM** at ~10 GB on A6000 leaves substantial headroom. KV-cache FP8
  (covered by `KILN_KV_CACHE_FP8`, opt-in) is the most obvious future row
  here.
- **Consistency**: post-#536 decode tok/s spread is 1.8 % across three
  seeds, well inside the 3–5 % run-to-run noise floor for kiln-bench on
  A6000 documented in the agent note `kiln-bench-median-of-3-noise-floor`.

The remaining bs=1 decode lever is verifier-step cost reduction inside the
self-spec path so that α=0.68 actually clears break-even (or, equivalently,
pulling break-even down toward current α). At larger batch / paged-prefill /
multi-turn-prefix workloads the dominant lever shifts to the radix prefix
cache (PR #520 / #521 / #523) and to longer-context KV residency.

## Reproducing

### Provision
Any A6000 / RTX 6000 Ada / L40S / A100 / H100 class GPU with driver ≥570 and
CUDA 12.4+. The `kiln-runpod` image
(`ghcr.io/ericflo/kiln-runpod:latest`) bakes the toolchain, sccache, and
build cache wiring; see `deploy/runpod/`.

### Fetch the model
```bash
hf download Qwen/Qwen3.5-4B --local-dir qwen3.5-4b
```

### Build kiln
```bash
export KILN_CUDA_ARCHS=86   # 86 on A6000/3090/4090, 89 on RTX 6000 Ada / L40S, 80 on A100, 90 on H100
cargo build --release --features cuda --bin kiln-bench
```

### Build llama.cpp
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j --target llama-bench llama-quantize
python3 convert_hf_to_gguf.py qwen3.5-4b --outfile qwen3.5-4b-bf16.gguf --outtype bf16
```

### Run
Current single-stream kiln protocol (matches PR #535 / PR #536):

```bash
KILN_W4A16=1 KILN_CUDA_GRAPHS=true ./target/release/kiln-bench \
  --model-path qwen3.5-4b \
  --paged \
  --prompt-tokens 512 --max-output-tokens 128 \
  --skip-training \
  --prompt-subset humaneval \
  --chat-template \
  --latency-only \
  --temperature 0.0 --seed 1 > kiln-bench.json
```

llama.cpp side (still on the `f3d5089` head-to-head shape, 512 → 256):

```bash
./llama.cpp/build/bin/llama-bench \
  -m qwen3.5-4b-bf16.gguf \
  -p 512 -n 256 -r 3 -o json > llama-bench.json
```

Raw JSON for the historical RTX 6000 Ada head-to-head is checked in under
[`bench-results/`](bench-results/). The post-#536 A6000 per-seed CSV is at
[`docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv`](docs/archive/phase-c/phase-c66/post-535-mtp-decode-bench.csv);
the post-#534 NVTX hotspot tables are in
[`docs/archive/phase-c/phase-c65/`](docs/archive/phase-c/phase-c65/).

## macOS / Apple Silicon (Metal)

Kiln also runs on Apple Silicon via candle-metal. Measured numbers are
not yet in this doc — the methodology below lets a contributor with
M3/M4 Max hardware drop them in.

### Provision
M3 Pro/Max, M4 Pro/Max, or M2 Ultra. Xcode Command Line Tools only
(full Xcode is **not** required — candle-metal-kernels JIT-compiles MSL
at runtime). Rust stable. No x86_64 Macs — Metal perf there is
unusable.

### Build kiln
```bash
cargo build --release --features metal --bin kiln-bench
```

### Run
```bash
./target/release/kiln-bench \
  --model-path qwen3.5-4b \
  --prompt-tokens 512 --max-output-tokens 256 \
  --paged --skip-training > kiln-bench-metal.json
```

The JSON output includes a top-level `"backend": "metal"` field so
mixed-platform reports can split runs without parsing GPU names.

### Compare against
- **llama.cpp Metal**: `cmake -B build -DGGML_METAL=ON` (no CUDA
  dependency), then the same `llama-bench` invocation.
- **MLX-LM**: Apple's reference inference stack; good baseline for
  Apple Silicon peak perf.

Kiln's Metal backend uses `candle_nn::ops::sdpa` for both prefill and
paged decode (the latter via an `index_select` gather from the paged
pool). GDN linear-attention layers run on the portable candle
composition.
