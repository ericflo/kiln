# Benchmarks

Head-to-head inference performance comparison between **kiln** (this project)
and **llama.cpp** on the same GPU, same model, same prompt/output lengths.

## Setup

| Component | Value |
|---|---|
| GPU | NVIDIA RTX 6000 Ada Generation (48 GB, compute capability 8.9) |
| Driver | 570.211.01 |
| CUDA | 12.8.93 |
| Host CPU | AMD EPYC 9654 96-Core |
| Model | Qwen3.5-4B (~4.2B params, 32 layers, hybrid GDN + GQA) |
| Weights | bf16 |
| Prompt length | 512 tokens (kiln measured 506 after tokenization) |
| Output length | 256 tokens |
| kiln commit | `f3d5089` |
| kiln features | `--features cuda`, `KILN_CUDA_ARCHS=89` |
| kiln binary | `kiln-bench` |
| llama.cpp commit | `408225b` |
| llama.cpp build | `cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86` |
| llama.cpp binary | `llama-bench` (default flags) |

Both binaries were built natively on the benchmark host with full CUDA optimization.
llama.cpp uses its default settings: `n_batch=2048`, `n_ubatch=512`, `flash_attn=false`,
`n_gpu_layers=99` (all layers offloaded), K/V cache in f16.

## Results

### Single-stream throughput (512 prompt → 256 generated)

| Engine | Prefill (tok/s) | Decode (tok/s) | Time-to-first-token | Peak VRAM |
|---|---:|---:|---:|---:|
| **llama.cpp** | **8224.45** | **88.43** | ~62 ms | **8820 MB** |
| **kiln** | 36.31 | 10.89 | 13934 ms | 10278 MB |
| *Ratio (llama.cpp / kiln)* | *226×* | *8.1×* | *222× faster* | *0.86×* |

Decode tok/s is reported as mean over the 256-token generation; llama.cpp's
stddev over 3 repetitions was 0.05 tok/s, kiln's per-token stddev was ≈ 91.8 ± 0.6 ms
(p50 91.8 ms, p99 94.7 ms).

VRAM is peak during inference as reported by each binary. For llama.cpp this
was sampled with `nvidia-smi` polling during `llama-bench`; for kiln it is the
engine's internal peak counter.

### kiln steady-state (repeated sequential runs)

`kiln-bench` runs the same 512-in / 256-out decode repeatedly to check for
throughput drift. Numbers are effectively flat across 1–16 repeated runs:

| Sequential runs | tok/s | Peak VRAM |
|---:|---:|---:|
| 1 | 10.21 | 10278 MB |
| 4 | 10.14 | 10278 MB |
| 8 | 10.09 | 10278 MB |
| 16 | 10.08 | 10278 MB |

These are sequential runs of the same single-stream workload, not concurrent
batch sizes. kiln today does not expose a concurrent-batch decode mode in its
`kiln-bench` binary, so a true batch-throughput comparison against
`llama-batched-bench` is not included here.

## Interpretation

- **llama.cpp is dramatically faster** on this workload — ~8× on decode and
  over 200× on prefill. The prefill gap is especially large: llama.cpp executes
  the 512-token prefill in ~62 ms while kiln takes ~13.9 s, which dominates
  TTFT.
- **VRAM** is comparable. llama.cpp uses ~8.6 GB for the model + workspace;
  kiln uses ~10.0 GB. Both fit comfortably on a 24 GB card.
- **Consistency**: both engines are very steady within a run (llama.cpp stddev
  0.06 % of mean on decode; kiln p99/p50 ratio ≈ 1.03).

kiln is an early-stage Rust-native inference engine. The 200× prefill gap and
8× decode gap against llama.cpp identify prefill kernels and decode step
overhead as the highest-leverage optimization targets. Fused QKV projection,
flash-attention on the full-attention layers, and tighter GDN step kernels are
likely where the gap is hiding.

## Reproducing

### Provision
Any RTX 6000 Ada / L40S / A6000 / A100 class GPU with driver ≥ 570 and
CUDA 12.8.

### Fetch the model
```bash
hf download Qwen/Qwen3.5-4B --local-dir qwen3.5-4b
```

### Build kiln
```bash
export KILN_CUDA_ARCHS=89   # 86 on A6000, 80 on A100
cargo build --release --features cuda --bin kiln-bench
```

### Build llama.cpp
```bash
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DBUILD_SHARED_LIBS=OFF
cmake --build build --config Release -j --target llama-bench llama-quantize
python3 convert_hf_to_gguf.py qwen3.5-4b --outfile qwen3.5-4b-bf16.gguf --outtype bf16
```

### Run
```bash
./target/release/kiln-bench \
  --model-path qwen3.5-4b \
  --prompt-tokens 512 --max-output-tokens 256 \
  --skip-training > kiln-bench.json

./llama.cpp/build/bin/llama-bench \
  -m qwen3.5-4b-bf16.gguf \
  -p 512 -n 256 -r 3 -o json > llama-bench.json
```

Raw JSON for this run is checked in under [`bench-results/`](bench-results/).

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
