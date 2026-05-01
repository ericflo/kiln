---
channel: discord-rust
status: draft
length: ~250 words
---

# Rust Discord #showcase post draft

> **DRAFT — do not post until reviewed by Eric AND the demo asciicast lands.**

## Post body (paste directly into #showcase)

> 🦀 **Showcase: Kiln — single-GPU LLM inference + live LoRA training, all in Rust**
>
> https://github.com/ericflo/kiln · https://ericflo.github.io/kiln/launch.html
>
> One binary that serves a language model and trains it in the same process, on the same GPU. You POST corrections or scored completions over HTTP, training runs on a background thread, and new LoRA weights hot-swap in atomically at the next iteration boundary — no restart, no second copy of the model in VRAM, no Python sidecar copying tensors over a socket.
>
> **What's interesting from a Rust-systems angle:**
>
> - Pure Rust, single binary. `axum` HTTP server, in-process training thread, `candle` for the autograd surface where it earns its keep, custom kernels where it didn't.
> - Vendored CUDA kernels with thin C-ABI wrappers + Rust FFI:
>     - `kiln-flash-attn` — flash-attention-2 (bf16, hdim 128/256, causal) with the PyTorch binding layer stripped out, our own ~300-line C-ABI shim, and `build.rs` driving `nvcc` via the `cc` crate.
>     - `kiln-gdn-kernel` — fused Gated DeltaNet linear attention.
>     - `kiln-marlin-gemm` — W4A16 GEMM for the MLP and projection paths.
>     - `kiln-rmsnorm-kernel` — fused pre-norm RMSNorm.
>     - `kiln-conv1d-kernel` — vendored mamba-ssm causal conv1d.
> - Sarathi-style chunked-prefill scheduler with continuous batching, paged KV cache, and a block manager that gives KV cache "virtual memory" semantics.
> - Targets one model (Qwen3.5-4B) deliberately. The scheduler knows exactly how much KV cache each layer needs because every layer is accounted for in the source.
>
> **Build matrix is intentionally small** — "has CUDA or doesn't." `cargo build --release --features cuda` on Linux+CUDA, `--features metal` on Apple Silicon, plain `cargo build --release` for the CPU/headless path. macOS Apple Silicon gets the Metal backend via a candle-metal JIT path.
>
> Reproducible builds with `--locked`, Sigstore-signed build provenance on every release artifact and the GHCR `kiln-server` image, MIT licensed, pre-1.0 (current production line is kiln-v0.2.8).
>
> Would love eyes on the kernel crates and the in-process training thread design. Also happy to talk about the tradeoffs of the single-model architectural choice — that's the thing I most expect pushback on.

## Posting checklist

- [ ] Demo asciicast linked from the launch post
- [ ] Verify the post is under the channel's character limit (`#showcase` rules sometimes cap at 2000 chars; trim the kernel bullet list if needed)
- [ ] Eric reviews this draft + the asciicast
- [ ] Post during US working hours, Tue–Thu (Discord is more time-zone-mixed than HN; aim for 12:00–14:00 ET to catch both EU late-afternoon and US lunch)
- [ ] Stay in the channel for ~1 hour to answer questions; Discord conversations are short-lived
