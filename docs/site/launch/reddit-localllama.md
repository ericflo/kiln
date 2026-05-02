---
channel: reddit-localllama
status: draft
length: ~140 chars title / ~620 words body
---

# /r/LocalLLaMA submission draft

> **DRAFT — do not post until reviewed by Eric AND the demo asciicast lands.**

## Submission

**Title** (max 300 chars; this is 138):

```
Kiln: a single-GPU inference server in Rust that also trains the model it's serving — live LoRA hot-swap, runs Qwen3.5-4B on a 3090
```

**Flair suggestion:** `Resources` or `New Model / New Tool` (whichever is canonical at post time).

## Body

> Hey r/LocalLLaMA. I built [Kiln](https://github.com/ericflo/kiln) because I got tired of the loop where you ship a local model, find a thing it does badly, collect failure examples, run a separate training job overnight, and redeploy. I wanted that loop to be one HTTP call.
>
> **What it is:** one Rust binary that serves a model AND trains it in the same process, on the same GPU, sharing the same weights in VRAM. You POST corrections to `/v1/train/sft` or scored completions to `/v1/train/grpo` and new LoRA weights hot-swap in atomically at the next iteration boundary. No restart, no second copy of the model, no Python sidecar.
>
> **What it runs:** [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) on a single 24 GB consumer GPU. RTX 3090, RTX 4090, A6000 on the CUDA side. M-series Mac with 16 GB+ unified memory on the Metal side.
>
> **The 24 GB story is unusually good** because Qwen3.5-4B's hybrid architecture is 24 linear-attention Gated DeltaNet layers + 8 full-attention layers. Only 8 layers need a KV cache at all — the other 24 have O(1) state. So 128K context for one sequence fits in ~13 GB. Even *training* at 128K context fits in ~18 GB. (Actual numbers, not back-of-envelope.)
>
> **The killer feature is GRPO, not SFT.** Generate candidates, score them with your own reward function, POST the batch. The model learns what "good" means for *your* use case:
>
> ```python
> import openai, requests
> client = openai.OpenAI(base_url="http://localhost:8420/v1", api_key="unused")
>
> # 1. Generate 8 candidates
> responses = [
>     client.chat.completions.create(
>         model="qwen3.5-4b-kiln",
>         messages=[{"role": "user", "content": prompt}],
>         temperature=0.7,
>     )
>     for _ in range(8)
> ]
>
> # 2. Score them however you want — regex, unit tests, another model, click-through
> scored = [{"text": r.choices[0].message.content, "reward": my_score(r)}
>           for r in responses]
>
> # 3. POST — the server trains and hot-swaps immediately
> requests.post("http://localhost:8420/v1/train/grpo", json={
>     "groups": [{"prompt": prompt, "completions": scored}]
> })
>
> # 4. Next inference already uses the improved weights
> ```
>
> **What's actually in the repo** (not wrapped Python, not vendored C++ ports): the forward pass, the Sarathi-style chunked-prefill scheduler, the paged KV cache, the LoRA training loop with gradient checkpointing, fused CUDA kernels for flash-attention-2, GDN linear-attention, Marlin W4A16 GEMM, RMSNorm, and Conv1d. Vendored where it made sense, written from scratch where it didn't.
>
> **What's the catch?**
>
> - **Single model.** Every line of code is tuned for Qwen3.5-4B. Model-agnostic mode is post-1.0 with the honest tradeoff that kernels need re-tuning per architecture.
> - **Single GPU.** Multi-GPU tensor parallelism is also post-1.0.
> - **Pre-1.0.** Current production line is kiln-v0.2.12. The phases that got us here closed core inference, LoRA serving, SFT, GRPO, production hardening, decode-perf optimization, developer experience, advanced adapter features, v0.1.0 release engineering, and the Liger long-context training pieces. The roadmap from here is deliberately small.
>
> **How to try it:**
>
> - Linux+CUDA, macOS Apple Silicon (Metal), and Windows+CUDA binaries on the [releases page](https://github.com/ericflo/kiln/releases). Sigstore-signed build provenance on every artifact.
> - `ghcr.io/ericflo/kiln-server:latest` on GHCR with the same provenance attestation.
> - [Quickstart](https://github.com/ericflo/kiln/blob/main/QUICKSTART.md) walks through downloading the model weights, the first chat completion, the first SFT POST, and the first GRPO loop.
> - Embedded `/ui` dashboard for live decode tok/s, p50/p99 ITL, recent requests, and adapter management. No extra service to run.
> - Full launch post (with the architecture deep-dive and the tradeoffs section): https://ericflo.github.io/kiln/launch.html
>
> MIT licensed. Built solo in Rust.
>
> Happy to answer questions — particularly interested in what reward functions y'all would want to plug into the GRPO loop. The single-model design is also the thing I most expect pushback on, so fire away.

## Posting checklist

- [ ] Demo asciicast linked from the launch post
- [ ] Confirm flair against current sub rules (the mods rotate accepted flair)
- [ ] Eric reviews this draft + the asciicast
- [ ] Submit Tuesday–Thursday, 9:00–11:00 ET (catches both US morning and EU afternoon)
- [ ] Reply to every top-level comment in the first 4 hours; this sub rewards founder presence
- [ ] If the post catches: cross-post to r/MachineLearning Project Announcements (separate weekly thread) — but only after r/LocalLLaMA discussion has settled
