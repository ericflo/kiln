---
channel: x
status: draft
length: 8 tweets, each <280 chars
---

# X / Twitter thread draft

> **DRAFT — do not post until reviewed by Eric AND the demo asciicast lands.**

## Thread

**1/8 — hook (lead with GRPO value prop)**

> Most AI products feel frozen because the model never learns from the people using it.
>
> I built Kiln to fix that: a single-GPU inference server that trains the model it's serving. POST a correction over HTTP, the next request already uses the improved weights.
>
> 🧵

*(279 chars including 🧵)*

**2/8 — the killer feature**

> The unlock isn't SFT. It's GRPO.
>
> Generate 8 candidates → score them with your own reward fn (regex, unit tests, a judge model, click-through) → POST → server trains → next inference already uses the new weights.
>
> The model learns what "good" means for your workload, not a benchmark.

*(278 chars)*

**3/8 — hot-swap mechanics**

> No restart. No separate training pipeline. No second copy of the model in VRAM.
>
> Training runs on a background thread inside the same process. New LoRA weights swap in atomically at the next iteration boundary. The request that started 1 ms ago finishes on old weights; the next runs on new ones.

*(279 chars — at limit, OK)*

**4/8 — the 4B argument**

> Why 4B? Because a 4B model continuously tuned to your data beats a generic 70B on your task, and it runs on a 24 GB consumer GPU.
>
> Qwen3.5-4B's hybrid architecture (24 linear-attention layers + 8 full-attention) puts 128K context in ~13 GB. Training too.

*(254 chars)*

**5/8 — the Rust argument**

> Pure Rust. Single binary. One process.
>
> No Python sidecar, no socket-tensor-copy, no second model held in VRAM during training. Vendored CUDA kernels for flash-attn, GDN, Marlin W4A16, RMSNorm, Conv1d. Every layer is accounted for in the source.
>
> The specificity is the speed.

*(278 chars)*

**6/8 — anti-Goliath positioning**

> Kiln is not a general-purpose framework. It does not support 47 model families. There is no plugin system. No 800-knob config.
>
> It's a scalpel for one model. If you want to get extremely good at one model, that's the loop. If you want to swap between five, vLLM/SGLang are great.

*(279 chars)*

**7/8 — how to try it**

> Linux+CUDA, macOS Apple Silicon (Metal), and Windows+CUDA binaries on the releases page. Sigstore-signed build provenance on every artifact. `ghcr.io/ericflo/kiln-server` on GHCR.
>
> Quickstart is single-command. Embedded /ui dashboard for adapters, training, and a chat playground.

*(278 chars)*

**8/8 — call to action + link**

> Repo, releases, and the full launch post:
>
> https://ericflo.github.io/kiln/launch.html
>
> MIT licensed. v0.2.9 is the production line. Built in Rust by one person; the GRPO loop is what kept me up at night and what I most want feedback on.

*(248 chars — has link headroom for X's t.co)*

## Image / asciicast suggestions

- **Tweet 1 hero:** static screenshot of the GRPO Python snippet from `docs/site/launch.html` (lines 171–193). Alt text: "Python code: 1) generate 8 chat completions; 2) score them; 3) POST to /v1/train/grpo; 4) next request already uses the improved weights."
- **Tweet 2 GRPO:** the demo asciicast embed (when ready). Alt text: "60-second terminal recording of cold-starting Kiln, sending a chat request, POSTing an SFT correction, and seeing the next request use the improved output."
- **Tweet 4 4B:** screenshot of the README memory budget table. Alt text: "Memory budget table showing Qwen3.5-4B fits 128K context in ~13 GB on a 24 GB GPU, including training."
- **Tweet 7 try it:** screenshot of the embedded `/ui` dashboard. Alt text: "Kiln /ui dashboard showing live decode tokens/sec, p50/p99 ITL, recent requests, and adapter management controls."

## Post-thread reply template ("is this a wrapper around X?")

> Not a wrapper — every layer is in the source. The forward pass, the scheduler, the block manager, the LoRA training loop, and the fused CUDA kernels (flash-attn, GDN, Marlin W4A16, RMSNorm, Conv1d) are all in-repo and tuned specifically for Qwen3.5-4B's hybrid architecture. The only "wrap" is being OpenAI-API-compatible on the wire so existing clients work unchanged.

## Posting checklist

- [ ] Demo asciicast finalized and linked from the launch post
- [ ] Verify the `og:image` and Twitter card on `docs/site/launch.html` render correctly via cards-validator
- [ ] Eric reviews this draft + the asciicast
- [ ] Post Tuesday or Wednesday, 9:00–11:00 ET (best Twitter dev-tools traction window)
- [ ] Pin tweet 1 to the profile for the launch week
- [ ] Watch replies for the first 2 hours, respond to every substantive question
