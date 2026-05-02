---
channel: lobste
status: draft
length: ~70 chars title / ~330 words comment
---

# lobste.rs submission draft

> **DRAFT — do not post until reviewed by Eric and the v0.2.12 ops gate stays green.**

## Submission

**Title** (max 100 chars; this is 67):

```
Kiln: a single-GPU inference server with live LoRA training in Rust
```

**URL field:**

```
https://ericflo.github.io/kiln/launch.html
```

**Tags suggestion** (in order of relevance — confirm against the active lobste.rs tag list before posting):

- `rust` — primary language
- `ml` — machine learning
- `ai` — AI / LLM
- `performance` — single-GPU optimization angle

(Do not use `show` unless lobste.rs has reinstated it. The tag list shifts.)

## Comment to post on the submission

> Author here. Some context on why I built this and where the tradeoffs live, since this audience tends to be skeptical and I'd rather front-load the honest version.
>
> The pitch: Kiln is one Rust binary that serves a model and trains it in the same process, on the same GPU, sharing the same model in VRAM. You POST corrections (`/v1/train/sft`) or scored completions (`/v1/train/grpo`) over HTTP, training runs on a background thread, and new LoRA weights hot-swap in atomically at the next iteration boundary. The serving thread never stops.
>
> Why it exists: the typical "improve a deployed model" loop today is collect failures → format → upload → wait hours → redeploy. That cost-per-iteration is so high that almost nobody iterates. The thesis is that if the loop becomes one HTTP call, you iterate constantly, and a 4B model continuously tuned to your workload outperforms a generic 70B on the tasks you actually care about.
>
> The honest tradeoffs:
>
> - **Single model.** It targets [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) and the scheduler, block manager, and CUDA kernels are tuned for that one architecture. Model-agnostic mode is on the post-1.0 roadmap with the explicit caveat that kernels need re-tuning per architecture.
> - **Single GPU.** No tensor parallel, no multi-node. Multi-GPU TP is also on the post-1.0 roadmap.
> - **Pre-1.0.** The current production line is kiln-v0.2.12. Phases 1–10 (core inference, LoRA serving, SFT, GRPO, production hardening, decode-perf optimization, dev experience, advanced adapter features, v0.1.0 release engineering, Liger long-context training) are all closed. The roadmap from here is deliberately small and chosen by what early users actually ask for.
>
> What's in-tree (not vendored Python or wrapped C++): the forward pass, paged KV cache + block manager, Sarathi-style chunked-prefill scheduler, LoRA training loop, Marlin W4A16 GEMM, fused RMSNorm, GDN linear-attention kernel, vendored flash-attn-2 with our own C-ABI wrapper, fused Conv1d. All in `crates/kiln-*-kernel/` and `crates/kiln-model/src/forward.rs`.
>
> Repo: https://github.com/ericflo/kiln. MIT. Sigstore-signed build provenance on every release artifact and the GHCR image.
>
> Happy to dig into any of the architecture choices or the GRPO loop. Particularly interested in feedback on the single-model design choice from people who've shipped general-purpose inference servers.

## Posting checklist

- [x] Demo asciicast landed and linked from the launch post
- [ ] Confirm tags against current lobste.rs taxonomy (they prune)
- [ ] Eric reviews this draft before posting
- [ ] Submit Tuesday–Thursday, 9:00–11:00 ET
- [ ] Watch the thread for the first 2 hours; lobste.rs threads die fast but the early discussion sets the tone
- [ ] Engage substantively on technical pushback — this audience values long-form replies, not one-liners
