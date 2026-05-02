---
channel: hn
status: draft
length: ~80 chars title / ~220 words first comment
---

# Show HN: Kiln draft

> **DRAFT — do not post until reviewed by Eric and the v0.2.12 ops gate stays green.**

## Submission

**Title** (max 80 chars; this is 73):

```
Show HN: Kiln – Single-GPU inference server with live LoRA training in Rust
```

**URL field:**

```
https://ericflo.github.io/kiln/launch.html
```

(HN strongly prefers a content URL over a bare GitHub repo for "Show HN". The launch post links to the repo, the README, the QUICKSTART, and the releases page.)

## First comment (post immediately after submission)

> Hi HN — author here. Kiln is a single-GPU inference server that also trains the model it's serving. Same process, same GPU, same model in VRAM. You POST corrections or scored completions over HTTP, training runs on a background thread, and new LoRA weights hot-swap in atomically at the next iteration boundary. There's no separate training pipeline, no second model loaded for the swap, and no restart.
>
> The piece I'm most excited about is the GRPO loop: generate candidates, score them with your own reward function (regex, unit tests, a judge model, click-through, whatever), and POST the scored batch. The model learns what "good" means for *your* workload, not a generic benchmark.
>
> Some things I expect people to ask up front:
>
> - **Why 4B?** Because a 4B model continuously tuned on your data beats a generic 70B on your task, and it fits on a 24 GB consumer GPU you might already own. The Qwen3.5-4B hybrid architecture (24 linear-attention layers + 8 full-attention) means 128K context fits in ~13 GB.
> - **Why Rust?** Training and serving share GPU memory cooperatively in a single process. No Python sidecar, no socket-tensor-copy, no second copy of the weights.
> - **Why one model?** Every line of code is tuned for Qwen3.5-4B. The scheduler, kernels, and block manager don't have to be general — that specificity is the source of the speed and the developer experience. Model-agnostic mode is on the post-1.0 roadmap with the honest tradeoff that kernels need a re-tune per architecture.
> - **Why not vLLM/SGLang/llama.cpp?** They're excellent at general-purpose serving. Kiln is a scalpel for the "I want this one model to be the best version of itself for my workload" loop. If that's not your loop, those projects are probably the right answer.
>
> MIT, public binaries for Linux+CUDA, macOS Apple Silicon (Metal), and Windows+CUDA, plus a `ghcr.io/ericflo/kiln-server` image. Sigstore-signed build provenance on every artifact. Repo: https://github.com/ericflo/kiln. Quickstart: https://github.com/ericflo/kiln/blob/main/QUICKSTART.md.
>
> Happy to answer questions about the architecture, the GRPO loop, or the single-model design choice.

## Posting checklist

- [x] Demo asciicast landed and linked from the launch post (`docs/site/launch.html`)
- [ ] Confirm `gh release view kiln-v0.2.12 -R ericflo/kiln` is clean
- [ ] Confirm `https://ericflo.github.io/kiln/launch.html` renders on mobile + desktop
- [ ] Eric reviews this draft before posting
- [ ] Submit Tuesday–Thursday, 9:00–11:00 ET (US morning peak for HN)
- [ ] Post the first comment within 60 seconds of submission
- [ ] Watch the thread for the first 90 minutes, answer every top-level question
