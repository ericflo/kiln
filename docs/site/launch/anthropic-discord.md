---
channel: anthropic-discord
status: draft
length: ~220 words
---

# Anthropic Discord launch post draft

> **DRAFT — do not post until reviewed by Eric and the v0.2.12 ops gate stays green.**

## Post body (paste directly into the appropriate Anthropic Discord channel)

> **Kiln: single-GPU inference server with live online learning**
>
> https://ericflo.github.io/kiln/launch.html · https://ericflo.github.io/kiln/demo/ · https://github.com/ericflo/kiln
>
> I built Kiln around a workflow I wanted as an AI-product builder: your model should get better from production feedback without a separate training service, a notebook handoff, or a redeploy.
>
> Kiln serves one model and trains LoRA adapters in the same process on the same GPU. Send normal OpenAI-compatible chat requests, then POST corrections to `/v1/train/sft` or scored completions to `/v1/train/grpo`. Training runs in the background, writes a new adapter, and hot-swaps it at an iteration boundary — no restart and no second copy of the model in VRAM.
>
> The part I think is most relevant here is the feedback loop: Claude can be the evaluator, rubric writer, or product copilot that scores generations, while Kiln is the self-hosted model that absorbs those scores into a domain-specific LoRA. That makes GRPO reward loops feel like an API integration instead of a batch ML project.
>
> It is deliberately single-model and single-GPU: Qwen3.5-4B, tuned for 24GB+ cards, with continuous batching, paged KV cache, live SFT/GRPO endpoints, adapter history, and a small enough footprint that the economics work for product teams experimenting outside managed fine-tuning platforms.
>
> Launch post: https://ericflo.github.io/kiln/launch.html
> 60-second demo: https://ericflo.github.io/kiln/demo/
> GitHub: https://github.com/ericflo/kiln
>
> I would especially love feedback from people building Claude-centered workflows: where would you want the reward/scoring boundary to live, and what would make the online-learning loop safe enough to trust in a real product?

## Posting checklist

- [x] Demo asciicast landed and linked from the launch post
- [ ] Confirm the target channel permits project launch drafts and GitHub links
- [ ] Eric reviews this draft before posting
- [ ] Post after Rust Discord as a targeted AI-builder follow-up, ideally while the HN/X threads are still active
- [ ] Stay in the channel for ~1 hour to answer questions about reward loops, safety boundaries, and Claude-in-the-loop scoring
