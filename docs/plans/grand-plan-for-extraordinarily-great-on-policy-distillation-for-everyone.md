# A Grand Plan for Extraordinarily Great On-Policy Distillation for Everyone

> *Frontier brilliance, distilled by anyone, on whatever hardware they own, into a 4B model that gets sharper every time they use it.*

**Status:** Vision document. Ready to be staged into engineering plans.
**Author:** Synthesised by Claude (Opus 4.7) from the on-policy-distillation paper corpus in `docs/papers/on-policy-distillation/` and the kiln codebase as of branch `on-policy-distillation`.
**Date:** 2026-05-15.

---

## 0. Executive summary

On-policy distillation (OPD) is the most undervalued technique in post-training right now. Three lines of evidence converged in the last 12 months:

1. **Lu/Thinking Machines (Oct 2025)** showed OPD reaches the same reasoning quality as RL at **1/10th** the GPU-hours, and the same quality as off-policy SFT at **1/9th to 1/30th** the cost — while remaining a one-line change on top of any RL framework.
2. **DeepSeek-V4 (early 2026)** *replaced its mixed-RL post-training stage entirely* with multi-teacher full-vocabulary OPD over 10+ domain experts. The state-of-the-art open model is now an OPD model.
3. **The 2026 paper wave** (Fu, Luo, Li, Song & Zheng) catalogued the failure modes — length inflation, flawed-prefix collapse, thinking-pattern mismatch, tokenizer drift — and provided clean fixes (Stable-OPD, Top-K local support, off-policy cold-start, teacher-aligned templates) so OPD is no longer a research curiosity. It is engineering.

Kiln is the **single best vehicle in the open-source world to deliver this to everyone**, and it is not close. The reasons:

- **One model, one tokenizer.** Kiln targets `Qwen/Qwen3.5-4B`. Frontier-grade `Qwen/Qwen3.6-27B` shares the same architecture and tokenizer. That makes the student/teacher pair a clean per-token KL — no cross-tokenizer hacks (DSKD, ULD, CSD) needed. *This is the most important fact in the entire plan.*
- **One process, one binary, one GPU.** Kiln already serves, trains, hot-swaps adapters, and evals in a single Rust process. OPD needs exactly that loop: rollout → score-each-token → grad → swap. No Python sidecar, no Ray cluster, no Triton dance.
- **Three backends, with a Vulkan trajectory that compounds.** CUDA + Vulkan + Metal already exist for both inference *and* training. Every other OPD framework on Earth is CUDA-only. Kiln can be the first OPD trainer that runs on a MacBook, a 7900 XTX, *and* a rack of H200s with the same code path. **`kiln-train` already keeps every activation, gradient, and optimizer state resident on Vulkan** (see `docs/vk_native_training.md` — CPU touchpoints in a step are reduced to input upload, loss readback for logging, and adapter save). The Vulkan *inference* engine is concurrently lifting the host↔device boundary to per-step (PR #1030 targets ~7× lower per-decode kernel overhead) — and because OPD reuses the inference engine for student rollouts (§9.3), every Vulkan inference win lands as an OPD speedup automatically. **The two workstreams compound.**
- **The primitives are already 80% there.** SFT-over-HTTP, GRPO-over-HTTP, LoRA hot-swap, paged KV, prefix caching, FP8 KV, weight-space adapter merge (`linear` / `ties` / `concat`), 128K context, eval suites, judgment flywheel. OPD is the natural third training mode — its UX is GRPO without the reward function, plus a teacher.
- **The 4B → 27B brain-pump is uniquely powerful.** Hosted Qwen3.6-27B logits exist on every major inference provider. Buying a million teacher logits costs roughly the price of a sandwich. A one-click "distill a slice of 27B into a LoRA" flow turns frontier capability into a downloadable adapter.

**The non-negotiable design principle — the pit of success.** Every default in this plan is chosen so that a user who reads no documentation, runs no diagnostics, and presses one button still ends up with a working adapter. Failure modes have automatic mitigations engaged *before* the user sees the failure. Knobs default to values that work for 95% of cases; sub-optimal values require explicit override; destructive choices require explicit confirmation. Cold-start is auto-injected when the student/teacher gap demands it. Cost is hard-capped before the first API call. Bad runs auto-rollback to the prior checkpoint. The OPD literature is not the user's homework — it is the manual kiln consults on the user's behalf, with every auto-decision linked to a one-paragraph "Why?" the user can click if curious. **The user's job is to express intent. Kiln's job is to make the right thing happen.** §8 elaborates this principle into the concrete defaults, guardrails, and surfaces that deliver it.

**The bet of this document:**

> If we ship the OPD primitives below in roughly the order described, kiln becomes the default tool for personalising small language models for the next five years, and we put a 4B model with frontier-quality reasoning on every laptop, every dev workstation, every corporate H200 rack — adaptable, private, fast, and *getting better every time it's used*.

---

## 1. Why now, why kiln, why OPD specifically

### 1.1. The state of the field as of 2026-05-15

Three primitives dominate post-training today:

| | Sampling | Reward density | Bits/episode | When it dominates |
|---|---|---|---|---|
| **SFT** (off-policy) | teacher | dense | O(N) tokens | small data, short tasks |
| **RL** (on-policy) | student | sparse | O(1) | when only the answer matters |
| **OPD** (on-policy + dense) | student | **dense** | **O(N)** | almost everywhere else |

That third row is dramatically under-shipped in open-source tooling. RL frameworks (verl, OpenRLHF, TRL-RL) treat OPD as a footnote. SFT frameworks (axolotl, llama-factory, unsloth) don't speak it at all. The Tinker cookbook recipe is a research demo, not a product. **There is no "axolotl for OPD" — and there should be, because OPD outperforms both other approaches for most personalisation-scale work.**

### 1.2. The Qwen3.5-4B / Qwen3.6-27B alignment is a structural gift

Most distillation work in the literature is hampered by *cross-family distillation*: different tokenizers, different chat templates, different positional schemes, different special tokens. Whole bodies of research (DSKD, ULD, CSD, SimCT) exist to paper over this gap.

Kiln is exempt from all of that. Same vocabulary, same chat template, same RoPE, same hybrid attention layout. Reverse KL between the two models is a clean per-token quantity. We can use **full-vocabulary** reverse KL when we have the teacher locally (DeepSeek-V4 style) and **teacher top-K local support matching** (Fu et al. +19.8% over sampled-token) when we get logits from an API. No tokenizer surgery. Ever.

### 1.3. Hosted logits are now a commodity

As of May 2026, top-K logprobs are returned by:

- **vLLM** (`top_logprobs` in OpenAI-compat completions, configurable up to ~20)
- **sglang** (same OpenAI surface, similar caps)
- **llama.cpp / `llama-server`** (full-vocab via `n_probs` or `--logits-all`; effectively unbounded if you control the server)
- **TGI** (`top_n_tokens`)
- **OpenRouter** (top-N passthrough where the upstream supports it)
- **Together, Fireworks, DeepInfra, Anyscale, Featherless, NIM** (varies; most expose top-K up to 5-20)
- **Hugging Face Inference Endpoints** (top-K)
- **Self-hosted Qwen on a peer's box** — the most interesting case (see §3.11)

The bandwidth math is striking: **a 1024-token reasoning rollout with top-32 logprobs is roughly 1024 × 32 × 8 bytes ≈ 256 KB of teacher signal.** A million such rollouts is 250 GB — a single overnight pull on a home connection. The compute is rentable for under $100 against most providers' pricing for 27B-class models.

OPD turns those logits into a downloadable LoRA. **The price of frontier-flavoured personalisation is now roughly the price of a few takeout meals.** Nobody has shipped the workflow that makes this concretely buildable. Kiln will.

### 1.4. The Lu/Stable-OPD/Rethinking trio means OPD is *engineerable*

A year ago, telling a hobbyist to do OPD would have meant inviting them into the failure-mode forest blindfolded. Today we know:

- Reverse-KL on student-sampled trajectories with γ=0 (Lu) — the simplest objective — works for the reasoning case, given a strong SFT initialisation.
- **Teacher Top-K local support matching with renormalisation + top-p rollouts + special-token masking** (Fu et al.) gives +19.8% over the naive sampled-token form and is the right default for non-trivial setups.
- **Stable-OPD** (Luo et al.) — `L_OPD + λ·L_SFT_golden + β·KL(π‖π_ref)` — eliminates the abrupt length-inflation collapse that otherwise kills runs around step 30.
- **Off-policy cold-start + teacher-aligned chat template + entropy-preserving prompt mixing** (Li et al.) closes the thinking-pattern gap that otherwise causes a stronger teacher to fail where a weaker one would have succeeded.
- **Diagnostic metrics that predict success** (Li et al.): overlap ratio, overlap-token advantage, entropy gap, truncation rate, repetition rate. These are computable in real time during training.

We can ship these as defaults *and* as auto-detected mitigations triggered by the metrics. **The user never has to know what "reverse KL between top-32 supports with renormalisation" means.** They press a button.

---

## 2. The three tiers we serve

Designing for one user, even an idealised one, would be a mistake. Kiln has to land cleanly across a 1000× hardware spread.

### 2.1. Tier 1 — Laptop user (8–16 GB VRAM, Apple Silicon, or just a strong CPU)

- **Cannot** load Qwen3.6-27B locally at any useful precision.
- **Can** train Qwen3.5-4B + a small LoRA (rank 16–64) thanks to kiln's gradient checkpointing, 24+8 hybrid attention KV-cache savings, and Metal/Vulkan compute paths.
- **Wants:** a personal assistant that knows their notes, code, and writing style. A coding LoRA for *their* repos. A judge LoRA for *their* preferences. None of this needs frontier scale; it needs frontier *behaviour* poured into 4B parameters.
- **Solution shape:** OPD with a **remote teacher** (vLLM/llama.cpp/sglang/OpenRouter) over their broadband, with **aggressive prefix and per-position logit caching** so a single $5–10 spend produces a permanent personal LoRA.

### 2.2. Tier 2 — Prosumer (1× 4090/5090/7900XTX/M-series Max, 24–48 GB VRAM)

- **Can** load Qwen3.6-27B at 4-bit (≈14 GB) or FP8 (≈27 GB) alongside the 4B student via aggressive offload, or run the teacher on a second card.
- **Wants:** experiment, fine-tune for narrow domains, host hobby agents, possibly serve a few external users.
- **Solution shape:** OPD with **local teacher mode** — async teacher scheduling, KV reuse across student rollouts, the full Stable-OPD loop, the LoRA composition feature, and an "always-up-to-date" continual learning workflow.

### 2.3. Tier 3 — Corporate (8× H200, multi-node)

- **Can** do anything DeepSeek-V4 did. They have the compute and they have the data privacy requirements that make hosted APIs a non-starter.
- **Wants:** a reliable pipeline to train N domain specialists and consolidate them into one unified 4B that beats their current generic 70B at every internal task. Reproducibility, governance, audit trails.
- **Solution shape:** OPD with **multi-teacher full-vocabulary** mode, **FP4 QAT** on MoE-style adapter stacks, **deterministic kernel paths** (kiln already targets bitwise reproducibility per the recent FLCE work), and the full DeepSeek-V4 specialist-then-consolidate recipe templated as a one-shot job.

**The same code path serves all three.** The teacher abstraction (§3.2) is what differs. The training loop (§3.1) does not.

---

## 3. The architectural pillars

Twelve pillars, each with a clear home in the kiln crate layout. The numbering mirrors implementation order roughly but not strictly — see §5 for the actual phasing.

### 3.1. The OPD core training loop — `kiln-train::opd`

**Where:** new module `crates/kiln-train/src/opd.rs`, sibling to `cuda_train.rs` and `vk_train.rs`. The trainer dispatches to either backend.

**Loss:** Per-token reverse KL with discount γ=0 (Lu's choice; we don't pay the variance cost of γ>0). Three available granularities, controlled by one config knob:

- `sampled_token` — the simplest, brittle on long rollouts (Fu et al.). Available for backwards-compat / debugging only.
- `teacher_top_k` (default) — renormalise both distributions over the teacher's top-K support, then KL. K=32 by default per Fu et al. ablation. **This is the default for laptop and prosumer tiers because it works with API-provided top-K logits.**
- `full_vocab` — exact full-vocab KL. Required for corporate-tier multi-teacher consolidation. **This is the default for Tier 3.**

**Stabilisers, on by default:**

```
L_total = L_OPD                                # base reverse KL
        + β_kl  · KL(π_θ ‖ π_ref)              # Stable-OPD reference penalty (Luo et al.)
        + λ_sft · L_SFT(golden_minibatch)      # mixture distillation against held-out goldens
```

`β_kl` and `λ_sft` are auto-tuned by the diagnostic stack (§3.8): if `RepRate > 0.05` for two consecutive validation passes, both weights get bumped. Manual override is possible.

**Sampling:**

- Top-p rollout sampling (p=0.9) by default — Fu et al. ablation table 3 shows this is essential. Naive temperature-1 sampling drives the failure modes.
- Special-token masking on by default for `<think>`, `</think>`, `<|im_start|>`, etc. — orthogonal fix from Fu et al.
- Group rollout: 4 samples per prompt, mirroring Lu's default; this is the same shape as kiln's existing GRPO `groups + completions` API, which is good news for code reuse.

**Importance-sampling correction** identical to the existing GRPO importance ratio code in `kiln-train`. The loss function literally swaps the per-token advantage from `reward - baseline` to `-reverse_kl_per_token`. Kiln's existing trainer infrastructure (replay buffer, optimizer state, hot-swap) is reused unchanged. **This is the one-line claim from Lu's blog made real.**

**Fused per-engine OPD-loss kernel.** The reverse-KL loss runs through a new `kiln-opd-loss-kernel` crate — top-K-restricted, bit-equivalent across CUDA/Vulkan/Metal, comparable in cost to cross-entropy rather than the 3–5× slowdown a naive implementation would impose by materialising full-vocab tensors. **§9 covers the per-engine efficiency story in full; the short version is that fast is a shipping gate, not an aspiration.** No engine ships OPD until it meets its perf gate.

**Auto-injected cold-start (default on).** Before any OPD run, the trainer fires a 50-prompt initial-overlap probe against the chosen teacher. If the median overlap ratio on top-32 supports is below 0.5 — meaning the student would be doing reverse-KL inside a near-empty support set, exactly the failure case Li et al. warn about and Lu's [^support] footnote explains formally (forward-KL SFT *adds* support; reverse-KL OPD does mode-seeking *inside* support) — kiln silently inserts a brief SFT cold-start phase (typically 2 epochs over 5–10K teacher rollouts) before the OPD loop begins. **The user sees one progress bar labelled "preparing your model" and never learns that two distinct training paradigms are running back-to-back.** Doing OPD without sufficient student-side support is the single most common silent failure on stylistically divergent student/teacher pairs; kiln removes this footgun by default. Power users can disable via `cold_start: "off"` after acknowledging the warning.

### 3.2. The Logit Source abstraction — `kiln-train::logit_source`

**Where:** new module. Three concrete implementations, common trait.

```rust
trait LogitSource: Send + Sync {
    /// For each token position in `tokens`, return the teacher's
    /// top-K logprobs (or full-vocab if K is None). May reorder
    /// internally for batch efficiency.
    async fn fetch_logprobs(
        &self,
        tokens: &[TokenId],
        position_offsets: &[usize],
        top_k: Option<usize>,
    ) -> Result<LogprobBatch>;

    /// Capability self-description so the trainer picks the right
    /// loss granularity automatically.
    fn capabilities(&self) -> LogitSourceCaps;  // max_top_k, supports_full_vocab, etc.
}
```

Implementations:

1. **`LocalTeacher`** — loads a second model into the same kiln process (or a sibling process behind a Unix socket for memory isolation). Uses the same kernel stack as inference. Default for Tier 2/3. Knows how to share KV with student rollouts at the prefix level when chat templates align.

2. **`RemoteTeacher`** — speaks the OpenAI-compatible `top_logprobs` schema. Adapters live in `logit_source/remote/{vllm.rs, sglang.rs, llama_cpp.rs, openrouter.rs, together.rs, fireworks.rs, deepinfra.rs, anyscale.rs, hf_endpoints.rs, tgi.rs, nim.rs}`. Each adapter encodes:
   - URL / auth conventions.
   - Top-K cap and whether `top_logprobs` returns log-softmax over the full vocab or top-K only.
   - Tokenization-stability check (kiln re-tokenises the prompt with its own tokenizer and compares against the provider's `usage.prompt_tokens` to catch silent drift).
   - Streaming support detection.
   - Cost-per-token metadata for the budget guard.

3. **`CachedTeacher`** — a wrapper that satisfies queries from the local logit cache (§3.3) before falling through to an inner `LogitSource`. Always present in the stack.

The `Logit Source` is independent of the loss granularity. The trainer asks the source what it can deliver and picks `full_vocab` if available, `teacher_top_k` otherwise.

**Why this matters for the masses:** *the same `POST /v1/train/opd` endpoint works whether you have a 27B model loaded locally, a $5/month OpenRouter sub, or a friend with an H200 sharing logits over Tailscale.* That is the unlock.

### 3.3. The Logit Cache — `kiln-train::logit_cache`

**Where:** new module backed by an on-disk store (RocksDB or sled).

**Key:** `(teacher_id, tokenizer_hash, prefix_hash, position_offset_within_response)`.
**Value:** top-K logprobs, with K being the maximum K the source ever returned for that key (so a later request asking for fewer K is satisfied trivially, and a later request asking for more K *misses* and re-fetches at the wider K).

**Why this destroys the cost objection:**

- A typical "distil a domain into a LoRA" job re-uses the same prompt set across 5–20 epochs of student rollouts. With prefix-keyed caching, the *teacher* logits for fixed prefixes are computed once and reused forever.
- For canonical seed corpora (math: OpenThoughts3, GSM8K; code: BigCodeBench; instruction: Tulu3) we ship the cache *prepopulated* — the kiln distribution comes with a tarball of ~50M cached top-32 logits from Qwen3.6-27B against a curated 100K-prompt seed set. That tarball is ~12 GB. Users get reproducible OPD against frontier teachers **with zero API spend**.
- For everyone-else cases, the cache is local and personal; no privacy concern.

**Optional: shared community cache.** A flagged-off, opt-in mode that uploads only `(prefix_hash, top_K_logprobs)` tuples — never the prompt or response text — to a community-run cache. First user to OPD against `(Qwen3.6-27B, "Solve the integral...")` pays the API call; the next million don't. **This is the Hugging Face datasets pattern, applied to teacher logits.** The privacy model is strong: one-way prefix hashes, no plaintext, no metadata about the requester.

### 3.4. Behaviour-space adapter merge — `kiln-server::api::adapters::distill_merge`

**This is the user's first seed idea, and it is the killer feature.**

Today, `POST /v1/adapters/merge` with `kind: linear|ties|concat` does *weight-space* arithmetic. Each of those merges loses information: linear averages away domain-specific peaks; TIES drops 80% of the weights below a density threshold; concat blows up rank.

**Behaviour-space merge via OPD does not.** Each source LoRA is a *teacher* over its original prompt distribution. The merged adapter is a student trained to match each teacher *on the prompts that teacher was good at.* This is the DeepSeek-V4 multi-teacher recipe applied at the laptop scale.

The API:

```http
POST /v1/adapters/distill_merge
Content-Type: application/json

{
  "name": "unified-coder",
  "sources": [
    {"adapter": "rust-helper",   "weight": 1.0},
    {"adapter": "python-helper", "weight": 1.0},
    {"adapter": "sql-helper",    "weight": 0.7}
  ],
  "student": "base",                   // or "rust-helper" for continual learning
  "rollout_budget": 5000,              // total student trajectories
  "loss": "teacher_top_k",             // auto-selected from source caps
  "stable_opd": "auto",                // engages automatically if needed
  "post_eval": ["coder-eval"]          // existing post_eval flywheel
}
```

What kiln does under the hood:

1. **Reconstructs the prompt distributions.** Every adapter trained in kiln is born with a provenance record (training history, eval scores). Extend the provenance to also retain the prompts that produced it (we already have them — they came in via `/v1/train/sft` or `/v1/train/grpo`). For each source, we know which prompts trained it and at roughly what density.
2. **Builds a routed prompt minibatch.** Each step samples a prompt according to a weighted mixture over the source distributions. The teacher for that prompt is whichever source LoRA "owns" it (resolved by source-of-origin, with weights as tie-breaker). Multi-teacher OPD weighting (`Σ w_i · KL(π_θ ‖ π_E_i)` from DeepSeek-V4) is used when a prompt is genuinely shared between sources.
3. **Runs OPD with full diagnostics.** Emits an *adapter-coverage report* showing per-source overlap-ratio convergence, so the user can see "rust-helper teacher overlap converged at 89%, sql-helper stalled at 61% — consider raising sql-helper weight or adding more sql prompts."
4. **Hot-swaps the merged adapter** atomically when training completes, identical to the existing merge endpoint's behaviour.

The retained-data argument is what makes this clean: we are not asking the user to find prompts post-hoc. *Kiln already has them.* Per the user's words: "how great that we keep all the training data for each LoRA!"

For the case where source LoRAs came from elsewhere (uploaded via `POST /v1/adapters/upload`), the merge endpoint accepts an explicit `prompts` field per source, or kiln synthesises prompts by sampling the LoRA against a generic seed corpus and using the LoRA's own outputs as a soft preference signal — degraded but workable.

### 3.5. The 27B → 4B Knowledge Pump — `kiln-server::api::distill::pump`

**This is the user's second seed idea, fully realised.**

Three modes, all behind one endpoint, each producing a downloadable LoRA:

#### 3.5.1. Targeted-domain pump

```http
POST /v1/distill/pump
{
  "name": "math-frontier-lora",
  "teacher": "qwen3.6-27b@openrouter",   // or any LogitSource alias
  "domain": "math_reasoning",            // canonical kiln-curated corpus
  "rank": 64,
  "rollout_budget": 50000,
  "stable_opd": "auto",
  "use_cache": true                      // public + local cache hit-through
}
```

`domain` resolves to a kiln-curated seed corpus — for `math_reasoning`, that's a deduplicated mix of DeepMath, GSM8K, MATH, OpenThoughts3 prompts (~100K prompts). The cache is prepopulated against the canonical teacher list. **For canonical (domain × teacher) pairs, the user pays $0 and waits hours, not days, for a frontier-quality math LoRA.**

Canonical domains shipped on day one:

- `math_reasoning` (DeepMath / OpenThoughts3 / MATH)
- `python_codegen` (BigCodeBench / HumanEval-Plus / CodeContests)
- `rust_codegen` (CommitPack-Rust / leetcode-rust / kiln's own Rust corpus)
- `instruction_following` (Tulu3)
- `chinese_writing` (DeepSeek-V4's evaluation domains)
- `clinical_notes` (subset of MIMIC-style synthetic clinical text)
- `legal_drafting` (Pile of Law subsets)
- `scientific_writing` (S2ORC abstracts + ArXiv body excerpts)
- `tool_calling` (Gorilla / xLAM)
- `long_context_summarization` (LongBench-V2 prompts)

Each domain ships with:
- A curated prompt seed (open license).
- A held-out eval suite registered automatically (`coder-eval`, `math-frontier-eval`, etc., wiring into the existing eval system).
- Recommended hyperparameters tuned per teacher (Stable-OPD weights, LoRA rank, learning rate from the LoRA Without Regret 10× rule).

#### 3.5.2. Wide-corpus generalist pump

A single overnight job that distils 27B's *general* behaviour into a generalist r=128 LoRA across a balanced multi-domain seed corpus (~200K prompts, balanced across the canonical domains above + chat). This is the "I just want my 4B to feel more like 27B" button.

The Lu trick of training many epochs on the *same* prompt also applies here — for prosumer/laptop tier we can squeeze more from a smaller seed by 5× repeating with fresh student rollouts each time. Lu's experiment showed near-teacher AIME'24 from a *single* prompt repeated 20 times. We let users dial the prompt-coverage / depth tradeoff.

#### 3.5.3. Auto-domain pump from user examples

```http
POST /v1/distill/pump
{
  "examples": [ ...10–50 example user prompts that characterise the user's domain... ],
  "teacher": "qwen3.6-27b@local"
}
```

Kiln embeds the examples (using its own embedding head — the existing FLCE work positions kiln to do this efficiently), retrieves the top-N nearest canonical seed prompts, optionally synthesises supplemental prompts via the teacher itself (with strong dedup), and runs the targeted-domain pump against the resulting custom seed.

#### 3.5.4. Data-multiplier mode (for users with tiny seed datasets)

Lu's most underappreciated experiment: training Qwen3-8B-Base on math with on-policy distillation against a **single, randomly chosen prompt**, repeated for 20 sequential steps with batch 256, approximately matched the teacher's AIME'24 score. **5,120 graded sequences from one prompt produced near-frontier performance.** This is not a fluke — it is a property of dense reverse-KL supervision: every rollout produces O(N tokens) of teacher signal, so a small prompt set repeated many times yields more bits than a large prompt set seen once with sparse RL.

Kiln exposes this as a first-class mode for the laptop user with 5–100 prompts to spare:

```http
POST /v1/distill/pump
{
  "name": "my-style",
  "prompts": [...10–100 prompts...],
  "teacher": "qwen3.6-27b@local-or-best-cached-source",
  "data_multiplier": "auto"          // engages when |prompts| < 200
}
```

What kiln does:

1. Detects small prompt count.
2. Switches sampling: instead of one-rollout-per-prompt-per-step, does *many rollouts per prompt per step*, sweeping the same prompts across many steps with fresh student samples each time.
3. Bumps `samples_per_prompt` from 4 → 16–64 depending on prompt count.
4. Tracks training KL per prompt independently — flatlined prompts get dropped early; still-learning prompts get more rollout budget.
5. Caps total rollout budget per prompt to prevent overfitting.

**Combined with the cache from §3.3, a user can produce a useful personal LoRA from a single afternoon of writing examples.** This puts OPD in reach of a 10-prompt "I want my model to write like this" use case — something no other open-source tool currently delivers.

**Why this combination works specifically for kiln's user base:**

- The single-model focus means we don't have to amortise pump infrastructure across an open zoo of student architectures. Every kiln user is a Qwen3.5-4B user. Cache hits are universal.
- The `same arch, same tokenizer` thing means the canonical seed corpora can be tokenised once and shipped with offsets baked in. No per-user retokenisation. No tokenizer-drift failure mode.
- Prosumers with the teacher locally and laptop users with the teacher remote produce identical LoRAs given identical seeds. **Reproducibility across the tier hierarchy.**

### 3.6. Continual learning & the "always up to date" loop — `kiln-server::api::distill::refresh`

Lu's paper shows that on-policy distillation is the cleanest known tool for continual learning: alternate phases of (1) midtrain on new knowledge, (2) OPD against the *previous* checkpoint of yourself to recover degraded post-training behaviours.

Kiln already preserves adapter checkpoints with provenance. We add a `refresh` endpoint:

```http
POST /v1/distill/refresh
{
  "name": "company-assistant",
  "new_data": {"dataset": "q4-2026-internal-docs"},
  "behavioural_teacher": "company-assistant@v17",   // a prior self
  "background_chat": "tulu3"                         // entropy-preserving mixer
}
```

What kiln does:

1. **Mid-trains** the named adapter on `new_data` mixed with `background_chat` per the Lu recipe (70/30 default, sweepable).
2. **Diagnoses** the IF-eval / instruction-following degradation against the eval suites registered for this adapter.
3. **OPD-recovers** using the prior checkpoint as teacher, on Tulu3-flavoured prompts (or whatever `background_chat` resolves to).
4. **Eval-validates** — refuses to publish the refreshed adapter unless IF-eval recovers within X% of the pre-refresh score *and* a new-knowledge eval suite improves by Y%. Both thresholds configurable.

This is what the user means by "your model gets better every time you use it" carried through to its logical end: the model never *forgets* either, because OPD is the forgetting antidote.

### 3.7. Recipes: the 1-click cookbook — `kiln-server::api::recipes`

We ship a small set of pre-baked recipes that compose the primitives. Each recipe is a single endpoint that orchestrates many under the hood. Example recipe spec:

```yaml
# crates/kiln-server/recipes/coding-assistant-from-repo.yaml
name: coding-assistant-from-repo
inputs:
  repo_path: { type: directory, required: true }
  teacher:   { type: logit_source, default: qwen3.6-27b@local-or-best-cache }
  rank:      { type: int, default: 64 }

steps:
  - kind: synthesise_seed
    from: repo_path
    output: prompts
    strategy: code_explain_and_complete   # uses the teacher to generate diverse coding prompts about the user's repo

  - kind: cold_start
    base: base-instruct
    sft_data:
      from: prompts
      teacher: ${inputs.teacher}
      rollouts_per_prompt: 1
    output: cold-start-lora

  - kind: opd
    base: cold-start-lora
    teacher: ${inputs.teacher}
    prompts: ${prompts}
    loss: teacher_top_k
    stable_opd: auto
    output: ${inputs.name}

  - kind: post_eval
    suite: coder-eval
    adapter: ${inputs.name}
    require_min_score: 0.55
```

Day-one recipes:

- `coding-assistant-from-repo` — the example above.
- `merge-my-loras` — wraps `distill_merge` with an interactive A/B picker.
- `update-with-new-docs` — wraps `distill/refresh`.
- `frontier-pump` — wraps `distill/pump` with a guided domain picker.
- `recover-instruction-following` — minimal single-step OPD against the base Instruct model; the Lu personalisation recipe.
- `make-a-judge-lora-from-my-picks` — connects the existing judgment flywheel to OPD: distil the implicit reward function from the user's A/B picks into a small judge LoRA.

The desktop app surfaces these as *cards* — pick one, fill in the blanks, watch the live diagnostic dashboard, get a downloadable LoRA at the end.

### 3.8. The Diagnostic Stack — `kiln-train::diagnostics`

Live, per-step metrics emitted to the existing `/metrics` Prometheus endpoint and rendered in the dashboard:

- **Overlap ratio** (Li et al. equation 6) — fraction of student top-K that overlaps with teacher top-K, averaged over the rollout. Healthy runs climb 70% → 90%; failing runs stagnate.
- **Overlap-token advantage** (Li et al. equation 7) — average advantage on the overlap region.
- **Entropy gap** |H(q) − H(p)| — narrows in healthy runs.
- **Truncation rate** (Luo et al. §3.2) — fraction of rollouts hitting the length budget without emitting EOS. Spike to 1.0 = collapse imminent.
- **Repetition rate** (Luo et al. compression-ratio detector) — `|raw_bytes(suffix)| / |zlib(suffix)|` > τ = repetition.
- **Reverse KL per position** — heatmap; identifies which positions in long rollouts the teacher is already failing to score reliably (Li et al. §6.1).
- **Teacher-cache hit rate** — operational metric.
- **Cost spent** (for remote teachers) — running tally with budget alerts.

The dashboard (the existing kiln `/ui`) gets a new **Distillation** tab with these as live charts, side-by-side with the loss curve. Same UX as the existing Training tab.

**Beyond passive display: kiln pushes.** Every diagnostic state change with user-actionable consequence triggers a desktop notification (via the existing kiln-desktop tray channel), an email if configured, and a webhook event. The notifications are exhaustively enumerated in §8.12 — there are exactly four, designed to be the bare minimum signal a user needs to act on a real change.

**Every metric in the dashboard has a "Why does this matter?" link** opening a one-paragraph plain-English explanation citing the originating paper. Overlap ratio links to Li et al. §4. Repetition rate links to Luo et al. §3.4. Per-position KL links to Li et al. §6.1. The user becomes an OPD expert by clicking around their own runs.

### 3.9. The Failure-Mode Rules Engine — `kiln-train::guardrails`

Every known OPD failure mode in the corpus → a rule with auto-mitigation. Rules fire on the diagnostic stream.

| Rule | Trigger | Auto-mitigation |
|---|---|---|
| `LengthInflation` | RepRate > 0.05 over 2 validation passes (Luo et al.) | Bump `λ_sft` and `β_kl`; engage golden-trajectory mixture if not already on |
| `OverlapStagnation` | Overlap ratio Δ < 0.01 over 30 steps (Li et al.) | Pause, recommend off-policy cold-start; refuse to continue without operator confirmation |
| `EntropyGapWidening` | |H(q)−H(p)| trending up after step 50 | Reduce learning rate by 0.5× automatically |
| `LongTailRewardDecay` | Per-position KL increases monotonically past position 7K (Li et al. §6.1) | Cap rollout length at the position where the teacher signal degrades; retrain |
| `TokenizerDrift` | Re-tokenisation of remote-teacher prompts disagrees by >0.5% on prompt_tokens count | Hard fail with a precise diff message; don't pretend it'll work |
| `CostCeiling` | Remote teacher spend exceeds budget | Pause; switch to cached-only mode if cache fills the gap; otherwise notify |
| `ThinkingPatternMismatch` | Initial overlap < 0.3 even after cold-start (Li et al. §3.1) | Refuse the run; explain the chosen teacher is stylistically incompatible; suggest alternates from kiln's compatibility table |

These run continuously and are visible in the dashboard. Every mitigation logs a rationale referenced back to the originating paper. **The user sees not just "we changed your hyperparameter" but "we changed it because Luo et al. (Stable-OPD) shows this prevents repetition collapse."**

**Auto-checkpoint and auto-rollback.** Luo et al.'s phase-transition collapse can wipe out hours of training in a 30-step window if unmonitored. Kiln auto-checkpoints the LoRA every 10 steps (configurable; 5 for corporate tier) and keeps the last 5 checkpoints in the adapter store. When any guardrail fires within 20 steps of the most recent checkpoint, the trainer:

1. Pauses optimisation.
2. Reports the failure mode in plain English on the dashboard and via desktop notification (§8.12).
3. Rolls back to the most recent checkpoint that passed the diagnostic gates.
4. Re-engages the appropriate mitigation (e.g., lowers LR, raises β_kl, switches to cold-start).
5. Either auto-resumes (if `on_collapse: "auto-resume"` is set) or holds for user confirmation (default).

**The user can leave a kiln OPD job running unattended and trust they will not return to a broken adapter.** The single most important UX guarantee in the plan, elaborated in §8.7.

### 3.10. The Adapter Library and the Logit-Cache CDN — `kiln-server::api::library`

Two layered network artefacts.

**Adapter Library** — a public, opt-in distribution of pre-trained kiln adapters. Each adapter ships with:

- The provenance trail (which teachers, which prompts, which hyperparameters).
- Eval suite results (kiln eval format).
- A reproducibility receipt: `kiln distill reproduce <adapter>` reconstructs it from the same teacher + same seed + same hyperparameters. When the source teacher is reachable (cached, hosted, or local), the receipt is *verifiable*.

**Logit-Cache CDN** — see §3.3. The shared community store that drives the cost of canonical-pump runs to zero.

Both are *opt-in* both ways: opt-in to download from, opt-in to upload to. We make uploading attractive by giving uploaders priority access to the latest pump runs / Adapter Library entries.

### 3.11. The Self-Hosted Teacher Marketplace (P2P, optional, Phase 5+)

For the ambitious version of the laptop tier: peer-to-peer logit serving. A user with an H200 can flag their kiln node as a teacher provider for Qwen3.6-27B (or any other supported teacher). Other users discover them via a directory and route their `RemoteTeacher` requests through.

Privacy model:

- Default: encrypted in transit only. Provider sees plaintext prompts. (This is identical to using OpenRouter today.)
- Optional: prompts go through a kiln-side **token-redaction stage** (PII, secrets) before being sent.
- Aspirational (Phase 6+): **logit-only oblivious inference** — a homomorphic / MPC-flavoured protocol where the provider returns logits without seeing the prompt. Out of scope for v1; mentioned because the literature is moving fast and we should not foreclose this option in the API design.

Pricing optional. The default mode is *gift economy*, modelled on the BitTorrent / Folding@Home ethos. Teacher providers earn priority on community cache pulls and Adapter Library access.

### 3.12. Privileged-Information self-distillation — `kiln-server::api::distill::self`

For the user with no teacher access and no compute budget: train against *yourself with extra information.*

Modes (from the survey):

- **GT-conditioning self-distill** (Zhao et al., OPSD): use user-verified examples as a privileged-information context for the teacher copy of the model.
- **Conciseness self-distill** (Sang et al., CRISP): condition on "be concise" to teach concision; the survey reports 57% token reduction at +9% accuracy.
- **Document-as-PI** (Stein et al., GATES): a retrieval context the teacher sees but the student doesn't.
- **Reverse-teacher self-distill** (Kim et al., RLRT): flip preference order to encourage self-driven reasoning.

These all degrade gracefully to "OPD with the model itself as teacher, plus an asymmetric information advantage." They cost nothing beyond a doubled rollout. They unlock a third use case: **OPD without any teacher at all.**

---

## 4. New API surface

The complete list of new endpoints, in keeping with kiln's `POST /v1/<verb>/<noun>` style:

| Method | Endpoint | Purpose |
|---|---|---|
| `POST` | `/v1/train/opd` | Mirror of `/v1/train/grpo` but with `teacher: <logit_source>` instead of `groups + scored completions` |
| `POST` | `/v1/distill/pump` | The 27B → LoRA knowledge pump (§3.5) |
| `POST` | `/v1/distill/refresh` | Continual-learning refresh (§3.6) |
| `POST` | `/v1/adapters/distill_merge` | Behaviour-space LoRA merge (§3.4) |
| `POST` | `/v1/distill/self` | PI-based self-distillation (§3.12) |
| `POST` | `/v1/recipes/run` | Run a named recipe (§3.7) |
| `GET`  | `/v1/recipes` | List available recipes |
| `GET`  | `/v1/teachers` | List configured `LogitSource`s and their capabilities (§3.2) |
| `POST` | `/v1/teachers` | Register a new `LogitSource` (e.g. an OpenRouter API key) |
| `GET`  | `/v1/library` | Browse the public Adapter Library (§3.10) |
| `POST` | `/v1/library/install/{adapter_id}` | Download and register a public adapter |
| `POST` | `/v1/library/publish/{adapter_name}` | Publish an adapter to the library (with reproducibility receipt) |
| `GET`  | `/v1/cache/stats` | Logit cache size, hit rate, hottest prefixes |
| `POST` | `/v1/cache/import` | Import a prebuilt cache tarball |
| `POST` | `/v1/cache/export` | Export the local cache (for sharing or backup) |

The shape of `POST /v1/train/opd`:

```http
POST /v1/train/opd
Content-Type: application/json

{
  "prompts": [
    {"messages": [{"role":"user","content":"..."}]},
    ...
  ],
  "teacher": "qwen3.6-27b@openrouter",      // alias resolved via /v1/teachers
  "loss": "teacher_top_k",                   // or "full_vocab" or "sampled_token"
  "top_k": 32,
  "samples_per_prompt": 4,
  "temperature": 1.0,
  "top_p": 0.9,
  "max_tokens": 4096,
  "stable_opd": "auto",                      // {auto, off, {kl: 0.01, sft: 0.1}}
  "rollouts": 5000,
  "post_eval": ["math-frontier-eval"],
  "name": "math-pump-2026-05-15"
}
```

Same hot-swap semantics as SFT/GRPO. The next inference call uses the new adapter.

---

## 5. Phased rollout

Six phases. Each phase ships independently usable value. Each phase *also* unlocks the next by contributing infrastructure.

### Phase 0 — Foundation (4–6 weeks)

The minimum that makes "OPD with a local teacher" work end-to-end.

- `kiln-train::opd` core loop: reverse KL, γ=0, top-K renormalised loss.
- `LocalTeacher` `LogitSource` impl.
- `POST /v1/train/opd` endpoint, mirroring `/v1/train/grpo`.
- Diagnostic metrics 1–4 from §3.8 wired into Prometheus + dashboard.
- `LengthInflation` guardrail wired in.
- One recipe: `recover-instruction-following`. The Lu personalisation example, faithfully reproduced. **First demonstrable user value: I have a niche-finetuned 4B and `distill/refresh` against the base Instruct model recovers IF-eval from 45 → 83.**

### Phase 1 — Local teacher polish + Stable-OPD (4 weeks)

- Async teacher scheduling, KV reuse across student rollouts, FP8/FP4 teacher quantisation.
- Full Stable-OPD: golden-trajectory mixture distillation, β_kl reference penalty, auto-tuning.
- All 7 guardrails active.
- Recipes: `merge-my-loras` (behaviour-space merge — pull this forward because it's the user's seed idea and it's killer with just local teachers).
- The `Distillation` tab in the dashboard with all live charts.

### Phase 2 — Hosted-logits abstraction + cache (6 weeks)

- `RemoteTeacher` adapters for vLLM, llama.cpp, sglang, OpenRouter, Together, Fireworks, DeepInfra, TGI (eight is the minimum useful set).
- `kiln-train::logit_cache` with prefix-keyed RocksDB store.
- Cost estimator + budget guards.
- Tokenizer-drift detection.
- `/v1/teachers` registration UI in the dashboard.
- Recipe: `frontier-pump` for prosumers/laptops who want to pull from hosted teachers.

### Phase 3 — Knowledge Pump + Adapter Library (8 weeks)

- All canonical domains shipped with seed corpora and prepopulated cache tarballs.
- `POST /v1/distill/pump` with all three modes (targeted / wide / auto-domain).
- Adapter Library hosting (initially a kiln-managed S3 bucket with a CDN; longer term, a federated index).
- `/v1/library/{install,publish}` flows.
- Reproducibility receipts.

### Phase 4 — Multi-teacher full-vocab + corporate features (8 weeks)

- `full_vocab` loss path with the DeepSeek-V4 efficient-teacher-scheduling design (cache last-layer hidden states; one prediction head loaded at a time).
- `POST /v1/adapters/distill_merge` extended to many-teacher (>2) consolidation.
- FP4 QAT path through kiln-marlin-gemm and the Vulkan kernel.
- Deterministic kernel paths for reproducibility (kiln's existing batch-invariance work positions this well).
- Corporate-tier templates: "DeepSeek-V4-style specialist-then-consolidate", "FP4 deployment ready".

### Phase 5 — Network effects (continuous after Phase 3)

- Community Logit Cache CDN.
- Self-Hosted Teacher Marketplace (the discovery directory + the gift-economy layer).
- Public benchmark leaderboard for distilled adapters: cost-per-eval-point as the primary metric, not raw score. **The first leaderboard in the industry that rewards efficiency, not just compute spend.**

### Phase 6 — Frontier (research-grade)

- Privileged-information self-distillation suite (§3.12).
- Logit-only oblivious inference protocol exploration.
- Continual-OPD scaling study — extend the Lu continual-learning experiment to multi-domain sequential learning over months of real user data.
- Active-learning OPD: kiln picks the next prompts to teach by maximising expected information gain on a held-out eval set.

---

## 6. Algorithmic choices, with citations

The defaults below are not arbitrary; each one is the conclusion of a paper in the corpus.

| Choice | Default | Source |
|---|---|---|
| Loss function | Per-token reverse KL | Lu (2025); MiniLLM (Gu et al. 2024) |
| Discount factor γ | 0 | Lu (2025) — variance dominates the bias gain |
| Granularity | Teacher top-K with support renormalisation, K=32 | Fu et al. (2026) — +19.8% over sampled-token |
| Rollout sampling | top-p=0.9, temp=1.0 | Fu et al. (2026) ablation table 3 |
| Special-token masking | On | Fu et al. (2026) — orthogonal but consistent gain |
| Reference-KL penalty β | Auto, default 0.01 | Luo et al. (2026) — Stable-OPD ablation |
| Golden-trajectory mixture λ | Auto, default 0.1 | Luo et al. (2026) |
| Cold-start before OPD | **Auto-injected** when initial-overlap probe < 0.5 (user never opts in) | Li et al. (2026) §5.1; Lu (2025) [^support] |
| Teacher-aligned chat template | Required | Li et al. (2026) §5.2 |
| Rollout length cap | 7K tokens default; raises require explicit flag and a degradation-curve run | Li et al. (2026) §6.1 — reward decays past this point |
| LoRA rank — laptop | 16 (capacity-calculator can lower to 8 in data-multiplier mode) | LoRA Without Regret (Schulman 2025) — OPD on rollouts is bits-per-episode bounded; rank 16 holds ample capacity for personal-data LoRAs |
| LoRA rank — prosumer general | 32 (capacity calculator can raise to 64–128 if bits-needed exceeds capacity) | LoRA Without Regret — comfortable headroom; smaller-than-folklore rank is correct because OPD bits-per-episode are dense but per-domain content is small |
| LoRA rank — corporate full vocab | 128–256 | LoRA Without Regret — large datasets approach the capacity floor |
| LoRA target | All linear layers (MLP + attention + MoE if present) | LoRA Without Regret — attention-only consistently underperforms |
| LoRA learning rate | 10× the FullFT optimum | LoRA Without Regret — across 14 model sweep |
| LoRA α | 32 | LoRA Without Regret — community standard, robust |
| Continual-learning recovery | Reverse-distill from prior self | Lu (2025) personalisation experiment |
| Multi-teacher consolidation | Weighted reverse KL with per-prompt routing | DeepSeek-V4 §5.1.2 |
| Teacher hidden-state caching | One prediction head loaded at a time | DeepSeek-V4 §5.2.2 |
| Loss kernel | Fused `kiln-opd-loss-kernel` (CUDA / Vulkan / Metal); single-submit; resident-buffer on Vulkan from day 0 | §9.2; tracks PR #1030 pattern |
| Samples per prompt | Adaptive: 4 if `|prompts|≥200`, 16 if 50–200, 64 if <50 | Lu (2025) data-multiplier — single-prompt-many-epoch matches teacher AIME |
| Batch size | Auto: largest power-of-2 fitting in VRAM, capped at 64 | LoRA Without Regret — both LoRA and FullFT achieve best loss at smaller batches |
| Auto-checkpoint cadence | Every 10 OPD steps; retain last 5 (5 steps for corporate tier) | Luo et al. (2026) — phase-transition collapses can wipe a 30-step window |
| Auto-rollback policy | On any guardrail trigger within 20 steps of last passing checkpoint | Pragmatic — user must never wake up to a broken adapter |
| Cost cap (remote teachers) | Hard cap at user-set ceiling, default $25; pause on cap; cache fall-through option | Pragmatic — no surprise bills, ever |
| First-run dry-run | Mandatory on first use of any remote `LogitSource`; estimates $ and wall-clock before any token sent | Pragmatic — informed consent before paid API calls |
| Adapter promotion gate | Eval gate required; auto-promote off by default | Pragmatic — bad runs do not regress production traffic |

When a default is wrong for a given user, kiln tells them which paper says so, and offers the alternative.

---

## 7. The three killer workflows, scripted out

### 7.1. "Pump frontier brilliance into a domain LoRA" (laptop, ~$5)

```bash
# 1. Register a hosted teacher (one-time).
curl -X POST http://localhost:8420/v1/teachers \
  -H 'content-type: application/json' \
  -d '{"alias":"qwen3.6-27b@openrouter","kind":"openrouter",
       "model":"qwen/qwen-3.6-27b","api_key_env":"OPENROUTER_API_KEY",
       "top_k_max":20}'

# 2. Pump.
curl -X POST http://localhost:8420/v1/distill/pump \
  -H 'content-type: application/json' \
  -d '{"name":"math-frontier","domain":"math_reasoning",
       "teacher":"qwen3.6-27b@openrouter","rank":64,
       "rollout_budget":50000,"use_cache":true,
       "post_eval":["math-frontier-eval"]}'

# 3. Wait. Watch the diagnostic dashboard. Approve the post-eval.
# 4. The next inference uses the new LoRA.
curl http://localhost:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"messages":[{"role":"user","content":"Evaluate ∫_0^∞ e^{-x²} dx"}],
       "adapter":"math-frontier"}'
```

**Cost:** with ~70% cache hit against canonical-domain prepopulated cache, ~15K teacher API calls × ~1K prompt tokens × $0.0003/1K = ~$5. Time: 2–6 hours on a 4090.

### 7.2. "Merge my last 5 LoRAs into one unified assistant" (prosumer, no API)

```bash
# Behaviour-space merge — uses each adapter's stored prompt history.
curl -X POST http://localhost:8420/v1/adapters/distill_merge \
  -H 'content-type: application/json' \
  -d '{"name":"unified",
       "sources":[
         {"adapter":"rust-helper"},
         {"adapter":"python-helper"},
         {"adapter":"sql-helper"},
         {"adapter":"writing-helper"},
         {"adapter":"math-frontier"}
       ],
       "rollout_budget":20000,
       "stable_opd":"auto",
       "post_eval":["coder-eval","math-frontier-eval"]}'
```

What the user sees in the dashboard:

```
Source coverage report (live):
  rust-helper      overlap  92%  ✔ converged
  python-helper    overlap  89%  ✔ converged
  sql-helper       overlap  61%  ⚠ stalling — 'sql' prompt subset only 8% of mix
  writing-helper   overlap  88%  ✔ converged
  math-frontier    overlap  95%  ✔ converged

  → kiln auto-suggestion: re-run with weights [1, 1, 2.0, 1, 1] to bring sql up
  → or: add prompts from your sql training history (kiln knows where they are)
```

One click to re-run with the suggested weights. No spreadsheets, no pickled state.

### 7.3. "Update with Q4 docs without losing instruction-following" (corporate, on-prem)

```bash
# 1. Upload the new corpus.
curl -F name=q4-2026 -F format=jsonl -F file=@q4_internal.jsonl \
  http://localhost:8420/v1/eval/datasets/upload

# 2. Refresh.
curl -X POST http://localhost:8420/v1/distill/refresh \
  -H 'content-type: application/json' \
  -d '{"name":"company-assistant",
       "new_data":{"dataset":"q4-2026"},
       "behavioural_teacher":"company-assistant@v17",
       "background_chat":"tulu3",
       "require_if_eval_recovery":0.95,
       "require_internal_qa_gain":0.05}'
```

Kiln midtrains, diagnoses the IF degradation, OPD-recovers against `v17`, then post-evals. **Refuses to publish v18 unless both gates pass.** v17 stays live until v18 ships, atomically.

---

## 8. The pit of success — how kiln auto-defends user success

§7 showed three users winning. §8 explains *how* — by enumerating the ways users hurt themselves with OPD today and the specific design choices that prevent each one. The principle from §0 made concrete.

### 8.1. The eight ways users currently fail at OPD

Every one is a real failure mode reported in the corpus or seen in production. Every one has an automatic mitigation in kiln.

| # | Who | What they do | What happens | What kiln does |
|---|---|---|---|---|
| 1 | Casual fine-tuner | Pulls a hosted teacher; runs naive sampled-token OPD; checks back next morning | Hits length-inflation collapse at step 30 (Luo et al.); trains for hours on garbage rollouts; final adapter unusable | Stable-OPD on by default + auto-checkpoint every 10 steps + auto-rollback + desktop notification at the moment of the trigger |
| 2 | Capacity-mismatched fine-tuner | Picks frontier teacher, picks rank-8 LoRA, expects miracle | Overlap ratio stagnates; user concludes "OPD doesn't work" | Capacity calculator (§8.5) runs *before* the job; suggests rank or scope adjustment with a number ("this run overflows rank-8 by ~3×") |
| 3 | Thinking-pattern-mismatched fine-tuner | Picks a Non-Think teacher to distil into a Think-mode student because "bigger model = better teacher" (Li et al. §3.1) | Initial overlap < 0.3; cold-start can't close the gap; final adapter is *worse* than the starting point | Compatibility table (§8.4) refuses the pair by default; suggests a known-good alternative; explicit `--force` to override |
| 4 | Over-eager merger | Tries to merge 10 unrelated-domain LoRAs into one rank-32 adapter | Single LoRA can't store all; result is mediocre across all domains | Pre-merge compatibility scoring (§3.4 coverage report) flags the over-stuff; suggests narrower merge OR higher rank with concrete bits-required-vs-available number |
| 5 | Forgetting fine-tuner | Continues training adapter on new data with raw SFT | IF-eval crashes (Lu's mid-train regression curve); model becomes useless on chat | The `/v1/distill/refresh` endpoint is the *default* path for any "add new knowledge" intent; raw SFT-on-instruct-adapter requires explicit acknowledgement of the regression risk |
| 6 | Cost-surprised user | Forgets budget cap; runs an expensive remote teacher overnight | Wakes to a $400 bill | Mandatory dry-run on first use of any `LogitSource`; hard cost cap at $25 default; pause on cap; cache fall-through option |
| 7 | Reproducibility-broken user | Trains a great adapter; tries to recreate it three months later | Cache pruned, teacher API updated, hyperparameters ad-hoc — unreproducible | Reproducibility receipt (§8.11) on every adapter; `kiln distill verify <adapter>` re-runs and bit-checks; teacher snapshots pinned by version-hash, not name |
| 8 | Diagnostics-blind user | Never opens the dashboard; doesn't see warnings | Bad runs ship to production | Notifications push to desktop tray, email, and webhook; high-severity alerts (collapse imminent, cost cap reached, eval regression) interrupt the user's normal channels |

### 8.2. The single front door — `POST /v1/train`

Today a user has to know whether they want SFT, GRPO, or OPD before they can pick an endpoint. That's a knowledge gate at the door. Kiln introduces a single intent-aware front door:

```http
POST /v1/train
{
  "intent":   "make this model better at writing in my style",
  "examples": [...],         // optional — triggers SFT path if alone
  "scored":   [...],         // optional — triggers GRPO path
  "teacher":  "...",         // optional — triggers OPD path
  "name":     "my-style"
}
```

Kiln picks the right pipeline from the inputs:

| Inputs present | Pipeline kiln picks |
|---|---|
| `examples` only | SFT |
| `scored` (groups + rewards) | GRPO |
| `teacher` + `prompts` | OPD |
| `teacher` + `prompts` + `scored` | Hybrid OPD-per-token + GRPO sequence advantage (§12 research direction; gated until validated) |
| `name` of an existing adapter + `new_data` | `distill/refresh` |
| `sources: [adapters...]` | `adapters/distill_merge` |

The granular endpoints (`/v1/train/sft`, `/v1/train/grpo`, `/v1/train/opd`, etc.) remain for power users and backward compatibility. **The default front door does not require the user to know which paradigm fits their problem.** Kiln picks. The dashboard shows which it picked and why, with a "switch to X instead" button for the curious.

### 8.3. The twelve knobs users should never need to touch

For each knob, kiln picks based on a rule. The rule is auditable in the dashboard. The knob is overridable for power users.

| # | Knob | Auto-pick rule |
|---|---|---|
| 1 | Loss type (`sampled_token` / `top_k` / `full_vocab`) | From teacher capabilities — `full_vocab` if local, `top_k` if remote |
| 2 | `top_k` | min(teacher_max_topk, 32) — Fu et al. ablation optimum |
| 3 | Learning rate | 10× FullFT-optimum-for-this-model from Schulman's 14-model sweep |
| 4 | Batch size | Largest power-of-2 fitting in VRAM, capped at 64 — Schulman's small-batch finding |
| 5 | LoRA rank | Capacity calculator (§8.5): bits-required(prompts × samples × top_k) → rank with 2× headroom |
| 6 | `samples_per_prompt` | Adaptive from prompt count: 4 / 16 / 64 — Lu's data-multiplier curve |
| 7 | β_kl (Stable-OPD reference) | Auto-tuned by repetition-rate guardrail — starts 0.01, doubles on RepRate>0.05 |
| 8 | λ_sft (golden mixture) | Auto-tuned similarly — starts 0.1, doubles on RepRate>0.05 |
| 9 | `top_p` | 0.9 — Fu et al. ablation table 3 |
| 10 | Temperature | 1.0 — same |
| 11 | Max rollout tokens | min(7K, teacher_reliable_length) — Li et al. §6.1 reward-decay curve |
| 12 | Cold-start | Auto-inject if initial-overlap probe < 0.5 — Li et al. §5.1 + Lu's [^support] |

For comparison: a typical OPD framework today exposes 8–12 of these as required user input with no defaults. **Kiln exposes zero as required.** Power users can set any subset.

### 8.4. The compatibility table, pre-populated and shipped

Kiln 1.0 ships with an empirical table of (teacher × student × domain) combinations validated end-to-end on representative hardware. Each row records: predicted overlap-at-step-50, recommended rank, recommended cold-start length (epochs), expected GPU-hours, expected $ cost on top-3 hosted teacher providers, and the eval suite the validation used.

When a user starts an OPD job, kiln looks up the closest row:

- **In-table:** kiln uses the validated config and tells the user "this is a validated combination; expected cost $X and Y hours; expected eval improvement Z points."
- **Out-of-table:** kiln finds the nearest neighbour, warns about the deviation, and runs an extended initial-overlap probe to estimate fitness before committing the user's budget.

Day-one entries: ≥30 combinations covering the full canonical-domain × hosted-provider × tier matrix. Continuously expanded by community OPD runs that opt-in to upload anonymised diagnostic traces (same opt-in path as the community logit cache).

**The user picks the validated row. The user does not pick hyperparameters.** This is "pit of success" in concrete terms.

### 8.5. The capacity calculator — a number before any commitment

Before any OPD job consumes budget, kiln computes:

```
bits_needed                  ≈ rollouts × tokens × log2(vocab_top_k)
bits_storable_in_lora        ≈ rank × layer_count × hidden_dim × 2_bits_per_param   # Allen-Zhu 2024
expected_overlap_at_step_50  ≈ f(initial_overlap_probe, capacity_ratio)             # learned regressor over the compatibility table
expected_eval_delta          ≈ g(expected_overlap, domain_difficulty)               # validated by §8.4
```

Two warnings can fire:

- `bits_storable < 0.3 × bits_needed` → "this run overflows rank-X by ~Yx; consider rank Z or shorter rollouts; relevant paper section: <link>"
- `expected_eval_delta < 1 point` → "this run is unlikely to materially improve evals; consider a different teacher or a stronger cold-start"

Both warnings are interruptible — the user can press through. **They cannot do so accidentally.**

### 8.6. The cost lock — no surprise bills, ever

For any `RemoteTeacher`, the first-ever run requires a *dry-run*. Kiln estimates total token spend (using cache hit-rate prediction, typically 60–95% for canonical-domain runs), multiplies by the provider's published $/token, and shows the user "this run will cost approximately $X over Y hours."

A hard cost cap (default $25, configurable) applies *per training job*. When the cap is hit, kiln pauses, surfaces a dashboard alert and a desktop notification, and offers three options:

- **Resume in cache-only mode** (fall through to the cache; no new logits fetched)
- **Raise the cap by $N** (re-confirm)
- **Cancel and keep the partial adapter** (still useful as a checkpoint)

Default response: pause and wait for human. **No surprise bills.**

For local teachers, no cost lock; instead a wall-clock cap with the same three options.

### 8.7. The auto-rollback contract

The contract: *the user can leave a kiln OPD job running unattended for any duration and trust that they will not return to a broken adapter.* Specifically:

- The active adapter serving traffic is **never** the in-flight training adapter; it is the most recent adapter that passed both the diagnostic gates *and* the post-eval gate.
- If the post-eval is missing or pending, the prior adapter remains active.
- A new adapter promotes to "active" only after explicit user approval from the dashboard A/B view *or* configured auto-promote thresholds (default off).

A 12-hour OPD job that collapses at hour 8 results in the user logging back in to find: the model still serving, the dashboard showing what happened with the rollback timeline, and a "Resume from rollback" / "Try with adjusted config" button. **Never downtime, never a regression in production traffic.**

### 8.8. The "Why?" surface — every auto-decision is auditable

Every panel in the dashboard, every line in the training log, every hyperparameter shown in adapter provenance has a "Why?" link. Clicking opens a modal with:

- The rule kiln used.
- The paper section that supports it.
- The specific values kiln observed (e.g., "initial overlap probe returned 0.41; threshold for cold-start auto-inject is 0.5; cold-start engaged with 2 epochs").
- A "Override this decision" button exposing the underlying knob with sensible bounds.

**Auditing kiln's choices does not require reading the OPD literature.** Reading the literature does, however, deepen understanding of the same auditable choices. The two are complementary, not prerequisite.

### 8.9. Data-multiplier mode — auto-engaged for tiny datasets

Documented in §3.5.4. Surfaced here because the UX win is structural: the floor for OPD usefulness drops from "you need 10K prompts" to "you need 10 prompts that represent what you want." That alone changes the user demographic kiln can serve. The user does not opt in; kiln detects small prompt count and engages silently, with a one-line dashboard banner: "Data-multiplier mode active because you have 47 prompts — kiln is sampling each prompt 32× per epoch instead of 4× (Lu et al. 2025 §Discussion)."

### 8.10. The auto-cold-start mandate — surfaced

Documented in §3.1. Surfaced here because it is the single most common silent failure of naive OPD: running reverse-KL when the student has near-zero probability mass on tokens the teacher prefers. The fix is forward-KL SFT first (Lu's [^support] footnote made operational by Li et al.'s phenomenology paper). Kiln auto-injects this when the initial-overlap probe demands it. **The user sees one progress bar, not two.**

### 8.11. The reproducibility receipt — every adapter is rebuildable

Every adapter shipped from kiln carries a JSON receipt:

```json
{
  "adapter": "math-frontier",
  "produced_at": "2026-05-15T18:43:11Z",
  "kiln_version": "0.42.7",
  "kernel_versions": {"cuda": "...", "vulkan": "...", "metal": "..."},
  "seed": 4218,
  "teacher": {
    "alias": "qwen3.6-27b@openrouter",
    "model_id": "qwen/qwen-3.6-27b",
    "model_version_hash": "sha256:...",
    "snapshot_url": "..."
  },
  "prompts": {
    "source": "kiln-canonical:math_reasoning:v3",
    "manifest_hash": "sha256:..."
  },
  "hyperparameters": {...},
  "diagnostic_summary": {
    "overlap_ratio_final": 0.92,
    "rep_rate_max": 0.01,
    "guardrail_triggers": []
  },
  "post_eval": {
    "math-frontier-eval": 0.71
  }
}
```

`kiln distill verify <adapter>` re-runs the recipe against the same teacher snapshot and reports either "bit-identical" (deterministic kernel paths, same hardware), "evaluations equivalent within 1%" (same recipe, hardware drift), or "diverged" (something in the receipt has changed). **Any kiln adapter can be rebuilt from its receipt** — given access to the same teacher (locally, via cache, or via the same paid provider). This is the foundation for the Adapter Library's trust model (§3.10).

### 8.12. The four desktop notifications kiln will send unprompted

These are the only push channels kiln uses by default. All are dismissable and disable-able. Designed to be the bare minimum signal a user needs to act on a real change.

1. **"Training collapsing — kiln will auto-rollback at step N if it doesn't recover. Click for details."** Pre-collapse warning; user has time to override or accept the rollback plan.
2. **"Cost cap reached on `<teacher>` — paused. Cache hit rate is N% — resume in cache-only mode?"** Cost-pause with a one-click resume option.
3. **"Adapter `<name>` is ready and improved your eval by N points — click to A/B against the current adapter."** Post-eval notification; the user is one click from a side-by-side comparison.
4. **"Cache update available: +N new logits for the canonical `<domain>` corpus from the community CDN — pull?"** Opt-in cache refresh; declining is silent.

**The user receives no other notifications by default.** No "training has started" pings, no metric churn updates. Every notification represents an action the user could take that materially affects outcome.

### 8.13. Tier-aware defaults — the concrete table

For engineering reference. These are the values kiln picks per tier when the user provides no overrides.

| Setting | Laptop tier | Prosumer tier | Corporate tier |
|---|---|---|---|
| Default `LogitSource` | Best-cached → `RemoteTeacher` | `LocalTeacher(qwen3.6-27b, fp8)` | `LocalTeacher(qwen3.6-27b, full)` ×N |
| Default loss | `teacher_top_k`, K=20 (most APIs cap) | `teacher_top_k`, K=32 | `full_vocab` |
| LoRA rank | 16 | 32 | 64–256 (capacity-calculator-set) |
| Batch size | 8 | 16 | 32–64 |
| `samples_per_prompt` (default-data path) | 4 | 4 | 4 |
| `samples_per_prompt` (data-multiplier triggered) | 32 | 32 | 16 (rarely needed at scale) |
| Max rollout tokens | 4K | 7K | 7K (hard) / 16K (with degradation probe) |
| Auto-checkpoint cadence | Every 10 steps | Every 10 steps | Every 5 steps |
| Cost cap default | $10 | $25 | wall-clock cap instead |
| Cold-start auto-trigger threshold | overlap < 0.5 | overlap < 0.5 | overlap < 0.5 |
| Mixture distillation default fraction | 25% goldens | 25% goldens | 10% goldens |
| Eval gate before adapter promotion | Required | Required | Required + manual A/B sign-off |
| Notifications channel | Desktop tray + email | Desktop tray + email + webhook | Webhook + Slack/Teams |

These are auditable in `kiln.toml` and overridable globally or per-job. **A new user on any tier can do nothing but pick "Distill math from Qwen 3.6" and get a working adapter on the first try.** That is the success criterion for "pit of success."

### 8.14. The contract with the user, in one sentence

> *Tell kiln what you want. Kiln will pick everything else, run it, watch it, fix it if it breaks, eval it, and only show you the model when it is better than what you had — and explain every decision it made along the way if you ask.*

Everything else in this plan exists to make that sentence true.

---

## 9. Engine-native efficiency from day 0 — fast or it won't be used

The pit of success is necessary but not sufficient. **People won't do this unless it's fast and easy.** §8 covered "easy". §9 covers "fast" — specifically: fast on every engine kiln supports, from day one, with no second-class platform.

### 9.1. Why per-engine efficiency is the make-or-break

OPD adds three new compute hotspots on top of normal training:

1. **Student rollouts** — the model samples its own continuations. ~50% of a typical OPD step.
2. **Teacher logprob computation** — a forward pass returning per-position top-K logprobs. ~30% of a step when local; network-bound when remote.
3. **Reverse-KL loss** — per-token KL between student and teacher distributions over the top-K renormalised support. ~10% of a step **but easily 3–5× slower with a naive implementation** that materialises full-vocab tensors.

Plus the existing gradient + optimizer cost, ~10%.

If any one of these is slow on any engine, the user perceives the whole experience as slow. The slowest engine becomes the brand. **Kiln cannot let that happen.**

### 9.2. The fused OPD-loss kernel — `kiln-opd-loss-kernel`

A new crate, sibling to `kiln-flce-kernel`, with three implementations and identical numerical results. The kernel does, in one fused pass:

1. Read student last-layer hidden state (already in registers from forward pass).
2. Project to logits in chunks (don't materialise full V × hidden product).
3. Extract teacher top-K support indices (passed in as a small int32 tensor).
4. Gather student logits at those K indices.
5. Renormalize student & teacher logprobs over the K support.
6. Compute per-token KL.
7. Compute the gradient with respect to student logits, restricted to K positions.
8. Backprop into the projection weight (chunked, no full V materialisation).

**Memory cost:** O(B × T × K) instead of O(B × T × V). For Qwen3.5-4B with V=152K and K=32, that's ~5000× less.
**Time cost:** ~ one cross-entropy forward+backward (comparable to FLCE).

**Per-engine implementation:**

| Engine | Approach | Existing kiln experience to leverage |
|---|---|---|
| **CUDA** | One fused CUDA kernel, modelled on the flash-attn-style "online softmax + matmul fused" pattern. Same blockwise reduction tricks `kiln-flce-kernel` already uses for cross-entropy. Reference for numerical correctness. | `kiln-flce-kernel`, `kiln-marlin-gemm`, `kiln-flash-attn` are directly transferable. |
| **Vulkan** | One fused compute shader, **shipping with `dispatch_opd_loss_resident` from day 0**, following the resident-buffer pattern that PR #1030 establishes for decode-hot kernels. Accepts `&VulkanBuffer` for student hidden states and teacher top-K logprobs; writes loss scalar and per-position gradient into caller-provided buffers. Single-submit per token batch; no host transfers. The existing vk-native training path (`docs/vk_native_training.md`) already keeps every activation, gradient, and optimizer state resident; the OPD loss kernel slots in as another `VkBackwardOp` in that framework. | Recent vulkan work (`causal_conv1d_prefill` single-submit, `gdn` chunkwise) is the closest analog. The vk-native training infrastructure exists. |
| **Metal** | One Metal Shading Language kernel using simdgroup primitives. MSL `simd_max` and `simd_sum` natively express the renormalization. Apple Silicon's unified memory means we avoid stage-and-copy patterns common on discrete GPUs. | Kiln's existing Metal inference path. |

**Numerical-equivalence test:** the same `(student_hidden, teacher_topk_logprobs, top_k_indices)` tuple must produce KL values within 1e-5 across all three engines. **A platform that isn't bit-equivalent doesn't ship OPD until it is.** No "tier-2 platform" stigma.

### 9.3. Reusing kiln's inference engine for student rollouts

The OPD trainer does **not** have its own sampling path. It calls into kiln's existing inference engine. OPD inherits, for free, on every engine:

- Paged KV cache (no fragmentation across rollouts)
- Prefix caching (rollouts that share a prompt share KV)
- FP8 KV cache option (doubles effective batch size)
- Continuous batching
- Chunked prefill

The hybrid 24 linear (GDN) + 8 full-attention architecture is *especially* favorable: GDN layers have constant per-token cost and small per-layer state, so long rollout sequences are cheap; the 8 full-attention layers are the only quadratic cost, and FP8 + paged KV compress that further.

**Critically, the OPD trainer's wall-clock improves automatically as the inference team's PRs land.** Vulkan PR #1030 (decode-side resident buffers) drops per-decode-call kernel overhead from ≈1.5 ms to ≤200 µs — a roughly 7× speedup on the rollout phase, which is ~50% of an OPD step. **OPD on Vulkan gets ~3× faster the day #1030 merges, with no OPD-side code changes.** This dynamic is a strategic asset; we explicitly architect around it (§9.10).

### 9.4. Local-teacher pipelining — three-way concurrency + UMA wins

When the teacher is loaded locally, three parallelism patterns are critical:

1. **Prefix-shared teacher prefill.** Multiple student rollouts off the same prompt → one teacher prefill, many teacher decodes. We re-use kiln's KV-cache infrastructure. Speedup: ~3–5× when `samples_per_prompt > 1`.

2. **Async overlap (NPD-style, survey §6.3).** Student rollout for batch N+1, teacher logprob compute for batch N, gradient compute for batch N-1 run concurrently. Three-way pipelined. CUDA: streams. Vulkan: queue-family separation. Metal: MTLCommandQueue concurrency. Target: ≥2× wall-clock vs sequential.

3. **Quantised teacher.** FP8 (CUDA marlin), Q4/INT4 (Vulkan + Metal), per-channel int8 (uniform fallback). Default teacher quantisation auto-picked from VRAM availability.

4. **One prediction head loaded at a time** (DeepSeek-V4 §5.2.2). For multi-teacher consolidation, each teacher's last-layer projection is loaded only when its rollouts come up in the dispatch queue. Critical for corporate-tier multi-teacher.

**Unified-memory hardware (Strix Halo APUs, Apple M-series) is structurally advantaged.** Teacher and student share a single memory pool — no inter-device copy, no host↔device staging. The vk-native training path's observation applies verbatim: *"the bytes never physically move — they just stop being accounted as anon-rss by the kernel."* For OPD specifically, a 4B student + an FP8 27B teacher can co-reside on a 64 GB Strix Halo or M3 Max box that *no discrete-GPU consumer hardware can match*. **A single AMD mini-PC or Apple Silicon laptop becomes a competitive OPD station.** This is a genuine kiln moat — and one of the strongest arguments for the Vulkan + Metal investment paying off independent of the CUDA path.

### 9.5. Remote-teacher batching — laptop tier optimization

When the teacher is an HTTP API, the bottleneck is request overhead and rate limits, not compute. Optimizations:

1. **Single-request batching.** Most providers accept N prompts per request. The `RemoteTeacher` adapter knows the largest accepted batch (vLLM ~256, OpenRouter ~16, llama.cpp configurable). Trainer sizes accordingly.

2. **Concurrent requests with backoff.** Per-provider rate-limit tracking with token-bucket backoff; users see steady throughput rather than oscillating crashes.

3. **Cache-pull-then-fill.** Local + community cache consulted first. For canonical-domain runs against popular teachers, 60–95% of positions hit cache. **The "fast" experience is cache hit; the "slow but cached for next time" experience is the miss.**

4. **Speculative-style prefetching** (survey §6.3 SKD inspiration). Trainer issues teacher requests for the next batch's prompts in the background while the current batch's gradient compute runs.

### 9.6. Loss-kernel determinism and reproducibility

Reproducibility receipts (§8.11) require bit-exact reproduction. Kiln's existing batch-invariance work (in `kiln-server`) is the foundation. The OPD-loss kernel inherits this design:

- All reductions use a fixed accumulation order (no atomic adds).
- All sampling uses a per-rollout PRNG seeded from `(global_seed, prompt_hash, rollout_idx)` — rollouts are reproducible regardless of ordering.
- A "fast mode" (opt-in, default off) allows non-deterministic atomics for ~10–15% speedup; receipts mark such adapters as "evaluations equivalent within 1%" rather than "bit-identical".

### 9.7. Per-engine performance targets — two columns, with the trajectory

For Qwen3.5-4B student + Qwen3.6-27B teacher in OPD `top_k_renormalised` mode, batch 16, sequence 4K:

| Engine | Hardware target | Today | Post-near-term-trajectory |
|---|---|---|---|
| CUDA | 1× RTX 4090 (24 GB) | ≥1500 t/s cached / ≥600 t/s local FP8 teacher | ≥2000 / ≥800 (FP4 paths) |
| CUDA | 8× H200 | ≥50,000 t/s multi-teacher full-vocab | ≥80,000 |
| Vulkan | 1× 7900 XTX (24 GB) | ≥600 t/s cached / ≥250 t/s local Q4 teacher | **≥1200 cached / ≥500 local** (post-PR #1030 ~2× from resident-decode) |
| Vulkan | Strix Halo (64 GB UMA) | ≥400 t/s cached / ≥150 t/s local FP8 27B (UMA wins) | ≥800 cached / ≥350 local |
| Metal | M3 Max (40-core, 64 GB) | ≥800 t/s cached / ≥300 t/s local Q4 teacher | ≥1100 / ≥450 |
| Metal | M2 (10-core, 16 GB) | ≥200 t/s cached only | ≥350 cached |

Gates for shipping each engine. **A platform that misses its gate doesn't ship until it does.** Bench infrastructure lives in `bench-results/` and `crates/kiln-server/src/bench.rs`; we extend it with a per-engine OPD bench.

### 9.8. The "is this run going to be fast enough?" preview

Tied into the §8.5 capacity calculator and §8.6 cost lock: before the user commits, kiln estimates wall-clock on their hardware *and on alternative hardware they could rent*:

```
Estimated time on this machine (4090, CUDA): 4h 12m
  - Student rollouts: 2h 30m
  - Teacher logprobs (cache 73% hit): 50m
  - Loss + gradient: 52m
Equivalent run on:
  - 8× H200 (corporate tier): 12m
  - M3 Max: 11h 40m
  - Strix Halo (UMA, FP8 local 27B): 6h 20m
  - 7900 XTX (Vulkan, post-#1030): 5h 50m
```

Sets expectations honestly, lets users decide overnight-here vs cloud-elsewhere. **No surprise wall-clock either.**

### 9.9. The bench gate — perf as a CI feature

We extend `bench-results/` with a per-engine OPD benchmark suite that runs every PR. It tracks tokens/sec for each §9.7 target and **fails CI if any engine regresses by >5%**. Performance is treated as a feature with the same regression-test rigor as correctness. This is how we keep the promise of fast-on-every-engine forever — by enforcing it mechanically, not by hoping.

### 9.10. Riding the Vulkan trajectory — strategic free wins

PR #1030 ("Vulkan Resident Decode") and its successors lift the host↔device boundary from per-kernel to per-step in the Vulkan inference engine. Targets in that PR:

- Per-decode-call kernel overhead 1.5 ms → ≤200 µs (~7× reduction).
- Decode tok/s at batch=1 ≥80% of llama.cpp.
- Decode tok/s at batch=64 ≥66% of vLLM/sglang CUDA.

**Because OPD reuses the inference engine for student rollouts (§9.3), every Vulkan inference improvement lands as an OPD speedup with zero OPD-side code change.** This is a 1+1=3 dynamic between the two workstreams — and one we explicitly architect around:

1. **Design the OPD loss kernel with the resident-buffer pattern from day 0**, composing cleanly with the resident decode that #1030 establishes. Single-submit kernels all the way through the OPD step.
2. **Ship perf gates per §9.7 as two columns** (today / post-trajectory) and explicitly track our distance to the upper column over time. Regression on the upper column is treated as importantly as the lower.
3. **The OPD product roadmap is partially de-risked by the Vulkan inference roadmap** — we can promise faster-OPD-on-Vulkan as a near-term roadmap item with high confidence, because the underlying work is already in flight by another team and our integration is just "use the new dispatch variant".
4. **The same logic applies to vk-native training improvements** — the GDN backward kernels, gradient checkpointing for hybrid models, the solve_tri shader replacement (Phase 7 follow-ups in `vk_native_training.md`) — all of them flow directly into OPD speed on Vulkan as they land.

The user will not see any of this complexity. They will see a steadily-decreasing wall-clock estimate in the dry-run preview as kiln updates ship. **The Vulkan-engine investment compounds with the OPD-engine investment, and vice versa.** The same dynamic exists weakly between CUDA-inference and OPD (we ride CUDA inference wins for free too) and especially strongly on Metal (kiln's Metal path is comparatively young; every kernel ported is a direct OPD speedup).

---

## 10. Failure modes anticipated, and where each is addressed

A fully populated table that the engineering team can tick off.

| Failure mode | Source paper | Mitigation in kiln | Where in this plan |
|---|---|---|---|
| Length inflation / repetition saturation | Luo et al. 2026 | Stable-OPD: golden mixture + KL ref penalty + auto-trigger | §3.1, §3.9 |
| Flawed-prefix collapse | Survey + Li et al. 2026 | Top-p sampling + truncation rate guardrail + length cap | §3.1, §3.9 |
| Thinking-pattern mismatch | Li et al. 2026 §3.1 | Off-policy cold-start; teacher compatibility table; hard refuse | §3.7, §3.9 |
| Tokenizer / special-token mismatch | Fu et al. 2026 | Same arch advantage + special-token masking + drift detector | §1.2, §3.1, §3.9 |
| Globally informative reward, locally cancelling gradients | Li et al. 2026 §6.2 | Per-position KL heatmap; capped rollout length | §3.8, §6 |
| Teacher reliability decays past 7K | Li et al. 2026 §6.1 | Auto length cap; explicit override required | §3.9, §6 |
| Self-play saturation | Survey §7.2 | PI self-distill is one-shot, not iterative; iterative variants behind explicit flags | §3.12 |
| Diversity collapse from reverse KL mode-seeking | Survey §7.2 | Adaptive divergence option (ToDi/AKL) for open-ended tasks | §6 (future default) |
| Cost runaway on hosted teachers | Pragmatic | Budget guard; cache fall-through; hard pause | §3.2, §3.9 |
| Variance from sampled-token estimator | Fu et al. 2026 | Top-K renormalised default | §3.1 |
| Forgetting on continual learning | Lu 2025 | OPD against prior self in the refresh recipe | §3.6 |
| Capacity gap (teacher too strong for tiny LoRA) | Busbridge et al. 2025 + LoRA Without Regret | Per-tier rank defaults; capacity warning when bits-required > rank-capacity | §6 |

We are not planning around hopes. Every known failure has a named, paper-cited mitigation that ships on by default.

---

## 11. The network effects

Three compounding loops kiln unlocks once Phase 3 ships.

**The cache loop.** First user to OPD against (Qwen3.6-27B, prompt P) pays the API call. Their cache contribution makes the next thousand users pay nothing. The marginal cost of frontier-quality OPD trends to zero with the size of the community.

**The library loop.** Adapters published with reproducibility receipts can be re-built from the cache. Adapters in the library get more downloads when their eval scores are good. Kiln becomes the place to get trustworthy, reproducible 4B specialists for any domain. The library acts as a discovery surface for the cache: popular adapters mean popular prompts mean populated cache.

**The judge loop.** The existing kiln judgment flywheel turns A/B picks into a local judge LoRA. Combined with OPD: the judge LoRA can serve as an *implicit reward model* in OPD-augmented RLHF (the AlignDistil idea from the survey). Now every kiln user is producing both a personal assistant *and* a personal judge — and the judge can be the basis for further OPD on the assistant. **This is the closest thing to a personal RLHF that anyone has shipped.**

The combined effect: kiln gets disproportionately better as more people use it, in a way frontier APIs cannot match because their economics don't permit free amortised teacher inference.

---

## 12. How we'll know it worked

Concrete success criteria, gated by phase.

**Phase 0 success (~6 weeks in):**
- Reproduce Lu's recovery experiment: a kiln 4B mid-trained on internal docs degrades IF-eval; `distill/refresh` recovers it from <50% to ≥80% with no new internal-QA loss.
- **`kiln-opd-loss-kernel` ships on CUDA, Vulkan, and Metal**, bit-equivalent within 1e-5, all three meeting the per-engine speed gates in §9.7's "today" column. **No engine ships before its gate is met.** A >5% perf regression on any engine fails CI thereafter.
- Vulkan path uses `dispatch_opd_loss_resident` (resident-buffer pattern from PR #1030) from the first commit — no later "Vulkan retrofit" needed.

**Phase 1 success (~10 weeks):**
- `distill_merge` of three domain LoRAs (math, code, instruction) outperforms the best of the three on each of their respective held-out evals while keeping all of them within 5% of single-source performance. **Beats `linear` and `ties` weight-space merges by ≥10 absolute points on average.**

**Phase 2 success (~16 weeks):**
- A laptop user on a 16GB MacBook completes `frontier-pump` against a hosted Qwen3.6-27B teacher in ≤8 hours and ≤$10, producing a domain LoRA that scores within 5% of the same pump run from a corporate H200 box (using the same cache). **The reproducibility-across-tiers guarantee, met empirically.**
- **Pit-of-success metric:** in a controlled study with 20 first-time kiln users (none with prior OPD experience), ≥18 successfully complete an OPD run that improves their target eval, with **zero manual hyperparameter tuning**, within their first 30 minutes of using kiln. Failures (if any) are auto-rolled-back; no user lands on a broken adapter.
- **Vulkan parity check:** the same canonical `frontier-pump` recipe completes on a 7900 XTX (Vulkan) within 25% of the wall-clock of an equivalent 4090 (CUDA) run, validating that the trajectory in §9.10 is materialising. By Phase 5 the gap should be ≤10%.

**Phase 3 success (~24 weeks):**
- Adapter Library has 50+ published adapters across the 10 canonical domains.
- ≥1M cache hits served in the first month after launch.
- Median cost-per-AIME'24-point on the public leaderboard is **half** that of the cheapest non-kiln distillation pipeline.

**Phase 4 success (~32 weeks):**
- Reproduce a DeepSeek-V4-style specialist-then-consolidate run on a single 8×H200 box with 6+ specialist teachers, producing a 4B unified adapter that beats Qwen3.6-27B on at least one corporate-scenario internal eval.

**Phase 5+ success (continuous):**
- More than half of new adapter publications use cached teacher logits (i.e., $0 marginal teacher cost).
- Cumulative API spend across the kiln community is *less than* the cumulative compute saved by the cache versus uncached baselines, by an order of magnitude.

---

## 13. Open research frontiers kiln can lead

Kiln is the right place for the field to advance because we control the full stack and can run cheap, repeatable studies.

1. **Distillation scaling laws for LoRA.** The Busbridge work doesn't yet have an answer for the LoRA-adapter regime. We have everything we need (knob: rank, knob: prompts, knob: teacher size via choice of `LogitSource`) to publish a clean `Quality ∝ N_T^α · N_S^β · D^γ · R^δ · LoRA_rank^ε` study using Qwen3.5-4B as the substrate.
2. **Active OPD.** Use a held-out eval suite to choose the next batch of prompts to roll out. We have the eval system, the rollout system, and the score system in one process. No one else does.
3. **Per-position teacher reliability calibration.** Li et al. show teacher signal degrades past 7K tokens. We can publish the full calibration curve and a method that *learns* per-position trust weights from cheap probe rollouts.
4. **Adapter compatibility prediction.** Given (rank, source domain, target domain), predict the overlap-ratio convergence point of a `distill_merge` run *before* spending the rollouts. Save users the merge that won't work. This is a regression problem on the data kiln itself generates.
5. **OPD + GRPO hybrid for verifiable tasks.** Combine per-token reverse KL (dense, from teacher) with sequence-level GRPO advantage (sparse, from a verifier) into a single update. The corpus calls for this; we are uniquely positioned to ship it as `POST /v1/train/grpo+opd`.
6. **Cross-tier reproducibility at scale.** Publish a benchmark showing that `(seed, teacher, prompts, hyperparameters)` reproduce the same adapter on a MacBook M4 Max, a 4090, and an H200 cluster. Open-source ML has no such benchmark today. Kiln deserves to own it.

Each of these is a publishable paper, with reproducible artefacts users can re-run for $5.

---

## 14. The closing vision

Today, AI personalisation looks like this: pay a hyperscaler $X/month, hand them your data, get a chatbot that may or may not do what you want, can't be inspected, can't be ported, will be deprecated when their pricing model changes.

Tomorrow — with kiln + OPD shipped well — it looks like this:

> Alice opens kiln on her MacBook. She drops in a folder of her writing. She picks `frontier-pump` against `qwen3.6-27b@best-cached-source`. Six hours and four dollars later she has a 200 MB LoRA that writes like her, runs entirely on her laptop, never phones home, and got verifiably better than the same run yesterday because of a quiet improvement to the teacher cache. She can publish it (with a reproducibility receipt — anyone can rebuild it from the same teacher and same data), sell it, gift it, version it, *or merge it with three more LoRAs from her colleagues into a team-wide adapter* with one button.
>
> Bob's startup has a 4090 in the back office. Their support agent runs on Qwen3.5-4B + a kiln adapter that was distilled from Qwen3.6-27B over their three years of support tickets. Each Friday, kiln runs `distill/refresh` against the past week's new tickets, recovers any IF degradation by self-distilling against last Friday's snapshot, and publishes the new adapter behind an A/B gate that the support team approves on Monday morning from `/ui`. Their model gets better every week. Their cost is electricity.
>
> Carol's bank has a rack of H200s. They run kiln in `full_vocab` mode with eight domain specialists each trained on a separate compliance vertical. The unified adapter, consolidated by `distill_merge`, beats their previous Qwen3.6-27B-based generic deployment on every internal eval at 1/7th the inference cost. The data never leaves the building. The auditor gets a reproducibility receipt for every model in production.

That world is not far away. The papers are written. The hardware exists. The base model exists. **Kiln is the missing piece — and it is 80% built.**

Let's build the other 20%.

---

## Appendix A — Bibliography (annotated)

The corpus this plan is grounded in. All seven files live in `docs/papers/on-policy-distillation/`.

- **`on_policy_distillation.md`** — Lu / Thinking Machines (Oct 2025). Canonical OPD framing, the recovery / personalisation experiment, the 9–30× and 50–100× compute-efficiency claims, the "RL searches in semantic-strategy space" intuition. **Foundational. Read first.**
- **`lora_without_regret.md`** — Schulman / TML (Sep 2025). The bits-per-episode argument, the rank-independence of optimal LoRA learning rate, the 10× LR multiplier rule, the "all linear layers" requirement, the ⅔ FLOP advantage. **Foundational for every LoRA decision in this plan.**
- **`2603.25562_revisiting_opd.md`** — Fu et al. (March 2026). Three failure modes of sampled-token OPD; teacher Top-K local support matching; +19.8% improvement; ablations on K, top-p, masking. **Source of the default loss granularity.**
- **`2604.08527_demystifying_opd.md`** — Luo et al. (April 2026). Length-inflation collapse mechanism; Stable-OPD with reference KL + golden mixture; +7.2% average accuracy. **Source of the auto-engaged stabilisers.**
- **`2604.13016_rethinking_opd.md`** — Li et al. (April 2026). Phenomenology / mechanism / recipe; thinking-pattern consistency; off-policy cold-start; reward decay with depth; diagnostic metrics (overlap, advantage, entropy gap). **Source of the diagnostic stack and the cold-start guardrail.**
- **`2604.00626_survey_on_policy_distillation.md`** — Song & Zheng (April 2026). The unified f-divergence framing; 100+ methods catalogued; KL-constrained RL equivalence; industrial deployments. **Source of the "what other methods we could swap in later" map.**
- **`deepseek_v4.md`** — DeepSeek-AI. Pure multi-teacher full-vocabulary OPD as the post-training paradigm; FP4 QAT; teacher hidden-state caching with one head at a time; specialist-then-consolidate pipeline. **The aspirational corporate-tier blueprint.**

A second-order reading list (for the engineering team), not in the corpus:

- *MiniLLM* (Gu et al. 2024) — original reverse-KL OPD via REINFORCE.
- *GKD* (Agarwal et al. 2024) — DAgger-for-LLMs framing.
- *DistiLLM-2* (Ko et al. 2024) — asymmetric per-source divergence; informs an optional knob.
- *Punica* (Chen, Ye et al. 2023) — multi-tenant LoRA serving; relevant when shipping the Adapter Library.
- *Busbridge et al. 2025* — distillation scaling laws; the open-question framework we extend.

---

## Appendix B — Glossary of one-line decisions

For the engineering kickoff:

- **Default loss:** teacher-Top-K reverse KL with renormalisation, K=32, γ=0.
- **Default sampler:** nucleus, top-p=0.9, temp=1.0.
- **Default stabilisers:** Stable-OPD `auto`; goldens at 25% of minibatch; β_kl=0.01.
- **Default LoRA:** rank 64, α=32, all linear layers, LR = 10× FullFT optimum.
- **Default rollout cap:** 7K tokens. Override requires explicit flag.
- **Default cache policy:** local on; community opt-in off; one-way prefix hashes only.
- **Default teacher when local 27B fits:** local FP8.
- **Default teacher when it doesn't:** highest-K source on the user's `/v1/teachers` registry, with cost-per-token tiebreak.
- **Default thinking-pattern check:** required; cold-start enforced when initial overlap < 0.5.
- **Default eval gate for `distill/refresh`:** require IF-eval recovery within 5% and new-knowledge gain ≥ 5%.
- **Default merge mode:** behaviour-space (`distill_merge`), not weight-space, when ≥2 source adapters and either has post-eval >0.6.

---

*End of plan. Godspeed.*
