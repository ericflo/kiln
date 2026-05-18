# The ECHO Integration Plan

> *Make environment-token cross-entropy a first-class loss term across the kiln agentic-RL stack — on by default, native everywhere, composed cleanly with OPD.*

**Status:** Draft. Branch `use-breakthrough-echo-grpo-technique-throughout`. The reference docs for the technique live at `docs/papers/echo/` (paper + blog).
**Authors:** Synthesised by Claude (Opus 4.7) from the ECHO paper (Shrivastava, Awadallah, Papailiopoulos — MSR AI Frontiers, 2026), the OPD grand plan, the agentic-grpo skill, and a deep read of the GRPO training stack as it exists on this branch.
**Date:** 2026-05-18.

---

## TL;DR

ECHO isn't an add-on to GRPO — it's the natural completion of the **agentic** training stack kiln was already moving toward. The OPD grand plan (`docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md` §10) already names the agentic loop as kiln's primary deployment shape, already requires multi-turn per-turn token masking, and already discusses tool-result tokens as "inputs the model didn't generate." ECHO takes the third step that plan leaves implicit: **those tool-result tokens aren't just inputs to mask out — they're a dense supervision target.** Mask out of the policy-gradient objective, mask *into* an auxiliary cross-entropy objective, share the same forward pass.

So the integration is less "bolt ECHO onto GRPO" and more "complete the masking primitive the agentic plan needs anyway, and turn it on by default." Once the mask-pair is plumbed through, the loss change is a few dozen lines per backend (and the kernel already exists in `kiln-flce-kernel` with CUDA / Vulkan / Metal / CPU coverage — that's the reason this is feasible at the timescale we'd want).

---

## §0 What we know after exploring (the load-bearing facts)

**The data path today** (file:line referenced throughout):

- `GrpoGroup { messages, completions: Vec<ScoredCompletion> }` and `ScoredCompletion { text: String, reward: f64 }` (`crates/kiln-train/src/lib.rs:209–232`). One string per multi-turn rollout, no role/turn structure preserved.
- `rollout.py` in the pi-doctest cap (`capabilities/agentic-grpo/pi-doctest/rollout.py:101–323`) concatenates assistant turns with a literal `<TURN_BREAK>` sentinel and drops tool-result content entirely.
- `tokenize_grpo_group` (`crates/kiln-train/src/trainer.rs:1915–1981`) emits `TokenizedGrpoCompletion { input_ids, completion_mask }` (`trainer.rs:1686`) where `completion_mask` is just "true after the prompt." It's used in four sites in `train_tokenized_grpo_group` (`trainer.rs:1744, 1794, 1808, 1844`).
- The same shape repeats verbatim in `crates/kiln-train/src/vk_train.rs:59–128` and threads through `vk_native_grpo_train` (`vk_train.rs:3471`) and `vk_native_grpo_train_jsonl` (`vk_train.rs:3749`). Both call `grpo_active_rows_and_labels(&comp.input_ids, &comp.completion_mask)` (`vk_train.rs:3623, 3959`).

**The kernel surface is already what we need.** `kiln-flce-kernel` exposes `fused_linear_cross_entropy_dispatch(hidden, head_t, input_ids, label_mask, …)` (`crates/kiln-flce-kernel/src/lib.rs:120`) and supports CUDA Phase B, Metal Phase B, Vulkan Phase B (via `CustomOp1`), and CPU Phase A. **It already accepts a per-token mask.** The env-CE term is literally a second invocation of this kernel with a different mask. Zero new kernel work.

**The right template already exists.** SFT's `label_mask_from_rendered_assistant_spans` (`trainer.rs:2423`) finds `<|im_start|>assistant\n…<|im_end|>` regions in chat-template-rendered text and builds the per-token mask SFT uses. ECHO's masks must be built the same way, just with one mask that fires on assistant spans and a second that fires on tool/observation spans.

**The blocker the agentic-GRPO skill itself flags is the foundation ECHO needs.** `capabilities/agentic-grpo/pi-doctest/kiln-polish-prerequisites.md` §1, and §0 of `.agents/skills/agentic-grpo-capability-creator/SKILL.md`, both already specify the exact change: extend `ScoredCompletion` with `trajectory: Option<Vec<TurnSegment>>` and have `tokenize_grpo_group` walk turns to build per-turn masks. The current pi-doctest cap shipped with v0=single-assistant-turn-per-rollout precisely to avoid this gap. **Filling the gap *is* the ECHO foundation; ECHO is what we earn by filling it.**

**The OPD plan precedent is right there.** `docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md` §10.4-A literally says "The teacher gets logprobs for the assistant tokens only; tool-result tokens are masked from the loss (they are inputs the model didn't generate)." That sentence already concedes the action/observation split; ECHO uses the same split to make observations a *target*.

### §0.1 Empirical findings (after the code audit)

These are claims verified by direct reads, not the original agent reports. They change a few details in §3 below.

- **`ScoredCompletion` / `GrpoGroup` / `GrpoRequest`** confirmed at `crates/kiln-train/src/lib.rs:201–232`. Field shape is exactly `text: String` + `reward: f64`. No `seed`, no `trajectory`, no per-token info.
- **`TokenizedGrpoCompletion { input_ids, completion_mask }`** at `crates/kiln-train/src/trainer.rs:1682–1687`. **Same struct exists in `vk_train.rs:57–60` as `TokenizedVkGrpoCompletion` with identical fields.** Two structs to update, not one.
- **`train_tokenized_grpo_group`** at `trainer.rs:1716`. The loss call site for the **uncheckpointed** path is line 1848 (`grpo_loss(&policy_log_probs, &ref_log_probs, loss_params, device)`). The full `[1, T, V]` logits tensor is materialized at line 1830 (`model_forward`) and is what `token_log_probs` (line 1841–1846) gathers active rows from. ECHO inserts as an extra term added to `loss` before `loss.backward()` at line 1851.
- **The checkpointed path is the harder one.** `train_tokenized_grpo_group` dispatches to `checkpointed_grpo_forward_backward` (`trainer.rs:7270–7281`) when `segments` is `Some`. That function computes the loss via `analytic_grpo_tail_loss_grad_pre_final_norm` at line 7434 — an analytic backward through the LM head + final RMSNorm that returns `(loss_val, upstream_grad)`. The upstream grad then propagates back through each segment. **ECHO has to fold into this analytic tail**: the env-CE term contributes both to `loss_val` and to `upstream_grad` (linear combination of two analytic gradients). This is a real implementation task, not "one new line." Likely a new function `analytic_grpo_echo_tail_loss_grad_pre_final_norm` taking `action_mask` and `env_mask`.
- **`tokenizer.encode_with_offsets`** exists today (`trainer.rs:2348-2349` in `tokenize_for_training`). We do **not** need a new `apply_chat_template_with_token_offsets` method — `apply_chat_template(...)` + `encode_with_offsets(...)` already give us what `build_masks_from_trajectory` needs. **Tokenizer extension scope shrinks accordingly.**
- **The existing assistant-span finder** (`label_mask_from_rendered_assistant_spans`, `trainer.rs:2423–2457`) finds `<|im_start|>assistant\n…<|im_end|>` blocks via byte-search. It has a fallback (`label_mask_by_prefix_tokenization`, line 2475) that re-tokenizes prefixes when the byte-span approach can't find all spans. ECHO's `build_masks_from_trajectory` generalizes this to find `<|im_start|>tool\n…<|im_end|>` blocks as well (or whatever the active chat template uses for tool roles — to be verified in a Phase 0 fixture test).
- **Metal training has no separate file.** Metal is selected as a candle `Device` and the same `trainer.rs::grpo_train` runs on it. There's a Metal-specific test at `trainer.rs:10359–10367` but no `metal_train.rs`. **ECHO's `trainer.rs` change covers CUDA / CPU / Metal simultaneously** — three of four backends in a single set of edits.
- **Vulkan training is genuinely separate.** `vk_train.rs::vk_native_grpo_train` (`vk_train.rs:3471`) and `vk_native_grpo_train_jsonl` (`vk_train.rs:3749`) run on Vulkan's own `VkTensor` type with a hand-rolled backward graph; `vk_train.rs` does **not** import or use `kiln-flce-kernel`. ECHO on Vulkan needs its own loss-term implementation using VkTensor ops (not a kernel rewrite — the math is `sum(log_softmax(logits)[env_positions, input_ids[env_positions+1]])` which is a few VK ops — but the wiring is not the same as the candle path). Treat this as ~half a day of focused VK work in Phase 1.
- **`opd.rs::opd_step_loss`** (`opd.rs:693`) is the precise structural template ECHO should follow. It takes `(student_hidden, head_t, label_mask, top_k, …)` and returns `OpdStepOutputs { per_position_kl, mean_kl, active_count }`. `echo_env_ce` should have the *exact same shape* — `(student_hidden, head_t, env_mask, chunk_size)` returning `EchoStepOutputs { per_position_ce, mean_ce, env_count }`. This makes the `LossConfig { policy, echo, opd }` composition not just nominal but structurally clean: two parallel `step_loss` calls with parallel signatures.
- **Pi session JSONL schema** confirmed by reading `capabilities/agentic-grpo/pi-doctest/rollout.py:85–98, 280–318`. Format is `{"type": "message", "message": {"role": "system|user|assistant|tool", "content": [{"type": "text|thinking|toolCall|toolResult", ...}]}}`. Tool-call args land under `b["input"]` (not `b["arguments"]` — earlier agent report was wrong; verified by reading the source). Tool result events are filtered out in the current rollout.py because v0 only trains on assistant turns — exactly the gap ECHO closes.

---

## §1 The framing: ECHO is what completes the agentic loop

If we squint at OPD plan §10:

| | Source of supervision | Where it lands in the rollout |
| --- | --- | --- |
| GRPO (today) | Outcome reward | Action tokens |
| OPD (§3.1) | Teacher logprobs | Action tokens |
| Verifier blend (§10.4 D) | Programmatic test | Whole trajectory |
| Local judge (§10.6) | Distilled GRM | Whole trajectory |
| **ECHO (proposed)** | **Environment itself** | **Observation tokens** |

ECHO fills the only remaining quadrant. Together with OPD, every position in the rollout now carries gradient: action tokens get policy-gradient + (optionally) teacher reverse-KL; observation tokens get environment-CE. **"Trajectories already contain a dense training signal waiting to be used"** (paper §7) and the existing kiln agentic deployment is *the* canonical place where that sentence is concretely true.

This is also why "make ECHO the default" is the right move and not premature: the paper's λ=0.05 self-anneals (as the model learns terminal structure, the CE term shrinks naturally) and the cost is genuinely zero — same forward pass, same activations, just a second masked log-prob sum. The downside risk is bounded; the upside is the same headline number that drove this whole branch (~2× TerminalBench-2.0 pass@1).

---

## §2 The big naming refactor (the "from day one" piece)

This is the change that does most of the "feels like the design from the beginning" work. Every other change is mechanical once this lands.

**Today's vocabulary** assumes there's only one kind of trainable position in a rollout — "the completion." So we have `completion_mask`, `ScoredCompletion`, and a tokenizer that builds the mask by "everything after the prompt." That vocabulary made sense when GRPO was a single-turn primitive.

**The replacement** distinguishes the two kinds of positions explicitly. The new vocabulary, used everywhere:

- `Trajectory` — replaces a single `text: String` with an ordered list of `TurnSegment { role, content, kind: TurnKind }` where `TurnKind` is `Action` (assistant generation), `Observation` (tool result / environment), or `Context` (system/user prompt). Already specified in `kiln-polish-prerequisites.md` §1.
- `action_mask: Vec<bool>` — true at positions the model generated; targets of policy gradient.
- `env_mask: Vec<bool>` — true at observation positions; targets of env-CE.
- (The legacy `completion_mask` becomes the union `action_mask | env_mask` and is kept around only as a compatibility shim during the deprecation window.)

The shape, in the core crate with a clean Serde schema:

```rust
// crates/kiln-core/src/trajectory.rs  (new file)

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind { Context, Action, Observation }

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TurnSegment {
    pub role: String,          // "system" | "user" | "assistant" | "tool"
    pub content: String,
    pub kind: TurnKind,        // Action | Observation | Context
    #[serde(default)]
    pub tool_call_id: Option<String>,
    #[serde(default)]
    pub warning_prefix_len: Option<usize>, // bytes of harness warning to exclude from env_mask
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScoredRollout {
    /// Authoritative representation.
    pub trajectory: Vec<TurnSegment>,
    pub reward: f64,
    /// Legacy fallback. Populated automatically from `trajectory` on serialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AgenticGroup {
    pub messages: Vec<ChatMessage>,
    pub rollouts: Vec<ScoredRollout>,
}
```

And then `GrpoGroup` / `ScoredCompletion` become *deprecated aliases* (with `#[serde(alias)]`) that convert into `AgenticGroup` / `ScoredRollout` on load. **Old payloads keep working; new ones get the masks for free.** The "from day one" feel comes from the new shape being the canonical one in code, with the old form reduced to a footnote in serde.

This same naming refactor flows into:

- `TokenizedGrpoCompletion` → `TokenizedRollout { input_ids, action_mask, env_mask }` (both `trainer.rs:1686` and `vk_train.rs:59`).
- `tokenize_grpo_group` → `tokenize_agentic_group` (both `trainer.rs:1915` and the VK parallel at `vk_train.rs:53–128`).
- `GrpoConfig` → `AgenticConfig` (with `Grpo` kept as a `#[deprecated]` alias), and the loss-relevant subset called `LossConfig` so OPD and ECHO knobs can compose orthogonally.

The principle: the old single-mask shape is the *special case* (`env_mask` is all-false; behavior identical to today). The new dual-mask shape is the *default*. This is the inversion that makes ECHO look like the design from the start.

---

## §3 The five architectural pillars

Numbered to mirror the OPD grand plan's §3, with file-and-line touch points so the scope is sanity-checkable.

### 3.1 — `kiln-train::echo` — the loss term, in two flavors

**Where:** new module `crates/kiln-train/src/echo.rs`, sibling to `opd.rs`. The shape mirrors `opd_step_loss` (`opd.rs:693`) deliberately — same signature shape, same return shape, so `LossConfig { policy, echo, opd }` composes structurally:

```rust
pub struct EchoConfig {
    pub lambda: f64,                  // default 0.05 (paper §3.3)
    pub env_mask_mode: EnvMaskMode,   // EnvOnly (default) | FullObs (debug-only)
    pub warning_filter: bool,         // default true; strip harness warning prefix
}

#[derive(Default)]
pub enum EnvMaskMode { #[default] EnvOnly, FullObs }

pub struct EchoStepInputs<'a> {
    pub tokens: &'a [u32],
    pub env_positions: &'a [usize],   // analog of OPD's active_positions
    pub student_hidden: &'a Tensor,   // [1, T, H] post-final-RMSNorm
    pub head_t: &'a Tensor,           // [H, V]
    pub total_obs_len: usize,         // |O| for length normalization per paper §3.1
    pub chunk_size: usize,
}

pub struct EchoStepOutputs {
    pub per_position_ce: Tensor,
    pub mean_ce: Tensor,
    pub env_count: usize,
}

pub fn echo_step_loss(inputs: EchoStepInputs<'_>) -> Result<EchoStepOutputs>;
```

For the **uncheckpointed path** (`trainer.rs:1830–1851`), the implementation is straightforward: call `fused_linear_cross_entropy_dispatch` from `kiln-flce-kernel` with `env_mask` as the label mask, then divide by `|O|` (paper §3.1 length normalization). The kernel already supports CUDA Phase B (`CustomOp1`), Metal Phase B (`CustomOp1`), and CPU Phase A. **One new line in `train_tokenized_grpo_group`** — `let loss = policy_loss + cfg.echo_lambda * echo_step_loss(...)?.mean_ce + ...;` — and the existing `loss.backward()` does the rest.

For the **checkpointed path** (`trainer.rs:7270 checkpointed_grpo_forward_backward`), it's harder. Today's flow:

```
final_hidden ─► analytic_grpo_tail_loss_grad_pre_final_norm
                  ├── compute loss_val (GRPO surrogate + KL)
                  └── compute upstream_grad = d(loss) / d(final_hidden)
                                                          │
                                                          ▼
              [back through 32 layers via per-segment recompute + backward]
```

The analytic tail materializes logits chunk-wise (via the same `head_t` matmul FLCE does), computes the GRPO surrogate analytically, and returns the gradient wrt `final_hidden`. For ECHO we need a sibling **`analytic_grpo_echo_tail_loss_grad_pre_final_norm`** that:

1. Computes the GRPO surrogate analytically on `action_mask` positions (same as today).
2. Computes the env-CE analytically on `env_mask` positions (new — but the math is just `-log p(input_ids[t+1] | logits[t])` summed and normalized).
3. Returns `(loss_total, upstream_grad_total)` where `upstream_grad_total = upstream_grad_grpo + λ · upstream_grad_env_ce`.

Both gradients are wrt `final_hidden`, so a linear combination is exact. Implementation: extend the existing analytic tail with two extra arguments (`env_mask: &[bool]`, `echo_lambda: f64`) and add the env-CE math inside the same vocab-chunk loop. Memory cost: zero additional intermediates — the env-CE per-chunk contribution overlaps with the existing per-chunk GRPO contribution.

**Why this is still ~"free":**

- `student_hidden` is already on device — it's exactly the tensor the existing analytic tail already takes.
- The vocab-chunk matmul (`head_t` × `hidden_chunk`) is amortized across action and env masks; the env-CE term piggybacks on the same logit chunks.
- Backward through the per-segment recompute is unchanged — `upstream_grad` shape is identical, just numerically different.
- Same FLCE toggles (`KILN_USE_FLCE`, `KILN_CUDA_FLCE`, `KILN_VULKAN_FLCE`) apply.

**For Vulkan-native** (`vk_train.rs::vk_native_grpo_train`, `vk_train.rs:3471`), the implementation is separate because `vk_train.rs` runs on `VkTensor`, not candle `Tensor`, and **does not import `kiln-flce-kernel`** (verified — `grep -n kiln_flce_kernel crates/kiln-train/src/vk_train.rs` returns nothing). The math is the same — log-softmax of the env-position logits, gather the target tokens, mean over `|O|` — but it's expressed via VK shader dispatches. This lands as a new `vk_train::echo_step_vk` function next to `grpo_active_rows_and_labels` (`vk_train.rs:181`). Treat as ~half a day of focused VK work in Phase 1; the algorithm is simple and reuses existing VK matmul / softmax dispatches.

### 3.2 — The masking layer — `kiln-train::trajectory_mask`

**Where:** new module `crates/kiln-train/src/trajectory_mask.rs`. Hosts the function that takes a `Trajectory` + tokenizer and emits `(input_ids, action_mask, env_mask)`.

This module is the *foundation* — both the existing multi-turn agentic-GRPO need and ECHO depend on it. It generalizes `label_mask_from_rendered_assistant_spans` (`trainer.rs:2423–2457`) from one role to multiple.

```rust
pub struct MaskConfig {
    pub warning_filter: bool,
    pub env_mask_mode: EnvMaskMode,
}

pub struct MaskedRollout {
    pub input_ids: Vec<u32>,
    pub action_mask: Vec<bool>,
    pub env_mask: Vec<bool>,
    pub segment_spans: Vec<(usize, usize, TurnKind)>,   // (token_start, token_end, kind) per segment
}

pub fn build_masks_from_trajectory(
    trajectory: &[TurnSegment],
    prompt_messages: &[ChatMessage],
    tokenizer: &KilnTokenizer,
    cfg: &MaskConfig,
) -> Result<MaskedRollout>;
```

**The implementation reuses the existing tokenizer methods.** `apply_chat_template` and `tokenizer.encode_with_offsets` both exist today (verified in `trainer.rs:1924, 2348`). The flow:

1. Concatenate `prompt_messages + trajectory` into one `Vec<ChatMessage>`.
2. `tokenizer.apply_chat_template(full_messages)` → `full_text: String`.
3. `tokenizer.encode_with_offsets(&full_text)` → `(input_ids, offsets: Vec<(usize, usize)>)`.
4. For each `Action` segment, find its rendered byte range by looking for `<|im_start|>assistant\n…<|im_end|>` blocks in `full_text` (the existing `label_mask_from_rendered_assistant_spans` byte-search logic, generalized over role).
5. For each `Observation` segment, find its rendered byte range by looking for `<|im_start|>tool\n…<|im_end|>` blocks (or whatever the active chat template uses for tool roles — verified in a Phase 0 fixture test).
6. Mark `action_mask[token]` true for every token whose `offsets[token]` overlaps an Action span, and `env_mask[token]` true for every token whose `offsets[token]` overlaps an Observation span.
7. The `warning_filter` (paper §3.2) trims the `env_mask` span: for each Observation segment with a non-None `warning_prefix_len`, advance the byte start by that many bytes before computing the overlap.

**Fallback path.** Mirror the existing SFT fallback (`label_mask_by_prefix_tokenization`, `trainer.rs:2475`): when the byte-search approach can't account for all expected role spans, re-tokenize cumulative prefixes to find each segment's exact token boundaries. This handles edge cases (custom chat templates, embedded XML, escape sequences) the same way SFT already does.

Then **`tokenize_grpo_group` becomes a thin shim** that calls `build_masks_from_trajectory` and packages the result. The single-string legacy path produces a `Trajectory` with one big `Action` segment, and behavior is identical to today. **No new tokenizer methods needed** — the audit found `encode_with_offsets` already exists.

**Open empirical question (resolved during Phase 0 fixture test).** Does the active chat template use `<|im_start|>tool\n` for tool-result blocks, or a different special-token pair? The plan handles both: `MaskConfig` carries the role→delimiter mapping, defaulting to `<|im_start|>{role}\n…<|im_end|>` for ChatML-style templates (which Qwen3 uses). A unit test on a real pi-session fixture in `crates/kiln-train/tests/fixtures/` pins the actual delimiters.

### 3.3 — `pi`-side trajectory capture — `capabilities/agentic-grpo/lib/`

**Where:** new shared Python module `capabilities/agentic-grpo/lib/pi_trajectory.py` (lifted from `pi-doctest/rollout.py`'s `parse_transcript` and generalized).

Today's `rollout.py:260–318` concatenates assistant turns into one string and drops tool results. The new module parses the pi session JSONL into the canonical `Trajectory` schema directly:

```python
def parse_pi_session(session_path: Path, harness: str = "pi") -> dict:
    """Returns {'trajectory': [{role, content, kind, warning_prefix_len?}, ...]}."""
```

`rollout.py` for any cap (not just pi-doctest) calls this and writes:

```json
{
  "messages": [...],
  "rollouts": [
    {"trajectory": [{role, content, kind}, ...], "reward": 1.0},
    ...
  ]
}
```

This is the *only* per-capability change required. After this lands, every existing capability gets ECHO for free — they just emit the new schema. The legacy `text`+`completions[]` schema continues to deserialize (per §2's `#[serde(alias)]`) for any external producer that hasn't migrated.

### 3.4 — Defaults, config, and the loss-knob surface — `LossConfig`

**Where:** add `loss: LossConfig` to `GrpoConfig` (`lib.rs`), with this shape:

```rust
pub struct LossConfig {
    /// Action-token policy-gradient term. Identical to today's GRPO.
    pub policy: PolicyLossConfig,         // clip_eps, kl_coeff, kl_estimator, etc.

    /// Observation-token auxiliary CE. Default: enabled at λ=0.05.
    pub echo: Option<EchoConfig>,         // Some(EchoConfig::default())

    /// Observation-token reverse-KL to a teacher. Default: None.
    pub opd: Option<OpdAuxConfig>,        // composes with ECHO; can both be on
}
```

**The defaults change.** `LossConfig::default()` returns `echo: Some(EchoConfig::default())`. To opt out you have to write `echo: null`. This is the inversion that makes ECHO native: you don't enable it, you disable it.

The TOML surface mirrors:

```toml
[training.loss]
# echo on by default (lambda = 0.05); set [training.loss.echo] = false to disable
[training.loss.echo]
lambda = 0.05
env_mask_mode = "env_only"     # | "full_obs"
warning_filter = true
```

CLI flags on `cuda_grpo_ablation` (`crates/kiln-train/examples/cuda_grpo_ablation.rs`):

- `--echo-lambda <f64>` (default 0.05)
- `--no-echo` (sets `echo: null`)
- `--echo-env-mask-mode <env_only|full_obs>` (default env_only)

Env-var overrides for experimentation: `KILN_ECHO_LAMBDA`, `KILN_ECHO_ENABLED`, `KILN_ECHO_ENV_MASK_MODE`. Same convention as `KILN_USE_FLCE` and friends.

### 3.5 — Backend coverage — what lives where

Verified by audit:

| Backend | Training path | ECHO touch points |
| --- | --- | --- |
| **CUDA** | `crates/kiln-train/src/trainer.rs::grpo_train` (with `BackendRuntime::CudaBackend` selected) | uncheckpointed loss (line 1848) + checkpointed analytic tail (line 7434) |
| **CPU** | `trainer.rs::grpo_train` (with `BackendRuntime::CpuBackend`) | same as CUDA (Phase A FLCE fallback) |
| **Metal** | `trainer.rs::grpo_train` (with `BackendRuntime::MetalBackend`) | same as CUDA (Metal Phase B FLCE) |
| **Vulkan-native** | `vk_train.rs::vk_native_grpo_train` and `vk_native_grpo_train_jsonl` | independent loss path on `VkTensor`; not via `kiln-flce-kernel` |

The CUDA / CPU / Metal triple is **one set of edits** in `trainer.rs` (no per-backend gating; candle `Device` dispatches naturally). Vulkan-native is genuinely separate code in `vk_train.rs` because it uses `VkTensor` and a hand-rolled backward graph instead of candle autograd. So the ECHO implementation has exactly two code surfaces:

1. **`trainer.rs`** — covers CUDA, CPU, Metal. The new module `crates/kiln-train/src/echo.rs` provides `echo_step_loss` (used by the uncheckpointed path) and `analytic_grpo_echo_tail_loss_grad_pre_final_norm` (used by the checkpointed path).
2. **`vk_train.rs`** — covers Vulkan. A new function `echo_step_vk` next to `grpo_active_rows_and_labels` (`vk_train.rs:181`) computes the env-CE term using VK matmul + log-softmax dispatches and folds it into the existing loss aggregation at the `grpo_active_rows_and_labels` call sites (`vk_train.rs:3623, 3959`).

There's no `metal_train.rs` and there isn't going to be one — the OPD plan §5 explicitly notes the Vulkan/Metal split is about kernel availability, and Metal training shares `trainer.rs` because candle's Metal device + the existing FLCE Phase B Metal path handle the loss math. **ECHO inherits this for free.**

### 3.6 — The diagnostic + receipt updates — `kiln-train::diagnostics`

**Where:** existing `crates/kiln-train/src/diagnostics.rs` (56KB). Add three streams:

- `env_token_ce` — per-step mean CE on env positions in the training rollouts. This is the *training-loss* version of the paper's §5.2 dynamics test. Should drop fast over the first ~200 steps if ECHO is doing anything.
- `env_token_ce_holdout` — same thing, computed periodically on a held-out trajectory set generated by a stronger teacher (paper's exact validation). Becomes part of the training receipt. This is the test that proves ECHO learned dynamics, not memorized.
- `lambda_effective` — `λ · L_env / L_grpo` so we can see the auto-anneal happen.

The receipt (`kiln-train/src/receipt.rs`) records `echo: { lambda, env_ce_initial, env_ce_final, dynamics_holdout_ce_initial, dynamics_holdout_ce_final }`. Adapters published with ECHO carry this in their reproducibility metadata.

---

## §4 The user-facing surface

Mirroring OPD plan §4.

### 4.1 — HTTP API

`/v1/train/grpo` keeps its name but its request body shape evolves:

```json
{
  "groups": [...],           // legacy schema (single string completions)
  "agentic_groups": [...],   // new schema (trajectory rollouts) — takes precedence when present
  "config": { ..., "loss": { "echo": { "lambda": 0.05 } } }
}
```

The legacy `groups` path stays bit-identical (it sets `env_mask = all-false`, ECHO contributes 0). The new `agentic_groups` path is what the migrated `rollout.py` writes. **No URL break, no MIME break — the JSON evolves under the hood.**

Long-term we add an explicit `/v1/train/agentic` alias to signal the canonical path, but only after we've validated ECHO ships well.

### 4.2 — Skills

- `.agents/skills/agentic-grpo-capability-creator/SKILL.md` gets two surgical changes:
  - **§0 Token-attribution** stops being "a known gap" and becomes "the foundation, here's the trajectory schema." The §19 "kiln gap #1" disappears.
  - **§4 Hypothesis families** gains a top-of-the-list note: "**ECHO is the default loss** (λ=0.05). H_no_echo (`--no-echo`) is the comparison hypothesis, not the other way around." Below that, the existing H1–H14 unchanged; ECHO composes with all of them.
  - **§7 Group statistics watch** gains `env_token_ce` and `env_token_ce_holdout` as first-class diagnostics.
- `.agents/skills/grpo-capability-creator/SKILL.md` (single-turn) gains the same defaults but is the no-op case: in single-turn caps, every rollout has zero observation tokens, ECHO contributes 0, behavior is identical to today. The skill explicitly notes this so capability authors don't get confused: "if your rollouts are single-turn, ECHO is a no-op for you; you can leave the default on."
- `.agents/skills/opd-capability-creator/SKILL.md` gains a half-page §8 "ECHO + OPD composition" with the formula `L = L_grpo + λ_opd · L_revKL + λ_echo · L_envCE`. Both contribute via the same FLCE-kernel path; both auto-anneal.

### 4.3 — The capability suite

- **`capabilities/agentic-grpo/pi-doctest`** — the existing cap is the migration trail. Its v0 (single-turn workaround) becomes the "before" baseline; v_echo is a fresh iter with full multi-turn trajectories + ECHO on. The capability.jsonl just gets new iter entries; the cap doesn't get rewritten.
- **`capabilities/agentic-grpo/pi-terminal-bench-lite`** — new cap, the actual paper reproduction. 100 tasks from OpenThoughts-TBLite, same harness, the headline ECHO-vs-GRPO ablation. This is the cap that proves the technique inside our infra.
- **`capabilities/agentic-grpo/pi-script-fixup`** — new cap, the "verifier-free env-only adaptation" demo from paper §5.5. Same starting checkpoint, mask GRPO term off, train only on env CE for 100 steps. Demonstrates the perpetual-improvement-without-judge loop on a PyTerm-like set.

### 4.4 — Docs

- New `docs/ECHO_GUIDE.md` (peer to existing `docs/GRPO_GUIDE.md`). Single page: what ECHO is, when to turn it off, how to interpret the diagnostics, how to read `dynamics_holdout_ce`.
- `README.md` GRPO section gets one paragraph: "kiln's GRPO is ECHO-by-default for agentic rollouts (trajectories with tool calls); for single-turn rewards it behaves identically to classical GRPO."
- `CHANGELOG.md`: one entry per phase below.
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md` — the long-form companion to the OPD plan, written when Phase 1 lands.

---

## §5 The phased rollout

**The shape:** four short phases, each independently mergeable, each ships visible user value, each gates on a real measurement.

### Phase 0 — Foundation: masks, naming, the whole bandaid (1 PR, foundational)

"Rip the bandaid and heal early." No ECHO loss term yet, but every type, name, and call site that ECHO needs is in its final shape. New code reads as if ECHO were always part of the design — it just hasn't been wired into the loss yet.

**Core schema (new crate module):**

- `crates/kiln-core/src/trajectory.rs` — new module exporting `TurnKind`, `TurnSegment`, `ScoredRollout`, `AgenticGroup` from §2.
- Tokenizer extension: `apply_chat_template_with_token_offsets` returning `(input_ids, Vec<(byte_start, byte_end)>)` for clean segment→token mapping.

**Masking primitive:**

- `crates/kiln-train/src/trajectory_mask.rs` — new module, `build_masks_from_trajectory(trajectory, prompt_messages, tokenizer, &MaskConfig) -> MaskedRollout { input_ids, action_mask, env_mask, segment_spans }`. The harness-warning-prefix exclusion (paper §3.2) lives here.

**The full rename (the bandaid):**

- `GrpoGroup → AgenticGroup`, `ScoredCompletion → ScoredRollout`, `GrpoConfig → AgenticConfig`, `tokenize_grpo_group → tokenize_agentic_group`, `TokenizedGrpoCompletion → TokenizedRollout`, `train_tokenized_grpo_group → train_tokenized_rollout`. All in `crates/kiln-train/src/lib.rs`, `trainer.rs`, `vk_train.rs`.
- Old names kept as `#[deprecated]` type aliases for one release cycle: `pub type GrpoGroup = AgenticGroup;` etc. Serde-side, `#[serde(alias = "completions")]` etc. so wire payloads with the old field names keep deserializing.
- `TokenizedRollout` carries `action_mask` and `env_mask` as separate fields from day one; `env_mask` is just all-false for any legacy single-string input.
- All four call sites in `trainer.rs` (1744, 1794, 1808, 1844) and three in `vk_train.rs` (3623, 3959) switch from `.completion_mask` to `.action_mask`. Loss math unchanged.
- `LossConfig` lands now (empty `echo: None`, empty `opd: None`) so Phase 1's additions are surgical.

**Shared Python lib:**

- `capabilities/agentic-grpo/lib/pi_trajectory.py` — extracted from `pi-doctest/rollout.py`'s `parse_transcript`, returns the canonical `Trajectory` schema. Both `pi-doctest/rollout.py` and any future agentic cap import from it.
- `pi-doctest/rollout.py` migrates to write the new `agentic_groups` / `rollouts` / `trajectory` shape. The on-disk JSONL becomes the canonical schema.

**Server-side:**

- `crates/kiln-server/src/api/training.rs` accepts both legacy (`groups`) and new (`agentic_groups`) request bodies. Internally everything becomes `AgenticGroup`.
- `POST /v1/train/agentic` lands as a canonical alias for `/v1/train/grpo` (both routes serve the same handler).

**Tests:**

- The `kiln-polish-prerequisites.md` §1 "Acceptance" test (4-turn trajectory: user/assistant/tool/assistant; assert `action_mask` is true only on assistant ranges and `env_mask` is true only on tool ranges, exactly).
- Backward-compat test: deserialize a payload using the old field names (`completions: [{text, reward}]`) and assert it round-trips through `AgenticGroup` correctly.
- Real pi-session fixture in `crates/kiln-train/tests/fixtures/` to catch chat-template byte-offset edge cases (the `<tool_call>` XML / OpenAI-normalized question from `kiln-polish-prerequisites.md` #5 gets resolved here).

**Validation gate:** existing pi-doctest replay (iter-5 strong-signal recipe) reproduces composite within ±0.005 of the published number on both CUDA and Vulkan paths. If it doesn't, the masking refactor has a bug. This is a 30-minute pod run.

### Phase 1 — ECHO kernel hookup + explicit OPD composition (1 PR, the core)

ECHO becomes the default, and `LossConfig` is shaped so OPD's eventual rebase onto main is a few-line extension, not a rewrite.

**The ECHO term:**

- `crates/kiln-train/src/echo.rs` — new module, `echo_env_ce` function (§3.1). One call into `kiln-flce-kernel::fused_linear_cross_entropy_dispatch_with_provider` with `env_mask` as the label mask, length-normalized by `|O|` per paper §3.1.

**The `LossConfig` shape (OPD composition baked in):**

```rust
pub struct LossConfig {
    pub policy: PolicyLossConfig,      // existing GRPO knobs: clip_eps, kl_coeff, etc.
    pub echo:   Option<EchoConfig>,    // DEFAULT: Some(EchoConfig::default())
    pub opd:    Option<OpdAuxConfig>,  // DEFAULT: None. Shape defined now, wiring in OPD merge.
}

pub struct OpdAuxConfig {
    pub lambda: f64,
    pub target: OpdTarget,             // Actions (Lu) | ActionsAndContext (debug)
    pub granularity: OpdGranularity,   // SampledToken | TeacherTopK { k: 32 } | FullVocab
    pub teacher: TeacherSource,        // resolved via /v1/teachers from OPD grand plan §3.2
}
```

`OpdAuxConfig` lives in `kiln-train/src/lib.rs` from day one, even though `opd.rs` doesn't read it until the OPD branch lands. This guarantees the composition formula `L = L_policy + λ_echo · L_envCE + λ_opd · L_revKL` is structurally encoded — when OPD merges, it picks up its config field instead of fighting for a separate top-level slot.

**Defaults make ECHO native:**

- `LossConfig::default()` returns `echo: Some(EchoConfig { lambda: 0.05, env_mask_mode: EnvOnly, warning_filter: true })`.
- To opt out: `loss.echo = null` in JSON, `[training.loss.echo] enabled = false` in TOML, or `--no-echo` on the CLI.

**Wiring:**

- Call site in `train_tokenized_rollout` (post-rename `trainer.rs:1848`): `let total_loss = policy_loss + cfg.echo_lambda() * echo_env_ce(...) + cfg.opd_lambda() * opd_aux(...)`. The `opd_aux` branch returns `Tensor::zeros` when `opd: None`, so it's effectively eliminated by the optimizer until OPD lands.
- Same hookup in `vk_native_grpo_train` and `vk_native_grpo_train_jsonl`.
- CLI flags on `cuda_grpo_ablation`: `--echo-lambda`, `--no-echo`, `--echo-env-mask-mode`, plus the placeholder `--opd-lambda` flag (errors with a "not yet wired" message but exists in the parser so capability scripts that set it won't fail to parse).
- `kiln.example.toml` adds the `[training.loss]`, `[training.loss.echo]`, and `[training.loss.opd]` blocks. The OPD block is documented but inert until OPD lands.
- Diagnostic streams: `env_token_ce`, `lambda_effective` (`λ · L_env / L_policy`).

**Validation gate:** paired run on pi-doctest, 3 seeds each, GRPO-only (`--no-echo`) vs ECHO (default). **Ship gate is +0.10 composite or better, with 3-seed std under 0.02** — matching the pi-doctest cap's own ship discipline. Plus binary-outcome-reward variant to confirm zero interaction between ECHO and composite scoring. Plus Vulkan smoke (one iter through `vk_native_grpo_train` to confirm the dispatcher hookup works).

### Phase 2 — Reproduce the paper, separately, with receipts (1 PR, evidence)

- `capabilities/agentic-grpo/pi-terminal-bench-lite` — new, fully separate cap shipped per the agentic-grpo skill discipline (hypothesis-driven, 3-seed-verified). 100 tasks from OpenThoughts-TBLite. Separate `capability.md`, separate `capability.oracle.sh`, separate `capability.jsonl`. This is the receipt-of-record for "ECHO works in kiln."
- Holdout dynamics test: generate trajectories from Qwen3-32B (or strongest available) on the held-out tasks, measure env-token CE on `Base`, `GRPO`, `ECHO` checkpoints. Paper §5.2's exact methodology. The receipt records the before/after numbers.
- Skill updates land in this PR: agentic-grpo skill §0 unblocked (the multi-turn masking gap is gone), §4 hypothesis families gains the "ECHO is the default loss" note, §7 diagnostics adds `env_token_ce` and `env_token_ce_holdout`. The grpo and opd skills get their respective updates (§4.2 of this plan).
- **Validation gate:** dynamics holdout CE drops by at least 30% on the ECHO checkpoint vs the GRPO-only checkpoint, AND pass-rate strictly improves on the held-out eval. If both hold, we have receipts that ECHO learned dynamics, not just got lucky.

### Phase 3 — Polish + the second proof (1 PR)

The naming refactor is already done in Phase 0; this phase is just the rounding-out of the user story.

- `capabilities/agentic-grpo/pi-script-fixup` — verifier-free env-only adaptation cap (paper §5.5 demo). Mask GRPO term off, train only on env CE for 100 steps from the strongest Phase 2 checkpoint. Demonstrates the perpetual-improvement-without-judge loop that pairs with OPD plan §10.6's self-distillation loop.
- `docs/ECHO_GUIDE.md`, README/CHANGELOG/QUICKSTART updates.
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md` — the long-form companion to the OPD plan.

### Phase 4 — Compose with OPD (subsequent, dependent on OPD branch state)

Only relevant once the OPD branch is on main. The composition: `L = L_grpo + λ_opd · L_revKL_action + λ_echo · L_envCE_observation`. Three nearly orthogonal axes:

- Outcome reward (sparse, action)
- Teacher reverse-KL (dense, action)
- Environment CE (dense, observation)

This is the OPD plan's §10 self-distillation loop with ECHO added as the third leg. The day this composition is wired and validated, kiln has *the* most complete agentic-RL stack in open source. It's called out as an explicit goal in the ECHO grand plan doc.

---

## §6 Validation strategy + the things to flag now

**The non-negotiable gates** (each phase has to pass to merge):

1. **Phase 0**: pi-doctest iter-5 reproduces composite within ±0.005 with action-mask-only behavior. Proves the refactor didn't break GRPO.
2. **Phase 1**: ECHO improves pi-doctest composite by ≥+0.10 over GRPO at 3-seed-verified variance. Proves the technique inside our infra.
3. **Phase 2**: holdout dynamics CE drops ≥30% (ECHO vs GRPO) AND pass-rate strictly improves on the new pi-tb-lite cap. Proves the mechanism, not just the number.

**Risks flagged up front:**

- **Tool-result chat template stability.** Per `kiln-polish-prerequisites.md` #5, there's no firsthand verification that pi's session JSONL records the raw `<tool_call>` XML vs the OpenAI-normalized form. Phase 0's first task is to verify this on a real pi session and document it. If pi normalizes, we need to invert the normalization at parse time. Half-day blocker if it's an issue.
- **The composite reward / ECHO interaction.** Paper uses binary outcome rewards. The pi-doctest uses a 4-component composite. ECHO doesn't care what the policy-gradient term is — it's a separate auxiliary CE — so zero interaction is the expectation, but the way to be sure is to run Phase 1 with both binary-outcome and composite, and only ship if both improve. Cheap to do.
- **Vulkan-path coverage.** The VK trainer is its own substantial file (`vk_train.rs`, 4,610 lines). The kernel-side coverage is already there (FLCE Phase B on VK), but the dispatcher / call site in `vk_native_grpo_train` and `vk_native_grpo_train_jsonl` needs the same wiring as the CUDA path. Mechanical work (it's the same `fused_linear_cross_entropy_dispatch_with_provider` call) but won't claim done until we've actually run a smoke test on the Vulkan path. Probably ~1 day of work in Phase 1.
- **Trajectory token-byte alignment.** Tokenizer offsets across chat templates with embedded `<tool_call>` XML can be subtle (special tokens, leading whitespace). The `apply_chat_template_with_token_offsets` extension has to be tested with the actual templates we run, not just toy strings. Phase 0 unit tests have to include real pi-session fixtures.
- **OPD branch merge state.** The big naming refactor in Phase 3 has interaction with OPD's existing `lib.rs` types. Coordinate with the OPD branch (or whichever lands first) to avoid two rounds of rename churn. If the OPD branch is months from main, land the rename on this branch and OPD rebases. If OPD is close to main, wait one round.

**What we ship even if ECHO underperforms:** the masking primitive (Phase 0) is independently valuable — the agentic-grpo skill explicitly flags it as a known gap. So even in the worst case where ECHO doesn't reproduce, we ship a strict improvement to the agentic-GRPO foundation. The downside is bounded.

---

## §7 Decisions (resolved)

Three decisions, answered 2026-05-18:

1. **Naming refactor scope.** Resolved: do the *whole* rename (`GrpoGroup → AgenticGroup`, `ScoredCompletion → ScoredRollout`, `tokenize_grpo_group → tokenize_agentic_group`, `TokenizedGrpoCompletion → TokenizedRollout`, etc.) with deprecated aliases. **Rip the bandaid and heal early.** The refactor lands in Phase 0, not Phase 3.

2. **Phase 2 cap split.** Resolved: ship `pi-terminal-bench-lite` as a separate cap. Separate `capability.md`, separate eval, separate receipt. The existing `pi-doctest` cap stays as the iter-5 baseline; the new cap is the paper-reproduction receipt.

3. **OPD coupling.** Resolved: ECHO config composes explicitly with OPD config in Phase 1. `LossConfig { policy, echo, opd }` lands with all three fields from day one; the `opd` branch is a `None`-default placeholder until the OPD branch's wiring rebases on top of it. The composition formula `L = L_policy + λ_echo · L_envCE + λ_opd · L_revKL` is structurally encoded from Phase 1.

---

## §8 Effort estimate

- Phase 0 — ~4–5 days. Larger scope now (full rename + masking primitive + shared Python lib + server route alias) but everything in its final shape from the start. Behind clean validation gate (pi-doctest reproduction within ±0.005 on CUDA + Vulkan).
- Phase 1 — ~2–3 days. Behind 3-seed-verified +0.10 composite gate + binary-outcome control + Vulkan smoke.
- Phase 2 — ~1 week. New `pi-terminal-bench-lite` cap + dynamics-holdout test + skill updates.
- Phase 3 — ~3–4 days. Verifier-free demo cap + docs + long-form grand-plan companion.

Total: ~3 weeks of focused work to fully native ECHO, with usable intermediate checkpoints throughout. Phase 0 is the biggest single PR (it carries the whole rename); Phases 1–3 are surgical.

---

## Appendix A — Phase 0 migration map

Concrete checklist of every file the Phase 0 PR touches. Numbers from `grep -rn` audits on this branch (2026-05-18).

### A.1 Type renames (the core)

**Where the old types are defined** (single source of truth): `crates/kiln-train/src/lib.rs:201–232`. Three structs: `ScoredCompletion`, `GrpoGroup`, `GrpoRequest`. Renames + deprecated aliases:

```rust
// crates/kiln-train/src/lib.rs — new

pub use kiln_core::trajectory::{TurnKind, TurnSegment, ScoredRollout, AgenticGroup};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgenticRequest {
    #[serde(default, alias = "groups")]
    pub agentic_groups: Vec<AgenticGroup>,
    #[serde(default)]
    pub dataset_path: Option<String>,
    #[serde(default)]
    pub config: AgenticConfig,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub post_eval: Option<kiln_eval::PostEvalConfig>,
}

// Deprecated aliases — keep for one release cycle.
#[deprecated(since = "x.y.z", note = "renamed to AgenticGroup")]
pub type GrpoGroup = AgenticGroup;
#[deprecated(since = "x.y.z", note = "renamed to ScoredRollout")]
pub type ScoredCompletion = ScoredRollout;
#[deprecated(since = "x.y.z", note = "renamed to AgenticRequest")]
pub type GrpoRequest = AgenticRequest;
#[deprecated(since = "x.y.z", note = "renamed to AgenticConfig")]
pub type GrpoConfig = AgenticConfig;
```

**Serde back-compat:** old wire payloads keep deserializing because:

- `AgenticRequest` accepts `groups` as an alias for `agentic_groups` (the renamed field).
- `ScoredRollout { trajectory, reward, text }` accepts the legacy single-string `text` form — when `trajectory` is missing, a one-segment `Action` trajectory is synthesized at deserialize time.
- `AgenticGroup` accepts `completions` as an alias for `rollouts`.

This means no JSON payload that works today stops working tomorrow.

### A.2 Rust file-by-file changes (Phase 0)

Counted on this branch:

| File | `GrpoGroup` refs | `completion_mask` refs | `GrpoConfig` refs | Change |
| --- | ---: | ---: | ---: | --- |
| `crates/kiln-train/src/trainer.rs` | 12 | 30 | many | Rename types; rename `tokenize_grpo_group → tokenize_agentic_group`; switch `.completion_mask` to `.action_mask`; replace logic that builds the mask with `build_masks_from_trajectory`. |
| `crates/kiln-train/src/vk_train.rs` | 13 | 9 | many | Same rename pattern; rename `tokenize_vk_grpo_group → tokenize_vk_agentic_group`; switch `grpo_active_rows_and_labels(...)` to take `action_mask` instead of `completion_mask`. |
| `crates/kiln-train/src/lib.rs` | 3 | 0 | many | Re-export trajectory types from `kiln-core`; add deprecated aliases. |
| `crates/kiln-train/src/opd.rs` | refs `GrpoConfig` | 0 | refs | Update any `GrpoConfig` references to `AgenticConfig`. |
| `crates/kiln-train/examples/cuda_grpo_ablation.rs` | refs | 0 | refs | Update Mode::apply and CLI to use new type names. |
| `crates/kiln-train/examples/cuda_grpo_behavioral_eval.rs` | 4 | 0 | refs | Mechanical rename. |
| `crates/kiln-server/src/api/training.rs` | 5 | 0 | refs | Accept both `groups` (alias) and `agentic_groups`; route handler unchanged below the body parse. |
| `crates/kiln-server/src/training_queue.rs` | refs | 0 | refs | `QueuedJob::Grpo` becomes `QueuedJob::Agentic` (deprecated alias kept). |
| `crates/kiln-server/src/training_preflight.rs` | 3 | 0 | refs | Mechanical rename. |
| `crates/kiln-server/src/api/eval.rs` | refs | 0 | 0 | Mechanical rename if any. |
| `crates/kiln-server/src/api/pit_of_success.rs` | refs | 0 | refs | Mechanical rename. |
| `crates/kiln-server/src/eval/datasets.rs` | refs | 0 | 0 | Mechanical rename if any. |
| `crates/kiln-server/tests/training_queue_cap.rs` | refs | 0 | refs | Mechanical rename + add a backward-compat test on the legacy `groups` field. |
| `crates/kiln-server/tests/training_tracked_cap.rs` | refs | 0 | refs | Mechanical rename. |
| `crates/kiln-eval/examples/doctest_cross_check.rs` | 2 | 0 | 0 | Mechanical rename. |
| `crates/kiln-eval/src/suite.rs` | refs | 0 | 0 | Mechanical rename if it references training types. |

**Total: 15 Rust files touched.** Of these, only `trainer.rs`, `vk_train.rs`, and `lib.rs` have substantive logic changes (mask plumbing). The remaining 12 are pure mechanical renames.

### A.3 New crate modules

- `crates/kiln-core/src/trajectory.rs` — `TurnKind`, `TurnSegment`, `ScoredRollout`, `AgenticGroup`.
- `crates/kiln-core/src/lib.rs` — export `pub mod trajectory; pub use trajectory::*;`.
- `crates/kiln-train/src/trajectory_mask.rs` — `MaskConfig`, `MaskedRollout`, `build_masks_from_trajectory`.
- `crates/kiln-train/src/lib.rs` — `mod trajectory_mask;` and re-exports.

### A.4 Capability + Python changes (Phase 0)

- `capabilities/agentic-grpo/lib/pi_trajectory.py` — **new file**. Lifted from `capabilities/agentic-grpo/pi-doctest/rollout.py::parse_transcript` (line 85–98) and `rollout.py:280–318`. Returns the canonical `{"trajectory": [{"role", "content", "kind"}, ...]}` schema.
- `capabilities/agentic-grpo/pi-doctest/rollout.py` — migrated to emit `agentic_groups` / `rollouts` / `trajectory` instead of `groups` / `completions` / `text`. The `parse_transcript` function moves into the shared lib.
- `capabilities/agentic-grpo/pi-doctest/kiln-polish-prerequisites.md` — strike-through the §1 gap (the masking primitive is now done); keep #2–#5 entries that are still open.

### A.5 Doc changes (Phase 0)

- `ARCHITECTURE.md` — §Training Pipeline / §GRPO subsection (around line 435). Update the schema example block to show the new agentic shape; keep the old shape in a `<details>` block labeled "Legacy single-turn schema."
- `CHANGELOG.md` — one Phase 0 entry under "Unreleased": "Renamed `GrpoGroup → AgenticGroup`, `ScoredCompletion → ScoredRollout` (with deprecated aliases for one cycle). Multi-turn trajectory schema lands as the canonical shape. `kiln-polish-prerequisites.md` §1 closed."
- `docs/site/demo/SCRIPTS.md` — update any sample JSON request bodies.

### A.6 Phase 0 acceptance tests (concrete)

In `crates/kiln-train/tests/trajectory_mask_test.rs` (new):

```rust
#[test]
fn build_masks_from_trajectory_four_turn_pi_session() {
    // Fixture: a real Qwen3.5-4B pi session with user → assistant
    // (tool_call) → tool → assistant (final text) — 4 turns.
    let traj = vec![
        TurnSegment { role: "user".into(), content: "Solve x^2 = 4".into(), kind: TurnKind::Context, ..Default::default() },
        TurnSegment { role: "assistant".into(), content: "<tool_call>{\"name\":\"calc\",\"arguments\":{\"expr\":\"sqrt(4)\"}}</tool_call>".into(), kind: TurnKind::Action, ..Default::default() },
        TurnSegment { role: "tool".into(), content: "2".into(), kind: TurnKind::Observation, ..Default::default() },
        TurnSegment { role: "assistant".into(), content: "x = ±2".into(), kind: TurnKind::Action, ..Default::default() },
    ];
    let result = build_masks_from_trajectory(&traj, &[], &tokenizer, &MaskConfig::default()).unwrap();
    // The action_mask must be true exactly on tokens inside the two assistant blocks.
    // The env_mask must be true exactly on tokens inside the tool block.
    // The two masks must be disjoint everywhere.
    for i in 0..result.action_mask.len() {
        assert!(!(result.action_mask[i] && result.env_mask[i]),
                "action_mask and env_mask overlap at token {i}");
    }
    // Span boundary assertions (token ranges from the fixture):
    assert_action_mask_spans(&result.action_mask, &[(13, 35), (42, 50)]);  // illustrative
    assert_env_mask_spans(&result.env_mask, &[(38, 41)]);                  // illustrative
}

#[test]
fn legacy_scored_completion_text_round_trips_through_scored_rollout() {
    // A legacy JSON payload deserializes into AgenticGroup with a one-segment Action trajectory.
    let json = r#"{ "groups": [{ "messages": [...], "completions": [{ "text": "foo", "reward": 1.0 }] }] }"#;
    let req: AgenticRequest = serde_json::from_str(json).unwrap();
    assert_eq!(req.agentic_groups.len(), 1);
    assert_eq!(req.agentic_groups[0].rollouts.len(), 1);
    assert_eq!(req.agentic_groups[0].rollouts[0].trajectory.len(), 1);
    assert_eq!(req.agentic_groups[0].rollouts[0].trajectory[0].kind, TurnKind::Action);
}

#[test]
fn pi_doctest_iter5_replay_reproduces_within_tolerance() {
    // The headline validation gate: re-run iter-5 strong-signal recipe end-to-end with the
    // refactored types and assert composite within ±0.005 of the published 0.8958.
    let composite = run_pi_doctest_replay(/* iter_5_strong_signal_config */);
    assert!((composite - 0.8958).abs() < 0.005,
            "iter-5 replay produced {composite}; expected 0.8958 ± 0.005");
}
```

These are the concrete acceptance tests that turn the validation gate from prose into code.

### A.7 What Phase 0 explicitly does NOT do

- **No new loss term.** ECHO's loss math lands in Phase 1. Phase 0 just gives ECHO somewhere to put `env_mask` when it arrives.
- **No `echo` field on `LossConfig`.** `LossConfig` doesn't exist yet — Phase 1 introduces it. Phase 0 keeps the existing `GrpoConfig`/`AgenticConfig` field layout intact.
- **No new capabilities.** `pi-terminal-bench-lite` and `pi-script-fixup` are Phase 2 / Phase 3.
- **No skill rewrites.** Skill files get one-line `kiln-polish-prerequisites.md` cross-reference updates only; the substantive ECHO sections land in Phase 2.
- **No `analytic_grpo_echo_tail_loss_grad_pre_final_norm`.** That's Phase 1's work too. The existing `analytic_grpo_tail_loss_grad_pre_final_norm` keeps using `completion_mask` (now wired as `action_mask`) and computes identical loss values.

The discipline: Phase 0 is a pure-refactor PR that ships a strict no-op on training behavior. The validation gate is *exactly* "iter-5 reproduces within ±0.005" because that's the only thing Phase 0 should prove.

---

## Appendix B — Concrete API sketches for the load-bearing pieces

These are not just shape sketches; they're the API I'd start with on day one of Phase 0 and Phase 1.

### B.1 `crates/kiln-core/src/trajectory.rs` (Phase 0)

```rust
//! Canonical trajectory schema for agentic rollouts.
//!
//! A trajectory is an ordered sequence of TurnSegments. Each segment belongs
//! to a Context (prompt), Action (assistant generation, target of policy
//! gradient), or Observation (tool result / environment, target of ECHO's
//! env-CE). The trajectory is the unit of training data; ScoredRollout
//! attaches a reward.

use serde::{Deserialize, Serialize};
use crate::tokenizer::ChatMessage;

/// What kind of supervision applies to tokens inside this segment.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum TurnKind {
    /// System / user / non-trainable scaffolding. No gradient.
    #[default]
    Context,
    /// Assistant-generated tokens. Target of policy gradient (and OPD).
    Action,
    /// Tool result / environment-observation tokens. Target of ECHO's env-CE.
    Observation,
}

/// One semantic turn in a trajectory.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TurnSegment {
    /// Chat role: "system" | "user" | "assistant" | "tool" (extensible).
    pub role: String,
    /// The raw content the model saw or emitted, before chat-template
    /// formatting. For tool-call turns this includes the <tool_call> XML.
    pub content: String,
    /// What kind of supervision applies inside this segment.
    pub kind: TurnKind,
    /// Optional tool-call correlation ID (paired with the corresponding
    /// Observation segment when available). Informational; not used by
    /// mask building.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    /// Bytes at the start of this segment's content that are harness
    /// warnings (e.g. "WARNINGS:\n- bad tool call format\n"). Excluded
    /// from env_mask when MaskConfig.warning_filter is true. Paper §3.2.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub warning_prefix_len: Option<usize>,
}

/// One scored rollout in a group.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct ScoredRollout {
    /// Canonical multi-turn structure.
    #[serde(default)]
    pub trajectory: Vec<TurnSegment>,
    /// Outcome reward (paper convention: binary 0/1; kiln convention:
    /// continuous composite [0, 1]).
    pub reward: f64,
    /// Legacy single-string completion text. When `trajectory` is empty
    /// and `text` is present, deserialization synthesizes a one-segment
    /// Action trajectory from `text`. New emitters should populate
    /// `trajectory` directly.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

impl ScoredRollout {
    /// Synthesize a legacy text-only payload into a one-segment trajectory.
    /// Called automatically by the deserializer; here exposed for tests.
    pub fn ensure_trajectory(&mut self) {
        if self.trajectory.is_empty() {
            if let Some(text) = self.text.take() {
                self.trajectory.push(TurnSegment {
                    role: "assistant".into(),
                    content: text,
                    kind: TurnKind::Action,
                    tool_call_id: None,
                    warning_prefix_len: None,
                });
            }
        }
    }
}

/// A group of rollouts sharing a prompt. (Group-relative advantage is
/// computed within this set.)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct AgenticGroup {
    /// The prompt seen by every rollout in this group (system + user).
    pub messages: Vec<ChatMessage>,
    /// Multiple scored rollouts from the same prompt.
    #[serde(alias = "completions")]    // legacy field name; one-cycle compat
    pub rollouts: Vec<ScoredRollout>,
}
```

### B.2 `crates/kiln-train/src/trajectory_mask.rs` (Phase 0)

```rust
//! Build (input_ids, action_mask, env_mask) from a trajectory.
//!
//! Generalizes label_mask_from_rendered_assistant_spans (trainer.rs:2423)
//! to cover both assistant (Action) and tool (Observation) roles, with
//! a paper §3.2 warning-prefix exclusion for env spans.

use anyhow::{Context, Result};
use kiln_core::tokenizer::{ChatMessage, KilnTokenizer};
use kiln_core::trajectory::{TurnKind, TurnSegment};

#[derive(Clone, Debug)]
pub struct MaskConfig {
    /// Whether to strip the harness "WARNINGS:\n- ..." prefix from env_mask.
    /// Paper §3.2: warning tokens memorize in ~60 steps and provide no
    /// useful gradient. Default: true.
    pub warning_filter: bool,
    /// EnvOnly (default; paper recommendation) — env_mask covers only
    /// the tool-output bytes. FullObs (debug) — env_mask covers the full
    /// observation including warnings.
    pub env_mask_mode: EnvMaskMode,
}

impl Default for MaskConfig {
    fn default() -> Self {
        Self {
            warning_filter: true,
            env_mask_mode: EnvMaskMode::EnvOnly,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EnvMaskMode {
    #[default]
    EnvOnly,
    FullObs,
}

#[derive(Clone, Debug)]
pub struct MaskedRollout {
    pub input_ids: Vec<u32>,
    /// True at positions the model generated (Action segments).
    pub action_mask: Vec<bool>,
    /// True at positions the environment produced (Observation segments).
    pub env_mask: Vec<bool>,
    /// Per-segment token span (token_start_inclusive, token_end_exclusive,
    /// kind). Useful for diagnostics and per-segment loss attribution.
    pub segment_spans: Vec<(usize, usize, TurnKind)>,
}

pub fn build_masks_from_trajectory(
    trajectory: &[TurnSegment],
    prompt_messages: &[ChatMessage],
    tokenizer: &KilnTokenizer,
    cfg: &MaskConfig,
) -> Result<MaskedRollout> {
    // 1. Compose prompt + trajectory into a single ChatMessage list.
    // 2. apply_chat_template(full_messages) → full_text
    // 3. encode_with_offsets(full_text) → (input_ids, byte_offsets)
    // 4. For each Action / Observation segment, find its rendered byte range
    //    via the same delimiter-finding logic SFT uses
    //    (label_mask_from_rendered_assistant_spans pattern).
    // 5. Apply warning_filter to Observation segments: advance the start
    //    of the byte range by segment.warning_prefix_len if set.
    // 6. Mark action_mask[tok] = true / env_mask[tok] = true for tokens
    //    whose offsets overlap the corresponding spans.
    // 7. Sanity-check disjointness (action_mask[t] && env_mask[t] is a bug).
    todo!("Phase 0 implementation")
}

/// Fallback when delimiter-finding fails to account for all segments;
/// re-tokenize cumulative prefixes to determine each segment's exact
/// token boundaries. Mirrors label_mask_by_prefix_tokenization
/// (trainer.rs:2475).
fn build_masks_by_prefix_tokenization(/* ... */) -> Result<MaskedRollout> {
    todo!()
}
```

### B.3 `crates/kiln-train/src/echo.rs` (Phase 1)

```rust
//! ECHO (Environment Cross-entropy Hybrid Objective) — paper:
//! docs/papers/echo/echo_paper.md
//!
//! Adds a length-normalized cross-entropy loss on env-observation tokens
//! to the standard GRPO policy-gradient loss on action tokens. Shares
//! the same forward pass; differs only in which mask gathers the logits.

use anyhow::{Context, Result};
use candle_core::{Device, Tensor};
use kiln_flce_kernel::{
    DEFAULT_CHUNK_SIZE, FlceProvider, fused_linear_cross_entropy_dispatch_with_provider,
};

#[derive(Clone, Debug)]
pub struct EchoConfig {
    /// Mixing coefficient for the env-CE term. Paper §3.3 default: 0.05.
    /// Productive range: 0.01–0.05. ≥0.1 risks degrading the policy
    /// objective; 0.2 collapses to predictable-output rollouts.
    pub lambda: f64,
    /// What positions in observations contribute to env_mask. EnvOnly
    /// (default) excludes the harness warning prefix; FullObs is debug.
    pub env_mask_mode: EnvMaskMode,
    /// Whether to strip the harness warning prefix from observation
    /// spans (paper §3.2). Default: true.
    pub warning_filter: bool,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            lambda: 0.05,
            env_mask_mode: EnvMaskMode::EnvOnly,
            warning_filter: true,
        }
    }
}

pub use crate::trajectory_mask::EnvMaskMode;

/// Inputs to the ECHO loss term. Mirrors OpdStepInputs (opd.rs:693).
pub struct EchoStepInputs<'a> {
    pub tokens: &'a [u32],
    /// Token positions where loss applies. Must be inside `tokens`.
    pub env_positions: &'a [usize],
    /// Post-final-RMSNorm hidden states from the policy forward.
    /// [1, T, H]. Same tensor the policy forward already produced.
    pub student_hidden: &'a Tensor,
    /// LM head weight transposed. [H, V]. Same head_t the FLCE kernel
    /// expects.
    pub head_t: &'a Tensor,
    /// Total observation length |O|, for paper §3.1 length normalization.
    /// (NOT the count of env positions; the full observation span.)
    pub total_obs_len: usize,
    pub chunk_size: usize,
    /// Optional FLCE matmul provider (Vulkan / accelerator-specific).
    pub provider: Option<FlceProvider>,
}

#[derive(Clone, Debug)]
pub struct EchoStepOutputs {
    pub per_position_ce: Tensor,
    pub mean_ce: Tensor,
    pub env_count: usize,
}

/// Compute the env-CE loss term. The returned tensor is autograd-tracked
/// off `student_hidden`'s parents (LoRA Vars), so .backward() flows
/// gradients into the LoRA parameters.
pub fn echo_step_loss(inputs: EchoStepInputs<'_>) -> Result<EchoStepOutputs> {
    let EchoStepInputs {
        tokens,
        env_positions,
        student_hidden,
        head_t,
        total_obs_len,
        chunk_size,
        provider,
    } = inputs;

    anyhow::ensure!(!env_positions.is_empty(),
        "echo_step_loss called with no env positions — caller should short-circuit");

    // Build the label mask from env_positions.
    let seq_len = tokens.len();
    let mut env_mask = vec![false; seq_len];
    for &p in env_positions {
        anyhow::ensure!(p < seq_len,
            "env position {} out of range for seq_len {}", p, seq_len);
        env_mask[p] = true;
    }

    // Single call into the existing FLCE kernel with env_mask.
    let chunk = if chunk_size == 0 { DEFAULT_CHUNK_SIZE } else { chunk_size };
    let device = student_hidden.device();
    let raw_ce_sum = fused_linear_cross_entropy_dispatch_with_provider(
        student_hidden,
        head_t,
        tokens,
        &env_mask,
        device,
        chunk,
        provider,
    )
    .context("FLCE dispatch for ECHO env-CE")?;

    // The kernel returns a scalar that is the sum of CE values divided
    // by the number of active positions. We want length-normalization
    // by total_obs_len (paper §3.1), so rescale:
    //   mean_ce = raw_ce_sum * (env_positions.len() / total_obs_len)
    // This makes the env-CE term auto-anneal as the model learns terminal
    // structure (raw_ce_sum drops, env_count usually stable).
    let env_count = env_positions.len() as f64;
    let scale = env_count / total_obs_len.max(1) as f64;
    let mean_ce = (raw_ce_sum * scale)?;

    // For diagnostics: also compute the per-position CE for the
    // env_token_ce_per_step diagnostic stream.
    let per_position_ce = compute_per_position_ce(student_hidden, head_t, tokens, &env_mask, chunk, provider)?;

    Ok(EchoStepOutputs {
        per_position_ce,
        mean_ce,
        env_count: env_positions.len(),
    })
}

fn compute_per_position_ce(/* ... */) -> Result<Tensor> {
    // Sibling of opd_top_k_reverse_kl_phase_a_per_position, but with
    // standard cross-entropy instead of reverse-KL. Used only for
    // diagnostics; not on the main loss path.
    todo!()
}
```

### B.4 `LossConfig` (Phase 1)

```rust
// crates/kiln-train/src/lib.rs — added in Phase 1

/// Composition of per-token training objectives. Each branch contributes
/// to L_total = L_policy + λ_echo · L_envCE + λ_opd · L_revKL where
/// inactive branches contribute zero.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LossConfig {
    /// Action-token policy-gradient term. Identical knobs to today's
    /// GRPO: clip_eps, kl_coeff, kl_estimator, advantage_mode, is_level,
    /// reference_policy, etc.
    #[serde(default)]
    pub policy: PolicyLossConfig,

    /// Observation-token cross-entropy (paper: Shrivastava et al. 2026).
    /// Default: Some(EchoConfig::default()) with lambda=0.05. Set to
    /// None to opt out.
    #[serde(default = "default_echo_some")]
    pub echo: Option<EchoConfig>,

    /// Action-token reverse-KL to a teacher (OPD; Lu 2025). Default: None.
    /// Wired in by the OPD branch rebase; LossConfig holds the field so
    /// the composition is structural.
    #[serde(default)]
    pub opd: Option<OpdAuxConfig>,
}

fn default_echo_some() -> Option<EchoConfig> {
    Some(EchoConfig::default())
}

impl LossConfig {
    pub fn echo_lambda(&self) -> f64 {
        self.echo.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
    pub fn opd_lambda(&self) -> f64 {
        self.opd.as_ref().map(|c| c.lambda).unwrap_or(0.0)
    }
}

// AgenticConfig (renamed from GrpoConfig in Phase 0) gains:
pub struct AgenticConfig {
    /* existing fields — learning_rate, kl_coeff, clip_epsilon, etc. */
    #[serde(default)]
    pub loss: LossConfig,
}
```

### B.5 The uncheckpointed-path diff (Phase 1, `trainer.rs` around line 1848)

```rust
// Before:
let policy_log_probs = token_log_probs(&policy_logits, &comp.input_ids, &comp.action_mask, device)?;
let loss = grpo_loss(&policy_log_probs, &ref_log_probs, loss_params, device)?;
loss_val = loss.to_scalar::<f32>()? as f64;
let grads = loss.backward().context("GRPO backward pass")?;

// After:
let policy_log_probs = token_log_probs(&policy_logits, &comp.input_ids, &comp.action_mask, device)?;
let policy_loss = grpo_loss(&policy_log_probs, &ref_log_probs, loss_params, device)?;

let total_loss = if let Some(echo_cfg) = &config.loss.echo {
    let env_positions: Vec<usize> = comp.env_mask.iter().enumerate()
        .filter_map(|(i, &m)| if m { Some(i) } else { None }).collect();
    if env_positions.is_empty() {
        policy_loss
    } else {
        // Extract hidden_pre_head from policy_logits if available; otherwise
        // re-run model_forward_no_head. Plumbing detail.
        let echo_out = echo::echo_step_loss(echo::EchoStepInputs {
            tokens: &comp.input_ids,
            env_positions: &env_positions,
            student_hidden: &policy_hidden_pre_head,
            head_t: &weights.embed_tokens_t,
            total_obs_len: comp.total_obs_len,
            chunk_size: DEFAULT_CHUNK_SIZE,
            provider: None,
        })?;
        let scaled = echo_out.mean_ce.affine(echo_cfg.lambda, 0.0)?;
        policy_loss.add(&scaled)?
    }
} else {
    policy_loss
};

loss_val = total_loss.to_scalar::<f32>()? as f64;
let grads = total_loss.backward().context("GRPO+ECHO backward pass")?;
```

**Plumbing note:** the current `train_tokenized_grpo_group` discards the pre-head hidden after `model_forward` produces logits. Phase 1 needs to keep `policy_hidden_pre_head` alive so `echo_step_loss` can call FLCE on it. Two options: (a) refactor to call `model_forward_no_head` then apply head separately; (b) keep a fork of the existing path. Option (a) is the cleaner long-term fix and what OPD's existing code already does — pattern-match on `opd.rs::opd_step_loss`'s `student_hidden` plumbing.

### B.6 The checkpointed-path change (Phase 1, `trainer.rs` around line 7434)

```rust
// Before:
let (loss_val, mut upstream_grad) = analytic_grpo_tail_loss_grad_pre_final_norm(
    &final_hidden,
    &weights.final_norm,
    &weights.embed_tokens_t,
    input_ids,
    completion_mask,      // becomes action_mask after Phase 0
    ref_log_probs,
    loss_params,
    model_config.rms_norm_eps,
    DEFAULT_CHUNK_SIZE,
)?;

// After:
let (loss_val, mut upstream_grad) = analytic_grpo_echo_tail_loss_grad_pre_final_norm(
    &final_hidden,
    &weights.final_norm,
    &weights.embed_tokens_t,
    input_ids,
    action_mask,
    env_mask,             // NEW
    ref_log_probs,
    loss_params,
    echo_lambda,          // NEW (0.0 disables the ECHO term)
    total_obs_len,        // NEW
    model_config.rms_norm_eps,
    DEFAULT_CHUNK_SIZE,
)?;
```

The new `analytic_grpo_echo_tail_loss_grad_pre_final_norm` walks vocab chunks the same way the existing function does, but adds, for each chunk: a separate gather over `env_mask[t] && t in chunk` positions, computes the per-position CE analytically (`logp = logit - log_sum_exp`; `ce_t = -logp[input_ids[t+1]]`), and accumulates both `env_ce_sum` and `d(env_ce_sum) / d(final_hidden)` per chunk. Returns `(loss_val_total, upstream_grad_total) = (loss_grpo + λ_echo · loss_env_ce, upstream_grpo + λ_echo · upstream_env_ce)`.

The chunked vocab matmul is shared between the GRPO and ECHO computations — memory cost is zero additional intermediates above what the existing function already pays.

### B.7 Vulkan-path implementation (Phase 1, `vk_train.rs`)

```rust
// New helper next to grpo_active_rows_and_labels (vk_train.rs:181).
fn echo_env_active_rows_and_labels(
    input_ids: &[u32],
    env_mask: &[bool],
) -> Result<(Vec<u32>, Vec<u32>)> {
    // Same pattern as grpo_active_rows_and_labels: shift by 1 (next-token
    // prediction), return (active_rows, labels) where each active_row
    // is a position in the policy_logits that we'll gather, and labels[i]
    // is input_ids[active_rows[i]+1].
    anyhow::ensure!(input_ids.len() == env_mask.len(),
        "input_ids/env_mask length mismatch");
    let active_rows: Vec<u32> = env_mask[1..].iter().enumerate()
        .filter_map(|(i, &m)| if m { Some(i as u32) } else { None }).collect();
    let labels: Vec<u32> = active_rows.iter()
        .map(|&i| input_ids[i as usize + 1]).collect();
    Ok((active_rows, labels))
}

// Inside vk_native_grpo_train at the call site (vk_train.rs:3623 / 3959),
// after the existing GRPO loss computation:
let echo_loss_vk: Option<VkTensor> = if let Some(echo_cfg) = config.loss.echo.as_ref() {
    let (env_rows, env_labels) = echo_env_active_rows_and_labels(&comp.input_ids, &comp.env_mask)?;
    if env_rows.is_empty() {
        None
    } else {
        // Gather hidden rows at env positions, run head matmul, log-softmax,
        // gather labels, mean (length-normalized by total_obs_len).
        Some(vk_echo_env_ce(
            &final_hidden_vk,
            &head_t_vk,
            &env_rows,
            &env_labels,
            comp.total_obs_len,
            &mut vk_state,
        )?)
    }
} else {
    None
};

// Fold into total VK loss with scaling by echo_cfg.lambda.
```

The full implementation of `vk_echo_env_ce` follows the same VkTensor pattern as `vk_native_grpo_train`'s existing loss computation — log-softmax, gather, mean — and is roughly 50–80 lines of VkTensor ops. Estimated half-day of work.

### B.8 `pi_trajectory.py` (Phase 0, shared Python lib)

```python
"""Parse pi session JSONL into the canonical kiln Trajectory schema.

The pi session format (verified against capabilities/agentic-grpo/pi-doctest/rollout.py
on 2026-05-18):

  {"type": "message", "message": {
       "role": "system" | "user" | "assistant" | "tool",
       "content": [{
           "type": "text" | "thinking" | "toolCall" | "toolResult",
           ...role-specific fields...
       }]
  }}

This module emits {"trajectory": [{"role", "content", "kind", ...}, ...]}.
"""

from pathlib import Path
import json
from typing import Iterator

ASSISTANT_BLOCK_RENDERERS = {
    "text": lambda b: b.get("text", ""),
    "thinking": lambda b: f"<think>{b.get('thinking', '')}</think>",
    "toolCall": lambda b: (
        f'<tool_call>{{"name": "{b.get("name", "")}", '
        f'"arguments": {json.dumps(b.get("input", {}))}}}</tool_call>'
    ),
}


def parse_pi_session(session_path: Path) -> list[dict]:
    """Parse a pi session JSONL file into kiln Trajectory segments.

    Returns a list of TurnSegment dicts:
        [{"role", "content", "kind", "tool_call_id"?, "warning_prefix_len"?}, ...]

    System / user turns are TurnKind=context.
    Assistant turns are TurnKind=action (content rendered as Qwen XML).
    Tool-result turns are TurnKind=observation (content is the raw output).
    """
    segments: list[dict] = []
    for event in _iter_events(session_path):
        if event.get("type") != "message":
            continue
        msg = event.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")

        if role in ("system", "user"):
            text = _stringify_content(content)
            segments.append({"role": role, "content": text, "kind": "context"})

        elif role == "assistant":
            text = _render_assistant_blocks(content or [])
            if text:
                segments.append({"role": role, "content": text, "kind": "action"})

        elif role == "tool":
            text = _render_tool_blocks(content or [])
            tool_call_id = _extract_tool_call_id(content or [])
            warning_prefix_len = _detect_warning_prefix(text)
            segments.append({
                "role": role,
                "content": text,
                "kind": "observation",
                "tool_call_id": tool_call_id,
                "warning_prefix_len": warning_prefix_len,
            })

    return segments


def _iter_events(path: Path) -> Iterator[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def _render_assistant_blocks(blocks: list) -> str:
    parts = []
    for b in blocks:
        if not isinstance(b, dict):
            continue
        renderer = ASSISTANT_BLOCK_RENDERERS.get(b.get("type"))
        if renderer:
            parts.append(renderer(b))
    return "".join(parts)


def _render_tool_blocks(blocks: list) -> str:
    parts = []
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "toolResult":
            parts.append(str(b.get("content", "")))
    return "".join(parts)


def _extract_tool_call_id(blocks: list) -> str | None:
    for b in blocks:
        if isinstance(b, dict) and b.get("type") == "toolResult":
            return b.get("toolCallId") or b.get("tool_call_id")
    return None


def _detect_warning_prefix(text: str) -> int | None:
    """Detect the 'WARNINGS:\n- ...' prefix harness emits when a tool
    call fails parsing. Returns the byte length of the prefix (so the
    masker can advance past it), or None if no warning prefix."""
    if text.startswith("WARNINGS:\n"):
        # Find the first non-warning content (typically <command_output>).
        idx = text.find("<command_output>")
        return idx if idx > 0 else len("WARNINGS:\n")
    return None


def _stringify_content(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        # Flatten list-of-blocks for system/user (rare).
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content) if content is not None else ""
```

These eight code blocks are the load-bearing pieces of the plan. Phase 0 implements B.1, B.2, B.8 + the deprecated-alias type renames. Phase 1 implements B.3, B.4, B.5, B.6, B.7.
