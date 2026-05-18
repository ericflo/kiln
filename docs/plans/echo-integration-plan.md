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

### 3.1 — `kiln-train::echo` — the loss term and its dispatcher

**Where:** new module `crates/kiln-train/src/echo.rs`, sibling to `opd.rs`. It owns nothing big — it's a small surface, intentionally:

```rust
pub struct EchoConfig {
    pub lambda: f64,                  // default 0.05
    pub env_mask_mode: EnvMaskMode,   // EnvOnly (default) | FullObs (debug-only)
    pub warning_filter: bool,         // default true; strip harness warning prefix
}

pub enum EnvMaskMode { EnvOnly, FullObs }

/// Compute the auxiliary cross-entropy loss term over env-positions.
/// Reuses the same `(hidden, head_t)` activations as the GRPO forward.
pub fn echo_env_ce(
    hidden: &Tensor,           // [1, T, H] post-final-RMSNorm
    head_t: &Tensor,           // [H, V]
    input_ids: &[u32],
    env_mask: &[bool],
    device: &Device,
) -> Result<Tensor>            // scalar; length-normalized by |O|, not |O'|, per paper §3.1
```

The body is one call into `fused_linear_cross_entropy_dispatch` from `kiln-flce-kernel` with `env_mask` as the label mask. Length-normalization by `|O|` (full observation length) not `|O'|` (env-tokens-only count) is the paper's choice (§3.1) and is one division at the end. That's it.

**Why this is "free":**

- `hidden` is already on device — it's exactly the tensor SFT/GRPO already build for the policy forward.
- The dispatcher already covers all four backends. CUDA & Metal: Phase B `CustomOp1`. Vulkan: Phase B `CustomOp1`. CPU: Phase A. **Zero per-backend kernel work.**
- The same `KILN_USE_FLCE`, `KILN_CUDA_FLCE`, `KILN_VULKAN_FLCE` toggles already control the kernel; ECHO inherits them.

**Where it's called:** in `train_tokenized_grpo_group` after the existing `grpo_loss` call (`trainer.rs:1848`). One new line, one addition to the scalar loss, one config field check.

For the Vulkan-native streaming path (`vk_train.rs:3471 vk_native_grpo_train` and `vk_train.rs:3749 vk_native_grpo_train_jsonl`), the same hook lands after the existing `grpo_active_rows_and_labels` call (`vk_train.rs:3623, 3959`). VK's `fused_linear_cross_entropy_dispatch_with_provider` path (with the VK matmul provider — `kiln-flce-kernel/src/lib.rs:141`) is what ECHO uses there.

### 3.2 — The masking layer — `kiln-train::trajectory_mask`

**Where:** new module `crates/kiln-train/src/trajectory_mask.rs`. Hosts the function that takes a `Trajectory` + tokenizer and emits `(input_ids, action_mask, env_mask)`.

This module is the *foundation* — both the existing multi-turn agentic-GRPO need and ECHO depend on it. It generalizes `label_mask_from_rendered_assistant_spans` (`trainer.rs:2423`) from one mask to two.

```rust
pub fn build_masks_from_trajectory(
    trajectory: &[TurnSegment],
    prompt_messages: &[ChatMessage],
    tokenizer: &KilnTokenizer,
    cfg: &MaskConfig,           // warning_filter, env_mask_mode
) -> Result<MaskedRollout>      // { input_ids, action_mask, env_mask, segment_spans }
```

Internally: render the full conversation with `apply_chat_template` once, walk each `TurnSegment` to find its rendered byte range (using the same delimiter-finding logic SFT already uses, just generalized to all four roles), map byte ranges to token spans via the tokenizer's `encode_with_offsets`, then fill the two masks. **The harness warning prefix exclusion** (paper §3.2 — warnings memorize in ~60 steps) is a span tweak: for each `Observation` segment with a non-None `warning_prefix_len`, advance the start of its env_mask span by that many bytes.

We need one tokenizer enhancement to make this clean: `apply_chat_template_with_token_offsets` returning `(input_ids, Vec<(byte_start, byte_end)>)` so the segment→token mapping is exact. The polish-prerequisites doc already calls this out.

Then **`tokenize_grpo_group` becomes a thin shim**: it calls `build_masks_from_trajectory` and packages the result. The single-string legacy path produces a `Trajectory` with one big `Action` segment, and behavior is identical to today.

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

### 3.5 — The diagnostic + receipt updates — `kiln-train::diagnostics`

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
