# Kiln gaps that block multi-turn agentic GRPO

This document is intentionally narrow: it lists kiln-train changes
that need to land **before** an agentic-GRPO cap can credibly train on
multi-turn pi sessions.

For v0 (`pi-doctest`) we constrain each rollout to a single assistant
turn so the existing `tokenize_grpo_group` machinery works without
change. The list below is the path forward for any cap whose rollouts
exceed one assistant turn.

## #1 — Per-turn assistant-token masking — ✅ RESOLVED (ECHO Phase 0)

> **Resolved 2026-05-18.** The trajectory schema and masking primitive
> ECHO needed (`crates/kiln-train/src/trajectory.rs`,
> `crates/kiln-train/src/trajectory_mask.rs`) gives `tokenize_grpo_group`
> a path that consumes a `trajectory: Vec<TurnSegment>` field on each
> rollout and emits separate `action_mask` (assistant tokens — policy
> gradient targets) and `env_mask` (tool-result tokens — ECHO env-CE
> targets). The shared `capabilities/agentic-grpo/lib/pi_trajectory.py`
> module converts pi's session JSONL into the canonical schema. See
> `docs/plans/echo-integration-plan.md` for the full design.

### Historical record

**Symptom (today).** `tokenize_grpo_group` (`crates/kiln-train/src/trainer.rs`)
builds a single-message `assistant` turn from `ScoredCompletion.text`.
For a multi-turn pi session — where the model produced N assistant
turns interleaved with N tool turns — packing everything into one
synthetic `assistant` message means tool-result text is treated as
model-emitted. Gradient flows through tokens the model never produced.

**Effect on training.** Per-token IS ratio computed against the
reference forward will be near 1 for tool-result tokens (both policy
and reference assign similar low probability). The clip-min may fire
and zero out the gradient through these tokens, but relying on the
clip to mask is fragile and biased.

**Proposed change.** Extend `GrpoGroup` with an optional
`trajectory: Option<Vec<TurnSegment>>` parallel to `completions[i].text`:

```rust
#[derive(Serialize, Deserialize)]
pub struct TurnSegment {
    pub role: String,          // "system" | "user" | "assistant" | "tool"
    pub content: String,
    pub train: bool,           // assistant turns: true; everything else: false
}

#[derive(Serialize, Deserialize)]
pub struct ScoredCompletion {
    pub text: String,                          // legacy single-turn
    pub trajectory: Option<Vec<TurnSegment>>,  // new multi-turn
    pub reward: f64,
}
```

`tokenize_grpo_group` then takes the trajectory path when present:

1. Concatenate `[prompt messages] + [trajectory turns]` into the full
   message list.
2. Run `apply_chat_template` once on the full message list.
3. Build a per-token `completion_mask` by:
   a. Tokenizing each turn independently to know its token count.
   b. Walking the cumulative offset and marking positions inside
      `train: true` turns as `completion_mask = true`.
   c. Marking everything else `false`.

The chat-template adds role markers and separators which won't appear
in per-turn tokenization. The cleanest robust approach: have
kiln-core's tokenizer expose `apply_chat_template_with_token_offsets`
returning the encoded ids + per-turn token spans.

**Acceptance.** A unit test in `kiln-train`:
- Build a `GrpoGroup` with one completion carrying a 4-turn trajectory
  (user, assistant, tool, assistant) and a single reward.
- Tokenize.
- Assert the `completion_mask` is `false` for system/user/tool ranges
  and `true` for assistant ranges, exactly.

**Estimated effort.** Half a day. The chat-template-with-offsets
extension is the bulk of it.

## #2 — Rollout-batch determinism for in-process eval

**Symptom (today).** GRPO rollouts via `pi -p` are out-of-process and
non-deterministic across machines (sampling RNG, kiln's batched
serving). For reproducible iter logs we need a deterministic rollout
mode.

**Proposed change.** Add `--seed-base <u64>` flag to the rollout
runner; thread it through pi → kiln's chat completion request as a
seed parameter (OpenAI-compatible APIs accept `seed`).

**Acceptance.** Two consecutive runs of the rollout pass with the same
seed produce byte-identical session JSONL.

**Estimated effort.** Half a day; depends on kiln-server already
honoring `seed` (verify).

## #3 — Wall-clock budget per pi session

**Symptom (today).** `pi -p` has no `--max-wall-clock-s` flag. A
pathological session can chew through the iter budget.

**Workaround for v0.** Wrap pi in `timeout 120 pi -p ...` at the OS
level. Sessions killed by timeout produce no session JSONL — the
rollout adapter treats this as `outcome=0.0`.

**Proposed change (upstream pi).** PR a `--max-wall-clock-s` flag.

**Estimated effort.** One day upstream.

## #4 — Sandbox lifecycle helper

**Symptom (today).** Each cap rolls its own sandbox scaffold/teardown.
Risk of bugs (e.g. forgetting to clean a dir, two rollouts sharing a
workdir).

**Proposed change.** A small `kiln-rollout-sandbox` Rust helper that
provisions a tempdir, initializes it from a per-cap `task_scaffold.py
init_workdir`, runs a single rollout, captures the workdir + transcript,
and tears down.

**Status.** v0 inlines this logic into `run_iter1.sh`. Promote to a
helper when a second cap demonstrates the pattern.

## #5 — kiln-server raw token capture (status unknown, verify on pod)

**Question.** Does `chat_completions_inner` log the *raw* (pre-Qwen→OpenAI-
normalized) assistant token sequence to disk? The rollout adapter
uses pi's session JSONL as canonical, which records the
OpenAI-normalized view. For training, we re-tokenize using
kiln-core's chat template — this should be deterministic, but verify
on the pod that:
- `pi`'s session JSONL records assistant `content` containing the
  Qwen XML `<tool_call>` blocks (raw) OR the OpenAI-normalized
  `tool_calls` array (in which case we have to invert the
  normalization to get the raw text the model emitted).

**Acceptance.** Look at one session JSONL line emitted by pi for a
tool-calling assistant turn. Decide whether to (a) train on what
pi recorded as-is, or (b) reconstruct the raw via the chat template.

Add the finding to `capability.md` Pi configuration section.

---

## v0 disposition

For `pi-doctest`: constrain rollouts to **one assistant turn** via the
task prompt. The model emits a single message containing either:
- a single tool call (`write` to solution.py); pi runs the tool; the
  model emits a final no-tool-call turn after seeing the result, OR
- the final answer directly (no tool call).

Empirically I expect the model to need the second turn to issue the
`bash` test, so v0 will probably need 2-3 turns even with the prompt
constraint. That makes #1 the load-bearing gap. **Lay the masking PR
before iter 1** rather than accept the gradient bias.

If we MUST run iter 1 before the masking PR lands: cap pi to
`--max-turns 2` (if pi supports it) and accept whatever signal we get,
flagged as v0.0 in the capability.jsonl.
