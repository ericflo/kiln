# A Grand Plan for Extraordinarily Great ECHO for Everyone

> *Every command an agent runs produces a response. Every response is supervision. Use it.*

**Status:** Companion document to the engineering plan in [`echo-integration-plan.md`](echo-integration-plan.md). Where that one is the technical contract — phases, file:line touch points, acceptance gates — this one is the aspirational arc: why ECHO matters for kiln, who it serves, what the user-facing surface looks like, and what we ship after Phase 3.

**Authors:** Synthesised by Claude (Opus 4.7) after implementing Phases 0 / 1 / 2 / 3 of the integration plan.
**Date:** 2026-05-19.

---

## 0. Executive summary

ECHO (Shrivastava, Awadallah, Papailiopoulos — MSR AI Frontiers, 2026) is the most undervalued idea in agentic post-training right now.

The intuition is one sentence: *trajectories already contain a dense training signal waiting to be used.* When a CLI agent runs `ls`, the shell returns a listing. When it runs `pytest`, the framework returns pass/fail per test. When it `cat`s a file, the file's content streams back. Those tokens enter the model's forward pass as context for the next action — and standard agent RL **throws them away** as gradient targets. ECHO doesn't.

The math is one line:

$$\mathcal{L}_{\text{ECHO}} = \mathcal{L}_{\text{GRPO}}(\text{actions}) + \lambda \cdot \mathcal{L}_{\text{Env}}(\text{observations})$$

The paper's headline numbers are extraordinary for "no extra forward pass":

| Model | GRPO pass@1 | ECHO pass@1 | Factor |
| --- | --- | --- | --- |
| Qwen3-8B | 2.70% | 5.17% | **1.9×** |
| Qwen3-14B | 5.17% | 10.79% | **2.1×** |

Plus secondary wins:
- 1.5–2.3× faster to peak GRPO performance.
- Closes ~half of the expert-SFT gap *without expert demonstrations* — paper §5.3.
- Verifier-free self-improvement (paper §5.5): a strong agent keeps improving from environment interaction alone, +10 pp on PyTerm in 100 steps.

### Why kiln is the right vehicle

The grand-OPD-plan made a case for kiln as the OPD vehicle of the era. The same case applies for ECHO, with a different emphasis:

- **The primary deployment of kiln-tuned Qwen3.5-4B is agentic.** Pi as the harness, terminal as the environment, kiln serving + training in one process. ECHO targets this loop precisely.
- **All four backends, one loss term.** CUDA + CPU + Metal share `trainer.rs`; Vulkan-native gets its own VkTensor-based implementation. Phase 1 of this integration shipped both with full backward graphs. **No other open-source GRPO framework has agentic ECHO across this many backends.**
- **The masking primitive was already needed.** The `kiln-polish-prerequisites.md` §1 multi-turn assistant-token masking gap was the blocker for every other agentic post-training technique kiln cared about: SCoRe earliest-error weighting, TIP token-importance profiling, agentic OPD. ECHO Phase 0 closes that gap *as a side effect* of making env_mask trainable.

### What this document promises

By the time everything in §3-§5 ships, kiln will have:

- ECHO as the default for agentic GRPO, with zero-cost short-circuit for legacy single-turn rollouts.
- All four backends (CUDA / CPU / Metal candle + Vulkan-native) computing the env-CE term correctly.
- Both the uncheckpointed loss path AND the checkpointed analytic-tail path applying ECHO (so long contexts get the technique).
- A dedicated paper-reproduction cap (`pi-terminal-bench-lite`) with a dynamics-holdout test that pins paper §5.2.
- A verifier-free adaptation cap (`pi-script-fixup`) demonstrating paper §5.5 on PyTerm-shaped tasks.
- Receipt-grade evidence (`receipt.json::echo`) on every adapter so the ECHO contribution is auditable.

---

## 1. Why now, why kiln, why ECHO specifically

### 1.1 The state of agentic RL as of 2026-05-19

Three primitives dominate post-training for agents today:

| | Sampling | Reward density | Bits/episode | When it dominates |
| --- | --- | --- | --- | --- |
| **SFT** (off-policy) | teacher | dense | O(N) tokens | small data, short tasks |
| **GRPO** (on-policy) | student | sparse | O(1) | when only the answer matters |
| **OPD** (on-policy + dense teacher) | student | dense (teacher logprobs) | O(N) | personalization, distillation |
| **ECHO** (on-policy + dense env) | student | dense (raw env tokens) | O(N) | **agentic; every command response is training data** |

That last row is where this plan lives. ECHO is to *environment-token supervision* what OPD was to *teacher-logprob supervision* — a one-line change with O(N) bits per episode that nobody is shipping in open source.

### 1.2 The 4B agentic deployment is the structural fit

Kiln's grand-OPD-plan §1.2 noted that Qwen3.5-4B / Qwen3.6-27B share a tokenizer and chat template — making cross-family distillation a non-issue. ECHO doesn't need a teacher at all (the supervision IS the environment), so this point is even more direct: **ECHO just needs the trajectory schema**, and kiln already has it (Phase 0).

The remaining structural fit is the harness: kiln serves pi, pi runs the agent loop, pi captures sessions, the sessions become training data. The integration plan shows that the **only** capability-side change needed was `pi_trajectory.build_scored_rollout(...)` — a single function call.

### 1.3 The Vulkan trajectory compounds with ECHO too

Kiln's vk-native training keeps every activation, gradient, and optimizer state resident on Vulkan (per `docs/vk_native_training.md`). The ECHO env-CE term was added without any new VK shader work — it reuses the existing `vk_flce_loss`, `vk_index_select_rows`, `vk_scale`, `vk_add` kernels. **Every Vulkan inference speedup compounds into ECHO training speedup automatically.**

### 1.4 The "ECHO can substitute for an expert teacher" finding is huge for prosumers

Paper §5.3 shows ECHO recovers ~half of the expert-SFT gap without expert demonstrations. For Tier 1 (laptop) and Tier 2 (prosumer) users, this matters disproportionately:

- They often **don't have access to a stronger teacher** to bootstrap from. The grand-OPD-plan offered hosted-teacher logits as the answer; ECHO offers a *cheaper* answer: just let the model train on what its own actions produced.
- Even when a teacher IS available, ECHO is a free addition. Phase 1 §LossConfig composition is structural — both terms apply to the same forward pass.

---

## 2. The three tiers we serve

Mirrors grand-OPD-plan §2.

### 2.1 Tier 1 — Laptop user (8–16 GB VRAM, Apple Silicon, or CPU)

- **Wants:** an assistant that gets better at their coding workflow without paying for a teacher.
- **What ECHO gives them:** the auxiliary loss term costs essentially nothing — same forward pass, same activations, just a second masked log-prob sum. Tier 1 hardware can already train GRPO on Qwen3.5-4B + small LoRA; ECHO doesn't change that picture.
- **What's new for them:** the dynamics-holdout test (paper §5.2). On Tier 1 you can't run a stronger teacher model locally, so the test falls back to a "stub" mode (see `capabilities/agentic-grpo/pi-terminal-bench-lite/calibration/dynamics_holdout.py`) that still emits a receipt structure for later replay against a hosted teacher.

### 2.2 Tier 2 — Prosumer (1× 4090/5090/7900XTX/M-series Max, 24–48 GB VRAM)

- **Wants:** experiment with multi-turn training, build agentic LoRAs for specific repos / workflows.
- **What ECHO gives them:** the same uplift as Tier 1 plus enough VRAM to actually train at TBLite-shaped scale. The Phase 2 `pi-terminal-bench-lite` cap is designed for this tier.
- **What's new for them:** the `pi-script-fixup` verifier-free cap (paper §5.5) — once they've trained a strong-but-stable adapter, they can keep adapting it on OOD tasks *without* needing to define new verifiers. Just point it at a fresh task corpus and let env-CE drive the updates.

### 2.3 Tier 3 — Corporate (8× H200, multi-node)

- **Wants:** reliable agentic-RL pipelines, governance, audit trails.
- **What ECHO gives them:** the receipt fields (`EchoDiagnosticSummary`) capture exactly the paper §5.2 evidence (`env_ce_initial`, `env_ce_final`, `env_ce_drop_pct`). When ECHO is composed with OPD on the same `LossConfig`, every adapter receipt records both the teacher contribution and the env-CE contribution separately — a clean ablation trail.
- **What's new for them:** the agentic route alias (`POST /v1/train/agentic`) makes the semantic intent of the endpoint explicit, and the `LossConfig.no_policy_loss` flag enables verifier-free fleet updates (every team's pi sessions become continuous training data).

**The same code path serves all three.** The only knob that differs is which adapters are loaded as starting points.

---

## 3. The architectural pillars

Six pillars, mirroring grand-OPD-plan §3 numbering. The integration plan §3 already describes the code-level seams; this section is about the user-facing story.

### 3.1 The ECHO training term — `kiln-train::echo`

- **One module, one function.** `echo_step_loss` mirrors `opd_step_loss`'s signature so the `LossConfig` composition is structural.
- **Backend coverage:** candle path (CUDA / CPU / Metal) via `fused_linear_cross_entropy_dispatch`; vk-native via `vk_flce_loss`. The math is bit-identical; only the kernel dispatch differs.
- **Both loss paths:** uncheckpointed (logits already materialized) AND checkpointed analytic tail (Phase 1 step 6 — the env-CE term folds into the existing vocab-chunk loop with zero additional intermediates).

### 3.2 The masking primitive — `kiln-train::trajectory_mask`

- **What it solves:** the agentic-grpo skill's §0 "multi-turn assistant-token masking" gap, which previously blocked any cap with more than one assistant turn per rollout.
- **What it provides:** `build_masks_from_trajectory(trajectory, prompt_messages, tokenizer, &MaskConfig) -> MaskedRollout` returns separate `action_mask` (policy-gradient targets) and `env_mask` (ECHO env-CE targets), plus per-segment token spans for diagnostics.
- **Chat-template intelligence:** verified against the real Qwen3.5-4B `chat_template.jinja` — handles Qwen's `<tool_response>` wrapper for tool turns (not naive `<|im_start|>tool`). Paper §3.2 warning-prefix exclusion lives here.

### 3.3 The pi-side shared lib — `capabilities/agentic-grpo/lib/pi_trajectory.py`

- **Single function:** `pi_trajectory.build_scored_rollout(session_path, reward=...)` returns a `ScoredRollout`-shaped dict ready to drop into a GRPO JSONL.
- **What replaces:** every cap previously had its own ad-hoc parse_transcript that flattened assistant turns into `<TURN_BREAK>`-joined text and dropped tool results entirely.
- **15 unit tests** verify the parser handles all pi content-block types (text/thinking/toolCall/toolResult), warning-prefix detection, tool-call ID correlation, and malformed-JSONL tolerance.

### 3.4 The capability templates

- **`pi-doctest`** — the original v0 cap; migrated to emit the canonical trajectory schema in Phase 0. The "before" baseline for ECHO ablations.
- **`pi-terminal-bench-lite`** — Phase 2 paper-reproduction cap. Separate `capability.md` / `capability.oracle.sh` / `capability.jsonl` so the ECHO receipt is clean. Includes the paper §5.2 dynamics-holdout test (currently stub-capable; full wiring once kiln's `prompt_logprobs` extension lands).
- **`pi-script-fixup`** — Phase 3 verifier-free adaptation cap. Demonstrates paper §5.5 on PyTerm-shaped Python tasks. Uses `--no-policy-loss` to mask out the GRPO term and run ECHO-only.

### 3.5 The CLI + env-var + HTTP surface

- **CLI:** `cuda_grpo_ablation --echo-lambda F | --no-echo | --no-policy-loss | --opd-lambda F` (the last reserved for OPD rebase).
- **Env vars:** `KILN_ECHO_ENABLED`, `KILN_ECHO_LAMBDA`, `KILN_ECHO_ENV_MASK_MODE`, `KILN_ECHO_WARNING_FILTER`. Env-var precedence is higher than CLI (CLI = dev tweaks; env = ops orchestration).
- **HTTP:** `POST /v1/train/agentic` (canonical) and `POST /v1/train/grpo` (legacy alias) — same handler. Wire format accepts both `agentic_groups` / `rollouts` / `trajectory` (canonical) and `groups` / `completions` / `text` (legacy) via serde aliases.

### 3.6 The receipt schema — `kiln-train::receipt::EchoDiagnosticSummary`

- **`echo.lambda`** — the λ used.
- **`echo.env_ce_initial` / `echo.env_ce_final`** — paper §5.2's headline diagnostic.
- **`echo.env_ce_drop_pct`** — direct comparison to paper Figure 3.
- **`echo.lambda_effective_final`** — paper §3.3 auto-anneal verification.
- **`echo.env_tokens_supervised`** — the "did ECHO actually fire" smoke check.

These fields live on `DiagnosticSummary.echo: Option<EchoDiagnosticSummary>` so legacy receipts (where ECHO didn't run) are byte-identical post-schema-update.

---

## 4. The API surface

```
POST /v1/train/grpo       — Existing endpoint. Now accepts trajectory rollouts.
POST /v1/train/agentic    — Canonical alias. Same handler.
```

Request body:

```json
{
  "agentic_groups": [
    {
      "messages": [...],
      "rollouts": [
        {
          "text": "<flattened>",
          "reward": 1.0,
          "trajectory": [
            {"role":"assistant","content":"...","kind":"action"},
            {"role":"tool","content":"...","kind":"observation","warning_prefix_len":16}
          ]
        }
      ]
    }
  ],
  "config": {
    "lora_rank": 16,
    "lora_alpha": 32,
    "loss": {
      "echo": { "lambda": 0.05, "env_mask_mode": "env_only", "warning_filter": true },
      "opd": null,
      "no_policy_loss": false
    }
  }
}
```

Same hot-swap semantics as SFT/GRPO. The next inference call uses the new adapter.

---

## 5. Phased rollout (status)

### Phase 0 — Foundation: masks, naming, the whole bandaid ✅ SHIPPED
- Trajectory schema in `kiln-train::trajectory`.
- Masking primitive in `kiln-train::trajectory_mask`.
- Type aliases for back-compat (`ScoredCompletion`, `GrpoGroup`).
- `pi_trajectory.py` shared lib.
- `pi-doctest/rollout.py` migrated.
- `kiln-polish-prerequisites.md` §1 marked RESOLVED.

### Phase 1 — ECHO loss term ✅ SHIPPED
- `kiln-train::echo::echo_step_loss`.
- `LossConfig { policy, echo, opd, no_policy_loss }` composition surface.
- Wired into the uncheckpointed candle GRPO path.
- Wired into the checkpointed analytic-tail path (`analytic_grpo_tail_loss_grad_pre_final_norm` grows the `echo: Option<EchoTailParams>` parameter).
- Wired into vk-native (`vk_recompute_grpo_train_step_with_state` grows `echo: Option<VkEchoStepParams>`).
- CLI flags on `cuda_grpo_ablation`.
- KILN_ECHO_* env-var overrides (serialized via `ENV_LOCK` in tests).
- `POST /v1/train/agentic` route alias + `agentic_groups` serde alias
  on `GrpoRequest::groups` so the route's body matches the route name.
- Receipt schema (`EchoDiagnosticSummary`) including
  `dynamics_holdout_ce_initial` and `dynamics_holdout_ce_final` (paper
  §5.2 dynamics-test diagnostics).
- Appendix C acceptance tests pinned:
  - C.1 #1 — `lambda=0.0` is bit-equivalent to `echo=None`.
  - C.1 #3 — paper §3.1 normalization `mean_ce ∝ |O'|/|O|`.
  - C.1 #4 — checkpointed analytic-tail ECHO matches uncheckpointed
    path within 1e-3 on the trainer-level loss.
  - Phase 3 e2e — `no_policy_loss=true` upholds the linearity
    invariant `loss_full ≈ loss_grpo_only + loss_vf`.
- ~50 ECHO-related Rust tests across `echo.rs`, `receipt.rs`,
  `trajectory.rs`, `trajectory_mask.rs`, `lib.rs`, `trainer.rs`
  (kernel + masking + serde wire format + LossConfig + env-var
  overrides + end-to-end trainer paths + Appendix C acceptance gates);
  16 legacy GRPO/SFT tests still pass; 7 kiln-server training-API
  tests still pass; 15 Python `pi_trajectory.py` unit tests pass.

### Phase 2 — Paper reproduction in a separate cap ✅ SHIPPED (scaffold)
- `capabilities/agentic-grpo/pi-terminal-bench-lite` cap directory.
- `capability.md` documenting hypotheses + adversarial rubric review.
- `capability.config.json` with ECHO defaults.
- `rubric.py` with outcome-as-hard-floor composite.
- `rollout.py` emitting AgenticGroup JSONL via the shared lib.
- `build_corpus.py` with `--synthesize` mode for placeholder corpora.
- `calibration/sanity.py` — passes 6/6.
- `calibration/dynamics_holdout.py` — paper §5.2 test (stub-capable until kiln `prompt_logprobs` extension wires).
- **Open follow-up:** actual TBLite task data + paired ECHO-vs-no-ECHO runs on a pod to validate the +0.10 composite gate.

### Phase 3 — Polish + the second proof ✅ SHIPPED (scaffold + docs)
- `capabilities/agentic-grpo/pi-script-fixup` verifier-free cap.
- `LossConfig.no_policy_loss` flag landed in trainer.
- `--no-policy-loss` CLI flag in `cuda_grpo_ablation`.
- `docs/ECHO_GUIDE.md` (this document's operational peer).
- `docs/plans/grand-plan-for-extraordinarily-great-echo-for-everyone.md` (this document).
- README + CHANGELOG ECHO entries.

### Phase 4 — Compose with OPD (subsequent, dependent on OPD branch state)
When the OPD branch rebases on top of ECHO, the composition is mechanical:
- `LossConfig.opd` field is already reserved in `kiln-train::lib`.
- `analytic_grpo_tail_loss_grad_pre_final_norm` already accepts heterogeneous active positions (Action / Env); adding an OPD branch (PosKind::OpdActionRevKL) is a third arm in the same loop.
- The composition formula `L = L_policy + λ_echo · L_envCE + λ_opd · L_revKL` is structurally encoded.

---

## 6. Algorithmic choices, with citations

The defaults below are paper §3.3 / §5 conclusions ported into kiln.

| Choice | Default | Source |
| --- | --- | --- |
| ECHO mixing coefficient λ | 0.05 | Paper §3.3 (productive range 0.01–0.05) |
| Env-mask mode | `env_only` (excludes harness warning prefix) | Paper §3.2 (warnings memorize in ~60 steps) |
| Warning-prefix detection | Detect `WARNINGS:\n...` then locate `<command_output>` or `\n\n` | Empirical: kiln's pi harness convention |
| Length normalization | `|O|` (full observation length), not `|O'|` (warning-filtered) | Paper §3.1 |
| Loss composition with GRPO | Additive: `L = L_policy + λ · L_envCE` | Paper §3.1, Eq. (1) |
| Loss kernel | Reuses `kiln-flce-kernel` (candle) and `vk_flce_loss` (Vulkan) | No new kernel work needed |
| Auto-anneal | Constant λ; the env-CE shrinks naturally as the model learns terminal structure | Paper §3.3 (no λ-schedule needed) |
| Verifier-free regime | `--no-policy-loss` masks GRPO term, only env-CE drives gradients | Paper §5.5 |
| Verifier-free trajectory filter | Drop rollouts with parse errors / malformed tool calls | Paper §5.5 |

---

## 7. The three killer workflows

### 7.1 "Train a Python agent on my repo with ECHO" (Tier 2, ~30 min on A6000)

```bash
cd /workspace/kiln
# (Assume capability.config.json + datasets/train.tasks.jsonl exist.)
ECHO_MODE=on bash capabilities/agentic-grpo/pi-terminal-bench-lite/run_iter1.sh

# Run the ablation:
ECHO_MODE=off bash capabilities/agentic-grpo/pi-terminal-bench-lite/run_iter1.sh

# Compare in capability.jsonl.
```

The two runs produce paired ECHO vs no-ECHO adapters from identical rollouts; the only difference is the `--no-echo` flag.

### 7.2 "Keep my agent improving on new tasks without writing verifiers" (Tier 2, ~10 min)

```bash
# Take the strongest cap adapter.
BASE_ADAPTER=echo-tblite-iter5 \
OUTPUT_ADAPTER=echo-verifier-free-1 \
bash capabilities/agentic-grpo/pi-script-fixup/run_verifier_free.sh

# Diff baseline-vs-post pass rates across val100 / ITD / PyTerm / TBLite.
```

100 steps of `--no-policy-loss --echo-lambda 0.05`. Paper §5.5 target: +10 pp on PyTerm.

### 7.3 "Show me the ECHO contribution per adapter" (any tier, instantaneous)

```bash
jq '.diagnostic_summary.echo' /workspace/adapters/echo-tblite-iter5/receipt.json
# {
#   "lambda": 0.05,
#   "env_ce_initial": 4.21,
#   "env_ce_final": 0.83,
#   "env_ce_drop_pct": 80.3,
#   "lambda_effective_final": 0.07,
#   "env_tokens_supervised": 24576
# }
```

Receipt-grade evidence of ECHO contribution, paper §5.2 normalization.

---

## 8. The pit of success — how kiln auto-defends ECHO success

Mirrors grand-OPD-plan §8 design principle.

- **Default-on at safe λ.** New caps get ECHO at λ=0.05 automatically. The paper §3.3 productive range is 0.01–0.05; defaults are at the upper end of productive, not the low end of risky.
- **Zero-cost short-circuit.** Legacy single-turn rollouts (no `trajectory` field) get ECHO contribution = 0. `loss.echo = Some(EchoConfig)` is safe for every existing GRPO caller.
- **Warnings excluded by default.** Paper §3.2 says warning tokens memorize quickly and stop teaching; `MaskConfig::warning_filter = true` is the default.
- **Auto-anneal.** Paper §3.3 says λ=0.05 self-anneals as the model learns terminal structure. No λ-schedule needed; the receipt `lambda_effective_final` exposes the anneal trajectory.
- **Disjointness invariant.** `MaskedRollout::assert_masks_disjoint()` fires on every mask build — a token can't be both an Action and an Observation.
- **Mask boundary validation.** `build_masks_against_real_qwen_tokenizer` test runs on every PR via CI when `/workspace/Qwen3.5-4B/tokenizer.json` is available.
- **Compositional safety.** `LossConfig::default()` has ECHO on, OPD off. When OPD rebases, it slots in with `opd: Some(OpdAuxConfig)` — never replaces ECHO.

---

## 9. The contract for agentic users, in one sentence

> *Drop in pi_trajectory.build_scored_rollout, leave loss.echo at its default, and your agentic-GRPO cap trains the model on what its own actions produced — automatically, on every backend, without paying for a teacher.*

That's the entire promise. Phase 0–3 ship the implementation; this document is the user-facing arc that backs it up.
