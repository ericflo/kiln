# §15 Closing-Vignette Reproduction Scripts

The grand-plan's closing vision (`docs/plans/grand-plan-for-extraordinarily-great-on-policy-distillation-for-everyone.md` §15) sketches three users — Alice, Bob, Carol. Each vignette must be *reproducible from this branch on real hardware*. This document lays out the exact sequence of `kiln` CLI commands and HTTP requests that reproduces each vignette, with citations to the §3-§10 pillars each step exercises.

Pod-level reproduction (acquiring hardware, downloading the model, running the experiment to convergence) is out of scope for this document — that's the §13 success-criterion validation pass and runs against a leased CUDA pod. This document is the recipe; §13 is the validation.

---

## Vignette 1 — Alice (laptop student with a separate vLLM teacher)

> Alice opens kiln on her MacBook, points it at a separately hosted vLLM
> teacher, and distils her writing prompts into a local LoRA.

**Hardware:** any kiln-supported student host plus a separately provisioned
vLLM teacher. The current remote-teacher protocol is vLLM-only; the hosted
OpenRouter version of this vignette remains aspirational until it has a
dedicated, qualified adapter.

**Reproduction:**

```bash
# 1. Boot kiln (any backend; Metal on a MacBook, CUDA on a 4090).
kiln serve

# 2. Start vLLM for the teacher on another device/process. Raising
#    --max-logprobs to 32 is an explicit operator contract; stock vLLM caps
#    at 20 and Kiln will resolve the executable OPD width to 16.
vllm serve Qwen/Qwen3.6-27B --port 8000 --max-logprobs 32

# 3. Register the explicit vLLM protocol. Kiln never infers a provider from
#    a hostname or port.
curl -X POST localhost:8420/v1/teachers -H content-type:application/json -d '{
  "alias":         "qwen3.6-27b@vllm",
  "kind":          "remote",
  "provider":      "vllm",
  "model_id":      "Qwen/Qwen3.6-27B",
  "url":           "http://teacher-host:8000",
  "max_top_k":     32,
  "vocab_size":    152064
}'

# 4. Drop the writing corpus as a JSONL dataset on disk (one
#    {messages:[...]} per line) and register it via the dataset API.
#    For tiny corpora (<200 prompts) the §6 data-multiplier mode
#    auto-engages — samples_per_prompt scales 4 → 16 → 64.

# 5. Run the `frontier-pump` recipe (§3.5 + §3.7). The recipe ships
#    with the §6 paper-cited defaults: top-K=32, top-p=0.9, temp=1.0,
#    K=10× FullFT LR, all-linear-layer LoRA targets, rank-32 on
#    prosumer / rank-16 on laptop (§8.13 tier_defaults).
curl -X POST localhost:8420/v1/recipes/run -d '{
  "recipe": "frontier-pump",
  "inputs": {
    "teacher":    "qwen3.6-27b@vllm",
    "prompts":    "alice-writing.jsonl",
    "name":       "alice-writes-like-her"
  }
}'

# 6. When the run finishes, publish to the Adapter Library (§3.10)
#    with the reproducibility receipt (§8.11). Anyone with the same
#    teacher + same prompts + same seed rebuilds the same adapter.
curl -X POST localhost:8420/v1/library/publish/alice-writes-like-her
```

**Pillars exercised:**
- §3.2 `RemoteTeacher` + `/v1/teachers`.
- §3.3 `LogitCache` (cache hits for popular `(prompt, teacher)` prefixes).
- §3.5 knowledge-pump recipe.
- §3.7 recipe runtime with auto-chained `base_adapter`.
- §3.10 adapter library publish.
- §6 paper-cited defaults (top-K=32, top-p=0.9, temp=1.0).
- Remote vLLM transport is self-hosted and has no Kiln-side billing meter;
  `max_cost_usd` is rejected until a metered provider adapter exists.
- §8.11 reproducibility receipt — `AdapterReceipt::write_to_adapter_dir` runs at the end of every training job.
- §8.13 tier-aware defaults (rank 16/32/128 per laptop/prosumer/corporate tier).
- §11 12-trigger guardrail cascade (`LengthInflationGuardrail`).

**Reproducibility receipt format:** `<adapter_dir>/.kiln-receipt.json` (schema v1, `kiln_train::AdapterReceipt`). Records seed, teacher alias + model id, hyperparameters, prompt source descriptor, diagnostic summary.

---

## Vignette 2 — Bob (4090 office, pi pointed at kiln, weekly self-improve)

> Bob's startup has a 4090 in the back office. Every developer's terminal runs
> pi pointed at the office Kiln server. They use their separately hosted vLLM
> teacher to distil a turn-judge LoRA, then schedule local self-improvement
> rounds from their own recorded work.

**Hardware:** any CUDA 4090 (or stronger) host running the kiln server.

**Reproduction (the §10.14 five-command pipeline):**

```bash
# 1. One-time pi configuration — kiln backs up and merges
#    ~/.pi/agent/models.json plus settings.json, pointing pi at the
#    office kiln server without deleting existing providers.
kiln pi-setup --kiln-url http://office-kiln:8420

# 2. One-time judge distil (§10.6.1). Scores a week of pi-session
#    contested cases through the explicitly registered vLLM teacher and
#    trains a small judge LoRA. Hosted-provider protocols are not wired yet.
kiln judge distill \
  --teacher qwen3.6-27b@vllm \
  --sessions ~/.pi/agent/sessions/ \
  --judge-name office-judge \
  --rank 16

# 3. Use pi normally for a week. Sessions are captured to
#    ~/.pi/agent/sessions/<id>.jsonl automatically.

# 4. Saturday morning: GRPO loop using the local judge as the reward
#    model. CRISP terseness pass (§10.6.4) compresses successful
#    trajectories. New adapter behind an A/B gate the team reviews
#    Monday morning in the Trajectory Studio (§10.9).
kiln self-improve \
  --judge office-judge \
  --sessions ~/.pi/agent/sessions/ \
  --output-name office-assistant-2026-w20 \
  --post-eval pi-mini-mcpatlas

# 5. Optional periodic drift check (§10.6.3). Auto-refreshes the
#    judge if agreement on 50 contested cases drops below 80%.
#    NOTE: the scoring run lands with the trainer body (#31); until
#    then the server validates the inputs and returns 501, so this
#    command exits non-zero.
kiln judge drift-check \
  --judge office-judge \
  --teacher qwen3.6-27b@vllm \
  --sample-size 50
```

**Pillars exercised:**
- §10.3 Agent-trace layer (`/v1/agent/traces`).
- §10.4 Agentic OPD loss-shaping (SCoRe earliest-error weighting + TIP tool-call upweight + verifier reward).
- §10.6.1 Judge distil — `POST /v1/agent/judge_distill`.
- §10.6.2 Weekly self-improve — `POST /v1/agent/self_improve`.
- §10.6.3 Drift-triggered judge refresh — `POST /v1/agent/judge_drift_check`.
- §10.8 Agent-shaped recipes (`learn-from-my-pi-history`, `coding-assistant-from-repo`, `make-a-judge-lora-from-my-picks`, `merge-my-agent-loras`, `pi-share-then-pump`, `recover-tool-following`).
- §10.12 Agent benchmark suite (`SWE_BENCH_MINI`, `TERMINAL_BENCH_MINI`, `PI_MINI_MCPATLAS`).
- §10.14 Five-command CLI (`pi-setup`, `judge distill`, `self-improve`, `judge drift-check`, plus implicit `kiln serve`).
- §11 Loop-retry detection guardrails.

---

## Vignette 3 — Carol (single-GPU multi-teacher consolidation)

> Carol's team trains separate compliance adapters, retains each adapter's
> replay prompts, and consolidates them with bounded top-K behaviour-space
> distillation. Every resulting adapter carries a receipt.

**Hardware:** one supported training GPU for Kiln plus a separately provisioned
vLLM teacher for the specialist pumps. The executable loss is
`teacher_top_k`; full-vocabulary and multi-GPU training remain roadmap items.

```bash
# 1. Train each domain specialist via the targeted-domain pump.
for domain in fcra aml kyc sec-compliance tax-treaty fed-reg accounting audit; do
  curl -X POST localhost:8420/v1/distill/pump -d '{
    "name":       "carol-specialist-'$domain'",
    "teacher":    "qwen3.6-27b@vllm",
    "mode":       {"domain": "'$domain'"},
    "config":     {
      "training_mode": "on_policy",
      "loss": "teacher_top_k",
      "top_k": 16,
      "lora_rank": 128
    }
  }'
done

# 2. Consolidate via §3.4 behaviour-space multi-tenant merge. Each
#    source LoRA is loaded in turn, the model is run forward with
#    that LoRA applied, top-K teacher logprobs at active positions
#    are stashed in a unified FixtureLogitSource keyed by tokens_hash.
#    The trainer queries the *correct* source's teacher for each
#    prompt — no per-step LoRA swap, no multi-tenant inference server.
curl -X POST localhost:8420/v1/adapters/distill_merge -d '{
  "name":    "carol-unified-2026-q2",
  "sources": [
    {"adapter": "carol-specialist-fcra",          "weight": 1.0},
    {"adapter": "carol-specialist-aml",           "weight": 1.0},
    {"adapter": "carol-specialist-kyc",           "weight": 1.0},
    {"adapter": "carol-specialist-sec-compliance","weight": 1.0},
    {"adapter": "carol-specialist-tax-treaty",    "weight": 1.0},
    {"adapter": "carol-specialist-fed-reg",       "weight": 1.0},
    {"adapter": "carol-specialist-accounting",    "weight": 1.0},
    {"adapter": "carol-specialist-audit",         "weight": 1.0}
  ],
  "student":        "base",
  "rollout_budget": 50000,
  "config":         {
    "training_mode": "off_policy",
    "loss": "teacher_top_k",
    "top_k": 32,
    "lora_rank": 128
  }
}'

# 3. The auditor gets a reproducibility receipt for every adapter.
curl localhost:8420/v1/adapters/carol-unified-2026-q2/receipt
```

**Pillars exercised:**
- §3.4 Multi-tenant LoRA-as-teacher (`run_distill_merge` →
  `build_multi_tenant_merge_teacher`).
- §3.5 Targeted-domain knowledge pump.
- Supported `teacher_top_k` loss with K=16 for stock-vLLM pumps and K=32
  for local per-source merge fixtures.
- §8.11 Reproducibility receipt (the auditor's hook).
- §8.13 Corporate-tier defaults (LoRA rank 128–256).

---

## Why the §13 success criteria can't all be auto-validated in CI

The §13 phase-success criteria fall into three buckets:

1. **Validated by the test suite + the §9.9 bench gate** — CUDA kernel parity, throughput within 5% of baseline, recipe round-trips, receipt generation, all 12 guardrail triggers. These run on every PR.
2. **Validatable on a single-pod budget but not in CI** — Phase 0 IF-eval recovery, Phase 1 ≥10pt merge delta. These are GPU-hour experiments against real datasets (IFEval, MMLU-Pro, etc.). The pod-validation script `scripts/opd_phase0_pod_validation.sh` covers the build + kernel + recipe-round-trip half; the eval-suite scoring half needs the dataset registry + the eval queue to be wired (the registry exists; per-suite scoring pre-installed).
3. **Human-in-the-loop or community-scale studies** — Phase 2 cross-tier reproduction (3 different rigs), Phase 2 pit-of-success controlled study (20 users), Phase 2 pi-week A/B (a developer using pi for a week), Phase 3+ leaderboard / library traction. These are non-CI by construction and are explicitly annotated as non-goals for this branch in §5 / §13.

The branch ships every primitive these studies sit on top of.
