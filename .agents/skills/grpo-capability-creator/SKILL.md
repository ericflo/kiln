# grpo-capability-creator

Stand up one **GRPO** capability per session and run it through enough
iterations that you either ship a kept adapter or retire the capability
with a written lesson. This is the sister skill to
`opd-capability-creator` and `sft-capability-creator`: same loop, same
information-firewall discipline, same per-iteration verdict gate.

GRPO swaps the supervisory signal:
- **SFT** = curated assistant turns; gradient flows token-by-token
  against ground-truth labels.
- **OPD** = a teacher LM produces target logprobs; reverse-KL pulls the
  student toward the teacher distribution.
- **GRPO** = the student generates `N` rollouts itself; a **reward
  function** scores each; group-relative advantages drive a clipped
  policy-gradient step.

There is no teacher in the loop. The reward function takes the teacher's
role and consequently has to be load-bearing — not vibes-graded.

## Skill inventory

- `sft-capability-creator` — supervised fine-tuning on curated assistant
  turns. Use when you have ground-truth pairs.
- `opd-capability-creator` — on-policy distillation against a teacher
  LM. Use when a stronger teacher exists and the capability is shape /
  format / refusal-pattern transfer.
- **`grpo-capability-creator`** (this skill) — group-relative
  policy-gradient on **scored** rollouts. Use when you have a
  *verifier* (deterministic check) or a *judge* (LLM with a rubric)
  that returns a reward for any complete response.
- `agentic-grpo-capability-creator` — GRPO with multi-turn pi rollouts.
  Use when the capability is *agentic* — the model must use tools, not
  just emit text.

---

## 0. Mental model

### When GRPO is the right tool

Reach for GRPO when **all** of these are true:

1. There exists a programmatic reward function that, given a complete
   response, returns a score in `[0, 1]`.
2. That reward function is **stable** under perturbations of input
   surface form — paraphrases of the same correct answer get similar
   scores.
3. The base model produces *some* signal — not pure noise — on the
   training distribution. (GRPO needs reward variance within groups;
   if every rollout scores 0.0 you have no advantage signal and
   dynamic sampling drops every group.)
4. The capability has a tractable **rollout token budget**. GRPO trains
   on the model's own outputs, so one training step ≈ `N × max_tokens`
   tokens per prompt. On A6000 with rank 16, ~6000 tokens/group is the
   practical ceiling.

Reach for **OPD instead** when (1) is hard but a teacher exists. Reach
for **SFT instead** when you can write ~200 ground-truth pairs.

### GRPO ≠ "magic free improvement"

GRPO is policy gradient on top of your reward function. **Everything
the reward function rewards, the model learns.** If your reward
function rewards "matches a regex," the model converges to "shortest
string matching that regex." If your reward function rewards a
multi-component composite where one component is `length_band` with
positive weight, the model converges to "the length the band centers
on, no matter what's inside."

The published GRPO failure modes are all reward-function failure modes:
- **Length drift** — DAPO §2 (arXiv:2503.14476). Per-sample averaging
  under-penalises long wrong outputs.
- **Mode collapse** — Magistral (arXiv:2506.10910). Symmetric clip plus
  KL anchor against base model collapses exploration.
- **Entropy collapse** — Cui et al. (arXiv:2506.01939). All tokens
  contribute KL, including low-uncertainty ones, killing exploratory
  tokens.

Kiln's Phase 1 defaults (`DrGrpo` + `TokenLevel` + `dynamic_sampling`)
fix #1. Clip-Higher (`clip_eps_high`) is in-tree as an opt-in for #2.
`entropy_aware_kl_quantile` is in-tree as an opt-in for #3. Use them.

### The reward function IS the spec

The same lesson as "the eval is the spec" from sft/opd, but with sharper
teeth. In SFT/OPD, the eval *measures* the capability after training.
In GRPO, the reward function *defines* the capability during training.
A bad reward function doesn't just mis-measure — it *trains the wrong
behaviour into the model*.

**You must do the adversarial review before iter 1.** For every
sub-score in your reward function, write down: *what's the cheapest
response that maximises this sub-score without doing the capability?*
Then either add an anti-shortcut sub-score that punishes that cheat or
move on with eyes open.

Lessons backported from sft / opd cap closeouts:

- **Baseline ≥ 0.95**: reward function too lax. The model already
  passes; GRPO will not lift it because there's no group-internal
  variance. Inspect 3-5 baseline rollouts; tighten the rubric.
- **Baseline < 0.30 on a capability the 4B should partially have**:
  reward function too strict. Inspect 3-5 baseline rollouts; loosen
  *without re-opening Goodhart holes* (cap #5 OPD: rubric rejected
  fenced-code wrapper that the model legitimately emits).
- **Baseline reward variance across rollouts == 0**: every rollout
  scores identically. Either the prompts are trivial, the prompts are
  impossible, or the rubric is granularity-blind. Either way, GRPO has
  no signal — fix the prompts before training.

**Inspect 3-5 baseline rollouts in writing before generating iter 1.**
Cap #5 closeout: trying to train against a mis-measured baseline
destroyed the LoRA. The all-zeros gate (§10) caught it ex-post; the
right thing was not to have started.

### Headroom and group-variance preview

Before iter 1, compute:

- `composite_baseline` — mean reward on the held-out eval set, base
  model
- `composite_headroom` = `1.0 - composite_baseline` — the upper bound
  on what GRPO can possibly buy you
- `group_var_baseline` — mean within-group reward variance on training
  prompts (sample 8 rollouts per prompt at training-time sampling
  params, compute variance, mean across prompts)

If `composite_headroom < 0.05`, no GRPO gradient is going to find
anything. If `group_var_baseline < 0.01`, dynamic sampling will drop
every group. In either case: stop and fix the rubric or the prompts.

---

## 1. Information firewall (non-negotiable)

Same as `opd-capability-creator` §1. Briefly:

- The agent driving the loop is the **outer agent**. It writes
  `rubric.py`, `capability.oracle.sh`, hypothesis docs, and verdicts.
- The model under training and the model used to sample rollouts at
  eval time MUST NOT see the rubric, the eval prompts, or the
  hypothesis text. If you want a calibration set, generate it before
  the model has ever seen any of it.
- Eval prompts and training prompts come from disjoint pools. Hold out
  ~20% of your prompt pool as the eval set on day 1 and never touch
  it.
- Sub-agents inherit the firewall. If you spawn a sub-agent to inspect
  rollouts, hand it the rollouts, not the rubric, not the eval set.

Violation = the cap is retired; the artifact is unsalvageable.

---

## 2. Session files

```
capabilities/grpo/<slug>/
├── capability.md              # Design doc — what + why + rubric + adversarial review
├── capability.config.json     # Model path, kiln URL, sampling params, GRPO hyperparams
├── capability.jsonl           # Append-only iteration log (one JSON line per iter)
├── capability.oracle.sh       # Blind eval wrapper: takes adapter name, prints SCORE=
├── rubric.py                  # The reward function. ALSO the eval rubric.
├── build_corpus.py            # Builds train + eval JSONL from upstream sources
├── kiln-polish.jsonl          # Observations about kiln itself surfaced during the cap
├── calibration/
│   ├── good.jsonl             # 3 "ideal" responses; rubric must score ≥0.7
│   └── bad.jsonl              # 3 "obviously bad" responses; rubric must score ≤0.3
├── datasets/
│   ├── train.grpo.jsonl       # GrpoGroup JSONL: one line per prompt, no completions
│   └── eval.jsonl             # Held-out eval prompts (NEVER seen by the model in training)
├── prompts/                   # Per-iter prompt sets (filtered subsets of train.grpo.jsonl)
├── hypotheses/                # One markdown per hypothesis (h1-foo.md, h2-bar.md, ...)
└── run_iter<N>.sh             # Per-iter training command — reproducible
```

### `capability.md` template

```markdown
# Capability: <slug>

## Description
<2-4 sentences: what the model should do, concrete failure modes the 4B
exhibits today.>

## Base model
Qwen3.5-4B (kiln serve on http://localhost:8420)

## Rollout source
Self-rollout via kiln /v1/chat/completions. Sampling params:
temperature=<T>, top_p=<P>, max_tokens=<M>. N=<G> rollouts per prompt
per training step.

## Reward function (designed with adversarial review applied — §0)

| Sub-score | Weight | What it measures | What it CANNOT be cheated by |
|-----------|--------|-------------------|-----------------------------|
| ...       | ...    | ...               | ...                         |

Composite = sum(weight_i × sub_score_i). Direction: higher is better.

### Adversarial design (§0)

Q: What's the cheapest way to score 1.0 without doing the capability?
A1: ...
   Mitigation: ...
A2: ...
   Mitigation: ...

Q: What does the within-group reward distribution look like at baseline?
A: <after sampling>

### Headroom

- baseline composite: <0.xx>
- headroom: <1.0 - composite>
- baseline group variance: <0.xx>

## Hypothesis log

| Iter | Slug | Family | Composite | Δ | Status | Notes |
|------|------|--------|-----------|---|--------|-------|
| 0    | baseline | — | 0.xx | — | — | — |
```

### `capability.jsonl` schema

```json
{
  "iter": 1,
  "slug": "h1-default-recipe",
  "ts": "2026-05-18T20:00:00Z",
  "status": "kept | discarded | infra-fail",
  "family": "H1 | H2 | ...",
  "target_sub_score": "applies_cleanly",
  "hypothesis": "single sentence",
  "verdict": "✓ kept (+0.04) | ✗ regressed (-0.02) | infra: ...",
  "composite": 0.78,
  "composite_delta": 0.03,
  "sub_scores": {"strict_format": 0.95, "applies_cleanly": 0.80, ...},
  "regressions": [{"sub_score": "...", "delta": -0.05, "severity": "minor|moderate|catastrophic"}],
  "training": {
    "advantage_mode": "dr_grpo",
    "loss_aggregation": "token_level",
    "kl_estimator": "k1",
    "kl_coeff": 0.1,
    "clip_epsilon": 0.20,
    "clip_eps_high": null,
    "dynamic_sampling": true,
    "is_level": "token",
    "reference_policy": "base_per_step",
    "entropy_aware_kl_quantile": null,
    "num_generations": 8,
    "lr": 1e-5,
    "rank": 16,
    "alpha": 32,
    "max_tokens": 512,
    "temperature": 0.9,
    "top_p": 0.95,
    "groups_seen": 30,
    "groups_dropped_degenerate": 18,
    "groups_kept": 12,
    "mean_within_group_variance": 0.12,
    "fraction_clipped_low": 0.04,
    "fraction_clipped_high": 0.11,
    "elapsed_s": 290
  },
  "rollout_stats": {
    "mean_reward": 0.42,
    "p50_reward": 0.50,
    "p95_reward": 1.00,
    "p05_reward": 0.00
  },
  "asi": {
    "what_worked": "...",
    "what_failed": "...",
    "next_focus": "..."
  },
  "kiln_polish_noted": true,
  "git_sha": "abc1234"
}
```

---

## 3. The loop

### Phase 0 — Intake (one-shot, once per session)

1. Pick the capability. Write `capability.md` description and the
   sub-score table.
2. **Adversarial review.** For each sub-score, name a shortcut. For
   each shortcut, name a mitigation.
3. Write `rubric.py` (`score_response(**fields) -> dict`). The dict
   must include every sub-score plus `composite`.
4. Hand-write `calibration/good.jsonl` + `calibration/bad.jsonl` (3
   each).
5. Run `python3 $SKILL/templates/rubric_sanity.py` — must exit 0.
6. Write `build_corpus.py`. Build `datasets/train.grpo.jsonl` and
   `datasets/eval.jsonl` (disjoint). Aim for 100-500 training prompts
   and 20-50 eval prompts.
7. Stand up `kiln serve` with the base model. Health-check it.
8. **Baseline eval.** `./capability.oracle.sh ""` (empty adapter =
   base model). Log iter 0.
9. **Group-variance baseline.** Sample 8 rollouts per prompt on 30
   random training prompts at training-time `temperature` + `top_p`,
   compute within-group reward variance, mean across prompts. Log to
   `capability.md`.
10. **Baseline sanity gate.** Apply the §0 framework:
    - ≥ 0.95: rubric too lax. Inspect 3-5, tighten rubric, redo.
    - < 0.30 with seed-form capability: rubric too strict. Inspect
      3-5, loosen *without re-opening Goodhart holes*.
    - mean_within_group_variance < 0.01: prompts too easy or too
      hard or rubric granularity-blind. Diagnose before iter 1.
11. **Tiny-smoke** (§17): 1 GRPO group, 1 optimizer step, no eval.
    Confirms training infra is up.
12. Commit `capability.md`, `rubric.py`, `build_corpus.py`,
    `datasets/`, `calibration/`, `capability.oracle.sh`,
    `capability.config.json`, `capability.jsonl` (with iter 0).

### Phase 1 — Hypothesise (every iteration)

Pick **one** hypothesis. Pick a target sub-score. Write
`hypotheses/<slug>.md` containing:

- **Family** — H1 / H2 / ... (see §4)
- **Claim** — one sentence. "Lifting `dynamic_sampling=true` with
  `clip_eps_high=0.28` raises `applies_cleanly` by ≥5pp without
  regressing `strict_format` by >2pp."
- **Mechanism** — one paragraph. *Why* this should lift the target
  sub-score, grounded in the literature or prior cap experience.
- **Falsification plan** — what observation would make you discard
  this hypothesis.
- **Verdict** — left blank until Phase 5.

### Phase 2 — Construct (every iteration)

Build `prompts/<slug>.jsonl` — the filtered subset of training prompts
for this iteration. Filtering: drop prompts whose baseline rollout
group has variance < 0.05 (no signal) AND prompts whose baseline mean
reward > 0.9 (already solved). The kept prompts are where GRPO has
purchase.

Per-iter prompt counts: 30-100 prompts × 8 generations = 240-800
rollouts per epoch. One epoch is usually plenty.

### Phase 3 — Train (every iteration)

Run `run_iter<N>.sh`. It must:
- Be self-contained — reproducible from the script alone.
- Log to `/tmp/grpo-<slug>.log` so loss/group-stats are tail-able.
- Set every GRPO hyperparameter that differs from the kiln default
  *explicitly* — no defaults that change under you.

### Phase 4 — Blind eval — ALL sub-scores

Run `./capability.oracle.sh <adapter-name>`. Record every sub-score, not
just the composite. The oracle is **blind** — the adapter name is
the only signal the eval gets. The eval set is the same one used at
iter 0; it never changes within a session.

### Phase 5 — Close the loop (the verdict gate)

This is the gate that prevents the iteration count from outrunning the
hypothesis count. **No iter is logged until Phase 5 completes.**

Mandatory artifacts before `log_iter.sh` accepts:

1. **Verdict line** in `hypotheses/<slug>.md` — one of `✓ kept`,
   `✗ regressed`, `? inconclusive`, `infra: <reason>`.
2. **Inspected ≥3 rollouts** — paste excerpts in the hypothesis file
   showing whether the model is doing what you hoped.
3. **Regression list** — every sub-score that regressed by >0.02 with
   severity (minor < 0.05, moderate 0.05-0.15, catastrophic > 0.15).
4. **`asi` block** — what_worked / what_failed / next_focus, written
   *before* you look at iter N+1 ideas.
5. **kiln-polish flag** — true if you noticed something about kiln
   itself (a bug, a missing knob, a confusing log line) and logged it
   to `kiln-polish.jsonl`.

`log_iter.sh` rejects appends with any of the above missing.

### Phase 6 — Iterate

- **Kept** iter → branch from this adapter for the next hypothesis.
- **Discarded** iter → branch from the prior kept adapter (or
  baseline) — never stack on a discard.
- **Infra fail** → don't log it as a "kept/discarded" iter; log it as
  `status: "infra-fail"` and rerun once infra is fixed.

### Phase 7 — Cadence checkpoints (every 3 iters)

Write a one-paragraph update to `capability.md` summarising what was
learned across the last 3 iters. If you can't, you're not learning —
stop and look at what the rollouts are actually doing.

---

## 4. Hypothesis families (GRPO-specific)

| Family | What you change | When to reach for it |
|--------|-----------------|---------------------|
| **H1 — Default recipe** | Phase 1 defaults; lr 1e-5, rank 16, alpha 32, num_generations 8, max_tokens equal to baseline p95 length × 1.2 | Iter 1, always. Establishes the kiln-default baseline before any clever lever. |
| **H2 — Clip-Higher** | `clip_eps_high: Some(0.28)` (asymmetric clip) | When iter 1 has a high `fraction_clipped_high` (>0.10) — model wants to up-weight tokens but the clip is preventing it. |
| **H3 — Entropy-aware KL** | `entropy_aware_kl_quantile: Some(0.8)` | When iter 1 shows entropy collapse — diverging entropy curve, mode-collapsed rollouts. |
| **H4 — Drop the KL anchor** | `reference_policy: None`, `kl_coeff: 0` | When iter 1 with base-anchored KL underperforms a known-good baseline; the base model is the wrong anchor for the capability. |
| **H5 — EMA reference** | `reference_policy: Ema { decay: 0.99, refresh_every: 32 }` | When you want gentle anchoring but the base model has anti-capability priors. **Decay = 0.0 is a footgun** — use ≥0.9. |
| **H6 — GSPO sequence-level IS** | `is_level: Sequence` | When per-token IS variance is exploding (loss spikes, gradient norm spikes correlated with completion length). |
| **H7 — Reward reshape** | Edit rubric.py — add an anti-shortcut sub-score, rebalance weights | When sub-score regression watch (§11) flags the model is moving toward a known cheat. Costs a full re-baseline. |
| **H8 — Prompt difficulty rebalance** | Edit `prompts/<slug>.jsonl` to over-sample harder prompts | When `groups_dropped_degenerate / groups_seen > 0.7` — most groups have no signal. |
| **H9 — Sampling temperature** | Raise rollout temperature (0.9 → 1.2) | When `group_variance` is too tight; exploration too narrow. |
| **H10 — Longer rollouts** | Increase `max_tokens` (rollout) | When rollouts are truncating mid-answer (`truncation_rate > 0.10`). |
| **H11 — More generations** | Increase `num_generations` (8 → 16) | When you can afford the VRAM and `group_variance` is too noisy for stable advantages. Quadratic memory cost. |

Always start at H1. **Never combine two hypotheses in a single iter.**
The iteration log loses signal — you can't tell which lever moved the
score.

---

## 5. Anti-laziness gates (consolidated)

- **No iter without a hypothesis doc.** `log_iter.sh` rejects.
- **No iter without inspected rollouts.** Same gate.
- **No two-knob iters.** One hypothesis = one knob change.
- **No iter on a 0.95+ baseline.** Re-baseline first.
- **No iter on a <0.30 baseline with seed-form capability.**
  Re-baseline first.
- **No iter when `groups_dropped_degenerate / groups_seen > 0.7`.**
  Rebalance prompts first.
- **No reward-function edit without re-running calibration.** §6.
- **No silent kiln-polish.** Log it or don't note it.

---

## 6. Reward design — the actual deliverable

A reward function is a Python callable:

```python
def score_response(response: str, **prompt_context) -> dict:
    """Return {sub_score_name: float in [0,1], ..., 'composite': float}."""
```

It runs in two places:
1. **At training time** — inside the GRPO rollout pipeline, scoring
   each of N rollouts per prompt. The composite becomes the reward.
2. **At eval time** — inside `capability.oracle.sh`, scoring rollouts
   on the eval set. Same function, same weights, same prompt
   context.

The two paths use the **same code**. If they diverge, the model trains
against one signal and is judged against another, and the iteration
loop loses meaning.

### Reward components (multi-component composite)

- **Outcome reward (mandatory)** — did the model do the thing? Pass /
  fail or a graded match. Highest weight.
- **Format reward** — is the response in the expected shape (valid
  JSON, valid diff, function definition, etc.)? Cap-#5 lesson: this
  one is full of Goodhart holes; accept the *natural* model output
  shape, reject prose-around-answer.
- **Anti-cheat sub-scores** — one per shortcut named in adversarial
  review. Subtract from composite if the cheat triggers.
- **Efficiency reward (optional)** — penalty proportional to wasted
  effort (excessive tool calls, repeated output, long but uninformative
  prefixes). Use only when you have a reason to suspect inefficiency.

### Anti-reward-hacking checklist

Before iter 1, ask of every sub-score:

- Can the model maximise this sub-score with an empty response?
- Can the model maximise this sub-score by repeating the prompt back?
- Can the model maximise this sub-score with one canonical answer
  regardless of input?
- Can the model maximise this sub-score with a short response that
  partially matches?

For each YES, add an anti-shortcut sub-score that catches it.

### Stability under paraphrase

A reward function must score paraphrases of the same correct answer
similarly. Test this manually on 3-5 known-correct responses in
different phrasings. If the rubric gives 0.4 / 0.8 / 0.6 on what are
obviously equivalent answers, your rollout statistics will be noise and
GRPO won't converge.

---

## 7. Group statistics watch

Pull these out of every training log; track them in `capability.jsonl`
per iter:

- **`groups_seen`** — total groups the loader yielded.
- **`groups_dropped_degenerate`** — groups dropped by `dynamic_sampling`
  (all rollouts had identical reward). Ratio > 0.5 is a yellow flag;
  > 0.7 is a red flag — your prompts have no within-group signal.
- **`groups_kept`** — groups that contributed gradient.
- **`mean_within_group_variance`** — mean reward variance across kept
  groups. Below 0.01 means the rollouts are too similar; raise
  temperature or rebalance prompts.
- **`fraction_clipped_low`** / **`fraction_clipped_high`** — per-token
  fraction of tokens whose IS ratio hit the clip range. Both should
  be < 0.10. `clipped_high > 0.10` is the Clip-Higher signal.
- **`mean_reward`** + **`p05_reward`** + **`p95_reward`** — rollout
  reward distribution. Watch for collapse: p95 falling = exploration
  shrinking; p05 rising = floor improving.

If these aren't logged, your iter is partly blind — half of "why did
this iter work / not work" lives in these numbers.

---

## 8. Memory budget for rollouts

Rough rule for A6000 (48GB) with rank-16 LoRA on Qwen3.5-4B:

- Base model + KV cache: ~16 GB
- LoRA + optimizer state: ~2 GB
- Rollout buffer (N × max_tokens × hidden_size × 2 [policy + ref]): 
  `8 × 512 × 2560 × 2 × 2 bytes ≈ 42 MB` per group
- Activation memory during training: scales with `group_total_active`

The practical ceiling per group is roughly `num_generations × max_tokens
≤ 6000 tokens`. Above that, expect OOM during backward. If you need
longer rollouts, drop `num_generations` first.

A100-80GB doubles all numbers.

---

## 9. Loss is deceptive

GRPO loss sign convention: `loss = -mean(advantage × clipped_ratio +
kl_coeff × kl_term)`. Negative loss = positive advantage × ratio
average — that's "the model is moving in the rewarded direction" — and
**loss going more negative is good**.

But:
- A flat loss curve does NOT mean nothing is happening. It can mean
  advantages are small (the model is already near-optimal on this batch)
  or that exploration has collapsed.
- A spiking loss curve does NOT necessarily mean catastrophe. Per-token
  IS variance is high during early training; spikes that resolve are
  normal. Persistent spikes (3+ steps in a row) are not normal.

Trust the **eval composite** and the **group statistics**, not the loss.

---

## 10. The all-zeros failure mode

Same as opd §10. If iter N composite < `0.5 × baseline`, the adapter
has been wrecked. Don't argue with the number — discard the iter,
inspect rollouts, find the wreck cause.

The cap-#5 closeout: this gate would have caught the catastrophic
iter even when the *baseline was mis-measured*. The gate IS the right
backstop; rubric-design failure is the upstream cause but the gate is
the last-line defense.

---

## 11. Sub-score regression watch

After every iter, for every sub-score:
- ≥ +0.02: improvement (call it out)
- 0.0 to +0.02: noise (no claim)
- -0.02 to -0.05: minor regression (note it; iterate)
- -0.05 to -0.15: moderate regression (note it explicitly in
  `regressions`; consider rolling back)
- > -0.15: catastrophic (discard adapter; investigate)

A composite uplift driven by a sub-score that's a Goodhart shortcut is
**not** a real lift. Cap-#3 closeout: `length_band` lifted +29.7pp
while `entity_recall` (the *target*) was flat — the composite went up
but the adapter learned the wrong thing.

---

## 12. Stop conditions

Stop the cap when **any** of:

- Composite lift ≥ +0.10 over baseline, with no regression > 0.05 →
  ship the adapter.
- 5 consecutive iters with no composite improvement → retire the cap;
  write a closeout lesson.
- A reward-function edit became necessary mid-session (rubric-design
  failure) → that's a Phase 0 redo; retire the cap, ship a new
  capability with the corrected rubric.
- Memory / time budget hit → close out with whatever you have.

---

## 13. Closeout checklist

Write `closeout.md` in the cap dir:

1. **Outcome** — kept adapter / retired with lesson / abandoned.
2. **Baseline → final composite** — the headline numbers.
3. **Iter table** — copy the hypothesis log from `capability.md`.
4. **Lessons backported** — list edits made to this SKILL.md or to
   `opd-capability-creator/SKILL.md` or `sft-capability-creator/SKILL.md`.
5. **kiln-polish forwarded** — link to `kiln-polish.jsonl` entries
   that became kiln bugs / feature requests.
6. **Next capability** — what to do differently next time.

---

## 14. Resuming a session

The session is fully recoverable from disk:

```bash
cd capabilities/grpo/<slug>
tail -1 capability.jsonl   # latest iter
ls hypotheses/             # what's been tried
cat capability.md          # design + adversarial review + hypothesis log
```

If `capability.jsonl` has an iter with `status: "in_progress"` and you
don't remember where it left off — discard it; re-run the iter.

---

## 15. Sanity-check rollouts periodically

Every 3 iters, inspect 5 rollouts from the latest training set by hand.
You're looking for:
- The model is responding to the prompt (not echoing, not refusing)
- The format is what the reward function expects
- The model is doing something semantically related to the capability,
  not just gaming a sub-score

If anything looks weird, look at the *training* prompt set — not just
the rollouts. The prompts shape what the model sees as "the task."

---

## 16. Kiln polish ledger

`kiln-polish.jsonl` — one observation per line. Fields:
`{ts, iter, severity, category, summary, repro_hint}`.

Categories: `bug`, `missing-knob`, `confusing-log`, `perf`, `docs`.

These are the input to GitHub issues / PRs against kiln itself. Every
cap should produce 0-5 entries — if zero, you didn't poke hard enough.

---

## 17. Tiny-smoke

Before paying for a full iter, run a smoke that exercises the full
training path with N=1 group and 1 optimizer step. Confirms:
- kiln is reachable
- the adapter dir is writable
- the rollout pipeline returns N completions
- the rubric runs on those completions and returns the right shape
- one optimizer step doesn't OOM

```bash
bash $SKILL/templates/tiny_smoke.sh
```

If it doesn't exit 0 in under 90s, your infra is wrong and no iter
will succeed.

---

## 18. One-screen quickstart

```bash
SKILL=.agents/skills/grpo-capability-creator

# 0. Intake (one-shot)
SLUG=python-doctest-passrate
$SKILL/templates/scaffold.sh $SLUG
cd capabilities/grpo/$SLUG
# edit capability.md, rubric.py, capability.oracle.sh, capability.config.json
python3 build_corpus.py     # produces datasets/{train,eval}.jsonl
python3 $SKILL/templates/rubric_sanity.py   # MUST exit 0

# Stand up kiln serve with base model
curl -sf http://localhost:8420/v1/models

# Baseline
./capability.oracle.sh ""             # logs iter 0
python3 $SKILL/templates/group_variance_baseline.py    # measures within-group variance

# Tiny-smoke
bash $SKILL/templates/tiny_smoke.sh   # 1 group, 1 step, no eval

# Iter 1 — always H1 (default recipe)
ABL=h1-default-recipe
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/$ABL.md
bash run_iter1.sh
RESULT=$(./capability.oracle.sh $ABL)

# Phase 5 — verdict gate
# - inspect ≥3 rollouts; paste excerpts into hypotheses/$ABL.md
# - fill verdict, what_worked, what_failed, next_focus
# - log kiln-polish if needed
bash $SKILL/templates/log_iter.sh $ABL <fields>
# log_iter.sh rejects if verdict/asi/regressions/polish-flag empty

git add -A && git commit -m "cap[$SLUG/$ABL]: kept (+0.04)"

# Iter 2... follow §3 Phase 6. Stop when §12 fires; closeout per §13.
```

The whole skill is the discipline that makes the loop close cleanly.

---

## Reference: kiln GRPO defaults (Phase 1 ablation-validated)

Set explicitly in every `run_iter*.sh` so iter logs are self-describing
even if defaults change:

```toml
advantage_mode = "dr_grpo"          # subtract group mean, no std normalization
loss_aggregation = "token_level"     # DAPO Token-Level Loss
kl_estimator = "k1"                  # Schulman k1
kl_coeff = 0.1
clip_epsilon = 0.20                  # symmetric clip
clip_eps_high = null                 # asymmetric Clip-Higher: opt-in
dynamic_sampling = true              # drop degenerate groups
is_level = "token"                   # per-token IS (default)
reference_policy = "base_per_step"   # KL anchored to base model
entropy_aware_kl_quantile = null     # full-token KL (default)
```

Phase 1 + 2 + 3 references:
- Phase 1 — Dr. GRPO (arXiv:2503.20783), DAPO (arXiv:2503.14476),
  SimpleRL-Zoo (arXiv:2503.18892), Magistral (arXiv:2506.10910).
- Phase 2 — GSPO (arXiv:2507.18071), CISPO (arXiv:2506.13585).
- Phase 3 — Cui et al. entropy-aware KL (arXiv:2506.01939),
  Open-Reasoner-Zero (arXiv:2503.24290).

Lessons backported from cap closeouts:
- cap #1 (faithful-code-summarization): baseline ≥0.95 = rubric too
  lax, not "capability solved."
- cap #5 (diff-patch-fluency): baseline <0.30 with seed-form capability
  = rubric too strict. Inspect responses BEFORE training.
- cap #4 (tool-call-arg-fidelity): composite uplift driven by
  Goodhart-able sub-score (presence) without target sub-score moving =
  not a real lift.
- cap #3 (transcript-compaction): same pattern, `length_band` confound.
