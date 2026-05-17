---
name: opd-capability-creator
description: Autonomous on-policy distillation loop for lifting one measurable sub-score of a capability rubric, by sampling rollouts from a kiln student and grading them token-by-token against a live teacher LM. Use when asked to "distill capability X", "lift the validates / faithfulness / passes sub-score on Y", "uplift a kiln LoRA from a 27B teacher", or "OPD this skill into a 4B". Each iteration is one independent (prompts, hyperparameters) attempt; the eval is blind; the agent is prevented by explicit gates from skipping verdicts, regression checks, or reflection.
---

# opd-capability-creator

> Autonomous loop for **uplifting** a target capability via on-policy distillation. The student samples its own trajectories under the active LoRA. The teacher grades each sampled token by reverse-KL against the student's distribution at that state. Gradients flow back into the LoRA. Every iteration is one independent (prompts, hyperparameters) attempt whose hypothesis is stated in a sentence and closed with a verdict before the next iteration can start.

This skill is the OPD sibling of `sft-capability-creator`. The disciplines that transfer:
- the experimental record (one row per attempt, immutable once kept)
- the blind oracle (you see one score, never its internals)
- hypothesis-before-data, falsification-plan-before-result
- triangulation as the path to causal claims

The disciplines that don't transfer (and why):
- **You don't curate datasets.** OPD's "data" is the prompt set. The assistant turn is decoration — the student samples its own. Your design space is *prompts, hyperparameters, and which sub-score you're targeting*.
- **You stand up a teacher.** Unlike SFT (teacher is a file), OPD needs a live model server. Standing it up and cleaning up after it is part of the loop.
- **Loss is deceptive.** OPD loss is spiky by nature — perfect agreement on some states, large disagreement on others. Trust the blind eval, never the loss.
- **Hypothesis families are different.** Not "prose vs. numeric"; rather "epoch curve vs. prompt diversity vs. rank capacity vs. teacher-warm-start".

---

## 0. Mental model

You are an experimentalist whose budget is mostly GPU memory and time. The lab has:

- a fixed **base student** (kiln-served, e.g. Qwen3.5-4B)
- a **target capability** described in plain English and instrumented with a rubric of weighted sub-scores
- a **live teacher** (vLLM-served, Q4-quantized, from the same model family at 14–32B)
- a **blind oracle** — a command that returns one composite scalar plus its breakdown by sub-score

You produce **independent** OPD adapters from the base. Each is an attempt: a prompt set + hyperparameters, trained for N epochs at rank R, then scored. You never stack OPD on OPD. Every adapter starts from base.

### Where OPD shines and where it doesn't

OPD bridges a *distribution gap*. The student already has the capability in some form; the teacher has it in better form; OPD trains the student to match the teacher at states the student actually visits.

OPD helps when:
- baseline composite is in `(0.4, 0.95)`
- at least one sub-score has visible headroom
- the student already produces text in roughly the right shape (it can fail at the capability, but it should at least *try*)

OPD does **not** help when:
- the student needs fundamentally new knowledge (use SFT or pretraining)
- the rubric is already saturated (no headroom to capture)
- the student samples in distributions the teacher considers junk (no overlap → no gradient signal — see §8 on student-teacher overlap)

### The headroom principle (read first, plan from this)

Composite is a weighted sum. Each sub-score `s_i` with weight `w_i` and baseline `b_i` contributes at most `w_i × (1 − b_i)` to a future composite uplift. **Headroom = `Σ w_i × (1 − b_i)`.** Most of it usually lives in one or two sub-scores; the rest are saturated.

Before writing a hypothesis, look at headroom and pick the sub-score you're targeting. If headroom is < 0.05, the rubric is too saturated for OPD to be interesting; flag to the user. If headroom is concentrated in one sub-score, you have a clear target; if spread evenly, expect composite to move slowly even when individual sub-scores move.

`templates/headroom.py` does this analysis; run it after every baseline.

---

## 1. Information firewall (non-negotiable)

The eval is **blind**. The oracle returns one composite score plus per-sub-score breakdown. You see those numbers and nothing else.

You MUST NOT:
- read the eval set's prompts or expected outputs
- read per-example judgments
- design prompts to invert the eval ("probe what passes; train to pass it")
- ask the user "what does the eval check for?"

You MAY:
- ask the user for a plain-English description of the capability
- read the score the oracle returns (composite + sub-score breakdown)
- ask the user for categorical hints if volunteered ("the eval is short-answer", "the eval scores faithfulness via entity match")
- inspect sample responses on a *non-eval* prompt of your own choosing, to debug structural failures

If you catch yourself peeking, log `firewall_breach` in the next entry's `notes` and stop the session.

**Sub-agents inherit the firewall.** Same rules.

---

## 2. Session files

Everything lives in `<workdir>/opd-cap.<slug>/`. A fresh agent reading these files alone must be able to resume.

```
opd-cap.<slug>/
├── capability.md              # Living session doc. Description, headroom, hypothesis history.
├── capability.jsonl           # Append-only experiment log; one line per attempt.
├── capability.config.json     # Workdir, base model, teacher config, max iterations, etc.
├── capability.oracle.sh       # Blind oracle wrapper. Prints `SCORE=<f> <SUB>=<f>...` on stdout.
├── kiln-polish.jsonl          # Separate ledger for kiln rough edges. One entry per iter (even "none").
├── prompts/<slug>.jsonl       # Prompt set for one attempt. {"messages":[...]} per line.
├── hypotheses/<slug>.md       # Why/what, falsification plan, AND VERDICT (gated).
├── adapters/<slug>.txt        # Adapter name returned by training.
└── responses/<slug>/*.txt     # Inspected raw responses (when debugging — never read from eval set).
```

The slug is your experimental identity — keep it stable across log entry, hypothesis file, prompt set, adapter name.

### capability.md template

```markdown
# Capability: <one-line title>

## Description
<2–4 sentences. The user's own words. What the model should do. What it
currently does wrong. What "good" looks like. Do NOT write the eval's
rubric here.>

## Base model
<model id>

## Teacher
<served-name, quantization, vLLM URL, max_logprobs>

## Rubric
<sub-score names, weights, what each measures — at the level the
user/agent already knows; do NOT include eval-set examples.>

## Baseline
| Sub-score | Weight | Baseline | Headroom (w×(1−b)) |
|-----------|--------|----------|---------------------|
| ...       | ...    | ...      | ...                 |
| **Total** |        |          | **<sum>**           |

## Target sub-score
<the one with the most headroom; what we're trying to lift.>

## Hypothesis log
<living table — one row per attempt with slug, family (H1–H8), verdict, composite Δ, target Δ.
Updated AFTER each iter's verdict is written.>

## Dead ends
<one line per retired family with the falsifying evidence.>

## Open questions
<things we couldn't answer this round; carry to next-session.md at close.>
```

### capability.jsonl schema

```json
{
  "iter": 4,
  "slug": "epoch9-r16-lr1e4",
  "ts": "2026-05-17T03:14:11Z",
  "status": "kept",
  "family": "H1",
  "target_sub_score": "faithfulness",
  "hypothesis": "More epochs continues the monotonic lift in faithfulness from iters 1–3.",
  "verdict": "confirmed (faithfulness +3.1pp; in-line with epoch curve).",
  "composite": 0.9362,
  "composite_delta": 0.0319,
  "headroom_used": 0.32,
  "sub_scores": {"parses": 1.0, "validates": 0.8511, "is_pure": 1.0, "is_substantive": 0.8085},
  "regressions": [],
  "training": {"rank": 16, "alpha": 32, "lr": 1e-4, "epochs": 9, "samples_per_prompt": 1,
               "max_tokens": 64, "top_k": 8, "effective_steps": 17, "nominal_steps": 243,
               "skip_rate": 0.93, "final_loss": null, "elapsed_s": 300},
  "asi": {
    "what_worked": "Epoch curve still has slope at 9; faithfulness now within 3pp of teacher.",
    "what_failed": null,
    "next_focus": "Try 12 epochs to confirm; if it plateaus, pivot to H2 (more prompts)."
  },
  "kiln_polish_noted": false,
  "git_sha": "ac39482f",
  "notes": ""
}
```

`status` ∈ `{kept, discarded, crash, firewall_breach, oracle_error, broken}`. `broken` is reserved for the all-zeros / structural-failure mode (§10).

---

## 3. The loop

### Phase 0 — Intake (one-shot, run once per session)

1. **Capture the user's verbal description** verbatim into `capability.md`. Don't paraphrase.
2. **Confirm a rubric exists.** Sub-score names, weights, what each measures. The rubric is the contract.
3. **Stand up the teacher** (§6). Health-check it.
4. **Run baseline.** `./capability.oracle.sh ""` — log it as iter 0, slug `baseline`. Record ALL sub-scores.
5. **Headroom analysis.** Run `headroom.py`. Pick the target sub-score (the one with the most movable weight).
6. If headroom < 0.05, **stop and report** — OPD won't be interesting here.
7. **Tiny-smoke** (§18) — 5-prompt training, no scoring. Confirms infra is up.
8. **Commit intake.**

### Phase 1 — Hypothesise (every iteration)

Before touching prompts or running training, copy the template:

```bash
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/<slug>.md
# Edit. Falsification plan is mandatory.
```

The hypothesis file has these required sections:
- **Family** — one of H1–H8 (§4).
- **Claim** — one sentence: "Bumping epochs from 6→9 will lift `faithfulness` by ≥3pp."
- **Mechanism** — *why* you expect this. One sentence.
- **Falsification plan** — "If `faithfulness` Δ < +1pp, the H1 curve has saturated; pivot to H2."
- **Verdict** — *left blank until after eval*. **Phase 5 fills this in.**

If you find yourself writing the hypothesis after seeing the score, stop. That's rationalisation, not science.

### Phase 2 — Construct (prompts, not assistant turns)

Build `prompts/<slug>.jsonl`. Each line: `{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"<dummy>"}]}`.

The assistant content is dummy — the student samples its own. But you still need the field for the kiln tokenizer to compute label_mask.

Rules:
- **Filter by token length** to fit your VRAM budget (typically ≤500–700 tokens of prompt; see §7).
- **Reflect the capability surface, not the eval set.** Vary domain, framing, complexity.
- **Smaller is fine.** 20–50 prompts is plenty for a first probe.
- **Reuse prompt sets across iters** unless your hypothesis is specifically about the prompt set (H2). Don't quietly change prompts between H1 iters.

### Phase 3 — Train

Use kiln's OPD trainer (`cuda_opd_remote` example or `kiln train opd` if wired). Always set `KILN_STREAMING_PREFILL=1`. Use §5's starting recipe unless your hypothesis is about a hyperparameter.

Capture:
- Adapter path → `adapters/<slug>.txt`
- Effective steps from log → counted by progress callbacks
- Nominal steps → epochs × prompts × samples_per_prompt
- Skip rate → 1 − effective/nominal

### Phase 4 — Blind eval — ALL sub-scores

Run the oracle. Record ALL sub-scores. Compute:
- Composite Δ vs baseline
- Each sub-score Δ vs baseline
- Headroom utilization: composite Δ / theoretical ceiling (from Phase 0)
- Regressions: any sub-score that dropped by >1pp

**If composite < 0.5 × baseline → STOP. Go to §10.** Don't iterate on a broken adapter.

### Phase 5 — Close the loop (the verdict gate)

Before logging:
1. **Write the verdict** in `hypotheses/<slug>.md`. ✓ confirmed / ✗ falsified / ? inconclusive + one sentence of justification, anchored to the actual numbers.
2. **Compute regressions.** Sub-scores that dropped. List them with severity (minor < 2pp, major ≥ 2pp).
3. **Inspect ≥3 raw responses** if it's a kept or if anything looks wrong. Sample outside the eval set.
4. **Note kiln polish.** `kiln_polish_noted: true/false`. If anything was awkward in kiln this iter, append a one-line entry to `kiln-polish.jsonl`.

Now call the log helper (`templates/log_iter.sh`). It will reject the entry if:
- `verdict` is empty
- `target_sub_score` is empty
- `asi.what_worked` AND `asi.what_failed` are both empty
- `asi.next_focus` is empty or boilerplate
- `regressions` is missing
- `kiln_polish_noted` is missing

**The agent literally cannot proceed without filling these fields.** This is the central anti-laziness gate.

After logging:
- Update `capability.md` hypothesis log table with the row for this iter.
- Commit (§13).

### Phase 6 — Iterate

Pick the next hypothesis. Order of preference:
1. **If H1 epoch curve still rising**, continue H1 (one more epoch step).
2. **If H1 has saturated** (3 consecutive iters with utilization rising <5pp), **must pivot families** — §5's gate enforces this.
3. **If a clear winner has emerged**, run one **triangulation** ablation from a different family before declaring causal.
4. **If a family produces 2 consecutive discards**, retire to dead-ends.

### Phase 7 — Cadence checkpoints (every 3 iters)

Every 3rd iteration, **post a progress report to the user** (and write it into `capability.md` as a "Checkpoint at iter N" subsection). Required content:
- Best composite so far + which adapter
- Headroom captured so far
- Families tried; families retired
- One-paragraph plan for next 2 iters

Not optional. The 4th iteration cannot start until the checkpoint is written.

---

## 4. Hypothesis families (OPD-specific)

These are the **only** families. Don't invent new ones until you've exhausted these.

**H1 — Epoch curve.** Same prompts, same hyperparameters, just bump epochs (3 → 6 → 9 → 12). Cheapest first probe; establishes monotonicity. **Always start at H1.**

**H2 — Prompt diversity / coverage.** More prompts spanning more of the capability surface. Useful when H1 has saturated — student has overfit on the current prompt set's states.

**H3 — Samples per prompt.** More student rollouts per prompt, averaging out variance. ⚠ **`samples_per_prompt` has an auto-bump landmine in kiln** — if you set the default value, kiln may auto-scale it to 64 for small datasets, causing a 64× cost blowup. Always set a non-default value (e.g. 2 or 8).

**H4 — Rank capacity.** Bigger LoRA (16 → 32 → 64). Useful only after H1 + H2 plateau; rank-limited capacity is rare in practice.

**H5 — Top-K width.** Wider teacher support (8 → 16 → 32). The teacher's top-K logprobs are the supervision; a wider window gives the student more "exits" the teacher accepts. The literature defaults to 32.

**H6 — Off-policy SFT cold-start.** Run a brief SFT pass on teacher-generated rollouts before OPD. Per Lu (2025), this reduces early student/teacher divergence and accelerates the curve. Useful when baseline gap is large and the student rarely overlaps teacher's distribution.

**H7 — Stable-OPD mixture.** Add `λ × SFT_gold` to the OPD loss to anchor against drift (Luo et al. 2026). Useful when an OPD adapter lifts the target sub-score but regresses others.

**H8 — Prompt pruning.** Remove prompts where the student already saturates (baseline composite on that prompt = 1.0). They contribute no useful gradient. Useful late in a session when efficient signal matters more than coverage.

**Sequencing.** First 4 iterations should usually be: H1, H1 (one more epoch step), H2 or H3 (broaden), triangulation back to H1 with the broader prompt set. After that, follow the data.

---

## 5. Anti-laziness gates (consolidated)

All gates are mechanical: a helper script enforces them, the agent cannot proceed without satisfying them. Listed here in one place for reference.

| Gate | Where enforced | What it blocks |
|------|----------------|----------------|
| **Verdict gate** | `log_iter.sh` rejects entry with empty `verdict` | next iteration |
| **Asi-substance gate** | `log_iter.sh` rejects empty or boilerplate `asi.*` | next iteration |
| **Regressions array gate** | `log_iter.sh` requires explicit `regressions: []` (possibly empty) | next iteration |
| **Headroom-utilization gate** | After 3 H1 iters with <5pp utilization rise, family pivot required | another H1 iter |
| **All-zeros gate** | Composite < 0.5 × baseline triggers `failure_mode.md` requirement | another training run |
| **Resume protocol gate** | `status.sh` must be run + acknowledged by the agent before iter N+1 | first action after resume |
| **Checkpoint cadence gate** | Every 3rd iter requires a progress checkpoint in capability.md | iter 4, 7, 10... |
| **Polish acknowledgement gate** | `kiln_polish_noted` must be explicit yes/no per iter | log entry |
| **Closeout checklist gate** | End-of-session emits structured summary | session end |

The pattern: the artifact you'd be tempted to skip is the next gate's input. Forgetting becomes mechanically impossible — the agent literally can't proceed.

---

## 6. Teacher hosting

OPD requires a live teacher. This skill assumes vLLM with `/v1/completions` and `prompt_logprobs=K`.

Trade-offs:
- **Bigger teacher = more capable, but more VRAM.** Quantize aggressively (Q4 AWQ or GPTQ).
- **Same-family teacher = cleanest gradients.** Cross-family OPD requires tokenizer alignment, a separate project.
- **vLLM is the default.** SGLang works with the same API.

The teacher serves at a URL (typically `http://localhost:8002`). kiln's `RemoteTeacher` queries it.

### Health check (mandatory before training)

1. Teacher responds to a sample prompt with a coherent answer (not gibberish). Spot-check by hand.
2. `prompt_logprobs=K` returns dense distributions at every prompt position.
3. The vLLM `--max-logprobs` flag is ≥ your training `--top-k`.

If any check fails, fix the teacher before training. OPD against a broken teacher produces a broken LoRA.

### Process hygiene

vLLM spawns child processes (`VLLM::EngineCore`) that don't die cleanly with the parent. After teardown, **always** verify GPU memory is free; orphan EngineCore processes hold VRAM and prevent the next vLLM start.

`teacher-down.sh` (when added) finds and kills these via `/proc/*/fd` inspection.

---

## 7. Memory budget

The hardest constraint. With one 48 GB card and a Q4-27B teacher + bf16-4B student:
- Teacher: ~22 GB resident
- Student + LoRA + optimizer: ~12 GB
- Training transients: ~10 GB
- Margin: ~4 GB

Use `KILN_STREAMING_PREFILL=1` always. Peak transient memory bounds with the GDN tile size, not prompt length. Without it, ~400-token prompts will OOM the student's prefill.

If you hit OOM:
1. Lower teacher's `--gpu-memory-utilization` in 0.05 steps.
2. Reduce training `max_tokens` (rollout length cap).
3. Reduce student rank (capacity tradeoff).
4. Filter prompts harder by token length.

Don't try to "page" by halting/resuming — vLLM doesn't support it cleanly. Pick a stable budget and stick to it for the session.

---

## 8. Skip rate watch (the most underrated metric)

OPD on small datasets has a high skip rate. The student samples EOS or otherwise fails to produce a usable rollout, and that iteration trains on nothing.

**Effective steps = (1 − skip_rate) × nominal_steps.**

In an under-tuned run, skip rate ≈ 0.87. So 81 nominal iterations → ~10 effective training steps. To get 30 effective steps you need ~225 nominal. **Bump epochs aggressively, not rank.**

Watch the trainer's progress callback. If it fires on <5% of nominal iterations, your student isn't sampling usefully. Possible causes:
- Rollout prompt structure mismatches the inference-time chat template
- Student sampling EOS immediately (no assistant cue in the prompt)
- max_tokens too low to produce any non-EOS

This is the OPD analog of SFT's answer-form drift — equally common, equally fixable, equally easy to miss.

---

## 9. Loss is deceptive

Stable OPD loss does not mean good training. Spiky loss does not mean bad training.

- **Spiky loss `0.0 / 22 / 0.0`**: student matches the teacher on some prompts, is far off on others. Healthy.
- **Smooth loss `0.3 / 0.5 / 0.4`**: could be averaging across many partial-matches (fine), or overfitting on a tiny subset that always succeeds (bad).

**Trust the blind eval. Never keep on loss alone.** Loss is a training-time artifact; the oracle is ground truth.

---

## 10. The all-zeros failure mode

If composite is 0.0 (or < 0.5 × baseline) after a non-trivial training run, **the adapter is broken at a structural level**. Common causes:
- Rollout prompt construction bug (e.g., synthetic prefixes encoded with the wrong tokenizer path)
- Active-position indexing bug (loss computed at wrong tokens)
- LoRA targeting wrong modules
- Teacher returning garbage logprobs

**Stop. Do not iterate. Write `failure_mode.md`:**
- Inspect ≥5 raw responses from the broken adapter
- Identify which sub-score(s) collapsed and how
- Hypothesize the structural cause
- Propose a fix to test in isolation

`log_iter.sh` enforces this — entries with `status: broken` cannot be followed by `status: kept` without `failure_mode.md` existing and being committed.

---

## 11. Sub-score regression watch

Composite-lift adapters can silently regress on saturated sub-scores. Every entry must enumerate regressions explicitly (even empty list).

A **Pyrrhic keep** = lift on the target sub-score with regression on another. Mark it kept (it moved composite) but flag the regression severity in `notes`. Next iteration should attempt the same hypothesis with lower rank, more diverse prompts, or H7 (Stable-OPD mixture) to anchor against drift.

---

## 12. Stop conditions

Stop and run the closeout checklist (§13) when **any** of:
- Headroom utilization > 80% (most of what's movable has been captured)
- 3 consecutive iters with no composite improvement
- Composite < 0.5 × baseline after a non-trivial run (broken — debug, don't iterate)
- 2 consecutive teacher errors (vLLM crash; investigate before retrying)
- Max iterations reached
- User interrupts

---

## 13. Closeout checklist (gated)

End-of-session emits a structured summary. Required output, not optional. `summary.sh` generates the skeleton; the agent fills in narrative:

- **Top 3 adapters** with full sub-score breakdown
- **All-iters table** (slug, family, composite, target Δ, regressions, headroom used)
- **Families dead or alive** (one line each)
- **Best mechanism** — one paragraph naming the route that worked
- **`next-session.md`** — first thing to try when resuming, with prior so far
- **`kiln-polish.jsonl` summary** — count of polish notes captured

`capability.md` gets a "## Closing summary" section appended with the above.

Commit the closeout as a single commit `cap[<slug>]: close (best <composite>, +<delta>)`.

---

## 14. Resuming a session

A fresh agent in a directory with an existing `opd-cap.<slug>/`:

1. **Re-read this SKILL.md.** Discipline is the whole game.
2. Run `templates/status.sh` (when added) — one-screen summary.
3. Read `capability.md` end to end.
4. Find any `hypotheses/<slug>.md` without a `Verdict:` line — that's an interrupted run; either complete the verdict (run eval, write verdict) or log it as `status: crash` with the available info.
5. Run the teacher health check (§6). If unreachable, start the teacher before continuing.
6. **Re-baseline.** The base model's score can drift across kiln versions/restarts. Append a new `slug=baseline` line; do not overwrite the old one. Later confidence math uses the most recent baseline.
7. Continue from where the log ends.

**No further questions to the user** unless the budget is exhausted, the oracle is misconfigured, or the rubric needs reinterpretation.

---

## 15. Sanity-check the student periodically

Every ~5 iterations (and at session start), sample a handful of rollouts from the current best adapter on a *non-eval* prompt and read them. Reasons:
- Catch silent style regressions the rubric doesn't measure
- Confirm the student is producing recognizable shape
- Notice if the student is sampling EOS / repeating itself / generating in the wrong language

This is the OPD analog of SFT's "look at your dataset". Skipping it is how you train for 6 hours and then discover the LoRA is producing gibberish.

---

## 16. Kiln polish ledger

OPD exercises every part of kiln's training stack. Every iteration has a `kiln_polish_noted: bool` field — required, not optional. When something feels rough (a confusing error, a missing CLI flag, a default that should be different, an undocumented env var), append a one-line entry to `kiln-polish.jsonl`:

```json
{"ts":"...","slug":"<current>","note":"`samples_per_prompt=4` auto-bumps to 64 silently; should warn or flag explicitly."}
```

Don't try to fix kiln mid-session. Capture for the next maintenance window.

The closeout checklist surfaces the count. A run with zero polish notes across 8 iterations is suspicious — either kiln has matured (great) or the agent stopped noticing (more likely; review and add what was missed).

---

## 17. Hypothesis-loop close: what "verdict" means

The verdict line in `hypotheses/<slug>.md` is the central anti-laziness artifact. It must be **anchored to the actual numbers** and decisive about the falsification plan.

Examples of acceptable verdicts:

✓ "Confirmed — faithfulness +3.1pp (predicted ≥3pp). Composite +1.4pp; headroom utilization rose 32%→47%."

✗ "Falsified — faithfulness Δ = +0.4pp, well below the ≥3pp threshold. The H1 epoch curve has saturated at epoch 9; pivoting to H2 next."

? "Inconclusive — composite moved +1.1pp but faithfulness was flat. Lift came from is_substantive, which we weren't targeting. Likely a confound; re-run with more iters of the same config to check."

Unacceptable: "looks good", "didn't work as well as expected", "see results above". These don't close the loop.

---

## 18. Tiny-smoke

Before the first real iteration, run a smoke: 5 prompts, 1 epoch, rank 4. Goals:
- Confirm vLLM teacher reachable and returning logprobs
- Confirm kiln OPD trainer builds gradients without OOM
- Confirm the adapter file gets written to disk
- Confirm the oracle script accepts the adapter name and returns sub-scores

A smoke run completes in ~3–5 minutes. If anything is wrong, you find out before paying for a real iteration.

---

## 19. One-screen quickstart

```bash
SKILL=.claude/skills/opd-capability-creator

# 0. Intake
SLUG=faithful-code-summarization
$SKILL/templates/scaffold.sh $SLUG
cd opd-cap.$SLUG
# edit capability.md (paste capability description + rubric weights)
# edit capability.config.json (teacher URL, max iters, etc.)
# edit capability.oracle.sh (wrap the user's eval; print SCORE=... + per-sub-score)

# Stand up teacher (vLLM, Q4, gpu-mem=0.45, max-logprobs ≥ top-K)
# Health check
curl -sf http://localhost:8002/health

# Baseline
./capability.oracle.sh ""
# logs iter 0; headroom analysis printed
python3 $SKILL/templates/headroom.py < capability.jsonl

# Tiny-smoke
bash $SKILL/templates/tiny_smoke.sh   # when added

# Iter 1 — always H1
ABL=h1-epoch6-r16
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/$ABL.md
# Fill hypothesis (claim, mechanism, falsification plan; verdict left blank)

# Build prompts/$ABL.jsonl

# Train + eval (when train_and_score.sh is added)
RESULT=$(bash $SKILL/templates/train_and_score.sh $ABL)

# Phase 5 — Verdict gate
# Edit hypotheses/$ABL.md to add Verdict line
# Inspect ≥3 responses
# Log
bash $SKILL/templates/log_iter.sh $ABL <fields>
# log_iter.sh rejects if verdict/asi/regressions/polish-flag empty

# Update capability.md hypothesis log table; commit
git add -A && git commit -m "cap[$ABL]: kept (+0.03)"

# Iter 2... follow §3 Phase 6 sequencing
# Every 3rd iter — checkpoint gate (post progress to user, write to capability.md)
# Stop when §12 condition fires; closeout per §13.
```

That is the entire skill. Everything else above is the discipline that makes the loop close cleanly.
