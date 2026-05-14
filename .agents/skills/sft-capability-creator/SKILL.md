---
name: sft-capability-creator
description: Autonomous SFT-dataset experiment loop for eliciting a verbally described capability through blind evaluation. Use when asked to "elicit capability X", "design SFT datasets for X", "run capability ablations on X", "iterate datasets for X", or "fish for transfer to X". Hypothesise → construct → train → blind-eval → reflect → repeat. Best datasets teach through words and elicit capabilities in other modalities (e.g. prose that lifts math accuracy).
---

# sft-capability-creator

> Autonomous loop for **eliciting** a target capability with SFT. Each iteration is one small, independent dataset whose hypothesis you can articulate in a sentence. The evaluator is **blind** — you never see what it scores. You only see one number.

Inspired by [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch). That tool optimises a metric whose definition the agent knows; this skill optimises a metric whose definition the agent **must not know**. The discipline is harder. The payoff is crystallised, transferable capability — datasets that teach a frame in words and elicit the frame across modalities.

---

## 0. Mental model

You are an experimentalist. The lab has:

- a fixed **base model** (kiln serves Qwen3.5-4B);
- a **target capability** described in plain English (e.g. *"two-digit arithmetic accuracy"*, *"refusing to fabricate citations"*, *"following ordered multi-step instructions"*);
- a **blind oracle** — a command that takes an adapter name and returns one scalar. You never read its source, suite, or outputs.

You produce **independent** SFT adapters. Each one is an attempt: a hypothesis instantiated as 8–256 chat examples, trained for 1–3 epochs at low rank, then scored. You append the score to a log and reflect. You never carry weights forward — every ablation starts from the base.

**Why independence matters.** Stacking adapters confounds the experiment. If A+B beats baseline you don't know whether A, B, or the interaction did the work. Independent ablations let causal claims survive past iteration 5.

**Why words can elicit non-word capabilities.** A model that has seen the *algorithm* for two-digit addition described in 30 different prose framings often executes that algorithm more reliably than one that has seen 300 numerical worked examples in a single style. SFT updates the same underlying parameters whether the supervision is symbolic or verbal; the verbal route can re-shape the routing of latent computation without overfitting the surface form the eval uses. **This is the most important asymmetry the skill exploits.** When in doubt, write English about the skill; do not drill the eval's surface form.

---

## 1. Information firewall (non-negotiable)

The eval is **blind**. You MUST treat the oracle as a black box that returns one floating-point number, optionally with a coarse `n` (number of items scored).

You MUST NOT:

- read `adapters/.eval/suites/**` or any registered suite JSON;
- read `adapters/.eval/judgments/**` or any per-example judgment file;
- pass `--json` to `kiln-eval` and parse `runs[].response` or per-example scores;
- ask the user *"what does the eval check?"* with the intent of memorising its surface;
- design a dataset by trying to **invert** the oracle (probe → infer → train-to-match);
- copy the eval's prompt template into your training data.

You MAY:

- ask the user once at intake for a **plain-English description** of the capability (1–3 sentences);
- read the score the oracle returns;
- read the n the oracle returns;
- ask the user for **categorical hints** if they volunteer them ("the eval is multi-turn", "the eval has tool calls") — but do not press for surface details.

If you catch yourself reading a suite file or looking at a per-example output, **stop, revert that step in your reasoning, and write `firewall_breach` in the next log entry's `notes`**. The user is trusting the skill to keep its hands clean; the experimental record loses meaning the moment you peek.

---

## 2. Session files

Everything lives in `<workdir>/sft-cap.<slug>/`. A fresh agent reading those files alone must be able to resume.

```
sft-cap.<slug>/
├── capability.md              # Living session document. Objective, hypotheses, what's been tried.
├── capability.jsonl           # Append-only experiment log, one line per ablation.
├── capability.ideas.md        # Backlog of hypotheses you didn't pursue this round.
├── capability.oracle.sh       # The blind oracle wrapper. Returns ONE number (+ optional n) on stdout.
├── capability.config.json     # workdir, base model, max iterations, adapter prefix.
├── datasets/<slug>.jsonl      # The SFT dataset for one ablation. One JSON object per line, {"messages":[...]}.
├── hypotheses/<slug>.md       # The why/what of one ablation, written BEFORE the dataset is generated.
└── adapters/<slug>.txt        # The adapter name returned by `kiln train sft`. Often just `cap-<slug>`.
```

The slug for an ablation is a short, descriptive kebab-case stub the agent picks before generation — `verbal-add-algorithm`, `numeric-drill-uniform`, `multi-frame-paraphrase`. The slug *is the experimental identity* — keep it stable across log entries, hypothesis file, dataset file, and adapter name.

### `capability.md` template

```markdown
# Capability: <one-line title>

## Description
<2–4 sentences. Plain English. What the model should be able to do.
What it currently does wrong. What "good" looks like. Do NOT
write the eval's prompts or rubric — write the user's words from intake.>

## Base model
<model id, e.g. Qwen/Qwen3.5-4B>

## Oracle
Command: `./capability.oracle.sh <adapter_name>`
Output contract: stdout `SCORE=<float>` on the last line, optionally
also `N=<int>`. Anything else on stdout is logged but ignored.
Higher is better unless `direction=lower` is set in capability.config.json.

## Budget
- Max iterations: <int>
- Per-ablation dataset cap: <int examples>
- Per-ablation training cap: <int epochs, float lr, int rank>

## Hypothesis taxonomy
<Living list. Each entry: short slug, one-sentence claim, status
(pending/running/kept/discarded), confidence note. Add as you learn.>

## What's been tried
<Update after each iteration. Two lines per ablation:
"#7 verbal-add-algorithm | 64 ex, 1ep, r=4 | score 0.41 (+0.06) | algorithm-in-prose moves the needle, paraphrase variety mattered."
Do NOT speculate about the eval's contents here — only the data
intervention and its effect size.>

## Dead ends
<Hypotheses we falsified. One line each with the falsifying evidence.>

## Open questions
<Things you couldn't answer this round.>
```

### `capability.jsonl` schema

One JSON object per line. Append-only. Never edit. Fields:

```json
{
  "iter": 7,
  "slug": "verbal-add-algorithm",
  "ts": "2026-05-14T14:23:11Z",
  "status": "kept",
  "score": 0.41,
  "delta": 0.06,
  "n": 25,
  "hypothesis": "Describing the column-by-column addition algorithm in English, with no numbers in the assistant turn, transfers to numeric accuracy.",
  "dataset": {
    "path": "datasets/verbal-add-algorithm.jsonl",
    "size": 64,
    "construction": "30 paraphrases of the algorithm + 34 worked examples whose answer is the algorithm description, not the sum",
    "modality": "prose",
    "source": "agent-generated"
  },
  "training": {
    "adapter": "cap-verbal-add-algorithm",
    "lr": 1e-4,
    "epochs": 1,
    "lora_rank": 4,
    "final_loss": 1.31,
    "elapsed_s": 38
  },
  "asi": {
    "what_worked": "diversity of framings",
    "what_failed": null,
    "next_focus": "add 'thinking-aloud' assistant turns that name each carry digit"
  },
  "notes": ""
}
```

`status` ∈ {`kept`, `discarded`, `crash`, `firewall_breach`, `oracle_error`}.
`delta` is `score − best_score_before_this_run` (signed, in primary direction). If this is iter 1, `delta = score − baseline_score`. `baseline_score` is iter 0 — see §3.

---

## 3. The loop

### Phase 0 — Intake (one-shot)

1. Capture the user's verbal description of the capability. **Do not paraphrase to "improve" it** — copy their words.
2. Ask only what you cannot infer:
   - Where is the blind oracle? (a command, suite name, or "I'll paste the score back")
   - What's the budget (max iterations, max examples per dataset, max epochs)?
   - Any hard constraints? (style, language, refusal behaviour the dataset must respect)
3. Write `capability.md`, `capability.config.json`, `capability.oracle.sh`. Commit.
4. Run the **baseline**: `./capability.oracle.sh ""` (empty adapter = base model). Record as iter 0 with `slug="baseline"`.

The oracle wrapper is *your* file, not the eval's. Its job is to (a) call the user's eval, (b) parse exactly one number out, (c) print `SCORE=<float>` on stdout, (d) tell you NOTHING ELSE. A reference wrapper for kiln's blind-eval mode is at `templates/oracle.sh` (see §6).

### Phase 1 — Hypothesise (every iteration)

Before touching data, write `hypotheses/<slug>.md`:

```markdown
# Hypothesis: <slug>

## Claim
<One sentence. "X intervention will raise the capability score because Y."
The Y matters — without a mechanism, your update after the result is just vibes.>

## Mechanism
<2–4 sentences. What latent computation are you trying to re-route? What
shortcut are you trying to break? Why would prose teach a non-prose skill?>

## Dataset shape
- Size: <N>
- Modality of supervision: <prose / symbolic / mixed / hybrid>
- Distribution: <how examples vary; what's held constant>
- Surface form held OUT: <anything you suspect the eval uses, kept *out* of training>

## Risk
<What would make this hypothesis wrong? What would the result look like
if it's only memorisation?>

## Falsification plan
<If this raises the score by Δ, the next iteration should X. If it doesn't,
the next iteration should Y. Decide BEFORE seeing the score.>
```

Falsification-plan-before-result is the single most important rule. It is the difference between iterating and rationalising.

### Phase 2 — Construct (every iteration)

Build the SFT JSONL at `datasets/<slug>.jsonl`. One JSON object per line. Schema:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Rules:

1. **No system prompt unless the user specified one in capability.md.** Spurious system prompts leak into the eval and inflate variance.
2. **Generate, don't curate.** It is fine — often best — to write the dataset by hand or with prompts. Each line should be intentional. Cite your generation strategy in the hypothesis file.
3. **Cap size aggressively.** Small (16–128) ablations train in seconds and yield cleaner causal signal. Only grow past 256 when you've ruled out small-N versions of the hypothesis.
4. **Hold out the eval's surface form.** If the user hinted the eval uses two-digit problems, your dataset for `verbal-add-algorithm` should contain *three*-digit worked examples or zero numbers at all. Transfer is the point.
5. **Prefer prose over symbols when both fit.** The skill's edge. Worked example: instead of `"5 + 8 = 13"`, write `"To add 5 and 8, notice that 5 needs 5 more to reach 10, take 5 from 8 leaving 3, so the sum is 10+3=13."` — and even better, write that *as the entire assistant turn*, leaving the user turn as `"How would you add 5 and 8?"`.
6. **Vary framing widely, vary content narrowly.** 30 paraphrases of the same algorithm beats 30 different algorithms applied once.

Validate the file before training:

```bash
jq -c '.messages | length' datasets/<slug>.jsonl | sort -u
jq -c '[.messages[].role] | tostring' datasets/<slug>.jsonl | sort | uniq -c
```

Both should look sane (consistent message counts, sensible role patterns).

### Phase 3 — Train (every iteration)

The base model is served by kiln (`kiln serve`). Training is one command:

```bash
kiln train sft \
  --file datasets/<slug>.jsonl \
  --adapter cap-<slug> \
  --lr 1e-4 \
  --epochs 1 \
  --lora-rank 4
```

Defaults are deliberate: low rank (4), single epoch, mild LR. These keep the ablation cheap and cleanly isolated. **Resist raising rank/epoch as an early intervention** — if the score didn't move, almost always the *data* is wrong, not the optimiser. Bump rank/epoch only as a deliberate, logged hypothesis (e.g. `same-data-rank-16`).

If a previous adapter by this slug exists, you have two choices:

- **Replay the slug** (overwrite): only when iterating on the *same* hypothesis with a small data tweak. Re-use the slug, append a new log line, leave `delta` honest.
- **Fork the slug**: rename to `<slug>-v2` and treat it as a new ablation. Default to this.

Capture the trained adapter name + final loss into `adapters/<slug>.txt` and the JSONL entry. Final loss is *advisory only* — it correlates weakly with capability uplift and you should never `keep` on loss alone.

### Phase 4 — Blind eval (every iteration)

```bash
./capability.oracle.sh cap-<slug>
```

Output **must** be one `SCORE=<float>` line on stdout. The wrapper enforces this; do not bypass it. If the wrapper prints anything that looks like a per-example transcript, treat the wrapper as compromised, fix it, and rerun.

The score is the only signal you have. Compare against the best-kept score so far. If it's a tie or worse, you `discard`. If it's better by more than the noise floor (see §5), you `keep`.

### Phase 5 — Reflect (every iteration)

Append a line to `capability.jsonl`. Update `capability.md`. Apply your pre-committed falsification plan from `hypotheses/<slug>.md`.

The `asi` block in the log entry is where future-you (or a resuming agent) recovers your thinking. Write what you learned, not what you did:

- ✗ *"Generated 64 examples and trained for 1 epoch."* — recoverable from the entry.
- ✓ *"Prose-only assistant turns moved the score; switching half of them to numeric worked examples in the next try reversed the gain. Verbal supervision is doing real work here."* — only survives if you write it down.

**Annotate failures heavily.** Discarded ablations leave only the JSONL line; their dataset and hypothesis file stay on disk for archaeology. Make sure those notes carry enough mechanism for a fresh agent to skip the dead end.

### Phase 6 — Iterate

Pick the next hypothesis. Order of preference:

1. **Refine the best family.** If `verbal-add-algorithm` worked, the next 1–2 iterations should be variants of it (longer paraphrases, more vs. fewer worked examples, with/without thinking turns).
2. **Triangulate.** Run an explicitly *different* hypothesis next, even if you have a winner, to confirm the win isn't coincidence. If a clearly-different ablation also helps, the capability has multiple handles; if a clearly-different ablation does not help, your winner is more specific and probably real.
3. **Cool down winners; warm up new families.** When a family stops yielding gains across 2 consecutive ablations, retire it to `capability.md`'s "dead ends" and pull a hypothesis from `capability.ideas.md`.

**Never thrash.** If you've discarded the same direction twice, write it off in the dead-ends section and try something structurally different.

---

## 4. Hypothesis taxonomy (start here when stuck)

Use these as seed families. Each is a sentence the agent can instantiate into a dataset.

**T — Teach by words, test by deed (the asymmetry the skill is named for)**
- *"Explain the algorithm in prose with no numeric worked example. The model learns the routing; the eval surfaces the routing as numeric accuracy."*
- *"Describe the failure mode of the wrong answer, not the right answer. The model learns what to avoid."*
- *"Use a different surface form than the eval (different language, different units, longer numbers, fewer numbers) so transfer is forced."*

**F — Framing diversity**
- *"30 paraphrases of one rule beat 1 statement of 30 rules."* Vary *how* you say it, hold *what* fixed.
- *"Same content, three voices (formal, conversational, terse). Voice diversity broadens the basin."*

**S — Structural priors**
- *"Always end the assistant turn with a single-line summary of the answer."* Encourages a stable output mould the eval can detect.
- *"Always think first in a `<think>...</think>` block, then answer."* If the eval cares about answer-only, the thinking turn doesn't leak; if it cares about reasoning visible in the response, it does.

**R — Refusals and constraints**
- *"Teach 'I don't know' as a positive output for clearly under-determined questions."*
- *"Teach the constraint explicitly: 'When asked X, do not do Y.' Then the inverse: 'When asked something that looks like X but is Y-shaped, do Y.'"*

**N — Negation and contrast**
- *"For every positive example, include one near-miss that fails. Contrast tightens the boundary."*

**D — Decomposition**
- *"Break a hard task into 3 named subtasks. Teach each subtask separately. The eval recomposes them."*

**A — Abstraction ladder**
- *"Start one ablation at maximum abstraction (a principle stated in 3 sentences), the next at minimum (a worked example). Bracket the right altitude."*

**M — Modality crossover (the most powerful when it works)**
- *"Teach math by talking. Teach prose by listing constraints. Teach tool use by describing why the tool is the right shape for the task."*

Pull from this list when you have no hypothesis. Always specialise to the user's verbal description before generating data.

---

## 5. Confidence and noise

Most blind evals are small (n=10–50) and noisy. Treat the first `keep` with suspicion.

- After 3 logged ablations, compute the **noise floor** as the median absolute deviation (MAD) of `score` across all logged ablations (including discards). The agent can do this with `jq` over the log; `templates/confidence.sh` ships a reference.
- Confidence = `|best_delta| / MAD`.
- **≥ 2.0×**: improvement is likely real. Mark `kept` confidently.
- **1.0–2.0×**: above noise but marginal. Re-run the same ablation with a different seed (vary the dataset's example order, or change a single hyperparameter) before declaring it `kept`.
- **< 1.0×**: within noise. Default to `discard`.

When confidence is borderline, **don't move on**. Spending one extra iteration to confirm a real win is cheaper than chasing a phantom for ten.

---

## 6. The oracle wrapper

The blind eval is **whatever the user provides**, normalised through `capability.oracle.sh`. The wrapper's job is *containment*: it must reduce arbitrarily noisy output to one line of `SCORE=<float>`.

`templates/oracle.sh` ships a reference for the common case of a kiln registered eval suite:

```bash
#!/bin/bash
set -euo pipefail
# capability.oracle.sh — blind eval wrapper. Prints SCORE=<float> N=<int>.
# Configured via capability.config.json: { "eval_suite": "<name>", "server": "..." }
SUITE="$(jq -r '.eval_suite' capability.config.json)"
SERVER="$(jq -r '.server // "http://localhost:8420"' capability.config.json)"
ADAPTER="${1:-}"

JOB_JSON=$(kiln-eval --server "$SERVER" run \
  --suite "$SUITE" --adapter "$ADAPTER" --watch --json)
SCORE=$(echo "$JOB_JSON" | jq -r '.summary.accuracy // .summary.mean_score // empty')
N=$(echo "$JOB_JSON" | jq -r '.summary.num_examples // empty')

if [ -z "$SCORE" ]; then
  echo "ORACLE_ERROR: could not parse score from kiln-eval output" >&2
  exit 2
fi
echo "SCORE=$SCORE${N:+ N=$N}"
```

The wrapper **strips** `runs[].response`, `runs[].judgment`, per-example anything. The agent never reads the JSON; only `SCORE=` is consumed.

**If the user wants a fully human-in-the-loop oracle**, `capability.oracle.sh` should `read` from stdin or paste-back. The skill should ask the user once at intake which mode and write the appropriate wrapper.

---

## 7. Logging discipline

Three rules. They are not optional.

1. **Hypothesis file is written *before* the dataset.** If you find yourself writing the hypothesis after generating data, you are rationalising. Stop, delete the rationalisation, write the hypothesis from your honest intuition, then regenerate (or admit "I wrote this after the data").
2. **Falsification plan is written *before* the score is read.** Same principle, stronger version. The plan is in `hypotheses/<slug>.md`, finalised before you call the oracle.
3. **`asi` describes what was *learned*, not what was *done*.** The action is recoverable from other fields. The lesson is not.

Read `capability.md` and the tail of `capability.jsonl` at the start of every iteration. If you've forgotten what dead ends you've explored, you will re-explore them.

---

## 8. Resuming

A fresh agent invoked in a directory with an existing `sft-cap.<slug>/`:

1. Read `capability.md` end to end.
2. Read the last 10 lines of `capability.jsonl`.
3. Read `capability.ideas.md` and `capability.config.json`.
4. Scan `hypotheses/*.md` for any iteration whose JSONL entry is missing — that is an interrupted run; recover or mark crashed.
5. `kiln adapters list | grep cap-` to confirm trained adapters match the log.
6. Continue from where the log ends.

No further questions to the user unless the budget is exhausted or the oracle is misconfigured.

---

## 9. Stop conditions

Stop and report when **any** of:

- Max iterations reached (from `capability.config.json`).
- The same family has produced 3 consecutive `discard`s and the ideas backlog is empty.
- An oracle error has occurred twice in a row.
- The user interrupts.

When stopping, write a final summary to the bottom of `capability.md`:

- Best adapter (slug, score, delta from baseline).
- Top 3 hypotheses that worked, with one-line mechanism for each.
- Top 3 dead ends, with falsifying evidence.
- One paragraph of advice for whoever runs the next round.

Crystallised intelligence is not "the best dataset I found" — it is **the map of which families are alive and dead in this capability**.

---

## 10. Commit boundaries

Commit after:

- Intake (capability.md, capability.config.json, capability.oracle.sh, baseline jsonl entry).
- Each kept ablation (dataset + hypothesis + jsonl line, in one commit).
- Each retired family (dead-end section update).
- Final summary.

Use short, structural messages — `cap[<slug>]: kept (+0.06)`, `cap[<slug>]: discard (within noise)`, `cap: dead-end family <name>`. The log + commit graph together are the experimental record.

Do **not** include trained adapter binaries in commits — kiln writes them under its adapter store and references them by name. `adapters/<slug>.txt` storing the adapter name is enough.

---

## 11. Anti-patterns (read this list every session)

- **Peeking.** Reading suite files, judgment outputs, per-example transcripts. Hard veto. See §1.
- **Eval-shaped training data.** Copying the eval's prompt template into your dataset. Even if you've never *read* the eval, if your data looks like the eval, transfer collapses into memorisation.
- **Compound interventions.** Changing dataset *and* rank *and* lr in one ablation. You'll never learn which knob did the work.
- **Rationalising results.** Discovering a mechanism *after* a positive score. Symptom: your hypothesis file is shorter than your `asi` block.
- **Loss-chasing.** Picking the adapter with the lower final loss when its blind score is worse. Loss is a training-time signal; the score is ground truth.
- **Big-data reflex.** Reaching for 1000-example datasets in iteration 2. Small ablations win; only scale a clearly-working hypothesis.
- **Stylistic clobber.** Training data with a strong style overwrites the base model's defaults even for unrelated queries. Watch for it; the eval may not test style but humans will notice the model went weird.
- **Eval contamination through the system prompt.** Adding `"You are an expert at <capability>"` to every system prompt. Usually a bug, not a feature; remove the system prompt unless the user asked for it.

---

## 12. One-screen quickstart (the actual loop)

```bash
# 0. Intake — agent fills this in interactively with the user
SLUG=capability-name        # short kebab; this names the whole session
mkdir -p sft-cap.$SLUG/{datasets,hypotheses,adapters}
cd sft-cap.$SLUG
# write capability.md, capability.config.json, capability.oracle.sh
chmod +x capability.oracle.sh
git add -A && git commit -m "cap[$SLUG]: intake"

# baseline
./capability.oracle.sh ""              # -> SCORE=0.35 N=25
# append iter 0 line to capability.jsonl

# loop — one iteration:
ABL=verbal-add-algorithm
# write hypotheses/$ABL.md (claim, mechanism, dataset shape, risk,
# falsification plan) BEFORE touching data
# write datasets/$ABL.jsonl (one {"messages":[...]} per line)
jq -c '.messages | length' datasets/$ABL.jsonl | sort -u  # sanity
kiln train sft --file datasets/$ABL.jsonl --adapter cap-$ABL \
  --lr 1e-4 --epochs 1 --lora-rank 4
echo cap-$ABL > adapters/$ABL.txt
./capability.oracle.sh cap-$ABL        # -> SCORE=0.41 N=25
# append iter line to capability.jsonl with score, delta, asi, status
# update capability.md (what's been tried, hypothesis taxonomy)
git add -A && git commit -m "cap[$ABL]: kept (+0.06)"

# repeat until a stop condition fires (§9)
```

That is the entire skill. Everything else above is discipline.
