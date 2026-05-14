---
name: sft-capability-creator
description: Autonomous SFT-dataset experiment loop for eliciting a verbally described capability through blind evaluation. Use when asked to "elicit capability X", "design SFT datasets for X", "run capability ablations on X", "iterate datasets for X", or "fish for transfer to X". Hypothesise → construct → train → blind-eval → reflect → repeat. Best datasets teach through words and elicit capabilities in other modalities (e.g. prose that lifts math accuracy).
---

# sft-capability-creator

> Autonomous loop for **eliciting** a target capability with SFT. Each iteration is one small, independent dataset whose hypothesis you can articulate in a sentence. The evaluator is **blind** — you never see what it scores. You only see one number.

Inspired by [pi-autoresearch](https://github.com/davebcn87/pi-autoresearch). That tool optimises a metric whose definition the agent knows; this skill optimises a metric whose definition the agent **must not know**. The discipline is harder. The payoff is crystallised, transferable capability — datasets that teach a frame in words and elicit the frame across modalities.

### Skill inventory

The skill ships with one document and seven helper scripts. Reference them by absolute path (`.agents/skills/sft-capability-creator/...`) or via a `$SKILL` shell var as in §18.

| File | Purpose |
|------|---------|
| `SKILL.md` | This document. Authoritative procedure. |
| `templates/scaffold.sh` | Create `sft-cap.<slug>/` with `capability.md`, `capability.config.json`, etc. (§2, §3 Phase 0) |
| `templates/oracle.sh` | Blind kiln-eval wrapper. Emits only `SCORE=<f> N=<i>`. (§1, §6) |
| `templates/oracle-paste.sh` | Human-in-the-loop oracle variant. (§6) |
| `templates/hypothesis.md.tmpl` | Pre-experiment hypothesis form. (§3 Phase 1) |
| `templates/train_and_score.sh` | Async SFT train → poll → blind-score → one summary line. (§3 Phases 3–4) |
| `templates/log_iter.sh` | Append one structured `capability.jsonl` line. (§3 Phase 5) |
| `templates/annotate.sh` | Add `asi.*` fields + `notes` to the most-recent log entry. (§3 Phase 5, §7) |
| `templates/confidence.sh` | MAD-based confidence stats from the log. (§5) |
| `templates/status.sh` | One-screen session summary for resume / between iterations. (§8) |
| `install.sh` | Symlink the skill into `.claude/skills/`. |

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

- read any file under `adapters/.eval/suites/**` (the registered suite JSONs);
- read any file under `adapters/.eval/judgments/**` (per-example judgment outputs);
- read any file under `adapters/.eval/datasets/**` *for the dataset that backs the active oracle suite* — even though kiln puts dataset JSONLs there, those files are the source corpus the eval was synthesised from and are off-limits;
- call `kiln-eval list` with intent to surface tags that reveal eval contents (tag *counts* leak rough composition);
- call `kiln-eval --json run ...` and read fields other than `summary.accuracy`, `summary.mean_score`, `summary.num_examples`;
- call `GET /v1/eval/jobs/{id}` and read `runs[]`, per-example responses, judgments, or any field outside `summary.*`;
- call `GET /v1/eval/suites/{name}` for the active suite;
- call `GET /v1/eval/datasets/{name}` or `GET /v1/eval/datasets/{name}/rows` for the dataset backing the active suite;
- `cat` the oracle wrapper's intermediate JSON tempfile after a run;
- copy the eval's prompt template into your training data;
- design a dataset by trying to **invert** the oracle (probe → infer → train-to-match);
- ask the user *"what does the eval check?"* with the intent of memorising its surface.

You MAY:

- ask the user once at intake for a **plain-English description** of the capability (1–3 sentences);
- read the score the oracle returns;
- read `n` the oracle returns;
- ask the user for **categorical hints** if they volunteer them ("the eval is multi-turn", "the eval scores short answers", "the eval has a tool-call check") — but do not press for surface details and **never** request the suite name, prompt text, or any example.

If you catch yourself reading a suite file or looking at a per-example output, **stop, revert that step in your reasoning, and write `firewall_breach` in the next log entry's `notes`** with one sentence describing what leaked. The user is trusting the skill to keep its hands clean; the experimental record loses meaning the moment you peek. If a breach is severe (you saw the eval's prompts or rubric verbatim), **the session is dead** — start a new one with a slug-suffix `-postbreach` and acknowledge the contamination in the new `capability.md`.

### Operational rule for the agent

Before any `Read` / `Bash cat` / `Bash grep` / `Bash jq` call against a path under `adapters/.eval/`, pause and write a one-line note to yourself ("am I about to peek?"). Almost every legitimate operation you need only touches `sft-cap.<slug>/`, `Qwen3.5-4B/adapters/cap-*/lineage.json` (your own training records), and the oracle wrapper's stdout. If you don't need it for the loop, don't read it.

### Sub-agents inherit the firewall

If you spawn a sub-agent (general-purpose, Explore, or any other) you MUST give it the same firewall in its prompt: *"Do not read any file under `adapters/.eval/`; the eval is blind."* A sub-agent that searches the codebase and helpfully surfaces eval contents in its summary breaks the experiment just as completely as if you read the files yourself. Prefer Bash + jq + grep for routine tasks over spawning a sub-agent on this skill's working directory — sub-agents are a poor fit for blind-eval discipline.

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

1. Capture the user's verbal description of the capability. **Do not paraphrase to "improve" it** — copy their words. If they describe a failure mode (*"model writes the wrong column in two-digit subtraction"*), keep the failure language too — it constrains hypothesis space usefully.
2. Ask only what you cannot infer (use `AskUserQuestion`, one batch, multi-question):
   - **Oracle**: registered suite name (preferred), an arbitrary shell command, or paste-back?
   - **Direction**: is higher score better or lower score better?
   - **Scorer field**: `accuracy` (exact-match style) or `mean_score` (graded)?
   - **Budget**: max iterations, max examples per dataset, max epochs.
   - **Anchor suite** (optional): a second blind eval for regression watch (§12).
   - **Hard constraints**: style, language, refusal behaviour, system-prompt requirements.
3. Run `templates/scaffold.sh <slug>`. Edit the produced `capability.md` and `capability.config.json` with the user's answers. The oracle is already `templates/oracle.sh` — if the user picked paste-back, replace it with `templates/oracle-paste.sh`.
4. Commit the intake.
5. Run **tiny-smoke** (§17) — 4-example training, no scoring. Confirms `kiln serve` is up and the helpers work. Delete the smoke adapter after.
6. Run the **baseline**: `./capability.oracle.sh ""` (empty adapter = base model). Log it as iter 0 with `slug="baseline"`.

The oracle wrapper is *your* file, not the eval's. Its job is to (a) call the user's eval, (b) parse exactly one number out, (c) print `SCORE=<float>` on stdout, (d) tell you NOTHING ELSE. See §6 for the contract.

### Phase 1 — Hypothesise (every iteration)

Before touching data, copy the template and fill it in:

```bash
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/<slug>.md
# then edit each section. Don't skip the falsification plan.
```

The template lives at `templates/hypothesis.md.tmpl`. **Falsification-plan-before-result is the single most important rule** — it is the difference between iterating and rationalising.

A hypothesis file thinner than ~12 lines of body is almost always under-specified. If you can't think of a mechanism, a risk, and a falsification plan, you don't have a hypothesis worth running — pull from `capability.ideas.md` or the §4 taxonomy instead.

### Phase 2 — Construct (every iteration)

Build the SFT JSONL at `datasets/<slug>.jsonl`. One JSON object per line. Schema:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

Rules:

1. **No system prompt unless the user specified one in capability.md.** Spurious system prompts leak into the eval and inflate variance.
2. **Generate, don't curate.** It is fine — often best — to write the dataset by hand or with prompts. Each line should be intentional. Cite your generation strategy in the hypothesis file.
3. **Write the JSONL directly with the Write tool.** Don't spawn a subagent to generate the dataset — its summary will be lossy, and (worse) it lacks the firewall context. If you need to draft 64 examples, draft 64 examples; that's what you're here for.
4. **Cap size aggressively.** Small (16–128) ablations train in seconds and yield cleaner causal signal. Only grow past 256 when you've ruled out small-N versions of the hypothesis.
5. **Hold out the eval's surface form.** If the user hinted the eval uses two-digit problems, your dataset for `verbal-add-algorithm` should contain *three*-digit worked examples or zero numbers at all. Transfer is the point.
6. **Prefer prose over symbols when both fit.** The skill's edge. Worked example: instead of `"5 + 8 = 13"`, write `"To add 5 and 8, notice that 5 needs 5 more to reach 10, take 5 from 8 leaving 3, so the sum is 10+3=13."` — and even better, write that *as the entire assistant turn*, leaving the user turn as `"How would you add 5 and 8?"`.
7. **Vary framing widely, vary content narrowly.** 30 paraphrases of the same algorithm beats 30 different algorithms applied once.
8. **Assistant-turn length: medium.** Aim for 50–300 tokens per assistant turn. Very short (<20 tokens) doesn't teach; very long (>500 tokens) wastes training compute and steers the model toward verbosity. Anchor examples (§11) are the explicit exception — they're meant to be short.
9. **Combining previously-kept datasets is a new ablation, not a stack.** If A and B both worked separately and you want to try A+B, concatenate the files into a new `datasets/a-plus-b.jsonl`, write a *fresh* hypothesis explaining what you expect from the combination, and treat the result as its own data point. Loading two adapters is not allowed; combining datasets is.

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

- **Replay** (overwrite): only when **all** of (a) the hypothesis is unchanged, (b) the data delta is a typo fix / order shuffle / single-example swap, (c) the previous log line was a `crash` or `oracle_error`. Re-use the slug, append a new log line, leave `delta` honest.
- **Fork** (new slug): when the hypothesis has changed at all, the data is intentionally different, or you're testing a hyperparameter. Suffix with the change: `-anchored`, `-rank16`, `-3ep`, `-shuffle1`, `-bigger`. **This is the default**, because forking is the only way an ablation's identity survives the experimental record.

Same-slug overwrite obliterates the previous adapter weights but **not** the previous log entry. The dataset file at `datasets/<slug>.jsonl` is the one on disk now; the dataset that produced the *previous* entry is gone. If you can't bear losing it, fork.

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

**Use the ideas backlog actively.** When you write a hypothesis file, you almost always think of 1–3 *other* directions you couldn't pursue this iteration. Append each as a bullet to `capability.ideas.md` **immediately** — `- <slug>: <one-sentence claim>`. The list is your safety net for steps 2 and 3 above. On resume, the agent skims the backlog before generating a fresh hypothesis. Prune entries when tried or proven irrelevant.

**Never thrash.** If you've discarded the same direction twice, write it off in the dead-ends section and try something structurally different.

### Kept datasets are immutable

A dataset whose log entry has `status="kept"` is part of the experimental record. **Do not edit `datasets/<slug>.jsonl` after the entry is written.** If you want to try a variant, fork the slug (§3 Phase 3). Mutating a kept dataset breaks every future result that compares against it and silently corrupts the noise floor. The same goes for the hypothesis file. If you spot a typo, fix only `notes` via `annotate.sh`, never the original files.

---

## 4. Hypothesis taxonomy (start here when stuck)

Use these as seed families. **Iteration 1 should always be a T-family ablation** — the asymmetry between verbal supervision and non-verbal evaluation is the *whole point* of the skill, and you cannot make later claims about transfer without first establishing whether prose alone moves the score.

**T — Teach by words, test by deed (the asymmetry the skill is named for) — DEFAULT FIRST ABLATION**
- *"Explain the algorithm in prose with no numeric worked example in the assistant turn. The model learns the routing; the eval surfaces the routing as numeric accuracy."*
- *"Describe the failure mode of the wrong answer, not the right answer. The model learns what to avoid."*
- *"Use a different surface form than the eval is likely to use (different language, different units, longer numbers, named entities instead of variables) so transfer is forced."*
- *"Have the assistant turn answer a meta-question about the skill (\"how would you approach this kind of problem?\") rather than an instance of the skill (\"what is 27+45?\"). Meta-supervision often transfers."*

**F — Framing diversity (run early, after one T)**
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

**M — Modality crossover (most powerful when it works; pair with T)**
- *"Teach math by talking. Teach prose by listing constraints. Teach tool use by describing why the tool is the right shape for the task."*

**Late-stage families** (only after 5+ iterations and a clear winner):

**H — Hyperparameter probes** (`<slug>-rank16`, `<slug>-3ep`) — keep dataset fixed, change one knob. Confirms the win wasn't an undertrained or overtrained accident.

**B — Bigger-data probe** (`<slug>-256`) — once a small ablation wins, scale the dataset by 4× holding the recipe fixed. Diminishing returns above 256 examples are common at low rank; surprises here are publishable.

Pull from this list when you have no hypothesis. Always specialise to the user's verbal description before generating data.

### Generation prompt template (use to produce prose examples)

When the agent itself is drafting the dataset (the common case), use this as a working scaffold. Adapt the bracketed parts to the capability:

```
For each of N=64 examples, produce one JSON line of the form
  {"messages":[{"role":"user","content":"<USER>"},{"role":"assistant","content":"<ASSISTANT>"}]}
where:
- <USER> is a natural question that touches the capability but does
  NOT match the eval's surface form (vary length, framing, vocabulary,
  named entities, units). Do NOT include the verbatim eval prompt.
- <ASSISTANT> teaches the algorithm/rule/frame in clear English. Avoid
  the surface form the eval likely tests (e.g. if the eval asks
  "What is 27+45?", the assistant should describe HOW to add two-digit
  numbers with carrying, not produce numbers). Use varied vocabulary
  across examples — do not echo a single canned phrasing.

Quality bar: read the dataset back to yourself. If you could not learn
the skill *from these examples alone*, the dataset is too thin; rewrite.
If you could win the eval just by template-matching three examples,
the dataset is too narrow; broaden.
```

This is the dataset's lift — a model that has read 64 sincere English explanations of an algorithm has been taught the algorithm. The eval then asks it to *apply* the algorithm. Apply ≠ recite. That gap is what we are paid to close.

---

## 5. Confidence and noise

Most blind evals are small (n=10–50) and noisy. Treat the first `keep` with suspicion.

- After 3 logged ablations, compute the **noise floor** as the median absolute deviation (MAD) of `score` across all logged ablations (including discards). The agent can do this with `jq` over the log; `templates/confidence.sh` ships a reference.
- Confidence = `|best_delta| / MAD`.
- **≥ 2.0×**: improvement is likely real. Mark `kept` confidently.
- **1.0–2.0×**: above noise but marginal. Re-run the same ablation with a different seed (vary the dataset's example order, or change a single hyperparameter) before declaring it `kept`.
- **< 1.0×**: within noise. Default to `discard`.

When confidence is borderline, **don't move on**. Spending one extra iteration to confirm a real win is cheaper than chasing a phantom for ten.

### Small-N evals (n < 10)

If the oracle reports `N < 10`, one example flipping is a huge swing — at n=4, a single fail moves accuracy by 0.25. **At this scale MAD is dominated by sampling noise, not the dataset intervention.** Override the table above:

- Discard anything with `delta < 1 / max(N, 4)` regardless of MAD. That's roughly "one example's worth of swing".
- Before `kept`, *require* a confirmation run with a shuffled dataset (slug `<slug>-shuffle`). Only `kept` if both score above the small-N threshold.
- Note small-N status in the entry's `notes` so a future skim doesn't take a 0.4→0.6 jump at face value.

If the user controls the eval and `N` could be raised, ask them once (without asking for content) whether expanding the eval to ≥20 items is feasible. More items costs eval time; less noise saves iterations. Net usually wins.

---

## 6. The oracle wrapper

The blind eval is **whatever the user provides**, normalised through `capability.oracle.sh`. The wrapper's job is *containment*: it must reduce arbitrarily noisy output to one line of `SCORE=<float>` (optionally with `N=<int>`).

Three references ship in `templates/`:

- **`oracle.sh`** — calls a kiln-registered eval suite via `kiln-eval run --watch --json`, then reads only `summary.<scorer_field>` and `summary.num_examples`, falling back to `runs[0].metrics.*` for single-adapter runs (the `summary.*` aggregate is only populated for multi-adapter `compare` runs in current kiln). Everything outside those three fields is dropped. This is the default after `scaffold.sh`.
- **`oracle-paste.sh`** — prompts the user on stderr, reads one float from stdin (or `SCORE=<f> N=<i>` form), prints exactly `SCORE=<f>` on stdout. The agent sees nothing it shouldn't.
- (Custom) — the user can replace `capability.oracle.sh` with any script that prints `SCORE=<float>` as its last stdout line. The contract is the only thing that matters.

**Contract.** The wrapper:

1. Receives one positional arg — the adapter name (empty string = base model).
2. Returns exit 0 on success and prints `SCORE=<float>` (and optionally ` N=<int>`) as its **last** stdout line. Other stdout is allowed but ignored — the consuming helpers (`train_and_score.sh`, log helpers) only `grep` for `SCORE=` / `N=`.
3. Returns a non-zero exit on failure and prints `ORACLE_ERROR: <reason>` on stderr. The skill agent treats this as `status="oracle_error"` and logs it via `log_iter.sh <slug> oracle_error`.

If the wrapper ever prints text that resembles per-example output (transcripts, judgments, the eval's prompts), it is **compromised** — fix it, recommit, and log a `firewall_breach` line.

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

1. Run `$SKILL/templates/status.sh` — one-screen summary (suite, scorer, best, MAD, confidence, recent ledger, slugs in use).
2. Read `capability.md` end to end.
3. Read `capability.ideas.md` and `capability.config.json`.
4. Scan `hypotheses/*.md` for any file whose slug does **not** appear in `capability.jsonl` — that is an interrupted run; either complete it or log it as `status=crash` with a note.
5. `kiln adapters list | grep cap-` to confirm trained adapters match the log. Adapters present without a log line are orphan training runs; either claim them by writing a hypothesis post-hoc (and marking the entry `recovered=true` in `notes`) or `kiln adapters delete cap-<slug>`.
6. Re-baseline (§13). Append as a new `slug=baseline` line — do not overwrite the old one. Later confidence math uses the most-recent `slug=="baseline" or iter==0`.
7. Continue from where the log ends.

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

## 11. Answer-form discipline (the hidden killer of transfer)

A common failure mode: the prose dataset *does* teach the algorithm, but the evaluator scores short, terminal answers like `"62"` or `"Yes"` — and the SFT-tuned model now responds with a paragraph of prose. The capability is there; the surface form is wrong; the eval reads it as a regression.

The defence is **mixed-form supervision**:

1. The first 80–90 % of the dataset is your hypothesis (prose, abstract, etc.).
2. The remaining 10–20 % is **anchor examples**: short user questions paired with very-short assistant answers that respect the *output form* you suspect the eval wants. Do **not** copy the eval's prompts — invent new ones in the same shape.

Without knowing the eval's exact form, default to short-answer anchors: terminal `"42"`-style replies for math, `"Yes."`/`"No."`/`"I don't know."` for classification, single-sentence summaries for QA. The anchor is a stylistic preservation prior, not a teaching example — vary the surface but keep the answer terminal.

If iter-1 (a T-family prose ablation) **drops** the score relative to baseline, the most likely cause is answer-form drift, not a failure of transfer. Iter-2 should be the same dataset with 20 % anchor examples added (slug `<slug>-anchored`).

## 12. Regression watch (optional, recommended)

The blind oracle scores one capability. Aggressive SFT can lift it while breaking unrelated things. Recommend the user register a second suite — call it the **anchor suite** — that covers general competence (something like kiln's `qwen3.5-agentic-core` works). Call it the same way as the primary oracle, log its score in the `notes` field of the JSONL entry.

A `kept` ablation that loses >2× MAD on the anchor suite is a *Pyrrhic keep*. Mark it `kept` for the primary metric but explicitly note the regression and treat it as a partial dead end. The next ablation should attempt the same hypothesis at lower rank or with more diverse data, since extreme overfitting is the usual cause.

The anchor suite is **also blind**. The skill is not allowed to read it. Same firewall, two oracles.

## 13. Reproducibility

- **Re-baseline at the start of every session.** Server drift, model swap, kernel re-build — any of these can move the baseline. The first iteration of every new session is `slug=baseline`, regardless of whether one exists in the log already; compare deltas only within a session unless you explicitly re-validate.
- **Seed the SFT job.** kiln's request body accepts `seed`; the CLI doesn't expose it yet, but a future ablation that wants to test variance can issue the API call directly with a fixed seed and re-run. Pin a seed before declaring a 2× MAD result "real".
- **Pin dataset order.** SFT is sensitive to example order at low rank. If you want a clean re-run, don't shuffle the JSONL between attempts; if you want a variance probe, shuffle and re-run with a distinct slug like `<slug>-shuffle1`.
- **Record the kiln server version.** `curl -s "$SERVER/health" | jq -r '.version'` into `capability.config.json` at scaffold time. If a result is anomalous after a server restart, the version may have moved.

## 14. Worked example (read once, follow the shape)

Capability description (intake): *"two-digit subtraction with borrowing — model often messes up the borrow and produces off-by-ten errors."*

```text
# iter 0 — baseline (slug=baseline)
./capability.oracle.sh ""
# -> SCORE=0.36 N=25
# log: status=kept (it's the baseline), score=0.36, delta=0, hypothesis="baseline".

# iter 1 — T-family (slug=verbal-borrow-algorithm)
# hypotheses/verbal-borrow-algorithm.md (BEFORE generation):
#   Claim: 48 prose-only assistant turns describing the column-by-column
#     subtract-with-borrow procedure will lift two-digit subtraction
#     accuracy, because the model has the arithmetic primitives and just
#     needs reliable routing into the borrow case.
#   Mechanism: prose stabilises the chain "ones-column < ones-digit-of-
#     subtrahend → borrow from tens column → subtract" without giving
#     the model any numbers to overfit on.
#   Held out: no two-digit subtraction problems in the dataset.
#   Falsification: if Δ < MAD, the issue is form not routing; iter-2 adds
#     anchor examples. If Δ < -MAD, the dataset clobbered style; iter-2
#     reduces dataset to 24 examples + adds 8 short-answer anchors.

# Generate 48 prose examples per the §4 template. Each user asks how
# subtraction works in some specific framing ("If I have 73 apples and
# eat 28, how do I work out what's left?"); each assistant explains the
# borrow procedure in 3-6 sentences without writing the answer.
# Validate: jq -c '.messages | length' shows all = 2. Roles = user/assistant.

# Train + score
bash <skill>/train_and_score.sh verbal-borrow-algorithm
# -> ADAPTER=cap-verbal-borrow-algorithm LOSS=1.42 ELAPSED=37 SCORE=0.32 N=25

# Δ = -0.04. Negative. Falsification plan said: form drift -> add anchors.

# iter 2 — slug=verbal-borrow-anchored
# hypotheses/verbal-borrow-anchored.md (BEFORE):
#   Claim: same prose dataset + 8 short-answer anchor examples (single-
#     digit subtraction asked with terminal-form answer like "5") will
#     preserve the lift while restoring output form.
# Dataset: 48 prose + 8 anchors (single-digit subtraction Q -> single-
# digit numeric A). Surface-form-of-eval still held out (no two-digit
# subtraction problems).

bash <skill>/train_and_score.sh verbal-borrow-anchored
# -> SCORE=0.44 N=25

# Δ = +0.08 over baseline. With 3 logged ablations, MAD ≈ 0.04 → confidence ≈ 2×.
# kept. Update capability.md "what's been tried" with one line.

# iter 3 — triangulate. slug=numeric-drill-control
# A *different* family (M crossover -> symbolic): 32 worked two-digit-
# subtraction problems with numeric assistant answers. NOT a refinement
# of the winner; a control that asks "could we have got here without the
# prose route?".
# Falsification: if numeric-drill-control matches or beats anchored,
# the prose explanation isn't doing causal work — we're just doing SFT.

bash <skill>/train_and_score.sh numeric-drill-control
# -> SCORE=0.40 N=25

# Δ = +0.04 over baseline, less than the prose+anchor combo.
# Prose route is doing real work. Log: status=discard (below best),
# notes="control: pure numeric drill underperforms prose+anchor".

# iter 4 — refine the winner. slug=verbal-borrow-anchored-paraphrased
# Same recipe but the 48 prose examples are deliberately *more diverse*
# in framing (formal/casual/terse/poetic).
# ...continue.
```

The pattern is: **T-prose → anchor-fix → triangulate with control → refine winner**. The first four iterations buy you a real causal claim. Everything after is gain-chasing within an understood mechanism.

## 15. Anti-patterns (read this list every session)

- **Peeking.** Reading suite files, judgment outputs, per-example transcripts. Hard veto. See §1.
- **Eval-shaped training data.** Copying the eval's prompt template into your dataset. Even if you've never *read* the eval, if your data looks like the eval, transfer collapses into memorisation.
- **Compound interventions.** Changing dataset *and* rank *and* lr in one ablation. You'll never learn which knob did the work.
- **Rationalising results.** Discovering a mechanism *after* a positive score. Symptom: your hypothesis file is shorter than your `asi` block.
- **Loss-chasing.** Picking the adapter with the lower final loss when its blind score is worse. Loss is a training-time signal; the score is ground truth.
- **Big-data reflex.** Reaching for 1000-example datasets in iteration 2. Small ablations win; only scale a clearly-working hypothesis.
- **Stylistic clobber.** Training data with a strong style overwrites the base model's defaults even for unrelated queries. Watch for it; the eval may not test style but humans will notice the model went weird. Anchor examples (§11) and the anchor suite (§12) both defend against this.
- **Eval contamination through the system prompt.** Adding `"You are an expert at <capability>"` to every system prompt. Usually a bug, not a feature; remove the system prompt unless the user asked for it.
- **Drifting baseline.** Comparing iter 12 to a baseline from a previous session without re-baselining. The hardware/model/server has likely changed.
- **Same-slug overwrite without thinking.** `kiln train sft --adapter cap-X` replaces the adapter binary if cap-X exists. Re-using a slug across honest re-runs is fine; re-using it for a *different* hypothesis is a logging crime.

---

## 16. Adapter hygiene

After 20 iterations you have 20 `cap-*` adapters on the server. They consume disk and clutter `kiln adapters list`. **Do not delete them mid-session** — `capability.jsonl` references them by name and your archaeology trail dies without them. After finalisation (§9 final summary), keep the best 3–5 adapters and delete the rest:

```bash
# Inspect what's allocatable.
kiln adapters list | grep '^  cap-'

# Delete a single ablation's adapter.
kiln adapters delete cap-<slug>
```

If you genuinely need to free space mid-session, prefer **unloading** (the active-memory step) over **deleting** (the on-disk step). `kiln adapters unload` reverts to base; the weights remain on disk for the next eval call.

## 17. Tiny-smoke discipline (validate infra before paying for an ablation)

Before the very first real ablation, run a **tiny smoke** — 4 examples, 1 epoch, rank 4 — using `train_and_score.sh --no-score`. Goals:

- Confirm `kiln serve` is up and accepting jobs.
- Confirm dataset format is valid.
- Confirm the adapter directory gets a `cap-smoke` entry.
- Confirm `train_and_score.sh` returns `ADAPTER=… LOSS=… ELAPSED=…`.

A 4-example training run finishes in ~20 s. If anything is wrong, you find out in seconds rather than after a 5-minute real ablation. Delete the smoke adapter after (`kiln adapters delete cap-smoke`).

## 18. One-screen quickstart (the actual loop)

```bash
# Skill ships at .agents/skills/sft-capability-creator/. Templates at
# .agents/skills/sft-capability-creator/templates/. Reference them via
# absolute path or symlink. We'll abbreviate $SKILL.

SKILL=.agents/skills/sft-capability-creator

# 0. Intake — agent fills this in interactively with the user.
SLUG=capability-name        # short kebab; names the whole session
$SKILL/templates/scaffold.sh $SLUG
cd sft-cap.$SLUG
# edit capability.md (paste the user's verbatim description)
# edit capability.config.json (set eval_suite, scorer_field, direction)
# capability.oracle.sh is already the kiln-eval wrapper; if the user
# wants paste-back mode, replace with $SKILL/templates/oracle-paste.sh.
git add -A && git commit -m "cap[$SLUG]: intake"

# Tiny-smoke (§17) — run once at the very start of a session.
echo '{"messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}' > datasets/smoke.jsonl
echo '{"messages":[{"role":"user","content":"ok"},{"role":"assistant","content":"yes"}]}' >> datasets/smoke.jsonl
bash $SKILL/templates/train_and_score.sh smoke --no-score   # confirms infra
kiln adapters delete cap-smoke
rm -f datasets/smoke.jsonl adapters/smoke.txt

# Re-baseline (§13). Always.
./capability.oracle.sh ""             # -> SCORE=0.36 N=25
bash $SKILL/templates/log_iter.sh baseline kept 0.36 25 "" "" "" 0 0

# Loop — one iteration:
bash $SKILL/templates/status.sh        # see what's been tried
ABL=verbal-borrow-algorithm
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/$ABL.md
# Fill in hypotheses/$ABL.md (claim, mechanism, dataset shape, risk,
# falsification plan). DO NOT touch data yet.

# Generate datasets/$ABL.jsonl per §4's prompt template.
jq -c '.messages | length' datasets/$ABL.jsonl | sort -u   # sanity

# Train async (kiln SFT job) + blind-score in one call.
RESULT=$(bash $SKILL/templates/train_and_score.sh $ABL)
echo "$RESULT"
SCORE=$(echo "$RESULT" | grep -oE 'SCORE=[-0-9.]+' | cut -d= -f2)
N=$(echo "$RESULT" | grep -oE 'N=[0-9]+' | cut -d= -f2)
LOSS=$(echo "$RESULT" | grep -oE 'LOSS=[-0-9.]+' | cut -d= -f2)
ELAPSED=$(echo "$RESULT" | grep -oE 'ELAPSED=[0-9]+' | cut -d= -f2)

# Apply falsification plan from hypotheses/$ABL.md. Decide kept|discard.
STATUS=kept                            # or discard
bash $SKILL/templates/log_iter.sh $ABL $STATUS $SCORE $N \
  hypotheses/$ABL.md datasets/$ABL.jsonl cap-$ABL $LOSS $ELAPSED

# Annotate the just-appended entry. asi.* fields survive resume.
bash $SKILL/templates/annotate.sh \
  --what_worked "prose stabilised borrow routing" \
  --what_failed "" \
  --next_focus "anchor 20% to fix answer-form drift if confidence < 2x" \
  --notes ""

# Update capability.md "what's been tried" with a one-liner.
bash $SKILL/templates/confidence.sh   # advisory only
git add -A && git commit -m "cap[$ABL]: $STATUS"

# repeat until a stop condition fires (§9).
```

That is the entire skill. Everything else above is discipline.
