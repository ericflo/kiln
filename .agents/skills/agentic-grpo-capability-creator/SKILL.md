# agentic-grpo-capability-creator

Stand up one **agentic GRPO** capability — a coding/tool-using
capability the model exercises via the `pi` terminal agent — and run
it through enough iterations that you either ship a kept adapter or
retire the capability with a written lesson.

This is the sister skill to `grpo-capability-creator`. Read that first;
the GRPO discipline (information firewall, verdict gate, all-zeros
gate, sub-score regression watch) carries over unchanged. This skill
documents what's *different* when the rollouts are **multi-turn pi
sessions** instead of single-turn text completions.

## The single-line summary

Replace `single rollout (one assistant turn)` with `pi session (many
turns; tools)`. Replace `reward = rubric(response)` with
`reward = rubric(final sandbox state, transcript)`. Everything else
is GRPO with the standard kiln defaults.

## Skill inventory (for context)

- `sft-capability-creator` — SFT on curated assistant turns.
- `opd-capability-creator` — on-policy distillation against a teacher.
- `grpo-capability-creator` — GRPO on single-turn rollouts.
- **`agentic-grpo-capability-creator`** (this skill) — GRPO on
  multi-turn pi sessions.

---

## 0. Mental model

### When agentic-GRPO is the right tool

Reach for agentic-GRPO when **all** of these are true:

1. The capability is **agentic** — solving it requires the model to
   choose tool calls (bash, file edit, file read, web fetch, ...),
   not just emit text.
2. There exists a **verifier or judge** that operates on the
   *post-session sandbox state* (the workdir after pi exits) or on
   the *transcript* (the JSONL of turns). The reward function reads
   that state and returns a number in `[0, 1]`.
3. The base model can complete at least *some* sessions correctly,
   even rarely. Pi exposes bash + file CRUD to the model via tool
   calls; if the base model never emits a single valid `<tool_call>`
   the capability is broken at the tool-following layer, not at the
   reasoning layer — drop down to OPD on tool-call-arg-fidelity
   (cap #4) first.
4. The capability is solvable in **bounded wall-clock and turn count**.
   A pi session that wants 30 turns of trial-and-error costs ~5K-10K
   assistant tokens per rollout; at N=8 rollouts that's 40-80K tokens
   per group. Set a session budget; reward truncation is its own
   gradient.

Reach for **grpo-capability-creator instead** when (1) is false —
the capability is "emit text in the right shape." Reach for **OPD
or SFT** if even the tool-call format is wrong.

### How agentic-GRPO differs from single-turn GRPO

| | single-turn GRPO | agentic-GRPO |
|--|------------------|--------------|
| Rollout | One assistant message | Pi session: many user/assistant/tool turns |
| Reward signal | `f(response_text)` | `f(transcript, final_sandbox_state)` |
| Reward variance source | Sampling temperature | Sampling + tool-result feedback + sandbox state divergence |
| Tokens trained on | All assistant tokens | **Assistant tokens only** — NOT tool-result tokens |
| Practical token budget | ~512 / rollout | ~2K-8K / rollout |
| Wall-clock / rollout | ~5s | ~30-120s |
| What the model "sees" during training | Same as eval | Same as eval — same pi context window |
| Failure mode #1 | Goodhart on rubric | Goodhart on rubric **OR** on tool-use efficiency |
| Failure mode #2 | Length drift | Loop / repeat (re-running the same tool) |

The reward-design discipline from §6 of `grpo-capability-creator`
applies. Add to the adversarial-review questions:

- Can the model maximise reward by running `ls` 100 times then guessing?
- Can the model maximise reward by never calling tools, just emitting
  a confident answer?
- Can the model maximise reward by copying a known-good answer from a
  file pi has access to but shouldn't be reading?

For each YES, add an anti-shortcut sub-score.

### The pi rollout discipline

Pi is invoked headless with `pi -p "<prompt>"`. It runs in a fresh
workdir (which you provide), executes against the model served by
kiln (point pi at kiln via `kiln pi-setup`), captures every turn to
`~/.pi/sessions/<uuid>.jsonl`, and exits when the model emits a final
assistant turn with no tool call.

Empirical inputs you MUST verify on your pod before Phase 0 step 3:

- Where pi writes session JSONL (`~/.pi/sessions/`? `$HOME/.pi/`? per
  workdir?)
- The JSONL turn schema (role, content, tool calls, tool results,
  timestamps)
- How pi signals session end (exit code? sentinel turn?)
- What the wall-clock and turn budgets look like for the base model
  on your simplest tasks
- Whether `pi -p` blocks until session end (it should, for headless)

Until those are verified empirically, the skill is hand-waving. See
§17 (Phase 0 pi-smoke) — that smoke is mandatory before iter 1.

### Token-attribution: train only on assistant tokens

Pi sessions interleave four roles:

1. **system** — pi's initial prompt + tool manifest
2. **user** — the task prompt (your `pi -p` argument)
3. **assistant** — model output, possibly with `<tool_call>` XML
4. **tool** — tool result (bash stdout, file content)

For GRPO, the per-token IS ratio + advantage must apply to **assistant
tokens only**. System / user / tool tokens are context the model saw,
not tokens it produced — gradient through them is wrong.

Kiln's stock `tokenize_grpo_group` (as of `97b43ae`) treats every
post-prompt token as model-emitted. **That assumption breaks for
multi-turn pi sessions.** Before iter 1, kiln-train needs either:

- An extended `GrpoTrajectory` type that takes a `Vec<TurnSegment {
  role, text, train: bool }>` and builds a per-turn completion_mask, or
- A simpler hack: encode the full multi-turn conversation, then mark
  every token outside an assistant-turn span as `completion_mask =
  false` during tokenization.

This is a **Phase 0 kiln gap.** Land it (or workaround it) before
iter 1, or you'll be training against the wrong gradient signal and
your iter logs will be meaningless.

See `agentic-grpo-capability-creator/kiln-polish-prerequisites.md`
(written during your Phase 0) for the exact API change you'd land.

### Sandbox isolation

Each of N rollouts MUST run in an independent workdir. Two rollouts
sharing a workdir is a critical bug: rollout-2 sees rollout-1's
side effects and the reward is noise.

Recommended layout:

```
/tmp/agentic-rollouts/<run_id>/
├── 00/    # rollout 0 workdir — pi cwd, isolated
├── 01/    # rollout 1 workdir — pi cwd, isolated
├── ...
└── 07/    # rollout 7 workdir — pi cwd, isolated
```

Each workdir is initialized with a *task scaffold* — files pi needs to
start, test fixtures, etc. After pi exits, the reward function reads
the workdir AND the session JSONL.

If a rollout can't reach the network or read outside its workdir,
say so explicitly in the system message — don't rely on filesystem
permissions, the model has no way to know what's off-limits.

### What goes in the `messages` field of GrpoGroup

Pi sessions begin with a system message (pi's prompt) and a user
message (the task). The GRPO group's `messages` field is the
pre-rollout context — what the model "sees" before generating. Use
`messages = [system, user]` and tokenize all subsequent turns into
the completion text with the multi-turn mask.

---

## 1. Information firewall

Identical to §1 of `grpo-capability-creator`. One addition: **the model
itself, during rollout, must not see the rubric or the eval set**.
Don't include them in the workdir scaffold. The model can read files
in its workdir — anything you put there is fair game and may
short-circuit the reward.

---

## 2. Session files

```
capabilities/agentic-grpo/<slug>/
├── capability.md                        # Design + rubric + adversarial review
├── capability.config.json               # Pi binary path, kiln URL, sandbox root, hyperparams
├── capability.jsonl                     # Iteration log (one JSON line per iter)
├── capability.oracle.sh                 # Blind eval wrapper: takes adapter name, prints SCORE=
├── rubric.py                            # score_rollout(transcript, workdir, task) -> dict
├── build_corpus.py                      # Builds {train,eval}.tasks.jsonl
├── task_scaffold.py                     # Initializes a fresh workdir for one task
├── kiln-polish.jsonl                    # Kiln-itself observations
├── kiln-polish-prerequisites.md         # Phase 0 kiln gap list (multi-turn masking, etc.)
├── calibration/
│   ├── good.sessions/                   # 3 hand-crafted "ideal" sessions
│   └── bad.sessions/                    # 3 "obviously bad" sessions
├── datasets/
│   ├── train.tasks.jsonl                # Task specs (NOT pre-rolled completions)
│   └── eval.tasks.jsonl                 # Held-out eval tasks
├── prompts/                             # Per-iter filtered task subsets
├── hypotheses/                          # One markdown per hypothesis
├── runs/                                # Per-iter rollout archives (transcripts + workdirs)
│   └── <iter>-<slug>/
│       ├── 00/transcript.jsonl
│       ├── 00/workdir.tar.gz
│       └── ...
└── run_iter<N>.sh                       # Per-iter command
```

### `capability.md` template

Same shape as `grpo-capability-creator`. Add these sections:

```markdown
## Pi configuration
- Pi binary: `which pi` → /workspace/pi/target/release/pi
- Model id served by kiln: `qwen-3.5-4b-kiln`
- Pi --workdir flag: yes / pi inherits cwd
- Session JSONL location: ~/.pi/sessions/<uuid>.jsonl (verified <ts>)
- Session end signal: exit 0 + final assistant turn (verified <ts>)
- Turn budget per session: <N> turns / <S> seconds

## Reward function (designed with adversarial review applied — §0)

| Sub-score | Weight | What it measures | What it CANNOT be cheated by |
|-----------|--------|-------------------|-----------------------------|
| `outcome` | 0.60 | Verifier exit code 0 (e.g. pytest passes) | empty workdir, no-op session |
| `tested_before_done` | 0.15 | Final session has a successful bash test run | guessing without testing |
| `tool_call_efficiency` | 0.15 | 1.0 - clip(num_tool_calls / expected, 0, 1) | spamming `ls` |
| `format_compliance` | 0.10 | All assistant turns parse as valid tool calls or final text | mid-turn malformed `<tool_call>` |

Composite = sum(weights × sub_scores).

### Adversarial design (§0)

Q: Can the model pass without running tests?
A: Yes if outcome alone weights ≥0.50. Mitigation: `tested_before_done`
   sub-score punishes guessers.

Q: Can the model loop until something passes?
A: Yes — spam ls/cat until reward triggers. Mitigation: tool_call_efficiency
   sub-score, also wall-clock budget.

Q: Can the model write a no-op solution that passes only the visible doctests?
A: Yes, depending on eval set. Mitigation: held-out hidden tests.

### Headroom + group-variance baseline

- baseline composite: <0.xx>
- headroom: <1.0 - composite>
- baseline group variance: <0.xx>
- typical wall-clock per rollout: <S> seconds
- typical assistant tokens per rollout: <T> tokens
```

### `capability.jsonl` schema

Same as `grpo-capability-creator` + agentic-specific fields:

```json
{
  "iter": 1,
  "rollout_stats": {
    "mean_reward": 0.42,
    "mean_wall_clock_s": 47.3,
    "mean_assistant_tokens": 1820,
    "mean_tool_calls": 6.4,
    "rollouts_truncated_by_turn_budget": 1,
    "rollouts_truncated_by_token_budget": 0,
    "sessions_with_test_run": 0.75
  },
  "training": {
    ...
    "tokens_trained_on_per_rollout_p50": 1450,
    "tokens_trained_on_per_rollout_p95": 3800
  }
}
```

---

## 3. The loop

### Phase 0 — Intake (one-shot, once per session)

1. Write `capability.md`: description, sub-score table, adversarial
   review.
2. **Pi smoke** (§17 below) — verify pi binary, kiln integration,
   session JSONL location, end signal. Pin the empirical answers in
   `capability.md`.
3. Write `rubric.py`. The signature is:
   `score_rollout(transcript: list[dict], workdir: str, task: dict) -> dict`.
4. Write `task_scaffold.py` — `init_workdir(task: dict, dir: str) -> None`.
   This sets up a fresh workdir for one task. Idempotent.
5. Write `build_corpus.py`. Build `datasets/train.tasks.jsonl` and
   `datasets/eval.tasks.jsonl` (disjoint).
6. Hand-write `calibration/good.sessions/{00,01,02}/` and
   `calibration/bad.sessions/{00,01,02}/` (3 each). Each contains
   `transcript.jsonl` + `workdir/`.
7. Run `python3 $SKILL/templates/rubric_sanity.py` — must exit 0.
8. **Kiln gap audit.** Write `kiln-polish-prerequisites.md`. At minimum
   it must answer: does `kiln-train` support multi-turn rollouts with
   tool-token masking? If no, name the change you need before iter 1.
9. Stand up `kiln serve` with the base model. `kiln pi-setup`. Health
   check pi against kiln (manually: `pi -p "echo hello"` and watch
   the session JSONL appear).
10. **Baseline eval.** `./capability.oracle.sh ""` — base model
    completes the eval tasks. Log iter 0.
11. **Group-variance baseline.** Sample 4 rollouts on 8 random
    training tasks at training-time sampling params, compute reward
    variance per task, mean across tasks.
12. **Baseline sanity gate.** See §0 of `grpo-capability-creator`.
    Same rules. Plus a new one specific to agentic:
    - If `sessions_with_test_run < 0.10` at baseline — the model is
      not learning to test before declaring done in seed form. The
      `tested_before_done` sub-score will not get a gradient signal
      because no rollout will trigger it. Either lower the weight or
      drop the sub-score until iter 2+.
13. **Tiny-smoke** (§17): 1 task, 2 rollouts, 1 GRPO step, no eval.
14. Commit Phase 0 artifacts.

### Phase 1 — Hypothesise

Same as `grpo-capability-creator`. The hypothesis families (§4 below)
have agentic-specific entries.

### Phase 2 — Construct

Build `prompts/<slug>.tasks.jsonl` — the filtered subset for this iter.
Filter rules: drop tasks where baseline mean reward > 0.9 (no
headroom) AND baseline reward variance < 0.05 (no signal).

### Phase 3 — Train

Run `run_iter<N>.sh`. The training command path is:

1. **Rollout pass.** For each task in `prompts/<slug>.tasks.jsonl`,
   for each of N generations:
   - Provision a fresh sandbox dir.
   - `task_scaffold.py init_workdir`
   - Spawn `pi -p "<task prompt>" --workdir <dir>` (or
     `cd <dir> && pi -p ...`).
   - Wait for exit. Snapshot the workdir. Capture the session JSONL.
   - Score with rubric: `reward = score_rollout(transcript, workdir,
     task)["composite"]`.
2. **Group assembly.** Build one `GrpoGroup` per task. The `messages`
   field is `[system, user]` from the pi session. Each `completion`
   has `text = <concat of assistant turns + tool turns, with
   completion_mask masking tool tokens out>` and `reward = <composite>`.
3. **GRPO step.** Feed the JSONL to `grpo_train_jsonl` (or the new
   `grpo_train_jsonl_trajectories` if you landed it in Phase 0 step 8).

Step 1 dominates wall-clock. With 8 generations × ~60s = ~8 minutes
per task, a 30-task iter is ~4 hours. Plan for that.

### Phase 4 — Blind eval

Same as `grpo-capability-creator`. `./capability.oracle.sh <adapter>`
runs the eval task set with the adapter loaded. Always 1 generation
at temperature 0.0 for determinism on the eval composite — but ALSO
report pass@4 with temperature 0.8 for a diversity signal.

### Phase 5 — Close the loop

Same as `grpo-capability-creator`. Inspected rollouts means **read 3
session transcripts end-to-end** and paste excerpts into the hypothesis
file. For agentic capabilities, "inspecting" includes:

- What tools did the model call?
- Did the model run tests before declaring done?
- Are there loops or repeats?
- Did the model emit anything that looks like reward gaming?

### Phase 6 / 7 — Iterate / Cadence

Same as `grpo-capability-creator`.

---

## 4. Hypothesis families (agentic-specific additions)

All families from `grpo-capability-creator` §4 apply. Add:

| Family | What you change | When to reach for it |
|--------|-----------------|---------------------|
| **HA1 — System prompt anchor** | Edit pi's system message (or your task system prompt) to nudge tool-use shape | When iter 1 rollouts show the model "knows" how to solve the task but emits poor tool-call shape. |
| **HA2 — Turn budget** | Lower `--max-turns` per pi session | When `mean_tool_calls > 2× expected` — the model is wandering. |
| **HA3 — Task difficulty rebalance** | Edit `train.tasks.jsonl` — drop trivially easy and unsolvable | Same as H8 from grpo but at task-shape granularity. |
| **HA4 — Stricter tested-before-done** | Raise `tested_before_done` weight; require the test passes (not just runs) | When iter 1 shows `sessions_with_test_run` rising but `outcome` flat — model "tests" without listening to results. |
| **HA5 — Trajectory-shaping reward** | Add a `made_progress_each_turn` sub-score that scores per-turn deltas | When sessions stall (same workdir state after 10 turns). Heavy compute cost. |
| **HA6 — Verifier strictness** | Tighten the verifier (more hidden tests) | When the model passes the visible eval but inspected transcripts show the implementation is partial. |

---

## 5. Anti-laziness gates

Same as `grpo-capability-creator` §5. Plus:

- **No iter without a pi-smoke** in the latest run.
- **No iter without reading 3 transcripts** by hand.
- **No iter that increases turn budget without also lowering
  `num_generations` or task count.** Compute cost balloons; you'll be
  paying for noise.

---

## 6. Reward design (agentic)

The single-turn `score_response(text) → composite` becomes
`score_rollout(transcript, workdir, task) → composite`.

### Composite sub-scores — recommended starter set

| Sub-score | What | Where computed |
|-----------|------|----------------|
| `outcome` | Did the task complete? Run `task["verify_cmd"]` in workdir; reward = exit-0. | workdir |
| `tested_before_done` | Did the *last successful* bash run happen BEFORE the final assistant turn? | transcript |
| `tool_call_efficiency` | `clip(1.0 - num_tool_calls / expected, 0, 1)` | transcript |
| `format_compliance` | Fraction of assistant turns where every `<tool_call>` block parses as a valid tool call. | transcript |
| `no_loop` | 1.0 minus the fraction of tool calls that are exact duplicates of an earlier call in the same session. | transcript |

### Designing `outcome` (the load-bearing one)

The outcome verifier MUST be:
- **Deterministic** — same workdir → same result.
- **Hermetic** — no external network, no time-dependent state, no
  randomness.
- **Fast** — runs in < 5s per rollout. You'll run it on every rollout
  for every iter.
- **Not cheatable by no-op** — an empty workdir should score 0.

For Python-shaped tasks: use the existing `PythonDoctest` scorer that
just landed in kiln-eval (PR #1048), or `pytest --tb=line --no-header
-q`.

For shell/file tasks: write a small bash script that runs in the
workdir and exits 0 on success.

### When to use an LLM judge

**Only when no verifier exists.** LLM judges introduce two failure modes
GRPO is bad at handling:
- Score noise across rollouts becomes advantage noise becomes wasted
  gradient.
- The model can learn to write text that *appears* correct to the
  judge but isn't.

If you must use a judge:
- Pin to a single strong model (Claude / GPT-class).
- Multi-sample 3× and take the median.
- Cache scores by (task, transcript-hash) — judges are deterministic
  per input, so re-scoring during a session is wasted.
- Add a "verifier-disagrees-with-judge" sub-score on the eval set if
  you can — measures judge drift.

---

## 7. Group statistics watch — agentic additions

Plus the standard set from `grpo-capability-creator` §7:

- **`mean_wall_clock_s`** per rollout. If it drifts up by >20% iter
  over iter, the model is using more turns — that's either good
  (better solutions) or bad (looping). Inspect.
- **`rollouts_truncated_by_turn_budget`** — if > 10% of rollouts hit
  the turn budget, your budget is too low or the model is looping.
- **`sessions_with_test_run`** — what fraction tested before declaring
  done. Should rise across iters if you weighted `tested_before_done`.

---

## 8. Memory / time budget for agentic rollouts

Rough rule for one A6000 (48GB) with rank-16 LoRA on Qwen3.5-4B:

- Rollout pass: each rollout runs pi against kiln serving the base
  model. kiln serves at ~50 tok/s on A6000. A 2000-token assistant
  session takes ~40s + tool exec time (variable).
- Sample budget: with 8 rollouts × 30 tasks × 60s = 4 hours per iter
  rollout pass.
- Training pass: per-group token count is `2 × 8 × max_tokens`. For
  agentic this is `2 × 8 × (assistant + tool tokens)` ≈ 80K tokens
  per group at p95. Train with `num_generations: 4` if you're
  budget-pinched.

Realistic iter cost on A6000: 5-8 hours wall-clock.

On A100-80GB: roughly half.

If your iter cost is north of 8 hours wall-clock, you're either
running too many tasks, too many generations, or too long a session
budget. Pick one and lower it.

---

## 9. Loss is deceptive (agentic addition)

Standard GRPO loss caveats apply. Plus: agentic GRPO loss is
dominated by the *longest* completion in the batch — a single
runaway session can dominate the per-token aggregation. If you see
a single iter with loss 10× the previous iter's, check
`rollouts_truncated_by_turn_budget` and `mean_assistant_tokens
p95` — there's probably a 10K-token outlier in the batch.

---

## 10. The all-zeros failure mode

Same gate as `grpo-capability-creator` §10. The threshold (`< 0.5 ×
baseline`) is just as important here; if anything more important
because agentic rollouts have higher iter cost and you can't afford
a wasted run.

---

## 11–16. (Same as grpo-capability-creator)

Sub-score regression watch (§11), stop conditions (§12), closeout
(§13), resuming (§14), periodic rollout sanity check (§15), and
kiln-polish ledger (§16) carry over unchanged. The kiln-polish
entries for agentic capabilities tend to skew toward "pi
integration" and "transcript handling" categories — note that.

---

## 17. Pi-smoke (mandatory before iter 1)

A pi-smoke is a sequence that proves your pod setup actually works
end-to-end. Run it before Phase 0 step 3.

```bash
# 1. Pi binary on PATH
command -v pi || (echo "install pi first"; exit 1)

# 2. Kiln serving the base model
curl -sf http://localhost:8420/v1/models | grep -q qwen-3.5-4b-kiln

# 3. Pi configured against kiln
test -f ~/.pi/agent/models.json && grep -q kiln-local ~/.pi/agent/models.json

# 4. Headless pi session: print HELLO and exit
mkdir -p /tmp/pi-smoke
cd /tmp/pi-smoke
pi -p "Print exactly the string HELLO and nothing else, then exit." 2>&1 | tee pi-smoke.log

# 5. Session JSONL appears
ls ~/.pi/sessions/ | tail -5
SESS=$(ls -t ~/.pi/sessions/*.jsonl | head -1)
wc -l "$SESS"
# Should be > 0; should contain an assistant turn

# 6. Tool-call session: write a file, read it back
cd /tmp/pi-smoke
pi -p "Create a file called marker.txt containing the text PASS, then exit." 2>&1 | tee tool-smoke.log
cat /tmp/pi-smoke/marker.txt  # should print PASS

# 7. Two pi instances in parallel don't interfere
mkdir -p /tmp/pi-smoke/a /tmp/pi-smoke/b
(cd /tmp/pi-smoke/a && pi -p "Write 'A' to result.txt and exit.") &
(cd /tmp/pi-smoke/b && pi -p "Write 'B' to result.txt and exit.") &
wait
[ "$(cat /tmp/pi-smoke/a/result.txt)" = "A" ] || (echo "isolation broken"; exit 1)
[ "$(cat /tmp/pi-smoke/b/result.txt)" = "B" ] || (echo "isolation broken"; exit 1)
```

If any step fails, fix the infra before writing any rubric.

**Tiny-smoke (training)**: one task, two pi rollouts, one GRPO step,
no eval. Must complete in < 5 minutes wall-clock.

---

## 18. One-screen quickstart

```bash
SKILL=.agents/skills/agentic-grpo-capability-creator

# 0. Intake (one-shot)
SLUG=pi-doctest
$SKILL/templates/scaffold.sh $SLUG
cd capabilities/agentic-grpo/$SLUG

# Pi-smoke (§17) — REQUIRED before going further
bash $SKILL/templates/pi_smoke.sh   # exits 0 only when pi + kiln are wired

# Edit capability.md, rubric.py, task_scaffold.py, capability.oracle.sh
python3 build_corpus.py
python3 $SKILL/templates/rubric_sanity.py    # MUST exit 0

# Kiln gap audit — see §0 token-attribution paragraph
edit kiln-polish-prerequisites.md
# Land the multi-turn masking change in kiln-train (or workaround) here.

# Baseline eval — base model
./capability.oracle.sh ""

# Group-variance baseline
python3 $SKILL/templates/group_variance_baseline.py   # NEW — measures multi-rollout reward variance

# Tiny-smoke (training)
bash $SKILL/templates/tiny_smoke.sh

# Iter 1 — always HA1 (default recipe + simplest task system prompt)
ABL=h1-default-recipe
cp $SKILL/templates/hypothesis.md.tmpl hypotheses/$ABL.md
bash run_iter1.sh    # Phases 3+4: rollouts → grpo step → eval
RESULT=$(./capability.oracle.sh $ABL)

# Phase 5 — verdict gate
# - read 3 transcripts; paste excerpts into hypotheses/$ABL.md
# - fill verdict, what_worked, what_failed, next_focus
# - log kiln-polish if needed
bash $SKILL/templates/log_iter.sh $ABL <fields>

git add -A && git commit -m "cap[$SLUG/$ABL]: kept (+0.04)"

# Iter 2... follow §3 Phase 6 sequencing.
```

That's the whole skill. The discipline is in the gates, not the loop.

---

## 19. Open kiln gaps (read every session)

These are agentic-grpo-specific kiln gaps. Read them before Phase 0
step 3.

1. **Multi-turn assistant masking.** `tokenize_grpo_group` currently
   treats every post-prompt token as model-emitted. Multi-turn pi
   sessions need per-turn masking. Status: not landed as of cap
   intake. Workarounds: write `GrpoGroup`s with the full multi-turn
   conversation in the user message and a single concatenated
   assistant text, and ignore the bias for v0 (NOT recommended); OR
   land the extension before iter 1 (recommended).

2. **Raw token capture from kiln-server.** The pi/openai-compatible
   path normalizes Qwen XML tool calls into OpenAI `tool_calls`. For
   GRPO training we need the *raw* assistant tokens (XML included).
   Status: kiln-server's chat completion handler logs both. Verify
   on your pod via `/v1/models` debug endpoint or direct log
   inspection.

3. **Sandbox lifecycle.** No kiln helper today for managing rollout
   sandboxes. Each cap rolls its own scaffold/teardown. Consider
   landing a `kiln-rollout-sandbox` helper if you find yourself doing
   this twice.

4. **Trajectory replay.** No kiln helper today for replaying a
   captured pi session against a new model. Useful for re-scoring
   when you change the rubric. Consider landing.

5. **Wall-clock budget per rollout.** Pi doesn't ship with a
   wall-clock budget flag (as of cap intake). Without one, a
   runaway session can chew through your iter budget. Either land a
   `--max-wall-clock-s` flag in pi or wrap pi in a `timeout`-shaped
   helper.

6. **H100 (SM90) needs two env-flag workarounds as of kiln main 2026-05-18.**
   The fused GDN gates kernel (`kiln_gdn_gates_bf16`) compiles and
   functions correctly in isolation on H100 (verified with
   `gates_bench`), but fails in production code paths
   (paged-decode inference + training reference-forward) with
   `kiln_gdn_gates_bf16 failed with status 500`. PR #1050 cleared
   `cudaGetLastError` before the launch, which silenced the
   "first-request" form of the symptom, but the production paths
   still trip something stream/context-related that single-launch
   benches don't. Workaround: `KILN_DISABLE_FUSED_GDN_GATES=1` for
   both `kiln serve` and `cuda_grpo_ablation`.
   
   The batching engine also fails on H100 with
   `batched-engine prefill forward pass failed` for every request.
   Workaround: `KILN_BATCHING_ENGINE=0`.
   
   Combined env for H100:
   `KILN_MODEL_PATH=... KILN_BATCHING_ENGINE=0 KILN_DISABLE_FUSED_GDN_GATES=1 kiln serve`
   
   Both are no-ops on A6000 / A100. Track in kiln-polish.jsonl.

7. **Pod hibernation loses artifacts.** RunPod pods can hibernate
   mid-iter (we lost iter 2's adapter when the pool reaped the pod
   that was running it). **Push adapter files to B2 or copy them
   locally as soon as they're trained**, before the next rollout
   pass. The training is fast (~5 min); recomputing it after a
   hibernation is fine — but rollouts cost real wall-clock and
   should not need to be re-run.

8. **Kiln serve grabs all VRAM.** When you switch from rollouts to
   training, `pkill -9 -f "kiln serve"` before launching
   `cuda_grpo_ablation` — otherwise training OOMs at model-load.
   The training process loads its own model copy and won't share
   VRAM with the running server.

9. **Adapter dir defaults to `model_path/adapters/`.** Not
   `/workspace/kiln/adapters/`. When training writes adapters to
   `/tmp/...`, symlink them into `$KILN_MODEL_PATH/adapters/` before
   calling `POST /v1/adapters/load`. Otherwise the endpoint 404s
   with "adapter not found" even though the file exists.

10. **Pi `--session-dir` is per-rollout.** Use a unique
    `--session-dir` per rollout (or per `pi -p` invocation) so two
    concurrent rollouts don't clobber each other's session files.
    `pi 0.75.x` defaults to `~/.pi/agent/sessions/<workdir-encoded>/`
    which collides if two rollouts use the same workdir.

Anything you hit during the cap goes to `kiln-polish.jsonl`.
