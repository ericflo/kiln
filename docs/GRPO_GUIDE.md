# GRPO Training Guide

This guide walks through GRPO end-to-end on Kiln: what it is, the loop you run,
and three runnable verifiable-reward examples (math correctness, JSON-validity,
code-runs). Everything below assumes a Kiln server running on
`http://localhost:8420` with at least one adapter slot free and Python 3.10+
on the client.

The end-to-end generate/train/hot-swap loop requires the explicit development
profile: start Kiln with `KILN_SERVING_PROFILE=experimental`. The default
`stable` profile returns `409 serving_profile_conflict` for training, while
`maintenance` disables the generation half of the loop. Production systems
should generate under `stable`, restart into drained `maintenance` for the
training mutation, then restart into `stable` to evaluate. See
[Serving Profiles](SERVING_PROFILES.md).

## What GRPO is, in 5 sentences

GRPO is **Group Relative Policy Optimization**, the reinforcement-learning
algorithm introduced by [DeepSeekMath](https://arxiv.org/abs/2402.03300). For
each prompt you generate a *group* of `n` completions, score them with a
reward function you write, and turn those rewards into a policy-gradient
update — no separate critic network and no replay buffer, which is what makes
it suit Kiln's single-process design. Within each group the rewards are
mean-zeroed and (optionally) normalized to produce per-completion *advantages*,
so the update only depends on which completions in the group beat the others.
When exact rollout provenance is available, a clipped importance-sampling
ratio corrects for drift from the policy that generated the rollouts. Without
that provenance, Kiln explicitly fixes the ratio at one rather than pretending
that the KL reference was the behavior policy. A separately configured KL
penalty can keep the adapter from collapsing onto one high-reward output.
**You write the reward function. That's the whole point.**

## The loop

Every GRPO iteration is the same four-step cycle. In the `experimental`
profile this is one process, two HTTP endpoints, and an atomic adapter
hot-swap:

```
   [ generate ]    kiln rollout-generate
        │            (scored JSONL + exact behavior provenance)
        ▼
   [  score   ]    your reward fn (Python, regex, json.loads, subprocess, …)
        │
        ▼
   [  train   ]    POST /v1/train/grpo
        │            (groups + scored completions → adapter delta)
        ▼
   [hot-swap +     next inference call already uses the new adapter
    repeat   ]
```

That's the whole loop. Run it 5–20 times for a toy task and watch the mean
reward trend up.

## Endpoint reference

The full schema for both endpoints lives in
[Quickstart §9](../QUICKSTART.md#9-advanced-api-examples). The fields used in
this guide are:

**`kiln rollout-generate`** — renders a chat request for every task and seed,
forces non-streaming single-choice generation with exact rollout provenance,
runs your scorer, and writes trainer-compatible JSONL. It validates the
returned schema, seed, adapter identity, prompt/content hashes, complete action
coverage, and usage counts before invoking the scorer. Output publication is
atomic: one missing or malformed record fails the command without replacing an
existing dataset. This is the supported source for
`behavior_policy: "recorded"`. Recorded tool-call rollouts remain unsupported:
tool definitions and tool choice fail admission until generated tool actions can
be represented in the scored completion. Prior assistant `tool_calls` and
`role: "tool"` responses are supported context: their `name` and `tool_call_id`
fields survive the output JSONL and are bound into the provenance prompt hash.

**`POST /v1/completions/batch`** — issues `prompts.len() × n` text completions
in one HTTP round-trip. It remains useful for fast online or explicitly
uncorrected experiments, including the compact worked examples below, but it
does not emit exact per-token behavior provenance. Train those rollouts only
with `behavior_policy: "no_importance_correction"`.

**`POST /v1/train/grpo`** — accepts `groups`, where each group is
`{"messages": [...], "completions": [{"text": "...", "reward": 0.0}, ...]}`.
The request is enqueued and returns a `job_id` immediately; training runs on a
background thread. Its output remains hidden until Kiln atomically publishes
and auto-loads the completed adapter at an iteration boundary. Every `config`
field has a server default, so `{"groups": [...]}` is a valid minimal payload.
The default `behavior_policy` is `"no_importance_correction"`, which is the
honest mode for the text-only examples below. Set it to `"recorded"` only when
every scored completion includes a validated `provenance` object from the
generation that produced it.

## Recommended recorded-policy workflow

Create one task per line and a normal chat request template. The CLI owns
`seed`, `adapter`, `n`, `stream`, and `rollout_provenance`, so template values
for those fields are overwritten deliberately.

```json
{"id":"sum-1","prompt":"What is 47 + 138? Reply with just the number.","answer":185}
```

```json
{
  "model": "Qwen3.5-4B",
  "messages": [{"role": "user", "content": "{{prompt}}"}],
  "temperature": 0.9,
  "max_tokens": 64
}
```

The scorer receives the task, exact request, full response, parsed content,
usage, and latency on stdin. A minimal executable scorer can print one number:

```python
#!/usr/bin/env python3
import json, re, sys

row = json.load(sys.stdin)
numbers = re.findall(r"-?\d+", row["content"])
print(float(bool(numbers) and int(numbers[-1]) == row["task"]["answer"]))
```

```bash
chmod +x score_math.py
kiln rollout-generate \
  --adapter base \
  --thinking false \
  --tasks tasks.jsonl \
  --seeds 8 \
  --seed-start 42 \
  --request-template request.json \
  --scorer ./score_math.py \
  --output math.rollouts.jsonl \
  --summary-output math.rollouts.summary.json
```

Each completion contains `text`, `reward`, and the server-issued
`kiln.rollout-provenance.v1` object. It intentionally has no synthetic
single-turn `trajectory`: adding one after generation would change the scored
payload identity. The JSONL contains only the canonical training schema, so it
can enter strict HF/TRL export without normalization. Per-request latency,
usage, seed, adapter, server performance, and raw scorer output live in the
separate summary JSON. When the JSONL path is visible to the server, submit it
with recorded importance correction:

```bash
curl -s http://localhost:8420/v1/train/grpo \
  -H 'Content-Type: application/json' \
  -d "{\"dataset_path\":\"$(realpath math.rollouts.jsonl)\",\"config\":{\"behavior_policy\":\"recorded\",\"output_name\":\"math-grpo\"}}" \
  | python3 -m json.tool
```

The server replays the recorded template invocation with its pinned tokenizer,
verifies the exact prompt prefix and scored payload again, and uses only sampled
actions as policy targets. Runtime-forced thinking-close tokens remain context
and never receive invented behavior probabilities.

## Worked example 1: Math correctness reward

The cheapest possible verifiable reward: was the final integer in the
completion equal to the ground-truth answer?

```python
# math_reward.py — runnable end-to-end against a kiln server on :8420
import json
import re
import requests

KILN = "http://localhost:8420"

PROBLEMS = [
    {"messages": [{"role": "user", "content": "What is 47 + 138? Reply with just the number."}],         "answer": 185},
    {"messages": [{"role": "user", "content": "What is 23 * 17? Reply with just the number."}],          "answer": 391},
    {"messages": [{"role": "user", "content": "What is 1024 - 376? Reply with just the number."}],       "answer": 648},
    {"messages": [{"role": "user", "content": "What is the sum of the integers from 1 to 20?"}],          "answer": 210},
]

def reward(completion_text: str, answer: int) -> float:
    """Extract the last integer in the completion. +1 if it equals `answer`, else 0."""
    nums = re.findall(r"-?\d+", completion_text)
    if not nums:
        return 0.0
    return 1.0 if int(nums[-1]) == answer else 0.0

# 1. Generate — 8 samples per prompt, single batch round-trip
batch = requests.post(f"{KILN}/v1/completions/batch", json={
    "prompts":     [p["messages"] for p in PROBLEMS],
    "n":           8,
    "temperature": 0.9,        # diverse rollouts
    "max_tokens":  64,
    "seed":        42,
}).json()

# Reshape: items[i*n + j] belongs to prompt i, completion j
n = 8
groups = [{"messages": p["messages"], "completions": []} for p in PROBLEMS]
for item in batch["completions"]:
    pi = item["prompt_index"]
    text = item["text"]
    r = reward(text, PROBLEMS[pi]["answer"])
    groups[pi]["completions"].append({"text": text, "reward": r})

mean_reward = sum(c["reward"] for g in groups for c in g["completions"]) / (len(PROBLEMS) * n)
print(f"mean reward this round: {mean_reward:.3f}")

# 3. Train — server enqueues the GRPO step and hot-swaps the resulting adapter
job = requests.post(f"{KILN}/v1/train/grpo", json={
    "groups": groups,
    "config": {
        "kl_coeff":      0.1,
        "clip_epsilon":  0.2,
        "lora_rank":     16,
        "output_name":   "math-correctness",
        "auto_load":     True,
    },
}).json()
print("queued:", job["job_id"], job["state"])
```

Run that script in a loop for 10–20 rounds. With Qwen3.5-4B as the base, the
mean reward typically climbs from ~0.4 (some completions already nail it) to
~0.85 within the first dozen rounds for arithmetic this simple. Re-running
with the same `seed` lets you compare runs directly.

### Serving during training

Server-submitted GRPO does not reserve the GPU for the lifetime of the job.
Model residency/setup, each complete optimizer group (including any EMA
reference refresh), device snapshots, and final smoke/cleanup work acquire
exclusive GPU ownership separately. The backend is synchronized before every
release, and a failed settlement or panic quarantines the process until restart.
Reward filtering, JSONL reads, tokenization, progress callbacks, safetensors
encoding, and filesystem publication run without the GPU writer, so healthy
inference can make progress between groups.

Each group is still one atomic training interval. A very large or long-context
group can therefore produce a correspondingly long attributed inference wait.
Inspect `phase_timings.gpu_writer_wait_ms`, `gpu_writer_held_ms`, and
`gpu_writer_acquisitions` in `train_receipt.json` when diagnosing pauses; these
fields distinguish expected group-level contention from time spent outside GPU
ownership.

### Exact checkpoint and resume

Set a cadence when a run must survive cancellation or process failure:

```bash
kiln train grpo \
  --file math.rollouts.jsonl \
  --adapter math-grpo \
  --checkpoint-interval 25
```

The cadence counts committed optimizer groups. Cooperative cancellation waits
for the current group to settle and publishes an immutable
`.kiln-checkpoint`; a process crash can lose only the in-flight group after the
newest committed checkpoint. `kiln train status --job-id JOB_ID` and
`GET /v1/train/jobs/{job_id}` report its direct basename. The browser job drill
also labels whether the checkpoint came from inline or JSONL GRPO and can
prepare the matching form.

Resume with the identical source, route, adapter name, and configuration:

```bash
kiln train grpo \
  --file math.rollouts.jsonl \
  --adapter math-grpo \
  --checkpoint-interval 25 \
  --resume-checkpoint math-grpo-checkpoint-step-00000025.kiln-checkpoint
```

For the API, put the same two fields under `config`:

```json
{
  "dataset_path": "/absolute/path/math.rollouts.jsonl",
  "config": {
    "behavior_policy": "recorded",
    "output_name": "math-grpo",
    "checkpoint_interval": 25,
    "resume_checkpoint": "math-grpo-checkpoint-step-00000025.kiln-checkpoint"
  }
}
```

Resume restores adapter and optimizer tensors, frozen/EMA reference state and
cadence, exact inline order or JSONL line/byte cursor, RNG streams, loss
history, policy/ECHO/gradient diagnostics, and phase timings. Before GPU setup,
Kiln validates the complete artifact set and checksums plus exact data,
configuration, model/base weights, tokenizer, precision, backend, and derived
gradient plan. A PEFT adapter snapshot is a serving/warm-start artifact, not a
resume point. See [Native Training Checkpoints](training-checkpoints.md) for the
full fail-closed and immutable-name contract.

## Worked example 2: JSON-validity reward (format compliance)

A reward function doesn't have to be binary. Partial credit for *almost*
right is often the difference between a stuck loop and a learning one.

```python
# json_reward.py
import json
import requests

KILN = "http://localhost:8420"
REQUIRED_KEYS = {"name", "age", "city"}

PROMPTS = [
    [{"role": "user", "content": "Return a JSON object with keys name, age, city for a 32-year-old "
                                  "named Mira living in Lisbon. Reply with only the JSON."}],
    [{"role": "user", "content": "Return a JSON object with keys name, age, city for a 19-year-old "
                                  "named Theo living in Cairo. Reply with only the JSON."}],
    [{"role": "user", "content": "Return a JSON object with keys name, age, city for a 47-year-old "
                                  "named Akemi living in Kyoto. Reply with only the JSON."}],
]

def reward(text: str) -> float:
    """1.0 = parses + has all keys, 0.5 = parses, 0.0 = doesn't parse."""
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return 0.0
    if not isinstance(obj, dict):
        return 0.0
    return 1.0 if REQUIRED_KEYS.issubset(obj.keys()) else 0.5

batch = requests.post(f"{KILN}/v1/completions/batch", json={
    "prompts": PROMPTS, "n": 8, "temperature": 0.9, "max_tokens": 96, "seed": 0,
}).json()

groups = [{"messages": p, "completions": []} for p in PROMPTS]
for item in batch["completions"]:
    groups[item["prompt_index"]]["completions"].append(
        {"text": item["text"], "reward": reward(item["text"])}
    )

requests.post(f"{KILN}/v1/train/grpo", json={
    "groups": groups,
    "config": {"lora_rank": 16, "output_name": "json-format"},
}).raise_for_status()
```

Format compliance is harder to learn than arithmetic — expect 20–40 rounds
before mean reward saturates. The 0.5/1.0 split matters: if you reward only
the perfect output the gradient is zero whenever the whole group fails, and
the loop stalls.

## Worked example 3: Code-runs reward (subprocess-based)

The most powerful verifiable reward is "run the code and see if it works."
Kiln's GRPO endpoint doesn't care how you produce the score, only that it's a
float per completion.

````python
# code_reward.py
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

import requests

KILN = "http://localhost:8420"

TASK = {
    "messages": [{"role": "user", "content":
        "Write a Python function `add(a, b)` that returns a + b. "
        "Reply with only the function definition, no prose."}],
    "tests": [
        ("add(1, 2)",     3),
        ("add(-5, 5)",    0),
        ("add(10, 100)",  110),
        ("add(0, 0)",     0),
    ],
}

CODE_BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)

def reward(text: str, tests: list[tuple[str, int]]) -> float:
    """Fraction of test cases that pass when the completion is exec'd in a subprocess.

    Security caveat: this exec's untrusted model output. For real workloads,
    run inside a sandbox (Docker, gVisor, firejail, …). The example below
    assumes you trust your own model's output during development.
    """
    m = CODE_BLOCK.search(text)
    src = m.group(1) if m else text
    harness = "\n".join(
        f"assert {expr} == {expected}, '{expr} expected {expected}'"
        for expr, expected in tests
    )
    program = textwrap.dedent(src) + "\n" + harness
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(program)
        path = f.name
    try:
        result = subprocess.run(
            ["python3", path], capture_output=True, timeout=5, text=True,
        )
    except subprocess.TimeoutExpired:
        return 0.0
    finally:
        Path(path).unlink(missing_ok=True)
    if result.returncode == 0:
        return 1.0
    # Partial credit: count how many of the assertions ran before the first failure
    failed_at = result.stderr.count("AssertionError")
    passed = max(0, len(tests) - failed_at)
    return passed / len(tests)

batch = requests.post(f"{KILN}/v1/completions/batch", json={
    "prompts": [TASK["messages"]], "n": 8, "temperature": 0.9, "max_tokens": 192, "seed": 7,
}).json()

group = {"messages": TASK["messages"], "completions": []}
for item in batch["completions"]:
    group["completions"].append({"text": item["text"], "reward": reward(item["text"], TASK["tests"])})

requests.post(f"{KILN}/v1/train/grpo", json={
    "groups": [group],
    "config": {"lora_rank": 16, "output_name": "code-runs"},
}).raise_for_status()
````

For the trivial `add` task, a base Qwen3.5-4B already nails most rollouts.
The interesting regime is harder problems (string parsing, recursion, small
data-structure manipulation) where the base model fails 60–80% of the time
and GRPO has room to push the success rate up.

## Tuning knobs

`config` on `/v1/train/grpo` accepts the following — every field has a
server-side default, so omit anything you don't want to override:

- **`n` (in the batch request)** — group size. Defaults to 1; for GRPO use
  `>= 4`. 8 is the usual starting point; smaller groups have higher variance,
  larger groups eat the 64-completion batch cap faster.
- **`learning_rate`** — omit it: the server resolves the default per
  optimizer (Muon, the default: `2e-3` for GRPO; AdamW/SGD: legacy `1e-5`)
  and the train receipt records the resolved value. If you do pin it, halve
  on reward oscillation / KL spikes, double if reward improves but slowly —
  and mind that Muon's band is ~200x AdamW's.
- **`optimizer`** — defaults to Muon (momentum-orthogonalized SGD with fused
  on-device kernels). Select AdamW/SGD per request via
  `{"optimizer": {"kind": "adam_w"}}` / `{"kind": "sgd"}`.
- **`kl_coeff`** — defaults to `0.1`. Higher keeps the adapter closer to the
  base model (more conservative, slower). Lower lets the adapter drift faster
  but risks mode collapse onto whatever scored highest in early rounds.
- **`behavior_policy`** — defaults to `"no_importance_correction"`, which fixes
  the policy ratio at one. `"recorded"` enables correction from each sampled
  action token's exact behavior log-probability and rejects any completion with
  missing or mismatched provenance. The behavior distribution is never inferred
  from the base model or KL reference.
- **`kl_reference_policy`** — independently selects the frozen KL anchor.
  `{"kind": "base_per_step"}` is the default. Use `{"kind": "ema",
  "decay": 0.9, "refresh_every": 32}` for a moving frozen snapshot, or
  `{"kind": "none"}` only with `"kl_estimator": "none"` or `kl_coeff: 0`.
  The old `reference_policy` key is accepted only as an input alias; new
  requests and receipts should use `kl_reference_policy`.
- **`kl_estimator`** — `"k1"` by default; `"k3"` selects the non-negative
  estimator and `"none"` disables both the penalty and KL-reference forward.
- **`is_level`** — `"token"` (default) applies PPO clipping per action token.
  `"sequence"` selects GSPO: it forms one geometric-mean ratio per completion,
  broadcasts one sequence surrogate, and normalizes by completion length once.
  `"cispo"` uses a detached per-token importance weight with an upper cap and
  no lower floor. These modes share the same independently configured KL term.
- **`clip_epsilon`** / **`clip_eps_high`** — `0.2` / `null` by default. Token
  PPO and sequence GSPO use `[1 - clip_epsilon, 1 + clip_eps_high]`; a null
  upper value uses `clip_epsilon` symmetrically. These bounds have no effect on
  CISPO, or on the fixed-one ratio in no-correction mode.
- **`cispo_max_weight`** — defaults to `5.0` and is used only with
  `is_level: "cispo"`. This is an absolute cap (`min(ratio, 5.0)`), not an
  additive epsilon (`1 + 5.0`). Ratios below one retain their natural weight;
  Kiln does not impose the PPO lower floor. This matches MiniMax-M1 and TRL's
  CISPO loss semantics.
- **`lora_rank`** / **`lora_alpha`** — defaults `16` / `32`. The capacity of
  the adapter. Rank 8 is faster and still works for narrow tasks; rank 32+
  for broader behavioral shifts.
- **`base_adapter`** — continue training from a previously trained adapter
  instead of starting fresh from the base model.
- **`output_name`** — name the resulting adapter on disk (defaults to
  `grpo-<job_id_prefix>`).
- **`auto_load`** — defaults `true`. When the job completes, the new adapter
  is hot-swapped in immediately. Set `false` if you want to load it manually
  via `/v1/adapters` (e.g., for A/B testing). If `output_name` is already
  physically loaded, Kiln must reload that same name at its revision barrier
  even when this is `false`; use a new versioned name for a truly idle output.

For full schema details, see
[QUICKSTART.md §9.4](../QUICKSTART.md#94-grpo-rollout-generation).

## Audit the policy update

Every non-dry GRPO run that reaches the training loop writes a versioned audit
at `train_receipt.json` -> `grpo.policy_audit`. The same object appears as
`report.policy_audit` in the long-context training benchmark and the trainer
emits a compact `GRPO policy audit` structured log event when it writes the
receipt. Retrieve a published adapter's receipt with:

```bash
ADAPTER=math-grpo
curl -s http://localhost:8420/v1/adapters/$ADAPTER/receipt \
  | jq '.grpo.policy_audit'
```

The object has schema `kiln.grpo-policy-audit.v1` and keeps the two policy
comparisons separate:

- `importance_sampling` uses `exp(log p_policy - log p_behavior)`. Token PPO
  and CISPO report one ratio observation per action token; sequence/GSPO
  reports one per completion while retaining the total action-token count.
  `no_importance_correction` reports an exact ratio of `1.0` and does not
  borrow the KL reference as a denominator. For token PPO and GSPO,
  `below_clip_count` and `above_clip_count` describe the two-sided interval.
  CISPO has no lower bound, so its below count is always zero and its above
  count/fraction report only ratios beyond `cispo_max_weight`.
- `kl_reference` uses `log p_policy - log p_reference`. Its K1/K3 means are
  reported before multiplying by `kl_coeff`; `mean_masked_estimator` includes
  zeros for entropy-masked tokens and remains normalized over every observed
  action token, matching the loss contribution's denominator.
- `recorded_provenance` counts sampled and controller-forced actions and lists
  content-addressed behavior sources. A source binds the behavior model and
  adapter revision, tokenizer/template invocation, sampling controls, and
  generation backend. `behavior_source_manifest_sha256` is stable regardless
  of input order and changes when any source identity changes.

Counts are runtime observations, so multiple epochs count a completion each
time it is trained. The receipt intentionally stores aggregate metrics and
source identities rather than duplicating every per-token log-probability from
the rollout dataset. Dry runs and failures before the training loop omit the
audit instead of inventing zero-work policy evidence.

## What to expect at the wall clock

On a single A6000 with rank-8 LoRA, end-to-end timing for the loops above is
roughly:

- **Generate (8 prompts × 8 completions, 64 total)**: 1–3 s with continuous
  batching and chunked prefill.
- **Score (Python-side)**: depends entirely on your reward fn — sub-millisecond
  for math/JSON, 5–20 s for the code-runs example because each subprocess pays
  Python startup.
- **Train (one GRPO step over 64 completions)**: 5–15 s with gradient
  checkpointing on, hot-swap is atomic at iteration boundary.

Reward trajectories on these toy tasks:

- Math correctness: noticeable improvement in 5–10 rounds, saturation by ~20.
- JSON-validity: 15–30 rounds to saturate; format compliance is harder.
- Code-runs: 30+ rounds for non-trivial problems; the variance is higher and
  the reward signal sparser.

Watch live training progress with `GET /v1/train/status`.

## Troubleshooting

- **Reward isn't budging.** Check that the rewards within each group aren't all
  identical — GRPO normalizes within-group, so a group where every completion
  got `1.0` (or every one got `0.0`) contributes zero gradient. Either increase
  temperature to get more variance, or reshape the reward to be continuous (the
  0.5/1.0 split in example 2 is a worked instance).
- **Adapter looks worse, not better.** Most often `kl_coeff` is too low — the
  adapter is overfitting to whatever scored highest in the first few rounds.
  Try `0.2` or `0.5`.
- **Mock-mode error on `/v1/train/grpo`.** The server was started without real
  model weights (`--model-path`/`KILN_MODEL_PATH` unset, or the path didn't
  resolve). Training requires real weights; mock inference is fine for API
  smoke tests but not for training.
- **`adapter_revision_conflict`.** Another upload, delete, gate action, or
  publisher changed `output_name` while this job was preparing its result. The
  newer on-disk revision was preserved; start the next iteration from the
  current adapter and resubmit. A gated (`post_eval.min_accuracy`) same-name
  rewrite also returns this before GPU work when that adapter is physically
  loaded; unload it or choose a versioned `output_name`.
- **`behavior_policy=recorded` is rejected.** Do not switch to
  `no_importance_correction` merely to suppress the error for off-policy data.
  Regenerate the dataset with `kiln rollout-generate`; it requires exact token
  IDs, sampled-token behavior log-probabilities, behavior model/adapter
  identity, tokenizer/template hashes, effective sampling controls, seed, and
  backend provenance before it publishes output. Kiln also rejects provenance
  whose canonical prompt messages, scored payload, exact token sequence,
  action positions, or tokenizer identity drifted before training.

## See also

- [Quickstart §9.4](../QUICKSTART.md#94-grpo-rollout-generation) — rollout
  generation paths and API constraints; [Quickstart §9](../QUICKSTART.md#9-advanced-api-examples)
  has the full schema for `/v1/completions/batch` and `/v1/train/grpo`, plus
  the fastest path to run Kiln before trying GRPO.
- [README.md `## The GRPO Loop`](../README.md#the-grpo-loop) — the 30-second
  overview of why the generate → score → train loop exists.
- [Website Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html) — setup,
  model-loading, adapter, and API recovery steps when a command or request does
  not behave as expected.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — the algorithm.
