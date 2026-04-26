# GRPO Training Guide

This guide walks through GRPO end-to-end on Kiln: what it is, the loop you run,
and three runnable verifiable-reward examples (math correctness, JSON-validity,
code-runs). Everything below assumes a Kiln server running on
`http://localhost:8420` with at least one adapter slot free and Python 3.10+
on the client.

## What GRPO is, in 5 sentences

GRPO is **Group Relative Policy Optimization**, the reinforcement-learning
algorithm introduced by [DeepSeekMath](https://arxiv.org/abs/2402.03300). For
each prompt you generate a *group* of `n` completions, score them with a
reward function you write, and turn those rewards into a policy-gradient
update — no separate critic network and no replay buffer, which is what makes
it suit Kiln's single-process design. Within each group the rewards are
mean-zeroed and (optionally) normalized to produce per-completion *advantages*,
so the update only depends on which completions in the group beat the others.
A clipped importance-sampling ratio keeps the step honest when the in-flight
adapter has drifted from the policy that generated the rollouts, and a KL
penalty toward the base model keeps the adapter from collapsing onto a single
high-reward output. **You write the reward function. That's the whole point.**

## The loop

Every GRPO iteration is the same four-step cycle. On Kiln this is one process,
two HTTP endpoints, and an atomic adapter hot-swap:

```
   [ generate ]    POST /v1/completions/batch
        │            (1 request → prompts × n completions)
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
[QUICKSTART.md §9](../QUICKSTART.md#9-phase-8-api-examples). The fields used in
this guide are:

**`POST /v1/completions/batch`** — issues `prompts.len() × n` completions in
one HTTP round-trip. The iteration-level scheduler batches the underlying
prefill/decode steps, so this is meaningfully cheaper than firing N parallel
calls. Hard cap: `prompts.len() * n <= 64`. Use a fixed `seed` to make
rollouts reproducible — Kiln derives a per-completion seed of
`seed.wrapping_add(prompt_idx * n + completion_idx)` so identical prompts in
the same batch still produce distinct outputs at `temperature > 0`. Streaming
is not supported on this endpoint.

**`POST /v1/train/grpo`** — accepts `groups`, where each group is
`{"messages": [...], "completions": [{"text": "...", "reward": 0.0}, ...]}`.
The request is enqueued and returns a `job_id` immediately; training runs on a
background thread, and the resulting adapter is auto-loaded (atomically, at an
iteration boundary) when it lands. Every `config` field has a server default,
so `{"groups": [...]}` is a valid minimal payload.

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
        "learning_rate": 1e-5,
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
    "config": {"learning_rate": 1e-5, "lora_rank": 16, "output_name": "json-format"},
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

```python
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
    "config": {"learning_rate": 1e-5, "lora_rank": 16, "output_name": "code-runs"},
}).raise_for_status()
```

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
- **`learning_rate`** — defaults to `1e-5`. Halve it if you see reward
  oscillate or KL spike; double it if reward improves but slowly.
- **`kl_coeff`** — defaults to `0.1`. Higher keeps the adapter closer to the
  base model (more conservative, slower). Lower lets the adapter drift faster
  but risks mode collapse onto whatever scored highest in early rounds.
- **`clip_epsilon`** — defaults to `0.2`. The PPO/GRPO clip range on the
  importance-sampling ratio. Leave alone unless you have a specific reason.
- **`lora_rank`** / **`lora_alpha`** — defaults `16` / `32`. The capacity of
  the adapter. Rank 8 is faster and still works for narrow tasks; rank 32+
  for broader behavioral shifts.
- **`base_adapter`** — continue training from a previously trained adapter
  instead of starting fresh from the base model.
- **`output_name`** — name the resulting adapter on disk (defaults to
  `grpo-<job_id_prefix>`).
- **`auto_load`** — defaults `true`. When the job completes, the new adapter
  is hot-swapped in immediately. Set `false` if you want to load it manually
  via `/v1/adapters` (e.g., for A/B testing).

For full schema details, see
[QUICKSTART.md §9.1](../QUICKSTART.md#91-batch-generation-efficient-for-grpo-rollouts).

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

## See also

- [QUICKSTART.md §9](../QUICKSTART.md#9-phase-8-api-examples) — full schema for
  `/v1/completions/batch` and `/v1/train/grpo`.
- [README.md `## The GRPO Loop`](../README.md#the-grpo-loop) — the 30-second
  pitch.
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — the algorithm.
