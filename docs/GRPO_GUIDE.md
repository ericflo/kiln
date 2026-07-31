# GRPO Training Guide

GRPO in Kiln is a generate → score → train → evaluate loop. You provide a
reward function; Kiln turns groups of scored completions into a LoRA update,
publishes the resulting adapter, and records enough evidence to audit the
policy change.

The generated [Training and Agent Control Plane API
Schema](../contracts/kiln-control-plane-v1.schema.json) owns request and
response fields. The generated [HTTP API
contract](../contracts/kiln-http-api-v1.openapi.json) owns routes, status codes,
and error shapes. This guide owns the workflow and the decisions around it.

## Before you start

Use the `experimental` serving profile for a controlled loop that generates,
trains, and evaluates in one process:

```bash
KILN_SERVER_SERVING_PROFILE=experimental kiln serve
```

The default `stable` profile allows inference but rejects training and live
adapter transitions. `maintenance` allows drained training and adapter changes
but rejects inference and evaluation. Therefore a
stable → maintenance → stable sequence cannot evaluate or serve the resulting
unmerged adapter: the last stable process cannot load it. Either use
`experimental` for the complete controlled loop, or train under `maintenance`
and start an `experimental` process to load and evaluate the result.

Serving profiles are GPU-ownership policies, not hardware selectors. They do
not contain device-name, vendor-ID, or model-specific route allowlists. See
[Serving Profiles](SERVING_PROFILES.md).

Before a real run, confirm:

- the server has real model weights and reports a healthy training backend;
- the reward function measures the behavior you actually want;
- each prompt can produce completions with different rewards;
- generation and held-out evaluation prompts are separate;
- untrusted model output is sandboxed before any reward function executes it.

## What GRPO updates

GRPO stands for Group Relative Policy Optimization. For one prompt, generate a
group of completions and assign each a numeric reward. Kiln converts the
within-group reward differences into advantages and updates the LoRA policy
without training a separate critic.

Four distinctions matter:

- **Reward signal:** a group whose rewards are all identical has no relative
  policy signal. Kiln's default dynamic-sampling filter skips such groups.
- **Behavior policy:** exact recorded rollout provenance enables importance
  correction. Without it, Kiln fixes the ratio at one explicitly.
- **KL reference:** the policy used for the KL penalty is configured
  separately. Kiln never pretends that the KL reference generated a rollout.
- **Environment tokens:** ECHO is enabled by default and can add an auxiliary
  environment-token objective when trajectories contain those tokens. See the
  [ECHO Guide](ECHO_GUIDE.md).

The default policy surrogate is token-level. Sequence-level GSPO and CISPO are
available as explicit alternatives; they change the importance-weighting
mechanics, not the reward function.

## The loop

```text
tasks + request template
          │
          ▼
generate completions ── exact behavior provenance
          │
          ▼
score every completion ── one numeric reward each
          │
          ▼
train one versioned adapter
          │
          ▼
run held-out evaluation ── promote, keep, or reject
          │
          └────────────── next round uses the accepted adapter
```

Training submission is asynchronous. A successful `POST /v1/train/grpo`
response means “queued,” not “trained.” Do not generate the next round until
the job reaches a terminal state and you have reviewed its adapter and
evaluation outcome.

Use a new output name for every round, such as `math-r001` and `math-r002`.
For round two and later, set `base_adapter` to the accepted adapter from the
previous round. Reusing one physically loaded output name creates avoidable
revision conflicts and makes rollback ambiguous.

## Choose the rollout contract

Kiln supports two explicit behavior-policy modes:

| Mode | Use it when | Policy ratio |
|---|---|---|
| `recorded` | Rollouts came from `kiln rollout-generate` and retain validated per-token provenance | Computed from policy and recorded behavior log-probabilities |
| `no_importance_correction` | You intentionally accept uncorrected data, such as hand-authored or text-only batches | Fixed at exactly 1 |

Prefer `recorded` for an iterative on-policy loop. Do not switch to
`no_importance_correction` merely to suppress a provenance error: that changes
the objective rather than fixing the dataset.

## Recommended recorded-policy workflow

This example uses one arithmetic task so the file shapes stay visible. A useful
training run needs a broader train set and a separate held-out suite.

### 1. Write tasks

`tasks.jsonl` contains one JSON object per line:

```json
{"id":"sum-1","prompt":"What is 47 + 138? Reply with just the number.","answer":185}
```

### 2. Write the request template

`request.json` is a chat-completions body. The CLI substitutes task fields and
owns `seed`, `adapter`, `n`, `stream`, and rollout-provenance controls.

```json
{
  "model": "Qwen3.5-4B",
  "messages": [{"role": "user", "content": "{{prompt}}"}],
  "temperature": 0.9,
  "max_tokens": 64
}
```

### 3. Write a scorer

The scorer receives the task, rendered request, full response, parsed content,
usage, adapter, seed, and latency as one JSON object on standard input. It
prints either one finite number or an object containing `reward`, `score`, or
`value`.

```python
#!/usr/bin/env python3
import json
import re
import sys

row = json.load(sys.stdin)
numbers = re.findall(r"-?\d+", row["content"])
correct = bool(numbers) and int(numbers[-1]) == row["task"]["answer"]
print(1.0 if correct else 0.0)
```

Make it executable:

```bash
chmod +x score_math.py
```

If a scorer runs code emitted by the model, execute that code in a dedicated
sandbox with bounded CPU, memory, wall time, filesystem access, syscalls, and
network access. A subprocess timeout alone is not a security boundary.

### 4. Generate scored rollouts

For the first round, force the base model:

```bash
kiln rollout-generate \
  --adapter base \
  --thinking false \
  --tasks tasks.jsonl \
  --seeds 8 \
  --seed-start 42 \
  --request-template request.json \
  --scorer ./score_math.py \
  --output math-r001.rollouts.jsonl \
  --summary-output math-r001.summary.json
```

Publication is atomic. The command validates every response's seed, adapter,
prompt and content hashes, token/action coverage, usage, tokenizer/template
identity, sampling controls, and backend provenance before replacing either
output. The training JSONL contains canonical groups and
`kiln.rollout-provenance.v1` records. Latency, token counts, raw scorer output,
and per-request summaries remain in the separate summary file.

Inspect the reward distribution before training:

```bash
jq '.stats | {mean_reward, min_reward, max_reward}' math-r001.summary.json
```

If every completion received the same reward, improve generation diversity or
the reward shape before spending a training step.

### 5. Submit recorded-policy training

The JSONL path is read by the server process, so it must be an absolute path
that the server can access:

```bash
rollouts_path="$(realpath math-r001.rollouts.jsonl)"
response="$(
  curl --fail-with-body -sS http://localhost:8420/v1/train/grpo \
    -H 'Content-Type: application/json' \
    -d "$(jq -n --arg path "$rollouts_path" '{
      dataset_path: $path,
      config: {
        behavior_policy: "recorded",
        output_name: "math-r001",
        auto_load: false,
        checkpoint_interval: 25
      }
    }')"
)"
job_id="$(jq -r '.job_id' <<<"$response")"
printf 'queued %s\n' "$job_id"
```

`auto_load: false` keeps training completion separate from promotion. Add a
held-out `post_eval` gate when automatic promotion is appropriate.

The convenience command `kiln train grpo --file FILE.jsonl` submits the
streamed route with the default `no_importance_correction` behavior policy.
For recorded-policy JSONL, use an HTTP request like the one above or a JSON
request file that explicitly sets `config.behavior_policy` to `recorded`.

### 6. Wait for a terminal state

```bash
while :; do
  detail="$(curl --fail-with-body -sS \
    "http://localhost:8420/v1/train/jobs/$job_id")"
  state="$(jq -r '.state' <<<"$detail")"
  printf '%s\n' "$state"
  case "$state" in
    completed) break ;;
    failed)
      jq '{state, error}' <<<"$detail"
      exit 1
      ;;
  esac
  sleep 2
done
```

Review `adapter_path`, `train_receipt`, `linked_eval_job_ids`, and the
promotion outcome in job detail. “Completed” proves that training published an
adapter; it does not by itself prove that the adapter improved.

### 7. Continue from the accepted adapter

After held-out evaluation accepts `math-r001`, generate the next rollouts with
that exact adapter:

```bash
kiln rollout-generate \
  --adapter math-r001 \
  --thinking false \
  --tasks tasks.jsonl \
  --seeds 8 \
  --seed-start 50 \
  --request-template request.json \
  --scorer ./score_math.py \
  --output math-r002.rollouts.jsonl \
  --summary-output math-r002.summary.json
```

Submit round two with a new output and the prior adapter as its starting
weights:

```json
{
  "dataset_path": "/absolute/path/math-r002.rollouts.jsonl",
  "config": {
    "behavior_policy": "recorded",
    "base_adapter": "math-r001",
    "output_name": "math-r002",
    "auto_load": false
  }
}
```

Never advance the loop merely because mean training reward rose. Compare the
new adapter against the previous accepted adapter on a held-out suite, then
promote, keep, or reject it deliberately.

## Group and source rules

Each GRPO group has one shared prompt and one or more scored completions:

```json
{
  "messages": [{"role": "user", "content": "Return valid JSON."}],
  "completions": [
    {"text": "{\"ok\":true}", "reward": 1.0},
    {"text": "ok", "reward": 0.0}
  ]
}
```

Use exactly one source per request:

- `groups` for an inline array;
- `dataset_path` for server-local canonical JSONL;
- `dataset` for a named uploaded dataset.

The streamed JSONL route performs bounded full-corpus admission and pins a
private read-only snapshot before queue publication. Named `grpo_groups`
datasets use the persisted `train` partition by default and accept an explicit
`dataset_split`. Held-out post-eval rejects content or declared source-group
overlap. See [Dataset Splits and Train/Eval
Separation](DATASET_SPLITS.md).

For recorded mode, do not hand-edit completions, prompts, sampling controls, or
provenance. Kiln replays the template invocation with its pinned tokenizer and
rejects drift before training.

## Tuning knobs

Start with defaults and change one decision at a time. The effective
configuration and receipt—not this summary—are authoritative for a run.

| Field | Default | What it changes |
|---|---:|---|
| `behavior_policy` | `no_importance_correction` | Whether the policy ratio uses recorded behavior probabilities |
| `kl_coeff` | `0.1` | Strength of the separately configured KL penalty |
| `kl_reference_policy` | `base_per_step` | Frozen policy used only for KL |
| `kl_estimator` | `k1` | KL estimator; `none` disables the reference forward when the coefficient is also zero |
| `is_level` | `token` | Token PPO, sequence GSPO, or CISPO importance weighting |
| `clip_epsilon` | `0.2` | Lower and, unless overridden, upper PPO/GSPO clipping width |
| `dynamic_sampling` | `true` | Skips groups with no relative reward signal |
| `lora_rank` / `lora_alpha` | `16` / `32` | Adapter capacity and scale |
| `optimizer` | Muon | Optimizer family; omission lets Kiln resolve its GRPO learning-rate default |
| `auto_load` | `true` | Whether a completed, canary-qualified adapter may become active |
| `shared_prefix_reference` | `true` | Reuses qualified prompt-side KL-reference state |
| `detect_anomaly` | `false` | Adds expensive per-operation NaN/Inf localization |

`clip_eps_high` makes PPO/GSPO clipping asymmetric. `cispo_max_weight` applies
only to CISPO and is an absolute upper cap, not an epsilon. `base_adapter`
loads weights from an earlier PEFT adapter but does not restore optimizer state;
use `resume_checkpoint` for exact continuation.

Vulkan currently uses the exact per-completion KL-reference fallback even when
`shared_prefix_reference` is true. That is a capability-derived fallback, not
a device-name exception.

## Observe and audit the update

Every non-dry run that enters the training loop writes
`train_receipt.json`. The `grpo.policy_audit` object has schema
`kiln.grpo-policy-audit.v1` and keeps three kinds of evidence separate:

- `importance_sampling` summarizes policy-versus-behavior ratios and clipping;
- `kl_reference` summarizes policy-versus-reference differences before
  multiplying by `kl_coeff`;
- `recorded_provenance` counts sampled and controller-forced actions and binds
  the behavior model, adapter revision, tokenizer/template invocation,
  sampling controls, and generation backend.

Retrieve a published adapter's receipt:

```bash
curl --fail-with-body -sS \
  http://localhost:8420/v1/adapters/math-r001/receipt \
  | jq '.grpo.policy_audit'
```

Also inspect:

- reward and dynamic-filter statistics;
- loss history and gradient diagnostics;
- `phase_timings.gpu_writer_wait_ms`, `gpu_writer_held_ms`, and
  `gpu_writer_acquisitions`;
- adapter smoke-test evidence;
- linked held-out evaluation results and promotion outcome.

Server-submitted GRPO acquires exclusive GPU ownership for setup, each complete
optimizer group, device snapshots, and final smoke/cleanup work. Tokenization,
reward filtering, JSONL reads, progress callbacks, encoding, and file
publication run outside the GPU writer. A large group can still create a long
inference pause; the phase timings distinguish writer contention from work
outside GPU ownership.

## Checkpoint and resume

Set `checkpoint_interval` to publish an immutable exact checkpoint every N
committed optimizer groups. Cooperative cancellation settles the current group
and publishes at the next safe boundary.

```bash
kiln train grpo \
  --file scored-groups.jsonl \
  --adapter math-r001 \
  --checkpoint-interval 25
```

Resume requires the identical source bytes, route, adapter name, and effective
configuration:

```bash
kiln train grpo \
  --file scored-groups.jsonl \
  --adapter math-r001 \
  --checkpoint-interval 25 \
  --resume-checkpoint math-r001-checkpoint-step-00000025.kiln-checkpoint
```

An exact checkpoint restores adapter and optimizer tensors, reference state,
cursor, RNG streams, loss and diagnostic history, and runtime planning. A PEFT
adapter is a warm-start or serving artifact, not a resume point. See [Native
Training Checkpoints](training-checkpoints.md).

## Promotion

Use versioned output names and compare the candidate with the previous accepted
adapter. With `auto_load: true` and no gate, Kiln may activate a completed
adapter after its serving canary passes. With a held-out `post_eval` accuracy
gate, activation is deferred and the previous adapter remains active until the
gate passes.

Training completion, evaluation success, and adapter activation are separate
states. Record all three. See the [Evaluation Guide](EVAL_GUIDE.md) for
comparison and promotion semantics.

## Troubleshooting

- **All groups were filtered.** Their rewards had no useful within-group
  contrast, or explicit reward-variance filters removed them. Inspect the
  receipt, then improve sampling diversity or reward shaping.
- **Recorded policy was rejected.** Regenerate with `kiln rollout-generate`.
  Do not weaken the objective to hide prompt, token, adapter, sampling, or
  provenance drift.
- **The next round does not build on the last one.** Set `base_adapter` to the
  previous accepted adapter and use a new `output_name`.
- **The adapter regressed.** Keep the previous adapter active, inspect the
  held-out comparison and policy audit, then change one reward or optimization
  decision at a time.
- **`adapter_revision_conflict`.** Another mutation changed the output name, or
  a gated same-name rewrite targeted physically loaded weights. Use a new
  versioned output name.
- **Training says real weights are unavailable.** Set `model.path` in TOML or
  `KILN_MODEL_PATH`, restart the server, and verify readiness before
  resubmitting.

## See also

- [ECHO Guide](ECHO_GUIDE.md)
- [Dataset Splits and Train/Eval Separation](DATASET_SPLITS.md)
- [Native Training Checkpoints](training-checkpoints.md)
- [Evaluation Guide](EVAL_GUIDE.md)
- [Quickstart](../QUICKSTART.md)
- [DeepSeekMath](https://arxiv.org/abs/2402.03300)
