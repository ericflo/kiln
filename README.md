<p align="center">
  <img src="assets/logo.png" alt="Kiln" width="200">
</p>

<h1 align="center">Kiln</h1>

<p align="center">
  <strong>Your model gets better every time you use it.</strong><br>
  A single-GPU inference server with live LoRA training and a first-class eval loop. Pure Rust. Single binary.
</p>

<p align="center">
  <a href="https://ericflo.github.io/kiln/">Website</a> &middot;
  <a href="https://ericflo.github.io/kiln/docs/">Documentation</a> &middot;
  <a href="https://ericflo.github.io/kiln/demo/">Demo</a> &middot;
  <a href="QUICKSTART.md">Quickstart</a> &middot;
  <a href="https://ericflo.github.io/kiln/cli.html">CLI Guide</a> &middot;
  <a href="docs/guides/OPENENV_GUIDE.md">OpenEnv Guide</a> &middot;
  <a href="https://ericflo.github.io/kiln/grpo.html">GRPO Guide</a> &middot;
  <a href="docs/guides/EVAL_GUIDE.md">Eval Guide</a> &middot;
  <a href="https://ericflo.github.io/kiln/api.html">API Reference</a> &middot;
  <a href="https://ericflo.github.io/kiln/troubleshooting.html">Troubleshooting</a> &middot;
  <a href="ARCHITECTURE.md">Architecture</a> &middot;
  <a href="BENCHMARKS.md">Benchmarks</a> &middot;
  <a href="docs/contracts/CONFIGURATION.md">Configuration</a> &middot;
  <a href="CHANGELOG.md">Changelog</a> &middot;
  <a href="CONTRIBUTING.md">Contributing</a> &middot;
  <a href="LICENSE">License</a>
</p>

---

Kiln serves a language model, trains it, and evaluates it on one GPU from one Rust binary. The default `stable` profile is the complete product: inference, training, evaluation, adapter transitions, and correctness-qualified acceleration work together without profile tuning. Use `maintenance` only when inference must be drained, and `experimental` only to qualify a quarantined backend route. See [Serving Profiles](docs/serving/SERVING_PROFILES.md).

It targets one model ([Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)) and optimizes everything for that model — the scheduler, the memory manager, the kernels. This isn't a general-purpose framework. It's a scalpel.

Kiln's current measured service envelope is correctness-first local work at
external concurrency 1 through 8 with relaxed latency requirements. It makes no
current performance-parity claim against vLLM; use vLLM for serving-only
capacity, strict latency SLOs, or concurrency 8 and above. The current ROCm and
Vulkan development receipts, exact limitations, and historical counterevidence
are published in [Benchmarks](BENCHMARKS.md#current-measured-service-envelope).

## Why

Today, improving a deployed model looks like: collect failure examples, format them, upload to a training service, wait hours, download new weights, build a separate eval harness in Python, redeploy, hope. Kiln collapses that into one local binary and artifact set. Start the normal server:

```bash
KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve

# Submit a correction — the model learns it in seconds
curl http://localhost:8420/v1/train/sft \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"messages": [
        {"role": "user", "content": "Summarize this contract clause..."},
        {"role": "assistant", "content": "The clause establishes..."}
      ]}
    ],
    "config": {"training_profile": "native_online_lora_v1"}
  }'

# The next request already uses the updated weights
curl http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Summarize this contract clause..."}]}'

# Did the new adapter actually get better? Run an eval suite against it
curl http://localhost:8420/v1/eval/run \
  -H "Content-Type: application/json" \
  -d '{"suite": "contract-summaries", "adapter": "active"}'
# → Returns a job_id; drill into per-example outcomes at /ui/ or via
#   GET /v1/eval/jobs/{job_id}. Suites can be hand-authored, synthesized
#   from an uploaded SFT dataset (POST /v1/eval/datasets/.../synthesize),
#   or built up from your A/B picks in the Judgments tab.
```

A 4B model continuously tuned to your specific workload — and continuously *measured* against the prompts you actually care about — will outperform a generic 70B model on those tasks. And it runs on hardware you already own.

## Features

- **OpenAI-compatible API** — drop in as a local replacement. SSE streaming, chat completions, tool use formatting, and first-class thinking budgets by token count or decode time.
- **pi integration** — `kiln pi-setup` backs up and merges pi's `models.json` + `settings.json`, then points pi at Kiln as an OpenAI-compatible tool-calling backend.
- **Embedded agent runs** — the server drives pi itself (`POST /v1/agent/runs`), streams the trajectory live with steer/abort, and auto-indexes finished sessions into the trace layer the flywheel trains on.
- **Bounded online-LoRA SFT** over HTTP under the fixed `native_online_lora_v1` profile, with atomic publication and live hot-swap under normal `stable` serving or drained `maintenance`.
- **GRPO training** over HTTP — submit scored completions and you control the reward function; multi-turn agentic rollouts are ECHO-by-default (λ=0.05 env-CE term, `--no-echo` opt-out).
- **Native OpenEnv training** — discover OpenEnv servers, run seed-matched stateful episodes, and collect or directly train a LoRA with `kiln openenv`.
- **On-policy distillation (OPD)** over HTTP against an identity-bound local or vLLM teacher, with exact candidate-boundary checkpoints and resume.
- **First-class evals** over HTTP — register suites, run them against any adapter, drill into per-example outcomes, strictly replay a completed seeded run, and turn A/B picks into a local judge LoRA.
- **Post-training auto-eval** — attach a held-out `post_eval` to SFT/GRPO and Kiln rejects contamination before queuing.
- **Capability-bound optimizers** — Muon is the default, with AdamW/SGD admitted only where the exact resident tuple and workload support them; `GET /v1/config` field `training.optimizer_support` reports implementation, admission, and memory checks.
- **Continuous batching** with token-budgeted prefill and a typed tiled-prefill policy that resolves once at startup with provenance.
- **128K+ context on 24GB** — Qwen3.5-4B's hybrid architecture keeps KV cache 4x smaller than a pure transformer; paged KV cache, optional FP8 KV, prefix caching, and gradient checkpointing.
- **Adapter management and composition** — load, unload, upload, download, version, stack, and merge LoRAs; `/ui/` shows provenance, training history, and eval scores.
- **Embedded web dashboard** at `/ui/` — live server status, training queue with loss curves, the full eval workflow, an A/B playground, and a `⌘K` command palette. No extra service to run.
- **Prometheus metrics** at `/metrics`, plus a durable size-rotated request log you can mine into SFT data or an eval suite — see [docs/guides/EVAL_GUIDE.md § Mine your own request log](docs/guides/EVAL_GUIDE.md#mine-your-own-request-log).
- **Training webhooks** — POST a JSON event to a configured URL on training job completion or failure.
- **Pure Rust** — single binary, single process. No Python. No sidecar. No second model in memory.

## The OpenEnv Loop

Interactive reinforcement-learning environments are a native Kiln data source,
not an external preprocessing step. Point `kiln openenv` at one or more OpenEnv
servers and Kiln owns discovery, deterministic task grouping, policy action
generation, seed-matched stateful episode sessions, reward aggregation,
canonical agentic trajectories, ECHO masks, GRPO submission, and audit
artifacts — with held-out environment evaluation and promotion gates after
training. The runtime is implementation-neutral: miniopenenv is a pinned CI
oracle, not a dependency, configuration namespace, or special execution path.

- [docs/guides/OPENENV_GUIDE.md](docs/guides/OPENENV_GUIDE.md) — multi-environment training, task catalogs, protected credentials, ECHO behavior, artifacts, and troubleshooting.
- [docs/training/OPENENV_REPLAY_REFERENCE.md](docs/training/OPENENV_REPLAY_REFERENCE.md) — replay and verify the exact environment exchanges a run used (512 MiB collection budget, 256 MiB artifact caps).

## The GRPO Loop

This is the killer feature: generate completions, score them with your own
reward function, and feed the results back — the model learns what "good"
means for your use case. The default `POST /v1/train/grpo` submission uses
`behavior_policy="no_importance_correction"`; for drift-corrected training,
`kiln rollout-generate` requests exact token/action provenance from the real
batching path, validates the seed, adapter, prompt, scored content, and usage
before scoring, and atomically writes JSONL for `behavior_policy="recorded"`.

See [docs/guides/GRPO_GUIDE.md](docs/guides/GRPO_GUIDE.md) for the loop, the recorded-policy workflow, and a worked verifiable-reward example (math).

### Agentic GRPO with ECHO (multi-turn rollouts)

For multi-turn agentic rollouts — tool calls, command output, file contents — kiln's GRPO is **ECHO-by-default**: the trajectory format carries environment segments end-to-end, from rollout JSONL through tokenization and masking to the training receipt, and the env-CE term **trains by default at `λ = 0.05`** on any trajectory with observation segments. ECHO adds a length-normalized cross-entropy loss on environment-observation tokens to the standard policy-gradient loss, reusing the rollout's forward pass — the env tokens are already in the rollout context. Legacy single-turn rollouts contribute exactly zero; opt out per job with `--no-echo` or `loss.echo: null`.

```python
import requests

# Canonical multi-turn shape (kiln >= 0.3):
requests.post("http://localhost:8420/v1/train/agentic", json={
    "agentic_groups": [{
        "messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}],
        "rollouts": [{
            "text": "<TURN_BREAK>-joined action text",
            "reward": 1.0,
            "trajectory": [
                {"role": "assistant", "content": "<tool_call>...</tool_call>", "kind": "action"},
                {"role": "tool", "content": "command output here", "kind": "observation"},
                {"role": "assistant", "content": "...", "kind": "action"}
            ]
        }]
    }],
    # ECHO env-CE applies by default (λ = 0.05) to the observation
    # segments above — no config needed. To tune it explicitly:
    #   "config": {"loss": {"echo": {"lambda": 0.02,
    #       "env_mask_mode": "env_only", "warning_filter": true}}}
    # or disable it with "loss": {"echo": null}.
    "config": {}
})
```

See [docs/guides/ECHO_GUIDE.md](docs/guides/ECHO_GUIDE.md) for the objective, trajectory contract, full usage (CLI flags, verifier-free adaptation per paper §5.5), the paper citation and its TerminalBench-2.0 headline, and the receipt-grade diagnostics.

### Off-policy OPD teacher data

For saturated tasks where reward-only GRPO is low signal, kiln-train also
accepts off-policy teacher distillation data: prompt messages plus a teacher
response, optionally with per-token teacher top-logprobs for reverse-KL.
OPD publishes immutable exact `.kiln-checkpoint` bundles every 25 committed
optimizer steps by default. See [docs/training/OPD_TEACHER_JSONL.md](docs/training/OPD_TEACHER_JSONL.md) for the JSONL schema and the `reverse_kl` vs `cross_entropy` objective contract, and [Native Training Checkpoints](docs/training/training-checkpoints.md#opd) for resume semantics.

### Remote vLLM teachers

Remote numeric prompt-logprobs are accepted only from an identity-aware vLLM
process launched with [`scripts/vllm_teacher.py`](scripts/vllm_teacher.py):
the launcher snapshots the model and optional static adapter into
`system_fingerprint`, re-hashes the runtime in a fresh child before spawn, and
can bound every provenance content read with
`--max-provenance-read-mib-per-second`. Dynamic LoRA mutation is disabled and stock `vllm-*` fingerprints are rejected; registration probes K=1 and the advertised maximum K against the loaded student.
See [docs/contracts/VLLM_TEACHER_IDENTITY.md](docs/contracts/VLLM_TEACHER_IDENTITY.md) for the full identity, registration, and recovery contract.

## The Eval Loop

Training is half the story; the other half is knowing whether your last
training run actually helped. Under the default `stable` profile, Kiln's eval
system runs in the same process against the same model weights — registered
suites, drillable per-example outcomes, A/B comparisons, and a judgment
flywheel that turns your A/B picks into a local judge LoRA, with strict byte
replay of completed runs. See [docs/guides/EVAL_GUIDE.md](docs/guides/EVAL_GUIDE.md) for the full scorer reference, dataset synthesis strategies, the strict replay contract, and the judge-LoRA workflow.

## Quick Start

**Supported hardware:** NVIDIA GPU with 24GB+ VRAM and CUDA 12+, AMD GPU with ROCm/HIP 7.2.4+ on Linux, AMD/Intel GPU with Vulkan 1.2+ on Linux, **or** Apple Silicon Mac with 16GB+ unified memory. Kiln targets `Qwen/Qwen3.5-4B` and needs about 20GB of free disk for the server, model weights, and adapters.

**Path 1 — Desktop App (recommended):** Install [Kiln Desktop](#desktop-app) on Windows, Linux, or macOS. The app downloads and verifies the matching prebuilt `kiln` server binary on first launch, then walks you through choosing or downloading `Qwen/Qwen3.5-4B`. No Rust toolchain, CUDA toolkit, or source build is required for this path. → [Desktop App quick path](QUICKSTART.md#quick-path-desktop-app-recommended-for-most-users)

**Path 2 — Server binary (terminal-first, no source build):** Download the latest `kiln-v*` server artifact when you want the `kiln` server in your terminal with no source build, Desktop App, or Docker — Linux x86_64 + NVIDIA CUDA 12.4 is the compact path (run the `Qwen/Qwen3.5-4B` weights step above first):

```bash
KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest | sed -n 's/.*"tag_name": "kiln-v\([^"]*\)".*/\1/p')
curl -L -o kiln-linux-cuda.tar.gz "https://github.com/ericflo/kiln/releases/download/kiln-v${KILN_VERSION}/kiln-${KILN_VERSION}-x86_64-unknown-linux-gnu-cuda124.tar.gz" && tar -xzf kiln-linux-cuda.tar.gz
KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve
```

Full release artifact matrix (Linux ROCm, Linux Vulkan, macOS Metal, Windows CUDA, SHA-256 sidecar checks): [QUICKSTART.md](QUICKSTART.md#quick-path-server-binary-terminal-first-no-source-build).

**Path 3 — Container:** Run the prebuilt `ghcr.io/ericflo/kiln-server:latest` image (NVIDIA Container Toolkit required) with `./Qwen3.5-4B` mounted read-only at `/models/Qwen3.5-4B`. → [Running with Docker](QUICKSTART.md#running-with-docker)

**Path 4 — Source / CLI:** Install Rust stable, then `cargo build --release` with `cuda`, `rocm`, `vulkan`, or `metal`. → [Build Kiln](QUICKSTART.md#1-optional-source-cli-branch-build-kiln)

**Get the model weights** (Paths 2–4 share this step; the Desktop App handles it automatically):

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
```


```
  ┌─────────────────────────────────────┐
  │           🔥 K I L N 🔥             │
  │   inference · training · adapters   │
  └─────────────────────────────────────┘

  Version: <workspace version>
  Mode:    GPU inference
  Model:   ./Qwen3.5-4B
  CUDA:    available ✓
  GPU:     NVIDIA RTX A6000
  VRAM:    49140 MiB total, 48891 MiB free
  Listen:  http://127.0.0.1:8420

  Endpoints: /ui/, /v1/chat/completions, /v1/train/sft, /health, /metrics
```

**Verify the server is up.** Run `kiln health` (binary at `./kiln` for Path 2/3 or `./target/release/kiln` for Path 4) before sending real requests — it prints a readable tree with model, scheduler, training, and GPU status, and exits non-zero if anything is wrong. See [troubleshooting](https://ericflo.github.io/kiln/troubleshooting.html#start-with-three-probes) for the full three-probe sequence.

```bash
curl http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The default server is `stable` and admits training, so submit directly:

```bash
curl http://localhost:8420/v1/train/sft \
  -H "Content-Type: application/json" \
  -d '{"examples": [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey there!"}]}], "config": {"training_profile": "native_online_lora_v1"}}'

# Check training
curl http://localhost:8420/v1/train/status
```

Every accepted SFT, GRPO, OPD, or distillation submission materializes one immutable
effective seed (exact decimal string, so all 64 bits survive browser clients) before the job
enters the queue; set `config.seed` or omit it, and exact resume inherits the checkpoint's
LoRA-initialization seed — a repeatable initial condition, not byte-identical output. See the [API reference's training section](https://ericflo.github.io/kiln/api.html#training).

The training deep-dive lives in the canonical docs:

- [Native SFT Profile](docs/training/NATIVE_SFT_PROFILE.md) — the fixed `native_online_lora_v1` microtrainer profile (one conversation, one optimizer update at a time, constant learning rate, no gradient accumulation, warmup, decay, or clipping; unsupported general-trainer fields fail closed), the optimizer tagged objects (omission selects Muon), the precision and optimizer-state matrix, and the rank ceilings (1024 for AdamW/SGD on canonical Qwen3.5-4B; 48 on CUDA/ROCm, 32 on Metal/Vulkan for Muon).
- [SFT Tokenization and Assistant-Only Loss](docs/training/sft-tokenization.md) — the source-pinned Qwen/Hugging Face token-and-label oracle: `add_generation_prompt=false`, complete assistant turn bodies plus their terminator supervised, assistant role headers and all system, user, and tool-response turns masked.
- [HF/TRL Interoperability](docs/training/HF_TRL_INTEROP.md) — the portable `export-sft` / `export-grpo` / `import-peft` bundles with the pinned correctness runner, the versioned identity/envelope/receipt model.
- [SFT Ingestion, Invalid Rows, and Row Identity](docs/training/sft-ingestion.md) — one row-admission contract across every source; `invalid_row_policy: fail|skip` semantics and receipt-recorded kept/rejected hashes.
- [Site API reference — training](https://ericflo.github.io/kiln/api.html#training) — the running contract, including `GET /v1/config` and `GET /v1/recipes`.

**Optional: use Kiln as pi's local agent model.** With the server running on
`localhost:8420`, `kiln pi-setup` adds a `kiln-local` provider to pi without
deleting your other providers or settings (existing files backed up first as
`models.json.bak-<timestamp>` / `settings.json.bak-<timestamp>`); Kiln accepts
Qwen3.5's native XML tool-call generations internally, but OpenAI-compatible
clients receive normal `tool_calls` in both streaming and non-streaming responses.
See [QUICKSTART.md §4](QUICKSTART.md#optional-point-pi-at-kiln) for the exact commands and the remote `--kiln-url` form:

```bash
./kiln pi-setup                                    # local server at localhost:8420
./kiln pi-setup --kiln-url http://office-kiln:8420  # remote kiln server
```

**Embedded agent runs: kiln drives pi itself.** The server can spawn `pi --mode rpc`
as a managed child, hand it a task, stream the trajectory live, and merge the
finished session straight into the agent-trace layer the self-improvement
flywheel reads — on-policy rollouts on demand. `POST /v1/agent/runs` plus the
events/steer/abort surface, the FIFO queue (`[agent].max_concurrent_runs` default
2, `run_timeout_secs` default 900), and the loopback-only default live in the
[API reference's Embedded agent runs section](https://ericflo.github.io/kiln/api.html#embedded-agent-runs); the
dashboard's **Distill → Agent runs** tab provides a launch form and a live trajectory view.

See [QUICKSTART.md](QUICKSTART.md) for the full walkthrough including Desktop App setup, source builds, GRPO, adapter management, Docker, and systemd setup. If setup stalls on binary downloads, CUDA/ROCm/Vulkan/Metal, model paths, `/health`, mock mode, training endpoints, or adapter directories, start with the [Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html). For tools-bearing workloads on older pinned releases, see [QUICKSTART.md §9.2](QUICKSTART.md#92-troubleshooting-older-release-long-prefill-timeouts) for the legacy `workers=1` / request-timeout troubleshooting note ([#664](https://github.com/ericflo/kiln/issues/664)).

## See it in action

Six short asciicasts captured on a single A6000 against `Qwen3.5-4B` show the main developer flows: first token from cold start, benchmark output, LoRA hot-swap, an OpenAI-compatible client, GRPO with a custom reward, and the full SFT online-learning loop. Watch them in the embedded player at **[ericflo.github.io/kiln/demo/](https://ericflo.github.io/kiln/demo/)** or browse the recording scripts and reference shell drivers under [`docs/site/demo/`](docs/site/demo/).

The kiln server also ships an embedded web dashboard at `http://localhost:8420/ui/` with live decode tok/s, p50/p99 ITL, VRAM breakdown, adapter management, training monitoring, and a chat playground — no extra service to run.

Here is the embedded server dashboard running with healthy status, active adapters, training progress, and the chat playground in one view:

![Kiln embedded server dashboard showing healthy status metrics, adapters panel, training progress, and chat quick-inference panels](docs/site/assets/server-ui-dashboard.png)

## Memory Budget (24GB GPU)

Qwen3.5-4B's hybrid KV architecture (only 8 of 32 layers cache) is what keeps 24 GB in reach; the per-scenario VRAM and Apple-Silicon unified-memory sizing tables (128K → ~13 GB, FP8 → ~19 GB) live in [BENCHMARKS.md](BENCHMARKS.md#memory-budget-24gb-gpu).

## API

The table below is a curated workflow index. The generated
[HTTP operation contract](contracts/kiln-http-api-v1.openapi.json) is the
complete route, transport, status, header, and error inventory; field-level
wire semantics live in the generated
[inference](contracts/kiln-inference-v1.schema.json),
[observability](contracts/kiln-observability-v1.schema.json),
[artifact lifecycle](contracts/kiln-artifacts-v1.schema.json),
[eval and judgment](contracts/kiln-evals-v1.schema.json), and
[training and agent control plane](contracts/kiln-control-plane-v1.schema.json)
schemas, and the [site's API reference](docs/site/api.html) walks through the
main endpoints.

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible), including per-request thinking budgets, bounded `ignore_eos`, and opt-in exact single-choice `rollout_provenance` |
| POST | `/v1/completions` | vLLM-shaped prompt-logprob subset with a canonical base-teacher identity fingerprint |
| POST | `/v1/completions/batch` | Text-only batch generation (up to 64 prompts per request), with the same thinking-budget and bounded `ignore_eos` controls |
| POST | `/v1/train/sft` | Submit bounded `native_online_lora_v1` SFT examples and return the exact effective seed (optionally with a `post_eval` hook) |
| POST | `/v1/train/grpo` | Submit GRPO scored completions; supports the `agentic_groups` shape with multi-turn `trajectory` fields and the default ECHO env-CE term |
| POST | `/v1/train/opd` | Submit on-policy or off-policy distillation against a registered, identity-bound teacher, with exact checkpoints every 25 committed optimizer steps |
| GET | `/v1/train/status/{job_id}` | Inspect one training job and its exact effective seed |
| GET | `/v1/adapters` | List saved/available LoRA adapters, identify the active adapter, and report the exact loaded name/content revision |
| POST | `/v1/eval/run` | Submit an eval and materialize one effective seed (registered suite or inline) |
| GET | `/ui/` | Embedded web dashboard (Overview / Adapters / Training / Evals / Playground) |

### Performance timing

Set `include_performance: true` on `POST /v1/chat/completions` to receive
`metadata.performance`; batching-engine responses separate the actor's
`actor_queue_ms`, `actor_admission_ms`, and `actor_prefill_wall_ms` phases
from model prefill/decode/TTFT totals. The request-local ITL percentiles,
bounded stall attribution, exact timing boundaries, rolling endpoint, and
Prometheus contract are documented in [`docs/serving/LATENCY_OBSERVABILITY.md`](docs/serving/LATENCY_OBSERVABILITY.md).

### Prompt logprobs

`POST /v1/completions` is a deliberately bounded vLLM-shaped scoring subset:
`max_tokens` `0` or `1`, no streaming, `prompt_logprobs` from `0` through
`256`, prompts (text or token IDs) capped at the smaller of 4096 tokens and
the served model's context window. The first result is `null`; position `i`
scores the observed prompt token at `i` from logits row `i-1` and returns that
token plus the requested top K (K entries when the observed token is already
top K, K+1 otherwise; K=0 returns only the observed token). The response is
additionally capped at 65,536 candidate entries: real scoring uses the
runner's resident backend and inference recurrent-state policy, omits the
unused final logits row, and takes exclusive GPU admission so it cannot
multiply vocabulary scratch allocations alongside generation or training.

Settlement is part of the contract: the scorer settles backend work before
reusing each projection chunk or releasing admission; a failed or panicked
settlement quarantines the backend and retains ownership rather than admitting
work against an unknown GPU state. Quarantine is process-wide and irreversible
— `/health` becomes `503 degraded` with `backend_runtime.restart_required=true`,
new completion/eval/adapter/training work rejects with HTTP 503
`backend_quarantined`, and queued jobs fail; restart the process, there is no
unsafe reset endpoint.

The [site API reference](https://ericflo.github.io/kiln/api.html#prompt-logprobs)
documents the full result shape, the CUDA/ROCm O(TK) vs O(TV) fallback routes,
the teacher identity binding, and the settlement/quarantine boundary; the
generated [training and agent control plane schema](contracts/kiln-control-plane-v1.schema.json)
owns the wire semantics.


### Chat adapter selection

`POST /v1/chat/completions` treats adapter selection as a per-request choice:
an omitted `adapter` uses the server default, `null` or `""` uses the base
model for that request only, and a name targets that loaded adapter for that
request. Only `POST /v1/adapters/load` and `/unload` change the server
default; a successful load returns `content_revision`, a canonical SHA-256
over the exact PEFT config and safetensor identities, published in the
`x-kiln-loaded-adapter-revision` response header and copied into prefix-cache
and response-cache keys so same-name rewrites cannot reuse stale entries.
See the [HTTP operation contract](contracts/kiln-http-api-v1.openapi.json) and the [site's API reference](docs/site/api.html) for full load/reload/quarantine and composition semantics.

### Adapter publication and conflicts

All adapter-changing operations share one serialized revision barrier: uploads
and merges stage dot-prefixed directories and become visible with one rename,
delete and rewrite operations refuse to touch bytes the runner still has
physically loaded, and a failed post-eval gate renames the rejected adapter to
`.failed`. Training that publishes into a name another publisher changed fails
with `adapter_revision_conflict` and preserves the newer revision.

- [docs/contracts/ADAPTER_MANIFEST.md](docs/contracts/ADAPTER_MANIFEST.md) — the adapter manifest schema (adapter/config/receipt hashes, base-weight shard manifest, kiln commit, training-data hash) and `kiln adapters restore` verification.
- [docs/training/training-checkpoints.md](docs/training/training-checkpoints.md) — exact `.kiln-checkpoint` bundles, resume semantics, and the planning identity shared by SFT/GRPO/OPD.
- [docs/contracts/REPLAY_INTEGRITY.md](docs/contracts/REPLAY_INTEGRITY.md) — the narrower `kiln-replay` request-lineage boundary.

## Architecture

Everything runs in one Rust binary: an axum HTTP front, a continuous-batching
scheduler with paged KV cache, the Qwen3.5-4B hybrid engine (24 Gated DeltaNet
layers + 8 full-attention layers), and background training and eval workers
that share the already-loaded model — no second copy in VRAM, no Python
sidecar. See [ARCHITECTURE.md](ARCHITECTURE.md) for the full deep-dive.

## Project Structure

The 33-crate workspace roll-call (verified against root `Cargo.toml` members) is in [ARCHITECTURE.md](ARCHITECTURE.md#workspace-layout).

## Configuration

Kiln uses a typed TOML config file; unknown fields and malformed environment
overrides stop startup instead of silently falling back to defaults.

- Complete field reference — every field, default, validation rule, and
  override: [docs/contracts/CONFIGURATION.md](docs/contracts/CONFIGURATION.md)
- Deployable starting point: [`kiln.example.toml`](kiln.example.toml)
- A running server publishes its immutable startup snapshot in
  `GET /v1/config` under `effective_configuration`; before startup,
  `kiln config --file kiln.toml --json` emits every post-precedence typed leaf
  with source, environment spellings, redaction, hash, and restart semantics.

Related contracts the configuration feeds: [serving profiles](docs/serving/SERVING_PROFILES.md), the [thinking-budget contract](docs/serving/THINKING_BUDGET_CONTRACT.md), and the versioned [runtime-defaults contract](contracts/runtime-defaults-v1.json) shared by the server, CLI, and desktop.

## Security model

Kiln has no built-in auth: the default listen address is `127.0.0.1:8420`, so a
fresh install isn't reachable from the network — to accept remote connections,
front kiln with a reverse proxy (nginx, Caddy) that adds auth, or run it on a
private network (WireGuard, Tailscale). **Training data is privileged**: Kiln
applies a faithful gradient update to anything you POST to `/v1/train/sft` or
`/v1/train/grpo` (it validates structure, not semantics), so review training
examples like code; bad adapters revert via `/v1/adapters/unload` + delete.

Full v0.1 threat model and per-finding analysis: [`docs/audits/security-audit-v0.1.md`](docs/audits/security-audit-v0.1.md).

## Desktop App

Kiln Desktop is a system-tray app that wraps the `kiln` server for people who
don't want to use a CLI: it spawns and supervises the `kiln` binary, shows
server status in the tray, and opens a dashboard, settings, and log viewer in
native windows. Windows drives the CUDA build; Linux chooses CUDA on NVIDIA
systems and Vulkan on AMD/Intel; macOS drives the native Metal build on
M-series hardware (Intel Macs are not supported). The installer bundles the
wrapper only and auto-downloads the matching prebuilt `kiln` binary with SHA-256
verification from the latest `kiln-v*` release.

Full release notes, installer matrix, screenshots, and build/dev docs: [desktop/README.md](desktop/README.md#releases).

## Deployment

Kiln ships as a single static binary; run it directly, or wrap it in Docker
or systemd. Each `kiln-v*` tag publishes a `linux/amd64` CUDA 12.4 image to
GHCR, and `deploy/kiln.service` is the systemd unit. See
[QUICKSTART.md — Running with Docker](QUICKSTART.md#running-with-docker) and
[Running with systemd](QUICKSTART.md#running-with-systemd) for the pull,
build-from-source, and unit installation walkthroughs.

## Status

Kiln is actively developed; the current release line follows the latest
`kiln-v*` GitHub release — see [`CHANGELOG.md`](CHANGELOG.md) for what
landed most recently and [`BENCHMARKS.md`](BENCHMARKS.md) for the serving
acceptance protocol and historical results. Not yet production-hardened for
multi-tenant use; designed for single-user, single-GPU deployments.

## Prior Art

Kiln builds on ideas from:

- [vLLM](https://github.com/vllm-project/vllm) — paged KV cache, continuous batching
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [S-LoRA](https://arxiv.org/abs/2311.03285) — multi-LoRA serving techniques
- [Tinker](https://thinkingmachines.ai/blog/announcing-tinker/) — the cloud-hosted version of this idea. Kiln is the self-hosted, open-source take.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — proof that the core can be simple

## Contributing

Bug reports, performance work, kernel ports, documentation, and dev-experience polish are all welcome; read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a non-trivial PR. Maintained by [@ericflo](https://github.com/ericflo).

## License

MIT — see [LICENSE](LICENSE).

Third-party dependency licenses: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
