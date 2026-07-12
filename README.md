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
  <a href="https://ericflo.github.io/kiln/demo/">Demo</a> &middot;
  <a href="QUICKSTART.md">Quickstart</a> &middot;
  <a href="https://ericflo.github.io/kiln/cli.html">CLI Guide</a> &middot;
  <a href="https://ericflo.github.io/kiln/grpo.html">GRPO Guide</a> &middot;
  <a href="docs/EVAL_GUIDE.md">Eval Guide</a> &middot;
  <a href="https://ericflo.github.io/kiln/api.html">API Reference</a> &middot;
  <a href="https://ericflo.github.io/kiln/troubleshooting.html">Troubleshooting</a> &middot;
  <a href="ARCHITECTURE.md">Architecture</a> &middot;
  <a href="BENCHMARKS.md">Benchmarks</a> &middot;
  <a href="kiln.example.toml">Configuration</a> &middot;
  <a href="CHANGELOG.md">Changelog</a> &middot;
  <a href="CONTRIBUTING.md">Contributing</a> &middot;
  <a href="LICENSE">License</a>
</p>

---

Kiln serves a language model, trains it, and evaluates it on one GPU from one Rust binary. The default `stable` profile isolates predictable inference from GPU-writer transitions. Interactive train/eval loops opt into `experimental`; production systems enter a drained `maintenance` process for training or adapter changes, then restart into `stable`. See [Serving Profiles](docs/SERVING_PROFILES.md).

It targets one model ([Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B)) and optimizes everything for that model — the scheduler, the memory manager, the kernels. This isn't a general-purpose framework. It's a scalpel.

## Why

Today, improving a deployed model looks like: collect failure examples, format them, upload to a training service, wait hours, download new weights, build a separate eval harness in Python, redeploy, hope. Kiln collapses that into one local binary and artifact set. For the interactive loop below, start the server with the explicit development profile:

```bash
# Development only: admit inference plus GPU-writer transitions in one process
KILN_SERVING_PROFILE=experimental KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve

# Submit a correction — the model learns it in seconds
curl http://localhost:8420/v1/train/sft \
  -H "Content-Type: application/json" \
  -d '{
    "examples": [
      {"messages": [
        {"role": "user", "content": "Summarize this contract clause..."},
        {"role": "assistant", "content": "The clause establishes..."}
      ]}
    ]
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
- **pi integration** — `kiln pi-setup` backs up and merges `~/.pi/agent/models.json` + `settings.json`, then points pi at Kiln as an OpenAI-compatible tool-calling backend.
- **Embedded agent runs** — the server drives pi itself (`POST /v1/agent/runs`): spawns `pi --mode rpc` against its own model, streams the trajectory live with steer/abort, and auto-indexes finished sessions into the trace layer the self-improvement flywheel trains on.
- **SFT training** over HTTP — submit examples in `experimental` or drained `maintenance`; publication is atomic.
- **GRPO training** over HTTP — submit scored completions for reinforcement learning. You control the reward function; GPU ownership follows the selected serving profile.
- **On-policy distillation (OPD)** over HTTP — train against an identity-bound local or vLLM teacher, with exact candidate-boundary checkpoints and resume.
- **First-class evals** over HTTP — register suites, run them against any adapter, drill into per-example outcomes. Auto-detect picks the right scorer per example (`numeric_tolerance`, `multiple_choice`, `json_validity`, `regex`, `contains`, `tool_call`, `code`, `llm_judge`, `all`/`any` composites).
- **Dataset → eval synthesis** — upload an SFT JSONL and Kiln decomposes it into an eval suite (final-assistant / first-turn / every-turn / tool-call-prediction strategies). No separate eval harness to write.
- **Judgment flywheel** — A/B-judge two adapters in `/ui/`, save your picks into a judgment dataset, compile to SFT, train a *local* judge LoRA, validate it on a held-out slice. The dashboard ships a streaming side-by-side viewer with `A`/`B`/`Tie`/`Skip` keyboard shortcuts.
- **Post-training auto-eval** — in `experimental`, attach `post_eval` to any SFT/GRPO/OPD request and the produced adapter is graded immediately, with results back-linked to the training job.
- **Adapter smoke tests** — pass `--adapter-smoke-test` on SFT/GRPO CLI submissions to record base-vs-adapter canary metrics in `train_receipt.json` before a full eval.
- **Muon optimizer (default)** — momentum-orthogonalized SGD with fused on-device Newton-Schulz kernels for every backend (CUDA, ROCm, Vulkan, Metal). Converges LoRA fine-tunes in fewer steps than AdamW at roughly half the optimizer state (one momentum buffer vs Adam's two moments). AdamW and SGD remain selectable per-request via `{"optimizer": {"kind": "adam_w"}}` / `{"kind": "sgd"}`. Omit `learning_rate` and the server picks the per-optimizer default (Muon ~2e-2 vs AdamW ~1e-4 for SFT LoRA).
- **Atomic LoRA transitions** — `experimental` supports live hot-swap; `maintenance` supports drained activation; `stable` rejects real weight transitions before GPU ownership changes.
- **Continuous batching** with token-budgeted prefill — long prompts yield after every bounded quantum so ready decode rows keep advancing.
- **128K+ context** on 24GB — Qwen3.5-4B's hybrid architecture (24 linear attention + 8 full attention layers) means KV cache is 4x smaller than a pure transformer.
- **Paged KV cache** — virtual memory-style block allocation eliminates fragmentation.
- **FP8 KV cache** — optional quantization doubles effective context length.
- **Prefix caching** — shared prompt prefixes reuse cached KV blocks.
- **Gradient checkpointing** — training fits on consumer 24GB GPUs (RTX 3090/4090).
- **Adapter management** — load, unload, upload (import), download (export), and version LoRA adapters; click any adapter in `/ui/` for its provenance (training history + eval scores against it).
- **Adapter composition** — stack multiple LoRAs per request with per-adapter scaling and source-revision-aware caching, or merge them server-side via weighted_average / TIES / concatenation.
- **Embedded web dashboard** at `/ui/` — live server status, VRAM donut, adapter cards, training queue with live loss curves, full eval workflow (datasets / suites / jobs / judgments) with drill-in per-example modal, A/B compare playground, and a `⌘K` command palette across all of it. No extra service to run.
- **Prometheus metrics** at `/metrics` — request latency, throughput, training progress, memory usage.
- **Durable request log** — every inference request/response (SSE streams reassembled) lands as one JSONL row under `<adapter_dir>/.requests`, size-rotated, gzipped, retention-capped, attributed to the serving adapter. Production traffic becomes a corpus you can mine into SFT data or a `kiln-eval trace-suite` eval with one `jq` line — see [docs/EVAL_GUIDE.md § Mine your own request log](docs/EVAL_GUIDE.md#mine-your-own-request-log).
- **Training webhooks** — POST a JSON event to a configured URL on training job completion or failure.
- **Pure Rust** — single binary, single process. No Python. No sidecar. No second model in memory.

## The GRPO Loop

This is the killer feature. Generate completions, score them with your own reward function, and feed the results back. The model learns what "good" means for your use case.

```python
import openai
import requests

client = openai.OpenAI(base_url="http://localhost:8420/v1", api_key="unused")

# 1. Generate candidates
responses = [
    client.chat.completions.create(
        model="Qwen3.5-4B",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    for _ in range(8)
]

# 2. Score them however you want — regex, unit tests, another model, human eval
scored = [{"text": r.choices[0].message.content, "reward": my_score(r)} for r in responses]

# 3. Submit — an experimental-profile server trains and hot-swaps atomically
requests.post("http://localhost:8420/v1/train/grpo", json={
    "groups": [{
        "messages": [{"role": "user", "content": prompt}],
        "completions": scored,
    }]
})

# 4. Next inference already uses the improved weights
```

The compact example above intentionally uses the default
`behavior_policy="no_importance_correction"`. For drift-corrected training,
use `kiln rollout-generate`: it requests exact token/action provenance from the
real batching path, validates the seed, adapter, prompt, scored content, and
usage before scoring, and atomically writes JSONL for
`behavior_policy="recorded"`. See [docs/GRPO_GUIDE.md](docs/GRPO_GUIDE.md) for
the recorded-policy workflow and worked verifiable-reward examples (math,
JSON, code).

### Agentic GRPO with ECHO (multi-turn rollouts)

For multi-turn agentic rollouts — tool calls, command output, file contents — kiln's GRPO is **ECHO-by-default** (Shrivastava, Awadallah, Papailiopoulos, MSR AI Frontiers 2026): the trajectory format carries environment segments end-to-end, from rollout JSONL through tokenization and masking to the training receipt, and the env-CE term **trains by default at `λ = 0.05`** on any trajectory with observation segments. ECHO adds a length-normalized cross-entropy loss on environment-observation tokens to the standard policy-gradient loss, with **zero extra forward-pass cost** — the env tokens are already in the rollout context. Paper headline: roughly doubles TerminalBench-2.0 pass@1. Opt out with `--no-echo` / `loss.echo: null` / `KILN_ECHO_ENABLED=0`.

When your rollouts carry a `trajectory` field with `kind: "action"` / `kind: "observation"` segments, kiln-train's `tokenize_grpo_group` builds separate `action_mask` (policy-gradient targets) and `env_mask` (ECHO env-CE targets, `λ_echo = 0.05` by default). The env-CE gradient lives on the same fused GRPO tape root as constant-coefficient `(softmax − one-hot)` rows, and the training receipt records `echo.enabled: true` plus the initial/final env-CE. Legacy single-turn rollouts (no `trajectory` field) behave bit-identically to the pre-ECHO loss — with no env tokens the term contributes exactly zero.

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

See [docs/ECHO_GUIDE.md](docs/ECHO_GUIDE.md) for full ECHO usage (CLI flags, env vars, verifier-free adaptation per paper §5.5, and the receipt-grade `env_ce_drop_pct` diagnostic).

### Off-policy OPD teacher data

For saturated tasks where reward-only GRPO is low signal, kiln-train also
accepts off-policy teacher distillation data: prompt messages plus a teacher
response, optionally with per-token teacher top-logprobs for reverse-KL. The
same agentic `trajectory` shape can be attached so action tokens receive OPD
supervision while observation tokens stay masked out of it. Numeric teacher
logits require a canonical first-line manifest copied from the registered
teacher's `off_policy_manifest`; Kiln rejects a missing or different identity.
ECHO env-CE can be composed with OPD and receipts count action and environment
tokens separately.

See [docs/OPD_TEACHER_JSONL.md](docs/OPD_TEACHER_JSONL.md) for the JSONL
schema and the `reverse_kl` vs `cross_entropy` objective contract.

OPD publishes immutable exact `.kiln-checkpoint` bundles every 25 committed
optimizer steps by default and on cooperative cancellation. Resume binds the
adapter and optimizer tensors, separate optimizer-step and source/sample
candidate cursors, RNG streams, diagnostics, effective configuration, exact
prompt/dataset identity, and authoritative teacher content revision. Use
`kiln train status --job-id JOB_ID` to get the basename, then resubmit the
identical request with `config.resume_checkpoint` or
`kiln train opd --resume-checkpoint BASENAME`. See
[Native Training Checkpoints](docs/training-checkpoints.md#opd).

### Remote vLLM teachers

Remote numeric prompt-logprobs are accepted only from an identity-aware vLLM
process launched with [`scripts/vllm_teacher.py`](scripts/vllm_teacher.py).
The launcher snapshots the model and optional static adapter, binds their exact
content plus tokenizer/runtime limits, the resolved Python executable, and the
installed vLLM/torch/Transformers/tokenizers content into
`system_fingerprint`, then re-hashes the runtime in a fresh child immediately
before spawn. Dynamic LoRA mutation is disabled and stock `vllm-*`
fingerprints are rejected.

```bash
python3 scripts/vllm_teacher.py \
  --model-path=/models/Qwen3.5-4B \
  --served-model-id=qwen35-teacher \
  --max-top-k=32 \
  --max-model-len=32768 \
  --max-prompt-logprob-candidates=1000000 \
  -- --host=127.0.0.1 --port=8000

curl -X POST http://localhost:8420/v1/teachers \
  -H 'content-type: application/json' \
  -d '{"alias":"qwen35@vllm","kind":"remote","provider":"vllm","model_id":"qwen35-teacher","url":"http://127.0.0.1:8000"}'
```

Kiln probes K=1 and the advertised maximum K during registration, verifies the
complete numeric vocabulary against the loaded student, persists the canonical
identity, and repeats the probe at every job start before GPU ownership or a
cache lookup. Off-host URLs require HTTPS and a server-owned `credential_id`
configured under `[teachers.credentials]`; API callers can never choose the
secret environment-variable name. `GET /v1/teachers` reports `status`,
`usable`, `identity_revision`, bounds, and the exact off-policy manifest.
On first startup after upgrading, Kiln removes legacy `api_key_env` fields from
`teachers.json`; those entries stay unusable until they are explicitly deleted
and re-registered with a configured credential handle and fresh identity.
See [docs/VLLM_TEACHER_IDENTITY.md](docs/VLLM_TEACHER_IDENTITY.md).

## The Eval Loop

Training is half the story; the other half is knowing whether your last training run actually helped. In the `experimental` development profile, Kiln's eval system runs in the same process against the same model weights. In production, train under `maintenance`, restart into `stable`, and evaluate before restoring traffic. Both workflows use the same first-class artifacts: registered suites, drillable per-example outcomes, A/B comparisons across adapters, and a judgment flywheel that turns your A/B picks into a *local* judge LoRA you can re-use.

```bash
# 1. Upload an SFT JSONL — Kiln will use it as the source of truth for examples
curl -F name=customer-support -F format=sft_chat -F file=@my-tickets.jsonl \
  http://localhost:8420/v1/eval/datasets/upload

# 2. Synthesize an eval suite from it (auto-detect picks the right scorer per example)
curl -X POST http://localhost:8420/v1/eval/datasets/customer-support/synthesize \
  -H 'content-type: application/json' \
  -d '{"suite_name":"support-eval","strategy":"final_assistant",
       "scorer":{"kind":"auto_detect"},
       "sampling":{"max_examples":100,"max_prompt_chars":32768,"max_target_chars":4096,"dedupe":true},
       "force":true,"run_against":["v1"]}'

# 3. Compare two adapters head-to-head
curl -X POST http://localhost:8420/v1/eval/compare \
  -H 'content-type: application/json' \
  -d '{"suite":"support-eval","adapters":["v1","v2"]}'

# 4. Drill into per-example outcomes at /ui/ — pass/fail/invalid badges with prompt + target + got
#    side-by-side, scorer detail, and a one-click "re-run failures" loop.
```

The judgment flywheel runs entirely on your machine — no frontier LLM, no API keys, no telemetry leaving the box. Click two replies in the playground, save them as an A/B preference, compile your picks into SFT data, train a small judge LoRA on them, then use that LoRA as the `judge_adapter` in any `LlmJudge` scorer. The judge gets better the more you use it; bad judgments are removable from the dataset and a retrain wipes their influence.

See [docs/EVAL_GUIDE.md](docs/EVAL_GUIDE.md) for the full scorer reference, dataset synthesis strategies, and the judge-LoRA workflow.

## Quick Start

**Supported hardware:** NVIDIA GPU with 24GB+ VRAM and CUDA 12+, AMD GPU with ROCm/HIP 7.2.4+ on Linux, AMD/Intel GPU with Vulkan 1.2+ on Linux, **or** Apple Silicon Mac with 16GB+ unified memory. Kiln targets `Qwen/Qwen3.5-4B` and needs about 20GB of free disk for the server, model weights, and adapters.

**Path 1 — Desktop App (recommended):** Install [Kiln Desktop](#desktop-app) on Windows, Linux, or macOS. The app downloads and verifies the matching prebuilt `kiln` server binary on first launch, then walks you through choosing or downloading `Qwen/Qwen3.5-4B`. No Rust toolchain, CUDA toolkit, or source build is required for this path.

**Get the model weights** (Paths 2–4 share this step; the Desktop App handles it automatically):

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
```

This downloads `Qwen/Qwen3.5-4B` into `./Qwen3.5-4B`, which the commands below reference directly.

**Path 2 — Server binary (terminal-first, no source build):** Download the latest `kiln-v*` server artifact when you want the `kiln` server in your terminal with no source build, Desktop App, or Docker. Linux x86_64 + NVIDIA CUDA 12.4 is the compact path (run the `Qwen/Qwen3.5-4B` weights step above first):

```bash
KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest | sed -n 's/.*"tag_name": "kiln-v\([^"]*\)".*/\1/p')
curl -L -o kiln-linux-cuda.tar.gz \
  "https://github.com/ericflo/kiln/releases/download/kiln-v${KILN_VERSION}/kiln-${KILN_VERSION}-x86_64-unknown-linux-gnu-cuda124.tar.gz"
tar -xzf kiln-linux-cuda.tar.gz

KILN_MODEL_PATH=./Qwen3.5-4B ./kiln serve
```

For Linux ROCm, Linux Vulkan, macOS Metal, Windows CUDA, and SHA-256 sidecar checks, use the full release artifact matrix in [QUICKSTART.md](QUICKSTART.md#quick-path-server-binary-terminal-first-no-source-build).

**Path 3 — Container:** Run the prebuilt GHCR image when you prefer containerized deployment. This path requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Make sure the weights step above has placed the model under `./Qwen3.5-4B` (or substitute your own absolute path), then mount that directory into the container:

```bash
docker pull ghcr.io/ericflo/kiln-server:latest
docker run --gpus all -p 8420:8420 \
  -e KILN_MODEL_PATH=/models/Qwen3.5-4B \
  -v "$(pwd)/Qwen3.5-4B:/models/Qwen3.5-4B:ro" \
  ghcr.io/ericflo/kiln-server:latest serve
```

Open http://127.0.0.1:8420/ui/ after the container starts.

**Path 4 — Source / CLI:** Install Rust stable, then build the CLI from source when you are contributing, scripting against a local checkout, or need to test unreleased changes.

```bash
git clone https://github.com/ericflo/kiln.git
cd kiln

# Linux / Windows + NVIDIA
cargo build --release --features cuda     # ~15-30 min first build (CUDA kernels)

# Linux + AMD / Intel
# Requires Vulkan 1.2+ runtime plus glslc or glslangValidator for shader embedding.
cargo build --release --features vulkan   # Vulkan compute kernels via ash + SPIR-V

# Linux + AMD ROCm/HIP
# Requires ROCm/HIP SDK plus hipBLASLt; the release build emits CDNA, RDNA3, and Strix Halo.
ROCM_PATH=/opt/rocm KILN_ROCM_ARCHS='gfx90a;gfx942;gfx1100;gfx1151' cargo build --release --no-default-features --features rocm

# macOS + Apple Silicon
cargo build --release --features metal    # Metal backend via candle
```

Start the source-built server (using the weights downloaded above):

```bash
KILN_MODEL_PATH=./Qwen3.5-4B ./target/release/kiln serve
```

Vulkan builds auto-select a Vulkan physical device at startup. Use `KILN_VULKAN_DEVICE=0` to pin a zero-based Vulkan device index, or `GGML_VK_VISIBLE_DEVICES=0,1` to reuse llama.cpp-style visibility; invalid values are ignored with a warning and Kiln falls back to automatic selection or CPU if no Vulkan device is usable. ROCm builds use the HIP backend compiled into the binary; set `KILN_ROCM_ARCHS` at build time to control emitted gfx targets.

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

The `GPU` and `VRAM` lines come from `nvidia-smi` and are skipped silently if it isn't installed.

**Verify the server is up.** Run `kiln health` (binary at `./kiln` for Path 2/3 or `./target/release/kiln` for Path 4) before sending real requests — it prints a readable tree with model, scheduler, training, and GPU status, and exits non-zero if anything is wrong. See [troubleshooting](https://ericflo.github.io/kiln/troubleshooting.html#start-with-three-probes) for the full three-probe sequence.

```bash
./kiln health
```

```bash
# Chat
curl http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "Hello!"}], "stream": true}'
```

The default server is `stable`. To run the development training example, stop
it and restart the same binary/model command with
`KILN_SERVING_PROFILE=experimental`, then submit:

```bash
# Train
curl http://localhost:8420/v1/train/sft \
  -H "Content-Type: application/json" \
  -d '{"examples": [{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hey there!"}]}]}'

# Check training
curl http://localhost:8420/v1/train/status
```

Every accepted SFT, GRPO, OPD, or distillation submission materializes one
immutable effective seed before the job becomes visible in the queue. The
submit response, status/detail APIs, CLI, and dashboard report it as an exact
decimal string so all 64 bits survive browser clients. Set `config.seed` to
choose it; when omitted, Kiln chooses and persists it. An exact-resume request
inherits the checkpoint's original LoRA-initialization seed and rejects a
conflicting caller value. A seed makes a run repeatable only inside the rest of
the declared deterministic environment; it is not a promise of byte-identical
output across different builds, drivers, devices, precision policies, or
backends.

**Optional: use Kiln as pi's local agent model.** With the server running on `localhost:8420`, `kiln pi-setup` adds a `kiln-local` provider to pi without deleting your other providers or settings. Existing files are backed up first as `models.json.bak-<timestamp>` and `settings.json.bak-<timestamp>`.

```bash
# Local server
./kiln pi-setup

# Remote office/server box
./kiln pi-setup --kiln-url http://office-kiln:8420

# pi now uses model Qwen3.5-4B through http://.../v1
pi -p "Use the bash tool to run: pwd"
```

Kiln accepts Qwen3.5's native XML tool-call generations internally, but OpenAI-compatible clients receive normal `tool_calls` in both streaming and non-streaming responses. pi should execute the tool call instead of printing raw `<tool_call>` XML.

**Embedded agent runs: kiln drives pi itself.** Beyond serving pi as a model backend, the server can spawn `pi --mode rpc` as a managed child, hand it a task, stream the trajectory live, and merge the finished session straight into the agent-trace layer the self-improvement flywheel reads — on-policy rollouts on demand, no human at the keyboard:

```bash
# Start a run (the spawned pi talks back to this same server)
curl http://localhost:8420/v1/agent/runs \
  -H "Content-Type: application/json" \
  -d '{"task": "Run the test suite and fix the first failure", "cwd": "/path/to/project"}'

# Watch it live (poll cursor), steer it, or abort it
curl "http://localhost:8420/v1/agent/runs/<id>/events?after=0"
curl http://localhost:8420/v1/agent/runs/<id>/steer -d '{"message": "Prefer a minimal fix"}' -H "Content-Type: application/json"
curl -X POST http://localhost:8420/v1/agent/runs/<id>/abort
```

Runs queue FIFO (`[agent].max_concurrent_runs`, default 2; `run_timeout_secs`, default 900), persist across restarts, and auto-index into `GET /v1/agent/traces` when they finish — so `kiln self-improve` trains on them in the same cycle. Because embedded runs execute code on the server, they are enabled only on loopback binds by default (`KILN_AGENT_RUNS=1` opts in elsewhere, `=0` force-disables). The dashboard's **Distill → Agent runs** tab provides a launch form and a live trajectory view.

See [QUICKSTART.md](QUICKSTART.md) for the full walkthrough including Desktop App setup, source builds, GRPO, adapter management, Docker, and systemd setup. If setup stalls on binary downloads, CUDA/ROCm/Vulkan/Metal, model paths, `/health`, mock mode, training endpoints, or adapter directories, start with the [Troubleshooting guide](https://ericflo.github.io/kiln/troubleshooting.html). For tools-bearing workloads on older pinned releases, see [QUICKSTART.md §9.2](QUICKSTART.md#92-troubleshooting-older-release-long-prefill-timeouts) for the legacy `workers=1` / request-timeout troubleshooting note ([#664](https://github.com/ericflo/kiln/issues/664)).

## See it in action

Six short asciicasts captured on a single A6000 against `Qwen3.5-4B` show the main developer flows: first token from cold start, benchmark output, LoRA hot-swap, an OpenAI-compatible client, GRPO with a custom reward, and the full SFT online-learning loop. Watch them in the embedded player at **[ericflo.github.io/kiln/demo/](https://ericflo.github.io/kiln/demo/)** or browse the recording scripts and reference shell drivers under [`docs/site/demo/`](docs/site/demo/).

The kiln server also ships an embedded web dashboard at `http://localhost:8420/ui/` with live decode tok/s, p50/p99 ITL, VRAM breakdown, adapter management, training monitoring, and a chat playground — no extra service to run.

Here is the embedded server dashboard running with healthy status, active adapters, training progress, and the chat playground in one view:

![Kiln embedded server dashboard showing healthy status metrics, adapters panel, training progress, and chat quick-inference panels](docs/site/assets/server-ui-dashboard.png)

## Memory Budget (24GB GPU)

Qwen3.5-4B's hybrid architecture is the key. Only 8 of 32 layers need KV cache, so long-context inference costs a fraction of what a pure transformer would.

| Scenario | Total VRAM | Fits 24GB? |
|---|---|---|
| 128K context, 1 sequence, inference only | ~13 GB | Yes |
| 128K context, 1 sequence, inference + training | ~18 GB | Yes |
| 64K context, 4 sequences, inference + training | ~22 GB | Yes |
| 32K context, 8 sequences, inference + training | ~22 GB | Yes |
| 128K context, 4 sequences, FP8 KV cache | ~19 GB | Yes |

### Apple Silicon (M3 Max / M4 Max 64GB unified memory)

On Apple Silicon, model weights, KV cache, and training state all live in unified memory shared with the OS. A 64 GB chip leaves generous headroom for long contexts and concurrent training.

| Scenario | Unified Memory | Fits 64GB? |
|---|---|---|
| 128K context, 1 sequence, inference only | ~13 GB | Yes |
| 128K context, 1 sequence, inference + training | ~18 GB | Yes |
| 64K context, 4 sequences, inference + training | ~22 GB | Yes |
| 128K context, 8 sequences, inference + training | ~32 GB | Yes |
| 128K context, 4 sequences, FP8 KV cache | ~19 GB | Yes |

16 GB M-series chips fit short-context inference; 32 GB fits 64K context comfortably; 64 GB+ matches or exceeds the 24 GB CUDA envelope.

## API

| Method | Path | Description |
|---|---|---|
| POST | `/v1/chat/completions` | Chat completions (OpenAI-compatible), including per-request thinking budgets and opt-in exact single-choice `rollout_provenance` |
| POST | `/v1/completions` | vLLM-shaped prompt-logprob subset with a canonical base-teacher identity fingerprint |
| POST | `/v1/completions/batch` | Text-only batch generation (up to 64 prompts per request), with the same thinking-budget controls but no recorded behavior-policy probabilities |
| POST | `/v1/train/sft` | Submit SFT training examples and return the exact effective seed under `experimental` or `maintenance` (optionally with a `post_eval` hook in `experimental`) |
| POST | `/v1/train/grpo` | Submit GRPO scored completions and return the exact effective seed under `experimental` or `maintenance` (optionally with a `post_eval` hook in `experimental`). Supports the new `agentic_groups` shape with multi-turn `trajectory` fields; action/observation masks are built end-to-end, and the ECHO env-CE term applies by default (λ=0.05) to trajectories with observation segments. |
| POST | `/v1/train/agentic` | Canonical alias of `/v1/train/grpo` — same handler, semantically-honest name for multi-turn rollouts |
| POST | `/v1/train/opd` | Submit on-policy or off-policy distillation against a registered, identity-bound teacher, return the exact effective seed, and default exact checkpoints to every 25 committed optimizer steps |
| GET | `/v1/train/status` | Training queue, job status, and exact effective seeds |
| GET | `/v1/train/status/{job_id}` | Inspect one training job and its exact effective seed |
| GET | `/v1/train/jobs/{job_id}` | Rich training job detail (effective seed, base-weight shard identity, loss curve, linked evals, and latest exact SFT/GRPO/OPD checkpoint metadata) |
| GET | `/v1/train/queue` | List queued training jobs |
| DELETE | `/v1/train/queue/{job_id}` | Cancel a job (queued: dequeued; running: stops at the next step boundary) |
| GET | `/v1/stats/mtp-acceptance` | Per-adapter MTP draft acceptance (live alpha) |
| DELETE | `/v1/train/queue/{job_id}` | Cancel a queued job |
| POST | `/v1/distill/refresh` | Continual-learning distillation refresh job with an exact effective seed |
| POST | `/v1/distill/pump` | Continual-learning distillation pump job with an exact effective seed |
| GET / POST | `/v1/corrections` | Durable corrections store — the basket survives the browser; pi can file corrections |
| DELETE | `/v1/corrections/{request_id}` | Remove a correction |
| POST | `/v1/corrections/mark_trained` | Mark correction rows as trained into an adapter (kept as history) |
| GET / POST | `/v1/teachers` | Teacher registry; remote registration performs authoritative identity and capability probes |
| DELETE | `/v1/teachers/{alias}` | Remove a registered teacher |
| GET | `/v1/cache/stats` | Off-thread, serialized validation and summary of identity-bound cache-v3 entries |
| GET | `/v1/cache/export` | Stream a deterministic validated export (16 GiB source / 1M-file limit); concurrent scans and archive import are rejected |
| POST | `/v1/agent/traces/discover` | Index pi/agent session traces for training |
| GET | `/v1/agent/traces` | List discovered agent session traces |
| POST | `/v1/agent/self_improve` | One-call agentic self-improvement (traces → OPD + judge distill), returning effective seeds keyed by job ID |
| POST | `/v1/agent/judge_distill` | Distill a judge LoRA from agent traces and return its exact effective seed |
| GET / POST | `/v1/agent/runs` | Embedded pi runs — the server spawns `pi --mode rpc`, streams the trajectory, auto-indexes the session as a trace |
| GET | `/v1/agent/runs/status` | Embedded-run gate state, pi availability, capacity |
| GET | `/v1/agent/runs/{id}` | One run record (status, counters, session link) |
| GET | `/v1/agent/runs/{id}/events` | Live event feed for a run (`?after=<seq>` poll cursor) |
| POST | `/v1/agent/runs/{id}/steer` | Queue a steering message into a live run |
| POST | `/v1/agent/runs/{id}/follow_up` | Queue a follow-up task into a live run |
| POST | `/v1/agent/runs/{id}/abort` | Abort a queued or running run |
| GET | `/v1/adapters` | List saved/available LoRA adapters, identify the active adapter, and report the exact loaded name/content revision |
| GET | `/v1/adapters/{name}/detail` | Files + training history + eval history for one adapter |
| POST | `/v1/adapters/load` | Load adapter from disk and return its exact content revision |
| POST | `/v1/adapters/unload` | Unload active adapter |
| DELETE | `/v1/adapters/{name}` | Delete an idle adapter; active or physically loaded adapters return 409 |
| POST | `/v1/adapters/upload` | Stage and atomically publish a multipart tar.gz adapter import |
| GET  | `/v1/adapters/{name}/download` | Stream adapter as tar.gz (export) |
| POST | `/v1/adapters/merge` | Stage and atomically publish an adapter merge (weighted_average, TIES, or concatenation modes) |
| POST | `/v1/adapters/distill_merge` | Behaviour-space adapter merge with an exact effective seed |
| POST | `/v1/recipes/run` | Queue a typed training recipe and return effective seeds keyed by job ID |
| GET / POST | `/v1/eval/suites` | List or register eval suites (body = `EvalSuite`) |
| GET / DELETE | `/v1/eval/suites/{name}` | Fetch / delete one suite |
| POST | `/v1/eval/run` | Submit an eval and materialize one effective seed (registered suite or inline) |
| POST | `/v1/eval/compare` | Run a suite head-to-head with identical derived seeds across adapters |
| GET | `/v1/eval/jobs` | List eval jobs, effective seeds, base-weight and execution identities, and headline results |
| GET / DELETE | `/v1/eval/jobs/{job_id}` | Per-job seed/base-weight/execution identity, status, outcomes, or cancel/delete |
| POST | `/v1/eval/jobs/{job_id}/rerun` | Re-run failures with the original effective seed by default |
| POST | `/v1/eval/datasets/upload` | Multipart upload of an SFT/GRPO JSONL dataset |
| GET / DELETE | `/v1/eval/datasets/{name}` | Dataset manifest / delete |
| GET | `/v1/eval/datasets/{name}/rows` | Stream first N rows (used by the SFT submit form's dataset picker) |
| POST | `/v1/eval/datasets/{name}/preview` | Preview synthesized examples before committing |
| POST | `/v1/eval/datasets/{name}/synthesize` | Decompose a dataset into an eval suite |
| GET / POST | `/v1/judgments` | List or create judgment datasets |
| DELETE | `/v1/judgments/{name}` | Delete a judgment dataset |
| POST | `/v1/judgments/{name}/rows` | Append one A/B/Tie/Skip preference |
| DELETE | `/v1/judgments/{name}/rows/{id}` | Prune a bad row (then retrain to wipe its influence) |
| POST | `/v1/judgments/{name}/compile` | Compile judgments into an SFT dataset for training a judge LoRA |
| POST | `/v1/judgments/{name}/validate` | Score a judge LoRA against a held-out judgment slice |
| POST | `/v1/judgments/render_prompt` | Render the canonical pairwise judging prompt (debug aid) |
| GET | `/v1/models` | List available models |
| GET | `/v1/config` | Current server configuration, serving-profile source, and every effective profile policy |
| GET | `/v1/debug/model-state` | Trusted eval/debug snapshot of the complete base-weight shard manifest and execution-provenance record, active model/adapters, config hashes, env flags, batching, thinking defaults, and cache counts; enabled only with `server.eval_mode=true` or `KILN_DEBUG_ENDPOINTS=1` |
| GET | `/ui/` | Embedded web dashboard (Overview / Adapters / Training / Evals / Playground) |
| GET | `/v1/stats/decode` | Live decode tokens/sec and inter-token latency stats used by the dashboard |
| GET | `/v1/stats/recent-requests` | Bounded recent chat-completion history, including effective thinking-budget provenance and outcomes, for the dashboard's request panel |
| GET | `/health` | Server readiness and diagnostics, including bounded base-weight and execution-identity summaries; maintenance or missing/invalid real-backend execution provenance returns 503 |
| GET | `/v1/health` | `/v1` compatibility alias for readiness and diagnostics |
| GET | `/metrics` | Prometheus metrics |

### Prompt logprobs

`POST /v1/completions` is a deliberately bounded vLLM-shaped scoring subset:
set `max_tokens` to `0` or `1`, disable streaming, and request
`prompt_logprobs` from `0` through `256`. Prompts may be text or token-ID
arrays and are capped at the smaller of 4096 tokens and the served model's
context window. Text prompts default to
`add_special_tokens: true`, matching vLLM; set it to `false` explicitly when
the supplied text already contains the required boundaries. Kiln does not
support vLLM's `prompt_logprobs: -1` all-vocabulary extension.

The first result is `null`. Position `i` scores the observed prompt token at
`i` from logits row `i-1` and returns that token plus the requested top K. The
map therefore contains K entries when the observed token is already top K and
K+1 otherwise; K=0 returns only the observed token. Extra observed tokens carry
their full-vocabulary rank. Scores remain F32, ties are deterministic by token
ID, and split UTF-8 display tokens are decoded independently against preceding
actual prompt context. Vocabulary drift, decoder failures, wrong-width rows,
and non-finite logits or scores fail the request instead of producing partial
JSON. The optional `model` must exactly match the one served model. Every real
response carries a canonical identity binding the loader-owned base bytes,
numeric tokenizer vocabulary/config, executable, numerical runtime, backend,
and scoring limits. Scoring remains base-model only and rejects an active LoRA
until the loaded-adapter revision barrier is implemented.
The runtime digest includes OS/kernel build, stable CPU model/features and
microcode, loaded numerical-library mappings, and accelerator/driver evidence.
External probes have a five-second deadline, bounded output, process-group
cleanup, and explicit missing/timeout markers in the digest.

Before parsing weights, Kiln creates a private read-only checkpoint snapshot and
loads only from that snapshot. Linux reflink and macOS clonefile make this
copy-on-write when the filesystem supports it; otherwise startup performs a
full bounded copy and requires enough free space plus a reserve. Set
`model.snapshot_dir` (or its `KILN_MODEL_SNAPSHOT_DIR` override) to a private
filesystem with suitable capacity when the model directory's parent is not
appropriate. The snapshot stays alive for
deferred MTP loading. The server explicitly removes it after model-backed work
drains; the loader lease also removes it on drop for startup errors and library
callers. Cleanup retries after restoring owner-only deletion permissions and a
server shutdown fails with the exact residual path if removal still fails.
Mutating the original checkpoint after startup cannot change loaded bytes or
their revision. A user
with the same UID (or root) can still discover and rewrite process-owned files;
Kiln treats that as part of the trusted host boundary.

That startup hash pass also retains a strict manifest for every safetensors
shard: portable filename, exact byte length, complete SHA-256, and one
relocation-independent aggregate. `/health` exposes its bounded digest/count/
byte summary; gated `/v1/debug/model-state` exposes the full list. Exact
checkpoints, training receipts, adapter manifests, and eval results persist the
same identity without re-reading weights during inference. See
[Base-Weight Provenance](docs/BASE_WEIGHT_PROVENANCE.md) for the schema,
aggregate algorithm, legacy behavior, and exact-resume rules.

After backend initialization, Kiln also binds the resident runner to a strict
self-verifying execution envelope: backend/device, numerical runtime,
executable and optional source identity, tokenizer/template, precision, kernel
contract, and effective configuration/environment digests. Health exposes a
bounded summary and requires a valid record for real-backend readiness; the
gated debug endpoint exposes the complete record. See
[Execution Provenance](docs/EXECUTION_PROVENANCE.md) for the schema, evidence
sources, secret-redaction policy, and integrity scope.

Exact native checkpoints require that complete record and compare its canonical
digest before resume. Successful model-backed training receipts and adapter
manifests persist it, together with the trainer's concrete parameter,
optimizer-state, activation, gradient, and stochastic-rounding precision
contract. Legacy serving artifacts remain readable, but a partial legacy
runtime string is not accepted as evidence for exact continuation.

The response is additionally capped at 65,536 candidate entries. Real scoring
uses the runner's resident backend and inference recurrent-state policy, omits
the unused final logits row, and takes exclusive GPU admission so it cannot
multiply vocabulary scratch allocations alongside generation or training. The
scorer explicitly settles backend work before recycling each projection chunk
or releasing admission; a failed or panicked settlement quarantines the
backend and retains ownership instead of admitting work against an unknown GPU
state. Timeouts and dropped requests signal cancellation, and timeout responses
wait for worker settlement. The
current correctness-first implementation still reads each scored vocabulary
row to the host, so its transfer work is O(TV), not vLLM's selected-only O(TK)
path; use it for bounded teacher queries rather than high-throughput serving.

Backend quarantine is process-wide and irreversible. Once completion cannot be
proven, `/health` becomes `503 degraded` with
`backend_runtime.restart_required=true`, and Prometheus reports
`kiln_backend_quarantined 1`. New completion and eval work, adapter mutations,
prewarm operations, and training submissions reject with HTTP 503 and error
code `backend_quarantined`. Jobs already queued transition to `failed`; an SFT
job between steps and every job-wide training writer poll the same health latch
instead of waiting forever behind the intentionally retained GPU read owner.
Restart the process to construct fresh backend state; there is no unsafe reset
endpoint.

```json
{
  "model": "Qwen3.5-4B",
  "prompt": [1, 42, 314],
  "max_tokens": 0,
  "prompt_logprobs": 5,
  "add_special_tokens": true
}
```

### Chat adapter selection

`POST /v1/chat/completions` treats adapter selection as a per-request choice unless you call the adapter management endpoints:

| Request field | Behavior |
|---|---|
| `adapter` omitted | Use the current server default adapter without changing it. |
| `"adapter": null` | Use the base model for this request only. |
| `"adapter": ""` | Use the base model for this request only. |
| `"adapter": "<name>"` | Use that named adapter for this request only; the name must be a loaded/available adapter directory under `adapter_dir`. |

Only `POST /v1/adapters/load` and `POST /v1/adapters/unload` change the
server default adapter reported by `GET /v1/adapters`. A successful load
returns `content_revision`; `GET /v1/adapters` publishes the authoritative
`loaded_adapter_identity` tuple, and `/health` plus debug state expose
`loaded_adapter_revision`. Completion responses carry the same value in
`x-kiln-loaded-adapter-revision` (`base` when no LoRA is loaded).

The revision is a canonical SHA-256 over the exact PEFT config and safetensor
identities consumed by the loader. It is published with the runner's weight
flip, copied into queued inference work, prefix-cache keys, and deterministic
response-cache keys, and checked again before queued prefill. Same-name
rewrites therefore cannot reuse old KV or responses. Cache purges advance a
generation fence and revoke in-flight owner claims, so a late old request
cannot restore an invalidated entry. Chat requests log adapter runtime
transitions with the old adapter, new adapter, request id, and transition
reason.

### Adapter publication and conflicts

All adapter-changing operations share one serialized revision barrier. Uploads
and merges prepare dot-prefixed staging directories and become visible with one
rename; an incomplete artifact never appears in `GET /v1/adapters`. Delete and
rewrite operations refuse to touch bytes that the runner still has physically
loaded. `DELETE /v1/adapters/{name}` returns 409 `adapter_active` for the server
default and 409 `adapter_loaded` for any physically loaded revision; call
`POST /v1/adapters/unload` and retry. A failed post-eval gate uses the same
barrier, swaps a loaded rejected adapter to base, then renames it to `.failed`.
Per-request composition also resolves exact source revisions and performs
synthesis, atomic cache publication, live swap, and eviction under this
barrier. Its cache identity includes names, scales, and source revisions, so a
same-name source rewrite cannot reuse a stale composed adapter.

Training writes weights, receipts, replay data, lineage, and checkpoints under
a hidden staging root. At job start Kiln captures the target content revision
and a filesystem-local starting snapshot. At completion it compares the target
again: an intervening upload, delete, gate action, or other publisher wins, the
training job fails with `adapter_revision_conflict`, and that newer revision is
preserved. For an ungated same-name target that is already loaded, the final
directory replacement, fresh weight flip, loaded identity, and cache purge run
inside one drained inference barrier. This reload is required even when
`auto_load=false`, because the old bytes were already serving under that name.
When `post_eval.min_accuracy` is set, a loaded same-name rewrite is rejected
before GPU training instead of serving unapproved weights; unload it or choose
a versioned `config.output_name`.

SFT, GRPO, and OPD can also publish immutable exact `.kiln-checkpoint` directories
directly beneath the adapter registry while the final adapter remains staged.
They restore optimizer, cursor/RNG, and objective-specific reference state, not
just PEFT weights; admission validates the complete bundle and exact data route
before GPU work. See [Native Training Checkpoints](docs/training-checkpoints.md)
for API, CLI, browser, cancellation, teacher-identity, and resume semantics.

## Architecture

```
Single Rust binary:
  HTTP (axum) ─── Scheduler (continuous batching, chunked prefill)
                      │
                  Block Manager (paged KV cache)
                      │
                  Engine (model forward + LoRA)
                  ├── 24× Gated DeltaNet layers (linear attention, O(1) state)
                  └──  8× GQA layers (full attention + KV cache)
                      │
                      ├── Training worker (background thread, shares GPU)
                      │   ├── SFT (cross-entropy on LoRA parameters)
                      │   ├── GRPO (advantage-weighted policy gradient)
                      │   └── OPD (identity-bound teacher distillation)
                      │
                      └── Eval worker (background thread, shares GPU)
                          ├── Suite registry + dataset registry + judgment store on disk
                          ├── Pluggable scorers (auto-detect, exact, regex, JSON,
                          │   numeric, MCQ, contains, tool_call, code, llm_judge, all/any)
                          └── Post-training auto-eval hook
```

Everything runs in one process. Training and evaluation share the already-loaded model — no second copy in VRAM, no Python sidecar, no frontier-LLM dependency for judging.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full deep-dive.

## Project Structure

```
crates/
  kiln-core/             Core types: block manager, prefix cache, config, request lifecycle
  kiln-model/            Model loading, forward pass, LoRA, sampling
  kiln-scheduler/        Continuous batching scheduler with chunked prefill
  kiln-server/           HTTP server, CLI, training queue, eval queue, metrics, config
  kiln-train/            SFT, GRPO, and OPD training loops with gradient checkpointing
  kiln-eval/             Suites, scorers, results, dataset → eval synthesis (pure CPU, no GPU dep)
  kiln-nvtx/             Thin NVTX range wrapper for nsys attribution (no-op when off)
  kiln-flce-kernel/      Fused Linear Cross-Entropy (chunked CE without [T, V] logits)
  kiln-flash-attn/       Vendored Flash-Attention-2 CUDA kernels (C-ABI + Rust FFI) [CUDA only]
  kiln-gdn-kernel/       Vendored Gated DeltaNet chunkwise + recurrent kernels [CUDA only]
  kiln-vulkan-kernel/    Vulkan compute shaders for AMD/Intel GDN hot paths [Vulkan only]
  kiln-conv1d-kernel/    Vendored mamba-ssm causal_conv1d_update decode kernel [CUDA only]
  kiln-rmsnorm-kernel/   Fused RMSNorm CUDA kernel (Liger-style) [CUDA only]
  kiln-marlin-gemm/      Vendored IST-DASLab Marlin W4A16 GEMM kernel [CUDA only]
```

## Configuration

Kiln uses a TOML config file. Environment variables override config values. See [`kiln.example.toml`](kiln.example.toml) for all options. The server, CLI, and desktop share the versioned [runtime-defaults contract](contracts/runtime-defaults-v1.json), including the local port. The default `stable` GPU ownership contract and the restart-only maintenance workflow are documented in [Serving Profiles](docs/SERVING_PROFILES.md).

| Setting | Env Var | Default | Description |
|---|---|---|---|
| `model.path` | `KILN_MODEL_PATH` | — | Path to model weights (required) |
| `server.port` | `KILN_PORT` | 8420 | Server listen port |
| `server.serving_profile` | `KILN_SERVING_PROFILE` | `stable` | Immutable process-lifetime GPU ownership policy: `stable`, `experimental`, or `maintenance`; malformed values stop startup, and health/config report the source and every effective policy field |
| `server.http_send_buffer_bytes` | `KILN_HTTP_SEND_BUFFER_BYTES` | OS default | Optional accepted-socket `SO_SNDBUF` request (1024–16777216 bytes); Kiln preflights it before readiness and reports requested, kernel-readback, and platform-normalized effective bytes in health/debug |
| `server.stream_stall_grace_ms` | `KILN_STREAM_STALL_GRACE_MS` | 2000 | Maximum continuous time a full 64-event response channel may make no delivery progress before that request is cancelled (10–2000 ms). Strict startup validation rejects malformed or out-of-range values; health/debug report the effective value and whether it came from the default, config file, or environment |
| `server.max_batch_tokens` | `KILN_MAX_BATCH_TOKENS` | 512 | Combined decode-plus-prefill tokens per batching-actor cycle (2–65536). Ready decode rows consume one token each first; admission and resumable prefill share the remainder. Invalid values stop startup; health/debug report value and source |
| `server.max_prefill_tokens_per_cycle` | `KILN_MAX_PREFILL_TOKENS_PER_CYCLE` | 64 | Prompt-only ceiling inside the combined actor-cycle budget (1–65536). Decode reserves its rows first, then partial prefills advance round-robin by at most this many tokens before the next decode cohort. Lower values favor ITL; higher values favor prefill throughput. Invalid values stop startup; health/debug report the effective value and source |
| `server.max_prefill_layers_per_cycle` | `KILN_MAX_PREFILL_LAYERS_PER_CYCLE` | 4 | Transformer-layer ceiling for each retained prompt chunk (1–1024). A partial chunk yields to ready decode after this many layers and later resumes from its hidden state without replay. Lower values favor ITL; higher values reduce scheduling and synchronization overhead. Invalid values stop startup; health/debug report value and source |
| `server.max_decode_batch` | `KILN_MAX_DECODE_BATCH` | `auto` (backend policy) | Concurrent decode-row ceiling (`auto` or 1–65536). Invalid values stop startup. Deterministic mode and `max_batch_tokens` may lower it; startup, health, config, debug, and `kiln_batching_engine_max_decode_batch` report the final effective value |
| `server.deterministic` | `KILN_DETERMINISTIC` | false | Serving repeatability envelope. Strict boolean parsing rejects malformed values. True freezes the process-wide determinism selector and forces effective decode width 1 even when a wider batch is configured, preventing request-cohort changes from selecting different BF16 batched-GEMM shapes at close greedy-logit boundaries. This does not by itself make every accelerator kernel bitwise deterministic |
| `server.default_thinking_enabled` | `KILN_DEFAULT_THINKING_ENABLED` | template default | Default `chat_template_kwargs.enable_thinking` when a request omits it |
| `server.default_thinking_budget_tokens` | `KILN_DEFAULT_THINKING_BUDGET_TOKENS` | unlimited | Default maximum generated tokens before Kiln closes an open thinking block. The environment value must be a non-negative base-10 integer or `unlimited`; malformed values stop startup |
| `server.default_thinking_budget_ms` | `KILN_DEFAULT_THINKING_BUDGET_MS` | unlimited | Default decode-time budget before Kiln closes an open thinking block. The environment value must be a non-negative base-10 integer or `unlimited`; malformed values stop startup |
| `server.fold_reasoning_into_content` | `KILN_FOLD_REASONING_INTO_CONTENT` | false | Also copy separated reasoning into chat `content` for compatibility |
| `memory.inference_memory_fraction` | — | 0.7 | VRAM fraction for inference vs training |
| `memory.kv_cache_fp8` | `KILN_KV_CACHE_FP8` | false | FP8 KV cache (2x context length) |
| — | `KILN_MEMORY_RECLAIM_MODE` | `off` | Device-pool reclaim policy: `off`, `on-demand`, or `automatic` |
| `logging.format` | `KILN_LOG_FORMAT` | auto | `auto` (default; pretty on TTY, JSON otherwise), `json`, `pretty`, `text`, or `human` |
| `prefix_cache.enabled` | `KILN_PREFIX_CACHE_ENABLED` | true | Reuse KV cache for shared prefixes |
| `prefix_cache.max_blocks` | `KILN_PREFIX_CACHE_MAX_BLOCKS` | auto | Cap retained KV blocks for shared prefixes (auto = 50% of KV block pool) |
| `prefix_cache.max_entries` | `KILN_PREFIX_CACHE_MAX_ENTRIES` | auto | Cap cached GDN state snapshots (~49 MiB each; auto budget ≤1 GiB) |
| `request_log.enabled` | `KILN_REQUEST_LOG_ENABLED` | true | Durable JSONL request/response log for the inference endpoints |
| `request_log.dir` | `KILN_REQUEST_LOG_DIR` | `<adapter_dir>/.requests` | Request log directory (rotated + gzipped, retention-capped) |

Kiln resolves the logging table and its environment overrides before full
configuration validation. Every file-read, TOML, environment, or validation
error therefore emits `configuration_load_failed` with the selected config path
and complete error chain in the configured pretty/JSON format before exiting.

Streaming response channels are serviced by a fair delivery worker outside the
compute actor. A full or disconnected client cannot park peer decode or control
commands; final tokens remain ordered before `Done` or `Error`, and only the
affected request retains its KV slot during the configured grace. Delivery
acknowledgements from one decode forward are published to the actor atomically,
so response handling cannot fragment a wide batch into per-row forwards.
Current in-flight, backpressured, and pending-terminal counts are reported by
`/health`, `/v1/debug/model-state`, and the
`kiln_batching_engine_response_delivery_*` Prometheus gauges.

`/v1/config` and `/health` expose `decode_runtime` diagnostics with the
deterministic value and source plus the configured, backend-policy, and final
effective decode ceilings. The effective source is one of `backend_policy`,
`config_file`, `environment`, `deterministic`, or `max_batch_tokens`. This makes
an intentional reproducibility run distinguishable from an accidentally
serialized throughput run without inspecting process environment.

The current deterministic serving contract is deliberately narrower than
cross-backend bitwise determinism: it removes concurrent decode-shape variation
and exposes one immutable selector to tensor/kernel implementations. It does not
automatically set CUDA library controls such as `CUBLAS_WORKSPACE_CONFIG`, and
not every tolerance-bounded backward kernel consumes that selector yet. Treat
exact training replay and cross-device equality as separate qualification gates.

Real-model serving initializes paged prefill ownership without running an
unbounded prompt forward. Each actor cycle reserves one token for every ready
decode row, then advances partial prefills round-robin within both the remaining
`server.max_batch_tokens` budget and the independent
`server.max_prefill_tokens_per_cycle` ceiling. Within that token chunk, prefill
yields after at most `server.max_prefill_layers_per_cycle` transformer layers;
the actor retains the intermediate hidden and position state, runs the next
ready decode cohort, and resumes without repeating completed layers. This layer
quantum controls latency without multiplying full-model prompt passes as a
smaller token chunk can. The actor charges a chunk's token width once, when it
selects that chunk; later layer groups resume the same width without competing
for a second new-token budget, shrinking the chunk, or replaying completed
layers. Three of every four prefill dispatches remain round-robin. The fourth
may accelerate the shortest remaining prompt tail only when it is no more than
four token chunks and its admission-time prompt work is strictly smaller than
another eligible active row's. Comparing immutable work classes keeps an
all-equal cohort aligned instead of creating an artificial readiness staircase,
while a genuinely shorter interactive request can receive the bounded extra
service even on its ordinary turn; the round-robin lane retains 75% of dispatch
capacity.
Cancellation and
shutdown release partial KV ownership
only after the backend synchronization boundary; an unsettled device failure is
quarantined instead of recycling pages. `/health` and `/v1/debug/model-state`
expose `active_prefill`, both effective budgets and their sources, the last token
and layer quantum, cumulative processed layers, and actual inter-layer yields.
Prometheus exports the corresponding `kiln_batching_engine_active_prefill`,
`kiln_batching_engine_max_batch_tokens`,
`kiln_batching_engine_max_prefill_tokens_per_cycle`,
`kiln_batching_engine_max_prefill_layers_per_cycle`,
`kiln_batching_engine_last_prefill_tokens`, and
`kiln_batching_engine_{last_prefill_layers,prefill_layers_total,prefill_layer_yields_total}`
series. Bounded short-tail service is counted by
`kiln_batching_engine_short_prefill_priority_forwards_total`. Admission,
bounded-prefill, and decode-forward wall time is also
available as cumulative, process-maximum, and 100 ms slow-phase counters under
`kiln_batching_engine_{admission,prefill_forward,decode_forward}_*`.
The same values appear in health and debug snapshots. A phase crossing 100 ms
emits one structured `slow_batching_actor_phase` event with the bounded phase
name and work size, which lets qualification correlate a token gap without
logging ordinary forwards or request content.

Device-pool reclaim is disabled by default because CUDA and ROCm reclaim hooks
may synchronize the accelerator. `on-demand` permits explicit startup and
training reclaim calls but does not start a timer. `automatic` also enables the
background pressure monitor and is experimental. Invalid values stop startup;
the resolved mode and automatic-monitor state are reported under
`/health.decode_runtime.memory_governor`.

Kiln boots with the built-in `Qwen3.5-4B` defaults profile. That profile
preserves Qwen3.5-4B's official chat-template thinking default for ordinary
serving: assistant turns start in thinking mode unless `enable_thinking=false`
is passed. In eval mode, the same profile injects
`chat_template_kwargs.enable_thinking=false` unless a request explicitly
overrides it, so tool-agent evals and Pi-style loops get final `content`
instead of long `reasoning_content`. Operators can also set
`server.default_thinking_enabled = false` or
`KILN_DEFAULT_THINKING_ENABLED=false` for non-eval serving. The legacy
`KILN_DEFAULT_NO_THINK` env var is still accepted as a compatibility alias.

The normative wire schema and executable cross-runtime vectors are in the
[Thinking Budget Contract](docs/THINKING_BUDGET_CONTRACT.md).

Thinking can also be bounded without disabling it. Set
`thinking_budget_tokens` and/or `thinking_budget_ms` on
`POST /v1/chat/completions` or `POST /v1/completions/batch`:

```json
{
  "messages": [{"role": "user", "content": "Solve this carefully."}],
  "max_tokens": 512,
  "thinking_budget_tokens": 128,
  "thinking_budget_ms": 3000
}
```

An omitted field inherits its corresponding server default; explicit `null`
means unlimited for that dimension, even when the server has a default. `0`
closes thinking immediately. When both limits are set, the first one reached
wins. The time budget starts when the first decode candidate is ready, so queue
and prefill time do not consume it, and it is checked between generated tokens.
If the model emits `</think>` naturally first, Kiln leaves it alone.
The Playground reads the effective server defaults from `GET /v1/config` and
previews the resolved token/time pair before send, with each dimension marked
as coming from the server or the request.

When a budget applies to an open thinking block, Kiln validates closure before
decode. The effective `max_tokens` value, after any context-window clamp, must
fit the active tokenizer's complete `</think>` token sequence. A smaller value
returns an invalid-request error; a value equal to the close length leaves no
room for an answer, so reserve additional tokens for visible output. Any `stop`
string that can match, contain, or overlap all or part of `</think>` is also
rejected because it could terminate generation before the forced close enters
KV history atomically. A budget larger than the completion limit is valid:
Kiln reserves the final slots for the close and reports trigger `max_tokens`.

On exhaustion, Kiln feeds the forced close-tag tokens into the model context and
continues decoding the final answer. Those tokens count toward `max_tokens` and
completion usage just like model-generated close-tag tokens. Budgets are inert
when the rendered prompt did not start inside a thinking block. Time-budgeted
requests bypass deterministic completion caches because their boundary depends
on runtime speed; token-budgeted requests remain cacheable under a budget-aware
cache key. Both server defaults are unlimited when omitted.

Non-streaming chat choices and batch items include a `thinking_budget` outcome
when a budget was applied:

```json
{
  "triggered": true,
  "trigger": "tokens",
  "closed": true,
  "thinking_tokens": 128,
  "thinking_time_ms": 742
}
```

`trigger` is `tokens`, `time`, or `max_tokens`; the last value means Kiln
reserved the remaining completion slots for an atomic close. A natural close
has `triggered=false` and `closed=true`. Chat and batch
`metadata.thinking_budget` report the effective request-wide limits and whether
each dimension came from the request (`request`), a server default
(`server_default`), explicit request unlimited (`request_unlimited`), or no
configured limit (`unlimited`). Batch outcomes remain completion-specific at
`completions[].thinking_budget`; Kiln does not report a misleading aggregate
trigger or close state at the batch root. For SSE, chat metadata and the final
outcome are attached to the chunk containing `finish_reason`, immediately before
the optional usage chunk and `[DONE]`. Cached responses recompute configuration
provenance from the current request while preserving the original per-completion
outcome. Durable request-log reassembly retains the same chat
`metadata.thinking_budget` fields and restores the outcome at
`response.choices[0].thinking_budget`, so streamed and non-streamed rows use the
same mining paths.

Every completed chat record returned by `/v1/stats/recent-requests` also has a
`thinking_budget` object. It carries `configured`, the effective `max_tokens`
and `max_time_ms` when finite, independent `tokens_source` and `time_source`,
and `applied`. Once a terminal budget status exists it also carries
`triggered`, `trigger`, `closed`, `thinking_tokens`, and `thinking_time_ms`.
Outcome fields stay absent for an inert budget instead of resembling a natural
close; `applied` stays absent only when a failure happened before applicability
could be established. The dashboard request drill renders this configuration,
provenance, and outcome directly.

`/metrics` exports the same recorded-chat population without unbounded labels:

- `kiln_thinking_budget_source_total{dimension,source}` counts token/time
  provenance. `source` is limited to `request`, `server_default`,
  `request_unlimited`, `unlimited`, or the defensive `unknown` fallback.
- `kiln_thinking_budget_outcomes_total{outcome}` assigns exactly one of
  `unconfigured`, `inert`, `natural_close`, `tokens`, `time`, `max_tokens`,
  `unclosed`, `interrupted`, or `unresolved` to each recorded chat completion.
- `kiln_thinking_budget_effective_tokens` and
  `kiln_thinking_budget_effective_seconds` are fixed-bucket histograms of
  finite effective limits. Request-supplied numbers never become labels.

The Qwen3.5-4B profile expects adapters in `model.adapter_dir` when configured,
otherwise `<model.path>/adapters`. Chat-template loading prefers
`chat_template.jinja` next to the tokenizer and falls back to the
`tokenizer_config.json` `chat_template` field; the supported template behavior
includes `chat_template_kwargs.enable_thinking` and OpenAI-style tool calls.

When the model emits reasoning separately, chat responses expose it as
`choices[].message.reasoning_content` while `content` contains only the final
answer. If final answer content is empty, response `metadata` includes
`final_content_empty=true` and a `content_empty_reason` such as
`reasoning_without_final_content`. Clients that cannot handle an empty
`content` field can set request `fold_reasoning_into_content=true`, or server
`fold_reasoning_into_content = true`, to duplicate the reasoning block into
`content` while still keeping `reasoning_content` available.

## Security model

Kiln has no built-in auth. The default listen address is `127.0.0.1:8420` so a fresh install isn't reachable from the network. To accept remote connections, set `server.host = "0.0.0.0"` (or `KILN_HOST=0.0.0.0`) and front kiln with a reverse proxy (nginx, Caddy) that adds auth, or run it on a private network (WireGuard, Tailscale).

**Training data is privileged.** Kiln applies a faithful gradient update to anything you POST to `/v1/train/sft` or `/v1/train/grpo` — it validates structure, not semantics. A poisoned training example will permanently influence the active adapter until you unload or reset it. Treat your training corpus as security-sensitive: do not accept training data from untrusted sources, and review examples before submission the same way you would review code before merging it.

Adapters are easy to revert if a bad training run lands. `POST /v1/adapters/unload` deactivates the current adapter; after it is no longer physically loaded, `DELETE /v1/adapters/{name}` removes it from disk. The base model is unaffected — only LoRA deltas are written.

Completed training runs also write `adapter_manifest.json` beside the adapter
weights. The manifest records adapter/config/receipt hashes, parent adapter,
model config hash, the exact base-weight shard manifest, kiln commit, and
training data hash. Use
`kiln adapters restore <path>/adapter_manifest.json --adapter-dir <registry>`
to copy an adapter into a registry and verify hashes after copy. See
[`docs/ADAPTER_MANIFEST.md`](docs/ADAPTER_MANIFEST.md) for the schema.

Full v0.1 threat model and per-finding analysis: [`docs/audits/security-audit-v0.1.md`](docs/audits/security-audit-v0.1.md).

## Desktop App

Kiln Desktop is a system-tray app that wraps the `kiln` server for people who don't want to use a CLI. It spawns and supervises the `kiln` binary as a child process, shows server status in the tray, and opens a dashboard, settings, and log viewer in native windows.

**Windows, Linux, and macOS (Apple Silicon).** Windows drives the CUDA build of `kiln`; Linux chooses CUDA on NVIDIA systems and Vulkan on AMD/Intel systems; macOS drives the candle-metal build on M-series hardware. Intel Macs are not supported.

**Download — [Kiln Desktop v0.2.16](https://github.com/ericflo/kiln/releases/tag/desktop-v0.2.16):**

**Release note:** Desktop and server binaries use separate GitHub release tags/version numbers. `desktop-v0.2.16` is the latest Desktop release; it downloads and verifies the matching server binary from the latest `kiln-v*` release line, so the version split is intentional.

See **[desktop/CHANGELOG.md](desktop/CHANGELOG.md)** for the full version history.

| Platform | Installer | Size |
|---|---|---|
| macOS (Apple Silicon) | [Kiln.Desktop_0.2.16_aarch64.dmg](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_aarch64.dmg) | 8.5 MB |
| Windows | [Kiln.Desktop_0.2.16_x64-setup.exe](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_x64-setup.exe) (NSIS) | 4.5 MB |
| Windows | [Kiln.Desktop_0.2.16_x64_en-US.msi](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_x64_en-US.msi) (MSI) | 6.8 MB |
| Linux | [Kiln.Desktop_0.2.16_amd64.deb](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_amd64.deb) | 8.8 MB |
| Linux | [Kiln.Desktop_0.2.16_amd64.AppImage](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.16/Kiln.Desktop_0.2.16_amd64.AppImage) | 85.7 MB |

The installer bundles the desktop wrapper only. On first launch the app offers to auto-download the matching prebuilt `kiln` server binary for your platform (macOS aarch64 / Metal, Linux x86_64 / CUDA 12.4 or Vulkan, Windows x86_64 / CUDA 12.4) from the latest `kiln-v*` GitHub release and verify it against the published SHA-256. The terminal-first server release matrix also includes a Linux x86_64 ROCm 7.2.4 artifact; you can point the app at an existing `kiln` binary from Settings. Model weights still need to be downloaded separately — the Settings window has a HuggingFace downloader, or you can use the CLI path in [QUICKSTART.md](QUICKSTART.md).

**Dashboard** — a toolbar across the top surfaces server state, model path, VRAM usage, active LoRA adapter, training status, and the OpenAI base URL as click-to-copy pills, alongside Start / Stop / Restart Server, View Logs, and Settings buttons. A first-run empty state walks you through setting a model path, and if the kiln server crashes while the dashboard is open an error screen surfaces it with a one-click recovery path. Keyboard shortcuts cover the common actions — <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>S</kbd> to start, <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>.</kbd> to stop, <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>R</kbd> to restart, <kbd>Ctrl/Cmd</kbd>+<kbd>Shift</kbd>+<kbd>C</kbd> to copy the base URL, <kbd>Ctrl/Cmd</kbd>+<kbd>L</kbd> for logs, <kbd>Ctrl/Cmd</kbd>+<kbd>,</kbd> for settings, and <kbd>?</kbd> for the full cheatsheet modal. The toolbar wraps gracefully at narrow window widths, and the kiln server's `/ui/` is embedded below.

![Dashboard](docs/desktop/dashboard.png)

**Settings** — pick a model path (or pull weights with the built-in HuggingFace downloader), configure the listening host and port, and tune the runtime: inference VRAM fraction, FP8 KV cache, CUDA graphs, prefix cache, and adapter directory. Startup options cover auto-start kiln on app launch, auto-restart on crash, and launch-at-login. A Check for Updates button (with the same check running automatically on launch) surfaces new `kiln-v*` releases and explains when an update is held back for CUDA driver or GPU SM-arch compatibility reasons.

![Settings](docs/desktop/settings.png)

**Logs** — tails the kiln server's stdout/stderr from an in-process ring buffer.

![Logs](docs/desktop/logs.png)

Build and dev docs for the desktop app live in [desktop/README.md](desktop/README.md).

## Deployment

### Docker — pull a prebuilt image

Each `kiln-v*` tag publishes a `linux/amd64` CUDA 12.4 image to GHCR:

```bash
docker pull ghcr.io/ericflo/kiln-server:latest
# or pin the current latest version programmatically:
KILN_VERSION=$(curl -fsSL https://api.github.com/repos/ericflo/kiln/releases/latest | sed -n 's/.*"tag_name": "kiln-v\([^"]*\)".*/\1/p')
docker pull ghcr.io/ericflo/kiln-server:${KILN_VERSION}
```

Run with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html):

```bash
docker run --gpus all -p 8420:8420 \
  -e KILN_MODEL_PATH=/models/Qwen3.5-4B \
  -v /path/to/Qwen3.5-4B:/models/Qwen3.5-4B:ro \
  ghcr.io/ericflo/kiln-server:latest serve
```

### Docker — build from source

```bash
docker build -f deploy/Dockerfile -t kiln .
docker run --gpus all -p 8420:8420 \
  -e KILN_MODEL_PATH=/models/Qwen3.5-4B \
  -v /path/to/Qwen3.5-4B:/models/Qwen3.5-4B:ro \
  kiln serve
```

### systemd

```bash
sudo cp target/release/kiln /usr/local/bin/
sudo cp deploy/kiln.service /etc/systemd/system/
sudo systemctl enable --now kiln
```

## Status

Kiln v0.1.0 shipped on 2026-04-19 and the current release line follows the latest `kiln-v*` GitHub release. Phases 1–10 are shipped or chapter-closed: core inference, LoRA serving, SFT and GRPO training over HTTP, production hardening, the Phase 6 performance sprint (FP8 KV cache, CUDA graphs, GPTQ + Marlin W4A16 quantization, fused decode kernels, SGLang-style radix prefix cache), Phase 7 developer experience, Phase 8 advanced features (adapter upload/download, TIES + concatenation merge modes, per-request adapter composition, batch completions for GRPO, training webhooks), Phase 9 public-release prep (Sigstore-signed provenance, GHCR image, signed binaries for Linux/macOS/Windows), and Phase 10 Liger-style long-context training kernels (closed by [`docs/audits/PHASE10_CLOSURE.md`](docs/audits/PHASE10_CLOSURE.md)). Inference on macOS / Apple Silicon runs via the candle-metal backend, with a fused Metal kernel family landed in v0.2.0. Phase 11 — onboarding, the first-class eval system (suites, scorers, dataset → eval synthesis, judgment flywheel, post-training auto-eval), and the dashboard overhaul (drill-in modals, live loss curves, A/B compare, ⌘K command palette) — is the active line. See [`CHANGELOG.md`](CHANGELOG.md) for what landed in the most recent release and [`BENCHMARKS.md`](BENCHMARKS.md) for the current serving acceptance protocol and historical results.

Not yet production-hardened for multi-tenant use. Designed for single-user, single-GPU deployments — your home server, your dev box, your dedicated cloud instance.

## Prior Art

Kiln builds on ideas from:

- [vLLM](https://github.com/vllm-project/vllm) — paged KV cache, continuous batching
- [DeepSeekMath](https://arxiv.org/abs/2402.03300) — GRPO algorithm
- [S-LoRA](https://arxiv.org/abs/2311.03285) — multi-LoRA serving techniques
- [Tinker](https://thinkingmachines.ai/blog/announcing-tinker/) — the cloud-hosted version of this idea. Kiln is the self-hosted, open-source take.
- [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) — proof that the core can be simple

## Contributing

Bug reports, performance work, kernel ports, documentation, and dev-experience polish are all welcome. Issues live at [github.com/ericflo/kiln/issues](https://github.com/ericflo/kiln/issues); read [CONTRIBUTING.md](CONTRIBUTING.md) before opening a non-trivial PR — Kiln is a deliberate scalpel (Qwen3.5-4B only, single-binary, no Python sidecar) and a 5-minute scope conversation up front saves a 5-day rewrite later. Serving-performance changes should attach the comparable receipt described in [`BENCHMARKS.md`](BENCHMARKS.md); isolated kernel changes should include their focused microbenchmark and cite the closed-PR history first.

Maintained by [@ericflo](https://github.com/ericflo). MIT-licensed.

## License

MIT — see [LICENSE](LICENSE).

Third-party dependency licenses: see [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
