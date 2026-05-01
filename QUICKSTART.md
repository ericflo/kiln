# Quickstart: Zero to Running in 5 Minutes

This guide gets you from a fresh machine to running Kiln with real GPU inference and live training.

## Prerequisites

Either an NVIDIA GPU **or** an Apple Silicon Mac.

- **NVIDIA path**: GPU with 24GB+ VRAM (RTX 3090, RTX 4090, A6000, etc.) and CUDA 12.0+ (`nvcc --version` to check)
- **Apple Silicon path**: M-series Mac with 16GB+ unified memory and Xcode Command Line Tools installed (`xcode-select --install`). Full Xcode is **not** required — `candle-metal-kernels` JIT-compiles MSL shaders at runtime.
- **Rust**: Stable toolchain (`curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`)
- **Disk**: ~20GB free (model weights + build artifacts)

## Quick path: Desktop App (recommended for most users)

Kiln Desktop ships prebuilt installers for Windows, Linux, and macOS. The installer bundles only the GUI wrapper — on first launch it auto-downloads the matching prebuilt `kiln` server binary for your platform and verifies it against the published SHA-256. **No Rust toolchain or CUDA build required for the GUI itself.**

**Download — [Kiln Desktop v0.2.2](https://github.com/ericflo/kiln/releases/tag/desktop-v0.2.2):**

| Platform | Installer | Size |
|----------|-----------|------|
| macOS (Apple Silicon) | [Kiln.Desktop_0.2.2_aarch64.dmg](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.2/Kiln.Desktop_0.2.2_aarch64.dmg) | 8.5 MB |
| Windows | [Kiln.Desktop_0.2.2_x64-setup.exe](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.2/Kiln.Desktop_0.2.2_x64-setup.exe) (NSIS) | 4.5 MB |
| Windows | [Kiln.Desktop_0.2.2_x64_en-US.msi](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.2/Kiln.Desktop_0.2.2_x64_en-US.msi) (MSI) | 6.8 MB |
| Linux | [Kiln.Desktop_0.2.2_amd64.deb](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.2/Kiln.Desktop_0.2.2_amd64.deb) | 8.8 MB |
| Linux | [Kiln.Desktop_0.2.2_amd64.AppImage](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.2/Kiln.Desktop_0.2.2_amd64.AppImage) | 85.7 MB |

Continue with section 1 (Build Kiln) only if you want to build from source, contribute, or use the CLI directly. Otherwise, install the desktop app, let it finish its first-launch downloads, then skip ahead to section 4 (Test Inference) once the server is running.

## 1. Build Kiln

```bash
git clone https://github.com/ericflo/kiln.git
cd kiln
```

**Linux / Windows + NVIDIA:**

```bash
cargo build --release --features cuda
```

The first build takes 15-30 minutes (CUDA kernel compilation). Subsequent builds are fast.

**macOS + Apple Silicon:**

```bash
cargo build --release --features metal
```

No nvcc needed — Xcode Command Line Tools are sufficient. `candle-metal-kernels` JIT-compiles Metal Shading Language at runtime, so there's no Xcode IDE dependency.

The binary is at `target/release/kiln`.

## 2. Download Model Weights

Kiln targets **Qwen3.5-4B**. Download the weights from Hugging Face:

```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B
```

This downloads ~8GB of safetensors weights plus the tokenizer.

## 3. Start the Server

```bash
./target/release/kiln serve --config kiln.example.toml
```

But first, tell Kiln where the model is. Set the model path via environment variable:

```bash
KILN_MODEL_PATH=./Qwen3.5-4B ./target/release/kiln serve
```

Or create a `kiln.toml` config file:

```toml
[model]
path = "./Qwen3.5-4B"

[server]
port = 8420
```

Then start with:

```bash
./target/release/kiln serve --config kiln.toml
```

By default, Kiln logs in colored "pretty" format when stderr is an interactive
terminal and switches to structured JSON when stderr is piped or redirected
(systemd, docker, CI). To force a specific format, set `KILN_LOG_FORMAT=json`
or `KILN_LOG_FORMAT=pretty` (or the equivalent `[logging] format = "..."` in
`kiln.toml`).

You'll see the startup banner:

```
  ┌─────────────────────────────────────┐
  │           🔥 K I L N 🔥             │
  │   inference · training · adapters   │
  └─────────────────────────────────────┘

  Version: 0.2.9
  Mode:    GPU inference
  Model:   ./Qwen3.5-4B
  CUDA:    available ✓
  GPU:     NVIDIA RTX A6000
  VRAM:    49140 MiB total, 48891 MiB free
  Listen:  http://127.0.0.1:8420

  Endpoints: /v1/chat/completions, /v1/train/sft, /health, /metrics
```

The `GPU` and `VRAM` lines come from `nvidia-smi` and are skipped silently if it isn't installed. If you launched with `--config kiln.toml`, a `Config:` line appears just below `Version:`.

Kiln binds to loopback (`127.0.0.1`) by default so a fresh install isn't reachable from the network. To accept connections from other hosts, set `server.host = "0.0.0.0"` in your TOML config or `KILN_HOST=0.0.0.0` and put Kiln behind a trusted reverse proxy (auth is out of scope for v0.1).

**Training endpoints are privileged.** `/v1/train/sft` and `/v1/train/grpo` apply a faithful gradient update to whatever structurally-valid examples you POST — kiln does not validate the *content* of training data. A poisoned example will permanently influence the active adapter until you unload it. Do not expose training endpoints to untrusted inputs, and treat your training corpus as security-sensitive. See the README's [Security model](README.md#security-model) section for the full picture.

On Apple Silicon, Kiln auto-detects Metal and logs `Metal available — using Apple Silicon GPU` at startup instead of the CUDA availability line. No config changes needed — the binary selects the backend that was compiled in.

## 4. Test Inference

Send a chat completion request (OpenAI-compatible):

```bash
curl -s http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 64,
    "temperature": 0.7
  }' | python3 -m json.tool
```

With streaming:

```bash
curl -N http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Write a haiku about rust programming"}],
    "max_tokens": 64,
    "stream": true
  }'
```

### 4.1 Tool / Function Calling

Kiln speaks the OpenAI `tools` schema on the request side. The server forwards
`tools` and `tool_choice` into the chat-template Jinja context, which means
Qwen3.5-4B's bundled template renders its `<tools>` schema block and the
tool-calling prelude verbatim — so the model sees your function definitions at
inference time. This is what PRs [#632](https://github.com/ericflo/kiln/pull/632)
and [#653](https://github.com/ericflo/kiln/pull/653) wired up.

**Tool call request.** Pass `tools: [...]` exactly as you would to the OpenAI
API:

```bash
curl -s http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"}
    ],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the current weather for a city",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "auto",
    "max_tokens": 256,
    "temperature": 0.0
  }' | python3 -m json.tool
```

**Where the tool call shows up in the response.** Qwen3.5-4B's official chat
template (the one that ships with the tokenizer) instructs the model to emit
tool calls in an **XML** form, not as a structured JSON object. The model's
output ends up inside `choices[0].message.content` and looks like this:

```
<tool_call>
<function=get_weather>
<parameter=location>
San Francisco
</parameter>
</function>
</tool_call>
```

Your client is responsible for parsing this XML out of `content`. Note the
asymmetry vs. the OpenAI Chat Completions spec: today kiln returns
`choices[0].message.tool_calls = null` even when the model produced a tool
call — the structured-output path is a known gap. (See
`crates/kiln-server/src/api/completions.rs` around line 1370 and line 2010 if
you want to confirm in source.) On the request side you already use the
OpenAI-shape `tool_calls` array; on the response side you parse the XML.

**Tool result follow-up turn.** When you submit the tool's result back to
the model, use the OpenAI shape — kiln deserializes it cleanly and the
template renders it as a `<tool_response>...</tool_response>` block. Note
that `tool_calls[*].function.arguments` is a **JSON-encoded string** (this
matches OpenAI's spec; see test
`message_tool_calls_round_trip_preserved` at
`crates/kiln-server/src/api/completions.rs:2496` for the canonical shape):

```bash
curl -s http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the weather in San Francisco?"},
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "call_001",
            "type": "function",
            "function": {
              "name": "get_weather",
              "arguments": "{\"location\":\"San Francisco\"}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "call_001",
        "name": "get_weather",
        "content": "{\"temp_f\": 62, \"conditions\": \"foggy\"}"
      }
    ],
    "max_tokens": 128,
    "temperature": 0.0
  }' | python3 -m json.tool
```

The assistant's next turn comes back as plain text in
`choices[0].message.content` ("It's currently 62°F and foggy in San
Francisco.").

> **Tool call format reference.** Qwen3.5-4B's chat template — including the
> exact XML tool-call grammar the model is trained to emit — lives in the
> tokenizer config at
> [`Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B/blob/main/chat_template.jinja)
> on Hugging Face. If a tool call format ever looks ambiguous, that file is
> the source of truth.

### 4.2 Known limitation: workers=1 for long tools-bearing prompts

For tools-bearing chat completions with input prompts longer than roughly
30k tokens, run your client with **at most one in-flight request at a
time** (`workers=1`). This is a client-side workaround — there is no
kiln-side flag to set; you control it by serializing requests in your
driver, worker pool, or load generator.

If you push two or more concurrent in-flight prefills under that workload,
a single 300-second prefill timeout (the default
`server.request_timeout_secs` in [`kiln.example.toml`](kiln.example.toml))
can leave kiln in a degraded prefill state. Subsequent requests then fail
with HTTP 500:

```
{"error":{"code":"generation_error","message":"Text generation failed: prefill forward pass (paged) failed: ..."}}
```

…until the server is restarted. With `workers=1` there is no concurrent
prefill, no degraded state, and the failure mode does not occur.

The cascade and its symptoms are tracked in
[issue #664](https://github.com/ericflo/kiln/issues/664); the underlying
KV-cache-exhaustion / prefill-state-cleanup investigation is in
[issue #656](https://github.com/ericflo/kiln/issues/656). This workaround
is expected to be removed once the prefill-watchdog and timeout-cleanup
fixes from #664 land.

### 4.3 Known limitation: long-prefill workloads on kiln-v0.2.9

On kiln-v0.2.9, long-prefill workloads (input prompts of roughly 20k
tokens or more) under `workers=2` with a tight KV-cache cap can hit
repeated HTTP 408 prefill timeouts at exactly the
`server.request_timeout_secs` boundary (default 300 seconds). The same
byte-identical inputs ran cleanly under v0.2.8 and under workers=1 on
v0.2.9, so this is a regression specific to the v0.2.9 long-prefill
path under concurrent prefill pressure.

Two workarounds, pick whichever fits your driver:

- **Workaround A (recommended): run with `workers=1`.** Same client-side
  serialization pattern as §4.2 — there is no kiln-side flag, you
  control it by keeping at most one in-flight request in your driver,
  worker pool, or load generator. With no concurrent prefill the
  regression does not trigger.
- **Workaround B: raise `server.request_timeout_secs` to at least 600.**
  If `workers=2` is required, bump `server.request_timeout_secs` in
  [`kiln.example.toml`](kiln.example.toml) to `600` or higher. The
  v0.2.8 round-6 p99 prefill latency on the same workload was at or
  below 300 seconds under loose KV; budget roughly 2× that headroom
  (≥600s) for the tighter KV cap on v0.2.9.

The diagnosis is tracked in
[issue #686](https://github.com/ericflo/kiln/issues/686) and the bisect
audit lives at
[`docs/audits/PHASE11_ISSUE_686_BISECT.md`](docs/audits/PHASE11_ISSUE_686_BISECT.md).
This workaround is expected to be removed once the prefill fix lands in
kiln-v0.2.10.

### 4.4 Troubleshooting: KV cache OOM auto-recovery

At startup, kiln's auto-sizer computes the paged KV cache budget as
`available_for_kv = (total_vram - model_bytes) * inference_memory_fraction`
(default `inference_memory_fraction = 0.7`). On A40 / A6000 / RTX 6000 Ada /
L40S running Qwen3.5-4B in BF16, the configured fraction occasionally OOMs at
allocation time even though the math says it should fit — driver overhead,
fragmentation, and other allocations on the device can shrink the actual free
pool below the auto-sizer's estimate.

Since kiln-v0.2.9 (PR #687), the auto-sizer **automatically retries with a
shrinking fraction** from the fallback list `[0.75, 0.65, 0.55, 0.45]` (only
values strictly below the configured fraction are attempted). On success after
a fallback, kiln logs a `WARN` line naming the actual fraction it landed on:

```
WARN kiln_server::state: KV cache auto-sizer fell back to a smaller
inference_memory_fraction because the configured value OOM'd; set
memory.inference_memory_fraction (or KILN_INFERENCE_MEMORY_FRACTION) to
this value to silence the warning configured_fraction=0.99
actual_fraction=0.45 num_blocks=56652 attempts=5
```

**What to do on the next restart:** pin the `actual_fraction` value so the
auto-sizer doesn't have to retry. Either set `memory.inference_memory_fraction`
in [`kiln.example.toml`](kiln.example.toml), or export
`KILN_INFERENCE_MEMORY_FRACTION=<value>` in your environment. Subsequent
restarts will hit the configured fraction on the first attempt and the WARN
line goes away.

If **every** fraction in the fallback list also OOMs, kiln aborts startup with
a structured panic message that lists each attempted fraction → `num_blocks` →
exact OOM error, names the detected GPU and model size, and recommends two
concrete remediations:

- **(a) `KILN_NUM_BLOCKS=N`** (or `[memory] num_blocks = N` in
  `kiln.example.toml`) — preferred. This bypasses the auto-sizer entirely and
  pins an exact block count, which is the canonical workaround documented in
  [issue #685](https://github.com/ericflo/kiln/issues/685). The panic message
  prints a concrete `N` value sized at ~30% of remaining VRAM (well below the
  retry floor, with headroom for driver overhead).
- **(b) `KILN_INFERENCE_MEMORY_FRACTION=X`** (or
  `[memory] inference_memory_fraction = X`) — equivalent fraction-based knob,
  if you'd rather size by ratio than by absolute block count.

Use (a) on first-run failures and you are unlikely to see this codepath again.

## 5. Open the Browser Dashboard

Kiln ships with an embedded web dashboard. Open [http://localhost:8420/ui](http://localhost:8420/ui) in any browser:

- **Server Status** — live GPU VRAM breakdown (model / KV cache / training / free) plus scheduler queue depth
- **Adapters** — list available LoRA adapters and switch the active one
- **Training** — submit SFT or GRPO jobs from a form, watch the queue, and review recently completed runs
- **Quick Inference** — chat with the model directly (per-request adapter and temperature pickers) without writing curl

It's a single HTML page served by the `kiln` binary itself — no extra process, no build step, no JS bundle to deploy.

## 6. Submit SFT Training

Create a training file `examples.jsonl` with chat-format examples:

```jsonl
{"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]}
{"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
{"messages": [{"role": "user", "content": "Translate 'hello' to Spanish"}, {"role": "assistant", "content": "Hola"}]}
```

Submit training via the CLI:

```bash
./target/release/kiln train sft --file examples.jsonl
```

Or via curl:

```bash
curl -s http://localhost:8420/v1/train/sft \
  -H "Content-Type: application/json" \
  -d '{
    "adapter_name": "default",
    "examples": [
      {"messages": [{"role": "user", "content": "What is the capital of France?"}, {"role": "assistant", "content": "The capital of France is Paris."}]},
      {"messages": [{"role": "user", "content": "What is 2+2?"}, {"role": "assistant", "content": "4"}]}
    ],
    "learning_rate": 1e-4,
    "num_epochs": 3
  }' | python3 -m json.tool
```

Training runs in the background. The model continues serving requests during training. When training completes, the new LoRA adapter is hot-swapped — all subsequent requests use the updated weights.

## 7. Check Training Status

Via CLI:

```bash
./target/release/kiln health
```

Via curl:

```bash
curl -s http://localhost:8420/v1/train/status | python3 -m json.tool
```

## 8. Manage Adapters

```bash
# List loaded adapters
./target/release/kiln adapters list

# Unload an adapter (revert to base model)
./target/release/kiln adapters unload default

# Reload it
./target/release/kiln adapters load default

# Delete an adapter permanently
./target/release/kiln adapters delete default
```

## 9. Phase 8 API Examples

### 9.1 Batch generation (efficient for GRPO rollouts)

`/v1/completions/batch` returns one HTTP response covering many prompts × `n` completions per prompt. The iteration-level scheduler batches the underlying prefill/decode steps, so this is far cheaper than N parallel calls. Hard cap: `prompts.len() * n <= 64`. `stream` is not supported on this endpoint. `seed` is per-batch — each completion uses `seed.wrapping_add(prompt_idx * n + completion_idx)` so identical prompts produce distinct outputs.

For the full generate→score→train loop with three worked verifiable-reward examples (math correctness, JSON-validity, code-runs), see [docs/GRPO_GUIDE.md](docs/GRPO_GUIDE.md).

```bash
curl -s http://localhost:8420/v1/completions/batch \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": [
      [{"role": "user", "content": "What is 2+2?"}],
      [{"role": "user", "content": "What is the capital of France?"}]
    ],
    "n": 4,
    "max_tokens": 32,
    "temperature": 0.7,
    "seed": 42
  }' | python3 -m json.tool
```

### 9.2 Export an adapter (download tar.gz)

```bash
curl -s -OJ http://localhost:8420/v1/adapters/default/download
# -> writes default.tar.gz to the current directory
```

The body is a streamed `application/gzip` tar archive containing the adapter directory.

### 9.3 Import an adapter (upload tar.gz)

Multipart fields: `name` (target adapter directory name) and `archive` (the tar.gz body).

```bash
curl -s http://localhost:8420/v1/adapters/upload \
  -F name=imported \
  -F archive=@default.tar.gz | python3 -m json.tool
# -> {"name":"imported","size_bytes":...,"files":...}
```

Body limit is 2 GB compressed / 4 GB extracted. Uploads fail with 409 if the target name already exists.

### 9.4 Merge adapters (TIES)

Combine multiple adapters into a new on-disk adapter. Modes: `weighted_average` (default), `ties`, `concat`. `weighted_average` and `ties` require identical rank, target_modules, and shapes; `concat` produces a higher-rank adapter (`r_total = Σᵢ rᵢ`) and accepts mismatched ranks.

```bash
curl -s http://localhost:8420/v1/adapters/merge \
  -H "Content-Type: application/json" \
  -d '{
    "sources": [
      {"name": "math", "weight": 0.6},
      {"name": "code", "weight": 0.4}
    ],
    "output_name": "math_code_ties",
    "mode": "ties",
    "density": 0.2
  }' | python3 -m json.tool
```

### 9.5 Compose adapters per-request

The standard `/v1/chat/completions` endpoint accepts a request-body `adapters` array as a Kiln extension (mutually exclusive with `adapter`). Each entry is `{"name", "scale"}`. The server merges the composed adapter once via `merge_concat`, caches it on disk under `adapter_dir/.composed/` keyed by `(name, scale)` hash, and reuses the cache on subsequent requests with the same composition. `/v1/completions/batch` accepts the same `adapters` field for the whole batch.

```bash
curl -s http://localhost:8420/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Refactor this Python loop into a comprehension"}],
    "max_tokens": 128,
    "adapters": [
      {"name": "code", "scale": 1.0},
      {"name": "style-eric", "scale": 0.5}
    ]
  }' | python3 -m json.tool
```

### 9.6 Training completion webhooks

Kiln POSTs a JSON event to a configured URL when an SFT or GRPO job reaches a terminal state. Configured via `[training] webhook_url` in `kiln.toml` or the `KILN_TRAINING_WEBHOOK_URL` environment variable (set to the empty string to clear a TOML-set value). Delivery is fire-and-forget with a 5-second timeout — webhook failures are logged but never affect job state.

```toml
# kiln.toml
[training]
webhook_url = "https://example.com/kiln-hooks/training"
```

Payload (`Content-Type: application/json`):

```json
{
  "job_id": "<uuid>",
  "job_type": "sft" | "grpo",
  "status": "completed" | "failed",
  "adapter_name": "<name>",
  "adapter_path": "<path or null>",
  "error": "<message or null>",
  "timestamp": "<RFC3339>"
}
```

## CLI Reference

```
kiln serve              Start the inference server (default)
kiln health             Check server health
kiln config             Validate a config file
kiln train sft          Submit SFT training examples
kiln train grpo         Submit GRPO training batch
kiln adapters list      List loaded adapters
kiln adapters load      Load an adapter from disk
kiln adapters unload    Unload an active adapter
kiln adapters delete    Delete an adapter

Global options:
  --config, -c <file>   Path to TOML config file
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/ui` | Embedded web dashboard (status, adapters, training, chat) |
| GET | `/health` | Server health and diagnostics |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible). Kiln extension: per-request `adapter` (single name) or `adapters: [{name, scale}, …]` for adapter composition (see [9.5](#95-compose-adapters-per-request)). |
| POST | `/v1/completions/batch` | Multi-prompt batch generation — efficient for GRPO rollouts (see [9.1](#91-batch-generation-efficient-for-grpo-rollouts)). |
| GET | `/v1/adapters` | List LoRA adapters |
| POST | `/v1/adapters/load` | Load adapter from disk |
| POST | `/v1/adapters/unload` | Unload active adapter |
| DELETE | `/v1/adapters/{name}` | Delete an adapter |
| GET | `/v1/adapters/{name}/download` | Stream adapter as `application/gzip` tar.gz (see [9.2](#92-export-an-adapter-download-targz)). |
| POST | `/v1/adapters/upload` | Import adapter from a multipart `archive` tar.gz (see [9.3](#93-import-an-adapter-upload-targz)). |
| POST | `/v1/adapters/merge` | Combine adapters via `weighted_average`, `ties`, or `concat` mode (see [9.4](#94-merge-adapters-ties)). |
| POST | `/v1/train/sft` | Submit SFT training examples |
| POST | `/v1/train/grpo` | Submit GRPO training batch |
| GET | `/v1/train/status` | Training queue status |
| GET | `/v1/train/status/{job_id}` | Individual job status |
| GET | `/v1/train/queue` | List queued training jobs |
| DELETE | `/v1/train/queue/{job_id}` | Cancel a queued job |
| (config) | `[training].webhook_url` | Fire-and-forget POST on training completion — set in `kiln.toml` or via `KILN_TRAINING_WEBHOOK_URL` (see [9.6](#96-training-completion-webhooks)). |
| GET | `/v1/config` | Current server configuration |

## Configuration

Kiln uses a TOML config file. See [`kiln.example.toml`](kiln.example.toml) for all options.

Key settings:

| Setting | Env Var | Default | Description |
|---------|---------|---------|-------------|
| `model.path` | `KILN_MODEL_PATH` | none | Path to model weights (required for real inference) |
| `server.port` | `KILN_PORT` | 8420 | Server listen port |
| `memory.inference_memory_fraction` | — | 0.7 | VRAM fraction for inference (rest for training) |
| `memory.kv_cache_fp8` | `KILN_KV_CACHE_FP8` | false | FP8 KV cache (halves memory, ~2x context) |
| `logging.format` | `KILN_LOG_FORMAT` | auto | Log format: `auto` (pretty on TTY, JSON otherwise), `json`, `pretty`, `text`, `human` |
| `prefix_cache.enabled` | — | true | Reuse KV cache for shared prefixes |

## Running with Docker

```bash
# Build the image
docker build -f deploy/Dockerfile -t kiln .

# Run with GPU access
docker run --gpus all \
  -v /path/to/Qwen3.5-4B:/models \
  -p 8420:8420 \
  kiln serve
```

## Running with systemd

See [`deploy/kiln.service`](deploy/kiln.service). Copy the binary to `/usr/local/bin/kiln`, create `/etc/kiln/kiln.toml`, and:

```bash
sudo systemctl enable --now kiln
```

## Next Steps

- **GRPO training**: Submit scored completions to `/v1/train/grpo` for reinforcement learning from feedback
- **FP8 KV cache**: Set `kv_cache_fp8 = true` in config to double your context length
- **Prefix caching**: Enabled by default — repeated prompt prefixes reuse cached KV blocks
- **Monitoring**: Scrape `/metrics` with Prometheus for request latency, throughput, and training progress
