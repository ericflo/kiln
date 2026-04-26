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

**Download — [Kiln Desktop v0.2.0](https://github.com/ericflo/kiln/releases/tag/desktop-v0.2.0):**

| Platform | Installer | Size |
|----------|-----------|------|
| macOS (Apple Silicon) | [Kiln.Desktop_0.2.0_aarch64.dmg](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.0/Kiln.Desktop_0.2.0_aarch64.dmg) | 8.1 MB |
| Windows | [Kiln.Desktop_0.2.0_x64-setup.exe](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.0/Kiln.Desktop_0.2.0_x64-setup.exe) (NSIS) | 4.3 MB |
| Windows | [Kiln.Desktop_0.2.0_x64_en-US.msi](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.0/Kiln.Desktop_0.2.0_x64_en-US.msi) (MSI) | 6.5 MB |
| Linux | [Kiln.Desktop_0.2.0_amd64.deb](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.0/Kiln.Desktop_0.2.0_amd64.deb) | 8.4 MB |
| Linux | [Kiln.Desktop_0.2.0_amd64.AppImage](https://github.com/ericflo/kiln/releases/download/desktop-v0.2.0/Kiln.Desktop_0.2.0_amd64.AppImage) | 81.7 MB |

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

  Version: 0.2.1
  Mode:    GPU inference
  Model:   ./Qwen3.5-4B
  CUDA:    available ✓
  GPU:     NVIDIA RTX A6000
  VRAM:    49140 MiB total, 48891 MiB free
  Listen:  http://0.0.0.0:8420

  Endpoints: /v1/chat/completions, /v1/train/sft, /health, /metrics
```

The `GPU` and `VRAM` lines come from `nvidia-smi` and are skipped silently if it isn't installed. If you launched with `--config kiln.toml`, a `Config:` line appears just below `Version:`.

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
