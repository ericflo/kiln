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

**Download — [Kiln Desktop v0.1.10](https://github.com/ericflo/kiln/releases/tag/desktop-v0.1.10):**

| Platform | Installer | Size |
|----------|-----------|------|
| Windows | [Kiln.Desktop_0.1.10_x64-setup.exe](https://github.com/ericflo/kiln/releases/download/desktop-v0.1.10/Kiln.Desktop_0.1.10_x64-setup.exe) (NSIS) | 4.3 MB |
| Windows | [Kiln.Desktop_0.1.10_x64_en-US.msi](https://github.com/ericflo/kiln/releases/download/desktop-v0.1.10/Kiln.Desktop_0.1.10_x64_en-US.msi) (MSI) | 6.5 MB |
| Linux | [Kiln.Desktop_0.1.10_amd64.deb](https://github.com/ericflo/kiln/releases/download/desktop-v0.1.10/Kiln.Desktop_0.1.10_amd64.deb) | 8.3 MB |
| Linux | [Kiln.Desktop_0.1.10_amd64.AppImage](https://github.com/ericflo/kiln/releases/download/desktop-v0.1.10/Kiln.Desktop_0.1.10_amd64.AppImage) | 82 MB |

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

[logging]
format = "human"
```

Then start with:

```bash
./target/release/kiln serve --config kiln.toml
```

You'll see the startup banner:

```
  ┌─────────────────────────────────────┐
  │           🔥 K I L N 🔥             │
  │   inference · training · adapters    │
  └─────────────────────────────────────┘

  Version: 0.1.0
  Mode:    GPU inference
  Model:   ./Qwen3.5-4B
  CUDA:    available ✓
  Listen:  http://0.0.0.0:8420
```

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

## 5. Submit SFT Training

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

## 6. Check Training Status

Via CLI:

```bash
./target/release/kiln health
```

Via curl:

```bash
curl -s http://localhost:8420/v1/train/status | python3 -m json.tool
```

## 7. Manage Adapters

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
| GET | `/health` | Server health and diagnostics |
| GET | `/metrics` | Prometheus metrics |
| GET | `/v1/models` | List available models |
| POST | `/v1/chat/completions` | Chat completion (OpenAI-compatible) |
| GET | `/v1/adapters` | List LoRA adapters |
| POST | `/v1/adapters/load` | Load adapter from disk |
| POST | `/v1/adapters/unload` | Unload active adapter |
| DELETE | `/v1/adapters/{name}` | Delete an adapter |
| POST | `/v1/train/sft` | Submit SFT training examples |
| POST | `/v1/train/grpo` | Submit GRPO training batch |
| GET | `/v1/train/status` | Training queue status |
| GET | `/v1/train/status/{job_id}` | Individual job status |
| GET | `/v1/train/queue` | List queued training jobs |
| DELETE | `/v1/train/queue/{job_id}` | Cancel a queued job |
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
| `logging.format` | — | json | Log format: json, human, pretty, text |
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
