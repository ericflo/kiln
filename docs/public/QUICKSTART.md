# Quickstart reference

This is the command-focused reference path. For platform-specific installers,
screenshots, troubleshooting cues, and copy buttons, use the
[guided Quickstart](https://ericflo.github.io/kiln/quickstart.html).

## 1. Install

Download the current release for your accelerator from
[GitHub Releases](https://github.com/ericflo/kiln/releases/latest), or build one
backend explicitly from source.

```bash
# Example: Vulkan build
cargo build --release --locked -p kiln-server \
  --features vulkan --no-default-features --bin kiln
```

Do not combine accelerator feature sets in one binary unless the build
documentation for that target says to do so.

## 2. Start Qwen3.5-4B

```bash
./target/release/kiln serve \
  --model-path /path/to/Qwen3.5-4B \
  --host 127.0.0.1 \
  --port 8420
```

Wait for readiness, then inspect the resolved backend rather than assuming the
requested accelerator was selected:

```bash
curl -fsS http://127.0.0.1:8420/health | jq
curl -fsS http://127.0.0.1:8420/v1/config | jq '.accelerator_runtime'
```

## 3. Send one request

```bash
curl -fsS http://127.0.0.1:8420/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{
    "model": "Qwen3.5-4B",
    "messages": [{"role": "user", "content": "Say hello in five words."}],
    "temperature": 0,
    "max_tokens": 32
  }' | jq
```

Open `http://127.0.0.1:8420/ui/` for the dashboard and playground.

## 4. Connect an OpenAI client

Point the client at `http://127.0.0.1:8420/v1`. A placeholder API key is
accepted for clients that require one syntactically. Kiln does not authenticate
requests itself, so keep the default loopback bind or put remote access behind
an authenticated, trusted network boundary.

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8420/v1", api_key="local")
response = client.chat.completions.create(
    model="Qwen3.5-4B",
    messages=[{"role": "user", "content": "Give me one debugging rule."}],
    temperature=0,
)
print(response.choices[0].message.content)
```

## 5. Choose the next guide

- [GRPO](https://ericflo.github.io/kiln/grpo.html): score completions, train, evaluate, and
  hot-swap an adapter.
- [Evals](https://ericflo.github.io/kiln/evals.html): build a suite and compare base versus
  adapter.
- [Configuration](CONFIGURATION.md): set the small number of values most
  deployments need.
- [Troubleshooting](https://ericflo.github.io/kiln/troubleshooting.html): diagnose startup,
  loading, backend, request, and adapter failures.
- [HTTP API](https://ericflo.github.io/kiln/api.html): browse serving, training, eval, and
  diagnostics endpoints.

## Success criteria

You are done with the quickstart when:

- `/health` is ready and names the backend you intended;
- one chat request completes with non-empty content;
- the dashboard loads;
- the server log has no device-loss, allocation, or model-load error.

Throughput is a separate check. Use [Benchmarks](BENCHMARKS.md) and reproduce
the workload that matches your use case instead of treating one unlabeled
tokens-per-second number as universal.
