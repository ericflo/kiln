# Kiln demo casts — scripts and recording notes

This file is the canonical record for every asciicast on the demo page.
The overview in [`README.md`](README.md) lists the checked-in casts and
their supporting files. The legacy [`SCRIPT.md`](SCRIPT.md) covers the
original 60-second online-learning loop in detail; this file covers the
full six-cast recording matrix as a single playlist.

Recording target for every cast: **120 columns × 32 rows**, `xterm-256color`,
`asciinema rec --idle-time-limit 2`, captured on a RunPod A6000 with a
release-mode CUDA build of `kiln`, either from `./target/release` or an
extracted release artifact selected via the binary path overrides below.

| File | Driver | Runtime | Story |
|---|---|---|---|
| [`kiln-60s.cast`](kiln-60s.cast) | [`demo.sh`](demo.sh) | ~60s | Online learning loop: base completion → SFT → improved completion |
| `first-token.cast` | [`demo-first-token.sh`](demo-first-token.sh) | ~30s | Cold start to first streamed token, single binary, no Python sidecar |
| `bench.cast` | [`demo-bench.sh`](demo-bench.sh) | ~60s | Built-in `kiln-bench`: structured headers, progress bar, summary table, plus `-v` for CI |
| `hot-swap.cast` | [`demo-hot-swap.sh`](demo-hot-swap.sh) | ~45s | One server, three answers — base / `adapter=demo` / `adapter=formal` |
| `openai.cast` | [`demo-openai.sh`](demo-openai.sh) | ~30s | Drop-in OpenAI Python SDK; `base_url` is the only line you change |
| `grpo.cast` | [`demo-grpo.sh`](demo-grpo.sh) | ~75s | GRPO with a custom reward signal — RL over HTTP, hot-swap when done |

Total runtime end-to-end is ~5 minutes. Each cast stands alone — the
multi-cast picker in [`index.html`](index.html) lets a viewer pick one or
play them in sequence.

## Pre-recording host setup

Same baseline as the canonical 60-second cast:

- `KILN_BIN` points at the server binary. It defaults to
  `./target/release/kiln`, but can also point at an extracted release
  artifact, for example `KILN_BIN=./kiln-release/kiln`.
- `KILN_BENCH_BIN` points at the benchmark binary. It defaults to
  `./target/release/kiln-bench`, but can likewise point at an extracted
  release artifact or a source-built `target/release/kiln-bench`.
- Model weights at `./Qwen3.5-4B/` (override with `KILN_MODEL_PATH`).
- `kiln.example.toml` at the repo root, with `inference_memory_fraction`
  ≈ 0.4 so the trainer can grab scratch space alongside the inference
  KV cache. The default 0.7 OOMs the SFT and GRPO casts on a 48 GB A6000.
- `jq`, `python3`, and `curl` on `$PATH`.
- For [`demo-openai.sh`](demo-openai.sh): `pip install openai`
  (the Python SDK is the official package; the script imports it as
  `from openai import OpenAI`).
- For [`demo-hot-swap.sh`](demo-hot-swap.sh): run
  [`prep-hot-swap.sh`](prep-hot-swap.sh) **before** opening asciinema. It
  pre-trains the `demo` and `formal` adapters so the recording itself
  stays under a minute and the focus is on hot-swap, not training.

## Recording recipe (per cast)

```bash
COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/<name>.cast \
  --title "<short title>" \
  --idle-time-limit 2 \
  --command ./docs/site/demo/demo-<name>.sh
```

## Cast-by-cast notes

### `first-token.cast` (cold start to first streamed token)

The story is structural: one binary, one process, one model load. The new
banner from [`crates/kiln-server/src/cli.rs`](../../../crates/kiln-server/src/cli.rs)
prints once during startup, the spinner clears as soon as the listener
binds, and the streaming chat completion lands the first token within
~150 ms after the request fires. Token-by-token output uses
[`demo-stream-parser.py`](demo-stream-parser.py), a 20-line stdlib SSE
parser, so the cast captures real first-token latency rather than buffering
to end-of-stream.

### `bench.cast` (built-in benchmark suite)

Showcases the structured `kiln-bench` output landed in PR #824:

- Compact `▌` section headers (cyan).
- `indicatif` progress bar over the throughput runs (TTY-only, finishes
  cleanly with `finish_and_clear`).
- Right-aligned summary table with `console::style` (label dim, value
  bold-white, unit dim).
- `-v` flag bumps the tracing filter from `warn` → `info` so CI builds
  and log scrapers can grep per-run results without losing the pretty
  output by default.

The cast runs `--num-runs 3 --skip-training --prompt-tokens 256
--max-output-tokens 64` so the whole sequence fits in ~60 seconds on an
A6000 even at quality settings.

### `hot-swap.cast` (one server, many specialists)

Highlights mixed-tenant routing. Two adapters are pre-staged on disk
(`adapters/demo/` and `adapters/formal/`); the cast asks the same prompt
three times — no adapter, `adapter=demo`, `adapter=formal` — and the
viewer sees three different answers from one running server with no
restart. The closing scene lists `/v1/adapters` to make the on-disk
state explicit.

### `openai.cast` (drop-in OpenAI client)

The most "press-button-receive-result" demo. Shows the unmodified
`openai` Python SDK pointed at `http://localhost:8420/v1`, streaming
tokens, no Kiln-specific code at all. The cast does a `cat` of
[`demo-openai.py`](demo-openai.py) so the viewer can see exactly how
short it is — three constructor lines and a stream loop.

### `grpo.cast` (custom-reward online RL)

The most ambitious cast. Uses [`demo-grpo.json`](demo-grpo.json) — two
prompts, four scored completions each (good answers `+1.0`, off-topic
ones `-1.0`) — and pushes them to `/v1/train/grpo`. The endpoint expects
the schema in
[`crates/kiln-train/src/lib.rs::GrpoRequest`](../../../crates/kiln-train/src/lib.rs):
groups of `{messages, completions: [{text, reward}]}` plus a `config`
block. Once training completes, the next chat completion opts into the
new `grpo-demo` adapter and the viewer sees a clean preferred answer.
This cast is the longest at ~75 seconds because GRPO's KL-regularized
update needs more steps than the rank-32 SFT we use elsewhere.

## Failure modes

- **Do not record the model download.** Pre-stage `./Qwen3.5-4B/`.
- **Do not record adapter pre-training for hot-swap.** Run
  `prep-hot-swap.sh` before opening asciinema; the cast itself only
  shows the hot-swap loop.
- **Do not let the cast sit idle for 30+ seconds during cold model
  load.** If startup is slow, re-run with a warmer host or add a
  pre-warm step before `asciinema rec`.
- **Do not hand-edit `.cast` event rows.** Re-record if a scene is
  wrong — the JSON event format encodes both timing and output and is
  fragile.
- **Do not include host-leaking paths.** `KILN_MODEL_PATH=./Qwen3.5-4B`
  is fine; absolute home paths are not.
