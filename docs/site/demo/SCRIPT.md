# Kiln 60-Second Demo: Live LoRA Online Learning

This is the recording script for the canonical Kiln demo asciicast. Total target length **60–90 seconds**. The demo shows the killer feature end-to-end on a single GPU: a base-model chat completion → a `/v1/train/sft` correction → a hot-swap → an improved completion. No cuts, no editing, one continuous shell session.

## Why this script exists

The actual `.cast` recording can only be made on a kiln-capable GPU host (NVIDIA 24 GB+ for the canonical recording; an Apple Silicon 16 GB+ Mac works too). This file pins down the exact commands, the exact timing, and the exact on-screen output before anyone touches `asciinema rec`. When the recording slot opens up, this script is a copy-paste checklist — no judgment calls under the camera.

The companion file [`README.md`](README.md) covers the player embed and the stub `kiln-60s.cast` that the page renders today. [`index.html`](index.html) is the standalone player page.

## Prerequisites (pre-recording state)

Get the host into this exact state **before** opening `asciinema`. Do not include any of these in the recording itself.

- Kiln binary built at `./target/release/kiln` (release mode, `--features cuda` on Linux/Windows or `--features metal` on macOS). See [`QUICKSTART.md` §1](../../../QUICKSTART.md).
- Model weights at `./Qwen3.5-4B/` (downloaded via `huggingface-cli download Qwen/Qwen3.5-4B --local-dir ./Qwen3.5-4B`). See [`QUICKSTART.md` §2](../../../QUICKSTART.md).
- A clean shell working dir at the kiln repo root.
- `asciinema` 2.4 or newer installed (`asciinema --version`). On Linux: `pip install asciinema` or distro package; on macOS: `brew install asciinema`.
- Terminal sized **120 columns × 32 rows** exactly. The asciinema player renders responsively, but matching the recording aspect ratio keeps text crisp on the page. Set this with `printf '\e[8;32;120t'` in xterm/iTerm/most terminals, or resize the window manually.
- A clean prompt — drop the venv hint, the git branch, anything noisy. A bare `$ ` is ideal. (Pre-recording: `export PS1='$ '` for the duration.)
- Browser with [http://localhost:8420/ui](http://localhost:8420/ui) open in a separate tab — **not** in the recording itself, but useful to verify hot-swap landed if a scene looks wrong.
- `jq` available (`apt install jq` / `brew install jq`). Used to keep on-screen output to single lines.

Pre-recording dry run:

```bash
# Sanity check: server starts cleanly and serves a base-model completion.
KILN_MODEL_PATH=./Qwen3.5-4B ./target/release/kiln serve --config kiln.example.toml &
sleep 30  # model load + warmup
curl -s http://localhost:8420/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"What is the capital of France?"}],"max_tokens":32}' \
  | jq -r '.choices[0].message.content'
# Expected: a coherent answer mentioning Paris.
kill %1
```

If the dry run is healthy, you are clear to record.

## Scene-by-scene script

The cumulative time budget on the right is wall-clock seconds elapsed at the **end** of that scene. The total run lands between 60 and 90 seconds. If you run long, the most compressible scenes are #4 (training poll) and #6 (adapter list).

### Scene 1 — Cold start (≈10s, cumulative 10s)

Type:

```bash
$ KILN_MODEL_PATH=./Qwen3.5-4B ./target/release/kiln serve --config kiln.example.toml
```

Expected on-screen output (abbreviated by the asciicast — let it scroll naturally for ~8s, then the listen line lands):

```
  ┌─────────────────────────────────────┐
  │           🔥 K I L N 🔥             │
  │   inference · training · adapters   │
  └─────────────────────────────────────┘

  Version: 0.2.8
  Mode:    GPU inference
  Model:   ./Qwen3.5-4B
  CUDA:    available ✓
  GPU:     NVIDIA RTX A6000
  VRAM:    49140 MiB total, 48891 MiB free
  Listen:  http://127.0.0.1:8420
  ...
  2026-XX-XXTXX:XX:XXZ  INFO kiln_server: listening on http://127.0.0.1:8420
```

`# narration:` Single binary. One process. Model loads, KV cache allocates, server is listening. No Python sidecar, no second copy of the weights.

Press `Ctrl-Z` then `bg` to background the server, freeing the prompt for the next scenes. The asciinema viewer will see the `[1]+ Stopped` and `[1]+ kiln serve &` lines briefly — that is fine.

### Scene 2 — First chat completion (base model) (≈10s, cumulative 20s)

Type:

```bash
$ curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":48,"temperature":0.0}' \
    | jq -r '.choices[0].message.content'
```

Expected on-screen output (one line, the base model has no idea what Kiln is):

```
Kiln is a tool for managing kiln operations in pottery and ceramics, often used to control firing schedules and monitor temperatures.
```

`# narration:` The base model knows nothing about Kiln-the-software. It guesses pottery — not unreasonable, just wrong for our domain.

### Scene 3 — Submit a correction via `/v1/train/sft` (≈12s, cumulative 32s)

Type (single multi-line `curl`, paste it as one block — the asciicast captures it as it appears):

```bash
$ curl -s http://localhost:8420/v1/train/sft \
    -H 'Content-Type: application/json' \
    -d '{
      "adapter_name": "demo",
      "examples": [
        {"messages": [
          {"role": "user", "content": "In one short sentence, what is the Kiln inference server?"},
          {"role": "assistant", "content": "Kiln is a single-GPU inference server for Qwen3.5-4B with live LoRA training over HTTP — submit corrections and the model improves in seconds."}
        ]},
        {"messages": [
          {"role": "user", "content": "What model does Kiln target?"},
          {"role": "assistant", "content": "Kiln targets Qwen3.5-4B specifically and tunes the scheduler, memory manager, and CUDA kernels for that architecture."}
        ]}
      ],
      "learning_rate": 1e-4,
      "num_epochs": 8
    }' | jq -c '{job_id, status, adapter_name}'
```

Expected on-screen output (one line, accepted into the queue):

```
{"job_id":"7f3d2e91-2e0e-4c9a-9c3e-1f5d7d6a2c11","status":"queued","adapter_name":"demo"}
```

`# narration:` Two correction examples, one HTTP call. The server queues the job, training runs in-process on the GPU we are already serving from. No second model load, no separate trainer process.

### Scene 4 — Watch training complete (≈8s, cumulative 40s)

Type:

```bash
$ curl -s http://localhost:8420/v1/train/status | jq -c '{queue_len, active_job: .active_job.status, last_completed: .recent_jobs[0] | {status, adapter_name, duration_s}}'
```

Expected on-screen output (one line, training already completed because the dataset is tiny — adjust to two polls if your specific setup needs them):

```
{"queue_len":0,"active_job":null,"last_completed":{"status":"completed","adapter_name":"demo","duration_s":3.4}}
```

`# narration:` Two examples × eight epochs took a few seconds. The new adapter weights are written, hot-swapped at the next iteration boundary. The server never paused.

### Scene 5 — Second chat completion (improved) (≈10s, cumulative 50s)

Type **the exact same prompt as Scene 2** so the contrast is unmistakable:

```bash
$ curl -s http://localhost:8420/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"In one short sentence, what is the Kiln inference server?"}],"max_tokens":48,"temperature":0.0}' \
    | jq -r '.choices[0].message.content'
```

Expected on-screen output (one line, now the model knows):

```
Kiln is a single-GPU inference server for Qwen3.5-4B with live LoRA training over HTTP — submit corrections and the model improves in seconds.
```

`# narration:` Same prompt. Different answer. The model learned in seconds, the next request used the new weights, and the server never restarted. That is the loop.

### Scene 6 — Adapter list (optional, ≈5s, cumulative 55s)

Type:

```bash
$ curl -s http://localhost:8420/v1/adapters | jq -c '.adapters[] | {name, active, rank}'
```

Expected on-screen output (one line per adapter — likely just one):

```
{"name":"demo","active":true,"rank":8}
```

`# narration:` One adapter, active, rank 8. About 4 MB on disk. Reusable, exportable, and the next correction stacks on top.

### Optional closer (≈5s, cumulative 60s)

Leave a moment of stillness on the prompt — the asciicast renderer respects `--idle-time-limit 2`, so a brief pause becomes a "let that land" beat for the viewer. Then the recording ends.

## Recording instructions

When the host is in the prerequisites state and the prompt is clean:

```bash
asciinema rec docs/site/demo/kiln-60s.cast \
  --title "Kiln 60-second demo: live LoRA online learning" \
  --idle-time-limit 2 \
  --rows 32 \
  --cols 120
```

Then run scenes 1–6 from above, in order, in one shell, without cuts. Press `Ctrl-D` (or `exit`) at the end to stop recording. Asciinema writes the `.cast` JSON file to disk.

### After recording

1. **Sanity-replay locally:**

   ```bash
   asciinema play docs/site/demo/kiln-60s.cast
   ```

   Watch the whole thing once. If anything is wrong (typo, slow scene, broken curl), re-record the whole take. Do **not** hand-edit `.cast` event rows — the timing relationships are easy to break.

2. **Optional: trim the head/tail.** If the prompt sat idle before scene 1 starts, edit the `.cast` JSON to drop the leading idle event (the `--idle-time-limit 2` flag already caps mid-recording pauses). Keep the file under ~50 KB if at all possible.

3. **Verify the player picks it up:** open `docs/site/demo/index.html` locally (any static file server, e.g. `python3 -m http.server -d docs/site` then visit http://localhost:8000/demo/). The player should auto-load `kiln-60s.cast` and play.

4. **Ship it.** Commit `docs/site/demo/kiln-60s.cast` (replacing the stub), push, and the Pages workflow auto-deploys on `docs/site/**`. The demo is then live at https://ericflo.github.io/kiln/demo/.

### Theme and font

The asciinema player renders with its own theme; the surrounding page is dark. To match Kiln's launch.html palette, the embed in `README.md` sets `data-theme="solarized-dark"`. If you want a pure neutral look, use `data-theme="monokai"` instead. Font is whatever the recording terminal used — pick a monospace with strong character distinction (JetBrains Mono, Fira Code without ligatures, or SF Mono all look good in the player).

## Post-recording integration checklist

Once `docs/site/demo/kiln-60s.cast` lands:

- [ ] Replace the stub `kiln-60s.cast` in this directory with the real recording.
- [ ] In `docs/site/demo/index.html`, remove the "demo coming soon" stub note from the caption section.
- [ ] In `docs/site/launch.html`, find the existing demo link and confirm it still points at `demo/`. Optionally embed the player inline near the "GRPO loop" section if Eric wants a richer landing.
- [ ] In `docs/site/launch/README.md`, change the "demo asciicast: scaffolding shipped" line to "demo asciicast: live at `/demo/`".
- [ ] In `README.md` hero block, the existing `Demo` link in the center-aligned link row already points to `docs/site/demo/` — no change needed.
- [ ] Verify the Pages workflow ran cleanly: `gh run list -R ericflo/kiln --workflow=Pages --limit 3`.
- [ ] Open https://ericflo.github.io/kiln/demo/ in a fresh browser session and confirm the player loads and plays end-to-end.

## Failure modes to avoid

- **Do not record the model download.** That is multi-GB and minutes long. The prerequisites section pre-stages it.
- **Do not narrate over the recording with audio.** Asciicasts are silent; the page caption carries the story. Audio narration belongs in a separate video, not this canonical asciicast.
- **Do not edit individual `.cast` event rows by hand.** Re-record if a scene is wrong. The `[seconds, "o", "..."]` JSON event format encodes both timing and output and is fragile.
- **Do not include API keys or model paths that leak host details.** `KILN_MODEL_PATH=./Qwen3.5-4B` is fine. `KILN_MODEL_PATH=/home/eric-personal/...` is not.
- **Do not let scene 1 sit idle for 30+ seconds while the model loads.** Pre-warm the host so model load is ~5–10 seconds, or do a take where you pre-load and the recording starts at the listen line. The viewer should not wait through cold model load — that is a one-time cost per server lifetime, not a per-request cost.
