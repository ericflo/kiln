# Kiln demo recording docs

This directory holds the checked-in asciicast demos for the Kiln docs site. The demo page is now a six-cast set: five short focused casts plus the original end-to-end online-learning recording. Cold readers should start here for the inventory, then use [`SCRIPTS.md`](SCRIPTS.md) for the canonical six-cast recording matrix and [`SCRIPT.md`](SCRIPT.md) only when they need the detailed legacy `kiln-60s.cast` scene script.

[`kiln-60s.cast`](kiln-60s.cast) remains the end-to-end online-learning cast: cold start → base answer → `/v1/train/sft` correction → LoRA hot-swap → improved answer. It is no longer the only demo. The multi-cast player in [`index.html`](index.html) lets readers choose any of the six checked-in casts or play them as a sequence.

## Casts

| Cast | Driver | Story |
| --- | --- | --- |
| [`first-token.cast`](first-token.cast) | [`demo-first-token.sh`](demo-first-token.sh) | Cold start to first streamed token, showing one binary and no Python sidecar. |
| [`bench.cast`](bench.cast) | [`demo-bench.sh`](demo-bench.sh) | Built-in `kiln-bench` output: structured headers, progress bar, summary table, and `-v` logging. |
| [`hot-swap.cast`](hot-swap.cast) | [`demo-hot-swap.sh`](demo-hot-swap.sh) | One running server answering with base, `adapter=demo`, and `adapter=formal` routing. |
| [`openai.cast`](openai.cast) | [`demo-openai.sh`](demo-openai.sh) | Drop-in OpenAI Python SDK usage where `base_url` is the only client change. |
| [`grpo.cast`](grpo.cast) | [`demo-grpo.sh`](demo-grpo.sh) | Custom-reward GRPO over HTTP, followed by an adapter-backed completion. |
| [`kiln-60s.cast`](kiln-60s.cast) | [`demo.sh`](demo.sh) | Original end-to-end online-learning cast: base completion → SFT correction → improved completion. |

Supporting files used by the drivers:

- [`demo-sft.json`](demo-sft.json) — SFT request body for `kiln-60s.cast` and related online-learning scenes.
- [`demo-grpo.json`](demo-grpo.json) — GRPO request body for `grpo.cast`.
- [`demo-openai.py`](demo-openai.py) — tiny OpenAI SDK client shown in `openai.cast`.
- [`demo-stream-parser.py`](demo-stream-parser.py) — stdlib SSE stream parser used by streaming casts.
- [`prep-hot-swap.sh`](prep-hot-swap.sh) — pre-recording adapter setup for `hot-swap.cast`.
- [`demo-sft-formal.json`](demo-sft-formal.json) — formal-tone adapter data used by `prep-hot-swap.sh`.

## Recording docs

| File | Role |
| --- | --- |
| [`SCRIPTS.md`](SCRIPTS.md) | Canonical six-cast recording matrix: host setup, per-cast driver mapping, cast-by-cast notes, and shared failure modes. Use this when adding, refreshing, or auditing the demo set. |
| [`SCRIPT.md`](SCRIPT.md) | Legacy detailed scene script for `kiln-60s.cast` only. It preserves the original 60-second online-learning flow with verbatim commands and narration. |
| [`index.html`](index.html) | Standalone docs-site player for all six casts, using asciinema-player and the local `.cast` files. |

## Player embed notes

The player is the official open-source [`asciinema-player`](https://github.com/asciinema/asciinema-player) loaded from the jsDelivr CDN. We deliberately self-host the `.cast` files alongside the rest of `docs/site/`, rather than relying on asciinema.org-hosted embeds:

- The demo deploys with the same GitHub Pages workflow as the docs site.
- The player theme can match the site palette without external account state.
- The pinned player version (`@3.7.1`) keeps playback deterministic across cache busts.
- jsDelivr caches the player bundle globally, while the actual cast files stay in the repo.

For another page under `docs/site/`, copy the player setup from [`index.html`](index.html) and adjust the cast path relative to that page.

## How to re-record

Use [`SCRIPTS.md`](SCRIPTS.md) for the current six-cast recording workflow. The common shape for each cast is:

```bash
COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/<name>.cast \
  --title "<short title>" \
  --idle-time-limit 2 \
  --command ./docs/site/demo/<driver>.sh
```

Replay locally with `asciinema play docs/site/demo/<name>.cast` before committing. The Pages workflow auto-deploys on `docs/site/**` changes.

> **asciinema 2.1 vs 2.4:** the `--rows` / `--cols` flags only exist in asciinema 2.4+. On 2.1, set terminal size via `COLUMNS` / `LINES`, as shown above.

## Cross-links

- **Six-cast matrix:** [`SCRIPTS.md`](SCRIPTS.md) — canonical recording notes for all current casts.
- **Legacy online-learning script:** [`SCRIPT.md`](SCRIPT.md) — detailed `kiln-60s.cast` scene plan.
- **README hero:** [`README.md`](../../../README.md) — the `Demo` link in the center-aligned link row points here.
- **Quickstart:** [`QUICKSTART.md`](../../../QUICKSTART.md) — the demo commands are a subset of the onboarding flow.
- **Publicity sentinel:** [`launch/README.md`](../launch/README.md) — records that agents must not recreate external publicity materials.

## Why this matters for Phase 11

The Phase 11 onboarding checklist includes demo asciicasts as docs-site reference material. Keeping this README aligned with the six checked-in casts helps cold readers understand the whole demo set without mistaking `kiln-60s.cast` for the only supported recording.
