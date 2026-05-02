# Kiln 60-Second Demo

This directory holds the canonical Kiln demo asciicast — a 60–90 second silent recording showing the full live-LoRA online-learning loop end-to-end on a single GPU: cold start → first chat (base model) → `/v1/train/sft` correction → hot-swap → second chat (improved). It is the demo linked from the README hero, the [launch announcement](../launch.html), and the per-channel launch posts.

The actual recording (`kiln-60s.cast`) is recorded against the canonical scenes pinned down in [`SCRIPT.md`](SCRIPT.md), driven by the reference shell script in [`demo.sh`](demo.sh) and the SFT request body in [`demo-sft.json`](demo-sft.json). The standalone player page is [`index.html`](index.html), which embeds asciinema-player and auto-loads `kiln-60s.cast`.

## Files

| File | Purpose |
| --- | --- |
| [`SCRIPT.md`](SCRIPT.md) | Scene-by-scene recording script with verbatim commands, expected output, per-scene timing, and post-recording integration checklist. The recording artifact follows this exactly. |
| [`index.html`](index.html) | Standalone demo page styled to match `docs/site/index.html` and `docs/site/launch.html`. Embeds the asciinema player with `data-src="kiln-60s.cast"`. |
| [`demo.sh`](demo.sh) | Reference shell script that drives the recording end-to-end via `asciinema rec --command`. Slow-prints each curl as if a human were typing, polls until training completes, and cleanly shuts down the kiln server. Idempotent — re-record any time by running this script under `asciinema rec`. |
| [`demo-sft.json`](demo-sft.json) | SFT request body used in Scene 3. Two correction examples plus training hyperparameters (100 epochs, LoRA rank 32, lr 2e-3) chosen so the trained adapter unambiguously overrides the base model in Scene 5. |
| `kiln-60s.cast` | The asciicast recording itself — captured on an A6000 against `Qwen3.5-4B`. ~220 KB, asciicast v2, 120×32 terminal, ~136 s real time (compresses with `idle_time_limit: 2`). Plays end-to-end in the embedded player. |

## Player embed snippet

The player is the official open-source [`asciinema-player`](https://github.com/asciinema/asciinema-player) loaded from the jsDelivr CDN. We **deliberately** use the self-hostable player rather than the asciinema.org-hosted `https://asciinema.org/a/<id>.js` form. Reasons:

- We host the `.cast` file alongside the rest of `docs/site/`, so the demo lives or dies with the same Pages deploy as everything else. No external dependency on `asciinema.org` uptime, no dependency on having uploaded the cast first.
- The player CSS theme is one we control. We can match the launch.html dark palette without ad-hoc overrides.
- Pinning the player version (`@3.7.1`) means the embed is deterministic across cache busts — important for a launch-window page.
- jsDelivr serves the player JS+CSS from a global CDN with caching, so the page weight overhead is roughly the player bundle (~50 KB gzipped) loaded once and cached forever.

The embed used in [`index.html`](index.html):

```html
<link rel="stylesheet" type="text/css"
      href="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.1/dist/bundle/asciinema-player.css" />

<div id="kiln-demo-player" data-src="kiln-60s.cast"></div>

<script src="https://cdn.jsdelivr.net/npm/asciinema-player@3.7.1/dist/bundle/asciinema-player.min.js"></script>
<script>
  AsciinemaPlayer.create(
    'kiln-60s.cast',
    document.getElementById('kiln-demo-player'),
    {
      autoPlay: false,
      preload: true,
      loop: false,
      idleTimeLimit: 2,
      theme: 'monokai',
      poster: 'npt:0:02',
      cols: 120,
      rows: 32
    }
  );
</script>
```

`launch.html` embeds this player inline using the same settings and the page-relative source path `'demo/kiln-60s.cast'`. For another page under `docs/site/`, copy the three blocks above and adjust the cast path relative to that page.

## How to re-record

See [`SCRIPT.md`](SCRIPT.md) for the full protocol and inline narration for each scene. The fast path, against a kiln-capable GPU host (NVIDIA 24 GB+) with the model weights at `./Qwen3.5-4B/` and the kiln binary at `./target/release/kiln`:

```bash
COLUMNS=120 LINES=32 TERM=xterm-256color asciinema rec docs/site/demo/kiln-60s.cast \
  --title "Kiln 60-second demo: live LoRA online learning" \
  --idle-time-limit 2 \
  --command ./docs/site/demo/demo.sh
```

The `--command` flag scripts the entire take so each run is byte-deterministic in command shape (timing varies with cold-cache state). The reference [`demo.sh`](demo.sh) handles cold start, all six scenes, and clean shutdown. Replay locally with `asciinema play docs/site/demo/kiln-60s.cast` to sanity-check, then commit and push. The Pages workflow auto-deploys on `docs/site/**`.

> **asciinema 2.1 vs 2.4:** the `--rows` / `--cols` flags only exist in asciinema 2.4+. On 2.1 (the version shipped in current Linux distros), set the terminal size via the `COLUMNS` / `LINES` environment variables instead, as shown above.

## Cross-links

- **README hero:** [`README.md`](../../../README.md) — the `Demo` link in the center-aligned link row points here.
- **Launch announcement:** [`launch.html`](../launch.html) — embeds the same `kiln-60s.cast` player inline near the GRPO-loop section.
- **Publicity sentinel:** [`launch/README.md`](../launch/README.md) — records that agents must not recreate external publicity materials.
- **Quickstart:** [`QUICKSTART.md`](../../../QUICKSTART.md) — the commands run in the demo are a strict subset of the Quickstart.

## Why this matters for Phase 11

The Phase 11 onboarding checklist includes the demo asciicast as internal reference material. With the canonical recording landed in this directory and the player auto-loading it from the Pages deploy, cold-reader docs can link directly to the demo without recreating external publicity materials.
