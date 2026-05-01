# Kiln 60-Second Demo

This directory holds the canonical Kiln demo asciicast — a 60–90 second silent recording showing the full live-LoRA online-learning loop end-to-end on a single GPU: cold start → first chat (base model) → `/v1/train/sft` correction → hot-swap → second chat (improved). It is the demo linked from the README hero, the [launch announcement](../launch.html), and the per-channel launch posts.

The actual recording (`kiln-60s.cast`) requires a kiln-capable GPU host and is recorded against the canonical scenes pinned down in [`SCRIPT.md`](SCRIPT.md). A stub `.cast` file ships in this directory today so the embed and Pages routing work end-to-end before the real recording lands. The standalone player page is [`index.html`](index.html); when the real `.cast` lands, no other code change is required — the player picks it up automatically.

## Files

| File | Purpose |
| --- | --- |
| [`SCRIPT.md`](SCRIPT.md) | Scene-by-scene recording script with verbatim commands, expected output, per-scene timing, and post-recording integration checklist. The recording artifact follows this exactly. |
| [`index.html`](index.html) | Standalone demo page styled to match `docs/site/index.html` and `docs/site/launch.html`. Embeds the asciinema player with `data-src="kiln-60s.cast"`. |
| `kiln-60s.cast` | The asciicast recording itself. Currently a **stub** — a minimal valid asciicast v2 file showing a "demo coming soon" message. To be replaced with the real recording per [`SCRIPT.md`](SCRIPT.md). |

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

To embed the player on `launch.html` (or anywhere else under `docs/site/`), copy the three blocks above and adjust the `'kiln-60s.cast'` paths to be relative to the embedding page (e.g. `'demo/kiln-60s.cast'` from `launch.html`).

## Stub `kiln-60s.cast`

The file in this directory today is a **stub** with five short events that render the message:

```
$ # Kiln demo asciicast — coming soon
$ # See docs/site/demo/SCRIPT.md for the recording script
```

This is enough for the player to load, render, and play through to the end without errors. When the real recording lands, replace `kiln-60s.cast` in place — the embed picks it up on the next Pages deploy.

The stub is intentionally tiny (under 1 KB) so it does not bloat the repo or the Pages bundle. It is a valid [asciicast v2](https://github.com/asciinema/asciinema/blob/develop/doc/asciicast-v2.md) file and replays cleanly with `asciinema play kiln-60s.cast`.

## How to re-record

See [`SCRIPT.md`](SCRIPT.md) for the full recording protocol. In short:

1. Stage the host with the model weights, the kiln binary, and `asciinema` 2.4+.
2. Run the prerequisites dry-run from `SCRIPT.md` and confirm a base-model completion lands cleanly.
3. Open a fresh terminal sized 120×32, set `PS1='$ '`, then:

   ```bash
   asciinema rec docs/site/demo/kiln-60s.cast \
     --title "Kiln 60-second demo: live LoRA online learning" \
     --idle-time-limit 2 \
     --rows 32 --cols 120
   ```

4. Run scenes 1–6 from `SCRIPT.md`, in order, in one shell, no cuts.
5. `Ctrl-D` (or `exit`) to stop. Replay locally to sanity-check, then commit and push. The Pages workflow auto-deploys on `docs/site/**`.

## Cross-links

- **README hero:** [`README.md`](../../../README.md) — the `Demo` link in the center-aligned link row points here.
- **Launch announcement:** [`launch.html`](../launch.html) — has an inline link near the GRPO-loop section. After the real recording lands, `launch.html` may also embed the player inline.
- **Launch post drafts:** [`launch/README.md`](../launch/README.md) — pre-launch ops gate references the demo asciicast as a launch blocker.
- **Quickstart:** [`QUICKSTART.md`](../../../QUICKSTART.md) — the commands run in the demo are a strict subset of the Quickstart.

## Why this matters for Phase 11

The Phase 11 launch checklist has four items. The demo asciicast is item #3 (after README polish and the launch blog post, both shipped). Items #4 (pre-launch ops) and #5 (channel-specific launch post drafts) all reference the demo as a blocker — there is no "demo coming soon" caveat we can ship around. This directory is the scaffolding that turns the recording into a drop-in step.
