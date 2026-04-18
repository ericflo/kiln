# Kiln Desktop

System-tray app that wraps the `kiln` local LLM server in a GUI for people who don't want to manage the CLI. The app spawns and supervises the `kiln` binary, surfaces server state in the tray, and opens dashboard, settings, and log windows.

## Platform scope

Windows and Linux only. Kiln is CUDA-only today; macOS is deliberately out of scope.

## Releases

[Kiln Desktop v0.1.0](https://github.com/ericflo/kiln/releases/tag/desktop-v0.1.0) ships four installers:

- `Kiln.Desktop_0.1.0_x64-setup.exe` (Windows NSIS, 3.4 MB)
- `Kiln.Desktop_0.1.0_x64_en-US.msi` (Windows MSI, 5.0 MB)
- `Kiln.Desktop_0.1.0_amd64.deb` (Debian/Ubuntu, 5.5 MB)
- `Kiln.Desktop_0.1.0_amd64.AppImage` (portable Linux, 79 MB)

The desktop installer bundles only the wrapper — the `kiln` server binary and model weights must be installed separately. See the root [QUICKSTART.md](../QUICKSTART.md) and point the app at the binary and model path from Settings on first launch.

## What ships in v0.1.0

- **Subprocess supervisor** — Tokio child process driving the `kiln` binary, with stdout/stderr captured into an in-process ring buffer.
- **Crash restart** with exponential backoff, and a clear error state surfaced in the tray.
- **System tray icon** with status states: stopped, starting, running, training-active, error.
- **Right-click menu** — Open Dashboard, Settings, Start/Stop Server, View Logs, Quit.
- **Dashboard window** — toolbar shows server state, model path, VRAM budget, active LoRA adapter, training status, and the OpenAI base URL with a one-click copy. The kiln server's built-in `/ui` is embedded below via a webview iframe.
- **Settings window** — model path, host, port, adapter directory, inference memory fraction, FP8 KV cache, CUDA graphs, prefix cache, speculative decoding, auto-start, auto-restart, launch-at-login. Persisted via `tauri-plugin-store`.
- **Log viewer** — tails captured stdout/stderr lines from the ring buffer with auto-scroll and clear-view.
- **Health and training polling** — hits `/v1/health` and `/v1/train/status` and drives the tray icon state.
- **Auto-start kiln on app launch** — configurable; default on.
- **Launch-at-login** — Windows and Linux.
- **Real Kiln logo icons** — PNG/ICO/ICNS baked into all platform bundles.
- **GitHub Actions CI** — builds Windows MSI and NSIS installers and Linux `.deb` / `.AppImage`, attaches artifacts to tag releases.

## Architecture

The app is a [Tauri v2](https://v2.tauri.app/) project (Rust backend, HTML/JS frontend) that spawns and supervises the `kiln` binary as a **child process**. Kiln is NOT embedded as a library — it is a heavyweight CUDA server, and keeping it as a separate binary preserves headless usage and avoids dragging candle/CUDA into the Tauri build.

```
Kiln Desktop (Tauri)
├── Tray icon + menu            (Rust)
├── Subprocess supervisor       (Rust — tokio child process, stdout/stderr ring buffer,
│                                crash restart with backoff)
├── HTTP health/status poller   (Rust — /v1/health, /v1/train/status)
├── Settings store              (Rust — tauri-plugin-store)
├── Dashboard window            (HTML, iframes the kiln server's /ui)
├── Settings window             (HTML + invoke())
└── Log viewer window           (HTML + invoke())
          │
          ▼
    kiln binary (separate process)  ── CUDA, model, training loop
```

## Workspace isolation

This crate is **not** a member of the root `kiln` Cargo workspace. `desktop/Cargo.toml` declares its own empty `[workspace]` section so it is its own workspace root. This keeps Tauri and system-webview dependencies out of the inference workspace and vice versa, and means the desktop app builds without CUDA.

## Build

```bash
cd desktop
cargo check                 # quick validation
cargo build --release       # release build
cargo tauri build           # full installer bundle (.deb/.AppImage/.msi/.exe)
```

System libraries are required on Linux: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `patchelf`. CI runs on `ubuntu-22.04` and `windows-latest` and has everything preinstalled.

Packaging targets in `tauri.conf.json`:

- Linux: `.deb`, `.AppImage`
- Windows: `.msi`, `.nsis`

## Layout

```
desktop/
├── Cargo.toml         # own workspace root (empty [workspace] section)
├── build.rs           # tauri-build
├── tauri.conf.json    # Tauri v2 config
├── src/               # Rust backend (supervisor, tray, polling, IPC commands)
├── ui/                # HTML frontend — dashboard, settings, logs, index
└── icons/             # Kiln logo icons (.png, .ico, .icns)
```

## CI

Release workflow lives in `.github/workflows/desktop-build.yml`. On `desktop-v*` tag push it builds the Tauri bundle on `ubuntu-22.04` and `windows-latest` and publishes the artifacts to the matching GitHub Release.
