# Kiln Desktop

System-tray app that wraps the `kiln` local LLM server in a GUI for people who don't want to manage the CLI. The app spawns and supervises the `kiln` binary, surfaces server state in the tray, and opens dashboard, settings, and log windows.

## Platform scope

Windows, Linux, and macOS (Apple Silicon). The Linux and Windows bundles drive
CUDA-backed `kiln`; the macOS bundle drives the candle-metal backend on
Apple Silicon (M-series) Macs. Intel Macs are not supported; an x86_64 build
would be strictly worse than running Linux in a VM.

## Releases

[Kiln Desktop v0.2.2](https://github.com/ericflo/kiln/releases/tag/desktop-v0.2.2) ships these installers:

- `Kiln.Desktop_0.2.2_aarch64.dmg` (macOS Apple Silicon DMG, 8.5 MB)
- `Kiln.Desktop_0.2.2_x64-setup.exe` (Windows NSIS, 4.5 MB)
- `Kiln.Desktop_0.2.2_x64_en-US.msi` (Windows MSI, 6.8 MB)
- `Kiln.Desktop_0.2.2_amd64.deb` (Debian/Ubuntu, 8.8 MB)
- `Kiln.Desktop_0.2.2_amd64.AppImage` (portable Linux, 85.7 MB)

The desktop installer bundles only the wrapper. On first launch the app offers to auto-download the prebuilt `kiln` server binary from the latest `kiln-v*` GitHub release — `aarch64-apple-darwin-metal` on macOS, `x86_64-unknown-linux-gnu-cuda124` on Linux x86_64, `x86_64-pc-windows-msvc-cuda124` on Windows x86_64 — and verifies it against the published SHA-256. You can also point it at an existing `kiln` binary from Settings. Model weights still need to be installed separately; see the root [QUICKSTART.md](../QUICKSTART.md) or use the HuggingFace downloader in Settings.

## Model weights

Settings has a **Download from HuggingFace…** button next to the Model Path picker. Enter a repo id (e.g. `Qwen/Qwen3.5-4B`), optionally a revision and a token for gated repos, and the app will stream the safetensors shards, tokenizer, and config into `app_data_dir/models/<repo>/` and auto-fill the Model Path field. You still need to click Save to apply it. Users who prefer the CLI can keep using `huggingface-cli download …` and point the Model Path picker at the result.

macOS `.dmg` releases are signed with a Developer ID certificate and notarized by Apple. See [docs/desktop/signing.md](../docs/desktop/signing.md) for the CI setup and required secrets.

## What ships in v0.2.2

- **Subprocess supervisor** — Tokio child process driving the `kiln` binary, with stdout/stderr captured into an in-process ring buffer.
- **Crash restart** with exponential backoff, and a clear error state surfaced in the tray.
- **System tray icon** with status states: stopped, starting, running, training-active, error.
- **Right-click menu** — Open Dashboard, Settings, Start/Stop Server, View Logs, Quit.
- **Dashboard window** — toolbar shows server state, model path, VRAM budget, active LoRA adapter, training status, and the OpenAI base URL with a one-click copy. The kiln server's built-in `/ui` is embedded below via a webview iframe.
- **Settings window** — model path, host, port, adapter directory, inference memory fraction, FP8 KV cache, CUDA graphs, prefix cache, speculative decoding, auto-start, auto-restart, launch-at-login. Persisted via `tauri-plugin-store`.
- **Log viewer** — tails captured stdout/stderr lines from the ring buffer with auto-scroll and clear-view.
- **Health and training polling** — hits `/v1/health` and `/v1/train/status` and drives the tray icon state.
- **Auto-start kiln on app launch** — configurable; default on.
- **Launch-at-login** — Windows, Linux, and macOS (via LaunchAgent).
- **Real Kiln logo icons** — PNG/ICO/ICNS baked into all platform bundles.
- **GitHub Actions CI** — builds Windows MSI and NSIS installers and Linux `.deb` / `.AppImage`, attaches artifacts to tag releases.

See **[CHANGELOG.md](CHANGELOG.md)** for older versions and full release history.

## Screenshots

**Dashboard** — toolbar pills surface server state, model path, VRAM, active adapter, training status, and the OpenAI base URL (click-to-copy), with the kiln server's `/ui` embedded below.

![Dashboard](../docs/desktop/dashboard.png)

**Settings** — model path picker, HuggingFace downloader, host/port, runtime knobs (FP8 KV cache, CUDA graphs, prefix cache, inference VRAM fraction), and startup options.

![Settings](../docs/desktop/settings.png)

**Logs** — tails kiln's stdout/stderr from the in-process ring buffer.

![Logs](../docs/desktop/logs.png)

## Architecture

The app is a [Tauri v2](https://v2.tauri.app/) project (Rust backend, HTML/JS frontend) that spawns and supervises the `kiln` binary as a **child process**. Kiln is NOT embedded as a library — it is a heavyweight GPU server (CUDA on Linux/Windows, Metal on macOS), and keeping it as a separate binary preserves headless usage and avoids dragging candle/CUDA/Metal into the Tauri build.

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
    kiln binary (separate process)  ── CUDA / Metal, model, training loop
```

## Workspace isolation

This crate is **not** a member of the root `kiln` Cargo workspace. `desktop/Cargo.toml` declares its own empty `[workspace]` section so it is its own workspace root. This keeps Tauri and system-webview dependencies out of the inference workspace and vice versa, and means the desktop app builds without CUDA or Metal — only the `kiln` child process needs a GPU toolchain.

## Build

```bash
cd desktop
cargo check                 # quick validation
cargo build --release       # release build
cargo tauri build           # full installer bundle (.deb/.AppImage/.msi/.exe/.dmg/.app)
```

System libraries are required on Linux: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `patchelf`. CI runs on `ubuntu-22.04`, `windows-latest`, and `macos-14` and has everything preinstalled.

Packaging targets in `tauri.conf.json`:

- Linux: `.deb`, `.AppImage`
- Windows: `.msi`, `.nsis`
- macOS: `.app`, `.dmg` (Apple Silicon only)

### macOS (Apple Silicon)

Prerequisites: Xcode Command Line Tools (`xcode-select --install`) and a
recent stable Rust toolchain. The Tauri CLI is a dev dependency you can
install once with `cargo install tauri-cli --locked`.

```bash
cd desktop
cargo check                                  # quick validation
cargo tauri dev                              # hot-reload dev run
cargo tauri build --target aarch64-apple-darwin   # produces .app + .dmg
```

Artifacts land in `desktop/target/aarch64-apple-darwin/release/bundle/`:

- `macos/Kiln Desktop.app` — the bundled app
- `dmg/Kiln Desktop_<version>_aarch64.dmg` — the drag-to-Applications installer

The app targets macOS 11.0 (Big Sur) or later — that's the earliest arm64-only
release, and we deliberately do not ship an x86_64 build.

**Code signing and notarization.** For public distribution you must sign with
an Apple Developer ID Application certificate and notarize via Apple's
notary service, otherwise Gatekeeper will block first-launch. Configure via
the standard Tauri env vars (`APPLE_CERTIFICATE`, `APPLE_SIGNING_IDENTITY`,
`APPLE_ID`, `APPLE_TEAM_ID`, `APPLE_PASSWORD`) — see the
[Tauri macOS signing docs](https://v2.tauri.app/distribute/sign/macos/).
Local dev and CI ad-hoc builds work unsigned; users will need to right-click
and "Open" to bypass Gatekeeper on an unsigned build, or run
`xattr -dr com.apple.quarantine /Applications/Kiln\ Desktop.app`.

**Menu-bar integration.** Tauri's `tray-icon` crate maps the same code path
we use for Windows/Linux onto `NSStatusItem` on macOS — no platform-specific
Rust is needed in this crate.

## Layout

```
desktop/
├── Cargo.toml         # own workspace root (empty [workspace] section)
├── build.rs           # tauri-build
├── tauri.conf.json    # Tauri v2 config
├── src/               # Rust backend (supervisor, tray, polling, IPC commands)
├── ui/                # HTML frontend — dashboard, settings, logs, index
└── icons/             # Kiln logo icons (.png for Linux, .ico for Windows, .icns for macOS)
```

## CI

Release workflow lives in `.github/workflows/desktop-build.yml`. On `desktop-v*` tag push it builds the Tauri bundle on `ubuntu-22.04`, `windows-latest`, and `macos-14` and publishes the artifacts to the matching GitHub Release. The macOS job runs `--target aarch64-apple-darwin` to produce an Apple Silicon `.dmg` + `.app`.

## Troubleshooting

Common first-run issues and how to recover.

- **Kiln server binary failed to download on first launch.** The desktop app fetches the matching `kiln` server binary from GitHub Releases on first run. If the download fails, check your network/firewall (especially proxies blocking `github.com` or `objects.githubusercontent.com`), then retry from Settings → "Check for updates / Re-download". As a manual fallback, grab the binary from <https://github.com/ericflo/kiln/releases/latest> and point Settings → "Kiln binary path" at the unpacked file.

- **"Model path not set" on the Dashboard.** No weights are bundled. Open Settings, then either use the HuggingFace downloader (default `Qwen/Qwen3.5-4B`) or point the Model path picker at an existing safetensors directory and click Save. The Dashboard will pick up the change once the server restarts.

- **CUDA driver too old.** Settings shows the detected NVIDIA driver and the minimum required by the selected kiln release. If the gate trips ("CUDA driver too old: …  Update blocked"), update your NVIDIA driver to at least the version shown — see <https://www.nvidia.com/drivers> — then restart the desktop app. The same gate blocks in-app server-binary updates until the driver is current.

- **Port already in use.** Default server port is `8080`. If another process is bound, open Settings, change "Port" to a free port (e.g. `8090`), and Save. Then restart the server from the tray menu and update any OpenAI-compatible client to the new `http://localhost:<port>/v1` base URL.

- **Server keeps crashing or restarts in a loop.** Open the Logs window (Ctrl/Cmd+L) to inspect the captured stderr. The most common causes are: corrupt or partial model weights (re-download), insufficient VRAM for the model + KV cache (lower the inference memory fraction in Settings, or enable FP8 KV cache), and a CUDA runtime/driver mismatch (see above). The supervisor backs off exponentially between restart attempts; "Stop server" from the tray will halt the loop.

- **How to uninstall.**
  - **Windows:** Settings → Apps → "Kiln Desktop" → Uninstall (or rerun the original `.exe` installer and choose Remove).
  - **Linux (.deb):** `sudo dpkg -r kiln-desktop`. For the AppImage, just delete the `.AppImage` file.
  - **macOS:** drag `Kiln Desktop.app` from `/Applications` to the Trash.
  - App data (settings, downloaded kiln binary, logs) lives in the OS-specific app data directory — `%APPDATA%\com.kiln.desktop` on Windows, `~/.local/share/com.kiln.desktop` on Linux, `~/Library/Application Support/com.kiln.desktop` on macOS. Remove it manually for a clean reinstall.
