# Kiln Desktop

System-tray app that wraps the `kiln` local LLM server in a GUI. Intended for users who don't want to manage the CLI.

## Platform scope

Windows and Linux only. Kiln is CUDA-only today; macOS is deliberately out of scope.

## Architecture

The desktop app is a [Tauri v2](https://v2.tauri.app/) project (Rust backend, HTML/JS frontend) that spawns and supervises the `kiln` binary as a **child process**. Kiln is NOT embedded as a library — it is a heavyweight CUDA server, and keeping it as a separate binary preserves headless usage and avoids dragging candle/CUDA into the Tauri build.

## Workspace isolation

This crate is **not** a member of the root `kiln` Cargo workspace. `desktop/Cargo.toml` declares its own empty `[workspace]` section so it is its own workspace root. This keeps Tauri and system-webview dependencies out of the inference workspace and vice versa.

## Build

```bash
cd desktop
cargo check
```

System libraries are required on Linux: `libwebkit2gtk-4.1-dev`, `libgtk-3-dev`, `libayatana-appindicator3-dev`, `librsvg2-dev`, `patchelf`. On environments without those, `cargo check` will fail during system-dep probing — that's expected; CI has the libraries installed.

Packaging targets are configured in `tauri.conf.json`:

- Linux: `.deb`, `.AppImage`
- Windows: `.msi`, `.nsis`

Actual bundling (`cargo tauri build`) comes in a later task alongside CI.

## Layout

```
desktop/
├── Cargo.toml         # own workspace root (empty [workspace] section)
├── build.rs           # tauri-build
├── tauri.conf.json    # Tauri v2 config
├── src/main.rs        # Rust entry
├── ui/index.html      # static frontend (no bundler)
└── icons/             # placeholder app icons (real icons ship later)
```

## Status

Scaffold only. The next tasks add tray icon, kiln subprocess supervision, dashboard window, settings persistence, log viewer, and CI packaging.
