# Kiln Desktop Changelog

## desktop-v0.2.2 — 2026-04-25

Coordinated release aligned with kiln-v0.2.2 server. No desktop-side
behavior changes since desktop-v0.2.0; this cut realigns the desktop
version number with the server after the unpublished kiln-v0.2.1 cycle
and ships the new auto-downloaded server binary with all of v0.2.2's
prefix-cache, Metal/CUDA fusion, MTP audit, dependency, and governance
work. See [CHANGELOG.md](../CHANGELOG.md) for the full server changelog.

## desktop-v0.2.0 — 2026-04-24
- Coordinated release aligned with kiln-v0.2.0 server
- Picks up macOS startup and KV prefill overhead reductions
- Gate desktop readiness on inference prewarm so the tray "server ready" state reflects actual readiness
- Speed up macOS default MTP decode

## desktop-v0.1.11 — 2026-04-21
- Notify when the Dashboard Copy URL button fails (#315)
- Notify when the About window Copy Diagnostics fails (#312)
- Notify when Copy/Save in the log viewer fails (#310)
- Notify when Open Logs / Open Settings invocations fail (#305)
- Notify when Check for Updates or Install Update fails (#300)
- Notify when dashboard Stop/Restart Server buttons fail (#296)
- Notify when the launch-at-login toggle fails to apply (#292)
- Notify when "Open Kiln UI in Browser" fails to launch the default browser (#285)
- Notify tray Start/Stop/Restart synchronous errors (#277)
- Keep desktop starting until health passes
- Guard Settings Reload button against unsaved changes (#270)
- Suppress spurious "Kiln server ready" toast on Training → Running transition (#264)
- Notify on Copy OpenAI Base URL success + clipboard failure (#262)
- Notify when Copy OpenAI Base URL clicked while server not running (#259)
- Preflight tray Restart Server — notify when model_path unset / binary missing (#254)
- Preflight tray Start Server when kiln binary missing (#252)
- Preflight tray Start Server — notify + open Settings when model_path unset (#250)
- Enrich About Copy diagnostic info with binary path, model path, and server state (#249)
- Add Troubleshooting section covering binary download, model path, CUDA driver, ports, logs, uninstall (#248)
- Add Screenshots section to `desktop/README.md` (#246)
- Link `desktop/CHANGELOG.md` from root + desktop READMEs (#242)
- Add CHANGELOG and auto-populate release notes (#239)
- Refresh `desktop/README.md` to v0.1.10 (#236)

## desktop-v0.1.10 — 2026-04-20
- Updater parses `supported_sm` from kiln release notes
- Updater parses `min_cuda` and refuses to install when the local CUDA driver is too old
- Surface `cuda_driver_too_old` error in the Settings updater UI

## desktop-v0.1.9 — 2026-04-20
- Picks up the kiln-v0.1.2 server fix
- Updater GPU arch compatibility gate via `nvidia-smi compute_cap`
- `current_target()` extended to Linux + Windows CUDA 12.4
- Auto-check for a new kiln binary on launch
- Wire the Install button to `install_staged_update`
- Docs: desktop app auto-downloads a prebuilt kiln on Linux + Windows

## desktop-v0.1.8 — 2026-04-19
- Updater slice 2: download the new kiln binary to a staging path
- Updater slice 3a: extract the staged tarball to `bin/kiln.new` + verify sha256
- Updater slice 3b: atomic swap of `bin/kiln.new` into `bin/kiln`
- Health-gated updater install with rollback to `kiln.bak`
- Check-for-updates command and Settings UI (detection only)
- Server + desktop: configurable served model ID

## desktop-v0.1.7 — 2026-04-19
- Click the base-URL pill to copy the OpenAI base URL
- Click the model-path pill to copy the model path
- Click the VRAM pill to copy the VRAM value
- Click the adapter pill to copy the active LoRA name
- Log viewer: stdout/stderr stream filter checkboxes
- Fix HuggingFace download placeholder to `Qwen/Qwen3.5-4B`
- Design doc for kiln binary auto-update

## desktop-v0.1.6 — 2026-04-19
- Supersedes the broken v0.1.5 draft
- Fix v0.1.4 crashloop by passing kiln server settings as `KILN_*` env vars
- Color stderr lines amber in the log viewer
- "Download from HuggingFace" button in Settings
- Persist window size and position across launches

## desktop-v0.1.4 — 2026-04-19
- Warn on non-existent paths in Settings
- "Restore defaults" button in the Settings window
- Inline help text for advanced settings
- Dirty-state indicator on the Settings Save button
- Confirm discard when closing Settings with unsaved changes
- Keyboard shortcut help modals (dashboard, settings, logs)
- Dashboard start/stop shortcuts (`Ctrl/Cmd+Shift+S`, `Ctrl/Cmd+Shift+.`)
- `Cmd/Ctrl+W` and `Esc` close shortcuts for settings and logs windows
- Auto-download the signed kiln server on first run
- Wrap the dashboard toolbar gracefully at narrow widths

## desktop-v0.1.2 — 2026-04-18
- Replace the stale "Scaffold only" placeholder in `ui/index.html`
- Notify on server-ready and server-stopped transitions
- Fix silent button no-ops (set `withGlobalTauri` + About close)
- Sync `Cargo.lock` with the `tauri-plugin-notification` dependency

## desktop-v0.1.1 — 2026-04-18
- Sign and notarize the macOS `.dmg` for Gatekeeper
- Gracefully stop the kiln server on tray Quit
- About window with version, repo link, and license
- About window: show OS/arch + Copy diagnostic button
- Dashboard first-run empty state when `model_path` is unset
- Dashboard toolbar: live server state badge, server uptime, OpenAI base URL, Stop/Restart Server buttons, configured model path, VRAM budget, active adapter, training status pill
- Dashboard toolbar regrouped into status (left) and actions (right)
- Dashboard Start Server button for the not-running state
- Auto-show the error screen when kiln crashes while the dashboard is open
- Native notification when the server enters Error state
- Include kiln port in the tray tooltip when running
- Per-state tray icons (stopped/starting/running/training/error)
- Open Kiln UI in an external browser from the tray
- Restart Server tray menu item + status indicator row at the top of the tray menu
- View Logs / Settings buttons in the dashboard toolbar
- Log viewer: filter input, copy button, Save-to-file button
- Keyboard shortcuts for the dashboard (Logs, Settings, Copy, Restart)
- Keyboard shortcuts for the settings window (`Cmd/Ctrl+S`, `Cmd/Ctrl+R`)
- Keyboard shortcuts for the logs window (Filter, Save, Clear)
- Copy OpenAI base URL quick action in the tray
- Native file pickers for kiln binary, model path, and adapter directory
- "Restart now" button in Settings after save
- Tauri auto-updater plumbing + Check-for-Updates flow (tray + dashboard)
- Show app version in the settings window footer
- Refreshed README screenshots and dashboard description
- macOS support via candle-metal

## desktop-v0.1.0 — 2026-04-17
- Initial release
- Tauri v2 `desktop/` project scaffold (not a Cargo workspace member)
- Subprocess supervisor for the kiln server
- System tray with status icon and right-click menu
- Settings window with persistence and Supervisor wiring
- Dashboard window embedding kiln's `/ui` via webview
- Launch-at-login via `tauri-plugin-autostart`
- Poll `/v1/health` and `/v1/train/status` to drive tray state
- Log viewer window tailing supervisor stdout/stderr
- CI building Windows MSI/NSIS and Linux `.deb`/AppImage packages
- Kiln logo desktop icons
