# Kiln binary auto-update (design)

Status: design only, not implemented. Follow-up task will wire this into
`desktop/src/installer.rs` and the dashboard.

## Problem

The desktop app spawns `kiln` as a subprocess. `desktop/src/installer.rs`
already downloads, verifies, extracts, and chmods the kiln binary on first
run (macOS, Linux, and Windows release targets — see `current_target()`
in installer.rs). Once
installed, that binary is frozen forever: the app does not notice new
`kiln-v*` releases, so users miss kernel and training improvements
(Phase 6 work, bug fixes, new endpoints) unless they reinstall the whole
desktop app.

We want a minimal, honest update path:

- Reuse the existing download/verify pipeline rather than reinvent it.
- Keep the app's supervisor in charge of stop/start — no magic hot-swap.
- Start manual (a button), not background-silent.
- Fail safe (rollback) on any step.

## Non-goals

- Auto-updating the Tauri app shell itself. That's a separate project
  (`tauri-plugin-updater` + signed `latest.json` / `.sig`; see the
  `tauri-updater-wiring-checklist` note).
- macOS updates when the user is running a source build — kiln's prebuilt
  Apple Silicon + Metal asset already exists, but users on other macOS
  configurations are out of scope.
- Delta / binary-diff updates. The kiln binary is tens of MB; full
  replacement is fine.
- Mandatory or scheduled updates.
- Adapter / LoRA updates. Listed as an open question at the bottom.

## Update source: GitHub Releases

Kiln already publishes on `kiln-v*` tags via
`.github/workflows/server-release.yml`. The desktop installer discovers the
newest release with a matching asset suffix. We extend the same mechanism.

Asset naming convention (extension of the existing
`aarch64-apple-darwin-metal`):

| Platform | Asset suffix | CUDA |
| --- | --- | --- |
| macOS aarch64 Metal | `aarch64-apple-darwin-metal` | n/a |
| Linux x86_64 CUDA 12.4 | `x86_64-unknown-linux-gnu-cuda124` | 12.4 |
| Linux x86_64 Vulkan | `x86_64-unknown-linux-gnu-vulkan` | n/a |
| Windows x86_64 CUDA 12.4 | `x86_64-pc-windows-msvc-cuda124` | 12.4 |

Each tarball carries a companion `.sha256`. The installer already refuses
to install an asset without one (installer.rs:445-452) — keep that
invariant.

On Linux, the desktop app chooses the CUDA asset when an NVIDIA GPU is
present and the Vulkan asset on AMD/Intel-only hosts. `KILN_DESKTOP_GPU_BACKEND`
or `KILN_DESKTOP_SERVER_TARGET` can override the default when a multi-GPU host
needs a specific binary.

### Why GitHub Releases and not a custom channel

- It's already where the binary lives. No new infra.
- Release notes are the natural place to publish per-release metadata
  (min CUDA version, supported SM archs, known-bad models).
- `GET /repos/ericflo/kiln/releases` is anonymous and rate-limits are
  comfortable for desktop polling (60/hr per IP). If we outgrow that we
  can add a conditional ETag check before escalating to a CDN.

## Trigger model

v1: **manual only**. Settings window adds a "Check for updates" button
next to the kiln binary version label. The button emits a Tauri command
that runs the check + (if offered) install pipeline.

v2 (follow-up, gated on v1 shipping): **check on launch**, non-blocking.
Runs in a tokio task after supervisor is healthy; if a newer version
exists, badge the tray icon + show a banner in the dashboard. Still
requires a click to install.

No silent background replacement in v1. Downloading a new binary and
restarting the server without user consent is a surprising behavior for
a local inference tool where users may be mid-conversation.

## Version resolution

### Current version

Invoked once at startup, cached in `SupervisorState`:

```rust
async fn current_kiln_version(bin: &Path) -> Option<String> {
    let out = tokio::process::Command::new(bin)
        .arg("--version")
        .output()
        .await
        .ok()?;
    if !out.status.success() { return None; }
    // `kiln --version` prints `kiln <semver>` or `kiln <semver> (<git-sha>)`.
    let stdout = String::from_utf8_lossy(&out.stdout);
    stdout.split_whitespace().nth(1).map(|s| s.to_string())
}
```

If `--version` fails, treat the install as "unknown version" — always
offer the latest release and let the user decide.

### Latest version

Reuse `discover_asset` from installer.rs:138. It already walks releases
in listing order (newest first) and returns `AssetPair { version, tarball,
sha256 }`.

### Compare

Use `semver` (already in the Cargo.lock transitively). Compare
`current < latest` by parsed `Version`. Pre-release suffixes compare
lexically per semver, which is the behavior we want for
`0.5.0-rc.1 < 0.5.0`.

If either side fails to parse as semver, fall back to string equality —
"nothing offered unless strings differ" — to avoid suggesting a downgrade.

## Download + swap flow

The `install_latest_server` function already handles steps 1-4. The new
`update_to_latest` function builds on it:

```
┌────────────────────────────────────────────────────────────────┐
│ 1. discover_asset()  — newest kiln-v* release for this target  │
│ 2. Compare versions; no-op if current >= latest                │
│ 3. download_with_progress()  → staging tarball + sha256        │
│ 4. fetch_expected_sha256() + compare; abort+delete on mismatch │
│ 5. supervisor.stop()  (awaited; see gotcha below)              │
│ 6. Atomic swap:                                                │
│      rename bin/kiln   → bin/kiln.bak                          │
│      extract tarball → bin/kiln.new                            │
│      rename bin/kiln.new → bin/kiln                            │
│ 7. chmod +x (unix) / clear quarantine (macos)                  │
│ 8. supervisor.start()                                          │
│ 9. Health check: wait up to 30s for /v1/health 200             │
│10. If healthy: delete kiln.bak. If not: rollback.              │
└────────────────────────────────────────────────────────────────┘
```

### Why stop before swap

Windows won't let you rename a running executable (it holds a share-lock
on the image). Linux will let you rename over a running binary but the
kernel keeps the old inode open; any subsequent `execve` of `bin/kiln`
will use the new file, which is what we want, but we still need a clean
stop+start so the supervisor's restart counter and health state reset
honestly. Easiest correct answer: stop first everywhere.

See the `tauri-supervised-subprocess-graceful-quit` note — the supervisor
already `await`s child shutdown before declaring `Stopped`
(supervisor.rs:160-168), so we get this for free as long as we
`supervisor.stop().await` before the rename.

### Atomic swap details

- Extract into a sibling file `kiln.new` rather than the final path, so
  a crash during extract never leaves a half-written `bin/kiln`.
- `std::fs::rename` is atomic within a filesystem. `bin/` is always on
  `app_data_dir`; keep the staging file there.
- On Windows, `rename` over an existing file fails with `ERROR_ALREADY_EXISTS`.
  Use `MoveFileExW` with `MOVEFILE_REPLACE_EXISTING`, exposed via
  `std::fs::rename` on stable Rust (works since 1.5 on Windows).

### Health check gate

After `supervisor.start()`, poll `/v1/health` every 500ms for 30s. The
kiln server normally comes up in ~3s after model load; slow paths
(first-time GPU context init, large model) can take 15-20s. 30s with a
hard ceiling is the right envelope.

If health never flips green: roll back (see below) and emit an
`update-failed` event to the dashboard with the captured stderr tail
from the supervisor's ring buffer.

### Rollback

Kept as `bin/kiln.bak` for the duration of the health check.

- Health green within 30s → delete `bin/kiln.bak`.
- Health red, or `kiln --version` on the new binary failed to parse →
  `supervisor.stop()`, `rename bin/kiln.bak → bin/kiln`,
  `supervisor.start()`, recheck.
- If rollback itself fails (very rare — filesystem error), surface a
  "kiln is broken, reinstall required" state in the dashboard and point
  to `install_latest_server` as recovery.

Only ever keep one `.bak`. Older backups are not useful — we always have
the release list on GitHub to retry from.

## Signing / trust

v1 relies on HTTPS + sha256 hash verification. The `.sha256` companion
asset is fetched over the same HTTPS GitHub Release endpoint, so an
attacker who can rewrite one can rewrite the other. This is the same
threat model as the existing first-install path. Good enough for v1,
called out as future work:

- **Windows**: Authenticode sign the `kiln.exe` binary in the release
  workflow with a code-signing certificate. Unsigned binaries trip
  SmartScreen on install and show "Windows protected your PC" on launch.
- **Linux**: sigstore / cosign. Publish a `.sig` alongside each tarball;
  the installer verifies against a public key pinned in the desktop app.
  Equivalent to what Tauri's updater plugin does for the app shell (see
  `tauri-updater-pubkey-format` — the embedded pubkey pattern).
- **macOS**: already covered. Developer ID signed + notarized in
  `.github/workflows/server-release.yml`.

A single shared signing story across all three OS would use cosign for
everything, with the desktop app embedding the kiln release signing
pubkey at build time. Out of scope for v1.

## Storage layout

Follows the existing `install_dir(&app)` helper at installer.rs:106.
Tauri resolves `app_data_dir` to:

| Platform | Path |
| --- | --- |
| Linux | `$XDG_DATA_HOME/com.ericflo.kiln/bin/` (default `~/.local/share/com.ericflo.kiln/bin/`) |
| Windows | `%APPDATA%\com.ericflo.kiln\bin\` |
| macOS | `~/Library/Application Support/com.ericflo.kiln/bin/` |

Inside that `bin/` directory:

```
bin/
  kiln            # current, executable
  kiln.bak        # previous (only during an in-flight update)
  kiln.new        # staging during extract (only during extract)
```

No versioned subdirs. Keeping only "current" + "previous" is simpler and
avoids disk-space surprises.

### Package vs download

- **macOS installer**: ship `kiln` inside the `.app` bundle? Not yet.
  Keep the existing "download on first run" flow. Lets us cut desktop
  releases without also cutting a kiln release, and the first-run
  progress UI is already polished.
- **Windows/Linux installers**: same — do not embed. This also keeps the
  MSI / AppImage / .deb small and lets us update the kiln binary
  without a desktop app release.

Trade-off: a first-run network is required. This matches the existing
macOS behavior and is how most local-LLM tools ship (Ollama, LM Studio,
etc.).

## CUDA / GPU compat

Kiln only supports CUDA 12.4 and these SM archs today (from the kiln
SKILL notes):

- SM 80 (A100), 86 (A6000, 3090), 89 (4090, L40), 90 (H100)
- Blackwell SM 120a **not** supported under CUDA 12.4

Before offering an update on Linux/Windows, the desktop should:

1. Detect the local GPU arch via `nvidia-smi --query-gpu=compute_cap
   --format=csv,noheader` (falls back gracefully when `nvidia-smi` is
   absent — no GPU = no update offered).
2. Parse the release notes for a machine-readable
   `supported_sm: [80, 86, 89, 90]` line (new convention — added to the
   release-notes template in the follow-up task).
3. Refuse to install a release whose `supported_sm` does not include the
   local arch, and surface the reason in the dashboard: "This kiln
   release does not support your GPU (SM 120). Staying on v0.4.2."
4. Also surface the CUDA runtime version. If the system CUDA is older
   than the release's minimum, refuse.

On macOS this whole check is skipped — only one Apple Silicon + Metal
target exists.

## Pipeline integration (sketch)

New module: `desktop/src/updater.rs`. Exposed Tauri commands:

```rust
#[tauri::command]
async fn check_for_updates(
    app: AppHandle,
    supervisor: State<'_, Arc<Supervisor>>,
) -> Result<UpdateStatus, String> { ... }

#[tauri::command]
async fn install_update(
    app: AppHandle,
    supervisor: State<'_, Arc<Supervisor>>,
) -> Result<InstalledVersion, String> { ... }
```

`UpdateStatus` mirrors the existing `InstallProgress` enum vocabulary so
the dashboard can reuse rendering:

```rust
#[derive(Serialize)]
#[serde(tag = "state", rename_all = "snake_case")]
pub enum UpdateStatus {
    UpToDate { current: String },
    UpdateAvailable { current: String, latest: String, asset: String, supported: bool },
    NotSupported { reason: String },
}
```

Events emitted during `install_update` match the existing installer
events plus a new `kiln-update-rollback`:

- `kiln-install-progress` (reused)
- `kiln-install-done` (reused, means "update installed")
- `kiln-install-failed` (reused)
- `kiln-update-rollback { from, to, reason }` (new)

## Open questions

1. **Release channels.** Should the desktop app expose a "stable" vs
   "nightly" toggle in Settings, with nightly pulling from tags like
   `kiln-v0.5.0-nightly.<sha>`? Keeps contributors dogfooding kernel
   work without breaking the default user. Mild yes from me, defer to
   Eric.
2. **Adapter auto-update.** LoRA adapters change more often than the
   kiln binary (training fine-tunes produce new adapters continuously).
   Should desktop also poll the user's adapter directory for
   server-pushed updates, or is that out of scope for the binary
   updater entirely? I lean "out of scope" — adapters are data, not
   code; treat them separately.
3. **"Update available" tray badge vs. dashboard banner.** Tray icon
   real estate is precious; the training-active indicator already uses
   it. Probably just dashboard + settings surface the offer, and we
   skip tray noise for updates.
4. **Rate limit.** GitHub anonymous `GET /releases` is 60/hr. Auto-check
   on launch is fine; auto-check every hour while the app is running
   would burn through this on long-lived sessions. Propose: on launch
   plus manual button only. Re-evaluate if we need faster rollout.
5. **CUDA runtime detection on Windows.** `nvidia-smi` is on `%PATH%`
   when the NVIDIA driver is installed, but not guaranteed. Do we fall
   back to reading registry keys, or accept "no `nvidia-smi`" → no
   update offered?
6. **Version pinning.** Do we want a Settings toggle to pin the current
   kiln version (skip this offered update, and remind me in N days)? I
   lean no — kiln is a hot-moving kernel project, the honest default
   is "always offer the latest."

## Appendix: References

- Existing first-install: `desktop/src/installer.rs:64-472`
- Supervisor stop/start semantics: `desktop/src/supervisor.rs:160-200`
- Release workflow: `.github/workflows/server-release.yml`
- macOS signing: `docs/desktop/signing.md`
- GPU compat matrix: kiln `SKILL.md` → "GPU compat (CUDA 12.4)"
- Tauri updater pattern (for later app-shell updates): agent note
  `tauri-updater-wiring-checklist`
