//! Auto-download, verify, and install the `kiln` server binary.
//!
//! When the desktop app is launched for the first time there is no `kiln`
//! binary on the system. Rather than show a raw `No such file or directory`
//! error, the dashboard surfaces an onboarding state and calls into this
//! module to fetch the signed, notarized binary from the latest
//! `kiln-v*` GitHub release, verify its SHA256, and install it into the
//! app's data directory. The supervisor then spawns from that path.
//!
//! Platform coverage today is macOS aarch64 + Metal only; see
//! [`current_target`]. Other platforms fall through to "build from source"
//! in the UI.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use tauri::{AppHandle, Emitter, Manager};
use tokio::fs;
use tokio::io::AsyncWriteExt;

pub const INSTALL_PROGRESS_EVENT: &str = "kiln-install-progress";
pub const INSTALL_DONE_EVENT: &str = "kiln-install-done";
pub const INSTALL_FAILED_EVENT: &str = "kiln-install-failed";

pub const UPDATE_DOWNLOAD_PROGRESS_EVENT: &str = "download_update_progress";
pub const UPDATE_DOWNLOAD_DONE_EVENT: &str = "download_update_done";
pub const UPDATE_DOWNLOAD_FAILED_EVENT: &str = "download_update_failed";

pub const EXTRACT_UPDATE_PROGRESS_EVENT: &str = "extract_update_progress";
pub const EXTRACT_UPDATE_DONE_EVENT: &str = "extract_update_done";
pub const EXTRACT_UPDATE_FAILED_EVENT: &str = "extract_update_failed";

pub const INSTALL_STAGED_UPDATE_DONE_EVENT: &str = "install_staged_update_done";
pub const INSTALL_STAGED_UPDATE_FAILED_EVENT: &str = "install_staged_update_failed";

/// Subdirectory under `app_data_dir` where update tarballs are staged
/// before the slice-3 atomic-swap step promotes them.
pub const UPDATE_STAGING_DIR: &str = "kiln-updates";

/// Filename of the staged extracted binary produced by slice 3a. Lives
/// next to `bin/kiln` so slice 3b's atomic rename stays inside the same
/// filesystem. Platform-independent on purpose: the atomic-swap step
/// targets `bin/kiln` on unix and `bin/kiln.exe` on Windows, but the
/// sibling staging file is always `bin/kiln.new`.
pub const NEW_BINARY_NAME: &str = "kiln.new";

/// Filename of the single retained backup of the prior `kiln` binary
/// kept next to `bin/kiln` for the duration of the post-swap health
/// gate. Only one `.bak` is ever kept — slice 3c promotes the swap to
/// success by deleting it, or rolls back by renaming it over
/// `bin/kiln`.
pub const BACKUP_BINARY_NAME: &str = "kiln.bak";

const RELEASES_URL: &str = "https://api.github.com/repos/ericflo/kiln/releases";
const USER_AGENT: &str = concat!("kiln-desktop/", env!("CARGO_PKG_VERSION"));
/// Upper bound on release-asset size we're willing to stream. A macOS Metal
/// build of kiln is tens of MB today; pathological values would indicate a
/// wrong asset match or a corrupted release manifest.
const MAX_ASSET_BYTES: u64 = 2 * 1024 * 1024 * 1024;

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum InstallProgress {
    FetchingManifest,
    Downloading {
        received: u64,
        total: Option<u64>,
    },
    Verifying,
    Extracting,
    Finalizing,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallDone {
    pub path: String,
    pub version: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallFailed {
    pub error: String,
    pub cancelled: bool,
}

/// Progress payload for the slice-2 "download update tarball to staging"
/// pipeline. Separate from [`InstallProgress`] because the update flow
/// is simpler (download-only, no extract/verify yet) and the UI wants to
/// key progress bars by `version`.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateDownloadProgress {
    pub version: String,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateDownloadDone {
    pub version: String,
    pub path: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct UpdateDownloadFailed {
    pub version: String,
    pub error: String,
    pub cancelled: bool,
}

/// Progress payload for the slice-3a "extract + verify" pipeline. The
/// `total_bytes` field is `None` until extraction completes because the
/// tar entry's header-declared size is read lazily; a later slice may
/// pre-scan the archive to report total up front if the UX wants a smooth
/// progress bar instead of a two-step animation.
#[derive(Debug, Clone, Serialize)]
pub struct ExtractUpdateProgress {
    pub version: String,
    pub extracted_bytes: u64,
    pub total_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtractUpdateDone {
    pub version: String,
    pub path: String,
    pub bytes: u64,
    pub sha256_ok: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ExtractUpdateFailed {
    pub version: String,
    pub error: String,
    pub cancelled: bool,
}

/// Terminal payload for the slice-3b "atomic swap + restart" pipeline.
/// `path` is the path that now answers as `bin/kiln` — i.e. the
/// previously-staged `kiln.new` under its new name. Health-check gating
/// and rollback are slice 3c and intentionally not represented here.
#[derive(Debug, Clone, Serialize)]
pub struct InstallStagedUpdateDone {
    pub path: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct InstallStagedUpdateFailed {
    pub error: String,
}

/// Release-asset suffix for the current OS+arch+features combination, or
/// `None` when no prebuilt release exists for this platform. The suffix
/// matches the naming convention in `.github/workflows/server-release.yml`
/// (e.g. `aarch64-apple-darwin-metal`).
///
/// The CUDA 12.4 suffix on Linux/Windows isn't the Rust target triple —
/// it's the triple plus a `-cuda124` feature tag that the release
/// workflow bakes into the archive filename so the desktop app can tell
/// a CUDA 12.4 build apart from a future CUDA 13 build without a
/// manifest lookup.
pub fn current_target() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Some("aarch64-apple-darwin-metal");
    }
    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    {
        return Some("x86_64-unknown-linux-gnu-cuda124");
    }
    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    {
        return Some("x86_64-pc-windows-msvc-cuda124");
    }
    #[allow(unreachable_code)]
    None
}

/// Whether a downloadable prebuilt kiln binary exists for this platform.
pub fn supports_auto_install() -> bool {
    current_target().is_some()
}

/// Resolve the configured `kiln` binary path. A bare name like `"kiln"`
/// is searched on `PATH`; absolute or multi-component paths are checked
/// as-is. Returns `None` when nothing resolvable exists at that location.
pub fn resolve_binary(configured: &Path) -> Option<PathBuf> {
    if configured.as_os_str().is_empty() {
        return None;
    }
    // Treat anything with more than one component or a leading `/` / `.` as
    // a direct filesystem path (not a PATH lookup).
    let direct = configured.is_absolute()
        || configured
            .components()
            .any(|c| matches!(c, std::path::Component::ParentDir))
        || configured.parent().map(|p| !p.as_os_str().is_empty()).unwrap_or(false);
    if direct {
        return configured.is_file().then(|| configured.to_path_buf());
    }
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(configured);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Directory under `app_data_dir` where auto-installed kiln binaries live.
pub fn install_dir(app: &AppHandle) -> Result<PathBuf, String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir unavailable: {}", e))?;
    Ok(base.join("bin"))
}

/// File-extension convention for the release tarball/zip for `target`.
/// Windows releases ship as `.zip`; all other targets ship as `.tar.gz`.
/// Detected purely by substring match on the target triple to keep this
/// helper testable without platform cfg gates.
fn release_archive_ext(target: &str) -> &'static str {
    if target.contains("windows") {
        "zip"
    } else {
        "tar.gz"
    }
}

/// Asset filename for a given kiln release, matching the naming convention
/// in `.github/workflows/server-release.yml`:
///     `kiln-<version>-<target>.<ext>`
///
/// Pure helper — exposed for unit tests and reused by
/// [`release_download_url`] and [`staging_tarball_path`].
pub fn release_asset_name(version: &str, target: &str) -> String {
    format!(
        "kiln-{}-{}.{}",
        version,
        target,
        release_archive_ext(target)
    )
}

/// Full GitHub Releases download URL for the kiln binary at `version` for
/// `target`. Uses the existing `kiln-v*` tag convention
/// (see `.github/workflows/server-release.yml`).
pub fn release_download_url(version: &str, target: &str) -> String {
    format!(
        "https://github.com/ericflo/kiln/releases/download/kiln-v{}/{}",
        version,
        release_asset_name(version, target)
    )
}

/// Staging path inside `app_data_dir` where the downloaded update tarball
/// is written before slice 3's atomic-swap step promotes it. Files land
/// under `<app_data_dir>/kiln-updates/kiln-v<version>.<ext>` so multiple
/// staged updates can coexist without colliding on filename.
pub fn staging_tarball_path(app_data_dir: &Path, version: &str, target: &str) -> PathBuf {
    app_data_dir
        .join(UPDATE_STAGING_DIR)
        .join(format!("kiln-v{}.{}", version, release_archive_ext(target)))
}

/// Path where slice 3a writes the extracted `kiln` binary before the
/// slice-3b atomic rename promotes it to `bin/kiln` (or `bin/kiln.exe`
/// on Windows). See `desktop/docs/binary-update.md` "Storage layout".
pub fn extracted_new_binary_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("bin").join(NEW_BINARY_NAME)
}

/// Filename the supervisor actually `execve`s on this platform. Always
/// `kiln.exe` on Windows, plain `kiln` everywhere else. Kept as a
/// platform-specific constant (rather than `cfg!()` inline at call
/// sites) so tests can match the same string unambiguously.
#[cfg(windows)]
const INSTALLED_BINARY_NAME: &str = "kiln.exe";
#[cfg(not(windows))]
const INSTALLED_BINARY_NAME: &str = "kiln";

/// Path of the `kiln` binary the supervisor actually runs —
/// `<app_data_dir>/bin/kiln` on unix, `<app_data_dir>/bin/kiln.exe` on
/// Windows.
pub fn installed_binary_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("bin").join(INSTALLED_BINARY_NAME)
}

/// Path of the single retained backup of the previous `kiln` binary.
/// Slice 3b renames the currently-installed binary here before
/// promoting `kiln.new`; slice 3c removes it on health-gate success or
/// rolls it back into place on failure.
pub fn backup_binary_path(app_data_dir: &Path) -> PathBuf {
    app_data_dir.join("bin").join(BACKUP_BINARY_NAME)
}

/// Verify that the file at `path` matches `expected` (64-char lowercase
/// hex sha256). Streams the file in 64 KiB chunks so large tarballs and
/// extracted binaries don't balloon memory.
pub fn verify_sha256(path: &Path, expected: &str) -> Result<(), String> {
    use std::io::Read;
    let expected = expected.trim().to_lowercase();
    if expected.len() != 64 || !expected.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!("expected sha256 is not 64-char hex: {}", expected));
    }
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("open {}: {}", path.display(), e))?;
    let mut hasher = Sha256::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| format!("read {}: {}", path.display(), e))?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    let actual = hex_lower(hasher.finalize().as_slice());
    if actual != expected {
        return Err(format!(
            "sha256 mismatch: expected {}, got {}",
            expected, actual
        ));
    }
    Ok(())
}

/// Extract the `kiln` (or `kiln.exe`) binary from a release archive into
/// `target_binary_path`, overwriting any existing file at that path.
/// Returns the number of bytes written.
///
/// tar.gz only today — Windows `.zip` support lands alongside the Windows
/// release asset build. Detection is purely on the archive filename so
/// this helper stays testable without platform cfg gates.
pub fn extract_tarball_to(
    tarball: &Path,
    target_binary_path: &Path,
) -> Result<u64, String> {
    let fname = tarball
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("");
    if fname.ends_with(".zip") {
        return Err(
            "windows .zip extraction lands alongside Windows release asset support"
                .to_string(),
        );
    }
    if !(fname.ends_with(".tar.gz") || fname.ends_with(".tgz")) {
        return Err(format!(
            "unsupported archive extension for {} — expected .tar.gz or .zip",
            tarball.display()
        ));
    }

    if let Some(parent) = target_binary_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }
    if target_binary_path.exists() {
        std::fs::remove_file(target_binary_path).map_err(|e| {
            format!(
                "remove stale {}: {}",
                target_binary_path.display(),
                e
            )
        })?;
    }

    let file = std::fs::File::open(tarball)
        .map_err(|e| format!("open {}: {}", tarball.display(), e))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);

    let entries = archive
        .entries()
        .map_err(|e| format!("read tar entries: {}", e))?;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("tar entry: {}", e))?;
        let rel = entry
            .path()
            .map_err(|e| format!("tar entry path: {}", e))?
            .into_owned();
        // Reject absolute paths and `..` traversal the same way the
        // first-install extractor does (installer.rs `extract_tarball`).
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(format!(
                "tar entry {} escapes archive root",
                rel.display()
            ));
        }
        let name = match rel.file_name().and_then(|s| s.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name != "kiln" && name != "kiln.exe" {
            continue;
        }
        entry.unpack(target_binary_path).map_err(|e| {
            format!("unpack {}: {}", target_binary_path.display(), e)
        })?;
        let size = std::fs::metadata(target_binary_path)
            .map(|m| m.len())
            .map_err(|e| {
                format!("stat {}: {}", target_binary_path.display(), e)
            })?;
        return Ok(size);
    }

    Err(format!(
        "tarball {} did not contain a `kiln` (or `kiln.exe`) binary",
        tarball.display()
    ))
}

#[derive(Debug, Clone, Deserialize)]
struct ReleaseAsset {
    name: String,
    browser_download_url: String,
    #[serde(default)]
    #[allow(dead_code)]
    size: u64,
}

#[derive(Debug, Deserialize)]
struct Release {
    tag_name: String,
    #[serde(default)]
    assets: Vec<ReleaseAsset>,
    /// Full release-notes body (markdown). Parsed by
    /// [`parse_supported_sm`] so the per-release supported-SM list can
    /// move with the release rather than being hardcoded in the desktop
    /// app. `None` when the field is absent (older releases predating the
    /// `supported_sm:` convention).
    #[serde(default)]
    body: Option<String>,
}

struct AssetPair {
    version: String,
    tarball: ReleaseAsset,
    sha256: Option<ReleaseAsset>,
    /// Release-notes body carried forward so callers can feed it to
    /// [`is_supported_sm_for_release`] / [`supported_sm_for_release`].
    release_body: Option<String>,
}

/// Fetch the newest `kiln-v*` release and find the asset pair matching
/// the current target.
async fn discover_asset(
    client: &reqwest::Client,
    target: &str,
) -> Result<AssetPair, String> {
    let resp = client
        .get(RELEASES_URL)
        .header("Accept", "application/vnd.github+json")
        .header("X-GitHub-Api-Version", "2022-11-28")
        .send()
        .await
        .map_err(|e| format!("github releases list failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "github releases list returned HTTP {}",
            resp.status()
        ));
    }
    let releases: Vec<Release> = resp
        .json()
        .await
        .map_err(|e| format!("github releases list parse failed: {}", e))?;
    let want_suffix = format!("-{}.tar.gz", target);
    for release in releases {
        if !release.tag_name.starts_with("kiln-v") {
            continue;
        }
        let Some(tar) = release
            .assets
            .iter()
            .find(|a| a.name.ends_with(&want_suffix))
            .cloned()
        else {
            continue;
        };
        let sha = release
            .assets
            .iter()
            .find(|a| a.name == format!("{}.sha256", tar.name))
            .cloned();
        let version = release
            .tag_name
            .strip_prefix("kiln-v")
            .unwrap_or(&release.tag_name)
            .to_string();
        return Ok(AssetPair {
            version,
            tarball: tar,
            sha256: sha,
            release_body: release.body.clone(),
        });
    }
    Err(format!(
        "no published kiln-v* release has a `-{}.tar.gz` asset",
        target
    ))
}

async fn fetch_expected_sha256(
    client: &reqwest::Client,
    asset: &ReleaseAsset,
) -> Result<String, String> {
    let resp = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .map_err(|e| format!("sha256 fetch failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("sha256 fetch returned HTTP {}", resp.status()));
    }
    let body = resp
        .text()
        .await
        .map_err(|e| format!("sha256 body read failed: {}", e))?;
    let hash = body.trim().split_whitespace().next().unwrap_or("").to_string();
    if hash.len() != 64 || !hash.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(format!(
            "sha256 asset {} did not contain a 64-char hex digest",
            asset.name
        ));
    }
    Ok(hash.to_lowercase())
}

fn emit_progress(app: &AppHandle, p: InstallProgress) {
    let _ = app.emit(INSTALL_PROGRESS_EVENT, p);
}

async fn download_with_progress(
    app: &AppHandle,
    client: &reqwest::Client,
    asset: &ReleaseAsset,
    dest: &Path,
    cancel: &AtomicBool,
) -> Result<String, String> {
    let mut resp = client
        .get(&asset.browser_download_url)
        .send()
        .await
        .map_err(|e| format!("download request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!("download returned HTTP {}", resp.status()));
    }
    let total = resp.content_length();
    if let Some(t) = total {
        if t > MAX_ASSET_BYTES {
            return Err(format!(
                "asset claims {} bytes, over {}-byte safety limit",
                t, MAX_ASSET_BYTES
            ));
        }
    }
    let mut file = fs::File::create(dest)
        .await
        .map_err(|e| format!("create {}: {}", dest.display(), e))?;
    let mut hasher = Sha256::new();
    let mut received: u64 = 0;
    let mut last_reported: u64 = 0;
    emit_progress(
        app,
        InstallProgress::Downloading {
            received: 0,
            total,
        },
    );
    loop {
        if cancel.load(Ordering::SeqCst) {
            return Err("download cancelled".to_string());
        }
        let chunk = resp
            .chunk()
            .await
            .map_err(|e| format!("chunk read failed: {}", e))?;
        let Some(bytes) = chunk else { break };
        received = received.saturating_add(bytes.len() as u64);
        if received > MAX_ASSET_BYTES {
            return Err(format!(
                "aborted — received {} bytes past {}-byte safety limit",
                received, MAX_ASSET_BYTES
            ));
        }
        hasher.update(&bytes);
        file.write_all(&bytes)
            .await
            .map_err(|e| format!("write chunk: {}", e))?;
        // Report at most ~20 times per MB to keep IPC cheap.
        if received.saturating_sub(last_reported) >= 64 * 1024 {
            last_reported = received;
            emit_progress(
                app,
                InstallProgress::Downloading {
                    received,
                    total,
                },
            );
        }
    }
    file.flush()
        .await
        .map_err(|e| format!("flush: {}", e))?;
    emit_progress(
        app,
        InstallProgress::Downloading {
            received,
            total,
        },
    );
    let digest = hasher.finalize();
    Ok(hex_lower(digest.as_slice()))
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

fn extract_tarball(tarball: &Path, dest_dir: &Path) -> Result<PathBuf, String> {
    let file = std::fs::File::open(tarball)
        .map_err(|e| format!("open tarball: {}", e))?;
    let gz = flate2::read::GzDecoder::new(file);
    let mut archive = tar::Archive::new(gz);
    std::fs::create_dir_all(dest_dir)
        .map_err(|e| format!("mkdir {}: {}", dest_dir.display(), e))?;
    // Extract each entry manually and reject absolute paths / .. traversal
    // rather than trusting tar's default unpack.
    let entries = archive
        .entries()
        .map_err(|e| format!("read tar entries: {}", e))?;
    let mut installed: Option<PathBuf> = None;
    for entry in entries {
        let mut entry = entry.map_err(|e| format!("tar entry: {}", e))?;
        let rel = entry
            .path()
            .map_err(|e| format!("tar entry path: {}", e))?
            .into_owned();
        if rel.is_absolute()
            || rel
                .components()
                .any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(format!(
                "tar entry {} escapes archive root",
                rel.display()
            ));
        }
        let out_path = dest_dir.join(&rel);
        if let Some(parent) = out_path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
        }
        entry
            .unpack(&out_path)
            .map_err(|e| format!("unpack {}: {}", out_path.display(), e))?;
        let wanted = if cfg!(windows) { "kiln.exe" } else { "kiln" };
        if rel.file_name().and_then(|s| s.to_str()) == Some(wanted) {
            installed = Some(out_path);
        }
    }
    installed.ok_or_else(|| {
        format!(
            "tarball {} did not contain a top-level `kiln` binary",
            tarball.display()
        )
    })
}

#[cfg(unix)]
fn make_executable(path: &Path) -> Result<(), String> {
    use std::os::unix::fs::PermissionsExt;
    let mut perms = std::fs::metadata(path)
        .map_err(|e| format!("stat {}: {}", path.display(), e))?
        .permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(path, perms)
        .map_err(|e| format!("chmod {}: {}", path.display(), e))
}
#[cfg(not(unix))]
fn make_executable(_path: &Path) -> Result<(), String> {
    Ok(())
}

/// Remove the `com.apple.quarantine` xattr on macOS so the binary runs
/// without a Gatekeeper prompt. `reqwest` doesn't set this xattr, but if
/// the user manually drops a DMG-exported binary into the install dir we
/// still want to be safe.
#[cfg(target_os = "macos")]
fn clear_quarantine(path: &Path) {
    // Best effort — failure is expected when the xattr isn't set.
    let _ = std::process::Command::new("xattr")
        .arg("-d")
        .arg("com.apple.quarantine")
        .arg(path)
        .status();
}
#[cfg(not(target_os = "macos"))]
fn clear_quarantine(_path: &Path) {}

/// Parse the second whitespace-separated token from `kiln --version`
/// output. Returns `None` when the output is empty or has only one token.
/// `kiln --version` prints either `kiln <semver>` or
/// `kiln <semver> (<git-sha>)`; we take the `<semver>` token.
fn parse_kiln_version_output(stdout: &str) -> Option<String> {
    stdout.split_whitespace().nth(1).map(|s| s.to_string())
}

/// Run `bin --version` and return the reported semver, or `None` if the
/// command fails, exits non-zero, or produces unparseable output.
pub async fn current_kiln_version(bin: &Path) -> Option<String> {
    let out = tokio::process::Command::new(bin)
        .arg("--version")
        .output()
        .await
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_kiln_version_output(&stdout)
}

/// Fetch the newest release version for the current target. Returns
/// `None` when the platform has no prebuilt asset or when the GitHub
/// releases listing fails. Callers should separately check
/// [`supports_auto_install`] to distinguish those two cases.
pub async fn discover_latest_version(client: &reqwest::Client) -> Option<String> {
    let target = current_target()?;
    discover_asset(client, target).await.ok().map(|p| p.version)
}

/// Fetch the newest release's version + notes body for the current
/// target. Feeds [`is_supported_sm_for_release`] so callers can apply the
/// per-release supported-SM list. Returns `None` under the same
/// conditions as [`discover_latest_version`].
pub async fn discover_latest_version_and_body(
    client: &reqwest::Client,
) -> Option<(String, Option<String>)> {
    let target = current_target()?;
    discover_asset(client, target)
        .await
        .ok()
        .map(|p| (p.version, p.release_body))
}

/// Decide whether `latest` is newer than `current` using semver.
///
/// Per design doc (`desktop/docs/binary-update.md`): fall back to string
/// inequality when either side fails to parse as semver, never suggest a
/// downgrade, and treat "unknown current version" as "offer latest".
pub fn is_update_available(current: Option<&str>, latest: &str) -> bool {
    let Some(current) = current else { return true; };
    match (
        semver::Version::parse(current),
        semver::Version::parse(latest),
    ) {
        (Ok(c), Ok(l)) => c < l,
        // One side isn't valid semver — best we can do is say "offer when
        // strings differ" so a genuine mismatch surfaces without claiming
        // a downgrade-as-update when parsing succeeds on both sides.
        _ => current != latest,
    }
}

// ---------- GPU arch compat gate (slice 6) ----------
//
// Before offering a kiln update on Linux/Windows, the desktop app detects
// the local GPU's SM arch via `nvidia-smi --query-gpu=compute_cap` and
// refuses to suggest a binary compiled for an unsupported arch. macOS
// has no SM arch concept (single release target), so the check
// short-circuits to `Supported`. Detection failure (no nvidia-smi,
// non-zero exit, timeout, unparseable output) is reported as `Unknown`
// and is treated as "don't block" by callers — preserving existing
// behavior on systems without nvidia-smi.
//
// See `desktop/docs/binary-update.md` (CUDA / GPU compat, lines 260-280).

/// SM arches that the current kiln release series supports. Hardcoded
/// here until a later slice parses `supported_sm` out of the GitHub
/// release notes (design doc lines 266-268).
pub const SUPPORTED_SM_ARCHS: &[u32] = &[80, 86, 89, 90];

/// Outcome of the local GPU arch compatibility check.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuCompat {
    /// The detected SM arch is in [`SUPPORTED_SM_ARCHS`].
    Supported(u32),
    /// The detected SM arch is NOT in [`SUPPORTED_SM_ARCHS`]. Callers
    /// should refuse to offer an update and surface the SM number to the
    /// user.
    Unsupported(u32),
    /// Detection failed (no nvidia-smi, non-zero exit, timeout, or
    /// unparseable output). Callers should NOT block the update offer —
    /// this preserves existing behavior on systems without nvidia-smi,
    /// which includes macOS and any non-CUDA host.
    Unknown,
}

/// Whether `arch` is in [`SUPPORTED_SM_ARCHS`].
pub fn is_supported_sm(arch: u32) -> bool {
    SUPPORTED_SM_ARCHS.contains(&arch)
}

/// Parse `supported_sm: [80, 86, 89, 90]` (or `supported_sm: 80, 86, 89, 90`)
/// from release-notes body. Returns `None` when the line is absent or
/// unparseable so callers fall back to [`SUPPORTED_SM_ARCHS`].
///
/// The key match is strict (lowercase `supported_sm`). Brackets are
/// optional; whitespace inside the list is tolerated. Any unparseable
/// element or an empty list returns `None` so the caller falls back to
/// the compiled-in default rather than silently locking out every arch.
///
/// Note on line terminators: `gh release create --notes "..."` passes
/// the string verbatim (no escape-sequence processing), so release
/// bodies emitted by the CI workflow may contain literal `\n` (two
/// characters) between fields rather than real newlines. We normalize
/// both forms before scanning for the key so either style works.
pub fn parse_supported_sm(body: &str) -> Option<Vec<u32>> {
    let normalized = body.replace("\\n", "\n");
    for line in normalized.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("supported_sm:") else {
            continue;
        };
        let rest = rest.trim();
        let inner = rest
            .strip_prefix('[')
            .and_then(|s| s.strip_suffix(']'))
            .unwrap_or(rest);
        let parts: Vec<&str> = inner
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();
        if parts.is_empty() {
            return None;
        }
        let mut out = Vec::with_capacity(parts.len());
        for p in parts {
            let n: u32 = p.parse().ok()?;
            out.push(n);
        }
        if out.is_empty() {
            return None;
        }
        return Some(out);
    }
    None
}

/// Pick the supported-SM list for a release: prefer the parsed list from
/// its notes body, fall back to [`SUPPORTED_SM_ARCHS`] when the body is
/// missing, malformed, or empty.
pub fn supported_sm_for_release(body: Option<&str>) -> Vec<u32> {
    body.and_then(parse_supported_sm)
        .filter(|v| !v.is_empty())
        .unwrap_or_else(|| SUPPORTED_SM_ARCHS.to_vec())
}

/// Whether `arch` is supported by the release whose notes body is
/// `body`. Per-release list takes precedence over [`SUPPORTED_SM_ARCHS`]
/// so a newer release can widen support (e.g. add Blackwell SM 120) by
/// publishing `supported_sm: [80, 86, 89, 90, 120]` in its notes without
/// waiting for a desktop update.
pub fn is_supported_sm_for_release(arch: u32, body: Option<&str>) -> bool {
    supported_sm_for_release(body).contains(&arch)
}

/// Parse the first non-empty line of `nvidia-smi --query-gpu=compute_cap
/// --format=csv,noheader` output into an integer SM arch number (e.g.
/// `"8.6"` -> `Some(86)`, `"12.0"` -> `Some(120)`). Ignores trailing
/// whitespace and subsequent GPU lines (take the first GPU). Returns
/// `None` when the input is empty, unparseable, or has an unexpected
/// shape.
pub fn parse_compute_cap(output: &str) -> Option<u32> {
    let first = output.lines().find(|line| !line.trim().is_empty())?;
    let trimmed = first.trim();
    let (major_s, minor_s) = trimmed.split_once('.')?;
    let major: u32 = major_s.trim().parse().ok()?;
    let minor: u32 = minor_s.trim().parse().ok()?;
    // `compute_cap` minor digit is always a single decimal; multiplying
    // the major by 10 matches the SM arch numbering used elsewhere in
    // this file (sm_86, sm_120, etc.).
    Some(major * 10 + minor)
}

/// Invoke `nvidia-smi --query-gpu=compute_cap --format=csv,noheader` and
/// parse the first GPU's compute capability. Returns `None` when the
/// command fails to spawn, exits non-zero, times out (5s), or produces
/// unparseable output.
#[cfg(not(target_os = "macos"))]
pub async fn detect_gpu_compute_cap() -> Option<u32> {
    let fut = tokio::process::Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_compute_cap(&stdout)
}

/// macOS has no SM arch concept — detection is never attempted.
#[cfg(target_os = "macos")]
pub async fn detect_gpu_compute_cap() -> Option<u32> {
    None
}

/// GPU arch compatibility check for the current host. On macOS this
/// always returns [`GpuCompat::Supported`] with a sentinel arch of `0`
/// (macOS has a single release target — SM arch doesn't apply). On
/// Linux/Windows this shells out to `nvidia-smi`; detection failure is
/// reported as [`GpuCompat::Unknown`] so callers don't block the update
/// on systems without nvidia-smi.
#[cfg(target_os = "macos")]
pub async fn gpu_compat() -> GpuCompat {
    GpuCompat::Supported(0)
}

#[cfg(not(target_os = "macos"))]
pub async fn gpu_compat() -> GpuCompat {
    match detect_gpu_compute_cap().await {
        Some(arch) if is_supported_sm(arch) => GpuCompat::Supported(arch),
        Some(arch) => GpuCompat::Unsupported(arch),
        None => GpuCompat::Unknown,
    }
}

/// Parse `min_cuda: 12.4` (or `min_cuda: 12`) from release-notes body.
/// Returns `Some((major, minor))` when the key is present and parseable,
/// `None` when the key is absent, empty, or unparseable. A bare major
/// (`min_cuda: 12`) is normalized to `(12, 0)`.
///
/// Mirrors [`parse_supported_sm`] for line-terminator handling: `gh
/// release create --notes "..."` passes the string verbatim without
/// escape-sequence processing, so bodies emitted by CI may contain
/// literal `\n` (two characters) between fields rather than real
/// newlines. Both forms are normalized before key scanning.
pub fn parse_min_cuda(body: &str) -> Option<(u32, u32)> {
    let normalized = body.replace("\\n", "\n");
    for line in normalized.lines() {
        let trimmed = line.trim();
        let Some(rest) = trimmed.strip_prefix("min_cuda:") else {
            continue;
        };
        let token = rest.trim();
        if token.is_empty() {
            return None;
        }
        return match token.split_once('.') {
            Some((major_s, minor_s)) => {
                let major: u32 = major_s.trim().parse().ok()?;
                let minor: u32 = minor_s.trim().parse().ok()?;
                Some((major, minor))
            }
            None => {
                let major: u32 = token.parse().ok()?;
                Some((major, 0))
            }
        };
    }
    None
}

/// Pick the minimum-CUDA requirement for a release from its notes body.
/// Passes through to [`parse_min_cuda`]; returns `None` when no
/// `min_cuda:` line is present so callers treat the release as
/// CUDA-agnostic (no gate).
pub fn min_cuda_for_release(body: Option<&str>) -> Option<(u32, u32)> {
    body.and_then(parse_min_cuda)
}

/// Whether the local CUDA driver version `local` satisfies the release's
/// `min_cuda:` requirement parsed from `body`. Returns `true` when:
/// - `body` has no `min_cuda:` line (CUDA-agnostic release, no gate), or
/// - `local` is `None` (detection failed — preserve the no-block policy
///   applied to [`GpuCompat::Unknown`] so systems without `nvidia-smi`
///   are not locked out), or
/// - `local >= required` by lexicographic `(major, minor)` comparison.
///
/// Returns `false` only when the release advertises a `min_cuda:` AND
/// the local driver was detected AND the local version is strictly
/// older.
pub fn is_cuda_compatible_for_release(
    local: Option<(u32, u32)>,
    body: Option<&str>,
) -> bool {
    let Some(required) = min_cuda_for_release(body) else {
        return true;
    };
    let Some(local) = local else {
        return true;
    };
    local >= required
}

/// Parse `nvidia-smi` stdout (plain invocation, no args) for the
/// "CUDA Version: X.Y" token in the header line. Returns
/// `Some((major, minor))` on success, `None` when the line is absent or
/// unparseable. Tolerates arbitrary whitespace around the token.
///
/// Typical header shape:
/// `| NVIDIA-SMI 535.86.10   Driver Version: 535.86.10   CUDA Version: 12.2     |`
pub fn parse_cuda_driver_version(stdout: &str) -> Option<(u32, u32)> {
    for line in stdout.lines() {
        let Some(idx) = line.find("CUDA Version:") else {
            continue;
        };
        let after = &line[idx + "CUDA Version:".len()..];
        // The next whitespace-delimited token is the version. Strip any
        // trailing `|` or other table-border punctuation before parsing.
        let token = after.split_whitespace().next()?;
        let cleaned: String = token
            .chars()
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if cleaned.is_empty() {
            return None;
        }
        return match cleaned.split_once('.') {
            Some((major_s, minor_s)) => {
                let major: u32 = major_s.parse().ok()?;
                let minor: u32 = minor_s.parse().ok()?;
                Some((major, minor))
            }
            None => {
                let major: u32 = cleaned.parse().ok()?;
                Some((major, 0))
            }
        };
    }
    None
}

/// Invoke plain `nvidia-smi` (no args) and parse the CUDA driver version
/// from its header. Returns `None` when the command fails to spawn,
/// exits non-zero, times out (5s), or produces unparseable output. On
/// macOS this always returns `None` — no nvidia-smi, no CUDA.
#[cfg(not(target_os = "macos"))]
pub async fn detect_cuda_driver_version() -> Option<(u32, u32)> {
    let fut = tokio::process::Command::new("nvidia-smi").output();
    let out = tokio::time::timeout(std::time::Duration::from_secs(5), fut)
        .await
        .ok()?
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_cuda_driver_version(&stdout)
}

#[cfg(target_os = "macos")]
pub async fn detect_cuda_driver_version() -> Option<(u32, u32)> {
    None
}

/// Download the release tarball for `version` to the staging path under
/// `<app_data_dir>/kiln-updates/`. Does NOT extract, verify sha256, or
/// touch the running kiln binary — that is slice 3 of
/// `desktop/docs/binary-update.md`.
///
/// Emits [`UPDATE_DOWNLOAD_PROGRESS_EVENT`] with a
/// [`UpdateDownloadProgress`] payload during streaming. Returns the full
/// staging file path on success.
pub async fn download_release_binary(
    app: &AppHandle,
    version: &str,
    cancel: Arc<AtomicBool>,
) -> Result<(PathBuf, u64), String> {
    let target = current_target().ok_or_else(|| {
        "No prebuilt kiln release exists for this platform (Windows + Linux \
         release assets land in a separate build task)."
            .to_string()
    })?;

    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir unavailable: {}", e))?;
    let dest = staging_tarball_path(&base, version, target);
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)
            .await
            .map_err(|e| format!("mkdir {}: {}", parent.display(), e))?;
    }

    let url = release_download_url(version, target);
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("build reqwest client: {}", e))?;

    let mut resp = client
        .get(&url)
        .send()
        .await
        .map_err(|e| format!("download request failed: {}", e))?;
    if !resp.status().is_success() {
        return Err(format!(
            "download from {} returned HTTP {}",
            url,
            resp.status()
        ));
    }
    let total_bytes = resp.content_length();
    if let Some(t) = total_bytes {
        if t > MAX_ASSET_BYTES {
            return Err(format!(
                "asset claims {} bytes, over {}-byte safety limit",
                t, MAX_ASSET_BYTES
            ));
        }
    }

    let mut file = fs::File::create(&dest)
        .await
        .map_err(|e| format!("create {}: {}", dest.display(), e))?;
    let mut downloaded: u64 = 0;
    let mut last_reported: u64 = 0;

    let _ = app.emit(
        UPDATE_DOWNLOAD_PROGRESS_EVENT,
        UpdateDownloadProgress {
            version: version.to_string(),
            downloaded_bytes: 0,
            total_bytes,
        },
    );

    loop {
        if cancel.load(Ordering::SeqCst) {
            // Best-effort cleanup of partial file on cancel.
            drop(file);
            let _ = fs::remove_file(&dest).await;
            return Err("download cancelled".to_string());
        }
        let chunk = resp
            .chunk()
            .await
            .map_err(|e| format!("chunk read failed: {}", e))?;
        let Some(bytes) = chunk else { break };
        downloaded = downloaded.saturating_add(bytes.len() as u64);
        if downloaded > MAX_ASSET_BYTES {
            return Err(format!(
                "aborted — received {} bytes past {}-byte safety limit",
                downloaded, MAX_ASSET_BYTES
            ));
        }
        file.write_all(&bytes)
            .await
            .map_err(|e| format!("write chunk: {}", e))?;
        // Keep IPC cost down — emit at 64KB granularity.
        if downloaded.saturating_sub(last_reported) >= 64 * 1024 {
            last_reported = downloaded;
            let _ = app.emit(
                UPDATE_DOWNLOAD_PROGRESS_EVENT,
                UpdateDownloadProgress {
                    version: version.to_string(),
                    downloaded_bytes: downloaded,
                    total_bytes,
                },
            );
        }
    }
    file.flush()
        .await
        .map_err(|e| format!("flush: {}", e))?;
    let _ = app.emit(
        UPDATE_DOWNLOAD_PROGRESS_EVENT,
        UpdateDownloadProgress {
            version: version.to_string(),
            downloaded_bytes: downloaded,
            total_bytes,
        },
    );
    Ok((dest, downloaded))
}

/// Fetch the sha256 digest for the tarball published alongside `version`
/// on the current target, by re-discovering the release asset pair and
/// downloading the companion `.sha256` file. Returned as a 64-char
/// lowercase hex digest.
///
/// Exposed as its own helper (vs. baking it into
/// [`extract_and_verify_update`]) so the UI can fetch the digest once
/// when the user clicks "Extract & Verify" and hand it to the
/// orchestrator — keeping slice 3a self-contained without forcing slice
/// 2's download step to persist the digest to disk.
pub async fn fetch_release_sha256(version: &str) -> Result<String, String> {
    let target = current_target().ok_or_else(|| {
        "No prebuilt kiln release exists for this platform.".to_string()
    })?;
    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("build reqwest client: {}", e))?;
    let pair = discover_asset(&client, target).await?;
    if pair.version != version {
        return Err(format!(
            "latest published release is {}, not {} — asset discovery drift",
            pair.version, version
        ));
    }
    let sha_asset = pair.sha256.ok_or_else(|| {
        format!(
            "release {} has no {}.sha256 asset — cannot verify",
            pair.version, pair.tarball.name
        )
    })?;
    fetch_expected_sha256(&client, &sha_asset).await
}

/// Slice 3a orchestrator: verify the staged tarball written by slice 2 at
/// [`staging_tarball_path`] against `expected_sha256`, then extract the
/// `kiln` binary to `<app_data_dir>/bin/kiln.new`. Does NOT swap
/// `bin/kiln`, stop the supervisor, or restart — that is slice 3b.
///
/// Sha256 verification runs BEFORE extraction so a corrupted download
/// never produces a garbage `kiln.new` on disk. Both the verify and the
/// extract run inside [`tokio::task::spawn_blocking`] so the reads don't
/// block the runtime. Emits [`EXTRACT_UPDATE_PROGRESS_EVENT`] at start
/// and finish with [`ExtractUpdateProgress`]; terminal `done`/`failed`
/// events are emitted by the Tauri command layer so cancellation state
/// lives with the cancel flag there.
pub async fn extract_and_verify_update(
    app: &AppHandle,
    version: &str,
    expected_sha256: &str,
    cancel: Arc<AtomicBool>,
) -> Result<(PathBuf, u64), String> {
    let target = current_target().ok_or_else(|| {
        "No prebuilt kiln release exists for this platform.".to_string()
    })?;
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir unavailable: {}", e))?;
    let tarball = staging_tarball_path(&base, version, target);
    if !tarball.is_file() {
        return Err(format!(
            "no staged update tarball at {} — run the download step first",
            tarball.display()
        ));
    }
    let new_path = extracted_new_binary_path(&base);

    let _ = app.emit(
        EXTRACT_UPDATE_PROGRESS_EVENT,
        ExtractUpdateProgress {
            version: version.to_string(),
            extracted_bytes: 0,
            total_bytes: None,
        },
    );

    if cancel.load(Ordering::SeqCst) {
        return Err("extract cancelled".to_string());
    }

    let tarball_for_verify = tarball.clone();
    let expected = expected_sha256.to_string();
    tokio::task::spawn_blocking(move || verify_sha256(&tarball_for_verify, &expected))
        .await
        .map_err(|e| format!("verify task join: {}", e))??;

    if cancel.load(Ordering::SeqCst) {
        return Err("extract cancelled".to_string());
    }

    let tarball_for_extract = tarball.clone();
    let new_path_for_extract = new_path.clone();
    let bytes = tokio::task::spawn_blocking(move || {
        extract_tarball_to(&tarball_for_extract, &new_path_for_extract)
    })
    .await
    .map_err(|e| format!("extract task join: {}", e))??;

    // Unix: chmod 0755 so slice 3b's atomic rename hands the supervisor a
    // runnable file without a follow-up chmod. No-op on Windows.
    make_executable(&new_path)?;

    let _ = app.emit(
        EXTRACT_UPDATE_PROGRESS_EVENT,
        ExtractUpdateProgress {
            version: version.to_string(),
            extracted_bytes: bytes,
            total_bytes: Some(bytes),
        },
    );

    Ok((new_path, bytes))
}

/// Full install pipeline: discover latest asset, download, verify sha256,
/// extract, chmod, clear quarantine. Returns the installed binary path.
pub async fn install_latest_server(
    app: AppHandle,
    cancel: Arc<AtomicBool>,
) -> Result<(PathBuf, String), String> {
    let target = current_target().ok_or_else(|| {
        "No prebuilt kiln release exists for this platform. Build from source (see QUICKSTART.md).".to_string()
    })?;

    emit_progress(&app, InstallProgress::FetchingManifest);

    let client = reqwest::Client::builder()
        .user_agent(USER_AGENT)
        .connect_timeout(std::time::Duration::from_secs(15))
        .build()
        .map_err(|e| format!("build reqwest client: {}", e))?;

    let pair = discover_asset(&client, target).await?;
    if cancel.load(Ordering::SeqCst) {
        return Err("download cancelled".to_string());
    }

    let dir = install_dir(&app)?;
    fs::create_dir_all(&dir)
        .await
        .map_err(|e| format!("mkdir {}: {}", dir.display(), e))?;
    let tarball_path = dir.join(&pair.tarball.name);

    let actual_sha = download_with_progress(
        &app,
        &client,
        &pair.tarball,
        &tarball_path,
        &cancel,
    )
    .await?;

    emit_progress(&app, InstallProgress::Verifying);

    if let Some(sha_asset) = &pair.sha256 {
        let expected = fetch_expected_sha256(&client, sha_asset).await?;
        if expected != actual_sha {
            let _ = fs::remove_file(&tarball_path).await;
            return Err(format!(
                "sha256 mismatch: expected {}, got {}",
                expected, actual_sha
            ));
        }
    } else {
        // No companion .sha256 asset — refuse to install an unverified
        // binary rather than silently trusting the download.
        let _ = fs::remove_file(&tarball_path).await;
        return Err(format!(
            "release {} has no {}.sha256 asset — refusing to install unverified binary",
            pair.version, pair.tarball.name
        ));
    }

    emit_progress(&app, InstallProgress::Extracting);
    let tarball_path_clone = tarball_path.clone();
    let dir_clone = dir.clone();
    let installed = tokio::task::spawn_blocking(move || {
        extract_tarball(&tarball_path_clone, &dir_clone)
    })
    .await
    .map_err(|e| format!("extract task join: {}", e))??;

    // Remove the staging tarball; we keep only the extracted binary.
    let _ = fs::remove_file(&tarball_path).await;

    emit_progress(&app, InstallProgress::Finalizing);
    make_executable(&installed)?;
    clear_quarantine(&installed);

    Ok((installed, pair.version))
}

/// Pure filesystem half of the slice-3b atomic swap. Promotes
/// `<app_data_dir>/bin/kiln.new` into `<app_data_dir>/bin/kiln`
/// (or `kiln.exe` on Windows), moving any currently-installed binary
/// aside to `bin/kiln.bak`. Caller is responsible for stopping the
/// supervisor beforehand — see [`atomic_swap_new_binary`] for the
/// coupled version. Health-check gating and rollback on failed health
/// are slice 3c (`desktop/docs/binary-update.md`) and intentionally
/// not handled here.
///
/// On any IO failure mid-swap the helper makes a best-effort attempt
/// to restore the prior `bin/kiln` from the `.bak` it just created so
/// callers don't hand a half-installed state to the supervisor.
pub fn swap_new_binary_into_place(app_data_dir: &Path) -> Result<(), String> {
    let new_path = extracted_new_binary_path(app_data_dir);
    let installed = installed_binary_path(app_data_dir);
    let backup = backup_binary_path(app_data_dir);

    if !new_path.is_file() {
        return Err(format!(
            "no staged kiln.new at {} — run the extract step first",
            new_path.display()
        ));
    }

    // Move any pre-existing installed binary aside. Remove a stale
    // `.bak` first so the rename works the same on Windows (which
    // refuses rename-over-existing via std::fs::rename) as on unix.
    let had_prior_install = installed.exists();
    if had_prior_install {
        if backup.exists() {
            std::fs::remove_file(&backup).map_err(|e| {
                format!(
                    "remove stale backup {}: {}",
                    backup.display(),
                    e
                )
            })?;
        }
        std::fs::rename(&installed, &backup).map_err(|e| {
            format!(
                "rename {} -> {}: {}",
                installed.display(),
                backup.display(),
                e
            )
        })?;
    }

    // Promote the staged binary into the installed slot.
    if let Err(e) = std::fs::rename(&new_path, &installed) {
        // Best-effort rollback so the prior binary stays usable.
        if had_prior_install {
            let _ = std::fs::rename(&backup, &installed);
        }
        return Err(format!(
            "rename {} -> {}: {}",
            new_path.display(),
            installed.display(),
            e
        ));
    }

    // chmod +x on unix so the supervisor can `execve` without a
    // follow-up chmod. No-op on Windows.
    if let Err(e) = make_executable(&installed) {
        // Best-effort rollback: drop the just-promoted binary and
        // restore the .bak over it.
        if had_prior_install {
            let _ = std::fs::remove_file(&installed);
            let _ = std::fs::rename(&backup, &installed);
        }
        return Err(e);
    }

    Ok(())
}

/// Stop the supervisor (awaiting child shutdown) and promote
/// `bin/kiln.new` into `bin/kiln` via [`swap_new_binary_into_place`].
/// Does **not** restart the supervisor — the caller is responsible for
/// calling `supervisor.start()` afterwards so slice 3c's health-gate
/// layer can sit between the swap and the "update done" signal.
pub async fn atomic_swap_new_binary(
    supervisor: &crate::supervisor::Supervisor,
    app_data_dir: &Path,
) -> Result<(), String> {
    supervisor.stop().await?;

    let dir = app_data_dir.to_path_buf();
    tokio::task::spawn_blocking(move || swap_new_binary_into_place(&dir))
        .await
        .map_err(|e| format!("swap task join: {}", e))??;

    Ok(())
}

/// Poll `url` until it returns a 2xx response or `timeout` elapses.
/// Polls every 500ms. Returns `true` on the first 2xx, `false` on
/// timeout. Network errors and non-2xx responses are treated as "not
/// ready yet" — only the final timeout produces `false`.
///
/// Pure HTTP polling — does not touch the supervisor — so the slice-3c
/// updater can hand it the `/v1/health` URL built from
/// `Supervisor::config_snapshot()` without coupling the helper to that
/// type. See `desktop/docs/binary-update.md` "Health check gate".
pub async fn wait_for_health(
    client: &reqwest::Client,
    url: &str,
    timeout: std::time::Duration,
) -> bool {
    let deadline = std::time::Instant::now() + timeout;
    loop {
        if let Ok(resp) = client.get(url).send().await {
            if resp.status().is_success() {
                return true;
            }
        }
        if std::time::Instant::now() >= deadline {
            return false;
        }
        tokio::time::sleep(std::time::Duration::from_millis(500)).await;
    }
}

/// Roll back the slice-3b atomic swap by renaming `bin/kiln.bak` over
/// `bin/kiln`. Used by slice 3c when the post-swap health gate never
/// goes green: the freshly-promoted binary is broken, the prior binary
/// is still safe under `.bak`, and the supervisor needs the old path
/// back before it can restart.
///
/// Returns `Err` if no `.bak` exists — the caller is supposed to invoke
/// this only after a swap that produced one. On Windows
/// `std::fs::rename` refuses to overwrite an existing destination, so
/// the broken `bin/kiln` is removed first to make the rename succeed
/// the same on every platform (mirrors [`swap_new_binary_into_place`]).
pub fn rollback_to_bak(app_data_dir: &Path) -> Result<(), String> {
    let installed = installed_binary_path(app_data_dir);
    let backup = backup_binary_path(app_data_dir);

    if !backup.exists() {
        return Err(format!(
            "no backup at {} — nothing to roll back",
            backup.display()
        ));
    }

    if installed.exists() {
        std::fs::remove_file(&installed).map_err(|e| {
            format!(
                "remove broken {} before rollback: {}",
                installed.display(),
                e
            )
        })?;
    }

    std::fs::rename(&backup, &installed).map_err(|e| {
        format!(
            "rename {} -> {}: {}",
            backup.display(),
            installed.display(),
            e
        )
    })?;

    Ok(())
}

/// Delete `bin/kiln.bak` if present. No-op when absent. Used by slice
/// 3c on the health-green path: once the new binary is verified
/// running, the backup is no longer load-bearing and only one `.bak`
/// is ever kept (see `desktop/docs/binary-update.md` "Rollback").
pub fn cleanup_bak(app_data_dir: &Path) -> Result<(), String> {
    let backup = backup_binary_path(app_data_dir);
    if !backup.exists() {
        return Ok(());
    }
    std::fs::remove_file(&backup)
        .map_err(|e| format!("remove {}: {}", backup.display(), e))
}

/// Updater slice 3c Tauri command: stop the supervisor, promote the
/// staged `bin/kiln.new` into `bin/kiln`, restart the supervisor, then
/// poll `/v1/health` for up to 30s. On a green health response the
/// retained `bin/kiln.bak` is deleted; on timeout the supervisor is
/// stopped, `bin/kiln.bak` is renamed back over `bin/kiln`, and the
/// supervisor is restarted on the prior binary. Emits
/// [`INSTALL_STAGED_UPDATE_DONE_EVENT`] with [`InstallStagedUpdateDone`]
/// on success and [`INSTALL_STAGED_UPDATE_FAILED_EVENT`] with
/// [`InstallStagedUpdateFailed`] on failure.
#[tauri::command]
pub async fn install_staged_update(
    app: tauri::AppHandle,
    sup: tauri::State<'_, std::sync::Arc<crate::supervisor::Supervisor>>,
) -> Result<(), String> {
    let base = app
        .path()
        .app_data_dir()
        .map_err(|e| format!("app_data_dir unavailable: {}", e))?;
    let supervisor = sup.inner().clone();
    let app_for_task = app.clone();

    tauri::async_runtime::spawn(async move {
        // Single client for the whole post-swap health-poll loop so
        // keep-alive amortizes connection setup across the ~60 polls.
        let client = match reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .connect_timeout(std::time::Duration::from_secs(2))
            .build()
        {
            Ok(c) => c,
            Err(err) => {
                let _ = app_for_task.emit(
                    INSTALL_STAGED_UPDATE_FAILED_EVENT,
                    InstallStagedUpdateFailed {
                        error: format!("build reqwest client: {}", err),
                    },
                );
                return;
            }
        };

        let swap = atomic_swap_new_binary(&supervisor, &base).await;
        if let Err(err) = swap {
            // Swap failed mid-way; try to restart the prior
            // supervisor config so the user isn't left with a
            // dead server just because extract/rename tripped.
            // The error from the swap is what the UI should see.
            let _ = supervisor.start().await;
            let _ = app_for_task.emit(
                INSTALL_STAGED_UPDATE_FAILED_EVENT,
                InstallStagedUpdateFailed { error: err },
            );
            return;
        }

        if let Err(err) = supervisor.start().await {
            let _ = app_for_task.emit(
                INSTALL_STAGED_UPDATE_FAILED_EVENT,
                InstallStagedUpdateFailed {
                    error: format!(
                        "supervisor start after swap failed: {}",
                        err
                    ),
                },
            );
            return;
        }

        let cfg = supervisor.config_snapshot().await;
        let url = format!("http://{}:{}/v1/health", cfg.host, cfg.port);
        let healthy =
            wait_for_health(&client, &url, std::time::Duration::from_secs(30))
                .await;

        if healthy {
            // Best-effort cleanup; a leftover .bak isn't load-bearing
            // and the next swap will overwrite it (see
            // `swap_new_binary_into_place`).
            if let Err(e) = cleanup_bak(&base) {
                eprintln!("[updater] cleanup_bak: {}", e);
            }
            let installed = installed_binary_path(&base);
            let _ = app_for_task.emit(
                INSTALL_STAGED_UPDATE_DONE_EVENT,
                InstallStagedUpdateDone {
                    path: installed.display().to_string(),
                },
            );
            return;
        }

        // Health gate timed out — roll back to the prior binary so the
        // user isn't left on a broken kiln.
        let _ = supervisor.stop().await;
        let rollback_err = match rollback_to_bak(&base) {
            Ok(()) => None,
            Err(e) => Some(e),
        };
        let _ = supervisor.start().await;

        let error = match rollback_err {
            None => "kiln server did not become healthy within 30s; \
                     rolled back to previous binary"
                .to_string(),
            Some(re) => format!(
                "kiln server did not become healthy within 30s; \
                 rollback also failed: {}",
                re
            ),
        };
        let _ = app_for_task.emit(
            INSTALL_STAGED_UPDATE_FAILED_EVENT,
            InstallStagedUpdateFailed { error },
        );
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        backup_binary_path, cleanup_bak, current_target, extract_tarball_to,
        extracted_new_binary_path, installed_binary_path,
        is_cuda_compatible_for_release, is_supported_sm,
        is_supported_sm_for_release, is_update_available, min_cuda_for_release,
        parse_compute_cap, parse_cuda_driver_version, parse_kiln_version_output,
        parse_min_cuda, parse_supported_sm, release_archive_ext,
        release_asset_name, release_download_url, rollback_to_bak,
        staging_tarball_path, supports_auto_install, swap_new_binary_into_place,
        verify_sha256, BACKUP_BINARY_NAME, NEW_BINARY_NAME, SUPPORTED_SM_ARCHS,
        UPDATE_STAGING_DIR,
    };
    use std::io::Write;
    use std::path::PathBuf;

    #[test]
    fn archive_ext_windows_is_zip() {
        assert_eq!(release_archive_ext("x86_64-pc-windows-msvc-cuda124"), "zip");
        assert_eq!(release_archive_ext("x86_64-pc-windows-gnu"), "zip");
    }

    #[test]
    fn archive_ext_unix_is_tar_gz() {
        assert_eq!(
            release_archive_ext("aarch64-apple-darwin-metal"),
            "tar.gz"
        );
        assert_eq!(
            release_archive_ext("x86_64-unknown-linux-gnu-cuda124"),
            "tar.gz"
        );
        assert_eq!(release_archive_ext("x86_64-apple-darwin"), "tar.gz");
    }

    #[test]
    fn asset_name_matches_release_workflow_macos() {
        assert_eq!(
            release_asset_name("0.2.0", "aarch64-apple-darwin-metal"),
            "kiln-0.2.0-aarch64-apple-darwin-metal.tar.gz"
        );
    }

    #[test]
    fn asset_name_matches_release_workflow_linux() {
        assert_eq!(
            release_asset_name("0.2.0", "x86_64-unknown-linux-gnu-cuda124"),
            "kiln-0.2.0-x86_64-unknown-linux-gnu-cuda124.tar.gz"
        );
    }

    #[test]
    fn asset_name_windows_uses_zip() {
        assert_eq!(
            release_asset_name("0.2.0", "x86_64-pc-windows-msvc-cuda124"),
            "kiln-0.2.0-x86_64-pc-windows-msvc-cuda124.zip"
        );
    }

    #[test]
    fn download_url_uses_kiln_v_tag_and_matching_asset() {
        assert_eq!(
            release_download_url("0.2.0", "aarch64-apple-darwin-metal"),
            "https://github.com/ericflo/kiln/releases/download/kiln-v0.2.0/\
             kiln-0.2.0-aarch64-apple-darwin-metal.tar.gz"
                .replace(' ', "")
        );
        assert_eq!(
            release_download_url("1.2.3", "x86_64-pc-windows-msvc-cuda124"),
            "https://github.com/ericflo/kiln/releases/download/kiln-v1.2.3/\
             kiln-1.2.3-x86_64-pc-windows-msvc-cuda124.zip"
                .replace(' ', "")
        );
        assert_eq!(
            release_download_url("1.0.0", "x86_64-unknown-linux-gnu-cuda124"),
            "https://github.com/ericflo/kiln/releases/download/kiln-v1.0.0/\
             kiln-1.0.0-x86_64-unknown-linux-gnu-cuda124.tar.gz"
                .replace(' ', "")
        );
    }

    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    #[test]
    fn current_target_on_macos_aarch64_is_metal() {
        assert_eq!(current_target(), Some("aarch64-apple-darwin-metal"));
    }

    #[cfg(all(target_os = "linux", target_arch = "x86_64"))]
    #[test]
    fn current_target_on_linux_x86_64_is_cuda124() {
        assert_eq!(current_target(), Some("x86_64-unknown-linux-gnu-cuda124"));
    }

    #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
    #[test]
    fn current_target_on_windows_x86_64_is_cuda124() {
        assert_eq!(current_target(), Some("x86_64-pc-windows-msvc-cuda124"));
    }

    #[cfg(any(
        all(target_os = "macos", target_arch = "aarch64"),
        all(target_os = "linux", target_arch = "x86_64"),
        all(target_os = "windows", target_arch = "x86_64"),
    ))]
    #[test]
    fn supports_auto_install_on_supported_platforms() {
        assert!(supports_auto_install());
    }

    #[test]
    fn staging_path_nests_under_update_staging_dir() {
        let base = PathBuf::from("/tmp/fake-app-data");
        let p = staging_tarball_path(&base, "0.2.0", "aarch64-apple-darwin-metal");
        assert!(
            p.starts_with(base.join(UPDATE_STAGING_DIR)),
            "staging path {} should be under {}",
            p.display(),
            base.join(UPDATE_STAGING_DIR).display()
        );
        assert_eq!(p.file_name().unwrap(), "kiln-v0.2.0.tar.gz");
    }

    #[test]
    fn staging_path_windows_uses_zip_extension() {
        let base = PathBuf::from("C:/fake/AppData");
        let p = staging_tarball_path(&base, "0.2.0", "x86_64-pc-windows-msvc-cuda124");
        assert_eq!(p.file_name().unwrap(), "kiln-v0.2.0.zip");
    }

    #[test]
    fn staging_path_version_is_embedded_verbatim() {
        // Version strings are injected from GitHub release tags, so they can
        // include pre-release / build metadata. The staging helper must not
        // drop or rewrite them.
        let base = PathBuf::from("/tmp/x");
        let p = staging_tarball_path(&base, "0.3.0-rc.1", "aarch64-apple-darwin-metal");
        assert_eq!(p.file_name().unwrap(), "kiln-v0.3.0-rc.1.tar.gz");
    }

    #[test]
    fn parse_version_plain() {
        assert_eq!(
            parse_kiln_version_output("kiln 0.1.7\n"),
            Some("0.1.7".to_string())
        );
    }

    #[test]
    fn parse_version_with_git_sha() {
        assert_eq!(
            parse_kiln_version_output("kiln 0.2.0 (abc1234)\n"),
            Some("0.2.0".to_string())
        );
    }

    #[test]
    fn parse_version_handles_empty_and_single_token() {
        assert_eq!(parse_kiln_version_output(""), None);
        assert_eq!(parse_kiln_version_output("kiln"), None);
        assert_eq!(parse_kiln_version_output("\n"), None);
    }

    #[test]
    fn update_available_when_current_unknown() {
        assert!(is_update_available(None, "0.2.0"));
    }

    #[test]
    fn update_available_for_newer_semver() {
        assert!(is_update_available(Some("0.1.7"), "0.2.0"));
        assert!(is_update_available(Some("0.1.7"), "0.1.8"));
        assert!(is_update_available(Some("0.1.0-rc.1"), "0.1.0"));
    }

    #[test]
    fn no_update_when_equal_or_newer() {
        assert!(!is_update_available(Some("0.2.0"), "0.2.0"));
        assert!(!is_update_available(Some("0.3.0"), "0.2.0"));
        assert!(!is_update_available(Some("0.1.0"), "0.1.0-rc.1"));
    }

    #[test]
    fn non_semver_falls_back_to_string_inequality() {
        // Only current is unparseable: offer when strings differ.
        assert!(is_update_available(Some("not-semver"), "0.2.0"));
        assert!(!is_update_available(Some("not-semver"), "not-semver"));
        // Only latest is unparseable: same rule.
        assert!(is_update_available(Some("0.1.0"), "not-semver"));
    }

    // ---- Slice 3a: extracted_new_binary_path / verify_sha256 / extract_tarball_to ----

    #[test]
    fn new_binary_path_is_bin_kiln_new() {
        let base = PathBuf::from("/tmp/fake-app-data");
        let p = extracted_new_binary_path(&base);
        assert_eq!(p, base.join("bin").join(NEW_BINARY_NAME));
        assert_eq!(p.file_name().unwrap(), "kiln.new");
        assert!(p.starts_with(base.join("bin")));
    }

    fn tmp_dir(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "kiln-installer-test-{}-{}-{}",
            tag,
            std::process::id(),
            // nanos since UNIX_EPOCH; good enough to avoid collisions
            // across parallel tests inside the same process.
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0)
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    #[test]
    fn verify_sha256_matches_known_digest() {
        // sha256("hello world") = b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9
        let dir = tmp_dir("verify-match");
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello world").unwrap();
        verify_sha256(
            &path,
            "b94d27b9934d3e08a52e52d7da7dabfac484efe37a5380ee9088f7ace2efcde9",
        )
        .expect("sha256 should match");
        // Also accepts uppercase + surrounding whitespace.
        verify_sha256(
            &path,
            "  B94D27B9934D3E08A52E52D7DA7DABFAC484EFE37A5380EE9088F7ACE2EFCDE9  ",
        )
        .expect("sha256 should be case/whitespace tolerant");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_sha256_rejects_mismatch() {
        let dir = tmp_dir("verify-mismatch");
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello world").unwrap();
        let err = verify_sha256(&path, &"0".repeat(64)).unwrap_err();
        assert!(err.contains("sha256 mismatch"), "got {}", err);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn verify_sha256_rejects_non_hex_expected() {
        let dir = tmp_dir("verify-nonhex");
        let path = dir.join("hello.txt");
        std::fs::write(&path, b"hello world").unwrap();
        // Wrong length.
        let err = verify_sha256(&path, "abc123").unwrap_err();
        assert!(err.contains("64-char hex"), "got {}", err);
        // Right length but contains a non-hex char.
        let mut bad = "a".repeat(63);
        bad.push('z');
        let err = verify_sha256(&path, &bad).unwrap_err();
        assert!(err.contains("64-char hex"), "got {}", err);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Build a gzipped tar archive containing a single `kiln` entry with
    /// `contents`, returning the tarball path.
    fn write_fake_tarball(dir: &std::path::Path, contents: &[u8]) -> PathBuf {
        use flate2::write::GzEncoder;
        use flate2::Compression;

        let tar_path = dir.join("kiln-fake.tar.gz");
        let file = std::fs::File::create(&tar_path).unwrap();
        let gz = GzEncoder::new(file, Compression::default());
        let mut builder = tar::Builder::new(gz);
        let mut header = tar::Header::new_gnu();
        header.set_size(contents.len() as u64);
        header.set_mode(0o755);
        header.set_cksum();
        builder
            .append_data(&mut header, "kiln", contents)
            .unwrap();
        builder.into_inner().unwrap().finish().unwrap();
        tar_path
    }

    #[test]
    fn extract_tarball_to_writes_kiln_binary() {
        let dir = tmp_dir("extract-happy");
        let tar_path = write_fake_tarball(&dir, b"fake-kiln-binary-contents");
        let target = dir.join("bin").join("kiln.new");
        let bytes = extract_tarball_to(&tar_path, &target)
            .expect("extract should succeed");
        assert!(target.is_file(), "target binary should exist");
        assert_eq!(bytes, std::fs::metadata(&target).unwrap().len());
        let got = std::fs::read(&target).unwrap();
        assert_eq!(got, b"fake-kiln-binary-contents");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_tarball_to_overwrites_existing_target() {
        let dir = tmp_dir("extract-overwrite");
        let tar_path = write_fake_tarball(&dir, b"new-contents");
        let target = dir.join("bin").join("kiln.new");
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        let mut stale = std::fs::File::create(&target).unwrap();
        stale.write_all(b"OLD-CONTENTS-that-should-be-replaced").unwrap();
        drop(stale);

        extract_tarball_to(&tar_path, &target).expect("extract should succeed");
        let got = std::fs::read(&target).unwrap();
        assert_eq!(got, b"new-contents");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_tarball_to_rejects_zip_archive() {
        let dir = tmp_dir("extract-zip");
        let zip_path = dir.join("kiln-fake.zip");
        std::fs::write(&zip_path, b"PK\x03\x04-not-a-real-zip").unwrap();
        let target = dir.join("bin").join("kiln.new");
        let err = extract_tarball_to(&zip_path, &target).unwrap_err();
        assert!(
            err.contains("windows .zip extraction"),
            "expected zip-not-supported error, got {}",
            err
        );
        assert!(!target.exists(), "no binary should be written on error");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn extract_tarball_to_rejects_archive_without_kiln_entry() {
        let dir = tmp_dir("extract-missing-kiln");
        // Build a tar.gz that contains a `README` entry but no `kiln` entry.
        use flate2::write::GzEncoder;
        use flate2::Compression;
        let tar_path = dir.join("kiln-nokiln.tar.gz");
        let file = std::fs::File::create(&tar_path).unwrap();
        let gz = GzEncoder::new(file, Compression::default());
        let mut builder = tar::Builder::new(gz);
        let body = b"just a readme";
        let mut header = tar::Header::new_gnu();
        header.set_size(body.len() as u64);
        header.set_mode(0o644);
        header.set_cksum();
        builder.append_data(&mut header, "README", &body[..]).unwrap();
        builder.into_inner().unwrap().finish().unwrap();

        let target = dir.join("bin").join("kiln.new");
        let err = extract_tarball_to(&tar_path, &target).unwrap_err();
        assert!(
            err.contains("did not contain a `kiln`"),
            "expected missing-kiln error, got {}",
            err
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Slice 3b: swap_new_binary_into_place / installed_binary_path / backup_binary_path ----

    #[test]
    fn installed_binary_path_is_bin_platform_kiln() {
        let base = PathBuf::from("/tmp/fake-app-data");
        let p = installed_binary_path(&base);
        assert!(p.starts_with(base.join("bin")));
        let expected = if cfg!(windows) { "kiln.exe" } else { "kiln" };
        assert_eq!(p.file_name().unwrap(), expected);
    }

    #[test]
    fn backup_binary_path_is_bin_kiln_bak() {
        let base = PathBuf::from("/tmp/fake-app-data");
        let p = backup_binary_path(&base);
        assert_eq!(p, base.join("bin").join(BACKUP_BINARY_NAME));
        assert_eq!(p.file_name().unwrap(), "kiln.bak");
    }

    /// Seed `<app_data_dir>/bin/` with a stand-in kiln binary (optional)
    /// and a fresh `kiln.new`. Returns the two paths the test will
    /// assert against after the swap.
    fn seed_swap_dir(
        app_data: &std::path::Path,
        prior: Option<&[u8]>,
        new_contents: &[u8],
    ) -> (PathBuf, PathBuf) {
        let bin_dir = app_data.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let installed = installed_binary_path(app_data);
        if let Some(bytes) = prior {
            std::fs::write(&installed, bytes).unwrap();
        }
        let new_path = extracted_new_binary_path(app_data);
        std::fs::write(&new_path, new_contents).unwrap();
        (installed, new_path)
    }

    #[test]
    fn swap_promotes_new_and_preserves_prior_as_bak() {
        let dir = tmp_dir("swap-happy");
        let (installed, new_path) =
            seed_swap_dir(&dir, Some(b"OLD-kiln-binary"), b"NEW-kiln-binary");

        swap_new_binary_into_place(&dir).expect("swap should succeed");

        // kiln.new is gone; installed kiln has the new bytes.
        assert!(!new_path.exists(), "kiln.new should have been consumed");
        assert!(installed.is_file(), "installed kiln should exist");
        let got = std::fs::read(&installed).unwrap();
        assert_eq!(got, b"NEW-kiln-binary");

        // Backup holds the old bytes.
        let backup = backup_binary_path(&dir);
        assert!(backup.is_file(), "backup should exist");
        let bak = std::fs::read(&backup).unwrap();
        assert_eq!(bak, b"OLD-kiln-binary");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(unix)]
    #[test]
    fn swap_sets_executable_bit_on_unix() {
        use std::os::unix::fs::PermissionsExt;
        let dir = tmp_dir("swap-chmod");
        let (installed, _) =
            seed_swap_dir(&dir, Some(b"OLD"), b"NEW");

        // Make kiln.new deliberately non-executable to prove the swap
        // restores +x rather than inheriting whatever mode extract
        // left it in.
        let new_path = extracted_new_binary_path(&dir);
        let mut perms = std::fs::metadata(&new_path).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&new_path, perms).unwrap();

        swap_new_binary_into_place(&dir).expect("swap should succeed");

        let mode = std::fs::metadata(&installed).unwrap().permissions().mode();
        assert!(
            mode & 0o111 != 0,
            "installed kiln should be executable (mode = {:o})",
            mode
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn swap_fresh_install_creates_no_bak() {
        let dir = tmp_dir("swap-fresh");
        // No prior binary — simulate first-run install.
        let (installed, new_path) =
            seed_swap_dir(&dir, None, b"FRESH-kiln-binary");
        assert!(!installed.exists());

        swap_new_binary_into_place(&dir).expect("swap should succeed");

        assert!(!new_path.exists(), "kiln.new should be consumed");
        let got = std::fs::read(&installed).unwrap();
        assert_eq!(got, b"FRESH-kiln-binary");

        let backup = backup_binary_path(&dir);
        assert!(
            !backup.exists(),
            "no .bak should be created on a fresh install"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn swap_overwrites_stale_bak_from_a_previous_swap() {
        let dir = tmp_dir("swap-stale-bak");
        let (_, _) = seed_swap_dir(&dir, Some(b"CURRENT"), b"NEXT");

        // Pretend a previous aborted swap left a stale .bak lying
        // around. The new swap should overwrite it with the current
        // installed binary rather than refusing to rename.
        let backup = backup_binary_path(&dir);
        std::fs::write(&backup, b"STALE-previous-bak").unwrap();

        swap_new_binary_into_place(&dir).expect("swap should succeed");

        let bak = std::fs::read(&backup).unwrap();
        assert_eq!(bak, b"CURRENT", "stale bak should be replaced");

        let installed = installed_binary_path(&dir);
        let got = std::fs::read(&installed).unwrap();
        assert_eq!(got, b"NEXT");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn swap_errors_when_no_kiln_new_present() {
        let dir = tmp_dir("swap-no-staged");
        // Only a prior installed binary, no kiln.new.
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        std::fs::write(installed_binary_path(&dir), b"IRRELEVANT").unwrap();

        let err = swap_new_binary_into_place(&dir).unwrap_err();
        assert!(
            err.contains("no staged kiln.new"),
            "expected missing-staged error, got {}",
            err
        );

        // Installed binary stays untouched; no .bak was created.
        let installed = installed_binary_path(&dir);
        assert!(installed.is_file());
        assert!(!backup_binary_path(&dir).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---- Slice 3c: rollback_to_bak / cleanup_bak ----

    #[test]
    fn rollback_to_bak_restores_prior_binary() {
        let dir = tmp_dir("rollback-restore");
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();

        // Stage a "new" (broken) installed binary plus the prior .bak.
        let installed = installed_binary_path(&dir);
        let backup = backup_binary_path(&dir);
        std::fs::write(&installed, b"NEW-BROKEN-BINARY").unwrap();
        std::fs::write(&backup, b"OLD-WORKING-BINARY").unwrap();

        rollback_to_bak(&dir).expect("rollback should succeed");

        // Installed slot now holds the old bytes; .bak is gone.
        assert!(installed.is_file(), "installed kiln should exist post-rollback");
        let got = std::fs::read(&installed).unwrap();
        assert_eq!(got, b"OLD-WORKING-BINARY");
        assert!(!backup.exists(), "backup should have been consumed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rollback_to_bak_errors_when_no_bak() {
        let dir = tmp_dir("rollback-no-bak");
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        // Only the (broken) installed binary exists; no .bak.
        let installed = installed_binary_path(&dir);
        std::fs::write(&installed, b"BROKEN-NO-BAK").unwrap();

        let err = rollback_to_bak(&dir).unwrap_err();
        assert!(
            err.contains("no backup"),
            "expected no-backup error, got {}",
            err
        );
        // Installed binary must stay untouched on rollback failure so
        // the caller can still surface what's there for diagnosis.
        let got = std::fs::read(&installed).unwrap();
        assert_eq!(got, b"BROKEN-NO-BAK");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_bak_is_noop_when_no_bak() {
        let dir = tmp_dir("cleanup-noop");
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        // No .bak — cleanup should still succeed.
        cleanup_bak(&dir).expect("cleanup should succeed when .bak is absent");
        assert!(!backup_binary_path(&dir).exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn cleanup_bak_removes_existing_bak() {
        let dir = tmp_dir("cleanup-remove");
        let bin_dir = dir.join("bin");
        std::fs::create_dir_all(&bin_dir).unwrap();
        let backup = backup_binary_path(&dir);
        std::fs::write(&backup, b"STALE-BAK").unwrap();
        assert!(backup.is_file());

        cleanup_bak(&dir).expect("cleanup should succeed when .bak exists");
        assert!(!backup.exists(), "backup should have been removed");

        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------- GPU arch compat gate (slice 6) ----------

    #[test]
    fn parse_compute_cap_ampere() {
        assert_eq!(parse_compute_cap("8.6\n"), Some(86));
    }

    #[test]
    fn parse_compute_cap_blackwell() {
        assert_eq!(parse_compute_cap("12.0\n"), Some(120));
    }

    #[test]
    fn parse_compute_cap_first_of_multiple() {
        // Multi-GPU hosts emit one line per GPU; take the first.
        assert_eq!(parse_compute_cap("8.0\n9.0\n"), Some(80));
    }

    #[test]
    fn parse_compute_cap_empty_is_none() {
        assert_eq!(parse_compute_cap(""), None);
    }

    #[test]
    fn parse_compute_cap_whitespace_only_is_none() {
        assert_eq!(parse_compute_cap("   \n\n"), None);
    }

    #[test]
    fn parse_compute_cap_garbage_is_none() {
        assert_eq!(parse_compute_cap("bad"), None);
    }

    #[test]
    fn parse_compute_cap_no_trailing_newline() {
        assert_eq!(parse_compute_cap("7.5"), Some(75));
    }

    #[test]
    fn parse_compute_cap_missing_minor_is_none() {
        // `compute_cap` always emits `major.minor`; anything else is a
        // shape we don't know how to interpret.
        assert_eq!(parse_compute_cap("8\n"), None);
    }

    #[test]
    fn is_supported_sm_table() {
        // SUPPORTED list from the design doc.
        for &arch in SUPPORTED_SM_ARCHS {
            assert!(is_supported_sm(arch), "SM {} should be supported", arch);
        }
        // Blackwell (SM 120) is NOT supported under CUDA 12.4.
        assert!(!is_supported_sm(120));
        // Older Turing / Volta are out of scope for kiln releases.
        assert!(!is_supported_sm(75));
        assert!(!is_supported_sm(70));
    }

    #[test]
    fn parse_compute_cap_supported_check_is_separate() {
        // parse_compute_cap should succeed on ANY `major.minor` input —
        // the SUPPORTED_SM_ARCHS check lives in is_supported_sm so an
        // unknown-but-parseable arch still surfaces to the caller.
        assert_eq!(parse_compute_cap("7.5\n"), Some(75));
        assert!(!is_supported_sm(75));
    }

    #[test]
    fn parse_supported_sm_bracketed() {
        assert_eq!(
            parse_supported_sm("foo\nsupported_sm: [80, 86, 89, 90]\nbar"),
            Some(vec![80, 86, 89, 90]),
        );
    }

    #[test]
    fn parse_supported_sm_unbracketed() {
        assert_eq!(
            parse_supported_sm("supported_sm: 80, 86"),
            Some(vec![80, 86]),
        );
    }

    #[test]
    fn parse_supported_sm_literal_backslash_n() {
        // `gh release create --notes "..."` passes its argument verbatim,
        // so workflow-emitted bodies may contain literal `\n` (two
        // characters: backslash + n) between fields instead of real
        // newlines. Normalization must handle that form too.
        let body =
            "Prebuilt kiln binaries for kiln-v1.0.0. See README for platforms.\\n\\nsupported_sm: [80, 86, 89, 90]";
        assert_eq!(
            parse_supported_sm(body),
            Some(vec![80, 86, 89, 90]),
        );
    }

    #[test]
    fn parse_supported_sm_extra_whitespace() {
        assert_eq!(
            parse_supported_sm("  supported_sm:   [ 80 , 89 ]  "),
            Some(vec![80, 89]),
        );
    }

    #[test]
    fn parse_supported_sm_missing_returns_none() {
        assert_eq!(parse_supported_sm("any text"), None);
    }

    #[test]
    fn parse_supported_sm_unparseable_element_returns_none() {
        assert_eq!(parse_supported_sm("supported_sm: [80, abc]"), None);
    }

    #[test]
    fn parse_supported_sm_empty_list_returns_none() {
        assert_eq!(parse_supported_sm("supported_sm: []"), None);
    }

    #[test]
    fn is_supported_sm_for_release_uses_parsed() {
        let body = "notes\nsupported_sm: [120]\n";
        assert!(is_supported_sm_for_release(120, Some(body)));
        assert!(!is_supported_sm_for_release(86, Some(body)));
    }

    #[test]
    fn is_supported_sm_for_release_falls_back() {
        assert!(is_supported_sm_for_release(86, None));
        assert!(!is_supported_sm_for_release(75, None));
    }

    #[test]
    fn parse_min_cuda_basic() {
        assert_eq!(parse_min_cuda("min_cuda: 12.4"), Some((12, 4)));
    }

    #[test]
    fn parse_min_cuda_bare_major() {
        assert_eq!(parse_min_cuda("min_cuda: 12"), Some((12, 0)));
    }

    #[test]
    fn parse_min_cuda_literal_backslash_n() {
        let body =
            "Prebuilt kiln binaries for kiln-v1.0.0. See README.\\n\\nsupported_sm: [80, 86, 89, 90]\\nmin_cuda: 12.4";
        assert_eq!(parse_min_cuda(body), Some((12, 4)));
    }

    #[test]
    fn parse_min_cuda_absent() {
        assert_eq!(parse_min_cuda("any text without the key"), None);
    }

    #[test]
    fn parse_min_cuda_malformed() {
        assert_eq!(parse_min_cuda("min_cuda: abc"), None);
        assert_eq!(parse_min_cuda("min_cuda: 12.abc"), None);
        assert_eq!(parse_min_cuda("min_cuda:"), None);
    }

    #[test]
    fn parse_min_cuda_extra_whitespace() {
        assert_eq!(parse_min_cuda("  min_cuda:   12.4  "), Some((12, 4)));
    }

    #[test]
    fn min_cuda_for_release_passthrough() {
        assert_eq!(
            min_cuda_for_release(Some("notes\nmin_cuda: 12.4\n")),
            Some((12, 4))
        );
        assert_eq!(min_cuda_for_release(Some("no key")), None);
        assert_eq!(min_cuda_for_release(None), None);
    }

    #[test]
    fn parse_cuda_driver_version_smi_header() {
        let out = "+-----------------------------------------------------------------------------------------+\n\
                   | NVIDIA-SMI 535.86.10   Driver Version: 535.86.10   CUDA Version: 12.2     |\n\
                   |-----------------------------------------+------------------------+----------------------+";
        assert_eq!(parse_cuda_driver_version(out), Some((12, 2)));
    }

    #[test]
    fn parse_cuda_driver_version_bare_major() {
        let out = "| Driver Version: 550.00   CUDA Version: 12     |";
        assert_eq!(parse_cuda_driver_version(out), Some((12, 0)));
    }

    #[test]
    fn parse_cuda_driver_version_missing() {
        let out = "+---------+\n| no cuda token here |\n+---------+";
        assert_eq!(parse_cuda_driver_version(out), None);
    }

    #[test]
    fn parse_cuda_driver_version_malformed() {
        let out = "| CUDA Version: abc |";
        assert_eq!(parse_cuda_driver_version(out), None);
    }

    #[test]
    fn is_cuda_compatible_for_release_no_body() {
        assert!(is_cuda_compatible_for_release(Some((11, 8)), None));
        assert!(is_cuda_compatible_for_release(None, None));
    }

    #[test]
    fn is_cuda_compatible_for_release_no_local() {
        // Detection failed → do not block the update.
        assert!(is_cuda_compatible_for_release(
            None,
            Some("min_cuda: 12.4"),
        ));
    }

    #[test]
    fn is_cuda_compatible_for_release_older_local() {
        assert!(!is_cuda_compatible_for_release(
            Some((11, 8)),
            Some("min_cuda: 12.4"),
        ));
        assert!(!is_cuda_compatible_for_release(
            Some((12, 3)),
            Some("min_cuda: 12.4"),
        ));
    }

    #[test]
    fn is_cuda_compatible_for_release_newer_local() {
        assert!(is_cuda_compatible_for_release(
            Some((12, 6)),
            Some("min_cuda: 12.4"),
        ));
        assert!(is_cuda_compatible_for_release(
            Some((13, 0)),
            Some("min_cuda: 12.4"),
        ));
    }

    #[test]
    fn is_cuda_compatible_for_release_equal() {
        assert!(is_cuda_compatible_for_release(
            Some((12, 4)),
            Some("min_cuda: 12.4"),
        ));
    }

    #[test]
    fn is_cuda_compatible_for_release_body_without_key() {
        // Release without a min_cuda: line is treated as CUDA-agnostic.
        assert!(is_cuda_compatible_for_release(
            Some((11, 0)),
            Some("supported_sm: [80, 86]"),
        ));
    }
}
