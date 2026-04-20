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

/// Subdirectory under `app_data_dir` where update tarballs are staged
/// before the slice-3 atomic-swap step promotes them.
pub const UPDATE_STAGING_DIR: &str = "kiln-updates";

/// Filename of the staged extracted binary produced by slice 3a. Lives
/// next to `bin/kiln` so slice 3b's atomic rename stays inside the same
/// filesystem. Platform-independent on purpose: the atomic-swap step
/// targets `bin/kiln` on unix and `bin/kiln.exe` on Windows, but the
/// sibling staging file is always `bin/kiln.new`.
pub const NEW_BINARY_NAME: &str = "kiln.new";

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

/// Release-asset suffix for the current OS+arch+features combination, or
/// `None` when no prebuilt release exists for this platform. The suffix
/// matches the naming convention in `.github/workflows/server-release.yml`
/// (e.g. `aarch64-apple-darwin-metal`).
pub fn current_target() -> Option<&'static str> {
    #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
    {
        return Some("aarch64-apple-darwin-metal");
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
}

struct AssetPair {
    version: String,
    tarball: ReleaseAsset,
    sha256: Option<ReleaseAsset>,
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

#[cfg(test)]
mod tests {
    use super::{
        extract_tarball_to, extracted_new_binary_path, is_update_available,
        parse_kiln_version_output, release_archive_ext, release_asset_name,
        release_download_url, staging_tarball_path, verify_sha256, NEW_BINARY_NAME,
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
}
