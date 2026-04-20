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
    use super::{is_update_available, parse_kiln_version_output};

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
}
