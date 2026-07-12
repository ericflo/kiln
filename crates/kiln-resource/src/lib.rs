//! Cross-process resource persistence primitives.
//!
//! Kiln has several process-shared resource files: autotune caches, registries,
//! and trace indexes. Direct overwrite writes are not acceptable for those
//! paths because concurrent starts can expose truncated files or clobber entries
//! tuned by a sibling process. This crate keeps the filesystem contract in one
//! place so each caller gets the same lock + atomic write behavior.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Write bytes to `path` under the resource lock for that path.
///
/// The write is cross-process serialized via a sibling lock file, written to a
/// process-unique temporary file, fsynced, and atomically renamed into place.
pub fn locked_atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    locked_update(path, |_| Ok(bytes.to_vec()))
}

/// Atomically move `source` to a new `target` without replacing any existing
/// filesystem entry at `target`.
///
/// Linux, Android, and Apple platforms provide a kernel-enforced no-replace
/// rename. Other targets fail closed instead of emulating the operation with a
/// racy existence check followed by an ordinary rename.
pub fn atomic_rename_noreplace(source: &Path, target: &Path) -> std::io::Result<()> {
    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    {
        rustix::fs::renameat_with(
            rustix::fs::CWD,
            source,
            rustix::fs::CWD,
            target,
            rustix::fs::RenameFlags::NOREPLACE,
        )
        .map_err(Into::into)
    }

    #[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
    {
        let _ = (source, target);
        Err(std::io::Error::new(
            std::io::ErrorKind::Unsupported,
            "atomic no-replace rename is unavailable on this platform",
        ))
    }
}

/// Update `path` under the resource lock for that path.
///
/// The existing bytes are read after the lock is held and passed to `update`.
/// Callers that need merge semantics should do that merge inside `update`; no
/// caller should implement its own read-modify-write loop around this function.
pub fn locked_update<F>(path: &Path, update: F) -> std::io::Result<()>
where
    F: FnOnce(Option<&[u8]>) -> std::io::Result<Vec<u8>>,
{
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let _lock = ResourceFileLock::acquire(path)?;
    let existing = match std::fs::read(path) {
        Ok(bytes) => Some(bytes),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => None,
        Err(e) => return Err(e),
    };
    let next = update(existing.as_deref())?;
    atomic_write(path, &next)
}

/// Sibling lock path used by [`locked_atomic_write`] and [`locked_update`].
pub fn lock_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("resource");
    path.with_file_name(format!(".{file_name}.lock"))
}

struct ResourceFileLock {
    path: PathBuf,
}

impl ResourceFileLock {
    const WAIT: Duration = Duration::from_millis(10);
    const MAX_ATTEMPTS: usize = 500;

    fn acquire(path: &Path) -> std::io::Result<Self> {
        let lock_path = lock_path_for(path);
        for _ in 0..Self::MAX_ATTEMPTS {
            match std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&lock_path)
            {
                Ok(mut file) => {
                    writeln!(file, "pid={}", std::process::id())?;
                    file.sync_all()?;
                    return Ok(ResourceFileLock { path: lock_path });
                }
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    if lock_owner_is_dead(&lock_path) {
                        let _ = std::fs::remove_file(&lock_path);
                    } else {
                        std::thread::sleep(Self::WAIT);
                    }
                }
                Err(e) => return Err(e),
            }
        }
        Err(std::io::Error::new(
            std::io::ErrorKind::WouldBlock,
            format!(
                "timed out waiting for resource lock {}",
                lock_path.display()
            ),
        ))
    }
}

impl Drop for ResourceFileLock {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

#[cfg(unix)]
fn lock_owner_is_dead(path: &Path) -> bool {
    let Some(pid) = lock_owner_pid(path) else {
        return false;
    };
    if pid <= 0 {
        return false;
    }
    let rc = unsafe { libc::kill(pid as libc::pid_t, 0) };
    if rc == 0 {
        return false;
    }
    std::io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH)
}

#[cfg(not(unix))]
fn lock_owner_is_dead(_path: &Path) -> bool {
    false
}

fn lock_owner_pid(path: &Path) -> Option<i32> {
    let text = std::fs::read_to_string(path).ok()?;
    text.lines()
        .find_map(|line| line.strip_prefix("pid=")?.trim().parse::<i32>().ok())
}

fn atomic_write(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let tmp = temp_path_for(path);
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&tmp)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    match std::fs::rename(&tmp, path) {
        Ok(()) => {
            if let Some(parent) = path.parent() {
                if let Ok(dir) = std::fs::File::open(parent) {
                    let _ = dir.sync_all();
                }
            }
            Ok(())
        }
        Err(e) => {
            let _ = std::fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn temp_path_for(path: &Path) -> PathBuf {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("resource");
    let thread = format!("{:?}", std::thread::current().id())
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect::<String>();
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(
        ".{file_name}.{}.{}.{}.tmp",
        std::process::id(),
        thread,
        seq
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locked_update_merges_concurrent_writers_without_partial_files() {
        let dir =
            std::env::temp_dir().join(format!("kiln-resource-concurrent-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("resource.json");

        let handles = (0..12u64)
            .map(|i| {
                let path = path.clone();
                std::thread::spawn(move || {
                    locked_update(&path, |existing| {
                        let mut values = existing
                            .map(|bytes| {
                                std::str::from_utf8(bytes)
                                    .unwrap()
                                    .lines()
                                    .filter_map(|line| line.parse::<u64>().ok())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();
                        values.push(i);
                        values.sort_unstable();
                        Ok(values
                            .into_iter()
                            .map(|v| format!("{v}\n"))
                            .collect::<String>()
                            .into_bytes())
                    })
                    .unwrap();
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }

        let text = std::fs::read_to_string(&path).unwrap();
        let values = text
            .lines()
            .map(|line| line.parse::<u64>().unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values, (0..12u64).collect::<Vec<_>>());
        assert_no_temp_or_lock_files(&dir, &path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn locked_atomic_write_cleans_up_lock_and_temp_file() {
        let dir = std::env::temp_dir().join(format!("kiln-resource-write-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("resource.json");

        locked_atomic_write(&path, b"{\"ok\":true}").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"{\"ok\":true}");
        assert_no_temp_or_lock_files(&dir, &path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
    #[test]
    fn atomic_rename_noreplace_publishes_new_target_and_preserves_collision() {
        let dir = std::env::temp_dir().join(format!(
            "kiln-resource-noreplace-{}-{}",
            std::process::id(),
            TEMP_SEQ.fetch_add(1, Ordering::Relaxed)
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir(&dir).unwrap();
        let source = dir.join("source");
        let target = dir.join("target");
        std::fs::create_dir(&source).unwrap();
        std::fs::write(source.join("payload"), b"first").unwrap();

        atomic_rename_noreplace(&source, &target).unwrap();
        assert!(!source.exists());
        assert_eq!(std::fs::read(target.join("payload")).unwrap(), b"first");

        let replacement = dir.join("replacement");
        std::fs::create_dir(&replacement).unwrap();
        std::fs::write(replacement.join("payload"), b"second").unwrap();
        let error = atomic_rename_noreplace(&replacement, &target).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert_eq!(std::fs::read(target.join("payload")).unwrap(), b"first");
        assert_eq!(
            std::fs::read(replacement.join("payload")).unwrap(),
            b"second"
        );

        let empty_target = dir.join("empty-target");
        std::fs::create_dir(&empty_target).unwrap();
        let error = atomic_rename_noreplace(&replacement, &empty_target).unwrap_err();
        assert_eq!(error.kind(), std::io::ErrorKind::AlreadyExists);
        assert!(empty_target.read_dir().unwrap().next().is_none());
        assert_eq!(
            std::fs::read(replacement.join("payload")).unwrap(),
            b"second"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn dead_owner_lock_is_reclaimed() {
        let dir =
            std::env::temp_dir().join(format!("kiln-resource-dead-owner-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("resource.json");
        std::fs::write(lock_path_for(&path), "pid=2147483647\n").unwrap();

        locked_atomic_write(&path, b"reclaimed").unwrap();

        assert_eq!(std::fs::read(&path).unwrap(), b"reclaimed");
        assert_no_temp_or_lock_files(&dir, &path);
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn assert_no_temp_or_lock_files(dir: &Path, path: &Path) {
        assert_eq!(
            std::fs::read_dir(dir)
                .unwrap()
                .filter_map(|entry| entry.ok())
                .filter(|entry| entry.file_name().to_string_lossy().contains(".tmp"))
                .count(),
            0
        );
        assert!(!lock_path_for(path).exists());
    }
}
