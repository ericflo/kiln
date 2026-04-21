//! Phase C1 — per-step MTP acceptance-rate attribution sink.
//!
//! Records one row per [`crate::speculative::speculative_mtp_decode_step`]
//! call so we can attribute a low overall α (acceptance rate) to one of two
//! classes under **greedy** decoding (temperature == 0):
//!
//! - **Class A — verification/sampling bug.** MTP top-1 equals the main-head
//!   top-1 at the same position, but the accept check still reports reject.
//!   Impossible under a correct greedy compare.
//! - **Class B — MTP head bug.** MTP top-1 disagrees with the main-head
//!   top-1. Either the MTP forward pass is wrong, or the pretrained head
//!   really is this noisy on this workload.
//!
//! Enabled only when `KILN_C1_ATTR_PATH` is set — both [`push_row`] and
//! [`drain_to_csv`] early-out cheaply when the env var is unset. The accept
//! check in [`crate::speculative::speculative_mtp_decode_step`] is the only
//! call site; the bench driver (`kiln-server/src/bench.rs`) calls
//! [`drain_to_csv`] once at the end of the MTP loop.
//!
//! For Qwen3.5-4B k=1 MTP (the only supported checkpoint today) there is
//! exactly one draft per step, so `pos_in_k` is always 0. The field is
//! retained so the CSV format stays future-proof if a k>1 MTP config lands
//! later.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};

/// One attribution row — one call to `speculative_mtp_decode_step`.
#[derive(Debug, Clone, PartialEq)]
pub struct C1Row {
    /// Monotonically increasing step index within the current bench run.
    pub step_idx: usize,
    /// Position within the K-token MTP draft window. Always 0 for Qwen3.5-4B
    /// k=1 MTP; kept for forward compatibility with k>1 configs.
    pub pos_in_k: usize,
    /// Absolute position in the base stream (input-side) for this step.
    pub base_pos: usize,
    /// Slot index inside the isolated MTP paged cache for this step.
    pub mtp_pos: usize,
    /// Input token fed to both the MTP draft and the base verify pass.
    pub last_token: u32,
    /// Token chosen by the MTP head (greedy top-1 over `mtp_logits`).
    pub mtp_top1: u32,
    /// Top-1 logit magnitude from the MTP draft distribution.
    pub mtp_top1_logit: f32,
    /// Token chosen by the main base head at `base_pos` (greedy top-1 over
    /// `verify_logits0`).
    pub main_top1: u32,
    /// Top-1 logit magnitude from the base verify-pos-0 distribution.
    pub main_top1_logit: f32,
    /// Accept decision recorded by `speculative_mtp_decode_step`.
    pub accepted: bool,
    /// Independent check `mtp_top1 == main_top1`. Under greedy this must
    /// equal `accepted`; any divergence is a Class-A bug signal.
    pub topk_match: bool,
}

static SINK: Mutex<Vec<C1Row>> = Mutex::new(Vec::new());
static NEXT_STEP_IDX: AtomicUsize = AtomicUsize::new(0);

/// Return the next monotonic step index and bump the counter. Callers fill
/// this into [`C1Row::step_idx`] before [`push_row`]. Reset to 0 by
/// [`clear`]; a fresh process also starts at 0.
pub fn next_step_idx() -> usize {
    NEXT_STEP_IDX.fetch_add(1, Ordering::Relaxed)
}

/// Returns true when `KILN_C1_ATTR_PATH` is set to a non-empty string. Read
/// on every call (no caching) so tests can toggle between runs without a
/// process restart.
pub fn is_enabled() -> bool {
    std::env::var("KILN_C1_ATTR_PATH")
        .map(|v| !v.is_empty())
        .unwrap_or(false)
}

/// Append a row to the global sink. No-op when [`is_enabled`] is false, so
/// callers can wrap the row construction in a cheap guard and pay zero cost
/// on production decode paths.
pub fn push_row(row: C1Row) {
    if !is_enabled() {
        return;
    }
    if let Ok(mut sink) = SINK.lock() {
        sink.push(row);
    }
}

/// Clear the global sink and reset [`next_step_idx`] to 0. Used between test
/// cases and between bench arms so a single process can capture multiple
/// independent attribution traces.
pub fn clear() {
    if let Ok(mut sink) = SINK.lock() {
        sink.clear();
    }
    NEXT_STEP_IDX.store(0, Ordering::Relaxed);
}

/// Take ownership of the current rows, leaving the sink empty. Used by the
/// CSV writer and by tests that want to inspect the captured rows directly.
pub fn drain() -> Vec<C1Row> {
    match SINK.lock() {
        Ok(mut sink) => std::mem::take(&mut *sink),
        Err(_) => Vec::new(),
    }
}

/// CSV header row — stable format consumed by
/// `scripts/mtp_c1_summarize.py`.
pub const CSV_HEADER: &str = "step_idx,pos_in_k,base_pos,mtp_pos,last_token,mtp_top1,mtp_top1_logit,main_top1,main_top1_logit,accepted,topk_match";

/// Serialize a single row to CSV, matching [`CSV_HEADER`].
pub fn row_to_csv_line(row: &C1Row) -> String {
    format!(
        "{},{},{},{},{},{},{:.6},{},{:.6},{},{}",
        row.step_idx,
        row.pos_in_k,
        row.base_pos,
        row.mtp_pos,
        row.last_token,
        row.mtp_top1,
        row.mtp_top1_logit,
        row.main_top1,
        row.main_top1_logit,
        row.accepted as u8,
        row.topk_match as u8,
    )
}

/// Parse one CSV line back into a [`C1Row`]. Used by the round-trip test and
/// by any future Rust-side analysis tool; the canonical consumer remains
/// `scripts/mtp_c1_summarize.py`.
pub fn row_from_csv_line(line: &str) -> Result<C1Row> {
    let parts: Vec<&str> = line.split(',').collect();
    anyhow::ensure!(
        parts.len() == 11,
        "expected 11 CSV fields, got {} in line: {line:?}",
        parts.len()
    );
    let u = |i: usize, name: &str| -> Result<u64> {
        parts[i]
            .parse::<u64>()
            .with_context(|| format!("parse field {i} ({name}): {:?}", parts[i]))
    };
    let f = |i: usize, name: &str| -> Result<f32> {
        parts[i]
            .parse::<f32>()
            .with_context(|| format!("parse field {i} ({name}): {:?}", parts[i]))
    };
    let b = |i: usize, name: &str| -> Result<bool> {
        match parts[i] {
            "0" => Ok(false),
            "1" => Ok(true),
            other => anyhow::bail!("field {i} ({name}) must be 0 or 1, got {other:?}"),
        }
    };
    Ok(C1Row {
        step_idx: u(0, "step_idx")? as usize,
        pos_in_k: u(1, "pos_in_k")? as usize,
        base_pos: u(2, "base_pos")? as usize,
        mtp_pos: u(3, "mtp_pos")? as usize,
        last_token: u(4, "last_token")? as u32,
        mtp_top1: u(5, "mtp_top1")? as u32,
        mtp_top1_logit: f(6, "mtp_top1_logit")?,
        main_top1: u(7, "main_top1")? as u32,
        main_top1_logit: f(8, "main_top1_logit")?,
        accepted: b(9, "accepted")?,
        topk_match: b(10, "topk_match")?,
    })
}

/// Drain the sink and write all rows as CSV to `path`. No-op when the sink
/// is empty. Always clears the sink, even on I/O error, so a failed write
/// never leaks into the next run.
pub fn drain_to_csv(path: &str) -> Result<usize> {
    let rows = drain();
    let n = rows.len();
    if n == 0 {
        return Ok(0);
    }
    let mut out = String::with_capacity(128 + n * 64);
    out.push_str(CSV_HEADER);
    out.push('\n');
    for row in &rows {
        out.push_str(&row_to_csv_line(row));
        out.push('\n');
    }
    std::fs::write(path, out).with_context(|| format!("write C1 attribution CSV to {path}"))?;
    Ok(n)
}

#[cfg(test)]
mod tests {
    use super::*;

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn sample_row(i: usize, accepted: bool) -> C1Row {
        C1Row {
            step_idx: i,
            pos_in_k: 0,
            base_pos: 100 + i,
            mtp_pos: i,
            last_token: 42,
            mtp_top1: 7 + i as u32,
            mtp_top1_logit: 12.5 - i as f32 * 0.1,
            main_top1: if accepted { 7 + i as u32 } else { 999 },
            main_top1_logit: 14.0 + i as f32 * 0.01,
            accepted,
            topk_match: accepted,
        }
    }

    #[test]
    fn csv_round_trip_single_row() {
        let row = sample_row(3, true);
        let line = row_to_csv_line(&row);
        let parsed = row_from_csv_line(&line).expect("parse");
        assert_eq!(parsed.step_idx, row.step_idx);
        assert_eq!(parsed.pos_in_k, row.pos_in_k);
        assert_eq!(parsed.base_pos, row.base_pos);
        assert_eq!(parsed.mtp_pos, row.mtp_pos);
        assert_eq!(parsed.last_token, row.last_token);
        assert_eq!(parsed.mtp_top1, row.mtp_top1);
        assert!((parsed.mtp_top1_logit - row.mtp_top1_logit).abs() < 1e-4);
        assert_eq!(parsed.main_top1, row.main_top1);
        assert!((parsed.main_top1_logit - row.main_top1_logit).abs() < 1e-4);
        assert_eq!(parsed.accepted, row.accepted);
        assert_eq!(parsed.topk_match, row.topk_match);
    }

    #[test]
    fn csv_round_trip_multi_row_via_file() {
        let _guard = ENV_LOCK.lock().unwrap();
        let tmp = std::env::temp_dir().join(format!(
            "kiln_c1_round_trip_{}.csv",
            std::process::id()
        ));
        let path = tmp.to_str().unwrap().to_string();
        // SAFETY: ENV_LOCK serializes tests that mutate this process-wide
        // environment variable; the env var toggle is the documented way to
        // enable / disable the sink.
        unsafe {
            std::env::set_var("KILN_C1_ATTR_PATH", &path);
        }
        clear();
        for i in 0..5 {
            push_row(sample_row(i, i % 2 == 0));
        }
        let n = drain_to_csv(&path).expect("drain_to_csv");
        assert_eq!(n, 5);
        let body = std::fs::read_to_string(&path).expect("read back");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines[0], CSV_HEADER);
        assert_eq!(lines.len(), 6, "header + 5 rows");
        for (i, line) in lines[1..].iter().enumerate() {
            let parsed = row_from_csv_line(line).expect("parse back");
            let expected = sample_row(i, i % 2 == 0);
            assert_eq!(parsed.step_idx, expected.step_idx);
            assert_eq!(parsed.mtp_top1, expected.mtp_top1);
            assert_eq!(parsed.accepted, expected.accepted);
            assert_eq!(parsed.topk_match, expected.topk_match);
        }
        // SAFETY: same as above — single-threaded env mutation.
        unsafe {
            std::env::remove_var("KILN_C1_ATTR_PATH");
        }
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn push_is_no_op_when_disabled() {
        let _guard = ENV_LOCK.lock().unwrap();
        // SAFETY: ENV_LOCK serializes tests that mutate this process-wide
        // environment variable.
        unsafe {
            std::env::remove_var("KILN_C1_ATTR_PATH");
        }
        clear();
        push_row(sample_row(0, true));
        let drained = drain();
        assert!(drained.is_empty(), "push_row must no-op when disabled");
    }
}
