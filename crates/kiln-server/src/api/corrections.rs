//! Durable corrections store — the server-side home of the dashboard's
//! corrections basket.
//!
//! The basket is, in the dashboard's own words, "the literal mechanism of
//! 'your model gets better every time you use it'" — and until this module
//! existed it lived ONLY in one browser's localStorage: invisible to other
//! machines, unreachable by pi or any script, and (worst) the hand-written
//! ideal answers were DELETED after training, so the user's most valuable
//! data — what the model should have said, in their words — ended up
//! existing nowhere. The repo's own philosophy is "the dataset is the
//! asset"; this store makes corrections one.
//!
//! Storage: one JSON row per line at
//! `<adapter_dir>/.eval/corrections/data.jsonl`, rewritten atomically on
//! mutation (the basket is human-scale — tens to hundreds of rows).
//! Trained rows are MARKED (`trained_into`), never deleted.
//!
//! | Method | Path                          | Purpose                          |
//! |--------|-------------------------------|----------------------------------|
//! | GET    | /v1/corrections               | List (active by default;         |
//! |        |                               | `?include_trained=1` for all)    |
//! | POST   | /v1/corrections               | Upsert one row by `request_id`   |
//! | DELETE | /v1/corrections/{request_id}  | Remove one row                   |
//! | DELETE | /v1/corrections               | Clear active (untrained) rows    |

use std::path::{Path, PathBuf};

use axum::extract::{Path as AxumPath, Query, State};
use axum::routing::{delete, get};
use axum::{Json, Router};
use serde::{Deserialize, Serialize};

use crate::error::ApiError;
use crate::state::AppState;

/// One captured correction. Field names mirror the dashboard basket so the
/// write-through is mechanical; pi (or any script) can file rows with the
/// same shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectionRow {
    /// Capture identity — the originating request id (chat completion id,
    /// eval outcome key, or a caller-chosen unique string). Upsert key.
    pub request_id: String,
    /// Which client produced the original answer (pi, opencode, eval:suite).
    #[serde(default)]
    pub agent: String,
    /// Serving adapter at capture time (`None`/"base" = base model).
    #[serde(default)]
    pub adapter: Option<String>,
    /// The user prompt the model answered.
    pub user: String,
    /// What the model actually answered.
    #[serde(default)]
    pub original: String,
    /// What it SHOULD have answered — the human-authored asset.
    #[serde(default)]
    pub ideal: String,
    /// Captured from a preview-only source (text may be cut short).
    #[serde(default)]
    pub truncated: bool,
    /// RFC3339 capture time. Stamped server-side on first insert.
    #[serde(default)]
    pub created_at: String,
    /// Adapter name this row trained into, once trained. Marked, not
    /// deleted — the ideal answer outlives the job that consumed it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_into: Option<String>,
    /// RFC3339 time of the training submission that consumed this row.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trained_at: Option<String>,
}

/// Append-only-in-spirit JSONL store, rewritten atomically per mutation.
pub struct CorrectionsStore {
    dir: PathBuf,
}

impl CorrectionsStore {
    pub fn for_state(state: &AppState) -> Self {
        Self {
            dir: state.adapter_dir.join(".eval").join("corrections"),
        }
    }

    fn data_path(&self) -> PathBuf {
        self.dir.join("data.jsonl")
    }

    pub fn list(&self) -> Vec<CorrectionRow> {
        let Ok(text) = std::fs::read_to_string(self.data_path()) else {
            return Vec::new();
        };
        text.lines()
            .filter(|l| !l.trim().is_empty())
            .filter_map(|l| match serde_json::from_str::<CorrectionRow>(l) {
                Ok(row) => Some(row),
                Err(e) => {
                    tracing::warn!(error = %e, "skipping malformed corrections row");
                    None
                }
            })
            .collect()
    }

    // Test-only: `trained_rows_survive_clear_and_are_hidden_from_active_list`
    // uses it to simulate the training worker marking rows trained; live
    // mutations all go through `locked_mutate`.
    #[allow(dead_code)]
    fn write_all(&self, rows: &[CorrectionRow]) -> Result<(), String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{e}"))?;
        let mut body = String::new();
        for row in rows {
            body.push_str(&serde_json::to_string(row).map_err(|e| format!("{e}"))?);
            body.push('\n');
        }
        kiln_resource::locked_atomic_write(&self.data_path(), body.as_bytes())
            .map_err(|e| format!("{e}"))
    }

    /// Locked read-modify-write over the whole store. EVERY mutation goes
    /// through here: the file lock covers the read AND the write, so two
    /// concurrent mutators (the training worker's completion-time
    /// mark_trained_into vs a dashboard upsert) merge instead of the
    /// later write_all clobbering the earlier one — the exact lost-update
    /// race the old unlocked list() → write_all() pattern had.
    fn locked_mutate<R>(
        &self,
        mutate: impl FnOnce(&mut Vec<CorrectionRow>) -> R,
    ) -> Result<R, String> {
        std::fs::create_dir_all(&self.dir).map_err(|e| format!("{e}"))?;
        let mut result: Option<R> = None;
        kiln_resource::locked_update(&self.data_path(), |existing| {
            let mut rows: Vec<CorrectionRow> = existing
                .map(|bytes| {
                    String::from_utf8_lossy(bytes)
                        .lines()
                        .filter(|l| !l.trim().is_empty())
                        .filter_map(|l| serde_json::from_str::<CorrectionRow>(l).ok())
                        .collect()
                })
                .unwrap_or_default();
            result = Some(mutate(&mut rows));
            let mut body = String::new();
            for row in &rows {
                body.push_str(&serde_json::to_string(row).map_err(std::io::Error::other)?);
                body.push('\n');
            }
            Ok(body.into_bytes())
        })
        .map_err(|e| format!("{e}"))?;
        result.ok_or_else(|| "corrections locked_mutate: closure did not run".to_string())
    }

    /// Insert or update by `request_id`. First insert stamps `created_at`;
    /// updates preserve it (and `trained_*` unless the caller sets them).
    pub fn upsert(&self, mut row: CorrectionRow) -> Result<CorrectionRow, String> {
        self.locked_mutate(move |rows| {
            match rows.iter_mut().find(|r| r.request_id == row.request_id) {
                Some(existing) => {
                    if row.created_at.is_empty() {
                        row.created_at = existing.created_at.clone();
                    }
                    if row.trained_into.is_none() {
                        row.trained_into = existing.trained_into.clone();
                        row.trained_at = existing.trained_at.clone();
                    }
                    *existing = row.clone();
                }
                None => {
                    if row.created_at.is_empty() {
                        row.created_at = chrono::Utc::now().to_rfc3339();
                    }
                    rows.insert(0, row.clone());
                }
            }
            row
        })
    }

    pub fn remove(&self, request_id: &str) -> Result<bool, String> {
        self.locked_mutate(|rows| {
            let before = rows.len();
            rows.retain(|r| r.request_id != request_id);
            rows.len() < before
        })
    }

    /// Remove every ACTIVE (untrained) row. Trained rows are history and
    /// survive a basket clear.
    pub fn clear_active(&self) -> Result<usize, String> {
        self.locked_mutate(|rows| {
            let before = rows.len();
            rows.retain(|r| r.trained_into.is_some());
            before - rows.len()
        })
    }
}

#[derive(Debug, Deserialize)]
struct ListQuery {
    #[serde(default)]
    include_trained: Option<String>,
}

#[derive(Debug, Serialize)]
struct ListResponse {
    corrections: Vec<CorrectionRow>,
}

async fn list_corrections(
    State(state): State<AppState>,
    Query(q): Query<ListQuery>,
) -> Json<ListResponse> {
    let include_trained = matches!(q.include_trained.as_deref(), Some("1" | "true" | "yes"));
    let mut corrections = CorrectionsStore::for_state(&state).list();
    if !include_trained {
        corrections.retain(|r| r.trained_into.is_none());
    }
    Json(ListResponse { corrections })
}

async fn upsert_correction(
    State(state): State<AppState>,
    Json(row): Json<CorrectionRow>,
) -> Result<Json<CorrectionRow>, ApiError> {
    if row.request_id.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "correction request_id must be non-empty".to_string(),
        ));
    }
    if row.user.trim().is_empty() {
        return Err(ApiError::training_invalid_request(
            "correction `user` (the prompt) must be non-empty".to_string(),
        ));
    }
    CorrectionsStore::for_state(&state)
        .upsert(row)
        .map(Json)
        .map_err(|e| ApiError::internal(format!("corrections store: {e}")))
}

async fn delete_correction(
    State(state): State<AppState>,
    AxumPath(request_id): AxumPath<String>,
) -> Result<Json<DeleteCorrectionResponse>, ApiError> {
    let removed = CorrectionsStore::for_state(&state)
        .remove(&request_id)
        .map_err(|e| ApiError::internal(format!("corrections store: {e}")))?;
    if removed {
        Ok(Json(DeleteCorrectionResponse {
            status: "deleted",
            request_id,
        }))
    } else {
        Err(ApiError::training_invalid_request(format!(
            "no correction with request_id {request_id:?}"
        )))
    }
}

async fn clear_corrections(
    State(state): State<AppState>,
) -> Result<Json<ClearCorrectionsResponse>, ApiError> {
    let removed = CorrectionsStore::for_state(&state)
        .clear_active()
        .map_err(|e| ApiError::internal(format!("corrections store: {e}")))?;
    Ok(Json(ClearCorrectionsResponse {
        status: "cleared",
        removed,
    }))
}

#[derive(Debug, Serialize)]
struct DeleteCorrectionResponse {
    status: &'static str,
    request_id: String,
}

#[derive(Debug, Serialize)]
struct ClearCorrectionsResponse {
    status: &'static str,
    removed: usize,
}

/// Mark a set of rows as trained into `adapter`. Called by the dashboard
/// right after a successful corrections-train submit; the rows stay in
/// the store as history.
#[derive(Debug, Deserialize)]
struct MarkTrainedRequest {
    request_ids: Vec<String>,
    adapter: String,
}

impl CorrectionsStore {
    /// The `corrections:active` training feed: rows with a usable
    /// hand-written ideal (non-empty, differs from the original) that
    /// have not already trained into an adapter. Returns
    /// (request_ids, SFT examples) in store order.
    pub fn trainable_rows(&self) -> (Vec<String>, Vec<kiln_train::SftExample>) {
        let mut ids = Vec::new();
        let mut examples = Vec::new();
        for row in self.list() {
            let ideal = row.ideal.trim();
            if row.trained_into.is_some()
                || ideal.is_empty()
                || ideal == row.original.trim()
                || row.user.trim().is_empty()
            {
                continue;
            }
            ids.push(row.request_id.clone());
            examples.push(kiln_train::SftExample {
                messages: vec![
                    kiln_train::ChatMessage::new("user", row.user.clone()),
                    kiln_train::ChatMessage::new("assistant", row.ideal.clone()),
                ],
            });
        }
        (ids, examples)
    }

    /// Mark rows trained into `adapter` — the training queue calls this at
    /// job COMPLETION (not submission), so failed jobs leave the basket
    /// intact. Returns how many rows flipped.
    pub fn mark_trained_into(&self, request_ids: &[String], adapter: &str) -> usize {
        let now = chrono::Utc::now().to_rfc3339();
        match self.locked_mutate(|rows| {
            let mut marked = 0usize;
            for row in rows.iter_mut() {
                if request_ids.iter().any(|id| id == &row.request_id) {
                    row.trained_into = Some(adapter.to_string());
                    row.trained_at = Some(now.clone());
                    marked += 1;
                }
            }
            marked
        }) {
            Ok(marked) => marked,
            Err(e) => {
                tracing::warn!(error = %e, "corrections store: mark_trained_into failed");
                0
            }
        }
    }
}

async fn mark_trained(
    State(state): State<AppState>,
    Json(req): Json<MarkTrainedRequest>,
) -> Result<Json<MarkTrainedResponse>, ApiError> {
    let store = CorrectionsStore::for_state(&state);
    // Locked RMW — this route races the training worker's
    // completion-time marking and dashboard upserts.
    let marked = store.mark_trained_into(&req.request_ids, &req.adapter);
    Ok(Json(MarkTrainedResponse {
        status: "marked",
        marked,
    }))
}

#[derive(Debug, Serialize)]
struct MarkTrainedResponse {
    status: &'static str,
    marked: usize,
}

pub fn routes() -> Router<AppState> {
    Router::new()
        .route(
            "/v1/corrections",
            get(list_corrections)
                .post(upsert_correction)
                .delete(clear_corrections),
        )
        .route("/v1/corrections/{request_id}", delete(delete_correction))
        .route(
            "/v1/corrections/mark_trained",
            axum::routing::post(mark_trained),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store(dir: &Path) -> CorrectionsStore {
        CorrectionsStore {
            dir: dir.join(".eval").join("corrections"),
        }
    }

    fn row(id: &str, ideal: &str) -> CorrectionRow {
        CorrectionRow {
            request_id: id.to_string(),
            agent: "pi".to_string(),
            adapter: None,
            user: "what is 2+2".to_string(),
            original: "5".to_string(),
            ideal: ideal.to_string(),
            truncated: false,
            created_at: String::new(),
            trained_into: None,
            trained_at: None,
        }
    }

    #[test]
    fn upsert_stamps_created_at_once_and_updates_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());

        let first = s.upsert(row("r1", "")).unwrap();
        assert!(!first.created_at.is_empty());

        let updated = s.upsert(row("r1", "4")).unwrap();
        assert_eq!(updated.created_at, first.created_at, "created_at preserved");
        let rows = s.list();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].ideal, "4");
    }

    #[test]
    fn trained_rows_survive_clear_and_are_hidden_from_active_list() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert(row("done", "4")).unwrap();
        s.upsert(row("pending", "")).unwrap();

        // Mark "done" trained.
        let mut rows = s.list();
        rows.iter_mut()
            .find(|r| r.request_id == "done")
            .unwrap()
            .trained_into = Some("fixes-v1".into());
        s.write_all(&rows).unwrap();

        // Clearing removes only the untrained row.
        assert_eq!(s.clear_active().unwrap(), 1);
        let remaining = s.list();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].request_id, "done");
        assert_eq!(remaining[0].trained_into.as_deref(), Some("fixes-v1"));
    }

    #[test]
    fn upsert_preserves_trained_marker_when_caller_omits_it() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        let mut r = row("r1", "4");
        r.trained_into = Some("fixes-v1".into());
        s.upsert(r).unwrap();
        // A later ideal-edit upsert without trained fields keeps history.
        s.upsert(row("r1", "four")).unwrap();
        let rows = s.list();
        assert_eq!(rows[0].ideal, "four");
        assert_eq!(rows[0].trained_into.as_deref(), Some("fixes-v1"));
    }

    #[test]
    fn remove_and_persistence_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let s = store(dir.path());
        s.upsert(row("a", "x")).unwrap();
        s.upsert(row("b", "y")).unwrap();
        assert!(s.remove("a").unwrap());
        assert!(!s.remove("a").unwrap());
        // Fresh store handle reads the same file.
        let s2 = store(dir.path());
        let rows = s2.list();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].request_id, "b");
    }
    /// The corrections:active feed contract: only rows with a usable
    /// hand-written ideal resolve; completion-time marking flips exactly
    /// the consumed rows and removes them from the next feed.
    #[test]
    fn trainable_rows_and_completion_marking_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let store = CorrectionsStore {
            dir: dir.path().to_path_buf(),
        };
        let row = |id: &str, user: &str, original: &str, ideal: &str| CorrectionRow {
            request_id: id.to_string(),
            agent: "pi".to_string(),
            adapter: None,
            user: user.to_string(),
            original: original.to_string(),
            ideal: ideal.to_string(),
            truncated: false,
            created_at: String::new(),
            trained_into: None,
            trained_at: None,
        };
        store
            .upsert(row("a", "fix the bug", "wrong", "right"))
            .unwrap();
        store.upsert(row("b", "explain", "same", "same")).unwrap(); // ideal == original
        store.upsert(row("c", "write docs", "meh", "")).unwrap(); // no ideal yet

        let (ids, examples) = store.trainable_rows();
        assert_eq!(ids, vec!["a".to_string()]);
        assert_eq!(examples.len(), 1);
        assert_eq!(examples[0].messages[1].content, "right");

        // Completion marks ONLY the consumed rows…
        assert_eq!(store.mark_trained_into(&ids, "fixes-v1"), 1);
        let rows = store.list();
        let a = rows.iter().find(|r| r.request_id == "a").unwrap();
        assert_eq!(a.trained_into.as_deref(), Some("fixes-v1"));
        assert!(a.trained_at.is_some());
        // …and the next feed no longer includes them.
        let (ids2, _) = store.trainable_rows();
        assert!(ids2.is_empty());
    }
    /// The lost-update race (round-5 discovery): a training-worker
    /// mark_trained_into racing a dashboard upsert must merge — the old
    /// unlocked list() → write_all() pattern dropped one side.
    #[test]
    fn concurrent_upsert_and_mark_trained_merge() {
        let dir = tempfile::tempdir().unwrap();
        let store = std::sync::Arc::new(CorrectionsStore {
            dir: dir.path().to_path_buf(),
        });
        let row = |id: &str| CorrectionRow {
            request_id: id.to_string(),
            agent: "pi".to_string(),
            adapter: None,
            user: format!("prompt {id}"),
            original: "wrong".to_string(),
            ideal: "right".to_string(),
            truncated: false,
            created_at: String::new(),
            trained_into: None,
            trained_at: None,
        };
        store.upsert(row("seed")).unwrap();

        let s1 = store.clone();
        let mark = std::thread::spawn(move || {
            for _ in 0..50 {
                s1.mark_trained_into(&["seed".to_string()], "fixes-v1");
            }
        });
        let s2 = store.clone();
        let upserts = std::thread::spawn(move || {
            for i in 0..50 {
                s2.upsert(row(&format!("new-{i}"))).unwrap();
            }
        });
        mark.join().unwrap();
        upserts.join().unwrap();

        let rows = store.list();
        assert_eq!(rows.len(), 51, "no upsert may be lost");
        let seed = rows.iter().find(|r| r.request_id == "seed").unwrap();
        assert_eq!(
            seed.trained_into.as_deref(),
            Some("fixes-v1"),
            "the trained marker must survive concurrent upserts"
        );
    }
}
