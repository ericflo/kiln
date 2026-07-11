//! Recent requests ring buffer for the /ui dashboard.
//!
//! Records a bounded history of completed chat-completion requests (both
//! streaming and non-streaming) along with prompt/completion previews, token
//! counts, and timing. The /ui dashboard polls `/v1/stats/recent-requests`
//! every 2 seconds to render the panel.

use std::collections::VecDeque;

use serde::Serialize;

/// Default upper bound on retained requests. Old entries are evicted FIFO.
pub const DEFAULT_CAPACITY: usize = 100;

/// One row in the recent-requests panel.
///
/// Previews are stored already-truncated to keep the list payload small. The
/// `*_full` fields carry a larger (but still bounded) copy used by the /ui
/// inspect modal — they are capped at [`FULL_BODY_MAX_CHARS`] characters so
/// a single 1M-token prompt cannot bloat the ring.
///
/// Optional fields are skipped from JSON when `None` so cold rings stay tiny.
#[derive(Debug, Clone, Serialize, Default)]
pub struct RequestRecord {
    pub id: String,
    pub timestamp_unix_ms: u64,
    pub model: String,
    pub prompt_preview: String,
    pub completion_preview: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub duration_ms: u64,
    pub streamed: bool,
    pub finish_reason: String,

    #[serde(skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ttft_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_prefill_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_decode_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_mode: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prefix_cache: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_full: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_full: Option<String>,
    /// Raw `User-Agent` of the calling client, captured so the /ui dashboard
    /// can attribute traffic to the agent that produced it (pi, opencode, the
    /// OpenAI SDKs, curl, …). The dashboard maps it to a friendly label.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user_agent: Option<String>,
    /// First-party self-identification from the `X-Kiln-Client` request
    /// header. The /ui dashboard sends `dashboard` on its own traffic
    /// (Test connection, Playground, Compare) so onboarding milestones like
    /// "Agent connected" can tell dashboard-originated requests apart from a
    /// real external agent. Additive: absent for every other caller.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub client: Option<String>,
    /// Effective request-wide thinking-budget configuration plus the final
    /// outcome when generation progressed far enough to know it.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_budget: Option<RequestThinkingBudget>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct RequestThinkingBudget {
    pub configured: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_time_ms: Option<u64>,
    pub tokens_source: String,
    pub time_source: String,
    /// `None` means the request failed before prompt rendering established
    /// whether a configured budget could apply.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub applied: Option<bool>,
    /// Outcome fields remain absent for an unlimited or inert budget. This
    /// distinguishes those states from a completed natural close.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub triggered: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trigger: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub closed: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_tokens: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thinking_time_ms: Option<u64>,
}

/// Cap on full prompt / completion body size kept in the ring (chars).
/// The ring caps at 100 records by default, so worst case ~12.8 MB of body
/// text — comfortable, and the modal can render arbitrarily large entries
/// from the API response.
pub const FULL_BODY_MAX_CHARS: usize = 64 * 1024;

/// Bounded FIFO ring of [`RequestRecord`]s, newest at the back.
pub struct RecentRequestsRing {
    deque: VecDeque<RequestRecord>,
    capacity: usize,
}

impl RecentRequestsRing {
    pub fn new(capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            deque: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Append a record, evicting the oldest if at capacity.
    pub fn record(&mut self, record: RequestRecord) {
        if self.deque.len() >= self.capacity {
            self.deque.pop_front();
        }
        self.deque.push_back(record);
    }

    /// Return all retained records in newest-first order.
    pub fn snapshot(&self) -> Vec<RequestRecord> {
        self.deque.iter().rev().cloned().collect()
    }

    pub fn len(&self) -> usize {
        self.deque.len()
    }

    pub fn is_empty(&self) -> bool {
        self.deque.is_empty()
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl Default for RecentRequestsRing {
    fn default() -> Self {
        Self::new(DEFAULT_CAPACITY)
    }
}

/// Truncate `s` to at most `max_chars` characters, appending `…` if it was
/// shortened. Slicing on a char boundary keeps multibyte sequences intact.
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    if max_chars == 0 {
        return String::new();
    }
    let mut count = 0usize;
    for (idx, _) in s.char_indices() {
        if count == max_chars {
            let mut out = String::with_capacity(idx + 3);
            out.push_str(&s[..idx]);
            out.push('…');
            return out;
        }
        count += 1;
    }
    // The whole string fits within `max_chars`.
    s.to_owned()
}

/// Current Unix epoch in milliseconds. Saturates at 0 if the system clock is
/// before the epoch.
pub fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_record(id: &str) -> RequestRecord {
        RequestRecord {
            user_agent: None,
            id: id.to_owned(),
            timestamp_unix_ms: 0,
            model: "kiln-test".to_owned(),
            prompt_preview: "p".to_owned(),
            completion_preview: "c".to_owned(),
            prompt_tokens: 0,
            completion_tokens: 0,
            duration_ms: 0,
            streamed: false,
            finish_reason: "stop".to_owned(),
            ..Default::default()
        }
    }

    #[test]
    fn empty_ring_snapshot_is_empty() {
        let ring = RecentRequestsRing::new(8);
        assert!(ring.is_empty());
        assert_eq!(ring.len(), 0);
        assert!(ring.snapshot().is_empty());
    }

    #[test]
    fn fifo_eviction_at_capacity() {
        let mut ring = RecentRequestsRing::new(3);
        for id in ["a", "b", "c", "d", "e"] {
            ring.record(make_record(id));
        }
        assert_eq!(ring.len(), 3);
        let snap = ring.snapshot();
        let ids: Vec<&str> = snap.iter().map(|r| r.id.as_str()).collect();
        // Capacity 3, inserted 5 — only the last 3 (c, d, e) survive.
        // Snapshot is newest-first.
        assert_eq!(ids, vec!["e", "d", "c"]);
    }

    #[test]
    fn snapshot_is_newest_first() {
        let mut ring = RecentRequestsRing::new(8);
        ring.record(make_record("first"));
        ring.record(make_record("second"));
        ring.record(make_record("third"));
        let snap = ring.snapshot();
        assert_eq!(snap[0].id, "third");
        assert_eq!(snap[1].id, "second");
        assert_eq!(snap[2].id, "first");
    }

    #[test]
    fn snapshot_does_not_mutate() {
        let mut ring = RecentRequestsRing::new(8);
        ring.record(make_record("a"));
        let _ = ring.snapshot();
        assert_eq!(ring.len(), 1);
        let _ = ring.snapshot();
        assert_eq!(ring.len(), 1);
    }

    #[test]
    fn capacity_is_clamped_to_at_least_one() {
        let ring = RecentRequestsRing::new(0);
        assert_eq!(ring.capacity(), 1);
    }

    #[test]
    fn client_field_is_additive_serialized_only_when_present() {
        // Absent for every normal caller — cold rings / old clients see no
        // new key. Present verbatim when a first-party caller (the /ui
        // dashboard) self-identifies via X-Kiln-Client.
        let mut record = make_record("a");
        let json = serde_json::to_value(&record).unwrap();
        assert!(json.get("client").is_none());

        record.client = Some("dashboard".to_owned());
        let json = serde_json::to_value(&record).unwrap();
        assert_eq!(json["client"], "dashboard");
    }

    #[test]
    fn thinking_budget_record_distinguishes_unresolved_inert_and_final_outcomes() {
        let mut record = make_record("budget");
        record.thinking_budget = Some(RequestThinkingBudget {
            configured: true,
            max_tokens: Some(64),
            max_time_ms: None,
            tokens_source: "server_default".to_string(),
            time_source: "request_unlimited".to_string(),
            applied: None,
            triggered: None,
            trigger: None,
            closed: None,
            thinking_tokens: None,
            thinking_time_ms: None,
        });
        let unresolved = serde_json::to_value(&record).unwrap();
        assert_eq!(unresolved["thinking_budget"]["max_tokens"], 64);
        assert_eq!(
            unresolved["thinking_budget"]["tokens_source"],
            "server_default"
        );
        assert!(unresolved["thinking_budget"].get("applied").is_none());
        assert!(unresolved["thinking_budget"].get("triggered").is_none());

        let budget = record.thinking_budget.as_mut().unwrap();
        budget.applied = Some(false);
        let inert = serde_json::to_value(&record).unwrap();
        assert_eq!(inert["thinking_budget"]["applied"], false);
        assert!(inert["thinking_budget"].get("triggered").is_none());

        let budget = record.thinking_budget.as_mut().unwrap();
        budget.applied = Some(true);
        budget.triggered = Some(false);
        budget.closed = Some(true);
        budget.thinking_tokens = Some(17);
        budget.thinking_time_ms = Some(42);
        let natural = serde_json::to_value(&record).unwrap();
        assert_eq!(natural["thinking_budget"]["triggered"], false);
        assert_eq!(natural["thinking_budget"]["closed"], true);
        assert_eq!(natural["thinking_budget"]["thinking_tokens"], 17);
        assert_eq!(natural["thinking_budget"]["thinking_time_ms"], 42);
    }

    #[test]
    fn truncate_chars_shorter_than_max_is_unchanged() {
        assert_eq!(truncate_chars("hello", 10), "hello");
        assert_eq!(truncate_chars("", 10), "");
    }

    #[test]
    fn truncate_chars_at_exact_length_is_unchanged() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_longer_than_max_appends_ellipsis() {
        assert_eq!(truncate_chars("hello world", 5), "hello…");
    }

    #[test]
    fn truncate_chars_respects_codepoint_boundaries() {
        // "héllo" — the é is two bytes but one char. Truncating at char 3
        // must produce a valid UTF-8 string, not slice mid-codepoint.
        let s = "héllo world";
        let out = truncate_chars(s, 3);
        assert_eq!(out, "hél…");
        assert!(out.is_char_boundary(out.len()));

        // Multi-byte CJK characters.
        let cjk = "你好世界你好";
        let out = truncate_chars(cjk, 3);
        assert_eq!(out, "你好世…");
    }

    #[test]
    fn truncate_chars_max_zero_returns_empty() {
        assert_eq!(truncate_chars("hello", 0), "");
    }
}
