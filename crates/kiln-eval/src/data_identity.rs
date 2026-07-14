//! Content identities and deterministic train/eval splits.
//!
//! Dataset rows have two identities: a canonical JSON identity and a
//! deliberately conservative normalized identity that folds string case and
//! whitespace. Split assignment operates on connected components linked by a
//! normalized row identity, `group_id`, or `session_id`, so duplicates and
//! related turns cannot leak across partitions.

use std::collections::{BTreeMap, BTreeSet, HashMap};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

pub const DATASET_IDENTITY_SCHEMA_V1: &str = "kiln.dataset-identity.v1";
pub const DATASET_SPLIT_SCHEMA_V1: &str = "kiln.dataset-split.v1";
pub const DATASET_PROVENANCE_METADATA_KEY: &str = "kiln_dataset_provenance";

#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DatasetSplit {
    #[default]
    Train,
    Validation,
    Holdout,
}

impl DatasetSplit {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Train => "train",
            Self::Validation => "validation",
            Self::Holdout => "holdout",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplitConfig {
    /// Stable split seed. It is serialized as decimal text so JavaScript
    /// clients cannot silently round values above 2^53.
    #[serde(with = "crate::result::u64_decimal")]
    pub seed: u64,
    /// Percentage assigned to training. Validation receives
    /// `validation_percent`; holdout receives the remainder.
    pub train_percent: u8,
    pub validation_percent: u8,
}

impl Default for DatasetSplitConfig {
    fn default() -> Self {
        Self {
            seed: 0,
            train_percent: 80,
            validation_percent: 10,
        }
    }
}

impl DatasetSplitConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.train_percent == 0 {
            return Err("train_percent must be greater than zero".to_string());
        }
        let assigned = u16::from(self.train_percent) + u16::from(self.validation_percent);
        if assigned >= 100 {
            return Err(
                "train_percent + validation_percent must be less than 100 to reserve a non-zero holdout percentage"
                    .to_string(),
            );
        }
        Ok(())
    }

    pub fn holdout_percent(&self) -> u8 {
        100 - self.train_percent - self.validation_percent
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetRowIdentity {
    /// One-indexed non-empty JSONL row number.
    pub row_number: u64,
    pub content_sha256: String,
    pub normalized_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetIdentityIndex {
    pub schema: String,
    pub dataset_name: String,
    pub corpus_sha256: String,
    pub normalized_corpus_sha256: String,
    pub rows: Vec<DatasetRowIdentity>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplitCounts {
    pub train: u64,
    pub validation: u64,
    pub holdout: u64,
}

impl DatasetSplitCounts {
    pub fn get(&self, split: DatasetSplit) -> u64 {
        match split {
            DatasetSplit::Train => self.train,
            DatasetSplit::Validation => self.validation,
            DatasetSplit::Holdout => self.holdout,
        }
    }

    fn increment(&mut self, split: DatasetSplit) {
        match split {
            DatasetSplit::Train => self.train += 1,
            DatasetSplit::Validation => self.validation += 1,
            DatasetSplit::Holdout => self.holdout += 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplitRow {
    pub row_number: u64,
    pub content_sha256: String,
    pub normalized_sha256: String,
    pub split: DatasetSplit,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetSplitManifest {
    pub schema: String,
    pub dataset_name: String,
    pub corpus_sha256: String,
    pub normalized_corpus_sha256: String,
    pub config: DatasetSplitConfig,
    pub counts: DatasetSplitCounts,
    pub rows: Vec<DatasetSplitRow>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(deny_unknown_fields)]
pub struct DatasetExampleIdentity {
    pub content_sha256: String,
    pub normalized_sha256: String,
}

/// Metadata attached to synthesized examples. This is source provenance, not
/// a claim that the example belongs to a held-out partition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct DatasetExampleProvenance {
    pub dataset: String,
    pub source_split: DatasetSplit,
    pub row: DatasetRowIdentity,
}

/// The first conservative overlap found between admitted training data and an
/// eval suite. Exact prompt/target overlap is checked before normalized
/// overlap; persisted row and grouping provenance provide a stronger signal
/// for suites synthesized from registered datasets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContaminationMatch {
    ExactExample,
    NormalizedExample,
    SourceRow,
    NormalizedSourceRow,
    Group,
    Session,
}

impl ContaminationMatch {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ExactExample => "exact prompt/target",
            Self::NormalizedExample => "normalized prompt/target",
            Self::SourceRow => "source row",
            Self::NormalizedSourceRow => "normalized source row",
            Self::Group => "source group",
            Self::Session => "source session",
        }
    }
}

/// Bounded index of identities present in one eval suite. Training admission
/// streams candidate rows/groups through this index, so even very large
/// training corpora do not require a second in-memory identity set.
#[derive(Debug, Clone, Default)]
pub struct EvalContaminationIndex {
    example_content_sha256: BTreeSet<String>,
    example_normalized_sha256: BTreeSet<String>,
    source_content_sha256: BTreeSet<String>,
    source_normalized_sha256: BTreeSet<String>,
    group_ids: BTreeSet<String>,
    session_ids: BTreeSet<String>,
}

impl EvalContaminationIndex {
    pub fn from_suite(suite: &crate::EvalSuite) -> Self {
        let mut index = Self::default();
        for example in &suite.examples {
            let identity = example_identity(&example.messages, example.target.as_deref());
            index.example_content_sha256.insert(identity.content_sha256);
            index
                .example_normalized_sha256
                .insert(identity.normalized_sha256);
            let provenance = example
                .metadata
                .as_ref()
                .and_then(serde_json::Value::as_object)
                .and_then(|metadata| metadata.get(DATASET_PROVENANCE_METADATA_KEY))
                .and_then(|value| {
                    serde_json::from_value::<DatasetExampleProvenance>(value.clone()).ok()
                });
            if let Some(provenance) = provenance {
                index
                    .source_content_sha256
                    .insert(provenance.row.content_sha256);
                index
                    .source_normalized_sha256
                    .insert(provenance.row.normalized_sha256);
                if let Some(group_id) = provenance.row.group_id {
                    index.group_ids.insert(group_id);
                }
                if let Some(session_id) = provenance.row.session_id {
                    index.session_ids.insert(session_id);
                }
            }
        }
        index
    }

    pub fn check_example(
        &self,
        messages: &[crate::EvalChatMessage],
        target: Option<&str>,
    ) -> Option<ContaminationMatch> {
        let identity = example_identity(messages, target);
        if self
            .example_content_sha256
            .contains(&identity.content_sha256)
        {
            return Some(ContaminationMatch::ExactExample);
        }
        self.example_normalized_sha256
            .contains(&identity.normalized_sha256)
            .then_some(ContaminationMatch::NormalizedExample)
    }

    pub fn check_source_row(&self, row: &DatasetSplitRow) -> Option<ContaminationMatch> {
        if self.source_content_sha256.contains(&row.content_sha256) {
            return Some(ContaminationMatch::SourceRow);
        }
        if self
            .source_normalized_sha256
            .contains(&row.normalized_sha256)
        {
            return Some(ContaminationMatch::NormalizedSourceRow);
        }
        if row
            .group_id
            .as_ref()
            .is_some_and(|group_id| self.group_ids.contains(group_id))
        {
            return Some(ContaminationMatch::Group);
        }
        row.session_id
            .as_ref()
            .is_some_and(|session_id| self.session_ids.contains(session_id))
            .then_some(ContaminationMatch::Session)
    }
}

pub fn canonical_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), canonical_json(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        other => other.clone(),
    }
}

pub fn normalized_json(value: &serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::String(value) => serde_json::Value::String(normalize_text(value)),
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.iter().map(normalized_json).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values
                .iter()
                .map(|(key, value)| (key.clone(), normalized_json(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        other => other.clone(),
    }
}

pub fn normalize_text(value: &str) -> String {
    value
        .split_whitespace()
        .map(|part| part.to_lowercase())
        .collect::<Vec<_>>()
        .join(" ")
}

pub fn sha256_json(value: &serde_json::Value) -> String {
    let bytes = serde_json::to_vec(&canonical_json(value)).unwrap_or_default();
    sha256_bytes(&bytes)
}

pub fn normalized_sha256_json(value: &serde_json::Value) -> String {
    let normalized = normalized_json(value);
    let bytes = serde_json::to_vec(&canonical_json(&normalized)).unwrap_or_default();
    sha256_bytes(&bytes)
}

pub fn sha256_bytes(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    sha256_digest_string(digest.as_slice())
}

fn sha256_digest_string(digest: &[u8]) -> String {
    use std::fmt::Write;

    let mut encoded = String::with_capacity("sha256:".len() + digest.len() * 2);
    encoded.push_str("sha256:");
    for byte in digest {
        write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
    }
    encoded
}

pub fn row_identity(row_number: u64, value: &serde_json::Value) -> DatasetRowIdentity {
    DatasetRowIdentity {
        row_number,
        content_sha256: sha256_json(value),
        normalized_sha256: normalized_sha256_json(value),
        group_id: identity_field(value, "group_id"),
        session_id: identity_field(value, "session_id"),
    }
}

fn identity_field(value: &serde_json::Value, field: &str) -> Option<String> {
    fn from_object(
        object: &serde_json::Map<String, serde_json::Value>,
        field: &str,
    ) -> Option<String> {
        object
            .get(field)
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    }

    let object = value.as_object()?;
    from_object(object, field).or_else(|| {
        ["metadata", "provenance", "extra"]
            .iter()
            .find_map(|container| {
                object
                    .get(*container)
                    .and_then(serde_json::Value::as_object)
                    .and_then(|nested| from_object(nested, field))
            })
    })
}

pub fn build_identity_index(
    dataset_name: impl Into<String>,
    rows: &[(u64, serde_json::Value)],
) -> DatasetIdentityIndex {
    let identities = rows
        .iter()
        .map(|(row_number, value)| row_identity(*row_number, value))
        .collect::<Vec<_>>();
    build_identity_index_from_rows(dataset_name, identities)
}

pub fn build_identity_index_from_rows(
    dataset_name: impl Into<String>,
    identities: Vec<DatasetRowIdentity>,
) -> DatasetIdentityIndex {
    let corpus_sha256 = aggregate_identities(
        identities.iter().map(|row| row.content_sha256.as_str()),
        b"kiln.dataset-corpus.v1\0",
    );
    let normalized_corpus_sha256 = aggregate_identities(
        identities.iter().map(|row| row.normalized_sha256.as_str()),
        b"kiln.dataset-normalized-corpus.v1\0",
    );
    DatasetIdentityIndex {
        schema: DATASET_IDENTITY_SCHEMA_V1.to_string(),
        dataset_name: dataset_name.into(),
        corpus_sha256,
        normalized_corpus_sha256,
        rows: identities,
    }
}

fn aggregate_identities<'a>(
    identities: impl IntoIterator<Item = &'a str>,
    domain: &[u8],
) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    for identity in identities {
        hasher.update((identity.len() as u64).to_be_bytes());
        hasher.update(identity.as_bytes());
    }
    sha256_digest_string(hasher.finalize().as_slice())
}

pub fn build_split_manifest(
    index: &DatasetIdentityIndex,
    config: DatasetSplitConfig,
) -> Result<DatasetSplitManifest, String> {
    config.validate()?;
    let mut sets = DisjointSets::new(index.rows.len());
    let mut witnesses: HashMap<String, usize> = HashMap::new();
    for (index, row) in index.rows.iter().enumerate() {
        let keys = [
            Some(format!("normalized:{}", row.normalized_sha256)),
            row.group_id.as_ref().map(|value| format!("group:{value}")),
            row.session_id
                .as_ref()
                .map(|value| format!("session:{value}")),
        ];
        for key in keys.into_iter().flatten() {
            if let Some(previous) = witnesses.insert(key, index) {
                sets.union(previous, index);
            }
        }
    }

    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for index in 0..index.rows.len() {
        components.entry(sets.find(index)).or_default().push(index);
    }
    let mut assignments = HashMap::new();
    let mut component_buckets = Vec::new();
    for (root, members) in &components {
        let key = members
            .iter()
            .map(|row_index| index.rows[*row_index].normalized_sha256.as_str())
            .min()
            .unwrap_or("");
        let bucket = split_bucket(config.seed, key);
        let split = if bucket < config.train_percent {
            DatasetSplit::Train
        } else if bucket < config.train_percent + config.validation_percent {
            DatasetSplit::Validation
        } else {
            DatasetSplit::Holdout
        };
        assignments.insert(*root, split);
        component_buckets.push((*root, bucket));
    }
    ensure_useful_small_dataset_assignments(&mut assignments, &mut component_buckets, &config);

    let mut counts = DatasetSplitCounts::default();
    let rows = index
        .rows
        .iter()
        .enumerate()
        .map(|(row_index, row)| {
            let split = assignments[&sets.find(row_index)];
            counts.increment(split);
            DatasetSplitRow {
                row_number: row.row_number,
                content_sha256: row.content_sha256.clone(),
                normalized_sha256: row.normalized_sha256.clone(),
                split,
                group_id: row.group_id.clone(),
                session_id: row.session_id.clone(),
            }
        })
        .collect();
    Ok(DatasetSplitManifest {
        schema: DATASET_SPLIT_SCHEMA_V1.to_string(),
        dataset_name: index.dataset_name.clone(),
        corpus_sha256: index.corpus_sha256.clone(),
        normalized_corpus_sha256: index.normalized_corpus_sha256.clone(),
        config,
        counts,
        rows,
    })
}

fn ensure_useful_small_dataset_assignments(
    assignments: &mut HashMap<usize, DatasetSplit>,
    component_buckets: &mut [(usize, u8)],
    config: &DatasetSplitConfig,
) {
    component_buckets.sort_by_key(|(_, bucket)| *bucket);
    if !assignments
        .values()
        .any(|split| *split == DatasetSplit::Train)
        && let Some((root, _)) = component_buckets.first()
    {
        assignments.insert(*root, DatasetSplit::Train);
    }
    if component_buckets.len() >= 2
        && config.holdout_percent() > 0
        && !assignments
            .values()
            .any(|split| *split == DatasetSplit::Holdout)
        && let Some(root) = movable_component(
            assignments,
            component_buckets.iter().rev(),
            if component_buckets.len() >= 3 && config.validation_percent > 0 {
                &[DatasetSplit::Train, DatasetSplit::Validation]
            } else {
                &[DatasetSplit::Train]
            },
        )
    {
        assignments.insert(root, DatasetSplit::Holdout);
    }
    if component_buckets.len() >= 3
        && config.validation_percent > 0
        && !assignments
            .values()
            .any(|split| *split == DatasetSplit::Validation)
        && let Some(root) = movable_component(
            assignments,
            component_buckets.iter(),
            &[DatasetSplit::Train, DatasetSplit::Holdout],
        )
    {
        assignments.insert(root, DatasetSplit::Validation);
    }
}

fn movable_component<'a>(
    assignments: &HashMap<usize, DatasetSplit>,
    candidates: impl IntoIterator<Item = &'a (usize, u8)>,
    protected: &[DatasetSplit],
) -> Option<usize> {
    let mut counts = HashMap::<DatasetSplit, usize>::new();
    for split in assignments.values() {
        *counts.entry(*split).or_default() += 1;
    }
    let candidates = candidates
        .into_iter()
        .map(|(root, _)| *root)
        .collect::<Vec<_>>();
    candidates
        .iter()
        .copied()
        .find(|root| counts.get(&assignments[root]).copied().unwrap_or(0) > 1)
        .or_else(|| {
            candidates
                .into_iter()
                .find(|root| !protected.contains(&assignments[root]))
        })
}

fn split_bucket(seed: u64, key: &str) -> u8 {
    let mut hasher = Sha256::new();
    hasher.update(b"kiln.dataset-split-assignment.v1\0");
    hasher.update(seed.to_be_bytes());
    hasher.update(key.as_bytes());
    let digest = hasher.finalize();
    u64::from_be_bytes(digest[..8].try_into().expect("sha256 prefix")).wrapping_rem(100) as u8
}

pub fn example_identity(
    messages: &[crate::EvalChatMessage],
    target: Option<&str>,
) -> DatasetExampleIdentity {
    let value = serde_json::json!({
        "messages": messages,
        "target": target,
    });
    DatasetExampleIdentity {
        content_sha256: sha256_json(&value),
        normalized_sha256: normalized_sha256_json(&value),
    }
}

struct DisjointSets {
    parent: Vec<usize>,
    rank: Vec<u8>,
}

impl DisjointSets {
    fn new(len: usize) -> Self {
        Self {
            parent: (0..len).collect(),
            rank: vec![0; len],
        }
    }

    fn find(&mut self, value: usize) -> usize {
        let parent = self.parent[value];
        if parent != value {
            self.parent[value] = self.find(parent);
        }
        self.parent[value]
    }

    fn union(&mut self, left: usize, right: usize) {
        let mut left = self.find(left);
        let mut right = self.find(right);
        if left == right {
            return;
        }
        if self.rank[left] < self.rank[right] {
            std::mem::swap(&mut left, &mut right);
        }
        self.parent[right] = left;
        if self.rank[left] == self.rank[right] {
            self.rank[left] += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rows(values: Vec<serde_json::Value>) -> Vec<(u64, serde_json::Value)> {
        values
            .into_iter()
            .enumerate()
            .map(|(index, value)| (index as u64 + 1, value))
            .collect()
    }

    #[test]
    fn canonical_identity_ignores_object_key_order_only() {
        let left = serde_json::json!({"b": 2, "a": "Hello  world"});
        let right = serde_json::json!({"a": "Hello  world", "b": 2});
        assert_eq!(sha256_json(&left), sha256_json(&right));
        assert_eq!(
            normalized_sha256_json(&left),
            normalized_sha256_json(&right)
        );
    }

    #[test]
    fn normalized_identity_folds_case_and_whitespace() {
        let left = serde_json::json!({"messages": [{"content": " Hello\nWORLD "}]});
        let right = serde_json::json!({"messages": [{"content": "hello world"}]});
        assert_ne!(sha256_json(&left), sha256_json(&right));
        assert_eq!(
            normalized_sha256_json(&left),
            normalized_sha256_json(&right)
        );
    }

    #[test]
    fn split_keeps_exact_normalized_and_grouped_duplicates_together() {
        let source = rows(vec![
            serde_json::json!({"group_id": "g1", "session_id": "s1", "text": "Alpha"}),
            serde_json::json!({"group_id": "g1", "session_id": "s2", "text": "Beta"}),
            serde_json::json!({"group_id": "g2", "session_id": "s2", "text": "Gamma"}),
            serde_json::json!({"text": " Exact duplicate "}),
            serde_json::json!({"text": " Exact duplicate "}),
            serde_json::json!({"text": "normalized   DUPLICATE"}),
            serde_json::json!({"text": "NORMALIZED duplicate"}),
            serde_json::json!({"text": "independent one"}),
            serde_json::json!({"text": "independent two"}),
        ]);
        let index = build_identity_index("fixture", &source);
        let manifest = build_split_manifest(&index, DatasetSplitConfig::default()).unwrap();
        let split = |row: usize| manifest.rows[row - 1].split;
        assert_eq!(split(1), split(2));
        assert_eq!(split(2), split(3));
        assert_eq!(split(4), split(5));
        assert_eq!(split(6), split(7));
    }

    #[test]
    fn split_is_order_independent_for_unrelated_rows() {
        let original = rows(vec![
            serde_json::json!({"id": "a"}),
            serde_json::json!({"id": "b"}),
            serde_json::json!({"id": "c"}),
            serde_json::json!({"id": "d"}),
        ]);
        let reversed = rows(
            original
                .iter()
                .rev()
                .map(|(_, value)| value.clone())
                .collect(),
        );
        let assignments = |rows: &[(u64, serde_json::Value)]| {
            let index = build_identity_index("fixture", rows);
            build_split_manifest(&index, DatasetSplitConfig::default())
                .unwrap()
                .rows
                .into_iter()
                .map(|row| (row.content_sha256, row.split))
                .collect::<BTreeMap<_, _>>()
        };
        assert_eq!(assignments(&original), assignments(&reversed));
    }

    #[test]
    fn small_dataset_repair_exhaustively_populates_every_possible_partition() {
        let splits = [
            DatasetSplit::Train,
            DatasetSplit::Validation,
            DatasetSplit::Holdout,
        ];
        for first in splits {
            for second in splits {
                let mut assignments = HashMap::from([(0, first), (1, second)]);
                let mut buckets = [(0, 10), (1, 90)];
                ensure_useful_small_dataset_assignments(
                    &mut assignments,
                    &mut buckets,
                    &DatasetSplitConfig::default(),
                );
                assert!(
                    assignments
                        .values()
                        .any(|split| *split == DatasetSplit::Train)
                );
                assert!(
                    assignments
                        .values()
                        .any(|split| *split == DatasetSplit::Holdout),
                    "two-component assignment {first:?}/{second:?} omitted holdout"
                );
            }
        }

        for first in splits {
            for second in splits {
                for third in splits {
                    let mut assignments = HashMap::from([(0, first), (1, second), (2, third)]);
                    let mut buckets = [(0, 10), (1, 50), (2, 90)];
                    ensure_useful_small_dataset_assignments(
                        &mut assignments,
                        &mut buckets,
                        &DatasetSplitConfig::default(),
                    );
                    for required in splits {
                        assert!(
                            assignments.values().any(|split| *split == required),
                            "three-component assignment {first:?}/{second:?}/{third:?} omitted {required:?}"
                        );
                    }
                }
            }
        }
    }

    fn contamination_suite(
        prompt: &str,
        target: &str,
        provenance: Option<DatasetExampleProvenance>,
    ) -> crate::EvalSuite {
        crate::EvalSuite {
            name: "contamination-fixture".to_string(),
            description: None,
            default_scorer: crate::Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: true,
            },
            generation: Default::default(),
            aggregation: Default::default(),
            system_prompt: None,
            examples: vec![crate::EvalExample {
                messages: vec![crate::EvalChatMessage::new("user", prompt)],
                target: Some(target.to_string()),
                metadata: provenance.map(
                    |provenance| serde_json::json!({DATASET_PROVENANCE_METADATA_KEY: provenance}),
                ),
                ..Default::default()
            }],
            schema_version: crate::SUITE_SCHEMA_VERSION,
            tools: None,
        }
    }

    #[test]
    fn contamination_index_detects_exact_and_normalized_examples() {
        let index = EvalContaminationIndex::from_suite(&contamination_suite(
            "Answer This",
            "Forty Two",
            None,
        ));
        assert_eq!(
            index.check_example(
                &[crate::EvalChatMessage::new("user", "Answer This")],
                Some("Forty Two")
            ),
            Some(ContaminationMatch::ExactExample)
        );
        assert_eq!(
            index.check_example(
                &[crate::EvalChatMessage::new("user", " answer\nTHIS ")],
                Some("forty   two")
            ),
            Some(ContaminationMatch::NormalizedExample)
        );
    }

    #[test]
    fn contamination_index_detects_grouped_source_provenance() {
        let source = DatasetRowIdentity {
            row_number: 1,
            content_sha256: "sha256:source".to_string(),
            normalized_sha256: "sha256:normalized-source".to_string(),
            group_id: Some("group-7".to_string()),
            session_id: Some("session-9".to_string()),
        };
        let index = EvalContaminationIndex::from_suite(&contamination_suite(
            "prompt",
            "target",
            Some(DatasetExampleProvenance {
                dataset: "fixture".to_string(),
                source_split: DatasetSplit::Holdout,
                row: source,
            }),
        ));
        let related = DatasetSplitRow {
            row_number: 2,
            content_sha256: "sha256:different".to_string(),
            normalized_sha256: "sha256:also-different".to_string(),
            split: DatasetSplit::Train,
            group_id: Some("group-7".to_string()),
            session_id: None,
        };
        assert_eq!(
            index.check_source_row(&related),
            Some(ContaminationMatch::Group)
        );
    }
}
