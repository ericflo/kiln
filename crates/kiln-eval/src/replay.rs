//! Self-validating eval replay records and byte-comparison verdicts.

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::result::{EVAL_SEED_DERIVATION_V1, EvalThinkingBudget, ExampleOutcome};
use crate::suite::{EvalGenerationParams, EvalSuite};

pub const EVAL_REPLAY_RECORD_SCHEMA_VERSION: u32 = 1;
pub const EVAL_REPLAY_RECORD_TYPE: &str = "kiln.eval-replay.v1";
pub const EVAL_REPLAY_EXPECTATION_TYPE: &str = "kiln.eval-replay-expectation.v1";

#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{message}")]
pub struct EvalReplayError {
    message: String,
}

impl EvalReplayError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

/// Exact model target used for candidate generation or LLM-judge scoring.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalModelTargetIdentity {
    /// `None` identifies the immutable base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter: Option<String>,
    /// Loader-derived identity of the exact adapter config and weight bytes.
    /// Required for a named adapter and absent for the base model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub adapter_content_sha256: Option<String>,
}

impl EvalModelTargetIdentity {
    pub fn base() -> Self {
        Self {
            adapter: None,
            adapter_content_sha256: None,
        }
    }

    pub fn adapter(
        name: impl Into<String>,
        content_revision: &str,
    ) -> Result<Self, EvalReplayError> {
        let adapter_content_sha256 = format!("sha256:{content_revision}");
        let identity = Self {
            adapter: Some(name.into()),
            adapter_content_sha256: Some(adapter_content_sha256),
        };
        identity.validate()?;
        Ok(identity)
    }

    pub fn validate(&self) -> Result<(), EvalReplayError> {
        match (
            self.adapter.as_deref(),
            self.adapter_content_sha256.as_deref(),
        ) {
            (None, None) => Ok(()),
            (Some(name), Some(digest)) if !name.trim().is_empty() => {
                validate_sha256("adapter_content_sha256", digest)
            }
            (Some(_), None) => Err(EvalReplayError::new(
                "named replay adapter is missing adapter_content_sha256",
            )),
            (None, Some(_)) => Err(EvalReplayError::new(
                "base replay target must not carry adapter_content_sha256",
            )),
            (Some(_), Some(_)) => Err(EvalReplayError::new(
                "replay adapter name must not be empty",
            )),
        }
    }
}

/// Exact serialized scorer configuration selected for one stable example ID.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalScorerIdentity {
    pub example_id: String,
    pub kind: String,
    pub config_sha256: String,
    pub requires_judge: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub judge_adapter: Option<String>,
}

/// Content-addressed pointer into the adjacent `SuiteResult.outcomes` array.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalRawCompletionReference {
    pub outcome_index: u32,
    pub example_id: String,
    pub completion_index: u32,
    #[serde(
        default,
        with = "crate::result::optional_u64_decimal",
        skip_serializing_if = "Option::is_none"
    )]
    pub generation_seed: Option<u64>,
    pub raw_completion_pointer: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_completion_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub raw_completion_bytes: Option<u64>,
    pub normalized_completion_sha256: String,
    pub normalized_completion_bytes: u64,
}

/// Complete immutable input and output identity for one suite/adapter run.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvalReplayRecordV1 {
    pub schema_version: u32,
    pub record_type: String,
    pub suite: EvalSuite,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generation_override: Option<EvalGenerationParams>,
    #[serde(with = "crate::result::u64_decimal")]
    pub effective_seed: u64,
    pub seed_derivation: String,
    pub resolved_thinking_budgets: Vec<EvalThinkingBudget>,
    /// Absent only when a synthetic generator cannot attest its target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_target: Option<EvalModelTargetIdentity>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub judge_targets: Vec<EvalModelTargetIdentity>,
    pub scorer_identities: Vec<EvalScorerIdentity>,
    pub raw_completions: Vec<EvalRawCompletionReference>,
    pub suite_sha256: String,
    pub effective_generation_sha256: String,
    pub raw_completion_set_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution_provenance_sha256: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub base_weight_manifest_sha256: Option<String>,
    pub record_sha256: String,
}

impl EvalReplayRecordV1 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        suite: EvalSuite,
        generation_override: Option<EvalGenerationParams>,
        effective_seed: u64,
        resolved_thinking_budgets: Vec<EvalThinkingBudget>,
        model_target: Option<EvalModelTargetIdentity>,
        mut judge_targets: Vec<EvalModelTargetIdentity>,
        execution_provenance_sha256: Option<String>,
        base_weight_manifest_sha256: Option<String>,
        outcomes: &[ExampleOutcome],
    ) -> Result<Self, EvalReplayError> {
        judge_targets.sort_by(|left, right| {
            left.adapter.cmp(&right.adapter).then_with(|| {
                left.adapter_content_sha256
                    .cmp(&right.adapter_content_sha256)
            })
        });
        judge_targets.dedup();
        let suite_sha256 = eval_suite_sha256(&suite)?;
        let effective_generation_sha256 = eval_effective_generation_sha256(
            &suite,
            generation_override.as_ref(),
            effective_seed,
            &resolved_thinking_budgets,
        )?;
        let scorer_identities = scorer_identities(&suite)?;
        let raw_completions = raw_completion_references(outcomes)?;
        let raw_completion_set_sha256 =
            hash_json(b"kiln.eval-raw-completion-set.v1\0", &raw_completions)?;
        let mut record = Self {
            schema_version: EVAL_REPLAY_RECORD_SCHEMA_VERSION,
            record_type: EVAL_REPLAY_RECORD_TYPE.to_string(),
            suite,
            generation_override,
            effective_seed,
            seed_derivation: EVAL_SEED_DERIVATION_V1.to_string(),
            resolved_thinking_budgets,
            model_target,
            judge_targets,
            scorer_identities,
            raw_completions,
            suite_sha256,
            effective_generation_sha256,
            raw_completion_set_sha256,
            execution_provenance_sha256,
            base_weight_manifest_sha256,
            record_sha256: prefixed_digest(&[0; 32]),
        };
        record.record_sha256 = record.compute_record_sha256()?;
        record.validate(outcomes)?;
        Ok(record)
    }

    pub fn validate(&self, outcomes: &[ExampleOutcome]) -> Result<(), EvalReplayError> {
        if self.schema_version != EVAL_REPLAY_RECORD_SCHEMA_VERSION {
            return Err(EvalReplayError::new(format!(
                "unsupported replay schema_version {}; expected {EVAL_REPLAY_RECORD_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.record_type != EVAL_REPLAY_RECORD_TYPE {
            return Err(EvalReplayError::new(format!(
                "invalid replay record_type {:?}",
                self.record_type
            )));
        }
        if self.seed_derivation != EVAL_SEED_DERIVATION_V1 {
            return Err(EvalReplayError::new(format!(
                "unsupported replay seed_derivation {:?}",
                self.seed_derivation
            )));
        }
        self.suite
            .validate()
            .map_err(|error| EvalReplayError::new(format!("invalid replay suite: {error}")))?;
        if self.resolved_thinking_budgets.len() != self.suite.examples.len() {
            return Err(EvalReplayError::new(format!(
                "replay thinking-budget count {} does not match {} suite examples",
                self.resolved_thinking_budgets.len(),
                self.suite.examples.len()
            )));
        }
        for (example, budget) in self
            .suite
            .examples
            .iter()
            .zip(&self.resolved_thinking_budgets)
        {
            budget.validate().map_err(|error| {
                EvalReplayError::new(format!(
                    "invalid replay thinking budget for example {:?}: {error}",
                    example.resolved_id()
                ))
            })?;
        }
        if let Some(target) = self.model_target.as_ref() {
            target.validate()?;
        }
        let mut judge_selectors = BTreeSet::new();
        for target in &self.judge_targets {
            target.validate()?;
            if !judge_selectors.insert(target.adapter.clone()) {
                return Err(EvalReplayError::new(format!(
                    "duplicate replay judge target {:?}",
                    target.adapter
                )));
            }
        }
        for (label, digest) in [
            ("suite_sha256", Some(self.suite_sha256.as_str())),
            (
                "effective_generation_sha256",
                Some(self.effective_generation_sha256.as_str()),
            ),
            (
                "raw_completion_set_sha256",
                Some(self.raw_completion_set_sha256.as_str()),
            ),
            (
                "execution_provenance_sha256",
                self.execution_provenance_sha256.as_deref(),
            ),
            (
                "base_weight_manifest_sha256",
                self.base_weight_manifest_sha256.as_deref(),
            ),
            ("record_sha256", Some(self.record_sha256.as_str())),
        ] {
            if let Some(digest) = digest {
                validate_sha256(label, digest)?;
            }
        }
        let expected_suite = eval_suite_sha256(&self.suite)?;
        if self.suite_sha256 != expected_suite {
            return Err(EvalReplayError::new(format!(
                "replay suite digest mismatch: record has {}, expected {expected_suite}",
                self.suite_sha256
            )));
        }
        let expected_generation = eval_effective_generation_sha256(
            &self.suite,
            self.generation_override.as_ref(),
            self.effective_seed,
            &self.resolved_thinking_budgets,
        )?;
        if self.effective_generation_sha256 != expected_generation {
            return Err(EvalReplayError::new(format!(
                "replay generation digest mismatch: record has {}, expected {expected_generation}",
                self.effective_generation_sha256
            )));
        }
        let expected_scorers = scorer_identities(&self.suite)?;
        if self.scorer_identities != expected_scorers {
            return Err(EvalReplayError::new(
                "replay scorer identities do not match the suite snapshot",
            ));
        }
        let expected_raw = raw_completion_references(outcomes)?;
        if self.raw_completions != expected_raw {
            return Err(EvalReplayError::new(
                "replay raw-completion references do not match the retained outcomes",
            ));
        }
        let expected_raw_set = hash_json(b"kiln.eval-raw-completion-set.v1\0", &expected_raw)?;
        if self.raw_completion_set_sha256 != expected_raw_set {
            return Err(EvalReplayError::new(format!(
                "replay raw-completion set digest mismatch: record has {}, expected {expected_raw_set}",
                self.raw_completion_set_sha256
            )));
        }
        let expected_record = self.compute_record_sha256()?;
        if self.record_sha256 != expected_record {
            return Err(EvalReplayError::new(format!(
                "replay record digest mismatch: record has {}, expected {expected_record}",
                self.record_sha256
            )));
        }
        Ok(())
    }

    /// Require every identity needed to claim a strict byte replay.
    pub fn validate_strict_replay(
        &self,
        outcomes: &[ExampleOutcome],
    ) -> Result<(), EvalReplayError> {
        self.validate(outcomes)?;
        if self.execution_provenance_sha256.is_none() {
            return Err(EvalReplayError::new(
                "strict replay requires execution_provenance_sha256",
            ));
        }
        if self.base_weight_manifest_sha256.is_none() {
            return Err(EvalReplayError::new(
                "strict replay requires base_weight_manifest_sha256",
            ));
        }
        if self.model_target.is_none() {
            return Err(EvalReplayError::new(
                "strict replay requires an attested model target",
            ));
        }
        let required_judges = self
            .scorer_identities
            .iter()
            .filter(|identity| identity.requires_judge)
            .map(|identity| identity.judge_adapter.clone())
            .collect::<BTreeSet<_>>();
        let recorded_judges = self
            .judge_targets
            .iter()
            .map(|target| target.adapter.clone())
            .collect::<BTreeSet<_>>();
        if required_judges != recorded_judges {
            return Err(EvalReplayError::new(format!(
                "strict replay judge identities are incomplete: required {required_judges:?}, recorded {recorded_judges:?}"
            )));
        }
        if let Some(reference) = self
            .raw_completions
            .iter()
            .find(|reference| reference.raw_completion_sha256.is_none())
        {
            return Err(EvalReplayError::new(format!(
                "strict replay cannot reproduce outcome {} ({}/{}): raw decoder bytes are absent",
                reference.outcome_index, reference.example_id, reference.completion_index
            )));
        }
        let mut expected_completions = BTreeSet::new();
        for (example_index, example) in self.suite.examples.iter().enumerate() {
            let params = example
                .generation
                .as_ref()
                .or(self.generation_override.as_ref())
                .unwrap_or(&self.suite.generation);
            let example_id = example.resolved_id();
            let seed_root = params.seed.unwrap_or(self.effective_seed);
            let budget = &self.resolved_thinking_budgets[example_index];
            budget.validate().map_err(|error| {
                EvalReplayError::new(format!(
                    "invalid replay thinking budget for example {example_id:?}: {error}"
                ))
            })?;
            for completion_index in 0..params.n {
                let completion_index = u32::try_from(completion_index)
                    .map_err(|_| EvalReplayError::new("replay completion index exceeds u32"))?;
                expected_completions.insert((example_id.clone(), completion_index));
            }
            for outcome in outcomes
                .iter()
                .filter(|outcome| outcome.example_id == example_id)
            {
                let expected_seed = crate::result::derive_eval_completion_seed(
                    seed_root,
                    &example_id,
                    outcome.completion_index,
                );
                if outcome.generation_seed != Some(expected_seed) {
                    return Err(EvalReplayError::new(format!(
                        "replay seed mismatch for {example_id:?} completion {}: expected {expected_seed}, got {:?}",
                        outcome.completion_index, outcome.generation_seed
                    )));
                }
                let observed_budget = outcome.thinking_budget.as_ref().ok_or_else(|| {
                    EvalReplayError::new(format!(
                        "strict replay is missing thinking-budget evidence for {example_id:?} completion {}",
                        outcome.completion_index
                    ))
                })?;
                observed_budget.validate().map_err(|error| {
                    EvalReplayError::new(format!(
                        "invalid outcome thinking budget for {example_id:?} completion {}: {error}",
                        outcome.completion_index
                    ))
                })?;
                if observed_budget.configured != budget.configured
                    || observed_budget.max_tokens != budget.max_tokens
                    || observed_budget.max_time_ms != budget.max_time_ms
                    || observed_budget.tokens_source != budget.tokens_source
                    || observed_budget.time_source != budget.time_source
                {
                    return Err(EvalReplayError::new(format!(
                        "thinking-budget identity mismatch for {example_id:?} completion {}",
                        outcome.completion_index
                    )));
                }
            }
        }
        let actual_completions = outcomes
            .iter()
            .map(|outcome| {
                u32::try_from(outcome.completion_index)
                    .map(|index| (outcome.example_id.clone(), index))
                    .map_err(|_| EvalReplayError::new("eval outcome index exceeds u32"))
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        if actual_completions != expected_completions
            || outcomes.len() != expected_completions.len()
        {
            return Err(EvalReplayError::new(format!(
                "strict replay completion coverage mismatch: expected {expected_completions:?}, got {actual_completions:?}"
            )));
        }
        Ok(())
    }

    fn compute_record_sha256(&self) -> Result<String, EvalReplayError> {
        #[derive(Serialize)]
        struct IdentityFields<'a> {
            schema_version: u32,
            record_type: &'a str,
            suite: &'a EvalSuite,
            generation_override: &'a Option<EvalGenerationParams>,
            #[serde(with = "crate::result::u64_decimal")]
            effective_seed: u64,
            seed_derivation: &'a str,
            resolved_thinking_budgets: &'a [EvalThinkingBudget],
            model_target: &'a Option<EvalModelTargetIdentity>,
            judge_targets: &'a [EvalModelTargetIdentity],
            scorer_identities: &'a [EvalScorerIdentity],
            raw_completions: &'a [EvalRawCompletionReference],
            suite_sha256: &'a str,
            effective_generation_sha256: &'a str,
            raw_completion_set_sha256: &'a str,
            execution_provenance_sha256: &'a Option<String>,
            base_weight_manifest_sha256: &'a Option<String>,
        }
        hash_json(
            b"kiln.eval-replay-record.v1\0",
            &IdentityFields {
                schema_version: self.schema_version,
                record_type: &self.record_type,
                suite: &self.suite,
                generation_override: &self.generation_override,
                effective_seed: self.effective_seed,
                seed_derivation: &self.seed_derivation,
                resolved_thinking_budgets: &self.resolved_thinking_budgets,
                model_target: &self.model_target,
                judge_targets: &self.judge_targets,
                scorer_identities: &self.scorer_identities,
                raw_completions: &self.raw_completions,
                suite_sha256: &self.suite_sha256,
                effective_generation_sha256: &self.effective_generation_sha256,
                raw_completion_set_sha256: &self.raw_completion_set_sha256,
                execution_provenance_sha256: &self.execution_provenance_sha256,
                base_weight_manifest_sha256: &self.base_weight_manifest_sha256,
            },
        )
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalReplayExpectationV1 {
    pub expectation_type: String,
    pub source_job_id: String,
    pub source_run_index: u32,
    pub expected_record_sha256: String,
    pub expected_raw_completion_set_sha256: String,
}

impl EvalReplayExpectationV1 {
    pub fn new(source_job_id: String, source_run_index: u32, record: &EvalReplayRecordV1) -> Self {
        Self {
            expectation_type: EVAL_REPLAY_EXPECTATION_TYPE.to_string(),
            source_job_id,
            source_run_index,
            expected_record_sha256: record.record_sha256.clone(),
            expected_raw_completion_set_sha256: record.raw_completion_set_sha256.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), EvalReplayError> {
        if self.expectation_type != EVAL_REPLAY_EXPECTATION_TYPE {
            return Err(EvalReplayError::new(format!(
                "invalid replay expectation_type {:?}",
                self.expectation_type
            )));
        }
        if self.source_job_id.trim().is_empty() {
            return Err(EvalReplayError::new(
                "replay source_job_id must not be empty",
            ));
        }
        validate_sha256("expected_record_sha256", &self.expected_record_sha256)?;
        validate_sha256(
            "expected_raw_completion_set_sha256",
            &self.expected_raw_completion_set_sha256,
        )
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EvalReplayStatus {
    Matched,
    Mismatch,
    Error,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct EvalReplayVerdict {
    pub status: EvalReplayStatus,
    pub source_job_id: String,
    pub source_run_index: u32,
    pub expected_record_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_record_sha256: Option<String>,
    pub expected_raw_completion_set_sha256: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub actual_raw_completion_set_sha256: Option<String>,
    pub message: String,
}

impl EvalReplayVerdict {
    pub fn validate(&self, expectation: &EvalReplayExpectationV1) -> Result<(), EvalReplayError> {
        expectation.validate()?;
        if self.source_job_id != expectation.source_job_id
            || self.source_run_index != expectation.source_run_index
            || self.expected_record_sha256 != expectation.expected_record_sha256
            || self.expected_raw_completion_set_sha256
                != expectation.expected_raw_completion_set_sha256
        {
            return Err(EvalReplayError::new(
                "replay verdict does not match its persisted expectation",
            ));
        }
        if self.message.trim().is_empty() {
            return Err(EvalReplayError::new(
                "replay verdict message must not be empty",
            ));
        }
        if let Some(digest) = self.actual_record_sha256.as_deref() {
            validate_sha256("actual_record_sha256", digest)?;
        }
        if let Some(digest) = self.actual_raw_completion_set_sha256.as_deref() {
            validate_sha256("actual_raw_completion_set_sha256", digest)?;
        }
        match self.status {
            EvalReplayStatus::Matched => {
                if self.actual_record_sha256.as_deref()
                    != Some(self.expected_record_sha256.as_str())
                    || self.actual_raw_completion_set_sha256.as_deref()
                        != Some(self.expected_raw_completion_set_sha256.as_str())
                {
                    return Err(EvalReplayError::new(
                        "matched replay verdict must carry both expected digests exactly",
                    ));
                }
            }
            EvalReplayStatus::Mismatch => {
                let Some(actual_record) = self.actual_record_sha256.as_deref() else {
                    return Err(EvalReplayError::new(
                        "mismatch replay verdict is missing actual_record_sha256",
                    ));
                };
                let Some(actual_raw) = self.actual_raw_completion_set_sha256.as_deref() else {
                    return Err(EvalReplayError::new(
                        "mismatch replay verdict is missing actual_raw_completion_set_sha256",
                    ));
                };
                if actual_record == self.expected_record_sha256
                    && actual_raw == self.expected_raw_completion_set_sha256
                {
                    return Err(EvalReplayError::new(
                        "mismatch replay verdict carries two matching digests",
                    ));
                }
            }
            EvalReplayStatus::Error => {}
        }
        Ok(())
    }
}

pub fn eval_suite_sha256(suite: &EvalSuite) -> Result<String, EvalReplayError> {
    hash_json(b"kiln.eval-suite.v1\0", suite)
}

pub fn eval_effective_generation_sha256(
    suite: &EvalSuite,
    generation_override: Option<&EvalGenerationParams>,
    effective_seed: u64,
    resolved_thinking_budgets: &[EvalThinkingBudget],
) -> Result<String, EvalReplayError> {
    #[derive(Serialize)]
    struct GenerationIdentity<'a> {
        seed_derivation: &'static str,
        #[serde(with = "crate::result::u64_decimal")]
        effective_seed: u64,
        suite_sha256: String,
        generation_override: Option<&'a EvalGenerationParams>,
        resolved_thinking_budgets: &'a [EvalThinkingBudget],
    }
    hash_json(
        b"kiln.eval-effective-generation.v1\0",
        &GenerationIdentity {
            seed_derivation: EVAL_SEED_DERIVATION_V1,
            effective_seed,
            suite_sha256: eval_suite_sha256(suite)?,
            generation_override,
            resolved_thinking_budgets,
        },
    )
}

fn scorer_identities(suite: &EvalSuite) -> Result<Vec<EvalScorerIdentity>, EvalReplayError> {
    suite
        .examples
        .iter()
        .map(|example| {
            let scorer = example.scorer.as_ref().unwrap_or(&suite.default_scorer);
            Ok(EvalScorerIdentity {
                example_id: example.resolved_id(),
                kind: scorer.kind_label().to_string(),
                config_sha256: hash_json(b"kiln.eval-scorer.v1\0", scorer)?,
                requires_judge: scorer.requires_judge(),
                judge_adapter: scorer.judge_adapter().map(str::to_string),
            })
        })
        .collect()
}

fn raw_completion_references(
    outcomes: &[ExampleOutcome],
) -> Result<Vec<EvalRawCompletionReference>, EvalReplayError> {
    outcomes
        .iter()
        .enumerate()
        .map(|(index, outcome)| {
            let outcome_index = u32::try_from(index)
                .map_err(|_| EvalReplayError::new("eval outcome index exceeds u32"))?;
            let completion_index = u32::try_from(outcome.completion_index)
                .map_err(|_| EvalReplayError::new("eval completion index exceeds u32"))?;
            let raw_completion_sha256 = outcome
                .raw_completion_text
                .as_deref()
                .map(|text| sha256_bytes(text.as_bytes()));
            let raw_completion_bytes = outcome
                .raw_completion_text
                .as_deref()
                .map(|text| u64::try_from(text.len()))
                .transpose()
                .map_err(|_| EvalReplayError::new("raw completion length exceeds u64"))?;
            let normalized_completion_bytes = u64::try_from(outcome.completion_text.len())
                .map_err(|_| EvalReplayError::new("normalized completion length exceeds u64"))?;
            Ok(EvalRawCompletionReference {
                outcome_index,
                example_id: outcome.example_id.clone(),
                completion_index,
                generation_seed: outcome.generation_seed,
                raw_completion_pointer: format!("/outcomes/{index}/raw_completion_text"),
                raw_completion_sha256,
                raw_completion_bytes,
                normalized_completion_sha256: sha256_bytes(outcome.completion_text.as_bytes()),
                normalized_completion_bytes,
            })
        })
        .collect()
}

fn hash_json(domain: &[u8], value: &impl Serialize) -> Result<String, EvalReplayError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| EvalReplayError::new(format!("serialize replay identity: {error}")))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    Ok(prefixed_digest(&hasher.finalize()))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    prefixed_digest(&Sha256::digest(bytes))
}

fn prefixed_digest(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(71);
    output.push_str("sha256:");
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(output, "{byte:02x}");
    }
    output
}

fn validate_sha256(label: &str, value: &str) -> Result<(), EvalReplayError> {
    let Some(raw) = value.strip_prefix("sha256:") else {
        return Err(EvalReplayError::new(format!(
            "{label} must start with sha256:"
        )));
    };
    if raw.len() != 64
        || !raw
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(EvalReplayError::new(format!(
            "{label} must contain exactly 64 lowercase hexadecimal characters"
        )));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::result::EvalOutcomeKind;
    use crate::scorers::Scorer;
    use crate::suite::{EvalAggregation, EvalChatMessage, EvalExample};

    fn suite() -> EvalSuite {
        EvalSuite {
            name: "replay".into(),
            description: None,
            default_scorer: Scorer::ExactMatch {
                case_sensitive: true,
                strip_whitespace: false,
            },
            generation: EvalGenerationParams::default(),
            aggregation: EvalAggregation::Single,
            system_prompt: None,
            examples: vec![EvalExample {
                id: Some("one".into()),
                messages: vec![EvalChatMessage::new("user", "say x")],
                target: Some("x".into()),
                ..Default::default()
            }],
            schema_version: 1,
            tools: None,
        }
    }

    fn outcomes() -> Vec<ExampleOutcome> {
        vec![ExampleOutcome {
            example_id: "one".into(),
            completion_index: 0,
            generation_seed: Some(crate::result::derive_eval_completion_seed(7, "one", 0)),
            completion_text: "x".into(),
            raw_completion_text: Some("x".into()),
            thinking_budget: Some(EvalThinkingBudget::default()),
            kind: EvalOutcomeKind::Pass,
            score: 1.0,
            detail: None,
            prompt_tokens: Some(2),
            completion_tokens: Some(1),
            latency_ms: Some(1.0),
            tags: Vec::new(),
            metadata: None,
            reasoning_text: None,
            unclosed_thinking: false,
        }]
    }

    fn record(outcomes: &[ExampleOutcome]) -> EvalReplayRecordV1 {
        EvalReplayRecordV1::new(
            suite(),
            None,
            7,
            vec![EvalThinkingBudget::default()],
            Some(EvalModelTargetIdentity::base()),
            Vec::new(),
            Some(sha256_bytes(b"execution")),
            Some(sha256_bytes(b"weights")),
            outcomes,
        )
        .unwrap()
    }

    #[test]
    fn replay_record_round_trips_and_validates_strictly() {
        let outcomes = outcomes();
        let baseline_record = record(&outcomes);
        baseline_record.validate_strict_replay(&outcomes).unwrap();
        let decoded: EvalReplayRecordV1 =
            serde_json::from_slice(&serde_json::to_vec(&baseline_record).unwrap()).unwrap();
        decoded.validate_strict_replay(&outcomes).unwrap();
        assert_eq!(decoded.raw_completions[0].raw_completion_bytes, Some(1));
    }

    #[test]
    fn replay_record_rejects_tampered_raw_completion() {
        let outcomes = outcomes();
        let record = record(&outcomes);
        let mut changed = outcomes.clone();
        changed[0].raw_completion_text = Some("y".into());
        assert!(record.validate(&changed).is_err());
    }

    #[test]
    fn strict_replay_rejects_identity_and_raw_byte_gaps() {
        let outcomes = outcomes();
        let mut missing_identity = record(&outcomes);
        missing_identity.execution_provenance_sha256 = None;
        missing_identity.record_sha256 = missing_identity.compute_record_sha256().unwrap();
        assert!(missing_identity.validate_strict_replay(&outcomes).is_err());

        let mut missing_raw = outcomes.clone();
        missing_raw[0].raw_completion_text = None;
        let record = record(&missing_raw);
        assert!(record.validate_strict_replay(&missing_raw).is_err());
    }

    #[test]
    fn strict_replay_rejects_seed_budget_and_completion_coverage_drift() {
        let outcomes = outcomes();
        let baseline_record = record(&outcomes);

        let mut wrong_seed = outcomes.clone();
        wrong_seed[0].generation_seed = Some(99);
        let wrong_seed_record = record(&wrong_seed);
        assert!(
            wrong_seed_record
                .validate_strict_replay(&wrong_seed)
                .is_err()
        );

        let mut missing_budget = outcomes.clone();
        missing_budget[0].thinking_budget = None;
        let missing_budget_record = record(&missing_budget);
        assert!(
            missing_budget_record
                .validate_strict_replay(&missing_budget)
                .is_err()
        );

        assert!(baseline_record.validate_strict_replay(&[]).is_err());
    }

    #[test]
    fn replay_verdict_validation_rejects_false_match_and_unbound_verdicts() {
        let outcomes = outcomes();
        let record = record(&outcomes);
        let expectation = EvalReplayExpectationV1::new("source".into(), 0, &record);
        let mut verdict = EvalReplayVerdict {
            status: EvalReplayStatus::Matched,
            source_job_id: "source".into(),
            source_run_index: 0,
            expected_record_sha256: record.record_sha256.clone(),
            actual_record_sha256: Some(record.record_sha256.clone()),
            expected_raw_completion_set_sha256: record.raw_completion_set_sha256.clone(),
            actual_raw_completion_set_sha256: Some(record.raw_completion_set_sha256.clone()),
            message: "matched".into(),
        };
        verdict.validate(&expectation).unwrap();
        verdict.actual_record_sha256 = Some(sha256_bytes(b"different"));
        assert!(verdict.validate(&expectation).is_err());
        verdict.actual_record_sha256 = Some(record.record_sha256.clone());
        verdict.source_job_id = "other".into();
        assert!(verdict.validate(&expectation).is_err());
    }

    #[test]
    fn model_target_requires_revision_for_named_adapter() {
        assert!(
            EvalModelTargetIdentity {
                adapter: Some("named".into()),
                adapter_content_sha256: None,
            }
            .validate()
            .is_err()
        );
    }
}
