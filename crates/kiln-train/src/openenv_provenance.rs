//! Corpus-level identity and validation for OpenEnv-backed GRPO data.
//!
//! Rollout provenance identifies one episode. This module lifts those records
//! into a compact training-corpus receipt while enforcing the invariants that
//! make a GRPO group one reproducible OpenEnv task.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    GrpoGroup, OpenEnvEpisodeTerminationV1, OpenEnvRolloutProvenanceV1,
    RolloutBehaviorPolicyIdentityV1,
};

pub const OPENENV_TRAINING_DATA_PROVENANCE_SCHEMA_V1: &str = "kiln.openenv-training-data.v1";

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTerminationCountsV1 {
    pub done: usize,
    pub max_steps: usize,
    pub invalid_model_action: usize,
    pub protocol_error: usize,
}

impl OpenEnvTerminationCountsV1 {
    fn observe(&mut self, termination: OpenEnvEpisodeTerminationV1) -> Result<(), String> {
        let count = match termination {
            OpenEnvEpisodeTerminationV1::Done => &mut self.done,
            OpenEnvEpisodeTerminationV1::MaxSteps => &mut self.max_steps,
            OpenEnvEpisodeTerminationV1::InvalidModelAction => &mut self.invalid_model_action,
            OpenEnvEpisodeTerminationV1::ProtocolError => &mut self.protocol_error,
        };
        *count = count
            .checked_add(1)
            .ok_or_else(|| "OpenEnv termination count overflow".to_string())?;
        Ok(())
    }

    fn total(&self) -> Option<usize> {
        self.done
            .checked_add(self.max_steps)?
            .checked_add(self.invalid_model_action)?
            .checked_add(self.protocol_error)
    }

    fn add_from(&mut self, other: &Self) -> Result<(), String> {
        self.done = self
            .done
            .checked_add(other.done)
            .ok_or_else(|| "OpenEnv done count overflow".to_string())?;
        self.max_steps = self
            .max_steps
            .checked_add(other.max_steps)
            .ok_or_else(|| "OpenEnv max-steps count overflow".to_string())?;
        self.invalid_model_action = self
            .invalid_model_action
            .checked_add(other.invalid_model_action)
            .ok_or_else(|| "OpenEnv invalid-action count overflow".to_string())?;
        self.protocol_error = self
            .protocol_error
            .checked_add(other.protocol_error)
            .ok_or_else(|| "OpenEnv protocol-error count overflow".to_string())?;
        Ok(())
    }
}

/// One protocol endpoint and immutable schema identity represented in a
/// training corpus. Environment-specific reset values are not retained.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingEnvironmentV1 {
    pub environment_name: String,
    pub environment_base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub openapi_version: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub discovery_sha256: Option<String>,
    pub environment_schema_sha256: String,
    pub action_schema_sha256: String,
    pub groups: usize,
    pub rollouts: usize,
    pub total_steps: usize,
    pub terminations: OpenEnvTerminationCountsV1,
}

/// Compact semantic identity for an all-OpenEnv GRPO corpus.
///
/// `group_plan_sha256` binds the ordered endpoint, schema, reset hash, seed,
/// and candidate count for every group. The ordinary training-data digest
/// independently binds every byte, including trajectories and rewards.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvTrainingDataProvenanceV1 {
    schema: String,
    pub groups: usize,
    pub rollouts: usize,
    pub unique_seeds: usize,
    #[serde(serialize_with = "serialize_u64_decimal")]
    pub seed_min: u64,
    #[serde(serialize_with = "serialize_u64_decimal")]
    pub seed_max: u64,
    pub total_steps: usize,
    pub terminations: OpenEnvTerminationCountsV1,
    pub group_plan_sha256: String,
    /// One immutable behavior-policy revision for the complete corpus. Legacy
    /// v1 corpora may omit it; current native collection always includes it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub behavior_policy: Option<RolloutBehaviorPolicyIdentityV1>,
    pub environments: Vec<OpenEnvTrainingEnvironmentV1>,
}

impl<'de> Deserialize<'de> for OpenEnvTrainingDataProvenanceV1 {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Wire {
            schema: String,
            groups: usize,
            rollouts: usize,
            unique_seeds: usize,
            seed_min: String,
            seed_max: String,
            total_steps: usize,
            terminations: OpenEnvTerminationCountsV1,
            group_plan_sha256: String,
            #[serde(default)]
            behavior_policy: Option<RolloutBehaviorPolicyIdentityV1>,
            environments: Vec<OpenEnvTrainingEnvironmentV1>,
        }

        let wire = Wire::deserialize(deserializer)?;
        let provenance = Self {
            schema: wire.schema,
            groups: wire.groups,
            rollouts: wire.rollouts,
            unique_seeds: wire.unique_seeds,
            seed_min: parse_decimal_u64("openenv_training_data.seed_min", &wire.seed_min)
                .map_err(serde::de::Error::custom)?,
            seed_max: parse_decimal_u64("openenv_training_data.seed_max", &wire.seed_max)
                .map_err(serde::de::Error::custom)?,
            total_steps: wire.total_steps,
            terminations: wire.terminations,
            group_plan_sha256: wire.group_plan_sha256,
            behavior_policy: wire.behavior_policy,
            environments: wire.environments,
        };
        provenance.validate().map_err(serde::de::Error::custom)?;
        Ok(provenance)
    }
}

impl OpenEnvTrainingDataProvenanceV1 {
    pub fn schema(&self) -> &str {
        &self.schema
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema != OPENENV_TRAINING_DATA_PROVENANCE_SCHEMA_V1 {
            return Err(format!(
                "unsupported OpenEnv training-data schema {:?}; expected {:?}",
                self.schema, OPENENV_TRAINING_DATA_PROVENANCE_SCHEMA_V1
            ));
        }
        if self.groups == 0 || self.rollouts < self.groups {
            return Err(
                "OpenEnv training data must contain at least one group and one rollout per group"
                    .to_string(),
            );
        }
        if self.unique_seeds == 0
            || self.unique_seeds > self.groups
            || self.seed_min > self.seed_max
        {
            return Err("OpenEnv training data has inconsistent seed statistics".to_string());
        }
        validate_sha256(
            "openenv_training_data.group_plan_sha256",
            &self.group_plan_sha256,
        )?;
        if let Some(behavior_policy) = &self.behavior_policy {
            behavior_policy.validate()?;
        }
        if self.terminations.total() != Some(self.rollouts) {
            return Err(
                "OpenEnv training-data termination counts do not equal rollouts".to_string(),
            );
        }
        if self.environments.is_empty() {
            return Err("OpenEnv training data must identify at least one environment".to_string());
        }
        let mut groups = 0usize;
        let mut rollouts = 0usize;
        let mut total_steps = 0usize;
        let mut terminations = OpenEnvTerminationCountsV1::default();
        let mut previous_url: Option<&str> = None;
        for environment in &self.environments {
            validate_environment(environment)?;
            if previous_url
                .is_some_and(|previous| previous >= environment.environment_base_url.as_str())
            {
                return Err(
                    "OpenEnv training environments must be sorted by unique base URL".to_string(),
                );
            }
            previous_url = Some(&environment.environment_base_url);
            groups = groups
                .checked_add(environment.groups)
                .ok_or_else(|| "OpenEnv environment group count overflow".to_string())?;
            rollouts = rollouts
                .checked_add(environment.rollouts)
                .ok_or_else(|| "OpenEnv environment rollout count overflow".to_string())?;
            total_steps = total_steps
                .checked_add(environment.total_steps)
                .ok_or_else(|| "OpenEnv environment step count overflow".to_string())?;
            terminations.add_from(&environment.terminations)?;
        }
        if groups != self.groups
            || rollouts != self.rollouts
            || total_steps != self.total_steps
            || terminations != self.terminations
        {
            return Err("OpenEnv per-environment totals do not equal corpus totals".to_string());
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct OpenEnvGroupPlanIdentityV1<'a> {
    schema: &'static str,
    group_index: usize,
    environment_name: &'a str,
    environment_base_url: &'a str,
    openapi_version: &'a Option<String>,
    discovery_sha256: &'a Option<String>,
    environment_schema_sha256: &'a str,
    action_schema_sha256: &'a str,
    reset_sha256: &'a str,
    seed: u64,
    rollouts: usize,
}

enum CorpusKind {
    Unknown,
    Ordinary,
    OpenEnv,
}

struct EnvironmentAccumulator {
    environment: OpenEnvTrainingEnvironmentV1,
}

/// Streaming validator shared by HTTP admission and both native GRPO routes.
pub struct OpenEnvTrainingDataAccumulator {
    kind: CorpusKind,
    groups: usize,
    rollouts: usize,
    seeds: BTreeSet<u64>,
    seed_min: u64,
    seed_max: u64,
    total_steps: usize,
    terminations: OpenEnvTerminationCountsV1,
    environments: BTreeMap<String, EnvironmentAccumulator>,
    behavior_policy: Option<Option<RolloutBehaviorPolicyIdentityV1>>,
    plan_hasher: Sha256,
}

impl Default for OpenEnvTrainingDataAccumulator {
    fn default() -> Self {
        let mut plan_hasher = Sha256::new();
        plan_hasher.update(b"kiln.openenv-group-plan.v1\0");
        Self {
            kind: CorpusKind::Unknown,
            groups: 0,
            rollouts: 0,
            seeds: BTreeSet::new(),
            seed_min: u64::MAX,
            seed_max: 0,
            total_steps: 0,
            terminations: OpenEnvTerminationCountsV1::default(),
            environments: BTreeMap::new(),
            behavior_policy: None,
            plan_hasher,
        }
    }
}

impl OpenEnvTrainingDataAccumulator {
    /// Observe one logical GRPO group. `group_index` is a stable, one-based
    /// source index (blank JSONL lines do not consume an index).
    pub fn observe_group(&mut self, group_index: usize, group: &GrpoGroup) -> Result<(), String> {
        if group_index == 0 {
            return Err("OpenEnv training-data group index must be one-based".to_string());
        }
        if group.completions.is_empty() {
            return Err(format!(
                "OpenEnv corpus group {group_index} has no completions"
            ));
        }
        let openenv_count = group
            .completions
            .iter()
            .filter(|completion| completion.openenv.is_some())
            .count();
        if openenv_count == 0 {
            if matches!(self.kind, CorpusKind::OpenEnv) {
                return Err(format!(
                    "GRPO corpus mixes OpenEnv and ordinary groups at group {group_index}"
                ));
            }
            self.kind = CorpusKind::Ordinary;
            return Ok(());
        }
        if openenv_count != group.completions.len() {
            return Err(format!(
                "OpenEnv corpus group {group_index} has provenance on {openenv_count} of {} completions",
                group.completions.len()
            ));
        }
        if matches!(self.kind, CorpusKind::Ordinary) {
            return Err(format!(
                "GRPO corpus mixes ordinary and OpenEnv groups at group {group_index}"
            ));
        }
        self.kind = CorpusKind::OpenEnv;

        let first = group.completions[0]
            .openenv
            .as_ref()
            .expect("OpenEnv count proved provenance exists");
        first.validate()?;
        for (candidate_index, completion) in group.completions.iter().enumerate() {
            let provenance = completion
                .openenv
                .as_ref()
                .expect("OpenEnv count proved provenance exists");
            provenance.validate()?;
            if !same_group_task(first, provenance) {
                return Err(format!(
                    "OpenEnv group {group_index} candidate {} does not share the group's endpoint, schema, reset, and seed",
                    candidate_index + 1
                ));
            }
            if completion.reward != provenance.episode_return {
                return Err(format!(
                    "OpenEnv group {group_index} candidate {} reward {} differs from episode_return {}",
                    candidate_index + 1,
                    completion.reward,
                    provenance.episode_return
                ));
            }
            match &self.behavior_policy {
                None => self.behavior_policy = Some(provenance.behavior_policy.clone()),
                Some(expected) if expected == &provenance.behavior_policy => {}
                Some(_) => {
                    return Err(format!(
                        "OpenEnv corpus behavior policy changed at group {group_index} candidate {}",
                        candidate_index + 1
                    ));
                }
            }
        }

        let identity = OpenEnvGroupPlanIdentityV1 {
            schema: "kiln.openenv-group-plan-item.v1",
            group_index,
            environment_name: &first.environment_name,
            environment_base_url: &first.environment_base_url,
            openapi_version: &first.openapi_version,
            discovery_sha256: &first.discovery_sha256,
            environment_schema_sha256: &first.environment_schema_sha256,
            action_schema_sha256: &first.action_schema_sha256,
            reset_sha256: &first.reset_sha256,
            seed: first.seed,
            rollouts: group.completions.len(),
        };
        let encoded = serde_json::to_vec(&identity)
            .map_err(|error| format!("serialize OpenEnv group plan: {error}"))?;
        self.plan_hasher
            .update((encoded.len() as u64).to_be_bytes());
        self.plan_hasher.update(encoded);

        self.groups = self
            .groups
            .checked_add(1)
            .ok_or_else(|| "OpenEnv group count overflow".to_string())?;
        self.rollouts = self
            .rollouts
            .checked_add(group.completions.len())
            .ok_or_else(|| "OpenEnv rollout count overflow".to_string())?;
        self.seeds.insert(first.seed);
        self.seed_min = self.seed_min.min(first.seed);
        self.seed_max = self.seed_max.max(first.seed);

        let environment = self
            .environments
            .entry(first.environment_base_url.clone())
            .or_insert_with(|| EnvironmentAccumulator {
                environment: OpenEnvTrainingEnvironmentV1 {
                    environment_name: first.environment_name.clone(),
                    environment_base_url: first.environment_base_url.clone(),
                    openapi_version: first.openapi_version.clone(),
                    discovery_sha256: first.discovery_sha256.clone(),
                    environment_schema_sha256: first.environment_schema_sha256.clone(),
                    action_schema_sha256: first.action_schema_sha256.clone(),
                    groups: 0,
                    rollouts: 0,
                    total_steps: 0,
                    terminations: OpenEnvTerminationCountsV1::default(),
                },
            });
        if environment.environment.environment_name != first.environment_name
            || environment.environment.openapi_version != first.openapi_version
            || environment.environment.discovery_sha256 != first.discovery_sha256
            || environment.environment.environment_schema_sha256 != first.environment_schema_sha256
            || environment.environment.action_schema_sha256 != first.action_schema_sha256
        {
            return Err(format!(
                "OpenEnv endpoint {:?} changed name or schema within the training corpus",
                first.environment_base_url
            ));
        }
        environment.environment.groups = environment
            .environment
            .groups
            .checked_add(1)
            .ok_or_else(|| "OpenEnv environment group count overflow".to_string())?;
        environment.environment.rollouts = environment
            .environment
            .rollouts
            .checked_add(group.completions.len())
            .ok_or_else(|| "OpenEnv environment rollout count overflow".to_string())?;
        for completion in &group.completions {
            let provenance = completion
                .openenv
                .as_ref()
                .expect("validated OpenEnv completion");
            self.total_steps = self
                .total_steps
                .checked_add(provenance.steps)
                .ok_or_else(|| "OpenEnv step count overflow".to_string())?;
            environment.environment.total_steps = environment
                .environment
                .total_steps
                .checked_add(provenance.steps)
                .ok_or_else(|| "OpenEnv environment step count overflow".to_string())?;
            self.terminations.observe(provenance.termination)?;
            environment
                .environment
                .terminations
                .observe(provenance.termination)?;
        }
        Ok(())
    }

    pub fn finish(self) -> Result<Option<OpenEnvTrainingDataProvenanceV1>, String> {
        if !matches!(self.kind, CorpusKind::OpenEnv) {
            return Ok(None);
        }
        let digest: [u8; 32] = self.plan_hasher.finalize().into();
        let provenance = OpenEnvTrainingDataProvenanceV1 {
            schema: OPENENV_TRAINING_DATA_PROVENANCE_SCHEMA_V1.to_string(),
            groups: self.groups,
            rollouts: self.rollouts,
            unique_seeds: self.seeds.len(),
            seed_min: self.seed_min,
            seed_max: self.seed_max,
            total_steps: self.total_steps,
            terminations: self.terminations,
            group_plan_sha256: format_sha256(&digest),
            behavior_policy: self.behavior_policy.flatten(),
            environments: self
                .environments
                .into_values()
                .map(|entry| entry.environment)
                .collect(),
        };
        provenance.validate()?;
        Ok(Some(provenance))
    }
}

pub fn openenv_training_data_provenance(
    groups: &[GrpoGroup],
) -> Result<Option<OpenEnvTrainingDataProvenanceV1>, String> {
    let mut accumulator = OpenEnvTrainingDataAccumulator::default();
    for (index, group) in groups.iter().enumerate() {
        accumulator.observe_group(index + 1, group)?;
    }
    accumulator.finish()
}

fn same_group_task(left: &OpenEnvRolloutProvenanceV1, right: &OpenEnvRolloutProvenanceV1) -> bool {
    left.environment_name == right.environment_name
        && left.environment_base_url == right.environment_base_url
        && left.openapi_version == right.openapi_version
        && left.discovery_sha256 == right.discovery_sha256
        && left.environment_schema_sha256 == right.environment_schema_sha256
        && left.action_schema_sha256 == right.action_schema_sha256
        && left.reset_sha256 == right.reset_sha256
        && left.seed == right.seed
}

fn validate_environment(environment: &OpenEnvTrainingEnvironmentV1) -> Result<(), String> {
    validate_identity_text(
        "openenv_training_data.environment_name",
        &environment.environment_name,
        256,
    )?;
    validate_identity_text(
        "openenv_training_data.environment_base_url",
        &environment.environment_base_url,
        2048,
    )?;
    if let Some(version) = environment.openapi_version.as_deref() {
        validate_identity_text("openenv_training_data.openapi_version", version, 256)?;
    }
    if let Some(discovery_sha256) = environment.discovery_sha256.as_deref() {
        validate_sha256("openenv_training_data.discovery_sha256", discovery_sha256)?;
    }
    validate_sha256(
        "openenv_training_data.environment_schema_sha256",
        &environment.environment_schema_sha256,
    )?;
    validate_sha256(
        "openenv_training_data.action_schema_sha256",
        &environment.action_schema_sha256,
    )?;
    if environment.groups == 0 || environment.rollouts < environment.groups {
        return Err(
            "OpenEnv environment totals require at least one rollout per group".to_string(),
        );
    }
    if environment.terminations.total() != Some(environment.rollouts) {
        return Err("OpenEnv environment termination counts do not equal rollouts".to_string());
    }
    Ok(())
}

fn validate_identity_text(field: &str, value: &str, max_bytes: usize) -> Result<(), String> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > max_bytes
        || value.chars().any(char::is_control)
    {
        Err(format!(
            "{field} must be non-empty, trimmed, control-free, and at most {max_bytes} bytes"
        ))
    } else {
        Ok(())
    }
}

fn serialize_u64_decimal<S>(value: &u64, serializer: S) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    serializer.serialize_str(&value.to_string())
}

fn parse_decimal_u64(field: &str, value: &str) -> Result<u64, String> {
    if value.is_empty()
        || (value.len() > 1 && value.starts_with('0'))
        || !value.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(format!(
            "{field} must be a canonical unsigned 64-bit decimal string"
        ));
    }
    value
        .parse::<u64>()
        .map_err(|_| format!("{field} exceeds the unsigned 64-bit range"))
}

fn validate_sha256(field: &str, value: &str) -> Result<(), String> {
    let valid = value.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    });
    if valid {
        Ok(())
    } else {
        Err(format!(
            "{field} must use the sha256:<64 lowercase hex> form"
        ))
    }
}

fn format_sha256(digest: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut value = String::with_capacity(71);
    value.push_str("sha256:");
    for byte in digest {
        value.push(HEX[(byte >> 4) as usize] as char);
        value.push(HEX[(byte & 0x0f) as usize] as char);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ChatMessage, ScoredCompletion};

    fn hash(byte: char) -> String {
        format!("sha256:{}", byte.to_string().repeat(64))
    }

    fn behavior_policy(byte: char) -> RolloutBehaviorPolicyIdentityV1 {
        RolloutBehaviorPolicyIdentityV1 {
            served_model_id: "test-model".to_string(),
            base_model_sha256: hash(byte),
            adapter: None,
            inference_config_sha256: hash('d'),
            implementation: "kiln/test".to_string(),
        }
    }

    fn completion(seed: u64, reward: f64) -> ScoredCompletion {
        let provenance = OpenEnvRolloutProvenanceV1::new(
            "math-env",
            "http://env.test",
            Some("3.1.0".to_string()),
            hash('d'),
            hash('a'),
            hash('b'),
            hash('c'),
            seed,
            2,
            reward,
            true,
            OpenEnvEpisodeTerminationV1::Done,
            None,
        )
        .unwrap()
        .with_behavior_policy(behavior_policy('e'))
        .unwrap();
        ScoredCompletion::legacy("answer".to_string(), reward).with_openenv(provenance)
    }

    fn group(seed: u64) -> GrpoGroup {
        GrpoGroup {
            messages: vec![ChatMessage::new("user", "solve")],
            completions: vec![completion(seed, 1.0), completion(seed, 1.0)],
        }
    }

    #[test]
    fn corpus_provenance_is_stable_and_aggregated() {
        let groups = vec![group(7), group(8)];
        let first = openenv_training_data_provenance(&groups).unwrap().unwrap();
        let second = openenv_training_data_provenance(&groups).unwrap().unwrap();
        assert_eq!(first, second);
        assert_eq!(first.schema(), OPENENV_TRAINING_DATA_PROVENANCE_SCHEMA_V1);
        assert_eq!(first.groups, 2);
        assert_eq!(first.rollouts, 4);
        assert_eq!(first.unique_seeds, 2);
        assert_eq!(first.total_steps, 8);
        assert_eq!(first.terminations.done, 4);
        assert_eq!(first.environments.len(), 1);
        assert_eq!(first.environments[0].discovery_sha256, Some(hash('d')));
        assert_eq!(first.behavior_policy, Some(behavior_policy('e')));
    }

    #[test]
    fn corpus_seed_range_round_trips_as_exact_decimal_text() {
        let provenance = openenv_training_data_provenance(&[group(u64::MAX)])
            .unwrap()
            .unwrap();
        let encoded = serde_json::to_value(&provenance).unwrap();
        assert_eq!(encoded["seed_min"], u64::MAX.to_string());
        assert_eq!(encoded["seed_max"], u64::MAX.to_string());
        assert_eq!(
            serde_json::from_value::<OpenEnvTrainingDataProvenanceV1>(encoded).unwrap(),
            provenance
        );
    }

    #[test]
    fn corpus_rejects_partial_provenance_and_reward_drift() {
        let mut partial = group(7);
        partial.completions[1].openenv = None;
        assert!(
            openenv_training_data_provenance(&[partial])
                .unwrap_err()
                .contains("provenance on 1 of 2")
        );

        let mut drifted = group(7);
        drifted.completions[1].reward = 0.5;
        assert!(
            openenv_training_data_provenance(&[drifted])
                .unwrap_err()
                .contains("differs from episode_return")
        );
    }

    #[test]
    fn corpus_rejects_behavior_policy_drift_or_missing_identity() {
        let mut changed = group(7);
        changed.completions[1]
            .openenv
            .as_mut()
            .unwrap()
            .behavior_policy = Some(behavior_policy('f'));
        let error = openenv_training_data_provenance(&[changed]).unwrap_err();
        assert!(error.contains("behavior policy changed"), "{error}");

        let mut missing = group(7);
        missing.completions[1]
            .openenv
            .as_mut()
            .unwrap()
            .behavior_policy = None;
        let error = openenv_training_data_provenance(&[missing]).unwrap_err();
        assert!(error.contains("behavior policy changed"), "{error}");
    }

    #[test]
    fn corpus_rejects_mixed_groups_and_group_task_drift() {
        let ordinary = GrpoGroup {
            messages: vec![ChatMessage::new("user", "ordinary")],
            completions: vec![ScoredCompletion::legacy("x".to_string(), 1.0)],
        };
        assert!(
            openenv_training_data_provenance(&[ordinary, group(7)])
                .unwrap_err()
                .contains("mixes ordinary and OpenEnv")
        );

        let mut drifted = group(7);
        drifted.completions[1].openenv.as_mut().unwrap().seed = 8;
        assert!(
            openenv_training_data_provenance(&[drifted])
                .unwrap_err()
                .contains("does not share")
        );

        let mut discovery_drifted = group(7);
        discovery_drifted.completions[1]
            .openenv
            .as_mut()
            .unwrap()
            .discovery_sha256 = Some(hash('f'));
        assert!(
            openenv_training_data_provenance(&[discovery_drifted])
                .unwrap_err()
                .contains("does not share")
        );
    }
}
