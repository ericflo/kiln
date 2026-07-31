//! Stable, bounded failure semantics for persisted OpenEnv workflows.

use anyhow::Error;
use kiln_openenv::{OpenEnvClientError, OpenEnvErrorCode};
use serde::{Deserialize, Serialize};

use super::OpenEnvRunState;
use crate::openenv_replay::OpenEnvCapacityTimeout;

pub const OPENENV_RUN_FAILURE_SCHEMA_V1: &str = "kiln.openenv-run-failure.v1";
const MAX_FAILURE_MESSAGE_BYTES: usize = 4 * 1024;

#[derive(Debug, thiserror::Error)]
#[error("OpenEnv trainer evidence validation failed: {source:#}")]
struct OpenEnvTrainingEvidenceFailure {
    #[source]
    source: Error,
}

pub(crate) fn training_evidence_failure(source: Error) -> Error {
    Error::new(OpenEnvTrainingEvidenceFailure { source })
}

/// The workflow boundary that owned the operation when a run failed.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvRunFailureStage {
    Restoration,
    Admission,
    Discovery,
    Collection,
    IdentityVerification,
    ArtifactPublication,
    TrainingSubmission,
    Training,
    PostEvaluation,
    EnvironmentEvaluation,
    Orchestration,
}

impl OpenEnvRunFailureStage {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::Restoration => "restoration",
            Self::Admission => "admission",
            Self::Discovery => "discovery",
            Self::Collection => "collection",
            Self::IdentityVerification => "identity_verification",
            Self::ArtifactPublication => "artifact_publication",
            Self::TrainingSubmission => "training_submission",
            Self::Training => "training",
            Self::PostEvaluation => "post_evaluation",
            Self::EnvironmentEvaluation => "environment_evaluation",
            Self::Orchestration => "orchestration",
        }
    }
}

/// Closed, automation-safe reason for a terminal OpenEnv workflow failure.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvRunFailureCode {
    RunAdmissionFailed,
    RunInterrupted,
    PersistedContractInvalid,
    EnvironmentUnavailable,
    EnvironmentCapacityExhausted,
    EnvironmentProtocolError,
    EnvironmentIdentityChanged,
    CollectionFailed,
    ArtifactPublicationFailed,
    TrainingSubmissionFailed,
    TrainingFailed,
    TrainingEvidenceInvalid,
    PostEvaluationFailed,
    EnvironmentEvaluationFailed,
    InternalError,
}

impl OpenEnvRunFailureCode {
    pub const fn as_label(self) -> &'static str {
        match self {
            Self::RunAdmissionFailed => "run_admission_failed",
            Self::RunInterrupted => "run_interrupted",
            Self::PersistedContractInvalid => "persisted_contract_invalid",
            Self::EnvironmentUnavailable => "environment_unavailable",
            Self::EnvironmentCapacityExhausted => "environment_capacity_exhausted",
            Self::EnvironmentProtocolError => "environment_protocol_error",
            Self::EnvironmentIdentityChanged => "environment_identity_changed",
            Self::CollectionFailed => "collection_failed",
            Self::ArtifactPublicationFailed => "artifact_publication_failed",
            Self::TrainingSubmissionFailed => "training_submission_failed",
            Self::TrainingFailed => "training_failed",
            Self::TrainingEvidenceInvalid => "training_evidence_invalid",
            Self::PostEvaluationFailed => "post_evaluation_failed",
            Self::EnvironmentEvaluationFailed => "environment_evaluation_failed",
            Self::InternalError => "internal_error",
        }
    }
}

/// Self-contained terminal diagnosis retained with an OpenEnv run.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvRunFailure {
    pub schema: String,
    pub code: OpenEnvRunFailureCode,
    pub stage: OpenEnvRunFailureStage,
    pub retryable: bool,
    pub message: String,
    pub hint: String,
    pub occurred_unix_ms: u64,
    /// Exact OpenEnv wire error when the peer supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub protocol_code: Option<OpenEnvErrorCode>,
    /// Exact upstream discovery status when an HTTP response supplied one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub http_status: Option<u16>,
}

impl OpenEnvRunFailure {
    pub(crate) fn explicit(
        code: OpenEnvRunFailureCode,
        stage: OpenEnvRunFailureStage,
        retryable: bool,
        message: impl AsRef<str>,
        hint: impl Into<String>,
        occurred_unix_ms: u64,
    ) -> Self {
        Self {
            schema: OPENENV_RUN_FAILURE_SCHEMA_V1.to_string(),
            code,
            stage,
            retryable,
            message: bounded_message(message.as_ref()),
            hint: hint.into(),
            occurred_unix_ms,
            protocol_code: None,
            http_status: None,
        }
    }

    pub(crate) fn from_error(
        state: OpenEnvRunState,
        collection_complete: bool,
        error: &Error,
        occurred_unix_ms: u64,
    ) -> Self {
        let stage = failure_stage(state, collection_complete);
        let message = format!("{error:#}");
        if error
            .downcast_ref::<OpenEnvTrainingEvidenceFailure>()
            .is_some()
        {
            return Self::explicit(
                OpenEnvRunFailureCode::TrainingEvidenceInvalid,
                OpenEnvRunFailureStage::Training,
                false,
                message,
                "Inspect the retained trainer receipt, adapter manifest, corpus digest, and OpenEnv semantic lineage before starting another run.",
                occurred_unix_ms,
            );
        }
        if error.downcast_ref::<OpenEnvCapacityTimeout>().is_some() {
            return Self::explicit(
                OpenEnvRunFailureCode::EnvironmentCapacityExhausted,
                stage,
                true,
                message,
                "Retry after environment capacity becomes available, or increase capacity_wait_seconds.",
                occurred_unix_ms,
            );
        }
        if let Some(client_error) = error.downcast_ref::<OpenEnvClientError>() {
            return client_failure(stage, client_error, message, occurred_unix_ms);
        }
        let (code, retryable, hint) = fallback_semantics(stage);
        Self::explicit(code, stage, retryable, message, hint, occurred_unix_ms)
    }
}

fn failure_stage(state: OpenEnvRunState, collection_complete: bool) -> OpenEnvRunFailureStage {
    match state {
        OpenEnvRunState::Queued => OpenEnvRunFailureStage::Admission,
        OpenEnvRunState::Discovering => OpenEnvRunFailureStage::Discovery,
        OpenEnvRunState::Collecting if collection_complete => {
            OpenEnvRunFailureStage::ArtifactPublication
        }
        OpenEnvRunState::Collecting => OpenEnvRunFailureStage::Collection,
        OpenEnvRunState::Revalidating => OpenEnvRunFailureStage::IdentityVerification,
        OpenEnvRunState::Submitting => OpenEnvRunFailureStage::TrainingSubmission,
        OpenEnvRunState::TrainingQueued | OpenEnvRunState::TrainingRunning => {
            OpenEnvRunFailureStage::Training
        }
        OpenEnvRunState::PostEvaluating => OpenEnvRunFailureStage::PostEvaluation,
        OpenEnvRunState::EnvironmentEvaluating => OpenEnvRunFailureStage::EnvironmentEvaluation,
        OpenEnvRunState::RolloutReady
        | OpenEnvRunState::Completed
        | OpenEnvRunState::Failed
        | OpenEnvRunState::Cancelled => OpenEnvRunFailureStage::Orchestration,
    }
}

fn client_failure(
    stage: OpenEnvRunFailureStage,
    error: &OpenEnvClientError,
    message: String,
    occurred_unix_ms: u64,
) -> OpenEnvRunFailure {
    let (code, retryable, hint, protocol_code, http_status) = match error {
        OpenEnvClientError::EnvironmentIdentityChanged { .. } => (
            OpenEnvRunFailureCode::EnvironmentIdentityChanged,
            true,
            "Stabilize or pin the OpenEnv deployment, inspect it again, and submit a new run; mixed-identity episodes are never published.",
            None,
            None,
        ),
        OpenEnvClientError::Protocol(protocol)
            if protocol.code == OpenEnvErrorCode::CapacityReached =>
        {
            (
                OpenEnvRunFailureCode::EnvironmentCapacityExhausted,
                true,
                "Retry after environment capacity becomes available, or increase capacity_wait_seconds.",
                Some(protocol.code),
                None,
            )
        }
        OpenEnvClientError::Protocol(protocol) => (
            OpenEnvRunFailureCode::EnvironmentProtocolError,
            false,
            "Inspect the retained message and the environment's advertised schemas before submitting a new run.",
            Some(protocol.code),
            None,
        ),
        OpenEnvClientError::HttpStatus { status, .. } => {
            let retryable = matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504);
            (
                OpenEnvRunFailureCode::EnvironmentUnavailable,
                retryable,
                if retryable {
                    "Retry with a new run after the OpenEnv HTTP endpoint recovers."
                } else {
                    "Check the OpenEnv URL, credential, and discovery endpoint before submitting a new run."
                },
                None,
                Some(status.as_u16()),
            )
        }
        OpenEnvClientError::AuthenticatedWebSocketStatus(status) => {
            let retryable = matches!(status.as_u16(), 408 | 425 | 429 | 500 | 502 | 503 | 504);
            (
                OpenEnvRunFailureCode::EnvironmentUnavailable,
                retryable,
                if retryable {
                    "Retry with a new run after the authenticated OpenEnv endpoint recovers."
                } else {
                    "Check the origin-scoped OpenEnv credential and endpoint policy before submitting a new run."
                },
                None,
                Some(status.as_u16()),
            )
        }
        OpenEnvClientError::Http(_)
        | OpenEnvClientError::WebSocket(_)
        | OpenEnvClientError::Timeout(_)
        | OpenEnvClientError::Closed => (
            OpenEnvRunFailureCode::EnvironmentUnavailable,
            true,
            "Check environment reachability and retry with a new run; retained OpenEnv episodes cannot be resumed.",
            None,
            error.http_status_code(),
        ),
        OpenEnvClientError::InvalidBaseUrl(_)
        | OpenEnvClientError::InvalidCredential
        | OpenEnvClientError::InsecureCredentialTransport
        | OpenEnvClientError::InvalidTaskSelector { .. }
        | OpenEnvClientError::InvalidTaskPageLimit { .. }
        | OpenEnvClientError::TaskEnvironmentRequired { .. }
        | OpenEnvClientError::UnknownTaskEnvironment { .. }
        | OpenEnvClientError::UnknownTaskSplit { .. } => (
            OpenEnvRunFailureCode::PersistedContractInvalid,
            false,
            "Correct the OpenEnv request or credential configuration before submitting a new run.",
            None,
            None,
        ),
        OpenEnvClientError::HttpBodyTooLarge { .. }
        | OpenEnvClientError::TaskCollectionTooLarge { .. }
        | OpenEnvClientError::InvalidTaskCount(_)
        | OpenEnvClientError::InvalidTaskPage { .. }
        | OpenEnvClientError::CredentialReflected
        | OpenEnvClientError::BinaryFrame
        | OpenEnvClientError::UnsolicitedApplicationMessage
        | OpenEnvClientError::MessageTooLarge(_)
        | OpenEnvClientError::InvalidMessage(_)
        | OpenEnvClientError::UnexpectedResponse { .. } => (
            OpenEnvRunFailureCode::EnvironmentProtocolError,
            false,
            "Inspect the environment's advertised protocol and schemas before submitting a new run.",
            None,
            None,
        ),
    };
    let mut failure =
        OpenEnvRunFailure::explicit(code, stage, retryable, message, hint, occurred_unix_ms);
    failure.protocol_code = protocol_code;
    failure.http_status = http_status;
    failure
}

fn fallback_semantics(
    stage: OpenEnvRunFailureStage,
) -> (OpenEnvRunFailureCode, bool, &'static str) {
    match stage {
        OpenEnvRunFailureStage::Restoration => (
            OpenEnvRunFailureCode::RunInterrupted,
            true,
            "Submit a new run; OpenEnv sessions and trainer ownership cannot be assumed resumable after restart.",
        ),
        OpenEnvRunFailureStage::Admission => (
            OpenEnvRunFailureCode::RunAdmissionFailed,
            true,
            "Inspect server capacity and submit a new run after admission recovers.",
        ),
        OpenEnvRunFailureStage::Discovery => (
            OpenEnvRunFailureCode::EnvironmentUnavailable,
            true,
            "Check OpenEnv discovery reachability and submit a new run.",
        ),
        OpenEnvRunFailureStage::Collection => (
            OpenEnvRunFailureCode::CollectionFailed,
            true,
            "Inspect the environment, policy, and retained progress, then submit a new run; episodes cannot be resumed.",
        ),
        OpenEnvRunFailureStage::IdentityVerification => (
            OpenEnvRunFailureCode::EnvironmentUnavailable,
            true,
            "Restore a stable OpenEnv discovery endpoint and submit a new run; collected episodes are never published without final identity verification.",
        ),
        OpenEnvRunFailureStage::ArtifactPublication => (
            OpenEnvRunFailureCode::ArtifactPublicationFailed,
            false,
            "Inspect artifact storage and disk capacity. The failed run is terminal and publishes no partial bundle.",
        ),
        OpenEnvRunFailureStage::TrainingSubmission => (
            OpenEnvRunFailureCode::TrainingSubmissionFailed,
            true,
            "Inspect native GRPO admission and capacity before submitting a new run.",
        ),
        OpenEnvRunFailureStage::Training => (
            OpenEnvRunFailureCode::TrainingFailed,
            false,
            "Inspect the projected trainer status and retained corpus evidence before starting another training run.",
        ),
        OpenEnvRunFailureStage::PostEvaluation => (
            OpenEnvRunFailureCode::PostEvaluationFailed,
            false,
            "Inspect linked evaluation status and suite configuration before starting another training run.",
        ),
        OpenEnvRunFailureStage::EnvironmentEvaluation => (
            OpenEnvRunFailureCode::EnvironmentEvaluationFailed,
            true,
            "Inspect the paired held-out evaluation verdict and environment availability before submitting a new run.",
        ),
        OpenEnvRunFailureStage::Orchestration => (
            OpenEnvRunFailureCode::InternalError,
            false,
            "Inspect server logs and the retained run status before retrying.",
        ),
    }
}

fn bounded_message(message: &str) -> String {
    if message.len() <= MAX_FAILURE_MESSAGE_BYTES {
        return message.to_string();
    }
    let mut end = MAX_FAILURE_MESSAGE_BYTES.saturating_sub(3);
    while !message.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}...", &message[..end])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_protocol_capacity_with_wire_evidence() {
        let error = anyhow::Error::new(OpenEnvClientError::Protocol(
            kiln_openenv::OpenEnvProtocolError {
                code: OpenEnvErrorCode::CapacityReached,
                message: "full".into(),
                errors: None,
                active_sessions: Some(2),
                max_sessions: Some(2),
                factory_name: None,
            },
        ))
        .context("reset environment");
        let failure = OpenEnvRunFailure::from_error(OpenEnvRunState::Collecting, false, &error, 17);
        assert_eq!(
            failure.code,
            OpenEnvRunFailureCode::EnvironmentCapacityExhausted
        );
        assert_eq!(failure.stage, OpenEnvRunFailureStage::Collection);
        assert_eq!(
            failure.protocol_code,
            Some(OpenEnvErrorCode::CapacityReached)
        );
        assert!(failure.retryable);
    }

    #[test]
    fn completed_collection_failure_is_artifact_publication() {
        let failure = OpenEnvRunFailure::from_error(
            OpenEnvRunState::Collecting,
            true,
            &anyhow::anyhow!("disk full"),
            19,
        );
        assert_eq!(
            failure.code,
            OpenEnvRunFailureCode::ArtifactPublicationFailed
        );
        assert_eq!(failure.stage, OpenEnvRunFailureStage::ArtifactPublication);
        assert!(!failure.retryable);
    }

    #[test]
    fn training_evidence_has_a_distinct_non_retryable_code() {
        let error = training_evidence_failure(anyhow::anyhow!("receipt digest drift"));
        let failure =
            OpenEnvRunFailure::from_error(OpenEnvRunState::TrainingRunning, false, &error, 21);
        assert_eq!(failure.code, OpenEnvRunFailureCode::TrainingEvidenceInvalid);
        assert_eq!(failure.stage, OpenEnvRunFailureStage::Training);
        assert!(!failure.retryable);
    }

    #[test]
    fn environment_identity_drift_has_a_distinct_revalidation_failure() {
        let error = anyhow::Error::new(OpenEnvClientError::EnvironmentIdentityChanged {
            endpoint: "https://environment.example/openenv".into(),
            expected_schema_sha256: format!("sha256:{}", "a".repeat(64)),
            actual_schema_sha256: format!("sha256:{}", "b".repeat(64)),
            changed_fields: vec!["identity.metadata", "schema.action"],
        });
        let failure =
            OpenEnvRunFailure::from_error(OpenEnvRunState::Revalidating, false, &error, 22);
        assert_eq!(
            failure.code,
            OpenEnvRunFailureCode::EnvironmentIdentityChanged
        );
        assert_eq!(failure.stage, OpenEnvRunFailureStage::IdentityVerification);
        assert!(failure.retryable);
        assert!(failure.hint.contains("mixed-identity episodes"));
        assert_eq!(failure.protocol_code, None);
        assert_eq!(failure.http_status, None);
    }

    #[test]
    fn authenticated_websocket_status_is_retained_without_a_body() {
        let error = anyhow::Error::new(OpenEnvClientError::AuthenticatedWebSocketStatus(
            reqwest::StatusCode::UNAUTHORIZED,
        ));
        let failure = OpenEnvRunFailure::from_error(OpenEnvRunState::Collecting, false, &error, 22);
        assert_eq!(failure.code, OpenEnvRunFailureCode::EnvironmentUnavailable);
        assert_eq!(failure.http_status, Some(401));
        assert!(!failure.retryable);
    }

    #[test]
    fn bounds_multibyte_failure_message() {
        let failure = OpenEnvRunFailure::explicit(
            OpenEnvRunFailureCode::InternalError,
            OpenEnvRunFailureStage::Orchestration,
            false,
            "🧯".repeat(2_000),
            "inspect logs",
            23,
        );
        assert!(failure.message.len() <= MAX_FAILURE_MESSAGE_BYTES);
        assert!(failure.message.ends_with("..."));
    }
}
