use serde::Serialize;

use crate::state::{AppState, ModelBackend};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum RocmGraphUnavailableReason {
    BackendWithoutGraphRunner,
    ModelRunnerBusy,
    ModelRunnerLockPoisoned,
    GraphRunnerBusy,
    GraphRunnerLockPoisoned,
}

impl RocmGraphUnavailableReason {
    pub(crate) const ALL: [Self; 5] = [
        Self::BackendWithoutGraphRunner,
        Self::ModelRunnerBusy,
        Self::ModelRunnerLockPoisoned,
        Self::GraphRunnerBusy,
        Self::GraphRunnerLockPoisoned,
    ];

    pub(crate) const fn is_busy(self) -> bool {
        matches!(self, Self::ModelRunnerBusy | Self::GraphRunnerBusy)
    }

    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::BackendWithoutGraphRunner => "backend_without_graph_runner",
            Self::ModelRunnerBusy => "model_runner_busy",
            Self::ModelRunnerLockPoisoned => "model_runner_lock_poisoned",
            Self::GraphRunnerBusy => "graph_runner_busy",
            Self::GraphRunnerLockPoisoned => "graph_runner_lock_poisoned",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct RocmGraphObservation {
    pub(crate) stats: Option<kiln_model::RocmGraphStats>,
    pub(crate) stats_unavailable_reason: Option<RocmGraphUnavailableReason>,
    pub(crate) telemetry: Option<kiln_model::RocmGraphLiveTelemetry>,
    pub(crate) telemetry_unavailable_reason: Option<RocmGraphUnavailableReason>,
}

pub(crate) fn observe_rocm_graphs(state: &AppState) -> RocmGraphObservation {
    match state.backend.as_ref() {
        ModelBackend::Mock { .. } => RocmGraphObservation {
            stats: None,
            stats_unavailable_reason: Some(RocmGraphUnavailableReason::BackendWithoutGraphRunner),
            telemetry: None,
            telemetry_unavailable_reason: Some(
                RocmGraphUnavailableReason::BackendWithoutGraphRunner,
            ),
        },
        ModelBackend::Real {
            runner,
            rocm_graph_telemetry,
            ..
        } => {
            let telemetry = Some(rocm_graph_telemetry.snapshot());
            match runner.try_read() {
                Ok(runner) => match runner.rocm_graph_stats() {
                    Ok(stats) => RocmGraphObservation {
                        stats: Some(stats),
                        stats_unavailable_reason: None,
                        telemetry,
                        telemetry_unavailable_reason: None,
                    },
                    Err(kiln_model::RocmGraphStatsUnavailable::Busy) => RocmGraphObservation {
                        stats: None,
                        stats_unavailable_reason: Some(RocmGraphUnavailableReason::GraphRunnerBusy),
                        telemetry,
                        telemetry_unavailable_reason: None,
                    },
                    Err(kiln_model::RocmGraphStatsUnavailable::Poisoned) => RocmGraphObservation {
                        stats: None,
                        stats_unavailable_reason: Some(
                            RocmGraphUnavailableReason::GraphRunnerLockPoisoned,
                        ),
                        telemetry,
                        telemetry_unavailable_reason: None,
                    },
                },
                Err(std::sync::TryLockError::WouldBlock) => RocmGraphObservation {
                    stats: None,
                    stats_unavailable_reason: Some(RocmGraphUnavailableReason::ModelRunnerBusy),
                    telemetry,
                    telemetry_unavailable_reason: None,
                },
                Err(std::sync::TryLockError::Poisoned(_)) => RocmGraphObservation {
                    stats: None,
                    stats_unavailable_reason: Some(
                        RocmGraphUnavailableReason::ModelRunnerLockPoisoned,
                    ),
                    telemetry,
                    telemetry_unavailable_reason: None,
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::RocmGraphUnavailableReason;

    #[test]
    fn only_lock_contention_is_busy() {
        assert!(RocmGraphUnavailableReason::ModelRunnerBusy.is_busy());
        assert!(RocmGraphUnavailableReason::GraphRunnerBusy.is_busy());
        assert!(!RocmGraphUnavailableReason::BackendWithoutGraphRunner.is_busy());
        assert!(!RocmGraphUnavailableReason::ModelRunnerLockPoisoned.is_busy());
        assert!(!RocmGraphUnavailableReason::GraphRunnerLockPoisoned.is_busy());
    }

    #[test]
    fn prometheus_reason_labels_match_the_serialized_api_contract() {
        for reason in RocmGraphUnavailableReason::ALL {
            assert_eq!(
                serde_json::to_value(reason).unwrap(),
                serde_json::Value::String(reason.as_str().to_string())
            );
        }
    }
}
