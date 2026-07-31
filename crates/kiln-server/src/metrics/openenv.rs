//! Fixed-cardinality OpenEnv rollout-quality telemetry.

use std::sync::atomic::Ordering;

use crate::openenv_cli::OpenEnvRolloutStats;

use super::{Metrics, prom_counter, push_line};

const TERMINATIONS: [&str; 4] = [
    "done",
    "max_steps",
    "invalid_model_action",
    "protocol_error",
];

impl Metrics {
    /// Record one artifact-published OpenEnv collection. Every value comes
    /// from the same retained summary exposed through run status; no endpoint,
    /// environment, adapter, or run identifier becomes a metric label.
    pub(crate) fn record_openenv_rollout_stats(&self, stats: &OpenEnvRolloutStats) {
        let termination_counts = [
            stats.done_count,
            stats.max_steps_count,
            stats.invalid_model_action_count,
            stats.protocol_error_count,
        ];
        let episodes = termination_counts
            .iter()
            .fold(0usize, |total, count| total.saturating_add(*count));
        self.openenv_episodes_collected.fetch_add(
            u64::try_from(episodes).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        for (counter, count) in self
            .openenv_episode_terminations
            .iter()
            .zip(termination_counts)
        {
            counter.fetch_add(u64::try_from(count).unwrap_or(u64::MAX), Ordering::Relaxed);
        }
        self.openenv_recoverable_protocol_errors.fetch_add(
            u64::try_from(stats.recoverable_protocol_error_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.openenv_capacity_retries.fetch_add(
            u64::try_from(stats.capacity_retry_count).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.openenv_environment_steps.fetch_add(
            u64::try_from(stats.total_environment_steps).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        self.openenv_model_tokens.fetch_add(
            u64::try_from(stats.total_model_tokens).unwrap_or(u64::MAX),
            Ordering::Relaxed,
        );
        let latency_us = (stats.mean_model_latency_ms * episodes as f64 * 1_000.0)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64;
        self.openenv_model_latency_us_total
            .fetch_add(latency_us, Ordering::Relaxed);
    }
}

pub(super) fn render_rollout_metrics(metrics: &Metrics, out: &mut String) {
    out.push_str(
        "# HELP kiln_openenv_episode_terminations_total Artifact-published OpenEnv episodes by closed terminal outcome.\n",
    );
    out.push_str("# TYPE kiln_openenv_episode_terminations_total counter\n");
    for (index, termination) in TERMINATIONS.iter().enumerate() {
        prom_counter(
            out,
            "kiln_openenv_episode_terminations_total",
            "termination",
            termination,
            metrics.openenv_episode_terminations[index].load(Ordering::Relaxed),
        );
    }
    for (name, help, value) in [
        (
            "kiln_openenv_recoverable_protocol_errors_total",
            "Recoverable OpenEnv protocol errors retained as corrective observation turns.",
            metrics
                .openenv_recoverable_protocol_errors
                .load(Ordering::Relaxed),
        ),
        (
            "kiln_openenv_capacity_retries_total",
            "Fresh OpenEnv sessions opened after a capacity-reached response.",
            metrics.openenv_capacity_retries.load(Ordering::Relaxed),
        ),
        (
            "kiln_openenv_environment_steps_total",
            "Environment step exchanges retained in canonical OpenEnv rollout artifacts.",
            metrics.openenv_environment_steps.load(Ordering::Relaxed),
        ),
        (
            "kiln_openenv_model_tokens_total",
            "Policy output tokens sampled for artifact-published OpenEnv actions.",
            metrics.openenv_model_tokens.load(Ordering::Relaxed),
        ),
    ] {
        out.push_str(&format!("# HELP {name} {help}\n# TYPE {name} counter\n"));
        push_line(out, &format!("{name} {value}"));
    }
    out.push_str(
        "# HELP kiln_openenv_model_latency_seconds_total Aggregate policy inference latency for artifact-published OpenEnv actions.\n",
    );
    out.push_str("# TYPE kiln_openenv_model_latency_seconds_total counter\n");
    push_line(
        out,
        &format!(
            "kiln_openenv_model_latency_seconds_total {:.6}",
            metrics
                .openenv_model_latency_us_total
                .load(Ordering::Relaxed) as f64
                / 1_000_000.0
        ),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rollout_metrics_use_only_closed_labels_and_exact_summary_totals() {
        let metrics = Metrics::new();
        metrics.record_openenv_rollout_stats(&OpenEnvRolloutStats {
            mean_episode_return: 0.25,
            min_episode_return: Some(-1.0),
            max_episode_return: Some(1.0),
            done_count: 3,
            max_steps_count: 2,
            invalid_model_action_count: 1,
            protocol_error_count: 1,
            recoverable_protocol_error_count: 4,
            capacity_retry_count: 5,
            total_environment_steps: 17,
            total_model_tokens: 23,
            mean_model_latency_ms: 12.5,
        });

        let mut rendered = String::new();
        render_rollout_metrics(&metrics, &mut rendered);
        assert_eq!(
            metrics.openenv_episodes_collected.load(Ordering::Relaxed),
            7
        );
        assert!(
            rendered.contains("kiln_openenv_episode_terminations_total{termination=\"done\"} 3")
        );
        assert!(
            rendered.contains(
                "kiln_openenv_episode_terminations_total{termination=\"protocol_error\"} 1"
            )
        );
        assert!(rendered.contains("kiln_openenv_recoverable_protocol_errors_total 4"));
        assert!(rendered.contains("kiln_openenv_capacity_retries_total 5"));
        assert!(rendered.contains("kiln_openenv_environment_steps_total 17"));
        assert!(rendered.contains("kiln_openenv_model_tokens_total 23"));
        assert!(rendered.contains("kiln_openenv_model_latency_seconds_total 0.087500"));
        assert!(!rendered.contains("environment="));
        assert!(!rendered.contains("adapter="));
        assert!(!rendered.contains("run_id="));
    }
}
