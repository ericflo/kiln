//! Shared user-facing thinking-budget configuration semantics.
//!
//! Runtime enforcement lives in [`crate::sampling::ThinkingBudget`]. This
//! module owns the portable contract used before decode: inherit versus
//! explicit unlimited versus a finite limit, effective-limit resolution, and
//! stable provenance names.

use serde::{Deserialize, Serialize};

use crate::sampling::{ThinkingBudgetStatus, ThinkingBudgetTrigger};

/// An optional request/config override where omission and explicit `null`
/// have different meanings.
///
/// On JSON objects, callers pair this with `#[serde(default,
/// skip_serializing_if = "ThinkingBudgetOverride::is_inherit")]`: an omitted
/// field becomes [`Self::Inherit`], explicit `null` becomes
/// [`Self::Unlimited`], and a number becomes [`Self::Limited`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ThinkingBudgetOverride<T> {
    #[default]
    Inherit,
    Unlimited,
    Limited(T),
}

impl<T: Copy> ThinkingBudgetOverride<T> {
    pub fn resolve(self, default: Option<T>) -> Option<T> {
        match self {
            Self::Inherit => default,
            Self::Unlimited => None,
            Self::Limited(value) => Some(value),
        }
    }

    pub fn resolve_with_source(
        self,
        default: Option<T>,
        scope: ThinkingBudgetScope,
    ) -> ResolvedThinkingBudgetLimit<T> {
        let source = match self {
            Self::Inherit if default.is_some() => ThinkingBudgetSource::ServerDefault,
            Self::Inherit => ThinkingBudgetSource::Unlimited,
            Self::Unlimited => scope.unlimited_source(),
            Self::Limited(_) => scope.limit_source(),
        };
        ResolvedThinkingBudgetLimit {
            limit: self.resolve(default),
            source,
        }
    }

    pub fn is_inherit(&self) -> bool {
        matches!(self, Self::Inherit)
    }
}

impl<T: Serialize> Serialize for ThinkingBudgetOverride<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Inherit | Self::Unlimited => serializer.serialize_none(),
            Self::Limited(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ThinkingBudgetOverride<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer)
            .map(|value| value.map_or(Self::Unlimited, Self::Limited))
    }
}

/// A present CLI/wire value. Absence is represented by `Option` and converts
/// to [`ThinkingBudgetOverride::Inherit`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExplicitThinkingBudget<T> {
    Unlimited,
    Limited(T),
}

impl<T> From<ExplicitThinkingBudget<T>> for ThinkingBudgetOverride<T> {
    fn from(value: ExplicitThinkingBudget<T>) -> Self {
        match value {
            ExplicitThinkingBudget::Unlimited => Self::Unlimited,
            ExplicitThinkingBudget::Limited(value) => Self::Limited(value),
        }
    }
}

impl<T> From<Option<ExplicitThinkingBudget<T>>> for ThinkingBudgetOverride<T> {
    fn from(value: Option<ExplicitThinkingBudget<T>>) -> Self {
        value.map_or(Self::Inherit, Into::into)
    }
}

impl<T> std::str::FromStr for ExplicitThinkingBudget<T>
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    type Err = String;

    fn from_str(raw: &str) -> Result<Self, Self::Err> {
        let value = raw.trim();
        if value.eq_ignore_ascii_case("unlimited") {
            return Ok(Self::Unlimited);
        }
        value.parse::<T>().map(Self::Limited).map_err(|err| {
            format!(
                "invalid thinking budget `{raw}`: expected a non-negative integer or `unlimited` ({err})"
            )
        })
    }
}

impl<T: Serialize> Serialize for ExplicitThinkingBudget<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Unlimited => serializer.serialize_none(),
            Self::Limited(value) => value.serialize(serializer),
        }
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for ExplicitThinkingBudget<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        Option::<T>::deserialize(deserializer)
            .map(|value| value.map_or(Self::Unlimited, Self::Limited))
    }
}

/// Stable origin names used by API metadata, eval results, recent requests,
/// durable logs, and bounded metrics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingBudgetSource {
    Unlimited,
    ServerDefault,
    Request,
    RequestUnlimited,
    Suite,
    SuiteUnlimited,
    RunOverride,
    RunOverrideUnlimited,
    Example,
    ExampleUnlimited,
}

impl ThinkingBudgetSource {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Unlimited => "unlimited",
            Self::ServerDefault => "server_default",
            Self::Request => "request",
            Self::RequestUnlimited => "request_unlimited",
            Self::Suite => "suite",
            Self::SuiteUnlimited => "suite_unlimited",
            Self::RunOverride => "run_override",
            Self::RunOverrideUnlimited => "run_override_unlimited",
            Self::Example => "example",
            Self::ExampleUnlimited => "example_unlimited",
        }
    }
}

impl std::fmt::Display for ThinkingBudgetSource {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

/// Surface that supplied an explicit override. Inherited values always resolve
/// to `server_default` or `unlimited`, independent of this scope.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThinkingBudgetScope {
    Request,
    Suite,
    RunOverride,
    Example,
}

impl ThinkingBudgetScope {
    const fn limit_source(self) -> ThinkingBudgetSource {
        match self {
            Self::Request => ThinkingBudgetSource::Request,
            Self::Suite => ThinkingBudgetSource::Suite,
            Self::RunOverride => ThinkingBudgetSource::RunOverride,
            Self::Example => ThinkingBudgetSource::Example,
        }
    }

    const fn unlimited_source(self) -> ThinkingBudgetSource {
        match self {
            Self::Request => ThinkingBudgetSource::RequestUnlimited,
            Self::Suite => ThinkingBudgetSource::SuiteUnlimited,
            Self::RunOverride => ThinkingBudgetSource::RunOverrideUnlimited,
            Self::Example => ThinkingBudgetSource::ExampleUnlimited,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResolvedThinkingBudgetLimit<T> {
    pub limit: Option<T>,
    pub source: ThinkingBudgetSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThinkingBudgetOverrides {
    pub tokens: ThinkingBudgetOverride<usize>,
    pub time_ms: ThinkingBudgetOverride<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ThinkingBudgetDefaults {
    pub tokens: Option<usize>,
    pub time_ms: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct EffectiveThinkingBudget {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_time_ms: Option<u64>,
    pub tokens_source: ThinkingBudgetSource,
    pub time_source: ThinkingBudgetSource,
}

impl EffectiveThinkingBudget {
    pub fn resolve(
        overrides: ThinkingBudgetOverrides,
        defaults: ThinkingBudgetDefaults,
        scope: ThinkingBudgetScope,
    ) -> Self {
        let tokens = overrides.tokens.resolve_with_source(defaults.tokens, scope);
        let time = overrides
            .time_ms
            .resolve_with_source(defaults.time_ms, scope);
        Self {
            max_tokens: tokens.limit,
            max_time_ms: time.limit,
            tokens_source: tokens.source,
            time_source: time.source,
        }
    }

    pub const fn configured(self) -> bool {
        self.max_tokens.is_some() || self.max_time_ms.is_some()
    }
}

/// Canonical terminal/runtime outcome serialized by chat, batch, eval, and
/// durable result surfaces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ThinkingBudgetOutcome {
    pub triggered: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub trigger: Option<ThinkingBudgetTrigger>,
    pub closed: bool,
    pub thinking_tokens: usize,
    pub thinking_time_ms: u64,
}

impl ThinkingBudgetOutcome {
    pub const fn new(
        trigger: Option<ThinkingBudgetTrigger>,
        closed: bool,
        thinking_tokens: usize,
        thinking_time_ms: u64,
    ) -> Self {
        Self {
            triggered: trigger.is_some(),
            trigger,
            closed,
            thinking_tokens,
            thinking_time_ms,
        }
    }
}

impl From<ThinkingBudgetStatus> for ThinkingBudgetOutcome {
    fn from(status: ThinkingBudgetStatus) -> Self {
        Self::new(
            status.trigger,
            status.closed,
            status.thinking_tokens,
            status.elapsed_ms,
        )
    }
}

impl From<&ThinkingBudgetStatus> for ThinkingBudgetOutcome {
    fn from(status: &ThinkingBudgetStatus) -> Self {
        (*status).into()
    }
}

impl<'de> Deserialize<'de> for ThinkingBudgetOutcome {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Wire {
            triggered: bool,
            #[serde(default)]
            trigger: Option<ThinkingBudgetTrigger>,
            closed: bool,
            thinking_tokens: usize,
            thinking_time_ms: u64,
        }

        let wire = Wire::deserialize(deserializer)?;
        if wire.triggered != wire.trigger.is_some() {
            return Err(serde::de::Error::custom(
                "thinking-budget triggered must agree with trigger presence",
            ));
        }
        Ok(Self::new(
            wire.trigger,
            wire.closed,
            wire.thinking_tokens,
            wire.thinking_time_ms,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Serialize, Deserialize, PartialEq, Eq)]
    struct Wire {
        #[serde(default, skip_serializing_if = "ThinkingBudgetOverride::is_inherit")]
        budget: ThinkingBudgetOverride<usize>,
    }

    #[test]
    fn override_wire_shape_preserves_omitted_null_zero_and_limit() {
        assert_eq!(
            serde_json::from_str::<Wire>(r#"{}"#).unwrap().budget,
            ThinkingBudgetOverride::Inherit
        );
        assert_eq!(
            serde_json::from_str::<Wire>(r#"{"budget":null}"#)
                .unwrap()
                .budget,
            ThinkingBudgetOverride::Unlimited
        );
        assert_eq!(
            serde_json::from_str::<Wire>(r#"{"budget":0}"#)
                .unwrap()
                .budget,
            ThinkingBudgetOverride::Limited(0)
        );
        assert_eq!(
            serde_json::to_value(Wire {
                budget: ThinkingBudgetOverride::Inherit
            })
            .unwrap(),
            serde_json::json!({})
        );
        assert_eq!(
            serde_json::to_value(Wire {
                budget: ThinkingBudgetOverride::Unlimited
            })
            .unwrap(),
            serde_json::json!({"budget": null})
        );
    }

    #[test]
    fn independent_dimensions_resolve_values_and_sources() {
        let effective = EffectiveThinkingBudget::resolve(
            ThinkingBudgetOverrides {
                tokens: ThinkingBudgetOverride::Limited(0),
                time_ms: ThinkingBudgetOverride::Unlimited,
            },
            ThinkingBudgetDefaults {
                tokens: Some(64),
                time_ms: Some(1_500),
            },
            ThinkingBudgetScope::Request,
        );
        assert_eq!(effective.max_tokens, Some(0));
        assert_eq!(effective.max_time_ms, None);
        assert_eq!(effective.tokens_source, ThinkingBudgetSource::Request);
        assert_eq!(
            effective.time_source,
            ThinkingBudgetSource::RequestUnlimited
        );
        assert!(effective.configured());
    }

    #[test]
    fn every_scope_has_stable_limited_and_unlimited_provenance() {
        let cases = [
            (ThinkingBudgetScope::Request, "request", "request_unlimited"),
            (ThinkingBudgetScope::Suite, "suite", "suite_unlimited"),
            (
                ThinkingBudgetScope::RunOverride,
                "run_override",
                "run_override_unlimited",
            ),
            (ThinkingBudgetScope::Example, "example", "example_unlimited"),
        ];
        for (scope, limited, unlimited) in cases {
            assert_eq!(
                ThinkingBudgetOverride::Limited(1)
                    .resolve_with_source(None, scope)
                    .source
                    .as_str(),
                limited
            );
            assert_eq!(
                ThinkingBudgetOverride::<usize>::Unlimited
                    .resolve_with_source(Some(2), scope)
                    .source
                    .as_str(),
                unlimited
            );
        }
        assert_eq!(
            ThinkingBudgetOverride::Inherit
                .resolve_with_source(Some(2), ThinkingBudgetScope::Example)
                .source,
            ThinkingBudgetSource::ServerDefault
        );
        assert_eq!(
            ThinkingBudgetOverride::<usize>::Inherit
                .resolve_with_source(None, ThinkingBudgetScope::Example)
                .source,
            ThinkingBudgetSource::Unlimited
        );
    }

    #[test]
    fn explicit_cli_value_converts_to_the_shared_override() {
        assert_eq!(
            "0".parse::<ExplicitThinkingBudget<usize>>().unwrap(),
            ExplicitThinkingBudget::Limited(0)
        );
        assert_eq!(
            "UNLIMITED"
                .parse::<ExplicitThinkingBudget<usize>>()
                .unwrap(),
            ExplicitThinkingBudget::Unlimited
        );
        assert!("-1".parse::<ExplicitThinkingBudget<usize>>().is_err());
        assert_eq!(
            ThinkingBudgetOverride::from(None::<ExplicitThinkingBudget<usize>>),
            ThinkingBudgetOverride::Inherit
        );
        assert_eq!(
            ThinkingBudgetOverride::from(Some(ExplicitThinkingBudget::<usize>::Unlimited)),
            ThinkingBudgetOverride::Unlimited
        );
    }

    #[test]
    fn outcome_wire_shape_is_shared_and_rejects_inconsistent_trigger_state() {
        let outcome = ThinkingBudgetOutcome::new(Some(ThinkingBudgetTrigger::Tokens), true, 12, 41);
        let json = serde_json::to_value(outcome).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "triggered": true,
                "trigger": "tokens",
                "closed": true,
                "thinking_tokens": 12,
                "thinking_time_ms": 41
            })
        );
        assert_eq!(
            serde_json::from_value::<ThinkingBudgetOutcome>(json).unwrap(),
            outcome
        );
        assert!(
            serde_json::from_value::<ThinkingBudgetOutcome>(serde_json::json!({
                "triggered": false,
                "trigger": "time",
                "closed": false,
                "thinking_tokens": 3,
                "thinking_time_ms": 7
            }))
            .is_err()
        );
    }
}
