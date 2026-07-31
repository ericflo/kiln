//! Typed OpenEnv wire values.
//!
//! OpenEnv has no negotiated wire version. These types follow the observed
//! HTTP/1.x profile described by miniopenenv's protocol corpus. In particular,
//! rewards remain tagged because `null`, booleans, integers, and floats are
//! observably distinct on `WS /ws`.

use std::fmt;

use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;

pub const OPENENV_CLIENT_PROFILE: &str = "openenv-http/1.x";
pub const OPENENV_MAX_SERVER_MESSAGE_BYTES: usize = 512 * 1024;
pub const OPENENV_MAX_CLIENT_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
pub const OPENENV_MAX_DISCOVERY_BYTES: usize = 2 * 1024 * 1024;

/// The tagged reward accepted on the OpenEnv WebSocket.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OpenEnvReward {
    Null,
    Bool(bool),
    Integer(i64),
    Float(f64),
}

impl OpenEnvReward {
    /// Convert a protocol reward into the scalar used by RL training.
    ///
    /// Boolean rewards are numeric predicates (`true = 1`, `false = 0`);
    /// `null` contributes zero to an episode return.
    pub fn training_value(self) -> f64 {
        match self {
            Self::Null => 0.0,
            Self::Bool(value) => f64::from(u8::from(value)),
            Self::Integer(value) => value as f64,
            Self::Float(value) => value,
        }
    }

    pub fn is_null(self) -> bool {
        matches!(self, Self::Null)
    }
}

impl Serialize for OpenEnvReward {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match *self {
            Self::Null => serializer.serialize_none(),
            Self::Bool(value) => serializer.serialize_bool(value),
            Self::Integer(value) => serializer.serialize_i64(value),
            Self::Float(value) if value.is_finite() => serializer.serialize_f64(value),
            Self::Float(value) => Err(serde::ser::Error::custom(format!(
                "OpenEnv reward must be finite, got {value}"
            ))),
        }
    }
}

impl<'de> Deserialize<'de> for OpenEnvReward {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        match value {
            Value::Null => Ok(Self::Null),
            Value::Bool(value) => Ok(Self::Bool(value)),
            Value::Number(number) => {
                if let Some(value) = number.as_i64() {
                    Ok(Self::Integer(value))
                } else {
                    let value = number.as_f64().ok_or_else(|| {
                        serde::de::Error::custom("OpenEnv reward is not representable as f64")
                    })?;
                    if !value.is_finite() {
                        return Err(serde::de::Error::custom("OpenEnv reward must be finite"));
                    }
                    Ok(Self::Float(value))
                }
            }
            other => Err(serde::de::Error::custom(format!(
                "OpenEnv reward must be null, boolean, integer, or float; got {other}"
            ))),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvObservation {
    pub observation: Value,
    pub reward: OpenEnvReward,
    pub done: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvSchema {
    pub action: Value,
    pub observation: Value,
    pub state: Value,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvMetadata {
    pub name: String,
    pub description: String,
    #[serde(default)]
    pub readme_content: Option<String>,
    #[serde(default)]
    pub version: Option<String>,
    #[serde(default)]
    pub author: Option<String>,
    #[serde(default)]
    pub documentation_url: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum OpenEnvErrorCode {
    #[serde(rename = "INVALID_JSON")]
    InvalidJson,
    #[serde(rename = "UNKNOWN_TYPE")]
    UnknownType,
    #[serde(rename = "VALIDATION_ERROR")]
    ValidationError,
    #[serde(rename = "EXECUTION_ERROR")]
    ExecutionError,
    #[serde(rename = "CAPACITY_REACHED")]
    CapacityReached,
    #[serde(rename = "FACTORY_ERROR")]
    FactoryError,
    #[serde(rename = "SESSION_ERROR")]
    SessionError,
}

impl OpenEnvErrorCode {
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            Self::CapacityReached | Self::FactoryError | Self::SessionError
        )
    }

    pub fn as_wire(self) -> &'static str {
        match self {
            Self::InvalidJson => "INVALID_JSON",
            Self::UnknownType => "UNKNOWN_TYPE",
            Self::ValidationError => "VALIDATION_ERROR",
            Self::ExecutionError => "EXECUTION_ERROR",
            Self::CapacityReached => "CAPACITY_REACHED",
            Self::FactoryError => "FACTORY_ERROR",
            Self::SessionError => "SESSION_ERROR",
        }
    }
}

impl fmt::Display for OpenEnvErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_wire())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvProtocolError {
    pub code: OpenEnvErrorCode,
    pub message: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub errors: Option<Vec<Value>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub active_sessions: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_sessions: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub factory_name: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum OpenEnvServerMessage {
    Observation(OpenEnvObservation),
    State(Value),
    Error(OpenEnvProtocolError),
    Mcp(Value),
}

impl<'de> Deserialize<'de> for OpenEnvServerMessage {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Envelope {
            #[serde(rename = "type")]
            kind: String,
            data: Value,
        }

        let envelope = Envelope::deserialize(deserializer)?;
        match envelope.kind.as_str() {
            "observation" => serde_json::from_value(envelope.data)
                .map(Self::Observation)
                .map_err(serde::de::Error::custom),
            "state" => Ok(Self::State(envelope.data)),
            "error" => serde_json::from_value(envelope.data)
                .map(Self::Error)
                .map_err(serde::de::Error::custom),
            "mcp" => Ok(Self::Mcp(envelope.data)),
            other => Err(serde::de::Error::custom(format!(
                "unknown OpenEnv server message type {other:?}"
            ))),
        }
    }
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub(crate) enum OpenEnvClientMessage<'a> {
    Reset { data: &'a Value },
    Step { data: &'a Value },
    State,
    Close,
    Mcp { data: &'a Value },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reward_preserves_all_five_wire_shapes() {
        let cases = [
            ("null", OpenEnvReward::Null),
            ("true", OpenEnvReward::Bool(true)),
            ("false", OpenEnvReward::Bool(false)),
            ("3", OpenEnvReward::Integer(3)),
            ("3.0", OpenEnvReward::Float(3.0)),
        ];
        for (wire, expected) in cases {
            let parsed: OpenEnvReward = serde_json::from_str(wire).unwrap();
            assert_eq!(parsed, expected);
            assert_eq!(serde_json::to_string(&parsed).unwrap(), wire);
        }
        assert!(serde_json::from_str::<OpenEnvReward>(r#""1""#).is_err());
    }

    #[test]
    fn client_frames_obey_the_bare_action_and_data_absence_rules() {
        let reset = serde_json::json!({"seed": 7});
        assert_eq!(
            serde_json::to_value(OpenEnvClientMessage::Reset { data: &reset }).unwrap(),
            serde_json::json!({"type": "reset", "data": {"seed": 7}})
        );
        let action = serde_json::json!({"answer": "B"});
        assert_eq!(
            serde_json::to_value(OpenEnvClientMessage::Step { data: &action }).unwrap(),
            serde_json::json!({"type": "step", "data": {"answer": "B"}})
        );
        assert_eq!(
            serde_json::to_value(OpenEnvClientMessage::State).unwrap(),
            serde_json::json!({"type": "state"})
        );
        assert_eq!(
            serde_json::to_value(OpenEnvClientMessage::Close).unwrap(),
            serde_json::json!({"type": "close"})
        );
    }

    #[test]
    fn observation_reads_reward_and_done_from_data_top_level() {
        let message: OpenEnvServerMessage = serde_json::from_value(serde_json::json!({
            "type": "observation",
            "data": {
                "observation": {"metadata": {}, "answer": "ok"},
                "reward": true,
                "done": false
            }
        }))
        .unwrap();
        let OpenEnvServerMessage::Observation(observation) = message else {
            panic!("expected observation");
        };
        assert_eq!(observation.reward, OpenEnvReward::Bool(true));
        assert!(!observation.done);
        assert_eq!(observation.metadata, None);
    }
}
