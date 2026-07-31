//! First-class OpenEnv support for Kiln.
//!
//! The crate owns the reusable protocol boundary: HTTP discovery, the exact
//! stateful `WS /ws` episode loop, tagged rewards, closed error vocabulary,
//! lock-step framing, bounded messages, and content-addressed environment
//! identity. Kiln's model rollout and GRPO orchestration live in
//! `kiln-server`, above this crate.

mod client;
mod types;

pub use client::{
    OpenEnvAuthentication, OpenEnvClient, OpenEnvClientError, OpenEnvIdentity, OpenEnvInspection,
    OpenEnvSession,
};
pub use types::{
    OPENENV_CLIENT_PROFILE, OPENENV_MAX_CLIENT_MESSAGE_BYTES, OPENENV_MAX_DISCOVERY_BYTES,
    OPENENV_MAX_SERVER_MESSAGE_BYTES, OPENENV_MAX_TASK_CATALOG_NAMES, OPENENV_MAX_TASK_ITEMS,
    OPENENV_MAX_TASK_SELECTOR_BYTES, OpenEnvErrorCode, OpenEnvMetadata, OpenEnvObservation,
    OpenEnvProtocolError, OpenEnvReward, OpenEnvSchema, OpenEnvServerMessage, OpenEnvTask,
    OpenEnvTaskApiSupport, OpenEnvTaskCatalog, OpenEnvTaskCount, OpenEnvTaskList, OpenEnvTaskRange,
    OpenEnvTaskSplit,
};
