//! Async OpenEnv discovery and stateful episode client.

use std::{future::Future, time::Duration};

use anyhow::Context;
use futures::{SinkExt, StreamExt};
use reqwest::{
    StatusCode,
    header::{AUTHORIZATION, HeaderValue},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use tokio::net::TcpStream;
use tokio_tungstenite::{
    MaybeTlsStream, WebSocketStream, connect_async_with_config,
    tungstenite::{self, Message, client::IntoClientRequest, protocol::WebSocketConfig},
};

use crate::types::{
    OPENENV_CLIENT_PROFILE, OPENENV_MAX_CLIENT_MESSAGE_BYTES, OPENENV_MAX_DISCOVERY_BYTES,
    OPENENV_MAX_SERVER_MESSAGE_BYTES, OPENENV_MAX_TASK_CATALOG_NAMES, OPENENV_MAX_TASK_ITEMS,
    OPENENV_MAX_TASK_SELECTOR_BYTES, OpenEnvClientMessage, OpenEnvMetadata, OpenEnvObservation,
    OpenEnvProtocolError, OpenEnvSchema, OpenEnvServerMessage, OpenEnvTask, OpenEnvTaskApiSupport,
    OpenEnvTaskCatalog, OpenEnvTaskCount, OpenEnvTaskList, OpenEnvTaskRange, OpenEnvTaskSplit,
};
use crate::{OpenEnvActionSchemaError, OpenEnvActionValidator};

const DEFAULT_REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, thiserror::Error)]
pub enum OpenEnvClientError {
    #[error("invalid OpenEnv base URL: {0}")]
    InvalidBaseUrl(String),
    #[error("invalid OpenEnv bearer credential")]
    InvalidCredential,
    #[error("OpenEnv bearer credentials require HTTPS/WSS unless the environment host is loopback")]
    InsecureCredentialTransport,
    #[error("OpenEnv HTTP request failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("OpenEnv endpoint {endpoint} returned HTTP {status}: {body}")]
    HttpStatus {
        endpoint: String,
        status: StatusCode,
        body: String,
    },
    #[error("OpenEnv endpoint {endpoint} exceeded the {limit} byte discovery limit")]
    HttpBodyTooLarge { endpoint: String, limit: usize },
    #[error("OpenEnv Task API {selector} must contain 1..={limit} UTF-8 bytes, got {actual}")]
    InvalidTaskSelector {
        selector: &'static str,
        actual: usize,
        limit: usize,
    },
    #[error("OpenEnv Task API {collection} returned {count} items; limit is {limit}")]
    TaskCollectionTooLarge {
        collection: &'static str,
        count: usize,
        limit: usize,
    },
    #[error(
        "OpenEnv Task API environment name is required because the server advertised {available:?}"
    )]
    TaskEnvironmentRequired { available: Vec<String> },
    #[error(
        "OpenEnv Task API environment {requested:?} was not advertised; available names: {available:?}"
    )]
    UnknownTaskEnvironment {
        requested: String,
        available: Vec<String>,
    },
    #[error(
        "OpenEnv Task API split {requested:?} was not advertised; available names: {available:?}"
    )]
    UnknownTaskSplit {
        requested: String,
        available: Vec<String>,
    },
    #[error("OpenEnv Task API provider returned invalid task count {0}")]
    InvalidTaskCount(i64),
    #[error("OpenEnv Task API page limit must be in 1..={max}, got {actual}")]
    InvalidTaskPageLimit { actual: usize, max: usize },
    #[error("OpenEnv Task API provider returned {actual} tasks for a requested page of {limit}")]
    InvalidTaskPage { actual: usize, limit: usize },
    #[error("OpenEnv WebSocket failed: {0}")]
    WebSocket(#[from] tungstenite::Error),
    #[error("authenticated OpenEnv WebSocket upgrade returned HTTP {0}; response redacted")]
    AuthenticatedWebSocketStatus(StatusCode),
    #[error("OpenEnv peer reflected the configured bearer credential; response rejected")]
    CredentialReflected,
    #[error("OpenEnv request timed out after {0:?}")]
    Timeout(Duration),
    #[error("OpenEnv peer closed the episode socket")]
    Closed,
    #[error("OpenEnv peer sent a binary frame; text JSON is required")]
    BinaryFrame,
    #[error("OpenEnv peer sent an unsolicited application message while the client was idle")]
    UnsolicitedApplicationMessage,
    #[error("OpenEnv frame exceeded the {0} byte client limit")]
    MessageTooLarge(usize),
    #[error("invalid OpenEnv message: {0}")]
    InvalidMessage(String),
    #[error("OpenEnv protocol error {0:?}")]
    Protocol(OpenEnvProtocolError),
    #[error("OpenEnv response type {actual} did not match expected {expected}")]
    UnexpectedResponse {
        expected: &'static str,
        actual: &'static str,
    },
    #[error(
        "OpenEnv environment {endpoint} changed identity during the bounded operation: discovery fields {changed_fields:?}; expected schema {expected_schema_sha256}, observed {actual_schema_sha256}"
    )]
    EnvironmentIdentityChanged {
        endpoint: String,
        expected_schema_sha256: String,
        actual_schema_sha256: String,
        changed_fields: Vec<&'static str>,
    },
    #[error("OpenEnv endpoint {endpoint} advertised an invalid action schema: {source}")]
    InvalidActionSchema {
        endpoint: String,
        #[source]
        source: OpenEnvActionSchemaError,
    },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum OpenEnvAuthentication {
    #[default]
    None,
    Bearer,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvIdentity {
    pub schema: String,
    pub client_profile: String,
    pub base_url: String,
    pub websocket_url: String,
    /// Authentication method applied to both discovery and WebSocket upgrade.
    /// Credential handles and secret values are intentionally excluded.
    #[serde(default)]
    pub authentication: OpenEnvAuthentication,
    pub openapi_version: Option<String>,
    pub environments: Vec<String>,
    pub metadata: OpenEnvMetadata,
    pub schema_sha256: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct OpenEnvInspection {
    pub identity: OpenEnvIdentity,
    pub schema: OpenEnvSchema,
}

impl OpenEnvInspection {
    /// Compile the advertised action schema into a reusable, self-contained
    /// validator. External HTTP and filesystem references are never resolved.
    pub fn action_validator(&self) -> Result<OpenEnvActionValidator, OpenEnvActionSchemaError> {
        OpenEnvActionValidator::compile(&self.schema.action)
    }

    /// Require a later discovery snapshot to be exactly the same environment
    /// identity used to generate actions and collect rewards.
    ///
    /// The error intentionally names only closed discovery-field labels and
    /// content digests. Environment-authored metadata or schemas are never
    /// copied into an error message.
    pub fn ensure_unchanged(&self, current: &Self) -> Result<(), OpenEnvClientError> {
        let mut changed_fields = Vec::new();
        for (changed, field) in [
            (
                self.identity.schema != current.identity.schema,
                "identity.schema",
            ),
            (
                self.identity.client_profile != current.identity.client_profile,
                "identity.client_profile",
            ),
            (
                self.identity.base_url != current.identity.base_url,
                "identity.base_url",
            ),
            (
                self.identity.websocket_url != current.identity.websocket_url,
                "identity.websocket_url",
            ),
            (
                self.identity.authentication != current.identity.authentication,
                "identity.authentication",
            ),
            (
                self.identity.openapi_version != current.identity.openapi_version,
                "identity.openapi_version",
            ),
            (
                self.identity.environments != current.identity.environments,
                "identity.environments",
            ),
            (
                self.identity.metadata != current.identity.metadata,
                "identity.metadata",
            ),
            (
                self.identity.schema_sha256 != current.identity.schema_sha256,
                "identity.schema_sha256",
            ),
            (self.schema.action != current.schema.action, "schema.action"),
            (
                self.schema.observation != current.schema.observation,
                "schema.observation",
            ),
            (self.schema.state != current.schema.state, "schema.state"),
        ] {
            if changed {
                changed_fields.push(field);
            }
        }
        if changed_fields.is_empty() {
            return Ok(());
        }
        Err(OpenEnvClientError::EnvironmentIdentityChanged {
            endpoint: self.identity.base_url.clone(),
            expected_schema_sha256: self.identity.schema_sha256.clone(),
            actual_schema_sha256: current.identity.schema_sha256.clone(),
            changed_fields,
        })
    }
}

#[derive(Clone)]
pub struct OpenEnvClient {
    base_url: String,
    websocket_url: String,
    http: reqwest::Client,
    request_timeout: Duration,
    /// A sensitive header value intentionally omitted from Debug, identities,
    /// receipts, replay manifests, and every serialized OpenEnv artifact.
    authorization: Option<HeaderValue>,
}

impl std::fmt::Debug for OpenEnvClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvClient")
            .field("base_url", &self.base_url)
            .field("websocket_url", &self.websocket_url)
            .field("request_timeout", &self.request_timeout)
            .field("authenticated", &self.authorization.is_some())
            .finish_non_exhaustive()
    }
}

impl OpenEnvClient {
    pub fn new(base_url: impl AsRef<str>) -> Result<Self, OpenEnvClientError> {
        let (base_url, websocket_url) = normalize_urls(base_url.as_ref())?;
        let http = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .timeout(DEFAULT_REQUEST_TIMEOUT)
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .map_err(OpenEnvClientError::Http)?;
        Ok(Self {
            base_url,
            websocket_url,
            http,
            request_timeout: DEFAULT_REQUEST_TIMEOUT,
            authorization: None,
        })
    }

    /// Authenticate both HTTP discovery and the stateful WebSocket upgrade
    /// with one standard bearer credential.
    ///
    /// The credential is held only as a sensitive request header and is never
    /// included in Debug output or any protocol identity.
    pub fn with_bearer_token(mut self, token: impl AsRef<str>) -> Result<Self, OpenEnvClientError> {
        let token = token.as_ref();
        if token.trim().is_empty()
            || token.trim() != token
            || token
                .bytes()
                .any(|byte| byte.is_ascii_whitespace() || byte.is_ascii_control())
        {
            return Err(OpenEnvClientError::InvalidCredential);
        }
        if !credential_transport_is_secure(&self.base_url) {
            return Err(OpenEnvClientError::InsecureCredentialTransport);
        }
        let mut authorization = HeaderValue::from_str(&format!("Bearer {token}"))
            .map_err(|_| OpenEnvClientError::InvalidCredential)?;
        authorization.set_sensitive(true);
        self.authorization = Some(authorization);
        Ok(self)
    }

    pub fn with_request_timeout(mut self, timeout: Duration) -> Self {
        self.request_timeout = timeout;
        self
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    pub fn websocket_url(&self) -> &str {
        &self.websocket_url
    }

    /// OpenEnv client requirement 8.B.2: health is status-only.
    pub async fn health(&self) -> Result<(), OpenEnvClientError> {
        let endpoint = self.endpoint("health");
        let response = self.authenticate(self.http.get(&endpoint)).send().await?;
        if response.status() != StatusCode::OK {
            return Err(http_status_error(endpoint, response, self.authorization.is_some()).await);
        }
        Ok(())
    }

    pub async fn metadata(&self) -> Result<OpenEnvMetadata, OpenEnvClientError> {
        self.get_json("metadata").await
    }

    pub async fn schema(&self) -> Result<OpenEnvSchema, OpenEnvClientError> {
        self.get_json("schema").await
    }

    pub async fn list_environments(&self) -> Result<Vec<String>, OpenEnvClientError> {
        self.get_json("list_environments").await
    }

    /// Build a bounded catalog from OpenEnv's optional Task API.
    ///
    /// The selected task values are never interpreted as reset data. OpenEnv
    /// does not define that relationship; portable episode scheduling remains
    /// an explicit reset payload plus deterministic seed.
    pub async fn task_catalog(
        &self,
        requested_environment: Option<&str>,
        requested_split: Option<&str>,
        start: u64,
        limit: usize,
    ) -> Result<OpenEnvTaskCatalog, OpenEnvClientError> {
        if limit == 0 || limit > OPENENV_MAX_TASK_ITEMS {
            return Err(OpenEnvClientError::InvalidTaskPageLimit {
                actual: limit,
                max: OPENENV_MAX_TASK_ITEMS,
            });
        }
        if let Some(environment) = requested_environment {
            validate_task_selector("environment name", environment)?;
        }
        if let Some(split) = requested_split {
            validate_task_selector("split name", split)?;
        }

        let environments = self.list_environments().await?;
        ensure_task_collection_limit("environment names", environments.len())?;
        ensure_task_catalog_name_limit("environment names", environments.len())?;
        for environment in &environments {
            validate_task_selector("advertised environment name", environment)?;
        }
        let environment_name = select_task_name(
            requested_environment,
            &environments,
            TaskNameKind::Environment,
        )?;
        let splits = match self.list_task_splits(&environment_name).await {
            Ok(splits) => splits,
            Err(error) if error.is_task_api_unsupported() => {
                return Ok(OpenEnvTaskCatalog {
                    environment_name,
                    task_api: OpenEnvTaskApiSupport::Unsupported,
                    splits: Vec::new(),
                    selected_split: None,
                    num_tasks: None,
                    start: None,
                    stop: None,
                    tasks: Vec::new(),
                });
            }
            Err(error) => return Err(error),
        };
        ensure_task_catalog_name_limit("split names", splits.len())?;
        for split in &splits {
            validate_task_selector("advertised split name", &split.name)?;
        }
        let Some(requested_split) = requested_split else {
            return Ok(OpenEnvTaskCatalog {
                environment_name,
                task_api: OpenEnvTaskApiSupport::Available,
                splits,
                selected_split: None,
                num_tasks: None,
                start: None,
                stop: None,
                tasks: Vec::new(),
            });
        };
        let split_names = splits
            .iter()
            .map(|split| split.name.clone())
            .collect::<Vec<_>>();
        let selected_split =
            select_task_name(Some(requested_split), &split_names, TaskNameKind::Split)?;
        let count = self.num_tasks(&environment_name, &selected_split).await?;
        let num_tasks = u64::try_from(count.num_tasks)
            .map_err(|_| OpenEnvClientError::InvalidTaskCount(count.num_tasks))?;
        let stop = start.saturating_add(limit as u64).min(num_tasks);
        let tasks = if start >= stop {
            Vec::new()
        } else {
            let start_i64 =
                i64::try_from(start).map_err(|_| OpenEnvClientError::InvalidTaskCount(i64::MAX))?;
            let stop_i64 =
                i64::try_from(stop).map_err(|_| OpenEnvClientError::InvalidTaskCount(i64::MAX))?;
            self.get_task_range(
                &environment_name,
                &selected_split,
                Some(start_i64),
                Some(stop_i64),
            )
            .await?
            .tasks
        };
        if tasks.len() > limit {
            return Err(OpenEnvClientError::InvalidTaskPage {
                actual: tasks.len(),
                limit,
            });
        }
        Ok(OpenEnvTaskCatalog {
            environment_name,
            task_api: OpenEnvTaskApiSupport::Available,
            splits,
            selected_split: Some(selected_split),
            num_tasks: Some(num_tasks),
            start: Some(start),
            stop: Some(stop),
            tasks,
        })
    }

    /// List the dataset splits advertised by an optional OpenEnv TaskProvider.
    ///
    /// HTTP 501 is a conforming result for environments without a task
    /// provider and remains available through `OpenEnvClientError`.
    pub async fn list_task_splits(
        &self,
        environment: &str,
    ) -> Result<Vec<OpenEnvTaskSplit>, OpenEnvClientError> {
        validate_task_selector("environment name", environment)?;
        let endpoint = self.endpoint_segments(&[environment, "splits"])?;
        let splits: Vec<OpenEnvTaskSplit> = self.get_json_endpoint(endpoint).await?;
        ensure_task_collection_limit("splits", splits.len())?;
        Ok(splits)
    }

    /// Fetch every task in a split. Prefer `num_tasks` plus `get_task_range`
    /// for interactive or control-plane use so a provider cannot force a large
    /// catalog transfer.
    pub async fn list_tasks(
        &self,
        environment: &str,
        split: &str,
    ) -> Result<OpenEnvTaskList, OpenEnvClientError> {
        validate_task_selectors(environment, split)?;
        let endpoint = self.endpoint_segments(&[environment, "tasks"])?;
        let response: OpenEnvTaskList = self
            .post_json_endpoint(endpoint, &TaskSplitRequest { split })
            .await?;
        ensure_task_collection_limit("tasks", response.tasks.len())?;
        Ok(response)
    }

    pub async fn num_tasks(
        &self,
        environment: &str,
        split: &str,
    ) -> Result<OpenEnvTaskCount, OpenEnvClientError> {
        validate_task_selectors(environment, split)?;
        let endpoint = self.endpoint_segments(&[environment, "num_tasks"])?;
        self.post_json_endpoint(endpoint, &TaskSplitRequest { split })
            .await
    }

    pub async fn get_task(
        &self,
        environment: &str,
        split: &str,
        index: i64,
    ) -> Result<OpenEnvTask, OpenEnvClientError> {
        validate_task_selectors(environment, split)?;
        let endpoint = self.endpoint_segments(&[environment, "task"])?;
        self.post_json_endpoint(endpoint, &TaskIndexRequest { split, index })
            .await
    }

    /// Fetch one bounded Python-style task slice. Signed optional bounds are
    /// preserved because negative indexes are observable OpenEnv behavior.
    pub async fn get_task_range(
        &self,
        environment: &str,
        split: &str,
        start: Option<i64>,
        stop: Option<i64>,
    ) -> Result<OpenEnvTaskRange, OpenEnvClientError> {
        validate_task_selectors(environment, split)?;
        let endpoint = self.endpoint_segments(&[environment, "task_range"])?;
        let response: OpenEnvTaskRange = self
            .post_json_endpoint(endpoint, &TaskRangeRequest { split, start, stop })
            .await?;
        ensure_task_collection_limit("task range", response.tasks.len())?;
        Ok(response)
    }

    pub async fn openapi(&self) -> Result<Value, OpenEnvClientError> {
        self.get_json("openapi.json").await
    }

    /// Read all stable discovery surfaces and derive a content-addressed
    /// environment identity. Health is checked first because an OpenEnv server
    /// may answer it independently of environment construction.
    pub async fn inspect(&self) -> Result<OpenEnvInspection, OpenEnvClientError> {
        self.health().await?;
        let (metadata, schema, environments, openapi) = tokio::try_join!(
            self.metadata(),
            self.schema(),
            self.list_environments(),
            self.openapi()
        )?;
        let schema_bytes = serde_json::to_vec(&schema)
            .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()))?;
        let schema_sha256 = format!(
            "sha256:{}",
            Sha256::digest(schema_bytes)
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>()
        );
        let openapi_version = openapi
            .pointer("/info/version")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let inspection = OpenEnvInspection {
            identity: OpenEnvIdentity {
                schema: "kiln.openenv-identity.v1".to_string(),
                client_profile: OPENENV_CLIENT_PROFILE.to_string(),
                base_url: self.base_url.clone(),
                websocket_url: self.websocket_url.clone(),
                authentication: if self.authorization.is_some() {
                    OpenEnvAuthentication::Bearer
                } else {
                    OpenEnvAuthentication::None
                },
                openapi_version,
                environments,
                metadata,
                schema_sha256,
            },
            schema,
        };
        // Discovery is not successful unless the action contract can actually
        // be enforced. Compile it before any session is opened or artifact is
        // attributed to this identity.
        inspection.action_validator().map_err(|source| {
            OpenEnvClientError::InvalidActionSchema {
                endpoint: self.endpoint("schema"),
                source,
            }
        })?;
        Ok(inspection)
    }

    /// Re-read every stable discovery surface and fail if the environment no
    /// longer has the exact identity captured before a bounded operation.
    pub async fn revalidate(&self, expected: &OpenEnvInspection) -> Result<(), OpenEnvClientError> {
        let current = self.inspect().await?;
        expected.ensure_unchanged(&current)
    }

    pub async fn connect(&self) -> Result<OpenEnvSession, OpenEnvClientError> {
        let socket_config = WebSocketConfig::default()
            .read_buffer_size(16 * 1024)
            .write_buffer_size(16 * 1024)
            .max_write_buffer_size(OPENENV_MAX_CLIENT_MESSAGE_BYTES + 16 * 1024)
            .max_message_size(Some(OPENENV_MAX_SERVER_MESSAGE_BYTES))
            .max_frame_size(Some(OPENENV_MAX_SERVER_MESSAGE_BYTES));
        let mut request = self.websocket_url.as_str().into_client_request()?;
        if let Some(authorization) = &self.authorization {
            request
                .headers_mut()
                .insert(AUTHORIZATION, authorization.clone());
        }
        let connect = connect_async_with_config(request, Some(socket_config), false);
        let connected = tokio::time::timeout(self.request_timeout, connect)
            .await
            .map_err(|_| OpenEnvClientError::Timeout(self.request_timeout))?;
        let (socket, response) = match connected {
            Ok(connected) => connected,
            Err(tungstenite::Error::Http(response)) if self.authorization.is_some() => {
                return Err(OpenEnvClientError::AuthenticatedWebSocketStatus(
                    response.status(),
                ));
            }
            Err(error) => return Err(OpenEnvClientError::WebSocket(error)),
        };
        if response.status() != StatusCode::SWITCHING_PROTOCOLS {
            return Err(OpenEnvClientError::InvalidMessage(format!(
                "OpenEnv WebSocket upgrade returned {}",
                response.status()
            )));
        }
        Ok(OpenEnvSession {
            socket,
            request_timeout: self.request_timeout,
            closed: false,
            credential_marker: bearer_token_bytes(self.authorization.as_ref())
                .map(ToOwned::to_owned),
        })
    }

    fn endpoint(&self, path: &str) -> String {
        format!("{}/{}", self.base_url, path.trim_start_matches('/'))
    }

    fn endpoint_segments(&self, segments: &[&str]) -> Result<String, OpenEnvClientError> {
        let mut endpoint = reqwest::Url::parse(&format!("{}/", self.base_url))
            .map_err(|error| OpenEnvClientError::InvalidBaseUrl(error.to_string()))?;
        {
            let mut path = endpoint.path_segments_mut().map_err(|_| {
                OpenEnvClientError::InvalidBaseUrl(
                    "OpenEnv base URL cannot hold path segments".to_string(),
                )
            })?;
            path.pop_if_empty();
            for segment in segments {
                path.push(segment);
            }
        }
        Ok(endpoint.into())
    }

    fn authenticate(&self, request: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        match &self.authorization {
            Some(authorization) => request.header(AUTHORIZATION, authorization.clone()),
            None => request,
        }
    }

    async fn get_json<T>(&self, path: &str) -> Result<T, OpenEnvClientError>
    where
        T: serde::de::DeserializeOwned,
    {
        let endpoint = self.endpoint(path);
        self.get_json_endpoint(endpoint).await
    }

    async fn get_json_endpoint<T>(&self, endpoint: String) -> Result<T, OpenEnvClientError>
    where
        T: serde::de::DeserializeOwned,
    {
        let response = self.authenticate(self.http.get(&endpoint)).send().await?;
        if !response.status().is_success() {
            return Err(http_status_error(endpoint, response, self.authorization.is_some()).await);
        }
        let bytes = read_http_body_bounded(&endpoint, response).await?;
        if credential_reflected(&bytes, self.authorization.as_ref()) {
            return Err(OpenEnvClientError::CredentialReflected);
        }
        serde_json::from_slice(&bytes)
            .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()))
    }

    async fn post_json_endpoint<T, B>(
        &self,
        endpoint: String,
        body: &B,
    ) -> Result<T, OpenEnvClientError>
    where
        T: serde::de::DeserializeOwned,
        B: Serialize + ?Sized,
    {
        let response = self
            .authenticate(self.http.post(&endpoint).json(body))
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(http_status_error(endpoint, response, self.authorization.is_some()).await);
        }
        let bytes = read_http_body_bounded(&endpoint, response).await?;
        if credential_reflected(&bytes, self.authorization.as_ref()) {
            return Err(OpenEnvClientError::CredentialReflected);
        }
        serde_json::from_slice(&bytes)
            .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()))
    }
}

impl OpenEnvClientError {
    pub fn http_status_code(&self) -> Option<u16> {
        match self {
            Self::HttpStatus { status, .. } => Some(status.as_u16()),
            Self::AuthenticatedWebSocketStatus(status) => Some(status.as_u16()),
            _ => None,
        }
    }

    /// A provider-less Task API route is explicitly conforming OpenEnv
    /// behavior, not evidence that the server itself is incompatible.
    pub fn is_task_api_unsupported(&self) -> bool {
        self.http_status_code() == Some(StatusCode::NOT_IMPLEMENTED.as_u16())
    }
}

#[derive(Serialize)]
struct TaskSplitRequest<'a> {
    split: &'a str,
}

#[derive(Serialize)]
struct TaskIndexRequest<'a> {
    split: &'a str,
    index: i64,
}

#[derive(Serialize)]
struct TaskRangeRequest<'a> {
    split: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    start: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<i64>,
}

#[derive(Clone, Copy)]
enum TaskNameKind {
    Environment,
    Split,
}

fn select_task_name(
    requested: Option<&str>,
    available: &[String],
    kind: TaskNameKind,
) -> Result<String, OpenEnvClientError> {
    if let Some(requested) = requested {
        let requested_lower = requested.to_lowercase();
        return available
            .iter()
            .find(|name| name.to_lowercase() == requested_lower)
            .cloned()
            .ok_or_else(|| match kind {
                TaskNameKind::Environment => OpenEnvClientError::UnknownTaskEnvironment {
                    requested: requested.to_string(),
                    available: available.to_vec(),
                },
                TaskNameKind::Split => OpenEnvClientError::UnknownTaskSplit {
                    requested: requested.to_string(),
                    available: available.to_vec(),
                },
            });
    }
    if available.len() == 1 {
        return Ok(available[0].clone());
    }
    Err(OpenEnvClientError::TaskEnvironmentRequired {
        available: available.to_vec(),
    })
}

fn validate_task_selectors(environment: &str, split: &str) -> Result<(), OpenEnvClientError> {
    validate_task_selector("environment name", environment)?;
    validate_task_selector("split name", split)
}

fn validate_task_selector(selector: &'static str, value: &str) -> Result<(), OpenEnvClientError> {
    let actual = value.len();
    if actual == 0 || actual > OPENENV_MAX_TASK_SELECTOR_BYTES {
        return Err(OpenEnvClientError::InvalidTaskSelector {
            selector,
            actual,
            limit: OPENENV_MAX_TASK_SELECTOR_BYTES,
        });
    }
    Ok(())
}

fn ensure_task_collection_limit(
    collection: &'static str,
    count: usize,
) -> Result<(), OpenEnvClientError> {
    if count > OPENENV_MAX_TASK_ITEMS {
        return Err(OpenEnvClientError::TaskCollectionTooLarge {
            collection,
            count,
            limit: OPENENV_MAX_TASK_ITEMS,
        });
    }
    Ok(())
}

fn ensure_task_catalog_name_limit(
    collection: &'static str,
    count: usize,
) -> Result<(), OpenEnvClientError> {
    if count > OPENENV_MAX_TASK_CATALOG_NAMES {
        return Err(OpenEnvClientError::TaskCollectionTooLarge {
            collection,
            count,
            limit: OPENENV_MAX_TASK_CATALOG_NAMES,
        });
    }
    Ok(())
}

pub struct OpenEnvSession {
    socket: WebSocketStream<MaybeTlsStream<TcpStream>>,
    request_timeout: Duration,
    closed: bool,
    /// Raw credential bytes used only to reject reflected secrets before
    /// parsing. Intentionally omitted from Debug and every serialized value.
    credential_marker: Option<Vec<u8>>,
}

impl std::fmt::Debug for OpenEnvSession {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvSession")
            .field("request_timeout", &self.request_timeout)
            .field("closed", &self.closed)
            .field("authenticated", &self.credential_marker.is_some())
            .finish_non_exhaustive()
    }
}

impl OpenEnvSession {
    pub async fn reset(&mut self, data: &Value) -> Result<OpenEnvObservation, OpenEnvClientError> {
        let response = self.exchange(OpenEnvClientMessage::Reset { data }).await?;
        self.expect_observation(response)
    }

    pub async fn step(&mut self, action: &Value) -> Result<OpenEnvObservation, OpenEnvClientError> {
        let response = self
            .exchange(OpenEnvClientMessage::Step { data: action })
            .await?;
        self.expect_observation(response)
    }

    pub async fn state(&mut self) -> Result<Value, OpenEnvClientError> {
        let response = self.exchange(OpenEnvClientMessage::State).await?;
        match response {
            OpenEnvServerMessage::State(state) => Ok(state),
            other => self.fail_closed(unexpected("state", &other)),
        }
    }

    pub async fn mcp(&mut self, request: &Value) -> Result<Value, OpenEnvClientError> {
        let response = self
            .exchange(OpenEnvClientMessage::Mcp { data: request })
            .await?;
        match response {
            OpenEnvServerMessage::Mcp(value) => Ok(value),
            other => self.fail_closed(unexpected("mcp", &other)),
        }
    }

    /// Keep the episode socket alive while policy work is in flight.
    ///
    /// OpenEnv permits WebSocket control frames while otherwise remaining
    /// strictly lock-step. Long model inference can exceed a server's ping
    /// timeout, so callers should wrap that work here between an observation
    /// and the next action. Ping frames are answered, Pong frames are absorbed,
    /// and any unsolicited application message poisons the session because it
    /// cannot be correlated with a request.
    pub async fn keep_alive_while<F>(&mut self, work: F) -> Result<F::Output, OpenEnvClientError>
    where
        F: Future,
    {
        if self.closed {
            return Err(OpenEnvClientError::Closed);
        }
        tokio::pin!(work);
        loop {
            tokio::select! {
                // If policy completion and a buffered peer frame become ready
                // together, validate the frame first. Otherwise an already
                // unsolicited response could be mistaken for the next step's
                // answer after this method returns.
                biased;
                frame = self.socket.next() => {
                    let frame = match frame {
                        Some(Ok(frame)) => frame,
                        Some(Err(error)) => {
                            return self.fail_closed(OpenEnvClientError::WebSocket(error));
                        }
                        None => return self.fail_closed(OpenEnvClientError::Closed),
                    };
                    match frame {
                        Message::Ping(payload) => {
                            if let Err(error) = self.socket.send(Message::Pong(payload)).await {
                                return self.fail_closed(OpenEnvClientError::WebSocket(error));
                            }
                        }
                        Message::Pong(_) => {}
                        Message::Close(_) => {
                            return self.fail_closed(OpenEnvClientError::Closed);
                        }
                        Message::Text(text) => {
                            if text.len() > OPENENV_MAX_SERVER_MESSAGE_BYTES {
                                return self.fail_closed(OpenEnvClientError::MessageTooLarge(text.len()));
                            }
                            if self.credential_marker.as_ref().is_some_and(|marker| {
                                contains_credential(text.as_bytes(), marker.as_slice())
                            }) {
                                return self.fail_closed(OpenEnvClientError::CredentialReflected);
                            }
                            return self.fail_closed(
                                OpenEnvClientError::UnsolicitedApplicationMessage,
                            );
                        }
                        Message::Binary(_) => {
                            return self.fail_closed(OpenEnvClientError::BinaryFrame);
                        }
                        Message::Frame(_) => {
                            return self.fail_closed(OpenEnvClientError::InvalidMessage(
                                "OpenEnv peer exposed an unexpected raw WebSocket frame".to_string(),
                            ));
                        }
                    }
                }
                output = &mut work => return Ok(output),
            }
        }
    }

    /// Whether this handle has observed a terminal or ambiguous transport
    /// outcome and therefore cannot safely issue another OpenEnv request.
    pub fn is_closed(&self) -> bool {
        self.closed
    }

    /// Send OpenEnv's close message best-effort and await no application
    /// response, as required by 8.B.17.
    pub async fn close(mut self) -> Result<(), OpenEnvClientError> {
        if !self.closed {
            let payload = serde_json::to_string(&OpenEnvClientMessage::Close)
                .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()))?;
            self.ensure_client_message_limit(payload.len())?;
            self.socket.send(Message::Text(payload.into())).await?;
            self.closed = true;
        }
        let _ = self.socket.close(None).await;
        Ok(())
    }

    async fn exchange(
        &mut self,
        request: OpenEnvClientMessage<'_>,
    ) -> Result<OpenEnvServerMessage, OpenEnvClientError> {
        if self.closed {
            return Err(OpenEnvClientError::Closed);
        }
        let payload = serde_json::to_string(&request)
            .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()))?;
        self.ensure_client_message_limit(payload.len())?;
        if let Err(error) = self.socket.send(Message::Text(payload.into())).await {
            return self.fail_closed(OpenEnvClientError::WebSocket(error));
        }
        let response =
            match tokio::time::timeout(self.request_timeout, self.read_application_message()).await
            {
                Ok(Ok(response)) => response,
                Ok(Err(error)) => return self.fail_closed(error),
                Err(_) => {
                    return self.fail_closed(OpenEnvClientError::Timeout(self.request_timeout));
                }
            };
        if let OpenEnvServerMessage::Error(error) = response {
            if error.code.is_terminal() {
                self.closed = true;
            }
            return Err(OpenEnvClientError::Protocol(error));
        }
        Ok(response)
    }

    async fn read_application_message(
        &mut self,
    ) -> Result<OpenEnvServerMessage, OpenEnvClientError> {
        loop {
            let Some(frame) = self.socket.next().await else {
                self.closed = true;
                return Err(OpenEnvClientError::Closed);
            };
            match frame? {
                Message::Text(text) => {
                    if text.len() > OPENENV_MAX_SERVER_MESSAGE_BYTES {
                        return Err(OpenEnvClientError::MessageTooLarge(text.len()));
                    }
                    if self.credential_marker.as_ref().is_some_and(|marker| {
                        contains_credential(text.as_bytes(), marker.as_slice())
                    }) {
                        return Err(OpenEnvClientError::CredentialReflected);
                    }
                    return serde_json::from_str(&text)
                        .map_err(|error| OpenEnvClientError::InvalidMessage(error.to_string()));
                }
                Message::Ping(payload) => {
                    self.socket.send(Message::Pong(payload)).await?;
                }
                Message::Pong(_) => {}
                Message::Close(_) => {
                    self.closed = true;
                    return Err(OpenEnvClientError::Closed);
                }
                Message::Binary(_) => return Err(OpenEnvClientError::BinaryFrame),
                Message::Frame(_) => {}
            }
        }
    }

    fn ensure_client_message_limit(&self, bytes: usize) -> Result<(), OpenEnvClientError> {
        if bytes > OPENENV_MAX_CLIENT_MESSAGE_BYTES {
            return Err(OpenEnvClientError::MessageTooLarge(bytes));
        }
        Ok(())
    }

    fn expect_observation(
        &mut self,
        response: OpenEnvServerMessage,
    ) -> Result<OpenEnvObservation, OpenEnvClientError> {
        match response {
            OpenEnvServerMessage::Observation(observation) => Ok(observation),
            other => self.fail_closed(unexpected("observation", &other)),
        }
    }

    fn fail_closed<T>(&mut self, error: OpenEnvClientError) -> Result<T, OpenEnvClientError> {
        self.closed = true;
        Err(error)
    }
}

fn bearer_token_bytes(authorization: Option<&HeaderValue>) -> Option<&[u8]> {
    authorization?
        .as_bytes()
        .strip_prefix(b"Bearer ")
        .filter(|token| !token.is_empty())
}

fn credential_transport_is_secure(base_url: &str) -> bool {
    let Ok(url) = reqwest::Url::parse(base_url) else {
        return false;
    };
    if url.scheme() == "https" {
        return true;
    }
    url.host_str().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

fn credential_reflected(payload: &[u8], authorization: Option<&HeaderValue>) -> bool {
    bearer_token_bytes(authorization)
        .is_some_and(|credential| contains_credential(payload, credential))
}

fn contains_credential(payload: &[u8], credential: &[u8]) -> bool {
    !credential.is_empty()
        && payload.len() >= credential.len()
        && payload
            .windows(credential.len())
            .any(|candidate| candidate == credential)
}

fn unexpected(expected: &'static str, message: &OpenEnvServerMessage) -> OpenEnvClientError {
    let actual = match message {
        OpenEnvServerMessage::Observation(_) => "observation",
        OpenEnvServerMessage::State(_) => "state",
        OpenEnvServerMessage::Error(_) => "error",
        OpenEnvServerMessage::Mcp(_) => "mcp",
    };
    OpenEnvClientError::UnexpectedResponse { expected, actual }
}

async fn http_status_error(
    endpoint: String,
    response: reqwest::Response,
    authenticated: bool,
) -> OpenEnvClientError {
    let status = response.status();
    let body = if authenticated {
        "<authenticated response body redacted>".to_string()
    } else {
        match read_http_body_bounded(&endpoint, response).await {
            Ok(bytes) => String::from_utf8_lossy(&bytes).into_owned(),
            Err(OpenEnvClientError::HttpBodyTooLarge { limit, .. }) => {
                format!("<body exceeded {limit} bytes>")
            }
            Err(error) => format!("<failed to read body: {error}>"),
        }
    };
    OpenEnvClientError::HttpStatus {
        endpoint,
        status,
        body,
    }
}

async fn read_http_body_bounded(
    endpoint: &str,
    mut response: reqwest::Response,
) -> Result<Vec<u8>, OpenEnvClientError> {
    if response
        .content_length()
        .is_some_and(|length| length > OPENENV_MAX_DISCOVERY_BYTES as u64)
    {
        return Err(OpenEnvClientError::HttpBodyTooLarge {
            endpoint: endpoint.to_string(),
            limit: OPENENV_MAX_DISCOVERY_BYTES,
        });
    }
    let mut body = Vec::new();
    while let Some(chunk) = response.chunk().await? {
        let next_len = body.len().checked_add(chunk.len()).ok_or_else(|| {
            OpenEnvClientError::HttpBodyTooLarge {
                endpoint: endpoint.to_string(),
                limit: OPENENV_MAX_DISCOVERY_BYTES,
            }
        })?;
        if next_len > OPENENV_MAX_DISCOVERY_BYTES {
            return Err(OpenEnvClientError::HttpBodyTooLarge {
                endpoint: endpoint.to_string(),
                limit: OPENENV_MAX_DISCOVERY_BYTES,
            });
        }
        body.extend_from_slice(&chunk);
    }
    Ok(body)
}

fn normalize_urls(input: &str) -> Result<(String, String), OpenEnvClientError> {
    let trimmed = input.trim().trim_end_matches('/');
    if trimmed.is_empty() {
        return Err(OpenEnvClientError::InvalidBaseUrl(
            "URL must not be empty".to_string(),
        ));
    }
    let with_scheme = if trimmed.contains("://") {
        trimmed.to_string()
    } else {
        format!("http://{trimmed}")
    };
    let mut base = reqwest::Url::parse(&with_scheme)
        .with_context(|| format!("parse {with_scheme:?}"))
        .map_err(|error| OpenEnvClientError::InvalidBaseUrl(error.to_string()))?;
    if !matches!(base.scheme(), "http" | "https") {
        return Err(OpenEnvClientError::InvalidBaseUrl(format!(
            "scheme must be http or https, got {:?}",
            base.scheme()
        )));
    }
    if base.host_str().is_none()
        || !base.username().is_empty()
        || base.password().is_some()
        || base.query().is_some()
        || base.fragment().is_some()
    {
        return Err(OpenEnvClientError::InvalidBaseUrl(
            "URL must name a host and must not contain credentials, a query, or a fragment"
                .to_string(),
        ));
    }
    let path = base.path().trim_end_matches('/').to_string();
    base.set_path(&path);
    let mut websocket = base.clone();
    websocket
        .set_scheme(if base.scheme() == "https" {
            "wss"
        } else {
            "ws"
        })
        .map_err(|_| {
            OpenEnvClientError::InvalidBaseUrl("could not map URL to WebSocket".to_string())
        })?;
    let ws_path = format!("{}/ws", websocket.path().trim_end_matches('/'));
    websocket.set_path(&ws_path);
    Ok((
        base.as_str().trim_end_matches('/').to_string(),
        websocket.as_str().to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Json, Router,
        extract::{Path, State},
        http::StatusCode as AxumStatusCode,
        response::{IntoResponse, Response},
        routing::{get, post},
    };
    use serde_json::json;
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };

    async fn task_splits(Path(environment): Path<String>) -> Response {
        if environment == "unsupported" {
            return (
                AxumStatusCode::NOT_IMPLEMENTED,
                Json(json!({
                    "detail": "list_splits is not supported for this environment"
                })),
            )
                .into_response();
        }
        Json(json!([
            {"name": "train", "type": "train"},
            {"name": "test", "type": "test"},
            {"name": "holdout", "type": "validation", "provider": "reference"}
        ]))
        .into_response()
    }

    async fn task_post(
        Path((_environment, operation)): Path<(String, String)>,
        Json(body): Json<Value>,
    ) -> Response {
        match operation.as_str() {
            "tasks" => {
                assert_eq!(body, json!({"split": "train"}));
                Json(json!({
                    "tasks": [
                        {"id": 0, "prompt": "1 + 1", "answer": "2"},
                        {"id": 1, "prompt": "2 + 2", "answer": "4"},
                        {"id": 2, "prompt": "3 + 3", "answer": "6"}
                    ],
                    "env_name": "task_env"
                }))
                .into_response()
            }
            "num_tasks" => {
                assert_eq!(body, json!({"split": "train"}));
                Json(json!({"num_tasks": 3})).into_response()
            }
            "task" => {
                if body == json!({"split": "train", "index": 99}) {
                    return (
                        AxumStatusCode::BAD_REQUEST,
                        Json(json!({"detail": "Invalid task index"})),
                    )
                        .into_response();
                }
                assert_eq!(body, json!({"split": "train", "index": 1}));
                Json(json!({
                    "task": {"id": 1, "prompt": "2 + 2", "answer": "4"}
                }))
                .into_response()
            }
            "task_range" => {
                assert_eq!(body, json!({"split": "train", "start": 1, "stop": 3}));
                Json(json!({
                    "tasks": [
                        {"id": 1, "prompt": "2 + 2", "answer": "4"},
                        {"id": 2, "prompt": "3 + 3", "answer": "6"}
                    ]
                }))
                .into_response()
            }
            _ => AxumStatusCode::NOT_FOUND.into_response(),
        }
    }

    async fn task_fixture() -> (String, tokio::task::JoinHandle<()>) {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let app = Router::new()
            .route(
                "/prefix/list_environments",
                get(|| async { Json(json!(["task_env"])) }),
            )
            .route("/prefix/{environment}/splits", get(task_splits))
            .route("/prefix/{environment}/{operation}", post(task_post));
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}/prefix"), server)
    }

    async fn mutable_discovery_fixture() -> (String, Arc<AtomicBool>, tokio::task::JoinHandle<()>) {
        let changed = Arc::new(AtomicBool::new(false));
        let app = Router::new()
            .route("/health", get(|| async { AxumStatusCode::OK }))
            .route(
                "/metadata",
                get(|State(changed): State<Arc<AtomicBool>>| async move {
                    Json(json!({
                        "name": "MutableEnvironment",
                        "description": if changed.load(Ordering::Relaxed) {
                            "deployment two"
                        } else {
                            "deployment one"
                        },
                        "version": if changed.load(Ordering::Relaxed) { "2" } else { "1" }
                    }))
                }),
            )
            .route(
                "/schema",
                get(|| async {
                    Json(json!({
                        "action": {"type": "object"},
                        "observation": {"type": "object"},
                        "state": {"type": "object"}
                    }))
                }),
            )
            .route(
                "/list_environments",
                get(|| async { Json(json!(["mutable"])) }),
            )
            .route(
                "/openapi.json",
                get(|| async { Json(json!({"info": {"version": "1.0"}})) }),
            )
            .with_state(changed.clone());
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{address}"), changed, server)
    }

    fn inspection_fixture() -> OpenEnvInspection {
        OpenEnvInspection {
            identity: OpenEnvIdentity {
                schema: "kiln.openenv-identity.v1".into(),
                client_profile: OPENENV_CLIENT_PROFILE.into(),
                base_url: "https://environment.example/openenv".into(),
                websocket_url: "wss://environment.example/openenv/ws".into(),
                authentication: OpenEnvAuthentication::Bearer,
                openapi_version: Some("1.0".into()),
                environments: vec!["arcade".into()],
                metadata: OpenEnvMetadata {
                    name: "Arcade".into(),
                    description: "Stable environment".into(),
                    readme_content: None,
                    version: Some("17".into()),
                    author: None,
                    documentation_url: None,
                },
                schema_sha256: format!("sha256:{}", "a".repeat(64)),
            },
            schema: OpenEnvSchema {
                action: json!({"type": "object", "required": ["move"]}),
                observation: json!({"type": "object"}),
                state: json!({"type": "object"}),
            },
        }
    }

    #[test]
    fn url_derivation_matches_openenv_client_requirement() {
        let client = OpenEnvClient::new("127.0.0.1:8000/").unwrap();
        assert_eq!(client.base_url(), "http://127.0.0.1:8000");
        assert_eq!(client.websocket_url(), "ws://127.0.0.1:8000/ws");

        let client = OpenEnvClient::new("https://example.test/prefix///").unwrap();
        assert_eq!(client.base_url(), "https://example.test/prefix");
        assert_eq!(client.websocket_url(), "wss://example.test/prefix/ws");
        assert!(OpenEnvClient::new("ftp://example.test").is_err());
        assert!(OpenEnvClient::new("http://token@example.test").is_err());
        assert!(OpenEnvClient::new("http://example.test?token=secret").is_err());
    }

    #[test]
    fn environment_identity_revalidation_is_exact_and_bounded() {
        let expected = inspection_fixture();
        expected.ensure_unchanged(&expected).unwrap();

        let mut current = expected.clone();
        current.identity.metadata.description = "secret changed description".into();
        current.schema.action = json!({"type": "object", "required": ["answer"]});
        current.identity.schema_sha256 = format!("sha256:{}", "b".repeat(64));
        let error = expected.ensure_unchanged(&current).unwrap_err();
        let OpenEnvClientError::EnvironmentIdentityChanged {
            endpoint,
            expected_schema_sha256,
            actual_schema_sha256,
            changed_fields,
        } = &error
        else {
            panic!("expected environment identity drift, got {error:?}");
        };
        assert_eq!(endpoint, "https://environment.example/openenv");
        assert_eq!(
            expected_schema_sha256,
            &format!("sha256:{}", "a".repeat(64))
        );
        assert_eq!(actual_schema_sha256, &format!("sha256:{}", "b".repeat(64)));
        assert_eq!(
            changed_fields,
            &[
                "identity.metadata",
                "identity.schema_sha256",
                "schema.action"
            ]
        );
        assert!(!error.to_string().contains("secret changed description"));
    }

    #[tokio::test]
    async fn client_revalidation_detects_a_redeployed_discovery_surface() {
        let (base_url, changed, server) = mutable_discovery_fixture().await;
        let client = OpenEnvClient::new(base_url).unwrap();
        let expected = client.inspect().await.unwrap();
        client.revalidate(&expected).await.unwrap();

        changed.store(true, Ordering::Relaxed);
        let error = client.revalidate(&expected).await.unwrap_err();
        assert!(matches!(
            &error,
            OpenEnvClientError::EnvironmentIdentityChanged { changed_fields, .. }
                if changed_fields == &["identity.metadata"]
        ));
        assert!(!error.to_string().contains("deployment two"));
        server.abort();
    }

    #[tokio::test]
    async fn inspection_rejects_an_unenforceable_action_schema_before_sessions() {
        let secret = "INVALID_SCHEMA_VALUE_MUST_NOT_LEAK";
        let app = Router::new()
            .route("/health", get(|| async { AxumStatusCode::OK }))
            .route(
                "/metadata",
                get(|| async {
                    Json(json!({
                        "name": "InvalidSchemaEnvironment",
                        "description": "fixture"
                    }))
                }),
            )
            .route(
                "/schema",
                get(move || async move {
                    Json(json!({
                        "action": {"type": secret},
                        "observation": {"type": "object"},
                        "state": {"type": "object"}
                    }))
                }),
            )
            .route(
                "/list_environments",
                get(|| async { Json(json!(["invalid_schema"])) }),
            )
            .route(
                "/openapi.json",
                get(|| async { Json(json!({"info": {"version": "1.0"}})) }),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        let error = OpenEnvClient::new(format!("http://{address}"))
            .unwrap()
            .inspect()
            .await
            .unwrap_err();
        assert!(matches!(
            error,
            OpenEnvClientError::InvalidActionSchema { .. }
        ));
        assert!(!error.to_string().contains(secret));
        server.abort();
    }

    #[test]
    fn bearer_credentials_are_validated_and_redacted_from_debug() {
        let secret = "super-secret-openenv-token";
        let client = OpenEnvClient::new("https://example.test/prefix")
            .unwrap()
            .with_bearer_token(secret)
            .unwrap();
        let debug = format!("{client:?}");
        assert!(debug.contains("authenticated: true"));
        assert!(!debug.contains(secret));
        assert!(
            OpenEnvClient::new("https://example.test")
                .unwrap()
                .with_bearer_token(" \t")
                .is_err()
        );
        assert!(
            OpenEnvClient::new("https://example.test")
                .unwrap()
                .with_bearer_token("line-one\nline-two")
                .is_err()
        );
        assert!(
            OpenEnvClient::new("https://example.test")
                .unwrap()
                .with_bearer_token(" token-with-leading-space")
                .is_err()
        );
        assert!(
            OpenEnvClient::new("https://example.test")
                .unwrap()
                .with_bearer_token("token with internal space")
                .is_err()
        );
        assert!(matches!(
            OpenEnvClient::new("http://environment.example")
                .unwrap()
                .with_bearer_token(secret),
            Err(OpenEnvClientError::InsecureCredentialTransport)
        ));
        assert!(
            OpenEnvClient::new("http://[::1]:8990")
                .unwrap()
                .with_bearer_token(secret)
                .is_ok()
        );
    }

    #[test]
    fn task_paths_encode_untrusted_environment_names_as_one_segment() {
        let client = OpenEnvClient::new("https://example.test/prefix").unwrap();
        assert_eq!(
            client
                .endpoint_segments(&["task env/β?#", "splits"])
                .unwrap(),
            "https://example.test/prefix/task%20env%2F%CE%B2%3F%23/splits"
        );
        assert!(matches!(
            validate_task_selector("environment name", ""),
            Err(OpenEnvClientError::InvalidTaskSelector { .. })
        ));
    }

    #[tokio::test]
    async fn task_api_preserves_reference_success_shapes_and_unsupported_status() {
        let (base_url, server) = task_fixture().await;
        let client = OpenEnvClient::new(base_url).unwrap();

        let splits = client.list_task_splits("task_env").await.unwrap();
        assert_eq!(
            splits
                .iter()
                .map(|split| (split.name.as_str(), split.split_type.as_str()))
                .collect::<Vec<_>>(),
            [
                ("train", "train"),
                ("test", "test"),
                ("holdout", "validation"),
            ]
        );
        assert_eq!(splits[2].extra["provider"], "reference");

        let tasks = client.list_tasks("task_env", "train").await.unwrap();
        assert_eq!(tasks.env_name, "task_env");
        assert_eq!(tasks.tasks.len(), 3);
        assert_eq!(
            client
                .num_tasks("task_env", "train")
                .await
                .unwrap()
                .num_tasks,
            3
        );
        assert_eq!(
            client.get_task("task_env", "train", 1).await.unwrap().task,
            json!({"id": 1, "prompt": "2 + 2", "answer": "4"})
        );
        let invalid_index = client.get_task("task_env", "train", 99).await.unwrap_err();
        assert_eq!(invalid_index.http_status_code(), Some(400));
        assert!(invalid_index.to_string().contains("Invalid task index"));
        assert_eq!(
            client
                .get_task_range("task_env", "train", Some(1), Some(3))
                .await
                .unwrap()
                .tasks,
            [
                json!({"id": 1, "prompt": "2 + 2", "answer": "4"}),
                json!({"id": 2, "prompt": "3 + 3", "answer": "6"}),
            ]
        );

        let unsupported = client.list_task_splits("unsupported").await.unwrap_err();
        assert_eq!(unsupported.http_status_code(), Some(501));
        assert!(unsupported.is_task_api_unsupported());

        let catalog = client
            .task_catalog(None, Some("TRAIN"), 1, 50)
            .await
            .unwrap();
        assert_eq!(catalog.environment_name, "task_env");
        assert_eq!(catalog.task_api, OpenEnvTaskApiSupport::Available);
        assert_eq!(catalog.selected_split.as_deref(), Some("train"));
        assert_eq!(catalog.num_tasks, Some(3));
        assert_eq!((catalog.start, catalog.stop), (Some(1), Some(3)));
        assert_eq!(catalog.tasks.len(), 2);
        server.abort();
    }

    #[test]
    fn authenticated_websocket_rejection_exposes_only_its_status() {
        let error = OpenEnvClientError::AuthenticatedWebSocketStatus(StatusCode::UNAUTHORIZED);
        assert_eq!(error.http_status_code(), Some(401));
        assert!(!error.to_string().contains("credential"));
    }
}
