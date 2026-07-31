//! Async OpenEnv discovery and stateful episode client.

use std::time::Duration;

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
    OPENENV_MAX_SERVER_MESSAGE_BYTES, OpenEnvClientMessage, OpenEnvMetadata, OpenEnvObservation,
    OpenEnvProtocolError, OpenEnvSchema, OpenEnvServerMessage,
};

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
        Ok(OpenEnvInspection {
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
        })
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
        expect_observation(response)
    }

    pub async fn step(&mut self, action: &Value) -> Result<OpenEnvObservation, OpenEnvClientError> {
        let response = self
            .exchange(OpenEnvClientMessage::Step { data: action })
            .await?;
        expect_observation(response)
    }

    pub async fn state(&mut self) -> Result<Value, OpenEnvClientError> {
        let response = self.exchange(OpenEnvClientMessage::State).await?;
        match response {
            OpenEnvServerMessage::State(state) => Ok(state),
            other => Err(unexpected("state", &other)),
        }
    }

    pub async fn mcp(&mut self, request: &Value) -> Result<Value, OpenEnvClientError> {
        let response = self
            .exchange(OpenEnvClientMessage::Mcp { data: request })
            .await?;
        match response {
            OpenEnvServerMessage::Mcp(value) => Ok(value),
            other => Err(unexpected("mcp", &other)),
        }
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
        self.socket.send(Message::Text(payload.into())).await?;
        let response = tokio::time::timeout(self.request_timeout, self.read_application_message())
            .await
            .map_err(|_| OpenEnvClientError::Timeout(self.request_timeout))??;
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

fn expect_observation(
    response: OpenEnvServerMessage,
) -> Result<OpenEnvObservation, OpenEnvClientError> {
    match response {
        OpenEnvServerMessage::Observation(observation) => Ok(observation),
        other => Err(unexpected("observation", &other)),
    }
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
}
