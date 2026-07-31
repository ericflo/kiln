//! Server-owned credentials for protected OpenEnv origins.
//!
//! The control plane accepts only opaque credential handles. This module binds
//! each handle to one exact origin and to the name of a trusted process
//! environment variable; bearer values are resolved only at client
//! construction and never enter typed configuration, run state, or artifacts.

use crate::config::{
    ConfigValueSource, EffectiveConfigurationField, validate_secret_environment_name,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// `[openenv]` server-owned rollout orchestration policy.
///
/// CLI-driven OpenEnv workflows remain available independently. These limits
/// govern only the asynchronous HTTP/dashboard control plane.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct OpenEnvConfig {
    /// Enable OpenEnv discovery and run lifecycle API routes.
    pub enabled: bool,
    /// Maximum concurrent rollout collectors. Training jobs submitted by a
    /// run subsequently use the normal bounded training queue.
    pub max_active_runs: usize,
    /// Maximum in-memory tracked run records, including terminal runs.
    pub max_tracked_runs: usize,
    /// Retention window for terminal run records.
    pub tracked_run_ttl_secs: u64,
    /// Permit the server control plane to connect to non-loopback OpenEnv
    /// origins. Disabled by default to make the HTTP API safe against SSRF;
    /// the local CLI remains unrestricted.
    pub allow_remote_environments: bool,
    /// Origin-scoped bearer credential handles accepted by OpenEnv run and
    /// inspection requests. Values name secret environment variables; bearer
    /// values never enter typed config, AppState, status, or artifacts.
    pub credentials: BTreeMap<String, OpenEnvCredentialConfig>,
}

/// One server-owned OpenEnv bearer credential and its exact authorized origin.
#[derive(Clone, Deserialize, Serialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct OpenEnvCredentialConfig {
    /// Canonical `scheme://host[:port]` origin. HTTPS is required unless the
    /// host is loopback.
    pub origin: String,
    /// Trusted environment-variable name containing the bearer token.
    pub bearer_token_env: String,
}

impl std::fmt::Debug for OpenEnvCredentialConfig {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("OpenEnvCredentialConfig")
            .field("origin", &self.origin)
            .field("bearer_token_env", &"<redacted>")
            .finish()
    }
}

impl Default for OpenEnvConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_active_runs: 4,
            max_tracked_runs: 128,
            tracked_run_ttl_secs: 604_800,
            allow_remote_environments: false,
            credentials: BTreeMap::new(),
        }
    }
}

impl OpenEnvConfig {
    /// Resolve API-visible credential handles to one credential source per
    /// environment URL. Empty input means every endpoint is unauthenticated.
    ///
    /// Handles are exact-origin scoped. Only the environment-variable name
    /// leaves this method; the secret itself is resolved immediately before
    /// client construction and is never persisted in run state.
    pub fn resolve_credential_envs(
        &self,
        credential_ids: &[Option<String>],
        environment_urls: &[String],
    ) -> std::result::Result<Vec<Option<String>>, String> {
        if credential_ids.is_empty() {
            return Ok(vec![None; environment_urls.len()]);
        }
        if credential_ids.len() != environment_urls.len() {
            return Err(format!(
                "credential_ids must be empty or contain exactly one entry per environment URL (expected {}, got {})",
                environment_urls.len(),
                credential_ids.len()
            ));
        }
        credential_ids
            .iter()
            .zip(environment_urls)
            .map(|(credential_id, environment_url)| {
                let Some(credential_id) = credential_id.as_deref() else {
                    return Ok(None);
                };
                validate_openenv_credential_id(credential_id)?;
                let credential = self.credentials.get(credential_id).ok_or_else(|| {
                    format!(
                        "OpenEnv credential_id {credential_id:?} is not configured on this server"
                    )
                })?;
                credential.validate_definition(credential_id)?;
                let requested_origin = canonical_openenv_origin(environment_url)?;
                if credential.origin != requested_origin {
                    return Err(format!(
                        "OpenEnv credential_id {credential_id:?} is not authorized for origin {requested_origin:?}"
                    ));
                }
                if kiln_train::validate_bearer_secret_environment(&credential.bearer_token_env)
                    .is_err()
                {
                    return Err(format!(
                        "OpenEnv credential_id {credential_id:?} is unavailable because its server-configured secret is missing or empty"
                    ));
                }
                Ok(Some(credential.bearer_token_env.clone()))
            })
            .collect()
    }

    pub(crate) fn validate_credentials(&self) -> Result<()> {
        for (credential_id, credential) in &self.credentials {
            validate_openenv_credential_id(credential_id)
                .map_err(anyhow::Error::msg)
                .with_context(|| {
                    format!("invalid openenv.credentials.{credential_id} credential id")
                })?;
            credential
                .validate_definition(credential_id)
                .map_err(anyhow::Error::msg)?;
            kiln_train::validate_bearer_secret_environment(&credential.bearer_token_env).map_err(
                |error| {
                    let detail = match error {
                        kiln_train::CredentialLookupError::Unavailable => "is not set",
                        kiln_train::CredentialLookupError::Empty => "has an empty value",
                    };
                    anyhow::anyhow!(
                        "openenv.credentials.{credential_id}.bearer_token_env names an environment variable that {detail}"
                    )
                },
            )?;
        }
        Ok(())
    }

    pub(crate) fn insert_effective_configuration_fields(
        &self,
        value_sources: &BTreeMap<String, ConfigValueSource>,
        fields: &mut BTreeMap<String, EffectiveConfigurationField>,
    ) {
        for (credential_id, credential) in &self.credentials {
            for (field, value, redacted) in [
                (
                    "origin",
                    serde_json::Value::String(credential.origin.clone()),
                    false,
                ),
                ("bearer_token_env", serde_json::Value::Null, true),
            ] {
                let path = format!("openenv.credentials.{credential_id}.{field}");
                fields.insert(
                    path.clone(),
                    EffectiveConfigurationField {
                        effective_value: value,
                        source: value_sources
                            .get(&path)
                            .copied()
                            .unwrap_or(ConfigValueSource::Default),
                        canonical_environment: None,
                        compatibility_environment: Vec::new(),
                        redacted,
                        restart_required_to_change: true,
                    },
                );
            }
        }
    }
}

impl OpenEnvCredentialConfig {
    fn validate_definition(&self, credential_id: &str) -> std::result::Result<(), String> {
        validate_secret_environment_name(&self.bearer_token_env).map_err(|message| {
            format!("openenv.credentials.{credential_id}.bearer_token_env {message}")
        })?;
        let canonical = canonical_openenv_origin(&self.origin)?;
        if self.origin != canonical {
            return Err(format!(
                "openenv.credentials.{credential_id}.origin must be the exact canonical origin {canonical:?}"
            ));
        }
        let parsed = reqwest::Url::parse(&self.origin)
            .map_err(|error| format!("invalid OpenEnv credential origin: {error}"))?;
        let host = parsed
            .host_str()
            .ok_or_else(|| "OpenEnv credential origin must include a host".to_string())?;
        let loopback = host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|address| address.is_loopback());
        if parsed.scheme() != "https" && !loopback {
            return Err(format!(
                "openenv.credentials.{credential_id}.origin must use HTTPS unless its host is loopback"
            ));
        }
        Ok(())
    }
}

/// Canonical origin used for exact OpenEnv credential scoping.
pub fn canonical_openenv_origin(url: &str) -> std::result::Result<String, String> {
    let parsed = reqwest::Url::parse(url.trim())
        .map_err(|error| format!("OpenEnv URL {url:?} is invalid: {error}"))?;
    if !matches!(parsed.scheme(), "http" | "https")
        || parsed.host_str().is_none()
        || !parsed.username().is_empty()
        || parsed.password().is_some()
        || parsed.query().is_some()
        || parsed.fragment().is_some()
    {
        return Err(
            "OpenEnv URL must be absolute HTTP(S) without credentials, query, or fragment"
                .to_string(),
        );
    }
    Ok(parsed.origin().ascii_serialization())
}

pub fn validate_openenv_credential_id(id: &str) -> std::result::Result<(), String> {
    let valid = !id.is_empty()
        && id != "-"
        && id.len() <= 64
        && id
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'));
    if !valid {
        return Err(
            "OpenEnv credential_id must be 1..=64 ASCII letters, digits, '_' or '-', and '-' alone is reserved for a public CLI/dashboard slot".to_string(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::tests::ScopedConfigEnvironment;

    fn protected_config(origin: &str, secret_env: &str) -> OpenEnvConfig {
        let mut config = OpenEnvConfig::default();
        config.credentials.insert(
            "production".into(),
            OpenEnvCredentialConfig {
                origin: origin.into(),
                bearer_token_env: secret_env.into(),
            },
        );
        config
    }

    #[test]
    fn credentials_are_origin_scoped_and_secret_free() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        const ENV: &str = "KILN_TEST_SCOPED_OPENENV_SECRET";
        let environment = ScopedConfigEnvironment::isolated_with(&[ENV]);
        environment.set(ENV, "openenv-test-secret");
        let config = protected_config("https://environment.example.com:8443", ENV);
        config.validate_credentials().unwrap();
        assert!(!format!("{config:?}").contains(ENV));
        assert_eq!(
            config
                .resolve_credential_envs(
                    &[Some("production".to_string())],
                    &["https://environment.example.com:8443/team/arcade".to_string()]
                )
                .unwrap(),
            [Some(ENV.to_string())]
        );
        assert_eq!(
            config
                .resolve_credential_envs(&[], &["https://public.example.com".to_string()])
                .unwrap(),
            [None]
        );
        let error = config
            .resolve_credential_envs(
                &[Some("production".to_string())],
                &["https://other.example.com:8443".to_string()],
            )
            .unwrap_err();
        assert!(error.contains("not authorized"), "{error}");
        assert!(!error.contains(ENV), "credential internals leaked: {error}");
        assert!(
            config
                .resolve_credential_envs(
                    &[Some("production".to_string()), None],
                    &["https://environment.example.com:8443".to_string()]
                )
                .unwrap_err()
                .contains("exactly one")
        );
        assert!(validate_openenv_credential_id("-").is_err());
    }

    #[test]
    fn credentials_reject_cleartext_remote_origins_and_missing_secrets() {
        let _env_guard = crate::TEST_ENV_LOCK
            .lock()
            .unwrap_or_else(|error| error.into_inner());
        const ENV: &str = "KILN_TEST_MISSING_OPENENV_SECRET";
        let environment = ScopedConfigEnvironment::isolated_with(&[ENV]);
        let mut config = protected_config("http://environment.example.com", ENV);
        let error = config.validate_credentials().unwrap_err().to_string();
        assert!(error.contains("must use HTTPS"), "{error}");

        config.credentials.get_mut("production").unwrap().origin =
            "https://environment.example.com".into();
        let error = config.validate_credentials().unwrap_err().to_string();
        assert!(error.contains("not set"), "{error}");

        environment.set(ENV, " ");
        let error = config.validate_credentials().unwrap_err().to_string();
        assert!(error.contains("empty"), "{error}");

        environment.set(ENV, "secret");
        config.validate_credentials().unwrap();
    }
}
