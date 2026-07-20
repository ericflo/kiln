//! Narrow process-environment credential-provider boundary.
//!
//! Typed server configuration owns each secret variable name and exact
//! authorized origin. This adapter resolves only the named bearer secret and
//! never serializes, logs, caches, or exposes its value.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CredentialLookupError {
    Unavailable,
    Empty,
}

/// Check that a configured bearer-secret source is currently usable without
/// retaining or exposing the secret value.
pub fn validate_bearer_secret_environment(name: &str) -> Result<(), CredentialLookupError> {
    bearer_secret_from_environment(name).map(drop)
}

pub(crate) fn bearer_secret_from_environment(name: &str) -> Result<String, CredentialLookupError> {
    let secret = std::env::var(name).map_err(|_| CredentialLookupError::Unavailable)?;
    if secret.trim().is_empty() {
        return Err(CredentialLookupError::Empty);
    }
    Ok(secret)
}
