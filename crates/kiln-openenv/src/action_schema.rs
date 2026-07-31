//! Compiled, self-contained validation for environment-advertised actions.

use serde_json::Value;

const MAX_VALIDATION_ISSUES: usize = 4;
const MAX_POINTER_BYTES: usize = 192;

/// One bounded, payload-free JSON Schema validation diagnostic.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OpenEnvActionValidationIssue {
    pub keyword: String,
    pub instance_path: String,
    pub schema_path: String,
}

impl std::fmt::Display for OpenEnvActionValidationIssue {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            formatter,
            "{} at action {} (schema {})",
            self.keyword, self.instance_path, self.schema_path
        )
    }
}

/// The environment advertised an action schema that cannot be compiled safely.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("advertised action JSON Schema could not be compiled: {issue}")]
pub struct OpenEnvActionSchemaError {
    pub issue: OpenEnvActionValidationIssue,
}

/// A model action did not satisfy the environment's advertised action schema.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("model action failed advertised JSON Schema validation: {summary}")]
pub struct OpenEnvActionValidationError {
    pub issues: Vec<OpenEnvActionValidationIssue>,
    pub truncated: bool,
    summary: String,
}

/// A reusable validator compiled once from one OpenEnv discovery identity.
///
/// `jsonschema` is built without its HTTP and filesystem resolvers. Internal
/// references remain supported, while external references fail compilation.
#[derive(Clone, Debug)]
pub struct OpenEnvActionValidator {
    validator: jsonschema::Validator,
}

impl OpenEnvActionValidator {
    pub fn compile(schema: &Value) -> Result<Self, OpenEnvActionSchemaError> {
        jsonschema::validator_for(schema)
            .map(|validator| Self { validator })
            .map_err(|error| OpenEnvActionSchemaError {
                issue: safe_issue(&error),
            })
    }

    pub fn validate(&self, action: &Value) -> Result<(), OpenEnvActionValidationError> {
        let mut errors = self.validator.iter_errors(action);
        let mut issues = errors
            .by_ref()
            .take(MAX_VALIDATION_ISSUES)
            .map(|error| safe_issue(&error))
            .collect::<Vec<_>>();
        if issues.is_empty() {
            return Ok(());
        }
        let truncated = errors.next().is_some();
        let mut summary = issues
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join("; ");
        if truncated {
            summary.push_str("; additional issues omitted");
        }
        // Keep a non-empty issue vector even if future refactoring changes
        // construction above; it is useful machine-readable evidence.
        issues.truncate(MAX_VALIDATION_ISSUES);
        Err(OpenEnvActionValidationError {
            issues,
            truncated,
            summary,
        })
    }
}

fn safe_issue(error: &jsonschema::ValidationError<'_>) -> OpenEnvActionValidationIssue {
    OpenEnvActionValidationIssue {
        keyword: bounded(error.kind().keyword(), MAX_POINTER_BYTES),
        instance_path: pointer_or_root(error.instance_path().as_str()),
        schema_path: pointer_or_root(error.schema_path().as_str()),
    }
}

fn pointer_or_root(pointer: &str) -> String {
    if pointer.is_empty() {
        "$".to_string()
    } else {
        bounded(pointer, MAX_POINTER_BYTES)
    }
}

fn bounded(value: &str, limit: usize) -> String {
    if value.len() <= limit {
        return value.to_string();
    }
    let mut end = limit.saturating_sub(3);
    while !value.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    format!("{}...", &value[..end])
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn representative_validator() -> OpenEnvActionValidator {
        OpenEnvActionValidator::compile(&json!({
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "type": "object",
            "properties": {
                "answer": {"type": "integer", "minimum": 1, "maximum": 4}
            },
            "required": ["answer"],
            "additionalProperties": false
        }))
        .unwrap()
    }

    #[test]
    fn accepts_actions_that_satisfy_the_advertised_schema() {
        representative_validator()
            .validate(&json!({"answer": 3}))
            .unwrap();
    }

    #[test]
    fn reports_only_bounded_keywords_and_json_pointers() {
        let secret = "NEVER_ECHO_THIS_MODEL_VALUE";
        let error = representative_validator()
            .validate(&json!({"answer": secret, "private": secret}))
            .unwrap_err();
        let rendered = error.to_string();
        assert!(error.issues.len() <= MAX_VALIDATION_ISSUES);
        assert!(error.issues.iter().any(|issue| issue.keyword == "type"));
        assert!(
            error
                .issues
                .iter()
                .any(|issue| issue.keyword == "additionalProperties")
        );
        assert!(!rendered.contains(secret));
    }

    #[test]
    fn rejects_invalid_and_external_schemas_without_echoing_them() {
        let invalid = OpenEnvActionValidator::compile(&json!({"type": "SECRET_TYPE"}))
            .unwrap_err()
            .to_string();
        assert!(!invalid.contains("SECRET_TYPE"));

        for external_uri in [
            "file:///etc/OPENENV_SHOULD_NOT_READ_THIS",
            "https://example.invalid/OPENENV_SHOULD_NOT_FETCH_THIS.json",
        ] {
            let external = OpenEnvActionValidator::compile(&json!({"$ref": external_uri}))
                .unwrap_err()
                .to_string();
            assert!(!external.contains(external_uri));
        }
    }

    #[test]
    fn supports_self_contained_internal_references() {
        let validator = OpenEnvActionValidator::compile(&json!({
            "$defs": {"move": {"enum": ["left", "right"]}},
            "type": "object",
            "properties": {"move": {"$ref": "#/$defs/move"}},
            "required": ["move"]
        }))
        .unwrap();
        validator.validate(&json!({"move": "left"})).unwrap();
        assert_eq!(
            validator
                .validate(&json!({"move": "SECRET_MOVE"}))
                .unwrap_err()
                .issues[0]
                .keyword,
            "enum"
        );
    }
}
