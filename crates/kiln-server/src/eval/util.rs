//! Cross-cutting helpers shared across the eval subsystem.
//!
//! Consolidating these here keeps the per-registry modules focused on their
//! own data layout and removes the temptation to subtly diverge a name
//! validator (which previously lived in three sibling files).

use std::path::Path;

/// True when `name` is a safe single-segment identifier — non-empty, no
/// path separators, no `..`, not absolute. Mirrors the rules used by
/// `api::adapters::validate_adapter_name`; identical errors but each
/// caller maps to its own typed error since the error enums differ.
pub fn is_valid_segment_name(name: &str) -> bool {
    if name.is_empty() || name == "." || name == ".." {
        return false;
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        return false;
    }
    if Path::new(name).is_absolute() {
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accepts_alphanumeric_names() {
        assert!(is_valid_segment_name("smoke"));
        assert!(is_valid_segment_name("math-200"));
        assert!(is_valid_segment_name("my_dataset.v1"));
    }

    #[test]
    fn rejects_path_separators_and_traversal() {
        assert!(!is_valid_segment_name(""));
        assert!(!is_valid_segment_name("."));
        assert!(!is_valid_segment_name(".."));
        assert!(!is_valid_segment_name("a/b"));
        assert!(!is_valid_segment_name("a\\b"));
        assert!(!is_valid_segment_name("a..b"));
        assert!(!is_valid_segment_name("/abs"));
    }
}
