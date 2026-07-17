//! Execution scope for the Qwen3-Next MTP draft block.
//!
//! The MTP block uses single-token self-attention even after the local draft
//! position advances. Attention helpers consult this thread-local scope to
//! bypass paged history and operate only on the current Q/K/V tensors.

use std::cell::Cell;

thread_local! {
    static SINGLE_TOKEN_SELF_ATTENTION_DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Guard that keeps MTP-only attention semantics confined to one draft block.
///
/// The depth counter makes nested entry well-defined, and `Drop` restores the
/// previous state on every return path, including error propagation and panic
/// unwinding.
#[must_use = "the scope must live for the duration of the MTP draft block"]
pub(crate) struct MtpAttentionScope;

impl MtpAttentionScope {
    pub(crate) fn enter() -> Self {
        SINGLE_TOKEN_SELF_ATTENTION_DEPTH.with(|depth| {
            depth.set(
                depth
                    .get()
                    .checked_add(1)
                    .expect("MTP attention scope depth overflow"),
            );
        });
        Self
    }
}

impl Drop for MtpAttentionScope {
    fn drop(&mut self) {
        SINGLE_TOKEN_SELF_ATTENTION_DEPTH.with(|depth| {
            let current = depth.get();
            debug_assert!(current > 0, "unbalanced MTP attention scope");
            depth.set(current.saturating_sub(1));
        });
    }
}

#[inline]
pub(crate) fn single_token_self_attention_active() -> bool {
    SINGLE_TOKEN_SELF_ATTENTION_DEPTH.with(|depth| depth.get() != 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_is_nested_and_restored() {
        assert!(!single_token_self_attention_active());
        {
            let _outer = MtpAttentionScope::enter();
            assert!(single_token_self_attention_active());
            {
                let _inner = MtpAttentionScope::enter();
                assert!(single_token_self_attention_active());
            }
            assert!(single_token_self_attention_active());
        }
        assert!(!single_token_self_attention_active());
    }

    #[test]
    fn scope_is_restored_during_unwind() {
        let _ = std::panic::catch_unwind(|| {
            let _scope = MtpAttentionScope::enter();
            assert!(single_token_self_attention_active());
            panic!("exercise scope unwinding");
        });
        assert!(!single_token_self_attention_active());
    }
}
