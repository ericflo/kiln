//! Cross-backend handle for a buffer that lives on a compute device.
//!
//! Backends keep their own native buffer types ([`kiln_vulkan_kernel::VulkanBuffer`],
//! and CUDA/Metal slots reserved for parity); [`DeviceBuffer`] is the
//! common Arc-shaped wrapper that callers in `kiln-model` (the resident
//! registry) and `kiln-train` (LoRA params, optimizer state, activations)
//! can pass around without depending directly on a specific backend
//! crate. Conversion to the underlying buffer happens through the typed
//! accessors (e.g. [`DeviceBuffer::as_vulkan`]) which return `None` when
//! the variant doesn't match.
//!
//! The CUDA and Metal arms are deliberately stubbed — the call sites
//! that need them are gated behind their own feature flags, so widening
//! the enum is a separate landing.

use std::sync::Arc;

/// A handle to a buffer that lives on a compute device.
///
/// The enum carries an `Arc<>` of the backend-specific buffer type;
/// dropping the [`DeviceBuffer`] decrements the buffer's refcount and
/// releases the device allocation when nothing else holds a reference.
#[derive(Debug, Clone)]
pub enum DeviceBuffer {
    /// Vulkan-native buffer. Available with `--features vulkan`.
    #[cfg(feature = "vulkan")]
    Vulkan(Arc<kiln_vulkan_kernel::VulkanBuffer>),
    /// CPU-side fallback buffer. Carries an `Arc<[u8]>` so the same
    /// type-erased flow works when no GPU backend is selected.
    Cpu(Arc<[u8]>),
}

impl DeviceBuffer {
    /// Length of the buffer in bytes.
    pub fn len_bytes(&self) -> u64 {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(buf) => buf.size(),
            Self::Cpu(bytes) => bytes.len() as u64,
        }
    }

    /// Short backend tag for diagnostics and logs.
    pub fn backend(&self) -> &'static str {
        match self {
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => "vulkan",
            Self::Cpu(_) => "cpu",
        }
    }

    /// Borrow the inner Vulkan buffer, if this variant is Vulkan.
    ///
    /// Returns `None` for any other variant — call sites that require
    /// a Vulkan handle should treat that as a precondition violation
    /// and fall back to the portable path.
    #[cfg(feature = "vulkan")]
    pub fn as_vulkan(&self) -> Option<&Arc<kiln_vulkan_kernel::VulkanBuffer>> {
        match self {
            Self::Vulkan(buf) => Some(buf),
            _ => None,
        }
    }

    /// Borrow the CPU bytes, if this variant is Cpu.
    pub fn as_cpu(&self) -> Option<&Arc<[u8]>> {
        match self {
            Self::Cpu(bytes) => Some(bytes),
            #[cfg(feature = "vulkan")]
            Self::Vulkan(_) => None,
        }
    }

    /// Construct a Vulkan-backed device buffer from an existing Arc.
    #[cfg(feature = "vulkan")]
    pub fn from_vulkan(buf: Arc<kiln_vulkan_kernel::VulkanBuffer>) -> Self {
        Self::Vulkan(buf)
    }

    /// Construct a CPU-backed device buffer.
    pub fn from_cpu_bytes(bytes: Arc<[u8]>) -> Self {
        Self::Cpu(bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_buffer_reports_len_and_backend() {
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 1024].into_boxed_slice());
        let buf = DeviceBuffer::from_cpu_bytes(bytes.clone());
        assert_eq!(buf.len_bytes(), 1024);
        assert_eq!(buf.backend(), "cpu");
        assert!(buf.as_cpu().is_some());
    }

    #[cfg(feature = "vulkan")]
    #[test]
    fn cpu_variant_does_not_resolve_as_vulkan() {
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 4].into_boxed_slice());
        let buf = DeviceBuffer::from_cpu_bytes(bytes);
        assert!(buf.as_vulkan().is_none());
    }

    #[test]
    fn dropping_buffer_releases_arc() {
        let bytes: Arc<[u8]> = Arc::from(vec![0u8; 8].into_boxed_slice());
        let weak = Arc::downgrade(&bytes);
        let buf = DeviceBuffer::from_cpu_bytes(bytes);
        drop(buf);
        // Original strong ref is still held by `weak`'s upgrade target;
        // we kept the original `Arc` only by cloning it into the buffer.
        // After dropping the buffer there is one strong ref left in the
        // weak's underlying storage iff we still hold the original `Arc`.
        assert!(weak.upgrade().is_none());
    }
}
