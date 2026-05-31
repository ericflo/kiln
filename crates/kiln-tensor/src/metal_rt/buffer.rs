//! Vendored from candle-metal-kernels 0.10.2 `src/metal/buffer.rs`
//! (MIT/Apache-2.0). objc2 wrapper over `MTLBuffer`. (#1082)

use objc2::{rc::Retained, runtime::ProtocolObject};
use objc2_foundation::NSRange;
use objc2_metal::{MTLBuffer, MTLResource};
use std::{collections::HashMap, sync::Arc};

pub type MetalResource = ProtocolObject<dyn MTLResource>;
pub type MTLResourceOptions = objc2_metal::MTLResourceOptions;

#[derive(Clone, Debug, Hash, PartialEq)]
pub struct Buffer {
    raw: Retained<ProtocolObject<dyn MTLBuffer>>,
}

unsafe impl Send for Buffer {}
unsafe impl Sync for Buffer {}

impl Buffer {
    pub fn new(raw: Retained<ProtocolObject<dyn MTLBuffer>>) -> Buffer {
        Buffer { raw }
    }

    pub fn contents(&self) -> *mut u8 {
        self.data()
    }

    pub fn data(&self) -> *mut u8 {
        use objc2_metal::MTLBuffer as _;
        self.as_ref().contents().as_ptr() as *mut u8
    }

    pub fn length(&self) -> usize {
        self.as_ref().length()
    }

    pub fn did_modify_range(&self, range: NSRange) {
        self.as_ref().didModifyRange(range);
    }
}

impl AsRef<ProtocolObject<dyn MTLBuffer>> for Buffer {
    fn as_ref(&self) -> &ProtocolObject<dyn MTLBuffer> {
        &self.raw
    }
}

impl<'a> From<&'a Buffer> for &'a MetalResource {
    fn from(val: &'a Buffer) -> Self {
        ProtocolObject::from_ref(val.as_ref())
    }
}

pub type BufferMap = HashMap<usize, Vec<Arc<Buffer>>>;

/// A `(buffer, byte-offset)` pair — the positional argument every MSL
/// kernel entry consumes for an input/output tensor. Vendored from
/// candle-metal-kernels 0.10.2 `src/utils.rs` `BufferOffset` (the
/// `EncoderParam` trait impl is dropped: kiln's kernels call
/// `encoder.set_buffer(..)` directly with the two fields). (#1082)
pub struct BufferOffset<'a> {
    pub buffer: &'a Buffer,
    pub offset_in_bytes: usize,
}

impl std::fmt::Debug for BufferOffset<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferOffset")
            .field("buffer", &self.buffer)
            .field("offset_in_bytes", &self.offset_in_bytes)
            .finish()
    }
}

impl<'a> BufferOffset<'a> {
    pub fn zero_offset(buffer: &'a Buffer) -> Self {
        Self {
            buffer,
            offset_in_bytes: 0,
        }
    }
}
