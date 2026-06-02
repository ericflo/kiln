//! `kiln_tensor::Element` — trait pinning the bytewise representation
//! of each scalar dtype the CPU constructors accept.
//!
//! This is a deliberately small trait. The conversion from `&[T]` to a
//! [`CpuStorage`](crate::CpuStorage) byte buffer goes through
//! [`bytemuck::Pod`] for the trivially-copyable cases (`f32`, `u32`,
//! `u8`, `i64`) and through a custom [`Element::to_bytes`] path for
//! `bf16` / `f16` which carry domain-specific bit patterns.
//!
//! Packed dtypes (`Int4Packed`, `Fp4Packed`) are intentionally
//! **not** representable as a Rust scalar — there is no `Int4` type
//! that fits in 4 bits — so the CPU constructors take a `&[u8]` byte
//! buffer at the packed-byte level and skip this trait.

use crate::DType;

/// A scalar type that maps 1:1 to one of the non-packed [`DType`]
/// variants. Used by the typed CPU constructors
/// (`Tensor::from_slice` / `Tensor::from_vec`).
pub trait Element: Copy + 'static {
    /// The [`DType`] this Rust scalar corresponds to.
    const DTYPE: DType;
    /// Bytes per element. Matches `Self::DTYPE.size_in_bytes()`.
    const SIZE: usize;
    /// Convert a slice of `Self` to the byte buffer the storage carries.
    fn to_bytes(values: &[Self]) -> Vec<u8>;
    /// Decode a byte buffer (little-endian, `Self::SIZE` bytes per
    /// element) back into a `Vec<Self>`. The inverse of [`to_bytes`].
    ///
    /// Used by [`Tensor::to_vec`](crate::Tensor::to_vec) to read a
    /// contiguous CPU storage back to host scalars — the candle-free
    /// replacement for candle's `Tensor::to_vec1::<T>()` /
    /// `to_scalar::<T>()`. `bytes.len()` must be a multiple of
    /// `Self::SIZE`.
    fn from_bytes(bytes: &[u8]) -> Vec<Self>;
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    const SIZE: usize = 4;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<f32, u8>(values).to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        // `cast_slice` panics (`TargetAlignmentGreaterAndInputNotAligned`) when
        // `bytes` is empty (its dangling ptr is only 1-aligned) or is an
        // unaligned subslice. Fast-path the aligned case via the zero-copy view;
        // fall back to an element-wise copy otherwise (handles len==0 → []). (#1082)
        match bytemuck::try_cast_slice::<u8, f32>(bytes) {
            Ok(s) => s.to_vec(),
            Err(_) => bytes
                .chunks_exact(4)
                .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

impl Element for half::bf16 {
    const DTYPE: DType = DType::BF16;
    const SIZE: usize = 2;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

impl Element for half::f16 {
    const DTYPE: DType = DType::F16;
    const SIZE: usize = 2;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        let mut out = Vec::with_capacity(values.len() * 2);
        for v in values {
            out.extend_from_slice(&v.to_le_bytes());
        }
        out
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]))
            .collect()
    }
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;
    const SIZE: usize = 4;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<u32, u8>(values).to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        // See the f32 impl: empty / unaligned slices make `cast_slice` panic. (#1082)
        match bytemuck::try_cast_slice::<u8, u32>(bytes) {
            Ok(s) => s.to_vec(),
            Err(_) => bytes
                .chunks_exact(4)
                .map(|c| u32::from_ne_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        }
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
    const SIZE: usize = 1;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        values.to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        bytes.to_vec()
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    const SIZE: usize = 8;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<i64, u8>(values).to_vec()
    }
    fn from_bytes(bytes: &[u8]) -> Vec<Self> {
        // See the f32 impl: empty / unaligned slices make `cast_slice` panic. (#1082)
        match bytemuck::try_cast_slice::<u8, i64>(bytes) {
            Ok(s) => s.to_vec(),
            Err(_) => bytes
                .chunks_exact(8)
                .map(|c| {
                    i64::from_ne_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]])
                })
                .collect(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // (#1082) `from_bytes` must not panic on an empty slice (dangling,
    // 1-aligned ptr) or an unaligned subslice — the cast_slice family panics
    // there. Regression for the empty-`rotary_inv_freq` -> to_vec1 panic
    // surfaced during the candle->kt GpuWeights flip.
    #[test]
    fn from_bytes_empty_returns_empty_no_panic() {
        assert_eq!(<f32 as Element>::from_bytes(&[]), Vec::<f32>::new());
        assert_eq!(<u32 as Element>::from_bytes(&[]), Vec::<u32>::new());
        assert_eq!(<i64 as Element>::from_bytes(&[]), Vec::<i64>::new());
    }

    #[test]
    fn from_bytes_unaligned_subslice_matches_aligned() {
        // Build an f32 byte buffer, then read it from a +1 offset (forced
        // 1-aligned) — must round-trip identically to the aligned read.
        let vals = vec![1.0_f32, -2.5, 3.25, 42.0];
        let mut buf = vec![0u8]; // 1-byte prefix forces misalignment
        buf.extend_from_slice(&<f32 as Element>::to_bytes(&vals));
        let unaligned = &buf[1..];
        assert_eq!(<f32 as Element>::from_bytes(unaligned), vals);
    }

    #[test]
    fn dtype_const_matches() {
        assert_eq!(<f32 as Element>::DTYPE, DType::F32);
        assert_eq!(<half::bf16 as Element>::DTYPE, DType::BF16);
        assert_eq!(<half::f16 as Element>::DTYPE, DType::F16);
        assert_eq!(<u32 as Element>::DTYPE, DType::U32);
        assert_eq!(<u8 as Element>::DTYPE, DType::U8);
        assert_eq!(<i64 as Element>::DTYPE, DType::I64);
    }

    #[test]
    fn size_const_matches() {
        for (size, dt_size) in [
            (<f32 as Element>::SIZE, DType::F32.size_in_bytes()),
            (<half::bf16 as Element>::SIZE, DType::BF16.size_in_bytes()),
            (<half::f16 as Element>::SIZE, DType::F16.size_in_bytes()),
            (<u32 as Element>::SIZE, DType::U32.size_in_bytes()),
            (<u8 as Element>::SIZE, DType::U8.size_in_bytes()),
            (<i64 as Element>::SIZE, DType::I64.size_in_bytes()),
        ] {
            assert_eq!(size, dt_size);
        }
    }

    #[test]
    fn f32_to_bytes_round_trip() {
        let v = vec![1.0_f32, -2.5, 3.14];
        let bytes = <f32 as Element>::to_bytes(&v);
        assert_eq!(bytes.len(), 12);
        let back: Vec<f32> = bytemuck::cast_slice::<u8, f32>(&bytes).to_vec();
        assert_eq!(back, v);
    }

    #[test]
    fn bf16_to_bytes_little_endian() {
        let v = vec![half::bf16::from_f32(1.0), half::bf16::from_f32(-0.5)];
        let bytes = <half::bf16 as Element>::to_bytes(&v);
        assert_eq!(bytes.len(), 4);
        // BF16(1.0) = 0x3F80 little-endian = [0x80, 0x3F]
        assert_eq!(&bytes[..2], &[0x80, 0x3F]);
    }

    #[test]
    fn from_bytes_inverts_to_bytes() {
        // f32
        let f = vec![1.0_f32, -2.5, 3.14, 0.0];
        assert_eq!(<f32 as Element>::from_bytes(&<f32 as Element>::to_bytes(&f)), f);
        // u32
        let u = vec![0_u32, 7, 4_000_000_000];
        assert_eq!(<u32 as Element>::from_bytes(&<u32 as Element>::to_bytes(&u)), u);
        // u8
        let b = vec![0_u8, 255, 17];
        assert_eq!(<u8 as Element>::from_bytes(&<u8 as Element>::to_bytes(&b)), b);
        // i64
        let i = vec![-1_i64, 0, 9_000_000_000];
        assert_eq!(<i64 as Element>::from_bytes(&<i64 as Element>::to_bytes(&i)), i);
        // bf16 / f16 (exactly-representable values round-trip)
        let bf = vec![half::bf16::from_f32(1.0), half::bf16::from_f32(-0.5)];
        assert_eq!(
            <half::bf16 as Element>::from_bytes(&<half::bf16 as Element>::to_bytes(&bf)),
            bf
        );
        let hf = vec![half::f16::from_f32(2.0), half::f16::from_f32(0.25)];
        assert_eq!(
            <half::f16 as Element>::from_bytes(&<half::f16 as Element>::to_bytes(&hf)),
            hf
        );
    }
}
