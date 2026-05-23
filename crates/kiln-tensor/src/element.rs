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
}

impl Element for f32 {
    const DTYPE: DType = DType::F32;
    const SIZE: usize = 4;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<f32, u8>(values).to_vec()
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
}

impl Element for u32 {
    const DTYPE: DType = DType::U32;
    const SIZE: usize = 4;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<u32, u8>(values).to_vec()
    }
}

impl Element for u8 {
    const DTYPE: DType = DType::U8;
    const SIZE: usize = 1;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        values.to_vec()
    }
}

impl Element for i64 {
    const DTYPE: DType = DType::I64;
    const SIZE: usize = 8;
    fn to_bytes(values: &[Self]) -> Vec<u8> {
        bytemuck::cast_slice::<i64, u8>(values).to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
