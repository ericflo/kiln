//! `MpsTilePolicy` — Metal-side tile + transpose configuration.
//!
//! `MPSMatrixMultiplication` doesn't expose an algo-id integer like
//! cublasLt; instead the per-shape "algo" is encoded in:
//! - tile dimensions (M, N, K splits)
//! - transpose flags (transposeLeft / transposeRight)
//! - alpha / beta scale type
//! - storage mode (Shared on UMA, Private on discrete)
//!
//! `MpsTilePolicy` carries this configuration in a compact form that
//! serializes cleanly to the algo_blob byte slot in
//! [`kiln_blas::AlgoCacheValue`].

use std::convert::TryInto;

/// Metal tile + transpose configuration for one matmul shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsTilePolicy {
    /// Tile width along the M axis. 0 means "let MPS pick".
    pub tile_m: u32,
    /// Tile width along the N axis. 0 means "let MPS pick".
    pub tile_n: u32,
    /// Tile width along the K axis. 0 means "let MPS pick".
    pub tile_k: u32,
    /// Transpose left operand?
    pub transpose_left: bool,
    /// Transpose right operand?
    pub transpose_right: bool,
}

impl MpsTilePolicy {
    /// MPS-chooses-tiles default. Useful as the initial cache miss.
    pub const AUTO: Self = MpsTilePolicy {
        tile_m: 0,
        tile_n: 0,
        tile_k: 0,
        transpose_left: false,
        transpose_right: false,
    };

    /// Serialize to a 14-byte blob:
    /// `[tile_m: u32 LE][tile_n: u32 LE][tile_k: u32 LE][transposes: 2 bytes]`.
    /// Stable across kiln versions.
    pub fn to_blob(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(14);
        out.extend_from_slice(&self.tile_m.to_le_bytes());
        out.extend_from_slice(&self.tile_n.to_le_bytes());
        out.extend_from_slice(&self.tile_k.to_le_bytes());
        out.push(self.transpose_left as u8);
        out.push(self.transpose_right as u8);
        out
    }

    /// Reverse of `to_blob`. Returns `None` on a malformed blob.
    pub fn from_blob(blob: &[u8]) -> Option<Self> {
        if blob.len() != 14 {
            return None;
        }
        let tile_m = u32::from_le_bytes(blob[0..4].try_into().ok()?);
        let tile_n = u32::from_le_bytes(blob[4..8].try_into().ok()?);
        let tile_k = u32::from_le_bytes(blob[8..12].try_into().ok()?);
        Some(MpsTilePolicy {
            tile_m,
            tile_n,
            tile_k,
            transpose_left: blob[12] != 0,
            transpose_right: blob[13] != 0,
        })
    }
}

impl Default for MpsTilePolicy {
    fn default() -> Self {
        Self::AUTO
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_is_all_zero_no_transpose() {
        let p = MpsTilePolicy::AUTO;
        assert_eq!(p.tile_m, 0);
        assert_eq!(p.tile_n, 0);
        assert_eq!(p.tile_k, 0);
        assert!(!p.transpose_left);
        assert!(!p.transpose_right);
    }

    #[test]
    fn blob_round_trip() {
        let p = MpsTilePolicy {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            transpose_left: false,
            transpose_right: true,
        };
        let blob = p.to_blob();
        assert_eq!(blob.len(), 14);
        let back = MpsTilePolicy::from_blob(&blob).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn blob_handles_auto() {
        let p = MpsTilePolicy::AUTO;
        let blob = p.to_blob();
        let back = MpsTilePolicy::from_blob(&blob).unwrap();
        assert_eq!(back, p);
    }

    #[test]
    fn blob_rejects_wrong_size() {
        assert!(MpsTilePolicy::from_blob(&[]).is_none());
        assert!(MpsTilePolicy::from_blob(&[0u8; 13]).is_none());
        assert!(MpsTilePolicy::from_blob(&[0u8; 15]).is_none());
    }
}
