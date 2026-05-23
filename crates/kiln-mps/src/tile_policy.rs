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

    /// Heuristic tile-size pick for an `M x N x K` matmul on Apple
    /// Silicon. Buckets per the Phase 0.9 `mps_mlp_probe` shape
    /// sweeps:
    ///
    /// - **GEMV-ish** (`M <= 2`) — narrow tile favoring N.
    /// - **Small square** (`M, N < 512`) — 32 / 32 / 16 tile.
    /// - **MLP-elongated** (`N > 8 * M`, the gate||up shape) —
    ///   wide tile favoring N (`128 / 256 / 32`).
    /// - **Square hot-path** (everything else) — `64 / 64 / 32` tile.
    ///
    /// Returned policy still serializes to the same 14-byte blob via
    /// [`Self::to_blob`].
    pub fn recommended_for(m: u64, n: u64, k: u64) -> Self {
        let (tile_m, tile_n, tile_k) = if m <= 2 {
            // GEMV — let MPS pick the M side; cap N + K.
            (0, 256, 16)
        } else if m < 512 && n < 512 {
            (32, 32, 16)
        } else if n > 8 * m.max(1) {
            // Wide / MLP-elongated.
            (128, 256, 32)
        } else {
            (64, 64, 32)
        };
        MpsTilePolicy {
            tile_m,
            tile_n,
            tile_k,
            transpose_left: false,
            transpose_right: false,
        }
        .with_k_hint(k)
    }

    /// Tweak `tile_k` based on the K axis (the contracted dim).
    /// Phase 0 mps probe finds the smaller-K shapes prefer 16-element
    /// tiles; larger ones favor 32.
    fn with_k_hint(mut self, k: u64) -> Self {
        if self.tile_k == 0 {
            return self;
        }
        if k < 1024 {
            self.tile_k = 16;
        }
        self
    }

    /// Alias for [`Self::to_blob`] used by the Phase 2.2
    /// `MpsBackendMatmul` adapter — `serialize` reads better at the
    /// call site.
    pub fn serialize(&self) -> Vec<u8> {
        self.to_blob()
    }

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

    #[test]
    fn recommended_picks_distinct_tiles_per_shape() {
        let gemv = MpsTilePolicy::recommended_for(1, 2560, 18432);
        let small = MpsTilePolicy::recommended_for(64, 64, 256);
        let mlp = MpsTilePolicy::recommended_for(2048, 18432, 2560);
        let square = MpsTilePolicy::recommended_for(2048, 2048, 4096);
        assert_eq!(gemv.tile_m, 0);
        assert_eq!(small.tile_m, 32);
        assert!(mlp.tile_n > mlp.tile_m); // Elongated.
        assert!(square.tile_m == square.tile_n);
        // Three distinct policies → three distinct blobs.
        let blobs = [gemv.to_blob(), small.to_blob(), mlp.to_blob(), square.to_blob()];
        for i in 0..blobs.len() {
            for j in i + 1..blobs.len() {
                assert_ne!(blobs[i], blobs[j], "blob {i} == blob {j}");
            }
        }
    }

    #[test]
    fn serialize_is_to_blob_alias() {
        let p = MpsTilePolicy {
            tile_m: 64,
            tile_n: 128,
            tile_k: 32,
            transpose_left: true,
            transpose_right: false,
        };
        assert_eq!(p.serialize(), p.to_blob());
    }

    #[test]
    fn small_k_picks_16_tile() {
        let p = MpsTilePolicy::recommended_for(2048, 2048, 256);
        assert_eq!(p.tile_k, 16);
    }
}
