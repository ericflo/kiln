//! Vendored Marlin W4A16 GEMM CUDA kernel.
//!
//! # Provenance: IST-DASLab/marlin
//!
//! This crate is the vendored port of
//! [`IST-DASLab/marlin`](https://github.com/IST-DASLab/marlin) at upstream
//! commit `1f25790bdd49fba53106164a24666dade68d7c90`. The single CUDA
//! translation unit `csrc/marlin_kernel.cu` is the upstream
//! `marlin/marlin_cuda_kernel.cu` verbatim, with one tiny addition at the
//! bottom of the file: an `extern "C"` shim, [`kiln_marlin_w4a16_gemm`],
//! that exposes the otherwise-`cudaStream_t`-typed entry point through a
//! C-ABI signature so cargo's `cc` build can link it from Rust.
//!
//! # Scope (kernel-only crate, Phase 6)
//!
//! Per the project's "minimal-scope vendoring" policy, this crate ships
//! the kernel and a parity test only. There is no `kiln-model` /
//! `forward.rs` integration yet — that lands in a follow-up PR.
//!
//! Within this crate the supported envelope is:
//!
//!   - **Activations:** bf16 in / bf16 out, as the rest of kiln. The kernel
//!     itself is FP16-only (it uses `mma.m16n8k16.f32.f16.f16.f32`), so the
//!     wrapper casts bf16 → fp16 on the way in and fp16 → bf16 on the way
//!     out. A future integration PR can revisit this if a fused bf16-cast
//!     kernel becomes worthwhile.
//!   - **Weights:** symmetric INT4, packed into Marlin's tiled / permuted
//!     `int32` layout (NOT raw GPTQ `qweight`). See [`pack`].
//!   - **Group size:** `-1` (per-column) or `128`. Marlin's
//!     `CALL_IF` table only instantiates `group_blocks ∈ {-1, 8}`.
//!   - **Shape constraints:** `prob_k % 128 == 0`, `prob_n % 256 == 0`,
//!     same as upstream `marlin/__init__.py::Layer.__init__`.
//!   - **Forward only.** No backward, no quant kernel.
//!
//! # API
//!
//! - [`marlin_w4a16_gemm`] — the BF16-facing matmul wrapper. Takes
//!   `[m, k] bf16`, `[k/16, n*16/8] i32`, `[k/groupsize, n] f16`, and
//!   returns `[m, n] bf16`.
//! - [`pack`] — pure-Rust port of the upstream Python packer
//!   (`marlin/__init__.py::Layer.pack`). Both the parity test and any
//!   future integration code use this to build the Marlin-format `B` and
//!   permuted scales from a fake-quantized fp32 weight matrix.

use candle_core::{
    DType, Result, Tensor, backend::BackendStorage, cuda_backend::cudarc::driver::DevicePtr,
};
use half::{bf16, f16};

unsafe extern "C" {
    fn kiln_marlin_w4a16_gemm(
        a: *const core::ffi::c_void,
        b: *const core::ffi::c_void,
        c: *mut core::ffi::c_void,
        s: *const core::ffi::c_void,
        prob_m: i32,
        prob_n: i32,
        prob_k: i32,
        workspace: *mut core::ffi::c_void,
        groupsize: i32,
        dev: i32,
        stream: *mut core::ffi::c_void,
        thread_k: i32,
        thread_n: i32,
        sms: i32,
        max_par: i32,
    ) -> i32;
}

/// Marlin's default `max_par` from `marlin/__init__.py`.
const DEFAULT_MAX_PAR: i32 = 16;

/// Workspace tile size in `n` (Marlin's smallest `thread_n`).
const WORKSPACE_TILE_N: usize = 128;

/// Run the vendored Marlin W4A16 GEMM with a BF16 activation interface.
///
/// Inputs (all CUDA, all contiguous):
///
///   - `a`           : `[m, k]` bf16 input matrix.
///   - `b_packed`    : `[k / 16, n * 16 / 8]` i32, in Marlin's tiled /
///                     permuted packed layout (see [`pack::pack_weights`]).
///   - `scales`      : `[k / groupsize, n]` f16, Marlin-permuted (see
///                     [`pack::permute_scales`]). Use `groupsize = k` for
///                     per-column quantization.
///   - `groupsize`   : Quantization group size. Must be `-1` (== `k`) or
///                     `128`.
///
/// Returns a freshly allocated `[m, n]` bf16 tensor.
///
/// # Constraints
///
///   - `k % 128 == 0`
///   - `n % 256 == 0`
///   - `m >= 1` (the kernel pads `m` up to a multiple of 16 internally)
///   - `groupsize in { -1, 128 }`
pub fn marlin_w4a16_gemm(
    a: &Tensor,
    b_packed: &Tensor,
    scales: &Tensor,
    groupsize: i32,
) -> Result<Tensor> {
    let device = a.device();
    if !device.is_cuda() {
        candle_core::bail!("kiln-marlin-gemm: a must be on CUDA");
    }

    let (m, k) = a.dims2()?;
    let (b_rows, b_cols) = b_packed.dims2()?;
    let (s_rows, n) = scales.dims2()?;

    if k % 128 != 0 {
        candle_core::bail!("kiln-marlin-gemm: k must be a multiple of 128 (got {k})");
    }
    if n % 256 != 0 {
        candle_core::bail!("kiln-marlin-gemm: n must be a multiple of 256 (got {n})");
    }

    if b_rows != k / 16 || b_cols != n * 16 / 8 {
        candle_core::bail!(
            "kiln-marlin-gemm: b_packed shape [{b_rows}, {b_cols}] does not match expected \
             [k/16={}, n*16/8={}] for k={k}, n={n}",
            k / 16,
            n * 16 / 8,
        );
    }

    if !(groupsize == -1 || groupsize == 128) {
        candle_core::bail!("kiln-marlin-gemm: groupsize must be -1 or 128 (got {groupsize})");
    }
    // `groupsize_for_dims` is the actual chunk size in K used for sizing the
    // scales tensor; `-1` means per-column, i.e. one row of scales total.
    // The kernel itself takes `-1` as a sentinel for per-column, so we keep
    // `groupsize` as-is when forwarding to the FFI call.
    let groupsize_for_dims: usize = if groupsize == -1 {
        k
    } else {
        groupsize as usize
    };
    if groupsize_for_dims > k || k % groupsize_for_dims != 0 {
        candle_core::bail!("kiln-marlin-gemm: k={k} must be divisible by groupsize={groupsize}");
    }

    let expected_s_rows = k / groupsize_for_dims;
    if s_rows != expected_s_rows {
        candle_core::bail!(
            "kiln-marlin-gemm: scales rows {s_rows} != expected {expected_s_rows} \
             (k={k}, groupsize={groupsize})"
        );
    }

    if a.dtype() != DType::BF16 {
        candle_core::bail!("kiln-marlin-gemm: a must be bf16 (got {:?})", a.dtype());
    }
    if b_packed.dtype() != DType::I32 {
        candle_core::bail!(
            "kiln-marlin-gemm: b_packed must be i32 (got {:?})",
            b_packed.dtype()
        );
    }
    if scales.dtype() != DType::F16 {
        candle_core::bail!(
            "kiln-marlin-gemm: scales must be f16 (got {:?})",
            scales.dtype()
        );
    }

    // The kernel is FP16-only; cast bf16 -> fp16 on the way in.
    let a_fp16 = a.to_dtype(DType::F16)?.contiguous()?;
    let b_packed = b_packed.contiguous()?;
    let scales = scales.contiguous()?;

    // Output is fp16 (kernel writes here), then we cast back to bf16.
    let c_fp16 = Tensor::zeros((m, n), DType::F16, device)?;

    // Workspace: int32 buffer of size at least (n / 128) * max_par, all zero.
    let workspace_len = (n / WORKSPACE_TILE_N) * DEFAULT_MAX_PAR as usize;
    let workspace = Tensor::zeros((workspace_len,), DType::I32, device)?;

    {
        let (a_storage, a_layout) = a_fp16.storage_and_layout();
        let (b_storage, b_layout) = b_packed.storage_and_layout();
        let (c_storage, c_layout) = c_fp16.storage_and_layout();
        let (s_storage, s_layout) = scales.storage_and_layout();
        let (w_storage, w_layout) = workspace.storage_and_layout();

        let a_cuda = match &*a_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-marlin-gemm: a must be on CUDA"),
        };
        let b_cuda = match &*b_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-marlin-gemm: b_packed must be on CUDA"),
        };
        let c_cuda = match &*c_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-marlin-gemm: c must be on CUDA"),
        };
        let s_cuda = match &*s_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-marlin-gemm: scales must be on CUDA"),
        };
        let w_cuda = match &*w_storage {
            candle_core::Storage::Cuda(c) => c,
            _ => candle_core::bail!("kiln-marlin-gemm: workspace must be on CUDA"),
        };

        let stream = a_cuda.device().cuda_stream();
        let raw_stream = stream.cu_stream() as *mut core::ffi::c_void;
        // candle 0.10's CudaDevice doesn't expose its ordinal publicly. The
        // Marlin kernel only consults `dev` to call cudaDeviceGetAttribute
        // when `sms == -1`. Defaulting to 0 is correct on single-GPU pods,
        // which is what kiln targets today; multi-GPU support would need a
        // candle API addition to surface the device index.
        let dev_ord: i32 = 0;

        let a_slice = a_cuda
            .as_cuda_slice::<f16>()?
            .slice(a_layout.start_offset()..);
        let b_slice = b_cuda
            .as_cuda_slice::<i32>()?
            .slice(b_layout.start_offset()..);
        let c_slice = c_cuda
            .as_cuda_slice::<f16>()?
            .slice(c_layout.start_offset()..);
        let s_slice = s_cuda
            .as_cuda_slice::<f16>()?
            .slice(s_layout.start_offset()..);
        let w_slice = w_cuda
            .as_cuda_slice::<i32>()?
            .slice(w_layout.start_offset()..);

        unsafe {
            let (a_ptr, _g1) = a_slice.device_ptr(&stream);
            let (b_ptr, _g2) = b_slice.device_ptr(&stream);
            let (c_ptr, _g3) = c_slice.device_ptr(&stream);
            let (s_ptr, _g4) = s_slice.device_ptr(&stream);
            let (w_ptr, _g5) = w_slice.device_ptr(&stream);

            let status = kiln_marlin_w4a16_gemm(
                a_ptr as *const _,
                b_ptr as *const _,
                c_ptr as *mut _,
                s_ptr as *const _,
                m as i32,
                n as i32,
                k as i32,
                w_ptr as *mut _,
                groupsize,
                dev_ord,
                raw_stream,
                /* thread_k */ -1,
                /* thread_n */ -1,
                /* sms */ -1,
                DEFAULT_MAX_PAR,
            );

            if status != 0 {
                candle_core::bail!(
                    "kiln_marlin_w4a16_gemm failed with status {status} \
                     (1=ERR_PROB_SHAPE, 2=ERR_KERN_SHAPE) for m={m}, n={n}, k={k}, \
                     groupsize={groupsize}"
                );
            }
        }
    }

    // Cast fp16 output back to bf16 to match the rest of the kiln pipeline.
    let c_bf16 = c_fp16.to_dtype(DType::BF16)?;

    // Keep types around to silence unused-import warnings on non-CUDA hosts.
    let _ = std::marker::PhantomData::<bf16>;

    Ok(c_bf16)
}

/// Pure-Rust port of the upstream Marlin Python packer
/// (`marlin/__init__.py::_get_perms` and `Layer.pack`).
///
/// These helpers exist so the parity test (and any future integration)
/// can build a Marlin-format `B` and permuted-scale tensor from a
/// fake-quantized fp32 weight matrix without depending on Python or the
/// upstream marlin package.
pub mod pack {
    use super::*;

    /// Marlin uses 4-bit symmetric quantization with an offset of 8.
    pub const MAX_Q: i32 = 15;
    pub const Q_OFFSET: i32 = 8;

    /// Build the Marlin weight permutation array (length 1024).
    ///
    /// Mirrors the Python reference in
    /// `marlin/__init__.py::_get_perms`. The interleave step at the end
    /// reorders each contiguous group of 8 indices by `[0, 2, 4, 6, 1, 3, 5, 7]`.
    pub fn build_weight_perm() -> Vec<usize> {
        let mut perm: Vec<usize> = Vec::with_capacity(1024);
        for i in 0..32usize {
            let mut perm1: Vec<usize> = Vec::with_capacity(8);
            let col = i / 4;
            for block in [0usize, 1] {
                let row_seeds = [
                    2 * (i % 4),
                    2 * (i % 4) + 1,
                    2 * (i % 4 + 4),
                    2 * (i % 4 + 4) + 1,
                ];
                for row in row_seeds {
                    perm1.push(16 * row + col + 8 * block);
                }
            }
            for j in 0..4usize {
                for &p in perm1.iter() {
                    perm.push(p + 256 * j);
                }
            }
        }
        let interleave: [usize; 8] = [0, 2, 4, 6, 1, 3, 5, 7];
        let mut out = Vec::with_capacity(perm.len());
        for chunk in perm.chunks_exact(8) {
            for &idx in &interleave {
                out.push(chunk[idx]);
            }
        }
        debug_assert_eq!(out.len(), 1024);
        out
    }

    /// Marlin per-group-scale permutation (length 64).
    ///
    /// Mirrors `_get_perms`'s `scale_perm`:
    /// `[i + 8*j for i in 0..8 for j in 0..8]`.
    pub fn build_scale_perm() -> Vec<usize> {
        let mut sp: Vec<usize> = Vec::with_capacity(64);
        for i in 0..8usize {
            for j in 0..8usize {
                sp.push(i + 8 * j);
            }
        }
        sp
    }

    /// Marlin per-column-scale permutation (length 32, used when
    /// `groupsize == -1`).
    ///
    /// Mirrors `_get_perms`'s `scale_perm_single`:
    /// `[2*i + j for i in 0..4 for j in [0,1,8,9,16,17,24,25]]`.
    pub fn build_scale_perm_single() -> Vec<usize> {
        let mut sps: Vec<usize> = Vec::with_capacity(32);
        let inner: [usize; 8] = [0, 1, 8, 9, 16, 17, 24, 25];
        for i in 0..4usize {
            for j in inner {
                sps.push(2 * i + j);
            }
        }
        sps
    }

    /// Quantize a fp32 weight matrix `[k, n]` (note: NOT transposed —
    /// row-major in input-feature x output-feature order) into Marlin's
    /// packed `B` layout `[k/16, n*16/8]` i32, plus the matching permuted
    /// scales `[k/groupsize, n]` f16.
    ///
    /// `groupsize == -1` means per-column quantization.
    ///
    /// Returns `(b_packed_i32, scales_f16, dequant_weights_f32)`.
    /// `dequant_weights_f32` is the round-tripped weight that the kernel
    /// will effectively multiply against (i.e. the "ground truth" for
    /// parity checks against a candle bf16 baseline).
    pub fn quantize_and_pack(
        weight: &[f32], // row-major [k, n]
        k: usize,
        n: usize,
        groupsize: i64,
    ) -> (Vec<i32>, Vec<f16>, Vec<f32>) {
        assert_eq!(weight.len(), k * n);
        assert_eq!(k % 128, 0, "k must be %128 == 0");
        assert_eq!(n % 256, 0, "n must be %256 == 0");

        let g = if groupsize == -1 {
            k
        } else {
            groupsize as usize
        };
        assert!(k % g == 0, "k={k} must be divisible by groupsize={g}");
        let num_groups = k / g;

        // 1) Compute per-(group, column) scales. We use upstream Marlin's
        //    convention: s = 2 * max(|w|) / 15. The dequant grid is then
        //    (q - 8) * s for q in 0..=15. We immediately round each scale
        //    through f16 so the candle baseline below uses the same numeric
        //    values the kernel will actually load.
        let mut scales = vec![0.0f32; num_groups * n];
        for grp in 0..num_groups {
            for col in 0..n {
                let mut max_abs = 0.0f32;
                for r in 0..g {
                    let v = weight[(grp * g + r) * n + col].abs();
                    if v > max_abs {
                        max_abs = v;
                    }
                }
                // Avoid divide-by-zero. 1.0 is harmless because every weight
                // in the group is exactly zero, so the round-trip is zero.
                if max_abs == 0.0 {
                    max_abs = 1.0;
                }
                let s_f32 = (2.0 * max_abs) / (MAX_Q as f32);
                // Round-trip through f16 so kernel-side and candle-side
                // dequant agree on the scale value bit-for-bit.
                scales[grp * n + col] = f16::from_f32(s_f32).to_f32();
            }
        }

        // 2) Quantize: w_int = clamp(round(w/s) + 8, 0, 15).
        let mut q = vec![0i32; k * n];
        let mut dequant = vec![0.0f32; k * n];
        for r in 0..k {
            let grp = r / g;
            for col in 0..n {
                let s = scales[grp * n + col];
                let raw = (weight[r * n + col] / s).round() as i32;
                let shifted = (raw + Q_OFFSET).clamp(0, MAX_Q);
                q[r * n + col] = shifted;
                dequant[r * n + col] = (shifted - Q_OFFSET) as f32 * s;
            }
        }

        // 3) Apply Marlin's 16x16 tile transpose:
        //    w = w.reshape(k/16, 16, n/16, 16).permute(0, 2, 1, 3)
        //          .reshape(k/16, n*16)
        // Then apply the perm[1024] re-ordering to each row of length n*16
        // (which is processed as `n*16 // 1024` chunks of 1024 each).
        let kt = k / 16;
        let nt = n / 16;
        let row_len = n * 16;
        let mut tiled = vec![0i32; kt * row_len];
        for k_blk in 0..kt {
            for n_blk in 0..nt {
                for tk in 0..16usize {
                    for tn in 0..16usize {
                        let src = (k_blk * 16 + tk) * n + (n_blk * 16 + tn);
                        let dst = k_blk * row_len + n_blk * 16 * 16 + tk * 16 + tn;
                        tiled[dst] = q[src];
                    }
                }
            }
        }

        let perm = build_weight_perm();
        assert_eq!(perm.len(), 1024);
        assert_eq!(row_len % perm.len(), 0);
        let chunks_per_row = row_len / perm.len();
        let mut permuted = vec![0i32; kt * row_len];
        for r in 0..kt {
            for chunk in 0..chunks_per_row {
                let base = r * row_len + chunk * perm.len();
                let src_base = base;
                for (i, &p) in perm.iter().enumerate() {
                    permuted[base + i] = tiled[src_base + p];
                }
            }
        }

        // 4) Pack 8 nibbles into one int32 along the last dim:
        //    res shape (kt, row_len), packed shape (kt, row_len/8).
        assert_eq!(row_len % 8, 0);
        let packed_cols = row_len / 8;
        let mut packed = vec![0i32; kt * packed_cols];
        for r in 0..kt {
            for c in 0..packed_cols {
                let mut acc: u32 = 0;
                for i in 0..8usize {
                    let nibble = permuted[r * row_len + c * 8 + i] as u32 & 0xF;
                    acc |= nibble << (4 * i);
                }
                packed[r * packed_cols + c] = acc as i32;
            }
        }

        // 5) Permute scales into Marlin's expected layout.
        let scales_perm = if groupsize == -1 {
            permute_scales_single(&scales, n)
        } else {
            permute_scales_grouped(&scales, num_groups, n)
        };

        let scales_f16: Vec<f16> = scales_perm.iter().map(|&s| f16::from_f32(s)).collect();

        (packed, scales_f16, dequant)
    }

    /// Apply Marlin's grouped scale permutation:
    /// reshape `[num_groups, n]` -> rows of length 64, permute by
    /// `scale_perm`, then reshape back.
    fn permute_scales_grouped(scales: &[f32], num_groups: usize, n: usize) -> Vec<f32> {
        let sp = build_scale_perm();
        let group_len = sp.len(); // 64
        assert!(n % group_len == 0);
        let mut out = vec![0.0f32; num_groups * n];
        for grp in 0..num_groups {
            // Each scale row has length n, broken into n/64 sub-rows of 64.
            let row = &scales[grp * n..(grp + 1) * n];
            let dst = &mut out[grp * n..(grp + 1) * n];
            for sub in 0..(n / group_len) {
                for (i, &p) in sp.iter().enumerate() {
                    dst[sub * group_len + i] = row[sub * group_len + p];
                }
            }
        }
        out
    }

    /// Apply Marlin's per-column (single-group) scale permutation:
    /// reshape `[1, n]` -> rows of length 32, permute by
    /// `scale_perm_single`, then reshape back.
    fn permute_scales_single(scales: &[f32], n: usize) -> Vec<f32> {
        let sps = build_scale_perm_single();
        let group_len = sps.len(); // 32
        assert!(n % group_len == 0);
        let mut out = vec![0.0f32; n];
        for sub in 0..(n / group_len) {
            for (i, &p) in sps.iter().enumerate() {
                out[sub * group_len + i] = scales[sub * group_len + p];
            }
        }
        out
    }
}
