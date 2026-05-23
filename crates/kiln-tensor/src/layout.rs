//! `kiln_tensor::Layout` — stride-aware tensor layout descriptor.
//!
//! Replaces candle's `Layout` at the shape / stride / start-offset level.
//! Carries the **logical** view of a tensor's data; the physical buffer
//! lives behind [`Storage`](crate::Storage) (Phase 1.4+).
//!
//! # Zero-copy views (anti-pattern 10)
//!
//! From the issue:
//!
//! > **`narrow` / `reshape` / `transpose` / `slice` are zero-copy.**
//! > Downstream kernels declare stride support in their `supports_*`
//! > function. We do not silently `.contiguous()` to satisfy a kernel;
//! > we teach the kernel about strides or we use the packed variant.
//!
//! Phase 0.1 measured 712 layout calls in `forward.rs` alone:
//! `.contiguous()` x 318, `.narrow(` x 179, `.reshape(` x 142,
//! `.transpose(` x 73. Migrating each off candle uses this `Layout` type
//! as the lossless view.
//!
//! # Memory model
//!
//! - `shape[i]` — number of logical elements along axis `i`.
//! - `strides[i]` — number of physical elements (NOT bytes) to skip
//!   to advance one step along axis `i`. May be zero (broadcast) or
//!   negative (reverse view) once Phase 1.4 supports signed strides.
//! - `start_offset` — physical-element offset into the underlying
//!   storage. Set by [`Layout::narrow_axis`] et al. Bytes-vs-elements
//!   resolution happens at the storage layer; `Layout` is dtype-free.

use crate::{Error, Result};

/// Stride-aware layout descriptor.
///
/// One element per axis. The empty layout (`shape == []`) represents a
/// scalar.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Layout {
    shape: Vec<usize>,
    strides: Vec<usize>,
    start_offset: usize,
}

impl Layout {
    // ------------------------------------------------------------------
    // Constructors
    // ------------------------------------------------------------------

    /// Construct a contiguous row-major layout from a shape.
    pub fn contiguous(shape: impl Into<Vec<usize>>) -> Self {
        let shape: Vec<usize> = shape.into();
        let strides = row_major_strides(&shape);
        Layout {
            shape,
            strides,
            start_offset: 0,
        }
    }

    /// Construct a layout from explicit `(shape, strides, start_offset)`.
    /// Returns an error if `shape.len() != strides.len()`.
    pub fn from_parts(
        shape: Vec<usize>,
        strides: Vec<usize>,
        start_offset: usize,
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(Error::Msg(format!(
                "Layout::from_parts: shape rank {} != strides rank {}",
                shape.len(),
                strides.len()
            )));
        }
        Ok(Layout {
            shape,
            strides,
            start_offset,
        })
    }

    // ------------------------------------------------------------------
    // Accessors
    // ------------------------------------------------------------------

    /// Borrow the shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Borrow the strides.
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the start offset (in logical elements, not bytes).
    pub fn start_offset(&self) -> usize {
        self.start_offset
    }

    /// Logical rank (number of axes).
    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    /// Total element count = product of `shape`.
    pub fn element_count(&self) -> usize {
        self.shape.iter().product()
    }

    /// `true` iff layout is row-major contiguous and `start_offset` is 0.
    /// This is the test downstream kernels gate `is_contiguous` on today.
    pub fn is_contiguous(&self) -> bool {
        self.start_offset == 0 && self.strides == row_major_strides(&self.shape)
    }

    /// Compute the byte footprint addressable by this layout, given
    /// `bytes_per_element`. **Includes `start_offset`**, so callers can
    /// use it to size a buffer big enough to hold the view.
    pub fn addressable_byte_size(&self, bytes_per_element: usize) -> usize {
        if self.shape.is_empty() {
            // Scalar: just 1 element starting at start_offset.
            return (self.start_offset + 1) * bytes_per_element;
        }
        let mut max_offset = self.start_offset;
        for (&dim, &stride) in self.shape.iter().zip(self.strides.iter()) {
            if dim == 0 {
                return 0;
            }
            max_offset += (dim - 1) * stride;
        }
        (max_offset + 1) * bytes_per_element
    }

    // ------------------------------------------------------------------
    // Zero-copy view operations (anti-pattern 10: must not copy)
    // ------------------------------------------------------------------

    /// Narrow along `axis` to `[offset .. offset + length]`. Zero-copy:
    /// only `start_offset` and `shape[axis]` change.
    ///
    /// Returns an error if `axis` is out of bounds or the requested
    /// range doesn't fit.
    pub fn narrow_axis(&self, axis: usize, offset: usize, length: usize) -> Result<Self> {
        if axis >= self.rank() {
            return Err(Error::Msg(format!(
                "Layout::narrow_axis: axis {} out of bounds (rank {})",
                axis,
                self.rank()
            )));
        }
        let dim = self.shape[axis];
        if offset + length > dim {
            return Err(Error::Msg(format!(
                "Layout::narrow_axis: offset={offset} + length={length} > dim={dim} on axis {axis}"
            )));
        }
        let mut shape = self.shape.clone();
        shape[axis] = length;
        let start_offset = self.start_offset + offset * self.strides[axis];
        Ok(Layout {
            shape,
            strides: self.strides.clone(),
            start_offset,
        })
    }

    /// Swap two axes. Zero-copy: shape and strides are permuted.
    pub fn transpose(&self, axis_a: usize, axis_b: usize) -> Result<Self> {
        let rank = self.rank();
        if axis_a >= rank || axis_b >= rank {
            return Err(Error::Msg(format!(
                "Layout::transpose: axes ({axis_a}, {axis_b}) out of bounds (rank {rank})"
            )));
        }
        let mut shape = self.shape.clone();
        let mut strides = self.strides.clone();
        shape.swap(axis_a, axis_b);
        strides.swap(axis_a, axis_b);
        Ok(Layout {
            shape,
            strides,
            start_offset: self.start_offset,
        })
    }

    /// Apply a full permutation of axes.
    ///
    /// `axes` is the new order: `result.shape[i] = self.shape[axes[i]]`.
    /// Returns an error if `axes` is not a permutation of `0..rank`.
    pub fn permute(&self, axes: &[usize]) -> Result<Self> {
        let rank = self.rank();
        if axes.len() != rank {
            return Err(Error::Msg(format!(
                "Layout::permute: axes length {} != rank {}",
                axes.len(),
                rank
            )));
        }
        let mut seen = vec![false; rank];
        for &a in axes {
            if a >= rank {
                return Err(Error::Msg(format!(
                    "Layout::permute: axis {a} out of bounds (rank {rank})"
                )));
            }
            if seen[a] {
                return Err(Error::Msg(format!(
                    "Layout::permute: axis {a} repeated in {axes:?}"
                )));
            }
            seen[a] = true;
        }
        let shape = axes.iter().map(|&a| self.shape[a]).collect();
        let strides = axes.iter().map(|&a| self.strides[a]).collect();
        Ok(Layout {
            shape,
            strides,
            start_offset: self.start_offset,
        })
    }

    /// Reshape into `new_shape`. **Only valid on a contiguous layout** —
    /// non-contiguous reshape would force a copy, which violates
    /// anti-pattern 10.
    ///
    /// Returns an error if the layout is not contiguous or if the new
    /// element count differs from the current one.
    pub fn reshape(&self, new_shape: impl Into<Vec<usize>>) -> Result<Self> {
        let new_shape: Vec<usize> = new_shape.into();
        if !self.is_contiguous() {
            return Err(Error::Msg(format!(
                "Layout::reshape: layout shape={:?} strides={:?} is not contiguous; \
                 callers must explicitly contiguous() first (logged via \
                 kiln_profile_contiguous_copy)",
                self.shape, self.strides
            )));
        }
        let want = new_shape.iter().product::<usize>();
        let have = self.element_count();
        if want != have {
            return Err(Error::Msg(format!(
                "Layout::reshape: shape {new_shape:?} has {want} elements; \
                 source has {have}"
            )));
        }
        Ok(Layout::contiguous(new_shape))
    }
}

/// Compute row-major strides for a shape.
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for (i, &dim) in shape.iter().enumerate().rev() {
        strides[i] = acc;
        acc *= dim;
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn contiguous_2d_strides() {
        let l = Layout::contiguous(vec![3, 4]);
        assert_eq!(l.shape(), &[3, 4]);
        assert_eq!(l.strides(), &[4, 1]);
        assert!(l.is_contiguous());
        assert_eq!(l.element_count(), 12);
        assert_eq!(l.rank(), 2);
    }

    #[test]
    fn contiguous_3d_strides() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        assert_eq!(l.strides(), &[12, 4, 1]);
        assert!(l.is_contiguous());
    }

    #[test]
    fn scalar_layout() {
        let l = Layout::contiguous(vec![]);
        assert_eq!(l.rank(), 0);
        assert_eq!(l.element_count(), 1); // empty product is 1
        assert!(l.is_contiguous());
    }

    #[test]
    fn narrow_advances_start_offset() {
        let l = Layout::contiguous(vec![4, 5]);
        let n = l.narrow_axis(0, 1, 2).unwrap();
        assert_eq!(n.shape(), &[2, 5]);
        assert_eq!(n.strides(), &[5, 1]);
        assert_eq!(n.start_offset(), 5); // skip 1 row of 5 elements
        // Not contiguous from start of buffer — start_offset != 0.
        assert!(!n.is_contiguous());
    }

    #[test]
    fn narrow_axis_out_of_bounds() {
        let l = Layout::contiguous(vec![4, 5]);
        let e = l.narrow_axis(2, 0, 1).unwrap_err();
        assert!(e.to_string().contains("axis 2 out of bounds"));
    }

    #[test]
    fn narrow_overrun_errors() {
        let l = Layout::contiguous(vec![4, 5]);
        let e = l.narrow_axis(1, 3, 5).unwrap_err();
        assert!(e.to_string().contains("> dim=5"));
    }

    #[test]
    fn transpose_swaps_shape_and_strides() {
        let l = Layout::contiguous(vec![3, 4]);
        let t = l.transpose(0, 1).unwrap();
        assert_eq!(t.shape(), &[4, 3]);
        assert_eq!(t.strides(), &[1, 4]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn permute_three_axes() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        let p = l.permute(&[2, 0, 1]).unwrap();
        assert_eq!(p.shape(), &[4, 2, 3]);
        assert_eq!(p.strides(), &[1, 12, 4]);
    }

    #[test]
    fn permute_rejects_non_permutation() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        assert!(l.permute(&[0, 0, 1]).is_err());
        assert!(l.permute(&[0, 1]).is_err());
        assert!(l.permute(&[0, 1, 5]).is_err());
    }

    #[test]
    fn reshape_contiguous_preserves_element_count() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        let r = l.reshape(vec![6, 4]).unwrap();
        assert_eq!(r.shape(), &[6, 4]);
        assert!(r.is_contiguous());
    }

    #[test]
    fn reshape_rejects_non_contiguous() {
        let l = Layout::contiguous(vec![3, 4]).transpose(0, 1).unwrap();
        let e = l.reshape(vec![12]).unwrap_err();
        assert!(e.to_string().contains("not contiguous"));
    }

    #[test]
    fn reshape_rejects_element_mismatch() {
        let l = Layout::contiguous(vec![2, 3, 4]);
        let e = l.reshape(vec![5, 5]).unwrap_err();
        assert!(e.to_string().contains("has 25 elements"));
    }

    #[test]
    fn addressable_byte_size_contiguous() {
        let l = Layout::contiguous(vec![3, 4]);
        assert_eq!(l.addressable_byte_size(4), 48); // 12 elements x 4 bytes
    }

    #[test]
    fn addressable_byte_size_narrowed() {
        // Start at offset 5, span shape [2,5] with strides [5,1] —
        // last addressable index is 5 + (2-1)*5 + (5-1)*1 = 14, so 15 elements.
        let n = Layout::contiguous(vec![4, 5]).narrow_axis(0, 1, 2).unwrap();
        assert_eq!(n.addressable_byte_size(4), 15 * 4);
    }

    #[test]
    fn addressable_byte_size_zero_dim() {
        let l = Layout::from_parts(vec![0, 5], vec![5, 1], 0).unwrap();
        assert_eq!(l.addressable_byte_size(4), 0);
    }

    #[test]
    fn from_parts_rejects_rank_mismatch() {
        let e = Layout::from_parts(vec![3], vec![1, 1], 0).unwrap_err();
        assert!(e.to_string().contains("shape rank"));
    }
}
