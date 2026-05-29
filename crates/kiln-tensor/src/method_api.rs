//! candle-API-compatible inherent-method façade for [`Tensor`].
//!
//! # Why this file exists (issue #1082)
//!
//! kiln-tensor's [`Tensor`] is a pure view/storage handle: every math
//! op lives as a *free function* in [`crate::ops`] with the receiver as
//! the first positional argument — `kiln_tensor::ops::matmul(&a, &b)`.
//! candle, by contrast, exposes the entire vocabulary as **methods** —
//! `a.matmul(&b)?`, `x.exp()?`, `t.to_dtype(dt)?`. That call-shape
//! mismatch is the single biggest blocker for the `forward.rs`
//! candle→kt type-flip: a mechanical `s/candle_core::Tensor/kiln_tensor::Tensor/`
//! only compiles if `kt::Tensor` answers the same method calls.
//!
//! This module adds those inherent methods. Each one **delegates** to
//! the existing `ops::` free function (or composes a few of them when
//! candle exposes a method kt only has as primitives). The signatures
//! are matched against `vendor/candle-core/src/tensor.rs` so the flip is
//! a no-brainer.
//!
//! # Purely additive
//!
//! Nothing in the tree calls these yet (the flip is a separate PR), so
//! this file cannot regress anything. Correctness is proven by the
//! `#[cfg(test)]` block below: every method is asserted equal to the
//! `ops::` free function (or a hand-computed small example) it wraps.
//!
//! # Out of scope (autograd island)
//!
//! Methods that would need kt-native tape wiring — `Var`,
//! `.backward()`, `.track_op()`, `.apply_op{1,2,3}()`, `from_storage`,
//! `slice_set`, in-place mutation — are **not** added here. They land in
//! a later #1082 phase. See the module-level skip-list in the PR notes.
//!
//! # The `Dim` / `D` shim
//!
//! candle types axis-taking methods over a `Dim` trait so callers can
//! write `x.sum(D::Minus1)` or `x.sum(2)` interchangeably. `forward.rs`
//! uses `D::Minus1` 19 times, so the flip needs the same ergonomics.
//! [`Dim`] (impl'd for `usize` and [`D`]) reproduces candle's
//! negative-axis resolution. The kt `ops::` reductions are single-axis
//! (`axis: usize`), and `forward.rs` only ever reduces a single axis, so
//! these methods take `D: Dim` (one axis) rather than candle's
//! multi-axis `Dims`. See the deviation note in the PR report.

use crate::ops;
use crate::{DType, Device, Element, Result, Tensor};

/// Negative-axis selector, mirroring `candle_core::D`.
///
/// Resolved against a tensor's rank by [`Dim::to_index`]. `forward.rs`
/// uses `D::Minus1` (last axis) pervasively; `D::Minus2` and
/// `D::Minus(n)` round out parity with candle's enum.
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum D {
    /// The last axis (`rank - 1`).
    Minus1,
    /// The second-to-last axis (`rank - 2`).
    Minus2,
    /// The `n`-th axis from the end (`rank - n`).
    Minus(usize),
}

/// An axis selector resolvable against a tensor's rank.
///
/// Mirrors `candle_core::shape::Dim`. Implemented for plain `usize`
/// (absolute axis) and [`D`] (negative axis). This is the trait that
/// lets `x.sum(2)` and `x.sum(D::Minus1)` both type-check, which is the
/// whole point of the shim for the `forward.rs` flip.
pub trait Dim {
    /// Resolve to an absolute axis index in `[0, rank)`.
    fn to_index(&self, rank: usize, op: &'static str) -> Result<usize>;
}

impl Dim for usize {
    fn to_index(&self, rank: usize, op: &'static str) -> Result<usize> {
        let dim = *self;
        if dim >= rank {
            return Err(crate::Error::Msg(format!(
                "{op}: axis {dim} out of range for rank {rank}"
            )));
        }
        Ok(dim)
    }
}

impl Dim for D {
    fn to_index(&self, rank: usize, op: &'static str) -> Result<usize> {
        let resolved = match self {
            D::Minus1 if rank >= 1 => Some(rank - 1),
            D::Minus2 if rank >= 2 => Some(rank - 2),
            D::Minus(u) if *u >= 1 && rank >= *u => Some(rank - *u),
            _ => None,
        };
        resolved.ok_or_else(|| {
            crate::Error::Msg(format!(
                "{op}: negative axis {self:?} out of range for rank {rank}"
            ))
        })
    }
}

/// Compute the numpy/candle right-aligned broadcast shape of two
/// shapes, or `Err` if they are incompatible.
///
/// Mirrors candle's `Shape::broadcast_shape_binary_op`. The two shapes
/// are right-aligned; for each aligned pair the output dim is the
/// non-1 value, equal values pass through, and missing leading dims of
/// the shorter shape are treated as 1.
fn broadcast_shape(lhs: &[usize], rhs: &[usize], op: &'static str) -> Result<Vec<usize>> {
    let out_rank = lhs.len().max(rhs.len());
    let mut out = vec![0usize; out_rank];
    for i in 0..out_rank {
        // Right-aligned: index from the end.
        let rev = out_rank - i;
        let l = if lhs.len() < rev {
            1
        } else {
            lhs[lhs.len() - rev]
        };
        let r = if rhs.len() < rev {
            1
        } else {
            rhs[rhs.len() - rev]
        };
        out[i] = if l == r {
            l
        } else if l == 1 {
            r
        } else if r == 1 {
            l
        } else {
            return Err(crate::Error::Msg(format!(
                "{op}: cannot broadcast shapes {lhs:?} and {rhs:?}"
            )));
        };
    }
    Ok(out)
}

/// Right-align `x` to `target` rank by left-padding with size-1 axes,
/// then materialize the broadcast via [`ops::broadcast_to`].
///
/// kt's `ops::broadcast_to` requires the input rank to equal the target
/// rank (it bails on rank mismatch). candle's `broadcast_as` left-pads
/// implicitly. This helper reproduces candle's behavior so the broadcast
/// binary ops and `broadcast_as`/`expand` work for rank-expanding cases.
fn broadcast_to_shape(x: &Tensor, target: &[usize]) -> Result<Tensor> {
    if x.shape() == target {
        return Ok(x.clone());
    }
    // Left-pad x's shape with leading 1s up to target rank.
    let cur = x.shape();
    if cur.len() > target.len() {
        return Err(crate::Error::Msg(format!(
            "broadcast_as: input rank {} exceeds target rank {} (shapes {cur:?} -> {target:?})",
            cur.len(),
            target.len()
        )));
    }
    let padded = if cur.len() == target.len() {
        x.clone()
    } else {
        let mut padded_shape = vec![1usize; target.len() - cur.len()];
        padded_shape.extend_from_slice(cur);
        x.reshape(padded_shape)?
    };
    ops::broadcast_to(&padded, target)
}

impl Tensor {
    // ==================================================================
    // Binary elementwise (same-shape) — candle `binary_op!`
    // candle: `pub fn add(&self, rhs: &Self) -> Result<Self>`
    // ==================================================================

    /// Elementwise `self + rhs` (same shape). Delegates to [`ops::add`].
    pub fn add(&self, rhs: &Self) -> Result<Self> {
        ops::add(self, rhs)
    }

    /// Elementwise `self - rhs` (same shape). Delegates to [`ops::sub`].
    pub fn sub(&self, rhs: &Self) -> Result<Self> {
        ops::sub(self, rhs)
    }

    /// Elementwise `self * rhs` (same shape). Delegates to [`ops::mul`].
    pub fn mul(&self, rhs: &Self) -> Result<Self> {
        ops::mul(self, rhs)
    }

    /// Elementwise `self / rhs` (same shape). Delegates to [`ops::div`].
    pub fn div(&self, rhs: &Self) -> Result<Self> {
        ops::div(self, rhs)
    }

    /// Elementwise `max(self, rhs)`. Delegates to [`ops::maximum`].
    ///
    /// candle types this `<T: TensorOrScalar>`; kt only needs the
    /// tensor-tensor form for the flip, so the arg is `&Self`.
    pub fn maximum(&self, rhs: &Self) -> Result<Self> {
        ops::maximum(self, rhs)
    }

    /// Elementwise `min(self, rhs)`. Delegates to [`ops::minimum`].
    pub fn minimum(&self, rhs: &Self) -> Result<Self> {
        ops::minimum(self, rhs)
    }

    // ==================================================================
    // Broadcasting binary elementwise — candle `broadcast_binary_op!`
    // candle: `pub fn broadcast_add(&self, rhs: &Self) -> Result<Self>`
    // ==================================================================

    /// Broadcasting `self + rhs` (numpy rules). Composes
    /// [`ops::broadcast_to`] + [`ops::add`].
    pub fn broadcast_add(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_add")?;
        ops::add(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    /// Broadcasting `self - rhs` (numpy rules).
    pub fn broadcast_sub(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_sub")?;
        ops::sub(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    /// Broadcasting `self * rhs` (numpy rules).
    pub fn broadcast_mul(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_mul")?;
        ops::mul(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    /// Broadcasting `self / rhs` (numpy rules).
    pub fn broadcast_div(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_div")?;
        ops::div(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    /// Broadcasting elementwise `max(self, rhs)` (numpy rules).
    pub fn broadcast_maximum(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_maximum")?;
        ops::maximum(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    /// Broadcasting elementwise `min(self, rhs)` (numpy rules).
    pub fn broadcast_minimum(&self, rhs: &Self) -> Result<Self> {
        let shape = broadcast_shape(self.shape(), rhs.shape(), "broadcast_minimum")?;
        ops::minimum(
            &broadcast_to_shape(self, &shape)?,
            &broadcast_to_shape(rhs, &shape)?,
        )
    }

    // ==================================================================
    // Matmul — candle `pub fn matmul(&self, rhs: &Self) -> Result<Self>`
    // ==================================================================

    /// Matrix multiply `self @ rhs`. Delegates to [`ops::matmul`].
    pub fn matmul(&self, rhs: &Self) -> Result<Self> {
        ops::matmul(self, rhs)
    }

    /// Batched matmul with broadcast batch dims.
    ///
    /// candle: `pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self>`.
    /// The trailing two axes are the matrix; leading batch axes are
    /// numpy-broadcast, then [`ops::matmul`] runs on the aligned
    /// operands.
    pub fn broadcast_matmul(&self, rhs: &Self) -> Result<Self> {
        let l = self.shape();
        let r = rhs.shape();
        if l.len() < 2 || r.len() < 2 {
            return Err(crate::Error::Msg(format!(
                "broadcast_matmul: both operands must be rank >= 2, got {l:?} and {r:?}"
            )));
        }
        let (m, lk) = (l[l.len() - 2], l[l.len() - 1]);
        let (rk, n) = (r[r.len() - 2], r[r.len() - 1]);
        if lk != rk {
            return Err(crate::Error::Msg(format!(
                "broadcast_matmul: inner dims differ ({lk} vs {rk}) for {l:?} and {r:?}"
            )));
        }
        // Broadcast the batch (leading) dims.
        let batch = broadcast_shape(
            &l[..l.len() - 2],
            &r[..r.len() - 2],
            "broadcast_matmul",
        )?;
        let mut lhs_shape = batch.clone();
        lhs_shape.extend_from_slice(&[m, lk]);
        let mut rhs_shape = batch;
        rhs_shape.extend_from_slice(&[rk, n]);
        ops::matmul(
            &broadcast_to_shape(self, &lhs_shape)?,
            &broadcast_to_shape(rhs, &rhs_shape)?,
        )
    }

    // ==================================================================
    // Broadcast-as / expand — candle aliases, delegate to broadcast_to.
    // candle: `pub fn broadcast_as<S: Into<Shape>>(&self, shape: S)`
    //         `pub fn expand<S: Into<Shape>>(&self, shape: S)`
    // ==================================================================

    /// Broadcast `self` to `shape` (size-1 axes replicated, leading
    /// axes left-padded). candle's `broadcast_as`.
    pub fn broadcast_as(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        broadcast_to_shape(self, &shape.into())
    }

    /// Alias for [`Tensor::broadcast_as`] (candle's `expand`).
    pub fn expand(&self, shape: impl Into<Vec<usize>>) -> Result<Self> {
        broadcast_to_shape(self, &shape.into())
    }

    // ==================================================================
    // Unary elementwise — candle `unary_op!`
    // candle: `pub fn exp(&self) -> Result<Self>` etc.
    // ==================================================================

    /// `exp(self)`. Delegates to [`ops::exp`].
    pub fn exp(&self) -> Result<Self> {
        ops::exp(self)
    }

    /// `sqrt(self)`. Delegates to [`ops::sqrt`].
    pub fn sqrt(&self) -> Result<Self> {
        ops::sqrt(self)
    }

    /// `-self`. Delegates to [`ops::neg`].
    pub fn neg(&self) -> Result<Self> {
        ops::neg(self)
    }

    /// `|self|`. Delegates to [`ops::abs`].
    pub fn abs(&self) -> Result<Self> {
        ops::abs(self)
    }

    /// Natural log `ln(self)`. candle names this `log`; delegates to
    /// [`ops::ln`].
    pub fn log(&self) -> Result<Self> {
        ops::ln(self)
    }

    /// `1/self`. candle names this `recip`; delegates to
    /// [`ops::reciprocal`].
    pub fn recip(&self) -> Result<Self> {
        ops::reciprocal(self)
    }

    /// `self * self`. candle's `sqr`; composed as [`ops::mul`]`(self, self)`.
    pub fn sqr(&self) -> Result<Self> {
        ops::mul(self, self)
    }

    /// `1/sqrt(self)`. Composed as `sqrt` then [`ops::reciprocal`].
    /// (candle has no `rsqrt` method, but `forward.rs` uses the idiom.)
    pub fn rsqrt(&self) -> Result<Self> {
        ops::reciprocal(&ops::sqrt(self)?)
    }

    /// `sin(self)`. Delegates to [`ops::sin`].
    pub fn sin(&self) -> Result<Self> {
        ops::sin(self)
    }

    /// `cos(self)`. Delegates to [`ops::cos`].
    pub fn cos(&self) -> Result<Self> {
        ops::cos(self)
    }

    /// `tanh(self)`. Delegates to [`ops::tanh`].
    pub fn tanh(&self) -> Result<Self> {
        ops::tanh(self)
    }

    /// `gelu(self)`. Delegates to [`ops::gelu`].
    pub fn gelu(&self) -> Result<Self> {
        ops::gelu(self)
    }

    /// `relu(self)`. Delegates to [`ops::relu`].
    pub fn relu(&self) -> Result<Self> {
        ops::relu(self)
    }

    /// `silu(self)` (a.k.a. swish). Delegates to [`ops::silu`].
    pub fn silu(&self) -> Result<Self> {
        ops::silu(self)
    }

    /// `sigmoid(self)`. Delegates to [`ops::sigmoid`].
    pub fn sigmoid(&self) -> Result<Self> {
        ops::sigmoid(self)
    }

    /// `log2(self)`. Delegates to [`ops::log2`].
    pub fn log2(&self) -> Result<Self> {
        ops::log2(self)
    }

    /// `log10(self)`. Delegates to [`ops::log10`].
    pub fn log10(&self) -> Result<Self> {
        ops::log10(self)
    }

    // ==================================================================
    // Scalar-arg ops
    // ==================================================================

    /// `self * e` (elementwise power). candle:
    /// `pub fn powf(&self, e: f64) -> Result<Self>`. Delegates to
    /// [`ops::pow`] (kt takes `f32`; the `f64` arg is narrowed to match
    /// candle's call shape).
    pub fn powf(&self, e: f64) -> Result<Self> {
        ops::pow(self, e as f32)
    }

    /// `clamp(self, min, max)`. candle:
    /// `pub fn clamp<T1: TensorOrScalar, T2: TensorOrScalar>(&self, min, max)`.
    /// kt only needs the scalar-scalar form for the flip. Delegates to
    /// [`ops::clamp`] (`f32` bounds).
    pub fn clamp(&self, min: f64, max: f64) -> Result<Self> {
        ops::clamp(self, min as f32, max as f32)
    }

    /// `self * mul + add` (affine). candle:
    /// `pub fn affine(&self, mul: f64, add: f64) -> Result<Self>`.
    /// Composed as [`ops::mul_scalar`] then [`ops::add_scalar`].
    pub fn affine(&self, mul: f64, add: f64) -> Result<Self> {
        ops::add_scalar(&ops::mul_scalar(self, mul as f32)?, add as f32)
    }

    /// L_p normalization. kt's `ops::normalize(x, p, eps)`. (Not a
    /// candle-core method — candle uses `candle_nn`'s `normalize`; this
    /// mirrors the kt free fn for call sites that already use it.)
    pub fn normalize(&self, p: f32, eps: f32) -> Result<Self> {
        ops::normalize(self, p, eps)
    }

    // ==================================================================
    // Cast / device
    // ==================================================================

    /// Cast to `dtype`. candle:
    /// `pub fn to_dtype(&self, dtype: DType) -> Result<Self>`. Delegates
    /// to [`ops::cast`].
    pub fn to_dtype(&self, dtype: DType) -> Result<Self> {
        ops::cast(self, dtype)
    }

    // NOTE: `to_device` already exists as an inherent method on Tensor
    // (see tensor.rs). candle's signature is `to_device(&self, &Device)`
    // vs kt's `to_device(&self, Device)` (kt Device is `Copy`). That
    // deviation is pre-existing and intentional; not re-added here.

    // ==================================================================
    // where_cond — candle: `mask.where_cond(&on_true, &on_false)`
    // kt free fn: `where_select(mask, t, f)`  (same arg order).
    // ==================================================================

    /// Select elementwise: where `self` (the mask) is nonzero take
    /// `on_true`, else `on_false`. candle's `where_cond`; delegates to
    /// [`ops::where_select`] with `self` as the mask.
    pub fn where_cond(&self, on_true: &Self, on_false: &Self) -> Result<Self> {
        ops::where_select(self, on_true, on_false)
    }

    // ==================================================================
    // softmax — candle uses candle_nn::ops::softmax_last_dim; kt has the
    // op directly. Expose a method so call sites read `x.softmax_last_dim()`.
    // ==================================================================

    /// Softmax over the last axis. Delegates to
    /// [`ops::softmax_last_dim`]. (candle has this as
    /// `candle_nn::ops::softmax_last_dim(&x)`, a free fn, not a
    /// `Tensor` method — callers using the candle-nn free fn keep that
    /// form; this method is a convenience for the kt side.)
    pub fn softmax_last_dim(&self) -> Result<Self> {
        ops::softmax_last_dim(self)
    }

    // ==================================================================
    // Axis reductions — candle `sum`/`mean`/`max`/`min` REMOVE the axis,
    // `*_keepdim` keeps it as size-1.
    // candle: `pub fn sum<D: Dims>(&self, dims: D)` (multi-dim).
    // kt deviation: single-axis `D: Dim` (matches all forward.rs usage).
    // ==================================================================

    /// Sum over `dim`, removing the axis. candle's `sum` (single-axis
    /// form). Delegates to [`ops::sum_axis`].
    pub fn sum<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "sum")?;
        ops::sum_axis(self, axis)
    }

    /// Sum over `dim`, keeping the axis as size-1. candle's
    /// `sum_keepdim`. Composed as [`ops::sum_axis`] then `unsqueeze`.
    pub fn sum_keepdim<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "sum_keepdim")?;
        ops::sum_axis(self, axis)?.unsqueeze(axis)
    }

    /// Mean over `dim`, removing the axis. candle's `mean` (single-axis
    /// form). Delegates to [`ops::mean_axis`].
    pub fn mean<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "mean")?;
        ops::mean_axis(self, axis)
    }

    /// Mean over `dim`, keeping the axis as size-1. candle's
    /// `mean_keepdim`. Composed as [`ops::mean_axis`] then `unsqueeze`.
    pub fn mean_keepdim<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "mean_keepdim")?;
        ops::mean_axis(self, axis)?.unsqueeze(axis)
    }

    /// Max over `dim`, removing the axis. candle's `max`. Delegates to
    /// [`ops::max_axis`].
    pub fn max<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "max")?;
        ops::max_axis(self, axis)
    }

    /// Max over `dim`, keeping the axis as size-1. candle's
    /// `max_keepdim`. Composed as [`ops::max_axis`] then `unsqueeze`.
    pub fn max_keepdim<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "max_keepdim")?;
        ops::max_axis(self, axis)?.unsqueeze(axis)
    }

    /// Min over `dim`, removing the axis. candle's `min`. Delegates to
    /// [`ops::min_axis`].
    pub fn min<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "min")?;
        ops::min_axis(self, axis)
    }

    /// Min over `dim`, keeping the axis as size-1. candle's
    /// `min_keepdim`. Composed as [`ops::min_axis`] then `unsqueeze`.
    pub fn min_keepdim<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "min_keepdim")?;
        ops::min_axis(self, axis)?.unsqueeze(axis)
    }

    /// Sum of all elements (scalar output). candle's `sum_all`.
    /// Delegates to [`ops::sum_all`].
    pub fn sum_all(&self) -> Result<Self> {
        ops::sum_all(self)
    }

    /// Mean of all elements (scalar output). candle's `mean_all` is the
    /// `mean` over every dim; this delegates to [`ops::mean_all`].
    pub fn mean_all(&self) -> Result<Self> {
        ops::mean_all(self)
    }

    // ==================================================================
    // cumsum — candle: `pub fn cumsum<D: Dim>(&self, dim: D)`
    // ==================================================================

    /// Cumulative sum along `dim`. Delegates to [`ops::cumsum`].
    pub fn cumsum<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "cumsum")?;
        ops::cumsum(self, axis)
    }

    // ==================================================================
    // index_select — candle: `index_select<D: Dim>(&self, indexes, dim)`
    // kt free fn arg order: `index_select(input, axis, indices)`.
    // ==================================================================

    /// Gather rows of `self` along `dim` using integer `indexes`.
    /// candle's `index_select` (note: indexes first, dim second — the kt
    /// free fn is `(input, axis, indices)`, so this reorders).
    pub fn index_select<Dm: Dim>(&self, indexes: &Self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "index_select")?;
        ops::index_select(self, axis, indexes)
    }

    // ==================================================================
    // argmax — candle: `pub fn argmax<D: Dim>(&self, dim: D) -> Result<Self>`
    // candle squeezes the reduced axis (its `argmax`, *not*
    // `argmax_keepdim`). kt's [`ops::argmax_last_dim`] already squeezes
    // the *trailing* axis, so for a general `dim` we move that axis to
    // the end first (zero-copy permute), reduce, and let the squeeze
    // drop it — leaving the remaining axes in their original relative
    // order (exactly candle's output layout). Output dtype is `I64`
    // (kt's `argmax_last_dim` convention), tie-broken by lowest index
    // (matches `candle_core::Tensor::argmax`).
    // ==================================================================

    /// Index of the maximum along `dim`, with `dim` removed from the
    /// output shape. candle's `argmax` (squeezes the reduced axis).
    ///
    /// Delegates to [`ops::argmax_last_dim`] directly when `dim` is the
    /// last axis; otherwise [`Tensor::move_axis`]es the target axis to
    /// the end (zero-copy), reduces, and the trailing-axis squeeze
    /// leaves the other axes in their original order.
    pub fn argmax<Dm: Dim>(&self, dim: Dm) -> Result<Self> {
        let rank = self.rank();
        let axis = dim.to_index(rank, "argmax")?;
        if axis == rank - 1 {
            ops::argmax_last_dim(self)
        } else {
            // Move `axis` to the trailing position; the squeeze in
            // argmax_last_dim then drops it, leaving the remaining axes
            // in their original relative order. `move_axis` is a permute
            // (non-contiguous view); `ops::argmax_last_dim` requires a
            // contiguous input, so materialize first.
            let moved = self.move_axis(axis, rank - 1)?.contiguous()?;
            ops::argmax_last_dim(&moved)
        }
    }

    // ==================================================================
    // index_add — candle:
    //   `pub fn index_add<D: Dim>(&self, indexes, source, dim) -> Result<Self>`
    // candle returns `self` with `source` *added into* `self` at the
    // positions named by `indexes` along `dim`. kt's
    // [`ops::scatter_add`] is the "into-zeros" sibling (the
    // index_select backward): it builds a zero output of the target
    // shape and scatters `source` in. So candle's index_add is
    // `self + scatter_add(source, dim, indexes, self.dims()[dim])`.
    // ==================================================================

    /// Add `source` into a copy of `self` at the rows named by
    /// `indexes` along `dim`. candle's `index_add` (arg order:
    /// `indexes`, `source`, `dim`).
    ///
    /// Composed as `self + ops::scatter_add(source, dim, indexes,
    /// self.dims()[dim])`: `scatter_add` produces a zero tensor of
    /// `self`'s shape with `source` scattered into the indexed
    /// positions, and the elementwise add folds that onto `self`.
    pub fn index_add<Dm: Dim>(&self, indexes: &Self, source: &Self, dim: Dm) -> Result<Self> {
        let axis = dim.to_index(self.rank(), "index_add")?;
        let target_dim = self.shape()[axis];
        let scattered = ops::scatter_add(source, axis, indexes, target_dim)?;
        ops::add(self, &scattered)
    }

    // ==================================================================
    // Autograd-free shims — candle methods that interrogate or detach
    // the computation graph. kt's [`Tensor`] is a pure view/storage
    // handle with **no autograd graph** (the tape lives in a separate
    // crate), so these collapse to constants / identity / deep-copy.
    // ==================================================================

    /// Whether the autograd graph should track ops on this tensor.
    /// candle: `self.is_variable || self.op.is_some()`.
    ///
    /// **Always `false` for kt.** kt's [`Tensor`] carries no autograd
    /// metadata — it is a storage+layout handle, and gradient tracking
    /// lives entirely in the separate `kiln-autograd` tape, which is
    /// keyed off explicit `Var`s rather than a per-tensor `op` field.
    /// `forward.rs` reads `track_op()` (28 sites) purely to route
    /// between the autograd-safe slow path and the inference fast path;
    /// with no in-tensor graph, every kt tensor takes the fast path,
    /// which is correct because the tape-authoritative training path
    /// detaches its intermediates (see the `kiln-candle-autograd-drops`
    /// notes on #1082).
    pub fn track_op(&self) -> bool {
        false
    }

    /// Deep copy — a fresh storage allocation holding the same values.
    /// candle: `pub fn copy(&self) -> Result<Tensor>` ("copies the
    /// actual storage but may fail because of running out of memory").
    ///
    /// Unlike [`Clone`] (which shares the storage `Arc`) and
    /// [`Tensor::contiguous`] (which returns a *shared* clone when the
    /// input is already contiguous), `copy` always materializes a new
    /// backing buffer, so later in-place mutation of the source does
    /// not alias the copy. `forward.rs` relies on this to snapshot the
    /// GDN recurrent / conv states before they are mutated in place.
    ///
    /// - CUDA: routes through `cuda_contiguous`, which always allocates
    ///   a fresh device buffer (even for contiguous inputs).
    /// - CPU: rebuilds a fresh [`crate::CpuStorage`] from the
    ///   materialized row-major bytes.
    /// - Metal / Vulkan: not yet implemented (errors) — no `copy` call
    ///   site reaches those backends today (#1082).
    pub fn copy(&self) -> Result<Self> {
        match self.device() {
            #[cfg(feature = "cuda")]
            Device::Cuda(_) => crate::cuda_storage::cuda_contiguous(self),
            Device::Cpu => {
                // Materialize a contiguous CPU view, then rebuild fresh
                // storage from its addressable bytes so the result never
                // aliases `self`'s buffer.
                let contig = self.contiguous()?;
                let per = contig.dtype().size_in_bytes();
                let n = contig.element_count();
                let start_bytes = contig.layout().start_offset() * per;
                let end_bytes = start_bytes + n * per;
                let storage = contig
                    .storage()
                    .as_any()
                    .downcast_ref::<crate::CpuStorage>()
                    .ok_or_else(|| {
                        crate::Error::from_str("Tensor::copy: CPU device must hold CpuStorage")
                    })?;
                let bytes = storage.as_bytes();
                if end_bytes > bytes.len() {
                    return Err(crate::Error::Msg(format!(
                        "Tensor::copy: byte range {start_bytes}..{end_bytes} exceeds CPU \
                         storage length {}",
                        bytes.len()
                    )));
                }
                let fresh = crate::CpuStorage::from_bytes(
                    contig.dtype(),
                    bytes[start_bytes..end_bytes].to_vec(),
                )?;
                // Explicit `Storage` (= `Arc<dyn StorageBackend>`) so the
                // `Arc<CpuStorage>` unsized-coerces at the binding, matching
                // the `from_slice` constructor pattern.
                let storage_arc: crate::Storage = std::sync::Arc::new(fresh);
                Self::from_parts(
                    storage_arc,
                    crate::Layout::contiguous(contig.shape().to_vec()),
                    crate::TensorId::next(),
                )
            }
            other => Err(crate::Error::Msg(format!(
                "Tensor::copy: deep copy on {other} is not yet implemented (#1082)"
            ))),
        }
    }

    /// Detach from the autograd graph. candle:
    /// `pub fn detach(&self) -> Tensor` (returns a graph-free view that
    /// **shares** the storage; identity if already detached).
    ///
    /// kt has no in-tensor autograd graph, so detach is a plain
    /// identity: it returns a `Clone` (shared-storage handle). Matches
    /// candle's "already detached → same tensor" fast path, and
    /// candle's by-value `Tensor` return (not `Result`).
    pub fn detach(&self) -> Self {
        self.clone()
    }

    // ==================================================================
    // Tensor::empty — candle:
    //   `pub unsafe fn empty<S: Into<Shape>>(shape, dtype, &device)`
    // candle hands back *uninitialized* memory; kt has no uninitialized
    // allocator, so this zero-fills (a strictly safer superset — every
    // candle `empty` call site immediately overwrites the buffer). The
    // method is still `unsafe` to keep candle's call shape
    // (`unsafe { Tensor::empty(...) }`) compiling unchanged at the flip
    // sites.
    // ==================================================================

    /// Allocate a tensor of `shape`/`dtype` on `device`. candle's
    /// `Tensor::empty` returns uninitialized memory; kt zero-fills
    /// instead (no uninit allocator), which is a safe superset since
    /// callers overwrite the buffer immediately.
    ///
    /// # Safety
    ///
    /// Kept `unsafe` to mirror candle's signature so flip sites that
    /// write `unsafe { Tensor::empty(...) }` type-check unchanged. The
    /// kt impl is in fact memory-safe (it zero-initializes), but the
    /// `unsafe` marker is part of the API-compat contract.
    ///
    /// kt deviation: `device` is taken **by value** (`Device` is
    /// `Copy`), matching kt's other ctors (`zeros`/`ones`/`arange`);
    /// candle takes `&Device`. The flip handles the `&`→value at the
    /// far fewer ctor sites.
    pub unsafe fn empty(
        shape: impl Into<Vec<usize>>,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        Self::zeros_on(device, shape.into(), dtype)
    }

    // ==================================================================
    // Shape accessors — candle name aliases
    // ==================================================================

    /// Borrow the shape as `&[usize]`. candle's `dims` (alias for kt's
    /// existing `shape()`).
    pub fn dims(&self) -> &[usize] {
        self.shape()
    }

    /// Size of axis `dim`. candle: `pub fn dim<D: Dim>(&self, dim: D)`.
    pub fn dim<Dm: Dim>(&self, dim: Dm) -> Result<usize> {
        let axis = dim.to_index(self.rank(), "dim")?;
        Ok(self.shape()[axis])
    }

    /// Rank-2 shape as a tuple, erroring on other ranks. candle's
    /// `dims2`.
    pub fn dims2(&self) -> Result<(usize, usize)> {
        let d = self.shape();
        if d.len() != 2 {
            return Err(crate::Error::Msg(format!(
                "dims2: expected rank 2, got rank {} (shape {d:?})",
                d.len()
            )));
        }
        Ok((d[0], d[1]))
    }

    /// Rank-3 shape as a tuple, erroring on other ranks. candle's
    /// `dims3`.
    pub fn dims3(&self) -> Result<(usize, usize, usize)> {
        let d = self.shape();
        if d.len() != 3 {
            return Err(crate::Error::Msg(format!(
                "dims3: expected rank 3, got rank {} (shape {d:?})",
                d.len()
            )));
        }
        Ok((d[0], d[1], d[2]))
    }

    /// Rank-4 shape as a tuple, erroring on other ranks. candle's
    /// `dims4`.
    pub fn dims4(&self) -> Result<(usize, usize, usize, usize)> {
        let d = self.shape();
        if d.len() != 4 {
            return Err(crate::Error::Msg(format!(
                "dims4: expected rank 4, got rank {} (shape {d:?})",
                d.len()
            )));
        }
        Ok((d[0], d[1], d[2], d[3]))
    }

    /// Total element count. candle's `elem_count` (alias for kt's
    /// existing `element_count()`).
    pub fn elem_count(&self) -> usize {
        self.element_count()
    }

    /// Flatten to a rank-1 tensor. candle's `flatten_all` (alias for
    /// kt's existing `flatten()`).
    pub fn flatten_all(&self) -> Result<Self> {
        self.flatten()
    }

    // ==================================================================
    // Readback — candle: `to_vec1::<S>() -> Result<Vec<S>>`
    // ==================================================================

    /// Read a rank-1 tensor back to a host `Vec<E>`. candle's `to_vec1`.
    /// Errors if the tensor is not rank-1; otherwise delegates to kt's
    /// existing [`Tensor::to_vec`].
    pub fn to_vec1<E: Element>(&self) -> Result<Vec<E>> {
        if self.rank() != 1 {
            return Err(crate::Error::Msg(format!(
                "to_vec1: expected rank 1, got rank {} (shape {:?})",
                self.rank(),
                self.shape()
            )));
        }
        self.to_vec::<E>()
    }

    /// Read a rank-2 tensor back to a host `Vec<Vec<E>>` (row-major).
    /// candle: `pub fn to_vec2<S: WithDType>(&self) -> Result<Vec<Vec<S>>>`.
    ///
    /// Rank is asserted via [`Tensor::dims2`] (mirrors candle's
    /// `self.dims2()?` rank-check), then the flat row-major readback from
    /// kt's existing [`Tensor::to_vec`] is sliced into `dim0` rows of
    /// `dim1` elements. `to_vec` already returns contiguous row-major
    /// data, so the row split is a plain chunking — no stride walk.
    pub fn to_vec2<E: Element>(&self) -> Result<Vec<Vec<E>>> {
        let (d0, d1) = self.dims2()?;
        let flat = self.to_vec::<E>()?;
        let mut rows = Vec::with_capacity(d0);
        for r in 0..d0 {
            rows.push(flat[r * d1..(r + 1) * d1].to_vec());
        }
        Ok(rows)
    }

    /// Read a rank-3 tensor back to a host `Vec<Vec<Vec<E>>>` (row-major).
    /// candle: `pub fn to_vec3<S: WithDType>(&self) -> Result<Vec<Vec<Vec<S>>>>`.
    ///
    /// Rank is asserted via [`Tensor::dims3`] (mirrors candle's
    /// `self.dims3()?`), then the flat contiguous readback from
    /// [`Tensor::to_vec`] is nested into `d0 × d1 × d2`. Outer index strides
    /// by `d1 * d2`, middle by `d2`.
    pub fn to_vec3<E: Element>(&self) -> Result<Vec<Vec<Vec<E>>>> {
        let (d0, d1, d2) = self.dims3()?;
        let flat = self.to_vec::<E>()?;
        let plane = d1 * d2;
        let mut out = Vec::with_capacity(d0);
        for i in 0..d0 {
            let base = i * plane;
            let mut rows = Vec::with_capacity(d1);
            for j in 0..d1 {
                let row_base = base + j * d2;
                rows.push(flat[row_base..row_base + d2].to_vec());
            }
            out.push(rows);
        }
        Ok(out)
    }

    // ==================================================================
    // Finite / infinite masks — elementwise predicate masks.
    //
    // candle-core has NO `is_finite`/`is_infinite` *method* on `Tensor`
    // (kt previously only had the reduce-to-bool `all_finite()`), so
    // there is no candle signature to copy here. These follow the
    // numpy/torch `isfinite`/`isinf` *elementwise-mask* convention the
    // #1082 flip needs, and the kt `ops::compare` mask convention: a U8
    // tensor of the same shape, `1` where the predicate holds, `0`
    // otherwise. Composed entirely from existing kt primitives
    // (`abs` + a single `lt`/`eq` against a `full_like(±inf)` constant).
    // ==================================================================

    /// Elementwise finite mask: U8, same shape, `1` where the element is
    /// finite (not ±inf, not NaN), `0` otherwise.
    ///
    /// Composition: `abs(self) < +inf`. IEEE-754 makes this exact in one
    /// compare — for a finite `v`, `|v| < inf` is true; for `±inf`,
    /// `inf < inf` is false; for `NaN`, every ordered compare (including
    /// `NaN < inf`) is false. So no NaN-specific term and no U8 logical-AND
    /// is needed (kt's `mul`/`minimum` reject U8 anyway). [`ops::abs`]
    /// materializes a contiguous tensor, so the [`ops::lt`] contiguity
    /// requirement holds regardless of `self`'s layout.
    ///
    /// dtype must be F32/BF16/F16 (the dtypes [`ops::abs`]/[`ops::lt`]
    /// accept); other dtypes error from the composed ops.
    pub fn is_finite(&self) -> Result<Self> {
        let absx = ops::abs(self)?;
        let inf = ops::full_like(&absx, f32::INFINITY)?;
        ops::lt(&absx, &inf)
    }

    /// Elementwise infinite mask: U8, same shape, `1` where the element is
    /// `+inf` or `-inf`, `0` everywhere else (including NaN).
    ///
    /// Composition: `abs(self) == +inf`. For `±inf`, `inf == inf` is true;
    /// for any finite value the equality is false; for `NaN`, `NaN == inf`
    /// is false (matching candle/torch `isinf`, which is `0` at NaN).
    /// Single `eq` against a `full_like(+inf)` constant.
    ///
    /// dtype must be F32/BF16/F16; other dtypes error from the composed ops.
    pub fn is_infinite(&self) -> Result<Self> {
        let absx = ops::abs(self)?;
        let inf = ops::full_like(&absx, f32::INFINITY)?;
        ops::eq(&absx, &inf)
    }

    // ==================================================================
    // Associated constructors — candle: `Tensor::zeros(shape, dt, &dev)`
    // kt Device is `Copy`, so these take `Device` by value (deviation
    // from candle's `&Device`; the flip handles the `&`→value at the
    // ctor call sites, which are far fewer than the method sites).
    // ==================================================================

    /// Zero-initialized tensor of `shape`/`dtype` on `device`. candle:
    /// `Tensor::zeros<S: Into<Shape>>(shape, dtype, &device)`. Delegates
    /// to [`Tensor::zeros_on`].
    pub fn zeros(shape: impl Into<Vec<usize>>, dtype: DType, device: Device) -> Result<Self> {
        Self::zeros_on(device, shape.into(), dtype)
    }

    /// Ones tensor of `shape`/`dtype` on `device`. candle:
    /// `Tensor::ones<S: Into<Shape>>(shape, dtype, &device)`. Builds via
    /// `zeros_on` then [`ops::full_like`] with value 1.
    pub fn ones(shape: impl Into<Vec<usize>>, dtype: DType, device: Device) -> Result<Self> {
        let z = Self::zeros_on(device, shape.into(), dtype)?;
        ops::full_like(&z, 1.0)
    }

    /// Zeros with the same shape/dtype/device as `self`. candle's
    /// `zeros_like`. Delegates to [`ops::zeros_like`].
    pub fn zeros_like(&self) -> Result<Self> {
        ops::zeros_like(self)
    }

    /// Ones with the same shape/dtype/device as `self`. candle's
    /// `ones_like`. Delegates to [`ops::ones_like`].
    pub fn ones_like(&self) -> Result<Self> {
        ops::ones_like(self)
    }

    /// Concatenate `tensors` along `dim`. candle:
    /// `Tensor::cat<A: AsRef<Tensor>, D: Dim>(args: &[A], dim: D)`.
    /// kt's free fn takes `&[&Tensor]`; this accepts `&[&Tensor]` to
    /// match the common flip call shape and resolves a single absolute
    /// axis. Delegates to [`ops::concat`].
    pub fn cat(tensors: &[&Self], dim: usize) -> Result<Self> {
        ops::concat(tensors, dim)
    }

    /// 1-D tensor `[start, start+1, …, end)` on `device`. candle:
    /// `Tensor::arange<D: WithDType>(start, end, &device)` (step 1).
    /// Built on CPU via [`ops::arange`] (step 1.0, F32) then moved to
    /// `device`.
    pub fn arange(start: f32, end: f32, device: Device) -> Result<Self> {
        let cpu = ops::arange(start, end, 1.0, DType::F32)?;
        cpu.to_device(device)
    }

    /// Normal-distributed tensor `N(mean, std²)`. candle:
    /// `Tensor::randn<S, T>(mean, std, shape, &device)` using a global
    /// RNG. **kt deviation:** kt's RNG is *seedable* and has no global
    /// state, so this uses a fixed deterministic seed
    /// (`KT_RANDN_DEFAULT_SEED`). Results are reproducible but will NOT
    /// match candle's global-RNG draws. Built on CPU then moved to
    /// `device`.
    pub fn randn(
        mean: f32,
        std: f32,
        shape: impl Into<Vec<usize>>,
        device: Device,
    ) -> Result<Self> {
        const KT_RANDN_DEFAULT_SEED: u64 = 0x6b696c6e_72616e64; // "kiln rand"
        let cpu = ops::rand_normal(shape.into(), mean, std, KT_RANDN_DEFAULT_SEED, DType::F32)?;
        cpu.to_device(device)
    }

    /// `Tensor::new(data, &device)` candle ctor for 1-D host data.
    ///
    /// candle's `new` accepts an `NdArray` (nested arrays of arbitrary
    /// rank). kt has no `NdArray` abstraction, so this façade covers the
    /// **rank-1 slice** case (`Tensor::new(&[1f32, 2., 3.], &dev)`),
    /// which is the dominant flip pattern. Higher-rank `new` call sites
    /// flip to `from_slice` + `reshape`. Built via
    /// [`Tensor::from_slice`] then moved to `device`.
    pub fn new<E: Element>(data: &[E], device: Device) -> Result<Self> {
        let cpu = Self::from_slice(data, vec![data.len()])?;
        cpu.to_device(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape.to_vec()).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    // --- D / Dim shim --------------------------------------------------

    #[test]
    fn dim_usize_resolves_absolute() {
        assert_eq!(Dim::to_index(&0usize, 3, "t").unwrap(), 0);
        assert_eq!(Dim::to_index(&2usize, 3, "t").unwrap(), 2);
        assert!(Dim::to_index(&3usize, 3, "t").is_err());
    }

    #[test]
    fn dim_d_resolves_negative() {
        assert_eq!(D::Minus1.to_index(3, "t").unwrap(), 2);
        assert_eq!(D::Minus2.to_index(3, "t").unwrap(), 1);
        assert_eq!(D::Minus(3).to_index(3, "t").unwrap(), 0);
        assert!(D::Minus2.to_index(1, "t").is_err());
    }

    // --- binary elementwise -------------------------------------------

    #[test]
    fn add_sub_mul_div_match_ops() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        assert_eq!(v(&a.add(&b).unwrap()), v(&ops::add(&a, &b).unwrap()));
        assert_eq!(v(&a.sub(&b).unwrap()), v(&ops::sub(&a, &b).unwrap()));
        assert_eq!(v(&a.mul(&b).unwrap()), v(&ops::mul(&a, &b).unwrap()));
        assert_eq!(v(&a.div(&b).unwrap()), v(&ops::div(&a, &b).unwrap()));
    }

    #[test]
    fn maximum_minimum_match_ops() {
        let a = t(&[1.0, 9.0, 3.0, 2.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 1.0, 8.0], &[2, 2]);
        assert_eq!(v(&a.maximum(&b).unwrap()), v(&ops::maximum(&a, &b).unwrap()));
        assert_eq!(v(&a.minimum(&b).unwrap()), v(&ops::minimum(&a, &b).unwrap()));
    }

    // --- broadcasting binary ------------------------------------------

    #[test]
    fn broadcast_add_size1_axis() {
        // [2,3] + [2,1] -> [2,3], second operand replicated across cols.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[10.0, 20.0], &[2, 1]);
        let got = a.broadcast_add(&b).unwrap();
        assert_eq!(got.shape(), &[2, 3]);
        assert_eq!(v(&got), vec![11.0, 12.0, 13.0, 24.0, 25.0, 26.0]);
    }

    #[test]
    fn broadcast_mul_rank_expansion() {
        // [3] * [2,3] -> [2,3] (leading axis left-padded to size 1).
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[1.0, 1.0, 1.0, 2.0, 2.0, 2.0], &[2, 3]);
        let got = a.broadcast_mul(&b).unwrap();
        assert_eq!(got.shape(), &[2, 3]);
        assert_eq!(v(&got), vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0]);
    }

    #[test]
    fn broadcast_sub_div_min_max_shapes() {
        let a = t(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[2, 3]);
        let b = t(&[1.0, 2.0], &[2, 1]);
        assert_eq!(a.broadcast_sub(&b).unwrap().shape(), &[2, 3]);
        assert_eq!(a.broadcast_div(&b).unwrap().shape(), &[2, 3]);
        assert_eq!(a.broadcast_maximum(&b).unwrap().shape(), &[2, 3]);
        assert_eq!(a.broadcast_minimum(&b).unwrap().shape(), &[2, 3]);
        // broadcast_sub spot value: row0 - 1, row1 - 2.
        assert_eq!(
            v(&a.broadcast_sub(&b).unwrap()),
            vec![9.0, 19.0, 29.0, 38.0, 48.0, 58.0]
        );
    }

    // --- matmul --------------------------------------------------------

    #[test]
    fn matmul_matches_ops() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        assert_eq!(v(&a.matmul(&b).unwrap()), v(&ops::matmul(&a, &b).unwrap()));
    }

    #[test]
    fn broadcast_matmul_batch_expansion() {
        // lhs [1,2,2] @ rhs [3,2,2] -> [3,2,2] (lhs batch broadcast to 3).
        let a = t(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]); // identity
        let b = t(
            &[
                1.0, 2.0, 3.0, 4.0, // batch 0
                5.0, 6.0, 7.0, 8.0, // batch 1
                9.0, 10.0, 11.0, 12.0, // batch 2
            ],
            &[3, 2, 2],
        );
        let got = a.broadcast_matmul(&b).unwrap();
        assert_eq!(got.shape(), &[3, 2, 2]);
        // identity @ b == b for every batch.
        assert_eq!(v(&got), v(&b));
    }

    // --- broadcast_as / expand ----------------------------------------

    #[test]
    fn broadcast_as_and_expand_alias() {
        let a = t(&[1.0, 2.0, 3.0], &[1, 3]);
        let ba = a.broadcast_as(vec![2, 3]).unwrap();
        let ex = a.expand(vec![2, 3]).unwrap();
        assert_eq!(ba.shape(), &[2, 3]);
        assert_eq!(v(&ba), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
        assert_eq!(v(&ba), v(&ex));
    }

    #[test]
    fn broadcast_as_left_pads_rank() {
        // [3] -> [2,3] via implicit leading size-1 axis.
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let got = a.broadcast_as(vec![2, 3]).unwrap();
        assert_eq!(got.shape(), &[2, 3]);
        assert_eq!(v(&got), vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0]);
    }

    // --- unary ---------------------------------------------------------

    #[test]
    fn unary_methods_match_ops() {
        let x = t(&[0.5, 1.0, 2.0, 4.0], &[2, 2]);
        assert_eq!(v(&x.exp().unwrap()), v(&ops::exp(&x).unwrap()));
        assert_eq!(v(&x.sqrt().unwrap()), v(&ops::sqrt(&x).unwrap()));
        assert_eq!(v(&x.neg().unwrap()), v(&ops::neg(&x).unwrap()));
        assert_eq!(v(&x.abs().unwrap()), v(&ops::abs(&x).unwrap()));
        assert_eq!(v(&x.log().unwrap()), v(&ops::ln(&x).unwrap()));
        assert_eq!(v(&x.recip().unwrap()), v(&ops::reciprocal(&x).unwrap()));
        assert_eq!(v(&x.sin().unwrap()), v(&ops::sin(&x).unwrap()));
        assert_eq!(v(&x.cos().unwrap()), v(&ops::cos(&x).unwrap()));
        assert_eq!(v(&x.tanh().unwrap()), v(&ops::tanh(&x).unwrap()));
        assert_eq!(v(&x.gelu().unwrap()), v(&ops::gelu(&x).unwrap()));
        assert_eq!(v(&x.relu().unwrap()), v(&ops::relu(&x).unwrap()));
        assert_eq!(v(&x.silu().unwrap()), v(&ops::silu(&x).unwrap()));
        assert_eq!(v(&x.sigmoid().unwrap()), v(&ops::sigmoid(&x).unwrap()));
        assert_eq!(v(&x.log2().unwrap()), v(&ops::log2(&x).unwrap()));
        assert_eq!(v(&x.log10().unwrap()), v(&ops::log10(&x).unwrap()));
    }

    #[test]
    fn sqr_equals_self_times_self() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(v(&x.sqr().unwrap()), vec![1.0, 4.0, 9.0, 16.0]);
        assert_eq!(v(&x.sqr().unwrap()), v(&ops::mul(&x, &x).unwrap()));
    }

    #[test]
    fn rsqrt_equals_recip_sqrt() {
        let x = t(&[1.0, 4.0, 16.0, 100.0], &[2, 2]);
        let got = v(&x.rsqrt().unwrap());
        let want = vec![1.0, 0.5, 0.25, 0.1];
        for (g, w) in got.iter().zip(want.iter()) {
            assert!((g - w).abs() < 1e-5, "rsqrt: got {g} want {w}");
        }
    }

    // --- scalar ops ----------------------------------------------------

    #[test]
    fn powf_matches_ops_pow() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(v(&x.powf(2.0).unwrap()), v(&ops::pow(&x, 2.0).unwrap()));
    }

    #[test]
    fn clamp_matches_ops() {
        let x = t(&[-1.0, 0.5, 2.0, 5.0], &[2, 2]);
        assert_eq!(
            v(&x.clamp(0.0, 3.0).unwrap()),
            v(&ops::clamp(&x, 0.0, 3.0).unwrap())
        );
    }

    #[test]
    fn affine_matches_mul_then_add_scalar() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let got = x.affine(2.0, 1.0).unwrap();
        // 2x + 1
        assert_eq!(v(&got), vec![3.0, 5.0, 7.0, 9.0]);
        let want = ops::add_scalar(&ops::mul_scalar(&x, 2.0).unwrap(), 1.0).unwrap();
        assert_eq!(v(&got), v(&want));
    }

    // --- cast ----------------------------------------------------------

    #[test]
    fn to_dtype_matches_ops_cast() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let m = x.to_dtype(DType::BF16).unwrap();
        let f = ops::cast(&x, DType::BF16).unwrap();
        assert_eq!(m.dtype(), DType::BF16);
        assert_eq!(m.shape(), f.shape());
    }

    // --- where_cond ----------------------------------------------------

    #[test]
    fn where_cond_matches_ops_where_select() {
        // kt's where_select requires a U8 mask (same shape as t/f).
        let mask = Tensor::from_slice(&[1u8, 0, 0, 1], vec![2, 2]).unwrap();
        let on_true = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let on_false = t(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
        let got = mask.where_cond(&on_true, &on_false).unwrap();
        let want = ops::where_select(&mask, &on_true, &on_false).unwrap();
        assert_eq!(v(&got), v(&want));
        assert_eq!(v(&got), vec![10.0, -2.0, -3.0, 40.0]);
    }

    // --- softmax -------------------------------------------------------

    #[test]
    fn softmax_last_dim_matches_ops() {
        let x = t(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]);
        assert_eq!(
            v(&x.softmax_last_dim().unwrap()),
            v(&ops::softmax_last_dim(&x).unwrap())
        );
    }

    // --- reductions ----------------------------------------------------

    #[test]
    fn sum_removes_axis_keepdim_keeps() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        // sum over last axis -> [2]
        let s = x.sum(D::Minus1).unwrap();
        assert_eq!(s.shape(), &[2]);
        assert_eq!(v(&s), vec![6.0, 15.0]);
        // keepdim -> [2,1]
        let sk = x.sum_keepdim(D::Minus1).unwrap();
        assert_eq!(sk.shape(), &[2, 1]);
        assert_eq!(v(&sk), vec![6.0, 15.0]);
        // usize axis form
        let s0 = x.sum(0usize).unwrap();
        assert_eq!(s0.shape(), &[3]);
        assert_eq!(v(&s0), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn mean_removes_axis_keepdim_keeps() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let m = x.mean(1usize).unwrap();
        assert_eq!(m.shape(), &[2]);
        assert_eq!(v(&m), vec![2.0, 5.0]);
        let mk = x.mean_keepdim(1usize).unwrap();
        assert_eq!(mk.shape(), &[2, 1]);
        assert_eq!(v(&mk), vec![2.0, 5.0]);
    }

    #[test]
    fn max_min_axis_and_keepdim() {
        let x = t(&[1.0, 9.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
        let mx = x.max(D::Minus1).unwrap();
        assert_eq!(mx.shape(), &[2]);
        assert_eq!(v(&mx), vec![9.0, 6.0]);
        let mxk = x.max_keepdim(D::Minus1).unwrap();
        assert_eq!(mxk.shape(), &[2, 1]);
        let mn = x.min(D::Minus1).unwrap();
        assert_eq!(v(&mn), vec![1.0, 2.0]);
        let mnk = x.min_keepdim(D::Minus1).unwrap();
        assert_eq!(mnk.shape(), &[2, 1]);
    }

    #[test]
    fn sum_all_mean_all_match_ops() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert_eq!(v(&x.sum_all().unwrap()), v(&ops::sum_all(&x).unwrap()));
        assert_eq!(v(&x.mean_all().unwrap()), v(&ops::mean_all(&x).unwrap()));
    }

    // --- cumsum --------------------------------------------------------

    #[test]
    fn cumsum_matches_ops() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(
            v(&x.cumsum(D::Minus1).unwrap()),
            v(&ops::cumsum(&x, 1).unwrap())
        );
        assert_eq!(v(&x.cumsum(1usize).unwrap()), vec![1.0, 3.0, 6.0, 4.0, 9.0, 15.0]);
    }

    // --- index_select --------------------------------------------------

    #[test]
    fn index_select_reorders_args() {
        // pick rows 0 and 0 along axis 0 of a [3,2] tensor.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let idx = Tensor::from_slice(&[2u32, 0u32], vec![2]).unwrap();
        let got = x.index_select(&idx, 0usize).unwrap();
        let want = ops::index_select(&x, 0, &idx).unwrap();
        assert_eq!(got.shape(), &[2, 2]);
        assert_eq!(v(&got), v(&want));
        assert_eq!(v(&got), vec![5.0, 6.0, 1.0, 2.0]);
    }

    // --- shape accessors ----------------------------------------------

    #[test]
    fn dims_dim_elem_count_aliases() {
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(x.dims(), &[2, 3]);
        assert_eq!(x.dims(), x.shape());
        assert_eq!(x.dim(0usize).unwrap(), 2);
        assert_eq!(x.dim(D::Minus1).unwrap(), 3);
        assert_eq!(x.elem_count(), 6);
        assert_eq!(x.elem_count(), x.element_count());
    }

    #[test]
    fn dims2_dims3_dims4_tuples() {
        let r2 = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        assert_eq!(r2.dims2().unwrap(), (2, 3));
        assert!(r2.dims3().is_err());
        let r3 = t(&[0.0; 24], &[2, 3, 4]);
        assert_eq!(r3.dims3().unwrap(), (2, 3, 4));
        assert!(r3.dims2().is_err());
        let r4 = t(&[0.0; 24], &[1, 2, 3, 4]);
        assert_eq!(r4.dims4().unwrap(), (1, 2, 3, 4));
        assert!(r4.dims3().is_err());
    }

    #[test]
    fn flatten_all_alias() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let f = x.flatten_all().unwrap();
        assert_eq!(f.shape(), &[4]);
        assert_eq!(v(&f), vec![1.0, 2.0, 3.0, 4.0]);
    }

    // --- readback ------------------------------------------------------

    #[test]
    fn to_vec1_rank1_only() {
        let x = t(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(x.to_vec1::<f32>().unwrap(), vec![1.0, 2.0, 3.0]);
        let r2 = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(r2.to_vec1::<f32>().is_err());
    }

    #[test]
    fn to_vec2_nests_row_major() {
        // [2,3] -> two rows of three, row-major.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let nested = x.to_vec2::<f32>().unwrap();
        assert_eq!(nested, vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]);
    }

    #[test]
    fn to_vec2_round_trips_from_known_array() {
        // Build via from_vec + reshape, read back, assert equal.
        let known = vec![vec![10.0f32, 20.0], vec![30.0, 40.0], vec![50.0, 60.0]];
        let flat: Vec<f32> = known.iter().flatten().copied().collect();
        let x = Tensor::from_vec(flat, vec![3, 2]).unwrap();
        assert_eq!(x.to_vec2::<f32>().unwrap(), known);
    }

    #[test]
    fn to_vec2_rank_mismatch_errors() {
        let r1 = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(r1.to_vec2::<f32>().is_err());
        let r3 = t(&[0.0; 8], &[2, 2, 2]);
        assert!(r3.to_vec2::<f32>().is_err());
    }

    #[test]
    fn to_vec3_nests_row_major() {
        // [2,2,2] -> 2 planes, each 2 rows of 2, row-major.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 2, 2]);
        let nested = x.to_vec3::<f32>().unwrap();
        assert_eq!(
            nested,
            vec![
                vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                vec![vec![5.0, 6.0], vec![7.0, 8.0]],
            ]
        );
    }

    #[test]
    fn to_vec3_round_trips_non_cubic() {
        // [1,2,3] exercises distinct d0/d1/d2 strides.
        let known = vec![vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ]];
        let flat: Vec<f32> = known
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let x = Tensor::from_vec(flat, vec![1, 2, 3]).unwrap();
        assert_eq!(x.to_vec3::<f32>().unwrap(), known);
    }

    #[test]
    fn to_vec3_rank_mismatch_errors() {
        let r2 = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(r2.to_vec3::<f32>().is_err());
        let r1 = t(&[1.0, 2.0], &[2]);
        assert!(r1.to_vec3::<f32>().is_err());
    }

    // --- is_finite / is_infinite masks --------------------------------

    fn read_u8(x: &Tensor) -> Vec<u8> {
        x.to_vec::<u8>().unwrap()
    }

    #[test]
    fn is_finite_mask_exact() {
        // [1.0, 2.0, +inf, -inf, NaN] -> finite mask 1,1,0,0,0.
        let x = t(
            &[1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            &[5],
        );
        let mask = x.is_finite().unwrap();
        assert_eq!(mask.dtype(), DType::U8);
        assert_eq!(mask.shape(), &[5]);
        assert_eq!(read_u8(&mask), vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn is_infinite_mask_exact() {
        // [1.0, 2.0, +inf, -inf, NaN] -> infinite mask 0,0,1,1,0.
        let x = t(
            &[1.0, 2.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN],
            &[5],
        );
        let mask = x.is_infinite().unwrap();
        assert_eq!(mask.dtype(), DType::U8);
        assert_eq!(mask.shape(), &[5]);
        assert_eq!(read_u8(&mask), vec![0, 0, 1, 1, 0]);
    }

    #[test]
    fn is_finite_and_is_infinite_are_complementary_off_nan() {
        // For non-NaN inputs, finite and infinite masks partition: their
        // sum is 1 everywhere. (NaN is 0 in both, so it's excluded here.)
        let x = t(&[0.0, -7.5, f32::INFINITY, f32::NEG_INFINITY], &[4]);
        let fin = read_u8(&x.is_finite().unwrap());
        let inf = read_u8(&x.is_infinite().unwrap());
        for (f, i) in fin.iter().zip(inf.iter()) {
            assert_eq!(f + i, 1, "finite={f} infinite={i} should partition");
        }
    }

    #[test]
    fn is_finite_preserves_multidim_shape() {
        let x = t(&[1.0, f32::INFINITY, f32::NAN, -3.0], &[2, 2]);
        let mask = x.is_finite().unwrap();
        assert_eq!(mask.shape(), &[2, 2]);
        assert_eq!(read_u8(&mask), vec![1, 0, 0, 1]);
    }

    // --- constructors --------------------------------------------------

    #[test]
    fn zeros_ones_ctors() {
        let z = Tensor::zeros(vec![2, 2], DType::F32, Device::Cpu).unwrap();
        assert_eq!(z.shape(), &[2, 2]);
        assert_eq!(v(&z), vec![0.0, 0.0, 0.0, 0.0]);
        let o = Tensor::ones(vec![2, 2], DType::F32, Device::Cpu).unwrap();
        assert_eq!(v(&o), vec![1.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn zeros_like_ones_like_match_ops() {
        let x = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        assert_eq!(v(&x.zeros_like().unwrap()), v(&ops::zeros_like(&x).unwrap()));
        assert_eq!(v(&x.ones_like().unwrap()), v(&ops::ones_like(&x).unwrap()));
    }

    #[test]
    fn cat_ctor_matches_ops_concat() {
        let a = t(&[1.0, 2.0], &[1, 2]);
        let b = t(&[3.0, 4.0], &[1, 2]);
        let got = Tensor::cat(&[&a, &b], 0).unwrap();
        let want = ops::concat(&[&a, &b], 0).unwrap();
        assert_eq!(got.shape(), &[2, 2]);
        assert_eq!(v(&got), v(&want));
        assert_eq!(v(&got), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn arange_ctor_step_one() {
        let a = Tensor::arange(0.0, 4.0, Device::Cpu).unwrap();
        assert_eq!(a.shape(), &[4]);
        assert_eq!(v(&a), vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn new_ctor_rank1() {
        let x = Tensor::new(&[1.0f32, 2.0, 3.0], Device::Cpu).unwrap();
        assert_eq!(x.shape(), &[3]);
        assert_eq!(v(&x), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn randn_is_deterministic_and_shaped() {
        let a = Tensor::randn(0.0, 1.0, vec![2, 3], Device::Cpu).unwrap();
        let b = Tensor::randn(0.0, 1.0, vec![2, 3], Device::Cpu).unwrap();
        assert_eq!(a.shape(), &[2, 3]);
        // Fixed seed => reproducible draws.
        assert_eq!(v(&a), v(&b));
    }

    // --- argmax (#1082 flip gaps) --------------------------------------

    #[test]
    fn argmax_last_dim_matches_ops_and_squeezes() {
        // [2,3]; per-row argmax over the last axis -> [2].
        let x = t(&[1.0, 9.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
        let got = x.argmax(D::Minus1).unwrap();
        let want = ops::argmax_last_dim(&x).unwrap();
        assert_eq!(got.dtype(), DType::I64);
        assert_eq!(got.shape(), &[2]); // last axis dropped
        assert_eq!(got.to_vec::<i64>().unwrap(), want.to_vec::<i64>().unwrap());
        // row0 max is 9.0 @ idx 1; row1 max is 6.0 @ idx 2.
        assert_eq!(got.to_vec::<i64>().unwrap(), vec![1, 2]);
        // usize axis form for the last axis takes the same fast path.
        let got_usize = x.argmax(1usize).unwrap();
        assert_eq!(got_usize.to_vec::<i64>().unwrap(), vec![1, 2]);
    }

    #[test]
    fn argmax_rank1_dim0_is_last_axis() {
        // The dominant forward.rs call shape: argmax(0) on a 1-D tensor.
        let logits = t(&[0.1, 0.3, 9.9, 0.2], &[4]);
        let got = logits.argmax(0usize).unwrap();
        assert_eq!(got.shape(), &[] as &[usize]); // scalar (rank-0)
        assert_eq!(got.to_vec::<i64>().unwrap(), vec![2]);
    }

    #[test]
    fn argmax_non_last_axis_drops_that_axis_in_order() {
        // [2,3]; argmax over axis 0 -> [3], one winner per column.
        // col0: max(1,4)=4 @ row1; col1: max(9,2)=9 @ row0; col2: max(3,6)=6 @ row1.
        let x = t(&[1.0, 9.0, 3.0, 4.0, 2.0, 6.0], &[2, 3]);
        let got = x.argmax(0usize).unwrap();
        assert_eq!(got.shape(), &[3]); // axis 0 removed, axis 1 preserved
        assert_eq!(got.to_vec::<i64>().unwrap(), vec![1, 0, 1]);
    }

    // --- index_add (#1082 flip gaps) -----------------------------------

    #[test]
    fn index_add_adds_source_rows_into_self() {
        // self: [3,2] zeros; add source rows into rows 0 and 2.
        // candle arg order: index_add(indexes, source, dim).
        let base = Tensor::zeros(vec![3, 2], DType::F32, Device::Cpu).unwrap();
        let indexes = Tensor::from_slice(&[0i64, 2i64], vec![2]).unwrap();
        let source = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]); // 2 rows of 2
        let got = base.index_add(&indexes, &source, 0usize).unwrap();
        assert_eq!(got.shape(), &[3, 2]);
        // row0 gets [10,20], row1 untouched [0,0], row2 gets [30,40].
        assert_eq!(v(&got), vec![10.0, 20.0, 0.0, 0.0, 30.0, 40.0]);
    }

    #[test]
    fn index_add_folds_onto_existing_self_values() {
        // Non-zero base verifies the `self + scatter_add(...)` composition.
        let base = t(&[1.0, 1.0, 1.0, 1.0, 1.0, 1.0], &[3, 2]);
        let indexes = Tensor::from_slice(&[1i64], vec![1]).unwrap();
        let source = t(&[100.0, 200.0], &[1, 2]);
        let got = base.index_add(&indexes, &source, 0usize).unwrap();
        // only row1 changes: [1,1] + [100,200] = [101,201].
        assert_eq!(v(&got), vec![1.0, 1.0, 101.0, 201.0, 1.0, 1.0]);
        // Cross-check against the underlying scatter_add composition.
        let scattered = ops::scatter_add(&source, 0, &indexes, 3).unwrap();
        let want = ops::add(&base, &scattered).unwrap();
        assert_eq!(v(&got), v(&want));
    }

    // --- track_op (#1082 flip gaps) ------------------------------------

    #[test]
    fn track_op_is_always_false() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(!x.track_op());
        // Derived tensors are equally untracked (no in-tensor graph).
        let y = x.add(&x).unwrap();
        assert!(!y.track_op());
    }

    // --- copy (#1082 flip gaps) ----------------------------------------

    #[test]
    fn copy_produces_independent_storage() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let c = x.copy().unwrap();
        // Same values + shape + dtype.
        assert_eq!(c.shape(), x.shape());
        assert_eq!(c.dtype(), x.dtype());
        assert_eq!(v(&c), v(&x));
        // Fresh storage allocation (not aliasing the source Arc), unlike
        // a plain Clone / contiguous() fast path.
        assert!(
            !std::sync::Arc::ptr_eq(c.storage(), x.storage()),
            "copy must allocate fresh storage, not share the source Arc"
        );
    }

    #[test]
    fn copy_of_noncontiguous_materializes_logical_order() {
        // Transpose makes a non-contiguous view; copy must read it in
        // logical (row-major) order, like candle's copy.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let xt = x.transpose(0, 1).unwrap(); // [3,2], non-contiguous
        let c = xt.copy().unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        // logical order of the transpose: columns of x.
        assert_eq!(v(&c), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert!(!std::sync::Arc::ptr_eq(c.storage(), xt.storage()));
    }

    // --- detach (#1082 flip gaps) --------------------------------------

    #[test]
    fn detach_is_identity_returns_tensor() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        // candle returns `Tensor` by value (not Result); kt mirrors that.
        let d: Tensor = x.detach();
        assert_eq!(v(&d), v(&x));
        assert_eq!(d.shape(), x.shape());
        // kt detach is a shared-storage identity (no autograd to sever).
        assert!(std::sync::Arc::ptr_eq(d.storage(), x.storage()));
    }

    // --- empty (#1082 flip gaps) ---------------------------------------

    #[test]
    fn empty_has_requested_shape_and_dtype() {
        // SAFETY: kt's `empty` zero-fills (safe superset of candle's
        // uninitialized memory); the `unsafe` is API-compat only.
        let e = unsafe { Tensor::empty(vec![2, 3], DType::F32, Device::Cpu) }.unwrap();
        assert_eq!(e.shape(), &[2, 3]);
        assert_eq!(e.dtype(), DType::F32);
        // kt zero-fills rather than leaving garbage.
        assert_eq!(v(&e), vec![0.0; 6]);
        // BF16 dtype also honored.
        let eb = unsafe { Tensor::empty(vec![4], DType::BF16, Device::Cpu) }.unwrap();
        assert_eq!(eb.shape(), &[4]);
        assert_eq!(eb.dtype(), DType::BF16);
    }
}
