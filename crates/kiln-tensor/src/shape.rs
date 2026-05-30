//! `kiln_tensor::Shape` — the candle-compatible shape-argument façade.
//!
//! # Why this exists (#1082 forward.rs flip)
//!
//! candle's tensor constructors and view ops are generic over
//! `S: Into<Shape>`, where `candle_core::Shape` provides `From` impls for
//! tuples (`(usize, usize)`, `(usize, usize, usize)`, …), the unit shape
//! `()`, a bare `usize` scalar, fixed arrays `[usize; N]`, slices
//! `&[usize]`, and `Vec<usize>`. `forward.rs` (and the other kiln-model /
//! kiln-train files being flipped off candle) call
//! `Tensor::zeros((d0, d1), dtype, &dev)`, `x.reshape((b, t, h, d))`,
//! `Tensor::from_slice(data, (batch,), &dev)`, etc. with those exact
//! shape literals.
//!
//! kiln-tensor's own constructors were originally spelled
//! `shape: impl Into<Vec<usize>>`, which accepts `Vec<usize>`,
//! `[usize; N]`, and `&[usize]` but **not** tuples, the unit `()`, or a
//! bare `usize`. This newtype closes that gap so the candle-style call
//! shapes type-check unchanged.
//!
//! # Design
//!
//! `Shape` is a thin newtype over `Vec<usize>`. Every form candle accepts
//! is given a `From<…> for Shape` impl, and `Shape` converts back to
//! `Vec<usize>` (`impl From<Shape> for Vec<usize>`) so the constructor
//! bodies keep their existing `let shape: Vec<usize> = …;` shape with a
//! single extra `.into()` hop. The conversion is allocation-cheap (a
//! `Vec` move or a small fixed-size collect).
//!
//! # Additivity
//!
//! All existing kiln-tensor callers pass `Vec<usize>`, `[usize; N]`,
//! `&[usize]`, or `&Vec<usize>` — every one of which has a `From` impl
//! here, so switching the public constructor/view signatures from
//! `impl Into<Vec<usize>>` to `impl Into<Shape>` is behavior-preserving
//! for them. This is the same dim-set candle's `Into<Shape>` covers.

/// A tensor shape: an ordered list of axis lengths.
///
/// Mirrors `candle_core::Shape` for the purpose of the #1082 façade. The
/// public surface is intentionally minimal — its job is to be the target
/// of `Into` for every shape literal candle accepts, then hand back a
/// `Vec<usize>` to the existing constructor bodies.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Shape(Vec<usize>);

impl Shape {
    /// Borrow the axis lengths.
    pub fn dims(&self) -> &[usize] {
        &self.0
    }

    /// Consume into the owned axis-length vector.
    pub fn into_dims(self) -> Vec<usize> {
        self.0
    }

    /// Number of axes (the rank).
    pub fn rank(&self) -> usize {
        self.0.len()
    }

    /// Total element count (product of axis lengths; `1` for a scalar).
    pub fn elem_count(&self) -> usize {
        self.0.iter().product()
    }
}

// --- back-conversion: Shape -> Vec<usize> --------------------------------
// Lets constructor bodies do `let shape: Vec<usize> = shape.into().into();`
// (or via the `impl Into<Shape>` param + `.into_dims()`).

impl From<Shape> for Vec<usize> {
    fn from(s: Shape) -> Self {
        s.0
    }
}

// --- forward conversions: every candle-accepted shape form -> Shape ------

impl From<Vec<usize>> for Shape {
    fn from(v: Vec<usize>) -> Self {
        Shape(v)
    }
}

impl From<&Vec<usize>> for Shape {
    fn from(v: &Vec<usize>) -> Self {
        Shape(v.clone())
    }
}

impl From<&[usize]> for Shape {
    fn from(v: &[usize]) -> Self {
        Shape(v.to_vec())
    }
}

impl<const N: usize> From<[usize; N]> for Shape {
    fn from(v: [usize; N]) -> Self {
        Shape(v.to_vec())
    }
}

impl<const N: usize> From<&[usize; N]> for Shape {
    fn from(v: &[usize; N]) -> Self {
        Shape(v.to_vec())
    }
}

/// The unit / scalar shape: `Tensor::zeros((), dtype, &dev)`.
impl From<()> for Shape {
    fn from(_: ()) -> Self {
        Shape(vec![])
    }
}

/// A bare axis length: `Tensor::zeros(head_dim, dtype, dev)`.
impl From<usize> for Shape {
    fn from(d: usize) -> Self {
        Shape(vec![d])
    }
}

impl From<(usize,)> for Shape {
    fn from(d: (usize,)) -> Self {
        Shape(vec![d.0])
    }
}

impl From<(usize, usize)> for Shape {
    fn from(d: (usize, usize)) -> Self {
        Shape(vec![d.0, d.1])
    }
}

impl From<(usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize)) -> Self {
        Shape(vec![d.0, d.1, d.2])
    }
}

impl From<(usize, usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize, usize)) -> Self {
        Shape(vec![d.0, d.1, d.2, d.3])
    }
}

impl From<(usize, usize, usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize, usize, usize)) -> Self {
        Shape(vec![d.0, d.1, d.2, d.3, d.4])
    }
}

impl From<(usize, usize, usize, usize, usize, usize)> for Shape {
    fn from(d: (usize, usize, usize, usize, usize, usize)) -> Self {
        Shape(vec![d.0, d.1, d.2, d.3, d.4, d.5])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_unit_is_rank0_scalar() {
        let s: Shape = ().into();
        assert_eq!(s.dims(), &[] as &[usize]);
        assert_eq!(s.rank(), 0);
        assert_eq!(s.elem_count(), 1);
    }

    #[test]
    fn from_scalar_usize() {
        let s: Shape = 7usize.into();
        assert_eq!(s.dims(), &[7]);
    }

    #[test]
    fn from_tuples() {
        assert_eq!(Shape::from((3,)).dims(), &[3]);
        assert_eq!(Shape::from((3, 4)).dims(), &[3, 4]);
        assert_eq!(Shape::from((3, 4, 5)).dims(), &[3, 4, 5]);
        assert_eq!(Shape::from((1, 2, 3, 4)).dims(), &[1, 2, 3, 4]);
        assert_eq!(Shape::from((1, 2, 3, 4, 5)).dims(), &[1, 2, 3, 4, 5]);
        assert_eq!(Shape::from((1, 2, 3, 4, 5, 6)).dims(), &[1, 2, 3, 4, 5, 6]);
    }

    #[test]
    fn from_vec_array_slice_roundtrip() {
        assert_eq!(Shape::from(vec![2usize, 3]).dims(), &[2, 3]);
        assert_eq!(Shape::from([2usize, 3]).dims(), &[2, 3]);
        let v = vec![4usize, 5];
        assert_eq!(Shape::from(v.as_slice()).dims(), &[4, 5]);
        assert_eq!(Shape::from(&v).dims(), &[4, 5]);
        let back: Vec<usize> = Shape::from((6usize, 7)).into();
        assert_eq!(back, vec![6, 7]);
    }
}
