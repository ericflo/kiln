//! `std::ops::{Add,Sub,Mul,Div,Neg}` trait impls for [`Tensor`] —
//! the candle-API operator-overload surface the `forward.rs`
//! candle→kt type-flip relies on (issue #1082).
//!
//! # Why this file exists
//!
//! candle implements the arithmetic operator traits for `Tensor`,
//! `&Tensor`, scalar (`f64`) RHS, and even `Result<Tensor>` RHS, all
//! with **`type Output = Result<Tensor>`**. `forward.rs` writes
//! `(&a + &b)?`, `x * scale`, `&logits - max`, etc., dozens of times.
//! A mechanical `s/candle_core::Tensor/kiln_tensor::Tensor/` only
//! type-checks if `kt::Tensor` answers the same operator overloads.
//! The error-inventory probe (flipping the `Tensor` alias and
//! building) attributed ~78 of the 1,450 flip errors to these missing
//! `impl std::ops::*` blocks — the single biggest mechanical bucket.
//!
//! # Faithful mirror of candle's `bin_trait!`
//!
//! candle-core's upstream `tensor.rs` (captured in the pre-#1082 vendor
//! tree) defines a `bin_trait!` macro that
//! emits, for each of `Add/Sub/Mul/Div`:
//!
//! - `Tensor      ⊕ B: Borrow<Tensor>` → `Result<Tensor>`
//! - `&Tensor     ⊕ B: Borrow<Tensor>` → `Result<Tensor>`
//! - `Result<B>   ⊕ Tensor`            → `Result<Tensor>`
//! - `Result<B>   ⊕ &Tensor`           → `Result<Tensor>`
//! - `Tensor      ⊕ Result<B>`         → `Result<Tensor>`
//! - `&Tensor     ⊕ Result<B>`         → `Result<Tensor>`
//! - `Tensor      ⊕ f64`               → `Result<Tensor>`
//! - `&Tensor     ⊕ f64`               → `Result<Tensor>`
//!
//! plus standalone `f64 ⊕ Tensor` / `f64 ⊕ &Tensor` impls. This file
//! reproduces that whole matrix.
//!
//! The `B: Borrow<Tensor>` bound is candle's trick that collapses the
//! `Tensor ⊕ Tensor`, `Tensor ⊕ &Tensor`, `&Tensor ⊕ Tensor`,
//! `&Tensor ⊕ &Tensor` quartet into two impl blocks (the LHS is the
//! `self` receiver; the RHS is any `Borrow<Tensor>`). We keep that
//! exact shape so the same RHS forms callers use compile here too.
//!
//! # Delegation
//!
//! - Tensor⊕Tensor → the existing free fns
//!   [`ops::add`] / [`ops::sub`] / [`ops::mul`] / [`ops::div`]
//!   (via the inherent [`Tensor::add`] … methods in `method_api.rs`,
//!   which themselves delegate to `ops::`).
//! - Tensor⊕f64 → candle computes these as `self.affine($mul(rhs),
//!   $add(rhs))`. kt's inherent [`Tensor::affine`] already exists and
//!   itself composes [`ops::mul_scalar`] + [`ops::add_scalar`], so we
//!   delegate to it and stay numerically identical to candle:
//!     * Add: `affine(1.0, rhs)`   = `x + rhs`   (≡ `ops::add_scalar`)
//!     * Sub: `affine(1.0, -rhs)`  = `x - rhs`   (≡ `ops::sub_scalar`)
//!     * Mul: `affine(rhs, 0.0)`   = `x * rhs`   (≡ `ops::mul_scalar`)
//!     * Div: `affine(1.0/rhs, 0.0)` = `x / rhs` (≡ `ops::div_scalar`)
//!
//!   **Deviation note:** kt's scalar `ops::` (and therefore `affine`)
//!   narrow the `f64` argument to `f32` internally — kt is an f32/bf16
//!   substrate and has no f64 storage. candle's `affine` keeps the
//!   `mul`/`add` coefficients in `f64` until the per-element kernel.
//!   The narrowing is lossless for every coefficient `forward.rs`
//!   actually uses (small integer / power-of-two scales), and matches
//!   the pre-existing [`Tensor::powf`] / [`Tensor::clamp`] /
//!   [`Tensor::affine`] façade convention.
//!
//! # `Neg`
//!
//! candle does **not** implement `std::ops::Neg` for `Tensor` (it only
//! has the inherent `Tensor::neg(&self) -> Result<Self>` method, which
//! kt already mirrors in `method_api.rs`). We therefore do **not** add
//! a `std::ops::Neg` impl — doing so would *exceed* candle's surface
//! and could change overload resolution at flip sites that write
//! `x.neg()?`. See the PR report's "SKIPPED" section.
//!
//! # Purely additive
//!
//! Nothing in the tree calls these operator overloads yet (the flip is
//! a separate PR), so this file cannot regress anything. Correctness is
//! proven by the `#[cfg(test)]` block below.

use crate::{Result, Tensor, ops};
use std::borrow::Borrow;

// ----------------------------------------------------------------------
// Tensor ⊕ Tensor (and all &/Borrow combinations) + Result<B> chaining
// + scalar f64 — one macro invocation per operator, mirroring candle's
// `bin_trait!`.
// ----------------------------------------------------------------------

macro_rules! kt_bin_trait {
    // $trait : the std::ops trait (Add/Sub/Mul/Div)
    // $fn     : the trait method (add/sub/mul/div) — also the kt free fn
    // $mul    : f64 -> f64 closure giving the affine multiplier for the scalar form
    // $add    : f64 -> f64 closure giving the affine addend for the scalar form
    ($trait:ident, $fn:ident, $mul:expr, $add:expr) => {
        // #1082: kt `ops::{add,sub,mul,div}` require contiguous operands
        // ("stride-aware path not in Phase 1.15"); candle's operators handled
        // strided/narrowed views implicitly. Contiguify each operand
        // (device-correct; O(1) no-op when already contiguous) so
        // `(a * b)?`-style call sites in forward.rs work verbatim.
        impl<B: Borrow<Tensor>> std::ops::$trait<B> for Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: B) -> Self::Output {
                ops::$fn(&self.contiguous()?, &rhs.borrow().contiguous()?)
            }
        }

        impl<B: Borrow<Tensor>> std::ops::$trait<B> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: B) -> Self::Output {
                ops::$fn(&self.contiguous()?, &rhs.borrow().contiguous()?)
            }
        }

        impl<B: Borrow<Tensor>> std::ops::$trait<Tensor> for Result<B> {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: Tensor) -> Self::Output {
                ops::$fn(&self?.borrow().contiguous()?, &rhs.contiguous()?)
            }
        }

        impl<B: Borrow<Tensor>> std::ops::$trait<&Tensor> for Result<B> {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: &Tensor) -> Self::Output {
                ops::$fn(&self?.borrow().contiguous()?, &rhs.contiguous()?)
            }
        }

        impl<B: Borrow<Tensor>> std::ops::$trait<Result<B>> for Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: Result<B>) -> Self::Output {
                ops::$fn(&self.contiguous()?, &rhs?.borrow().contiguous()?)
            }
        }

        impl<B: Borrow<Tensor>> std::ops::$trait<Result<B>> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: Result<B>) -> Self::Output {
                ops::$fn(&self.contiguous()?, &rhs?.borrow().contiguous()?)
            }
        }

        impl std::ops::$trait<f64> for Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: f64) -> Self::Output {
                // #1082: contiguify for the strict scalar/affine op (no-op when
                // already contiguous).
                self.contiguous()?.affine($mul(rhs), $add(rhs))
            }
        }

        impl std::ops::$trait<f64> for &Tensor {
            type Output = Result<Tensor>;

            fn $fn(self, rhs: f64) -> Self::Output {
                self.contiguous()?.affine($mul(rhs), $add(rhs))
            }
        }
    };
}

// Same affine coefficients candle uses (`bin_trait!(...)` call sites):
//   Add: x*1 + rhs           Sub: x*1 + (-rhs)
//   Mul: x*rhs + 0           Div: x*(1/rhs) + 0
kt_bin_trait!(Add, add, |_| 1., |v| v);
kt_bin_trait!(Sub, sub, |_| 1., |v: f64| -v);
kt_bin_trait!(Mul, mul, |v| v, |_| 0.);
kt_bin_trait!(Div, div, |v| 1. / v, |_| 0.);

// ----------------------------------------------------------------------
// f64 ⊕ Tensor — candle's standalone impls (commutative re-dispatch for
// Add/Mul; explicit affine for the non-commutative Sub/Div).
// ----------------------------------------------------------------------

impl std::ops::Add<Tensor> for f64 {
    type Output = Result<Tensor>;

    fn add(self, rhs: Tensor) -> Self::Output {
        rhs + self
    }
}

impl std::ops::Add<&Tensor> for f64 {
    type Output = Result<Tensor>;

    fn add(self, rhs: &Tensor) -> Self::Output {
        rhs + self
    }
}

impl std::ops::Mul<Tensor> for f64 {
    type Output = Result<Tensor>;

    fn mul(self, rhs: Tensor) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Mul<&Tensor> for f64 {
    type Output = Result<Tensor>;

    fn mul(self, rhs: &Tensor) -> Self::Output {
        rhs * self
    }
}

impl std::ops::Sub<Tensor> for f64 {
    type Output = Result<Tensor>;

    // candle: `rhs.affine(-1., self)` => `self - rhs` elementwise.
    fn sub(self, rhs: Tensor) -> Self::Output {
        rhs.affine(-1., self)
    }
}

impl std::ops::Sub<&Tensor> for f64 {
    type Output = Result<Tensor>;

    fn sub(self, rhs: &Tensor) -> Self::Output {
        rhs.affine(-1., self)
    }
}

impl std::ops::Div<Tensor> for f64 {
    type Output = Result<Tensor>;

    // candle: `rhs.recip()? * self` => `self / rhs` elementwise.
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: Tensor) -> Self::Output {
        rhs.recip()? * self
    }
}

impl std::ops::Div<&Tensor> for f64 {
    type Output = Result<Tensor>;

    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &Tensor) -> Self::Output {
        rhs.recip()? * self
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;
    use crate::ops;

    fn t(data: &[f32], shape: &[usize]) -> Tensor {
        Tensor::from_slice(data, shape.to_vec()).unwrap()
    }

    fn v(x: &Tensor) -> Vec<f32> {
        x.to_vec::<f32>().unwrap()
    }

    // --- Tensor ⊕ Tensor across all owned/borrowed combinations -------

    #[test]
    fn add_tensor_tensor_all_ref_combos() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let want = v(&ops::add(&a, &b).unwrap());

        // owned ⊕ owned
        assert_eq!(v(&(a.clone() + b.clone()).unwrap()), want);
        // owned ⊕ &
        assert_eq!(v(&(a.clone() + &b).unwrap()), want);
        // & ⊕ owned
        assert_eq!(v(&(&a + b.clone()).unwrap()), want);
        // & ⊕ &
        assert_eq!(v(&(&a + &b).unwrap()), want);
    }

    #[test]
    fn sub_mul_div_tensor_tensor_match_ops() {
        let a = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let b = t(&[2.0, 4.0, 5.0, 8.0], &[2, 2]);
        assert_eq!(v(&(&a - &b).unwrap()), v(&ops::sub(&a, &b).unwrap()));
        assert_eq!(v(&(&a * &b).unwrap()), v(&ops::mul(&a, &b).unwrap()));
        assert_eq!(v(&(&a / &b).unwrap()), v(&ops::div(&a, &b).unwrap()));
    }

    // --- Result<B> chaining: (a + b) + c without intermediate `?` ------

    #[test]
    fn result_chaining_left_and_right() {
        let a = t(&[1.0, 1.0], &[2]);
        let b = t(&[2.0, 2.0], &[2]);
        let c = t(&[3.0, 3.0], &[2]);

        // Result<Tensor> ⊕ &Tensor  (LHS is the Result)
        let lhs_result = (&a + &b) + &c; // ((a+b)?) + c
        assert_eq!(v(&lhs_result.unwrap()), vec![6.0, 6.0]);

        // Tensor ⊕ Result<Tensor>  (RHS is the Result)
        let rhs_result = a.clone() + (&b + &c); // a + ((b+c)?)
        assert_eq!(v(&rhs_result.unwrap()), vec![6.0, 6.0]);

        // &Tensor ⊕ Result<Tensor>
        let ref_rhs_result = &a + (&b + &c);
        assert_eq!(v(&ref_rhs_result.unwrap()), vec![6.0, 6.0]);

        // Result<Tensor> ⊕ Tensor (owned RHS)
        let lhs_result_owned = (&a + &b) + c.clone();
        assert_eq!(v(&lhs_result_owned.unwrap()), vec![6.0, 6.0]);
    }

    // --- Tensor ⊕ f64 (and &Tensor ⊕ f64), all four operators ---------

    #[test]
    fn scalar_rhs_matches_scalar_ops() {
        let a = t(&[2.0, 4.0, 6.0, 8.0], &[2, 2]);

        // Add: x + c  ≡ ops::add_scalar
        assert_eq!(
            v(&(a.clone() + 3.0f64).unwrap()),
            v(&ops::add_scalar(&a, 3.0).unwrap())
        );
        assert_eq!(
            v(&(&a + 3.0f64).unwrap()),
            v(&ops::add_scalar(&a, 3.0).unwrap())
        );

        // Sub: x - c  ≡ ops::sub_scalar
        assert_eq!(
            v(&(&a - 1.5f64).unwrap()),
            v(&ops::sub_scalar(&a, 1.5).unwrap())
        );

        // Mul: x * c  ≡ ops::mul_scalar
        assert_eq!(
            v(&(&a * 2.0f64).unwrap()),
            v(&ops::mul_scalar(&a, 2.0).unwrap())
        );

        // Div: x / c  ≡ ops::div_scalar
        assert_eq!(
            v(&(&a / 2.0f64).unwrap()),
            v(&ops::div_scalar(&a, 2.0).unwrap())
        );
    }

    // --- f64 ⊕ Tensor -------------------------------------------------

    #[test]
    fn scalar_lhs_add_mul_commute() {
        let a = t(&[2.0, 4.0], &[2]);
        // 3 + a == a + 3
        assert_eq!(v(&(3.0f64 + a.clone()).unwrap()), vec![5.0, 7.0]);
        assert_eq!(v(&(3.0f64 + &a).unwrap()), vec![5.0, 7.0]);
        // 2 * a == a * 2
        assert_eq!(v(&(2.0f64 * a.clone()).unwrap()), vec![4.0, 8.0]);
        assert_eq!(v(&(2.0f64 * &a).unwrap()), vec![4.0, 8.0]);
    }

    #[test]
    fn scalar_lhs_sub_div_are_non_commutative() {
        let a = t(&[2.0, 4.0], &[2]);
        // 10 - a == [8, 6]  (NOT a - 10)
        assert_eq!(v(&(10.0f64 - a.clone()).unwrap()), vec![8.0, 6.0]);
        assert_eq!(v(&(10.0f64 - &a).unwrap()), vec![8.0, 6.0]);
        // 8 / a == [4, 2]   (NOT a / 8)
        assert_eq!(v(&(8.0f64 / a.clone()).unwrap()), vec![4.0, 2.0]);
        assert_eq!(v(&(8.0f64 / &a).unwrap()), vec![4.0, 2.0]);
    }
}
