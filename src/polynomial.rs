//! Polynomial arithmetic over a [`GaloisField`], ported from
//! `src/reed-solomon/polynomial.c` in libcorrect's `short_rs` branch.
//!
//! A polynomial is represented as a plain coefficient slice, `&[u8]`,
//! ordered from lowest to highest degree (`coeff[i]` is the coefficient
//! of `x^i`); its order is simply `coeff.len() - 1`. This is a
//! deliberate simplification of the C `polynomial_t`, which pairs a
//! `coeff` pointer with a separate, sometimes-smaller-than-the-buffer
//! `order` field: that separation exists in C so decode-time scratch
//! buffers can be reused across calls without reallocating even as their
//! logical length changes step to step. Here, that reuse is expressed
//! directly with slicing (`&buf[..=order]`) at the call site instead of
//! being threaded through every function signature, which keeps this
//! module's functions plain and easy to test in isolation. The RS
//! decoder (a later phase) is where that scratch-buffer reuse will
//! actually happen.
//!
//! Output parameters (`res`, `der`, `poly`) are passed as `&mut [u8]`
//! rather than returned, again mirroring the C functions: the caller
//! picks the buffer size (which, for [`mul`] in particular, doubles as
//! the truncation length -- see its docs).

use alloc::vec;
use alloc::vec::Vec;

use crate::field::GaloisField;

/// Multiplies polynomials `l` and `r`, writing the result into `res`.
///
/// `res` need not be long enough to hold the full product
/// (`l.len() + r.len() - 1` coefficients): any coefficients of the
/// product at or above `res.len()` are simply not computed, which is the
/// same as computing the product modulo `x^res.len()`. Truncating this
/// way (rather than computing the full product and discarding the top)
/// is how the RS decoder computes things like the error evaluator
/// polynomial, which is only ever needed modulo `x^(2t)`.
pub fn mul(field: &GaloisField, l: &[u8], r: &[u8], res: &mut [u8]) {
    res.fill(0);
    for (i, &li) in l.iter().enumerate() {
        if i >= res.len() {
            // i only grows from here, so no later i can write into res
            // either.
            break;
        }
        if li == 0 {
            continue;
        }
        for (j, &rj) in r.iter().enumerate() {
            if i + j >= res.len() {
                break;
            }
            res[i + j] = field.add(res[i + j], field.mul(li, rj));
        }
    }
}

/// Computes `dividend mod divisor` (polynomial long division, keeping
/// only the remainder), writing the result into `remainder`.
///
/// `remainder` must be at least as long as `dividend`; it is used both
/// as scratch space during the division and to hold the final result.
/// `divisor`'s leading (highest-order) coefficient must be nonzero.
///
/// Coefficients of `remainder` at or above `divisor.len() - 1` are only
/// ever produced transiently during the division and are always
/// cancelled back to zero by the time it completes -- callers only need
/// to look at `remainder[..divisor.len() - 1]` for the true remainder,
/// which is what [`crate::field`]-level callers like RS encoding do.
pub fn poly_mod(field: &GaloisField, dividend: &[u8], divisor: &[u8], remainder: &mut [u8]) {
    let dividend_order = dividend.len() - 1;
    let divisor_order = divisor.len() - 1;
    assert!(
        remainder.len() >= dividend.len(),
        "remainder buffer must be at least as long as dividend"
    );

    remainder[..dividend.len()].copy_from_slice(dividend);

    let divisor_leading_log = field.log_table()[divisor[divisor_order] as usize];

    // Walk down from the highest-order term, cancelling it at each step
    // by subtracting an appropriately-shifted and -scaled copy of the
    // divisor. Stops once the remaining order drops below the divisor's,
    // since no further reduction is possible past that point.
    let mut i = dividend_order;
    while i > 0 && i >= divisor_order {
        if remainder[i] != 0 {
            let q_order = i - divisor_order;
            let q_coeff = field.div_log(field.log_table()[remainder[i] as usize], divisor_leading_log);
            for (j, &dj) in divisor.iter().enumerate() {
                if dj != 0 {
                    let contribution = field.mul_log_element(field.log_table()[dj as usize], q_coeff);
                    remainder[j + q_order] = field.add(remainder[j + q_order], contribution);
                }
            }
        }
        i -= 1;
    }
}

/// Computes the formal derivative of `poly`, writing the result into
/// `der`. `der` must have length exactly `poly.len() - 1`.
///
/// Since this field has characteristic 2, the usual "multiply by the
/// exponent" rule for polynomial differentiation
/// (`d/dx a_n x^n = n * a_n * x^(n-1)`) collapses `n * a_n` down to a
/// parity check on `n` (see [`GaloisField::sum`]): even-degree terms
/// vanish entirely, and odd-degree terms pass their coefficient through
/// unchanged.
pub fn formal_derivative(field: &GaloisField, poly: &[u8], der: &mut [u8]) {
    assert_eq!(der.len(), poly.len() - 1, "der must have length poly.len() - 1");
    for i in 0..der.len() {
        der[i] = field.sum(poly[i + 1], (i + 1) as u32);
    }
}

/// Evaluates `poly` at `val` via Horner-free repeated multiplication:
/// accumulates `val^i` in log form as it goes, rather than recomputing
/// it from scratch for each term.
pub fn eval(field: &GaloisField, poly: &[u8], val: u8) -> u8 {
    if val == 0 {
        return poly[0];
    }

    let mut res = 0u8;
    // log(1) -- see GaloisField's docs for why this is stored as
    // largest_element rather than 0. Either value is correct here since
    // mul_log's arithmetic is mod largest_element.
    let mut val_exponentiated = field.log_table()[1];
    let val_log = field.log_table()[val as usize];

    for &c in poly {
        if c != 0 {
            res = field.add(res, field.mul_log_element(field.log_table()[c as usize], val_exponentiated));
        }
        val_exponentiated = field.mul_log(val_exponentiated, val_log);
    }
    res
}

/// Like [`eval`], but takes a precomputed table of the successive
/// logarithms of the evaluation point (`val_exp[i] == log(val^i)`, as
/// produced by [`build_exp_lut`]) instead of recomputing them. Useful
/// when evaluating multiple polynomials at the same point, e.g.
/// computing syndromes at each root of the generator polynomial.
pub fn eval_lut(field: &GaloisField, poly: &[u8], val_exp: &[u8]) -> u8 {
    // build_exp_lut's val==0 special case produces all-zero val_exp; that
    // can never happen for val != 0, since val_exp[0] would then be
    // log(1) == largest_element, which is never 0 (every field in scope
    // has at least 2 elements, so largest_element >= 1).
    if val_exp[0] == 0 {
        return poly[0];
    }

    let mut res = 0u8;
    for (i, &c) in poly.iter().enumerate() {
        if c != 0 {
            res = field.add(res, field.mul_log_element(field.log_table()[c as usize], val_exp[i]));
        }
    }
    res
}

/// Like [`eval_lut`], but `poly_log` holds the *logarithms* of the
/// polynomial's coefficients rather than the coefficients themselves
/// (with `0` used as a sentinel for "this coefficient is 0", since
/// `log(0)` doesn't otherwise exist). Used when the polynomial in
/// question -- e.g. an error locator built up during Chien search -- is
/// naturally already available in log form, to skip a table lookup per
/// coefficient.
pub fn eval_log_lut(field: &GaloisField, poly_log: &[u8], val_exp: &[u8]) -> u8 {
    if val_exp[0] == 0 {
        return if poly_log[0] == 0 {
            0
        } else {
            field.exp_table()[poly_log[0] as usize]
        };
    }

    let mut res = 0u8;
    for (i, &lc) in poly_log.iter().enumerate() {
        if lc != 0 {
            res = field.add(res, field.mul_log_element(lc, val_exp[i]));
        }
    }
    res
}

/// Fills `val_exp` with the successive logarithms of `val`:
/// `val_exp[i] == log(val^i)`, for `i` in `0..val_exp.len()`.
///
/// `val == 0` is special-cased to fill `val_exp` with all zeros, which
/// [`eval_lut`] and [`eval_log_lut`] both recognize (via `val_exp[0]`) as
/// meaning "the evaluation point is 0" and handle by returning the
/// constant term directly, since `log(0)` has no real value to give it.
pub fn build_exp_lut(field: &GaloisField, val: u8, val_exp: &mut [u8]) {
    let mut val_exponentiated = field.log_table()[1];
    let val_log = field.log_table()[val as usize];
    for slot in val_exp.iter_mut() {
        if val == 0 {
            *slot = 0;
        } else {
            *slot = val_exponentiated;
            val_exponentiated = field.mul_log(val_exponentiated, val_log);
        }
    }
}

/// Builds the monic polynomial `product((x + roots[i]) for i in roots)`
/// into `poly` (which must have length `roots.len() + 1`), using the two
/// buffers in `scratch` as ping-pong working space instead of
/// allocating.
///
/// Each buffer in `scratch` must have length at least `roots.len() + 1`;
/// their contents afterward are unspecified. This does no allocation of
/// its own, which matters for callers like the RS decoder that need to
/// rebuild a locator polynomial from roots on every decode call, on
/// hardware where allocating on that path isn't acceptable.
///
/// See [`from_roots`] for an allocating convenience wrapper.
pub fn init_from_roots(field: &GaloisField, roots: &[u8], poly: &mut [u8], scratch: &mut [Vec<u8>; 2]) {
    let nroots = roots.len();
    assert_eq!(poly.len(), nroots + 1, "poly must have length roots.len() + 1");

    if nroots == 0 {
        // The empty product is the constant polynomial 1.
        poly[0] = 1;
        return;
    }

    for buf in scratch.iter() {
        assert!(
            buf.len() > nroots,
            "each scratch buffer must have length >= roots.len() + 1"
        );
    }

    // Ping-pong between the two scratch buffers: at each step, multiply
    // the current result (order `order`) by (x + roots[i]) into the
    // other buffer (order `order + 1`), then swap which buffer is
    // "current". This lets each step reuse both buffers' storage instead
    // of growing a single accumulator.
    let [r0, r1] = scratch;
    r0[0] = roots[0];
    r0[1] = 1;
    let mut order = 1usize;
    let mut current_is_r0 = true;

    for &root in &roots[1..] {
        let l = [root, 1]; // represents (x + root)
        let next_order = order + 1;
        if current_is_r0 {
            mul(field, &l, &r0[..=order], &mut r1[..=next_order]);
        } else {
            mul(field, &l, &r1[..=order], &mut r0[..=next_order]);
        }
        current_is_r0 = !current_is_r0;
        order = next_order;
    }

    let result = if current_is_r0 { &r0[..=order] } else { &r1[..=order] };
    poly.copy_from_slice(result);
}

/// Allocating convenience wrapper around [`init_from_roots`]: builds and
/// returns the monic polynomial `product((x + roots[i]) for i in roots)`.
///
/// Used where the allocation-free discipline [`init_from_roots`] exists
/// for doesn't matter, e.g. building the RS generator polynomial once at
/// codec construction time.
pub fn from_roots(field: &GaloisField, roots: &[u8]) -> Vec<u8> {
    let mut poly = vec![0u8; roots.len() + 1];
    let mut scratch = [vec![0u8; roots.len() + 1], vec![0u8; roots.len() + 1]];
    init_from_roots(field, roots, &mut poly, &mut scratch);
    poly
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// x^4 + x + 1 -- GF(16), used by CCSDS AOS Frame Header Error
    /// Control's shortened RS(10,6) code (quiet/libcorrect#17).
    const POLY_GF16: u16 = 0x13;
    /// x^8 + x^4 + x^3 + x^2 + 1 -- GF(256), the usual RS(255,223) field.
    const POLY_GF256: u16 = 0x11d;

    fn gf16_element() -> impl Strategy<Value = u8> {
        0u8..=15u8
    }

    fn gf16_poly(max_order: usize) -> impl Strategy<Value = Vec<u8>> {
        proptest::collection::vec(gf16_element(), 1..=max_order + 1)
    }

    #[test]
    fn mul_matches_hand_computed_example() {
        // Over GF(16) (0x13): (x + 2) * (x + 4) = x^2 + (2^4)x + (2*4)
        //   = x^2 + (2 xor 4) x + 8 = x^2 + 6x + 8, using field add = xor
        //   for the linear coefficient and field mul for the constant
        //   term.
        let gf = GaloisField::new(POLY_GF16);
        let l = [2u8, 1]; // x + 2
        let r = [4u8, 1]; // x + 4
        let mut res = [0u8; 3];
        mul(&gf, &l, &r, &mut res);
        assert_eq!(res, [gf.mul(2, 4), gf.add(2, 4), 1]);
    }

    #[test]
    fn mul_truncates_to_output_length() {
        let gf = GaloisField::new(POLY_GF16);
        let l = [1u8, 1, 1]; // 1 + x + x^2
        let r = [1u8, 1]; // 1 + x
        let mut full = [0u8; 4];
        mul(&gf, &l, &r, &mut full);
        let mut truncated = [0u8; 2];
        mul(&gf, &l, &r, &mut truncated);
        assert_eq!(truncated, full[..2]);
    }

    proptest! {
        #[test]
        fn mod_by_linear_divisor_matches_eval(
            poly in gf16_poly(8),
            root in gf16_element(),
        ) {
            let gf = GaloisField::new(POLY_GF16);
            let divisor = [root, 1]; // (x + root)
            let mut remainder = vec![0u8; poly.len()];
            poly_mod(&gf, &poly, &divisor, &mut remainder);

            // Remainder theorem: poly(x) mod (x + root) == poly(root).
            prop_assert_eq!(remainder[0], eval(&gf, &poly, root));
            // Everything at or above the divisor's order must have fully
            // cancelled out.
            prop_assert!(remainder[1..].iter().all(|&c| c == 0));
        }

        #[test]
        fn eval_lut_and_log_lut_match_eval(
            poly in gf16_poly(8),
            val in gf16_element(),
        ) {
            let gf = GaloisField::new(POLY_GF16);
            let mut val_exp = vec![0u8; poly.len()];
            build_exp_lut(&gf, val, &mut val_exp);

            let expected = eval(&gf, &poly, val);
            prop_assert_eq!(eval_lut(&gf, &poly, &val_exp), expected);

            let poly_log: Vec<u8> = poly
                .iter()
                .map(|&c| if c == 0 { 0 } else { gf.log_table()[c as usize] })
                .collect();
            prop_assert_eq!(eval_log_lut(&gf, &poly_log, &val_exp), expected);
        }

        #[test]
        fn from_roots_produces_a_monic_polynomial_with_those_roots(
            roots in proptest::collection::vec(gf16_element(), 1..=6),
        ) {
            let gf = GaloisField::new(POLY_GF16);
            let poly = from_roots(&gf, &roots);

            prop_assert_eq!(poly.len(), roots.len() + 1);
            prop_assert_eq!(*poly.last().unwrap(), 1, "leading coefficient must be 1 (monic)");
            for &root in &roots {
                prop_assert_eq!(eval(&gf, &poly, root), 0, "root {} does not evaluate to 0", root);
            }
        }

        #[test]
        fn init_from_roots_matches_from_roots(
            roots in proptest::collection::vec(gf16_element(), 1..=6),
        ) {
            let gf = GaloisField::new(POLY_GF16);
            let expected = from_roots(&gf, &roots);

            let mut poly = vec![0u8; roots.len() + 1];
            let mut scratch = [vec![0u8; roots.len() + 1], vec![0u8; roots.len() + 1]];
            init_from_roots(&gf, &roots, &mut poly, &mut scratch);

            prop_assert_eq!(poly, expected);
        }
    }

    #[test]
    fn from_roots_of_empty_set_is_the_constant_one() {
        let gf = GaloisField::new(POLY_GF16);
        assert_eq!(from_roots(&gf, &[]), vec![1]);
    }

    #[test]
    fn formal_derivative_hand_computed_cases() {
        let gf = GaloisField::new(POLY_GF16);

        // f(x) = x -> f'(x) = 1
        let mut der = [0u8; 1];
        formal_derivative(&gf, &[0, 1], &mut der);
        assert_eq!(der, [1]);

        // f(x) = x^2 -> f'(x) = 2x = 0 (char 2)
        let mut der = [0u8; 2];
        formal_derivative(&gf, &[0, 0, 1], &mut der);
        assert_eq!(der, [0, 0]);

        // f(x) = 5 + 3x + 7x^2 + 2x^3 -> f'(x) = 3 + 0*x + 3*2*x^2
        //   = 3 (odd n=1 keeps a1) + 0 (even n=2 kills a2) + 2 (odd n=3
        //   keeps a3, landing at x^2)
        let mut der = [0u8; 3];
        formal_derivative(&gf, &[5, 3, 7, 2], &mut der);
        assert_eq!(der, [3, 0, 2]);
    }

    #[test]
    fn works_over_gf256_too() {
        let gf = GaloisField::new(POLY_GF256);
        let roots = [3u8, 200, 17, 1];
        let poly = from_roots(&gf, &roots);
        for &root in &roots {
            assert_eq!(eval(&gf, &poly, root), 0);
        }

        let mut val_exp = vec![0u8; poly.len()];
        build_exp_lut(&gf, 42, &mut val_exp);
        assert_eq!(eval_lut(&gf, &poly, &val_exp), eval(&gf, &poly, 42));
    }
}
