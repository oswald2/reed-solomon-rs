//! Arithmetic over `GF(2^k)` for `k <= 8`, built from a primitive
//! polynomial via log/antilog (Zech logarithm) tables.
//!
//! This is a direct port of `field_create` and the `field_*` inline
//! functions in libcorrect's `include/correct/reed-solomon/field.h`. The
//! table construction, the "wraparound" doubled-size `exp` table, and the
//! log-value aliasing at `0` (which really means "undefined" for `log(0)`,
//! and also happens to be where `log(1)` lands once you wrap past
//! `largest_element`) are all preserved bit-for-bit, since later
//! Berlekamp-Massey/Chien-search/Forney code depends on these exact
//! conventions, and the goal is wire compatibility with the C
//! implementation.

use alloc::vec;
use alloc::vec::Vec;

/// A Galois field `GF(2^k)`, `k` in `1..=8`, represented by its
/// exponentiation (`exp`, i.e. `alpha^i`) and discrete logarithm (`log`)
/// tables with respect to a primitive element `alpha`, a root of the
/// field's primitive polynomial.
///
/// Elements and logarithms are represented as `u8`, which holds for any
/// `k <= 8` since `field_size <= 256` and `largest_element <= 255`.
#[derive(Debug, Clone)]
pub struct GaloisField {
    /// `exp[i] == alpha^i`, for `i` in `0..2*field_size`. The table is
    /// twice as long as strictly necessary so that products of two
    /// logarithms (which can sum to as much as `2 * largest_element`)
    /// can be looked up directly, without a modulo/wraparound check.
    exp: Vec<u8>,
    /// `log[e] == i` such that `alpha^i == e`, for `e` in `1..field_size`.
    /// `log[0]` is meaningless (there is no such `i`) and is set to `0` as
    /// a sentinel; callers must never look up the logarithm of `0`.
    log: Vec<u8>,
    /// Number of elements in the field, `2^k`.
    field_size: u16,
    /// `field_size - 1`, both the largest field element and the order of
    /// the multiplicative group (i.e. `alpha^largest_element == 1`).
    largest_element: u8,
}

impl GaloisField {
    /// Builds the field `GF(2^k)` defined by `primitive_poly`, an
    /// irreducible polynomial over `GF(2)` of degree `k` (`k <= 8`),
    /// packed as an integer (bit `i` is the coefficient of `x^i`).
    ///
    /// For example, `0x11d` (`x^8 + x^4 + x^3 + x^2 + 1`) gives the usual
    /// `GF(256)` used by CCITT/CCSDS-style RS(255,223) codes, and `0x13`
    /// (`x^4 + x + 1`) gives the `GF(16)` used by the CCSDS AOS Frame
    /// Header Error Control shortened RS(10,6) code.
    ///
    /// The field width `k` is inferred from `primitive_poly`'s bit
    /// length, exactly as `field_create` does in the C implementation.
    pub fn new(primitive_poly: u16) -> Self {
        let mut width: u32 = 0;
        let mut temp_poly = primitive_poly >> 1;
        while temp_poly != 0 {
            temp_poly >>= 1;
            width += 1;
        }

        let field_size: u16 = 1u16 << width;
        // Fits in a u16 always, and in a u8 whenever width <= 8, which is
        // the documented constraint on this field implementation.
        let largest_element: u16 = field_size - 1;

        // exp is sized 2 * field_size so that field_mul (which adds two
        // logarithms each in [0, largest_element]) can look up results up
        // to alpha^(2*largest_element) without wraparound arithmetic.
        let mut exp = vec![0u8; 2 * field_size as usize];
        let mut log = vec![0u8; field_size as usize];

        let mut element: u16 = 1;
        exp[0] = element as u8;
        log[0] = 0; // undefined; never read for a well-formed program

        for i in 1..(2 * field_size) {
            element *= 2;
            if element > largest_element {
                element ^= primitive_poly;
            }
            exp[i as usize] = element as u8;
            if i <= largest_element {
                log[element as usize] = i as u8;
            }
        }

        GaloisField {
            exp,
            log,
            field_size,
            largest_element: largest_element as u8,
        }
    }

    /// Number of elements in the field, `2^k`.
    #[inline]
    pub fn field_size(&self) -> u16 {
        self.field_size
    }

    /// `field_size() - 1`: the largest field element, and the order of
    /// the multiplicative group.
    #[inline]
    pub fn largest_element(&self) -> u8 {
        self.largest_element
    }

    /// Direct access to the exponentiation table, `exp[i] == alpha^i`,
    /// for `i` in `0..2*field_size()`. Exposed `pub(crate)` for
    /// polynomial/RS code that needs to skip the bounds/zero checks
    /// `mul`/`div` do, the same way the C code indexes `field.exp[...]`
    /// directly in its hot paths.
    #[inline]
    pub(crate) fn exp_table(&self) -> &[u8] {
        &self.exp
    }

    /// Direct access to the logarithm table. `log[0]` is a sentinel and
    /// must not be treated as a real logarithm.
    #[inline]
    pub(crate) fn log_table(&self) -> &[u8] {
        &self.log
    }

    /// Field addition. In `GF(2^k)`, addition and subtraction are both
    /// bytewise XOR.
    #[inline]
    pub fn add(&self, l: u8, r: u8) -> u8 {
        l ^ r
    }

    /// Field subtraction. Identical to [`GaloisField::add`] in
    /// characteristic 2.
    #[inline]
    pub fn sub(&self, l: u8, r: u8) -> u8 {
        l ^ r
    }

    /// Sums `elem` with itself `n` times (e.g. as used by the formal
    /// derivative of a polynomial). Since repeated XOR of a value with
    /// itself alternates between `0` and the value, this collapses to a
    /// parity check on `n` rather than actually looping `n` times.
    #[inline]
    pub fn sum(&self, elem: u8, n: u32) -> u8 {
        if n % 2 == 1 {
            elem
        } else {
            0
        }
    }

    /// Field multiplication, via `exp[log(l) + log(r)]`.
    pub fn mul(&self, l: u8, r: u8) -> u8 {
        if l == 0 || r == 0 {
            return 0;
        }
        let res = self.log[l as usize] as u16 + self.log[r as usize] as u16;
        self.exp[res as usize]
    }

    /// Field division `l / r`, via `exp[largest_element + log(l) - log(r)]`.
    ///
    /// Division by zero is undefined in a field; matching the C
    /// implementation, this returns `0` rather than panicking.
    pub fn div(&self, l: u8, r: u8) -> u8 {
        if l == 0 || r == 0 {
            return 0;
        }
        let res = self.largest_element as u16 + self.log[l as usize] as u16
            - self.log[r as usize] as u16;
        self.exp[res as usize]
    }

    /// Multiplies two *logarithms* (not field elements) as `field_mul_log`
    /// does: adds them and wraps the result back into `[0, largest_element]`.
    /// Useful when a caller already has values in log form and wants to
    /// avoid the round-trip through `exp`/`log`.
    #[inline]
    pub fn mul_log(&self, l: u8, r: u8) -> u8 {
        let res = l as u16 + r as u16;
        if res > self.largest_element as u16 {
            (res - self.largest_element as u16) as u8
        } else {
            res as u8
        }
    }

    /// Divides two logarithms, the log-domain analogue of
    /// [`GaloisField::div`].
    #[inline]
    pub fn div_log(&self, l: u8, r: u8) -> u8 {
        let res = self.largest_element as u16 + l as u16 - r as u16;
        if res > self.largest_element as u16 {
            (res - self.largest_element as u16) as u8
        } else {
            res as u8
        }
    }

    /// Like [`GaloisField::mul`], but takes two logarithms and returns a
    /// field element directly: `exp[l + r]`. Safe to skip the wraparound
    /// check here because `exp` is sized for exactly this case.
    #[inline]
    pub fn mul_log_element(&self, l: u8, r: u8) -> u8 {
        let res = l as u16 + r as u16;
        self.exp[res as usize]
    }

    /// Raises `elem` to `power` (which may be negative).
    ///
    /// Note: mirrors the C implementation's behavior for `elem == 0`
    /// exactly, including for `power == 0`: since `log(0)` is the
    /// undefined sentinel `0`, this returns `exp[0] == 1` rather than the
    /// mathematically-correct `0^n == 0` (`n != 0`). This is preserved
    /// deliberately for bit-for-bit parity with upstream; callers in this
    /// crate never call `pow` on a zero base with a nonzero exponent in a
    /// way that would expose the discrepancy.
    pub fn pow(&self, elem: u8, power: i32) -> u8 {
        let log = self.log[elem as usize] as i32;
        let res_log = log * power;
        let mut m = res_log % self.largest_element as i32;
        if m < 0 {
            m += self.largest_element as i32;
        }
        self.exp[m as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// x^8 + x^4 + x^3 + x^2 + 1 -- the primitive polynomial CCSDS uses
    /// for its GF(256) RS(255,223) code.
    const POLY_GF256: u16 = 0x11d;

    /// x^4 + x + 1 -- the primitive polynomial for the GF(16) field used
    /// by the CCSDS AOS Frame Header Error Control shortened RS(10,6)
    /// code (quiet/libcorrect#17).
    const POLY_GF16: u16 = 0x13;

    #[test]
    fn field_size_and_largest_element() {
        let gf256 = GaloisField::new(POLY_GF256);
        assert_eq!(gf256.field_size(), 256);
        assert_eq!(gf256.largest_element(), 255);

        let gf16 = GaloisField::new(POLY_GF16);
        assert_eq!(gf16.field_size(), 16);
        assert_eq!(gf16.largest_element(), 15);
    }

    #[test]
    fn gf16_tables_match_known_values() {
        // Hand-verified against the field_create recurrence for 0x13:
        // alpha^i for i in 0..15 (period 15, since largest_element == 15).
        let expected_exp: [u8; 15] = [1, 2, 4, 8, 3, 6, 12, 11, 5, 10, 7, 14, 15, 13, 9];
        // log[e] for e in 1..=15.
        let expected_log: [u8; 16] = [0, 15, 1, 4, 2, 8, 5, 10, 3, 14, 9, 7, 6, 13, 11, 12];

        let gf16 = GaloisField::new(POLY_GF16);
        assert_eq!(&gf16.exp_table()[..15], &expected_exp[..]);
        // exp is periodic with period largest_element == 15.
        assert_eq!(gf16.exp_table()[15], expected_exp[0]);
        assert_eq!(gf16.log_table(), &expected_log[..]);
    }

    #[test]
    fn exp_and_log_are_inverses() {
        for &poly in &[POLY_GF16, POLY_GF256] {
            let gf = GaloisField::new(poly);
            let largest = gf.largest_element();
            for e in 1..=largest {
                let l = gf.log_table()[e as usize];
                assert_eq!(gf.exp_table()[l as usize], e, "exp(log({e})) != {e}");
            }
            // log(exp(i)) == i for i in 1..=largest, but *not* at i == 0:
            // exp[0] == alpha^0 == 1, yet log[1] is stored as
            // `largest_element`, not `0` -- `0` is reserved as the
            // "log(0) is undefined" sentinel, and alpha^largest_element
            // is also 1, so log[1] == largest_element is the convention
            // field_create actually picks (see the module docs).
            for i in 1..=largest {
                let elem = gf.exp_table()[i as usize];
                assert_ne!(elem, 0, "alpha^{i} must never be 0");
                assert_eq!(gf.log_table()[elem as usize], i, "log(exp({i})) != {i}");
            }
            assert_eq!(gf.exp_table()[0], 1, "alpha^0 must be 1");
            assert_eq!(
                gf.log_table()[1], largest,
                "log(1) is expected to alias to largest_element, not 0"
            );
        }
    }

    #[test]
    fn mul_by_zero_and_one() {
        for &poly in &[POLY_GF16, POLY_GF256] {
            let gf = GaloisField::new(poly);
            for e in 0..=gf.largest_element() {
                assert_eq!(gf.mul(e, 0), 0);
                assert_eq!(gf.mul(0, e), 0);
                assert_eq!(gf.mul(e, 1), e);
                assert_eq!(gf.mul(1, e), e);
            }
        }
    }

    #[test]
    fn mul_is_commutative_and_div_inverts_mul() {
        for &poly in &[POLY_GF16, POLY_GF256] {
            let gf = GaloisField::new(poly);
            let n = gf.largest_element();
            for l in 0..=n {
                for r in 1..=n {
                    assert_eq!(gf.mul(l, r), gf.mul(r, l));
                    if l != 0 {
                        assert_eq!(gf.div(gf.mul(l, r), r), l);
                    }
                }
            }
        }
    }

    #[test]
    fn add_and_sub_are_self_inverse_xor() {
        let gf = GaloisField::new(POLY_GF16);
        for l in 0..=gf.largest_element() {
            for r in 0..=gf.largest_element() {
                assert_eq!(gf.add(l, r), l ^ r);
                assert_eq!(gf.sub(gf.add(l, r), r), l);
            }
        }
    }

    #[test]
    fn sum_is_parity_of_n() {
        let gf = GaloisField::new(POLY_GF16);
        for elem in 0..=gf.largest_element() {
            assert_eq!(gf.sum(elem, 0), 0);
            assert_eq!(gf.sum(elem, 1), elem);
            assert_eq!(gf.sum(elem, 2), 0);
            assert_eq!(gf.sum(elem, 3), elem);
        }
    }

    #[test]
    fn pow_matches_repeated_mul() {
        let gf = GaloisField::new(POLY_GF16);
        for elem in 1..=gf.largest_element() {
            let mut acc = 1u8;
            for p in 0..6 {
                assert_eq!(gf.pow(elem, p), acc, "elem={elem} power={p}");
                acc = gf.mul(acc, elem);
            }
        }
    }

    #[test]
    fn mul_log_and_div_log_match_element_domain_ops() {
        let gf = GaloisField::new(POLY_GF16);
        let n = gf.largest_element();
        for l in 1..=n {
            for r in 1..=n {
                let ll = gf.log_table()[l as usize];
                let lr = gf.log_table()[r as usize];
                assert_eq!(gf.exp_table()[gf.mul_log(ll, lr) as usize], gf.mul(l, r));
                assert_eq!(gf.exp_table()[gf.div_log(ll, lr) as usize], gf.div(l, r));
                assert_eq!(gf.mul_log_element(ll, lr), gf.mul(l, r));
            }
        }
    }
}
