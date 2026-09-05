//! Chien search: brute-force root-finding for the error locator
//! polynomial, plus mapping the roots found back to byte positions in
//! the codeword. Ported from `reed_solomon_factorize_error_locator` and
//! `reed_solomon_find_error_locations` in `src/reed-solomon/decode.c`.

use alloc::vec::Vec;

use crate::field::GaloisField;
use crate::polynomial;

/// Finds every root of `locator_log` (the error locator polynomial, in
/// log-coefficient form -- see [`polynomial::eval_log_lut`]) by brute
/// force: the field is small enough that just trying every element is
/// fast, and reliably finds every root, unlike general polynomial
/// root-finding.
///
/// Writes the roots found into `roots[num_skip..]` and returns whether
/// exactly `locator_log.len() - 1` roots were found (its degree): if
/// fewer were found, Berlekamp-Massey built a locator that happens to
/// fit the syndromes but doesn't factor completely, which is how this
/// crate (like the C it's ported from) detects "too many errors to
/// recover from" rather than silently returning a wrong answer.
///
/// `num_skip` reserves the first `num_skip` slots of `roots` for the
/// erasure locator's roots, filled in separately by the erasure-decoding
/// path (phase 5); pass `0` for the plain (no-erasures) decode path.
/// `element_exp[e]` must hold the successive log-powers of field element
/// `e`, at least `locator_log.len()` of them (see
/// [`polynomial::build_exp_lut`]).
pub(crate) fn factorize_error_locator(
    field: &GaloisField,
    num_skip: usize,
    locator_log: &[u8],
    roots: &mut [u8],
    element_exp: &[Vec<u8>],
) -> bool {
    let order = locator_log.len() - 1;
    let mut found = num_skip;
    for slot in roots[num_skip..num_skip + order].iter_mut() {
        *slot = 0;
    }
    for (i, exp) in element_exp.iter().enumerate().take(field.field_size() as usize) {
        if polynomial::eval_log_lut(field, locator_log, &exp[..locator_log.len()]) == 0 {
            roots[found] = i as u8;
            found += 1;
        }
    }
    found == order + num_skip
}

/// Converts the error locator's roots (found by [`factorize_error_locator`])
/// into byte positions within the codeword, writing them into
/// `error_locations`.
///
/// An error root is the *reciprocal* of `alpha^(generator_root_gap * location)`,
/// so recovering `location` means dividing it back out and then searching
/// for which power of `alpha^generator_root_gap` produced it -- again by
/// brute force over the field, same as the root search itself.
///
/// A root of exactly `0` is skipped (left unwritten) as a defensive
/// no-op: `0` can never actually be a root of the error locator (its
/// constant term is always `1`, by construction -- see
/// `berlekamp_massey`'s docs), so this only matters if it's ever called
/// with the buffer in an inconsistent state.
pub(crate) fn find_error_locations(
    field: &GaloisField,
    generator_root_gap: u8,
    error_roots: &[u8],
    error_locations: &mut [u8],
) {
    for (location, &root) in error_locations.iter_mut().zip(error_roots.iter()) {
        if root == 0 {
            continue;
        }
        let target = field.div(1, root);
        for j in 0..field.field_size() as usize {
            if field.pow(j as u8, generator_root_gap as i32) == target {
                *location = field.log_table()[j];
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    const POLY_GF16: u16 = 0x13;

    fn build_element_exp(field: &GaloisField, width: usize) -> Vec<Vec<u8>> {
        (0..field.field_size())
            .map(|e| {
                let mut lut = vec![0u8; width];
                polynomial::build_exp_lut(field, e as u8, &mut lut);
                lut
            })
            .collect()
    }

    #[test]
    fn finds_the_known_roots_of_a_product_of_linear_factors() {
        let gf = GaloisField::new(POLY_GF16);
        // Lambda(x) = (1 + r1*x)(1 + r2*x), roots are 1/r1 and 1/r2.
        let r1 = gf.exp_table()[3];
        let r2 = gf.exp_table()[7];
        let root1 = gf.div(1, r1);
        let root2 = gf.div(1, r2);
        let locator = polynomial::from_roots(&gf, &[root1, root2]);
        let locator_log: Vec<u8> = locator
            .iter()
            .map(|&c| gf.log_table()[c as usize])
            .collect();

        let element_exp = build_element_exp(&gf, locator_log.len());
        let mut roots = vec![0u8; 2];
        let ok = factorize_error_locator(&gf, 0, &locator_log, &mut roots, &element_exp);

        assert!(ok);
        let mut found = roots.clone();
        found.sort_unstable();
        let mut expected = [root1, root2];
        expected.sort_unstable();
        assert_eq!(found, expected);
    }

    #[test]
    fn reports_failure_when_the_locator_does_not_fully_factor() {
        let gf = GaloisField::new(POLY_GF16);
        // An irreducible quadratic 1 + b*x + x^2 (no roots in the field
        // at all) reliably exercises the "not enough roots found"
        // failure path. Search for one by brute force rather than
        // asserting a hand-picked example is irreducible -- a field this
        // small makes that cheap, and it keeps the test honest about why
        // it's expected to fail to factor.
        let locator = (0..=gf.largest_element())
            .find_map(|b| {
                let candidate = [1u8, b, 1];
                let has_root = (0..=gf.largest_element()).any(|x| polynomial::eval(&gf, &candidate, x) == 0);
                if has_root {
                    None
                } else {
                    Some(candidate)
                }
            })
            .expect("GF(16) must contain at least one irreducible quadratic");

        let locator_log: Vec<u8> = locator
            .iter()
            .map(|&c| gf.log_table()[c as usize])
            .collect();
        let element_exp = build_element_exp(&gf, locator_log.len());
        let mut roots = vec![0u8; 2];
        let ok = factorize_error_locator(&gf, 0, &locator_log, &mut roots, &element_exp);
        assert!(!ok, "expected this quadratic to not fully factor over GF(16)");
    }

    #[test]
    fn find_error_locations_inverts_the_root_construction() {
        let gf = GaloisField::new(POLY_GF16);
        let generator_root_gap = 1u8;
        // An error at codeword position `location` produces a root of
        // 1 / alpha^(generator_root_gap * location) -- mirror that here
        // and check find_error_locations recovers `location`.
        for location in 0u8..15 {
            let power = gf.exp_table()[(generator_root_gap as usize * location as usize) % gf.largest_element() as usize];
            let root = gf.div(1, power);
            let mut locations = [0u8];
            find_error_locations(&gf, generator_root_gap, &[root], &mut locations);
            assert_eq!(locations[0], location, "mismatch for location {location}");
        }
    }
}
