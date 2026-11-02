//! Chien search: brute-force root-finding for the error locator
//! polynomial over the positions of the codeword. Replaces
//! `reed_solomon_factorize_error_locator` and
//! `reed_solomon_find_error_locations` in `src/reed-solomon/decode.c`,
//! which search the whole field and then map each root back to its
//! position by another search over the field.

use crate::field::GaloisField;

/// Finds the roots of `locator` (the error locator polynomial, lowest
/// order coefficient first) among the positions of a `len` symbol
/// (possibly shortened) codeword: an error at position `p` (polynomial
/// coefficient index, i.e. `encoded[len - 1 - p]`) makes
/// `1 / alpha^(generator_root_gap * p)` a root.
///
/// Trying every position reliably finds every root, unlike general
/// polynomial root-finding. The terms `locator[t] * x^t` are carried
/// from one position to the next in log form, `x` changing by the
/// constant factor `alpha^-generator_root_gap`, and the search stops as
/// soon as all roots are found.
///
/// Writes the roots found into `roots` and their positions into
/// `positions` (both of length `locator.len() - 1`) and returns whether
/// all `locator.len() - 1` (its degree) roots were found: if fewer
/// were, Berlekamp-Massey built a locator that happens to fit the
/// syndromes but doesn't factor completely (or has roots outside the
/// codeword), which is how this crate detects "too many errors to
/// recover from" rather than silently returning a wrong answer.
///
/// `terms` is scratch space of at least `locator.len()` entries.
pub(crate) fn find_error_positions(
    field: &GaloisField,
    generator_root_gap: u8,
    len: usize,
    locator: &[u8],
    terms: &mut [[u8; 2]],
    roots: &mut [u8],
    positions: &mut [u8],
) -> bool {
    let order = locator.len() - 1;
    if order == 0 {
        return true;
    }
    let largest = field.largest_element() as u16;
    let exp = field.exp_table();
    let log = field.log_table();
    // log(alpha^-generator_root_gap)
    let step = (largest - generator_root_gap as u16 % largest) % largest;

    // (log(locator[t] * x^t), log of its factor to the next position)
    // per nonzero term, starting at position 0 (x = 1). Logs stay in
    // 0..=largest, where both ends mean 1.
    let mut num_terms = 0;
    for (t, &c) in locator.iter().enumerate() {
        if c != 0 {
            terms[num_terms] = [log[c as usize], ((t as u32 * step as u32) % largest as u32) as u8];
            num_terms += 1;
        }
    }
    let terms = &mut terms[..num_terms];

    let mut found = 0;
    let mut x_log = 0u16;
    for p in 0..len {
        let mut sum = 0u8;
        for [term_log, term_step] in terms.iter_mut() {
            sum ^= exp[*term_log as usize];
            let next = *term_log as u16 + *term_step as u16;
            *term_log = if next >= largest { next - largest } else { next } as u8;
        }
        if sum == 0 {
            roots[found] = exp[x_log as usize];
            positions[found] = p as u8;
            found += 1;
            if found == order {
                return true;
            }
        }
        x_log += step;
        if x_log >= largest {
            x_log -= largest;
        }
    }
    false
}

/// The inverse of [`find_error_positions`]' position mapping: given known codeword
/// positions (erasures, whose locations the receiver already knows by
/// some means outside this codec -- e.g. from a demodulator's confidence
/// signal), computes the error roots those positions correspond to, so
/// an erasure locator polynomial can be built from them via
/// [`crate::polynomial::init_from_roots`].
///
/// A codeword position `p` and its corresponding root are related by
/// `root = 1 / alpha^(generator_root_gap * p)`; recovering `root` from
/// `p` is thus direct exponentiation.
pub(crate) fn find_error_roots_from_locations(
    field: &GaloisField,
    generator_root_gap: u8,
    error_locations: &[u8],
    error_roots: &mut [u8],
) {
    for (root, &location) in error_roots.iter_mut().zip(error_locations.iter()) {
        let alpha_to_location = field.exp_table()[location as usize];
        let loc = field.pow(alpha_to_location, generator_root_gap as i32);
        *root = field.div(1, loc);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial;
    use alloc::vec;

    const POLY_GF16: u16 = 0x13;
    const POLY_GF256: u16 = 0x187;

    fn search(gf: &GaloisField, gap: u8, len: usize, locator: &[u8]) -> Option<(vec::Vec<u8>, vec::Vec<u8>)> {
        let order = locator.len() - 1;
        let mut terms = vec![[0u8; 2]; locator.len()];
        let mut roots = vec![0u8; order];
        let mut positions = vec![0u8; order];
        find_error_positions(gf, gap, len, locator, &mut terms, &mut roots, &mut positions)
            .then_some((roots, positions))
    }

    #[test]
    fn finds_the_roots_and_positions_of_a_product_of_linear_factors() {
        for (poly, gap, len) in [(POLY_GF16, 1u8, 15usize), (POLY_GF256, 11, 255), (POLY_GF256, 11, 40)] {
            let gf = GaloisField::new(poly);
            let error_positions = [3u8, 7, (len - 1) as u8];
            let mut roots = [0u8; 3];
            find_error_roots_from_locations(&gf, gap, &error_positions, &mut roots);
            let locator = polynomial::from_roots(&gf, &roots);

            let (found_roots, found_positions) = search(&gf, gap, len, &locator).expect("all roots found");
            assert_eq!(found_positions, error_positions, "poly {poly:#x} len {len}");
            assert_eq!(found_roots, roots, "poly {poly:#x} len {len}");
        }
    }

    #[test]
    fn roots_outside_a_shortened_codeword_are_not_found() {
        let gf = GaloisField::new(POLY_GF256);
        let mut roots = [0u8; 2];
        find_error_roots_from_locations(&gf, 11, &[5, 100], &mut roots);
        let locator = polynomial::from_roots(&gf, &roots);
        assert!(search(&gf, 11, 101, &locator).is_some());
        assert!(search(&gf, 11, 100, &locator).is_none());
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

        assert!(
            search(&gf, 1, 15, &locator).is_none(),
            "expected this quadratic to not fully factor over GF(16)"
        );
    }

    #[test]
    fn find_error_roots_from_locations_matches_the_root_construction() {
        let gf = GaloisField::new(POLY_GF16);
        let generator_root_gap = 1u8;
        // An error at codeword position `location` produces a root of
        // 1 / alpha^(generator_root_gap * location).
        for location in 0u8..15 {
            let power = gf.exp_table()[(generator_root_gap as usize * location as usize) % gf.largest_element() as usize];
            let mut roots = [0u8];
            find_error_roots_from_locations(&gf, generator_root_gap, &[location], &mut roots);
            assert_eq!(roots[0], gf.div(1, power), "mismatch for location {location}");
        }
    }
}
