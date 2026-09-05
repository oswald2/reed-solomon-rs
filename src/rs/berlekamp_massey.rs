//! Berlekamp-Massey algorithm: finds the shortest LFSR (linear feedback
//! shift register) that generates a given syndrome sequence, i.e. the
//! error locator polynomial. Ported from
//! `reed_solomon_find_error_locator` in `src/reed-solomon/decode.c`.

use crate::field::GaloisField;

/// Finds the error locator polynomial for `syndromes`, writing it into
/// `error_locator` and returning its order (the number of errors).
///
/// `num_erasures` erasures are assumed to already be accounted for
/// elsewhere (in the erasure-decoding path, phase 5): only the first
/// `syndromes.len() - num_erasures` syndromes are consulted here. Pass
/// `0` for the plain (no-erasures) decode path.
///
/// `error_locator` and `last_error_locator` (scratch space -- its
/// contents afterward are unspecified) must each have length
/// `syndromes.len() + 1`; sizing them once and reusing them across
/// decode calls is how this avoids allocating.
pub(crate) fn find_error_locator(
    field: &GaloisField,
    syndromes: &[u8],
    num_erasures: usize,
    error_locator: &mut [u8],
    last_error_locator: &mut [u8],
) -> usize {
    let min_distance = syndromes.len();
    assert_eq!(error_locator.len(), min_distance + 1);
    assert_eq!(last_error_locator.len(), min_distance + 1);

    let mut numerrors: usize = 0;

    // Initialize to f(x) = 1: no errors found yet.
    error_locator.fill(0);
    error_locator[0] = 1;
    let mut order: usize = 0;

    last_error_locator.copy_from_slice(error_locator);
    let mut last_order = order;

    let mut last_discrepancy: u8 = 1;
    let mut delay_length: usize = 1;

    let limit = min_distance - num_erasures;
    let mut i = order;
    while i < limit {
        // The discrepancy is how far the current LFSR's prediction is
        // from the actual next syndrome.
        let mut discrepancy = syndromes[i];
        for j in 1..=numerrors {
            discrepancy = field.add(discrepancy, field.mul(error_locator[j], syndromes[i - j]));
        }

        if discrepancy == 0 {
            // The existing LFSR already describes this syndrome; just
            // track how long it's been since we last needed to correct
            // it, so a future correction can be shifted by the right
            // amount.
            delay_length += 1;
            i += 1;
            continue;
        }

        if 2 * numerrors <= i {
            // There's room to lengthen the LFSR by one tap to eliminate
            // this discrepancy. Shift the previous locator up by
            // delay_length places (scaled to cancel the discrepancy),
            // swap it with the current locator, and grow the order.
            //
            // The shift walks downward (from last_order to 0) so that
            // writing to index j + delay_length (always > j, since
            // delay_length >= 1) never clobbers a value at some smaller
            // index that's still waiting to be read.
            for j in (0..=last_order).rev() {
                last_error_locator[j + delay_length] =
                    field.div(field.mul(last_error_locator[j], discrepancy), last_discrepancy);
            }
            for slot in last_error_locator[..delay_length].iter_mut() {
                *slot = 0;
            }

            // locator = locator - last_locator (both directions -- the
            // old locator becomes the new last_locator).
            for j in 0..=(last_order + delay_length) {
                let temp = error_locator[j];
                error_locator[j] = field.add(error_locator[j], last_error_locator[j]);
                last_error_locator[j] = temp;
            }
            let temp_order = order;
            order = last_order + delay_length;
            last_order = temp_order;

            numerrors = i + 1 - numerrors;
            last_discrepancy = discrepancy;
            delay_length = 1;
            i += 1;
            continue;
        }

        // No more taps available: preserve last_locator as-is, but still
        // apply the correction to locator (this is the same update as
        // above, just without also swapping in a new last_locator).
        for j in (0..=last_order).rev() {
            error_locator[j + delay_length] = field.add(
                error_locator[j + delay_length],
                field.div(field.mul(last_error_locator[j], discrepancy), last_discrepancy),
            );
        }
        order = order.max(last_order + delay_length);
        delay_length += 1;
        i += 1;
    }

    order
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::polynomial;
    use alloc::vec;
    use alloc::vec::Vec;

    const POLY_GF16: u16 = 0x13;

    /// Builds syndromes for a codeword with known errors, by directly
    /// evaluating the error pattern's polynomial at each generator root
    /// -- i.e. simulates "what syndromes would decode() compute" without
    /// needing encode/decode wired up yet.
    fn syndromes_for_errors(field: &GaloisField, roots: &[u8], error_locations_and_vals: &[(usize, u8)]) -> Vec<u8> {
        roots
            .iter()
            .map(|&root| {
                let mut acc = 0u8;
                for &(loc, val) in error_locations_and_vals {
                    // Contribution of one error at position `loc` with
                    // value `val` to the syndrome at this root:
                    // val * root^loc.
                    acc = field.add(acc, field.mul(val, field.pow(root, loc as i32)));
                }
                acc
            })
            .collect()
    }

    #[test]
    fn no_errors_gives_order_zero() {
        let gf = GaloisField::new(POLY_GF16);
        let syndromes = [0u8; 4];
        let mut error_locator = vec![0u8; syndromes.len() + 1];
        let mut last_error_locator = vec![0u8; syndromes.len() + 1];
        let order = find_error_locator(&gf, &syndromes, 0, &mut error_locator, &mut last_error_locator);
        assert_eq!(order, 0);
        assert_eq!(error_locator[0], 1);
    }

    #[test]
    fn single_error_gives_a_linear_locator_with_that_root() {
        let gf = GaloisField::new(POLY_GF16);
        // Generator roots alpha^1..alpha^4 (min_distance = 4).
        let roots: Vec<u8> = (1..=4).map(|p| gf.exp_table()[p]).collect();
        // A single error of value 5 at position 3: the error locator's
        // root should be the reciprocal of alpha^3.
        let error_root = gf.exp_table()[3];
        let syndromes = syndromes_for_errors(&gf, &roots, &[(3, 5)]);

        let mut error_locator = vec![0u8; syndromes.len() + 1];
        let mut last_error_locator = vec![0u8; syndromes.len() + 1];
        let order = find_error_locator(&gf, &syndromes, 0, &mut error_locator, &mut last_error_locator);

        assert_eq!(order, 1);
        // Lambda(x) = 1 + error_root*x, so its root is 1/error_root.
        let expected_root = gf.div(1, error_root);
        assert_eq!(
            polynomial::eval(&gf, &error_locator[..=order], expected_root),
            0
        );
    }

    #[test]
    fn two_errors_gives_a_quadratic_locator_with_both_roots() {
        let gf = GaloisField::new(POLY_GF16);
        let roots: Vec<u8> = (1..=4).map(|p| gf.exp_table()[p]).collect();
        let syndromes = syndromes_for_errors(&gf, &roots, &[(1, 3), (5, 7)]);

        let mut error_locator = vec![0u8; syndromes.len() + 1];
        let mut last_error_locator = vec![0u8; syndromes.len() + 1];
        let order = find_error_locator(&gf, &syndromes, 0, &mut error_locator, &mut last_error_locator);

        assert_eq!(order, 2);
        for &loc in &[1usize, 5] {
            let error_root = gf.exp_table()[loc];
            let expected_root = gf.div(1, error_root);
            assert_eq!(
                polynomial::eval(&gf, &error_locator[..=order], expected_root),
                0,
                "expected root for error at location {loc} not found"
            );
        }
    }
}
