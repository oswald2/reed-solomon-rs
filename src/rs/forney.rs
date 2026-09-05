//! Forney's algorithm: given the error locator polynomial and its roots,
//! computes the actual correction value at each error location. Ported
//! from `reed_solomon_find_error_evaluator`/`reed_solomon_find_error_values`
//! in `src/reed-solomon/decode.c`.

use alloc::vec::Vec;

use crate::field::GaloisField;
use crate::polynomial;

/// Computes the correction value for each error, writing them into
/// `error_vals` at the same index as the corresponding entry of
/// `error_roots`.
///
/// The error value at root `X^-1` is
/// `-(X^(1-c) * omega(X^-1)) / lambda'(X^-1)`, where `omega` is the
/// *error evaluator* polynomial (`error_locator * syndrome_polynomial`,
/// truncated to `min_distance` terms), `lambda'` is the error locator's
/// formal derivative, and `c` is the generator's first consecutive root
/// (`first_consecutive_root`). Negation is a no-op in this field
/// (characteristic 2), so it's omitted.
///
/// `error_evaluator` is scratch space for computing `omega`; its length
/// determines how many terms of the (otherwise unbounded) product
/// `error_locator * syndromes` are kept, and must equal `syndromes.len()`
/// (`min_distance`). `error_locator_derivative` is scratch space for
/// `lambda'`, and must have length `error_locator.len() - 1`.
/// `element_exp[e]` must hold the successive log-powers of field element
/// `e` (see [`polynomial::build_exp_lut`]), at least
/// `error_evaluator.len()` of them.
///
/// A root of exactly `0` is skipped, for the same reason (and with the
/// same "should be unreachable in practice" caveat) as in
/// [`super::chien::find_error_locations`].
#[allow(clippy::too_many_arguments)]
pub(crate) fn find_error_values(
    field: &GaloisField,
    error_locator: &[u8],
    syndromes: &[u8],
    error_roots: &[u8],
    first_consecutive_root: u8,
    element_exp: &[Vec<u8>],
    error_evaluator: &mut [u8],
    error_locator_derivative: &mut [u8],
    error_vals: &mut [u8],
) {
    // omega(x) = Lambda(x) * S(x) mod x^min_distance, where S(x) is
    // built directly from the syndromes (not the polynomial whose roots
    // are the syndromes -- just their values used as coefficients).
    polynomial::mul(field, error_locator, syndromes, error_evaluator);

    polynomial::formal_derivative(field, error_locator, error_locator_derivative);

    for (i, &root) in error_roots.iter().enumerate() {
        if root == 0 {
            continue;
        }
        let val_exp = &element_exp[root as usize];
        error_vals[i] = field.mul(
            field.pow(root, first_consecutive_root as i32 - 1),
            field.div(
                polynomial::eval_lut(field, error_evaluator, &val_exp[..error_evaluator.len()]),
                polynomial::eval_lut(field, error_locator_derivative, &val_exp[..error_locator_derivative.len()]),
            ),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rs::berlekamp_massey::find_error_locator;
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

    /// End-to-end sanity check tying berlekamp_massey, chien, and forney
    /// together directly (without the full `ReedSolomon` decode path,
    /// which doesn't exist yet): given a known error pattern, recompute
    /// its syndromes by hand, run the three decode stages in sequence,
    /// and check the recovered (location, value) pairs match what was
    /// injected.
    #[test]
    fn recovers_known_error_locations_and_values() {
        let gf = GaloisField::new(POLY_GF16);
        let min_distance = 4;
        let first_consecutive_root = 1u8;
        let generator_root_gap = 1u8;
        let roots: Vec<u8> = (0..min_distance)
            .map(|i| gf.exp_table()[(generator_root_gap as usize * (i + first_consecutive_root as usize)) % gf.largest_element() as usize])
            .collect();

        // Two injected errors: value 5 at codeword position 2, value 9
        // at position 10.
        let injected = [(2usize, 5u8), (10usize, 9u8)];
        let syndromes: Vec<u8> = roots
            .iter()
            .map(|&r| {
                injected.iter().fold(0u8, |acc, &(loc, val)| {
                    gf.add(acc, gf.mul(val, gf.pow(r, loc as i32)))
                })
            })
            .collect();

        let mut error_locator = vec![0u8; min_distance + 1];
        let mut last_error_locator = vec![0u8; min_distance + 1];
        let order = find_error_locator(&gf, &syndromes, 0, &mut error_locator, &mut last_error_locator);
        assert_eq!(order, injected.len());

        let locator_log: Vec<u8> = error_locator[..=order]
            .iter()
            .map(|&c| gf.log_table()[c as usize])
            .collect();
        let element_exp = build_element_exp(&gf, min_distance.max(order + 1));

        let mut error_roots = vec![0u8; order];
        let ok = super::super::chien::factorize_error_locator(&gf, 0, &locator_log, &mut error_roots, &element_exp);
        assert!(ok);

        let mut error_locations = vec![0u8; order];
        super::super::chien::find_error_locations(&gf, generator_root_gap, &error_roots, &mut error_locations);

        let mut error_evaluator = vec![0u8; min_distance];
        let mut error_locator_derivative = vec![0u8; order];
        let mut error_vals = vec![0u8; order];
        find_error_values(
            &gf,
            &error_locator[..=order],
            &syndromes,
            &error_roots,
            first_consecutive_root,
            &element_exp,
            &mut error_evaluator,
            &mut error_locator_derivative,
            &mut error_vals,
        );

        let mut recovered: Vec<(u8, u8)> = error_locations
            .iter()
            .zip(error_vals.iter())
            .map(|(&l, &v)| (l, v))
            .collect();
        recovered.sort_unstable();
        let mut expected: Vec<(u8, u8)> = injected.iter().map(|&(l, v)| (l as u8, v)).collect();
        expected.sort_unstable();
        assert_eq!(recovered, expected);
    }
}
