//! CCSDS 131.0-B-5, SS4.3.9 / Annex F: conversion between the
//! "conventional" GF(256) symbol representation (the one this crate's
//! [`crate::field::GaloisField`] and every other part of this codec
//! natively use -- byte bit `j` is the coefficient of `alpha^j`) and the
//! "dual basis" (Berlekamp) representation the standard mandates for
//! actual transmission (SS4.3.9.1: "Dual basis representation shall be
//! used").
//!
//! The RS math itself is basis-independent -- encoding/decoding produces
//! the same field elements regardless of which basis you write them
//! down in -- so this conversion only matters when interoperating with
//! something that actually puts dual-basis-encoded bytes on the wire
//! (as CCSDS 131.0-B requires). A [`crate::rs::ReedSolomon`] built with
//! [`super::tm_channel_coding`]'s parameters and fed conventional bytes
//! throughout, with [`to_dual_basis`]/[`from_dual_basis`] applied only
//! at the point of actual transmission/reception, is fully conformant.
//!
//! The conversion is a fixed linear (GF(2)) transform, given in the
//! standard as two 8x8 bit matrices, `T` and `T^-1`. Both are
//! implemented here as one XOR per set bit against a precomputed
//! per-bit-position table -- there are only 8 bits to examine, so this
//! is already about as fast as a conversion like this gets without a
//! full 256-entry lookup table (which would just be this same
//! computation, precomputed).
//!
//! To convert whole buffers (codeblocks), use [`DualBasisConverter`],
//! which uses the same SIMD tiers as the codec (see [`crate::rs`]).

use crate::rs::simd::{LinearMap, Tier};

/// `ROWS_TO_DUAL[i]` is the dual-basis representation of `alpha^(7-i)`
/// (row `i+1` of the standard's matrix `T`), i.e. what
/// [`to_dual_basis`] should XOR in when bit `7-i` of the conventional
/// input byte is set.
const ROWS_TO_DUAL: [u8; 8] = [0x8D, 0xEF, 0xEC, 0x86, 0xFA, 0x99, 0xAF, 0x7B];

/// `ROWS_FROM_DUAL[i]` is the conventional representation of dual-basis
/// vector `l_i` (row `i+1` of the standard's matrix `T^-1`), i.e. what
/// [`from_dual_basis`] should XOR in when bit `7-i` of the dual-basis
/// input byte is set.
const ROWS_FROM_DUAL: [u8; 8] = [0xC5, 0x42, 0x2E, 0xFD, 0xF0, 0x79, 0xAC, 0xCC];

/// Converts a GF(256) element from this crate's native (conventional)
/// representation to the dual basis (Berlekamp) representation CCSDS
/// 131.0-B mandates for transmission.
///
/// The result's bit 7 is `z0` (transmitted first, per SS4.3.9.2) down to
/// bit 0 as `z7`, the same MSB-first convention the conventional
/// representation itself uses for `u7..u0`.
pub fn to_dual_basis(conventional: u8) -> u8 {
    apply_transform(conventional, &ROWS_TO_DUAL)
}

/// The inverse of [`to_dual_basis`]: converts a dual-basis byte (as
/// received off the wire) back to this crate's native conventional
/// representation.
pub fn from_dual_basis(dual: u8) -> u8 {
    apply_transform(dual, &ROWS_FROM_DUAL)
}

fn apply_transform(input: u8, rows: &[u8; 8]) -> u8 {
    let mut output = 0u8;
    for (i, &row) in rows.iter().enumerate() {
        if input & (0x80 >> i) != 0 {
            output ^= row;
        }
    }
    output
}

/// Converts whole buffers between the conventional and the dual basis
/// representation, with SIMD where the CPU has it (picked once in
/// [`DualBasisConverter::new`], like for [`crate::rs::ReedSolomon`]).
#[derive(Debug, Clone)]
pub struct DualBasisConverter {
    to_dual: LinearMap,
    from_dual: LinearMap,
}

impl Default for DualBasisConverter {
    fn default() -> Self {
        Self::new()
    }
}

impl DualBasisConverter {
    pub fn new() -> Self {
        Self::with_tier(Tier::detect())
    }

    fn with_tier(tier: Tier) -> Self {
        DualBasisConverter {
            to_dual: LinearMap::new(to_dual_basis, tier),
            from_dual: LinearMap::new(from_dual_basis, tier),
        }
    }

    /// [`to_dual_basis`] of every symbol of `conventional`, written to
    /// `dual`. Panics if the lengths differ.
    pub fn to_dual(&self, conventional: &[u8], dual: &mut [u8]) {
        self.to_dual.apply(conventional, dual)
    }

    /// [`from_dual_basis`] of every symbol of `dual`, written to
    /// `conventional`. Panics if the lengths differ.
    pub fn from_dual(&self, dual: &[u8], conventional: &mut [u8]) {
        self.from_dual.apply(dual, conventional)
    }

    /// [`to_dual_basis`] of every symbol of `buf`, in place.
    pub fn to_dual_in_place(&self, buf: &mut [u8]) {
        self.to_dual.apply_in_place(buf)
    }

    /// [`from_dual_basis`] of every symbol of `buf`, in place.
    pub fn from_dual_in_place(&self, buf: &mut [u8]) {
        self.from_dual.apply_in_place(buf)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn the_converter_matches_the_symbol_functions_in_every_tier() {
        // Long enough for whole SIMD chunks and a tail, at an odd offset.
        let data: Vec<u8> = (0..=255u8).chain(0..=100).collect();
        let data = &data[3..];
        let to: Vec<u8> = data.iter().map(|&b| to_dual_basis(b)).collect();
        let from: Vec<u8> = data.iter().map(|&b| from_dual_basis(b)).collect();
        for tier in Tier::ALL {
            let conv = DualBasisConverter::with_tier(tier);
            let mut out = alloc::vec![0u8; data.len()];
            conv.to_dual(data, &mut out);
            assert_eq!(out, to, "{tier:?}");
            conv.from_dual(data, &mut out);
            assert_eq!(out, from, "{tier:?}");
            out.copy_from_slice(data);
            conv.to_dual_in_place(&mut out);
            assert_eq!(out, to, "{tier:?}");
            conv.from_dual_in_place(&mut out);
            assert_eq!(out, data, "{tier:?}");
        }
    }

    /// CCSDS 131.0-B-5 annex F, Example 1: dual-basis `10111001`
    /// converts to conventional `00101010` (published as `alpha^213`).
    #[test]
    fn matches_the_standards_worked_example_1() {
        assert_eq!(from_dual_basis(0b1011_1001), 0b0010_1010);
    }

    /// CCSDS 131.0-B-5 annex F, Example 2: conventional `01011001`
    /// (published as `alpha^152`) converts to dual-basis `11101000`.
    #[test]
    fn matches_the_standards_worked_example_2() {
        assert_eq!(to_dual_basis(0b0101_1001), 0b1110_1000);
    }

    #[test]
    fn the_two_worked_examples_are_also_correct_in_the_other_direction() {
        // Since both examples give a matched (conventional, dual) pair,
        // each direction's function should invert the other's example
        // too, not just its own.
        assert_eq!(to_dual_basis(0b0010_1010), 0b1011_1001);
        assert_eq!(from_dual_basis(0b1110_1000), 0b0101_1001);
    }

    #[test]
    fn the_two_transforms_are_exact_inverses_over_the_whole_field() {
        for byte in 0u16..256 {
            let byte = byte as u8;
            assert_eq!(from_dual_basis(to_dual_basis(byte)), byte);
            assert_eq!(to_dual_basis(from_dual_basis(byte)), byte);
        }
    }

    #[test]
    fn zero_maps_to_zero() {
        // The all-zero vector is fixed by any linear transform.
        assert_eq!(to_dual_basis(0), 0);
        assert_eq!(from_dual_basis(0), 0);
    }
}
