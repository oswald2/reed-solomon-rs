//! CCSDS 131.0-B-5, section 4: TM Synchronization and Channel Coding's
//! Reed-Solomon code, over GF(256), with two selectable error-correction
//! strengths and optional symbol interleaving.
//!
//! Parameters, quoted/derived from the standard (SS4.3.3, SS4.3.4):
//!
//! - Field generator polynomial: `F(x) = x^8 + x^7 + x^2 + x + 1` over
//!   GF(2) (`0x187` -- also `correct_rs_primitive_polynomial_ccsds` in
//!   the original C library this crate is ported from).
//! - `J = 8` bits per symbol, `n = 255` symbols per codeword.
//! - Code generator polynomial `g(x) = product(x - alpha^(11*j))` for
//!   `j` in `128-E ..= 127+E`, i.e. `generator_root_gap = 11`,
//!   `first_consecutive_root = 128 - E`, `num_roots = 2*E`.
//! - `E = 16`: a (255,223) code ([`codec_e16`]). `E = 8`: a (255,239)
//!   code ([`codec_e8`]), lower overhead but weaker correction.
//! - Symbol interleaving depths `I` in `{1, 2, 3, 4, 5, 8}` (SS4.3.5.1),
//!   implemented generically as [`encode_interleaved`]/
//!   [`decode_interleaved`] on top of either codec (SS4.4.1).
//! - Shortened codeblocks via "virtual fill" (SS4.3.7) are just this
//!   crate's existing shortened-code support (see
//!   [`crate::rs::ReedSolomon::encode`]) -- pass a message shorter than
//!   `message_length()`.
//!
//! Not yet covered here: the dual-basis symbol representation the
//! standard requires for transmission (SS4.3.9) -- see [`super::dual_basis`],
//! applied separately at the point of actual transmission/reception.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::RsError;
use crate::rs::ReedSolomon;

/// The primitive polynomial for CCSDS TM channel coding's GF(256):
/// `x^8 + x^7 + x^2 + x + 1`. Identical to
/// `correct_rs_primitive_polynomial_ccsds` in the original C library.
pub const PRIMITIVE_POLYNOMIAL: u16 = 0x187;
/// The generator polynomial's root gap: its roots are consecutive
/// powers of `alpha^11`, not `alpha` itself.
pub const GENERATOR_ROOT_GAP: u8 = 11;

/// Interleaving depths CCSDS 131.0-B allows (SS4.3.5.1). `I = 1` is
/// equivalent to no interleaving.
pub const ALLOWED_INTERLEAVING_DEPTHS: [usize; 6] = [1, 2, 3, 4, 5, 8];

/// Builds the `E = 16` code: `(255,223)`, correcting up to 16 symbol
/// errors per codeword.
pub fn codec_e16() -> ReedSolomon {
    // first_consecutive_root = 128 - E = 112, num_roots = 2*E = 32.
    ReedSolomon::new(PRIMITIVE_POLYNOMIAL, 112, GENERATOR_ROOT_GAP, 32)
        .expect("CCSDS TM channel coding's E=16 parameters are always valid")
}

/// Builds the `E = 8` code: `(255,239)`, correcting up to 8 symbol
/// errors per codeword, at lower parity overhead than [`codec_e16`].
pub fn codec_e8() -> ReedSolomon {
    // first_consecutive_root = 128 - E = 120, num_roots = 2*E = 16.
    ReedSolomon::new(PRIMITIVE_POLYNOMIAL, 120, GENERATOR_ROOT_GAP, 16)
        .expect("CCSDS TM channel coding's E=8 parameters are always valid")
}

/// Encodes `msg` across `depth` interleaved codewords (CCSDS 131.0-B-5
/// SS4.4.1), using `rs` for each sub-codeword in turn.
///
/// `depth` must be one of [`ALLOWED_INTERLEAVING_DEPTHS`], and
/// `msg.len()` must be a multiple of `depth` (so each sub-codeword gets
/// the same message length -- shorter than `rs.message_length()` is
/// fine, corresponding to the standard's "virtual fill", as long as the
/// total shortfall divides evenly across the `depth` sub-codewords).
///
/// `msg`'s symbols are assumed already demultiplexed in the standard's
/// round-robin order (symbol `i` belongs to sub-codeword `i % depth`);
/// the output places every message symbol first (unchanged, in the same
/// order they arrived), followed by all `depth * rs.min_distance()`
/// parity symbols, themselves interleaved round-robin by parity
/// position across sub-codewords, exactly as SS4.4.1 describes.
///
/// Unlike the core [`ReedSolomon`] methods, this allocates two small,
/// `depth`-independent scratch buffers per call: the `depth`-fold loop
/// over full RS encodes this performs already dominates the cost, so
/// avoiding two small `Vec`s here wouldn't meaningfully help, and doing
/// so would need `depth` times as much caller-provided scratch instead.
pub fn encode_interleaved(
    rs: &mut ReedSolomon,
    depth: usize,
    msg: &[u8],
    encoded: &mut [u8],
) -> Result<usize, RsError> {
    if !ALLOWED_INTERLEAVING_DEPTHS.contains(&depth) || !msg.len().is_multiple_of(depth) {
        return Err(RsError::InvalidInterleavingDepth);
    }
    let per_codeword_msg_len = msg.len() / depth;
    let written = msg.len() + depth * rs.min_distance();
    if encoded.len() < written {
        return Err(RsError::BufferTooSmall);
    }

    // Message symbols pass straight through, unchanged, in their
    // original (already-interleaved) order.
    encoded[..msg.len()].copy_from_slice(msg);

    let mut codeword_msg = vec![0u8; per_codeword_msg_len];
    let mut codeword_encoded = vec![0u8; per_codeword_msg_len + rs.min_distance()];
    for j in 0..depth {
        for (i, slot) in codeword_msg.iter_mut().enumerate() {
            *slot = msg[i * depth + j];
        }
        rs.encode(&codeword_msg, &mut codeword_encoded)?;

        let parity = &codeword_encoded[per_codeword_msg_len..];
        for (p, &sym) in parity.iter().enumerate() {
            encoded[msg.len() + p * depth + j] = sym;
        }
    }
    Ok(written)
}

/// The inverse of [`encode_interleaved`]: recovers the original,
/// still-interleaved message symbols from `encoded` (message symbols
/// followed by round-robin-interleaved parity, as `encode_interleaved`
/// produces), correcting up to `rs.min_distance() / 2` symbol errors
/// *per sub-codeword* (i.e. up to that many errors in each of the
/// `depth` independent codewords, not just that many in the block as a
/// whole -- interleaving spreads a burst of consecutive corrupted
/// symbols across multiple sub-codewords, which is what makes it
/// effective against burst noise).
///
/// `depth` must be one of [`ALLOWED_INTERLEAVING_DEPTHS`], and
/// `encoded.len()` must be a multiple of `depth`.
pub fn decode_interleaved(
    rs: &mut ReedSolomon,
    depth: usize,
    encoded: &[u8],
    msg: &mut [u8],
) -> Result<usize, RsError> {
    if !ALLOWED_INTERLEAVING_DEPTHS.contains(&depth) || !encoded.len().is_multiple_of(depth) {
        return Err(RsError::InvalidInterleavingDepth);
    }
    let per_codeword_len = encoded.len() / depth;
    if per_codeword_len < rs.min_distance() {
        return Err(RsError::EncodedTooShort);
    }
    let per_codeword_msg_len = per_codeword_len - rs.min_distance();
    let msg_len = per_codeword_msg_len * depth;
    if msg.len() < msg_len {
        return Err(RsError::BufferTooSmall);
    }

    let mut codeword: Vec<u8> = vec![0u8; per_codeword_len];
    let mut codeword_msg = vec![0u8; per_codeword_msg_len];
    for j in 0..depth {
        for (i, slot) in codeword[..per_codeword_msg_len].iter_mut().enumerate() {
            *slot = encoded[i * depth + j];
        }
        for p in 0..rs.min_distance() {
            codeword[per_codeword_msg_len + p] = encoded[msg_len + p * depth + j];
        }

        rs.decode(&codeword, &mut codeword_msg)?;

        for (i, &sym) in codeword_msg.iter().enumerate() {
            msg[i * depth + j] = sym;
        }
    }
    Ok(msg_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// CCSDS 131.0-B-5 annex G's full published generator polynomial
    /// coefficients for E=16, `G0..G32`, transcribed directly from the
    /// standard's `alpha7 alpha6 ... alpha0` bit columns (so this is a
    /// byte-for-byte match against the standard, not just an
    /// internally-consistent computation). Passing this is a much
    /// stronger check than the AOS FHEC generator check (33
    /// coefficients here, vs. 5 there).
    const E16_GENERATOR: [u8; 33] = [
        0b0000_0001, // G0  = alpha^0
        0b0101_1011, // G1  = alpha^249
        0b0111_1111, // G2  = alpha^59
        0b0101_0110, // G3  = alpha^66
        0b0001_0000, // G4  = alpha^4
        0b0001_1110, // G5  = alpha^43
        0b0000_1101, // G6  = alpha^126
        0b1110_1011, // G7  = alpha^251
        0b0110_0001, // G8  = alpha^97
        0b1010_0101, // G9  = alpha^30
        0b0000_1000, // G10 = alpha^3
        0b0010_1010, // G11 = alpha^213
        0b0011_0110, // G12 = alpha^50
        0b0101_0110, // G13 = alpha^66 (== G3, per the standard's own note)
        0b1010_1011, // G14 = alpha^170
        0b0010_0000, // G15 = alpha^5
        0b0111_0001, // G16 = alpha^24
        0b0010_0000, // G17 = G15
        0b1010_1011, // G18 = G14
        0b0101_0110, // G19 = G3 = G13
        0b0011_0110, // G20 = G12
        0b0010_1010, // G21 = G11
        0b0000_1000, // G22 = G10
        0b1010_0101, // G23 = G9
        0b0110_0001, // G24 = G8
        0b1110_1011, // G25 = G7
        0b0000_1101, // G26 = G6
        0b0001_1110, // G27 = G5
        0b0001_0000, // G28 = G4
        0b0101_0110, // G29 = G3
        0b0111_1111, // G30 = G2
        0b0101_1011, // G31 = G1
        0b0000_0001, // G32 = G0
    ];

    /// CCSDS 131.0-B-5 annex G's full published generator polynomial
    /// coefficients for E=8, `G0..G16`.
    const E8_GENERATOR: [u8; 17] = [
        0b0000_0001, // G0  = alpha^0
        0b1010_0101, // G1  = alpha^30
        0b0110_1001, // G2  = alpha^230
        0b0001_1011, // G3  = alpha^49
        0b1001_1111, // G4  = alpha^235
        0b0110_1000, // G5  = alpha^129
        0b1001_1000, // G6  = alpha^81
        0b0110_0101, // G7  = alpha^76
        0b0100_1010, // G8  = alpha^173
        0b0110_0101, // G9  = G7
        0b1001_1000, // G10 = G6
        0b0110_1000, // G11 = G5
        0b1001_1111, // G12 = G4
        0b0001_1011, // G13 = G3
        0b0110_1001, // G14 = G2
        0b1010_0101, // G15 = G1
        0b0000_0001, // G16 = G0
    ];

    #[test]
    fn e16_generator_polynomial_matches_the_published_standard() {
        let rs = codec_e16();
        assert_eq!(rs.generator(), &E16_GENERATOR[..]);
    }

    #[test]
    fn e8_generator_polynomial_matches_the_published_standard() {
        let rs = codec_e8();
        assert_eq!(rs.generator(), &E8_GENERATOR[..]);
    }

    #[test]
    fn construction_parameters() {
        let e16 = codec_e16();
        assert_eq!(e16.block_length(), 255);
        assert_eq!(e16.min_distance(), 32);
        assert_eq!(e16.message_length(), 223);

        let e8 = codec_e8();
        assert_eq!(e8.block_length(), 255);
        assert_eq!(e8.min_distance(), 16);
        assert_eq!(e8.message_length(), 239);
    }

    #[test]
    fn round_trips_with_no_errors_non_interleaved() {
        let mut rs = codec_e16();
        let msg: Vec<u8> = (0..223).map(|i| (i * 7) as u8).collect();
        let mut encoded = vec![0u8; 255];
        rs.encode(&msg, &mut encoded).unwrap();

        let mut recovered = vec![0u8; 223];
        let n = rs.decode(&encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn interleaved_round_trip_with_no_errors() {
        let mut rs = codec_e16();
        let depth = 5;
        let msg: Vec<u8> = (0..223 * depth).map(|i| (i * 3 + 1) as u8).collect();

        let mut encoded = vec![0u8; msg.len() + depth * rs.min_distance()];
        let written = encode_interleaved(&mut rs, depth, &msg, &mut encoded).unwrap();
        assert_eq!(written, encoded.len());

        let mut recovered = vec![0u8; msg.len()];
        let n = decode_interleaved(&mut rs, depth, &encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn interleaving_survives_a_burst_that_would_defeat_a_single_codeword() {
        // A burst of `depth` consecutive corrupted symbols lands one
        // error in each of the `depth` sub-codewords when interleaved,
        // rather than `depth` errors concentrated in a single codeword
        // -- this is the entire point of interleaving against burst
        // noise. Use E=8 (min_distance=16, corrects up to 8 errors per
        // sub-codeword) with a burst of depth=5 consecutive symbols:
        // fatal for one un-interleaved codeword already at 5 errors
        // only if repeated past its own capacity, but here each
        // sub-codeword only picks up 1 of the 5.
        let mut rs = codec_e8();
        let depth = 5;
        let msg: Vec<u8> = (0..239 * depth).map(|i| (i * 11 + 5) as u8).collect();

        let mut encoded = vec![0u8; msg.len() + depth * rs.min_distance()];
        encode_interleaved(&mut rs, depth, &msg, &mut encoded).unwrap();

        // Burst-corrupt 5 consecutive symbols right at the start.
        for byte in encoded.iter_mut().take(depth) {
            *byte ^= 0xff;
        }

        let mut recovered = vec![0u8; msg.len()];
        let n = decode_interleaved(&mut rs, depth, &encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn interleaved_encode_supports_shortened_sub_codewords() {
        // CCSDS's "virtual fill": each sub-codeword can be shortened,
        // as long as the total shortfall divides evenly across depth.
        let mut rs = codec_e16();
        let depth = 4;
        let per_codeword_msg_len = 50; // well under message_length() (223)
        let msg: Vec<u8> = (0..per_codeword_msg_len * depth).map(|i| i as u8).collect();

        let mut encoded = vec![0u8; msg.len() + depth * rs.min_distance()];
        let written = encode_interleaved(&mut rs, depth, &msg, &mut encoded).unwrap();
        assert_eq!(written, msg.len() + depth * rs.min_distance());

        let mut recovered = vec![0u8; msg.len()];
        let n = decode_interleaved(&mut rs, depth, &encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn rejects_disallowed_interleaving_depth() {
        let mut rs = codec_e16();
        let msg = vec![0u8; 223 * 6];
        let mut encoded = vec![0u8; msg.len() + 6 * rs.min_distance()];
        assert_eq!(
            encode_interleaved(&mut rs, 6, &msg, &mut encoded), // 6 is not in ALLOWED_INTERLEAVING_DEPTHS
            Err(RsError::InvalidInterleavingDepth)
        );
    }

    #[test]
    fn rejects_message_length_not_divisible_by_depth() {
        let mut rs = codec_e16();
        let msg = vec![0u8; 100]; // not a multiple of depth 3
        let mut encoded = vec![0u8; 300];
        assert_eq!(
            encode_interleaved(&mut rs, 3, &msg, &mut encoded),
            Err(RsError::InvalidInterleavingDepth)
        );
    }
}
