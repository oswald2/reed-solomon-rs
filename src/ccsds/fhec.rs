//! CCSDS 732.0-B-4, SS4.1.2.6.5: AOS Frame Header Error Control.
//!
//! A shortened RS(10,6) code over GF(2^4), used to protect 24 bits (the
//! 10-bit Master Channel Identifier, 6-bit Virtual Channel Identifier,
//! and 8-bit Signaling Field) of an AOS Transfer Frame Primary Header,
//! with the 4 resulting parity symbols (16 bits) carried in a dedicated
//! Frame Header Error Control field. This is the code
//! quiet/libcorrect#17 asked for.
//!
//! Parameters, quoted from the standard:
//!
//! - Field generator polynomial: `F(x) = x^4 + x + 1` over GF(2)
//!   (packed as `0x13`).
//! - `J = 4` bits per R-S symbol.
//! - `E = 2` symbol error correction capability per codeword
//!   (`min_distance = 2*E = 4`).
//! - Code generator polynomial:
//!   `g(x) = (x + alpha^6)(x + alpha^7)(x + alpha^8)(x + alpha^9)`
//!   over GF(2^4) -- i.e. `first_consecutive_root = 6`,
//!   `generator_root_gap = 1`, `num_roots = 4`. (Earlier, less-verified
//!   attempts at this preset elsewhere in this crate's history assumed
//!   `first_consecutive_root = 1`; the standard's own worked example for
//!   `g(x)`, checked in this module's tests, confirms `6` is correct.)
//!
//! Each symbol is a 4-bit nibble in `0..=15`, transmitted MSB-first
//! within the symbol (the standard's own example: `alpha^3 = 0b1000` is
//! transmitted as a `1` followed by three `0`s). The six message symbols
//! aren't a standalone field -- they're drawn directly from the 24 bits
//! of header fields being protected (symbols 0-3 from the 16-bit
//! MCID+VCID field, symbols 4-5 from the 8-bit Signaling Field); only
//! the four parity symbols occupy a dedicated field. Extracting/
//! inserting those bits from/into a full AOS Primary Header is a framing
//! concern for the caller (or a higher-level AOS-framing crate) -- this
//! module provides the codec itself, operating on symbols already
//! unpacked into one nibble per byte.

use crate::error::RsError;
use crate::rs::ReedSolomon;

/// The primitive polynomial for CCSDS AOS FHEC's GF(16): `x^4 + x + 1`.
pub const PRIMITIVE_POLYNOMIAL: u16 = 0x13;
/// The generator polynomial's first consecutive root, `alpha^6`.
pub const FIRST_CONSECUTIVE_ROOT: u8 = 6;
/// The generator polynomial's root gap: its roots are consecutive
/// powers of `alpha` (`alpha^6, alpha^7, alpha^8, alpha^9`).
pub const GENERATOR_ROOT_GAP: u8 = 1;
/// Number of parity symbols, `2 * E` for `E = 2` correctable symbol
/// errors per codeword.
pub const NUM_ROOTS: usize = 4;
/// Message length in symbols (nibbles): the 6 symbols drawn from the
/// protected header fields.
pub const MESSAGE_LENGTH: usize = 6;
/// Codeword length in symbols (nibbles): 6 message + 4 parity.
pub const CODEWORD_LENGTH: usize = MESSAGE_LENGTH + NUM_ROOTS;

/// Builds the CCSDS AOS Frame Header Error Control codec.
pub fn codec() -> ReedSolomon {
    ReedSolomon::new(
        PRIMITIVE_POLYNOMIAL,
        FIRST_CONSECUTIVE_ROOT,
        GENERATOR_ROOT_GAP,
        NUM_ROOTS,
    )
    .expect("CCSDS AOS FHEC's fixed parameters are always valid")
}

/// Encodes the 6 message symbols (each a nibble, `0..=15`) into the 4
/// parity symbols the Frame Header Error Control field carries.
///
/// `msg` holds one symbol per byte (`0..=15`); packing those symbols
/// into (or, for `msg`, extracting them from) actual header bits is the
/// caller's responsibility. Fails with [`RsError::InvalidSymbol`] if any
/// entry of `msg` is outside `0..=15`.
pub fn encode(rs: &mut ReedSolomon, msg: &[u8; MESSAGE_LENGTH]) -> Result<[u8; NUM_ROOTS], RsError> {
    let mut encoded = [0u8; CODEWORD_LENGTH];
    rs.encode(msg, &mut encoded)?;
    let mut parity = [0u8; NUM_ROOTS];
    parity.copy_from_slice(&encoded[MESSAGE_LENGTH..]);
    Ok(parity)
}

/// Recovers the original 6 message symbols from a possibly-corrupted
/// codeword (6 message symbols followed by 4 parity symbols, one nibble
/// per byte), correcting up to `E = 2` symbol errors.
pub fn decode(
    rs: &mut ReedSolomon,
    codeword: &[u8; CODEWORD_LENGTH],
) -> Result<[u8; MESSAGE_LENGTH], RsError> {
    let mut msg = [0u8; MESSAGE_LENGTH];
    rs.decode(codeword, &mut msg)?;
    Ok(msg)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn to_codeword(msg: &[u8; MESSAGE_LENGTH], parity: &[u8; NUM_ROOTS]) -> [u8; CODEWORD_LENGTH] {
        let mut codeword = [0u8; CODEWORD_LENGTH];
        codeword[..MESSAGE_LENGTH].copy_from_slice(msg);
        codeword[MESSAGE_LENGTH..].copy_from_slice(parity);
        codeword
    }

    /// The standard's own worked example (CCSDS 732.0-B-4 SS4.1.2.6.5c):
    /// `g(x) = x^4 + alpha^3*x^3 + alpha*x^2 + alpha^3*x + 1`, with
    /// `alpha^3 = 0b1000 = 8` and `alpha = 0b0010 = 2`. This is the
    /// strongest conformance check available without a second
    /// independent implementation to compare against: if this matches,
    /// the field, the root convention, and the generator-from-roots
    /// construction are all simultaneously correct for this code.
    #[test]
    fn generator_polynomial_matches_the_published_standard() {
        let rs = codec();
        assert_eq!(rs.generator(), &[1, 8, 2, 8, 1]);
    }

    #[test]
    fn construction_parameters() {
        let rs = codec();
        assert_eq!(rs.block_length(), 15); // natural (unshortened) GF(16) block length
        assert_eq!(rs.min_distance(), NUM_ROOTS);
        assert_eq!(rs.message_length(), 11); // natural (15,11) code, shortened here to (10,6)
        assert_eq!(rs.generator_roots().len(), NUM_ROOTS);
    }

    #[test]
    fn round_trips_with_no_errors() {
        let mut rs = codec();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let parity = encode(&mut rs, &msg).unwrap();
        let codeword = to_codeword(&msg, &parity);

        assert_eq!(decode(&mut rs, &codeword).unwrap(), msg);
    }

    #[test]
    fn corrects_up_to_e_2_symbol_errors() {
        let mut rs = codec();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let parity = encode(&mut rs, &msg).unwrap();
        let mut codeword = to_codeword(&msg, &parity);

        // Corrupt 2 symbols (E = 2, this code's guaranteed correction
        // capacity), keeping values in the valid 0..=15 nibble range.
        codeword[1] ^= 0x0a;
        codeword[8] ^= 0x03;

        assert_eq!(decode(&mut rs, &codeword).unwrap(), msg);
    }

    #[test]
    fn detects_corruption_beyond_e_2_symbol_errors() {
        let mut rs = codec();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let parity = encode(&mut rs, &msg).unwrap();
        let mut codeword = to_codeword(&msg, &parity);

        for (i, byte) in codeword.iter_mut().enumerate().take(5) {
            *byte = (*byte ^ (i as u8 + 1)) & 0x0f;
        }

        assert_eq!(decode(&mut rs, &codeword), Err(RsError::TooManyErrors));
    }

    #[test]
    fn rejects_out_of_range_symbol() {
        let mut rs = codec();
        let msg = [16u8, 0, 0, 0, 0, 0]; // 16 is out of range (valid: 0..=15)
        assert_eq!(encode(&mut rs, &msg), Err(RsError::InvalidSymbol));
    }
}
