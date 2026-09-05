//! The Reed-Solomon codec itself: construction and systematic encode, port
//! of `reed_solomon_build_generator`/`correct_reed_solomon_create` (from
//! `src/reed-solomon/reed-solomon.c`) and `correct_reed_solomon_encode`
//! (from `src/reed-solomon/encode.c`).
//!
//! Decode (Berlekamp-Massey, Chien search, Forney, and the erasure path)
//! is a later phase; this file will likely become a `rs/` directory with
//! one submodule per decode stage once that lands, per `PORTING_PLAN.md`.

use alloc::vec;
use alloc::vec::Vec;

use crate::error::RsError;
use crate::field::GaloisField;
use crate::polynomial;

/// A Reed-Solomon encoder/decoder over `GF(2^k)`, `k <= 8`.
///
/// Parameterized exactly like the C `correct_reed_solomon_create`: the
/// field width comes from `primitive_polynomial`'s bit length, and
/// `first_consecutive_root`/`generator_root_gap`/`num_roots` pick which
/// code within that field this is. This generality is what covers every
/// Reed-Solomon code CCSDS defines, not just the usual GF(256)
/// RS(255,223): see the crate-level docs and `PORTING_PLAN.md`.
#[derive(Debug)]
pub struct ReedSolomon {
    field: GaloisField,

    /// `field.largest_element()` as a `usize`: the natural (unshortened)
    /// codeword length for this field.
    block_length: usize,
    /// `block_length - min_distance`: the natural (unshortened) message
    /// length. Shorter messages are supported directly by `encode`
    /// (see its docs), which is how shortened codes like CCSDS AOS
    /// FHEC's RS(10,6) are built on top of the natural (15,11) code.
    message_length: usize,
    /// Number of parity symbols (`2t`, in the usual `t`-error-correcting
    /// terminology), a.k.a. `num_roots`.
    min_distance: usize,

    // Not read anywhere yet -- decode (phase 4+) needs both of these
    // (generator_root_gap to map error roots back to locations, and
    // first_consecutive_root in the Forney error-value calculation).
    #[allow(dead_code)]
    first_consecutive_root: u8,
    #[allow(dead_code)]
    generator_root_gap: u8,
    /// The `min_distance` roots of the generator polynomial:
    /// `alpha^(generator_root_gap * (i + first_consecutive_root))` for
    /// `i` in `0..min_distance`.
    generator_roots: Vec<u8>,
    /// The generator polynomial itself, `product(x + generator_roots[i])`.
    generator: Vec<u8>,

    // Steady-state scratch buffers for encode, sized once here so
    // `encode` itself never allocates.
    encoded_polynomial: Vec<u8>, // len == block_length
    encoded_remainder: Vec<u8>,  // len == block_length; poly_mod's scratch/output
}

impl ReedSolomon {
    /// Builds a Reed-Solomon codec.
    ///
    /// `primitive_polynomial` selects the field `GF(2^k)` (see
    /// [`GaloisField::new`]); `first_consecutive_root` and
    /// `generator_root_gap` select which consecutive roots of unity form
    /// the generator polynomial (the C header's docs note that `1` and
    /// `1` are sane defaults, though CCSDS-standardized codes may use
    /// other values -- see `PORTING_PLAN.md`'s note on verifying those
    /// against the actual blue books); `num_roots` is the number of
    /// parity symbols (`min_distance`), which must be at least `1` and
    /// less than the field's block length.
    pub fn new(
        primitive_polynomial: u16,
        first_consecutive_root: u8,
        generator_root_gap: u8,
        num_roots: usize,
    ) -> Result<Self, RsError> {
        let field = GaloisField::new(primitive_polynomial);
        let block_length = field.largest_element() as usize;

        if num_roots == 0 || num_roots >= block_length {
            return Err(RsError::InvalidParameters);
        }
        let min_distance = num_roots;
        let message_length = block_length - min_distance;

        let mut generator_roots = vec![0u8; min_distance];
        for (i, root) in generator_roots.iter_mut().enumerate() {
            let exponent = (generator_root_gap as usize * (i + first_consecutive_root as usize))
                % field.largest_element() as usize;
            *root = field.exp_table()[exponent];
        }
        let generator = polynomial::from_roots(&field, &generator_roots);

        Ok(ReedSolomon {
            encoded_polynomial: vec![0u8; block_length],
            encoded_remainder: vec![0u8; block_length],
            field,
            block_length,
            message_length,
            min_distance,
            first_consecutive_root,
            generator_root_gap,
            generator_roots,
            generator,
        })
    }

    /// The field this code is defined over.
    pub fn field(&self) -> &GaloisField {
        &self.field
    }

    /// The natural (unshortened) codeword length, `field.largest_element()`.
    pub fn block_length(&self) -> usize {
        self.block_length
    }

    /// The natural (unshortened) message length, `block_length() -
    /// min_distance()`. This is the longest message `encode` will
    /// accept; shorter messages are encoded as a shortened code (see
    /// [`ReedSolomon::encode`]).
    pub fn message_length(&self) -> usize {
        self.message_length
    }

    /// Number of parity symbols this code adds. Can repair as many as
    /// `min_distance() / 2` corrupted symbols (with unknown locations),
    /// or up to `min_distance()` symbols if all their locations are
    /// known (erasures) -- or some combination, per
    /// `2*num_errors + num_erasures < min_distance()`.
    pub fn min_distance(&self) -> usize {
        self.min_distance
    }

    /// The roots of the generator polynomial, in field-element form.
    pub fn generator_roots(&self) -> &[u8] {
        &self.generator_roots
    }

    /// Encodes `msg` (`msg.len() <= message_length()`) into `encoded`,
    /// which must have room for at least `msg.len() + min_distance()`
    /// bytes (the C implementation does not check this, and will write
    /// out of bounds if `encoded` is too small).
    ///
    /// If `msg` is shorter than `message_length()`, it's encoded as a
    /// *shortened* code: conceptually, the message is padded with
    /// leading zero symbols up to `message_length()` before encoding,
    /// but those padding symbols are never written to `encoded` (both
    /// sides just have to agree they're there). This is exactly the
    /// mechanism CCSDS's AOS Frame Header Error Control code uses to
    /// build a shortened RS(10,6) out of a GF(16) field whose natural
    /// code is (15,11) -- see `PORTING_PLAN.md`.
    ///
    /// Returns the number of bytes written to `encoded`,
    /// `msg.len() + min_distance()`. Note this differs from the C
    /// implementation, which always returns the fixed `block_length()`
    /// regardless of `msg.len()` -- that appears to be a bug (it
    /// contradicts the function's own documented contract, "returns the
    /// number of bytes written to encoded") that just happens not to
    /// matter for full-length (unshortened) messages, where
    /// `msg.len() + min_distance() == block_length()` anyway. It would
    /// matter a great deal for a shortened code, i.e. exactly the CCSDS
    /// AOS FHEC case this crate exists for, so this port returns the
    /// value the docs actually promise.
    pub fn encode(&mut self, msg: &[u8], encoded: &mut [u8]) -> Result<usize, RsError> {
        if msg.len() > self.message_length {
            return Err(RsError::MessageTooLong);
        }
        let written = msg.len() + self.min_distance;
        if encoded.len() < written {
            return Err(RsError::BufferTooSmall);
        }

        // The message goes from high order to low order (byte 0 is the
        // most significant symbol), but polynomial coefficients go low
        // to high, so we reverse on the way in and again on the way out.
        // A shortened code just means pad_length virtual leading-zero
        // symbols are never written into (or read out of) the buffer.
        let pad_length = self.message_length - msg.len();
        let block_len = self.block_length;

        // Equivalent to the C implementation's two memsets (one for the
        // padding zone above the message, one for the parity zone below
        // it): zero everything, then place the message in between.
        self.encoded_polynomial.fill(0);
        for (i, &b) in msg.iter().enumerate() {
            if b > self.field.largest_element() {
                return Err(RsError::InvalidSymbol);
            }
            self.encoded_polynomial[block_len - 1 - (i + pad_length)] = b;
        }

        polynomial::poly_mod(
            &self.field,
            &self.encoded_polynomial,
            &self.generator,
            &mut self.encoded_remainder,
        );

        // Return byte order to highest order to lowest order: the
        // message symbols live at polynomial indices
        // [min_distance, min_distance + msg.len()), highest order first,
        // so reverse that range on the way out.
        let msg_region = &self.encoded_polynomial[self.min_distance..self.min_distance + msg.len()];
        for (dst, &src) in encoded[..msg.len()].iter_mut().zip(msg_region.iter().rev()) {
            *dst = src;
        }
        let parity_region = self.encoded_remainder[..self.min_distance].iter().rev();
        for (dst, src) in encoded[msg.len()..written].iter_mut().zip(parity_region) {
            *dst = *src;
        }

        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    /// x^8 + x^7 + x^2 + x + 1, `correct_rs_primitive_polynomial_ccsds`
    /// in the C header: the primitive polynomial libcorrect's own tests
    /// use for GF(256) codes.
    const POLY_GF256: u16 = 0x187;
    /// x^4 + x + 1, GF(16): the field for CCSDS AOS Frame Header Error
    /// Control's shortened RS(10,6) code (quiet/libcorrect#17), and the
    /// same field libcorrect's own GF(16) test in `tests/reed-solomon.c`
    /// uses.
    const POLY_GF16: u16 = 0x13;

    /// Rebuilds the full `block_length`-symbol codeword polynomial (low
    /// order to high, matching `polynomial`'s convention) that `encode`
    /// must have internally produced, from its output plus the message
    /// length -- i.e. undoes the high-to-low reversal and re-inserts the
    /// (unshortened) padding zeros, without needing access to
    /// `ReedSolomon`'s private scratch buffers.
    fn reconstruct_codeword_poly(rs: &ReedSolomon, encoded: &[u8], msg_len: usize) -> Vec<u8> {
        let pad_length = rs.message_length() - msg_len;
        let block_len = rs.block_length();
        let mut poly = vec![0u8; block_len];
        for i in 0..msg_len {
            poly[block_len - 1 - (i + pad_length)] = encoded[i];
        }
        for i in 0..rs.min_distance() {
            poly[rs.min_distance() - 1 - i] = encoded[msg_len + i];
        }
        poly
    }

    /// A valid codeword is, by construction, a multiple of the generator
    /// polynomial, which means it must evaluate to 0 at every one of the
    /// generator's roots. This doesn't require a decoder to check --
    /// it's the same algebraic fact the decoder's syndrome calculation
    /// relies on (all syndromes are 0 exactly when there's no error) --
    /// so it's a strong end-to-end check of `encode` on its own.
    fn assert_is_valid_codeword(rs: &ReedSolomon, encoded: &[u8], msg_len: usize) {
        let poly = reconstruct_codeword_poly(rs, encoded, msg_len);
        for &root in rs.generator_roots() {
            assert_eq!(
                polynomial::eval(rs.field(), &poly, root),
                0,
                "codeword does not evaluate to 0 at generator root {root}"
            );
        }
    }

    #[test]
    fn construction_parameters_gf256() {
        let rs = ReedSolomon::new(POLY_GF256, 1, 1, 32).unwrap();
        assert_eq!(rs.block_length(), 255);
        assert_eq!(rs.min_distance(), 32);
        assert_eq!(rs.message_length(), 223);
        assert_eq!(rs.generator_roots().len(), 32);
    }

    #[test]
    fn construction_parameters_gf16() {
        // Matches tests/reed-solomon.c's GF(16) case exactly: block
        // length 15, min_distance 2, message_length 13.
        let rs = ReedSolomon::new(POLY_GF16, 1, 1, 2).unwrap();
        assert_eq!(rs.block_length(), 15);
        assert_eq!(rs.min_distance(), 2);
        assert_eq!(rs.message_length(), 13);
    }

    #[test]
    fn rejects_invalid_num_roots() {
        assert_eq!(
            ReedSolomon::new(POLY_GF16, 1, 1, 0).unwrap_err(),
            RsError::InvalidParameters
        );
        assert_eq!(
            ReedSolomon::new(POLY_GF16, 1, 1, 15).unwrap_err(),
            RsError::InvalidParameters
        );
        assert_eq!(
            ReedSolomon::new(POLY_GF16, 1, 1, 100).unwrap_err(),
            RsError::InvalidParameters
        );
    }

    #[test]
    fn encode_rejects_message_too_long() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 2).unwrap();
        let msg = [0u8; 14]; // message_length() is 13
        let mut out = [0u8; 15];
        assert_eq!(rs.encode(&msg, &mut out), Err(RsError::MessageTooLong));
    }

    #[test]
    fn encode_rejects_undersized_output_buffer() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 2).unwrap();
        let msg = [1u8, 2, 3];
        let mut out = [0u8; 4]; // needs 3 + min_distance(2) == 5
        assert_eq!(rs.encode(&msg, &mut out), Err(RsError::BufferTooSmall));
    }

    #[test]
    fn encode_rejects_out_of_range_symbol() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 2).unwrap();
        let msg = [16u8]; // 16 is out of range for GF(16) (valid: 0..=15)
        let mut out = [0u8; 3];
        assert_eq!(rs.encode(&msg, &mut out), Err(RsError::InvalidSymbol));
    }

    #[test]
    fn encode_returns_actual_bytes_written_for_a_shortened_code() {
        // The motivating case (quiet/libcorrect#17): a GF(16) code
        // shortened well below its natural message_length, as CCSDS AOS
        // FHEC's RS(10,6) shortens the natural (15,11) code down from
        // 11-symbol messages to 6.
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        assert_eq!(rs.message_length(), 11);
        let msg = [1u8, 2, 3, 4, 5, 6];
        let mut out = [0u8; 10];
        let written = rs.encode(&msg, &mut out).unwrap();
        assert_eq!(written, 10, "6-symbol message + 4 parity symbols == 10, not block_length() (15)");
    }

    #[test]
    fn ccsds_aos_fhec_shortened_rs_10_6_produces_a_valid_codeword() {
        // CCSDS 732.0-B AOS Frame Header Error Control: shortened
        // RS(10,6) over GF(16), primitive polynomial 0x13.
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        let written = rs.encode(&msg, &mut encoded).unwrap();
        assert_eq!(written, 10);
        assert_is_valid_codeword(&rs, &encoded, msg.len());
    }

    proptest! {
        #[test]
        fn encode_gf16_always_produces_a_valid_codeword(
            min_distance in 1usize..14,
            msg in proptest::collection::vec(0u8..=15u8, 0..13),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, min_distance).unwrap();
            // Only exercise messages that fit this min_distance's message_length.
            prop_assume!(msg.len() <= rs.message_length());

            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            let written = rs.encode(&msg, &mut encoded).unwrap();
            prop_assert_eq!(written, msg.len() + rs.min_distance());
            assert_is_valid_codeword(&rs, &encoded, msg.len());
        }

        #[test]
        fn encode_gf256_always_produces_a_valid_codeword(
            msg in proptest::collection::vec(any::<u8>(), 0..223),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF256, 1, 1, 32).unwrap();
            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            let written = rs.encode(&msg, &mut encoded).unwrap();
            prop_assert_eq!(written, msg.len() + rs.min_distance());
            assert_is_valid_codeword(&rs, &encoded, msg.len());
        }
    }
}
