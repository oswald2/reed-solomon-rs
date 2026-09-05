//! The Reed-Solomon codec itself: construction, systematic encode, and
//! decode. Construction and encode port
//! `reed_solomon_build_generator`/`correct_reed_solomon_create` (from
//! `src/reed-solomon/reed-solomon.c`) and `correct_reed_solomon_encode`
//! (from `src/reed-solomon/encode.c`); decode ports
//! `correct_reed_solomon_decode` (from `src/reed-solomon/decode.c`),
//! with its three main stages split into the [`berlekamp_massey`],
//! [`chien`], and [`forney`] submodules.
//!
//! Decoding with erasures (`correct_reed_solomon_decode_with_erasures`)
//! is a later phase (see `PORTING_PLAN.md`).

mod berlekamp_massey;
mod chien;
mod forney;

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

    first_consecutive_root: u8,
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

    // Steady-state scratch buffers and precomputed tables for decode,
    // likewise sized once here so `decode` never allocates. Comments
    // give each buffer's length; `min_distance`/`block_length`/
    // `field_size` refer to this codec's own values throughout.
    received_polynomial: Vec<u8>, // len == block_length
    syndromes: Vec<u8>,           // len == min_distance
    error_locator: Vec<u8>,       // len == min_distance + 1
    last_error_locator: Vec<u8>,  // len == min_distance + 1; berlekamp_massey scratch
    error_locator_log: Vec<u8>,   // len == min_distance + 1
    error_roots: Vec<u8>,         // len == min_distance
    error_locations: Vec<u8>,     // len == min_distance
    error_evaluator: Vec<u8>,     // len == min_distance; forney scratch
    error_locator_derivative: Vec<u8>, // len == min_distance; forney scratch
    error_vals: Vec<u8>,          // len == min_distance

    // Erasure-decoding-only scratch (see `decode_with_erasures`).
    erasure_locator: Vec<u8>,     // len == min_distance + 1
    /// The combined locator (erasures and any additional errors found
    /// beyond them), `erasure_locator * error_locator`. Kept as a
    /// separate buffer from `error_locator` rather than reusing it in
    /// place (as the C implementation's pointer-swapping does), since
    /// both are needed simultaneously and this is far easier to follow.
    combined_locator: Vec<u8>,    // len == min_distance + 1
    modified_syndromes: Vec<u8>,  // len == min_distance
    /// A snapshot of the true syndromes, taken before they're
    /// temporarily overwritten (shifted by the modified-syndrome
    /// technique) to run Berlekamp-Massey on the erasure-adjusted
    /// sequence; restored before Forney's algorithm runs, which needs
    /// the real syndromes.
    syndrome_snapshot: Vec<u8>,   // len == min_distance
    /// Ping-pong scratch for `polynomial::init_from_roots`, used to
    /// build `erasure_locator` from the caller-supplied erasure
    /// locations without allocating.
    init_from_roots_scratch: [Vec<u8>; 2], // each len == min_distance + 1

    /// `generator_root_exp[i]` holds the successive log-powers of
    /// `generator_roots[i]`, `block_length` of them: used to evaluate
    /// the received polynomial at each generator root (i.e. compute
    /// syndromes) without recomputing those powers from scratch.
    generator_root_exp: Vec<Vec<u8>>, // min_distance entries, each len == block_length
    /// `element_exp[e]` holds the successive log-powers of field element
    /// `e`, `min_distance` of them: used by Chien search and Forney's
    /// algorithm, both of which need this for every field element, not
    /// just the generator's roots.
    element_exp: Vec<Vec<u8>>, // field_size entries, each len == min_distance
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

        // Precompute the successive log-powers of each generator root
        // (for syndrome calculation) and of every field element (for
        // Chien search and Forney's algorithm), so decode() itself never
        // needs to build these on the fly.
        let generator_root_exp: Vec<Vec<u8>> = generator_roots
            .iter()
            .map(|&root| {
                let mut lut = vec![0u8; block_length];
                polynomial::build_exp_lut(&field, root, &mut lut);
                lut
            })
            .collect();
        let element_exp: Vec<Vec<u8>> = (0..field.field_size())
            .map(|e| {
                let mut lut = vec![0u8; min_distance];
                polynomial::build_exp_lut(&field, e as u8, &mut lut);
                lut
            })
            .collect();

        Ok(ReedSolomon {
            encoded_polynomial: vec![0u8; block_length],
            encoded_remainder: vec![0u8; block_length],
            received_polynomial: vec![0u8; block_length],
            syndromes: vec![0u8; min_distance],
            error_locator: vec![0u8; min_distance + 1],
            last_error_locator: vec![0u8; min_distance + 1],
            error_locator_log: vec![0u8; min_distance + 1],
            error_roots: vec![0u8; min_distance],
            error_locations: vec![0u8; min_distance],
            error_evaluator: vec![0u8; min_distance],
            error_locator_derivative: vec![0u8; min_distance],
            error_vals: vec![0u8; min_distance],
            erasure_locator: vec![0u8; min_distance + 1],
            combined_locator: vec![0u8; min_distance + 1],
            modified_syndromes: vec![0u8; min_distance],
            syndrome_snapshot: vec![0u8; min_distance],
            init_from_roots_scratch: [vec![0u8; min_distance + 1], vec![0u8; min_distance + 1]],
            generator_root_exp,
            element_exp,
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

    /// The generator polynomial itself, `product(x + generator_roots[i])`,
    /// coefficients ordered lowest to highest degree. Useful mainly for
    /// checking a construction against a spec's own published generator
    /// polynomial (as `ccsds::fhec`'s tests do against CCSDS 732.0-B-4).
    pub fn generator(&self) -> &[u8] {
        &self.generator
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

    /// Finds and corrects errors in `encoded` (a full or shortened
    /// codeword produced by [`ReedSolomon::encode`]), writing the
    /// recovered message into `msg`.
    ///
    /// `encoded.len()` must be in `min_distance()..=block_length()`,
    /// and `msg` must have room for the resulting message length,
    /// `encoded.len() - min_distance()`.
    ///
    /// Can recover from as many as `min_distance() / 2` corrupted
    /// symbols at unknown locations, and returns
    /// `Err(RsError::TooManyErrors)` if there were more than that --
    /// though, as in the C implementation, it's possible (if unlikely)
    /// for a sufficiently corrupted block to be mistaken for a
    /// different, valid codeword instead of being detected as
    /// uncorrectable at all.
    ///
    /// Returns the number of bytes written to `msg`,
    /// `encoded.len() - min_distance()`.
    pub fn decode(&mut self, encoded: &[u8], msg: &mut [u8]) -> Result<usize, RsError> {
        if encoded.len() > self.block_length {
            return Err(RsError::EncodedTooLong);
        }
        if encoded.len() < self.min_distance {
            return Err(RsError::EncodedTooShort);
        }
        let msg_length = encoded.len() - self.min_distance;
        if msg.len() < msg_length {
            return Err(RsError::BufferTooSmall);
        }

        // Undo encode()'s high-to-low reversal, and zero-fill the
        // (unshortened) padding region above the message -- the mirror
        // image of encode()'s own setup.
        for (dst, &src) in self.received_polynomial[..encoded.len()]
            .iter_mut()
            .zip(encoded.iter().rev())
        {
            if src > self.field.largest_element() {
                return Err(RsError::InvalidSymbol);
            }
            *dst = src;
        }
        for slot in self.received_polynomial[encoded.len()..self.block_length].iter_mut() {
            *slot = 0;
        }

        let all_zero = find_syndromes(
            &self.field,
            &self.received_polynomial,
            &self.generator_root_exp,
            &mut self.syndromes,
        );
        if all_zero {
            // All syndromes are 0, so the received block is already a
            // valid codeword: no error occurred (or, vanishingly
            // unlikely, one did but happened to land exactly on another
            // valid codeword). Nothing to correct.
            for (dst, &src) in msg[..msg_length]
                .iter_mut()
                .zip(self.received_polynomial[..encoded.len()].iter().rev())
            {
                *dst = src;
            }
            return Ok(msg_length);
        }

        let order = berlekamp_massey::find_error_locator(
            &self.field,
            &self.syndromes,
            0,
            &mut self.error_locator,
            &mut self.last_error_locator,
        );

        // Log form of the locator, for the Chien search below (see
        // polynomial::eval_log_lut's docs for why 0 safely doubles as
        // "no term here" in this form).
        for i in 0..=order {
            self.error_locator_log[i] = self.field.log_table()[self.error_locator[i] as usize];
        }

        if !chien::factorize_error_locator(
            &self.field,
            0,
            &self.error_locator_log[..=order],
            &mut self.error_roots,
            &self.element_exp,
        ) {
            // Berlekamp-Massey built a locator that's consistent with
            // the syndromes but doesn't fully factor over this field:
            // there were too many errors to recover from.
            return Err(RsError::TooManyErrors);
        }

        chien::find_error_locations(
            &self.field,
            self.generator_root_gap,
            &self.error_roots[..order],
            &mut self.error_locations[..order],
        );

        forney::find_error_values(
            &self.field,
            &self.error_locator[..=order],
            &self.syndromes,
            &self.error_roots[..order],
            self.first_consecutive_root,
            &self.element_exp,
            &mut self.error_evaluator,
            &mut self.error_locator_derivative[..order],
            &mut self.error_vals[..order],
        );

        for i in 0..order {
            let location = self.error_locations[i] as usize;
            self.received_polynomial[location] =
                self.field.sub(self.received_polynomial[location], self.error_vals[i]);
        }

        for (dst, &src) in msg[..msg_length]
            .iter_mut()
            .zip(self.received_polynomial[..encoded.len()].iter().rev())
        {
            *dst = src;
        }
        Ok(msg_length)
    }

    /// Like [`ReedSolomon::decode`], but additionally accepts
    /// `erasure_locations`: byte indices into `encoded` that the caller
    /// already suspects are corrupted (e.g. flagged by a demodulator's
    /// confidence signal). Knowing *where* some errors are, rather than
    /// having to find them, buys more total correction capacity: this
    /// can recover from any mix of erasures and (unlocated) errors
    /// satisfying `2 * num_errors + erasure_locations.len() <
    /// min_distance()`, versus plain `decode`'s `2 * num_errors <
    /// min_distance()`.
    ///
    /// If `erasure_locations` is empty, this is exactly
    /// [`ReedSolomon::decode`]. Otherwise, `erasure_locations.len()`
    /// must not exceed `min_distance()`, and every entry must be a valid
    /// index into `encoded` (`< encoded.len()`).
    ///
    /// Returns the number of bytes written to `msg`,
    /// `encoded.len() - min_distance()`, same as `decode`.
    pub fn decode_with_erasures(
        &mut self,
        encoded: &[u8],
        erasure_locations: &[u8],
        msg: &mut [u8],
    ) -> Result<usize, RsError> {
        if erasure_locations.is_empty() {
            return self.decode(encoded, msg);
        }
        if encoded.len() > self.block_length {
            return Err(RsError::EncodedTooLong);
        }
        if encoded.len() < self.min_distance {
            return Err(RsError::EncodedTooShort);
        }
        if erasure_locations.len() > self.min_distance {
            return Err(RsError::TooManyErasures);
        }
        let msg_length = encoded.len() - self.min_distance;
        if msg.len() < msg_length {
            return Err(RsError::BufferTooSmall);
        }
        let erasure_length = erasure_locations.len();

        // Same setup as decode(): undo the high-to-low reversal, and
        // zero-fill the (unshortened) padding region.
        for (dst, &src) in self.received_polynomial[..encoded.len()]
            .iter_mut()
            .zip(encoded.iter().rev())
        {
            if src > self.field.largest_element() {
                return Err(RsError::InvalidSymbol);
            }
            *dst = src;
        }
        for slot in self.received_polynomial[encoded.len()..self.block_length].iter_mut() {
            *slot = 0;
        }

        // Map each erasure's byte index (in encoded's high-to-low byte
        // order) to the low-to-high polynomial coefficient index used
        // internally, then convert those positions into error roots and
        // multiply them out into the erasure locator polynomial.
        for (i, &loc) in erasure_locations.iter().enumerate() {
            if loc as usize >= encoded.len() {
                return Err(RsError::InvalidErasureLocation);
            }
            self.error_locations[i] = (encoded.len() - 1 - loc as usize) as u8;
        }
        chien::find_error_roots_from_locations(
            &self.field,
            self.generator_root_gap,
            &self.error_locations[..erasure_length],
            &mut self.error_roots[..erasure_length],
        );
        polynomial::init_from_roots(
            &self.field,
            &self.error_roots[..erasure_length],
            &mut self.erasure_locator[..=erasure_length],
            &mut self.init_from_roots_scratch,
        );

        let all_zero = find_syndromes(
            &self.field,
            &self.received_polynomial,
            &self.generator_root_exp,
            &mut self.syndromes,
        );
        if all_zero {
            // The received block is already a valid codeword, whatever
            // was flagged as erased notwithstanding.
            for (dst, &src) in msg[..msg_length]
                .iter_mut()
                .zip(self.received_polynomial[..encoded.len()].iter().rev())
            {
                *dst = src;
            }
            return Ok(msg_length);
        }

        // The modified-syndrome technique: multiplying the syndrome
        // polynomial by the (already-known) erasure locator produces a
        // sequence whose first erasure_length terms are eliminated,
        // leaving min_distance - erasure_length usable terms that
        // Berlekamp-Massey can run on to find any *additional* errors
        // beyond the known erasures. Since this overwrites `syndromes`
        // in place, save the real syndromes first -- Forney's algorithm
        // needs those, not the modified ones.
        self.syndrome_snapshot.copy_from_slice(&self.syndromes);
        polynomial::mul(
            &self.field,
            &self.erasure_locator[..=erasure_length],
            &self.syndromes,
            &mut self.modified_syndromes,
        );
        let remaining = self.min_distance - erasure_length;
        for i in 0..remaining {
            self.syndromes[i] = self.modified_syndromes[erasure_length + i];
        }

        let order = berlekamp_massey::find_error_locator(
            &self.field,
            &self.syndromes,
            erasure_length,
            &mut self.error_locator,
            &mut self.last_error_locator,
        );

        for i in 0..=order {
            self.error_locator_log[i] = self.field.log_table()[self.error_locator[i] as usize];
        }

        if !chien::factorize_error_locator(
            &self.field,
            erasure_length,
            &self.error_locator_log[..=order],
            &mut self.error_roots,
            &self.element_exp,
        ) {
            // Consistent with too-many-errors detection in decode():
            // the found root count didn't match the locator's degree.
            return Err(RsError::TooManyErrors);
        }

        // The full error locator -- covering both the known erasures and
        // whatever additional errors were just found -- is the product
        // of the two.
        let combined_order = erasure_length + order;
        polynomial::mul(
            &self.field,
            &self.erasure_locator[..=erasure_length],
            &self.error_locator[..=order],
            &mut self.combined_locator[..=combined_order],
        );

        chien::find_error_locations(
            &self.field,
            self.generator_root_gap,
            &self.error_roots[..combined_order],
            &mut self.error_locations[..combined_order],
        );

        // Restore the real syndromes before Forney's algorithm, which
        // needs them (not the modified ones used above).
        self.syndromes.copy_from_slice(&self.syndrome_snapshot);

        forney::find_error_values(
            &self.field,
            &self.combined_locator[..=combined_order],
            &self.syndromes,
            &self.error_roots[..combined_order],
            self.first_consecutive_root,
            &self.element_exp,
            &mut self.error_evaluator,
            &mut self.error_locator_derivative[..combined_order],
            &mut self.error_vals[..combined_order],
        );

        for i in 0..combined_order {
            let location = self.error_locations[i] as usize;
            self.received_polynomial[location] =
                self.field.sub(self.received_polynomial[location], self.error_vals[i]);
        }

        for (dst, &src) in msg[..msg_length]
            .iter_mut()
            .zip(self.received_polynomial[..encoded.len()].iter().rev())
        {
            *dst = src;
        }
        Ok(msg_length)
    }
}

/// Evaluates `received` at each of the generator's roots (i.e. computes
/// the syndromes): because a valid codeword is, by construction, a
/// multiple of the generator polynomial, it evaluates to 0 at every one
/// of those roots, so any nonzero syndrome directly reveals the error
/// polynomial's value there. Returns whether every syndrome was 0 (no
/// detected error). Ported from `reed_solomon_find_syndromes`.
fn find_syndromes(
    field: &GaloisField,
    received: &[u8],
    generator_root_exp: &[Vec<u8>],
    syndromes: &mut [u8],
) -> bool {
    let mut all_zero = true;
    for (syndrome, root_exp) in syndromes.iter_mut().zip(generator_root_exp.iter()) {
        let eval = polynomial::eval_lut(field, received, root_exp);
        if eval != 0 {
            all_zero = false;
        }
        *syndrome = eval;
    }
    all_zero
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
        // Shaped like the motivating case (quiet/libcorrect#17): a
        // GF(16) code shortened well below its natural message_length,
        // the same way CCSDS AOS FHEC's RS(10,6) shortens its natural
        // (15,11) code down from 11-symbol messages to 6 (root
        // convention doesn't matter for this particular check, so this
        // uses 1/1 rather than CCSDS's actual 6/1 -- see
        // src/ccsds/fhec.rs for the spec-exact preset).
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        assert_eq!(rs.message_length(), 11);
        let msg = [1u8, 2, 3, 4, 5, 6];
        let mut out = [0u8; 10];
        let written = rs.encode(&msg, &mut out).unwrap();
        assert_eq!(written, 10, "6-symbol message + 4 parity symbols == 10, not block_length() (15)");
    }

    #[test]
    fn ccsds_aos_fhec_shortened_rs_10_6_produces_a_valid_codeword() {
        // CCSDS 732.0-B-4 SS4.1.2.6.5: shortened RS(10,6) over GF(16),
        // primitive polynomial x^4+x+1 (0x13), generator roots
        // alpha^6..alpha^9 (first_consecutive_root=6, generator_root_gap=1).
        // See src/ccsds/fhec.rs for the spec-verified preset and its own
        // conformance test against the standard's published generator
        // polynomial.
        let mut rs = ReedSolomon::new(POLY_GF16, 6, 1, 4).unwrap();
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

    #[test]
    fn decode_with_no_errors_returns_the_original_message() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();

        let mut recovered = [0u8; 6];
        let n = rs.decode(&encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn ccsds_aos_fhec_round_trips_through_encode_and_decode() {
        // The quiet/libcorrect#17 case itself, with the actual CCSDS
        // 732.0-B-4 SS4.1.2.6.5 parameters: GF(16), generator roots
        // alpha^6..alpha^9, a message shortened to 6 symbols (RS(10,6)).
        let mut rs = ReedSolomon::new(POLY_GF16, 6, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();

        // Corrupt 2 symbols -- min_distance/2, this code's guaranteed
        // correction capacity.
        encoded[1] ^= 0x0a;
        encoded[8] ^= 0x03;

        let mut recovered = [0u8; 6];
        let n = rs.decode(&encoded, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn decode_detects_corruption_beyond_correction_capacity() {
        // min_distance = 4 guarantees correction of at most 2 errors;
        // corrupt 5 of the 10 codeword bytes (well beyond capacity) and
        // expect this to be detected rather than silently "corrected"
        // into the wrong message.
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();
        for (i, byte) in encoded.iter_mut().enumerate().take(5) {
            *byte = (*byte ^ (i as u8 + 1)) & 0x0f;
        }
        let mut recovered = [0u8; 6];
        assert_eq!(
            rs.decode(&encoded, &mut recovered),
            Err(RsError::TooManyErrors)
        );
    }

    /// Corrupts up to `max_errors` distinct positions in `encoded`,
    /// picking positions from `raw_positions` (reduced mod
    /// `encoded.len()`, deduplicated) and pairing each with the value at
    /// the same index in `values`. Shared by the GF(16) and GF(256)
    /// round-trip properties below.
    fn corrupt(encoded: &mut [u8], max_errors: usize, raw_positions: &[u16], values: &[u8]) {
        let mut positions: Vec<usize> = Vec::new();
        for &p in raw_positions {
            if positions.len() >= max_errors || positions.len() >= values.len() {
                break;
            }
            let pos = (p as usize) % encoded.len();
            if !positions.contains(&pos) {
                positions.push(pos);
            }
        }
        for (i, &pos) in positions.iter().enumerate() {
            encoded[pos] ^= values[i];
        }
    }

    proptest! {
        /// Mirrors tests/rs_tester.c's test_rs_errors: encode a random
        /// message, corrupt up to this code's guaranteed correction
        /// capacity (min_distance / 2) at random positions, and check
        /// decode recovers the exact original message.
        #[test]
        fn decode_recovers_from_errors_within_capacity_gf16(
            min_distance in 2usize..14,
            msg in proptest::collection::vec(0u8..=15u8, 0..13),
            raw_positions in proptest::collection::vec(any::<u16>(), 0..8),
            values in proptest::collection::vec(1u8..=15u8, 0..8),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, min_distance).unwrap();
            prop_assume!(msg.len() <= rs.message_length());
            let max_errors = rs.min_distance() / 2;
            prop_assume!(max_errors > 0);

            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            rs.encode(&msg, &mut encoded).unwrap();
            corrupt(&mut encoded, max_errors, &raw_positions, &values);

            let mut recovered = vec![0u8; msg.len()];
            let n = rs.decode(&encoded, &mut recovered).unwrap();
            prop_assert_eq!(n, msg.len());
            prop_assert_eq!(recovered, msg);
        }

        #[test]
        fn decode_recovers_from_errors_within_capacity_gf256(
            msg in proptest::collection::vec(any::<u8>(), 0..223),
            raw_positions in proptest::collection::vec(any::<u16>(), 0..16),
            values in proptest::collection::vec(1u8..=255u8, 0..16),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF256, 1, 1, 32).unwrap();
            let max_errors = rs.min_distance() / 2;

            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            rs.encode(&msg, &mut encoded).unwrap();
            corrupt(&mut encoded, max_errors, &raw_positions, &values);

            let mut recovered = vec![0u8; msg.len()];
            let n = rs.decode(&encoded, &mut recovered).unwrap();
            prop_assert_eq!(n, msg.len());
            prop_assert_eq!(recovered, msg);
        }
    }

    #[test]
    fn decode_with_erasures_with_no_erasures_matches_decode() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();
        encoded[2] ^= 0x05;

        let mut recovered = [0u8; 6];
        let n = rs.decode_with_erasures(&encoded, &[], &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn decode_with_erasures_rejects_too_many_erasures() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let encoded = [0u8; 10];
        let erasures = [0u8, 1, 2, 3, 4]; // 5 > min_distance (4)
        let mut recovered = [0u8; 6];
        assert_eq!(
            rs.decode_with_erasures(&encoded, &erasures, &mut recovered),
            Err(RsError::TooManyErasures)
        );
    }

    #[test]
    fn decode_with_erasures_rejects_out_of_range_location() {
        let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();

        let mut recovered = [0u8; 6];
        assert_eq!(
            rs.decode_with_erasures(&encoded, &[10], &mut recovered), // encoded.len() == 10
            Err(RsError::InvalidErasureLocation)
        );
    }

    #[test]
    fn ccsds_aos_fhec_recovers_via_pure_erasures() {
        // min_distance = 4 allows up to 3 (min_distance - 1) erasures
        // when their locations are fully known, more than the 2 errors
        // decode() alone could correct blind. Uses the real CCSDS
        // 732.0-B-4 SS4.1.2.6.5 generator roots (alpha^6..alpha^9).
        let mut rs = ReedSolomon::new(POLY_GF16, 6, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();

        encoded[0] ^= 0x0a;
        encoded[4] ^= 0x03;
        encoded[9] ^= 0x0c;
        let erasures = [0u8, 4, 9];

        let mut recovered = [0u8; 6];
        let n = rs.decode_with_erasures(&encoded, &erasures, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    #[test]
    fn ccsds_aos_fhec_recovers_via_mixed_errors_and_erasures() {
        // 2*num_errors + num_erasures < min_distance (4): 1 unknown
        // error plus 1 known erasure fits (2*1 + 1 == 3 < 4).
        let mut rs = ReedSolomon::new(POLY_GF16, 6, 1, 4).unwrap();
        let msg = [3u8, 7, 1, 15, 0, 9];
        let mut encoded = [0u8; 10];
        rs.encode(&msg, &mut encoded).unwrap();

        encoded[3] ^= 0x0a; // known erasure
        encoded[7] ^= 0x05; // unlocated error
        let erasures = [3u8];

        let mut recovered = [0u8; 6];
        let n = rs.decode_with_erasures(&encoded, &erasures, &mut recovered).unwrap();
        assert_eq!(n, msg.len());
        assert_eq!(recovered, msg);
    }

    /// Corrupts up to `num_erasures + num_errors` distinct positions in
    /// `encoded` (same position-selection scheme as `corrupt`), and
    /// returns the byte indices of the first `num_erasures` of them --
    /// the subset a caller would already know about and pass to
    /// `decode_with_erasures`. The rest are corrupted the same way but
    /// their locations aren't returned, standing in for unlocated
    /// errors. If fewer than `num_erasures + num_errors` distinct
    /// positions were available from `raw_positions`, fewer total
    /// corruptions happen (which can only make recovery easier); callers
    /// should assert the returned erasure count actually matches
    /// `num_erasures` before relying on the error count too.
    fn corrupt_with_erasures(
        encoded: &mut [u8],
        num_erasures: usize,
        num_errors: usize,
        raw_positions: &[u16],
        values: &[u8],
    ) -> Vec<u8> {
        let total = num_erasures + num_errors;
        let mut positions: Vec<usize> = Vec::new();
        for &p in raw_positions {
            if positions.len() >= total || positions.len() >= values.len() {
                break;
            }
            let pos = (p as usize) % encoded.len();
            if !positions.contains(&pos) {
                positions.push(pos);
            }
        }
        for (i, &pos) in positions.iter().enumerate() {
            encoded[pos] ^= values[i];
        }
        positions.iter().take(num_erasures).map(|&p| p as u8).collect()
    }

    proptest! {
        /// Mirrors tests/rs_tester.c's combined errors+erasures case:
        /// erasures alone, up to the provable maximum (min_distance - 1).
        #[test]
        fn decode_with_erasures_recovers_pure_erasures_gf16(
            min_distance in 2usize..14,
            msg in proptest::collection::vec(0u8..=15u8, 0..13),
            raw_positions in proptest::collection::vec(any::<u16>(), 0..14),
            values in proptest::collection::vec(1u8..=15u8, 0..14),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, min_distance).unwrap();
            prop_assume!(msg.len() <= rs.message_length());
            let num_erasures = min_distance - 1;
            prop_assume!(num_erasures > 0);

            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            rs.encode(&msg, &mut encoded).unwrap();
            let erasure_locations = corrupt_with_erasures(&mut encoded, num_erasures, 0, &raw_positions, &values);
            prop_assume!(erasure_locations.len() == num_erasures);

            let mut recovered = vec![0u8; msg.len()];
            let n = rs.decode_with_erasures(&encoded, &erasure_locations, &mut recovered).unwrap();
            prop_assert_eq!(n, msg.len());
            prop_assert_eq!(recovered, msg);
        }

        /// Mirrors tests/rs_tester.c's combined errors+erasures case: a
        /// roughly even split of the min_distance budget between known
        /// erasures and unlocated errors.
        #[test]
        fn decode_with_erasures_recovers_mixed_errors_and_erasures_gf16(
            min_distance in 3usize..14,
            msg in proptest::collection::vec(0u8..=15u8, 0..13),
            raw_positions in proptest::collection::vec(any::<u16>(), 0..14),
            values in proptest::collection::vec(1u8..=15u8, 0..14),
        ) {
            let mut rs = ReedSolomon::new(POLY_GF16, 1, 1, min_distance).unwrap();
            prop_assume!(msg.len() <= rs.message_length());
            let num_erasures = min_distance / 2;
            let num_errors = (min_distance - 1 - num_erasures) / 2;
            prop_assume!(num_errors > 0);

            let mut encoded = vec![0u8; msg.len() + rs.min_distance()];
            rs.encode(&msg, &mut encoded).unwrap();
            let erasure_locations = corrupt_with_erasures(&mut encoded, num_erasures, num_errors, &raw_positions, &values);
            prop_assume!(erasure_locations.len() == num_erasures);

            let mut recovered = vec![0u8; msg.len()];
            let n = rs.decode_with_erasures(&encoded, &erasure_locations, &mut recovered).unwrap();
            prop_assert_eq!(n, msg.len());
            prop_assert_eq!(recovered, msg);
        }
    }
}
