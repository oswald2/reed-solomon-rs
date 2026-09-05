//! Error type for Reed-Solomon operations.

use core::fmt;

/// Errors that can occur when constructing or using a
/// [`crate::rs::ReedSolomon`] codec.
///
/// Marked `#[non_exhaustive]` since later phases (decode, decode with
/// erasures) will add variants (e.g. "too many errors/erasures to
/// recover from") without that being a breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum RsError {
    /// `num_roots` passed to [`crate::rs::ReedSolomon::new`] was `0`, or
    /// was not less than the field's block length, so no valid code
    /// exists for these parameters. The C implementation doesn't check
    /// this at all: passing an out-of-range `num_roots` there computes
    /// `message_length` as an unsigned underflow (`block_length -
    /// num_roots` wrapping around) instead of failing.
    InvalidParameters,
    /// The message passed to `encode` is longer than this code's
    /// message length (`block_length() - min_distance()`).
    MessageTooLong,
    /// A message byte's value exceeds the field's largest element, i.e.
    /// it is not a valid symbol for this code's field width. This can
    /// only happen for fields smaller than GF(256), e.g. the GF(16)
    /// field used by the CCSDS AOS Frame Header Error Control code,
    /// where only byte values `0..=15` are valid symbols.
    InvalidSymbol,
    /// The output buffer passed to `encode`/`decode` is too small to
    /// hold the result. The C implementation doesn't check this either,
    /// and will write out of bounds if the caller under-allocates.
    BufferTooSmall,
    /// The block passed to `decode`/`decode_with_erasures` is longer
    /// than this code's `block_length()`.
    EncodedTooLong,
    /// The block passed to `decode`/`decode_with_erasures` is shorter
    /// than this code's `min_distance()`, so it can't even contain all
    /// the parity symbols, let alone any message. The C implementation
    /// doesn't check this, and computes the message length as an
    /// unsigned underflow instead of failing.
    EncodedTooShort,
    /// Decoding failed: there were more errors and/or erasures than this
    /// code's `min_distance()` can guarantee recovery from. As in the C
    /// implementation, it's possible but unlikely for this to go
    /// undetected and instead return an incorrectly "recovered" message.
    TooManyErrors,
    /// The number of erasures passed to `decode_with_erasures` exceeds
    /// this code's `min_distance()`, leaving no room for Berlekamp-Massey
    /// to run at all.
    TooManyErasures,
    /// One of the erasure locations passed to `decode_with_erasures` is
    /// not a valid index into the encoded block (i.e. is `>= encoded.len()`).
    /// The C implementation doesn't check this, and silently computes a
    /// nonsensical position via unsigned underflow instead of failing.
    InvalidErasureLocation,
}

impl fmt::Display for RsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let msg = match self {
            RsError::InvalidParameters => {
                "invalid parameters: num_roots must be in 1..block_length"
            }
            RsError::MessageTooLong => "message is longer than this code's message length",
            RsError::InvalidSymbol => "a message byte is not a valid symbol for this code's field",
            RsError::BufferTooSmall => "output buffer is too small to hold the result",
            RsError::EncodedTooLong => "encoded block is longer than this code's block length",
            RsError::EncodedTooShort => "encoded block is shorter than this code's min_distance",
            RsError::TooManyErrors => "too many errors to recover from",
            RsError::TooManyErasures => "too many erasures for this code's min_distance",
            RsError::InvalidErasureLocation => "an erasure location is out of range for the encoded block",
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RsError {}
