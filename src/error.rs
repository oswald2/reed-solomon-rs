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
        };
        f.write_str(msg)
    }
}

#[cfg(feature = "std")]
impl std::error::Error for RsError {}
