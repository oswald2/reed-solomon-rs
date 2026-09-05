//! Reed-Solomon encoding/decoding over `GF(2^k)`, `k <= 8`.
//!
//! This is a Rust port of the Reed-Solomon implementation in
//! [libcorrect](https://github.com/quiet/libcorrect)'s `short_rs` branch,
//! generalized so the field width, primitive polynomial, generator roots,
//! and number of parity symbols are all runtime parameters. That
//! generality is what makes it possible to implement the various
//! Reed-Solomon codes CCSDS standardizes -- see [`ccsds`].
//!
//! Ported: [`field`] (GF(2^k) construction and arithmetic), [`polynomial`]
//! (polynomial arithmetic over a field), and [`rs`] (codec construction,
//! systematic encode, and decode with and without erasures). See
//! `PORTING_PLAN.md` for what's still to come (CCSDS 131.0-B TM channel
//! coding, interleaving, dual-basis representation).
#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod ccsds;
pub mod error;
pub mod field;
pub mod polynomial;
pub mod rs;
