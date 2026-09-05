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
//! (polynomial arithmetic over a field), [`rs`] (codec construction,
//! systematic encode, and decode with and without erasures), and
//! [`ccsds`] (spec-verified presets: AOS Frame Header Error Control and
//! TM Synchronization and Channel Coding, including symbol interleaving
//! and the dual-basis representation the latter requires). See
//! `PORTING_PLAN.md` for the full history of what was verified against
//! which standard, and how.
#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod ccsds;
pub mod error;
pub mod field;
pub mod polynomial;
pub mod rs;
