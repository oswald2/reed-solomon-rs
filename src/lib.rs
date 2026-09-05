//! Reed-Solomon encoding/decoding over `GF(2^k)`, `k <= 8`.
//!
//! This is a Rust port of the Reed-Solomon implementation in
//! [libcorrect](https://github.com/quiet/libcorrect)'s `short_rs` branch,
//! generalized so the field width, primitive polynomial, generator roots,
//! and number of parity symbols are all runtime parameters. That
//! generality is what makes it possible to implement the various
//! Reed-Solomon codes CCSDS standardizes (a `ccsds` module with presets
//! for those is a later phase -- see `PORTING_PLAN.md`).
//!
//! Ported so far: [`field`] (GF(2^k) construction and arithmetic),
//! [`polynomial`] (polynomial arithmetic over a field), and [`rs`]
//! (codec construction and systematic encode; decode is still to come).
#![no_std]

extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

pub mod error;
pub mod field;
pub mod polynomial;
pub mod rs;

// Phase 4+: decode (Berlekamp-Massey, Chien search, Forney, erasures)
// and CCSDS presets. Left undeclared until their contents exist, so an
// empty file doesn't silently compile as a real (but vacuous) module.
// pub mod ccsds;
