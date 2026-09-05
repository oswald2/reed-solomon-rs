//! Reed-Solomon encoding/decoding over `GF(2^k)`, `k <= 8`.
//!
//! This is a Rust port of the Reed-Solomon implementation in
//! [libcorrect](https://github.com/quiet/libcorrect)'s `short_rs` branch,
//! generalized so the field width, primitive polynomial, generator roots,
//! and number of parity symbols are all runtime parameters. That
//! generality is what makes it possible to implement the various
//! Reed-Solomon codes CCSDS standardizes -- see the [`ccsds`] module.
//!
//! Ported so far: [`field`] (GF(2^k) construction and arithmetic).
//! See `PORTING_PLAN.md` in the repo root for the rest of the roadmap.
#![no_std]

extern crate alloc;

pub mod field;

// Phase 2+: polynomial operations, the RS codec itself, and CCSDS presets.
// Left undeclared until their contents exist, so an empty file doesn't
// silently compile as a real (but vacuous) module.
// pub mod polynomial;
// pub mod error;
// pub mod rs;
// pub mod ccsds;
