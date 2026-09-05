//! CCSDS-standardized Reed-Solomon codes, built on top of the generic
//! [`crate::rs::ReedSolomon`] engine.
//!
//! Currently covers:
//!
//! - [`fhec`]: CCSDS 732.0-B-4 (AOS Space Data Link Protocol) Frame
//!   Header Error Control, a shortened RS(10,6) code over GF(16). This
//!   is the code quiet/libcorrect#17 asked for, and the reason the
//!   `short_rs` branch this crate is ported from generalized the field
//!   width beyond GF(256) in the first place.
//!
//! Not yet covered: CCSDS 131.0-B (TM Synchronization and Channel
//! Coding)'s RS(255,223) code, its interleaving, and its dual-basis
//! symbol representation. See `PORTING_PLAN.md`.

pub mod fhec;
