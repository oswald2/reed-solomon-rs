//! CCSDS-standardized Reed-Solomon codes, built on top of the generic
//! [`crate::rs::ReedSolomon`] engine.
//!
//! - [`fhec`]: CCSDS 732.0-B-4 (AOS Space Data Link Protocol) Frame
//!   Header Error Control, a shortened RS(10,6) code over GF(16). This
//!   is the code quiet/libcorrect#17 asked for, and the reason the
//!   `short_rs` branch this crate is ported from generalized the field
//!   width beyond GF(256) in the first place.
//! - [`tm_channel_coding`]: CCSDS 131.0-B-5 (TM Synchronization and
//!   Channel Coding), the "standard" GF(256) RS(255,223)/RS(255,239)
//!   codes, plus symbol interleaving.
//! - [`dual_basis`]: the symbol-representation conversion
//!   `tm_channel_coding` requires for actual transmission.

pub mod dual_basis;
pub mod fhec;
pub mod tm_channel_coding;
