# reed-solomon-rs

A `no_std` + `alloc` Rust port of the Reed-Solomon codec from
[libcorrect](https://github.com/quiet/libcorrect)'s `short_rs` branch,
generalized to `GF(2^k)` for `k <= 8`.

This generalization is what makes it possible to implement the Reed-Solomon
codes CCSDS standardizes, which use field sizes other than the usual
`GF(2^8)`:

- **CCSDS 732.0-B, AOS Space Data Link Protocol** — Frame Header Error
  Control: a shortened RS(10,6) code over `GF(2^4)`, primitive polynomial
  `0x13` (`x^4 + x + 1`). See
  [quiet/libcorrect#17](https://github.com/quiet/libcorrect/issues/17).
- **CCSDS 131.0-B, TM Synchronization and Channel Coding** — RS(255,223) or
  RS(255,239) over `GF(2^8)`, with optional symbol interleaving (depths
  1–5, 8) and the mandatory dual-basis symbol representation.

## Status

All planned phases are done: the generic RS engine (field, polynomial,
encode, decode with and without erasures) and both CCSDS presets, each
verified against its own standard's published worked examples/generator
polynomials rather than only self-consistency. See `PORTING_PLAN.md` for
the phased history and exactly what was checked against what.

Not covered, and not currently planned: convolutional codes, SSE
variants, and the `fec_shim` compatibility layer from the original C
library (out of scope from the start -- this crate only ports the
Reed-Solomon half), and CCSDS TC Synchronization and Channel Coding
(uses BCH, not Reed-Solomon).

- [x] Phase 1: `field` — GF(2^k) construction and arithmetic
- [x] Phase 2: `polynomial` — polynomial operations over a field
- [x] Phase 3: RS construction + systematic encode
- [x] Phase 4: RS decode (Berlekamp-Massey, Chien search, Forney)
- [x] Phase 5: RS decode with erasures
- [x] Phase 6a: CCSDS AOS Frame Header Error Control preset
      (`ccsds::fhec`), verified against CCSDS 732.0-B-4's own published
      generator polynomial
- [x] Phase 6b: CCSDS TM Synchronization and Channel Coding preset
      (`ccsds::tm_channel_coding`: RS(255,223)/RS(255,239), symbol
      interleaving; `ccsds::dual_basis`: the required symbol-
      representation conversion), verified against CCSDS 131.0-B-5's
      own published generator polynomials (33 and 17 coefficients) and
      both of its dual-basis worked examples

## License

BSD-2-Clause, matching the license of the original libcorrect
implementation this crate is ported from. See `LICENSE`.
