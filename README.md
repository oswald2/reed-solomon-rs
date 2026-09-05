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
- **CCSDS 131.0-B, TM Synchronization and Channel Coding** — RS(255,223)
  over `GF(2^8)`, with optional interleaving and a dual-basis symbol
  representation.

## Status

Early port, in progress. See `PORTING_PLAN.md` for the phased plan and
current progress.

- [x] Phase 1: `field` — GF(2^k) construction and arithmetic
- [x] Phase 2: `polynomial` — polynomial operations over a field
- [x] Phase 3: RS construction + systematic encode
- [ ] Phase 4: RS decode (Berlekamp-Massey, Chien search, Forney)
- [ ] Phase 5: RS decode with erasures
- [ ] Phase 6: CCSDS conformance presets (AOS FHEC, TM channel coding,
      interleaving, dual-basis conversion)

## License

BSD-2-Clause, matching the license of the original libcorrect
implementation this crate is ported from. See `LICENSE`.
