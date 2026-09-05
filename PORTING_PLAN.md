# Porting plan: libcorrect (short_rs) Reed-Solomon -> Rust

Source: https://github.com/quiet/libcorrect, branch `short_rs`, RS-only
sources (`include/correct/reed-solomon*`, `src/reed-solomon/*.c`).
Motivating use case: quiet/libcorrect#17 (CCSDS AOS TM Frame Header Error
Control, a shortened RS(10,6) code over GF(2^4)).

## Scope

Port the generic RS engine (field, polynomial, encode, decode, decode with
erasures), parameterized exactly like the C API (field width, primitive
polynomial, first consecutive root, root gap, `num_roots` all runtime
values). On top of that, add CCSDS conformance presets:

- CCSDS 732.0-B AOS Frame Header Error Control: RS(10,6), GF(2^4), `0x13`.
- CCSDS 131.0-B TM Synchronization and Channel Coding: RS(255,223), GF(2^8),
  interleaving depths I in {1,2,3,4,5,8}, dual-basis <-> conventional-basis
  symbol conversion (Annex G).

Out of scope: convolutional codes, SSE variants, the `fec_shim` compat
layer, CCSDS TC (which uses BCH, not RS).

## Design

- `no_std` + `alloc`, allocate only in constructors, zero allocation at
  steady state (matches the C code's own design intent).
- `Result<usize, RsError>` instead of `ssize_t` sentinel `-1` returns.
- Eager initialization of decode tables in the constructor instead of the
  C code's lazy `has_init_decode` flag — simpler, no manual free()/Drop
  bookkeeping, negligible cost difference for the field sizes in scope.
- Keep byte-order reversal and log-table aliasing quirks (e.g. `log(1)` as
  `0` vs `largest_element`) bit-for-bit identical to the C: the
  Forney/Chien-search math and CCSDS wire compatibility depend on them.
- Polynomials are plain `&[u8]`/`&mut [u8]` coefficient slices (order is
  just `len() - 1`), not a stateful `Polynomial` type. The C `polynomial_t`
  pairs a buffer with a separately-mutable `order` so decode-time scratch
  buffers can be reused across calls without reallocating even as their
  logical length changes; here that's expressed with slicing
  (`&buf[..=order]`) at the call site instead, which keeps `polynomial.rs`
  itself simple and independently testable. The scratch-buffer reuse this
  was for happens in the RS decoder (phase 4+), which owns the buffers and
  slices into them.
- Where the C decoder reuses a single buffer for two different meanings
  at different points in a call (e.g. `decode_with_erasures` swapping
  `rs->error_locator` for a combined-locator buffer via pointer
  assignment, then swapping back), the Rust port just uses two
  separate, clearly-named persistent buffers instead. Slightly more
  memory (all buffers here are already tiny -- at most a few hundred
  bytes even for GF(256)), no swap-back bookkeeping to get wrong.

## Phases

1. `field` — GF(2^k) table construction (`field_create`) + arithmetic
   (`add/sub/sum/mul/div/mul_log/div_log/mul_log_element/pow`). Unit tests
   against known tables for `0x11d` (GF(256)) and `0x13` (GF(16), CCSDS
   AOS FHEC).
2. `polynomial` — `mul`, `mod` (long division), `formal_derivative`, `eval`,
   `eval_lut`, `eval_log_lut`, `build_exp_lut`, `create_from_roots`.
   Property tests: LUT-based eval must match naive eval for random
   polynomials/points.
3. RS construction + systematic encode (generator polynomial from roots,
   virtual padding for shortened codes). Validate against (255,223) and
   the GF(16) (15,11) / shortened (10,6) cases.
4. Decode without erasures: syndromes, Berlekamp-Massey, Chien search,
   Forney error values, each as its own module.
5. Decode with erasures: modified syndromes, erasure-locator-from-roots,
   combined error+erasure Forney step.
6. CCSDS conformance layer:
   a. `ccsds::fhec` (done) -- CCSDS 732.0-B-4 SS4.1.2.6.5, AOS Frame
      Header Error Control. Verified (not assumed) against the standard
      itself: fetched the actual PDF, confirmed `F(x) = x^4 + x + 1`
      (`0x13`) and generator roots `alpha^6..alpha^9`
      (`first_consecutive_root = 6`, `generator_root_gap = 1`, contrary
      to an earlier unverified guess of `1` elsewhere in this crate's
      test suite), and cross-checked our `ReedSolomon::generator()`
      output against the standard's own published `g(x)` coefficients
      byte-for-byte -- see `ccsds::fhec`'s
      `generator_polynomial_matches_the_published_standard` test.
   b. `ccsds::tm_channel_coding` (+ interleaving) and `ccsds::dual_basis`
      -- not started. CCSDS 131.0-B's RS(255,223) root convention,
      interleaving depths, and the dual-basis <-> conventional-basis
      symbol conversion table all need the same fetch-and-verify
      treatment phase 6a got, not assumption.

## Correctness strategy

- Differential fixtures captured from the C implementation (encode
  outputs, syndromes, error locators) for fixed inputs, committed under
  `tests/`.
- `proptest`-based round-trip tests mirroring `tests/rs_tester.c`'s
  `test_rs_errors`: random message, inject up to `floor(min_distance/2)`
  errors and/or up to `min_distance` erasures (respecting
  `2*errors + erasures < min_distance`), assert round-trip, for each of
  the field/min_distance configurations the C test sweeps (255,223 GF(256)
  with min_distance in {32,16,8,4}; GF(16) (15,11)/(10,6)).
- CCSDS conformance tests against published blue-book worked examples
  where available, since self-consistent round-trip tests alone can't
  catch a wrong root convention.

## Stretch (not in v1)

- Port `tools/find_rs_primitive_poly.c` as an example/binary.
- `criterion` benchmarks.
- SIMD field ops (no C equivalent to port from; new work).
