# Benchmarks: Rust port vs. the original C (libcorrect)

`cargo bench` runs criterion benchmarks (`benches/rs_benchmarks.rs`) for
the two configurations this crate is built around:

- `gf256_255_223`: CCSDS TM Channel Coding's E=16 code
  (`ccsds::tm_channel_coding::codec_e16()`).
- `ccsds_aos_fhec_10_6`: the CCSDS AOS Frame Header Error Control code
  (`ccsds::fhec::codec()`), the motivating case for this whole crate.

For each, three or four cases: `encode`, `decode_no_errors` (all
syndromes zero, cheapest decode path), `decode_*_errors_at_capacity`
(the maximum number of errors the code guarantees correcting -- the most
expensive decode path, since Berlekamp-Massey/Chien search/Forney all
scale with the number of errors found), and (GF(256) only)
`decode_*_erasures_*_errors` to exercise the erasure-decoding path.

## Comparison methodology

To compare against the original C implementation, a small standalone C
benchmark harness (not part of either repo -- see below) was built
directly against libcorrect's `short_rs` branch sources
(`src/reed-solomon/{polynomial,reed-solomon,encode,decode}.c`) with
`gcc -O3`, using the *same* code parameters as the Rust presets above
(same primitive polynomial, root convention, `min_distance`, and the
same encoded messages/corruption patterns), so the two sides are doing
identical work. It times each operation in a tight loop with
`clock_gettime(CLOCK_MONOTONIC, ...)` for at least 2 seconds per case
and reports `ns/iter`.

This is **not** an apples-to-apples statistical comparison: criterion
does outlier detection, warm-up, and reports a confidence interval;
the C harness is a plain loop-and-divide. Treat both as
order-of-magnitude/relative numbers from one run on one machine, not
precise benchmarks. The C harness lives at `/tmp/bench_c/bench.c` during
development and isn't committed to either repo, since it exists purely
to answer "how does the port compare" and isn't part of either
library's own test/build story; regenerate it from this file's
description if you want to rerun the comparison.

Both sides built with full optimization (`cargo bench`'s `bench`
profile, which inherits `release`'s `opt-level = 3`, vs. `gcc -O3`).

## Results (one run, x86_64, `rustc 1.98.1` / `gcc`, September 2026)

| Case | Rust (ns/iter) | C (ns/iter) | Rust / C |
|---|---:|---:|---:|
| gf256_255_223 / encode | 5 727 | 3 857 | 1.48x |
| gf256_255_223 / decode_no_errors | 4 420 | 3 475 | 1.27x |
| gf256_255_223 / decode_16_errors_at_capacity | 10 906 | 8 933 | 1.22x |
| gf256_255_223 / decode_20_erasures_5_errors | 11 595 | 9 901 | 1.17x |
| ccsds_aos_fhec_10_6 / encode | 41.2 | 39.2 | 1.05x |
| ccsds_aos_fhec_10_6 / decode_no_errors | 37.7 | 37.8 | 0.997x |
| ccsds_aos_fhec_10_6 / decode_2_errors_at_capacity | 158.5 | 136.6 | 1.16x |

## Takeaways

- The Rust port is consistently in the same ballpark as the C
  implementation -- never more than ~1.5x slower, and for the small
  GF(16) FHEC code (the actual motivating use case, and small enough
  that per-call overhead matters more than per-symbol work) it's
  essentially tied, even matching C on `decode_no_errors`.
- The largest gap is GF(256) `encode` (1.48x). The likely cause is bounds
  checking on slice indexing in the hot copy/reversal loops in
  `ReedSolomon::encode` and `polynomial::poly_mod`, versus C's raw
  pointer arithmetic; this is exactly the kind of gap that`get_unchecked`
  in a few hot spots (with a safety comment justifying it, and only
  after confirming it actually helps) could close, but that's a
  targeted optimization pass, not something done as part of this
  benchmark run.
- The gap shrinks as the error/erasure count grows (1.48x at encode down
  to 1.17x with the most decode work), consistent with the difference
  being dominated by a roughly constant per-call/per-symbol overhead
  rather than an algorithmically worse decode path.

## Reproducing

```sh
# Rust
cd reed-solomon-rs
cargo bench

# C comparison harness (adjust the libcorrect path)
mkdir -p /tmp/bench_c && cd /tmp/bench_c
# write bench.c per the description above, then:
gcc -O3 -I/path/to/libcorrect/include bench.c \
  /path/to/libcorrect/src/reed-solomon/{polynomial,reed-solomon,encode,decode}.c \
  -o bench_c
./bench_c
```
