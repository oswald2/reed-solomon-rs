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
- The largest gap was GF(256) `encode` (1.48x). That was closed by the
  encoder rewrite below, not by removing bounds checks: the gap came from
  the algorithm (generic polynomial division), not from indexing.
- The gap shrinks as the error/erasure count grows (1.48x at encode down
  to 1.17x with the most decode work), consistent with the difference
  being dominated by a roughly constant per-call/per-symbol overhead
  rather than an algorithmically worse decode path.

## Encoder rewrite (November 2026)

`ReedSolomon::encode` used to port libcorrect's encoder: it padded and
reversed the message into a polynomial and ran a generic `poly_mod`, with
two log/exp lookups per generator coefficient. It is now a systematic
LFSR with a precomputed `field_size x min_distance` table
(`row[fb][j] = fb * g[min_distance - 1 - j]`), and the common register
lengths (32, 16, 4) are const generic so the shift and XOR stay in
vector registers. `encode_parity_interleaved` encodes all `depth`
sub-codewords of an interleaved codeblock in one pass. The independent
register chains overlap, and no de-interleaving copies are needed.
Output is bit-identical. A proptest checks it against `poly_mod`, and
space-rs checks it against golden files generated with libcorrect.

The numbers below come from a different, slower machine than the table
above, so compare only within this table. Each pair was measured in the
same run.

| Case | Before | After | libcorrect |
|---|---:|---:|---:|
| gf256_255_223 / encode (one codeword) | 19.8 µs | 5.2 µs | |
| ccsds_aos_fhec_10_6 / encode | 147 ns | 41 ns | |
| CADU check symbols, E=16, I=5, 1115 byte frame, incl. dual basis (space-rs `RsCodeblockCoder::encode_parity`) | 102 µs | 8.4 µs | 75 µs |

The libcorrect column is the old space-rs `RSCADU::encode` (5x
`correct_reed_solomon_encode` plus dual-basis tables), linked against
the `libcorrect.a` that space-rs shipped before it switched to this
crate.

## Decoder rewrite and SIMD (November 2026)

See `SIMD_PLAN.md` for the design. In short:

- Decoding computes the remainder of the received codeword with the
  encoder and derives the syndromes from it. A codeword without errors is
  detected after about one encode.
- The Chien search only tries the positions of the (possibly shortened)
  codeword and returns positions directly.
- `decode_interleaved` checks all sub-codewords in one pass, without
  copies.
- A GF(256) multiply-accumulate kernel computes the remainder (and so
  encodes), the syndromes and the Chien search. Its tier is picked at
  runtime with the `std` feature: SSSE3, AVX2 or AVX2 + GFNI. Without
  SIMD, the scalar code runs.
- `DualBasisConverter` converts whole buffers with the same tiers.

Output is bit-identical. Proptests check every SIMD tier against the
scalar code, and space-rs checks the results against golden files
generated with libcorrect.

The numbers below are from a Xeon Silver 4309Y (Ice Lake-SP, which has all
tiers), so compare only within this section. "Before" is the code before
this rewrite on the same machine. The CADU rows are an E=16, I=5
codeblock with a 1115 byte frame, including the dual basis (space-rs
`RsCodeblockCoder`).

| Case | Before | After (GFNI) | libcorrect |
|---|---:|---:|---:|
| gf256_255_223 / decode, no errors | 8.43 µs | 0.23 µs | |
| gf256_255_223 / decode, 16 errors | 19.0 µs | 2.9 µs | |
| gf256_255_223 / encode | 1.17 µs | 0.23 µs | |
| CADU decode, no errors | 46.2 µs | 1.72 µs | 33.0 µs |
| CADU decode, 16 errors per codeword | 93.5 µs | 13.7 µs | 79.0 µs |
| CADU check symbols | 3.59 µs | 1.46 µs | 34.5 µs |

Per tier, one E=16 codeword and an I=5 codeblock without the dual
basis (`benches/decode_stages.rs`):

| Case | Scalar | SSSE3 | AVX2 | GFNI |
|---|---:|---:|---:|---:|
| Decode, no errors | 1.00 µs | 0.45 µs | 0.32 µs | 0.23 µs |
| Decode, 16 errors | 8.1 µs | 3.3 µs | 3.0 µs | 2.9 µs |
| Encode | 0.98 µs | 0.46 µs | 0.32 µs | 0.23 µs |
| I=5 decode, no errors | 3.41 µs | 2.74 µs | 2.14 µs | 1.56 µs |
| I=5 decode, 16 errors per codeword | 39 µs | 17.1 µs | 15.9 µs | 14.9 µs |
| I=5 encode | 3.19 µs | 2.62 µs | 2.02 µs | 1.43 µs |

With 16 errors, about 2.4 µs of the 2.9 µs per codeword is
Berlekamp-Massey and Forney, which are still scalar.

## Reproducing

```sh
# Rust
cd reed-solomon-rs
cargo bench --features std --bench rs_benchmarks
# every SIMD tier this machine has, and the decode stages
cargo bench --features std,bench-internals --bench decode_stages

# C comparison harness (adjust the libcorrect path)
mkdir -p /tmp/bench_c && cd /tmp/bench_c
# write bench.c per the description above, then:
gcc -O3 -I/path/to/libcorrect/include bench.c \
  /path/to/libcorrect/src/reed-solomon/{polynomial,reed-solomon,encode,decode}.c \
  -o bench_c
./bench_c
```
