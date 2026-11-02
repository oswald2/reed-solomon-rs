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

## Usage

```toml
[dependencies]
reed-solomon-rs = { git = "ssh://git@git.terma.com/library/reed-solomon-rs.git", rev = "<commit>", features = ["std"] }
```

Enable `std` unless you are on a `no_std` target. With `std`, the fastest
SIMD code the CPU supports is picked at runtime (see "Performance").

```rust
use reed_solomon_rs::ccsds::dual_basis::DualBasisConverter;
use reed_solomon_rs::ccsds::tm_channel_coding;

// CCSDS TM channel coding, E=16, interleaving depth 5
let mut rs = tm_channel_coding::codec_e16();
let depth = 5;
let frame: Vec<u8> = (0..1115).map(|i| i as u8).collect();

let mut codeblock = vec![0u8; frame.len() + depth * rs.min_distance()];
tm_channel_coding::encode_interleaved(&rs, depth, &frame, &mut codeblock)?;

codeblock[10] ^= 0xff; // a symbol error
let mut decoded = vec![0u8; frame.len()];
tm_channel_coding::decode_interleaved(&mut rs, depth, &codeblock, &mut decoded)?;
assert_eq!(decoded, frame);

// On the wire, CCSDS uses the dual basis representation
let dual = DualBasisConverter::new();
dual.to_dual_in_place(&mut codeblock);
```

- **Building codecs:** build a codec (`ReedSolomon::new` or the presets)
  once and reuse it. Construction precomputes tables and matrices so that
  encode and decode never allocate.
- **Generic codes:** `ReedSolomon` also takes arbitrary parameters (field
  polynomial, first root, root gap, number of parity symbols), encodes
  shortened codes (messages shorter than `message_length()`) and decodes
  with erasures (`decode_with_erasures`).
- **Errors outside a shortened codeword:** if decoding a shortened
  codeword would place an error in the virtual fill, the codeword is
  reported as uncorrectable (`RsError::TooManyErrors`).

## Features

| Feature | Effect |
|---|---|
| `std` | `std::error::Error` impls, and runtime CPU detection for the SIMD code. Without it, SIMD is only used for target features enabled at compile time (e.g. `-C target-feature=+avx2,+gfni` or `-C target-cpu=...`). |
| `force-scalar` | Never use SIMD, e.g. to test the portable code on any machine. |
| `bench-internals` | Exposes decode stages and a SIMD tier override to the `decode_stages` benchmark. Not a stable API. |

## Performance

GF(2^8) codes (CCSDS TM channel coding and any other code over a 256-element
field) use a SIMD kernel for:

- encoding
- the decoder's remainder/syndrome computation
- the Chien search
- Forney's algorithm
- `DualBasisConverter`

The kernel tier is picked when a codec is built:

| Tier | Needs | Typical CPUs |
|---|---|---|
| Scalar | nothing (rustc's default `x86_64` target) | anything, including CPUs without SSSE3 |
| SSSE3 | SSSE3 | Core 2 and later, e.g. Nehalem/Westmere |
| AVX2 | AVX2 | Haswell and later, Zen 1–3 |
| GFNI | AVX2 + GFNI (no AVX-512 needed) | Ice Lake and later, Alder Lake and later, Zen 4 and later |

GF(16) codes (AOS FHEC) are tiny and always use the scalar code. The
scalar code is the portable fallback, also on non-x86 targets (NEON for
aarch64 is not implemented yet). All tiers give bit-identical results.

Some numbers (Xeon Silver 4309Y, GFNI tier; a 1115-byte frame at E=16,
depth 5, including the dual basis conversion, via space-rs):

| Case | This crate | libcorrect |
|---|---:|---:|
| Check symbols | 1.5 µs | 34.5 µs |
| Decode, no errors | 1.7 µs | 33.0 µs |
| Decode, 16 errors per codeword | 8.9 µs | 79.1 µs |

See `BENCHMARKS.md` for all tiers and the methodology, and `SIMD_PLAN.md`
for the design.

## Testing and benchmarks

```sh
cargo test --features std               # all SIMD tiers this CPU has
cargo test --features std,force-scalar   # the scalar code only
cargo bench --features std --bench rs_benchmarks
cargo bench --features std,bench-internals --bench decode_stages   # per tier and per decode stage
```

The proptests check every available SIMD tier against the scalar code,
including uncorrectable inputs. Without `std`, the tests only cover the
tiers enabled at compile time.

## Status

All planned phases are done: the generic RS engine (field, polynomial,
encode, decode with and without erasures) and both CCSDS presets, each
verified against its own standard's published worked examples/generator
polynomials rather than only self-consistency. See `PORTING_PLAN.md` for
the phased history and exactly what was checked against what.

Since then, the encoder and decoder have been rewritten for speed and use
SIMD where the CPU has it (see "Performance").

Not covered, and not currently planned:

- libcorrect's convolutional codes, including its SSE variants, and its
  `fec_shim` compatibility layer. These were out of scope from the
  start: this crate only ports the Reed-Solomon half.
- CCSDS TC Synchronization and Channel Coding, which uses BCH, not
  Reed-Solomon.

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
