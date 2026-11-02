# SIMD and algorithmic optimization plan

This is the plan for speeding up Reed-Solomon decoding (and possibly encoding) in
this crate, with SIMD where it pays off. It was written on a machine without
AVX2. The measurements and the SIMD work are meant to be done on a machine
that has it.

## Status (2 November 2026)

Phases 0, 1 and 2 (x86) are implemented, measured on a Xeon Silver 4309Y
(Ice Lake-SP), see "Results" below. Not done yet:

- Phase 3 (space-rs integration, `rev` bump, `BENCHMARKS.md`)
- NEON for aarch64
- measurements on real Zen 3 and Zen 4 hardware

## Targets

- **Floor: the rustc default `x86_64` target (SSE2), no `-C target-cpu`.**
  The scalar code is the guaranteed path. Everything above it is picked at
  runtime (with `std`) or from the enabled target features (without `std`).
- **Old machines** (Nehalem/Westmere, SSE4.2): the SSSE3 tier. CPUs with SSE3
  but no SSSE3 (AMD K10, Pentium 4) get scalar code.
- **Zen 3** (x86-64-v3): the AVX2 tier.
- **Zen 4**, also Zen 5, Ice Lake and newer, and Alder Lake and newer: the
  GFNI tier. It uses the 256-bit VEX encoding, so it needs only AVX2 + GFNI
  and no AVX-512.
- **No AVX-512 tier.** 2t = 32 for E=16 fills exactly one ymm register, and
  Zen 4 splits 512-bit operations into two 256-bit halves anyway.

## Where we are (November 2026)

The encoder was already rewritten, see `BENCHMARKS.md`, "Encoder rewrite":

- a table-driven LFSR
- `encode_parity_interleaved`
- no SIMD intrinsics; the compiler auto-vectorizes the fixed-size register shift and XOR

Decoding is still a straight port of libcorrect and is now the slower side.

Numbers from the old machine (Xeon E5620, Westmere, SSSE3 only, no AVX).
The task is a 1115-byte frame, E=16, interleaving depth 5, with dual basis:

| Case | Rust | libcorrect |
|---|---:|---:|
| Check symbols (encode) | 8.4 µs | 75 µs |
| Decode, no errors | 106 µs | 72 µs |
| Decode, 16 errors per codeword | 238 µs | 188 µs |
| One codeword, `gf256_255_223/decode_no_errors` | 19.4 µs | |
| One codeword, `gf256_255_223/decode_16_errors_at_capacity` | 41.7 µs | |

Re-measure all of these on the new machine first. Absolute numbers from the
old machine are not comparable.

## Findings that shape the plan

- **The biggest decode costs are algorithmic, not a lack of SIMD:**
  - **Syndromes** (`find_syndromes` in `src/rs/mod.rs`): the full 255-symbol
    received polynomial is evaluated at each of the 32 roots. That is about
    8k dependent log/exp table lookups and accounts for nearly all of the
    no-error decode time.
  - **`chien::find_error_locations`**: maps each error root to its position by
    brute-forcing the field with `field.pow()` for every error.
  - **`chien::factorize_error_locator`**: evaluates the locator at all 256
    field elements, even for shortened codewords, with no early exit.
  - **`tm_channel_coding::decode_interleaved`**: allocates two `Vec`s and
    de-interleaves with a copy on every call.
- **GFNI:** `gf2p8mulb` is hard-wired to the AES polynomial 0x11B and is
  unusable for CCSDS (0x187). Only `gf2p8affineqb` (multiply by a constant via
  an 8x8 bit matrix) works for arbitrary fields. It needs Ice Lake / Zen 4 or
  newer.
- **Instruction sets:** Rust's default `x86_64` target only assumes SSE2. Even
  `pshufb` (SSSE3) needs runtime detection or `-C target-cpu`.
- **The crate is `no_std` + `alloc`:** runtime CPU detection is only available
  with the `std` feature, which space-rs enables.

## Phase 0: Measure each decode stage (done)

As built:

- The `bench-internals` feature exposes `rs::bench_internals`: `Stages` and
  `set_tier`.
- `benches/decode_stages.rs` runs every available tier: full encode/decode,
  E16/I5 blocks and each stage. Run it with
  `cargo bench --features std,bench-internals --bench decode_stages`.
- `benches/rs_benchmarks.rs` has a `tm_e16_i5` group.
- Pass `--save-baseline` with `--bench <name>`, not with `--benches`: the lib
  target's libtest harness rejects the option.

Original plan:

- Add a `bench-internals` feature with a `#[doc(hidden)]` module that exposes
  the decode stages to criterion:
  - syndromes
  - Berlekamp-Massey
  - Chien root search
  - root-to-position mapping
  - Forney
- Add an interleaved E16/I5 decode case to `benches/rs_benchmarks.rs`.
- Rebuild the libcorrect comparison harness (see the appendix) and record the
  baseline for:
  - `cargo bench -p reed-solomon-rs`
  - `cargo bench -p space-rs --bench cadu`
  - the harness

## Phase 1: Algorithmic fixes (portable, no `unsafe`, done first) (done)

As built, the decoder no longer needs the received polynomial:

- The remainder gives the syndromes, and corrections are applied directly
  to the output message.
- The Chien search (`chien::find_error_positions`) walks the codeword's
  positions with a log-domain recurrence. It returns roots and positions
  together, which replaces the root-to-position table, and stops early.
- `decode_interleaved` computes all remainders in one pass
  (`encode_parity_interleaved`) and only decodes codewords with errors.
- `decode` is now `decode_with_erasures` with no erasures.

**Behavior change:** roots outside a shortened codeword now count as "too
many errors". Before, they "corrected" virtual fill symbols and reported
success.

Original plan:

1. **Syndromes from the remainder.**
   - Run the encoder LFSR (`encode_parity`) over the received message part and
     XOR the result with the received parity symbols. This gives
     `R(x) = r(x) mod g(x)`, with only 2t coefficients.
   - If `R = 0`, the codeword is valid and decoding is done. The no-error path
     then costs about one encode instead of a full syndrome evaluation.
   - Otherwise `S_j = R(α_j)`, because `g(α_j) = 0`. That is 2t×2t instead of
     2t×n operations.
   - Watch the coefficient order. The LFSR register holds the highest-order
     remainder coefficient first.
   - Add a proptest against the current `find_syndromes`, which stays as the
     reference.
2. **Root-to-position mapping:** build a `field_size`-entry table
   (root -> codeword position) in `ReedSolomon::new` to replace the brute
   force in `find_error_locations`.
3. **Chien search:**
   - evaluate only the positions that exist in the (possibly shortened)
     codeword
   - write positions directly instead of roots that need mapping afterwards
   - stop as soon as `order` roots are found
4. **`decode_interleaved`:** remove the per-call allocations, in the same way
   as `encode_interleaved`.

Benchmark after Phase 1 and look at the stage breakdown again. It decides how
much of Phase 2 is still worth doing.

## Phase 2: GF(256) SIMD kernel (done for x86)

As built, in `src/rs/simd.rs`:

- One kernel, `Kernel::mul_acc`, computes `out[32] = Σ c_j · rows[j]` and
  returns a zero-lane mask.
- The coefficients can be read with a stride, so interleaved codewords are
  used in place and need no de-interleave copy.
- Tiers: SSSE3 and AVX2 use `pshufb` nibble tables; GFNI uses
  `vgf2p8affineqb` with one 8x8 matrix per symbol.
- `ReedSolomon` holds three matrices: syndrome, parity (`x^(u+2t) mod g`
  rows) and Chien (`x_p^t`).
- Any SIMD tier uses them for:
  - the remainder
  - the syndromes
  - Chien
  - **encode**: matrix form beats the LFSR on every tier, also for
    interleaved blocks
- The `force-scalar` feature turns the SIMD tiers off.
- The proptest `every_simd_tier_matches_scalar` compares every available
  tier with scalar, errors included.

Original plan:

**The kernel.** One multiply-accumulate kernel covers most uses:
`acc ^= Σ s_i · V_i`, where each `s_i` is a single field symbol and each `V_i`
is a 16- or 32-byte vector.
- It uses split-nibble product tables:
  `s·v = LO_s[v & 0x0F] ^ HI_s[v >> 4]`, with `pshufb` / `vpshufb` on x86 and
  `vqtbl1q_u8` on aarch64.
- The tables take 256 × 32 bytes = 8 KB per field and are built once in
  `ReedSolomon::new`.

**Uses:**

| Use | `s_i` | `V_i` | Note |
|---|---|---|---|
| Syndromes from `R` (Phase 1.1) | the `R` coefficients | precomputed powers `[α_j^i]` for each syndrome `j` | 32 lanes = 32 syndromes |
| Chien search | locator coefficients `Λ_t` | `[x_k^t]` for 16/32 candidate positions | root lanes via compare-to-zero + `movemask` |
| Encode in matrix form | message symbols `m_i` | rows of the precomputed `k × 2t` parity matrix (223×32 = 7 KB for E16) | no serial dependency, unlike the LFSR |

- Adopt matrix-form encode only if it beats the current
  `encode_parity_interleaved`, measured on the new machine.
- **Stays scalar:**
  - Berlekamp-Massey and Forney: they are sequential or tiny.
  - The GF(16) FHEC code: 10-symbol codewords, so per-call overhead dominates.

**Dispatch:**
- Add a private `simd` module with these tiers:
  - scalar: the existing code, always available
  - SSSE3
  - AVX2
  - NEON, always present on aarch64
  - optionally GFNI `gf2p8affineqb`, if the target hardware has it
- With the `std` feature, `is_x86_feature_detected!` picks the tier once in
  `ReedSolomon::new` and stores it in the codec (as an enum or function
  pointers).
- Without `std`, the tier is chosen at compile time with
  `cfg(target_feature = ...)`.
- A `force-scalar` feature lets tests and CI exercise the fallback.

**Safety:** `unsafe` stays confined to the `simd` module. Every
`#[target_feature(enable = ...)]` function gets a SAFETY comment saying which
detection guarantees it is only called when the feature is present.

## Phase 3: space-rs integration

- `rs::RsCodeblockCoder::decode` (space-rs): avoid the `conventional` buffer
  copy where the new API allows it, for example by feeding dual-basis-mapped
  symbols through an iterator as the encode side already does.
- Dual-basis conversion is GF(2)-linear. It can be one `gf2p8affineqb` per
  32 bytes (or two `pshufb` nibble lookups), or folded into the kernel's
  per-symbol matrices.
- Commit reed-solomon-rs and bump the `rev` pin in `space-rs/Cargo.toml`.
- Add the results to `BENCHMARKS.md`.

## Verification

- Proptests check every SIMD tier available on the machine against the scalar
  path, both per kernel and for full decode with errors and erasures.
- The existing tests must stay green:
  - `cargo test -p reed-solomon-rs`
  - `cargo test -p space-rs`, in particular the libcorrect golden tests
    `rs::codeblock_matches_libcorrect` and `cadu::cadu_matches_libcorrect`
  - `cargo test -p galileo-rs`
- Run the tests twice: once normally and once with `--features force-scalar`.
- Test tiers the machine lacks under Intel SDE (x86) or QEMU (aarch64), or on
  real hardware.
- Benchmark before and after each phase (criterion `--save-baseline` /
  `--baseline`), including the libcorrect harness.

## Results (Xeon Silver 4309Y, Ice Lake-SP)

One E=16 codeword and E16/I5 blocks (1115-byte message), from
`decode_stages`:

| Case | Before | Scalar | SSSE3 | AVX2 | GFNI |
|---|---:|---:|---:|---:|---:|
| Encode, 1 codeword | 1.17 µs | 0.98 µs | 0.46 µs | 0.32 µs | 0.23 µs |
| Decode, no errors | 8.43 µs | 1.00 µs | 0.45 µs | 0.32 µs | 0.23 µs |
| Decode, 16 errors | 19.0 µs | 8.1 µs | 3.3 µs | 3.0 µs | 2.9 µs |
| I5 encode | 3.23 µs | 3.19 µs | 2.62 µs | 2.02 µs | 1.43 µs |
| I5 decode, no errors | 43.4 µs | 3.41 µs | 2.74 µs | 2.14 µs | 1.56 µs |
| I5 decode, 16 errors/codeword | 101.6 µs | 39 µs | 17.1 µs | 15.9 µs | 14.9 µs |

Stages for 16 errors:

| Stage | Scalar | SSSE3 | AVX2 | GFNI |
|---|---:|---:|---:|---:|
| Remainder | 0.98 µs | 450 ns | 314 ns | 218 ns |
| Syndromes | 893 ns | 71 ns | 51 ns | 35 ns |
| Chien | 3.9–4.5 µs | 324 ns | 224 ns | 187 ns |
| Berlekamp-Massey | 1.28 µs | same | same | same |
| Forney | 1.0–1.1 µs | same | same | same |

With SIMD, the 16-error decode is now mostly Berlekamp-Massey and Forney.
Both are scalar and could be the next target. Forney could evaluate Ω and
Λ' at all error positions at once with the kernel.

space-rs, GFNI picked at runtime:

| Case | Before | Now | libcorrect |
|---|---:|---:|---:|
| `RsCodeblockCoder` check symbols | 3.60 µs | 2.47 µs | 34.9 µs |
| `RsCodeblockCoder` decode, no errors | 46.4 µs | 4.0 µs | 33.1 µs |
| `RsCodeblockCoder` decode, 16 errors/codeword | 93.9 µs | 16.9 µs | 79.0 µs |

About 2.4 µs of the 4.0 µs no-error decode is space-rs's own dual-basis table
mapping and output loop (Phase 3).

## Rough expectations (not measured, written before the work)

| Case | Old machine now | After the plan |
|---|---:|---:|
| No-error CADU decode | 106 µs | about 10 µs or less, mostly from Phase 1.1 |
| 16 errors per codeword | 238 µs | well under libcorrect's 188 µs |

Encode gains from SIMD are uncertain because it is already about 9× faster
than libcorrect.

## Appendix: libcorrect comparison harness

This is a standalone crate, not part of the workspace. It benchmarks the old
libcorrect-based space-rs CADU coding (`RSCADU`) against the current
`RsCodeblockCoder` in the same process. Its build checks that both produce
identical check symbols.

```sh
mkdir -p lcbench/{lib,src,benches}
git -C <workspace>/space-rs show 8b648827^:lib/libcorrect.a > lcbench/lib/libcorrect.a
# write the four files below (adjust the absolute paths in Cargo.toml)
cd lcbench && cargo bench --bench vs
```

`libcorrect.a` is the prebuilt x86_64 Linux archive that space-rs shipped. On
another architecture, build it from `lib/libcorrect` in the same commit
(CMake).

`Cargo.toml`:

```toml
[package]
name = "lcbench"
version = "0.1.0"
edition = "2024"

[dependencies]
space-rs = { path = "<workspace>/space-rs" }
reed-solomon-rs = { path = "<workspace>/reed-solomon-rs", features = ["std"] }
bytes = "1"

[dev-dependencies]
criterion = "0.8"

[patch."ssh://git@git.terma.com/library/reed-solomon-rs.git"]
reed-solomon-rs = { path = "<workspace>/reed-solomon-rs" }

[profile.release]
lto = "fat"
codegen-units = 1

# keeps it out of any enclosing workspace
[workspace]

[[bench]]
name = "vs"
harness = false
```

`build.rs`:

```rust
fn main() {
    let lib = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("lib");
    println!("cargo::rustc-link-search={}", lib.display());
    println!("cargo::rustc-link-lib=static=correct");
}
```

`src/lib.rs`:

```rust
//! The old libcorrect based CADU RS encoding/decoding (space-rs before 8b648827)
use reed_solomon_rs::ccsds::dual_basis::{from_dual_basis, to_dual_basis};

#[repr(C)]
pub struct CorrectRs {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn correct_reed_solomon_create(poly: u16, fcr: u8, gap: u8, roots: usize) -> *mut CorrectRs;
    fn correct_reed_solomon_encode(rs: *mut CorrectRs, msg: *const u8, len: usize, enc: *mut u8) -> isize;
    fn correct_reed_solomon_decode(rs: *mut CorrectRs, enc: *const u8, len: usize, msg: *mut u8) -> isize;
}

const RS_N: usize = 255;
const RS_K: usize = 223;
const I: usize = 5;

pub struct LibcorrectCadu {
    rs: *mut CorrectRs,
    to_dual: [u8; 256],
    from_dual: [u8; 256],
}

impl LibcorrectCadu {
    pub fn new() -> Self {
        LibcorrectCadu {
            rs: unsafe { correct_reed_solomon_create(0x187, 112, 11, 32) },
            to_dual: std::array::from_fn(|i| to_dual_basis(i as u8)),
            from_dual: std::array::from_fn(|i| from_dual_basis(i as u8)),
        }
    }

    /// Same as the old `RSCADU::encode`: check symbols for a 1115 byte frame
    pub fn encode(&self, frame: &[u8], check: &mut Vec<u8>) {
        let mut msg = [[0u8; RS_K]; I];
        for j in 0..RS_K {
            for k in 0..I {
                msg[k][j] = self.from_dual[frame[j * I + k] as usize];
            }
        }
        let mut emsg = [[0u8; RS_N]; I];
        for k in 0..I {
            unsafe { correct_reed_solomon_encode(self.rs, msg[k].as_ptr(), RS_K, emsg[k].as_mut_ptr()) };
        }
        for j in RS_K..RS_N {
            for k in 0..I {
                check.push(self.to_dual[emsg[k][j] as usize]);
            }
        }
    }

    /// Same structure as the old `RSCADU::decode`
    pub fn decode(&self, block: &[u8], frame: &mut Vec<u8>) -> bool {
        let mut cw = [[0u8; RS_N]; I];
        for j in 0..RS_N {
            for k in 0..I {
                cw[k][j] = self.from_dual[block[j * I + k] as usize];
            }
        }
        let mut msg = [[0u8; RS_K]; I];
        for k in 0..I {
            if unsafe { correct_reed_solomon_decode(self.rs, cw[k].as_ptr(), RS_N, msg[k].as_mut_ptr()) } < 0 {
                return false;
            }
        }
        for j in 0..RS_K {
            for k in 0..I {
                frame.push(self.to_dual[msg[k][j] as usize]);
            }
        }
        true
    }
}
```

`benches/vs.rs`:

```rust
use std::hint::black_box;

use bytes::BytesMut;
use criterion::{Criterion, criterion_group, criterion_main};
use lcbench::LibcorrectCadu;
use space_rs::rs::{RsCodeblockCoder, RsConfig};

fn bench(c: &mut Criterion) {
    let frame: Vec<u8> = (0..1115).map(|i| (i * 7 + 3) as u8).collect();
    let lc = LibcorrectCadu::new();
    #[allow(unused_mut)]
    let mut coder = RsCodeblockCoder::new(&RsConfig::DEFAULT, 1115).unwrap();

    // sanity: both produce the same check symbols
    let mut a = Vec::new();
    lc.encode(&frame, &mut a);
    let mut b = BytesMut::new();
    coder.encode_parity(&frame, &mut b).unwrap();
    assert_eq!(&a[..], &b[..]);
    let mut block = frame.clone();
    block.extend_from_slice(&a);
    let mut corrupted = block.clone();
    corrupted[..80].iter_mut().for_each(|b| *b = !*b);

    let mut g = c.benchmark_group("cadu_parity_e16_i5");
    g.bench_function("libcorrect", |bch| {
        let mut out = Vec::with_capacity(160);
        bch.iter(|| {
            out.clear();
            lc.encode(black_box(&frame), &mut out)
        })
    });
    g.bench_function("rust", |bch| {
        let mut out = BytesMut::with_capacity(160);
        bch.iter(|| {
            out.clear();
            coder.encode_parity(black_box(&frame), &mut out).unwrap()
        })
    });
    g.finish();

    let mut g = c.benchmark_group("cadu_decode_e16_i5");
    for (name, blk) in [("no_errors", &block), ("16_errors", &corrupted)] {
        g.bench_function(format!("libcorrect_{name}"), |bch| {
            let mut out = Vec::with_capacity(1115);
            bch.iter(|| {
                out.clear();
                assert!(lc.decode(black_box(blk), &mut out))
            })
        });
        g.bench_function(format!("rust_{name}"), |bch| {
            let mut out = BytesMut::with_capacity(1115);
            bch.iter(|| {
                out.clear();
                coder.decode(black_box(blk), &mut out).unwrap()
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
```
