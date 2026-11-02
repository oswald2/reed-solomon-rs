//! GF(256) SIMD kernel: one multiply-accumulate of a block of 32 field
//! elements, `out = sum_j c_j * rows[j]`, where each `c_j` is a single
//! symbol and each `rows[j]` a 32-symbol row of a
//! precomputed matrix. Syndromes, the Chien search and the matrix form
//! of the encoder's remainder are all this operation (see the matrices
//! in [`super::ReedSolomon`]).
//!
//! Tiers, picked at runtime (with the `std` feature) or at compile time
//! (without it, from the enabled target features):
//!
//! - [`Tier::Scalar`]: no SIMD; [`super::ReedSolomon`] keeps its
//!   log-table code and does not use this kernel at all.
//! - [`Tier::Ssse3`]: split-nibble product tables, `s * v = LO_s[v & 15]
//!   ^ HI_s[v >> 4]`, with `pshufb` on two 16-byte halves.
//! - [`Tier::Avx2`]: the same with `vpshufb` on 32 bytes.
//! - [`Tier::Gfni`]: `vgf2p8affineqb` (AVX2 + GFNI, VEX encoded, no
//!   AVX-512 needed) multiplies by `s` as an 8x8 bit matrix. `gf2p8mulb`
//!   can't be used: it is hard-wired to the AES polynomial.
//!
//! `unsafe` is confined to this module: [`Kernel`] can only be built for
//! a tier [`Tier::is_available`] confirms, which is what makes calling
//! the `#[target_feature]` functions sound.

use alloc::vec::Vec;

use crate::field::GaloisField;

/// Width of one block, in symbols.
pub(crate) const BLOCK: usize = 32;

/// An instruction set tier of the GF(256) kernel. See the module docs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    Scalar,
    Ssse3,
    Avx2,
    Gfni,
}

impl Tier {
    #[cfg(any(test, feature = "bench-internals"))]
    pub const ALL: [Tier; 4] = [Tier::Scalar, Tier::Ssse3, Tier::Avx2, Tier::Gfni];

    /// Whether this CPU (with `std`) or this build's target features
    /// (without `std`) support the tier.
    pub fn is_available(self) -> bool {
        match self {
            Tier::Scalar => true,
            Tier::Ssse3 => has_ssse3(),
            Tier::Avx2 => has_avx2(),
            Tier::Gfni => has_avx2() && has_gfni(),
        }
    }

    /// The best available tier, or [`Tier::Scalar`] with the
    /// `force-scalar` feature.
    pub fn detect() -> Tier {
        if cfg!(feature = "force-scalar") {
            return Tier::Scalar;
        }
        [Tier::Gfni, Tier::Avx2, Tier::Ssse3]
            .into_iter()
            .find(|t| t.is_available())
            .unwrap_or(Tier::Scalar)
    }
}

#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn has_ssse3() -> bool {
    std::is_x86_feature_detected!("ssse3")
}
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn has_avx2() -> bool {
    std::is_x86_feature_detected!("avx2")
}
#[cfg(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64")))]
fn has_gfni() -> bool {
    std::is_x86_feature_detected!("gfni")
}
#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn has_ssse3() -> bool {
    cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "ssse3"))
}
#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn has_avx2() -> bool {
    cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "avx2"))
}
#[cfg(not(all(feature = "std", any(target_arch = "x86", target_arch = "x86_64"))))]
fn has_gfni() -> bool {
    cfg!(all(any(target_arch = "x86", target_arch = "x86_64"), target_feature = "gfni"))
}

/// The kernel for one field and tier, with its per-symbol multiplication
/// tables.
#[derive(Debug)]
pub(crate) struct Kernel {
    tier: Tier,
    /// Per symbol `s`: `s * v` for `v` in `0..16` (low nibble table),
    /// then `s * (v << 4)` for `v` in `0..16` (high nibble table).
    nibble: Vec<[u8; 32]>, // len == 256 for Ssse3/Avx2
    /// Per symbol `s`: multiplication by `s` as a `gf2p8affineqb` matrix.
    affine: Vec<u64>, // len == 256 for Gfni
}

impl Kernel {
    /// Builds the kernel for `field`, which must be a GF(256), or
    /// returns `None` if `tier` is [`Tier::Scalar`] or not available.
    pub(crate) fn new(field: &GaloisField, tier: Tier) -> Option<Kernel> {
        if tier == Tier::Scalar || !tier.is_available() || field.field_size() != 256 {
            return None;
        }
        let mut kernel = Kernel { tier, nibble: Vec::new(), affine: Vec::new() };
        match tier {
            Tier::Ssse3 | Tier::Avx2 => {
                kernel.nibble = (0..=255u8)
                    .map(|s| {
                        let mut t = [0u8; 32];
                        for v in 0..16u8 {
                            t[v as usize] = field.mul(s, v);
                            t[16 + v as usize] = field.mul(s, v << 4);
                        }
                        t
                    })
                    .collect();
            }
            Tier::Gfni => kernel.affine = (0..=255u8).map(|s| affine_matrix(field, s)).collect(),
            Tier::Scalar => unreachable!(),
        }
        Some(kernel)
    }

    #[cfg(any(test, feature = "bench-internals"))]
    pub(crate) fn tier(&self) -> Tier {
        self.tier
    }

    /// `out = sum_j c_j * rows[j * stride..j * stride + BLOCK]`, where
    /// `c_j = coeffs[j * step]` (`step > 1` reads interleaved
    /// coefficients in place). Returns a mask of the zero symbols of
    /// `out`, bit `i` for `out[i]`.
    #[inline]
    pub(crate) fn mul_acc(
        &self,
        coeffs: &[u8],
        step: usize,
        rows: &[u8],
        stride: usize,
        out: &mut [u8; BLOCK],
    ) -> u32 {
        assert!(step > 0);
        if let Some(last) = coeffs.len().div_ceil(step).checked_sub(1) {
            assert!(rows.len() >= last * stride + BLOCK, "rows too short");
        }
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        // SAFETY: `self.tier` was confirmed by `Tier::is_available` in
        // `Kernel::new`, its table was built there, and the assert above
        // keeps all row loads within `rows`.
        unsafe {
            match self.tier {
                Tier::Ssse3 => x86::mul_acc_ssse3(&self.nibble, coeffs, step, rows.as_ptr(), stride, out),
                Tier::Avx2 => x86::mul_acc_avx2(&self.nibble, coeffs, step, rows.as_ptr(), stride, out),
                Tier::Gfni => x86::mul_acc_gfni(&self.affine, coeffs, step, rows.as_ptr(), stride, out),
                Tier::Scalar => unreachable!(),
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        unreachable!()
    }
}

/// Multiplication by `s` as the 8x8 bit matrix `gf2p8affineqb` takes:
/// bit `i` of the product is the parity of `x & matrix.byte[7 - i]`, so
/// byte `7 - i` has bit `j` set when `s * alpha^j` has bit `i` set.
fn affine_matrix(field: &GaloisField, s: u8) -> u64 {
    let mut matrix = 0u64;
    for j in 0..8 {
        let column = field.mul(s, 1 << j);
        for i in 0..8 {
            if column & (1 << i) != 0 {
                matrix |= 1u64 << (8 * (7 - i) + j);
            }
        }
    }
    matrix
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
mod x86 {
    #[cfg(target_arch = "x86")]
    use core::arch::x86::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;

    use super::BLOCK;

    /// Runs `$body` with `$j` and `$s` bound to each index and coefficient
    /// `coeffs[j * step]`, with a plain loop for the common `step == 1`.
    macro_rules! for_each_coeff {
        ($coeffs:expr, $step:expr, |$j:ident, $s:ident| $body:block) => {
            if $step == 1 {
                for ($j, &$s) in $coeffs.iter().enumerate() $body
            } else {
                for ($j, &$s) in $coeffs.iter().step_by($step).enumerate() $body
            }
        };
    }

    /// # Safety
    /// Needs SSSE3, `step > 0`. `nibble` must have 256 entries, and
    /// `rows` must be readable for
    /// `(coeffs.len().div_ceil(step) - 1) * stride + BLOCK` bytes.
    #[target_feature(enable = "ssse3")]
    pub(super) unsafe fn mul_acc_ssse3(
        nibble: &[[u8; 32]],
        coeffs: &[u8],
        step: usize,
        rows: *const u8,
        stride: usize,
        out: &mut [u8; BLOCK],
    ) -> u32 {
        let mask = _mm_set1_epi8(0x0f);
        let mut acc0 = _mm_setzero_si128();
        let mut acc1 = _mm_setzero_si128();
        for_each_coeff!(coeffs, step, |j, s| {
            if s == 0 {
                continue;
            }
            let t = nibble.get_unchecked(s as usize).as_ptr() as *const __m128i;
            let lo = _mm_loadu_si128(t);
            let hi = _mm_loadu_si128(t.add(1));
            let r = rows.add(j * stride) as *const __m128i;
            let v0 = _mm_loadu_si128(r);
            let v1 = _mm_loadu_si128(r.add(1));
            let p0 = _mm_xor_si128(
                _mm_shuffle_epi8(lo, _mm_and_si128(v0, mask)),
                _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16(v0, 4), mask)),
            );
            let p1 = _mm_xor_si128(
                _mm_shuffle_epi8(lo, _mm_and_si128(v1, mask)),
                _mm_shuffle_epi8(hi, _mm_and_si128(_mm_srli_epi16(v1, 4), mask)),
            );
            acc0 = _mm_xor_si128(acc0, p0);
            acc1 = _mm_xor_si128(acc1, p1);
        });
        let o = out.as_mut_ptr() as *mut __m128i;
        _mm_storeu_si128(o, acc0);
        _mm_storeu_si128(o.add(1), acc1);
        let zero = _mm_setzero_si128();
        let z0 = _mm_movemask_epi8(_mm_cmpeq_epi8(acc0, zero)) as u32 & 0xffff;
        let z1 = _mm_movemask_epi8(_mm_cmpeq_epi8(acc1, zero)) as u32 & 0xffff;
        z0 | (z1 << 16)
    }

    /// # Safety
    /// Needs AVX2. Otherwise as [`mul_acc_ssse3`].
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn mul_acc_avx2(
        nibble: &[[u8; 32]],
        coeffs: &[u8],
        step: usize,
        rows: *const u8,
        stride: usize,
        out: &mut [u8; BLOCK],
    ) -> u32 {
        let mask = _mm256_set1_epi8(0x0f);
        let mut acc = _mm256_setzero_si256();
        for_each_coeff!(coeffs, step, |j, s| {
            if s == 0 {
                continue;
            }
            let t = nibble.get_unchecked(s as usize).as_ptr() as *const __m128i;
            let lo = _mm256_broadcastsi128_si256(_mm_loadu_si128(t));
            let hi = _mm256_broadcastsi128_si256(_mm_loadu_si128(t.add(1)));
            let v = _mm256_loadu_si256(rows.add(j * stride) as *const __m256i);
            let p = _mm256_xor_si256(
                _mm256_shuffle_epi8(lo, _mm256_and_si256(v, mask)),
                _mm256_shuffle_epi8(hi, _mm256_and_si256(_mm256_srli_epi16(v, 4), mask)),
            );
            acc = _mm256_xor_si256(acc, p);
        });
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc);
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(acc, _mm256_setzero_si256())) as u32
    }

    /// # Safety
    /// Needs AVX2 and GFNI. `affine` must have 256 entries, `rows` as for
    /// [`mul_acc_ssse3`].
    #[target_feature(enable = "avx2,gfni")]
    pub(super) unsafe fn mul_acc_gfni(
        affine: &[u64],
        coeffs: &[u8],
        step: usize,
        rows: *const u8,
        stride: usize,
        out: &mut [u8; BLOCK],
    ) -> u32 {
        let mut acc = _mm256_setzero_si256();
        for_each_coeff!(coeffs, step, |j, s| {
            if s == 0 {
                continue;
            }
            let m = _mm256_set1_epi64x(*affine.get_unchecked(s as usize) as i64);
            let v = _mm256_loadu_si256(rows.add(j * stride) as *const __m256i);
            acc = _mm256_xor_si256(acc, _mm256_gf2p8affine_epi64_epi8::<0>(v, m));
        });
        _mm256_storeu_si256(out.as_mut_ptr() as *mut __m256i, acc);
        _mm256_movemask_epi8(_mm256_cmpeq_epi8(acc, _mm256_setzero_si256())) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    use proptest::prelude::*;

    fn reference(field: &GaloisField, coeffs: &[u8], step: usize, rows: &[u8], stride: usize) -> [u8; BLOCK] {
        let mut out = [0u8; BLOCK];
        for (j, &s) in coeffs.iter().step_by(step).enumerate() {
            for (o, &v) in out.iter_mut().zip(&rows[j * stride..j * stride + BLOCK]) {
                *o ^= field.mul(s, v);
            }
        }
        out
    }

    #[test]
    fn affine_matrices_multiply_like_the_field() {
        let field = GaloisField::new(0x187);
        // Emulates gf2p8affineqb bit by bit.
        for s in 0..=255u8 {
            let m = affine_matrix(&field, s);
            for x in 0..=255u8 {
                let mut y = 0u8;
                for i in 0..8 {
                    let row = (m >> (8 * (7 - i))) as u8;
                    y |= (((row & x).count_ones() & 1) as u8) << i;
                }
                assert_eq!(y, field.mul(s, x), "s={s} x={x}");
            }
        }
    }

    proptest! {
        #[test]
        fn every_available_tier_matches_the_reference(
            poly in prop::sample::select(vec![0x187u16, 0x11d]),
            coeffs in proptest::collection::vec(any::<u8>(), 0..40),
            extra_stride in 0usize..40,
            step in 1usize..=8,
            rows in proptest::collection::vec(any::<u8>(), 40 * 80),
        ) {
            let field = GaloisField::new(poly);
            let stride = BLOCK + extra_stride;
            let expected = reference(&field, &coeffs, step, &rows, stride);
            let expected_mask = expected
                .iter()
                .enumerate()
                .fold(0u32, |m, (i, &v)| m | (((v == 0) as u32) << i));
            for tier in Tier::ALL {
                let Some(kernel) = Kernel::new(&field, tier) else { continue };
                let mut out = [0xa5u8; BLOCK];
                let mask = kernel.mul_acc(&coeffs, step, &rows, stride, &mut out);
                prop_assert_eq!(out, expected, "{:?}", tier);
                prop_assert_eq!(mask, expected_mask, "{:?}", tier);
            }
        }
    }
}
