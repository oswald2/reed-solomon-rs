//! Criterion benchmarks for the two main configurations this crate
//! targets: the CCSDS TM Synchronization and Channel Coding E=16 code
//! (GF(256), (255,223)) and the CCSDS AOS Frame Header Error Control
//! code (GF(16), shortened (10,6)) that motivated this crate in the
//! first place.
//!
//! Run with `cargo bench`. See `BENCHMARKS.md` for a comparison against
//! the original C (libcorrect) implementation.

use criterion::{criterion_group, criterion_main, Criterion};
use reed_solomon_rs::ccsds::dual_basis::DualBasisConverter;
use reed_solomon_rs::ccsds::{fhec, tm_channel_coding};
use reed_solomon_rs::rs::ReedSolomon;

fn bench_gf256_255_223(c: &mut Criterion) {
    let mut rs = tm_channel_coding::codec_e16();
    let msg: Vec<u8> = (0..223).map(|i| (i * 7) as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();

    // 16 errors is min_distance/2 for this code: the guaranteed
    // correction capacity, and the most expensive case decode() has to
    // handle (a longer error locator means more work in
    // Berlekamp-Massey, Chien search, and Forney's algorithm).
    let mut at_capacity = encoded.clone();
    for i in 0..16 {
        at_capacity[i * 8] ^= 0xff;
    }

    // 20 known erasures plus 5 unlocated errors (2*5 + 20 == 30 < 32):
    // exercises the modified-syndrome erasure-decoding path.
    let mut with_erasures = encoded.clone();
    let erasure_locations: Vec<u8> = (0..20).map(|i| (i * 3) as u8).collect();
    for &loc in &erasure_locations {
        with_erasures[loc as usize] ^= 0xaa;
    }
    for i in 0..5 {
        with_erasures[200 + i] ^= 0x55;
    }

    let mut group = c.benchmark_group("gf256_255_223");
    let mut out = vec![0u8; 255];
    group.bench_function("encode", |b| {
        b.iter(|| rs.encode(std::hint::black_box(&msg), &mut out).unwrap())
    });
    let mut out = vec![0u8; 223];
    group.bench_function("decode_no_errors", |b| {
        b.iter(|| rs.decode(std::hint::black_box(&encoded), &mut out).unwrap())
    });
    group.bench_function("decode_16_errors_at_capacity", |b| {
        b.iter(|| rs.decode(std::hint::black_box(&at_capacity), &mut out).unwrap())
    });
    group.bench_function("decode_20_erasures_5_errors", |b| {
        b.iter(|| {
            rs.decode_with_erasures(
                std::hint::black_box(&with_erasures),
                &erasure_locations,
                &mut out,
            )
            .unwrap()
        })
    });
    group.finish();
}

/// A CADU-sized codeblock: 1115 message bytes at E=16, interleaving
/// depth 5, as in space-rs.
fn bench_tm_e16_i5(c: &mut Criterion) {
    const DEPTH: usize = 5;
    let mut rs = tm_channel_coding::codec_e16();
    let msg: Vec<u8> = (0..1115).map(|i| (i * 7 + 3) as u8).collect();
    let mut encoded = vec![0u8; 1275];
    tm_channel_coding::encode_interleaved(&rs, DEPTH, &msg, &mut encoded).unwrap();
    // A burst of 80 symbols: 16 errors in each of the 5 codewords.
    let mut corrupted = encoded.clone();
    corrupted[..80].iter_mut().for_each(|b| *b = !*b);

    let mut group = c.benchmark_group("tm_e16_i5");
    let mut out = vec![0u8; 1275];
    group.bench_function("encode", |b| {
        b.iter(|| {
            tm_channel_coding::encode_interleaved(&rs, DEPTH, std::hint::black_box(&msg), &mut out)
                .unwrap()
        })
    });
    let mut out = vec![0u8; 1115];
    group.bench_function("decode_no_errors", |b| {
        b.iter(|| {
            tm_channel_coding::decode_interleaved(&mut rs, DEPTH, std::hint::black_box(&encoded), &mut out)
                .unwrap()
        })
    });
    group.bench_function("decode_16_errors_per_codeword", |b| {
        b.iter(|| {
            tm_channel_coding::decode_interleaved(&mut rs, DEPTH, std::hint::black_box(&corrupted), &mut out)
                .unwrap()
        })
    });
    group.finish();
}

/// Dual-basis conversion of a CADU-sized codeblock (1275 bytes).
fn bench_dual_basis(c: &mut Criterion) {
    let conv = DualBasisConverter::new();
    let data: Vec<u8> = (0..1275).map(|i| (i * 7 + 3) as u8).collect();
    let mut out = vec![0u8; 1275];
    let mut group = c.benchmark_group("dual_basis_1275");
    group.bench_function("from_dual", |b| b.iter(|| conv.from_dual(std::hint::black_box(&data), &mut out)));
    group.bench_function("to_dual_in_place", |b| b.iter(|| conv.to_dual_in_place(std::hint::black_box(&mut out))));
    group.finish();
}

fn bench_ccsds_aos_fhec(c: &mut Criterion) {
    let mut rs: ReedSolomon = fhec::codec();
    let msg = [3u8, 7, 1, 15, 0, 9];
    let parity = fhec::encode(&mut rs, &msg).unwrap();

    let mut codeword = [0u8; fhec::CODEWORD_LENGTH];
    codeword[..fhec::MESSAGE_LENGTH].copy_from_slice(&msg);
    codeword[fhec::MESSAGE_LENGTH..].copy_from_slice(&parity);

    // E=2 for this code: the guaranteed correction capacity.
    let mut at_capacity = codeword;
    at_capacity[1] ^= 0x0a;
    at_capacity[8] ^= 0x03;

    let mut group = c.benchmark_group("ccsds_aos_fhec_10_6");
    group.bench_function("encode", |b| {
        b.iter(|| fhec::encode(&mut rs, std::hint::black_box(&msg)).unwrap())
    });
    group.bench_function("decode_no_errors", |b| {
        b.iter(|| fhec::decode(&mut rs, std::hint::black_box(&codeword)).unwrap())
    });
    group.bench_function("decode_2_errors_at_capacity", |b| {
        b.iter(|| fhec::decode(&mut rs, std::hint::black_box(&at_capacity)).unwrap())
    });
    group.finish();
}

criterion_group!(benches, bench_gf256_255_223, bench_tm_e16_i5, bench_dual_basis, bench_ccsds_aos_fhec);
criterion_main!(benches);
