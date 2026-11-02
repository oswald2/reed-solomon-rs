//! The CCSDS TM E=16 code with every SIMD tier this machine has, and its
//! decode stages one by one, to see where decode time goes. Run with
//! `cargo bench --features std,bench-internals --bench decode_stages`
//! (without `std`, only the tiers enabled at compile time are found).

use std::hint::black_box;

use criterion::{criterion_group, criterion_main, Criterion};
use reed_solomon_rs::ccsds::tm_channel_coding;
use reed_solomon_rs::rs::bench_internals::{set_tier, Stages, Tier};

fn bench_tiers(c: &mut Criterion) {
    let mut rs = tm_channel_coding::codec_e16();
    let msg: Vec<u8> = (0..223).map(|i| (i * 7) as u8).collect();
    let mut encoded = vec![0u8; 255];
    rs.encode(&msg, &mut encoded).unwrap();
    let mut corrupted = encoded.clone();
    for i in 0..16 {
        corrupted[i * 8] ^= 0xff;
    }

    const DEPTH: usize = 5;
    let block_msg: Vec<u8> = (0..1115).map(|i| (i * 7 + 3) as u8).collect();
    let mut block = vec![0u8; 1275];
    tm_channel_coding::encode_interleaved(&rs, DEPTH, &block_msg, &mut block).unwrap();
    let mut corrupted_block = block.clone();
    corrupted_block[..80].iter_mut().for_each(|b| *b = !*b);

    for tier in Tier::ALL {
        if !set_tier(&mut rs, tier) {
            continue;
        }
        let mut group = c.benchmark_group(format!("{tier:?}"));
        let mut out = vec![0u8; 255];
        group.bench_function("encode", |b| b.iter(|| rs.encode(black_box(&msg), &mut out).unwrap()));
        group.bench_function("decode_no_errors", |b| {
            b.iter(|| rs.decode(black_box(&encoded), &mut out).unwrap())
        });
        group.bench_function("decode_16_errors", |b| {
            b.iter(|| rs.decode(black_box(&corrupted), &mut out).unwrap())
        });
        let mut out = vec![0u8; 1275];
        group.bench_function("i5_encode", |b| {
            b.iter(|| tm_channel_coding::encode_interleaved(&rs, DEPTH, black_box(&block_msg), &mut out).unwrap())
        });
        group.bench_function("i5_decode_no_errors", |b| {
            b.iter(|| tm_channel_coding::decode_interleaved(&mut rs, DEPTH, black_box(&block), &mut out).unwrap())
        });
        group.bench_function("i5_decode_16_errors", |b| {
            b.iter(|| {
                tm_channel_coding::decode_interleaved(&mut rs, DEPTH, black_box(&corrupted_block), &mut out).unwrap()
            })
        });

        // The stages, each leaving its state in place for the next one.
        let mut st = Stages::new(&mut rs);
        group.bench_function("stage_remainder", |b| b.iter(|| st.remainder(black_box(&encoded)).unwrap()));
        assert!(st.remainder(&corrupted).unwrap());
        group.bench_function("stage_syndromes", |b| b.iter(|| st.syndromes()));
        group.bench_function("stage_berlekamp_massey", |b| b.iter(|| black_box(st.berlekamp_massey())));
        let order = st.berlekamp_massey();
        assert_eq!(order, 16);
        group.bench_function("stage_chien", |b| b.iter(|| black_box(st.chien(black_box(order)))));
        assert!(st.chien(order));
        group.bench_function("stage_forney", |b| b.iter(|| st.forney(black_box(order))));
        group.finish();
    }
}

criterion_group!(benches, bench_tiers);
criterion_main!(benches);
