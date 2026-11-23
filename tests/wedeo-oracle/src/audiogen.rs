//! Bitexact port of FFmpeg's tests/audiogen.c
//!
//! Generates a synthetic test WAV file using integer-only math and a
//! deterministic PRNG. Output is bit-identical to FFmpeg's audiogen.

use std::env;
use std::fs::File;
use std::io::{BufWriter, Write};

const MAX_CHANNELS: usize = 12;
const FRAC_BITS: i32 = 16;
const FRAC_ONE: i32 = 1 << FRAC_BITS;
const COS_TABLE_BITS: i32 = 7;
const CSHIFT: i32 = FRAC_BITS - COS_TABLE_BITS - 2;

static COS_TABLE: [u16; 130] = [
    0x8000, 0x7ffe, 0x7ff6, 0x7fea, 0x7fd9, 0x7fc2, 0x7fa7, 0x7f87, 0x7f62, 0x7f38, 0x7f0a, 0x7ed6,
    0x7e9d, 0x7e60, 0x7e1e, 0x7dd6, 0x7d8a, 0x7d3a, 0x7ce4, 0x7c89, 0x7c2a, 0x7bc6, 0x7b5d, 0x7aef,
    0x7a7d, 0x7a06, 0x798a, 0x790a, 0x7885, 0x77fb, 0x776c, 0x76d9, 0x7642, 0x75a6, 0x7505, 0x7460,
    0x73b6, 0x7308, 0x7255, 0x719e, 0x70e3, 0x7023, 0x6f5f, 0x6e97, 0x6dca, 0x6cf9, 0x6c24, 0x6b4b,
    0x6a6e, 0x698c, 0x68a7, 0x67bd, 0x66d0, 0x65de, 0x64e9, 0x63ef, 0x62f2, 0x61f1, 0x60ec, 0x5fe4,
    0x5ed7, 0x5dc8, 0x5cb4, 0x5b9d, 0x5a82, 0x5964, 0x5843, 0x571e, 0x55f6, 0x54ca, 0x539b, 0x5269,
    0x5134, 0x4ffb, 0x4ec0, 0x4d81, 0x4c40, 0x4afb, 0x49b4, 0x486a, 0x471d, 0x45cd, 0x447b, 0x4326,
    0x41ce, 0x4074, 0x3f17, 0x3db8, 0x3c57, 0x3af3, 0x398d, 0x3825, 0x36ba, 0x354e, 0x33df, 0x326e,
    0x30fc, 0x2f87, 0x2e11, 0x2c99, 0x2b1f, 0x29a4, 0x2827, 0x26a8, 0x2528, 0x23a7, 0x2224, 0x209f,
    0x1f1a, 0x1d93, 0x1c0c, 0x1a83, 0x18f9, 0x176e, 0x15e2, 0x1455, 0x12c8, 0x113a, 0x0fab, 0x0e1c,
    0x0c8c, 0x0afb, 0x096b, 0x07d9, 0x0648, 0x04b6, 0x0324, 0x0192, 0x0000, 0x0000,
];

fn myrnd(seed: &mut u32, n: u32) -> u32 {
    *seed = seed.wrapping_mul(314159).wrapping_add(1);
    if n == 256 { *seed >> 24 } else { *seed % n }
}

fn int_cos(mut a: i32) -> i32 {
    a &= FRAC_ONE - 1;
    if a >= FRAC_ONE / 2 {
        a = FRAC_ONE - a;
    }
    let mut neg = 0i32;
    if a > FRAC_ONE / 4 {
        neg = -1;
        a = FRAC_ONE / 2 - a;
    }
    let idx = (a >> CSHIFT) as usize;
    let p0 = COS_TABLE[idx] as i32;
    let p1 = COS_TABLE[idx + 1] as i32;
    let f = a & ((1 << CSHIFT) - 1);
    let mut v = p0 + (((p1 - p0) * f + (1 << (CSHIFT - 1))) >> CSHIFT);
    v = (v ^ neg) - neg;
    v <<= FRAC_BITS - 15;
    v
}

fn put16(w: &mut impl Write, v: i16) {
    w.write_all(&v.to_le_bytes()).unwrap();
}

fn put32(w: &mut impl Write, v: u32) {
    w.write_all(&v.to_le_bytes()).unwrap();
}

fn put_wav_header(w: &mut impl Write, sample_rate: u32, channels: u16, nb_samples: u32) {
    let block_align: u16 = 2 * channels; // SAMPLE_SIZE=2
    let data_size = block_align as u32 * nb_samples;

    w.write_all(b"RIFF").unwrap();
    put32(w, 38 + data_size); // HEADER_SIZE=38
    w.write_all(b"WAVEfmt ").unwrap();
    put32(w, 18); // FMT_SIZE=18
    put16(w, 1); // WFORMAT_PCM
    put16(w, channels as i16);
    put32(w, sample_rate);
    put32(w, block_align as u32 * sample_rate);
    put16(w, block_align as i16);
    put16(w, 16); // SAMPLE_SIZE * 8
    put16(w, 0); // cbSize
    w.write_all(b"data").unwrap();
    put32(w, data_size);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 || args.len() > 5 {
        eprintln!(
            "usage: {} file [<sample rate> [<channels>] [<random seed>]]",
            args[0]
        );
        eprintln!("generate a test raw 16 bit audio stream");
        eprintln!("If the file extension is .wav a WAVE header will be added.");
        eprintln!("default: 44100 Hz stereo");
        std::process::exit(1);
    }

    let sample_rate: i32 = if args.len() > 2 {
        args[2].parse().unwrap()
    } else {
        44100
    };
    let nb_channels: i32 = if args.len() > 3 {
        args[3].parse().unwrap()
    } else {
        2
    };
    let mut seed: u32 = if args.len() > 4 {
        args[4].parse().unwrap()
    } else {
        1
    };

    let file = File::create(&args[1]).unwrap();
    let mut w = BufWriter::new(file);

    if args[1].ends_with(".wav") {
        put_wav_header(
            &mut w,
            sample_rate as u32,
            nb_channels as u16,
            6 * sample_rate as u32,
        );
    }

    // 1 second of single freq sine at 1000 Hz
    let mut a: i32 = 0;
    for _i in 0..sample_rate {
        let v = (int_cos(a) * 10000) >> FRAC_BITS;
        for _j in 0..nb_channels {
            put16(&mut w, v as i16);
        }
        a += (1000 * FRAC_ONE) / sample_rate;
    }

    // 1 second of varying frequency between 100 and 10000 Hz
    a = 0;
    for i in 0..sample_rate {
        let v = (int_cos(a) * 10000) >> FRAC_BITS;
        for _j in 0..nb_channels {
            put16(&mut w, v as i16);
        }
        let f = 100 + (((10000 - 100) * i) / sample_rate);
        a += (f * FRAC_ONE) / sample_rate;
    }

    // 0.5 second of low amplitude white noise
    for _i in 0..sample_rate / 2 {
        let v = myrnd(&mut seed, 20000) as i32 - 10000;
        for _j in 0..nb_channels {
            put16(&mut w, v as i16);
        }
    }

    // 0.5 second of high amplitude white noise
    for _i in 0..sample_rate / 2 {
        let v = myrnd(&mut seed, 65535) as i32 - 32768;
        for _j in 0..nb_channels {
            put16(&mut w, v as i16);
        }
    }

    // 1 second of unrelated ramps for each channel
    let mut taba = [0i32; MAX_CHANNELS];
    let mut tabf1 = [0i32; MAX_CHANNELS];
    let mut tabf2 = [0i32; MAX_CHANNELS];
    for j in 0..nb_channels as usize {
        taba[j] = 0;
        tabf1[j] = 100 + myrnd(&mut seed, 5000) as i32;
        tabf2[j] = 100 + myrnd(&mut seed, 5000) as i32;
    }
    for i in 0..sample_rate {
        for j in 0..nb_channels as usize {
            let v = (int_cos(taba[j]) * 10000) >> FRAC_BITS;
            put16(&mut w, v as i16);
            let f = tabf1[j] + (((tabf2[j] - tabf1[j]) * i) / sample_rate);
            taba[j] += (f * FRAC_ONE) / sample_rate;
        }
    }

    // 2 seconds of 500 Hz with varying volume
    a = 0;
    let mut ampa: i32 = 0;
    for _i in 0..2 * sample_rate {
        for j in 0..nb_channels {
            let mut amp = ((FRAC_ONE + int_cos(ampa)) * 5000) >> FRAC_BITS;
            if j & 1 != 0 {
                amp = 10000 - amp;
            }
            let v = (int_cos(a) * amp) >> FRAC_BITS;
            put16(&mut w, v as i16);
            a += (500 * FRAC_ONE) / sample_rate;
            ampa += (2 * FRAC_ONE) / sample_rate;
        }
    }

    w.flush().unwrap();
}
