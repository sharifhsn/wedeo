//! Integration tests for `wedeo-symphonia` via `wedeo::default_registry()`.
//!
//! Uses locally generated samples via `wedeo-audiogen` + `/usr/bin/ffmpeg` so
//! ZERO tests silently skip:
//! - FLAC (16-bit & 24-bit): bit-exact decode vs FFmpeg `n9.1` oracle
//! - Vorbis (stereo & mono): exact sample count & >100 dB SNR vs FFmpeg `n9.1`
//! - MP3 (`libmp3lame`): aligned >100 dB SNR & bounded priming delta vs FFmpeg `n9.1`
//! - AAC (M4A & raw ADTS): aligned SNR & bounded priming delta vs FFmpeg `n9.1`
//! - AIFF (PCM): bit-exact decode vs FFmpeg `n9.1`
//! - Seeking on WAV, FLAC, and MP3 via `AudioReader::seek`
//! - Priority invariants: native WAV/PCM/MP4/MKV always win over Symphonia

use std::path::{Path, PathBuf};
use std::process::Command;

use wedeo::core::{
    CodecContext, CodecId, ErrorPolicy, FileSource, Packet, Rational, RecvStatus, SampleFormat,
    SeekTarget, SendStatus, Timestamp,
};
use wedeo::default_registry;
use wedeo_oracle::{ffmpeg_oracle_bin, find_oracle_bin, system_ffmpeg_bin, temp_test_dir};

fn wedeo_audiogen_bin() -> PathBuf {
    find_oracle_bin("wedeo-audiogen", Some("CARGO_BIN_EXE_wedeo-audiogen"))
        .expect("wedeo-audiogen binary must exist")
}

fn generate_base_wav(dir: &Path, name: &str, sample_rate: u32, channels: u32) -> PathBuf {
    let path = dir.join(name);
    let status = Command::new(wedeo_audiogen_bin())
        .args([
            path.to_str().unwrap(),
            &sample_rate.to_string(),
            &channels.to_string(),
            "1",
        ])
        .status()
        .expect("run wedeo-audiogen");
    assert!(status.success(), "wedeo-audiogen failed");
    path
}

fn transcode_fixture(src: &Path, dst: &Path, extra_args: &[&str]) {
    let mut args = vec![
        "-y",
        "-v",
        "error",
        "-i",
        src.to_str().unwrap(),
        "-t",
        "1.5",
    ];
    args.extend_from_slice(extra_args);
    args.push(dst.to_str().unwrap());

    let st = Command::new(system_ffmpeg_bin())
        .args(&args)
        .status()
        .expect("run system ffmpeg to build audio fixture");
    assert!(
        st.success(),
        "failed creating fixture {} with args {extra_args:?}",
        dst.display()
    );
}

fn decode_wedeo_samples_raw(path: &Path) -> (SampleFormat, u32, u32, Vec<u8>) {
    let reg = default_registry();
    let mut reader = wedeo::audio::AudioReader::open(&reg, path)
        .unwrap_or_else(|e| panic!("AudioReader failed on {}: {e}", path.display()));
    let mut fmt = reader.params().format;
    let sr = reader.params().sample_rate;
    let ch = reader.params().channel_layout.nb_channels;
    let mut out = Vec::new();
    while let Some(block) = reader.next_block().unwrap() {
        fmt = block.format;
        assert_eq!(block.planes.len(), 1, "expected packed output block");
        out.extend_from_slice(block.planes[0].data());
    }
    (fmt, sr, ch, out)
}

fn decode_wedeo_to_f32(path: &Path) -> Vec<f32> {
    let (fmt, _sr, _ch, raw) = decode_wedeo_samples_raw(path);
    match fmt {
        SampleFormat::F32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| f32::from_le_bytes(c))
            .collect(),
        SampleFormat::S16 => raw
            .as_chunks::<2>()
            .0
            .iter()
            .map(|&c| i16::from_le_bytes(c) as f32 / 32768.0)
            .collect(),
        SampleFormat::S32 => raw
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&c| i32::from_le_bytes(c) as f32 / 2_147_483_648.0)
            .collect(),
        other => panic!("unexpected sample format {other:?} from {}", path.display()),
    }
}

fn decode_ffmpeg_oracle_raw(path: &Path, fmt: &str, acodec: &str) -> Vec<u8> {
    let out = Command::new(ffmpeg_oracle_bin())
        .args([
            "-v",
            "error",
            "-bitexact",
            "-i",
            path.to_str().unwrap(),
            "-f",
            fmt,
            "-acodec",
            acodec,
            "-",
        ])
        .output()
        .expect("run ffmpeg oracle audio decode");
    assert!(
        out.status.success(),
        "ffmpeg oracle decode failed for {}: {}",
        path.display(),
        String::from_utf8_lossy(&out.stderr)
    );
    out.stdout
}

fn decode_ffmpeg_oracle_f32(path: &Path) -> Vec<f32> {
    let bytes = decode_ffmpeg_oracle_raw(path, "f32le", "pcm_f32le");
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&c| f32::from_le_bytes(c))
        .collect()
}

fn compute_snr_f32(reference: &[f32], actual: &[f32]) -> (f64, f32) {
    let n = reference.len().min(actual.len());
    let mut sig_pow = 0.0f64;
    let mut err_pow = 0.0f64;
    let mut max_err = 0.0f32;
    for i in 0..n {
        let r = reference[i] as f64;
        let e = (reference[i] - actual[i]) as f64;
        sig_pow += r * r;
        err_pow += e * e;
        max_err = max_err.max((reference[i] - actual[i]).abs());
    }
    let snr = if err_pow == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig_pow / err_pow).log10()
    };
    (snr, max_err)
}

fn find_alignment_offset(reference: &[f32], actual: &[f32], max_offset: usize) -> usize {
    let search_len = reference.len().min(4096);
    let mut best_offset = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for off in 0..=max_offset {
        if off + search_len > actual.len() {
            break;
        }
        let mut corr = 0.0f64;
        for i in 0..search_len {
            corr += reference[i] as f64 * actual[off + i] as f64;
        }
        if corr > best_corr {
            best_corr = corr;
            best_offset = off;
        }
    }
    best_offset
}

#[test]
fn symphonia_priority_and_registration_invariants() {
    let reg = default_registry();

    // Native WAV + PCM + MP4 + MKV must always beat Symphonia.
    let wav_head = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00";
    assert_eq!(
        reg.probe_demuxer(wav_head, Some("wav"))
            .unwrap()
            .descriptor()
            .name,
        "wav"
    );
    assert_eq!(
        reg.find_decoder(CodecId::PcmS16le)
            .unwrap()
            .descriptor()
            .name,
        "pcm_s16le"
    );

    // Symphonia fallback decoders & demuxer are present.
    assert!(reg.find_demuxer_by_name("symphonia").is_some());
    for (codec_id, expected_name) in [
        (CodecId::Flac, "flac_symphonia"),
        (CodecId::Mp3, "mp3_symphonia"),
        (CodecId::Aac, "aac_symphonia"),
        (CodecId::Vorbis, "vorbis_symphonia"),
        (CodecId::Alac, "alac_symphonia"),
        (CodecId::Opus, "opus_symphonia"),
    ] {
        let dec = reg
            .find_decoder(codec_id)
            .unwrap_or_else(|| panic!("missing decoder for {codec_id:?}"));
        assert_eq!(dec.descriptor().name, expected_name);
    }

    // Symphonia probes OGG and FLAC cleanly.
    assert_eq!(
        reg.probe_demuxer(b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00", Some("ogg"))
            .unwrap()
            .descriptor()
            .name,
        "symphonia"
    );
    assert_eq!(
        reg.probe_demuxer(b"fLaC\x00\x00\x00\x22", Some("flac"))
            .unwrap()
            .descriptor()
            .name,
        "symphonia"
    );
}

#[test]
fn symphonia_flac_16bit_and_24bit_decode_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("symphonia_flac");
    let wav = generate_base_wav(&dir, "src.wav", 44100, 2);

    let flac16 = dir.join("test_16bit.flac");
    transcode_fixture(&wav, &flac16, &["-c:a", "flac", "-sample_fmt", "s16"]);

    let (fmt16, sr16, ch16, wd16) = decode_wedeo_samples_raw(&flac16);
    assert_eq!((sr16, ch16), (44100, 2));
    let ff16 = match fmt16 {
        SampleFormat::S16 => decode_ffmpeg_oracle_raw(&flac16, "s16le", "pcm_s16le"),
        SampleFormat::S32 => decode_ffmpeg_oracle_raw(&flac16, "s32le", "pcm_s32le"),
        other => panic!("unexpected FLAC 16-bit format: {other:?}"),
    };
    assert_eq!(wd16.len(), ff16.len(), "FLAC 16-bit length mismatch");
    assert_eq!(
        wd16, ff16,
        "FLAC 16-bit decode not bit-exact vs FFmpeg n9.1"
    );

    let flac24 = dir.join("test_24bit.flac");
    transcode_fixture(
        &wav,
        &flac24,
        &["-c:a", "flac", "-sample_fmt", "s32", "-ar", "48000"],
    );
    let (fmt24, sr24, ch24, wd24) = decode_wedeo_samples_raw(&flac24);
    assert_eq!((sr24, ch24), (48000, 2));
    assert_eq!(fmt24, SampleFormat::S32);
    let ff24 = decode_ffmpeg_oracle_raw(&flac24, "s32le", "pcm_s32le");
    assert_eq!(wd24.len(), ff24.len(), "FLAC 24-bit length mismatch");
    assert_eq!(
        wd24, ff24,
        "FLAC 24-bit decode not bit-exact vs FFmpeg n9.1"
    );
}

#[test]
fn symphonia_vorbis_stereo_and_mono_decode_near_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("symphonia_vorbis");
    let wav_stereo = generate_base_wav(&dir, "src_stereo.wav", 44100, 2);
    let wav_mono = generate_base_wav(&dir, "src_mono.wav", 44100, 1);

    for (name, src, ch) in [
        ("stereo.ogg", &wav_stereo, 2u32),
        ("mono.ogg", &wav_mono, 1u32),
    ] {
        let ogg = dir.join(name);
        transcode_fixture(src, &ogg, &["-c:a", "libvorbis", "-q:a", "4"]);

        let (fmt, sr, got_ch, _) = decode_wedeo_samples_raw(&ogg);
        assert_eq!(fmt, SampleFormat::F32);
        assert_eq!(sr, 44100);
        assert_eq!(got_ch, ch);

        let wd = decode_wedeo_to_f32(&ogg);
        let ff = decode_ffmpeg_oracle_f32(&ogg);
        assert_eq!(wd.len(), ff.len(), "Vorbis {name} sample count differs");

        let (snr, max_err) = compute_snr_f32(&ff, &wd);
        assert!(snr > 100.0, "Vorbis {name} SNR too low: {snr:.2} dB");
        assert!(
            max_err < 1e-4,
            "Vorbis {name} max error too high: {max_err}"
        );
    }
}

#[test]
fn symphonia_mp3_decode_aligned_snr_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("symphonia_mp3");
    let wav = generate_base_wav(&dir, "src.wav", 44100, 2);
    let mp3 = dir.join("test.mp3");
    transcode_fixture(&wav, &mp3, &["-c:a", "libmp3lame", "-b:a", "192k"]);

    let wd = decode_wedeo_to_f32(&mp3);
    let ff = decode_ffmpeg_oracle_f32(&mp3);
    assert!(!wd.is_empty());

    let count_diff = (wd.len() as i64 - ff.len() as i64).unsigned_abs() as usize;
    assert!(
        count_diff < 1152 * 2 * 4,
        "MP3 sample count diff too large: {count_diff}"
    );

    let offset = find_alignment_offset(&ff, &wd, 4096);
    let len = ff.len().min(wd.len() - offset);
    let (snr, max_err) = compute_snr_f32(&ff[..len], &wd[offset..offset + len]);
    assert!(snr > 100.0, "MP3 aligned SNR too low: {snr:.2} dB");
    assert!(max_err < 1e-3, "MP3 max err too high: {max_err}");
}

#[test]
fn symphonia_aac_m4a_and_adts_decode_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("symphonia_aac");
    let wav = generate_base_wav(&dir, "src.wav", 44100, 2);

    let m4a = dir.join("test.m4a");
    transcode_fixture(&wav, &m4a, &["-c:a", "aac", "-b:a", "192k"]);
    let wd_m4a = decode_wedeo_to_f32(&m4a);
    let ff_m4a = decode_ffmpeg_oracle_f32(&m4a);
    assert!(!wd_m4a.is_empty());
    let m4a_off = find_alignment_offset(&ff_m4a, &wd_m4a, 6000);
    let m4a_len = ff_m4a.len().min(wd_m4a.len() - m4a_off);
    let (m4a_snr, _) = compute_snr_f32(&ff_m4a[..m4a_len], &wd_m4a[m4a_off..m4a_off + m4a_len]);
    assert!(
        m4a_snr > 40.0,
        "M4A AAC aligned SNR too low: {m4a_snr:.2} dB"
    );

    let adts = dir.join("test.aac");
    transcode_fixture(&wav, &adts, &["-c:a", "aac", "-b:a", "192k", "-f", "adts"]);
    let reg = default_registry();
    let adts_bytes = std::fs::read(&adts).unwrap();
    let head = &adts_bytes[..adts_bytes.len().min(8192)];
    let demux_factory = reg.probe_demuxer(head, Some("aac")).unwrap();
    assert_eq!(demux_factory.descriptor().name, "aac");
    let mut demuxer = demux_factory
        .open(
            Box::new(FileSource::open(&adts).unwrap()),
            ErrorPolicy::Strict,
        )
        .unwrap();
    let track = demuxer.header().tracks[0].clone();
    assert_eq!(track.codec_id, CodecId::Aac);
    let mut decoder = reg
        .find_decoder(CodecId::Aac)
        .unwrap()
        .open(&track, CodecContext::default())
        .unwrap();

    let mut wd_adts = Vec::new();
    let push_block = |b: wedeo::core::AudioBlock, out: &mut Vec<f32>| {
        assert_eq!(b.format, SampleFormat::F32);
        for &c in b.planes[0].data().as_chunks::<4>().0 {
            out.push(f32::from_le_bytes(c));
        }
    };
    loop {
        match demuxer.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                // `AacAdtsFramer` emits ADTS-framed packets; strip ADTS header for raw AAC AU decoder.
                let hdr = wedeo::syntax::aac::parse_adts_header(pkt.data.data()).unwrap();
                let mut raw_pkt = Packet::new(
                    pkt.data.subslice(hdr.header_size()..pkt.len()).unwrap(),
                    pkt.track_id,
                    pkt.time_base,
                );
                raw_pkt.pts = pkt.pts;
                raw_pkt.dts = pkt.dts;
                raw_pkt.duration = pkt.duration;
                assert_eq!(
                    decoder.send_packet(Some(&raw_pkt)).unwrap(),
                    SendStatus::Accepted
                );
                while let RecvStatus::Ready(u) = decoder.receive().unwrap() {
                    push_block(u.into_audio().unwrap(), &mut wd_adts);
                }
            }
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }
    assert_eq!(decoder.send_packet(None).unwrap(), SendStatus::Accepted);
    while let RecvStatus::Ready(u) = decoder.receive().unwrap() {
        push_block(u.into_audio().unwrap(), &mut wd_adts);
    }

    let ff_adts = decode_ffmpeg_oracle_f32(&adts);
    assert_eq!(
        wd_adts.len(),
        ff_adts.len(),
        "ADTS AAC sample count differs: wd={}, ff={}",
        wd_adts.len(),
        ff_adts.len()
    );
    let (adts_snr, adts_max_err) = compute_snr_f32(&ff_adts, &wd_adts);
    assert!(
        adts_snr > 100.0,
        "ADTS AAC SNR too low: {adts_snr:.2} dB (max_err={adts_max_err})"
    );
    assert!(
        adts_max_err < 1e-4,
        "ADTS AAC max err too high: {adts_max_err}"
    );
}

#[test]
fn symphonia_aiff_and_seek_roundtrips() {
    let dir = temp_test_dir("symphonia_seek_aiff");
    let wav = generate_base_wav(&dir, "src.wav", 44100, 2);

    // AIFF PCM bit-exact check
    let aiff = dir.join("test.aiff");
    transcode_fixture(&wav, &aiff, &["-c:a", "pcm_s16be"]);
    let (fmt, _, _, wd_aiff) = decode_wedeo_samples_raw(&aiff);
    assert_eq!(fmt, SampleFormat::S16);
    let ff_aiff = decode_ffmpeg_oracle_raw(&aiff, "s16le", "pcm_s16le");
    assert_eq!(wd_aiff, ff_aiff, "AIFF PCM decode mismatch vs FFmpeg n9.1");

    // Seek check on WAV, FLAC, MP3
    let flac = dir.join("seek.flac");
    transcode_fixture(&wav, &flac, &["-c:a", "flac"]);
    let mp3 = dir.join("seek.mp3");
    transcode_fixture(&wav, &mp3, &["-c:a", "libmp3lame", "-b:a", "128k"]);

    let reg = default_registry();
    for path in [&wav, &flac, &mp3] {
        let mut reader = wedeo::audio::AudioReader::open(&reg, path).unwrap();
        let target_sec = 0.5f64;
        let tb = Rational::new(1, reader.params().sample_rate as i32);
        let ts = Timestamp::new((target_sec * tb.den as f64) as i64);
        reader
            .seek(SeekTarget {
                track: None,
                timestamp: ts,
                time_base: tb,
                backward: true,
            })
            .unwrap_or_else(|e| panic!("seek failed on {}: {e}", path.display()));
        let block = reader
            .next_block()
            .unwrap()
            .unwrap_or_else(|| panic!("no block after seek on {}", path.display()));
        assert!(block.nb_samples > 0);
        if let Some(pts_sec) = block.pts.to_seconds(block.time_base) {
            assert!(
                (pts_sec - target_sec).abs() < 1.0,
                "seek landed too far on {}: {pts_sec}s vs {target_sec}s",
                path.display()
            );
        }
    }
}
