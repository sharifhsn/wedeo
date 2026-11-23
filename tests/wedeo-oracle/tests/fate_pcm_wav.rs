//! FATE-style bit-exact oracle tests for native `WavDemuxer` + `PcmDecoder`
//! + `PcmEncoder` + `WavMuxer` + `wedeo-audiogen` against FFmpeg `n9.1`.

use std::path::{Path, PathBuf};
use std::process::Command;

use wedeo::core::{
    CodecContext, CodecId, ContainerHeader, ErrorPolicy, FileSink, FileSource, MediaUnit, PassMode,
    RecvStatus, SendStatus, StreamParams, TrackDescriptor, TrackId, adler32_ffmpeg,
};
use wedeo::{default_registry, native_registry};
use wedeo_oracle::{
    FramecrcCompareOpts, compare_framecrc_full, ffmpeg_oracle_bin, find_oracle_bin, md5_bytes,
    parse_framecrc_headers, parse_framecrc_record, run_ffmpeg_packet_framecrc, run_wedeo_framecrc,
    temp_test_dir, workspace_root,
};

fn wedeo_audiogen_bin() -> PathBuf {
    find_oracle_bin("wedeo-audiogen", Some("CARGO_BIN_EXE_wedeo-audiogen"))
        .expect("wedeo-audiogen binary must exist")
}

fn generate_wav(dir: &Path, name: &str, sample_rate: u32, channels: u32, seed: u32) -> PathBuf {
    let path = dir.join(name);
    let status = Command::new(wedeo_audiogen_bin())
        .args([
            path.to_str().unwrap(),
            &sample_rate.to_string(),
            &channels.to_string(),
            &seed.to_string(),
        ])
        .status()
        .expect("run wedeo-audiogen");
    assert!(status.success(), "wedeo-audiogen failed");
    path
}

/// Demux all raw packet payloads via native `WavDemuxer`, verify continuous
/// `(dts, pts, duration)` timestamps, and verify chunked `adler32_ffmpeg` matches
/// FFmpeg `n9.1` `-c copy -f framecrc` byte-for-byte.
fn assert_wav_packets_match_ffmpeg(wav: &Path, expected_codec: CodecId) {
    let reg = default_registry();
    let wedeo_lines = run_wedeo_framecrc(wav).expect("wedeo-framecrc");
    let ffmpeg_lines = run_ffmpeg_packet_framecrc(wav).expect("ffmpeg -c copy framecrc");

    // 1. Stream headers (#tb, #media_type, #codec_id, #sample_rate, #channel_layout_name)
    //    must match FFmpeg n9.1 identically.
    let w_hdrs = parse_framecrc_headers(&wedeo_lines);
    let f_hdrs = parse_framecrc_headers(&ffmpeg_lines);
    assert_eq!(
        w_hdrs,
        f_hdrs,
        "WAV stream headers differ for {}",
        wav.display()
    );

    // 2. Collect FFmpeg packet records and verify Wedeo's native WavDemuxer emits the exact
    //    same total byte stream, sample CRCs, and contiguous timeline.
    let ff_records: Vec<_> = ffmpeg_lines
        .iter()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| parse_framecrc_record(l).unwrap())
        .collect();
    let wd_records: Vec<_> = wedeo_lines
        .iter()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| parse_framecrc_record(l).unwrap())
        .collect();

    assert!(!wd_records.is_empty());
    let mut expected_ts = 0i64;
    for rec in &wd_records {
        assert_eq!(rec.dts, rec.pts);
        assert_eq!(rec.dts, expected_ts);
        expected_ts += rec.duration;
    }

    let ff_total_bytes: usize = ff_records.iter().map(|r| r.size).sum();
    let wd_total_bytes: usize = wd_records.iter().map(|r| r.size).sum();
    assert_eq!(
        wd_total_bytes, ff_total_bytes,
        "total WAV payload byte count differs"
    );

    // 3. Read raw packets directly via native WavDemuxer and verify per-chunk Adler-32
    //    against FFmpeg's exact packet boundaries.
    let src = FileSource::open(wav).unwrap();
    let mut head = [0u8; 64];
    {
        use std::io::Read;
        let mut peek = FileSource::open(wav).unwrap();
        let n = peek.read(&mut head).unwrap();
        let factory = reg.probe_demuxer(&head[..n], Some("wav")).unwrap();
        assert_eq!(factory.descriptor().name, "wav");
    }
    let factory = reg.find_demuxer_by_name("wav").unwrap();
    let mut demux = factory.open(Box::new(src), ErrorPolicy::Strict).unwrap();
    assert_eq!(demux.header().tracks[0].codec_id, expected_codec);

    let mut all_pkt_bytes = Vec::with_capacity(wd_total_bytes);
    let mut pkt_idx = 0usize;
    loop {
        match demux.read_packet().unwrap() {
            RecvStatus::Ready(pkt) => {
                let data = pkt.data.data();
                let crc = format!("0x{:08x}", adler32_ffmpeg(0, data));
                assert_eq!(wd_records[pkt_idx].size, data.len());
                assert_eq!(wd_records[pkt_idx].crc, crc);
                all_pkt_bytes.extend_from_slice(data);
                pkt_idx += 1;
            }
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }
    assert_eq!(pkt_idx, wd_records.len());

    let mut off = 0usize;
    for (i, ff) in ff_records.iter().enumerate() {
        let chunk = &all_pkt_bytes[off..off + ff.size];
        let crc = format!("0x{:08x}", adler32_ffmpeg(0, chunk));
        assert_eq!(
            crc, ff.crc,
            "FFmpeg chunk {i} Adler-32 mismatch (expected {}, got {crc})",
            ff.crc
        );
        off += ff.size;
    }
    assert_eq!(off, all_pkt_bytes.len());
}

#[test]
fn audiogen_is_bitexact_vs_ffmpeg_c_source() {
    let audiogen_c = workspace_root().join("FFmpeg/tests/audiogen.c");
    assert!(
        audiogen_c.exists(),
        "missing FFmpeg/tests/audiogen.c at {}",
        audiogen_c.display()
    );

    let dir = temp_test_dir("pcm_wav_audiogen");
    let c_bin = dir.join("ffmpeg-audiogen");
    let compile = Command::new("cc")
        .args([
            "-O2",
            "-o",
            c_bin.to_str().unwrap(),
            audiogen_c.to_str().unwrap(),
        ])
        .status()
        .expect("run cc on FFmpeg/tests/audiogen.c");
    assert!(compile.success(), "cc failed to compile audiogen.c");

    for (sr, ch, seed) in [
        (44100u32, 2u32, 1u32),
        (44100, 1, 1),
        (96000, 1, 1),
        (48000, 6, 42),
    ] {
        let ff_wav = dir.join(format!("ff-{sr}-{ch}-{seed}.wav"));
        let wd_wav = dir.join(format!("wd-{sr}-{ch}-{seed}.wav"));

        let st_ff = Command::new(&c_bin)
            .args([
                ff_wav.to_str().unwrap(),
                &sr.to_string(),
                &ch.to_string(),
                &seed.to_string(),
            ])
            .status()
            .unwrap();
        assert!(st_ff.success(), "C audiogen failed");

        let st_wd = Command::new(wedeo_audiogen_bin())
            .args([
                wd_wav.to_str().unwrap(),
                &sr.to_string(),
                &ch.to_string(),
                &seed.to_string(),
            ])
            .status()
            .unwrap();
        assert!(st_wd.success(), "Rust wedeo-audiogen failed");

        let ff_bytes = std::fs::read(&ff_wav).unwrap();
        let wd_bytes = std::fs::read(&wd_wav).unwrap();
        assert_eq!(
            ff_bytes.len(),
            wd_bytes.len(),
            "WAV size mismatch for ({sr}, {ch}, {seed})"
        );
        assert_eq!(
            ff_bytes, wd_bytes,
            "wedeo-audiogen output differs from FFmpeg/tests/audiogen.c for ({sr}, {ch}, {seed})"
        );
    }
}

#[test]
fn wav_packet_framecrc_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("pcm_wav_pkt_crc");
    for (name, sr, ch) in [
        ("asynth-44100-2.wav", 44100u32, 2u32),
        ("asynth-44100-1.wav", 44100, 1),
        ("asynth-96000-1.wav", 96000, 1),
    ] {
        let wav = generate_wav(&dir, name, sr, ch, 1);
        assert_wav_packets_match_ffmpeg(&wav, CodecId::PcmS16le);
    }
}

#[test]
fn wav_decoded_samples_bitexact_vs_ffmpeg_oracle() {
    let dir = temp_test_dir("pcm_wav_samples");
    let wav = generate_wav(&dir, "decoded-44100-2.wav", 44100, 2, 1);

    // Decode with Wedeo's AudioReader using default_registry (must use native WAV + native PCM).
    let reg = default_registry();
    let mut reader = wedeo::audio::AudioReader::open(&reg, &wav).expect("AudioReader open WAV");
    assert_eq!(reader.header().format_name, "wav");
    assert_eq!(reader.params().sample_rate, 44100);
    assert_eq!(reader.params().channel_layout.nb_channels, 2);

    let mut wedeo_pcm = Vec::new();
    while let Some(block) = reader.next_block().expect("next_block") {
        assert_eq!(block.planes.len(), 1);
        wedeo_pcm.extend_from_slice(block.planes[0].data());
    }
    assert!(reader.health().is_clean());
    assert!(reader.demuxer_health().is_clean());

    // Decode with FFmpeg n9.1 oracle to raw s16le PCM.
    let out = Command::new(ffmpeg_oracle_bin())
        .args([
            "-v",
            "error",
            "-bitexact",
            "-i",
            wav.to_str().unwrap(),
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-",
        ])
        .output()
        .expect("run ffmpeg oracle raw pcm");
    assert!(
        out.status.success(),
        "ffmpeg oracle pcm decode failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let ffmpeg_pcm = out.stdout;

    assert_eq!(wedeo_pcm.len(), ffmpeg_pcm.len());
    assert_eq!(
        md5_bytes(&wedeo_pcm),
        md5_bytes(&ffmpeg_pcm),
        "decoded PCM sample MD5 differs from FFmpeg n9.1 oracle"
    );
    assert_eq!(wedeo_pcm, ffmpeg_pcm);
}

#[test]
fn wav_native_roundtrip_demux_decode_encode_mux_bitexact() {
    let dir = temp_test_dir("pcm_wav_roundtrip");
    let wav = generate_wav(&dir, "rt-src.wav", 48000, 2, 7);

    let reg = native_registry();
    let mut reader = wedeo::audio::AudioReader::open(&reg, &wav).expect("open source WAV");
    let audio_params = reader.params().clone();

    let track = TrackDescriptor::new(
        TrackId(0),
        CodecId::PcmS16le,
        wedeo::core::Rational::new(1, audio_params.sample_rate as i32),
        StreamParams::Audio(audio_params),
    );
    let enc_factory = reg.find_encoder(CodecId::PcmS16le).expect("pcm_s16le enc");
    let mut encoder = enc_factory
        .open(&track, CodecContext::default(), PassMode::Single)
        .expect("open pcm encoder");

    let mut encoded_pkts = Vec::new();
    let mut original_pcm = Vec::new();

    while let Some(block) = reader.next_block().unwrap() {
        original_pcm.extend_from_slice(block.planes[0].data());
        assert_eq!(
            encoder
                .send_unit(
                    Some(&MediaUnit::Audio(block)),
                    wedeo::core::EncodeHint::default()
                )
                .unwrap(),
            SendStatus::Accepted
        );
        while let RecvStatus::Ready(pkt) = encoder.receive_packet().unwrap() {
            encoded_pkts.push(pkt);
        }
    }
    assert_eq!(
        encoder
            .send_unit(None, wedeo::core::EncodeHint::default())
            .unwrap(),
        SendStatus::Accepted
    );
    while let RecvStatus::Ready(pkt) = encoder.receive_packet().unwrap() {
        encoded_pkts.push(pkt);
    }

    let mux_factory = reg.find_muxer_by_name("wav").expect("wav muxer");
    let rt_path = dir.join("rt-out.wav");
    let file_sink = FileSink::create(&rt_path).expect("create rt-out.wav");
    let mut muxer = mux_factory.create(Box::new(file_sink)).unwrap();
    let mut hdr = ContainerHeader::new("wav");
    hdr.tracks.push(track);
    muxer.write_header(&hdr).unwrap();
    for pkt in &encoded_pkts {
        muxer.write_packet(pkt).unwrap();
    }
    muxer.finalize().unwrap();

    let mut rt_reader = wedeo::audio::AudioReader::open(&reg, &rt_path).expect("re-open rt WAV");
    let mut roundtrip_pcm = Vec::new();
    while let Some(block) = rt_reader.next_block().unwrap() {
        roundtrip_pcm.extend_from_slice(block.planes[0].data());
    }
    assert_eq!(original_pcm, roundtrip_pcm);

    // Cross-check the roundtripped WAV against original WAV via wedeo-framecrc!
    let orig_crc = run_wedeo_framecrc(&wav).unwrap();
    let rt_crc = run_wedeo_framecrc(&rt_path).unwrap();
    compare_framecrc_full(
        &rt_crc,
        &orig_crc,
        FramecrcCompareOpts::STRICT_TIMING_AND_HEADERS,
    )
    .expect("roundtripped WAV must match original WAV framecrc");

    assert_wav_packets_match_ffmpeg(&rt_path, CodecId::PcmS16le);
}

#[test]
fn wav_native_decode_matches_deterministic_and_supports_all_pcm_subtypes() {
    let dir = temp_test_dir("pcm_wav_subtypes");
    let src_wav = generate_wav(&dir, "sub-src.wav", 22050, 1, 3);

    for (codec_name, codec_id, ff_fmt) in [
        ("pcm_u8", CodecId::PcmU8, "u8"),
        ("pcm_s16le", CodecId::PcmS16le, "s16le"),
        ("pcm_s24le", CodecId::PcmS24le, "s32le"),
        ("pcm_s32le", CodecId::PcmS32le, "s32le"),
        ("pcm_f32le", CodecId::PcmF32le, "f32le"),
    ] {
        let out_wav = dir.join(format!("sub-{codec_name}.wav"));
        let status = Command::new(ffmpeg_oracle_bin())
            .args([
                "-v",
                "error",
                "-y",
                "-i",
                src_wav.to_str().unwrap(),
                "-t",
                "0.5",
                "-c:a",
                codec_name,
                out_wav.to_str().unwrap(),
            ])
            .status()
            .expect("convert wav subtype");
        assert!(status.success(), "ffmpeg failed converting to {codec_name}");

        assert_wav_packets_match_ffmpeg(&out_wav, codec_id);

        // Also confirm AudioReader decoded samples match FFmpeg oracle raw PCM byte-for-byte.
        let reg = native_registry();
        let mut reader = wedeo::audio::AudioReader::open(&reg, &out_wav).unwrap();
        assert_eq!(reader.header().tracks[0].codec_id, codec_id);
        let mut wedeo_pcm = Vec::new();
        while let Some(b) = reader.next_block().unwrap() {
            wedeo_pcm.extend_from_slice(b.planes[0].data());
        }
        assert!(!wedeo_pcm.is_empty(), "expected samples for {codec_name}");

        let ff_out = Command::new(ffmpeg_oracle_bin())
            .args([
                "-v",
                "error",
                "-bitexact",
                "-i",
                out_wav.to_str().unwrap(),
                "-f",
                ff_fmt,
                "-acodec",
                &format!("pcm_{ff_fmt}"),
                "-",
            ])
            .output()
            .unwrap();
        assert!(ff_out.status.success());
        assert_eq!(
            wedeo_pcm, ff_out.stdout,
            "decoded PCM samples differ from FFmpeg oracle for {codec_name}"
        );
    }
}
