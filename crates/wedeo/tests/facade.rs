//! Integration tests for the `wedeo` facade crate:
//! - Deterministic `default_registry()` / `native_registry()` resolution
//! - `AudioReader` end-to-end (+ `AudioOpList` resampling when `audio-ops` is enabled)
//! - `VideoReader` end-to-end on bundled VP9 IVF fixture

use std::io::{self, Cursor, Read};

use wedeo::{
    ByteSource, CodecId, ErrorPolicy, MemorySource, SeekTarget, Timestamp, default_registry,
    native_registry,
};

#[cfg(all(feature = "wav", feature = "pcm"))]
fn make_pcm_s16le_wav(sample_rate: u32, channels: u16, frames: usize) -> Vec<u8> {
    let bytes_per_sample = 2u32;
    let block_align = channels as u32 * bytes_per_sample;
    let byte_rate = sample_rate * block_align;
    let data_size = frames as u32 * block_align;
    let riff_size = 36 + data_size;

    let mut out = Vec::with_capacity(44 + data_size as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&(block_align as u16).to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_size.to_le_bytes());

    for i in 0..frames {
        for ch in 0..channels {
            let sample = ((i as i32 * 13 + ch as i32 * 500) % 16000) as i16;
            out.extend_from_slice(&sample.to_le_bytes());
        }
    }
    out
}

/// Non-seekable `ByteSource` to exercise `PrefixedSource` probing + streaming.
struct PipeSource {
    inner: Cursor<Vec<u8>>,
}

impl Read for PipeSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.inner.read(buf)
    }
}

impl ByteSource for PipeSource {
    fn is_seekable(&self) -> bool {
        false
    }
}

#[test]
fn native_registry_excludes_external_adapters() {
    let reg = native_registry();
    assert!(reg.find_demuxer_by_name("symphonia").is_none());
    assert!(reg.find_decoder_by_name("av1_rav1d").is_none());
    assert!(reg.find_encoder_by_name("av1_rav1e").is_none());
}

#[cfg(all(feature = "wav", feature = "pcm", feature = "symphonia"))]
#[test]
fn default_registry_prefers_native_wav_and_pcm_over_symphonia() {
    let reg = default_registry();

    // Native WAV demuxer scores 100 on RIFF/WAVE vs Symphonia's 50.
    let wav = make_pcm_s16le_wav(48_000, 2, 64);
    let demuxer = reg
        .probe_demuxer(&wav[..64], Some("wav"))
        .expect("WAV demuxer should match");
    assert_eq!(demuxer.descriptor().name, "wav");
    assert_eq!(demuxer.probe(&wav[..64], Some("wav")), 100);

    let sym = reg
        .find_demuxer_by_name("symphonia")
        .expect("symphonia demuxer should be registered");
    assert!(sym.probe(&wav[..64], Some("wav")) <= 50);

    // Native PCM decoder wins over Symphonia's fallback PCM decoder.
    let dec = reg
        .find_decoder(CodecId::PcmS16le)
        .expect("PCM S16LE decoder should be registered");
    assert_ne!(dec.descriptor().name, "symphonia");
}

#[cfg(feature = "mp4")]
#[test]
fn default_registry_resolves_mp4() {
    let reg = default_registry();
    // Minimal ftyp box: size=20, "ftyp", "isom"
    let mut head = Vec::new();
    head.extend_from_slice(&20u32.to_be_bytes());
    head.extend_from_slice(b"ftypisom");
    head.extend_from_slice(&0u32.to_be_bytes());
    head.extend_from_slice(b"isom");

    let demuxer = reg
        .probe_demuxer(&head, Some("mp4"))
        .expect("MP4 demuxer should match");
    assert_eq!(demuxer.descriptor().name, "mp4");
}

#[cfg(feature = "mkv")]
#[test]
fn default_registry_resolves_mkv() {
    let reg = default_registry();
    // EBML magic 0x1A 0x45 0xDF 0xA3 + DocType "matroska"
    let head = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\x82\x88matroska";
    let demuxer = reg
        .probe_demuxer(head, Some("mkv"))
        .expect("MKV demuxer should match");
    assert_eq!(demuxer.descriptor().name, "matroska");
}

#[cfg(feature = "ivf")]
#[test]
fn default_registry_resolves_ivf() {
    let reg = default_registry();
    let mut head = [0u8; 32];
    head[0..4].copy_from_slice(b"DKIF");
    head[4..6].copy_from_slice(&0u16.to_le_bytes());
    head[6..8].copy_from_slice(&32u16.to_le_bytes());
    head[8..12].copy_from_slice(b"VP90");
    let demuxer = reg
        .probe_demuxer(&head, Some("ivf"))
        .expect("IVF demuxer should match");
    assert_eq!(demuxer.descriptor().name, "ivf");
}

#[cfg(feature = "h264")]
#[test]
fn default_registry_resolves_h264() {
    let reg = default_registry();
    let dec = reg
        .find_decoder(CodecId::H264)
        .expect("H.264 decoder should be registered");
    assert_eq!(dec.descriptor().name, "h264");
}

#[cfg(feature = "vp9")]
#[test]
fn default_registry_resolves_vp9() {
    let reg = default_registry();
    let dec = reg
        .find_decoder(CodecId::Vp9)
        .expect("VP9 decoder should be registered");
    assert_eq!(dec.descriptor().name, "vp9");
}

#[cfg(feature = "rav1e")]
#[test]
fn default_registry_resolves_rav1e_av1_encoder() {
    let reg = default_registry();
    let enc = reg
        .find_encoder(CodecId::Av1)
        .expect("rav1e AV1 encoder should be registered");
    assert_eq!(enc.descriptor().name, "av1_rav1e");
}

#[cfg(feature = "rav1d")]
#[test]
fn default_registry_resolves_rav1d_av1_decoder() {
    let reg = default_registry();
    let dec = reg
        .find_decoder(CodecId::Av1)
        .expect("rav1d AV1 decoder should be registered");
    assert_eq!(dec.descriptor().name, "av1_rav1d");
}

#[cfg(all(feature = "wav", feature = "pcm"))]
#[test]
fn audio_reader_decodes_in_memory_wav_and_pipe_source() {
    let reg = default_registry();
    let total_frames = 2048usize;
    let wav = make_pcm_s16le_wav(48_000, 2, total_frames);

    // Seekable MemorySource + seek back to start
    let mut reader = wedeo::audio::AudioReader::from_source(
        &reg,
        Box::new(MemorySource::new(wav.clone())),
        ErrorPolicy::Strict,
    )
    .expect("AudioReader should open in-memory WAV");

    assert_eq!(reader.header().format_name, "wav");
    assert_eq!(reader.params().sample_rate, 48_000);
    assert_eq!(reader.params().channel_layout.nb_channels, 2);

    let mut decoded_frames = 0usize;
    while let Some(block) = reader.next_block().expect("next_block should succeed") {
        assert_eq!(block.sample_rate, 48_000);
        assert_eq!(block.channel_layout.nb_channels, 2);
        decoded_frames += block.nb_samples as usize;
    }
    assert_eq!(decoded_frames, total_frames);
    assert_eq!(reader.health().corrupt_packets, 0);

    // Seek back to 0 and re-read
    reader
        .seek(SeekTarget {
            track: None,
            timestamp: Timestamp::new(0),
            time_base: wedeo::Rational::new(1, 48_000),
            backward: true,
        })
        .expect("seek to 0 should succeed");
    let first_after_seek = reader
        .next_block()
        .expect("next_block after seek should succeed")
        .expect("should yield block after seek");
    assert!(first_after_seek.nb_samples > 0);

    // Non-seekable pipe source
    let mut pipe_reader = wedeo::audio::AudioReader::from_source(
        &reg,
        Box::new(PipeSource {
            inner: Cursor::new(wav),
        }),
        ErrorPolicy::Strict,
    )
    .expect("AudioReader should open non-seekable WAV pipe");
    let mut pipe_frames = 0usize;
    while let Some(block) = pipe_reader.next_block().expect("pipe next_block") {
        pipe_frames += block.nb_samples as usize;
    }
    assert_eq!(pipe_frames, total_frames);
}

#[cfg(all(feature = "wav", feature = "pcm", feature = "audio-ops"))]
#[test]
fn audio_reader_with_audio_op_list_resample() {
    use wedeo::BufferPool;
    use wedeo::audio::{AudioOpList, CompiledAudioOps};
    use wedeo::audio_ops::AudioContract;

    let reg = default_registry();
    let wav = make_pcm_s16le_wav(48_000, 2, 4800);
    let mut reader = wedeo::audio::AudioReader::from_source(
        &reg,
        Box::new(MemorySource::new(wav)),
        ErrorPolicy::Strict,
    )
    .expect("AudioReader should open WAV");

    let params = reader.params().clone();
    let src_contract = AudioContract::new(
        params.format,
        params.sample_rate,
        params.channel_layout.clone(),
    );
    let ops = AudioOpList::builder(src_contract)
        .sample_rate(44_100)
        .build()
        .expect("AudioOpList build");
    let mut compiled = CompiledAudioOps::compile(&ops).expect("CompiledAudioOps::compile");
    let pool = BufferPool::new();

    let mut out_samples = 0usize;
    while let Some(block) = reader.next_block().expect("next_block") {
        for out_block in compiled.process(block, &pool).expect("process") {
            assert_eq!(out_block.sample_rate, 44_100);
            out_samples += out_block.nb_samples as usize;
        }
    }
    for out_block in compiled.flush(&pool).expect("flush") {
        assert_eq!(out_block.sample_rate, 44_100);
        out_samples += out_block.nb_samples as usize;
    }
    // 4800 samples @ 48kHz -> ~4410 samples @ 44.1kHz
    assert!(
        (4300..=4500).contains(&out_samples),
        "unexpected resampled count: {out_samples}"
    );
}

#[cfg(all(feature = "ivf", feature = "vp9"))]
#[test]
fn video_reader_decodes_bundled_vp9_ivf() {
    let reg = default_registry();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/wedeo-oracle/data/vp9_red_64x64.ivf"
    );

    let mut reader =
        wedeo::video::VideoReader::open(&reg, path).expect("VideoReader should open VP9 IVF");
    assert_eq!(reader.header().format_name, "ivf");
    assert_eq!(reader.params().width, 64);
    assert_eq!(reader.params().height, 64);

    let mut frame_count = 0usize;
    while let Some(frame) = reader.next_frame().expect("next_frame should succeed") {
        assert_eq!(frame.width, 64);
        assert_eq!(frame.height, 64);
        frame_count += 1;
    }
    assert!(frame_count >= 1, "expected at least one decoded VP9 frame");
    assert_eq!(reader.health().corrupt_packets, 0);
}
