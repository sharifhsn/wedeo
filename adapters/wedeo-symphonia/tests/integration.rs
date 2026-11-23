use std::io::{self, Cursor, Read};
use std::sync::Arc;

use wedeo_core::{
    AudioStreamParams, Buffer, ByteSource, ChannelLayout, CodecContext, CodecId, Demuxer,
    DemuxerDescriptor, DemuxerFactory, Error, ErrorKind, ErrorPolicy, HealthIssue, MemorySource,
    Packet, Rational, RecvStatus, Registry, Result, SampleFormat, SeekTarget, SendStatus, SideData,
    StreamParams, Timestamp, TrackDescriptor, TrackId,
};
use wedeo_symphonia::demuxer::probe_symphonia;
use wedeo_symphonia::register_symphonia;

/// Build a valid in-memory PCM S16LE WAV file from interleaved `i16` samples.
fn build_s16le_wav(sample_rate: u32, channels: u16, frames: &[[i16; 2]]) -> Vec<u8> {
    assert_eq!(channels, 2);
    let bits_per_sample: u16 = 16;
    let block_align: u16 = channels * (bits_per_sample / 8);
    let byte_rate: u32 = sample_rate * block_align as u32;
    let data_len: u32 = (frames.len() * block_align as usize) as u32;
    let riff_chunk_size: u32 = 36 + data_len;

    let mut out = Vec::with_capacity(44 + data_len as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&riff_chunk_size.to_le_bytes());
    out.extend_from_slice(b"WAVE");

    // fmt chunk
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format tag = 1
    out.extend_from_slice(&channels.to_le_bytes());
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&byte_rate.to_le_bytes());
    out.extend_from_slice(&block_align.to_le_bytes());
    out.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_len.to_le_bytes());
    for frame in frames {
        out.extend_from_slice(&frame[0].to_le_bytes());
        out.extend_from_slice(&frame[1].to_le_bytes());
    }

    out
}

/// Generate deterministic stereo `[left, right]` test samples for `n_frames`.
fn generate_stereo_frames(n_frames: usize) -> Vec<[i16; 2]> {
    (0..n_frames)
        .map(|i| {
            let left = ((i as i32 * 37 - 12_000) % 30_000) as i16;
            let right = (-(i as i32 * 53 - 8_000) % 30_000) as i16;
            [left, right]
        })
        .collect()
}

/// Extract interleaved stereo `[i16; 2]` frames from a decoded `S16` `AudioBlock`.
fn extract_s16_stereo_frames(block: &wedeo_core::AudioBlock) -> Vec<[i16; 2]> {
    assert_eq!(block.format, SampleFormat::S16);
    assert_eq!(block.channel_layout.nb_channels, 2);
    assert_eq!(block.planes.len(), 1);
    let bytes = block.planes[0].data();
    assert_eq!(bytes.len(), block.nb_samples as usize * 4);

    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| {
            let l = i16::from_ne_bytes([c[0], c[1]]);
            let r = i16::from_ne_bytes([c[2], c[3]]);
            [l, r]
        })
        .collect()
}

#[test]
fn wav_s16le_probe_demux_and_decode_exact_samples() {
    let expected_frames = generate_stereo_frames(1000);
    let wav_bytes = build_s16le_wav(48_000, 2, &expected_frames);

    let mut reg = Registry::new();
    register_symphonia(&mut reg);

    // Probe using Registry containing only symphonia.
    let demuxer_factory = reg
        .probe_demuxer(&wav_bytes, Some("wav"))
        .expect("symphonia should probe WAV bytes");
    assert_eq!(demuxer_factory.descriptor().name, "symphonia");
    assert_eq!(demuxer_factory.probe(&wav_bytes, Some("wav")), 50);

    // Open demuxer on in-memory WAV via MemorySource.
    let source = Box::new(MemorySource::new(wav_bytes));
    let mut demuxer = demuxer_factory
        .open(source, ErrorPolicy::Resilient)
        .expect("symphonia should open WAV");

    let header = demuxer.header().clone();
    assert_eq!(header.tracks.len(), 1);
    let track = &header.tracks[0];
    assert_eq!(track.id, TrackId(0));
    assert_eq!(track.codec_id, CodecId::PcmS16le);
    assert_eq!(track.time_base, Rational::new(1, 48_000));

    let audio_params = track.params.audio().expect("track must be audio");
    assert_eq!(audio_params.sample_rate, 48_000);
    assert_eq!(audio_params.channel_layout, ChannelLayout::stereo());
    assert_eq!(audio_params.format, SampleFormat::S16);

    // Open symphonia decoder from Registry.
    let decoder_factory = reg
        .find_decoder(track.codec_id)
        .expect("symphonia should register PcmS16le decoder");
    let mut decoder = decoder_factory
        .open(track, CodecContext::default())
        .expect("decoder should open");

    let mut decoded_frames = Vec::new();
    let mut expected_pts = 0i64;

    loop {
        match demuxer.read_packet().expect("read_packet should succeed") {
            RecvStatus::Ready(pkt) => {
                assert_eq!(pkt.track_id, TrackId(0));
                assert_eq!(pkt.time_base, Rational::new(1, 48_000));
                assert_eq!(pkt.pts, Timestamp::new(expected_pts));
                assert!(pkt.duration > 0);
                expected_pts += pkt.duration;

                let send_status = decoder
                    .send_packet(Some(&pkt))
                    .expect("send_packet should succeed");
                assert_eq!(send_status, SendStatus::Accepted);

                loop {
                    match decoder.receive().expect("receive should succeed") {
                        RecvStatus::Ready(unit) => {
                            let block = unit.into_audio().expect("unit must be AudioBlock");
                            assert_eq!(block.sample_rate, 48_000);
                            assert_eq!(block.time_base, Rational::new(1, 48_000));
                            assert_eq!(block.pts, pkt.pts);
                            assert_eq!(block.duration, pkt.duration);
                            decoded_frames.extend(extract_s16_stereo_frames(&block));
                        }
                        RecvStatus::NeedMore => break,
                        RecvStatus::Eof => panic!("unexpected Eof before flushing"),
                    }
                }
            }
            RecvStatus::NeedMore => continue,
            RecvStatus::Eof => break,
        }
    }

    // Drain decoder at EOF.
    assert_eq!(
        decoder.send_packet(None).expect("flush send_packet(None)"),
        SendStatus::Accepted
    );
    assert!(matches!(
        decoder.receive().expect("receive after flush"),
        RecvStatus::Eof
    ));

    assert_eq!(decoded_frames.len(), expected_frames.len());
    assert_eq!(decoded_frames, expected_frames);
    assert!(demuxer.health().is_clean());
    assert!(decoder.health().is_clean());
}

#[test]
fn wav_seek_with_timebase_rescale_and_reset() {
    // 3000 stereo frames at 48,000 Hz spans 3 Symphonia WAV packets (1152 + 1152 + 696).
    let expected_frames = generate_stereo_frames(3000);
    let wav_bytes = build_s16le_wav(48_000, 2, &expected_frames);

    let mut reg = Registry::new();
    register_symphonia(&mut reg);

    let demuxer_factory = reg.find_demuxer_by_name("symphonia").unwrap();
    let mut demuxer = demuxer_factory
        .open(Box::new(MemorySource::new(wav_bytes)), ErrorPolicy::Strict)
        .unwrap();
    let track = demuxer.header().tracks[0].clone();
    let mut decoder = reg
        .find_decoder(track.codec_id)
        .unwrap()
        .open(&track, CodecContext::default())
        .unwrap();

    // Read the first packet (frames 0..1152), then seek to 24 ms (= 1152 samples at 48 kHz)
    // expressed in a 1/1000 (millisecond) time base to exercise SeekTarget rescaling.
    let first = demuxer.read_packet().unwrap();
    assert!(matches!(first, RecvStatus::Ready(_)));

    demuxer
        .seek(SeekTarget {
            track: Some(TrackId(0)),
            timestamp: Timestamp::new(24),
            time_base: Rational::new(1, 1000),
            backward: true,
        })
        .expect("seek to 24ms (sample 1152) should succeed");
    decoder.flush();

    let mut after_seek_frames = Vec::new();
    let mut first_pkt_pts = None;
    while let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() {
        if first_pkt_pts.is_none() {
            first_pkt_pts = pkt.pts.get();
        }
        decoder.send_packet(Some(&pkt)).unwrap();
        while let RecvStatus::Ready(unit) = decoder.receive().unwrap() {
            let block = unit.into_audio().unwrap();
            after_seek_frames.extend(extract_s16_stereo_frames(&block));
        }
    }

    let start_sample = first_pkt_pts.expect("should read packets after seek") as usize;
    assert!(
        start_sample <= 1152,
        "backward seek should land at or before sample 1152, got {start_sample}"
    );
    assert_eq!(
        after_seek_frames.as_slice(),
        &expected_frames[start_sample..]
    );

    // Seek back to start (0) and verify full stream can be read again.
    demuxer
        .seek(SeekTarget {
            track: None,
            timestamp: Timestamp::new(0),
            time_base: track.time_base,
            backward: true,
        })
        .expect("seek back to 0 should succeed");
    decoder.flush();

    let mut re_read_frames = Vec::new();
    while let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() {
        decoder.send_packet(Some(&pkt)).unwrap();
        while let RecvStatus::Ready(unit) = decoder.receive().unwrap() {
            re_read_frames.extend(extract_s16_stereo_frames(&unit.into_audio().unwrap()));
        }
    }
    assert_eq!(re_read_frames, expected_frames);

    // Seeking to nonexistent track or Timestamp::NONE returns appropriate errors.
    let err_track = demuxer
        .seek(SeekTarget {
            track: Some(TrackId(99)),
            timestamp: Timestamp::new(0),
            time_base: track.time_base,
            backward: true,
        })
        .unwrap_err();
    assert_eq!(err_track.kind, ErrorKind::NotFound);

    let err_nopts = demuxer
        .seek(SeekTarget {
            track: Some(TrackId(0)),
            timestamp: Timestamp::NONE,
            time_base: track.time_base,
            backward: true,
        })
        .unwrap_err();
    assert_eq!(err_nopts.kind, ErrorKind::InvalidArgument);
}

/// Non-seekable `ByteSource` wrapper (simulating a pipe or network stream).
struct NonSeekableSource {
    cursor: Cursor<Vec<u8>>,
}

impl NonSeekableSource {
    fn new(data: Vec<u8>) -> Self {
        Self {
            cursor: Cursor::new(data),
        }
    }
}

impl Read for NonSeekableSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.cursor.read(buf)
    }
}

impl ByteSource for NonSeekableSource {
    fn is_seekable(&self) -> bool {
        false
    }
}

#[test]
fn non_seekable_byte_source_demuxes_sequentially_and_rejects_seek() {
    let expected_frames = generate_stereo_frames(1000);
    let wav_bytes = build_s16le_wav(44_100, 2, &expected_frames);

    let mut reg = Registry::new();
    register_symphonia(&mut reg);

    let demuxer_factory = reg.find_demuxer_by_name("symphonia").unwrap();
    let mut demuxer = demuxer_factory
        .open(
            Box::new(NonSeekableSource::new(wav_bytes)),
            ErrorPolicy::Resilient,
        )
        .expect("non-seekable WAV should open sequentially");

    let track = demuxer.header().tracks[0].clone();
    let mut decoder = reg
        .find_decoder(track.codec_id)
        .unwrap()
        .open(&track, CodecContext::default())
        .unwrap();

    // Attempting to seek a non-seekable source must return ErrorKind::NotSeekable.
    let seek_err = demuxer
        .seek(SeekTarget {
            track: Some(TrackId(0)),
            timestamp: Timestamp::new(0),
            time_base: track.time_base,
            backward: true,
        })
        .unwrap_err();
    assert_eq!(seek_err.kind, ErrorKind::NotSeekable);

    // Sequential demuxing and decoding still works end-to-end.
    let mut decoded_frames = Vec::new();
    while let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() {
        decoder.send_packet(Some(&pkt)).unwrap();
        while let RecvStatus::Ready(unit) = decoder.receive().unwrap() {
            decoded_frames.extend(extract_s16_stereo_frames(&unit.into_audio().unwrap()));
        }
    }
    assert_eq!(decoded_frames, expected_frames);
}

struct FakeNativeWavDemuxerFactory;

impl DemuxerFactory for FakeNativeWavDemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        static DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "wav_native_fake",
            long_name: "Fake Native WAV Demuxer",
            extensions: &["wav"],
            mime_types: &["audio/wav"],
        };
        &DESC
    }

    fn probe(&self, head: &[u8], _ext: Option<&str>) -> u8 {
        if head.len() >= 12 && &head[0..4] == b"RIFF" && &head[8..12] == b"WAVE" {
            100
        } else {
            0
        }
    }

    fn open(&self, _source: Box<dyn ByteSource>, _policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Err(Error::unsupported("test/fake_wav", "fake native demuxer"))
    }
}

#[test]
fn fake_native_demuxer_with_probe_100_beats_symphonia() {
    let frames = generate_stereo_frames(16);
    let wav_bytes = build_s16le_wav(48_000, 2, &frames);

    // Order 1: Native registered first (recommended order).
    let mut reg1 = Registry::new();
    reg1.register_demuxer(Arc::new(FakeNativeWavDemuxerFactory));
    register_symphonia(&mut reg1);

    let winner1 = reg1
        .probe_demuxer(&wav_bytes, Some("wav"))
        .expect("should match a demuxer");
    assert_eq!(winner1.descriptor().name, "wav_native_fake");

    // Order 2: Symphonia registered first — native (100) still beats Symphonia (50).
    let mut reg2 = Registry::new();
    register_symphonia(&mut reg2);
    reg2.register_demuxer(Arc::new(FakeNativeWavDemuxerFactory));

    let winner2 = reg2
        .probe_demuxer(&wav_bytes, Some("wav"))
        .expect("should match a demuxer");
    assert_eq!(winner2.descriptor().name, "wav_native_fake");
}

#[test]
fn probe_scores_never_exceed_50() {
    let cases: &[(&[u8], Option<&str>, u8)] = &[
        (b"RIFF\x24\x00\x00\x00WAVEfmt ", Some("wav"), 50),
        (b"RIFX\x24\x00\x00\x00WAVEfmt ", None, 50),
        (b"RF64\xFF\xFF\xFF\xFFWAVEds64", None, 50),
        (b"BW64\xFF\xFF\xFF\xFFWAVEds64", None, 50),
        (b"FORM\x00\x00\x01\x00AIFFCOMM", None, 50),
        (b"OggS\x00\x02\x00\x00", None, 50),
        (b"fLaC\x00\x00\x00\x22", None, 50),
        (&[0x1A, 0x45, 0xDF, 0xA3, 0x01, 0x00], None, 50),
        (b"caff\x00\x01\x00\x00", None, 50),
        (b"\x00\x00\x00\x18ftypM4A ", None, 45),
        (b"ID3\x04\x00\x00", None, 45),
        (&[0xFF, 0xFB, 0x90, 0x64], None, 30),
        (b"unknown_header", Some("flac"), 20),
        (b"unknown_header", Some("xyz"), 0),
    ];

    for &(head, ext, expected) in cases {
        let score = probe_symphonia(head, ext);
        assert!(score <= 50, "score {score} exceeded 50 for {head:?}");
        assert_eq!(score, expected, "unexpected score for {head:?} / {ext:?}");
    }
}

#[test]
fn decoder_error_policy_resilient_vs_strict_and_sample_trim() {
    let mut reg = Registry::new();
    register_symphonia(&mut reg);

    // 1. Test corrupt MP3 packet under Strict vs Resilient policy.
    let mp3_params = AudioStreamParams::new(48_000, SampleFormat::F32, ChannelLayout::stereo());
    let mp3_track = TrackDescriptor::new(
        TrackId(0),
        CodecId::Mp3,
        Rational::new(1, 48_000),
        StreamParams::Audio(mp3_params),
    );
    let mp3_factory = reg.find_decoder(CodecId::Mp3).unwrap();

    let strict_ctx = CodecContext {
        error_policy: ErrorPolicy::Strict,
        ..Default::default()
    };
    let mut strict_dec = mp3_factory.open(&mp3_track, strict_ctx).unwrap();

    let bad_pkt = Packet::new(
        Buffer::from_slice(&[0x00, 0x01, 0x02, 0x03]),
        TrackId(0),
        Rational::new(1, 48_000),
    );
    strict_dec.send_packet(Some(&bad_pkt)).unwrap();
    let strict_err = strict_dec.receive().unwrap_err();
    assert_eq!(strict_err.kind, ErrorKind::CorruptStream);

    let resilient_ctx = CodecContext {
        error_policy: ErrorPolicy::Resilient,
        ..Default::default()
    };
    let mut resilient_dec = mp3_factory.open(&mp3_track, resilient_ctx).unwrap();
    resilient_dec.send_packet(Some(&bad_pkt)).unwrap();
    assert!(matches!(
        resilient_dec.receive().unwrap(),
        RecvStatus::NeedMore
    ));
    assert_eq!(resilient_dec.health().corrupt_packets, 1);
    assert_eq!(
        resilient_dec.health().events[0].kind,
        HealthIssue::CorruptSyntax
    );

    // 2. Test SampleTrim side_data on a PCM packet.
    let pcm_frames = generate_stereo_frames(100);
    let mut raw_bytes = Vec::with_capacity(pcm_frames.len() * 4);
    for f in &pcm_frames {
        raw_bytes.extend_from_slice(&f[0].to_le_bytes());
        raw_bytes.extend_from_slice(&f[1].to_le_bytes());
    }
    let pcm_params = AudioStreamParams::new(48_000, SampleFormat::S16, ChannelLayout::stereo());
    let pcm_track = TrackDescriptor::new(
        TrackId(0),
        CodecId::PcmS16le,
        Rational::new(1, 48_000),
        StreamParams::Audio(pcm_params),
    );
    let mut pcm_dec = reg
        .find_decoder(CodecId::PcmS16le)
        .unwrap()
        .open(&pcm_track, CodecContext::default())
        .unwrap();

    let mut pkt = Packet::new(
        Buffer::from_slice(&raw_bytes),
        TrackId(0),
        Rational::new(1, 48_000),
    );
    pkt.pts = Timestamp::new(500);
    pkt.duration = 100;
    pkt.side_data.push(SideData::SampleTrim {
        skip_start: 10,
        skip_end: 15,
    });

    pcm_dec.send_packet(Some(&pkt)).unwrap();
    let RecvStatus::Ready(unit) = pcm_dec.receive().unwrap() else {
        panic!("expected Ready audio unit");
    };
    let block = unit.into_audio().unwrap();
    assert_eq!(block.nb_samples, 75);
    assert_eq!(block.duration, 75);
    assert_eq!(block.pts, Timestamp::new(500));
    assert_eq!(
        extract_s16_stereo_frames(&block).as_slice(),
        &pcm_frames[10..85]
    );
}
