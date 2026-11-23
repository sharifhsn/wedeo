//! Complete native Matroska (`.mkv`, `.mka`, `.mks`) and WebM (`.webm`) EBML demuxer,
//! muxer, and shared EBML AST inspector.

#![forbid(unsafe_code)]

pub mod demuxer;
pub mod ebml;
pub mod muxer;

use std::sync::Arc;

use wedeo_core::Registry;

pub use demuxer::{CueEntry, MkvDemuxer, MkvDemuxerFactory};
pub use ebml::{EbmlElement, EbmlHeader, inspect_ebml_bytes, parse_ebml_tree};
pub use muxer::{MkvMuxer, MkvMuxerFactory, WebmMuxerFactory};

/// Register Matroska and WebM demuxer and muxer factories into `registry`.
pub fn register_mkv(registry: &mut Registry) {
    registry.register_demuxer(Arc::new(MkvDemuxerFactory));
    registry.register_muxer(Arc::new(MkvMuxerFactory));
    registry.register_muxer(Arc::new(WebmMuxerFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use wedeo_core::{
        Attachment, AudioStreamParams, Buffer, ByteSink, ChannelLayout, Chapter, ChromaLocation,
        CodecId, ColorDesc, ColorPrimaries, ColorRange, ContainerHeader, CropRect, Demuxer,
        ErrorPolicy, FileSink, FileSource, MatrixCoefficients, MemorySink, MemorySource, Metadata,
        Muxer, Packet, PacketFlags, PixelLayout, Rational, RecvStatus, SampleFormat, SeekTarget,
        SideData, StreamParams, SubtitleFormat, SubtitleStreamParams, Timestamp, TrackDescriptor,
        TrackDisposition, TrackId, TransferCharacteristic, VideoStreamParams,
    };

    #[derive(Clone, Default)]
    struct NonSeekableVecSink(Arc<std::sync::Mutex<Vec<u8>>>);

    impl Write for NonSeekableVecSink {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    impl ByteSink for NonSeekableVecSink {
        fn is_seekable(&self) -> bool {
            false
        }
    }

    fn make_test_avcc() -> Vec<u8> {
        vec![
            0x01, 0x42, 0xC0, 0x1E, 0xFF, 0xE1, 0x00, 0x04, 0x67, 0x42, 0xC0, 0x1E, 0x01, 0x00,
            0x02, 0x68, 0xCE,
        ]
    }

    #[test]
    fn roundtrip_video_audio_subtitle_chapters_colour_and_cues() {
        let mut hdr = ContainerHeader::new("matroska");
        hdr.metadata.set("title", "Wedeo MKV Test");
        hdr.metadata.set("artist", "Wedeo Team");

        // Track 0: Video (VP9 with HDR BT.2020 PQ Colour + Crop + SAR)
        let mut vparams = VideoStreamParams::new(1920, 1080, PixelLayout::YUV420P10LE);
        vparams.color = ColorDesc {
            primaries: ColorPrimaries::Bt2020,
            transfer: TransferCharacteristic::Pq,
            matrix: MatrixCoefficients::Bt2020Ncl,
            range: ColorRange::Limited,
            chroma_location: ChromaLocation::TopLeft,
        };
        vparams.crop = CropRect {
            top: 4,
            bottom: 4,
            left: 8,
            right: 8,
        };
        vparams.sar = Rational::new(4, 3);
        vparams.avg_frame_rate = Rational::new(25, 1);
        let mut vtrack = TrackDescriptor::new(
            TrackId(0),
            CodecId::Vp9,
            Rational::new(1, 1000),
            StreamParams::Video(vparams),
        );
        vtrack.language = Some("eng".into());
        vtrack.disposition = TrackDisposition::DEFAULT;
        vtrack.metadata.set("title", "Main Video");
        vtrack.metadata.set("ENCODER_SETTINGS", "crf=18");
        hdr.tracks.push(vtrack);

        // Track 1: Audio (Opus 48kHz stereo with initial_padding)
        let mut aparams =
            AudioStreamParams::new(48_000, SampleFormat::F32Planar, ChannelLayout::stereo());
        aparams.initial_padding = 312;
        aparams.extradata = b"OpusHead\x01\x02\x38\x01\x80\xBB\x00\x00\x00\x00\x00".to_vec();
        let mut atrack = TrackDescriptor::new(
            TrackId(1),
            CodecId::Opus,
            Rational::new(1, 1000),
            StreamParams::Audio(aparams),
        );
        atrack.language = Some("jpn".into());
        atrack.disposition = TrackDisposition::DEFAULT;
        atrack.metadata.set("title", "Stereo Audio");
        hdr.tracks.push(atrack);

        // Track 2: Subtitle (ASS with header_script)
        let sparams = SubtitleStreamParams {
            format: SubtitleFormat::Ass,
            canvas_width: 1920,
            canvas_height: 1080,
            header_script: b"[Script Info]\nTitle: Test\n".to_vec(),
        };
        let mut strack = TrackDescriptor::new(
            TrackId(2),
            CodecId::SubAss,
            Rational::new(1, 1000),
            StreamParams::Subtitle(sparams),
        );
        strack.language = Some("eng".into());
        strack.disposition = TrackDisposition::FORCED;
        hdr.tracks.push(strack);

        // Chapters
        let mut ch1_meta = Metadata::new();
        ch1_meta.set("title", "Intro");
        hdr.chapters.push(Chapter {
            id: 101,
            start: Timestamp::new(0),
            end: Timestamp::new(1_000),
            time_base: Rational::new(1, 1000),
            metadata: ch1_meta,
        });
        let mut ch2_meta = Metadata::new();
        ch2_meta.set("title", "Main Feature");
        hdr.chapters.push(Chapter {
            id: 102,
            start: Timestamp::new(1_000),
            end: Timestamp::new(5_000),
            time_base: Rational::new(1, 1000),
            metadata: ch2_meta,
        });

        // Attachments
        hdr.attachments.push(Attachment {
            name: "cover.jpg".into(),
            mime_type: "image/jpeg".into(),
            data: Buffer::from_slice(&[0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4]),
        });

        // Mux into MemorySink
        let path = "/tmp/wedeo_test_mkv_full_roundtrip.mkv";
        let sink = FileSink::create(path).unwrap();
        let mut muxer: Box<dyn Muxer> = Box::new(MkvMuxer::new(Box::new(sink)));
        muxer.write_header(&hdr).unwrap();

        let tb = Rational::new(1, 1000);
        // Cluster 1 at t=0ms
        let mut v0 = Packet::new(
            Buffer::from_slice(&[0x10, 0x20, 0x30, 0x40]),
            TrackId(0),
            tb,
        );
        v0.pts = Timestamp::new(0);
        v0.dts = Timestamp::new(0);
        v0.duration = 40;
        v0.flags = PacketFlags::KEY;
        muxer.write_packet(&v0).unwrap();

        let mut a0 = Packet::new(Buffer::from_slice(&[0xAA, 0xBB, 0xCC]), TrackId(1), tb);
        a0.pts = Timestamp::new(0);
        a0.dts = Timestamp::new(0);
        a0.duration = 20;
        a0.flags = PacketFlags::KEY;
        muxer.write_packet(&a0).unwrap();

        let mut s0 = Packet::new(Buffer::from_slice(b"Hello Subtitle!"), TrackId(2), tb);
        s0.pts = Timestamp::new(100);
        s0.dts = Timestamp::new(100);
        s0.duration = 1500;
        s0.flags = PacketFlags::KEY;
        muxer.write_packet(&s0).unwrap();

        let mut v1 = Packet::new(Buffer::from_slice(&[0x11, 0x22, 0x33]), TrackId(0), tb);
        v1.pts = Timestamp::new(45); // VFR timestamp (45ms instead of 40ms)
        v1.dts = Timestamp::new(45);
        v1.duration = 55;
        muxer.write_packet(&v1).unwrap();

        // Cluster 2 at t=1000ms (new video keyframe triggers a new Cluster and CuePoint)
        let mut v2 = Packet::new(
            Buffer::from_slice(&[0x99, 0x88, 0x77, 0x66]),
            TrackId(0),
            tb,
        );
        v2.pts = Timestamp::new(1000);
        v2.dts = Timestamp::new(1000);
        v2.duration = 40;
        v2.flags = PacketFlags::KEY;
        muxer.write_packet(&v2).unwrap();

        let mut a1 = Packet::new(Buffer::from_slice(&[0xDD, 0xEE]), TrackId(1), tb);
        a1.pts = Timestamp::new(1020);
        a1.dts = Timestamp::new(1020);
        a1.duration = 20;
        a1.flags = PacketFlags::KEY;
        muxer.write_packet(&a1).unwrap();

        muxer.finalize().unwrap();

        // Inspect EBML AST
        let raw_bytes = std::fs::read(path).unwrap();
        let ast = inspect_ebml_bytes(&raw_bytes, 3).unwrap();
        assert_eq!(ast[0].name, "EBML");
        assert_eq!(ast[1].name, "Segment");
        assert!(ast[1].children.iter().any(|c| c.name == "Info"));
        assert!(ast[1].children.iter().any(|c| c.name == "Tracks"));
        assert!(ast[1].children.iter().any(|c| c.name == "Cluster"));
        assert!(ast[1].children.iter().any(|c| c.name == "Cues"));

        // Demux and verify everything
        let source = FileSource::open(path).unwrap();
        let mut demuxer = MkvDemuxer::open(Box::new(source), ErrorPolicy::Strict).unwrap();
        let dh = demuxer.header().clone();

        assert_eq!(dh.format_name, "matroska");
        assert_eq!(dh.metadata.get("title"), Some("Wedeo MKV Test"));
        assert_eq!(dh.metadata.get("artist"), Some("Wedeo Team"));
        assert_eq!(dh.tracks.len(), 3);

        // Verify Video track
        let vt = &dh.tracks[0];
        assert_eq!(vt.codec_id, CodecId::Vp9);
        assert_eq!(vt.language.as_deref(), Some("eng"));
        assert_eq!(vt.metadata.get("title"), Some("Main Video"));
        assert_eq!(vt.metadata.get("ENCODER_SETTINGS"), Some("crf=18"));
        let vp = vt.params.video().unwrap();
        assert_eq!(vp.width, 1920);
        assert_eq!(vp.height, 1080);
        assert_eq!(vp.layout, PixelLayout::YUV420P10LE);
        assert_eq!(vp.color.primaries, ColorPrimaries::Bt2020);
        assert_eq!(vp.color.transfer, TransferCharacteristic::Pq);
        assert_eq!(vp.color.matrix, MatrixCoefficients::Bt2020Ncl);
        assert_eq!(vp.color.range, ColorRange::Limited);
        assert_eq!(vp.color.chroma_location, ChromaLocation::TopLeft);
        assert_eq!(
            vp.crop,
            CropRect {
                top: 4,
                bottom: 4,
                left: 8,
                right: 8
            }
        );
        assert_eq!(vp.sar, Rational::new(4, 3));

        // Verify Audio track (never dropped!)
        let at = &dh.tracks[1];
        assert_eq!(at.codec_id, CodecId::Opus);
        assert_eq!(at.language.as_deref(), Some("jpn"));
        let ap = at.params.audio().unwrap();
        assert_eq!(ap.sample_rate, 48_000);
        assert_eq!(ap.channel_layout.nb_channels, 2);
        assert_eq!(ap.initial_padding, 312);

        // Verify Subtitle track
        let st = &dh.tracks[2];
        assert_eq!(st.codec_id, CodecId::SubAss);
        assert!(st.disposition.contains(TrackDisposition::FORCED));
        match &st.params {
            StreamParams::Subtitle(sp) => {
                assert_eq!(sp.format, SubtitleFormat::Ass);
                assert_eq!(sp.header_script, b"[Script Info]\nTitle: Test\n");
            }
            _ => panic!("expected subtitle params"),
        }

        // Verify Chapters & Attachments
        assert_eq!(dh.chapters.len(), 2);
        assert_eq!(dh.chapters[0].metadata.get("title"), Some("Intro"));
        assert_eq!(dh.chapters[1].metadata.get("title"), Some("Main Feature"));
        assert_eq!(dh.attachments.len(), 1);
        assert_eq!(dh.attachments[0].name, "cover.jpg");
        assert_eq!(dh.attachments[0].mime_type, "image/jpeg");

        // Verify Cues parsed via SeekHead
        assert_eq!(demuxer.cues().len(), 2);

        // Read all 6 packets
        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demuxer.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 6);
        assert_eq!(pkts[0].track_id, TrackId(0));
        assert_eq!(pkts[0].pts, Timestamp::new(0));
        assert!(pkts[0].flags.contains(PacketFlags::KEY));
        assert_eq!(pkts[1].track_id, TrackId(1));
        assert_eq!(pkts[1].pts, Timestamp::new(0));
        assert_eq!(pkts[2].track_id, TrackId(2));
        assert_eq!(pkts[2].pts, Timestamp::new(100));
        assert_eq!(pkts[2].duration, 1500); // preserved via BlockGroup + BlockDuration
        assert_eq!(pkts[3].track_id, TrackId(0));
        assert_eq!(pkts[3].pts, Timestamp::new(45));
        assert!(!pkts[3].flags.contains(PacketFlags::KEY));
        assert_eq!(pkts[4].track_id, TrackId(0));
        assert_eq!(pkts[4].pts, Timestamp::new(1000));
        assert!(pkts[4].flags.contains(PacketFlags::KEY));
        assert_eq!(pkts[5].track_id, TrackId(1));
        assert_eq!(pkts[5].pts, Timestamp::new(1020));

        // Verify Cues seeking to t=1000ms
        demuxer
            .seek(SeekTarget {
                track: Some(TrackId(0)),
                timestamp: Timestamp::new(1000),
                time_base: Rational::new(1, 1000),
                backward: true,
            })
            .unwrap();
        let RecvStatus::Ready(seek_pkt) = demuxer.read_packet().unwrap() else {
            panic!("expected packet after seek");
        };
        assert_eq!(seek_pkt.track_id, TrackId(0));
        assert_eq!(seek_pkt.pts, Timestamp::new(1000));
        assert!(seek_pkt.flags.contains(PacketFlags::KEY));

        assert!(demuxer.health().is_clean());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn lacing_modes_xiph_fixed_and_ebml_plus_bframe_vfr_timestamps() {
        use crate::ebml::*;

        // Test all 3 lacing modes in parse_laced_frames directly first:
        // 1. Xiph lacing: 3 frames of sizes 300 (255 + 45), 10, and 20
        let mut xiph_payload = vec![2, 255, 45, 10];
        xiph_payload.extend(std::iter::repeat_n(0x11, 300));
        xiph_payload.extend(std::iter::repeat_n(0x22, 10));
        xiph_payload.extend(std::iter::repeat_n(0x33, 20));
        let xiph_ranges = parse_laced_frames(&xiph_payload, 1).unwrap();
        assert_eq!(xiph_ranges.len(), 3);
        assert_eq!(xiph_ranges[0].len(), 300);
        assert_eq!(xiph_ranges[1].len(), 10);
        assert_eq!(xiph_ranges[2].len(), 20);

        // 2. Fixed-size lacing: 3 frames of 16 bytes each
        let mut fixed_payload = vec![2];
        fixed_payload.extend(std::iter::repeat_n(0x44, 48));
        let fixed_ranges = parse_laced_frames(&fixed_payload, 2).unwrap();
        assert_eq!(fixed_ranges.len(), 3);
        assert_eq!(fixed_ranges[0].len(), 16);
        assert_eq!(fixed_ranges[1].len(), 16);
        assert_eq!(fixed_ranges[2].len(), 16);

        // 3. EBML lacing: 3 frames of sizes 100, 90 (delta -10 -> 63 - 10 = 53 -> 0x80 | 53 = 0xB5), 110
        let mut ebml_payload = vec![2, 0x80 | 100, 0x80 | (63 - 10)];
        ebml_payload.extend(std::iter::repeat_n(0x55, 100));
        ebml_payload.extend(std::iter::repeat_n(0x66, 90));
        ebml_payload.extend(std::iter::repeat_n(0x77, 110));
        let ebml_ranges = parse_laced_frames(&ebml_payload, 3).unwrap();
        assert_eq!(ebml_ranges.len(), 3);
        assert_eq!(ebml_ranges[0].len(), 100);
        assert_eq!(ebml_ranges[1].len(), 90);
        assert_eq!(ebml_ranges[2].len(), 110);

        // Now build a synthetic Matroska stream containing:
        // - B-frame reordered PTS on SimpleBlock (0, 80, 40) to verify no corrupt non-monotonic DTS is fabricated!
        // - A BlockGroup with DiscardPadding and Xiph-laced audio frames!
        let mut hdr = ContainerHeader::new("matroska");
        let mut vp = VideoStreamParams::new(320, 240, PixelLayout::YUV420P);
        vp.extradata = make_test_avcc();
        hdr.tracks.push(TrackDescriptor::new(
            TrackId(0),
            CodecId::H264,
            Rational::new(1, 1000),
            StreamParams::Video(vp),
        ));

        let pipe_sink = NonSeekableVecSink::default();
        // Write header + reordered B-frame packets (PTS: 0, 80, 40)
        {
            let tb = Rational::new(1, 1000);
            let mut muxer: Box<dyn Muxer> = Box::new(MkvMuxer::new(Box::new(pipe_sink.clone())));
            muxer.write_header(&hdr).unwrap();

            for &(pts, is_key) in &[(0i64, true), (80, false), (40, false), (120, false)] {
                let mut pkt = Packet::new(
                    Buffer::from_slice(&[0, 0, 0, 2, 0x65, 0x88]),
                    TrackId(0),
                    tb,
                );
                pkt.pts = Timestamp::new(pts);
                pkt.dts = Timestamp::NONE;
                pkt.duration = 40;
                if is_key {
                    pkt.flags = PacketFlags::KEY;
                }
                muxer.write_packet(&pkt).unwrap();
            }
            muxer.finalize().unwrap();
        }
        let _ = MemorySink::new();

        // Demux the non-seekable output and verify B-frame PTS=40 after PTS=80 has dts=Timestamp::NONE!
        let muxed_bytes = pipe_sink.0.lock().unwrap().clone();
        let mut demuxer = MkvDemuxer::open(
            Box::new(MemorySource::new(muxed_bytes)),
            ErrorPolicy::Strict,
        )
        .unwrap();
        let mut pkts = Vec::new();
        while let RecvStatus::Ready(p) = demuxer.read_packet().unwrap() {
            pkts.push(p);
        }
        assert_eq!(pkts.len(), 4);
        assert_eq!(pkts[0].pts, Timestamp::new(0));
        assert_eq!(pkts[0].dts, Timestamp::new(0));
        assert_eq!(pkts[1].pts, Timestamp::new(80));
        assert_eq!(pkts[1].dts, Timestamp::new(80));
        // Packet 2 has PTS=40 < 80: DTS must NOT be fabricated as 40 (< 80)!
        assert_eq!(pkts[2].pts, Timestamp::new(40));
        assert_eq!(pkts[2].dts, Timestamp::NONE);
        // Once reordering was seen on this PTS-only SimpleBlock track, subsequent packets keep DTS=NONE
        assert_eq!(pkts[3].pts, Timestamp::new(120));
        assert_eq!(pkts[3].dts, Timestamp::NONE);

        let _ = SideData::SampleTrim {
            skip_start: 0,
            skip_end: 0,
        };
    }

    #[test]
    fn ffprobe_oracle_validates_muxed_mkv_video_and_audio() {
        let path = "/tmp/wedeo_test_mkv_ffprobe_oracle.mkv";
        let mut hdr = ContainerHeader::new("matroska");

        // Video track: H.264 320x240 with valid avcC
        let mut vp = VideoStreamParams::new(320, 240, PixelLayout::YUV420P);
        vp.color = ColorDesc::BT709_LIMITED;
        vp.avg_frame_rate = Rational::new(25, 1);
        vp.extradata = make_test_avcc();
        hdr.tracks.push(TrackDescriptor::new(
            TrackId(0),
            CodecId::H264,
            Rational::new(1, 1000),
            StreamParams::Video(vp),
        ));

        // Audio track: PCM S16LE 48kHz stereo
        let mut ap = AudioStreamParams::new(48_000, SampleFormat::S16, ChannelLayout::stereo());
        ap.bits_per_coded_sample = 16;
        hdr.tracks.push(TrackDescriptor::new(
            TrackId(1),
            CodecId::PcmS16le,
            Rational::new(1, 1000),
            StreamParams::Audio(ap),
        ));

        let sink = FileSink::create(path).unwrap();
        let mut muxer: Box<dyn Muxer> = Box::new(MkvMuxer::new(Box::new(sink)));
        muxer.write_header(&hdr).unwrap();

        let tb = Rational::new(1, 1000);
        for i in 0..5i64 {
            // Video packet (length-prefixed NAL)
            let mut vdata = vec![0, 0, 0, 4, 0x65, 0x88, 0x80, 0x40];
            if i > 0 {
                vdata[4] = 0x41;
            }
            let mut vpkt = Packet::new(Buffer::from_slice(&vdata), TrackId(0), tb);
            vpkt.pts = Timestamp::new(i * 40);
            vpkt.dts = Timestamp::new(i * 40);
            vpkt.duration = 40;
            if i == 0 {
                vpkt.flags = PacketFlags::KEY;
            }
            muxer.write_packet(&vpkt).unwrap();

            // Audio packet (40ms of 48kHz stereo S16LE = 1920 * 4 = 7680 bytes)
            let mut apkt = Packet::new(Buffer::from_slice(&[0u8; 7680]), TrackId(1), tb);
            apkt.pts = Timestamp::new(i * 40);
            apkt.dts = Timestamp::new(i * 40);
            apkt.duration = 40;
            apkt.flags = PacketFlags::KEY;
            muxer.write_packet(&apkt).unwrap();
        }
        muxer.finalize().unwrap();

        let ffprobe_bin = "/home/sharif/wedeo/target/ffmpeg-oracle/ffprobe";
        if std::path::Path::new(ffprobe_bin).exists() {
            let output = std::process::Command::new(ffprobe_bin)
                .args([
                    "-v",
                    "error",
                    "-show_format",
                    "-show_streams",
                    "-of",
                    "json",
                    path,
                ])
                .output()
                .expect("failed to execute ffprobe oracle");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "ffprobe oracle failed on muxed MKV: {stderr}"
            );
            assert!(
                stdout.contains("\"codec_name\": \"h264\""),
                "stdout: {stdout}"
            );
            assert!(
                stdout.contains("\"codec_name\": \"pcm_s16le\""),
                "stdout: {stdout}"
            );
            assert!(stdout.contains("\"width\": 320"), "stdout: {stdout}");
            assert!(stdout.contains("\"height\": 240"), "stdout: {stdout}");
            assert!(
                stdout.contains("\"sample_rate\": \"48000\""),
                "stdout: {stdout}"
            );
        }

        std::fs::remove_file(path).ok();
    }
}
