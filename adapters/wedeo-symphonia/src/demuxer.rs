use symphonia::core::formats::{FormatOptions, FormatReader, Track};
use symphonia::core::io::MediaSourceStream;
use symphonia::core::meta::MetadataOptions;
use symphonia::core::probe::Hint;

use wedeo_codec::decoder::CodecParameters;
use wedeo_core::codec_id::CodecId;
use wedeo_core::error::{Error, Result};
use wedeo_core::media_type::MediaType;
use wedeo_core::metadata::Metadata;
use wedeo_core::packet::{Packet, PacketFlags};
use wedeo_core::rational::Rational;
use wedeo_core::sample_format::SampleFormat;
use wedeo_format::demuxer::{
    Demuxer, DemuxerHeader, InputFormatDescriptor, InputFormatFlags, PROBE_SCORE_MAX, ProbeData,
    SeekFlags, Stream,
};
use wedeo_format::io::BufferedIo;
use wedeo_format::registry::DemuxerFactory;

use crate::channel_layout::channels_to_layout;
use crate::codec_map::symphonia_to_wedeo;
use crate::error::from_symphonia;
use crate::io_bridge::WedeoMediaSource;

/// Parse the iTunSMPB metadata tag used by Apple encoders to signal
/// encoder delay and padding for AAC in MP4/M4A containers.
///
/// Format: " 00000000 XXXXXXXX YYYYYYYY ZZZZZZZZZZZZZZZZ"
/// where XXXXXXXX is encoder delay (hex) and YYYYYYYY is padding (hex).
fn parse_itunsmpb(metadata: &Metadata) -> (u32, u32) {
    for (key, value) in metadata.iter() {
        if key.contains("iTunSMPB") {
            let parts: Vec<&str> = value.split_whitespace().collect();
            if parts.len() >= 3 {
                let delay = u32::from_str_radix(parts[1], 16).unwrap_or(0);
                let padding = u32::from_str_radix(parts[2], 16).unwrap_or(0);
                return (delay, padding);
            }
        }
    }
    (0, 0)
}

/// Symphonia demuxer wrapper.
struct SymphoniaDemuxerWrapper {
    reader: Option<Box<dyn FormatReader>>,
    /// Remaining encoder delay samples to trim across packets (from iTunSMPB).
    /// Decremented as trim_start is applied to successive packets.
    remaining_trim_start: u32,
    /// Encoder padding from iTunSMPB (AAC in MP4/M4A).
    /// Not currently applied (would need to know the last packet).
    #[allow(dead_code)]
    aac_priming_end: u32,
}

impl SymphoniaDemuxerWrapper {
    fn new() -> Self {
        Self {
            reader: None,
            remaining_trim_start: 0,
            aac_priming_end: 0,
        }
    }
}

/// Convert a symphonia Track to a wedeo Stream.
fn track_to_stream(track: &Track, index: usize) -> Stream {
    let codec_id = symphonia_to_wedeo(track.codec_params.codec);
    let mut params = CodecParameters::new(codec_id, MediaType::Audio);

    if let Some(sr) = track.codec_params.sample_rate {
        params.sample_rate = sr;
    }

    if let Some(channels) = track.codec_params.channels {
        let count = channels.count();
        params.channel_layout = channels_to_layout(channels, count);
    } else {
        // Container didn't provide channel info. Try codec-specific extradata.
        let ch_count = guess_channels_from_extradata(codec_id, &track.codec_params.extra_data);
        if ch_count > 0 {
            params.channel_layout = match ch_count {
                1 => wedeo_core::channel_layout::ChannelLayout::mono(),
                2 => wedeo_core::channel_layout::ChannelLayout::stereo(),
                6 => wedeo_core::channel_layout::ChannelLayout::surround_5_1(),
                8 => wedeo_core::channel_layout::ChannelLayout::surround_7_1(),
                n => wedeo_core::channel_layout::ChannelLayout::unspec(n as i32),
            };
        }
    }

    if let Some(bps) = track.codec_params.bits_per_coded_sample {
        params.bits_per_coded_sample = bps;
    }

    if let Some(bps) = track.codec_params.bits_per_sample {
        params.bits_per_raw_sample = bps;
    }

    if let Some(extra) = &track.codec_params.extra_data {
        params.extradata = extra.to_vec();
    }

    // Determine sample format from codec and bits
    params.sample_format = guess_sample_format(codec_id, params.bits_per_raw_sample);

    let mut stream = Stream::new(index, params);

    // Use symphonia's time base if available
    if let Some(tb) = track.codec_params.time_base {
        stream.time_base = Rational::new(tb.numer as i32, tb.denom as i32);
    } else if let Some(sr) = track.codec_params.sample_rate {
        stream.time_base = Rational::new(1, sr as i32);
    }

    if let Some(n_frames) = track.codec_params.n_frames {
        stream.nb_frames = n_frames as i64;
        stream.duration = n_frames as i64;
    }

    stream
}

/// Try to determine channel count from codec-specific extradata when the
/// container doesn't provide it.
fn guess_channels_from_extradata(codec_id: CodecId, extra: &Option<Box<[u8]>>) -> usize {
    let Some(data) = extra else { return 0 };
    match codec_id {
        // AAC AudioSpecificConfig: 5 bits audioObjectType + 4 bits frequencyIndex +
        // 4 bits channelConfiguration. The channel config is at bits 9-12.
        CodecId::Aac if data.len() >= 2 => {
            let channel_config = (data[1] >> 3) & 0x0F;
            match channel_config {
                1 => 1, // mono
                2 => 2, // stereo
                3 => 3, // 3.0
                4 => 4, // 4.0
                5 => 5, // 5.0
                6 => 6, // 5.1
                7 => 8, // 7.1
                _ => 0,
            }
        }
        // Opus ID header: byte 9 = channel count (RFC 7845)
        CodecId::Opus if data.len() >= 10 && data.starts_with(b"OpusHead") => data[9] as usize,
        _ => 0,
    }
}

/// Guess the output sample format based on codec and bit depth.
fn guess_sample_format(codec_id: CodecId, bits_per_sample: u32) -> SampleFormat {
    match codec_id {
        CodecId::PcmU8 => SampleFormat::U8,
        CodecId::PcmS16le | CodecId::PcmS16be | CodecId::PcmU16le | CodecId::PcmU16be => {
            SampleFormat::S16
        }
        CodecId::PcmS24le | CodecId::PcmS24be | CodecId::PcmU24le | CodecId::PcmU24be => {
            SampleFormat::S32
        }
        CodecId::PcmS32le | CodecId::PcmS32be | CodecId::PcmU32le | CodecId::PcmU32be => {
            SampleFormat::S32
        }
        CodecId::PcmF32le | CodecId::PcmF32be => SampleFormat::Flt,
        CodecId::PcmF64le | CodecId::PcmF64be => SampleFormat::Dbl,
        CodecId::Flac | CodecId::Alac | CodecId::WavPack => match bits_per_sample {
            0..=16 => SampleFormat::S16,
            17..=32 => SampleFormat::S32,
            _ => SampleFormat::S32,
        },
        // Lossy codecs generally decode to float
        CodecId::Mp1
        | CodecId::Mp2
        | CodecId::Mp3
        | CodecId::Aac
        | CodecId::Vorbis
        | CodecId::Opus => SampleFormat::Flt,
        CodecId::AdpcmImaWav | CodecId::AdpcmMs => SampleFormat::S16,
        _ => SampleFormat::S32,
    }
}

impl Demuxer for SymphoniaDemuxerWrapper {
    fn read_header(&mut self, io: &mut BufferedIo) -> Result<DemuxerHeader> {
        let io_ctx = io.take_inner();
        let source = WedeoMediaSource::new(io_ctx);
        let mss = MediaSourceStream::new(Box::new(source), Default::default());

        let fmt_opts = FormatOptions {
            enable_gapless: true,
            ..Default::default()
        };
        let meta_opts = MetadataOptions::default();
        let hint = Hint::new();

        let probed = symphonia::default::get_probe()
            .format(&hint, mss, &fmt_opts, &meta_opts)
            .map_err(from_symphonia)?;

        let mut reader = probed.format;

        // Convert tracks to streams
        let streams: Vec<Stream> = reader
            .tracks()
            .iter()
            .enumerate()
            .map(|(i, t)| track_to_stream(t, i))
            .collect();

        // Convert metadata
        let metadata = if let Some(rev) = reader.metadata().current() {
            crate::metadata::convert_metadata(rev)
        } else {
            Metadata::new()
        };

        // Parse iTunSMPB for AAC gapless trimming in MP4/M4A containers.
        // Symphonia's gapless mode handles MP3 and OGG natively, but not AAC in MP4.
        let (aac_start, aac_end) = parse_itunsmpb(&metadata);
        self.remaining_trim_start = aac_start;
        self.aac_priming_end = aac_end;

        self.reader = Some(reader);

        Ok(DemuxerHeader {
            streams,
            metadata,
            duration: 0,
            start_time: 0,
        })
    }

    fn read_packet(&mut self, _io: &mut BufferedIo) -> Result<Packet> {
        let reader = self.reader.as_mut().ok_or(Error::Bug)?;

        match reader.next_packet() {
            Ok(sym_pkt) => {
                let data = sym_pkt.buf().to_vec();
                let mut pkt = Packet::from_slice(&data);
                pkt.stream_index = sym_pkt.track_id() as usize;
                pkt.pts = sym_pkt.ts() as i64;
                pkt.dts = sym_pkt.ts() as i64;
                pkt.duration = sym_pkt.dur() as i64;
                pkt.flags = PacketFlags::KEY;

                // Copy gapless trim info from symphonia (handles MP3, OGG natively)
                pkt.trim_start = sym_pkt.trim_start();
                pkt.trim_end = sym_pkt.trim_end();

                // Apply AAC priming from iTunSMPB across successive packets.
                // The encoder delay (e.g. 2112 samples) may span multiple AAC
                // frames (1024 samples each), so we spread trim_start across
                // packets until the full delay is consumed.
                if self.remaining_trim_start > 0 && pkt.trim_start == 0 {
                    let packet_dur = pkt.duration.max(0) as u32;
                    let trim = self.remaining_trim_start.min(packet_dur);
                    pkt.trim_start = trim;
                    self.remaining_trim_start -= trim;
                }

                Ok(pkt)
            }
            Err(symphonia::core::errors::Error::IoError(ref e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                Err(Error::Eof)
            }
            Err(e) => Err(from_symphonia(e)),
        }
    }

    fn seek(
        &mut self,
        _io: &mut BufferedIo,
        _stream_index: usize,
        timestamp: i64,
        _flags: SeekFlags,
    ) -> Result<()> {
        let reader = self.reader.as_mut().ok_or(Error::Bug)?;

        let seek_to = symphonia::core::formats::SeekTo::TimeStamp {
            ts: timestamp as u64,
            track_id: 0,
        };

        reader
            .seek(symphonia::core::formats::SeekMode::Accurate, seek_to)
            .map_err(from_symphonia)?;

        Ok(())
    }
}

// --- Factory registration macro ---

macro_rules! register_symphonia_demuxer {
    ($factory_name:ident, $name:expr, $long_name:expr, $extensions:expr, $mime:expr, $probe_fn:expr) => {
        register_symphonia_demuxer_pri!(
            $factory_name,
            $name,
            $long_name,
            $extensions,
            $mime,
            $probe_fn,
            50
        );
    };
}

macro_rules! register_symphonia_demuxer_pri {
    ($factory_name:ident, $name:expr, $long_name:expr, $extensions:expr, $mime:expr, $probe_fn:expr, $pri:expr) => {
        struct $factory_name;

        impl DemuxerFactory for $factory_name {
            fn descriptor(&self) -> &InputFormatDescriptor {
                static DESC: InputFormatDescriptor = InputFormatDescriptor {
                    name: $name,
                    long_name: $long_name,
                    extensions: $extensions,
                    mime_types: $mime,
                    flags: InputFormatFlags::empty(),
                    priority: $pri,
                };
                &DESC
            }

            fn probe(&self, data: &ProbeData<'_>) -> i32 {
                $probe_fn(data)
            }

            fn create(&self) -> Result<Box<dyn Demuxer>> {
                Ok(Box::new(SymphoniaDemuxerWrapper::new()))
            }
        }

        inventory::submit!(&$factory_name as &dyn DemuxerFactory);
    };
}

fn probe_ogg(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() >= 4 && &data.buf[0..4] == b"OggS" {
        90
    } else {
        0
    }
}

fn probe_flac(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() >= 4 && &data.buf[0..4] == b"fLaC" {
        90
    } else {
        0
    }
}

fn probe_mp4(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() < 8 {
        return 0;
    }
    let tag = &data.buf[4..8];
    if tag == b"ftyp" || tag == b"moov" || tag == b"mdat" || tag == b"free" || tag == b"wide" {
        80
    } else {
        0
    }
}

fn probe_mkv(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() >= 4 && data.buf[0..4] == [0x1A, 0x45, 0xDF, 0xA3] {
        90
    } else {
        0
    }
}

fn probe_aiff(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() >= 12
        && &data.buf[0..4] == b"FORM"
        && (&data.buf[8..12] == b"AIFF" || &data.buf[8..12] == b"AIFC")
    {
        90
    } else {
        0
    }
}

fn probe_caf(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() >= 4 && &data.buf[0..4] == b"caff" {
        90
    } else {
        0
    }
}

fn probe_mp3(data: &ProbeData<'_>) -> i32 {
    if data.buf.len() < 3 {
        return 0;
    }
    // ID3v2 tag
    if &data.buf[0..3] == b"ID3" {
        80
    }
    // MPEG sync word: 0xFF followed by 0xE0+ (sync bits)
    else if data.buf[0] == 0xFF && (data.buf[1] & 0xE0) == 0xE0 {
        50
    } else {
        // Check file extension
        if data
            .filename
            .rsplit('.')
            .next()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("mp3"))
        {
            30
        } else {
            0
        }
    }
}

fn probe_wav(data: &ProbeData<'_>) -> i32 {
    // FFmpeg requires buf_size > 32 (i.e., at least 33 bytes)
    if data.buf.len() <= 32 {
        return 0;
    }

    let magic = &data.buf[0..4];
    let form = &data.buf[8..12];

    // RIFF or RIFX (big-endian RIFF) + WAVE
    // Since the ACT demuxer has a standard WAV header at the top of
    // its own, the returned score is decreased to avoid a probe
    // conflict between ACT and WAV.
    if (magic == b"RIFF" || magic == b"RIFX") && form == b"WAVE" {
        return PROBE_SCORE_MAX - 1;
    }

    // RF64 or BW64 + WAVE + ds64
    if (magic == b"RF64" || magic == b"BW64")
        && form == b"WAVE"
        && data.buf.len() > 16
        && &data.buf[12..16] == b"ds64"
    {
        return PROBE_SCORE_MAX;
    }

    // Check file extension as fallback
    if data
        .filename
        .rsplit('.')
        .next()
        .is_some_and(|ext| ext.eq_ignore_ascii_case("wav"))
    {
        return 50; // PROBE_SCORE_EXTENSION
    }

    0
}

// Register per-format demuxer factories
register_symphonia_demuxer!(
    SymphoniaOggDemuxerFactory,
    "ogg_symphonia",
    "OGG (Ogg container) [symphonia]",
    "ogg,oga,ogx,spx,opus",
    "audio/ogg,application/ogg",
    probe_ogg
);

register_symphonia_demuxer!(
    SymphoniaFlacDemuxerFactory,
    "flac_symphonia",
    "FLAC (Free Lossless Audio Codec) [symphonia]",
    "flac",
    "audio/flac,audio/x-flac",
    probe_flac
);

register_symphonia_demuxer!(
    SymphoniaMp4DemuxerFactory,
    "mp4_symphonia",
    "MP4 (MPEG-4 Part 14) [symphonia]",
    "mp4,m4a,m4b,m4v,mov",
    "audio/mp4,video/mp4",
    probe_mp4
);

register_symphonia_demuxer!(
    SymphoniaMkvDemuxerFactory,
    "mkv_symphonia",
    "MKV (Matroska) [symphonia]",
    "mkv,mka,webm",
    "video/x-matroska,audio/x-matroska,video/webm",
    probe_mkv
);

register_symphonia_demuxer!(
    SymphoniaAiffDemuxerFactory,
    "aiff_symphonia",
    "AIFF (Audio Interchange File Format) [symphonia]",
    "aiff,aif,aifc",
    "audio/aiff,audio/x-aiff",
    probe_aiff
);

register_symphonia_demuxer!(
    SymphoniaCafDemuxerFactory,
    "caf_symphonia",
    "CAF (Core Audio Format) [symphonia]",
    "caf",
    "audio/x-caf",
    probe_caf
);

register_symphonia_demuxer!(
    SymphoniaMp3DemuxerFactory,
    "mp3_symphonia",
    "MP3 (MPEG audio layer 3) [symphonia]",
    "mp3",
    "audio/mpeg",
    probe_mp3
);

register_symphonia_demuxer_pri!(
    SymphoniaWavDemuxerFactory,
    "wav",
    "WAV / WAVE [symphonia]",
    "wav,wave",
    "audio/x-wav,audio/wav",
    probe_wav,
    100
);
