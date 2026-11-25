#![forbid(unsafe_code)]
//! Native bit-exact FFmpeg RIFF/RIFX/RF64/BW64 WAV demuxer and muxer for Wedeo.

use std::io::{self, Read, SeekFrom};
use std::sync::Arc;

use wedeo_core::{
    AudioStreamParams, BufByteSource, Buffer, ByteSink, ByteSource, Channel, ChannelLayout,
    CodecId, ContainerHeader, Demuxer, DemuxerDescriptor, DemuxerFactory, Error, ErrorPolicy,
    HealthEvent, HealthIssue, Muxer, MuxerDescriptor, MuxerFactory, Packet, PacketFlags, Rational,
    RecvStatus, Registry, Result, SampleFormat, SeekTarget, StreamHealth, StreamParams,
    TIME_BASE_Q, Timestamp, TrackDescriptor, TrackId, rescale_q,
};

const COMPONENT: &str = "wav";

/// Default maximum packet payload size matching FFmpeg `wavdec.c` (`4096`).
const MAX_SIZE: usize = 4096;

/// WAVE format codes.
const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_ADPCM_MS: u16 = 0x0002;
const WAVE_FORMAT_IEEE_FLOAT: u16 = 0x0003;
const WAVE_FORMAT_ALAW: u16 = 0x0006;
const WAVE_FORMAT_MULAW: u16 = 0x0007;
const WAVE_FORMAT_ADPCM_IMA_WAV: u16 = 0x0011;
const WAVE_FORMAT_MP2: u16 = 0x0050;
const WAVE_FORMAT_MP3: u16 = 0x0055;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;

/// `KSDATAFORMAT_SUBTYPE_WAVEFORMATEX` base GUID tail (bytes 4..16).
const MEDIASUBTYPE_BASE_GUID: [u8; 12] = [
    0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38, 0x9b, 0x71,
];

/// `FF_AMBISONIC_BASE_GUID` tail (bytes 4..16).
const AMBISONIC_BASE_GUID: [u8; 12] = [
    0x21, 0x07, 0xd3, 0x11, 0x86, 0x44, 0xc8, 0xc1, 0xca, 0x00, 0x00, 0x00,
];

/// `FF_BROKEN_BASE_GUID` tail (bytes 4..16).
const BROKEN_BASE_GUID: [u8; 12] = [
    0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xaa, 0x00, 0x38,
];

/// Map `CodecId` to `(wFormatTag, wBitsPerSample)`.
fn codec_id_to_wav_format(codec_id: CodecId) -> Option<(u16, u16)> {
    match codec_id {
        CodecId::PcmU8 => Some((WAVE_FORMAT_PCM, 8)),
        CodecId::PcmS16le => Some((WAVE_FORMAT_PCM, 16)),
        CodecId::PcmS24le => Some((WAVE_FORMAT_PCM, 24)),
        CodecId::PcmS32le => Some((WAVE_FORMAT_PCM, 32)),
        CodecId::PcmF32le => Some((WAVE_FORMAT_IEEE_FLOAT, 32)),
        CodecId::PcmF64le => Some((WAVE_FORMAT_IEEE_FLOAT, 64)),
        CodecId::PcmAlaw => Some((WAVE_FORMAT_ALAW, 8)),
        CodecId::PcmMulaw => Some((WAVE_FORMAT_MULAW, 8)),
        _ => None,
    }
}

/// Map `(wFormatTag, wBitsPerSample, big_endian)` to `(CodecId, SampleFormat)`.
fn wav_format_to_codec(
    tag: u16,
    bits_per_sample: u16,
    big_endian: bool,
) -> Option<(CodecId, SampleFormat)> {
    match tag {
        WAVE_FORMAT_PCM => match bits_per_sample {
            8 => Some((CodecId::PcmU8, SampleFormat::U8)),
            16 => Some((
                if big_endian {
                    CodecId::PcmS16be
                } else {
                    CodecId::PcmS16le
                },
                SampleFormat::S16,
            )),
            24 => Some((
                if big_endian {
                    CodecId::PcmS24be
                } else {
                    CodecId::PcmS24le
                },
                SampleFormat::S32,
            )),
            32 => Some((
                if big_endian {
                    CodecId::PcmS32be
                } else {
                    CodecId::PcmS32le
                },
                SampleFormat::S32,
            )),
            _ => None,
        },
        WAVE_FORMAT_IEEE_FLOAT => match bits_per_sample {
            32 => Some((
                if big_endian {
                    CodecId::PcmF32be
                } else {
                    CodecId::PcmF32le
                },
                SampleFormat::F32,
            )),
            64 => Some((
                if big_endian {
                    CodecId::PcmF64be
                } else {
                    CodecId::PcmF64le
                },
                SampleFormat::F64,
            )),
            _ => None,
        },
        WAVE_FORMAT_ALAW => Some((CodecId::PcmAlaw, SampleFormat::S16)),
        WAVE_FORMAT_MULAW => Some((CodecId::PcmMulaw, SampleFormat::S16)),
        WAVE_FORMAT_ADPCM_MS => Some((CodecId::AdpcmMs, SampleFormat::S16)),
        WAVE_FORMAT_ADPCM_IMA_WAV => Some((CodecId::AdpcmImaWav, SampleFormat::S16)),
        WAVE_FORMAT_MP2 => Some((CodecId::Mp2, SampleFormat::S16)),
        WAVE_FORMAT_MP3 => Some((CodecId::Mp3, SampleFormat::S16)),
        _ => None,
    }
}

/// Compute `WAVEFORMATEXTENSIBLE` `dwChannelMask` from a `ChannelLayout`.
/// Returns `None` if the layout is unspecified or contains channels outside the 18-bit WAV mask.
pub fn channel_layout_to_wav_mask(layout: &ChannelLayout) -> Option<u32> {
    if layout.channels.is_empty() {
        return None;
    }
    let mut mask = 0u32;
    let mut prev_bit = 0u32;
    for &ch in &layout.channels {
        let bit = match ch {
            Channel::FrontLeft => 1 << 0,
            Channel::FrontRight => 1 << 1,
            Channel::FrontCenter => 1 << 2,
            Channel::LowFrequency => 1 << 3,
            Channel::BackLeft => 1 << 4,
            Channel::BackRight => 1 << 5,
            Channel::FrontLeftOfCenter => 1 << 6,
            Channel::FrontRightOfCenter => 1 << 7,
            Channel::BackCenter => 1 << 8,
            Channel::SideLeft => 1 << 9,
            Channel::SideRight => 1 << 10,
            Channel::TopCenter => 1 << 11,
            Channel::TopFrontLeft => 1 << 12,
            Channel::TopFrontCenter => 1 << 13,
            Channel::TopFrontRight => 1 << 14,
            Channel::TopBackLeft => 1 << 15,
            Channel::TopBackCenter => 1 << 16,
            Channel::TopBackRight => 1 << 17,
            _ => return None,
        };
        if bit <= prev_bit {
            return None;
        }
        prev_bit = bit;
        mask |= bit;
    }
    Some(mask)
}

fn needs_waveformatextensible(layout: &ChannelLayout, mask: Option<u32>) -> bool {
    let Some(mask) = mask else {
        return false;
    };
    match layout.nb_channels {
        1 => mask != (1 << 2),
        2 => mask != ((1 << 0) | (1 << 1)),
        n if n > 2 => true,
        _ => false,
    }
}

fn riff_info_tag_to_key(tag: &[u8; 4]) -> Option<&'static str> {
    match tag {
        b"IART" => Some("artist"),
        b"ICMT" => Some("comment"),
        b"ICOP" => Some("copyright"),
        b"ICRD" => Some("date"),
        b"IGNR" => Some("genre"),
        b"INAM" => Some("title"),
        b"IPRD" => Some("album"),
        b"ISFT" => Some("encoder"),
        b"ITRK" => Some("track"),
        b"IALB" => Some("album"),
        b"IENG" => Some("engineer"),
        b"ISRC" => Some("source"),
        b"ISBJ" => Some("subject"),
        b"ITCH" => Some("technician"),
        _ => None,
    }
}

// --- WAV Muxer ---

/// Native WAV/RF64 muxer owning its `ByteSink`.
pub struct WavMuxer {
    sink: Box<dyn ByteSink>,
    format_tag: u16,
    nb_channels: u16,
    sample_rate: u32,
    byte_rate: u32,
    block_align: u16,
    bits_per_sample: u16,
    channel_mask: Option<u32>,
    extensible: bool,
    bytes_written: u64,
    data_size: u64,
    fmt_offset: u64,
    data_size_pos: u64,
    data_offset: u64,
    reserved_ds64_junk: bool,
    header_written: bool,
}

impl WavMuxer {
    pub fn new(sink: Box<dyn ByteSink>) -> Self {
        Self {
            sink,
            format_tag: 0,
            nb_channels: 0,
            sample_rate: 0,
            byte_rate: 0,
            block_align: 0,
            bits_per_sample: 0,
            channel_mask: None,
            extensible: false,
            bytes_written: 0,
            data_size: 0,
            fmt_offset: 0,
            data_size_pos: 0,
            data_offset: 0,
            reserved_ds64_junk: false,
            header_written: false,
        }
    }

    fn write_all(&mut self, buf: &[u8]) -> Result<()> {
        self.sink
            .write_all(buf)
            .map_err(|e| Error::io(COMPONENT, &e))?;
        self.bytes_written = self.bytes_written.saturating_add(buf.len() as u64);
        Ok(())
    }

    fn write_u16le(&mut self, v: u16) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    fn write_u32le(&mut self, v: u32) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    fn write_u64le(&mut self, v: u64) -> Result<()> {
        self.write_all(&v.to_le_bytes())
    }

    fn write_fmt_chunk(&mut self) -> Result<()> {
        self.write_all(b"fmt ")?;
        if self.extensible {
            self.write_u32le(40)?;
            self.write_u16le(WAVE_FORMAT_EXTENSIBLE)?;
            self.write_u16le(self.nb_channels)?;
            self.write_u32le(self.sample_rate)?;
            self.write_u32le(self.byte_rate)?;
            self.write_u16le(self.block_align)?;
            self.write_u16le(self.bits_per_sample)?;
            self.write_u16le(22)?; // cbSize
            self.write_u16le(self.bits_per_sample)?; // wValidBitsPerSample
            self.write_u32le(self.channel_mask.unwrap_or(0))?;
            self.write_u32le(u32::from(self.format_tag))?;
            self.write_all(&MEDIASUBTYPE_BASE_GUID)?;
        } else {
            self.write_u32le(16)?;
            self.write_u16le(self.format_tag)?;
            self.write_u16le(self.nb_channels)?;
            self.write_u32le(self.sample_rate)?;
            self.write_u32le(self.byte_rate)?;
            self.write_u16le(self.block_align)?;
            self.write_u16le(self.bits_per_sample)?;
        }
        Ok(())
    }
}

impl Muxer for WavMuxer {
    fn write_header(&mut self, header: &ContainerHeader) -> Result<()> {
        let track = header.tracks.first().ok_or_else(|| {
            Error::invalid_argument(COMPONENT, "wav muxer requires 1 audio track")
        })?;
        let audio = track.params.audio().ok_or_else(|| {
            Error::invalid_argument(COMPONENT, "wav muxer only supports audio tracks")
        })?;
        let (format_tag, bits_per_sample) =
            codec_id_to_wav_format(track.codec_id).ok_or_else(|| {
                Error::unsupported(
                    COMPONENT,
                    format!("unsupported codec for wav: {}", track.codec_id.name()),
                )
            })?;

        let nb_channels = u16::try_from(audio.channel_layout.nb_channels)
            .map_err(|_| Error::invalid_argument(COMPONENT, "invalid channel count for wav"))?;
        if nb_channels == 0 || audio.sample_rate == 0 {
            return Err(Error::invalid_argument(
                COMPONENT,
                "wav sample_rate and channel count must be non-zero",
            ));
        }

        let block_align = nb_channels * (bits_per_sample / 8);
        let byte_rate = audio.sample_rate.saturating_mul(u32::from(block_align));
        let channel_mask = channel_layout_to_wav_mask(&audio.channel_layout);
        let extensible = needs_waveformatextensible(&audio.channel_layout, channel_mask);

        self.format_tag = format_tag;
        self.nb_channels = nb_channels;
        self.sample_rate = audio.sample_rate;
        self.byte_rate = byte_rate;
        self.block_align = block_align;
        self.bits_per_sample = bits_per_sample;
        self.channel_mask = channel_mask;
        self.extensible = extensible;

        // RIFF header (12 bytes)
        self.write_all(b"RIFF")?;
        self.write_u32le(0xFFFF_FFFF)?;
        self.write_all(b"WAVE")?;

        // fmt chunk
        self.fmt_offset = self.bytes_written;
        self.write_fmt_chunk()?;

        // Reserve 36-byte JUNK chunk on seekable sinks so RF64 (>4GB) can upgrade in-place
        // by overwriting fmt+JUNK with ds64+fmt without shifting the data chunk.
        if self.sink.is_seekable() {
            self.write_all(b"JUNK")?;
            self.write_u32le(28)?;
            self.write_all(&[0u8; 28])?;
            self.reserved_ds64_junk = true;
        }

        // data chunk header
        self.write_all(b"data")?;
        self.data_size_pos = self.bytes_written;
        self.write_u32le(0xFFFF_FFFF)?;
        self.data_offset = self.bytes_written;
        self.header_written = true;

        self.sink.flush().map_err(|e| Error::io(COMPONENT, &e))?;
        Ok(())
    }

    fn write_packet(&mut self, pkt: &Packet) -> Result<()> {
        if !self.header_written {
            return Err(Error::invalid_argument(
                COMPONENT,
                "write_packet called before write_header",
            ));
        }
        let data = pkt.data.data();
        self.write_all(data)?;
        self.data_size = self.data_size.saturating_add(data.len() as u64);
        Ok(())
    }

    fn finalize(mut self: Box<Self>) -> Result<()> {
        if !self.header_written {
            return Ok(());
        }

        // Pad to even boundary if data size is odd (RIFF/RF64 requirement).
        if !self.data_size.is_multiple_of(2) {
            self.write_all(&[0u8])?;
        }
        self.sink.flush().map_err(|e| Error::io(COMPONENT, &e))?;

        if self.sink.is_seekable() {
            let file_size = self.bytes_written;
            let riff_payload_size = file_size.saturating_sub(8);
            let use_rf64 = self.reserved_ds64_junk
                && (riff_payload_size > u64::from(u32::MAX)
                    || self.data_size >= u64::from(u32::MAX));

            if use_rf64 {
                self.sink
                    .seek(SeekFrom::Start(0))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                self.sink
                    .write_all(b"RF64")
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                self.sink
                    .write_all(&0xFFFF_FFFFu32.to_le_bytes())
                    .map_err(|e| Error::io(COMPONENT, &e))?;

                // Write ds64 + fmt over fmt + JUNK
                self.sink
                    .seek(SeekFrom::Start(self.fmt_offset))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                let sample_count = if self.block_align > 0 {
                    self.data_size / u64::from(self.block_align)
                } else {
                    0
                };
                self.write_all(b"ds64")?;
                self.write_u32le(28)?;
                self.write_u64le(riff_payload_size)?;
                self.write_u64le(self.data_size)?;
                self.write_u64le(sample_count)?;
                self.write_u32le(0)?;
                self.write_fmt_chunk()?;

                self.sink
                    .seek(SeekFrom::Start(self.data_size_pos))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                self.sink
                    .write_all(&0xFFFF_FFFFu32.to_le_bytes())
                    .map_err(|e| Error::io(COMPONENT, &e))?;
            } else {
                self.sink
                    .seek(SeekFrom::Start(4))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                self.sink
                    .write_all(&(riff_payload_size as u32).to_le_bytes())
                    .map_err(|e| Error::io(COMPONENT, &e))?;

                self.sink
                    .seek(SeekFrom::Start(self.data_size_pos))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
                self.sink
                    .write_all(&(self.data_size as u32).to_le_bytes())
                    .map_err(|e| Error::io(COMPONENT, &e))?;
            }

            self.sink.flush().map_err(|e| Error::io(COMPONENT, &e))?;
        }

        Ok(())
    }
}

// --- WAV Demuxer ---

/// Native bit-exact FFmpeg RIFF/RIFX/RF64/BW64 WAV demuxer.
pub struct WavDemuxer {
    src: BufByteSource,
    header: ContainerHeader,
    health: StreamHealth,
    policy: ErrorPolicy,
    rifx: bool,
    data_ofs: u64,
    data_size: u64,
    data_end: u64,
    block_align: usize,
    max_size: usize,
    current_sample: u64,
    eof: bool,
}

impl WavDemuxer {
    fn open(source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Self> {
        let mut src = BufByteSource::new(source);
        let start_pos = src.tell();
        let unaligned = (start_pos & 1) != 0;

        let riff_tag = src
            .read_tag()
            .map_err(|e| Error::invalid_data(COMPONENT, format!("truncated WAV header: {e}")))?;
        let (rifx, rf64) = match &riff_tag {
            b"RIFF" => (false, false),
            b"RIFX" => (true, false),
            b"RF64" | b"BW64" => (false, true),
            _ => {
                return Err(Error::invalid_data(
                    COMPONENT,
                    "invalid start code in RIFF header",
                ));
            }
        };

        let _riff_size = read_tag_size(&mut src, rifx)?;
        let wave_tag = src
            .read_tag()
            .map_err(|e| Error::invalid_data(COMPONENT, format!("truncated WAVE tag: {e}")))?;
        if &wave_tag != b"WAVE" {
            return Err(Error::invalid_data(
                COMPONENT,
                "invalid format in RIFF header (expected WAVE)",
            ));
        }

        let mut ds64_data_size: Option<u64> = None;
        let mut sample_count: u64 = 0;

        if rf64 {
            let ds64_tag = src.read_tag().map_err(|e| {
                Error::invalid_data(COMPONENT, format!("missing ds64 chunk in RF64: {e}"))
            })?;
            if &ds64_tag != b"ds64" {
                return Err(Error::invalid_data(
                    COMPONENT,
                    "expected ds64 chunk after RF64 WAVE header",
                ));
            }
            let ds64_size = u64::from(src.read_u32le().map_err(|e| Error::io(COMPONENT, &e))?);
            if ds64_size < 24 {
                return Err(Error::invalid_data(COMPONENT, "ds64 chunk too small"));
            }
            let _rf64_riff_size = src.read_u64le().map_err(|e| Error::io(COMPONENT, &e))?;
            let data_sz = src.read_u64le().map_err(|e| Error::io(COMPONENT, &e))?;
            let samp_cnt = src.read_u64le().map_err(|e| Error::io(COMPONENT, &e))?;
            ds64_data_size = Some(data_sz);
            sample_count = samp_cnt;
            let rem = ds64_size - 24;
            if rem > 0 {
                src.skip(rem + (ds64_size & 1))
                    .map_err(|e| Error::io(COMPONENT, &e))?;
            }
        }

        let mut header = ContainerHeader::new("wav");
        let mut health = StreamHealth::new();
        let mut track: Option<TrackDescriptor> = None;
        let mut data_ofs: Option<u64> = None;
        let mut data_size: u64 = 0;
        let mut data_end: u64 = u64::MAX;

        let file_size = src.size_hint();

        loop {
            if let Some(fs) = file_size
                && src.tell() >= fs
            {
                break;
            }
            if ((src.tell() + u64::from(unaligned)) & 1) != 0 && src.skip(1).is_err() {
                break;
            }

            let tag = match src.read_tag() {
                Ok(t) => t,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(Error::io(COMPONENT, &e)),
            };
            let size = match read_tag_size(&mut src, rifx) {
                Ok(s) => u64::from(s),
                Err(e) => {
                    if policy == ErrorPolicy::Strict {
                        return Err(e);
                    }
                    health.record(HealthEvent {
                        track: TrackId(0),
                        pts: Timestamp::NONE,
                        byte_offset: Some(src.tell()),
                        kind: HealthIssue::CorruptSyntax,
                        detail: "truncated chunk size in WAV header",
                    });
                    break;
                }
            };

            let chunk_start = src.tell();
            let next_tag_ofs = chunk_start.saturating_add(size).saturating_add(size & 1);

            match &tag {
                b"fmt " => {
                    if track.is_none() {
                        track = Some(parse_fmt_chunk(&mut src, size, rifx)?);
                    }
                }
                b"fact" => {
                    if size >= 4 && sample_count == 0 {
                        let cnt = if rifx {
                            src.read_u32be()
                        } else {
                            src.read_u32le()
                        }
                        .map_err(|e| Error::io(COMPONENT, &e))?;
                        sample_count = u64::from(cnt);
                    }
                }
                b"LIST" | b"list" => {
                    if size >= 4 {
                        let list_type = src.read_tag().map_err(|e| Error::io(COMPONENT, &e))?;
                        if &list_type == b"INFO" {
                            parse_riff_info(&mut src, size - 4, &mut header, &mut health, policy)?;
                        }
                    }
                }
                b"data" => {
                    if !src.is_seekable() && track.is_none() {
                        return Err(Error::invalid_data(
                            COMPONENT,
                            "found no 'fmt ' tag before 'data' tag",
                        ));
                    }
                    data_ofs = Some(chunk_start);
                    if let Some(ds64_sz) = ds64_data_size {
                        data_size = ds64_sz;
                        data_end = chunk_start.saturating_add(ds64_sz);
                    } else if size > 0 && size != u64::from(u32::MAX) {
                        data_size = size;
                        data_end = chunk_start.saturating_add(size);
                    } else {
                        data_size = u64::MAX;
                        data_end = u64::MAX;
                    }

                    // On non-seekable streams or unbounded data chunks, stop scanning so we can
                    // start streaming audio packets immediately from `data_ofs`.
                    if !src.is_seekable() || data_size == u64::MAX {
                        break;
                    }
                    let next_data_ofs = data_end.saturating_add(data_size & 1);
                    if let Some(fs) = file_size
                        && next_data_ofs >= fs
                    {
                        break;
                    }
                    if src.seek_abs(next_data_ofs).is_err() {
                        break;
                    }
                    continue;
                }
                _ => {}
            }

            if let Some(fs) = file_size
                && next_tag_ofs >= fs
            {
                break;
            }
            if src.is_seekable() {
                if src.seek_abs(next_tag_ofs).is_err() {
                    break;
                }
            } else {
                let consumed = src.tell().saturating_sub(chunk_start);
                let to_skip = (size + (size & 1)).saturating_sub(consumed);
                if to_skip > 0 && src.skip(to_skip).is_err() {
                    break;
                }
            }
        }

        let mut track =
            track.ok_or_else(|| Error::invalid_data(COMPONENT, "no 'fmt ' tag found"))?;
        let data_ofs =
            data_ofs.ok_or_else(|| Error::invalid_data(COMPONENT, "no 'data' tag found"))?;

        if src.is_seekable() && src.tell() != data_ofs {
            src.seek_abs(data_ofs)
                .map_err(|e| Error::io(COMPONENT, &e))?;
        }

        let audio = track
            .params
            .audio()
            .ok_or_else(|| Error::internal(COMPONENT, "wav track missing audio stream params"))?;
        let block_align = audio.block_align as usize;
        let nb_channels = u64::from(audio.channel_layout.nb_channels);

        if sample_count == 0 && data_size != u64::MAX && block_align > 0 && nb_channels > 0 {
            sample_count = data_size / block_align as u64;
        }
        if sample_count > 0 {
            track.duration = sample_count as i64;
            header.duration = Some(rescale_q(sample_count as i64, track.time_base, TIME_BASE_Q));
        }

        let max_size = if block_align > 1 {
            if MAX_SIZE < block_align {
                block_align
            } else {
                (MAX_SIZE / block_align) * block_align
            }
        } else {
            MAX_SIZE
        };

        header.tracks.push(track);

        Ok(Self {
            src,
            header,
            health,
            policy,
            rifx,
            data_ofs,
            data_size,
            data_end,
            block_align,
            max_size,
            current_sample: 0,
            eof: false,
        })
    }

    fn find_next_data_chunk(&mut self) -> Result<Option<u64>> {
        if !self.src.is_seekable() {
            return Ok(None);
        }
        loop {
            if (self.src.tell() & 1) != 0 && self.src.skip(1).is_err() {
                return Ok(None);
            }
            let tag = match self.src.read_tag() {
                Ok(t) => t,
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
                Err(e) => return Err(Error::io(COMPONENT, &e)),
            };
            let size = match read_tag_size(&mut self.src, self.rifx) {
                Ok(s) => u64::from(s),
                Err(_) => return Ok(None),
            };
            if &tag == b"data" {
                return Ok(Some(size));
            }
            let skip_len = size.saturating_add(size & 1);
            if self.src.skip(skip_len).is_err() {
                return Ok(None);
            }
        }
    }
}

impl Demuxer for WavDemuxer {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        if self.eof {
            return Ok(RecvStatus::Eof);
        }

        let mut left = self.data_end.saturating_sub(self.src.tell());
        if left == 0 {
            match self.find_next_data_chunk()? {
                Some(new_size) if new_size > 0 => {
                    self.data_end = self.src.tell().saturating_add(new_size);
                    left = new_size;
                }
                _ => {
                    self.eof = true;
                    return Ok(RecvStatus::Eof);
                }
            }
        }

        let to_read = (self.max_size as u64).min(left) as usize;
        let pkt_pos = self.src.tell();
        let mut raw = vec![0u8; to_read];
        let mut filled = 0usize;
        let mut truncated = false;

        while filled < to_read {
            match self.src.read(&mut raw[filled..]) {
                Ok(0) => {
                    truncated = self.data_size != u64::MAX;
                    break;
                }
                Ok(n) => filled += n,
                Err(e) => return Err(Error::io(COMPONENT, &e)),
            }
        }

        if filled == 0 {
            if truncated {
                self.health.record(HealthEvent {
                    track: TrackId(0),
                    pts: Timestamp::new(self.current_sample as i64),
                    byte_offset: Some(pkt_pos),
                    kind: HealthIssue::TruncatedPacket,
                    detail: "unexpected EOF inside WAV data chunk",
                });
                if self.policy == ErrorPolicy::Strict {
                    return Err(Error::corrupt(
                        COMPONENT,
                        "unexpected EOF inside WAV data chunk",
                    ));
                }
            }
            self.eof = true;
            return Ok(RecvStatus::Eof);
        }

        raw.truncate(filled);
        let samples = filled.checked_div(self.block_align).unwrap_or(0) as u64;

        let track = &self.header.tracks[0];
        let mut pkt = Packet::new(Buffer::from_slice(&raw), track.id, track.time_base);
        pkt.pts = Timestamp::new(self.current_sample as i64);
        pkt.dts = pkt.pts;
        pkt.duration = samples as i64;
        pkt.pos = Some(pkt_pos);
        if truncated {
            pkt.flags |= PacketFlags::CORRUPT;
            self.health.record(HealthEvent {
                track: track.id,
                pts: pkt.pts,
                byte_offset: Some(pkt_pos),
                kind: HealthIssue::TruncatedPacket,
                detail: "truncated WAV audio packet at EOF",
            });
            if self.policy == ErrorPolicy::Strict {
                return Err(Error::corrupt(
                    COMPONENT,
                    "truncated WAV audio packet at EOF",
                ));
            }
        }

        self.current_sample = self.current_sample.saturating_add(samples);
        Ok(RecvStatus::Ready(pkt))
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.src.is_seekable() {
            return Err(Error::not_seekable(COMPONENT, "WAV source is not seekable"));
        }
        let track = &self.header.tracks[0];
        let target_samples = if target.time_base.is_valid() && track.time_base.is_valid() {
            rescale_q(target.timestamp.raw(), target.time_base, track.time_base)
        } else {
            target.timestamp.raw()
        };
        let target_samples = target_samples.max(0) as u64;
        let align = self.block_align.max(1) as u64;
        let mut byte_off = target_samples.saturating_mul(align);
        if self.data_size != u64::MAX {
            byte_off = byte_off.min(self.data_size);
        }
        byte_off = (byte_off / align) * align;

        self.src
            .seek_abs(self.data_ofs.saturating_add(byte_off))
            .map_err(|e| Error::io(COMPONENT, &e))?;
        self.current_sample = byte_off / align;
        self.eof = false;
        Ok(())
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

fn read_tag_size(src: &mut BufByteSource, rifx: bool) -> Result<u32> {
    if rifx {
        src.read_u32be()
    } else {
        src.read_u32le()
    }
    .map_err(|e| Error::io(COMPONENT, &e))
}

fn parse_fmt_chunk(src: &mut BufByteSource, mut size: u64, rifx: bool) -> Result<TrackDescriptor> {
    if size < 14 {
        return Err(Error::invalid_data(COMPONENT, "wav fmt chunk size < 14"));
    }

    let (format_tag, nb_channels, sample_rate, byte_rate, block_align) = if rifx {
        (
            src.read_u16be().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u16be().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u32be().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u32be().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u16be().map_err(|e| Error::io(COMPONENT, &e))?,
        )
    } else {
        (
            src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u32le().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u32le().map_err(|e| Error::io(COMPONENT, &e))?,
            src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?,
        )
    };

    let mut bits_per_coded_sample = if size == 14 {
        8
    } else if rifx {
        src.read_u16be().map_err(|e| Error::io(COMPONENT, &e))?
    } else {
        src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?
    };

    if sample_rate == 0 || nb_channels == 0 {
        return Err(Error::invalid_data(
            COMPONENT,
            "invalid sample_rate or channel count in fmt chunk",
        ));
    }

    let mut codec_tag = u32::from(format_tag);
    let mut channel_layout = ChannelLayout::default_for_count(u32::from(nb_channels));
    let mut extradata = Vec::new();

    if size >= 18 {
        let cb_size = u64::from(src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?);
        size -= 18;
        let mut cb_rem = cb_size.min(size);

        if format_tag == WAVE_FORMAT_EXTENSIBLE && cb_rem >= 22 {
            let valid_bps = src.read_u16le().map_err(|e| Error::io(COMPONENT, &e))?;
            if valid_bps != 0 {
                bits_per_coded_sample = valid_bps;
            }
            let mask = src.read_u32le().map_err(|e| Error::io(COMPONENT, &e))?;
            let mut guid = [0u8; 16];
            src.read_exact_bytes(&mut guid)
                .map_err(|e| Error::io(COMPONENT, &e))?;

            let parsed_layout = ChannelLayout::from_wav_channel_mask(mask);
            if parsed_layout.nb_channels == u32::from(nb_channels) {
                channel_layout = parsed_layout;
            }

            if guid[4..16] == MEDIASUBTYPE_BASE_GUID
                || guid[4..16] == AMBISONIC_BASE_GUID
                || guid[4..16] == BROKEN_BASE_GUID
            {
                codec_tag = u32::from_le_bytes([guid[0], guid[1], guid[2], guid[3]]);
            }

            cb_rem -= 22;
            size -= 22;
        }

        if cb_rem > 0 {
            extradata = vec![0u8; cb_rem as usize];
            src.read_exact_bytes(&mut extradata)
                .map_err(|e| Error::io(COMPONENT, &e))?;
            size -= cb_rem;
        }
        if size > 0 {
            src.skip(size).map_err(|e| Error::io(COMPONENT, &e))?;
        }
    }

    let (codec_id, sample_format) =
        wav_format_to_codec(codec_tag as u16, bits_per_coded_sample, rifx).ok_or_else(|| {
            Error::unsupported(
                COMPONENT,
                format!(
                    "unsupported WAV format tag 0x{codec_tag:04x} ({bits_per_coded_sample} bps)"
                ),
            )
        })?;

    let mut audio = AudioStreamParams::new(sample_rate, sample_format, channel_layout);
    audio.block_align = u32::from(block_align);
    audio.bits_per_coded_sample = u32::from(bits_per_coded_sample);
    audio.bit_rate = i64::from(byte_rate) * 8;
    audio.extradata = extradata;

    let mut track = TrackDescriptor::new(
        TrackId(0),
        codec_id,
        Rational::new(1, sample_rate as i32),
        StreamParams::Audio(audio),
    );
    track.codec_tag = codec_tag;
    Ok(track)
}

fn parse_riff_info(
    src: &mut BufByteSource,
    size: u64,
    header: &mut ContainerHeader,
    health: &mut StreamHealth,
    policy: ErrorPolicy,
) -> Result<()> {
    let end = src.tell().saturating_add(size);
    while src.tell().saturating_add(8) <= end {
        let tag = match src.read_tag() {
            Ok(t) => t,
            Err(_) => break,
        };
        let sub_size = match src.read_u32le() {
            Ok(s) => u64::from(s),
            Err(_) => break,
        };
        let aligned_size = sub_size.saturating_add(sub_size & 1);
        if src.tell().saturating_add(sub_size) > end {
            health.record(HealthEvent {
                track: TrackId(0),
                pts: Timestamp::NONE,
                byte_offset: Some(src.tell()),
                kind: HealthIssue::CorruptSyntax,
                detail: "truncated LIST/INFO subchunk",
            });
            if policy == ErrorPolicy::Strict {
                return Err(Error::corrupt(COMPONENT, "truncated LIST/INFO subchunk"));
            }
            break;
        }
        let mut buf = vec![0u8; sub_size as usize];
        src.read_exact_bytes(&mut buf)
            .map_err(|e| Error::io(COMPONENT, &e))?;
        if aligned_size > sub_size {
            src.skip(1).map_err(|e| Error::io(COMPONENT, &e))?;
        }
        while buf.last() == Some(&0) {
            buf.pop();
        }
        let val = String::from_utf8_lossy(&buf).into_owned();
        if !val.is_empty() {
            if let Some(key) = riff_info_tag_to_key(&tag) {
                header.metadata.set(key, val);
            } else {
                let raw_key = String::from_utf8_lossy(&tag).into_owned();
                header.metadata.set(raw_key, val);
            }
        }
    }
    Ok(())
}

// --- Factory & Explicit Registration ---

/// Factory for the native WAV/RF64/BW64 demuxer.
pub struct WavDemuxerFactory;

impl DemuxerFactory for WavDemuxerFactory {
    fn descriptor(&self) -> &DemuxerDescriptor {
        static DESC: DemuxerDescriptor = DemuxerDescriptor {
            name: "wav",
            long_name: "WAV / WAVE (Waveform Audio)",
            extensions: &["wav", "wave", "rf64", "bw64"],
            mime_types: &["audio/wav", "audio/x-wav", "audio/wave"],
        };
        &DESC
    }

    fn probe(&self, head: &[u8], _ext: Option<&str>) -> u8 {
        if head.len() < 12 {
            return 0;
        }
        let is_riff_family = matches!(&head[0..4], b"RIFF" | b"RIFX" | b"RF64" | b"BW64");
        if is_riff_family && &head[8..12] == b"WAVE" {
            100
        } else {
            0
        }
    }

    fn open(&self, source: Box<dyn ByteSource>, policy: ErrorPolicy) -> Result<Box<dyn Demuxer>> {
        Ok(Box::new(WavDemuxer::open(source, policy)?))
    }
}

/// Factory for the native WAV/RF64 muxer.
pub struct WavMuxerFactory;

impl MuxerFactory for WavMuxerFactory {
    fn descriptor(&self) -> &MuxerDescriptor {
        static DESC: MuxerDescriptor = MuxerDescriptor {
            name: "wav",
            long_name: "WAV / WAVE (Waveform Audio)",
            extensions: &["wav"],
            mime_types: &["audio/wav", "audio/x-wav"],
            default_video: CodecId::None,
            default_audio: CodecId::PcmS16le,
        };
        &DESC
    }

    fn create(&self, sink: Box<dyn ByteSink>) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(WavMuxer::new(sink)))
    }
}

/// Register the native WAV demuxer and muxer into `reg`.
pub fn register_wav(reg: &mut Registry) {
    reg.register_demuxer(Arc::new(WavDemuxerFactory));
    reg.register_muxer(Arc::new(WavMuxerFactory));
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{MemorySink, MemorySource};

    #[test]
    fn probe_scores_100_on_riff_rifx_rf64_bw64() {
        let f = WavDemuxerFactory;
        assert_eq!(f.probe(b"RIFF\x24\x00\x00\x00WAVEfmt ", None), 100);
        assert_eq!(f.probe(b"RIFX\x24\x00\x00\x00WAVEfmt ", None), 100);
        assert_eq!(f.probe(b"RF64\xff\xff\xff\xffWAVEds64", None), 100);
        assert_eq!(f.probe(b"BW64\xff\xff\xff\xffWAVEds64", None), 100);
        assert_eq!(f.probe(b"RIFF\x24\x00\x00\x00AVI ", None), 0);
        assert_eq!(f.probe(b"fLaC\x00\x00\x00\x22", None), 0);
    }

    #[test]
    fn wav_mux_demux_roundtrip_and_odd_byte_pad_and_seek() {
        let mut reg = Registry::new();
        register_wav(&mut reg);

        let muxer_factory = reg.find_muxer_by_ext("wav").unwrap();
        let sink = MemorySink::new();
        let mut muxer = muxer_factory.create(Box::new(sink)).unwrap();

        let mut header = ContainerHeader::new("wav");
        // 3 channels (2.1) to exercise WAVEFORMATEXTENSIBLE + 3-byte samples for odd data length
        let layout = ChannelLayout {
            order: wedeo_core::ChannelOrder::Native,
            nb_channels: 3,
            channels: vec![
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::LowFrequency,
            ],
        };
        let audio = AudioStreamParams::new(48_000, SampleFormat::U8, layout.clone());
        let track = TrackDescriptor::new(
            TrackId(0),
            CodecId::PcmU8,
            Rational::new(1, 48_000),
            StreamParams::Audio(audio),
        );
        header.tracks.push(track);

        muxer.write_header(&header).unwrap();
        // 3 bytes = 1 sample across 3 channels -> odd byte count, triggers 1-byte pad
        let payload = vec![0x11u8, 0x22, 0x33];
        let mut pkt = Packet::from_slice(&payload);
        pkt.pts = Timestamp::new(0);
        pkt.duration = 1;
        muxer.write_packet(&pkt).unwrap();

        // Write 4098 more bytes (1366 samples) so total > 4096 to exercise MAX_SIZE chunking
        let big = vec![0x7Fu8; 4098];
        let pkt2 = Packet::from_slice(&big);
        muxer.write_packet(&pkt2).unwrap();
        muxer.finalize().unwrap();
    }

    #[test]
    fn wav_mux_demux_roundtrip_full_verification() {
        let mut reg = Registry::new();
        register_wav(&mut reg);

        // Build WAV into a shared Vec<u8> via a custom sink or file
        let tmp_path = std::env::temp_dir().join("wedeo_wav_roundtrip_test.wav");
        {
            let file_sink = wedeo_core::FileSink::create(&tmp_path).unwrap();
            let mut muxer = reg
                .find_muxer_by_name("wav")
                .unwrap()
                .create(Box::new(file_sink))
                .unwrap();

            let mut header = ContainerHeader::new("wav");
            let layout = ChannelLayout::surround_5_1();
            let audio = AudioStreamParams::new(44_100, SampleFormat::S16, layout);
            header.tracks.push(TrackDescriptor::new(
                TrackId(0),
                CodecId::PcmS16le,
                Rational::new(1, 44_100),
                StreamParams::Audio(audio),
            ));
            muxer.write_header(&header).unwrap();

            // 500 samples * 6 channels * 2 bytes = 6000 bytes (> 4096)
            let samples: Vec<u8> = (0..6000u32).map(|i| (i & 0xFF) as u8).collect();
            muxer.write_packet(&Packet::from_slice(&samples)).unwrap();
            muxer.finalize().unwrap();
        }

        let file_src = wedeo_core::FileSource::open(&tmp_path).unwrap();
        let mut demuxer = reg
            .find_demuxer_by_name("wav")
            .unwrap()
            .open(Box::new(file_src), ErrorPolicy::Strict)
            .unwrap();

        assert!(demuxer.health().is_clean());
        let hdr = demuxer.header();
        assert_eq!(hdr.tracks.len(), 1);
        let t = &hdr.tracks[0];
        assert_eq!(t.codec_id, CodecId::PcmS16le);
        assert_eq!(t.duration, 500);
        let a = t.params.audio().unwrap();
        assert_eq!(a.sample_rate, 44_100);
        assert_eq!(a.channel_layout, ChannelLayout::surround_5_1());

        let mut reassembled = Vec::new();
        loop {
            match demuxer.read_packet().unwrap() {
                RecvStatus::Ready(p) => {
                    assert!(p.len() <= MAX_SIZE);
                    assert_eq!(p.len() % 12, 0); // block_align = 6 * 2 = 12
                    reassembled.extend_from_slice(p.data.data());
                }
                RecvStatus::Eof => break,
                RecvStatus::NeedMore => unreachable!(),
            }
        }
        assert_eq!(reassembled.len(), 6000);
        for (i, &b) in reassembled.iter().enumerate() {
            assert_eq!(b, (i & 0xFF) as u8);
        }

        // Seek back to sample 250 (byte 3000)
        demuxer
            .seek(SeekTarget {
                track: Some(TrackId(0)),
                timestamp: Timestamp::new(250),
                time_base: Rational::new(1, 44_100),
                backward: true,
            })
            .unwrap();
        let RecvStatus::Ready(p_after_seek) = demuxer.read_packet().unwrap() else {
            panic!("expected packet after seek");
        };
        assert_eq!(p_after_seek.pts, Timestamp::new(250));
        assert_eq!(p_after_seek.data.data()[0], (3000 & 0xFF) as u8);

        let _ = std::fs::remove_file(&tmp_path);
    }

    #[test]
    fn wav_ffmpeg_oracle_compatibility() {
        let ffmpeg = std::path::Path::new("/home/sharif/wedeo/target/ffmpeg-oracle/ffmpeg");
        if !ffmpeg.exists() {
            return;
        }
        let tmp_pcm = std::env::temp_dir().join("wedeo_oracle_in.pcm");
        let tmp_wav = std::env::temp_dir().join("wedeo_oracle_test.wav");
        let tmp_wedeo_wav = std::env::temp_dir().join("wedeo_oracle_muxed.wav");
        let tmp_decoded_pcm = std::env::temp_dir().join("wedeo_oracle_decoded.pcm");

        // 0.1s * 48000 Hz * 2 channels * 2 bytes = 19200 bytes
        let pcm_in: Vec<u8> = (0..19200u32).map(|i| ((i * 7) & 0xFF) as u8).collect();
        std::fs::write(&tmp_pcm, &pcm_in).unwrap();

        // 1. FFmpeg muxes raw PCM -> WAV with metadata, then WavDemuxer reads it bit-exact
        let status = std::process::Command::new(ffmpeg)
            .args([
                "-y",
                "-f",
                "s16le",
                "-ar",
                "48000",
                "-ac",
                "2",
                "-i",
                tmp_pcm.to_str().unwrap(),
                "-c:a",
                "pcm_s16le",
                "-metadata",
                "title=OracleTone",
                "-f",
                "wav",
                tmp_wav.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(status.success(), "ffmpeg oracle failed to generate wav");

        let mut reg = Registry::new();
        register_wav(&mut reg);

        let src = wedeo_core::FileSource::open(&tmp_wav).unwrap();
        let mut demuxer = reg
            .find_demuxer_by_name("wav")
            .unwrap()
            .open(Box::new(src), ErrorPolicy::Strict)
            .unwrap();
        assert_eq!(demuxer.header().metadata.get("title"), Some("OracleTone"));
        assert_eq!(demuxer.header().tracks[0].codec_id, CodecId::PcmS16le);

        let mut demuxed_pcm = Vec::new();
        while let RecvStatus::Ready(p) = demuxer.read_packet().unwrap() {
            demuxed_pcm.extend_from_slice(p.data.data());
        }
        assert_eq!(demuxed_pcm, pcm_in);

        // 2. WavMuxer muxes PCM -> WAV, then FFmpeg n9.1 decodes it back bit-exact
        {
            let sink = wedeo_core::FileSink::create(&tmp_wedeo_wav).unwrap();
            let mut muxer = reg
                .find_muxer_by_name("wav")
                .unwrap()
                .create(Box::new(sink))
                .unwrap();
            let mut header = ContainerHeader::new("wav");
            let audio = AudioStreamParams::new(48_000, SampleFormat::S16, ChannelLayout::stereo());
            header.tracks.push(TrackDescriptor::new(
                TrackId(0),
                CodecId::PcmS16le,
                Rational::new(1, 48_000),
                StreamParams::Audio(audio),
            ));
            muxer.write_header(&header).unwrap();
            muxer.write_packet(&Packet::from_slice(&pcm_in)).unwrap();
            muxer.finalize().unwrap();
        }

        let status2 = std::process::Command::new(ffmpeg)
            .args([
                "-y",
                "-i",
                tmp_wedeo_wav.to_str().unwrap(),
                "-f",
                "s16le",
                tmp_decoded_pcm.to_str().unwrap(),
            ])
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .unwrap();
        assert!(
            status2.success(),
            "ffmpeg oracle failed to decode wedeo wav"
        );
        let decoded_pcm = std::fs::read(&tmp_decoded_pcm).unwrap();
        assert_eq!(decoded_pcm, pcm_in);

        let _ = std::fs::remove_file(&tmp_pcm);
        let _ = std::fs::remove_file(&tmp_wav);
        let _ = std::fs::remove_file(&tmp_wedeo_wav);
        let _ = std::fs::remove_file(&tmp_decoded_pcm);
    }

    #[test]
    fn rf64_demux_parses_ds64_and_reads_audio() {
        // Construct a minimal RF64 + ds64 + fmt + data file in memory
        let mut buf = Vec::new();
        buf.extend_from_slice(b"RF64");
        buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf.extend_from_slice(b"WAVE");
        // ds64
        buf.extend_from_slice(b"ds64");
        buf.extend_from_slice(&28u32.to_le_bytes());
        buf.extend_from_slice(&100u64.to_le_bytes()); // riffSize
        buf.extend_from_slice(&8u64.to_le_bytes()); // dataSize
        buf.extend_from_slice(&4u64.to_le_bytes()); // sampleCount
        buf.extend_from_slice(&0u32.to_le_bytes()); // tableLength
        // fmt
        buf.extend_from_slice(b"fmt ");
        buf.extend_from_slice(&16u32.to_le_bytes());
        buf.extend_from_slice(&1u16.to_le_bytes()); // PCM
        buf.extend_from_slice(&1u16.to_le_bytes()); // 1 ch
        buf.extend_from_slice(&48_000u32.to_le_bytes());
        buf.extend_from_slice(&96_000u32.to_le_bytes());
        buf.extend_from_slice(&2u16.to_le_bytes());
        buf.extend_from_slice(&16u16.to_le_bytes());
        // data
        buf.extend_from_slice(b"data");
        buf.extend_from_slice(&0xFFFF_FFFFu32.to_le_bytes());
        buf.extend_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8]);

        let mut reg = Registry::new();
        register_wav(&mut reg);
        let factory = reg.probe_demuxer(&buf[..16], Some("wav")).unwrap();
        assert_eq!(factory.descriptor().name, "wav");
        assert_eq!(factory.probe(&buf[..16], None), 100);

        let mut demuxer = factory
            .open(Box::new(MemorySource::new(buf)), ErrorPolicy::Strict)
            .unwrap();
        assert_eq!(demuxer.header().tracks[0].duration, 4);
        let RecvStatus::Ready(pkt) = demuxer.read_packet().unwrap() else {
            panic!("expected packet");
        };
        assert_eq!(pkt.data.data(), &[1, 2, 3, 4, 5, 6, 7, 8]);
        assert!(matches!(demuxer.read_packet().unwrap(), RecvStatus::Eof));
    }
}
