//! Generic raw elementary stream framing (`ElementaryFramer` + `RawStreamDemuxer`).
//!
//! Provides a single, clean `wedeo_core::Demuxer` implementation (~150 LOC) backed by
//! a rolling byte buffer with amortized O(1) compaction, EOF drain, exact byte-offset
//! tracking, VFR-native timing (`Timestamp::NONE` or framer-provided duration/time_base),
//! and `StreamHealth` accounting.

use std::io::SeekFrom;
use std::ops::Range;

use wedeo_core::{
    Buffer, ByteSource, CodecId, ContainerHeader, Demuxer, Error, ErrorPolicy, HealthEvent,
    HealthIssue, Packet, PacketFlags, Rational, RecvStatus, Result, SeekTarget, StreamHealth,
    StreamParams, Timestamp, TrackDescriptor, TrackId,
};

/// Default read chunk size pulled from the underlying `ByteSource`.
const READ_CHUNK_SIZE: usize = 32 * 1024;

/// Compact the rolling buffer once `head` reaches this threshold or half the buffer length.
const COMPACT_THRESHOLD: usize = 64 * 1024;

/// Hard cap on buffered unframed bytes (32 MiB) to prevent unbounded memory growth on corrupt streams.
const MAX_BUFFER_BYTES: usize = 32 * 1024 * 1024;

/// Optional per-unit metadata emitted by an [`ElementaryFramer`].
#[derive(Debug, Clone, Default)]
pub struct FrameParams {
    /// Unit duration in `time_base` ticks (e.g. `1024` samples for an AAC ADTS frame).
    pub duration: Option<i64>,
    /// Explicit time base for `duration` (e.g. `1/sample_rate` for AAC ADTS).
    pub time_base: Option<Rational>,
    /// Updated track parameters discovered from bitstream headers (SPS, Sequence Header, ADTS, etc.).
    pub stream_params: Option<StreamParams>,
}

/// Action returned by [`ElementaryFramer::next_unit`].
#[derive(Debug, Clone)]
pub enum FrameStep {
    /// More bytes are needed before a complete unit can be delimited.
    NeedMore,
    /// Emit a complete elementary unit from `buf[range]` and advance the buffer by `consume` bytes.
    Emit {
        /// Byte range within the slice passed to `next_unit` forming the packet payload.
        range: Range<usize>,
        /// Total number of bytes to consume from the front of the buffer (must be `>= range.end`).
        consume: usize,
        /// Whether this unit is a random-access / keyframe unit.
        is_key: bool,
        /// Optional timing or updated stream parameters discovered for this unit.
        params: Option<FrameParams>,
    },
    /// Discard `usize` leading garbage / unsynchronized bytes and continue scanning.
    Skip(usize),
}

/// Codec-specific raw elementary stream framer (H.264 Annex B, HEVC Annex B, AV1 OBU, AAC ADTS).
pub trait ElementaryFramer: Send {
    /// Short canonical format name for [`ContainerHeader::format_name`].
    fn format_name(&self) -> &'static str;

    /// Codec identifier for the elementary track.
    fn codec_id(&self) -> CodecId;

    /// Initial track parameters before the first unit header is parsed.
    fn initial_params(&self) -> StreamParams;

    /// Default packet time base (`0/1` for untimed VFR elementary video streams).
    fn default_time_base(&self) -> Rational {
        Rational::new(0, 1)
    }

    /// Confidence score in `0..=100` for format probing on the leading bytes of a stream.
    fn probe(head: &[u8]) -> u8
    where
        Self: Sized;

    /// Inspect `buf` (the unread window of the rolling buffer) and return the next framing step.
    ///
    /// When `eof` is `true`, no further bytes will arrive; the framer should flush any remaining
    /// complete unit or return [`FrameStep::NeedMore`] to signal end-of-stream.
    fn next_unit(&mut self, buf: &[u8], eof: bool) -> Result<FrameStep>;

    /// Reset internal parser state (used when seeking back to the start of a seekable source).
    fn reset(&mut self);
}

/// Generic [`Demuxer`] wrapping any [`ElementaryFramer`] over a rolling byte buffer.
pub struct RawStreamDemuxer<F: ElementaryFramer> {
    source: Box<dyn ByteSource>,
    framer: F,
    header: ContainerHeader,
    health: StreamHealth,
    policy: ErrorPolicy,
    buf: Vec<u8>,
    head: usize,
    stream_offset: u64,
    eof: bool,
    pending: Option<Packet>,
}

impl<F: ElementaryFramer> RawStreamDemuxer<F> {
    /// Open a raw elementary stream demuxer and inspect the first unit (if present) so
    /// [`Demuxer::header`] reflects discovered codec parameters immediately.
    pub fn open(source: Box<dyn ByteSource>, framer: F, policy: ErrorPolicy) -> Result<Self> {
        let mut header = ContainerHeader::new(framer.format_name());
        let track = TrackDescriptor::new(
            TrackId(0),
            framer.codec_id(),
            framer.default_time_base(),
            framer.initial_params(),
        );
        header.tracks.push(track);

        let mut demuxer = Self {
            source,
            framer,
            header,
            health: StreamHealth::new(),
            policy,
            buf: Vec::with_capacity(READ_CHUNK_SIZE * 2),
            head: 0,
            stream_offset: 0,
            eof: false,
            pending: None,
        };

        // Prime the first packet so `header()` immediately exposes parsed SPS/ADTS/SeqHdr params.
        if let RecvStatus::Ready(pkt) = demuxer.read_next_internal()? {
            demuxer.pending = Some(pkt);
        }

        Ok(demuxer)
    }

    /// Borrow the underlying framer.
    pub fn framer(&self) -> &F {
        &self.framer
    }

    /// Compact the rolling buffer when enough leading bytes have been consumed.
    fn compact_if_needed(&mut self) {
        if self.head == 0 {
            return;
        }
        let remaining = self.buf.len() - self.head;
        if remaining == 0 {
            self.buf.clear();
            self.head = 0;
        } else if self.head >= COMPACT_THRESHOLD || self.head * 2 >= self.buf.len() {
            self.buf.copy_within(self.head.., 0);
            self.buf.truncate(remaining);
            self.head = 0;
        }
    }

    /// Pull more bytes from `source` into the rolling buffer. Returns `Ok(true)` if bytes were read.
    fn refill(&mut self) -> Result<bool> {
        if self.eof {
            return Ok(false);
        }
        self.compact_if_needed();
        if self.buf.len() - self.head >= MAX_BUFFER_BYTES {
            return Err(Error::invalid_data(
                self.framer.format_name(),
                "elementary unit exceeds maximum buffer limit",
            ));
        }
        let old_len = self.buf.len();
        self.buf.resize(old_len + READ_CHUNK_SIZE, 0);
        let n = self
            .source
            .read(&mut self.buf[old_len..])
            .map_err(|e| Error::io(self.framer.format_name(), &e))?;
        self.buf.truncate(old_len + n);
        if n == 0 {
            self.eof = true;
            return Ok(false);
        }
        Ok(true)
    }

    fn read_next_internal(&mut self) -> Result<RecvStatus<Packet>> {
        loop {
            let unread = &self.buf[self.head..];
            if unread.is_empty() && self.eof {
                return Ok(RecvStatus::Eof);
            }

            if unread.is_empty() {
                if !self.refill()? {
                    return Ok(RecvStatus::Eof);
                }
                continue;
            }

            match self.framer.next_unit(unread, self.eof)? {
                FrameStep::NeedMore => {
                    if self.eof {
                        // Trailing unframable bytes at EOF.
                        let leftover = self.buf.len() - self.head;
                        if leftover > 0 {
                            self.record_dropped(leftover, "trailing unframed bytes at EOF")?;
                            self.head = self.buf.len();
                            self.compact_if_needed();
                        }
                        return Ok(RecvStatus::Eof);
                    }
                    self.refill()?;
                }
                FrameStep::Skip(n) => {
                    let avail = self.buf.len() - self.head;
                    let skip = n.min(avail);
                    if skip == 0 {
                        if self.eof {
                            return Ok(RecvStatus::Eof);
                        }
                        self.refill()?;
                        continue;
                    }
                    self.record_dropped(skip, "skipped unsynchronized bytes")?;
                    self.head += skip;
                    self.stream_offset = self.stream_offset.saturating_add(skip as u64);
                    self.compact_if_needed();
                }
                FrameStep::Emit {
                    range,
                    consume,
                    is_key,
                    params,
                } => {
                    let avail = self.buf.len() - self.head;
                    if range.start > range.end || range.end > avail || consume > avail {
                        return Err(Error::internal(
                            self.framer.format_name(),
                            "framer returned out-of-bounds range or consume count",
                        ));
                    }
                    let pkt_pos = self.stream_offset.saturating_add(range.start as u64);
                    let payload = Buffer::from_slice(
                        &self.buf[self.head + range.start..self.head + range.end],
                    );

                    let mut time_base = self.header.tracks[0].time_base;
                    let mut duration = 0i64;

                    if let Some(fp) = params {
                        if let Some(tb) = fp.time_base
                            && tb.is_valid()
                        {
                            time_base = tb;
                            self.header.tracks[0].time_base = tb;
                        }
                        if let Some(dur) = fp.duration {
                            duration = dur;
                        }
                        if let Some(sp) = fp.stream_params {
                            self.header.tracks[0].params = sp;
                        }
                    }

                    let mut pkt = Packet::new(payload, TrackId(0), time_base);
                    pkt.pts = Timestamp::NONE;
                    pkt.dts = Timestamp::NONE;
                    pkt.duration = duration;
                    pkt.pos = Some(pkt_pos);
                    if is_key {
                        pkt.flags |= PacketFlags::KEY;
                    }

                    self.head += consume;
                    self.stream_offset = self.stream_offset.saturating_add(consume as u64);
                    self.compact_if_needed();
                    return Ok(RecvStatus::Ready(pkt));
                }
            }
        }
    }

    fn record_dropped(&mut self, count: usize, detail: &'static str) -> Result<()> {
        self.health.dropped_bytes = self.health.dropped_bytes.saturating_add(count as u64);
        self.health.record(HealthEvent {
            track: TrackId(0),
            pts: Timestamp::NONE,
            byte_offset: Some(self.stream_offset),
            kind: HealthIssue::CorruptSyntax,
            detail,
        });
        if self.policy == ErrorPolicy::Strict {
            return Err(Error::corrupt(self.framer.format_name(), detail));
        }
        Ok(())
    }
}

impl<F: ElementaryFramer> Demuxer for RawStreamDemuxer<F> {
    fn header(&self) -> &ContainerHeader {
        &self.header
    }

    fn read_packet(&mut self) -> Result<RecvStatus<Packet>> {
        if let Some(pkt) = self.pending.take() {
            return Ok(RecvStatus::Ready(pkt));
        }
        self.read_next_internal()
    }

    fn seek(&mut self, target: SeekTarget) -> Result<()> {
        if !self.source.is_seekable() {
            return Err(Error::not_seekable(
                self.framer.format_name(),
                "raw elementary stream source is not seekable",
            ));
        }
        if !target.timestamp.is_none() && target.timestamp.raw() > 0 {
            return Err(Error::unsupported(
                self.framer.format_name(),
                "non-zero timestamp seek is unsupported on unindexed raw elementary streams",
            ));
        }
        self.source
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::io(self.framer.format_name(), &e))?;
        self.buf.clear();
        self.head = 0;
        self.stream_offset = 0;
        self.eof = false;
        self.pending = None;
        self.framer.reset();
        Ok(())
    }

    fn health(&self) -> &StreamHealth {
        &self.health
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::{MemorySource, PixelLayout, VideoStreamParams};

    /// Simple delimiter-based framer (`0xAA` starts a 4-byte unit `[0xAA, len, payload...]`) for testing `RawStreamDemuxer`.
    struct TestFramer;

    impl ElementaryFramer for TestFramer {
        fn format_name(&self) -> &'static str {
            "test_raw"
        }

        fn codec_id(&self) -> CodecId {
            CodecId::Rawvideo
        }

        fn initial_params(&self) -> StreamParams {
            StreamParams::Video(VideoStreamParams::new(0, 0, PixelLayout::YUV420P))
        }

        fn probe(head: &[u8]) -> u8 {
            if head.first() == Some(&0xAA) { 90 } else { 0 }
        }

        fn next_unit(&mut self, buf: &[u8], _eof: bool) -> Result<FrameStep> {
            let Some(pos) = buf.iter().position(|&b| b == 0xAA) else {
                return Ok(FrameStep::Skip(buf.len()));
            };
            if pos > 0 {
                return Ok(FrameStep::Skip(pos));
            }
            if buf.len() < 2 {
                return Ok(FrameStep::NeedMore);
            }
            let total = 2 + buf[1] as usize;
            if buf.len() < total {
                return Ok(FrameStep::NeedMore);
            }
            let is_key = buf[1] > 0 && buf[2] == 1;
            Ok(FrameStep::Emit {
                range: 0..total,
                consume: total,
                is_key,
                params: Some(FrameParams {
                    duration: Some(10),
                    time_base: Some(Rational::new(1, 1000)),
                    stream_params: None,
                }),
            })
        }

        fn reset(&mut self) {}
    }

    #[test]
    fn raw_demuxer_skips_junk_tracks_offsets_and_rewinds() {
        let mut data = vec![0x00, 0x11, 0x22]; // 3 junk bytes
        data.extend_from_slice(&[0xAA, 2, 1, 0xFF]); // key unit at offset 3
        data.extend_from_slice(&[0xAA, 1, 0]); // non-key unit at offset 7
        data.extend_from_slice(&[0xAA, 5, 0]); // truncated trailing unit at EOF

        let mut demux = RawStreamDemuxer::open(
            Box::new(MemorySource::new(data)),
            TestFramer,
            ErrorPolicy::Resilient,
        )
        .unwrap();

        let p0 = match demux.read_packet().unwrap() {
            RecvStatus::Ready(p) => p,
            other => panic!("expected packet 0, got {other:?}"),
        };
        assert_eq!(p0.data.as_slice().unwrap(), &[0xAA, 2, 1, 0xFF]);
        assert_eq!(p0.pos, Some(3));
        assert!(p0.flags.contains(PacketFlags::KEY));
        assert_eq!(p0.pts, Timestamp::NONE);
        assert_eq!(p0.duration, 10);
        assert_eq!(p0.time_base, Rational::new(1, 1000));

        let p1 = match demux.read_packet().unwrap() {
            RecvStatus::Ready(p) => p,
            other => panic!("expected packet 1, got {other:?}"),
        };
        assert_eq!(p1.data.as_slice().unwrap(), &[0xAA, 1, 0]);
        assert_eq!(p1.pos, Some(7));
        assert!(!p1.flags.contains(PacketFlags::KEY));

        assert!(matches!(demux.read_packet().unwrap(), RecvStatus::Eof));
        assert_eq!(demux.health().dropped_bytes, 6); // 3 leading junk + 3 trailing truncated

        // Rewind to start
        demux
            .seek(SeekTarget {
                track: None,
                timestamp: Timestamp::new(0),
                time_base: Rational::new(1, 1000),
                backward: true,
            })
            .unwrap();
        let p0_again = match demux.read_packet().unwrap() {
            RecvStatus::Ready(p) => p,
            other => panic!("expected packet 0 after rewind, got {other:?}"),
        };
        assert_eq!(p0_again.pos, Some(3));
    }

    #[test]
    fn raw_demuxer_strict_policy_fails_on_junk() {
        let data = vec![0xFF, 0xAA, 1, 1];
        let res = RawStreamDemuxer::open(
            Box::new(MemorySource::new(data)),
            TestFramer,
            ErrorPolicy::Strict,
        );
        assert!(res.is_err());
    }

    /// Tiny chunked `ByteSource` returning at most `chunk_len` bytes per `read` call
    /// to exercise rolling-buffer refill, compaction, and chunk-boundary straddling.
    struct ChunkedSource {
        data: Vec<u8>,
        pos: usize,
        chunk_len: usize,
    }

    impl std::io::Read for ChunkedSource {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let rem = self.data.len().saturating_sub(self.pos);
            let n = rem.min(self.chunk_len).min(buf.len());
            buf[..n].copy_from_slice(&self.data[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    impl ByteSource for ChunkedSource {}

    #[test]
    fn raw_demuxer_streams_across_tiny_chunks_and_compacts() {
        let mut data = Vec::new();
        for i in 0..200u8 {
            data.extend_from_slice(&[0xAA, 3, i % 2, i, i.wrapping_mul(3)]);
        }
        let src = ChunkedSource {
            data,
            pos: 0,
            chunk_len: 3,
        };
        let mut demux =
            RawStreamDemuxer::open(Box::new(src), TestFramer, ErrorPolicy::Strict).unwrap();
        let mut count = 0u64;
        while let RecvStatus::Ready(p) = demux.read_packet().unwrap() {
            assert_eq!(p.pos, Some(count * 5));
            assert_eq!(p.data.len(), 5);
            count += 1;
        }
        assert_eq!(count, 200);
        assert!(demux.health().is_clean());
    }
}
