use std::io::{self, Read, SeekFrom};

use wedeo_core::{
    BufferPool, ByteSource, CodecContext, ContainerHeader, Decoder, Demuxer, Error, ErrorPolicy,
    MediaType, MediaUnit, Packet, RecvStatus, Registry, Result, SeekTarget, SendStatus,
    StreamHealth, TrackDescriptor, TrackId,
};

const PROBE_BYTES: usize = 8192;

/// Non-seekable `ByteSource` wrapper that replays already-probed leading bytes before
/// delegating to the underlying stream.
struct PrefixedSource {
    prefix: Vec<u8>,
    pos: usize,
    inner: Box<dyn ByteSource>,
}

impl Read for PrefixedSource {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.pos < self.prefix.len() {
            let avail = self.prefix.len() - self.pos;
            let n = avail.min(buf.len());
            buf[..n].copy_from_slice(&self.prefix[self.pos..self.pos + n]);
            self.pos += n;
            return Ok(n);
        }
        self.inner.read(buf)
    }
}

impl ByteSource for PrefixedSource {
    fn is_seekable(&self) -> bool {
        false
    }

    fn size_hint(&self) -> Option<u64> {
        self.inner.size_hint()
    }
}

/// Read up to `PROBE_BYTES` from `source`, probe `reg`, rewind (or wrap in `PrefixedSource`),
/// and open the winning demuxer.
pub(crate) fn open_demuxer(
    reg: &Registry,
    mut source: Box<dyn ByteSource>,
    ext: Option<&str>,
    policy: ErrorPolicy,
) -> Result<Box<dyn Demuxer>> {
    let mut head = Vec::with_capacity(PROBE_BYTES);
    let mut buf = [0u8; 4096];
    while head.len() < PROBE_BYTES {
        let want = (PROBE_BYTES - head.len()).min(buf.len());
        match source.read(&mut buf[..want]) {
            Ok(0) => break,
            Ok(n) => head.extend_from_slice(&buf[..n]),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(e) => return Err(Error::io("wedeo::open_demuxer", &e)),
        }
    }

    let factory = reg.probe_demuxer(&head, ext).ok_or_else(|| {
        Error::not_found(
            "wedeo::open_demuxer",
            "no registered demuxer matched the input stream",
        )
    })?;

    let rewound: Box<dyn ByteSource> = if source.is_seekable() {
        source
            .seek(SeekFrom::Start(0))
            .map_err(|e| Error::io("wedeo::open_demuxer", &e))?;
        source
    } else {
        Box::new(PrefixedSource {
            prefix: head,
            pos: 0,
            inner: source,
        })
    };

    factory.open(rewound, policy)
}

/// Shared single-track demux + decode state machine backing [`crate::audio::AudioReader`]
/// and [`crate::video::VideoReader`].
pub(crate) struct TrackReader {
    demuxer: Box<dyn Demuxer>,
    decoder: Box<dyn Decoder>,
    track_id: TrackId,
    pending_pkt: Option<Packet>,
    demux_eof: bool,
    flush_sent: bool,
    decode_eof: bool,
}

impl TrackReader {
    pub(crate) fn open(
        reg: &Registry,
        source: Box<dyn ByteSource>,
        ext: Option<&str>,
        media_type: MediaType,
        policy: ErrorPolicy,
    ) -> Result<(Self, TrackDescriptor)> {
        let demuxer = open_demuxer(reg, source, ext, policy)?;
        let track = demuxer
            .header()
            .tracks
            .iter()
            .find(|t| t.params.media_type() == media_type)
            .cloned()
            .ok_or_else(|| {
                Error::not_found(
                    "wedeo::TrackReader",
                    format!("container has no {media_type:?} track"),
                )
            })?;

        let dec_factory = reg.find_decoder(track.codec_id).ok_or_else(|| {
            Error::not_found(
                "wedeo::TrackReader",
                format!("no registered decoder for {:?}", track.codec_id),
            )
        })?;

        let mut ctx = CodecContext::new(BufferPool::new());
        ctx.error_policy = policy;
        let decoder = dec_factory.open(&track, ctx)?;
        let track_id = track.id;

        Ok((
            Self {
                demuxer,
                decoder,
                track_id,
                pending_pkt: None,
                demux_eof: false,
                flush_sent: false,
                decode_eof: false,
            },
            track,
        ))
    }

    pub(crate) fn header(&self) -> &ContainerHeader {
        self.demuxer.header()
    }

    pub(crate) fn health(&self) -> &StreamHealth {
        self.decoder.health()
    }

    pub(crate) fn demuxer_health(&self) -> &StreamHealth {
        self.demuxer.health()
    }

    pub(crate) fn seek(&mut self, target: SeekTarget) -> Result<()> {
        self.demuxer.seek(target)?;
        self.decoder.flush();
        self.pending_pkt = None;
        self.demux_eof = false;
        self.flush_sent = false;
        self.decode_eof = false;
        Ok(())
    }

    pub(crate) fn next_unit(&mut self) -> Result<Option<MediaUnit>> {
        if self.decode_eof {
            return Ok(None);
        }

        loop {
            match self.decoder.receive()? {
                RecvStatus::Ready(unit) => return Ok(Some(unit)),
                RecvStatus::Eof => {
                    self.decode_eof = true;
                    return Ok(None);
                }
                RecvStatus::NeedMore => {}
            }

            if let Some(pkt) = self.pending_pkt.take() {
                match self.decoder.send_packet(Some(&pkt))? {
                    SendStatus::Accepted => continue,
                    SendStatus::Full => {
                        self.pending_pkt = Some(pkt);
                        continue;
                    }
                }
            }

            if self.demux_eof {
                if self.flush_sent {
                    self.decode_eof = true;
                    return Ok(None);
                }
                match self.decoder.send_packet(None)? {
                    SendStatus::Accepted => self.flush_sent = true,
                    SendStatus::Full => {}
                }
                continue;
            }

            match self.demuxer.read_packet()? {
                RecvStatus::Ready(pkt) => {
                    if pkt.track_id != self.track_id {
                        continue;
                    }
                    if self.decoder.send_packet(Some(&pkt))? == SendStatus::Full {
                        self.pending_pkt = Some(pkt);
                    }
                }
                RecvStatus::NeedMore => continue,
                RecvStatus::Eof => {
                    self.demux_eof = true;
                }
            }
        }
    }
}
