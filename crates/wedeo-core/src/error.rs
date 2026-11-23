use std::borrow::Cow;
use std::fmt;

use crate::media::TrackId;
use crate::time::Timestamp;

/// Maximum number of concrete health events retained in `StreamHealth`.
pub const MAX_HEALTH_EVENTS: usize = 64;

/// High-level error classification for programmatic matching.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
    /// Malformed container, bitstream, or packet payload.
    InvalidData,
    /// Invalid parameter or configuration supplied by caller.
    InvalidArgument,
    /// Requested codec, format, pixel layout, or operation is unsupported.
    Unsupported,
    /// Requested decoder, encoder, demuxer, muxer, or track was not found.
    NotFound,
    /// Attempted to seek a non-seekable source or sink.
    NotSeekable,
    /// Underlying I/O error.
    Io(std::io::ErrorKind),
    /// Operation was cancelled / interrupted.
    Interrupted,
    /// Corrupt stream encountered under `ErrorPolicy::Strict`.
    CorruptStream,
    /// Unexpected internal invariant violation.
    Internal,
}

/// Contextual error type carrying kind, originating component, and human message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Error {
    pub kind: ErrorKind,
    pub component: &'static str,
    pub message: Cow<'static, str>,
}

impl Error {
    pub fn new(
        kind: ErrorKind,
        component: &'static str,
        message: impl Into<Cow<'static, str>>,
    ) -> Self {
        Self {
            kind,
            component,
            message: message.into(),
        }
    }

    pub fn invalid_data(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::InvalidData, component, msg)
    }

    pub fn invalid_argument(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::InvalidArgument, component, msg)
    }

    pub fn unsupported(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Unsupported, component, msg)
    }

    pub fn not_found(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::NotFound, component, msg)
    }

    pub fn not_seekable(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::NotSeekable, component, msg)
    }

    pub fn corrupt(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::CorruptStream, component, msg)
    }

    pub fn internal(component: &'static str, msg: impl Into<Cow<'static, str>>) -> Self {
        Self::new(ErrorKind::Internal, component, msg)
    }

    pub fn io(component: &'static str, err: &std::io::Error) -> Self {
        let kind = match err.kind() {
            std::io::ErrorKind::Interrupted => ErrorKind::Interrupted,
            std::io::ErrorKind::NotSeekable => ErrorKind::NotSeekable,
            other => ErrorKind::Io(other),
        };
        Self::new(kind, component, err.to_string())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}] {}", self.component, self.message)
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::io("io", &err)
    }
}

/// Convenience `Result` alias for `wedeo-core`.
pub type Result<T> = std::result::Result<T, Error>;

/// How demuxers and decoders react to stream corruption.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorPolicy {
    /// FFmpeg-compatible recovery: mark affected frames corrupt, record in `StreamHealth`, keep decoding.
    #[default]
    Resilient,
    /// Fail immediately on the first detected corruption or timestamp discontinuity.
    Strict,
}

/// Specific category of stream health issue.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HealthIssue {
    CorruptSyntax,
    MissingRef,
    ConcealedBlock,
    NonMonotonicDts,
    TimestampGap,
    TruncatedPacket,
    AudioClip,
}

/// Concrete stream health diagnostic event (zero heap allocation on static detail).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HealthEvent {
    pub track: TrackId,
    pub pts: Timestamp,
    pub byte_offset: Option<u64>,
    pub kind: HealthIssue,
    pub detail: &'static str,
}

/// Lightweight inline corruption & stream-integrity ledger.
/// Zero heap allocations on healthy streams; event log is capped at `MAX_HEALTH_EVENTS`.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StreamHealth {
    pub corrupt_packets: u64,
    pub concealed_frames: u64,
    pub concealed_blocks: u64,
    pub timestamp_glitches: u64,
    pub dropped_bytes: u64,
    pub clipped_audio_samples: u64,
    pub events: Vec<HealthEvent>,
}

impl StreamHealth {
    pub const fn new() -> Self {
        Self {
            corrupt_packets: 0,
            concealed_frames: 0,
            concealed_blocks: 0,
            timestamp_glitches: 0,
            dropped_bytes: 0,
            clipped_audio_samples: 0,
            events: Vec::new(),
        }
    }

    /// Returns `true` if no health issues have been recorded.
    #[inline]
    pub fn is_clean(&self) -> bool {
        self.corrupt_packets == 0
            && self.concealed_frames == 0
            && self.concealed_blocks == 0
            && self.timestamp_glitches == 0
            && self.dropped_bytes == 0
            && self.clipped_audio_samples == 0
            && self.events.is_empty()
    }

    /// Record a health event, updating the corresponding counter and retaining
    /// up to `MAX_HEALTH_EVENTS` concrete entries.
    pub fn record(&mut self, event: HealthEvent) {
        match event.kind {
            HealthIssue::CorruptSyntax | HealthIssue::TruncatedPacket => {
                self.corrupt_packets = self.corrupt_packets.saturating_add(1);
            }
            HealthIssue::MissingRef => {
                self.concealed_frames = self.concealed_frames.saturating_add(1);
            }
            HealthIssue::ConcealedBlock => {
                self.concealed_blocks = self.concealed_blocks.saturating_add(1);
            }
            HealthIssue::NonMonotonicDts | HealthIssue::TimestampGap => {
                self.timestamp_glitches = self.timestamp_glitches.saturating_add(1);
            }
            HealthIssue::AudioClip => {
                self.clipped_audio_samples = self.clipped_audio_samples.saturating_add(1);
            }
        }
        if self.events.len() < MAX_HEALTH_EVENTS {
            self.events.push(event);
        }
    }

    /// Merge another `StreamHealth` ledger into `self`, respecting the event cap.
    pub fn merge(&mut self, other: &Self) {
        self.corrupt_packets = self.corrupt_packets.saturating_add(other.corrupt_packets);
        self.concealed_frames = self.concealed_frames.saturating_add(other.concealed_frames);
        self.concealed_blocks = self.concealed_blocks.saturating_add(other.concealed_blocks);
        self.timestamp_glitches = self
            .timestamp_glitches
            .saturating_add(other.timestamp_glitches);
        self.dropped_bytes = self.dropped_bytes.saturating_add(other.dropped_bytes);
        self.clipped_audio_samples = self
            .clipped_audio_samples
            .saturating_add(other.clipped_audio_samples);
        let remaining = MAX_HEALTH_EVENTS.saturating_sub(self.events.len());
        if remaining > 0 {
            self.events
                .extend(other.events.iter().take(remaining).cloned());
        }
    }
}

/// Non-error control flow returned by `Decoder::send_packet` / `Encoder::send_unit`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendStatus {
    /// Input accepted.
    Accepted,
    /// Internal queue is full; drain outputs via `receive` and retry the same input.
    Full,
}

/// Non-error control flow returned by `Demuxer::read_packet` / `Decoder::receive` / `Encoder::receive_packet`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecvStatus<T> {
    /// An output item is ready.
    Ready(T),
    /// More input is needed before another output can be produced.
    NeedMore,
    /// End of stream reached cleanly.
    Eof,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stream_health_event_cap_and_counters() {
        let mut h = StreamHealth::new();
        assert!(h.is_clean());
        assert_eq!(h.events.capacity(), 0); // zero alloc on clean stream

        for i in 0..100u64 {
            h.record(HealthEvent {
                track: TrackId(0),
                pts: Timestamp::new(i as i64 * 1000),
                byte_offset: Some(i * 512),
                kind: HealthIssue::CorruptSyntax,
                detail: "bad slice header",
            });
        }

        assert!(!h.is_clean());
        assert_eq!(h.corrupt_packets, 100);
        assert_eq!(h.events.len(), MAX_HEALTH_EVENTS);
        assert_eq!(h.events[0].pts, Timestamp::new(0));
        assert_eq!(
            h.events[MAX_HEALTH_EVENTS - 1].pts,
            Timestamp::new((MAX_HEALTH_EVENTS as i64 - 1) * 1000)
        );
    }

    #[test]
    fn stream_health_merge_respects_cap() {
        let mut a = StreamHealth::new();
        let mut b = StreamHealth::new();
        for i in 0..40 {
            a.record(HealthEvent {
                track: TrackId(0),
                pts: Timestamp::new(i),
                byte_offset: None,
                kind: HealthIssue::ConcealedBlock,
                detail: "mb concealed",
            });
            b.record(HealthEvent {
                track: TrackId(1),
                pts: Timestamp::new(i),
                byte_offset: None,
                kind: HealthIssue::AudioClip,
                detail: "clip",
            });
        }
        a.merge(&b);
        assert_eq!(a.concealed_blocks, 40);
        assert_eq!(a.clipped_audio_samples, 40);
        assert_eq!(a.events.len(), MAX_HEALTH_EVENTS);
    }

    #[test]
    fn error_formatting_and_io_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Interrupted, "cancelled");
        let err: Error = io_err.into();
        assert_eq!(err.kind, ErrorKind::Interrupted);
        assert!(err.to_string().contains("cancelled"));
    }
}
