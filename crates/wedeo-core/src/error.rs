use std::fmt;

/// Wedeo error type, modeled after FFmpeg's AVERROR codes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// End of file / stream.
    Eof,
    /// Resource temporarily unavailable (try again).
    Again,
    /// Invalid data found when processing input.
    InvalidData,
    /// Decoder not found.
    DecoderNotFound,
    /// Encoder not found.
    EncoderNotFound,
    /// Demuxer not found.
    DemuxerNotFound,
    /// Muxer not found.
    MuxerNotFound,
    /// Filter not found.
    FilterNotFound,
    /// Protocol not found.
    ProtocolNotFound,
    /// Stream not found.
    StreamNotFound,
    /// Bug detected (should not happen, indicates a bug in wedeo).
    Bug,
    /// Option not found.
    OptionNotFound,
    /// Not yet implemented.
    PatchwelcomeNotImplemented,
    /// Exit requested.
    Exit,
    /// Invalid argument.
    InvalidArgument,
    /// Out of memory.
    OutOfMemory,
    /// I/O error with message.
    Io(std::io::ErrorKind),
    /// Other error with message.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Eof => write!(f, "End of file"),
            Error::Again => write!(f, "Resource temporarily unavailable"),
            Error::InvalidData => write!(f, "Invalid data found when processing input"),
            Error::DecoderNotFound => write!(f, "Decoder not found"),
            Error::EncoderNotFound => write!(f, "Encoder not found"),
            Error::DemuxerNotFound => write!(f, "Demuxer not found"),
            Error::MuxerNotFound => write!(f, "Muxer not found"),
            Error::FilterNotFound => write!(f, "Filter not found"),
            Error::ProtocolNotFound => write!(f, "Protocol not found"),
            Error::StreamNotFound => write!(f, "Stream not found"),
            Error::Bug => write!(f, "Internal bug"),
            Error::OptionNotFound => write!(f, "Option not found"),
            Error::PatchwelcomeNotImplemented => write!(f, "Not yet implemented"),
            Error::Exit => write!(f, "Exit requested"),
            Error::InvalidArgument => write!(f, "Invalid argument"),
            Error::OutOfMemory => write!(f, "Out of memory"),
            Error::Io(kind) => write!(f, "I/O error: {kind}"),
            Error::Other(msg) => write!(f, "{msg}"),
        }
    }
}

impl std::error::Error for Error {}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            std::io::ErrorKind::UnexpectedEof => Error::Eof,
            std::io::ErrorKind::WouldBlock => Error::Again,
            std::io::ErrorKind::InvalidData => Error::InvalidData,
            std::io::ErrorKind::InvalidInput => Error::InvalidArgument,
            std::io::ErrorKind::OutOfMemory => Error::OutOfMemory,
            kind => Error::Io(kind),
        }
    }
}

/// Convenience type alias for wedeo results.
pub type Result<T> = std::result::Result<T, Error>;
