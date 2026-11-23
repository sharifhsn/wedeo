//! Core media types, zero-copy aligned buffers, Miri-sound row-progress planes,
//! bitstream primitives, I/O contracts, codec/container traits, and explicit registry.

#[forbid(unsafe_code)]
pub mod audio;
#[forbid(unsafe_code)]
pub mod bitstream;
pub mod buffer;
#[forbid(unsafe_code)]
pub mod clock;
#[forbid(unsafe_code)]
pub mod error;
#[forbid(unsafe_code)]
pub mod io;
#[forbid(unsafe_code)]
pub mod media;
#[forbid(unsafe_code)]
pub mod registry;
pub mod sync_plane;
#[forbid(unsafe_code)]
pub mod time;
#[forbid(unsafe_code)]
pub mod traits;
#[forbid(unsafe_code)]
pub mod video;

// Convenience root re-exports.
pub use audio::{
    AudioBlock, AudioStreamParams, Channel, ChannelLayout, ChannelOrder, SampleFormat, SampleTrim,
};
pub use bitstream::{BitReader, BitReaderLe, BitWriter, adler32_ffmpeg, crc32};
pub use buffer::{
    BorderPadding, BorderedPlane, Buffer, BufferPool, ExternalCpuBuffer, GpuDeviceId, GpuSurface,
    MemoryDomain, NativeGpuHandle, SIMD_ALIGN, SIMD_TAIL_PADDING,
};
pub use clock::{AudioClock, SeekEpoch};
pub use error::{
    Error, ErrorKind, ErrorPolicy, HealthEvent, HealthIssue, MAX_HEALTH_EVENTS, RecvStatus, Result,
    SendStatus, StreamHealth,
};
pub use io::{
    BufByteSource, ByteSink, ByteSource, FileSink, FileSource, InterruptibleSource, MemorySink,
    MemorySource,
};
pub use media::{
    Attachment, Chapter, CodecId, ContainerHeader, DataStreamParams, EditSegment,
    MasteringDisplayMeta, MediaType, MediaUnit, Metadata, Packet, PacketFlags, PositionedPlane,
    ReplayGain, SeekTarget, SideData, StreamParams, SubtitleEvent, SubtitleFormat, SubtitlePayload,
    SubtitleStreamParams, TimedData, TrackDescriptor, TrackDisposition, TrackId,
};
pub use registry::Registry;
pub use sync_plane::{
    PlaneRowReader, PlaneRowWriter, PublishedPlaneView, UnpublishedRowsMut, row_progress_plane,
};
pub use time::{
    ROUND_PASS_MINMAX, Rational, Rounding, TIME_BASE, TIME_BASE_Q, Timestamp, WrappingExt, cmp_q,
    gcd, reduce, rescale, rescale_q, rescale_q_rnd, rescale_rnd,
};
pub use traits::{
    CodecContext, Decoder, DecoderDescriptor, DecoderFactory, Demuxer, DemuxerDescriptor,
    DemuxerFactory, EncodeHint, Encoder, EncoderDescriptor, EncoderFactory, Muxer, MuxerDescriptor,
    MuxerFactory, OptionKind, OptionSchema, PassMode,
};
pub use video::{
    AlphaMode, ChromaLocation, ColorDesc, ColorModel, ColorPrimaries, ColorRange, CompOrder,
    CropRect, Endian, FrameFlags, MatrixCoefficients, PictureType, PixelLayout, PlanePacking,
    SampleDepth, Subsampling, TransferCharacteristic, VideoFrame, VideoStorage, VideoStreamParams,
};
