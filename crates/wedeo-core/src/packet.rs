use bitflags::bitflags;

use crate::buffer::Buffer;
use crate::timestamp::NOPTS_VALUE;

bitflags! {
    /// Packet flags, matching FFmpeg's AV_PKT_FLAG_*.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct PacketFlags: u32 {
        const KEY     = 0x0001;
        const CORRUPT = 0x0002;
        const DISCARD = 0x0004;
        const TRUSTED = 0x0008;
        const DISPOSABLE = 0x0010;
    }
}

/// Packet side data type, matching FFmpeg's AVPacketSideDataType.
/// Values are sequential starting from 0, matching libavcodec/packet.h.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum PacketSideDataType {
    Palette = 0,
    NewExtradata = 1,
    ParamChange = 2,
    H263MbInfo = 3,
    ReplayGain = 4,
    DisplayMatrix = 5,
    Stereo3d = 6,
    AudioServiceType = 7,
    QualityStats = 8,
    FallbackTrack = 9,
    CpbProperties = 10,
    SkipSamples = 11,
    JpDualmono = 12,
    StringsMetadata = 13,
    SubtitlePosition = 14,
    MatroskaBlockadditional = 15,
    WebvttIdentifier = 16,
    WebvttSettings = 17,
    MetadataUpdate = 18,
    MpegtsStreamId = 19,
    MasteringDisplayMetadata = 20,
    Spherical = 21,
    ContentLightLevel = 22,
    A53Cc = 23,
    EncryptionInitInfo = 24,
    EncryptionInfo = 25,
    Afd = 26,
    Prft = 27,
    IccProfile = 28,
    DoviConf = 29,
    S12mTimecode = 30,
    DynamicHdr10Plus = 31,
    IamfMixGainParam = 32,
    IamfDemixingInfoParam = 33,
    IamfReconGainInfoParam = 34,
    AmbientViewingEnvironment = 35,
    FrameCropping = 36,
    Lcevc = 37,
    ReferenceDisplays3d = 38,
    RtcpSr = 39,
    Exif = 40,
}

/// A piece of side data attached to a packet.
#[derive(Debug, Clone)]
pub struct PacketSideData {
    pub data_type: PacketSideDataType,
    pub data: Vec<u8>,
}

/// Encoded packet, matching FFmpeg's AVPacket concept.
#[derive(Debug, Clone)]
pub struct Packet {
    /// Compressed data buffer.
    pub data: Buffer,
    /// Presentation timestamp in time_base units.
    pub pts: i64,
    /// Decompression timestamp in time_base units.
    pub dts: i64,
    /// Duration in time_base units.
    pub duration: i64,
    /// Stream index this packet belongs to.
    pub stream_index: usize,
    pub flags: PacketFlags,
    /// Position in the stream (byte offset), or -1 if unknown.
    pub pos: i64,
    pub side_data: Vec<PacketSideData>,
    /// Number of decoded samples to trim from the start of the decoded packet
    /// (encoder delay / priming). Used for gapless playback.
    pub trim_start: u32,
    /// Number of decoded samples to trim from the end of the decoded packet
    /// (encoder padding). Used for gapless playback.
    pub trim_end: u32,
}

impl Packet {
    /// Create a new empty packet.
    pub fn new() -> Self {
        Self {
            data: Buffer::new(0),
            pts: NOPTS_VALUE,
            dts: NOPTS_VALUE,
            duration: 0,
            stream_index: 0,
            flags: PacketFlags::empty(),
            pos: -1,
            side_data: Vec::new(),
            trim_start: 0,
            trim_end: 0,
        }
    }

    /// Create a packet from raw data.
    pub fn from_slice(data: &[u8]) -> Self {
        Self {
            data: Buffer::from_slice(data),
            pts: NOPTS_VALUE,
            dts: NOPTS_VALUE,
            duration: 0,
            stream_index: 0,
            flags: PacketFlags::empty(),
            pos: -1,
            side_data: Vec::new(),
            trim_start: 0,
            trim_end: 0,
        }
    }

    /// Get the size of the packet data.
    pub fn size(&self) -> usize {
        self.data.size()
    }
}

impl Default for Packet {
    fn default() -> Self {
        Self::new()
    }
}
