/// Frame side data types, matching FFmpeg's `AVFrameSideDataType` from `libavutil/frame.h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u32)]
pub enum FrameSideDataType {
    /// The data is the AVPanScan struct defined in libavcodec.
    Panscan = 0,
    /// ATSC A53 Part 4 Closed Captions.
    /// A53 CC bitstream is stored as uint8_t in data.
    A53Cc = 1,
    /// Stereoscopic 3d metadata.
    Stereo3d = 2,
    /// The data is the AVMatrixEncoding enum.
    MatrixEncoding = 3,
    /// Metadata relevant to a downmix procedure.
    DownmixInfo = 4,
    /// ReplayGain information.
    ReplayGain = 5,
    /// 3x3 transformation matrix describing an affine transformation
    /// for correct presentation.
    DisplayMatrix = 6,
    /// Active Format Description data consisting of a single byte.
    Afd = 7,
    /// Motion vectors exported by some codecs.
    MotionVectors = 8,
    /// Recommends skipping the specified number of samples.
    SkipSamples = 9,
    /// Audio service type.
    AudioServiceType = 10,
    /// Mastering display metadata associated with a video frame.
    MasteringDisplayMetadata = 11,
    /// The GOP timecode in 25 bit timecode format.
    GopTimecode = 12,
    /// Spherical mapping metadata.
    Spherical = 13,
    /// Content light level (based on CTA-861.3).
    ContentLightLevel = 14,
    /// ICC profile as an opaque octet buffer following ISO 15076-1.
    IccProfile = 15,
    /// SMPTE ST 12-1 timecode.
    S12mTimecode = 16,
    /// HDR dynamic metadata (SMPTE 2094-40:2016).
    DynamicHdrPlus = 17,
    /// Regions Of Interest.
    RegionsOfInterest = 18,
    /// Encoding parameters for a video frame.
    VideoEncParams = 19,
    /// User data unregistered metadata (H.26[45] UDU SEI message).
    SeiUnregistered = 20,
    /// Film grain parameters for a frame.
    FilmGrainParams = 21,
    /// Bounding boxes for object detection and classification.
    DetectionBboxes = 22,
    /// Dolby Vision RPU raw data, suitable for passing to x265 or other libraries.
    DoviRpuBuffer = 23,
    /// Parsed Dolby Vision metadata.
    DoviMetadata = 24,
    /// HDR Vivid dynamic metadata (CUVA 005.1-2021).
    DynamicHdrVivid = 25,
    /// Ambient viewing environment metadata (H.274).
    AmbientViewingEnvironment = 26,
    /// Encoder-specific hinting information about changed/unchanged portions of a frame.
    VideoHint = 27,
    /// Raw LCEVC payload data.
    Lcevc = 28,
    /// View ID for multi-view video streams (e.g. stereoscopic 3D content).
    ViewId = 29,
    /// 3D reference displays information.
    ThreeDReferenceDisplays = 30,
    /// Extensible image file format metadata (EXIF).
    Exif = 31,
}

/// Side data associated with a frame, matching FFmpeg's `AVFrameSideData`.
#[derive(Debug, Clone)]
pub struct FrameSideData {
    pub data_type: FrameSideDataType,
    pub data: Vec<u8>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_side_data_type_discriminants() {
        // Verify key discriminant values match FFmpeg's enum.
        assert_eq!(FrameSideDataType::Panscan as u32, 0);
        assert_eq!(FrameSideDataType::A53Cc as u32, 1);
        assert_eq!(FrameSideDataType::DisplayMatrix as u32, 6);
        assert_eq!(FrameSideDataType::MasteringDisplayMetadata as u32, 11);
        assert_eq!(FrameSideDataType::ContentLightLevel as u32, 14);
        assert_eq!(FrameSideDataType::FilmGrainParams as u32, 21);
        assert_eq!(FrameSideDataType::DoviRpuBuffer as u32, 23);
        assert_eq!(FrameSideDataType::DoviMetadata as u32, 24);
        assert_eq!(FrameSideDataType::DynamicHdrVivid as u32, 25);
        assert_eq!(FrameSideDataType::AmbientViewingEnvironment as u32, 26);
        assert_eq!(FrameSideDataType::VideoHint as u32, 27);
        assert_eq!(FrameSideDataType::Lcevc as u32, 28);
        assert_eq!(FrameSideDataType::ViewId as u32, 29);
        assert_eq!(FrameSideDataType::ThreeDReferenceDisplays as u32, 30);
        assert_eq!(FrameSideDataType::Exif as u32, 31);
    }

    #[test]
    fn test_frame_side_data_creation() {
        let sd = FrameSideData {
            data_type: FrameSideDataType::A53Cc,
            data: vec![0x01, 0x02, 0x03],
        };
        assert_eq!(sd.data_type, FrameSideDataType::A53Cc);
        assert_eq!(sd.data.len(), 3);
    }

    #[test]
    fn test_frame_side_data_clone() {
        let sd = FrameSideData {
            data_type: FrameSideDataType::IccProfile,
            data: vec![0xFF; 100],
        };
        let sd2 = sd.clone();
        assert_eq!(sd2.data_type, FrameSideDataType::IccProfile);
        assert_eq!(sd2.data.len(), 100);
    }
}
