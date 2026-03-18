// H.264 SEI (Supplemental Enhancement Information) parsing.
//
// Most SEI messages are parsed and ignored for Baseline conformance.
// Only those that affect decode behavior (like recovery_point) are retained.
//
// Reference: ITU-T H.264 Annex D, FFmpeg libavcodec/h264_sei.c

/// SEI message types (subset relevant to Baseline).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeiType {
    BufferingPeriod,
    PicTiming,
    UserDataRegistered,
    UserDataUnregistered,
    RecoveryPoint,
    Other(u32),
}

impl From<u32> for SeiType {
    fn from(val: u32) -> Self {
        match val {
            0 => SeiType::BufferingPeriod,
            1 => SeiType::PicTiming,
            4 => SeiType::UserDataRegistered,
            5 => SeiType::UserDataUnregistered,
            6 => SeiType::RecoveryPoint,
            other => SeiType::Other(other),
        }
    }
}

/// Parsed SEI recovery point data.
#[derive(Debug, Clone)]
pub struct RecoveryPoint {
    pub recovery_frame_cnt: u32,
    pub exact_match_flag: bool,
    pub broken_link_flag: bool,
}

/// Parse SEI NAL unit data, returning any recovery point info.
///
/// SEI messages are structured as:
///   while (more_data) {
///       payloadType = sum of 0xFF bytes + final byte
///       payloadSize = sum of 0xFF bytes + final byte
///       payload[payloadSize]
///   }
///
/// Most messages are silently ignored. Only recovery_point is returned.
pub fn parse_sei(data: &[u8]) -> Option<RecoveryPoint> {
    let mut pos = 0;
    let mut recovery = None;

    while pos < data.len() {
        // Read payload type
        let mut payload_type: u32 = 0;
        while pos < data.len() && data[pos] == 0xFF {
            payload_type += 255;
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }
        payload_type += data[pos] as u32;
        pos += 1;

        // Read payload size
        let mut payload_size: u32 = 0;
        while pos < data.len() && data[pos] == 0xFF {
            payload_size += 255;
            pos += 1;
        }
        if pos >= data.len() {
            break;
        }
        payload_size += data[pos] as u32;
        pos += 1;

        let payload_end = pos + payload_size as usize;
        if payload_end > data.len() {
            break;
        }

        let sei_type = SeiType::from(payload_type);
        if sei_type == SeiType::RecoveryPoint && payload_size >= 1 {
            recovery = Some(RecoveryPoint {
                recovery_frame_cnt: 0,
                exact_match_flag: false,
                broken_link_flag: false,
            });
        }

        pos = payload_end;
    }

    recovery
}
