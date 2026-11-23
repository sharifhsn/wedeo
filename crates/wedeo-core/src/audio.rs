use std::fmt;

use crate::buffer::Buffer;
use crate::media::SideData;
use crate::time::{Rational, Timestamp};

/// Audio sample format supporting both interleaved (packed) and planar layouts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SampleFormat {
    U8,
    S16,
    S32,
    S64,
    F32,
    F64,
    U8Planar,
    S16Planar,
    S32Planar,
    S64Planar,
    F32Planar,
    F64Planar,
}

impl SampleFormat {
    #[inline]
    pub const fn bytes_per_sample(self) -> usize {
        match self {
            Self::U8 | Self::U8Planar => 1,
            Self::S16 | Self::S16Planar => 2,
            Self::S32 | Self::S32Planar | Self::F32 | Self::F32Planar => 4,
            Self::S64 | Self::S64Planar | Self::F64 | Self::F64Planar => 8,
        }
    }

    #[inline]
    pub const fn is_planar(self) -> bool {
        matches!(
            self,
            Self::U8Planar
                | Self::S16Planar
                | Self::S32Planar
                | Self::S64Planar
                | Self::F32Planar
                | Self::F64Planar
        )
    }

    #[inline]
    pub const fn packed(self) -> Self {
        match self {
            Self::U8Planar => Self::U8,
            Self::S16Planar => Self::S16,
            Self::S32Planar => Self::S32,
            Self::S64Planar => Self::S64,
            Self::F32Planar => Self::F32,
            Self::F64Planar => Self::F64,
            other => other,
        }
    }

    #[inline]
    pub const fn planar(self) -> Self {
        match self {
            Self::U8 => Self::U8Planar,
            Self::S16 => Self::S16Planar,
            Self::S32 => Self::S32Planar,
            Self::S64 => Self::S64Planar,
            Self::F32 => Self::F32Planar,
            Self::F64 => Self::F64Planar,
            other => other,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            Self::U8 => "u8",
            Self::S16 => "s16",
            Self::S32 => "s32",
            Self::S64 => "s64",
            Self::F32 => "flt",
            Self::F64 => "dbl",
            Self::U8Planar => "u8p",
            Self::S16Planar => "s16p",
            Self::S32Planar => "s32p",
            Self::S64Planar => "s64p",
            Self::F32Planar => "fltp",
            Self::F64Planar => "dblp",
        }
    }
}

impl fmt::Display for SampleFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

/// Individual audio speaker position, matching FFmpeg's `AVChannel`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Channel {
    FrontLeft = 0,
    FrontRight = 1,
    FrontCenter = 2,
    LowFrequency = 3,
    BackLeft = 4,
    BackRight = 5,
    FrontLeftOfCenter = 6,
    FrontRightOfCenter = 7,
    BackCenter = 8,
    SideLeft = 9,
    SideRight = 10,
    TopCenter = 11,
    TopFrontLeft = 12,
    TopFrontCenter = 13,
    TopFrontRight = 14,
    TopBackLeft = 15,
    TopBackCenter = 16,
    TopBackRight = 17,
    StereoLeft = 29,
    StereoRight = 30,
    WideLeft = 31,
    WideRight = 32,
    SurroundDirectLeft = 33,
    SurroundDirectRight = 34,
    LowFrequency2 = 35,
    TopSideLeft = 36,
    TopSideRight = 37,
    BottomFrontCenter = 38,
    BottomFrontLeft = 39,
    BottomFrontRight = 40,
    BinauralLeft = 61,
    BinauralRight = 62,
}

impl Channel {
    pub const fn name(self) -> &'static str {
        match self {
            Self::FrontLeft => "FL",
            Self::FrontRight => "FR",
            Self::FrontCenter => "FC",
            Self::LowFrequency => "LFE",
            Self::BackLeft => "BL",
            Self::BackRight => "BR",
            Self::FrontLeftOfCenter => "FLC",
            Self::FrontRightOfCenter => "FRC",
            Self::BackCenter => "BC",
            Self::SideLeft => "SL",
            Self::SideRight => "SR",
            Self::TopCenter => "TC",
            Self::TopFrontLeft => "TFL",
            Self::TopFrontCenter => "TFC",
            Self::TopFrontRight => "TFR",
            Self::TopBackLeft => "TBL",
            Self::TopBackCenter => "TBC",
            Self::TopBackRight => "TBR",
            Self::StereoLeft => "DL",
            Self::StereoRight => "DR",
            Self::WideLeft => "WL",
            Self::WideRight => "WR",
            Self::SurroundDirectLeft => "SDL",
            Self::SurroundDirectRight => "SDR",
            Self::LowFrequency2 => "LFE2",
            Self::TopSideLeft => "TSL",
            Self::TopSideRight => "TSR",
            Self::BottomFrontCenter => "BFC",
            Self::BottomFrontLeft => "BFL",
            Self::BottomFrontRight => "BFR",
            Self::BinauralLeft => "BinL",
            Self::BinauralRight => "BinR",
        }
    }
}

/// Channel ordering scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelOrder {
    Native,
    Custom,
    Unspec,
}

/// Audio channel layout supporting native, custom, and unspecified layouts.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ChannelLayout {
    pub order: ChannelOrder,
    pub nb_channels: u32,
    pub channels: Vec<Channel>,
}

impl ChannelLayout {
    pub fn mono() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 1,
            channels: vec![Channel::FrontCenter],
        }
    }

    pub fn stereo() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 2,
            channels: vec![Channel::FrontLeft, Channel::FrontRight],
        }
    }

    pub fn surround_5_1() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 6,
            channels: vec![
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::FrontCenter,
                Channel::LowFrequency,
                Channel::BackLeft,
                Channel::BackRight,
            ],
        }
    }

    pub fn surround_7_1() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 8,
            channels: vec![
                Channel::FrontLeft,
                Channel::FrontRight,
                Channel::FrontCenter,
                Channel::LowFrequency,
                Channel::BackLeft,
                Channel::BackRight,
                Channel::SideLeft,
                Channel::SideRight,
            ],
        }
    }

    pub fn unspec(nb_channels: u32) -> Self {
        Self {
            order: ChannelOrder::Unspec,
            nb_channels,
            channels: Vec::new(),
        }
    }

    /// Construct a default FFmpeg layout for a given channel count (`1 -> mono`, `2 -> stereo`, etc.).
    pub fn default_for_count(nb_channels: u32) -> Self {
        match nb_channels {
            1 => Self::mono(),
            2 => Self::stereo(),
            6 => Self::surround_5_1(),
            8 => Self::surround_7_1(),
            n => Self::unspec(n),
        }
    }

    /// Construct a layout from a WAV / `WAVEFORMATEXTENSIBLE` channel mask.
    pub fn from_wav_channel_mask(mask: u32) -> Self {
        const WAV_CHANNELS: [(u32, Channel); 18] = [
            (1 << 0, Channel::FrontLeft),
            (1 << 1, Channel::FrontRight),
            (1 << 2, Channel::FrontCenter),
            (1 << 3, Channel::LowFrequency),
            (1 << 4, Channel::BackLeft),
            (1 << 5, Channel::BackRight),
            (1 << 6, Channel::FrontLeftOfCenter),
            (1 << 7, Channel::FrontRightOfCenter),
            (1 << 8, Channel::BackCenter),
            (1 << 9, Channel::SideLeft),
            (1 << 10, Channel::SideRight),
            (1 << 11, Channel::TopCenter),
            (1 << 12, Channel::TopFrontLeft),
            (1 << 13, Channel::TopFrontCenter),
            (1 << 14, Channel::TopFrontRight),
            (1 << 15, Channel::TopBackLeft),
            (1 << 16, Channel::TopBackCenter),
            (1 << 17, Channel::TopBackRight),
        ];

        if mask == 0 {
            return Self::unspec(0);
        }

        let mut channels = Vec::new();
        for &(bit, ch) in &WAV_CHANNELS {
            if mask & bit != 0 {
                channels.push(ch);
            }
        }
        let nb_channels = channels.len() as u32;
        Self {
            order: ChannelOrder::Native,
            nb_channels,
            channels,
        }
    }

    /// Return the FFmpeg canonical layout name if this matches a standard layout
    /// (`libavutil/channel_layout.c`).
    pub fn standard_name(&self) -> Option<&'static str> {
        use Channel::*;
        match self.channels.as_slice() {
            [FrontCenter] => Some("mono"),
            [FrontLeft, FrontRight] => Some("stereo"),
            [FrontLeft, FrontRight, LowFrequency] => Some("2.1"),
            [FrontLeft, FrontRight, FrontCenter] => Some("3.0"),
            [FrontLeft, FrontRight, BackCenter] => Some("3.0(back)"),
            [FrontLeft, FrontRight, FrontCenter, BackCenter] => Some("4.0"),
            [FrontLeft, FrontRight, BackLeft, BackRight] => Some("quad"),
            [FrontLeft, FrontRight, SideLeft, SideRight] => Some("quad(side)"),
            [FrontLeft, FrontRight, FrontCenter, LowFrequency] => Some("3.1"),
            [FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight] => Some("5.0"),
            [FrontLeft, FrontRight, FrontCenter, SideLeft, SideRight] => Some("5.0(side)"),
            [FrontLeft, FrontRight, FrontCenter, LowFrequency, BackCenter] => Some("4.1"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
            ] => Some("5.1"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                SideLeft,
                SideRight,
            ] => Some("5.1(side)"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackCenter,
                SideLeft,
                SideRight,
            ] => Some("6.0"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackCenter,
                SideLeft,
                SideRight,
            ] => Some("6.1"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackLeft,
                BackRight,
                SideLeft,
                SideRight,
            ] => Some("7.0"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                SideLeft,
                SideRight,
            ] => Some("7.1"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
            ] => Some("5.1.2"),
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
                TopBackLeft,
                TopBackRight,
            ] => Some("7.1.4"),
            [StereoLeft, StereoRight] => Some("downmix"),
            [BinauralLeft, BinauralRight] => Some("binaural"),
            _ => None,
        }
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.channels.is_empty() {
            return write!(f, "{} channels", self.nb_channels);
        }
        if let Some(name) = self.standard_name() {
            return f.write_str(name);
        }
        for (i, ch) in self.channels.iter().enumerate() {
            if i > 0 {
                f.write_str("+")?;
            }
            f.write_str(ch.name())?;
        }
        Ok(())
    }
}

/// Gapless audio priming and trailing sample trim counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct SampleTrim {
    pub skip_start: u32,
    pub skip_end: u32,
}

/// Audio stream-level parameters.
#[derive(Debug, Clone)]
pub struct AudioStreamParams {
    pub sample_rate: u32,
    pub format: SampleFormat,
    pub channel_layout: ChannelLayout,
    pub frame_size: u32,
    pub block_align: u32,
    pub bits_per_coded_sample: u32,
    pub bit_rate: i64,
    pub initial_padding: u32,
    pub trailing_padding: u32,
    pub extradata: Vec<u8>,
}

impl AudioStreamParams {
    pub fn new(sample_rate: u32, format: SampleFormat, channel_layout: ChannelLayout) -> Self {
        Self {
            sample_rate,
            format,
            channel_layout,
            frame_size: 0,
            block_align: 0,
            bits_per_coded_sample: 0,
            bit_rate: 0,
            initial_padding: 0,
            trailing_padding: 0,
            extradata: Vec::new(),
        }
    }
}

/// Decoded audio sample block (planar or interleaved) with explicit VFR timing.
#[derive(Debug, Clone)]
pub struct AudioBlock {
    /// One `Buffer` per channel for planar formats; single interleaved `Buffer` for packed formats.
    pub planes: Vec<Buffer>,
    pub nb_samples: u32,
    pub format: SampleFormat,
    pub sample_rate: u32,
    pub channel_layout: ChannelLayout,
    pub pts: Timestamp,
    pub duration: i64,
    pub time_base: Rational,
    pub side_data: Vec<SideData>,
}

impl AudioBlock {
    pub fn new(
        planes: Vec<Buffer>,
        nb_samples: u32,
        format: SampleFormat,
        sample_rate: u32,
        channel_layout: ChannelLayout,
    ) -> Self {
        Self {
            planes,
            nb_samples,
            format,
            sample_rate,
            channel_layout,
            pts: Timestamp::NONE,
            duration: 0,
            time_base: Rational::new(1, sample_rate as i32),
            side_data: Vec::new(),
        }
    }

    /// Expected number of plane buffers for this block's format and channel count.
    #[inline]
    pub fn expected_plane_count(&self) -> usize {
        if self.format.is_planar() {
            self.channel_layout.nb_channels as usize
        } else {
            1
        }
    }

    /// Byte size of each plane buffer.
    #[inline]
    pub fn bytes_per_plane(&self) -> usize {
        let samples = self.nb_samples as usize;
        let bps = self.format.bytes_per_sample();
        if self.format.is_planar() {
            samples * bps
        } else {
            samples * bps * self.channel_layout.nb_channels as usize
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn channel_layout_standard_names_and_wav_mask() {
        assert_eq!(ChannelLayout::mono().to_string(), "mono");
        assert_eq!(ChannelLayout::stereo().to_string(), "stereo");
        assert_eq!(ChannelLayout::surround_5_1().to_string(), "5.1");
        assert_eq!(ChannelLayout::surround_7_1().to_string(), "7.1");
        assert_eq!(
            ChannelLayout::from_wav_channel_mask(0x3F).to_string(),
            "5.1"
        );
        assert_eq!(
            ChannelLayout::from_wav_channel_mask(0x04).to_string(),
            "mono"
        );
    }

    #[test]
    fn sample_format_planar_packed_roundtrip() {
        assert!(SampleFormat::F32Planar.is_planar());
        assert_eq!(SampleFormat::F32Planar.packed(), SampleFormat::F32);
        assert_eq!(SampleFormat::S16.planar(), SampleFormat::S16Planar);
        assert_eq!(SampleFormat::F64.bytes_per_sample(), 8);
    }
}
