use std::fmt;

/// Individual audio channel, matching FFmpeg's AVChannel.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(i32)]
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
    pub fn name(self) -> &'static str {
        match self {
            Channel::FrontLeft => "FL",
            Channel::FrontRight => "FR",
            Channel::FrontCenter => "FC",
            Channel::LowFrequency => "LFE",
            Channel::BackLeft => "BL",
            Channel::BackRight => "BR",
            Channel::FrontLeftOfCenter => "FLC",
            Channel::FrontRightOfCenter => "FRC",
            Channel::BackCenter => "BC",
            Channel::SideLeft => "SL",
            Channel::SideRight => "SR",
            Channel::TopCenter => "TC",
            Channel::TopFrontLeft => "TFL",
            Channel::TopFrontCenter => "TFC",
            Channel::TopFrontRight => "TFR",
            Channel::TopBackLeft => "TBL",
            Channel::TopBackCenter => "TBC",
            Channel::TopBackRight => "TBR",
            Channel::StereoLeft => "DL",
            Channel::StereoRight => "DR",
            Channel::WideLeft => "WL",
            Channel::WideRight => "WR",
            Channel::SurroundDirectLeft => "SDL",
            Channel::SurroundDirectRight => "SDR",
            Channel::LowFrequency2 => "LFE2",
            Channel::TopSideLeft => "TSL",
            Channel::TopSideRight => "TSR",
            Channel::BottomFrontCenter => "BFC",
            Channel::BottomFrontLeft => "BFL",
            Channel::BottomFrontRight => "BFR",
            Channel::BinauralLeft => "BinL",
            Channel::BinauralRight => "BinR",
        }
    }
}

/// Channel ordering scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChannelOrder {
    /// The native channel order, i.e. the channels are in the same order in
    /// which they are defined in the Channel enum.
    Native,
    /// Custom order — channels are described by the map in ChannelLayout.
    Custom,
    /// Unspecified order — only the channel count is known.
    Unspec,
}

/// Audio channel layout.
#[derive(Debug, Clone)]
pub struct ChannelLayout {
    pub order: ChannelOrder,
    pub nb_channels: i32,
    pub channels: Vec<Channel>,
}

impl ChannelLayout {
    /// Mono layout: front center.
    pub fn mono() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 1,
            channels: vec![Channel::FrontCenter],
        }
    }

    /// Stereo layout: front left + front right.
    pub fn stereo() -> Self {
        Self {
            order: ChannelOrder::Native,
            nb_channels: 2,
            channels: vec![Channel::FrontLeft, Channel::FrontRight],
        }
    }

    /// 5.1 surround layout.
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

    /// 7.1 surround layout.
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

    /// Unspecified layout with the given channel count.
    pub fn unspec(nb_channels: i32) -> Self {
        Self {
            order: ChannelOrder::Unspec,
            nb_channels,
            channels: Vec::new(),
        }
    }

    /// Construct a channel layout from a WAV/WAVEFORMATEXTENSIBLE channel mask.
    ///
    /// The WAV channel mask bits correspond to:
    /// - bit 0: FRONT_LEFT
    /// - bit 1: FRONT_RIGHT
    /// - bit 2: FRONT_CENTER
    /// - bit 3: LOW_FREQUENCY
    /// - bit 4: BACK_LEFT
    /// - bit 5: BACK_RIGHT
    /// - bit 6: FRONT_LEFT_OF_CENTER
    /// - bit 7: FRONT_RIGHT_OF_CENTER
    /// - bit 8: BACK_CENTER
    /// - bit 9: SIDE_LEFT
    /// - bit 10: SIDE_RIGHT
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
        for &(bit, channel) in &WAV_CHANNELS {
            if mask & bit != 0 {
                channels.push(channel);
            }
        }

        let nb_channels = channels.len() as i32;
        Self {
            order: ChannelOrder::Native,
            nb_channels,
            channels,
        }
    }
}

impl fmt::Display for ChannelLayout {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.channels.is_empty() {
            return write!(f, "{} channels", self.nb_channels);
        }
        // Use FFmpeg's standard layout names for common configurations
        if let Some(name) = self.standard_name() {
            return write!(f, "{name}");
        }
        let names: Vec<&str> = self.channels.iter().map(|c| c.name()).collect();
        write!(f, "{}", names.join("+"))
    }
}

impl ChannelLayout {
    /// Return the FFmpeg standard layout name if this matches a known configuration.
    ///
    /// The layout names and channel compositions match FFmpeg's `channel_layout_map`
    /// table in `libavutil/channel_layout.c`. Channels are listed in native order
    /// (ascending enum/bit-position order).
    pub fn standard_name(&self) -> Option<&'static str> {
        use Channel::*;
        match self.channels.as_slice() {
            // mono: FC
            [FrontCenter] => Some("mono"),
            // stereo: FL+FR
            [FrontLeft, FrontRight] => Some("stereo"),
            // 2.1: FL+FR+LFE
            [FrontLeft, FrontRight, LowFrequency] => Some("2.1"),
            // 3.0: FL+FR+FC  (AV_CH_LAYOUT_SURROUND)
            [FrontLeft, FrontRight, FrontCenter] => Some("3.0"),
            // 3.0(back): FL+FR+BC  (AV_CH_LAYOUT_2_1)
            [FrontLeft, FrontRight, BackCenter] => Some("3.0(back)"),
            // 4.0: FL+FR+FC+BC
            [FrontLeft, FrontRight, FrontCenter, BackCenter] => Some("4.0"),
            // quad: FL+FR+BL+BR
            [FrontLeft, FrontRight, BackLeft, BackRight] => Some("quad"),
            // quad(side): FL+FR+SL+SR  (AV_CH_LAYOUT_2_2)
            [FrontLeft, FrontRight, SideLeft, SideRight] => Some("quad(side)"),
            // 3.1: FL+FR+FC+LFE
            [FrontLeft, FrontRight, FrontCenter, LowFrequency] => Some("3.1"),
            // 5.0: FL+FR+FC+BL+BR  (AV_CH_LAYOUT_5POINT0_BACK)
            [FrontLeft, FrontRight, FrontCenter, BackLeft, BackRight] => Some("5.0"),
            // 5.0(side): FL+FR+FC+SL+SR  (AV_CH_LAYOUT_5POINT0)
            [FrontLeft, FrontRight, FrontCenter, SideLeft, SideRight] => Some("5.0(side)"),
            // 4.1: FL+FR+FC+LFE+BC
            [FrontLeft, FrontRight, FrontCenter, LowFrequency, BackCenter] => Some("4.1"),
            // 5.1: FL+FR+FC+LFE+BL+BR  (AV_CH_LAYOUT_5POINT1_BACK)
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
            ] => Some("5.1"),
            // 5.1(side): FL+FR+FC+LFE+SL+SR  (AV_CH_LAYOUT_5POINT1)
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                SideLeft,
                SideRight,
            ] => Some("5.1(side)"),
            // 6.0: FL+FR+FC+BC+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackCenter,
                SideLeft,
                SideRight,
            ] => Some("6.0"),
            // 6.0(front): FL+FR+FLC+FRC+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
            ] => Some("6.0(front)"),
            // 3.1.2: FL+FR+FC+LFE+TFL+TFR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                TopFrontLeft,
                TopFrontRight,
            ] => Some("3.1.2"),
            // hexagonal: FL+FR+FC+BL+BR+BC
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackLeft,
                BackRight,
                BackCenter,
            ] => Some("hexagonal"),
            // 6.1: FL+FR+FC+LFE+BC+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackCenter,
                SideLeft,
                SideRight,
            ] => Some("6.1"),
            // 6.1(back): FL+FR+FC+LFE+BL+BR+BC
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                BackCenter,
            ] => Some("6.1(back)"),
            // 6.1(front): FL+FR+LFE+FLC+FRC+SL+SR
            [
                FrontLeft,
                FrontRight,
                LowFrequency,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
            ] => Some("6.1(front)"),
            // 7.0: FL+FR+FC+BL+BR+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackLeft,
                BackRight,
                SideLeft,
                SideRight,
            ] => Some("7.0"),
            // 7.0(front): FL+FR+FC+FLC+FRC+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
            ] => Some("7.0(front)"),
            // 7.1: FL+FR+FC+LFE+BL+BR+SL+SR
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
            // 7.1(wide): FL+FR+FC+LFE+BL+BR+FLC+FRC  (AV_CH_LAYOUT_7POINT1_WIDE_BACK)
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                FrontLeftOfCenter,
                FrontRightOfCenter,
            ] => Some("7.1(wide)"),
            // 7.1(wide-side): FL+FR+FC+LFE+FLC+FRC+SL+SR  (AV_CH_LAYOUT_7POINT1_WIDE)
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
            ] => Some("7.1(wide-side)"),
            // 5.1.2: FL+FR+FC+LFE+SL+SR+TFL+TFR
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
            // 5.1.2(back): FL+FR+FC+LFE+BL+BR+TFL+TFR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                TopFrontLeft,
                TopFrontRight,
            ] => Some("5.1.2(back)"),
            // octagonal: FL+FR+FC+BL+BR+BC+SL+SR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackLeft,
                BackRight,
                BackCenter,
                SideLeft,
                SideRight,
            ] => Some("octagonal"),
            // cube: FL+FR+BL+BR+TFL+TFR+TBL+TBR
            [
                FrontLeft,
                FrontRight,
                BackLeft,
                BackRight,
                TopFrontLeft,
                TopFrontRight,
                TopBackLeft,
                TopBackRight,
            ] => Some("cube"),
            // 5.1.4: FL+FR+FC+LFE+SL+SR+TFL+TFR+TBL+TBR  (AV_CH_LAYOUT_5POINT1POINT4_BACK)
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
                TopBackLeft,
                TopBackRight,
            ] => Some("5.1.4"),
            // 7.1.2: FL+FR+FC+LFE+BL+BR+SL+SR+TFL+TFR
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
            ] => Some("7.1.2"),
            // 7.1.4: FL+FR+FC+LFE+BL+BR+SL+SR+TFL+TFR+TBL+TBR  (AV_CH_LAYOUT_7POINT1POINT4_BACK)
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
            // 7.2.3: FL+FR+FC+LFE+BL+BR+SL+SR+TFL+TFR+TBC+LFE2  (AV_CH_LAYOUT_7POINT2POINT3)
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
                TopBackCenter,
                LowFrequency2,
            ] => Some("7.2.3"),
            // 9.1.4: FL+FR+FC+LFE+BL+BR+FLC+FRC+SL+SR+TFL+TFR+TBL+TBR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
                TopBackLeft,
                TopBackRight,
            ] => Some("9.1.4"),
            // 9.1.6: FL+FR+FC+LFE+BL+BR+FLC+FRC+SL+SR+TFL+TFR+TBL+TBR+TSL+TSR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontRight,
                TopBackLeft,
                TopBackRight,
                TopSideLeft,
                TopSideRight,
            ] => Some("9.1.6"),
            // hexadecagonal: FL+FR+FC+BL+BR+BC+SL+SR+TFL+TFC+TFR+TBL+TBC+TBR+WL+WR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                BackLeft,
                BackRight,
                BackCenter,
                SideLeft,
                SideRight,
                TopFrontLeft,
                TopFrontCenter,
                TopFrontRight,
                TopBackLeft,
                TopBackCenter,
                TopBackRight,
                WideLeft,
                WideRight,
            ] => Some("hexadecagonal"),
            // downmix: DL+DR  (AV_CH_LAYOUT_STEREO_DOWNMIX)
            [StereoLeft, StereoRight] => Some("downmix"),
            // binaural: BinL+BinR
            [BinauralLeft, BinauralRight] => Some("binaural"),
            // 22.2: FL+FR+FC+LFE+BL+BR+FLC+FRC+BC+SL+SR+TC+TFL+TFC+TFR+TBL+TBC+TBR+LFE2+TSL+TSR+BFC+BFL+BFR
            [
                FrontLeft,
                FrontRight,
                FrontCenter,
                LowFrequency,
                BackLeft,
                BackRight,
                FrontLeftOfCenter,
                FrontRightOfCenter,
                BackCenter,
                SideLeft,
                SideRight,
                TopCenter,
                TopFrontLeft,
                TopFrontCenter,
                TopFrontRight,
                TopBackLeft,
                TopBackCenter,
                TopBackRight,
                LowFrequency2,
                TopSideLeft,
                TopSideRight,
                BottomFrontCenter,
                BottomFrontLeft,
                BottomFrontRight,
            ] => Some("22.2"),
            _ => None,
        }
    }
}
