use symphonia::core::audio::Channels;
use wedeo_core::{Channel, ChannelLayout, ChannelOrder};

/// Convert a wedeo `ChannelLayout` back to symphonia `Channels`.
///
/// This is the reverse of `channels_to_layout`. Used when creating a
/// symphonia decoder from wedeo track parameters.
pub fn layout_to_channels(layout: &ChannelLayout) -> Channels {
    use symphonia::core::audio::Channels as C;

    let mapping: &[(Channel, Channels)] = &[
        (Channel::FrontLeft, C::FRONT_LEFT),
        (Channel::FrontRight, C::FRONT_RIGHT),
        (Channel::FrontCenter, C::FRONT_CENTRE),
        (Channel::LowFrequency, C::LFE1),
        (Channel::BackLeft, C::REAR_LEFT),
        (Channel::BackRight, C::REAR_RIGHT),
        (Channel::FrontLeftOfCenter, C::FRONT_LEFT_CENTRE),
        (Channel::FrontRightOfCenter, C::FRONT_RIGHT_CENTRE),
        (Channel::BackCenter, C::REAR_CENTRE),
        (Channel::SideLeft, C::SIDE_LEFT),
        (Channel::SideRight, C::SIDE_RIGHT),
        (Channel::TopCenter, C::TOP_CENTRE),
        (Channel::TopFrontLeft, C::TOP_FRONT_LEFT),
        (Channel::TopFrontCenter, C::TOP_FRONT_CENTRE),
        (Channel::TopFrontRight, C::TOP_FRONT_RIGHT),
        (Channel::TopBackLeft, C::TOP_REAR_LEFT),
        (Channel::TopBackCenter, C::TOP_REAR_CENTRE),
        (Channel::TopBackRight, C::TOP_REAR_RIGHT),
        (Channel::LowFrequency2, C::LFE2),
    ];

    let mut channels = Channels::empty();
    for ch in &layout.channels {
        for &(wedeo_ch, sym_ch) in mapping {
            if *ch == wedeo_ch {
                channels |= sym_ch;
                break;
            }
        }
    }

    // Fallback: if no channels matched (e.g. mono mapped to FrontCenter is matched above,
    // or Unspec layout with count) use symphonia's expected front channel placeholders.
    if channels.is_empty() {
        match layout.nb_channels {
            1 => channels = C::FRONT_LEFT,
            2 => channels = C::FRONT_LEFT | C::FRONT_RIGHT,
            _ => {}
        }
    }

    channels
}

/// Convert symphonia `Channels` to a wedeo `ChannelLayout`.
///
/// Symphonia uses `FRONT_LEFT` as a placeholder for formats that don't carry
/// explicit channel mapping (e.g. basic WAV without `WAVEFORMATEXTENSIBLE`).
/// We detect the common mono/stereo cases and return the standard FFmpeg
/// layouts so that output matches.
pub fn channels_to_layout(channels: Channels, count: usize) -> ChannelLayout {
    use symphonia::core::audio::Channels as C;

    if count == 1 && (channels == C::FRONT_LEFT || channels == C::FRONT_CENTRE) {
        return ChannelLayout::mono();
    }
    if count == 2 && channels == (C::FRONT_LEFT | C::FRONT_RIGHT) {
        return ChannelLayout::stereo();
    }

    let mut layout_channels = Vec::new();

    let mapping: &[(Channels, Channel)] = &[
        (C::FRONT_LEFT, Channel::FrontLeft),
        (C::FRONT_RIGHT, Channel::FrontRight),
        (C::FRONT_CENTRE, Channel::FrontCenter),
        (C::LFE1, Channel::LowFrequency),
        (C::REAR_LEFT, Channel::BackLeft),
        (C::REAR_RIGHT, Channel::BackRight),
        (C::FRONT_LEFT_CENTRE, Channel::FrontLeftOfCenter),
        (C::FRONT_RIGHT_CENTRE, Channel::FrontRightOfCenter),
        (C::REAR_CENTRE, Channel::BackCenter),
        (C::SIDE_LEFT, Channel::SideLeft),
        (C::SIDE_RIGHT, Channel::SideRight),
        (C::TOP_CENTRE, Channel::TopCenter),
        (C::TOP_FRONT_LEFT, Channel::TopFrontLeft),
        (C::TOP_FRONT_CENTRE, Channel::TopFrontCenter),
        (C::TOP_FRONT_RIGHT, Channel::TopFrontRight),
        (C::TOP_REAR_LEFT, Channel::TopBackLeft),
        (C::TOP_REAR_CENTRE, Channel::TopBackCenter),
        (C::TOP_REAR_RIGHT, Channel::TopBackRight),
        (C::LFE2, Channel::LowFrequency2),
    ];

    for &(flag, channel) in mapping {
        if channels.contains(flag) {
            layout_channels.push(channel);
        }
    }

    if layout_channels.is_empty() {
        ChannelLayout::unspec(count as u32)
    } else {
        ChannelLayout {
            order: ChannelOrder::Native,
            nb_channels: layout_channels.len() as u32,
            channels: layout_channels,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symphonia::core::audio::Channels as C;

    #[test]
    fn channel_layout_roundtrips() {
        assert_eq!(channels_to_layout(C::FRONT_LEFT, 1), ChannelLayout::mono());
        assert_eq!(
            channels_to_layout(C::FRONT_LEFT | C::FRONT_RIGHT, 2),
            ChannelLayout::stereo()
        );

        let l51 = ChannelLayout::surround_5_1();
        let sym51 = layout_to_channels(&l51);
        assert_eq!(channels_to_layout(sym51, 6), l51);

        let l71 = ChannelLayout::surround_7_1();
        let sym71 = layout_to_channels(&l71);
        assert_eq!(channels_to_layout(sym71, 8), l71);
    }
}
