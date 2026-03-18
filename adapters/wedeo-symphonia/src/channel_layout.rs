use symphonia::core::audio::Channels;
use wedeo_core::channel_layout::{Channel, ChannelLayout, ChannelOrder};

/// Convert a wedeo ChannelLayout back to symphonia Channels.
///
/// This is the reverse of `channels_to_layout`. Used when creating a
/// symphonia decoder from wedeo codec parameters.
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

    // Fallback: if no channels matched but we know the count, use front channels
    if channels.is_empty() {
        match layout.nb_channels {
            1 => channels = C::FRONT_LEFT,
            2 => channels = C::FRONT_LEFT | C::FRONT_RIGHT,
            _ => {}
        }
    }

    channels
}

/// Convert symphonia Channels to a wedeo ChannelLayout.
///
/// Symphonia uses FRONT_LEFT as a placeholder for formats that don't carry
/// explicit channel mapping (e.g. basic WAV without WAVEFORMATEXTENSIBLE).
/// We detect the common mono/stereo cases and return the standard FFmpeg
/// layouts so that framecrc output matches.
pub fn channels_to_layout(channels: Channels, count: usize) -> ChannelLayout {
    use symphonia::core::audio::Channels as C;

    // Handle common cases where symphonia uses placeholder channels:
    // mono WAV reports FRONT_LEFT with count=1, stereo reports FL+FR with count=2.
    // Map these to the standard FFmpeg layouts (mono=FC, stereo=FL+FR).
    if count == 1 && channels == C::FRONT_LEFT {
        return ChannelLayout::mono();
    }
    if count == 2 && channels == (C::FRONT_LEFT | C::FRONT_RIGHT) {
        return ChannelLayout::stereo();
    }

    let mut layout_channels = Vec::new();

    // Map symphonia channel flags to wedeo Channel enum.
    // The order matches FFmpeg's native channel order (ascending bit positions).
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
        ChannelLayout::unspec(count as i32)
    } else {
        ChannelLayout {
            order: ChannelOrder::Native,
            nb_channels: layout_channels.len() as i32,
            channels: layout_channels,
        }
    }
}
