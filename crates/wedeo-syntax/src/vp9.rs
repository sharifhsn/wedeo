//! VP9 uncompressed frame header parser, superframe splitter/builder, and `vpcC` extradata helpers.
//!
//! Faithfully ports FFmpeg `n9.1` `libavcodec/vp9.c` (`decode_frame_header` &
//! `read_colorspace_details`), `libavcodec/bsf/vp9_superframe_split.c`, and
//! `libavformat/vpcc.c`.

use std::ops::Range;

use wedeo_core::{
    AlphaMode, BitReader, ChromaLocation, ColorDesc, ColorModel, ColorPrimaries, ColorRange,
    CompOrder, Endian, Error, MatrixCoefficients, PixelLayout, PlanePacking, Result, SampleDepth,
    Subsampling, TransferCharacteristic, VideoStreamParams,
};

/// 24-bit VP9 frame sync code (`0x49 0x83 0x42`).
pub const VP9_SYNC_CODE: u32 = 0x49_83_42;

/// VP9 3-bit color space field (`read_colorspace_details` in `vp9.c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Vp9ColorSpace {
    Unknown = 0,
    Bt601 = 1,
    Bt709 = 2,
    Smpte170 = 3,
    Smpte240 = 4,
    Bt2020 = 5,
    Reserved = 6,
    Rgb = 7,
}

impl Vp9ColorSpace {
    pub const fn from_u8(v: u8) -> Self {
        match v & 0x07 {
            0 => Self::Unknown,
            1 => Self::Bt601,
            2 => Self::Bt709,
            3 => Self::Smpte170,
            4 => Self::Smpte240,
            5 => Self::Bt2020,
            6 => Self::Reserved,
            _ => Self::Rgb,
        }
    }

    pub const fn to_matrix_coefficients(self) -> MatrixCoefficients {
        match self {
            Self::Unknown | Self::Reserved => MatrixCoefficients::Unspecified,
            Self::Bt601 => MatrixCoefficients::Bt470bg,
            Self::Bt709 => MatrixCoefficients::Bt709,
            Self::Smpte170 => MatrixCoefficients::Smpte170m,
            Self::Smpte240 => MatrixCoefficients::Smpte240m,
            Self::Bt2020 => MatrixCoefficients::Bt2020Ncl,
            Self::Rgb => MatrixCoefficients::Identity,
        }
    }
}

/// VP9 interpolation filter mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum Vp9InterpFilter {
    EightTap = 0,
    EightTapSmooth = 1,
    EightTapSharp = 2,
    Bilinear = 3,
    Switchable = 4,
}

/// Parsed VP9 uncompressed frame header (`vp9.c` `decode_frame_header`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Vp9FrameHeader {
    /// VP9 profile (`0..=3`).
    pub profile: u8,
    /// When `Some(idx)`, this frame simply displays existing reference buffer `idx` (`0..=7`).
    pub show_existing_frame: Option<u8>,
    pub is_keyframe: bool,
    pub show_frame: bool,
    pub error_resilient_mode: bool,
    pub intra_only: bool,
    pub reset_frame_context: u8,
    pub refresh_frame_flags: u8,

    /// Bit depth (`8`, `10`, or `12`). `None` on inter frames without embedded color config.
    pub bit_depth: Option<u8>,
    /// Horizontal chroma subsampling (`true` = 2x subsampled, `false` = full width).
    pub subsampling_x: Option<bool>,
    /// Vertical chroma subsampling (`true` = 2x subsampled, `false` = full height).
    pub subsampling_y: Option<bool>,
    pub vp9_color_space: Option<Vp9ColorSpace>,
    pub color: ColorDesc,
    pub pixel_layout: Option<PixelLayout>,

    /// Coded frame width/height (`None` for inter frames inheriting dimensions from a reference frame).
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Optional explicit display/render size when `render_and_frame_size_different == 1`.
    pub render_width: Option<u32>,
    pub render_height: Option<u32>,

    /// Inter-frame reference indices and sign biases (`None` on keyframes / intra-only).
    pub ref_frame_idx: Option<[u8; 3]>,
    pub ref_frame_sign_bias: Option<[bool; 3]>,
    pub allow_high_precision_mv: bool,
    pub interpolation_filter: Option<Vp9InterpFilter>,

    pub refresh_frame_context: bool,
    pub frame_parallel_decoding_mode: bool,
    pub frame_context_idx: u8,
}

impl Vp9FrameHeader {
    /// Convert parsed keyframe / intra-only header into [`VideoStreamParams`].
    pub fn to_video_stream_params(&self) -> Option<VideoStreamParams> {
        let (w, h, layout) = (self.width?, self.height?, self.pixel_layout?);
        let mut params = VideoStreamParams::new(w, h, layout);
        params.color = self.color;
        Some(params)
    }
}

fn vp9_pixel_layout(
    bit_depth: u8,
    is_rgb: bool,
    subsampling_x: bool,
    subsampling_y: bool,
) -> Option<PixelLayout> {
    let depth = match bit_depth {
        8 => SampleDepth::U8,
        10 => SampleDepth::U10,
        12 => SampleDepth::U12,
        _ => return None,
    };
    if is_rgb {
        return Some(PixelLayout {
            model: ColorModel::Rgb,
            packing: PlanePacking::Planar,
            subsampling: Subsampling::Cs444,
            depth,
            endian: Endian::Little,
            alpha: AlphaMode::None,
            order: CompOrder::Gbr,
        });
    }
    let subsampling = match (subsampling_x, subsampling_y) {
        (true, true) => Subsampling::Cs420,
        (true, false) => Subsampling::Cs422,
        (false, false) => Subsampling::Cs444,
        // VP9 4:4:0 (subsampling_x=0, subsampling_y=1) has no dedicated Subsampling variant in wedeo-core;
        // callers can inspect `subsampling_x`/`subsampling_y` directly.
        (false, true) => return None,
    };
    Some(PixelLayout {
        model: ColorModel::Ycbcr,
        packing: PlanePacking::Planar,
        subsampling,
        depth,
        endian: Endian::Little,
        alpha: AlphaMode::None,
        order: CompOrder::Yuv,
    })
}

fn parse_color_config(
    br: &mut BitReader<'_>,
    profile: u8,
) -> Result<(
    u8,
    bool,
    bool,
    Vp9ColorSpace,
    ColorDesc,
    Option<PixelLayout>,
)> {
    let bit_depth = if profile <= 1 {
        8u8
    } else if br.read_bool()? {
        12u8
    } else {
        10u8
    };

    let cs = Vp9ColorSpace::from_u8(br.read_bits(3)? as u8);
    let mut color = ColorDesc::UNSPECIFIED;
    color.matrix = cs.to_matrix_coefficients();
    // Populate matching primaries/transfer when VP9 unambiguously implies them.
    match cs {
        Vp9ColorSpace::Bt709 => {
            color.primaries = ColorPrimaries::Bt709;
            color.transfer = TransferCharacteristic::Bt709;
        }
        Vp9ColorSpace::Bt601 => {
            color.primaries = ColorPrimaries::Bt470bg;
            color.transfer = TransferCharacteristic::Smpte170m;
        }
        Vp9ColorSpace::Smpte170 => {
            color.primaries = ColorPrimaries::Smpte170m;
            color.transfer = TransferCharacteristic::Smpte170m;
        }
        Vp9ColorSpace::Smpte240 => {
            color.primaries = ColorPrimaries::Smpte240m;
            color.transfer = TransferCharacteristic::Smpte240m;
        }
        Vp9ColorSpace::Bt2020 => {
            color.primaries = ColorPrimaries::Bt2020;
            color.transfer = if bit_depth == 12 {
                TransferCharacteristic::Bt2020_12
            } else {
                TransferCharacteristic::Bt2020_10
            };
        }
        Vp9ColorSpace::Rgb => {
            color.primaries = ColorPrimaries::Bt709;
            color.transfer = TransferCharacteristic::Srgb;
        }
        Vp9ColorSpace::Unknown | Vp9ColorSpace::Reserved => {}
    }

    if cs == Vp9ColorSpace::Rgb {
        if (profile & 1) == 0 {
            return Err(Error::invalid_data(
                "vp9/header",
                "RGB colorspace is invalid in even VP9 profiles (0 or 2)",
            ));
        }
        if br.read_bool()? {
            return Err(Error::invalid_data(
                "vp9/header",
                "reserved bit set in VP9 RGB color config",
            ));
        }
        color.range = ColorRange::Full;
        let layout = vp9_pixel_layout(bit_depth, true, false, false);
        Ok((bit_depth, false, false, cs, color, layout))
    } else {
        color.range = if br.read_bool()? {
            ColorRange::Full
        } else {
            ColorRange::Limited
        };
        let (ss_x, ss_y) = if (profile & 1) != 0 {
            let sx = br.read_bool()?;
            let sy = br.read_bool()?;
            if sx && sy {
                return Err(Error::invalid_data(
                    "vp9/header",
                    "4:2:0 subsampling is invalid in VP9 profiles 1 and 3",
                ));
            }
            if br.read_bool()? {
                return Err(Error::invalid_data(
                    "vp9/header",
                    "reserved bit set in VP9 profile 1/3 color config",
                ));
            }
            (sx, sy)
        } else {
            (true, true)
        };
        let layout = vp9_pixel_layout(bit_depth, false, ss_x, ss_y);
        Ok((bit_depth, ss_x, ss_y, cs, color, layout))
    }
}

/// Parse a VP9 uncompressed frame header (matching FFmpeg `n9.1` `libavcodec/vp9.c`).
pub fn parse_frame_header(data: &[u8]) -> Result<Vp9FrameHeader> {
    if data.is_empty() {
        return Err(Error::invalid_data("vp9/header", "empty VP9 frame"));
    }
    let mut br = BitReader::new(data);

    let frame_marker = br.read_bits(2)?;
    if frame_marker != 0x2 {
        return Err(Error::invalid_data(
            "vp9/header",
            "invalid VP9 frame marker (expected 0b10)",
        ));
    }

    let profile_low = br.read_bit()?;
    let profile_high = br.read_bit()?;
    let mut profile = (profile_low | (profile_high << 1)) as u8;
    if profile == 3 {
        profile += br.read_bit()? as u8;
    }
    if profile > 3 {
        return Err(Error::invalid_data(
            "vp9/header",
            "unsupported VP9 profile > 3",
        ));
    }

    if br.read_bool()? {
        let ref_idx = br.read_bits(3)? as u8;
        return Ok(Vp9FrameHeader {
            profile,
            show_existing_frame: Some(ref_idx),
            is_keyframe: false,
            show_frame: true,
            error_resilient_mode: false,
            intra_only: false,
            reset_frame_context: 0,
            refresh_frame_flags: 0,
            bit_depth: None,
            subsampling_x: None,
            subsampling_y: None,
            vp9_color_space: None,
            color: ColorDesc::UNSPECIFIED,
            pixel_layout: None,
            width: None,
            height: None,
            render_width: None,
            render_height: None,
            ref_frame_idx: None,
            ref_frame_sign_bias: None,
            allow_high_precision_mv: false,
            interpolation_filter: None,
            refresh_frame_context: false,
            frame_parallel_decoding_mode: false,
            frame_context_idx: 0,
        });
    }

    let is_keyframe = !br.read_bool()?;
    let show_frame = br.read_bool()?;
    let error_resilient_mode = br.read_bool()?;

    let mut intra_only = false;
    let mut reset_frame_context = 0u8;
    let refresh_frame_flags: u8;
    let mut bit_depth = None;
    let mut subsampling_x = None;
    let mut subsampling_y = None;
    let mut vp9_color_space = None;
    let mut color = ColorDesc::UNSPECIFIED;
    let mut pixel_layout = None;
    let width: Option<u32>;
    let height: Option<u32>;
    let mut render_width = None;
    let mut render_height = None;
    let mut ref_frame_idx = None;
    let mut ref_frame_sign_bias = None;
    let mut allow_high_precision_mv = false;
    let mut interpolation_filter = None;

    if is_keyframe {
        if br.read_bits(24)? != VP9_SYNC_CODE {
            return Err(Error::invalid_data("vp9/header", "invalid VP9 sync code"));
        }
        let (bd, sx, sy, cs, col, lay) = parse_color_config(&mut br, profile)?;
        bit_depth = Some(bd);
        subsampling_x = Some(sx);
        subsampling_y = Some(sy);
        vp9_color_space = Some(cs);
        color = col;
        pixel_layout = lay;

        refresh_frame_flags = 0xFF;
        width = Some(br.read_bits(16)? + 1);
        height = Some(br.read_bits(16)? + 1);
        if br.read_bool()? {
            render_width = Some(br.read_bits(16)? + 1);
            render_height = Some(br.read_bits(16)? + 1);
        }
    } else {
        intra_only = if !show_frame { br.read_bool()? } else { false };
        reset_frame_context = if error_resilient_mode {
            0
        } else {
            br.read_bits(2)? as u8
        };

        if intra_only {
            if br.read_bits(24)? != VP9_SYNC_CODE {
                return Err(Error::invalid_data(
                    "vp9/header",
                    "invalid VP9 intra-only sync code",
                ));
            }
            if profile >= 1 {
                let (bd, sx, sy, cs, col, lay) = parse_color_config(&mut br, profile)?;
                bit_depth = Some(bd);
                subsampling_x = Some(sx);
                subsampling_y = Some(sy);
                vp9_color_space = Some(cs);
                color = col;
                pixel_layout = lay;
            } else {
                // Matches FFmpeg vp9.c lines 598-605 for profile 0 intra-only frames.
                bit_depth = Some(8);
                subsampling_x = Some(true);
                subsampling_y = Some(true);
                vp9_color_space = Some(Vp9ColorSpace::Bt601);
                color = ColorDesc {
                    primaries: ColorPrimaries::Bt470bg,
                    transfer: TransferCharacteristic::Smpte170m,
                    matrix: MatrixCoefficients::Bt470bg,
                    range: ColorRange::Limited,
                    chroma_location: ChromaLocation::Unspecified,
                };
                pixel_layout = Some(PixelLayout::YUV420P);
            }
            refresh_frame_flags = br.read_bits(8)? as u8;
            width = Some(br.read_bits(16)? + 1);
            height = Some(br.read_bits(16)? + 1);
            if br.read_bool()? {
                render_width = Some(br.read_bits(16)? + 1);
                render_height = Some(br.read_bits(16)? + 1);
            }
        } else {
            refresh_frame_flags = br.read_bits(8)? as u8;
            let mut ridx = [0u8; 3];
            let mut sbias = [false; 3];
            for i in 0..3 {
                ridx[i] = br.read_bits(3)? as u8;
                sbias[i] = br.read_bool()? && !error_resilient_mode;
            }
            ref_frame_idx = Some(ridx);
            ref_frame_sign_bias = Some(sbias);

            // Frame size: either inherited from one of the 3 refs or coded inline.
            let found_ref = br.read_bool()? || br.read_bool()? || br.read_bool()?;
            if found_ref {
                width = None;
                height = None;
            } else {
                width = Some(br.read_bits(16)? + 1);
                height = Some(br.read_bits(16)? + 1);
            }
            if br.read_bool()? {
                render_width = Some(br.read_bits(16)? + 1);
                render_height = Some(br.read_bits(16)? + 1);
            }
            allow_high_precision_mv = br.read_bool()?;
            let filter = if br.read_bool()? {
                Vp9InterpFilter::Switchable
            } else {
                match br.read_bits(2)? {
                    0 => Vp9InterpFilter::EightTap,
                    1 => Vp9InterpFilter::EightTapSmooth,
                    2 => Vp9InterpFilter::EightTapSharp,
                    _ => Vp9InterpFilter::Bilinear,
                }
            };
            interpolation_filter = Some(filter);
        }
    }

    let refresh_frame_context = if error_resilient_mode {
        false
    } else if br.bits_left() > 0 {
        br.read_bool()?
    } else {
        false
    };
    let frame_parallel_decoding_mode = if error_resilient_mode {
        true
    } else if br.bits_left() > 0 {
        br.read_bool()?
    } else {
        false
    };
    let mut frame_context_idx = if br.bits_left() >= 2 {
        br.read_bits(2)? as u8
    } else {
        0
    };
    if is_keyframe || intra_only {
        // Matches FFmpeg vp9.c line 672: libvpx ignores frame_context_idx in keyframes/intra-only.
        frame_context_idx = 0;
    }

    Ok(Vp9FrameHeader {
        profile,
        show_existing_frame: None,
        is_keyframe,
        show_frame,
        error_resilient_mode,
        intra_only,
        reset_frame_context,
        refresh_frame_flags,
        bit_depth,
        subsampling_x,
        subsampling_y,
        vp9_color_space,
        color,
        pixel_layout,
        width,
        height,
        render_width,
        render_height,
        ref_frame_idx,
        ref_frame_sign_bias,
        allow_high_precision_mv,
        interpolation_filter,
        refresh_frame_context,
        frame_parallel_decoding_mode,
        frame_context_idx,
    })
}

/// Split a VP9 packet into its constituent sub-frame byte ranges if it is a VP9 superframe,
/// or return `[0..packet.len()]` for a regular single-frame packet.
///
/// Faithfully ports FFmpeg `n9.1` `libavcodec/bsf/vp9_superframe_split.c`.
pub fn parse_superframe(packet: &[u8]) -> Result<Vec<Range<usize>>> {
    if packet.is_empty() {
        return Ok(Vec::new());
    }
    let marker = packet[packet.len() - 1];
    if (marker & 0xE0) == 0xC0 {
        let length_size = 1usize + (((marker >> 3) & 0x03) as usize);
        let nb_frames = 1usize + ((marker & 0x07) as usize);
        let idx_size = 2 + nb_frames * length_size;

        if packet.len() >= idx_size && packet[packet.len() - idx_size] == marker {
            let idx_start = packet.len() - idx_size + 1;
            let payload_limit = packet.len() - idx_size;
            let mut ranges = Vec::with_capacity(nb_frames);
            let mut offset = 0usize;

            for i in 0..nb_frames {
                let mut frame_size = 0usize;
                for j in 0..length_size {
                    frame_size |= (packet[idx_start + i * length_size + j] as usize) << (j * 8);
                }
                if frame_size == 0 || offset.saturating_add(frame_size) > payload_limit {
                    return Err(Error::invalid_data(
                        "vp9/superframe",
                        "invalid frame size in VP9 superframe index",
                    ));
                }
                ranges.push(offset..offset + frame_size);
                offset += frame_size;
            }
            return Ok(ranges);
        }
    }

    Ok(std::iter::once(0..packet.len()).collect())
}

/// Pack `1..=8` non-empty VP9 frames into a standard VP9 superframe packet.
pub fn build_superframe(frames: &[&[u8]]) -> Result<Vec<u8>> {
    if frames.is_empty() || frames.len() > 8 {
        return Err(Error::invalid_argument(
            "vp9/superframe",
            "VP9 superframe must contain 1..=8 frames",
        ));
    }
    let mut max_len = 0usize;
    let mut total_payload = 0usize;
    for f in frames {
        if f.is_empty() {
            return Err(Error::invalid_argument(
                "vp9/superframe",
                "VP9 superframe cannot contain an empty sub-frame",
            ));
        }
        max_len = max_len.max(f.len());
        total_payload = total_payload
            .checked_add(f.len())
            .ok_or_else(|| Error::invalid_data("vp9/superframe", "superframe payload overflow"))?;
    }
    let length_size: usize = if max_len <= 0xFF {
        1
    } else if max_len <= 0xFFFF {
        2
    } else if max_len <= 0x00FF_FFFF {
        3
    } else if max_len <= u32::MAX as usize {
        4
    } else {
        return Err(Error::invalid_data(
            "vp9/superframe",
            "sub-frame exceeds 4 GiB limit",
        ));
    };

    let marker = 0xC0 | (((length_size - 1) as u8) << 3) | ((frames.len() - 1) as u8);
    let mut out = Vec::with_capacity(total_payload + 2 + frames.len() * length_size);
    for f in frames {
        out.extend_from_slice(f);
    }
    out.push(marker);
    for f in frames {
        let sz = f.len() as u32;
        for j in 0..length_size {
            out.push(((sz >> (j * 8)) & 0xFF) as u8);
        }
    }
    out.push(marker);
    Ok(out)
}

/// VP9 `vpcC` ISO BMFF FullBox configuration (`VPCodecConfigurationRecord`, version 1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VpccConfig {
    pub profile: u8,
    pub level: u8,
    pub bit_depth: u8,
    /// `0` = 4:2:0 vertical, `1` = 4:2:0 colocated, `2` = 4:2:2, `3` = 4:4:4.
    pub chroma_subsampling: u8,
    pub full_range: bool,
    pub color_primaries: ColorPrimaries,
    pub transfer_characteristics: TransferCharacteristic,
    pub matrix_coefficients: MatrixCoefficients,
    pub codec_init_data: Vec<u8>,
}

impl VpccConfig {
    /// Convert `vpcC` color fields into a [`ColorDesc`].
    pub fn color_desc(&self) -> ColorDesc {
        let chroma_location = match self.chroma_subsampling {
            0 => ChromaLocation::Left,
            1 => ChromaLocation::TopLeft,
            _ => ChromaLocation::Unspecified,
        };
        ColorDesc {
            primaries: self.color_primaries,
            transfer: self.transfer_characteristics,
            matrix: self.matrix_coefficients,
            range: if self.full_range {
                ColorRange::Full
            } else {
                ColorRange::Limited
            },
            chroma_location,
        }
    }
}

/// Parse an ISO BMFF FullBox `vpcC` record (12+ bytes starting with version=1, flags=0).
pub fn parse_vpcc(data: &[u8]) -> Result<VpccConfig> {
    if data.len() < 12 {
        return Err(Error::invalid_data(
            "vp9/vpcc",
            "vpcC record shorter than 12 bytes",
        ));
    }
    if data[0] != 1 {
        return Err(Error::invalid_data(
            "vp9/vpcc",
            "unsupported vpcC version (expected 1)",
        ));
    }
    let profile = data[4];
    let level = data[5];
    let bit_depth = (data[6] >> 4) & 0x0F;
    let chroma_subsampling = (data[6] >> 1) & 0x07;
    let full_range = (data[6] & 0x01) != 0;
    let color_primaries = ColorPrimaries::from_cicp(data[7]);
    let transfer_characteristics = TransferCharacteristic::from_cicp(data[8]);
    let matrix_coefficients = MatrixCoefficients::from_cicp(data[9]);
    let init_len = u16::from_be_bytes([data[10], data[11]]) as usize;
    if data.len() - 12 < init_len {
        return Err(Error::invalid_data(
            "vp9/vpcc",
            "truncated vpcC codecInitializationData",
        ));
    }
    Ok(VpccConfig {
        profile,
        level,
        bit_depth,
        chroma_subsampling,
        full_range,
        color_primaries,
        transfer_characteristics,
        matrix_coefficients,
        codec_init_data: data[12..12 + init_len].to_vec(),
    })
}

/// Serialize a [`VpccConfig`] into the 12-byte (+ init data) ISO BMFF FullBox `vpcC` payload
/// matching FFmpeg's `ff_isom_write_vpcc`.
pub fn build_vpcc(cfg: &VpccConfig) -> Result<Vec<u8>> {
    if cfg.codec_init_data.len() > u16::MAX as usize {
        return Err(Error::invalid_argument(
            "vp9/vpcc",
            "vpcC codec_init_data exceeds 65535 bytes",
        ));
    }
    let mut out = Vec::with_capacity(12 + cfg.codec_init_data.len());
    out.push(1); // version
    out.extend_from_slice(&[0, 0, 0]); // flags
    out.push(cfg.profile);
    out.push(cfg.level);
    out.push(
        ((cfg.bit_depth & 0x0F) << 4)
            | ((cfg.chroma_subsampling & 0x07) << 1)
            | u8::from(cfg.full_range),
    );
    out.push(cfg.color_primaries as u8);
    out.push(cfg.transfer_characteristics as u8);
    out.push(cfg.matrix_coefficients as u8);
    out.extend_from_slice(&(cfg.codec_init_data.len() as u16).to_be_bytes());
    out.extend_from_slice(&cfg.codec_init_data);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wedeo_core::BitWriter;

    fn build_vp9_keyframe(
        profile: u8,
        ten_or_twelve: Option<bool>,
        color_space: u8,
        full_range: bool,
        ss: Option<(bool, bool)>,
        width: u32,
        height: u32,
    ) -> Vec<u8> {
        let mut w = BitWriter::new();
        w.write_bits(0x2, 2); // frame_marker
        w.write_bit((profile & 1) != 0);
        w.write_bit((profile & 2) != 0);
        if profile == 3 {
            w.write_bit(false);
        }
        w.write_bit(false); // show_existing_frame = 0
        w.write_bit(false); // frame_type = 0 (KEY_FRAME)
        w.write_bit(true); // show_frame = 1
        w.write_bit(false); // error_resilient = 0
        w.write_bits(VP9_SYNC_CODE, 24);
        if let Some(twelve) = ten_or_twelve {
            w.write_bit(twelve);
        }
        w.write_bits(u32::from(color_space), 3);
        if color_space == 7 {
            w.write_bit(false); // reserved bit for RGB
        } else {
            w.write_bit(full_range);
            if let Some((sx, sy)) = ss {
                w.write_bit(sx);
                w.write_bit(sy);
                w.write_bit(false); // reserved bit
            }
        }
        w.write_bits(width - 1, 16);
        w.write_bits(height - 1, 16);
        w.write_bit(false); // render_and_frame_size_different
        w.write_bit(true); // refresh_frame_context
        w.write_bit(false); // frame_parallel_decoding_mode
        w.write_bits(0, 2); // frame_context_idx
        w.finish()
    }

    #[test]
    fn vp9_profile_1_gbrp_and_profile_2_10bit_hdr() {
        // Profile 1 RGB (gbrp, odd dimensions 177x99)
        let kf_rgb = build_vp9_keyframe(1, None, 7, true, None, 177, 99);
        let hdr = parse_frame_header(&kf_rgb).unwrap();
        assert_eq!(hdr.profile, 1);
        assert!(hdr.is_keyframe);
        assert!(hdr.show_frame);
        assert_eq!(hdr.bit_depth, Some(8));
        assert_eq!(hdr.vp9_color_space, Some(Vp9ColorSpace::Rgb));
        assert_eq!(hdr.color.matrix, MatrixCoefficients::Identity);
        assert_eq!(hdr.color.range, ColorRange::Full);
        assert_eq!(hdr.pixel_layout, Some(PixelLayout::GBRP));
        assert_eq!(hdr.width, Some(177));
        assert_eq!(hdr.height, Some(99));

        // Profile 2 10-bit BT.2020 4:2:0
        let kf_hdr = build_vp9_keyframe(2, Some(false), 5, false, None, 3840, 2160);
        let hdr2 = parse_frame_header(&kf_hdr).unwrap();
        assert_eq!(hdr2.profile, 2);
        assert_eq!(hdr2.bit_depth, Some(10));
        assert_eq!(hdr2.pixel_layout, Some(PixelLayout::YUV420P10LE));
        assert_eq!(hdr2.color.primaries, ColorPrimaries::Bt2020);
        assert_eq!(hdr2.color.matrix, MatrixCoefficients::Bt2020Ncl);
        assert_eq!(hdr2.color.range, ColorRange::Limited);

        // Profile 3 12-bit 4:2:2
        let kf_p3 = build_vp9_keyframe(3, Some(true), 2, false, Some((true, false)), 640, 360);
        let hdr3 = parse_frame_header(&kf_p3).unwrap();
        assert_eq!(hdr3.profile, 3);
        assert_eq!(hdr3.bit_depth, Some(12));
        assert_eq!(hdr3.subsampling_x, Some(true));
        assert_eq!(hdr3.subsampling_y, Some(false));
        assert_eq!(hdr3.pixel_layout.unwrap().depth, SampleDepth::U12);
        assert_eq!(hdr3.pixel_layout.unwrap().subsampling, Subsampling::Cs422);
    }

    #[test]
    fn vp9_superframe_roundtrip_and_single_frame_passthrough() {
        let f0 = build_vp9_keyframe(0, None, 2, false, None, 320, 240);
        // Show-existing-frame sub-frame (1 byte)
        let mut w = BitWriter::new();
        w.write_bits(0x2, 2);
        w.write_bits(0, 2); // profile 0
        w.write_bit(true); // show_existing_frame = 1
        w.write_bits(3, 3); // idx 3
        let f1 = w.finish();

        let sf = build_superframe(&[&f0, &f1]).unwrap();
        let ranges = parse_superframe(&sf).unwrap();
        assert_eq!(ranges.len(), 2);
        assert_eq!(&sf[ranges[0].clone()], f0.as_slice());
        assert_eq!(&sf[ranges[1].clone()], f1.as_slice());

        let hdr1 = parse_frame_header(&sf[ranges[1].clone()]).unwrap();
        assert_eq!(hdr1.show_existing_frame, Some(3));

        // Non-superframe packet passes through untouched
        let single = parse_superframe(&f0).unwrap();
        assert_eq!(single, vec![0..f0.len()]);
    }

    #[test]
    fn vpcc_roundtrip() {
        let cfg = VpccConfig {
            profile: 2,
            level: 50,
            bit_depth: 10,
            chroma_subsampling: 1,
            full_range: false,
            color_primaries: ColorPrimaries::Bt2020,
            transfer_characteristics: TransferCharacteristic::Pq,
            matrix_coefficients: MatrixCoefficients::Bt2020Ncl,
            codec_init_data: Vec::new(),
        };
        let bytes = build_vpcc(&cfg).unwrap();
        assert_eq!(bytes.len(), 12);
        let parsed = parse_vpcc(&bytes).unwrap();
        assert_eq!(parsed, cfg);
        assert_eq!(parsed.color_desc().chroma_location, ChromaLocation::TopLeft);
    }

    #[test]
    fn vp9_intra_only_inter_and_invalid_profile_rejections() {
        // Profile 0 intra-only frame (invisible, sync code, no color config => BT.601 4:2:0 8-bit)
        let mut w = BitWriter::new();
        w.write_bits(0x2, 2); // marker
        w.write_bits(0, 2); // profile 0
        w.write_bit(false); // show_existing = 0
        w.write_bit(true); // frame_type = 1 (inter/intra-only)
        w.write_bit(false); // show_frame = 0 (invisible)
        w.write_bit(false); // error_resilient = 0
        w.write_bit(true); // intra_only = 1
        w.write_bits(0, 2); // reset_frame_context
        w.write_bits(VP9_SYNC_CODE, 24);
        w.write_bits(0x01, 8); // refresh_frame_flags
        w.write_bits(639, 16); // width = 640
        w.write_bits(359, 16); // height = 360
        w.write_bit(false); // render_size_different
        w.write_bit(false); // refresh_ctx
        w.write_bit(false); // parallel_mode
        w.write_bits(1, 2); // frame_ctx_idx (ignored on intra_only => 0)
        let intra_only_bytes = w.finish();
        let hdr = parse_frame_header(&intra_only_bytes).unwrap();
        assert!(!hdr.is_keyframe);
        assert!(hdr.intra_only);
        assert!(!hdr.show_frame);
        assert_eq!(hdr.width, Some(640));
        assert_eq!(hdr.height, Some(360));
        assert_eq!(hdr.pixel_layout, Some(PixelLayout::YUV420P));
        assert_eq!(hdr.frame_context_idx, 0);

        // Inter frame inheriting dimensions from ref[0]
        let mut wi = BitWriter::new();
        wi.write_bits(0x2, 2);
        wi.write_bits(0, 2); // profile 0
        wi.write_bit(false); // show_existing
        wi.write_bit(true); // inter
        wi.write_bit(true); // show_frame
        wi.write_bit(false); // error_resilient
        wi.write_bits(0, 2); // reset_ctx
        wi.write_bits(0x02, 8); // refresh_frame_flags
        for idx in [0u32, 1, 2] {
            wi.write_bits(idx, 3);
            wi.write_bit(false); // sign_bias
        }
        wi.write_bit(true); // found_ref[0] => inherit size
        wi.write_bit(false); // render_size_different
        wi.write_bit(true); // allow_high_precision_mv
        wi.write_bit(true); // is_filter_switchable
        wi.write_bit(true); // refresh_ctx
        wi.write_bit(false); // parallel_mode
        wi.write_bits(2, 2); // frame_ctx_idx = 2
        let inter = parse_frame_header(&wi.finish()).unwrap();
        assert!(!inter.is_keyframe);
        assert!(!inter.intra_only);
        assert_eq!(inter.width, None);
        assert_eq!(inter.ref_frame_idx, Some([0, 1, 2]));
        assert_eq!(
            inter.interpolation_filter,
            Some(Vp9InterpFilter::Switchable)
        );
        assert_eq!(inter.frame_context_idx, 2);

        // Invalid: RGB in Profile 0 must fail
        let bad_rgb_p0 = build_vp9_keyframe(0, None, 7, true, None, 64, 64);
        assert!(parse_frame_header(&bad_rgb_p0).is_err());

        // Invalid: 4:2:0 in Profile 1 must fail
        let bad_420_p1 = build_vp9_keyframe(1, None, 2, false, Some((true, true)), 64, 64);
        assert!(parse_frame_header(&bad_420_p1).is_err());
    }
}
