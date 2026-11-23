//! Port contracts, packet framing descriptors, auto-adapter conversion notes,
//! graph compilation options, and human/JSON `ExecutionPlan` inspection.

use std::fmt;

use wedeo_audio_ops::AudioContract;
use wedeo_core::{
    AlphaMode, CodecId, ColorModel, ColorPrimaries, ColorRange, GpuDeviceId, MemoryDomain,
    SampleFormat, SubtitleFormat,
};
use wedeo_video_ops::VideoContract;

/// Flat-indexed node identifier inside a `GraphBuilder` or `CompiledGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct NodeId(pub u32);

impl NodeId {
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for NodeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "n{}", self.0)
    }
}

/// Flat-indexed directed edge identifier inside a `GraphBuilder` or `CompiledGraph`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct EdgeId(pub u32);

impl EdgeId {
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

impl fmt::Display for EdgeId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// Compressed bitstream packet framing convention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum PacketFraming {
    /// ITU-T H.264 / HEVC Annex B start-code (`00 00 01` / `00 00 00 01`) byte stream.
    AnnexB,
    /// ISO/IEC 14496-15 (`avcC` / `hvcC`) length-prefixed NAL framing with `1..=4` byte prefix length.
    Avcc(u8),
    /// AV1 Open Bitstream Unit framing.
    Obu,
    /// AAC Audio Data Transport Stream framing (7/9-byte syncword header per frame).
    Adts,
    /// Raw unframed codec access unit payload (e.g. VP9, Opus, FLAC, PCM, MP4 raw AAC).
    #[default]
    Raw,
}

impl PacketFraming {
    pub const fn name(self) -> &'static str {
        match self {
            Self::AnnexB => "annexb",
            Self::Avcc(_) => "avcc",
            Self::Obu => "obu",
            Self::Adts => "adts",
            Self::Raw => "raw",
        }
    }
}

impl fmt::Display for PacketFraming {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::AnnexB => f.write_str("annexb"),
            Self::Avcc(n) => write!(f, "avcc({n})"),
            Self::Obu => f.write_str("obu"),
            Self::Adts => f.write_str("adts"),
            Self::Raw => f.write_str("raw"),
        }
    }
}

/// How codec configuration parameter sets (`SPS`/`PPS`, `AudioSpecificConfig`, etc.) are carried.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum ExtradataMode {
    /// No separate extradata required or unspecified.
    #[default]
    None,
    /// Parameter sets are repeated in-band ahead of keyframes (e.g. Annex B broadcast/raw streams).
    InBand,
    /// Parameter sets are carried out-of-band in container headers (`avcC`, `hvcC`, `av1C`, `esds`).
    OutOfBand,
}

impl ExtradataMode {
    pub const fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::InBand => "inband",
            Self::OutOfBand => "out_of_band",
        }
    }
}

/// Contract for a compressed packet port.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PacketContract {
    pub codec_id: CodecId,
    pub framing: PacketFraming,
    pub extradata_mode: ExtradataMode,
    /// Optional out-of-band extradata record (`avcC`, `hvcC`, etc.) used when auto-bridging to Annex B.
    pub extradata: Vec<u8>,
}

impl PacketContract {
    pub const fn new(codec_id: CodecId, framing: PacketFraming) -> Self {
        let extradata_mode = match framing {
            PacketFraming::AnnexB | PacketFraming::Adts => ExtradataMode::InBand,
            PacketFraming::Avcc(_) => ExtradataMode::OutOfBand,
            PacketFraming::Obu | PacketFraming::Raw => ExtradataMode::None,
        };
        Self {
            codec_id,
            framing,
            extradata_mode,
            extradata: Vec::new(),
        }
    }

    pub const fn with_extradata_mode(mut self, mode: ExtradataMode) -> Self {
        self.extradata_mode = mode;
        self
    }

    pub fn with_extradata(mut self, extradata: Vec<u8>) -> Self {
        self.extradata = extradata;
        self
    }
}

/// Typed media contract on a node input or output port.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PortContract {
    Packet(PacketContract),
    Video(VideoContract),
    Audio(AudioContract),
    Subtitle(SubtitleFormat),
    TimedData(&'static str),
}

impl PortContract {
    pub fn kind_name(&self) -> &'static str {
        match self {
            Self::Packet(_) => "packet",
            Self::Video(_) => "video",
            Self::Audio(_) => "audio",
            Self::Subtitle(_) => "subtitle",
            Self::TimedData(_) => "timed_data",
        }
    }

    pub fn memory_domain(&self) -> MemoryDomain {
        match self {
            Self::Video(v) => v.domain,
            Self::Packet(_) | Self::Audio(_) | Self::Subtitle(_) | Self::TimedData(_) => {
                MemoryDomain::Cpu
            }
        }
    }

    pub fn summary(&self) -> String {
        match self {
            Self::Packet(p) => format!("packet({}, {})", p.codec_id.name(), p.framing),
            Self::Video(v) => {
                let dom = match v.domain {
                    MemoryDomain::Cpu => "cpu".to_string(),
                    MemoryDomain::Gpu(GpuDeviceId(id)) => format!("gpu:{id}"),
                };
                format!(
                    "video({}x{}, {}, {}/{}/{}, {})",
                    v.width,
                    v.height,
                    v.layout.name(),
                    v.color.primaries.name(),
                    v.color.transfer.name(),
                    v.color.matrix.name(),
                    dom
                )
            }
            Self::Audio(a) => match a.frame_size {
                Some(fs) => format!(
                    "audio({}, {}Hz, {}, frame={fs})",
                    a.format.name(),
                    a.sample_rate,
                    a.channel_layout
                ),
                None => format!(
                    "audio({}, {}Hz, {})",
                    a.format.name(),
                    a.sample_rate,
                    a.channel_layout
                ),
            },
            Self::Subtitle(fmt) => format!("subtitle({fmt:?})"),
            Self::TimedData(tag) => format!("timed_data({tag})"),
        }
    }
}

impl fmt::Display for PortContract {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.summary())
    }
}

/// Concise 1-line explanation emitted when the auto-adapter inserts an implicit bridge
/// (especially highlighting lossy conversions such as HDR→SDR tone-mapping, downscaling,
/// bit-depth quantization, chroma subsampling, or channel downmixing).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConversionNote {
    pub from_node: NodeId,
    pub to_node: NodeId,
    pub bridge_node: NodeId,
    pub is_lossy: bool,
    pub summary: String,
    pub reasons: Vec<String>,
}

impl ConversionNote {
    pub fn new(
        from_node: NodeId,
        to_node: NodeId,
        bridge_node: NodeId,
        is_lossy: bool,
        summary: impl Into<String>,
        reasons: Vec<String>,
    ) -> Self {
        Self {
            from_node,
            to_node,
            bridge_node,
            is_lossy,
            summary: summary.into(),
            reasons,
        }
    }

    /// Concise single-line human-readable representation.
    pub fn one_line(&self) -> String {
        let kind = if self.is_lossy { "lossy" } else { "lossless" };
        if self.reasons.is_empty() {
            format!(
                "[{kind}] {} -> {} via {}: {}",
                self.from_node, self.to_node, self.bridge_node, self.summary
            )
        } else {
            format!(
                "[{kind}] {} -> {} via {}: {} ({})",
                self.from_node,
                self.to_node,
                self.bridge_node,
                self.summary,
                self.reasons.join(", ")
            )
        }
    }
}

impl fmt::Display for ConversionNote {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.one_line())
    }
}

/// Inspect a `VideoContract` pair `(src, dst)` and return the list of lossy conversion reasons,
/// if any (empty vector means the conversion is mathematically or structurally lossless).
pub fn classify_video_conversion_loss(src: &VideoContract, dst: &VideoContract) -> Vec<String> {
    let mut reasons = Vec::new();

    // 1. HDR -> SDR transfer tone-mapping
    if src.color.transfer.is_hdr() && !dst.color.transfer.is_hdr() {
        reasons.push(format!(
            "HDR->SDR tone-map ({} -> {})",
            src.color.transfer.name(),
            dst.color.transfer.name()
        ));
    }

    // 2. Wide gamut -> narrower gamut (e.g. BT.2020 -> BT.709 / SMPTE 170M)
    if src.color.primaries != dst.color.primaries
        && src.color.primaries != ColorPrimaries::Unspecified
        && dst.color.primaries != ColorPrimaries::Unspecified
        && gamut_rank(dst.color.primaries) < gamut_rank(src.color.primaries)
    {
        reasons.push(format!(
            "color gamut narrowing ({} -> {})",
            src.color.primaries.name(),
            dst.color.primaries.name()
        ));
    }

    // 3. Spatial downscaling
    let (src_w, src_h) = src
        .effective_dimensions()
        .unwrap_or((src.width, src.height));
    let (dst_w, dst_h) = dst
        .effective_dimensions()
        .unwrap_or((dst.width, dst.height));
    if dst_w < src_w || dst_h < src_h {
        reasons.push(format!("downscale ({src_w}x{src_h} -> {dst_w}x{dst_h})"));
    }

    // 4. Bit depth reduction
    let src_bits = src.layout.depth.bits();
    let dst_bits = dst.layout.depth.bits();
    if dst_bits < src_bits {
        reasons.push(format!(
            "bit-depth reduction ({src_bits}-bit -> {dst_bits}-bit)"
        ));
    }

    // 5. Color model reduction (Color -> Grayscale)
    if !matches!(src.layout.model, ColorModel::Gray) && matches!(dst.layout.model, ColorModel::Gray)
    {
        reasons.push("color->grayscale luma projection".to_string());
    } else if subsampling_rank(dst.layout.subsampling) < subsampling_rank(src.layout.subsampling) {
        // 6. Chroma subsampling reduction (e.g. 4:4:4 -> 4:2:0)
        reasons.push(format!(
            "chroma downsampling ({:?} -> {:?})",
            src.layout.subsampling, dst.layout.subsampling
        ));
    }

    // 7. Alpha channel discard
    if !matches!(src.layout.alpha, AlphaMode::None) && matches!(dst.layout.alpha, AlphaMode::None) {
        reasons.push("alpha channel discarded".to_string());
    }

    // 8. Full -> Limited range compression at same/lower bit depth
    if src.color.range == ColorRange::Full
        && dst.color.range == ColorRange::Limited
        && dst_bits <= src_bits
    {
        reasons.push("full->limited range quantization".to_string());
    }

    reasons
}

const fn gamut_rank(p: ColorPrimaries) -> u8 {
    match p {
        ColorPrimaries::Bt2020 | ColorPrimaries::Smpte428 => 3,
        ColorPrimaries::Smpte431 | ColorPrimaries::Smpte432 => 2,
        ColorPrimaries::Bt709
        | ColorPrimaries::Smpte170m
        | ColorPrimaries::Smpte240m
        | ColorPrimaries::Bt470m
        | ColorPrimaries::Bt470bg
        | ColorPrimaries::Ebu3213
        | ColorPrimaries::Film => 1,
        ColorPrimaries::Reserved0 | ColorPrimaries::Unspecified | ColorPrimaries::Reserved => 0,
    }
}

const fn subsampling_rank(s: wedeo_core::Subsampling) -> u8 {
    use wedeo_core::Subsampling;
    match s {
        Subsampling::Cs444 => 4,
        Subsampling::Cs422 => 3,
        Subsampling::Cs420 | Subsampling::Cs411 => 2,
        Subsampling::Cs410 => 1,
        Subsampling::Cs400 => 0,
    }
}

/// Inspect an `AudioContract` pair `(src, dst)` and return the list of lossy conversion reasons,
/// if any (planar/packed layout changes, reframing, and up-precision conversions are lossless).
pub fn classify_audio_conversion_loss(src: &AudioContract, dst: &AudioContract) -> Vec<String> {
    let mut reasons = Vec::new();

    if dst.channel_layout.nb_channels < src.channel_layout.nb_channels {
        reasons.push(format!(
            "channel downmix ({} -> {})",
            src.channel_layout, dst.channel_layout
        ));
    }

    if dst.sample_rate < src.sample_rate {
        reasons.push(format!(
            "audio downsample ({}Hz -> {}Hz)",
            src.sample_rate, dst.sample_rate
        ));
    } else if dst.sample_rate != src.sample_rate {
        reasons.push(format!(
            "polyphase resample ({}Hz -> {}Hz)",
            src.sample_rate, dst.sample_rate
        ));
    }

    if audio_precision_rank(dst.format) < audio_precision_rank(src.format) {
        reasons.push(format!(
            "sample quantization ({} -> {})",
            src.format.name(),
            dst.format.name()
        ));
    }

    reasons
}

const fn audio_precision_rank(fmt: SampleFormat) -> u8 {
    match fmt.packed() {
        SampleFormat::U8 => 1,
        SampleFormat::S16 => 2,
        SampleFormat::F32 => 3,
        SampleFormat::S32 => 4,
        SampleFormat::F64 | SampleFormat::S64 => 5,
        _ => 0,
    }
}

/// Compilation options controlling auto-adapter strictness, track/metadata preservation,
/// and GPU residency domain placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompileOpts {
    /// When `true`, reject any implicit lossy auto-adapter bridge with an error.
    pub strict: bool,
    /// Preserve all tracks, chapters, attachments, HDR side data, and display matrices by default.
    pub preserve_all: bool,
    /// Optional preferred GPU device ID for flexible video operations (`SwsOpList`).
    pub prefer_gpu: Option<GpuDeviceId>,
}

impl Default for CompileOpts {
    fn default() -> Self {
        Self {
            strict: false,
            preserve_all: true,
            prefer_gpu: None,
        }
    }
}

impl CompileOpts {
    pub const fn new() -> Self {
        Self {
            strict: false,
            preserve_all: true,
            prefer_gpu: None,
        }
    }

    pub const fn with_strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }

    pub const fn with_preserve_all(mut self, preserve_all: bool) -> Self {
        self.preserve_all = preserve_all;
        self
    }

    pub const fn with_prefer_gpu(mut self, prefer_gpu: Option<GpuDeviceId>) -> Self {
        self.prefer_gpu = prefer_gpu;
        self
    }
}

/// Summary of one node in a compiled [`ExecutionPlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedNode {
    pub id: NodeId,
    pub name: String,
    pub domain: MemoryDomain,
    pub is_auto_bridge: bool,
    pub inputs: Vec<PortContract>,
    pub outputs: Vec<PortContract>,
}

/// Summary of one directed edge in a compiled [`ExecutionPlan`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedEdge {
    pub id: EdgeId,
    pub from_node: NodeId,
    pub from_port: usize,
    pub to_node: NodeId,
    pub to_port: usize,
    pub contract: PortContract,
}

/// Compiled DAG execution plan supporting human-readable `explain()` and clean JSON serialization.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    pub order: Vec<NodeId>,
    pub nodes: Vec<PlannedNode>,
    pub edges: Vec<PlannedEdge>,
    pub auto_bridges: Vec<NodeId>,
    pub conversion_notes: Vec<ConversionNote>,
    pub gpu_warnings: Vec<String>,
    pub strict: bool,
    pub preserve_all: bool,
}

impl ExecutionPlan {
    /// Human-readable multi-line explanation of the compiled graph, auto-inserted bridges,
    /// lossy conversion notes, and GPU residency warnings.
    pub fn explain(&self) -> String {
        let mut out = String::new();
        out.push_str(&format!(
            "ExecutionPlan (nodes={}, edges={}, bridges={}, strict={}, preserve_all={})\n",
            self.nodes.len(),
            self.edges.len(),
            self.auto_bridges.len(),
            self.strict,
            self.preserve_all
        ));

        out.push_str("  Topological Order: [");
        for (i, id) in self.order.iter().enumerate() {
            if i > 0 {
                out.push_str(" -> ");
            }
            out.push_str(&id.to_string());
        }
        out.push_str("]\n");

        out.push_str("  Nodes:\n");
        for &node_id in &self.order {
            if let Some(node) = self.nodes.iter().find(|n| n.id == node_id) {
                let tag = if node.is_auto_bridge { " [auto]" } else { "" };
                let dom = match node.domain {
                    MemoryDomain::Cpu => "cpu".to_string(),
                    MemoryDomain::Gpu(GpuDeviceId(id)) => format!("gpu:{id}"),
                };
                out.push_str(&format!(
                    "    {} ({}{}, domain={})\n",
                    node.id, node.name, tag, dom
                ));
            }
        }

        if !self.edges.is_empty() {
            out.push_str("  Edges:\n");
            for edge in &self.edges {
                out.push_str(&format!(
                    "    {}: {}:{} -> {}:{} [{}]\n",
                    edge.id,
                    edge.from_node,
                    edge.from_port,
                    edge.to_node,
                    edge.to_port,
                    edge.contract.summary()
                ));
            }
        }

        if !self.conversion_notes.is_empty() {
            out.push_str("  Conversion Notes:\n");
            for note in &self.conversion_notes {
                out.push_str(&format!("    - {}\n", note.one_line()));
            }
        }

        if !self.gpu_warnings.is_empty() {
            out.push_str("  GPU Residency Warnings:\n");
            for warn in &self.gpu_warnings {
                out.push_str(&format!("    ! {warn}\n"));
            }
        }

        out
    }

    /// Clean, deterministic JSON representation of this [`ExecutionPlan`] (zero external crate deps).
    pub fn to_json(&self) -> String {
        let mut s = String::with_capacity(512);
        s.push_str("{\n");
        s.push_str(&format!("  \"strict\": {},\n", self.strict));
        s.push_str(&format!("  \"preserve_all\": {},\n", self.preserve_all));

        s.push_str("  \"order\": [");
        for (i, id) in self.order.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push_str(&id.0.to_string());
        }
        s.push_str("],\n");

        s.push_str("  \"nodes\": [\n");
        for (i, n) in self.nodes.iter().enumerate() {
            let dom = match n.domain {
                MemoryDomain::Cpu => "cpu".to_string(),
                MemoryDomain::Gpu(GpuDeviceId(id)) => format!("gpu:{id}"),
            };
            s.push_str(&format!(
                "    {{\"id\": {}, \"name\": \"{}\", \"domain\": \"{}\", \"auto_bridge\": {}}}",
                n.id.0,
                escape_json(&n.name),
                dom,
                n.is_auto_bridge
            ));
            if i + 1 < self.nodes.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");

        s.push_str("  \"edges\": [\n");
        for (i, e) in self.edges.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"id\": {}, \"from_node\": {}, \"from_port\": {}, \"to_node\": {}, \"to_port\": {}, \"contract\": \"{}\"}}",
                e.id.0,
                e.from_node.0,
                e.from_port,
                e.to_node.0,
                e.to_port,
                escape_json(&e.contract.summary())
            ));
            if i + 1 < self.edges.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");

        s.push_str("  \"conversion_notes\": [\n");
        for (i, note) in self.conversion_notes.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"from_node\": {}, \"to_node\": {}, \"bridge_node\": {}, \"is_lossy\": {}, \"note\": \"{}\"}}",
                note.from_node.0,
                note.to_node.0,
                note.bridge_node.0,
                note.is_lossy,
                escape_json(&note.one_line())
            ));
            if i + 1 < self.conversion_notes.len() {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");

        s.push_str("  \"gpu_warnings\": [");
        for (i, w) in self.gpu_warnings.iter().enumerate() {
            if i > 0 {
                s.push_str(", ");
            }
            s.push('"');
            s.push_str(&escape_json(w));
            s.push('"');
        }
        s.push_str("]\n");
        s.push('}');
        s
    }
}

impl fmt::Display for ExecutionPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.explain())
    }
}

fn escape_json(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => out.push_str(&format!("\\u{:04x}", c as u32)),
            c => out.push(c),
        }
    }
    out
}
