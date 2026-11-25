//! Flat-indexed synchronous Data-Oriented Design (DOD) media graph builder, auto-adapter
//! compiler, GPU residency domain placement pass, and deterministic execution engine.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use wedeo_audio_ops::{AudioContract, AudioOpList, CompiledAudioOps};
use wedeo_core::{
    AudioBlock, BorderedPlane, Buffer, BufferPool, CodecId, ContainerHeader, Decoder, Demuxer,
    EncodeHint, Encoder, Error, GpuDeviceId, GpuSurface, MediaUnit, MemoryDomain, Muxer, Packet,
    PacketFlags, PixelLayout, RecvStatus, Result, SendStatus, StreamHealth, StreamParams, TrackId,
    VideoFrame, VideoStorage,
};
use wedeo_video_ops::{
    CompiledSws, CpuSwsBackend, SwsBackend, SwsOpList, SwsOptions, VideoContract,
};

use crate::contract::{
    CompileOpts, ConversionNote, EdgeId, ExecutionPlan, ExtradataMode, NodeId, PacketContract,
    PacketFraming, PlannedEdge, PlannedNode, PortContract, classify_audio_conversion_loss,
    classify_video_conversion_loss,
};
use crate::filter::{AudioFilter, VideoFilter, WgslFilter};

/// Unified item flowing along edges in a [`CompiledGraph`].
#[derive(Debug, Clone)]
pub enum GraphItem {
    Packet(Packet),
    Unit(MediaUnit),
}

impl GraphItem {
    pub fn into_packet(self) -> Option<Packet> {
        match self {
            Self::Packet(p) => Some(p),
            Self::Unit(_) => None,
        }
    }

    pub fn into_unit(self) -> Option<MediaUnit> {
        match self {
            Self::Unit(u) => Some(u),
            Self::Packet(_) => None,
        }
    }

    pub fn into_video(self) -> Option<VideoFrame> {
        match self {
            Self::Unit(MediaUnit::Video(v)) => Some(v),
            _ => None,
        }
    }

    pub fn into_audio(self) -> Option<AudioBlock> {
        match self {
            Self::Unit(MediaUnit::Audio(a)) => Some(a),
            _ => None,
        }
    }
}

/// Summary report returned by [`CompiledGraph::run`].
#[derive(Debug, Clone)]
pub struct RunReport {
    /// Packet emissions from all graph stages, including intermediate conversions.
    pub packets_processed: u64,
    pub units_processed: u64,
    pub health: StreamHealth,
    pub plan: ExecutionPlan,
}

/// Core synchronous node execution contract in a flat-indexed media DAG.
pub trait GraphNode: Send {
    fn name(&self) -> &str;
    fn input_contracts(&self) -> &[PortContract];
    fn output_contracts(&self) -> &[PortContract];

    /// Process a single [`GraphItem`] arriving on `input_port`, returning zero or more
    /// `(output_port, GraphItem)` pairs to forward downstream.
    fn process(
        &mut self,
        input_port: usize,
        item: GraphItem,
        pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>>;

    /// Flush internal state at end-of-stream, returning any remaining `(output_port, GraphItem)` pairs.
    fn flush(&mut self, pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>>;

    /// Optional stream health ledger maintained by this node.
    fn health(&self) -> Option<&StreamHealth> {
        None
    }

    /// Pull the next item from a source node (`input_contracts().is_empty()`).
    fn pull(&mut self, _pool: &BufferPool) -> Result<RecvStatus<(usize, GraphItem)>> {
        Ok(RecvStatus::Eof)
    }

    /// Finalize node after all items have been processed and flushed (e.g. `Muxer::finalize`).
    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Primary memory residency domain of this node (`Cpu` or `Gpu(device_id)`).
    fn domain(&self) -> MemoryDomain {
        for c in self.output_contracts().iter().chain(self.input_contracts()) {
            if let PortContract::Video(vc) = c {
                return vc.domain;
            }
        }
        MemoryDomain::Cpu
    }

    /// Whether the GPU residency domain placement pass may relocate this node between `Cpu` and `Gpu`.
    fn is_domain_flexible(&self) -> bool {
        false
    }

    /// Update this node's memory domain during the domain placement pass.
    fn set_memory_domain(&mut self, _domain: MemoryDomain) -> Result<()> {
        Ok(())
    }

    /// Access collected items if this node is a collector [`SinkNode`].
    fn sink_items(&self) -> Option<&[GraphItem]> {
        None
    }

    /// Drain collected items if this node is a collector [`SinkNode`].
    fn take_sink_items(&mut self) -> Option<Vec<GraphItem>> {
        None
    }

    /// Downcast reference for inspecting node state in tests and callers.
    fn as_any(&self) -> &dyn Any;

    /// Mutable downcast reference for inspecting node state in tests and callers.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

// ============================================================================
// Headless CPU-backed GpuSurface for zero-wgpu domain transfer & WGSL testing
// ============================================================================

/// Lightweight CPU-backed [`GpuSurface`] allowing GPU residency domain placement,
/// `Cpu ↔ Gpu` transfers, and [`WgslFilter`] nodes to execute end-to-end in headless builds.
#[derive(Debug, Clone)]
pub struct CpuBackedGpuSurface {
    pub device_id: GpuDeviceId,
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub planes: Vec<BorderedPlane>,
}

impl GpuSurface for CpuBackedGpuSurface {
    fn device_id(&self) -> GpuDeviceId {
        self.device_id
    }

    fn layout(&self) -> PixelLayout {
        self.layout
    }

    fn width(&self) -> u32 {
        self.width
    }

    fn height(&self) -> u32 {
        self.height
    }

    fn readback_planes(&self) -> Result<Vec<BorderedPlane>> {
        Ok(self.planes.clone())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Place CPU pixels in a CPU-backed GPU domain for graph emulation. An external
/// surface without readable pixels retains its original device and contents.
/// Graph execution uses the fallible transfer below to report that case.
pub fn upload_frame_to_gpu(mut frame: VideoFrame, device_id: GpuDeviceId) -> VideoFrame {
    let _ = place_in_gpu_domain(&mut frame, device_id);
    frame
}

fn place_in_gpu_domain(frame: &mut VideoFrame, device_id: GpuDeviceId) -> Result<()> {
    let planes = match &mut frame.storage {
        VideoStorage::Gpu(surface) if surface.device_id() == device_id => return Ok(()),
        VideoStorage::Gpu(surface) => surface.readback_planes()?,
        VideoStorage::Cpu(planes) => std::mem::take(planes),
    };
    frame.storage = VideoStorage::Gpu(Arc::new(CpuBackedGpuSurface {
        device_id,
        width: frame.width,
        height: frame.height,
        layout: frame.layout,
        planes,
    }));
    Ok(())
}

/// Download pixels using the GPU surface's readback provider.
pub fn download_frame_to_cpu(mut frame: VideoFrame) -> Result<VideoFrame> {
    if let VideoStorage::Gpu(surface) = &frame.storage {
        frame.storage = VideoStorage::Cpu(surface.readback_planes()?);
    }
    Ok(frame)
}

// ============================================================================
// GraphNode Wrappers
// ============================================================================

/// Source node wrapping a [`Demuxer`] with one output port per container track.
pub struct DemuxerNode {
    name: String,
    demuxer: Box<dyn Demuxer>,
    track_ids: Vec<TrackId>,
    outputs: Vec<PortContract>,
}

impl DemuxerNode {
    pub fn new(demuxer: Box<dyn Demuxer>) -> Self {
        let hdr = demuxer.header();
        let name = format!("demuxer({})", hdr.format_name);
        let mut track_ids = Vec::with_capacity(hdr.tracks.len());
        let mut outputs = Vec::with_capacity(hdr.tracks.len());

        for t in &hdr.tracks {
            track_ids.push(t.id);
            let extradata = match &t.params {
                StreamParams::Video(v) => v.extradata.clone(),
                StreamParams::Audio(a) => a.extradata.clone(),
                StreamParams::Subtitle(s) => s.header_script.clone(),
                StreamParams::Data(d) => d.extradata.clone(),
            };
            let framing = infer_packet_framing(t.codec_id, &extradata);
            let contract = PacketContract::new(t.codec_id, framing).with_extradata(extradata);
            outputs.push(PortContract::Packet(contract));
        }

        Self {
            name,
            demuxer,
            track_ids,
            outputs,
        }
    }
}

fn infer_packet_framing(codec_id: CodecId, extradata: &[u8]) -> PacketFraming {
    match codec_id {
        CodecId::H264 => {
            if extradata.len() >= 7 && extradata[0] == 1 {
                let len_size = (extradata[4] & 0x03) + 1;
                PacketFraming::Avcc(len_size)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Hevc => {
            if extradata.len() >= 23 && extradata[0] == 1 {
                let len_size = (extradata[21] & 0x03) + 1;
                PacketFraming::Avcc(len_size)
            } else {
                PacketFraming::AnnexB
            }
        }
        CodecId::Av1 => PacketFraming::Obu,
        CodecId::Aac => {
            if extradata.is_empty() {
                PacketFraming::Adts
            } else {
                PacketFraming::Raw
            }
        }
        _ => PacketFraming::Raw,
    }
}

impl GraphNode for DemuxerNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &[]
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        _item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        Err(Error::invalid_argument(
            "demuxer_node",
            "DemuxerNode has no input ports",
        ))
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn health(&self) -> Option<&StreamHealth> {
        Some(self.demuxer.health())
    }

    fn pull(&mut self, _pool: &BufferPool) -> Result<RecvStatus<(usize, GraphItem)>> {
        loop {
            match self.demuxer.read_packet()? {
                RecvStatus::Ready(pkt) => {
                    if let Some(port) = self.track_ids.iter().position(|&id| id == pkt.track_id) {
                        return Ok(RecvStatus::Ready((port, GraphItem::Packet(pkt))));
                    }
                }
                RecvStatus::NeedMore => return Ok(RecvStatus::NeedMore),
                RecvStatus::Eof => return Ok(RecvStatus::Eof),
            }
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Sink node wrapping a [`Muxer`].
pub struct MuxerNode {
    name: String,
    muxer: Option<Box<dyn Muxer>>,
    header: ContainerHeader,
    header_written: bool,
    packets_written: u64,
    inputs: Vec<PortContract>,
}

impl MuxerNode {
    pub fn new(muxer: Box<dyn Muxer>, header: ContainerHeader, inputs: Vec<PortContract>) -> Self {
        let name = format!("muxer({})", header.format_name);
        Self {
            name,
            muxer: Some(muxer),
            header,
            header_written: false,
            packets_written: 0,
            inputs,
        }
    }
}

impl GraphNode for MuxerNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &[]
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Packet(pkt) = item else {
            return Err(Error::invalid_argument(
                "muxer_node",
                "MuxerNode expects GraphItem::Packet inputs",
            ));
        };
        let Some(muxer) = self.muxer.as_mut() else {
            return Err(Error::internal("muxer_node", "Muxer already finalized"));
        };
        if !self.header_written {
            muxer.write_header(&self.header)?;
            self.header_written = true;
        }
        muxer.write_packet(&pkt)?;
        self.packets_written += 1;
        Ok(Vec::new())
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn finalize(&mut self) -> Result<()> {
        if let Some(mut muxer) = self.muxer.take() {
            if !self.header_written {
                muxer.write_header(&self.header)?;
                self.header_written = true;
            }
            muxer.finalize()?;
        }
        Ok(())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Transform node wrapping a [`Decoder`] (`Packet -> MediaUnit`).
pub struct DecoderNode {
    name: String,
    decoder: Box<dyn Decoder>,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
}

impl DecoderNode {
    pub fn new(decoder: Box<dyn Decoder>, input: PacketContract, output: PortContract) -> Self {
        let name = format!("decoder({})", input.codec_id.name());
        Self {
            name,
            decoder,
            inputs: [PortContract::Packet(input)],
            outputs: [output],
        }
    }

    fn drain_ready(&mut self, out: &mut Vec<(usize, GraphItem)>) -> Result<()> {
        while let RecvStatus::Ready(unit) = self.decoder.receive()? {
            out.push((0, GraphItem::Unit(unit)));
        }
        Ok(())
    }
}

impl GraphNode for DecoderNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Packet(pkt) = item else {
            return Err(Error::invalid_argument(
                "decoder_node",
                "DecoderNode expects GraphItem::Packet",
            ));
        };
        let mut out = Vec::new();
        if self.decoder.send_packet(Some(&pkt))? == SendStatus::Full {
            self.drain_ready(&mut out)?;
            if self.decoder.send_packet(Some(&pkt))? == SendStatus::Full {
                return Err(Error::internal(
                    "decoder_node",
                    "decoder remained full after draining receive queue",
                ));
            }
        }
        self.drain_ready(&mut out)?;
        Ok(out)
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let mut out = Vec::new();
        let _ = self.decoder.send_packet(None)?;
        self.drain_ready(&mut out)?;
        Ok(out)
    }

    fn health(&self) -> Option<&StreamHealth> {
        Some(self.decoder.health())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Transform node wrapping an [`Encoder`] (`MediaUnit -> Packet`).
pub struct EncoderNode {
    name: String,
    encoder: Box<dyn Encoder>,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
}

impl EncoderNode {
    pub fn new(encoder: Box<dyn Encoder>, input: PortContract, mut output: PacketContract) -> Self {
        if output.extradata.is_empty() && !encoder.extradata().is_empty() {
            output.extradata = encoder.extradata().to_vec();
            output.extradata_mode = ExtradataMode::OutOfBand;
        }
        let name = format!("encoder({})", output.codec_id.name());
        Self {
            name,
            encoder,
            inputs: [input],
            outputs: [PortContract::Packet(output)],
        }
    }

    pub fn extradata(&self) -> &[u8] {
        self.encoder.extradata()
    }

    fn drain_ready(&mut self, out: &mut Vec<(usize, GraphItem)>) -> Result<()> {
        while let RecvStatus::Ready(pkt) = self.encoder.receive_packet()? {
            out.push((0, GraphItem::Packet(pkt)));
        }
        Ok(())
    }
}

impl GraphNode for EncoderNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(unit) = item else {
            return Err(Error::invalid_argument(
                "encoder_node",
                "EncoderNode expects GraphItem::Unit",
            ));
        };
        let mut out = Vec::new();
        let hint = EncodeHint::default();
        if self.encoder.send_unit(Some(&unit), hint)? == SendStatus::Full {
            self.drain_ready(&mut out)?;
            if self.encoder.send_unit(Some(&unit), hint)? == SendStatus::Full {
                return Err(Error::internal(
                    "encoder_node",
                    "encoder remained full after draining receive_packet queue",
                ));
            }
        }
        self.drain_ready(&mut out)?;
        Ok(out)
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let mut out = Vec::new();
        let _ = self.encoder.send_unit(None, EncodeHint::default())?;
        self.drain_ready(&mut out)?;
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Video scaling / color / layout / tone-mapping node executing an [`SwsOpList`] via a pluggable [`SwsBackend`].
pub struct SwsNode {
    name: String,
    ops: SwsOpList,
    compiled: Box<dyn CompiledSws>,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    internal_pool: Arc<BufferPool>,
    assigned_domain: MemoryDomain,
}

impl SwsNode {
    pub fn new(ops: SwsOpList) -> Result<Self> {
        Self::with_backend(ops, &CpuSwsBackend::new())
    }

    pub fn with_backend(ops: SwsOpList, backend: &dyn SwsBackend) -> Result<Self> {
        let assigned_domain = ops.dst.domain;
        // Compile CPU kernel on CPU contracts so even if assigned to GPU domain in headless tests,
        // fallback execution succeeds cleanly.
        let mut cpu_ops = ops.clone();
        cpu_ops.src.domain = MemoryDomain::Cpu;
        cpu_ops.dst.domain = MemoryDomain::Cpu;
        let compiled = backend.compile(&cpu_ops)?;
        let inputs = [PortContract::Video(ops.src.clone())];
        let outputs = [PortContract::Video(ops.dst.clone())];
        Ok(Self {
            name: "sws_op_list".to_string(),
            ops,
            compiled,
            inputs,
            outputs,
            internal_pool: BufferPool::new(),
            assigned_domain,
        })
    }

    pub fn ops(&self) -> &SwsOpList {
        &self.ops
    }
}

impl GraphNode for SwsNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn domain(&self) -> MemoryDomain {
        self.assigned_domain
    }

    fn is_domain_flexible(&self) -> bool {
        true
    }

    fn set_memory_domain(&mut self, domain: MemoryDomain) -> Result<()> {
        self.assigned_domain = domain;
        self.ops.src.domain = domain;
        self.ops.dst.domain = domain;
        if let PortContract::Video(vc) = &mut self.inputs[0] {
            vc.domain = domain;
        }
        if let PortContract::Video(vc) = &mut self.outputs[0] {
            vc.domain = domain;
        }
        Ok(())
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Video(frame)) = item else {
            return Err(Error::invalid_argument(
                "sws_node",
                "SwsNode expects MediaUnit::Video",
            ));
        };

        let cpu_in = if matches!(frame.domain(), MemoryDomain::Gpu(_)) {
            download_frame_to_cpu(frame)?
        } else {
            frame
        };

        let mut out_frame = self.compiled.execute(&cpu_in, &self.internal_pool)?;
        if let MemoryDomain::Gpu(dev_id) = self.assigned_domain {
            place_in_gpu_domain(&mut out_frame, dev_id)?;
        }

        Ok(vec![(0, GraphItem::Unit(MediaUnit::Video(out_frame)))])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Audio conversion / rematrix / resample / dither / reframe node executing an [`AudioOpList`].
pub struct AudioOpsNode {
    name: String,
    ops: AudioOpList,
    compiled: CompiledAudioOps,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    internal_pool: Arc<BufferPool>,
}

impl AudioOpsNode {
    pub fn new(ops: AudioOpList) -> Result<Self> {
        let compiled = CompiledAudioOps::compile(&ops)?;
        let inputs = [PortContract::Audio(ops.src.clone())];
        let outputs = [PortContract::Audio(ops.dst.clone())];
        Ok(Self {
            name: "audio_op_list".to_string(),
            ops,
            compiled,
            inputs,
            outputs,
            internal_pool: BufferPool::new(),
        })
    }

    pub fn ops(&self) -> &AudioOpList {
        &self.ops
    }
}

impl GraphNode for AudioOpsNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Audio(block)) = item else {
            return Err(Error::invalid_argument(
                "audio_ops_node",
                "AudioOpsNode expects MediaUnit::Audio",
            ));
        };
        if block.format != self.ops.src.format
            || block.sample_rate != self.ops.src.sample_rate
            || block.channel_layout != self.ops.src.channel_layout
        {
            let actual_src = AudioContract {
                format: block.format,
                sample_rate: block.sample_rate,
                channel_layout: block.channel_layout.clone(),
                frame_size: self.ops.src.frame_size,
            };
            let rebuilt = AudioOpList::build(actual_src, self.ops.dst.clone())?;
            self.compiled = CompiledAudioOps::compile(&rebuilt)?;
            self.ops = rebuilt;
        }
        let blocks = self.compiled.process(block, &self.internal_pool)?;
        Ok(blocks
            .into_iter()
            .map(|b| (0, GraphItem::Unit(MediaUnit::Audio(b))))
            .collect())
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let blocks = self.compiled.flush(&self.internal_pool)?;
        Ok(blocks
            .into_iter()
            .map(|b| (0, GraphItem::Unit(MediaUnit::Audio(b))))
            .collect())
    }

    fn health(&self) -> Option<&StreamHealth> {
        Some(self.compiled.health())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Lossless Bitstream Filter (`BSF`) node converting `Avcc(len_size) ↔ AnnexB` for H.264 and HEVC
/// via `wedeo-syntax`.
pub struct BsfNode {
    name: String,
    src: PacketContract,
    dst: PacketContract,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    annexb_param_prefix: Vec<u8>,
}

impl BsfNode {
    pub fn new(src: PacketContract, dst: PacketContract) -> Result<Self> {
        if src.codec_id != dst.codec_id {
            return Err(Error::invalid_argument(
                "bsf_node",
                "BsfNode cannot transcode between different CodecIds",
            ));
        }
        if !matches!(src.codec_id, CodecId::H264 | CodecId::Hevc) {
            return Err(Error::unsupported(
                "bsf_node",
                "BsfNode supports H.264 and HEVC Avcc <-> AnnexB conversions only",
            ));
        }

        let mut annexb_param_prefix = Vec::new();
        if matches!(src.framing, PacketFraming::Avcc(_))
            && matches!(dst.framing, PacketFraming::AnnexB)
            && !src.extradata.is_empty()
        {
            match src.codec_id {
                CodecId::H264 => {
                    if let Ok(cfg) = wedeo_syntax::h264::parse_avcc(&src.extradata) {
                        annexb_param_prefix = wedeo_syntax::h264::avcc_extradata_to_annexb(&cfg);
                    }
                }
                CodecId::Hevc => {
                    if let Ok(cfg) = wedeo_syntax::hevc::parse_hvcc(&src.extradata) {
                        annexb_param_prefix = wedeo_syntax::hevc::hvcc_extradata_to_annexb(&cfg);
                    }
                }
                _ => {}
            }
        }

        let name = format!(
            "bsf({}: {} -> {})",
            src.codec_id.name(),
            src.framing,
            dst.framing
        );
        let inputs = [PortContract::Packet(src.clone())];
        let outputs = [PortContract::Packet(dst.clone())];
        Ok(Self {
            name,
            src,
            dst,
            inputs,
            outputs,
            annexb_param_prefix,
        })
    }

    fn convert_bytes(&self, data: &[u8], is_key: bool) -> Result<Vec<u8>> {
        match (self.src.framing, self.dst.framing) {
            (a, b) if a == b => Ok(data.to_vec()),
            (PacketFraming::Avcc(len_size), PacketFraming::AnnexB) => {
                let converted = match self.src.codec_id {
                    CodecId::H264 => wedeo_syntax::h264::avcc_to_annexb(data, len_size)?,
                    CodecId::Hevc => wedeo_syntax::hevc::hvcc_to_annexb(data, len_size)?,
                    _ => {
                        return Err(Error::unsupported(
                            "bsf_node",
                            "Avcc -> AnnexB unsupported for codec",
                        ));
                    }
                };
                if is_key && !self.annexb_param_prefix.is_empty() {
                    let mut out =
                        Vec::with_capacity(self.annexb_param_prefix.len() + converted.len());
                    out.extend_from_slice(&self.annexb_param_prefix);
                    out.extend_from_slice(&converted);
                    Ok(out)
                } else {
                    Ok(converted)
                }
            }
            (PacketFraming::AnnexB, PacketFraming::Avcc(len_size)) => match self.src.codec_id {
                CodecId::H264 => wedeo_syntax::h264::annexb_to_avcc(data, len_size),
                CodecId::Hevc => wedeo_syntax::hevc::annexb_to_hvcc(data, len_size),
                _ => Err(Error::unsupported(
                    "bsf_node",
                    "AnnexB -> Avcc unsupported for codec",
                )),
            },
            (PacketFraming::Avcc(src_len), PacketFraming::Avcc(dst_len)) => {
                let annexb = wedeo_syntax::h264::avcc_to_annexb(data, src_len)?;
                wedeo_syntax::h264::annexb_to_avcc(&annexb, dst_len)
            }
            _ => Err(Error::unsupported(
                "bsf_node",
                "unsupported packet framing conversion",
            )),
        }
    }
}

impl GraphNode for BsfNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Packet(mut pkt) = item else {
            return Err(Error::invalid_argument(
                "bsf_node",
                "BsfNode expects GraphItem::Packet",
            ));
        };
        let is_key = pkt.flags.contains(PacketFlags::KEY);
        let new_bytes = self.convert_bytes(pkt.data.data(), is_key)?;
        pkt.data = Buffer::from_slice(&new_bytes);
        Ok(vec![(0, GraphItem::Packet(pkt))])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Node wrapper around any custom or built-in [`VideoFilter`].
pub struct VideoFilterNode<F: VideoFilter> {
    name: String,
    filter: F,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
}

impl<F: VideoFilter + 'static> VideoFilterNode<F> {
    pub fn new(name: impl Into<String>, input: VideoContract, mut filter: F) -> Result<Self> {
        let output = filter.negotiate(&input)?;
        Ok(Self {
            name: name.into(),
            filter,
            inputs: [PortContract::Video(input)],
            outputs: [PortContract::Video(output)],
        })
    }

    pub fn filter(&self) -> &F {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut F {
        &mut self.filter
    }
}

impl<F: VideoFilter + 'static> GraphNode for VideoFilterNode<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Video(frame)) = item else {
            return Err(Error::invalid_argument(
                "video_filter_node",
                "VideoFilterNode expects MediaUnit::Video",
            ));
        };
        let mut out = Vec::new();
        if let Some(f) = self.filter.process(frame, pool)? {
            out.push((0, GraphItem::Unit(MediaUnit::Video(f))));
        }
        Ok(out)
    }

    fn flush(&mut self, pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let mut out = Vec::new();
        while let Some(f) = self.filter.flush(pool)? {
            out.push((0, GraphItem::Unit(MediaUnit::Video(f))));
        }
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Node wrapper around any custom or built-in [`AudioFilter`].
pub struct AudioFilterNode<F: AudioFilter> {
    name: String,
    filter: F,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
}

impl<F: AudioFilter + 'static> AudioFilterNode<F> {
    pub fn new(name: impl Into<String>, input: AudioContract, mut filter: F) -> Result<Self> {
        let output = filter.negotiate(&input)?;
        Ok(Self {
            name: name.into(),
            filter,
            inputs: [PortContract::Audio(input)],
            outputs: [PortContract::Audio(output)],
        })
    }

    pub fn filter(&self) -> &F {
        &self.filter
    }

    pub fn filter_mut(&mut self) -> &mut F {
        &mut self.filter
    }
}

impl<F: AudioFilter + 'static> GraphNode for AudioFilterNode<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Audio(block)) = item else {
            return Err(Error::invalid_argument(
                "audio_filter_node",
                "AudioFilterNode expects MediaUnit::Audio",
            ));
        };
        let mut out = Vec::new();
        if let Some(b) = self.filter.process(block, pool)? {
            out.push((0, GraphItem::Unit(MediaUnit::Audio(b))));
        }
        Ok(out)
    }

    fn flush(&mut self, pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        let mut out = Vec::new();
        while let Some(b) = self.filter.flush(pool)? {
            out.push((0, GraphItem::Unit(MediaUnit::Audio(b))));
        }
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Node wrapper around a GPU [`WgslFilter`].
pub struct WgslFilterNode<F: WgslFilter> {
    name: String,
    filter: F,
    device_id: GpuDeviceId,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
    last_uniforms: Vec<u8>,
}

impl<F: WgslFilter + 'static> WgslFilterNode<F> {
    pub fn new(name: impl Into<String>, mut input: VideoContract, filter: F) -> Result<Self> {
        let device_id = match input.domain {
            MemoryDomain::Gpu(id) => id,
            MemoryDomain::Cpu => {
                let id = GpuDeviceId(0);
                input.domain = MemoryDomain::Gpu(id);
                id
            }
        };
        let mut output = filter.output_contract(&input)?;
        output.domain = MemoryDomain::Gpu(device_id);
        Ok(Self {
            name: name.into(),
            filter,
            device_id,
            inputs: [PortContract::Video(input)],
            outputs: [PortContract::Video(output)],
            last_uniforms: Vec::new(),
        })
    }

    pub fn filter(&self) -> &F {
        &self.filter
    }

    pub fn last_uniforms(&self) -> &[u8] {
        &self.last_uniforms
    }
}

impl<F: WgslFilter + 'static> GraphNode for WgslFilterNode<F> {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn domain(&self) -> MemoryDomain {
        MemoryDomain::Gpu(self.device_id)
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Video(frame)) = item else {
            return Err(Error::invalid_argument(
                "wgsl_filter_node",
                "WgslFilterNode expects MediaUnit::Video",
            ));
        };
        let pts_sec = frame.pts.to_seconds(frame.time_base).unwrap_or(0.0);
        self.last_uniforms = self
            .filter
            .uniforms(pts_sec, frame.width, frame.height)
            .to_vec();
        let mut gpu_frame = frame;
        place_in_gpu_domain(&mut gpu_frame, self.device_id)?;
        Ok(vec![(0, GraphItem::Unit(MediaUnit::Video(gpu_frame)))])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Auto-inserted `Cpu ↔ Gpu` memory residency domain transfer node.
pub struct DomainTransferNode {
    name: String,
    dst_domain: MemoryDomain,
    inputs: [PortContract; 1],
    outputs: [PortContract; 1],
}

impl DomainTransferNode {
    pub fn new(src: VideoContract, dst_domain: MemoryDomain) -> Self {
        let mut dst = src.clone();
        dst.domain = dst_domain;
        let name = match (src.domain, dst_domain) {
            (MemoryDomain::Cpu, MemoryDomain::Gpu(GpuDeviceId(id))) => {
                format!("gpu_upload(cpu->gpu:{id})")
            }
            (MemoryDomain::Gpu(GpuDeviceId(id)), MemoryDomain::Cpu) => {
                format!("gpu_download(gpu:{id}->cpu)")
            }
            (MemoryDomain::Gpu(GpuDeviceId(a)), MemoryDomain::Gpu(GpuDeviceId(b))) => {
                format!("gpu_copy(gpu:{a}->gpu:{b})")
            }
            (MemoryDomain::Cpu, MemoryDomain::Cpu) => "cpu_noop".to_string(),
        };
        Self {
            name,
            dst_domain,
            inputs: [PortContract::Video(src)],
            outputs: [PortContract::Video(dst)],
        }
    }
}

impl GraphNode for DomainTransferNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn domain(&self) -> MemoryDomain {
        self.dst_domain
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        let GraphItem::Unit(MediaUnit::Video(frame)) = item else {
            return Err(Error::invalid_argument(
                "domain_transfer",
                "DomainTransferNode expects MediaUnit::Video",
            ));
        };
        let transferred = match self.dst_domain {
            MemoryDomain::Cpu => download_frame_to_cpu(frame)?,
            MemoryDomain::Gpu(dev_id) => {
                let mut frame = frame;
                place_in_gpu_domain(&mut frame, dev_id)?;
                frame
            }
        };
        Ok(vec![(0, GraphItem::Unit(MediaUnit::Video(transferred)))])
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Synthetic/in-memory source node emitting queued [`GraphItem`]s one at a time.
pub struct SourceNode {
    name: String,
    outputs: [PortContract; 1],
    items: VecDeque<GraphItem>,
}

impl SourceNode {
    pub fn new(
        name: impl Into<String>,
        contract: PortContract,
        items: impl IntoIterator<Item = GraphItem>,
    ) -> Self {
        Self {
            name: name.into(),
            outputs: [contract],
            items: items.into_iter().collect(),
        }
    }

    pub fn video(
        name: impl Into<String>,
        contract: VideoContract,
        frames: impl IntoIterator<Item = VideoFrame>,
    ) -> Self {
        Self::new(
            name,
            PortContract::Video(contract),
            frames
                .into_iter()
                .map(|f| GraphItem::Unit(MediaUnit::Video(f))),
        )
    }

    pub fn audio(
        name: impl Into<String>,
        contract: AudioContract,
        blocks: impl IntoIterator<Item = AudioBlock>,
    ) -> Self {
        Self::new(
            name,
            PortContract::Audio(contract),
            blocks
                .into_iter()
                .map(|b| GraphItem::Unit(MediaUnit::Audio(b))),
        )
    }

    pub fn packets(
        name: impl Into<String>,
        contract: PacketContract,
        packets: impl IntoIterator<Item = Packet>,
    ) -> Self {
        Self::new(
            name,
            PortContract::Packet(contract),
            packets.into_iter().map(GraphItem::Packet),
        )
    }
}

impl GraphNode for SourceNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &[]
    }

    fn output_contracts(&self) -> &[PortContract] {
        &self.outputs
    }

    fn process(
        &mut self,
        _input_port: usize,
        _item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        Err(Error::invalid_argument(
            "source_node",
            "SourceNode has no input ports",
        ))
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn pull(&mut self, _pool: &BufferPool) -> Result<RecvStatus<(usize, GraphItem)>> {
        match self.items.pop_front() {
            Some(item) => Ok(RecvStatus::Ready((0, item))),
            None => Ok(RecvStatus::Eof),
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// Collector sink node capturing output [`GraphItem`]s for inspection.
pub struct SinkNode {
    name: String,
    inputs: [PortContract; 1],
    collected: Vec<GraphItem>,
}

impl SinkNode {
    pub fn new(name: impl Into<String>, contract: PortContract) -> Self {
        Self {
            name: name.into(),
            inputs: [contract],
            collected: Vec::new(),
        }
    }

    pub fn items(&self) -> &[GraphItem] {
        &self.collected
    }

    pub fn take_items(&mut self) -> Vec<GraphItem> {
        std::mem::take(&mut self.collected)
    }
}

impl GraphNode for SinkNode {
    fn name(&self) -> &str {
        &self.name
    }

    fn input_contracts(&self) -> &[PortContract] {
        &self.inputs
    }

    fn output_contracts(&self) -> &[PortContract] {
        &[]
    }

    fn process(
        &mut self,
        _input_port: usize,
        item: GraphItem,
        _pool: &BufferPool,
    ) -> Result<Vec<(usize, GraphItem)>> {
        self.collected.push(item);
        Ok(Vec::new())
    }

    fn flush(&mut self, _pool: &BufferPool) -> Result<Vec<(usize, GraphItem)>> {
        Ok(Vec::new())
    }

    fn sink_items(&self) -> Option<&[GraphItem]> {
        Some(&self.collected)
    }

    fn take_sink_items(&mut self) -> Option<Vec<GraphItem>> {
        Some(std::mem::take(&mut self.collected))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

// ============================================================================
// Flat-Indexed GraphBuilder -> CompiledGraph
// ============================================================================

#[derive(Debug, Clone, Copy)]
struct RawEdge {
    from_node: NodeId,
    from_port: usize,
    to_node: NodeId,
    to_port: usize,
}

/// Flat-indexed synchronous media DAG builder.
pub struct GraphBuilder {
    nodes: Vec<Box<dyn GraphNode>>,
    is_auto_bridge: Vec<bool>,
    edges: Vec<RawEdge>,
    sws_backend: Box<dyn SwsBackend>,
}

impl Default for GraphBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphBuilder {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            is_auto_bridge: Vec::new(),
            edges: Vec::new(),
            sws_backend: Box::new(CpuSwsBackend::new()),
        }
    }

    /// Override the [`SwsBackend`] used when compiling auto-inserted [`SwsNode`] bridges.
    pub fn set_sws_backend(&mut self, backend: Box<dyn SwsBackend>) {
        self.sws_backend = backend;
    }

    /// Add any [`GraphNode`] implementation and return its flat [`NodeId`].
    pub fn add_node(&mut self, node: Box<dyn GraphNode>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.is_auto_bridge.push(false);
        id
    }

    fn add_auto_bridge_node(&mut self, node: Box<dyn GraphNode>) -> NodeId {
        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(node);
        self.is_auto_bridge.push(true);
        id
    }

    pub fn add_demuxer(&mut self, demuxer: Box<dyn Demuxer>) -> NodeId {
        self.add_node(Box::new(DemuxerNode::new(demuxer)))
    }

    pub fn add_muxer(
        &mut self,
        muxer: Box<dyn Muxer>,
        header: ContainerHeader,
        inputs: Vec<PortContract>,
    ) -> NodeId {
        self.add_node(Box::new(MuxerNode::new(muxer, header, inputs)))
    }

    pub fn add_decoder(
        &mut self,
        decoder: Box<dyn Decoder>,
        input: PacketContract,
        output: PortContract,
    ) -> NodeId {
        self.add_node(Box::new(DecoderNode::new(decoder, input, output)))
    }

    pub fn add_encoder(
        &mut self,
        encoder: Box<dyn Encoder>,
        input: PortContract,
        output: PacketContract,
    ) -> NodeId {
        self.add_node(Box::new(EncoderNode::new(encoder, input, output)))
    }

    pub fn add_sws(&mut self, ops: SwsOpList) -> Result<NodeId> {
        let node = SwsNode::with_backend(ops, self.sws_backend.as_ref())?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_sws_with_backend(
        &mut self,
        ops: SwsOpList,
        backend: &dyn SwsBackend,
    ) -> Result<NodeId> {
        let node = SwsNode::with_backend(ops, backend)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_audio_ops(&mut self, ops: AudioOpList) -> Result<NodeId> {
        let node = AudioOpsNode::new(ops)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_bsf(&mut self, src: PacketContract, dst: PacketContract) -> Result<NodeId> {
        let node = BsfNode::new(src, dst)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_video_filter<F: VideoFilter + 'static>(
        &mut self,
        name: impl Into<String>,
        input: VideoContract,
        filter: F,
    ) -> Result<NodeId> {
        let node = VideoFilterNode::new(name, input, filter)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_audio_filter<F: AudioFilter + 'static>(
        &mut self,
        name: impl Into<String>,
        input: AudioContract,
        filter: F,
    ) -> Result<NodeId> {
        let node = AudioFilterNode::new(name, input, filter)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_wgsl_filter<F: WgslFilter + 'static>(
        &mut self,
        name: impl Into<String>,
        input: VideoContract,
        filter: F,
    ) -> Result<NodeId> {
        let node = WgslFilterNode::new(name, input, filter)?;
        Ok(self.add_node(Box::new(node)))
    }

    pub fn add_source(
        &mut self,
        name: impl Into<String>,
        contract: PortContract,
        items: impl IntoIterator<Item = GraphItem>,
    ) -> NodeId {
        self.add_node(Box::new(SourceNode::new(name, contract, items)))
    }

    pub fn add_sink(&mut self, name: impl Into<String>, contract: PortContract) -> NodeId {
        self.add_node(Box::new(SinkNode::new(name, contract)))
    }

    /// Connect `(from_node, from_port)` to `(to_node, to_port)`.
    pub fn connect(
        &mut self,
        from_node: NodeId,
        from_port: usize,
        to_node: NodeId,
        to_port: usize,
    ) -> Result<EdgeId> {
        let n_len = self.nodes.len();
        if from_node.index() >= n_len || to_node.index() >= n_len {
            return Err(Error::invalid_argument(
                "wedeo-graph",
                "connect references out-of-range NodeId",
            ));
        }
        if from_node == to_node {
            return Err(Error::invalid_argument(
                "wedeo-graph",
                "self-loop edge on a single node is forbidden",
            ));
        }
        let out_len = self.nodes[from_node.index()].output_contracts().len();
        if from_port >= out_len {
            return Err(Error::invalid_argument(
                "wedeo-graph",
                format!(
                    "node {} ('{}') has {} output ports, cannot connect from port {}",
                    from_node,
                    self.nodes[from_node.index()].name(),
                    out_len,
                    from_port
                ),
            ));
        }
        let in_len = self.nodes[to_node.index()].input_contracts().len();
        if to_port >= in_len {
            return Err(Error::invalid_argument(
                "wedeo-graph",
                format!(
                    "node {} ('{}') has {} input ports, cannot connect to port {}",
                    to_node,
                    self.nodes[to_node.index()].name(),
                    in_len,
                    to_port
                ),
            ));
        }
        if self
            .edges
            .iter()
            .any(|e| e.to_node == to_node && e.to_port == to_port)
        {
            return Err(Error::invalid_argument(
                "wedeo-graph",
                format!(
                    "input port {}:{} already has an incoming edge",
                    to_node, to_port
                ),
            ));
        }

        let id = EdgeId(self.edges.len() as u32);
        self.edges.push(RawEdge {
            from_node,
            from_port,
            to_node,
            to_port,
        });
        Ok(id)
    }

    /// Compile the DAG:
    /// 1. Topological sort & cycle verification.
    /// 2. GPU residency domain placement pass (minimizing `Cpu ↔ Gpu` transfers and flagging forced bounces).
    /// 3. Auto-adapter bridge insertion (`BsfNode`, `SwsNode`, `AudioOpsNode`, `DomainTransferNode`)
    ///    with `ConversionNote` emission and `opts.strict` enforcement.
    pub fn compile(mut self, opts: CompileOpts) -> Result<(CompiledGraph, ExecutionPlan)> {
        let initial_order = topological_sort(self.nodes.len(), &self.edges)?;

        // --- Pass 1: ~50-LOC GPU residency domain placement pass ---
        let mut gpu_warnings = Vec::new();
        self.place_gpu_domains(&initial_order, opts.prefer_gpu, &mut gpu_warnings)?;

        // --- Pass 2: Auto-adapter bridge insertion across edges ---
        let original_edges = std::mem::take(&mut self.edges);
        let mut expanded_edges: Vec<RawEdge> = Vec::with_capacity(original_edges.len() * 2);
        let mut conversion_notes: Vec<ConversionNote> = Vec::new();
        let mut auto_bridges: Vec<NodeId> = Vec::new();

        for edge in original_edges {
            let src_contract =
                self.nodes[edge.from_node.index()].output_contracts()[edge.from_port].clone();
            let dst_contract =
                self.nodes[edge.to_node.index()].input_contracts()[edge.to_port].clone();

            match (&src_contract, &dst_contract) {
                (PortContract::Packet(src_p), PortContract::Packet(dst_p)) => {
                    if src_p.codec_id != dst_p.codec_id {
                        return Err(Error::invalid_argument(
                            "wedeo-graph",
                            format!(
                                "cannot auto-bridge compressed packets of different codecs ({} -> {}) without Decoder+Encoder",
                                src_p.codec_id.name(),
                                dst_p.codec_id.name()
                            ),
                        ));
                    }
                    if src_p.framing == dst_p.framing {
                        expanded_edges.push(edge);
                    } else {
                        let bsf = BsfNode::new(src_p.clone(), dst_p.clone())?;
                        let summary = format!(
                            "{} packet framing {} -> {}",
                            src_p.codec_id.name(),
                            src_p.framing,
                            dst_p.framing
                        );
                        let bsf_id = self.add_auto_bridge_node(Box::new(bsf));
                        auto_bridges.push(bsf_id);
                        conversion_notes.push(ConversionNote::new(
                            edge.from_node,
                            edge.to_node,
                            bsf_id,
                            false,
                            summary,
                            Vec::new(),
                        ));
                        expanded_edges.push(RawEdge {
                            from_node: edge.from_node,
                            from_port: edge.from_port,
                            to_node: bsf_id,
                            to_port: 0,
                        });
                        expanded_edges.push(RawEdge {
                            from_node: bsf_id,
                            from_port: 0,
                            to_node: edge.to_node,
                            to_port: edge.to_port,
                        });
                    }
                }
                (PortContract::Video(src_v), PortContract::Video(dst_v)) => {
                    let mut src_cpu = src_v.clone();
                    src_cpu.domain = MemoryDomain::Cpu;
                    let mut dst_cpu = dst_v.clone();
                    dst_cpu.domain = MemoryDomain::Cpu;

                    let pixel_or_color_differs = src_cpu != dst_cpu;
                    let domain_differs = src_v.domain != dst_v.domain;

                    if !pixel_or_color_differs && !domain_differs {
                        expanded_edges.push(edge);
                    } else {
                        let mut cur_node = edge.from_node;
                        let mut cur_port = edge.from_port;
                        let mut cur_contract = src_v.clone();

                        if pixel_or_color_differs {
                            let loss_reasons = classify_video_conversion_loss(src_v, dst_v);
                            let is_lossy = !loss_reasons.is_empty();
                            let summary = format!(
                                "SwsOpList {}x{} {} ({}/{}) -> {}x{} {} ({}/{})",
                                src_v.width,
                                src_v.height,
                                src_v.layout.name(),
                                src_v.color.primaries.name(),
                                src_v.color.transfer.name(),
                                dst_v.width,
                                dst_v.height,
                                dst_v.layout.name(),
                                dst_v.color.primaries.name(),
                                dst_v.color.transfer.name()
                            );
                            if is_lossy && opts.strict {
                                return Err(Error::invalid_argument(
                                    "wedeo-graph",
                                    format!(
                                        "strict mode rejected implicit lossy video bridge ({} -> {}): {} [{}]",
                                        edge.from_node,
                                        edge.to_node,
                                        summary,
                                        loss_reasons.join(", ")
                                    ),
                                ));
                            }

                            let sws_domain = match (src_v.domain, dst_v.domain, opts.prefer_gpu) {
                                (MemoryDomain::Gpu(id), _, _) => MemoryDomain::Gpu(id),
                                (_, MemoryDomain::Gpu(id), _) => MemoryDomain::Gpu(id),
                                (_, _, Some(id)) => MemoryDomain::Gpu(id),
                                _ => MemoryDomain::Cpu,
                            };

                            if cur_contract.domain != sws_domain {
                                let xfer =
                                    DomainTransferNode::new(cur_contract.clone(), sws_domain);
                                let xfer_id = self.add_auto_bridge_node(Box::new(xfer));
                                auto_bridges.push(xfer_id);
                                expanded_edges.push(RawEdge {
                                    from_node: cur_node,
                                    from_port: cur_port,
                                    to_node: xfer_id,
                                    to_port: 0,
                                });
                                cur_node = xfer_id;
                                cur_port = 0;
                            }

                            let mut sws_ops =
                                SwsOpList::build(&src_cpu, &dst_cpu, SwsOptions::default())?;
                            let mut sws_node =
                                SwsNode::with_backend(sws_ops.clone(), self.sws_backend.as_ref())?;
                            sws_node.set_memory_domain(sws_domain)?;
                            sws_ops.src.domain = sws_domain;
                            sws_ops.dst.domain = sws_domain;

                            let sws_id = self.add_auto_bridge_node(Box::new(sws_node));
                            auto_bridges.push(sws_id);
                            conversion_notes.push(ConversionNote::new(
                                edge.from_node,
                                edge.to_node,
                                sws_id,
                                is_lossy,
                                summary,
                                loss_reasons,
                            ));
                            expanded_edges.push(RawEdge {
                                from_node: cur_node,
                                from_port: cur_port,
                                to_node: sws_id,
                                to_port: 0,
                            });
                            cur_node = sws_id;
                            cur_port = 0;
                            cur_contract = dst_v.clone();
                            cur_contract.domain = sws_domain;
                        }

                        if cur_contract.domain != dst_v.domain {
                            let xfer = DomainTransferNode::new(cur_contract.clone(), dst_v.domain);
                            let summary = format!(
                                "domain transfer {:?} -> {:?}",
                                cur_contract.domain, dst_v.domain
                            );
                            let xfer_id = self.add_auto_bridge_node(Box::new(xfer));
                            auto_bridges.push(xfer_id);
                            conversion_notes.push(ConversionNote::new(
                                edge.from_node,
                                edge.to_node,
                                xfer_id,
                                false,
                                summary,
                                Vec::new(),
                            ));
                            expanded_edges.push(RawEdge {
                                from_node: cur_node,
                                from_port: cur_port,
                                to_node: xfer_id,
                                to_port: 0,
                            });
                            cur_node = xfer_id;
                            cur_port = 0;
                        }

                        expanded_edges.push(RawEdge {
                            from_node: cur_node,
                            from_port: cur_port,
                            to_node: edge.to_node,
                            to_port: edge.to_port,
                        });
                    }
                }
                (PortContract::Audio(src_a), PortContract::Audio(dst_a)) => {
                    if src_a == dst_a {
                        expanded_edges.push(edge);
                    } else {
                        let loss_reasons = classify_audio_conversion_loss(src_a, dst_a);
                        let is_lossy = !loss_reasons.is_empty();
                        let summary = format!(
                            "AudioOpList {} {}Hz {} -> {} {}Hz {}",
                            src_a.format.name(),
                            src_a.sample_rate,
                            src_a.channel_layout,
                            dst_a.format.name(),
                            dst_a.sample_rate,
                            dst_a.channel_layout
                        );
                        if is_lossy && opts.strict {
                            return Err(Error::invalid_argument(
                                "wedeo-graph",
                                format!(
                                    "strict mode rejected implicit lossy audio bridge ({} -> {}): {} [{}]",
                                    edge.from_node,
                                    edge.to_node,
                                    summary,
                                    loss_reasons.join(", ")
                                ),
                            ));
                        }
                        let ops = AudioOpList::build(src_a.clone(), dst_a.clone())?;
                        let node = AudioOpsNode::new(ops)?;
                        let bridge_id = self.add_auto_bridge_node(Box::new(node));
                        auto_bridges.push(bridge_id);
                        conversion_notes.push(ConversionNote::new(
                            edge.from_node,
                            edge.to_node,
                            bridge_id,
                            is_lossy,
                            summary,
                            loss_reasons,
                        ));
                        expanded_edges.push(RawEdge {
                            from_node: edge.from_node,
                            from_port: edge.from_port,
                            to_node: bridge_id,
                            to_port: 0,
                        });
                        expanded_edges.push(RawEdge {
                            from_node: bridge_id,
                            from_port: 0,
                            to_node: edge.to_node,
                            to_port: edge.to_port,
                        });
                    }
                }
                (PortContract::Subtitle(a), PortContract::Subtitle(b)) if a == b => {
                    expanded_edges.push(edge);
                }
                (PortContract::TimedData(a), PortContract::TimedData(b)) if a == b => {
                    expanded_edges.push(edge);
                }
                _ => {
                    return Err(Error::invalid_argument(
                        "wedeo-graph",
                        format!(
                            "incompatible port contracts on edge {}:{} ({}) -> {}:{} ({})",
                            edge.from_node,
                            edge.from_port,
                            src_contract,
                            edge.to_node,
                            edge.to_port,
                            dst_contract
                        ),
                    ));
                }
            }
        }

        self.edges = expanded_edges;
        let final_order = topological_sort(self.nodes.len(), &self.edges)?;

        let planned_nodes: Vec<PlannedNode> = (0..self.nodes.len())
            .map(|i| {
                let n = &self.nodes[i];
                PlannedNode {
                    id: NodeId(i as u32),
                    name: n.name().to_string(),
                    domain: n.domain(),
                    is_auto_bridge: self.is_auto_bridge[i],
                    inputs: n.input_contracts().to_vec(),
                    outputs: n.output_contracts().to_vec(),
                }
            })
            .collect();

        let planned_edges: Vec<PlannedEdge> = self
            .edges
            .iter()
            .enumerate()
            .map(|(i, e)| {
                let contract =
                    self.nodes[e.from_node.index()].output_contracts()[e.from_port].clone();
                PlannedEdge {
                    id: EdgeId(i as u32),
                    from_node: e.from_node,
                    from_port: e.from_port,
                    to_node: e.to_node,
                    to_port: e.to_port,
                    contract,
                }
            })
            .collect();

        let plan = ExecutionPlan {
            order: final_order.clone(),
            nodes: planned_nodes,
            edges: planned_edges,
            auto_bridges,
            conversion_notes,
            gpu_warnings,
            strict: opts.strict,
            preserve_all: opts.preserve_all,
        };

        // Pre-index outgoing adjacency lists per node for O(1) dispatch during `run()`.
        let mut outgoing = vec![Vec::new(); self.nodes.len()];
        let mut in_degrees = vec![0usize; self.nodes.len()];
        for e in &self.edges {
            outgoing[e.from_node.index()].push((e.from_port, e.to_node, e.to_port));
            in_degrees[e.to_node.index()] += 1;
        }
        let sources: Vec<NodeId> = final_order
            .iter()
            .copied()
            .filter(|id| in_degrees[id.index()] == 0)
            .collect();

        let compiled = CompiledGraph {
            nodes: self.nodes,
            order: final_order,
            sources,
            outgoing,
            plan: plan.clone(),
        };

        Ok((compiled, plan))
    }

    /// ~50-LOC GPU residency domain placement pass:
    /// - Relocates flexible nodes (`SwsNode`) onto `MemoryDomain::Gpu` when adjacent to GPU nodes
    ///   or when `prefer_gpu` is set, preventing avoidable `Gpu -> Cpu -> Gpu` roundtrips.
    /// - Detects forced `GPU -> CPU -> GPU` bounces across pinned CPU-only nodes and records
    ///   actionable warnings in `ExecutionPlan::gpu_warnings`.
    fn place_gpu_domains(
        &mut self,
        order: &[NodeId],
        prefer_gpu: Option<GpuDeviceId>,
        warnings: &mut Vec<String>,
    ) -> Result<()> {
        for &node_id in order {
            let idx = node_id.index();
            if self.nodes[idx].is_domain_flexible() {
                let upstream_gpu =
                    self.edges
                        .iter()
                        .filter(|e| e.to_node == node_id)
                        .find_map(|e| match self.nodes[e.from_node.index()].domain() {
                            MemoryDomain::Gpu(dev) => Some(dev),
                            MemoryDomain::Cpu => None,
                        });
                let downstream_gpu = self
                    .edges
                    .iter()
                    .filter(|e| e.from_node == node_id)
                    .find_map(|e| match self.nodes[e.to_node.index()].domain() {
                        MemoryDomain::Gpu(dev) => Some(dev),
                        MemoryDomain::Cpu => None,
                    });
                if let Some(dev) = upstream_gpu.or(downstream_gpu).or(prefer_gpu) {
                    self.nodes[idx].set_memory_domain(MemoryDomain::Gpu(dev))?;
                }
            }
        }

        // Detect forced GPU -> CPU -> GPU ping-pong across non-flexible CPU nodes
        for &mid_id in order {
            let mid_idx = mid_id.index();
            if self.nodes[mid_idx].domain() == MemoryDomain::Cpu {
                let gpu_preds: Vec<(NodeId, GpuDeviceId)> = self
                    .edges
                    .iter()
                    .filter(|e| e.to_node == mid_id)
                    .filter_map(|e| match self.nodes[e.from_node.index()].domain() {
                        MemoryDomain::Gpu(dev) => Some((e.from_node, dev)),
                        MemoryDomain::Cpu => None,
                    })
                    .collect();
                let gpu_succs: Vec<(NodeId, GpuDeviceId)> = self
                    .edges
                    .iter()
                    .filter(|e| e.from_node == mid_id)
                    .filter_map(|e| match self.nodes[e.to_node.index()].domain() {
                        MemoryDomain::Gpu(dev) => Some((e.to_node, dev)),
                        MemoryDomain::Cpu => None,
                    })
                    .collect();

                for &(pred_id, pred_dev) in &gpu_preds {
                    for &(succ_id, succ_dev) in &gpu_succs {
                        warnings.push(format!(
                            "forced GPU->CPU->GPU bounce across CPU-only node {} ('{}') between {} ('{}', gpu:{}) and {} ('{}', gpu:{})",
                            mid_id,
                            self.nodes[mid_idx].name(),
                            pred_id,
                            self.nodes[pred_id.index()].name(),
                            pred_dev.0,
                            succ_id,
                            self.nodes[succ_id.index()].name(),
                            succ_dev.0
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Kahn's algorithm topological sort returning a deterministic [`NodeId`] execution schedule
/// or `Error::invalid_argument` if a cycle is present.
fn topological_sort(node_count: usize, edges: &[RawEdge]) -> Result<Vec<NodeId>> {
    let mut in_degree = vec![0usize; node_count];
    let mut adj = vec![Vec::new(); node_count];
    for e in edges {
        adj[e.from_node.index()].push(e.to_node);
        in_degree[e.to_node.index()] += 1;
    }

    let mut queue: VecDeque<NodeId> = VecDeque::with_capacity(node_count);
    for (i, &deg) in in_degree.iter().enumerate() {
        if deg == 0 {
            queue.push_back(NodeId(i as u32));
        }
    }

    let mut order = Vec::with_capacity(node_count);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for &v in &adj[u.index()] {
            let d = &mut in_degree[v.index()];
            *d -= 1;
            if *d == 0 {
                queue.push_back(v);
            }
        }
    }

    if order.len() != node_count {
        return Err(Error::invalid_argument(
            "wedeo-graph",
            "cycle detected in media graph; DAG must be acyclic",
        ));
    }
    Ok(order)
}

/// Compiled, flat-indexed synchronous media DAG ready for execution.
pub struct CompiledGraph {
    nodes: Vec<Box<dyn GraphNode>>,
    order: Vec<NodeId>,
    sources: Vec<NodeId>,
    /// Per-node outgoing edges: `(from_port, to_node, to_port)`.
    outgoing: Vec<Vec<(usize, NodeId, usize)>>,
    plan: ExecutionPlan,
}

impl CompiledGraph {
    /// Total successful writes across the graph's container muxers.
    pub fn packets_written(&self) -> u64 {
        self.nodes
            .iter()
            .filter_map(|node| node.as_any().downcast_ref::<MuxerNode>())
            .map(|muxer| muxer.packets_written)
            .sum()
    }

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    pub fn node(&self, id: NodeId) -> Option<&dyn GraphNode> {
        self.nodes.get(id.index()).map(Box::as_ref)
    }

    pub fn node_mut(&mut self, id: NodeId) -> Option<&mut (dyn GraphNode + 'static)> {
        match self.nodes.get_mut(id.index()) {
            Some(b) => Some(b.as_mut()),
            None => None,
        }
    }

    /// Inspect items collected by a [`SinkNode`] at `sink_id`.
    pub fn sink_items(&self, sink_id: NodeId) -> Option<&[GraphItem]> {
        self.node(sink_id)?.sink_items()
    }

    /// Take items collected by a [`SinkNode`] at `sink_id`.
    pub fn take_sink_items(&mut self, sink_id: NodeId) -> Option<Vec<GraphItem>> {
        self.node_mut(sink_id)?.take_sink_items()
    }

    /// Execute the graph synchronously to completion, collecting packet/unit counts and
    /// aggregated [`StreamHealth`].
    pub fn run(&mut self, pool: &BufferPool) -> Result<RunReport> {
        let mut packets_processed = 0u64;
        let mut units_processed = 0u64;
        let mut work_queue: VecDeque<(NodeId, usize, GraphItem)> = VecDeque::new();

        // Phase 1: Pull items from all source nodes in streaming lockstep
        let mut active_sources = self.sources.clone();
        while !active_sources.is_empty() {
            let mut next_sources = Vec::with_capacity(active_sources.len());
            for src_id in active_sources {
                match self.nodes[src_id.index()].pull(pool)? {
                    RecvStatus::Ready((out_port, item)) => {
                        Self::count_item(&item, &mut packets_processed, &mut units_processed);
                        self.enqueue_outgoing(src_id, out_port, item, &mut work_queue);
                        self.drain_work_queue(
                            pool,
                            &mut work_queue,
                            &mut packets_processed,
                            &mut units_processed,
                        )?;
                        next_sources.push(src_id);
                    }
                    RecvStatus::NeedMore | RecvStatus::Eof => {}
                }
            }
            active_sources = next_sources;
        }

        // Phase 2: Flush all nodes in topological order and propagate flushed items downstream
        let order = self.order.clone();
        for node_id in order {
            let flushed = self.nodes[node_id.index()].flush(pool)?;
            for (out_port, item) in flushed {
                Self::count_item(&item, &mut packets_processed, &mut units_processed);
                self.enqueue_outgoing(node_id, out_port, item, &mut work_queue);
            }
            self.drain_work_queue(
                pool,
                &mut work_queue,
                &mut packets_processed,
                &mut units_processed,
            )?;
        }

        // Phase 3: Finalize all nodes and aggregate StreamHealth
        let mut health = StreamHealth::new();
        for &node_id in &self.order {
            let node = &mut self.nodes[node_id.index()];
            node.finalize()?;
            if let Some(h) = node.health() {
                health.merge(h);
            }
        }

        Ok(RunReport {
            packets_processed,
            units_processed,
            health,
            plan: self.plan.clone(),
        })
    }

    #[inline]
    fn count_item(item: &GraphItem, packets: &mut u64, units: &mut u64) {
        match item {
            GraphItem::Packet(_) => *packets += 1,
            GraphItem::Unit(_) => *units += 1,
        }
    }

    fn enqueue_outgoing(
        &self,
        from_node: NodeId,
        from_port: usize,
        item: GraphItem,
        queue: &mut VecDeque<(NodeId, usize, GraphItem)>,
    ) {
        let edges = &self.outgoing[from_node.index()];
        let matching: Vec<(NodeId, usize)> = edges
            .iter()
            .filter_map(|&(fp, to_n, to_p)| (fp == from_port).then_some((to_n, to_p)))
            .collect();
        for (i, (to_n, to_p)) in matching.iter().copied().enumerate() {
            if i + 1 == matching.len() {
                queue.push_back((to_n, to_p, item));
                break;
            } else {
                queue.push_back((to_n, to_p, item.clone()));
            }
        }
    }

    fn drain_work_queue(
        &mut self,
        pool: &BufferPool,
        queue: &mut VecDeque<(NodeId, usize, GraphItem)>,
        packets: &mut u64,
        units: &mut u64,
    ) -> Result<()> {
        while let Some((target_node, target_port, item)) = queue.pop_front() {
            let outputs = self.nodes[target_node.index()].process(target_port, item, pool)?;
            for (out_port, out_item) in outputs {
                Self::count_item(&out_item, packets, units);
                self.enqueue_outgoing(target_node, out_port, out_item, queue);
            }
        }
        Ok(())
    }
}
