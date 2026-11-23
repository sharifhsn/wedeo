//! Native ISOBMFF / MP4 / MOV / M4A demuxer, faststart/auto-fMP4 muxer, and box AST (`wedeo inspect`).

#![forbid(unsafe_code)]

mod atoms;
pub mod boxes;
pub mod demuxer;
pub mod muxer;

use std::sync::Arc;

use wedeo_core::Registry;

/// Register the native MP4/MOV/M4A demuxer and muxer into `reg`.
pub fn register_mp4(reg: &mut Registry) {
    reg.register_demuxer(Arc::new(demuxer::Mp4DemuxerFactory));
    reg.register_muxer(Arc::new(muxer::Mp4MuxerFactory));
}
