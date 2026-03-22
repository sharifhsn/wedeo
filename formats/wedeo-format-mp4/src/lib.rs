mod atoms;
mod boxes;
pub mod demuxer;
pub mod muxer;

use wedeo_core::codec_id::CodecId;
use wedeo_core::error::Result;
use wedeo_format::muxer::{Muxer, OutputFormatDescriptor, OutputFormatFlags};
use wedeo_format::registry::MuxerFactory;

struct Mp4MuxerFactory;

impl MuxerFactory for Mp4MuxerFactory {
    fn descriptor(&self) -> &OutputFormatDescriptor {
        static DESC: OutputFormatDescriptor = OutputFormatDescriptor {
            name: "mp4",
            long_name: "MP4 (MPEG-4 Part 14)",
            extensions: "mp4,m4a,m4v",
            mime_types: "video/mp4",
            flags: OutputFormatFlags::GLOBALHEADER,
            audio_codec: CodecId::Aac,
            video_codec: CodecId::H264,
        };
        &DESC
    }

    fn create(&self) -> Result<Box<dyn Muxer>> {
        Ok(Box::new(muxer::Mp4Muxer::new()))
    }
}

inventory::submit!(&Mp4MuxerFactory as &dyn MuxerFactory);
