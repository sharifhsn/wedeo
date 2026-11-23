//! Pure-Rust FFmpeg `n9.1` H.264/AVC video decoder for Wedeo.

#![deny(unsafe_code)]
#![allow(rustdoc::broken_intra_doc_links)]

#[cfg(feature = "dev-trace")]
macro_rules! dev_trace {
    ($($arg:tt)*) => {
        eprintln!($($arg)*);
    };
}

#[cfg(not(feature = "dev-trace"))]
#[allow(unused_macros)]
macro_rules! dev_trace {
    ($($arg:tt)*) => {
        if false {
            let _ = format_args!($($arg)*);
        }
    };
}

#[allow(unused_imports)]
pub(crate) use dev_trace;

pub mod bitstream;
pub mod cabac;
pub mod cabac_tables;
pub mod cavlc;
pub mod cavlc_tables;
pub mod deblock;
pub mod decoder;
pub mod dequant;
pub mod dpb;
pub mod idct;
pub mod intra_pred;
pub mod mb;
pub mod mc;
pub mod mvpred;
pub mod nal;
pub mod pps;
pub mod refs;
pub mod sei;
pub mod shared_picture;
pub mod simd;
pub mod slice;
pub mod sps;
pub mod tables;
pub mod thread_pool;

pub use decoder::{H264Decoder, H264DecoderFactory, register_h264};
