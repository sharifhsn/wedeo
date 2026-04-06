use wedeo_core::error::Result;

use crate::demuxer::{Demuxer, InputFormatDescriptor, ProbeData};
use crate::muxer::{Muxer, OutputFormatDescriptor};

/// Factory for creating demuxer instances. Implementations register via `inventory`.
pub trait DemuxerFactory: Send + Sync {
    fn descriptor(&self) -> &InputFormatDescriptor;

    /// Probe the data and return a score (0 = no match, PROBE_SCORE_MAX = definite match).
    fn probe(&self, data: &ProbeData<'_>) -> i32;

    /// Create a new demuxer instance.
    fn create(&self) -> Result<Box<dyn Demuxer>>;
}

inventory::collect!(&'static dyn DemuxerFactory);

/// Factory for creating muxer instances. Implementations register via `inventory`.
pub trait MuxerFactory: Send + Sync {
    fn descriptor(&self) -> &OutputFormatDescriptor;
    fn create(&self) -> Result<Box<dyn Muxer>>;
}

inventory::collect!(&'static dyn MuxerFactory);

/// Probe a data source and find the best-matching demuxer.
/// When multiple demuxers return the same probe score, the one with the
/// highest priority wins (native implementations over wrappers).
pub fn probe(data: &ProbeData<'_>) -> Option<&'static dyn DemuxerFactory> {
    let mut best: Option<(&'static dyn DemuxerFactory, i32, i32)> = None;

    for factory in inventory::iter::<&'static dyn DemuxerFactory>() {
        let score = factory.probe(data);
        if score > 0 {
            let priority = factory.descriptor().priority;
            if let Some((_, best_score, best_priority)) = best {
                if score > best_score || (score == best_score && priority > best_priority) {
                    best = Some((*factory, score, priority));
                }
            } else {
                best = Some((*factory, score, priority));
            }
        }
    }

    best.map(|(f, _, _)| f)
}

/// Find a demuxer factory by name.
pub fn find_demuxer_by_name(name: &str) -> Option<&'static dyn DemuxerFactory> {
    inventory::iter::<&'static dyn DemuxerFactory>()
        .find(|f| f.descriptor().name == name)
        .copied()
}

/// Find a muxer factory by name.
pub fn find_muxer_by_name(name: &str) -> Option<&'static dyn MuxerFactory> {
    inventory::iter::<&'static dyn MuxerFactory>()
        .find(|f| f.descriptor().name == name)
        .copied()
}

/// Iterate over all registered demuxer factories.
pub fn demuxers() -> impl Iterator<Item = &'static dyn DemuxerFactory> {
    inventory::iter::<&'static dyn DemuxerFactory>().copied()
}

/// Iterate over all registered muxer factories.
pub fn muxers() -> impl Iterator<Item = &'static dyn MuxerFactory> {
    inventory::iter::<&'static dyn MuxerFactory>().copied()
}

/// Find the best muxer for a given file extension (e.g. "mp4", "wav").
pub fn guess_muxer_by_extension(ext: &str) -> Option<&'static dyn MuxerFactory> {
    let ext_lower = ext.to_ascii_lowercase();
    inventory::iter::<&'static dyn MuxerFactory>()
        .find(|f| {
            f.descriptor()
                .extensions
                .split(',')
                .any(|e| e.trim() == ext_lower)
        })
        .copied()
}
