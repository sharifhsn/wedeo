pub mod graph;
pub mod registry;

use bitflags::bitflags;

use wedeo_core::error::Result;
use wedeo_core::media_type::MediaType;

bitflags! {
    /// Filter flags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct FilterFlags: u32 {
        const DYNAMIC_INPUTS       = 1 << 0;
        const DYNAMIC_OUTPUTS      = 1 << 1;
        const SLICE_THREADS        = 1 << 2;
        const METADATA_ONLY        = 1 << 3;
        const SUPPORT_TIMELINE_GENERIC  = 1 << 16;
        const SUPPORT_TIMELINE_INTERNAL = 1 << 17;
    }
}

/// Filter pad direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterPadDirection {
    Input,
    Output,
}

/// Descriptor for a filter pad (input or output).
#[derive(Debug, Clone)]
pub struct FilterPadDescriptor {
    pub name: &'static str,
    pub media_type: MediaType,
    pub direction: FilterPadDirection,
}

/// Descriptor for a filter type.
#[derive(Debug, Clone)]
pub struct FilterDescriptor {
    pub name: &'static str,
    pub description: &'static str,
    pub inputs: &'static [FilterPadDescriptor],
    pub outputs: &'static [FilterPadDescriptor],
    pub flags: FilterFlags,
}

/// Filter trait — the main abstraction for filter implementations.
pub trait Filter: Send {
    fn descriptor(&self) -> &FilterDescriptor;
    fn activate(&mut self) -> Result<()>;
}
