use wedeo_core::error::Result;

use crate::{Filter, FilterDescriptor};

/// Factory for creating filter instances. Implementations register via `inventory`.
pub trait FilterFactory: Send + Sync {
    fn descriptor(&self) -> &FilterDescriptor;
    fn create(&self) -> Result<Box<dyn Filter>>;
}

inventory::collect!(&'static dyn FilterFactory);

/// Find a filter factory by name.
pub fn find_filter_by_name(name: &str) -> Option<&'static dyn FilterFactory> {
    inventory::iter::<&'static dyn FilterFactory>()
        .find(|f| f.descriptor().name == name)
        .copied()
}

/// Iterate over all registered filter factories.
pub fn filters() -> impl Iterator<Item = &'static dyn FilterFactory> {
    inventory::iter::<&'static dyn FilterFactory>().copied()
}
