use wedeo_core::error::Result;

use crate::io::IoContext;

/// Protocol trait for URL-based I/O.
pub trait Protocol: Send + Sync {
    fn name(&self) -> &'static str;
    fn open(&self, url: &str, flags: ProtocolFlags) -> Result<Box<dyn IoContext>>;
}

/// Protocol open flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProtocolFlags {
    pub read: bool,
    pub write: bool,
}

impl ProtocolFlags {
    pub const READ: Self = Self {
        read: true,
        write: false,
    };
    pub const WRITE: Self = Self {
        read: false,
        write: true,
    };
    pub const READ_WRITE: Self = Self {
        read: true,
        write: true,
    };
}

inventory::collect!(&'static dyn Protocol);

/// Open a URL using the registered protocol handlers.
pub fn open_url(url: &str, flags: ProtocolFlags) -> Result<Box<dyn IoContext>> {
    // Extract scheme from URL
    let scheme = url.find("://").map(|i| &url[..i]).unwrap_or("file");

    for proto in inventory::iter::<&'static dyn Protocol>() {
        if proto.name() == scheme {
            return proto.open(url, flags);
        }
    }

    // Default: try as file path
    if flags.write {
        Ok(Box::new(crate::io::FileIo::create(url)?))
    } else {
        Ok(Box::new(crate::io::FileIo::open(url)?))
    }
}
