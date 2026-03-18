/// Log level, matching FFmpeg's AV_LOG_* levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(i32)]
pub enum LogLevel {
    Quiet = -8,
    Panic = 0,
    Fatal = 8,
    Error = 16,
    Warning = 24,
    Info = 32,
    Verbose = 40,
    Debug = 48,
    Trace = 56,
}

impl LogLevel {
    /// Convert to a `tracing::Level`.
    pub fn to_tracing_level(self) -> tracing::Level {
        match self {
            LogLevel::Quiet | LogLevel::Panic | LogLevel::Fatal => tracing::Level::ERROR,
            LogLevel::Error => tracing::Level::ERROR,
            LogLevel::Warning => tracing::Level::WARN,
            LogLevel::Info | LogLevel::Verbose => tracing::Level::INFO,
            LogLevel::Debug => tracing::Level::DEBUG,
            LogLevel::Trace => tracing::Level::TRACE,
        }
    }
}
