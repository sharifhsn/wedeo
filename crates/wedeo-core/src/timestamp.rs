use crate::rational::Rational;

/// Undefined/unknown timestamp value, matching FFmpeg's AV_NOPTS_VALUE.
pub const NOPTS_VALUE: i64 = i64::MIN;

/// Internal time base, matching FFmpeg's AV_TIME_BASE (1MHz).
pub const TIME_BASE: i32 = 1_000_000;

/// Internal time base as a rational, matching FFmpeg's AV_TIME_BASE_Q.
pub const TIME_BASE_Q: Rational = Rational::new(1, TIME_BASE);
