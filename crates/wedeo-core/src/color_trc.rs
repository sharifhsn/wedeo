use std::fmt;

/// Color transfer characteristic, matching FFmpeg's AVColorTransferCharacteristic.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(i32)]
pub enum ColorTransferCharacteristic {
    Reserved0 = 0,
    Bt709 = 1,
    #[default]
    Unspecified = 2,
    Reserved = 3,
    Gamma22 = 4,
    Gamma28 = 5,
    Smpte170m = 6,
    Smpte240m = 7,
    Linear = 8,
    Log = 9,
    LogSqrt = 10,
    Iec61966_2_4 = 11,
    Bt1361Ecg = 12,
    Iec61966_2_1 = 13,
    Bt2020_10 = 14,
    Bt2020_12 = 15,
    Smpte2084 = 16,
    Smpte428 = 17,
    AribStdB67 = 18,
}

impl ColorTransferCharacteristic {
    /// Returns the FFmpeg-compatible string name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Reserved0 => "reserved",
            Self::Bt709 => "bt709",
            Self::Unspecified => "unknown",
            Self::Reserved => "reserved",
            Self::Gamma22 => "bt470m",
            Self::Gamma28 => "bt470bg",
            Self::Smpte170m => "smpte170m",
            Self::Smpte240m => "smpte240m",
            Self::Linear => "linear",
            Self::Log => "log100",
            Self::LogSqrt => "log316",
            Self::Iec61966_2_4 => "iec61966-2-4",
            Self::Bt1361Ecg => "bt1361e",
            Self::Iec61966_2_1 => "iec61966-2-1",
            Self::Bt2020_10 => "bt2020-10",
            Self::Bt2020_12 => "bt2020-12",
            Self::Smpte2084 => "smpte2084",
            Self::Smpte428 => "smpte428",
            Self::AribStdB67 => "arib-std-b67",
        }
    }
}

impl fmt::Display for ColorTransferCharacteristic {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_is_unspecified() {
        assert_eq!(
            ColorTransferCharacteristic::default(),
            ColorTransferCharacteristic::Unspecified
        );
    }

    #[test]
    fn test_repr_values() {
        assert_eq!(ColorTransferCharacteristic::Reserved0 as i32, 0);
        assert_eq!(ColorTransferCharacteristic::Bt709 as i32, 1);
        assert_eq!(ColorTransferCharacteristic::Smpte2084 as i32, 16);
        assert_eq!(ColorTransferCharacteristic::AribStdB67 as i32, 18);
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", ColorTransferCharacteristic::Bt709), "bt709");
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::Unspecified),
            "unknown"
        );
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::Gamma22),
            "bt470m"
        );
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::Gamma28),
            "bt470bg"
        );
        assert_eq!(format!("{}", ColorTransferCharacteristic::Log), "log100");
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::LogSqrt),
            "log316"
        );
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::Bt1361Ecg),
            "bt1361e"
        );
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::AribStdB67),
            "arib-std-b67"
        );
        assert_eq!(
            format!("{}", ColorTransferCharacteristic::Iec61966_2_1),
            "iec61966-2-1"
        );
    }
}
