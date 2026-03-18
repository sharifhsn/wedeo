/// Codec-private options, matching FFmpeg's AVOption system.
///
/// Options are stored as string key-value pairs, parsed by each codec.
/// Uses `Vec<(String, String)>` for deterministic ordering (required for
/// FATE test parity — see CLAUDE.md).
#[derive(Debug, Clone, Default)]
pub struct CodecOptions {
    options: Vec<(String, String)>,
}

impl CodecOptions {
    pub fn new() -> Self {
        Self {
            options: Vec::new(),
        }
    }

    /// Set an option. If the key already exists, its value is replaced in-place
    /// to preserve insertion order.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(entry) = self.options.iter_mut().find(|(k, _)| k == &key) {
            entry.1 = value;
        } else {
            self.options.push((key, value));
        }
    }

    /// Get the string value for a key.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.options
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Get an option parsed as `i64`.
    pub fn get_i64(&self, key: &str) -> Option<i64> {
        self.get(key).and_then(|v| v.parse().ok())
    }

    /// Get an option parsed as `f64`.
    pub fn get_f64(&self, key: &str) -> Option<f64> {
        self.get(key).and_then(|v| v.parse().ok())
    }

    /// Get an option parsed as `bool`.
    ///
    /// Recognizes "1", "true", "yes" as true and "0", "false", "no" as false,
    /// matching FFmpeg's `av_opt_set` boolean parsing.
    pub fn get_bool(&self, key: &str) -> Option<bool> {
        self.get(key).and_then(|v| match v {
            "1" | "true" | "yes" => Some(true),
            "0" | "false" | "no" => Some(false),
            _ => None,
        })
    }

    /// Iterate over all options as `(&str, &str)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.options.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Returns true if no options have been set.
    pub fn is_empty(&self) -> bool {
        self.options.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_set_and_get() {
        let mut opts = CodecOptions::new();
        opts.set("bitrate", "128000");
        assert_eq!(opts.get("bitrate"), Some("128000"));
        assert_eq!(opts.get("missing"), None);
    }

    #[test]
    fn test_overwrite_preserves_order() {
        let mut opts = CodecOptions::new();
        opts.set("a", "1");
        opts.set("b", "2");
        opts.set("a", "3");

        let keys: Vec<&str> = opts.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["a", "b"]);
        assert_eq!(opts.get("a"), Some("3"));
    }

    #[test]
    fn test_typed_getters() {
        let mut opts = CodecOptions::new();
        opts.set("threads", "4");
        opts.set("quality", "0.95");
        opts.set("strict", "true");
        opts.set("experimental", "1");
        opts.set("disabled", "no");
        opts.set("bad_int", "abc");

        assert_eq!(opts.get_i64("threads"), Some(4));
        assert_eq!(opts.get_f64("quality"), Some(0.95));
        assert_eq!(opts.get_bool("strict"), Some(true));
        assert_eq!(opts.get_bool("experimental"), Some(true));
        assert_eq!(opts.get_bool("disabled"), Some(false));
        assert_eq!(opts.get_i64("bad_int"), None);
        assert_eq!(opts.get_i64("missing"), None);
    }

    #[test]
    fn test_is_empty() {
        let mut opts = CodecOptions::new();
        assert!(opts.is_empty());
        opts.set("key", "value");
        assert!(!opts.is_empty());
    }

    #[test]
    fn test_default() {
        let opts = CodecOptions::default();
        assert!(opts.is_empty());
    }
}
