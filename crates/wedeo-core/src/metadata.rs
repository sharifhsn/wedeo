/// Ordered metadata container using `Vec<(String, String)>` for deterministic
/// ordering (required for FATE test parity). Case-insensitive key lookup.
#[derive(Debug, Clone, Default)]
pub struct Metadata {
    entries: Vec<(String, String)>,
}

impl Metadata {
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Set a metadata key-value pair. If the key already exists (case-insensitive),
    /// the existing value is replaced. Otherwise, the pair is appended.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) {
        let key = key.into();
        let value = value.into();
        if let Some(entry) = self
            .entries
            .iter_mut()
            .find(|(k, _)| k.eq_ignore_ascii_case(&key))
        {
            entry.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    /// Get the value for a key (case-insensitive).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(key))
            .map(|(_, v)| v.as_str())
    }

    /// Remove a key (case-insensitive). Returns the removed value if found.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        if let Some(pos) = self
            .entries
            .iter()
            .position(|(k, _)| k.eq_ignore_ascii_case(key))
        {
            Some(self.entries.remove(pos).1)
        } else {
            None
        }
    }

    /// Iterate over all entries in insertion order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.entries.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Number of entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metadata_basic() {
        let mut m = Metadata::new();
        m.set("title", "Test Video");
        m.set("artist", "Wedeo");
        assert_eq!(m.get("title"), Some("Test Video"));
        assert_eq!(m.get("TITLE"), Some("Test Video")); // case-insensitive
        assert_eq!(m.len(), 2);
    }

    #[test]
    fn test_metadata_overwrite() {
        let mut m = Metadata::new();
        m.set("title", "First");
        m.set("TITLE", "Second"); // should overwrite
        assert_eq!(m.get("title"), Some("Second"));
        assert_eq!(m.len(), 1);
    }

    #[test]
    fn test_metadata_order() {
        let mut m = Metadata::new();
        m.set("b", "2");
        m.set("a", "1");
        m.set("c", "3");
        let keys: Vec<&str> = m.iter().map(|(k, _)| k).collect();
        assert_eq!(keys, vec!["b", "a", "c"]); // insertion order preserved
    }

    #[test]
    fn test_metadata_remove() {
        let mut m = Metadata::new();
        m.set("key", "value");
        assert_eq!(m.remove("KEY"), Some("value".to_string()));
        assert!(m.is_empty());
    }
}
