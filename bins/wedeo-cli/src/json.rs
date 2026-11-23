//! Minimal, deterministic pure-Rust JSON serializer and RFC 8259 parser (zero external deps).
//!
//! Used by `ops.rs` (`--json` reports) and `mcp.rs` (stdio JSON-RPC 2.0 server).

use std::fmt;

/// A dynamically-typed JSON value.
#[derive(Debug, Clone, PartialEq)]
pub enum JsonValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Arr(Vec<JsonValue>),
    Obj(Vec<(String, JsonValue)>),
}

impl JsonValue {
    pub fn obj(fields: Vec<(&str, JsonValue)>) -> Self {
        Self::Obj(
            fields
                .into_iter()
                .map(|(k, v)| (k.to_string(), v))
                .collect(),
        )
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            Self::Float(f) if f.is_finite() && f.fract() == 0.0 => Some(*f as i64),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Int(i) if *i >= 0 => Some(*i as u64),
            Self::Float(f) if f.is_finite() && *f >= 0.0 && f.fract() == 0.0 => Some(*f as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Self::Int(i) => Some(*i as f64),
            Self::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_arr(&self) -> Option<&[JsonValue]> {
        match self {
            Self::Arr(a) => Some(a.as_slice()),
            _ => None,
        }
    }

    pub fn as_obj(&self) -> Option<&[(String, JsonValue)]> {
        match self {
            Self::Obj(o) => Some(o.as_slice()),
            _ => None,
        }
    }

    pub fn get(&self, key: &str) -> Option<&JsonValue> {
        self.as_obj()?
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }

    pub fn to_compact(&self) -> String {
        let mut out = String::with_capacity(256);
        self.write_fmt(&mut out, None, 0);
        out
    }

    pub fn to_pretty(&self) -> String {
        let mut out = String::with_capacity(512);
        self.write_fmt(&mut out, Some(2), 0);
        out
    }

    fn write_fmt(&self, out: &mut String, indent: Option<usize>, depth: usize) {
        match self {
            Self::Null => out.push_str("null"),
            Self::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
            Self::Int(i) => out.push_str(&i.to_string()),
            Self::Float(f) => {
                if f.is_finite() {
                    let s = format!("{f}");
                    out.push_str(&s);
                } else {
                    out.push_str("null");
                }
            }
            Self::Str(s) => {
                out.push('"');
                write_escaped_json(s, out);
                out.push('"');
            }
            Self::Arr(items) => {
                if items.is_empty() {
                    out.push_str("[]");
                    return;
                }
                out.push('[');
                for (idx, item) in items.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    if let Some(step) = indent {
                        out.push('\n');
                        push_spaces(out, (depth + 1) * step);
                    }
                    item.write_fmt(out, indent, depth + 1);
                }
                if let Some(step) = indent {
                    out.push('\n');
                    push_spaces(out, depth * step);
                }
                out.push(']');
            }
            Self::Obj(entries) => {
                if entries.is_empty() {
                    out.push_str("{}");
                    return;
                }
                out.push('{');
                for (idx, (k, v)) in entries.iter().enumerate() {
                    if idx > 0 {
                        out.push(',');
                    }
                    if let Some(step) = indent {
                        out.push('\n');
                        push_spaces(out, (depth + 1) * step);
                    }
                    out.push('"');
                    write_escaped_json(k, out);
                    out.push('"');
                    out.push(':');
                    if indent.is_some() {
                        out.push(' ');
                    }
                    v.write_fmt(out, indent, depth + 1);
                }
                if let Some(step) = indent {
                    out.push('\n');
                    push_spaces(out, depth * step);
                }
                out.push('}');
            }
        }
    }

    /// Parse a single JSON value from `input`.
    pub fn parse(input: &str) -> Result<Self, JsonParseError> {
        let mut p = Parser {
            bytes: input.as_bytes(),
            pos: 0,
        };
        p.skip_ws();
        let val = p.parse_value()?;
        p.skip_ws();
        if p.pos != p.bytes.len() {
            return Err(JsonParseError::new(p.pos, "trailing characters after JSON"));
        }
        Ok(val)
    }
}

impl From<bool> for JsonValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}

impl From<i32> for JsonValue {
    fn from(v: i32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<i64> for JsonValue {
    fn from(v: i64) -> Self {
        Self::Int(v)
    }
}

impl From<u64> for JsonValue {
    fn from(v: u64) -> Self {
        match i64::try_from(v) {
            Ok(i) => Self::Int(i),
            Err(_) => Self::Float(v as f64),
        }
    }
}

impl From<u32> for JsonValue {
    fn from(v: u32) -> Self {
        Self::Int(i64::from(v))
    }
}

impl From<usize> for JsonValue {
    fn from(v: usize) -> Self {
        match i64::try_from(v) {
            Ok(i) => Self::Int(i),
            Err(_) => Self::Float(v as f64),
        }
    }
}

impl From<f64> for JsonValue {
    fn from(v: f64) -> Self {
        Self::Float(v)
    }
}

impl From<&str> for JsonValue {
    fn from(v: &str) -> Self {
        Self::Str(v.to_string())
    }
}

impl From<String> for JsonValue {
    fn from(v: String) -> Self {
        Self::Str(v)
    }
}

impl<T: Into<JsonValue>> From<Option<T>> for JsonValue {
    fn from(v: Option<T>) -> Self {
        match v {
            Some(inner) => inner.into(),
            None => Self::Null,
        }
    }
}

impl<T: Into<JsonValue>> From<Vec<T>> for JsonValue {
    fn from(v: Vec<T>) -> Self {
        Self::Arr(v.into_iter().map(Into::into).collect())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonParseError {
    pub offset: usize,
    pub message: String,
}

impl JsonParseError {
    fn new(offset: usize, msg: impl Into<String>) -> Self {
        Self {
            offset,
            message: msg.into(),
        }
    }
}

impl fmt::Display for JsonParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "JSON parse error at byte {}: {}",
            self.offset, self.message
        )
    }
}

impl std::error::Error for JsonParseError {}

struct Parser<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn next(&mut self) -> Option<u8> {
        let b = self.peek()?;
        self.pos += 1;
        Some(b)
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn parse_value(&mut self) -> Result<JsonValue, JsonParseError> {
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(JsonParseError::new(self.pos, "unexpected end of input"));
        };
        match b {
            b'n' => self.parse_null(),
            b't' => self.parse_true(),
            b'f' => self.parse_false(),
            b'"' => self.parse_string().map(JsonValue::Str),
            b'[' => self.parse_array(),
            b'{' => self.parse_object(),
            b'-' | b'0'..=b'9' => self.parse_number(),
            other => Err(JsonParseError::new(
                self.pos,
                format!("unexpected byte '{}'", other as char),
            )),
        }
    }

    fn expect_bytes(&mut self, expected: &[u8]) -> Result<(), JsonParseError> {
        let start = self.pos;
        if self.bytes[self.pos..].starts_with(expected) {
            self.pos += expected.len();
            Ok(())
        } else {
            Err(JsonParseError::new(
                start,
                format!("expected '{}'", String::from_utf8_lossy(expected)),
            ))
        }
    }

    fn parse_null(&mut self) -> Result<JsonValue, JsonParseError> {
        self.expect_bytes(b"null")?;
        Ok(JsonValue::Null)
    }

    fn parse_true(&mut self) -> Result<JsonValue, JsonParseError> {
        self.expect_bytes(b"true")?;
        Ok(JsonValue::Bool(true))
    }

    fn parse_false(&mut self) -> Result<JsonValue, JsonParseError> {
        self.expect_bytes(b"false")?;
        Ok(JsonValue::Bool(false))
    }

    fn parse_string(&mut self) -> Result<String, JsonParseError> {
        let start = self.pos;
        if self.next() != Some(b'"') {
            return Err(JsonParseError::new(start, "expected '\"'"));
        }
        let mut out = String::new();
        loop {
            let Some(b) = self.next() else {
                return Err(JsonParseError::new(self.pos, "unterminated string"));
            };
            match b {
                b'"' => return Ok(out),
                b'\\' => {
                    let Some(esc) = self.next() else {
                        return Err(JsonParseError::new(self.pos, "unterminated escape"));
                    };
                    match esc {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{0008}'),
                        b'f' => out.push('\u{000C}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let code = self.parse_hex4()?;
                            if (0xD800..=0xDBFF).contains(&code) {
                                if !self.bytes[self.pos..].starts_with(b"\\u") {
                                    return Err(JsonParseError::new(
                                        self.pos,
                                        "missing low surrogate",
                                    ));
                                }
                                self.pos += 2;
                                let low = self.parse_hex4()?;
                                if !(0xDC00..=0xDFFF).contains(&low) {
                                    return Err(JsonParseError::new(
                                        self.pos,
                                        "invalid low surrogate",
                                    ));
                                }
                                let cp = 0x10000
                                    + (((u32::from(code) - 0xD800) << 10)
                                        | (u32::from(low) - 0xDC00));
                                let ch = char::from_u32(cp).ok_or_else(|| {
                                    JsonParseError::new(self.pos, "invalid Unicode scalar")
                                })?;
                                out.push(ch);
                            } else {
                                let ch = char::from_u32(u32::from(code)).ok_or_else(|| {
                                    JsonParseError::new(self.pos, "invalid Unicode escape")
                                })?;
                                out.push(ch);
                            }
                        }
                        other => {
                            return Err(JsonParseError::new(
                                self.pos - 1,
                                format!("invalid escape '\\{}'", other as char),
                            ));
                        }
                    }
                }
                b if b < 0x20 => {
                    return Err(JsonParseError::new(
                        self.pos - 1,
                        "unescaped control character in string",
                    ));
                }
                first => {
                    let utf8_start = self.pos - 1;
                    let width = utf8_char_width(first);
                    let end = utf8_start + width;
                    if end > self.bytes.len() {
                        return Err(JsonParseError::new(utf8_start, "truncated UTF-8 sequence"));
                    }
                    let s = std::str::from_utf8(&self.bytes[utf8_start..end]).map_err(|_| {
                        JsonParseError::new(utf8_start, "invalid UTF-8 sequence in string")
                    })?;
                    out.push_str(s);
                    self.pos = end;
                }
            }
        }
    }

    fn parse_hex4(&mut self) -> Result<u16, JsonParseError> {
        if self.pos + 4 > self.bytes.len() {
            return Err(JsonParseError::new(self.pos, "truncated \\uXXXX escape"));
        }
        let hex = std::str::from_utf8(&self.bytes[self.pos..self.pos + 4])
            .map_err(|_| JsonParseError::new(self.pos, "invalid \\uXXXX escape"))?;
        let val = u16::from_str_radix(hex, 16)
            .map_err(|_| JsonParseError::new(self.pos, "invalid hex digit in \\uXXXX"))?;
        self.pos += 4;
        Ok(val)
    }

    fn parse_number(&mut self) -> Result<JsonValue, JsonParseError> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            _ => return Err(JsonParseError::new(start, "invalid number")),
        }
        let mut is_float = false;
        if self.peek() == Some(b'.') {
            is_float = true;
            self.pos += 1;
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(JsonParseError::new(self.pos, "expected digit after '.'"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            is_float = true;
            self.pos += 1;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if !matches!(self.peek(), Some(b'0'..=b'9')) {
                return Err(JsonParseError::new(self.pos, "expected exponent digit"));
            }
            while matches!(self.peek(), Some(b'0'..=b'9')) {
                self.pos += 1;
            }
        }
        let slice = std::str::from_utf8(&self.bytes[start..self.pos])
            .map_err(|_| JsonParseError::new(start, "invalid UTF-8 in number"))?;
        if !is_float && let Ok(i) = slice.parse::<i64>() {
            return Ok(JsonValue::Int(i));
        }
        let f = slice
            .parse::<f64>()
            .map_err(|_| JsonParseError::new(start, "invalid floating-point number"))?;
        Ok(JsonValue::Float(f))
    }

    fn parse_array(&mut self) -> Result<JsonValue, JsonParseError> {
        self.pos += 1; // '['
        let mut items = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(JsonValue::Arr(items));
        }
        loop {
            items.push(self.parse_value()?);
            self.skip_ws();
            match self.next() {
                Some(b',') => {}
                Some(b']') => return Ok(JsonValue::Arr(items)),
                _ => return Err(JsonParseError::new(self.pos, "expected ',' or ']'")),
            }
        }
    }

    fn parse_object(&mut self) -> Result<JsonValue, JsonParseError> {
        self.pos += 1; // '{'
        let mut entries = Vec::new();
        self.skip_ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(JsonValue::Obj(entries));
        }
        loop {
            self.skip_ws();
            let key = self.parse_string()?;
            self.skip_ws();
            if self.next() != Some(b':') {
                return Err(JsonParseError::new(self.pos, "expected ':' after key"));
            }
            let val = self.parse_value()?;
            entries.push((key, val));
            self.skip_ws();
            match self.next() {
                Some(b',') => {}
                Some(b'}') => return Ok(JsonValue::Obj(entries)),
                _ => return Err(JsonParseError::new(self.pos, "expected ',' or '}'")),
            }
        }
    }
}

fn utf8_char_width(first: u8) -> usize {
    if first < 0x80 {
        1
    } else if (first & 0xE0) == 0xC0 {
        2
    } else if (first & 0xF0) == 0xE0 {
        3
    } else {
        4
    }
}

fn push_spaces(out: &mut String, count: usize) {
    for _ in 0..count {
        out.push(' ');
    }
}

pub fn write_escaped_json(input: &str, out: &mut String) {
    use std::fmt::Write;
    for ch in input.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{0008}' => out.push_str("\\b"),
            '\u{000C}' => out.push_str("\\f"),
            c if c.is_control() => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn json_roundtrip_and_escapes() {
        let val = JsonValue::obj(vec![
            ("jsonrpc", "2.0".into()),
            ("id", 42i64.into()),
            ("ok", true.into()),
            ("ratio", 1.5f64.into()),
            ("msg", "hello \"wedeo\"\nline2".into()),
            ("items", JsonValue::Arr(vec![1i64.into(), JsonValue::Null])),
        ]);
        let compact = val.to_compact();
        let parsed = JsonValue::parse(&compact).unwrap();
        assert_eq!(parsed.get("id").and_then(JsonValue::as_i64), Some(42));
        assert_eq!(parsed.get("ok").and_then(JsonValue::as_bool), Some(true));
        assert_eq!(
            parsed.get("msg").and_then(JsonValue::as_str),
            Some("hello \"wedeo\"\nline2")
        );
        let pretty = val.to_pretty();
        assert_eq!(JsonValue::parse(&pretty).unwrap(), parsed);
    }
}
