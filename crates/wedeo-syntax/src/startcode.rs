//! Shared Annex B start-code scanning, RBSP emulation-prevention handling, and
//! length-prefixed <-> Annex B conversion helpers for H.264 and HEVC.

use wedeo_core::{Error, Result};

/// Find the earliest Annex B start code (`00 00 01` or `00 00 00 01`) at or after `from`.
///
/// Returns `Some((start_code_offset, start_code_len))` where `start_code_len` is `3` or `4`.
pub fn find_start_code(data: &[u8], from: usize) -> Option<(usize, usize)> {
    if data.len() < from.saturating_add(3) {
        return None;
    }
    let mut i = from + 2;
    while i < data.len() {
        if data[i] == 0x01 && data[i - 1] == 0x00 && data[i - 2] == 0x00 {
            if i >= from + 3 && data[i - 3] == 0x00 {
                return Some((i - 3, 4));
            }
            return Some((i - 2, 3));
        }
        if data[i] > 0x01 {
            // Skip ahead: neither `data[i]` nor `data[i-1]` can be the two leading zeros
            // of a start code ending at `i + 1` or `i + 2`.
            i += 3;
        } else {
            i += 1;
        }
    }
    None
}

/// A raw NAL unit located inside an Annex B byte stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RawAnnexBNal<'a> {
    /// Byte offset of the start code prefix (`00 00 01` or `00 00 00 01`).
    pub start_code_offset: usize,
    /// Length of the start code prefix (`3` or `4`).
    pub start_code_len: usize,
    /// Raw NAL bytes starting at the NAL unit header (excluding the start code).
    pub payload: &'a [u8],
}

/// Zero-allocation iterator over Annex B NAL units in a byte slice.
pub struct AnnexBIter<'a> {
    data: &'a [u8],
    next_sc: Option<(usize, usize)>,
}

impl<'a> AnnexBIter<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        let next_sc = find_start_code(data, 0);
        Self { data, next_sc }
    }
}

impl<'a> Iterator for AnnexBIter<'a> {
    type Item = RawAnnexBNal<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let (sc_off, sc_len) = self.next_sc?;
        let payload_start = sc_off + sc_len;
        let next = find_start_code(self.data, payload_start);
        let payload_end = next.map_or(self.data.len(), |(off, _)| off);
        self.next_sc = next;
        Some(RawAnnexBNal {
            start_code_offset: sc_off,
            start_code_len: sc_len,
            payload: &self.data[payload_start..payload_end],
        })
    }
}

/// Strip H.264/HEVC emulation-prevention bytes (`00 00 03` -> `00 00`) into an RBSP buffer.
///
/// Matches ITU-T H.264 §7.3.1 and FFmpeg `ff_h2645_extract_rbsp`.
pub fn ebsp_to_rbsp(ebsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(ebsp.len());
    ebsp_to_rbsp_into(ebsp, &mut out);
    out
}

/// Strip emulation-prevention bytes into a caller-supplied reusable buffer.
pub fn ebsp_to_rbsp_into(ebsp: &[u8], out: &mut Vec<u8>) {
    out.clear();
    out.reserve(ebsp.len());
    let mut zeros = 0u32;
    for &b in ebsp {
        if zeros >= 2 && b == 0x03 {
            zeros = 0;
            continue;
        }
        out.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
}

/// Insert emulation-prevention bytes (`00 00 {00..=03}` -> `00 00 03 {00..=03}`) to produce EBSP.
pub fn rbsp_to_ebsp(rbsp: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + rbsp.len() / 32);
    let mut zeros = 0u32;
    for &b in rbsp {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0x00 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

/// Compute the logical RBSP bit length excluding the trailing `rbsp_stop_one_bit` and zero padding,
/// matching FFmpeg's `get_bit_length` in `h2645_parse.c`.
pub fn rbsp_bit_length(rbsp: &[u8]) -> usize {
    let mut len = rbsp.len();
    while len > 0 && rbsp[len - 1] == 0 {
        len -= 1;
    }
    if len == 0 {
        return 0;
    }
    let last = rbsp[len - 1];
    len * 8 - (last.trailing_zeros() as usize + 1)
}

/// Losslessly convert a length-prefixed packet (AVCC / HVCC) to Annex B (4-byte start codes).
pub fn length_prefixed_to_annexb(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(Error::invalid_argument(
            "syntax/annexb",
            "nal_length_size must be in 1..=4",
        ));
    }
    let nls = nal_length_size as usize;
    let mut out = Vec::with_capacity(packet.len() + 16);
    let mut pos = 0usize;
    while pos < packet.len() {
        if packet.len() - pos < nls {
            return Err(Error::invalid_data(
                "syntax/annexb",
                "truncated NAL length prefix",
            ));
        }
        let mut nal_len = 0usize;
        for &b in &packet[pos..pos + nls] {
            nal_len = (nal_len << 8) | (b as usize);
        }
        pos += nls;
        if packet.len() - pos < nal_len {
            return Err(Error::invalid_data(
                "syntax/annexb",
                "NAL length exceeds remaining packet bytes",
            ));
        }
        if nal_len > 0 {
            out.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            out.extend_from_slice(&packet[pos..pos + nal_len]);
        }
        pos += nal_len;
    }
    Ok(out)
}

/// Losslessly convert an Annex B packet to length-prefixed format (AVCC / HVCC).
pub fn annexb_to_length_prefixed(packet: &[u8], nal_length_size: u8) -> Result<Vec<u8>> {
    if !(1..=4).contains(&nal_length_size) {
        return Err(Error::invalid_argument(
            "syntax/avcc",
            "nal_length_size must be in 1..=4",
        ));
    }
    let nls = nal_length_size as usize;
    let max_nal_len: usize = match nal_length_size {
        1 => u8::MAX as usize,
        2 => u16::MAX as usize,
        3 => 0x00FF_FFFF,
        4 => u32::MAX as usize,
        _ => unreachable!(),
    };

    let mut out = Vec::with_capacity(packet.len());
    for unit in AnnexBIter::new(packet) {
        if unit.payload.is_empty() {
            continue;
        }
        if unit.payload.len() > max_nal_len {
            return Err(Error::invalid_data(
                "syntax/avcc",
                "NAL unit length exceeds nal_length_size capacity",
            ));
        }
        let len_be = (unit.payload.len() as u32).to_be_bytes();
        out.extend_from_slice(&len_be[4 - nls..4]);
        out.extend_from_slice(unit.payload);
    }
    Ok(out)
}
