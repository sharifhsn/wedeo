// NAL unit parsing for H.264/AVC.
//
// Handles both Annex B (start-code delimited) and NALFF/avcC (length-prefixed)
// NAL unit streams. Performs emulation prevention byte removal to produce RBSP.
//
// Reference: ITU-T H.264 Section 7.3.1, FFmpeg libavcodec/h2645_parse.c

use wedeo_core::error::{Error, Result};

/// H.264 NAL unit types (ITU-T H.264 Table 7-1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum NalUnitType {
    Slice = 1,
    SliceA = 2,
    SliceB = 3,
    SliceC = 4,
    Idr = 5,
    Sei = 6,
    Sps = 7,
    Pps = 8,
    Aud = 9,
    EndSequence = 10,
    EndStream = 11,
    Filler = 12,
}

impl TryFrom<u8> for NalUnitType {
    type Error = Error;

    fn try_from(value: u8) -> Result<Self> {
        match value {
            1 => Ok(NalUnitType::Slice),
            2 => Ok(NalUnitType::SliceA),
            3 => Ok(NalUnitType::SliceB),
            4 => Ok(NalUnitType::SliceC),
            5 => Ok(NalUnitType::Idr),
            6 => Ok(NalUnitType::Sei),
            7 => Ok(NalUnitType::Sps),
            8 => Ok(NalUnitType::Pps),
            9 => Ok(NalUnitType::Aud),
            10 => Ok(NalUnitType::EndSequence),
            11 => Ok(NalUnitType::EndStream),
            12 => Ok(NalUnitType::Filler),
            _ => Err(Error::InvalidData),
        }
    }
}

/// A parsed NAL unit with header fields and RBSP data (emulation prevention bytes removed).
#[derive(Debug, Clone)]
pub struct NalUnit {
    pub nal_type: NalUnitType,
    pub nal_ref_idc: u8,
    pub data: Vec<u8>,
}

/// Remove emulation prevention bytes from raw NAL data to produce RBSP.
///
/// In H.264, the byte sequence `0x00, 0x00, 0x03` is an emulation prevention pattern.
/// The `0x03` byte is stripped, leaving the two zero bytes followed by whatever byte
/// came after the `0x03`. This prevents NAL payload data from accidentally containing
/// start code patterns (`0x00, 0x00, 0x00` or `0x00, 0x00, 0x01`).
fn remove_emulation_prevention_bytes(data: &[u8]) -> Vec<u8> {
    let mut rbsp = Vec::with_capacity(data.len());
    let mut i = 0;
    while i < data.len() {
        // Check for emulation prevention byte pattern: 0x00 0x00 0x03
        if i + 2 < data.len() && data[i] == 0x00 && data[i + 1] == 0x00 && data[i + 2] == 0x03 {
            rbsp.push(0x00);
            rbsp.push(0x00);
            // Skip the 0x03 emulation prevention byte
            i += 3;
        } else {
            rbsp.push(data[i]);
            i += 1;
        }
    }
    rbsp
}

/// Parse the NAL header byte, returning `(nal_ref_idc, nal_unit_type)`.
///
/// NAL header byte layout:
///   bit 7:    forbidden_zero_bit (must be 0)
///   bits 5-6: nal_ref_idc
///   bits 0-4: nal_unit_type
fn parse_nal_header(header_byte: u8) -> Result<(u8, NalUnitType)> {
    let forbidden_zero_bit = (header_byte >> 7) & 1;
    if forbidden_zero_bit != 0 {
        return Err(Error::InvalidData);
    }
    let nal_ref_idc = (header_byte >> 5) & 0x03;
    let nal_type_raw = header_byte & 0x1F;
    let nal_type = NalUnitType::try_from(nal_type_raw)?;
    Ok((nal_ref_idc, nal_type))
}

/// Parse a single raw NAL unit (header byte + payload) into a `NalUnit`.
///
/// The input `raw` must be the bytes of a single NAL unit (after start code or
/// length prefix removal), starting with the header byte.
fn parse_nal_unit(raw: &[u8]) -> Result<NalUnit> {
    if raw.is_empty() {
        return Err(Error::InvalidData);
    }
    let (nal_ref_idc, nal_type) = parse_nal_header(raw[0])?;
    let rbsp = remove_emulation_prevention_bytes(&raw[1..]);
    Ok(NalUnit {
        nal_type,
        nal_ref_idc,
        data: rbsp,
    })
}

/// Find the next Annex B start code in `data` starting at position `pos`.
///
/// Returns `Some((start_code_pos, start_code_len))` where `start_code_len` is
/// 3 for `0x000001` or 4 for `0x00000001`. Returns `None` if no start code is found.
fn find_start_code(data: &[u8], pos: usize) -> Option<(usize, usize)> {
    let mut i = pos;
    while i + 2 < data.len() {
        if data[i] == 0x00 && data[i + 1] == 0x00 {
            if data[i + 2] == 0x01 {
                return Some((i, 3));
            }
            if i + 3 < data.len() && data[i + 2] == 0x00 && data[i + 3] == 0x01 {
                return Some((i, 4));
            }
        }
        i += 1;
    }
    None
}

/// Split an Annex B byte stream into NAL units.
///
/// Scans for start codes (`0x000001` or `0x00000001`), extracts each NAL unit,
/// removes emulation prevention bytes, and parses the NAL header.
///
/// NAL units with unrecognized `nal_unit_type` values (0, 13-31) are silently
/// skipped, as they may appear in valid streams but are not relevant for decoding.
pub fn split_annex_b(data: &[u8]) -> Vec<NalUnit> {
    let mut nalus = Vec::new();

    // Find the first start code
    let Some((first_sc_pos, first_sc_len)) = find_start_code(data, 0) else {
        return nalus;
    };

    let mut nal_start = first_sc_pos + first_sc_len;

    loop {
        // Find the next start code (or end of data)
        let nal_end;
        let next_sc;
        if let Some((sc_pos, sc_len)) = find_start_code(data, nal_start) {
            nal_end = sc_pos;
            next_sc = Some(sc_pos + sc_len);
        } else {
            nal_end = data.len();
            next_sc = None;
        }

        let raw = &data[nal_start..nal_end];
        if !raw.is_empty()
            && let Ok(nalu) = parse_nal_unit(raw)
        {
            nalus.push(nalu);
        }

        match next_sc {
            Some(next) => nal_start = next,
            None => break,
        }
    }

    nalus
}

/// Split a NALFF (MP4/avcC) byte stream into NAL units.
///
/// In NALFF format, each NAL unit is prefixed by a big-endian length field.
/// The `length_size` parameter specifies the number of bytes in the length field
/// (typically 1, 2, or 4; derived from `lengthSizeMinusOne + 1` in the avcC box).
///
/// NAL units with unrecognized `nal_unit_type` values are silently skipped.
pub fn split_nalff(data: &[u8], length_size: u8) -> Vec<NalUnit> {
    let mut nalus = Vec::new();
    let ls = length_size as usize;
    let mut i = 0;

    while i + ls <= data.len() {
        // Read the length prefix (big-endian)
        let nal_len = match ls {
            1 => data[i] as usize,
            2 => u16::from_be_bytes([data[i], data[i + 1]]) as usize,
            4 => u32::from_be_bytes([data[i], data[i + 1], data[i + 2], data[i + 3]]) as usize,
            _ => break,
        };
        i += ls;

        if nal_len == 0 || i + nal_len > data.len() {
            break;
        }

        let raw = &data[i..i + nal_len];
        if let Ok(nalu) = parse_nal_unit(raw) {
            nalus.push(nalu);
        }

        i += nal_len;
    }

    nalus
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- EPB removal tests ---

    #[test]
    fn epb_removal_00_00_03_00() {
        let input = [0x00, 0x00, 0x03, 0x00];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, vec![0x00, 0x00, 0x00]);
    }

    #[test]
    fn epb_removal_00_00_03_01() {
        let input = [0x00, 0x00, 0x03, 0x01];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, vec![0x00, 0x00, 0x01]);
    }

    #[test]
    fn epb_removal_00_00_03_02() {
        let input = [0x00, 0x00, 0x03, 0x02];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, vec![0x00, 0x00, 0x02]);
    }

    #[test]
    fn epb_removal_00_00_03_03() {
        let input = [0x00, 0x00, 0x03, 0x03];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, vec![0x00, 0x00, 0x03]);
    }

    #[test]
    fn epb_removal_no_epb() {
        let input = [0x01, 0x02, 0x03, 0x04];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, input.to_vec());
    }

    #[test]
    fn epb_removal_multiple() {
        // Two EPB sequences in a row
        let input = [0x00, 0x00, 0x03, 0x00, 0x00, 0x00, 0x03, 0x01];
        let result = remove_emulation_prevention_bytes(&input);
        assert_eq!(result, vec![0x00, 0x00, 0x00, 0x00, 0x00, 0x01]);
    }

    #[test]
    fn epb_removal_empty() {
        let result = remove_emulation_prevention_bytes(&[]);
        assert!(result.is_empty());
    }

    // --- NAL header parsing tests ---

    #[test]
    fn parse_header_sps() {
        // nal_ref_idc=3, nal_unit_type=7 (SPS) -> 0b0110_0111 = 0x67
        let (ref_idc, nal_type) = parse_nal_header(0x67).unwrap();
        assert_eq!(ref_idc, 3);
        assert_eq!(nal_type, NalUnitType::Sps);
    }

    #[test]
    fn parse_header_pps() {
        // nal_ref_idc=3, nal_unit_type=8 (PPS) -> 0b0110_1000 = 0x68
        let (ref_idc, nal_type) = parse_nal_header(0x68).unwrap();
        assert_eq!(ref_idc, 3);
        assert_eq!(nal_type, NalUnitType::Pps);
    }

    #[test]
    fn parse_header_idr() {
        // nal_ref_idc=3, nal_unit_type=5 (IDR) -> 0b0110_0101 = 0x65
        let (ref_idc, nal_type) = parse_nal_header(0x65).unwrap();
        assert_eq!(ref_idc, 3);
        assert_eq!(nal_type, NalUnitType::Idr);
    }

    #[test]
    fn parse_header_sei() {
        // nal_ref_idc=0, nal_unit_type=6 (SEI) -> 0b0000_0110 = 0x06
        let (ref_idc, nal_type) = parse_nal_header(0x06).unwrap();
        assert_eq!(ref_idc, 0);
        assert_eq!(nal_type, NalUnitType::Sei);
    }

    #[test]
    fn parse_header_slice() {
        // nal_ref_idc=2, nal_unit_type=1 (Slice) -> 0b0100_0001 = 0x41
        let (ref_idc, nal_type) = parse_nal_header(0x41).unwrap();
        assert_eq!(ref_idc, 2);
        assert_eq!(nal_type, NalUnitType::Slice);
    }

    #[test]
    fn parse_header_aud() {
        // nal_ref_idc=0, nal_unit_type=9 (AUD) -> 0b0000_1001 = 0x09
        let (ref_idc, nal_type) = parse_nal_header(0x09).unwrap();
        assert_eq!(ref_idc, 0);
        assert_eq!(nal_type, NalUnitType::Aud);
    }

    #[test]
    fn parse_header_forbidden_bit_set() {
        // forbidden_zero_bit=1 -> 0b1000_0111 = 0x87
        let result = parse_nal_header(0x87);
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_unknown_type() {
        // nal_unit_type=0 is not in our enum
        let result = parse_nal_header(0x60); // ref_idc=3, type=0
        assert!(result.is_err());
    }

    #[test]
    fn parse_header_type_13_unknown() {
        // nal_unit_type=13 (SPS Extension) not in our enum
        let result = parse_nal_header(0x6D); // ref_idc=3, type=13
        assert!(result.is_err());
    }

    // --- NalUnitType TryFrom tests ---

    #[test]
    fn nal_unit_type_valid_values() {
        assert_eq!(NalUnitType::try_from(1).unwrap(), NalUnitType::Slice);
        assert_eq!(NalUnitType::try_from(5).unwrap(), NalUnitType::Idr);
        assert_eq!(NalUnitType::try_from(7).unwrap(), NalUnitType::Sps);
        assert_eq!(NalUnitType::try_from(8).unwrap(), NalUnitType::Pps);
        assert_eq!(NalUnitType::try_from(12).unwrap(), NalUnitType::Filler);
    }

    #[test]
    fn nal_unit_type_invalid_values() {
        assert!(NalUnitType::try_from(0).is_err());
        assert!(NalUnitType::try_from(13).is_err());
        assert!(NalUnitType::try_from(31).is_err());
        assert!(NalUnitType::try_from(255).is_err());
    }

    // --- Annex B split tests ---

    #[test]
    fn annex_b_3byte_start_code() {
        // 3-byte start code + SPS NAL (header 0x67) + some data
        let data = [0x00, 0x00, 0x01, 0x67, 0xAA, 0xBB];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
        assert_eq!(nalus[0].nal_ref_idc, 3);
        assert_eq!(nalus[0].data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn annex_b_4byte_start_code() {
        // 4-byte start code + PPS NAL (header 0x68) + some data
        let data = [0x00, 0x00, 0x00, 0x01, 0x68, 0xCC, 0xDD];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Pps);
        assert_eq!(nalus[0].nal_ref_idc, 3);
        assert_eq!(nalus[0].data, vec![0xCC, 0xDD]);
    }

    #[test]
    fn annex_b_multiple_nalus() {
        // SPS + PPS + IDR, mixed 3-byte and 4-byte start codes
        #[rustfmt::skip]
        let data = [
            // 4-byte start code + SPS
            0x00, 0x00, 0x00, 0x01, 0x67, 0x01,
            // 3-byte start code + PPS
            0x00, 0x00, 0x01, 0x68, 0x02,
            // 4-byte start code + IDR
            0x00, 0x00, 0x00, 0x01, 0x65, 0x03,
        ];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 3);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
        assert_eq!(nalus[0].data, vec![0x01]);
        assert_eq!(nalus[1].nal_type, NalUnitType::Pps);
        assert_eq!(nalus[1].data, vec![0x02]);
        assert_eq!(nalus[2].nal_type, NalUnitType::Idr);
        assert_eq!(nalus[2].data, vec![0x03]);
    }

    #[test]
    fn annex_b_with_epb() {
        // Start code + SPS NAL with EPB inside
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x01,       // start code
            0x67,                     // SPS header
            0x00, 0x00, 0x03, 0x00,  // EPB: 0x00 0x00 0x03 0x00 -> 0x00 0x00 0x00
            0xFF,
        ];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
        assert_eq!(nalus[0].data, vec![0x00, 0x00, 0x00, 0xFF]);
    }

    #[test]
    fn annex_b_empty_input() {
        let nalus = split_annex_b(&[]);
        assert!(nalus.is_empty());
    }

    #[test]
    fn annex_b_no_start_code() {
        let nalus = split_annex_b(&[0x67, 0xAA, 0xBB]);
        assert!(nalus.is_empty());
    }

    #[test]
    fn annex_b_start_code_at_end_no_data() {
        // Start code with nothing after it
        let nalus = split_annex_b(&[0x00, 0x00, 0x01]);
        assert!(nalus.is_empty());
    }

    #[test]
    fn annex_b_consecutive_start_codes() {
        // Two start codes with nothing between them, then a valid NAL
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x01,        // start code 1 (no data follows)
            0x00, 0x00, 0x01,        // start code 2
            0x67, 0xAA,             // SPS NAL
        ];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
    }

    #[test]
    fn annex_b_leading_zeros() {
        // Some leading zeros before the start code
        let data = [0x00, 0x00, 0x00, 0x00, 0x01, 0x67, 0xAA];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
    }

    #[test]
    fn annex_b_skips_unknown_nal_types() {
        // NAL type 0 (unspecified) should be skipped
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x01, 0x60, 0xAA,  // type=0, ref_idc=3 -> skipped
            0x00, 0x00, 0x01, 0x67, 0xBB,  // SPS -> kept
        ];
        let nalus = split_annex_b(&data);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
    }

    // --- NALFF split tests ---

    #[test]
    fn nalff_4byte_length() {
        // 4-byte length prefix (big-endian) + SPS NAL
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x00, 0x03,  // length = 3
            0x67, 0xAA, 0xBB,        // SPS header + data
        ];
        let nalus = split_nalff(&data, 4);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
        assert_eq!(nalus[0].data, vec![0xAA, 0xBB]);
    }

    #[test]
    fn nalff_2byte_length() {
        #[rustfmt::skip]
        let data = [
            0x00, 0x03,        // length = 3
            0x68, 0xCC, 0xDD,  // PPS header + data
        ];
        let nalus = split_nalff(&data, 2);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Pps);
        assert_eq!(nalus[0].data, vec![0xCC, 0xDD]);
    }

    #[test]
    fn nalff_1byte_length() {
        #[rustfmt::skip]
        let data = [
            0x02,        // length = 2
            0x06, 0xFF,  // SEI header + data
        ];
        let nalus = split_nalff(&data, 1);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sei);
        assert_eq!(nalus[0].data, vec![0xFF]);
    }

    #[test]
    fn nalff_multiple_nalus() {
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x00, 0x02,  // length = 2
            0x67, 0x01,              // SPS
            0x00, 0x00, 0x00, 0x02,  // length = 2
            0x68, 0x02,              // PPS
        ];
        let nalus = split_nalff(&data, 4);
        assert_eq!(nalus.len(), 2);
        assert_eq!(nalus[0].nal_type, NalUnitType::Sps);
        assert_eq!(nalus[1].nal_type, NalUnitType::Pps);
    }

    #[test]
    fn nalff_with_epb() {
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x00, 0x06,                // length = 6
            0x67,                                    // SPS header
            0x00, 0x00, 0x03, 0x01,                 // EPB -> 0x00 0x00 0x01
            0xEE,
        ];
        let nalus = split_nalff(&data, 4);
        assert_eq!(nalus.len(), 1);
        assert_eq!(nalus[0].data, vec![0x00, 0x00, 0x01, 0xEE]);
    }

    #[test]
    fn nalff_empty_input() {
        let nalus = split_nalff(&[], 4);
        assert!(nalus.is_empty());
    }

    #[test]
    fn nalff_truncated_length() {
        // Only 2 bytes but length_size=4
        let nalus = split_nalff(&[0x00, 0x00], 4);
        assert!(nalus.is_empty());
    }

    #[test]
    fn nalff_truncated_nal() {
        // Length says 10 bytes but only 3 are available
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x00, 0x0A,  // length = 10
            0x67, 0xAA, 0xBB,        // only 3 bytes
        ];
        let nalus = split_nalff(&data, 4);
        assert!(nalus.is_empty());
    }

    #[test]
    fn nalff_zero_length() {
        // Zero-length NAL should be skipped
        #[rustfmt::skip]
        let data = [
            0x00, 0x00, 0x00, 0x00,  // length = 0
            0x67, 0xAA,
        ];
        let nalus = split_nalff(&data, 4);
        assert!(nalus.is_empty());
    }
}
