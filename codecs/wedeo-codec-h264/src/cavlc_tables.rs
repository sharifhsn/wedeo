// CAVLC (Context-Adaptive Variable-Length Coding) lookup tables and VLC readers.
//
// Implements the VLC decoding functions for coeff_token, total_zeros, and
// run_before as specified in ITU-T H.264 Tables 9-5 through 9-10.
//
// The tables are transcribed from the H.264 spec and cross-referenced against
// FFmpeg's h264_cavlc.c (coeff_token_len/bits, total_zeros_len/bits, run_len/bits).
//
// Rather than building VLC lookup tables at init time (FFmpeg's approach), we use
// the (len, bits) table pairs directly: peek N bits, scan for a match. This is
// simpler and avoids global mutable state. Performance can be improved later with
// precomputed lookup tables if profiling shows this is a bottleneck.

use wedeo_codec::bitstream::{BitRead, BitReadBE};
use wedeo_core::error::{Error, Result};

// ---------------------------------------------------------------------------
// coeff_token tables (H.264 spec Table 9-5)
// ---------------------------------------------------------------------------
//
// Four nC contexts plus chroma DC. Each table is indexed as [total_coeff * 4 + trailing_ones].
// coeff_token_len[ctx][i] = bit length, coeff_token_bits[ctx][i] = codeword value.
// total_coeff ranges from 0..16 (17 entries), trailing_ones from 0..3 (4 entries).
// Entry layout: row = total_coeff, column = trailing_ones.

/// Codeword lengths for coeff_token, nC 0-1 (Table 9-5(a)).
/// Indexed as [total_coeff * 4 + trailing_ones].
const COEFF_TOKEN_LEN_0: [u8; 4 * 17] = [
    1, 0, 0, 0, 6, 2, 0, 0, 8, 6, 3, 0, 9, 8, 7, 5, 10, 9, 8, 6, 11, 10, 9, 7, 13, 11, 10, 8, 13,
    13, 11, 9, 13, 13, 13, 10, 14, 14, 13, 11, 14, 14, 14, 13, 15, 15, 14, 14, 15, 15, 15, 14, 16,
    15, 15, 15, 16, 16, 16, 15, 16, 16, 16, 16, 16, 16, 16, 16,
];

/// Codeword values for coeff_token, nC 0-1.
const COEFF_TOKEN_BITS_0: [u8; 4 * 17] = [
    1, 0, 0, 0, 5, 1, 0, 0, 7, 4, 1, 0, 7, 6, 5, 3, 7, 6, 5, 3, 7, 6, 5, 4, 15, 6, 5, 4, 11, 14, 5,
    4, 8, 10, 13, 4, 15, 14, 9, 4, 11, 10, 13, 12, 15, 14, 9, 12, 11, 10, 13, 8, 15, 1, 9, 12, 11,
    14, 13, 8, 7, 10, 9, 12, 4, 6, 5, 8,
];

/// Codeword lengths for coeff_token, nC 2-3 (Table 9-5(b)).
const COEFF_TOKEN_LEN_1: [u8; 4 * 17] = [
    2, 0, 0, 0, 6, 2, 0, 0, 6, 5, 3, 0, 7, 6, 6, 4, 8, 6, 6, 4, 8, 7, 7, 5, 9, 8, 8, 6, 11, 9, 9,
    6, 11, 11, 11, 7, 12, 11, 11, 9, 12, 12, 12, 11, 12, 12, 12, 11, 13, 13, 13, 12, 13, 13, 13,
    13, 13, 14, 13, 13, 14, 14, 14, 13, 14, 14, 14, 14,
];

/// Codeword values for coeff_token, nC 2-3.
const COEFF_TOKEN_BITS_1: [u8; 4 * 17] = [
    3, 0, 0, 0, 11, 2, 0, 0, 7, 7, 3, 0, 7, 10, 9, 5, 7, 6, 5, 4, 4, 6, 5, 6, 7, 6, 5, 8, 15, 6, 5,
    4, 11, 14, 13, 4, 15, 10, 9, 4, 11, 14, 13, 12, 8, 10, 9, 8, 15, 14, 13, 12, 11, 10, 9, 12, 7,
    11, 6, 8, 9, 8, 10, 1, 7, 6, 5, 4,
];

/// Codeword lengths for coeff_token, nC 4-7 (Table 9-5(c)).
const COEFF_TOKEN_LEN_2: [u8; 4 * 17] = [
    4, 0, 0, 0, 6, 4, 0, 0, 6, 5, 4, 0, 6, 5, 5, 4, 7, 5, 5, 4, 7, 5, 5, 4, 7, 6, 6, 4, 7, 6, 6, 4,
    8, 7, 7, 5, 8, 8, 7, 6, 9, 8, 8, 7, 9, 9, 8, 8, 9, 9, 9, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10,
];

/// Codeword values for coeff_token, nC 4-7.
const COEFF_TOKEN_BITS_2: [u8; 4 * 17] = [
    15, 0, 0, 0, 15, 14, 0, 0, 11, 15, 13, 0, 8, 12, 14, 12, 15, 10, 11, 11, 11, 8, 9, 10, 9, 14,
    13, 9, 8, 10, 9, 8, 15, 14, 13, 13, 11, 14, 10, 12, 15, 10, 13, 12, 11, 14, 9, 12, 8, 10, 13,
    8, 13, 7, 9, 12, 9, 12, 11, 10, 5, 8, 7, 6, 1, 4, 3, 2,
];

/// Codeword lengths for coeff_token, nC >= 8 (Table 9-5(d)).
/// Fixed 6-bit codes.
const COEFF_TOKEN_LEN_3: [u8; 4 * 17] = [
    6, 0, 0, 0, 6, 6, 0, 0, 6, 6, 6, 0, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6,
    6, 6, 6, 6,
];

/// Codeword values for coeff_token, nC >= 8.
const COEFF_TOKEN_BITS_3: [u8; 4 * 17] = [
    3, 0, 0, 0, 0, 1, 0, 0, 4, 5, 6, 0, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
];

/// All four coeff_token length tables, indexed by context 0..3.
const COEFF_TOKEN_LEN: [&[u8; 68]; 4] = [
    &COEFF_TOKEN_LEN_0,
    &COEFF_TOKEN_LEN_1,
    &COEFF_TOKEN_LEN_2,
    &COEFF_TOKEN_LEN_3,
];

/// All four coeff_token bit tables, indexed by context 0..3.
const COEFF_TOKEN_BITS: [&[u8; 68]; 4] = [
    &COEFF_TOKEN_BITS_0,
    &COEFF_TOKEN_BITS_1,
    &COEFF_TOKEN_BITS_2,
    &COEFF_TOKEN_BITS_3,
];

/// Maps nC value (0..16) to coeff_token table index (0..3).
/// From FFmpeg's `coeff_token_table_index`.
const COEFF_TOKEN_TABLE_INDEX: [u8; 17] = [0, 0, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3];

// ---------------------------------------------------------------------------
// Chroma DC coeff_token tables (H.264 spec Table 9-5(e) for 4:2:0)
// ---------------------------------------------------------------------------

/// Codeword lengths for chroma DC coeff_token (4:2:0, max 4 coeffs).
const CHROMA_DC_COEFF_TOKEN_LEN: [u8; 4 * 5] =
    [2, 0, 0, 0, 6, 1, 0, 0, 6, 6, 3, 0, 6, 7, 7, 6, 6, 8, 8, 7];

/// Codeword values for chroma DC coeff_token (4:2:0).
const CHROMA_DC_COEFF_TOKEN_BITS: [u8; 4 * 5] =
    [1, 0, 0, 0, 7, 1, 0, 0, 4, 6, 1, 0, 3, 3, 2, 5, 2, 3, 2, 0];

// ---------------------------------------------------------------------------
// total_zeros tables (H.264 spec Tables 9-7, 9-8)
// ---------------------------------------------------------------------------

/// Codeword lengths for total_zeros (4x4 blocks), indexed by [total_coeff - 1][total_zeros].
const TOTAL_ZEROS_LEN: [[u8; 16]; 15] = [
    [1, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 9],
    [3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 6, 6, 6, 6, 0],
    [4, 3, 3, 3, 4, 4, 3, 3, 4, 5, 5, 6, 5, 6, 0, 0],
    [5, 3, 4, 4, 3, 3, 3, 4, 3, 4, 5, 5, 5, 0, 0, 0],
    [4, 4, 4, 3, 3, 3, 3, 3, 4, 5, 4, 5, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 3, 3, 3, 4, 3, 6, 0, 0, 0, 0, 0],
    [6, 5, 3, 3, 3, 2, 3, 4, 3, 6, 0, 0, 0, 0, 0, 0],
    [6, 4, 5, 3, 2, 2, 3, 3, 6, 0, 0, 0, 0, 0, 0, 0],
    [6, 6, 4, 2, 2, 3, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0],
    [5, 5, 3, 2, 2, 2, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 3, 3, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 4, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

/// Codeword values for total_zeros (4x4 blocks).
const TOTAL_ZEROS_BITS: [[u8; 16]; 15] = [
    [1, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 1],
    [7, 6, 5, 4, 3, 5, 4, 3, 2, 3, 2, 3, 2, 1, 0, 0],
    [5, 7, 6, 5, 4, 3, 4, 3, 2, 3, 2, 1, 1, 0, 0, 0],
    [3, 7, 5, 4, 6, 5, 4, 3, 3, 2, 2, 1, 0, 0, 0, 0],
    [5, 4, 3, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0],
    [1, 1, 7, 6, 5, 4, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 5, 4, 3, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 3, 3, 2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 1, 3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 2, 1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
];

// Number of valid total_zeros values for each total_coeff (1..15).
// For total_coeff = k, total_zeros can be 0..(16-k), so count = 16 - k + 1 = 17 - k.
// But the table rows have exactly the right number of non-zero-length entries.
const TOTAL_ZEROS_COUNT: [u8; 15] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2];

// ---------------------------------------------------------------------------
// Chroma DC total_zeros tables (H.264 spec Table 9-9(a) for 4:2:0)
// ---------------------------------------------------------------------------

/// Codeword lengths for chroma DC total_zeros (4:2:0, max 4 coeffs).
const CHROMA_DC_TOTAL_ZEROS_LEN: [[u8; 4]; 3] = [[1, 2, 3, 3], [1, 2, 2, 0], [1, 1, 0, 0]];

/// Codeword values for chroma DC total_zeros (4:2:0).
const CHROMA_DC_TOTAL_ZEROS_BITS: [[u8; 4]; 3] = [[1, 1, 1, 0], [1, 1, 0, 0], [1, 0, 0, 0]];

const CHROMA_DC_TOTAL_ZEROS_COUNT: [u8; 3] = [4, 3, 2];

// ---------------------------------------------------------------------------
// run_before tables (H.264 spec Table 9-10)
// ---------------------------------------------------------------------------

/// Codeword lengths for run_before, indexed by [min(zeros_left, 7) - 1][run_before].
const RUN_BEFORE_LEN: [[u8; 16]; 7] = [
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 3, 3, 3, 3, 3, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0],
];

/// Codeword values for run_before.
const RUN_BEFORE_BITS: [[u8; 16]; 7] = [
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 2, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [3, 0, 1, 3, 2, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [7, 6, 5, 4, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
];

// Maximum run_before value for each zeros_left context.
const RUN_BEFORE_COUNT: [u8; 7] = [2, 3, 4, 5, 6, 7, 15];

// ---------------------------------------------------------------------------
// Generic VLC decoder
// ---------------------------------------------------------------------------

/// Decode a VLC symbol given (len, bits) table arrays.
///
/// Peeks up to `max_bits` from the bitstream, then scans the table to find
/// the matching codeword. Returns the table index of the match.
///
/// This is a straightforward O(n) scan. For the small tables used in CAVLC
/// this is fast enough; a precomputed lookup table can replace this later.
fn decode_vlc(
    br: &mut BitReadBE<'_>,
    lens: &[u8],
    bits: &[u8],
    count: usize,
    max_bits: u32,
) -> Result<usize> {
    let peeked = br.peek_bits_32(max_bits as usize);

    for i in 0..count {
        let len = lens[i];
        if len == 0 {
            continue;
        }
        let code = bits[i] as u32;
        // Extract the top `len` bits from the peeked value.
        let shift = max_bits - len as u32;
        if (peeked >> shift) == code {
            br.skip_bits(len as usize);
            return Ok(i);
        }
    }

    Err(Error::InvalidData)
}

// ---------------------------------------------------------------------------
// Public API: coeff_token
// ---------------------------------------------------------------------------

/// Read coeff_token for the given nC context.
///
/// Returns `(total_coeff, trailing_ones)`.
///
/// nC selection (H.264 spec 9.2.1):
/// - nC = -1: chroma DC (4:2:0, 2x2 block)
/// - nC = -2: chroma DC (4:2:2, 2x4 block) — not yet supported
/// - nC 0..16: luma / chroma AC, mapped to table index via COEFF_TOKEN_TABLE_INDEX
pub fn read_coeff_token(br: &mut BitReadBE<'_>, nc: i32) -> Result<(u8, u8)> {
    if nc == -1 {
        // Chroma DC 4:2:0
        let max_bits = 8;
        let idx = decode_vlc(
            br,
            &CHROMA_DC_COEFF_TOKEN_LEN,
            &CHROMA_DC_COEFF_TOKEN_BITS,
            4 * 5,
            max_bits,
        )?;
        let total_coeff = (idx / 4) as u8;
        let trailing_ones = (idx % 4) as u8;
        return Ok((total_coeff, trailing_ones));
    }

    if nc < -1 {
        // Chroma DC 4:2:2 — not supported for now (Baseline is 4:2:0 only).
        return Err(Error::InvalidData);
    }

    let nc_clamped = nc.min(16) as usize;
    let table_idx = COEFF_TOKEN_TABLE_INDEX[nc_clamped] as usize;
    let lens = COEFF_TOKEN_LEN[table_idx];
    let bits_tab = COEFF_TOKEN_BITS[table_idx];

    // Maximum codeword length per table: 16, 14, 10, 6
    let max_bits: u32 = match table_idx {
        0 => 16,
        1 => 14,
        2 => 10,
        3 => 6,
        _ => unreachable!(),
    };

    let idx = decode_vlc(br, lens, bits_tab, 4 * 17, max_bits)?;
    let total_coeff = (idx / 4) as u8;
    let trailing_ones = (idx % 4) as u8;
    Ok((total_coeff, trailing_ones))
}

// ---------------------------------------------------------------------------
// Public API: total_zeros
// ---------------------------------------------------------------------------

/// Read total_zeros for a 4x4 block.
///
/// `total_coeff` must be 1..15.
pub fn read_total_zeros(br: &mut BitReadBE<'_>, total_coeff: u8) -> Result<u8> {
    if total_coeff == 0 || total_coeff > 15 {
        return Err(Error::InvalidData);
    }

    let table_idx = (total_coeff - 1) as usize;
    let lens = &TOTAL_ZEROS_LEN[table_idx];
    let bits_tab = &TOTAL_ZEROS_BITS[table_idx];
    let count = TOTAL_ZEROS_COUNT[table_idx] as usize;

    // Max codeword length in total_zeros tables is 9.
    let max_bits = 9;

    let val = decode_vlc(br, lens, bits_tab, count, max_bits)?;
    Ok(val as u8)
}

/// Read total_zeros for a 2x2 chroma DC block (4:2:0).
///
/// `total_coeff` must be 1..3.
pub fn read_total_zeros_chroma_dc(br: &mut BitReadBE<'_>, total_coeff: u8) -> Result<u8> {
    if total_coeff == 0 || total_coeff > 3 {
        return Err(Error::InvalidData);
    }

    let table_idx = (total_coeff - 1) as usize;
    let lens = &CHROMA_DC_TOTAL_ZEROS_LEN[table_idx];
    let bits_tab = &CHROMA_DC_TOTAL_ZEROS_BITS[table_idx];
    let count = CHROMA_DC_TOTAL_ZEROS_COUNT[table_idx] as usize;

    let max_bits = 3;

    let val = decode_vlc(br, lens, bits_tab, count, max_bits)?;
    Ok(val as u8)
}

// ---------------------------------------------------------------------------
// Public API: run_before
// ---------------------------------------------------------------------------

/// Read run_before value.
///
/// `zeros_left` must be >= 1.
pub fn read_run_before(br: &mut BitReadBE<'_>, zeros_left: u8) -> Result<u8> {
    if zeros_left == 0 {
        return Err(Error::InvalidData);
    }

    let table_idx = (zeros_left.min(7) - 1) as usize;
    let lens = &RUN_BEFORE_LEN[table_idx];
    let bits_tab = &RUN_BEFORE_BITS[table_idx];
    let count = RUN_BEFORE_COUNT[table_idx] as usize;

    // Max codeword length: 11 for zeros_left >= 7, otherwise <= 3.
    let max_bits: u32 = if zeros_left >= 7 { 11 } else { 3 };

    let val = decode_vlc(br, lens, bits_tab, count, max_bits)?;
    Ok(val as u8)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a BitReadBE from a byte slice (padded to 8 bytes for cache safety).
    fn make_reader(data: &[u8]) -> BitReadBE<'_> {
        BitReadBE::new(data)
    }

    /// Build a byte array from a binary string like "10110000".
    fn bits_to_bytes(bits_str: &str) -> Vec<u8> {
        let mut bytes = Vec::new();
        let chars: Vec<char> = bits_str.chars().collect();
        // Pad to multiple of 8
        let padded_len = ((chars.len() + 7) / 8) * 8;
        let mut padded = chars.clone();
        padded.resize(padded_len, '0');
        // Add 8 bytes of padding for bitstream reader
        padded.resize(padded_len + 64, '0');

        for chunk in padded.chunks(8) {
            let mut byte = 0u8;
            for (j, &c) in chunk.iter().enumerate() {
                if c == '1' {
                    byte |= 1 << (7 - j);
                }
            }
            bytes.push(byte);
        }
        bytes
    }

    // --- coeff_token tests ---

    #[test]
    fn coeff_token_nc0_total0() {
        // nC=0: total_coeff=0, trailing_ones=0 => len=1, bits=1 => "1"
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 0).unwrap();
        assert_eq!((tc, to), (0, 0));
    }

    #[test]
    fn coeff_token_nc0_total1_trailing1() {
        // nC=0: total_coeff=1, trailing_ones=1 => len=2, bits=1 => "01"
        let data = bits_to_bytes("01");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 0).unwrap();
        assert_eq!((tc, to), (1, 1));
    }

    #[test]
    fn coeff_token_nc0_total1_trailing0() {
        // nC=0: total_coeff=1, trailing_ones=0 => len=6, bits=5 => "000101"
        let data = bits_to_bytes("000101");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 0).unwrap();
        assert_eq!((tc, to), (1, 0));
    }

    #[test]
    fn coeff_token_nc0_total2_trailing2() {
        // nC=0: total_coeff=2, trailing_ones=2 => len=3, bits=1 => "001"
        let data = bits_to_bytes("001");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 0).unwrap();
        assert_eq!((tc, to), (2, 2));
    }

    #[test]
    fn coeff_token_nc8_fixed_length() {
        // nC=8 (table 3): total_coeff=0, trailing_ones=0 => len=6, bits=3 => "000011"
        let data = bits_to_bytes("000011");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 8).unwrap();
        assert_eq!((tc, to), (0, 0));
    }

    #[test]
    fn coeff_token_nc8_total3_trailing2() {
        // nC=8 (table 3): total_coeff=3, trailing_ones=2 => index=3*4+2=14
        // len=6, bits=10 => "001010"
        let data = bits_to_bytes("001010");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, 8).unwrap();
        assert_eq!((tc, to), (3, 2));
    }

    #[test]
    fn coeff_token_chroma_dc_total0() {
        // Chroma DC (nC=-1): total_coeff=0, trailing_ones=0 => len=2, bits=1 => "01"
        let data = bits_to_bytes("01");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, -1).unwrap();
        assert_eq!((tc, to), (0, 0));
    }

    #[test]
    fn coeff_token_chroma_dc_total1_trailing1() {
        // Chroma DC (nC=-1): total_coeff=1, trailing_ones=1 => len=1, bits=1 => "1"
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, -1).unwrap();
        assert_eq!((tc, to), (1, 1));
    }

    #[test]
    fn coeff_token_chroma_dc_total2_trailing2() {
        // Chroma DC: total_coeff=2, trailing_ones=2 => len=3, bits=1 => "001"
        let data = bits_to_bytes("001");
        let mut br = make_reader(&data);
        let (tc, to) = read_coeff_token(&mut br, -1).unwrap();
        assert_eq!((tc, to), (2, 2));
    }

    // --- total_zeros tests ---

    #[test]
    fn total_zeros_tc1_tz0() {
        // total_coeff=1: total_zeros=0 => len=1, bits=1 => "1"
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        let tz = read_total_zeros(&mut br, 1).unwrap();
        assert_eq!(tz, 0);
    }

    #[test]
    fn total_zeros_tc1_tz1() {
        // total_coeff=1: total_zeros=1 => len=3, bits=3 => "011"
        let data = bits_to_bytes("011");
        let mut br = make_reader(&data);
        let tz = read_total_zeros(&mut br, 1).unwrap();
        assert_eq!(tz, 1);
    }

    #[test]
    fn total_zeros_tc1_tz15() {
        // total_coeff=1: total_zeros=15 => len=9, bits=1 => "000000001"
        let data = bits_to_bytes("000000001");
        let mut br = make_reader(&data);
        let tz = read_total_zeros(&mut br, 1).unwrap();
        assert_eq!(tz, 15);
    }

    #[test]
    fn total_zeros_chroma_dc_tc1_tz0() {
        // chroma DC total_coeff=1: total_zeros=0 => len=1, bits=1 => "1"
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        let tz = read_total_zeros_chroma_dc(&mut br, 1).unwrap();
        assert_eq!(tz, 0);
    }

    #[test]
    fn total_zeros_chroma_dc_tc1_tz3() {
        // chroma DC total_coeff=1: total_zeros=3 => len=3, bits=0 => "000"
        let data = bits_to_bytes("000");
        let mut br = make_reader(&data);
        let tz = read_total_zeros_chroma_dc(&mut br, 1).unwrap();
        assert_eq!(tz, 3);
    }

    // --- run_before tests ---

    #[test]
    fn run_before_zl1_rb0() {
        // zeros_left=1: run_before=0 => len=1, bits=1 => "1"
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 1).unwrap();
        assert_eq!(rb, 0);
    }

    #[test]
    fn run_before_zl1_rb1() {
        // zeros_left=1: run_before=1 => len=1, bits=0 => "0"
        let data = bits_to_bytes("0");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 1).unwrap();
        assert_eq!(rb, 1);
    }

    #[test]
    fn run_before_zl3_rb0() {
        // zeros_left=3: run_before=0 => len=2, bits=3 => "11"
        let data = bits_to_bytes("11");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 3).unwrap();
        assert_eq!(rb, 0);
    }

    #[test]
    fn run_before_zl3_rb3() {
        // zeros_left=3: run_before=3 => len=2, bits=0 => "00"
        let data = bits_to_bytes("00");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 3).unwrap();
        assert_eq!(rb, 3);
    }

    #[test]
    fn run_before_zl7_rb0() {
        // zeros_left>=7: run_before=0 => len=3, bits=7 => "111"
        let data = bits_to_bytes("111");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 7).unwrap();
        assert_eq!(rb, 0);
    }

    #[test]
    fn run_before_zl7_rb7() {
        // zeros_left>=7: run_before=7 => len=4, bits=1 => "0001"
        let data = bits_to_bytes("0001");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 7).unwrap();
        assert_eq!(rb, 7);
    }

    #[test]
    fn run_before_zl7_rb14() {
        // zeros_left>=7: run_before=14 => len=11, bits=1 => "00000000001"
        let data = bits_to_bytes("00000000001");
        let mut br = make_reader(&data);
        let rb = read_run_before(&mut br, 10).unwrap();
        assert_eq!(rb, 14);
    }

    // --- Error cases ---

    #[test]
    fn total_zeros_invalid_total_coeff() {
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        assert!(read_total_zeros(&mut br, 0).is_err());
        assert!(read_total_zeros(&mut br, 16).is_err());
    }

    #[test]
    fn run_before_invalid_zeros_left() {
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        assert!(read_run_before(&mut br, 0).is_err());
    }

    #[test]
    fn chroma_dc_total_zeros_invalid() {
        let data = bits_to_bytes("1");
        let mut br = make_reader(&data);
        assert!(read_total_zeros_chroma_dc(&mut br, 0).is_err());
        assert!(read_total_zeros_chroma_dc(&mut br, 4).is_err());
    }

    // --- Exhaustive round-trip: verify every coeff_token entry is decodable ---

    #[test]
    fn coeff_token_all_entries_table0() {
        verify_all_coeff_token_entries(0, &COEFF_TOKEN_LEN_0, &COEFF_TOKEN_BITS_0, 17);
    }

    #[test]
    fn coeff_token_all_entries_table1() {
        verify_all_coeff_token_entries(1, &COEFF_TOKEN_LEN_1, &COEFF_TOKEN_BITS_1, 17);
    }

    #[test]
    fn coeff_token_all_entries_table2() {
        verify_all_coeff_token_entries(2, &COEFF_TOKEN_LEN_2, &COEFF_TOKEN_BITS_2, 17);
    }

    #[test]
    fn coeff_token_all_entries_table3() {
        verify_all_coeff_token_entries(3, &COEFF_TOKEN_LEN_3, &COEFF_TOKEN_BITS_3, 17);
    }

    /// Verify that every valid (len != 0) entry in a coeff_token table can be
    /// decoded back to the correct (total_coeff, trailing_ones).
    fn verify_all_coeff_token_entries(
        table_idx: usize,
        lens: &[u8; 68],
        bits: &[u8; 68],
        num_tc: usize,
    ) {
        // Map table_idx to an nC value that selects this table.
        let nc: i32 = match table_idx {
            0 => 0,
            1 => 2,
            2 => 4,
            3 => 8,
            _ => unreachable!(),
        };

        for tc in 0..num_tc {
            for to in 0..4u8 {
                let idx = tc * 4 + to as usize;
                let len = lens[idx];
                if len == 0 {
                    continue; // Invalid entry (trailing_ones > total_coeff)
                }
                let code = bits[idx] as u32;

                // Build a bitstream with this codeword left-aligned.
                let mut data = [0u8; 16];
                // Place the code at the MSB of a u32.
                let shifted = code << (32 - len);
                data[0] = (shifted >> 24) as u8;
                data[1] = (shifted >> 16) as u8;
                data[2] = (shifted >> 8) as u8;
                data[3] = shifted as u8;

                let mut br = make_reader(&data);
                let result = read_coeff_token(&mut br, nc);
                assert!(
                    result.is_ok(),
                    "Failed to decode coeff_token table={table_idx} tc={tc} to={to} len={len} code={code:#b}"
                );
                let (dec_tc, dec_to) = result.unwrap();
                assert_eq!(
                    (dec_tc, dec_to),
                    (tc as u8, to),
                    "Mismatch for coeff_token table={table_idx} tc={tc} to={to}"
                );
            }
        }
    }

    #[test]
    fn coeff_token_chroma_dc_all_entries() {
        for tc in 0..5u8 {
            for to in 0..4u8 {
                let idx = tc as usize * 4 + to as usize;
                let len = CHROMA_DC_COEFF_TOKEN_LEN[idx];
                if len == 0 {
                    continue;
                }
                let code = CHROMA_DC_COEFF_TOKEN_BITS[idx] as u32;

                let mut data = [0u8; 16];
                let shifted = code << (32 - len);
                data[0] = (shifted >> 24) as u8;
                data[1] = (shifted >> 16) as u8;
                data[2] = (shifted >> 8) as u8;
                data[3] = shifted as u8;

                let mut br = make_reader(&data);
                let (dec_tc, dec_to) = read_coeff_token(&mut br, -1).unwrap();
                assert_eq!(
                    (dec_tc, dec_to),
                    (tc, to),
                    "Mismatch for chroma DC coeff_token tc={tc} to={to}"
                );
            }
        }
    }
}
