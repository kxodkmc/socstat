//! Data records: uncompressed / bytecode / zlib element decoding and
//! bytecode encoding (compressions 0/1/2 on read, compression 1 on write).
//!
//! Bytecode (compression 1) layout: data is a stream of 8-byte command
//! blocks. Each block holds 8 one-byte command codes followed by an 8-byte
//! literal for every `253` in the block, in order of appearance.

use flate2::{Decompress, FlushDecompress, Status};

use crate::error::SocStatResult;

use super::{Cursor, sav_err, truncated};

/// One decoded 8-byte element.
#[derive(Debug, Clone, Copy)]
pub(super) enum RawElement {
    Numeric(f64),
    Bytes([u8; 8]),
}

/// One encoded element for the writer.
#[derive(Debug, Clone, Copy)]
pub(super) struct Unit {
    pub cmd: u8,
    pub lit: Option<[u8; 8]>,
}

impl Unit {
    pub fn cmd(c: u8) -> Self {
        Self { cmd: c, lit: None }
    }

    pub fn lit(c: u8, lit: [u8; 8]) -> Self {
        Self { cmd: c, lit: Some(lit) }
    }
}

// -------------------------------------------------------------------------
// Compression 0 (uncompressed): 8-byte elements read directly.
// -------------------------------------------------------------------------

pub(super) struct FlatDecoder {
    data: Vec<u8>,
    pos: usize,
    le: bool,
}

impl FlatDecoder {
    pub fn new(data: Vec<u8>, le: bool) -> Self {
        Self { data, pos: 0, le }
    }

    pub fn next(&mut self, is_string: bool) -> SocStatResult<Option<RawElement>> {
        if self.pos + 8 > self.data.len() {
            return Ok(None);
        }
        let b: [u8; 8] = self.data[self.pos..self.pos + 8].try_into().unwrap();
        self.pos += 8;
        Ok(Some(if is_string {
            RawElement::Bytes(b)
        } else {
            RawElement::Numeric(super::bytes_to_f64(&b, self.le))
        }))
    }
}

// -------------------------------------------------------------------------
// Compression 1 (bytecode).
// -------------------------------------------------------------------------

/// Streaming bytecode decoder.
///
/// Command blocks are loaded on demand; the literals belonging to a block are
/// buffered right after the block's 8 command bytes and consumed in order.
pub(super) struct ByteDecoder {
    data: Vec<u8>,
    pos: usize,
    le: bool,
    bias: f64,
    block: [u8; 8],
    block_idx: usize, // == 8 → a new block must be loaded
    literals: Vec<[u8; 8]>,
    lit_idx: usize,
    done: bool,
}

impl ByteDecoder {
    pub fn new(data: Vec<u8>, le: bool, bias: f64) -> Self {
        Self {
            data,
            pos: 0,
            le,
            bias,
            block: [0; 8],
            block_idx: 8,
            literals: Vec::new(),
            lit_idx: 0,
            done: false,
        }
    }

    /// Decode the next element, or `None` at the bytecode EOF (code 252) or
    /// end of input.
    pub fn next(&mut self, is_string: bool) -> SocStatResult<Option<RawElement>> {
        if self.block_idx == 8 && (self.done || !self.load_block()?) {
            return Ok(None);
        }
        let code = self.block[self.block_idx];
        self.block_idx += 1;
        match code {
            252 => {
                self.done = true;
                Ok(None)
            }
            255 => {
                if is_string {
                    Err(sav_err("system-missing command (255) in a string element"))
                } else {
                    Ok(Some(RawElement::Numeric(super::header::SYSMIS)))
                }
            }
            254 => Ok(Some(RawElement::Bytes([b' '; 8]))),
            0 => Ok(Some(if is_string {
                RawElement::Bytes([0u8; 8]) // NUL byte(s) in a string
            } else {
                RawElement::Numeric(0.0)
            })),
            253 => {
                let raw = self.literals.get(self.lit_idx).copied().ok_or_else(|| {
                    sav_err("bytecode: literal command (253) with no buffered value")
                })?;
                self.lit_idx += 1;
                Ok(Some(if is_string {
                    RawElement::Bytes(raw)
                } else {
                    RawElement::Numeric(super::bytes_to_f64(&raw, self.le))
                }))
            }
            1..=251 => {
                if is_string {
                    Err(sav_err("bytecode: numeric command (1..=251) in a string element"))
                } else {
                    Ok(Some(RawElement::Numeric(code as f64 - self.bias)))
                }
            }
        }
    }

    /// Load the next 8 command bytes and their literal payload.
    fn load_block(&mut self) -> SocStatResult<bool> {
        if self.pos + 8 > self.data.len() {
            return Ok(false);
        }
        self.block.copy_from_slice(&self.data[self.pos..self.pos + 8]);
        self.pos += 8;
        self.block_idx = 0;
        let n = self.block.iter().filter(|&&c| c == 253).count();
        let need = n * 8;
        if self.pos + need > self.data.len() {
            return Err(truncated(self.pos, need));
        }
        self.literals.clear();
        self.lit_idx = 0;
        for i in 0..n {
            let start = self.pos + i * 8;
            self.literals.push(self.data[start..start + 8].try_into().unwrap());
        }
        self.pos += need;
        Ok(true)
    }
}

// -------------------------------------------------------------------------
// Compression 2 (zlib / `$FL3`): inflate the bytecode stream.
// -------------------------------------------------------------------------

/// Inflate the `$FL3` bytecode stream (zlib) from a data section.
///
/// The section starts with a 24-byte zheader (offsets), then compressed
/// zlib blocks, then a 32-byte trailer (`bias`, `0`, `block_size`,
/// `n_blocks`) followed by one 24-byte descriptor per block. `ztrailer_ofs`
/// is relative to the start of the data section; each block's
/// `compressed_ofs` is relative to the start of the zlib stream (right after
/// the zheader). Each block is decompressed independently and concatenated
/// into the bytecode stream used by [`ByteDecoder`].
pub(super) fn inflate_zlib(buf: &[u8], data_start: usize, le: bool) -> SocStatResult<Vec<u8>> {
    const ZHEADER_LEN: usize = 24;
    let mut c = Cursor::new(&buf[data_start..], le);
    c.read_i64()?; // zheader offset (typically 0; we are already there)
    let ztrailer_ofs = c.read_i64()?;
    c.read_i64()?; // ztrailer length
    c.seek(ztrailer_ofs as usize)?;

    c.read_i64()?; // bias stored in the trailer (header bias already validated)
    c.read_i64()?; // reserved (0)
    c.read_i64()?; // block size
    let n_blocks = c.read_i64()?;

    if !(0..=1_000_000).contains(&n_blocks) {
        return Err(sav_err(format!("implausible zlib block count {n_blocks}")));
    }

    let mut out = Vec::new();
    for _ in 0..n_blocks {
        c.read_i64()?; // uncompressed offset (cumulative; unused)
        let cmp_ofs = c.read_i64()?;
        let expected = c.read_i32()?; // uncompressed size
        let cmp_size = c.read_i32()?; // compressed size
        if cmp_size < 0 || expected < 0 {
            return Err(sav_err("zlib block descriptor carries a negative size"));
        }
        let start = data_start + ZHEADER_LEN + cmp_ofs as usize;
        if start + cmp_size as usize > buf.len() {
            return Err(truncated(data_start, ZHEADER_LEN + cmp_ofs as usize + cmp_size as usize));
        }
        let block = &buf[start..start + cmp_size as usize];
        out.extend_from_slice(&inflate_block(block, expected as usize)?);
    }
    Ok(out)
}

/// Decompress one RFC 1950 zlib block to exactly `expected` bytes.
fn inflate_block(compressed: &[u8], expected: usize) -> SocStatResult<Vec<u8>> {
    let mut dec = Decompress::new(true); // true = expect a zlib header
    let mut out = vec![0u8; expected];
    let status = dec
        .decompress(compressed, &mut out, FlushDecompress::None)
        .map_err(|e| sav_err(format!("zlib decompression failed: {e}")))?;
    if status != Status::StreamEnd {
        return Err(sav_err(format!(
            "zlib block ended prematurely (status {status:?})"
        )));
    }
    Ok(out)
}

// -------------------------------------------------------------------------
// Compression 1 writer.
// -------------------------------------------------------------------------

/// Encode bytecode units: chunk commands into 8-byte blocks, append each
/// block's literals after its command bytes, then emit the EOF marker.
pub(super) fn encode_bytecode(units: &[Unit]) -> Vec<u8> {
    let mut out = Vec::with_capacity(units.len() + 16);
    let mut idx = 0;
    while idx < units.len() {
        let n = (units.len() - idx).min(8);
        for u in &units[idx..idx + n] {
            out.push(u.cmd);
        }
        out.resize(out.len() + (8 - n), 0); // pad a partial block
        for u in &units[idx..idx + n] {
            if let Some(lit) = u.lit {
                out.extend_from_slice(&lit);
            }
        }
        idx += n;
    }
    out.extend_from_slice(&[252, 0, 0, 0, 0, 0, 0, 0]);
    out
}