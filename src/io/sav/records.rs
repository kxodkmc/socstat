//! Dictionary records: variable records, value labels, extension records,
//! and the `999` dictionary terminator.

use crate::data::{DataType, MissingSpec, Variable};
use crate::error::SocStatResult;

use super::header::OLD_LOWEST_BITS;
use super::{Cursor, f64_to_bytes, i32_to_bytes, sav_err, truncated};

/// A raw variable record (record types 1 and 2 share this layout).
pub(super) struct VariableEntry {
    /// Trimmed 8-byte short name.
    pub name: String,
    /// `0` = numeric, `>0` = string width, `-1` = continuation segment.
    pub width: i32,
    /// Raw label bytes, if any (decoded once the encoding is known).
    pub label: Option<Vec<u8>>,
    pub missing: MissingSpec,
}

/// A raw extension record (record type 7): subtype + opaque payload.
pub(super) struct ExtensionRecord {
    pub subtype: i32,
    pub data: Vec<u8>,
}

/// Parse a variable record (rec type 1/2). The record-type field is already
/// consumed by the caller.
pub(super) fn parse_variable(c: &mut Cursor) -> SocStatResult<VariableEntry> {
    let width = c.read_i32()?;
    let has_label = c.read_i32()?;
    let n_missing = c.read_i32()?;
    let _print = c.read_i32()?;
    let _write = c.read_i32()?;
    let name_slice = c.take(8)?;
    let name = trim_field(name_slice).to_string();

    let label = if has_label != 0 {
        let n = c.read_i32()?;
        if n < 0 {
            return Err(sav_err(format!("variable '{name}': negative label length {n}")));
        }
        let n = n as usize;
        let bytes = c.take(n)?;
        let pad = (4 - n % 4) % 4;
        c.skip(pad)?; // label is padded to a 4-byte boundary
        Some(bytes.to_vec())
    } else {
        None
    };

    let missing = parse_missing(c, n_missing)?;
    Ok(VariableEntry { name, width, label, missing })
}

/// Parse the `n_missing` field of a variable record.
///
/// `1..=3` are discrete values, `-2` a range, `-3` a range plus one discrete
/// value. The old (`0xffeffffffffffffe`) and modern (`-DBL_MAX`) encodings of
/// an unbounded low bound are both recognized and normalized.
fn parse_missing(c: &mut Cursor, n: i32) -> SocStatResult<MissingSpec> {
    match n {
        0 => Ok(MissingSpec::None),
        1..=3 => {
            let mut vals = Vec::with_capacity(n as usize);
            for _ in 0..n {
                vals.push(c.read_f64()?);
            }
            Ok(MissingSpec::Discrete(vals))
        }
        -2 | -3 => {
            let low = c.read_f64()?;
            let high = c.read_f64()?;
            let low = if low.to_bits() == OLD_LOWEST_BITS || low == -f64::MAX {
                -f64::MAX // unbounded low
            } else {
                low
            };
            let high = if high == f64::MAX { f64::MAX } else { high };
            let discrete = if n == -3 { Some(c.read_f64()?) } else { None };
            Ok(MissingSpec::Range { low, high, discrete })
        }
        other => Err(sav_err(format!("invalid missing-value count {other}"))),
    }
}

/// Parse value-label pairs of record type 3.
///
/// The record stores `(value[8], label_len, label[])` triples with `label`
/// padded to a multiple of 8 bytes, and has no explicit count: pairs run
/// until the next int32 is a recognized record type (record 4 normally
/// follows immediately).
pub(super) fn parse_value_labels(c: &mut Cursor) -> SocStatResult<Vec<(Vec<u8>, Vec<u8>)>> {
    let mut pairs = Vec::new();
    loop {
        if c.remaining() < 4 {
            break;
        }
        if is_record_type(c.peek_i32()?) {
            break;
        }
        let tag = c.take(8)?.to_vec();
        let n = c.read_i32()?;
        if n < 0 {
            return Err(sav_err(format!("negative value-label length {n}")));
        }
        let n = n as usize;
        let label = c.take(n)?.to_vec();
        let pad = (8 - n % 8) % 8;
        c.skip(pad)?; // label is padded to an 8-byte boundary
        pairs.push((tag, label));
    }
    Ok(pairs)
}

/// Parse the variable index list of record type 4 (`var_count + int32[]`).
pub(super) fn parse_value_label_vars(c: &mut Cursor) -> SocStatResult<Vec<i32>> {
    let count = c.read_i32()?;
    if count < 0 {
        return Err(sav_err(format!("negative value-label variable count {count}")));
    }
    let mut idxs = Vec::with_capacity(count as usize);
    for _ in 0..count {
        idxs.push(c.read_i32()?);
    }
    Ok(idxs)
}

/// Parse an extension record (rec type 7): `subtype, size, count, data[size*count]`.
pub(super) fn parse_extension(c: &mut Cursor) -> SocStatResult<ExtensionRecord> {
    let subtype = c.read_i32()?;
    let size = c.read_i32()?;
    let count = c.read_i32()?;
    if size < 0 || count < 0 {
        return Err(sav_err(format!(
            "invalid extension record (subtype {subtype}, size {size}, count {count})"
        )));
    }
    let n = (size as u64).saturating_mul(count as u64);
    if n > c.remaining() as u64 {
        return Err(truncated(c.pos(), n as usize));
    }
    let data = c.take(n as usize)?.to_vec();
    Ok(ExtensionRecord { subtype, data })
}

/// Skip a document record (rec type 6): `lines * 80` bytes.
pub(super) fn skip_document(c: &mut Cursor) -> SocStatResult<()> {
    let lines = c.read_i32()?;
    if lines < 0 {
        return Err(sav_err(format!("negative document line count {lines}")));
    }
    c.skip((lines as usize).saturating_mul(80))
}

// -------------------------------------------------------------------------
// Writers
// -------------------------------------------------------------------------

/// Serialize a variable record (main record, not a continuation).
pub(super) fn write_variable(
    short: &[u8; 8],
    v: &Variable,
    print: i32,
    le: bool,
) -> SocStatResult<Vec<u8>> {
    let mut out = Vec::with_capacity(96);
    out.extend_from_slice(&i32_to_bytes(2, le)); // rec_type
    let width = if v.data_type == DataType::Numeric { 0 } else { v.width as i32 };
    out.extend_from_slice(&i32_to_bytes(width, le)); // type
    out.extend_from_slice(&i32_to_bytes(v.label.is_some() as i32, le)); // has_var_label
    let (n_missing, miss_bytes) = encode_missing(&v.missing)?;
    out.extend_from_slice(&i32_to_bytes(n_missing, le));
    out.extend_from_slice(&i32_to_bytes(print, le)); // print
    out.extend_from_slice(&i32_to_bytes(print, le)); // write
    out.extend_from_slice(short);
    if let Some(label) = &v.label {
        let bytes = label.as_bytes();
        out.extend_from_slice(&i32_to_bytes(bytes.len() as i32, le));
        out.extend_from_slice(bytes);
        let pad = (4 - bytes.len() % 4) % 4;
        out.extend_from_slice(&[0u8; 4][..pad]);
    }
    out.extend_from_slice(&miss_bytes);
    Ok(out)
}

/// Serialize a continuation record of a long string (width > 8) variable.
pub(super) fn write_variable_continuation(short: &[u8; 8], le: bool) -> Vec<u8> {
    let mut out = Vec::with_capacity(32);
    out.extend_from_slice(&i32_to_bytes(2, le)); // rec_type
    out.extend_from_slice(&i32_to_bytes(-1, le)); // type = continuation
    out.extend_from_slice(&i32_to_bytes(0, le)); // has_var_label
    out.extend_from_slice(&i32_to_bytes(0, le)); // n_missing_values
    out.extend_from_slice(&i32_to_bytes(0, le)); // print
    out.extend_from_slice(&i32_to_bytes(0, le)); // write
    out.extend_from_slice(short);
    out
}

/// Serialize value labels as the record 3 + 4 pair.
fn encode_missing(m: &MissingSpec) -> SocStatResult<(i32, Vec<u8>)> {
    let mut b = Vec::new();
    match m {
        MissingSpec::None => Ok((0, b)),
        MissingSpec::Discrete(vals) => {
            if vals.len() > 3 {
                return Err(sav_err(format!(
                    "at most 3 discrete missing values, got {}",
                    vals.len()
                )));
            }
            if vals.is_empty() {
                return Ok((0, b));
            }
            for x in vals {
                b.extend_from_slice(&f64_to_bytes(*x, true));
            }
            Ok((vals.len() as i32, b))
        }
        MissingSpec::Range { low, high, discrete } => {
            b.extend_from_slice(&f64_to_bytes(*low, true));
            b.extend_from_slice(&f64_to_bytes(*high, true));
            let n = match discrete {
                Some(d) => {
                    b.extend_from_slice(&f64_to_bytes(*d, true));
                    -3
                }
                None => -2,
            };
            Ok((n, b))
        }
    }
}

/// Serialize one value-label set as records 3 + 4.
pub(super) fn write_value_labels(
    pairs: &[(Vec<u8>, Vec<u8>)],
    var_indexes: &[i32],
    le: bool,
) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&i32_to_bytes(3, le)); // rec_type 3
    for (tag, label) in pairs {
        out.extend_from_slice(tag); // 8-byte value
        out.extend_from_slice(&i32_to_bytes(label.len() as i32, le));
        out.extend_from_slice(label);
        let pad = (8 - label.len() % 8) % 8;
        out.extend_from_slice(&[0u8; 8][..pad]);
    }
    out.extend_from_slice(&i32_to_bytes(4, le)); // rec_type 4
    out.extend_from_slice(&i32_to_bytes(var_indexes.len() as i32, le));
    for idx in var_indexes {
        out.extend_from_slice(&i32_to_bytes(*idx, le));
    }
    out
}

/// Serialize an extension record (rec type 7).
pub(super) fn write_extension(
    subtype: i32,
    size: i32,
    count: i32,
    data: &[u8],
    le: bool,
) -> Vec<u8> {
    let mut out = Vec::with_capacity(16 + data.len());
    out.extend_from_slice(&i32_to_bytes(7, le)); // rec_type
    out.extend_from_slice(&i32_to_bytes(subtype, le));
    out.extend_from_slice(&i32_to_bytes(size, le));
    out.extend_from_slice(&i32_to_bytes(count, le));
    out.extend_from_slice(data);
    out
}

/// Serialize the dictionary terminator (rec type 999).
pub(super) fn write_terminator(le: bool) -> Vec<u8> {
    i32_to_bytes(999, le).to_vec()
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn is_record_type(v: i32) -> bool {
    matches!(v, 1 | 2 | 3 | 4 | 6 | 7 | 999)
}

fn trim_field(b: &[u8]) -> &str {
    let mut end = b.len();
    while end > 0 && matches!(b[end - 1], b' ' | 0) {
        end -= 1;
    }
    std::str::from_utf8(&b[..end]).unwrap_or("")
}