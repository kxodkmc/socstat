//! SPSS `.sav` (System File Format) binary reader and writer.
//!
//! This module compiles only when the `sav` feature is enabled. It follows
//! the GNU PSPP System File Format documentation —the de-facto public
//! specification of the proprietary `.sav` format.
//!
//! # Support matrix
//!
//! | Direction | Compression                                   |
//! |-----------|-----------------------------------------------|
//! | read      | 0 (uncompressed), 1 (bytecode), 2 (zlib / `$FL3`) |
//! | write     | 1 (bytecode, `$FL2`) —readable by SAS, PSPP, R |
//!
//! Strings are decoded as UTF-8 when the file declares `UTF-8` (extension
//! record subtype 20); otherwise each byte is mapped to a Latin-1 character
//! so no byte is lost. Byte-identical round-trips are therefore guaranteed
//! only for UTF-8 files.
//!
//! Metadata round-trips: variable names (including >8-byte long names),
//! variable labels, value labels, user missing values (discrete and range),
//! measurement level, the case-weight variable, and string width.
//!
//! # Explicitly unsupported (errors, never silent corruption)
//!
//! - EBCDIC files and non-IEEE-754 floats (`bias != 100.0`).
//! - Very long strings (width > 255, extension record subtype 14).
//! - `.zsav` writing and `.por` files.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! # fn main() -> SocStatResult<()> {
//! let ds = socstat::read().sav("data.sav")?;
//! socstat::write().sav(&ds, "out.sav")?;
//! # Ok(())
//! # }
//! ```

use std::collections::{BTreeMap, HashSet};
use std::fs::File;
use std::io::{Read, Write};
use std::path::Path;

use crate::data::{
    ColumnData, DataType, Dataset, MeasureType, MissingSpec, ValueFormat, Variable,
};
use crate::error::{SocStatError, SocStatResult};

use super::{Reader, Writer};

pub(super) mod data;
pub(super) mod header;
pub(super) mod records;

/// Reader for SPSS `.sav` files.
///
/// Access through [`crate::read().sav(path)`][crate::read].
pub struct SavReader;

/// Writer for SPSS `.sav` files (bytecode compression).
///
/// Access through [`crate::write().sav(&ds, path)`][crate::write].
pub struct SavWriter;

impl Reader for SavReader {
    fn read_path(&self, path: &Path) -> SocStatResult<Dataset> {
        let mut f = File::open(path)?;
        let mut buf = Vec::new();
        f.read_to_end(&mut buf)?;
        read_dataset(&buf)
    }
}

impl Writer for SavWriter {
    fn write_path(&self, dataset: &Dataset, path: &Path) -> SocStatResult<()> {
        let bytes = write_dataset(dataset)?;
        let mut f = File::create(path)?;
        f.write_all(&bytes)?;
        Ok(())
    }
}

// -------------------------------------------------------------------------
// Shared byte helpers (used by header / records / data)
// -------------------------------------------------------------------------

/// Decode an int32 according to the file's endianness.
pub(super) fn bytes_to_i32(b: &[u8], le: bool) -> i32 {
    let a: [u8; 4] = b[..4].try_into().unwrap();
    if le { i32::from_le_bytes(a) } else { i32::from_be_bytes(a) }
}

/// Decode an f64 according to the file's endianness.
pub(super) fn bytes_to_f64(b: &[u8], le: bool) -> f64 {
    let a: [u8; 8] = b[..8].try_into().unwrap();
    if le { f64::from_le_bytes(a) } else { f64::from_be_bytes(a) }
}

/// Encode an int32 (writers always emit little-endian).
pub(super) fn i32_to_bytes(v: i32, le: bool) -> [u8; 4] {
    if le { v.to_le_bytes() } else { v.to_be_bytes() }
}

/// Encode an f64 (writers always emit little-endian).
pub(super) fn f64_to_bytes(v: f64, le: bool) -> [u8; 8] {
    if le { v.to_le_bytes() } else { v.to_be_bytes() }
}

/// Build a [`SocStatError::Sav`].
pub(super) fn sav_err(msg: impl Into<String>) -> SocStatError {
    SocStatError::Sav(msg.into())
}

/// Build a [`SocStatError::UnsupportedFormat`].
pub(super) fn unsupported(msg: impl Into<String>) -> SocStatError {
    SocStatError::UnsupportedFormat(msg.into())
}

/// Build a "file ended unexpectedly" error at `pos`.
pub(super) fn truncated(pos: usize, need: usize) -> SocStatError {
    sav_err(format!("unexpected end of file at offset {pos}, need {need} more bytes"))
}

/// A byte cursor over an in-memory buffer, with endianness.
pub(super) struct Cursor<'a> {
    data: &'a [u8],
    pos: usize,
    le: bool,
}

impl<'a> Cursor<'a> {
    pub fn new(data: &'a [u8], le: bool) -> Self {
        Self { data, pos: 0, le }
    }

    pub fn pos(&self) -> usize { self.pos }

    pub fn remaining(&self) -> usize { self.data.len() - self.pos }

    pub fn seek(&mut self, pos: usize) -> SocStatResult<()> {
        if pos > self.data.len() {
            return Err(truncated(self.pos, pos - self.pos));
        }
        self.pos = pos;
        Ok(())
    }

    pub fn take(&mut self, n: usize) -> SocStatResult<&'a [u8]> {
        if self.remaining() < n {
            return Err(truncated(self.pos, n));
        }
        let s = &self.data[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn read_i32(&mut self) -> SocStatResult<i32> {
        Ok(bytes_to_i32(self.take(4)?, self.le))
    }

    pub fn read_f64(&mut self) -> SocStatResult<f64> {
        Ok(bytes_to_f64(self.take(8)?, self.le))
    }

    pub fn read_i64(&mut self) -> SocStatResult<i64> {
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.le { i64::from_le_bytes(b) } else { i64::from_be_bytes(b) })
    }

    /// Peek at the next int32 without consuming it.
    pub fn peek_i32(&mut self) -> SocStatResult<i32> {
        let v = self.read_i32()?;
        self.pos -= 4;
        Ok(v)
    }

    pub fn skip(&mut self, n: usize) -> SocStatResult<()> {
        self.take(n).map(|_| ())
    }
}

/// Text encoding of a `.sav` file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Encoding {
    Utf8,
    Latin1,
}

/// Decode bytes using the file's declared encoding.
///
/// A UTF-8 declaration with malformed bytes falls back to Latin-1 so the
/// read never loses information.
pub(super) fn decode_text(bytes: &[u8], enc: Encoding) -> String {
    match enc {
        Encoding::Utf8 => String::from_utf8(bytes.to_vec()).unwrap_or_else(|_| latin1(bytes)),
        Encoding::Latin1 => latin1(bytes),
    }
}

fn latin1(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| char::from(b)).collect()
}

// -------------------------------------------------------------------------
// Reader
// -------------------------------------------------------------------------

/// An assembled variable from the dictionary section.
pub(super) struct RawVar {
    /// Display name (short name, overridden by the long name if present).
    pub name: String,
    /// Storage type: `0` for numeric, otherwise the string width in bytes.
    pub width: usize,
    pub label: Option<Vec<u8>>,
    pub missing: MissingSpec,
    /// Number of 8-byte data elements per case (1, or `ceil(width / 8)`).
    pub segments: usize,
    /// Raw value-label map (value tag bytes →label bytes), filled from
    /// records 3 + 4 once the dictionary is complete.
    pub value_labels: BTreeMap<Vec<u8>, Vec<u8>>,
}

/// One decoded cell for a case.
enum Cell {
    Number(Option<f64>),
    Text(Option<String>),
}

fn read_dataset(buf: &[u8]) -> SocStatResult<Dataset> {
    let hdr = header::parse(buf)?;
    let le = hdr.le;

    let data_offset = header::HEADER_SIZE;
    let mut cur = Cursor::new(&buf[data_offset..], le);
    let (slots, mut vars, exts) = parse_dictionary(&mut cur)?;
    let data_start = data_offset + cur.pos();
    let n_vars = vars.len();

    // --- Extension resolution (requires the full dictionary) ---
    let encoding = resolve_encoding(&exts);
    let long_names = resolve_long_names(&exts);
    let display = resolve_display_params(&exts, n_vars, &slots, le);

    for v in vars.iter_mut() {
        v.name = decode_text(v.name.as_bytes(), Encoding::Latin1);
        if let Some(long) = long_names.get(&v.name) {
            v.name = long.clone();
        }
        // Continuation validity: segments must match the declared width.
        if v.width != 0 && v.segments != div_ceil(v.width, 8) {
            return Err(sav_err(format!(
                "variable '{}': declared width {} needs {} data segments, found {}",
                v.name, v.width, div_ceil(v.width, 8), v.segments
            )));
        }
    }

    // --- Data section ---
    let mut src = match hdr.compression {
        header::Compression::None => DataSource::Flat(data::FlatDecoder::new(
            buf[data_start..].to_vec(), hdr.le,
        )),
        header::Compression::Bytecode => DataSource::Bytecode(data::ByteDecoder::new(
            buf[data_start..].to_vec(), hdr.le, hdr.bias,
        )),
        header::Compression::Zlib => {
            let inflated = data::inflate_zlib(buf, data_start, hdr.le)?;
            DataSource::Bytecode(data::ByteDecoder::new(inflated, hdr.le, hdr.bias))
        }
    };

    let mut cols: Vec<ColumnData> = vars
        .iter()
        .map(|v| ColumnData::empty(if v.width == 0 { DataType::Numeric } else { DataType::Text }))
        .collect();

    let strict = hdr.ncases >= 0;
    let target = if strict { hdr.ncases as usize } else { usize::MAX };
    let mut rows = 0usize;
    if n_vars > 0 {
        while rows < target {
            let Some(cells) = read_case(&mut src, &vars, encoding)? else {
                if strict {
                    return Err(sav_err(format!(
                        "case data truncated: expected {target} cases, read {rows}"
                    )));
                }
                break;
            };
            for (vi, cell) in cells.into_iter().enumerate() {
                match (&mut cols[vi], cell) {
                    (ColumnData::Numeric(v), Cell::Number(x)) => v.push(x),
                    (ColumnData::Text(v), Cell::Text(x)) => v.push(x),
                    _ => unreachable!("column/type mismatch from dictionary assembly"),
                }
            }
            rows += 1;
        }
    }

    // --- Assemble the dataset ---
    // `weight_index` is 1-based over dictionary slots (continuations count),
    // `0` meaning no weight variable. Convert it back to a model index.
    let weight_var: Option<usize> = if hdr.weight_index > 0 {
        let slot = hdr.weight_index as usize - 1;
        slots.get(slot).copied().flatten().filter(|&vi| vars[vi].width == 0)
    } else {
        None
    };

    let mut ds = Dataset::new();
    for (vi, v) in vars.iter().enumerate() {
        let data_type = if v.width == 0 { DataType::Numeric } else { DataType::Text };
        let measure = match display[vi].map(|d| d.0) {
            Some(3) => MeasureType::Scale,
            Some(2) => MeasureType::Ordinal,
            Some(1) => MeasureType::Nominal,
            // Unknown (0) measure: keep the type default.
            _ => default_measure(data_type),
        };
        let var = Variable {
            name: v.name.clone(),
            label: v.label.as_deref().map(|b| decode_text(b, encoding)),
            data_type,
            measure,
            format: ValueFormat::General,
            missing: v.missing.clone(),
            value_labels: decode_value_labels(v, encoding, le),
            width: v.width,
            is_weight: weight_var == Some(vi),
        };
        ds.add_var(var)?;
    }
    for (vi, col) in cols.into_iter().enumerate() {
        if let Some(c) = ds.column_mut(vi) {
            *c = col;
        }
    }
    Ok(ds)
}

fn default_measure(data_type: DataType) -> MeasureType {
    match data_type {
        DataType::Numeric => MeasureType::Scale,
        DataType::Text => MeasureType::Nominal,
    }
}

/// Read one complete case from the element source. Returns `None` when the
/// data ends (EOF or bytecode command 252); the partially-read case is
/// discarded by the caller.
fn read_case(
    src: &mut DataSource,
    vars: &[RawVar],
    encoding: Encoding,
) -> SocStatResult<Option<Vec<Cell>>> {
    let mut cells = Vec::with_capacity(vars.len());
    for v in vars {
        if v.width == 0 {
            match src.next(false)? {
                Some(data::RawElement::Numeric(x)) => {
                    let val = if x == header::SYSMIS { None } else { Some(x) };
                    cells.push(Cell::Number(val));
                }
                Some(_) => return Err(sav_err("numeric element carried string bytes")),
                None => return Ok(None),
            }
        } else {
            let mut bytes = Vec::with_capacity(v.segments * 8);
            for _ in 0..v.segments {
                match src.next(true)? {
                    Some(data::RawElement::Bytes(b8)) => bytes.extend_from_slice(&b8),
                    Some(_) => return Err(sav_err("string element carried a number")),
                    None => return Ok(None),
                }
            }
            let content = trim_field(&bytes, v.width);
            let val = if content.is_empty() {
                None
            } else {
                Some(decode_text(content, encoding))
            };
            cells.push(Cell::Text(val));
        }
    }
    Ok(Some(cells))
}

/// Trim a string field to its declared width and drop trailing spaces/NULs.
/// An all-blank field yields an empty slice (→`None` value).
fn trim_field(bytes: &[u8], width: usize) -> &[u8] {
    let mut end = bytes.len().min(width);
    while end > 0 && matches!(bytes[end - 1], b' ' | 0) {
        end -= 1;
    }
    &bytes[..end]
}

/// Format an f64 as a value-label key: integers without a decimal point.
fn fmt_f64(n: f64) -> String {
    if n.fract() == 0.0 && n.abs() < 1e15 {
        format!("{}", n as i64)
    } else {
        format!("{}", n)
    }
}

fn decode_value_labels(v: &RawVar, encoding: Encoding, le: bool) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    for (k, l) in &v.value_labels {
        let label = decode_text(l, encoding);
        if v.width == 0 {
            let n = bytes_to_f64(k, le);
            if n == header::SYSMIS { continue; } // SYSMIS cannot carry a label
            out.insert(fmt_f64(n), label);
        } else {
            let key = decode_text(trim_field(k, 8), encoding);
            if !key.is_empty() {
                out.insert(key, label);
            }
        }
    }
    out
}

// -------------------------------------------------------------------------
// Dictionary parsing
// -------------------------------------------------------------------------

/// Per-slot dictionary index. `None` marks a continuation slot of a long
/// string variable (its segment data belongs to the preceding variable).
type SlotMap = Vec<Option<usize>>;

/// Extension records collected during dictionary parsing.
pub(super) struct ExtRecords {
    pub long_names: Vec<Vec<u8>>,
    pub display: Vec<Vec<u8>>,
    pub encoding: Option<Vec<u8>>,
}

fn parse_dictionary(cur: &mut Cursor) -> SocStatResult<(SlotMap, Vec<RawVar>, ExtRecords)> {
    let mut vars: Vec<RawVar> = Vec::new();
    let mut slots: SlotMap = Vec::new();
    let mut exts = ExtRecords {
        long_names: Vec::new(),
        display: Vec::new(),
        encoding: None,
    };

    let mut pending_labels: Option<Vec<(Vec<u8>, Vec<u8>)>> = None;

    loop {
        if cur.remaining() < 4 {
            return Err(sav_err("dictionary does not end with a record type 999"));
        }
        let rec_type = cur.read_i32()?;
        match rec_type {
            // Record types 1 (SJF legacy) and 2 are variable records.
            1 | 2 => {
                let entry = records::parse_variable(cur)?;
                if entry.width == -1 {
                    // Continuation segment of the preceding string variable.
                    let Some(prev) = vars.last_mut() else {
                        return Err(sav_err("orphan continuation record (no preceding variable)"));
                    };
                    if prev.width == 0 {
                        return Err(sav_err("continuation record following a numeric variable"));
                    }
                    prev.segments += 1;
                    slots.push(None);
                } else {
                    let width = entry.width as usize;
                    vars.push(RawVar {
                        name: entry.name,
                        width,
                        label: entry.label,
                        missing: entry.missing,
                        segments: 1,
                        value_labels: BTreeMap::new(),
                    });
                    slots.push(Some(vars.len() - 1));
                }
            }
            3 => {
                pending_labels = Some(records::parse_value_labels(cur)?);
            }
            4 => {
                let idxs = records::parse_value_label_vars(cur)?;
                let pairs = pending_labels
                    .take()
                    .ok_or_else(|| {
                        sav_err("value-label variables record (4) without value labels (3)")
                    })?;
                for idx in idxs {
                    // Indices in record 4 are 1-based over dictionary slots.
                    let Some(vi) = slots.get((idx - 1) as usize).copied().flatten() else {
                        continue; // continuation slot or out of range →skip
                    };
                    if vars[vi].width >= 8 {
                        continue; // SPSS forbids labels on string width ≥8
                    }
                    for (k, l) in &pairs {
                        vars[vi].value_labels.insert(k.clone(), l.clone());
                    }
                }
            }
            6 => records::skip_document(cur)?,
            7 => {
                let ext = records::parse_extension(cur)?;
                match ext.subtype {
                    14 => {
                        return Err(unsupported(
                            "very long strings (record subtype 14) are not supported",
                        ))
                    }
                    11 => exts.display.push(ext.data),
                    13 => exts.long_names.push(ext.data),
                    20 => exts.encoding = Some(ext.data),
                    _ => {} // unknown extension records are skipped (spec recommendation)
                }
            }
            999 => return Ok((slots, vars, exts)),
            other => return Err(sav_err(format!("unsupported dictionary record type {other}"))),
        }
    }
}

pub(super) fn div_ceil(a: usize, b: usize) -> usize {
    a.div_ceil(b)
}

/// Determine the file's text encoding from extension record subtype 20.
fn resolve_encoding(exts: &ExtRecords) -> Encoding {
    let Some(raw) = &exts.encoding else { return Encoding::Latin1 };
    let norm: Vec<u8> = raw
        .iter()
        .take_while(|&&b| b != 0)
        .copied()
        .filter(|&b| b.is_ascii_alphanumeric())
        .map(|b| b.to_ascii_lowercase())
        .collect();
    match norm.as_slice() {
        b"utf8" | b"utf" | b"ascii" | b"usascii" => Encoding::Utf8,
        _ => Encoding::Latin1,
    }
}

/// Parse extension record subtype 13: `short=long` pairs, tab-separated.
fn resolve_long_names(exts: &ExtRecords) -> BTreeMap<String, String> {
    let mut map = BTreeMap::new();
    for raw in &exts.long_names {
        let text = String::from_utf8_lossy(raw);
        for part in text.split('\t') {
            let Some(eq) = part.find('=') else { continue };
            map.insert(part[..eq].to_string(), part[eq + 1..].to_string());
        }
    }
    map
}

/// Parse extension record subtype 11 into per-variable display parameters.
///
/// The record uses 2 or 3 int32 fields per variable (older writers omit the
/// width field). The field count is derived from the byte length, matching
/// either the number of real variables or the number of dictionary slots.
fn resolve_display_params(
    exts: &ExtRecords,
    n_vars: usize,
    slot_to_var: &[Option<usize>],
    le: bool,
) -> Vec<Option<(i32, i32, i32)>> {
    let out_len = n_vars.max(1);
    let n_slots = slot_to_var.len();
    let mut out = vec![None; out_len];
    for raw in &exts.display {
        let n_ints = raw.len() / 4;
        let (per, n_entries, use_slots) = if n_ints == 2 * n_vars {
            (2, n_vars, false)
        } else if n_ints == 3 * n_vars {
            (3, n_vars, false)
        } else if n_ints == 2 * n_slots {
            (2, n_slots, true)
        } else if n_ints == 3 * n_slots {
            (3, n_slots, true)
        } else {
            continue; // unrecognized layout →ignore, keep type defaults
        };
        for i in 0..n_entries {
            let b = &raw[i * per * 4..(i + 1) * per * 4];
            let measure = bytes_to_i32(b, le);
            let width = if per == 3 { bytes_to_i32(&b[4..8], le) } else { 0 };
            let align = if per == 3 { bytes_to_i32(&b[8..12], le) } else { 0 };
            let vi = if use_slots {
                slot_to_var.get(i).copied().flatten()
            } else {
                Some(i)
            };
            if let Some(vi) = vi
                .filter(|&vi| vi < out.len())
                .filter(|&vi| out[vi].is_none())
            {
                out[vi] = Some((measure, width, align));
            }
        }
    }
    out
}

enum DataSource {
    Flat(data::FlatDecoder),
    Bytecode(data::ByteDecoder),
}

impl DataSource {
    fn next(&mut self, is_string: bool) -> SocStatResult<Option<data::RawElement>> {
        match self {
            DataSource::Flat(dec) => dec.next(is_string),
            DataSource::Bytecode(dec) => dec.next(is_string),
        }
    }
}

// -------------------------------------------------------------------------
// Writer
// -------------------------------------------------------------------------

/// Per-variable write plan.
struct Plan {
    short: [u8; 8],
    segments: usize,
    data_type: DataType,
    width: usize,
    slot: usize,
}

fn write_dataset(ds: &Dataset) -> SocStatResult<Vec<u8>> {
    let le = true;
    let n_vars = ds.n_vars();
    let mut out = Vec::new();

    // --- Preflight validation (Hard Rules 4 & 5: no silent coercion) ---
    for v in ds.variables() {
        match v.data_type {
            DataType::Numeric => {
                if let MissingSpec::Discrete(vals) = &v.missing
                    && vals.len() > 3
                {
                    return Err(sav_err(format!(
                        "variable '{}': at most 3 discrete missing values, got {}",
                        v.name, vals.len()
                    )));
                }
                for key in v.value_labels.keys() {
                    key.parse::<f64>().map_err(|_| {
                        sav_err(format!(
                            "variable '{}': value-label key '{key}' is not a number",
                            v.name
                        ))
                    })?;
                }
            }
            DataType::Text => {
                if v.width == 0 || v.width > 255 {
                    return Err(sav_err(format!(
                        "variable '{}': string width must be 1..=255, got {}",
                        v.name, v.width
                    )));
                }
                if v.missing != MissingSpec::None {
                    return Err(sav_err(format!(
                        "variable '{}': string variables cannot carry missing-value rules in .sav",
                        v.name
                    )));
                }
                if !v.value_labels.is_empty() && v.width >= 8 {
                    return Err(sav_err(format!(
                        "variable '{}': value labels require string width < 8, got {}",
                        v.name, v.width
                    )));
                }
            }
        }
        if let Some(label) = &v.label {
            let l = label.len();
            if l > 255 {
                return Err(sav_err(format!(
                    "variable '{}': label exceeds 255 bytes ({l})",
                    v.name
                )));
            }
        }
    }

    // --- Short names (≤ chars; long names go to extension record 13) ---
    let mut used: HashSet<String> = HashSet::new();
    let mut short_names: Vec<[u8; 8]> = Vec::with_capacity(n_vars);
    let mut long_pairs: Vec<(String, String)> = Vec::new();
    for v in ds.variables() {
        let short = make_short_name(&v.name, &mut used);
        let mut pad = [b' '; 8];
        pad[..short.len()].copy_from_slice(short.as_bytes());
        short_names.push(pad);
        if short != v.name {
            long_pairs.push((short.clone(), v.name.clone()));
        }
    }

    // --- Slots, case size, header ---
    let mut plans: Vec<Plan> = Vec::with_capacity(n_vars);
    let mut case_size = 0usize;
    for (vi, v) in ds.variables().iter().enumerate() {
        let segments = if v.data_type == DataType::Text { div_ceil(v.width, 8) } else { 1 };
        plans.push(Plan {
            short: short_names[vi],
            segments,
            data_type: v.data_type,
            width: v.width,
            slot: case_size,
        });
        case_size += segments;
    }

    let ncases = ds.n_rows();
    if ncases > i32::MAX as usize {
        return Err(sav_err(format!("too many rows to write: {ncases}")));
    }
    // 1-based dictionary slot of the weight variable, accounting for
    // continuation segments; 0 = no weight variable.
    let weight_index = ds.weight_var_index().map(|wi| plans[wi].slot + 1).unwrap_or(0) as i32;

    out.extend_from_slice(&header::write(
        header::Compression::Bytecode,
        weight_index,
        ncases as i32,
        case_size as i32,
        ds.name().unwrap_or(""),
    ));

    // --- Variable records (long strings split into continuation segments) ---
    for (plan, v) in plans.iter().zip(ds.variables()) {
        let print = format_code(v, plan.width);
        out.extend(records::write_variable(&plan.short, v, print, le)?);
        for _ in 1..plan.segments {
            out.extend(records::write_variable_continuation(&plan.short, le));
        }
    }

    // --- Value labels (records 3 + 4) ---
    for (plan, v) in plans.iter().zip(ds.variables()) {
        if v.value_labels.is_empty() {
            continue;
        }
        let pairs: Vec<(Vec<u8>, Vec<u8>)> = v
            .value_labels
            .iter()
            .map(|(k, l)| Ok((value_tag(v.data_type, k)?, l.as_bytes().to_vec())))
            .collect::<SocStatResult<_>>()?;
        out.extend(records::write_value_labels(&pairs, &[(plan.slot + 1) as i32], le));
    }

    // --- Extension records (ascending subtype order) ---
    let mut display = Vec::with_capacity(plans.len() * 12);
    for (plan, v) in plans.iter().zip(ds.variables()) {
        let measure = match v.measure {
            MeasureType::Nominal => 1,
            MeasureType::Ordinal => 2,
            MeasureType::Scale => 3,
        };
        let width = if plan.data_type == DataType::Text { plan.width as i32 } else { 8 };
        for x in [measure, width, 0] {
            display.extend_from_slice(&i32_to_bytes(x, le));
        }
    }
    out.extend(records::write_extension(11, 4, (display.len() / 4) as i32, &display, le));

    if !long_pairs.is_empty() {
        let mut text = String::new();
        for (short, long) in &long_pairs {
            text.push_str(&format!("{short}={long}\t"));
        }
        if text.ends_with('\t') {
            text.pop();
        }
        out.extend(records::write_extension(13, 1, text.len() as i32, text.as_bytes(), le));
    }

    out.extend(records::write_extension(20, 1, 5, b"UTF-8", le));

    // --- Dictionary terminator ---
    out.extend(records::write_terminator(le));

    // --- Data (bytecode, compression 1) ---
    let units = encode_dataset(ds, &plans)?;
    out.extend(data::encode_bytecode(&units));

    Ok(out)
}

/// Encode one 8-byte value tag for a value-label record.
fn value_tag(data_type: DataType, key: &str) -> SocStatResult<Vec<u8>> {
    match data_type {
        DataType::Numeric => key
            .parse::<f64>()
            .map(|n| f64_to_bytes(n, true).to_vec())
            .map_err(|_| sav_err(format!("value-label key '{key}' is not a number"))),
        DataType::Text => {
            if key.len() >= 8 {
                return Err(sav_err(format!(
                    "value-label key '{key}' does not fit in an 8-byte string tag"
                )));
            }
            let mut tag = [b' '; 8];
            tag[..key.len()].copy_from_slice(key.as_bytes());
            Ok(tag.to_vec())
        }
    }
}

/// Map a [`Variable`]'s display format to an SPSS format code
/// (`type | width<<8 | decimals<<16`).
fn format_code(v: &Variable, width: usize) -> i32 {
    const A: i32 = 1;
    const F: i32 = 5;
    const DOLLAR: i32 = 4;
    const E: i32 = 17;
    const DATE: i32 = 20;
    const DATETIME: i32 = 22;
    const PCT: i32 = 31;
    if v.data_type == DataType::Text {
        return A | ((width as i32).clamp(1, 255) << 8);
    }
    let (t, w, d) = match v.format {
        ValueFormat::General => (F, 8, 2),
        ValueFormat::Fixed { width: w, decimals: d } => (F, w, d),
        ValueFormat::Scientific { width: w, decimals: d } => (E, w, d),
        ValueFormat::Percent { decimals: d } => (PCT, 8, d),
        ValueFormat::Currency { decimals: d } => (DOLLAR, 8, d),
        ValueFormat::Date => (DATE, 10, 0),
        ValueFormat::DateTime => (DATETIME, 20, 0),
    };
    t | ((w.clamp(1, 40) as i32) << 8) | ((d.clamp(0, 16) as i32) << 16)
}

fn make_short_name(orig: &str, used: &mut HashSet<String>) -> String {
    let upper = orig.to_ascii_uppercase();
    let mut cand = String::new();
    for (i, ch) in upper.chars().enumerate() {
        let ok = if i == 0 {
            ch.is_ascii_alphabetic() || ch == '@'
        } else {
            ch.is_ascii_alphanumeric() || matches!(ch, '_' | '#' | '$' | '.')
        };
        if !ok {
            break;
        }
        cand.push(ch);
        if cand.len() == 8 {
            break;
        }
    }
    if cand.is_empty() {
        cand.push_str("VAR");
    }
    if used.insert(cand.clone()) {
        return cand;
    }
    // Name collision: shorten the base and append a unique numeric suffix.
    let base_len = cand.len().min(6);
    for i in 1..=9999usize {
        let tail = i.to_string();
        let keep = (10 - tail.len()).max(1).min(base_len);
        let s = format!("{}{tail}", &cand[..keep]);
        if used.insert(s.clone()) {
            return s;
        }
    }
    unreachable!("short-name generator exhausted its namespace")
}

/// Encode every case into a flat list of bytecode units.
fn encode_dataset(ds: &Dataset, plans: &[Plan]) -> SocStatResult<Vec<data::Unit>> {
    let cols = ds.columns();
    let mut units = Vec::new();
    for row in 0..ds.n_rows() {
        for (plan, col) in plans.iter().zip(cols) {
            if plan.data_type == DataType::Numeric {
                let v = col.as_numeric().unwrap()[row];
                match v {
                    Some(x) if is_small_int(x) => units.push(data::Unit::cmd((x as i32 + 100) as u8)),
                    Some(x) => units.push(data::Unit::lit(253, f64_to_bytes(x, true))),
                    None => units.push(data::Unit::cmd(255)),
                }
            } else {
                let width = plan.width;
                let text = col.as_text().unwrap()[row].clone();
                let mut buf = vec![b' '; plan.segments * 8];
                if let Some(s) = text {
                    if s.len() > width {
                        return Err(sav_err(format!(
                            "corruption guard: string of {} bytes exceeds declared width {width}",
                            s.len()
                        )));
                    }
                    buf[..s.len()].copy_from_slice(s.as_bytes());
                }
                for chunk in buf.chunks(8) {
                    if chunk.iter().all(|&b| b == b' ') {
                        units.push(data::Unit::cmd(254));
                    } else {
                        units.push(data::Unit::lit(253, chunk.try_into().unwrap()));
                    }
                }
            }
        }
    }
    Ok(units)
}

fn is_small_int(x: f64) -> bool {
    x.fract() == 0.0 && (-99.0..=151.0).contains(&x)
}

#[cfg(test)]
mod tests {
    use std::io::Write as _;

    use flate2::write::ZlibEncoder;
    use flate2::Compression;

    use super::*;
    use crate::data::Value;

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("socstat_sav_{}_{}.sav", name, std::process::id()))
    }

    fn push_i32(v: &mut Vec<u8>, x: i32) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn push_i64(v: &mut Vec<u8>, x: i64) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    fn push_f64(v: &mut Vec<u8>, x: f64) {
        v.extend_from_slice(&x.to_le_bytes());
    }

    /// Build a 176-byte header (little-endian, ASCII family).
    fn raw_header(
        rec_type: &[u8; 4],
        nominal: i32,
        compression: i32,
        weight: i32,
        ncases: i32,
    ) -> Vec<u8> {
        const PROD: &[u8] = b"@(#) SPSS DATA FILE        socstat";
        let mut h = vec![0u8; 176];
        h[0..4].copy_from_slice(rec_type);
        h[4..4 + PROD.len()].copy_from_slice(PROD);
        h[4..64].fill(b' ');
        h[64..68].copy_from_slice(&2i32.to_le_bytes());
        h[68..72].copy_from_slice(&nominal.to_le_bytes());
        h[72..76].copy_from_slice(&compression.to_le_bytes());
        h[76..80].copy_from_slice(&weight.to_le_bytes());
        h[80..84].copy_from_slice(&ncases.to_le_bytes());
        h[84..92].copy_from_slice(&100.0f64.to_le_bytes());
        h[92..101].copy_from_slice(b"01 Jan 70");
        h[101..109].copy_from_slice(b"00:00:00");
        h[109..173].fill(b' ');
        h
    }

    /// Build a plain variable record (no label, no missing).
    fn var_record(name: &str, width: i32) -> Vec<u8> {
        let mut r = Vec::new();
        push_i32(&mut r, 2); // rec_type
        push_i32(&mut r, width);
        push_i32(&mut r, 0); // has_var_label
        push_i32(&mut r, 0); // n_missing_values
        push_i32(&mut r, 5 | (8 << 8) | (2 << 16)); // print: F8.2
        push_i32(&mut r, 5 | (8 << 8) | (2 << 16)); // write
        let mut n = [b' '; 8];
        n[..name.len()].copy_from_slice(name.as_bytes());
        r.extend_from_slice(&n);
        r
    }

    /// Hand-built compression 0 fixture: X numeric (2 cases) + S string width 8.
    fn comp0_fixture() -> Vec<u8> {
        let mut b = raw_header(b"$FL2", 2, 0, 0, 2);
        b.extend(var_record("X", 0));
        b.extend(var_record("S", 8));
        push_i32(&mut b, 999);
        push_f64(&mut b, 1.0);
        let mut s1 = [b' '; 8];
        s1[..2].copy_from_slice(b"ab");
        b.extend_from_slice(&s1);
        push_f64(&mut b, -f64::MAX); // SYSMIS
        let mut s2 = [b' '; 8];
        s2[..2].copy_from_slice(b"cd");
        b.extend_from_slice(&s2);
        b
    }

    /// Hand-built compression 1 fixture: X numeric, 3 cases
    /// (5.0 encoded, 200.0 literal, missing). Validates literal ordering.
    fn comp1_fixture() -> Vec<u8> {
        let mut b = raw_header(b"$FL2", 1, 1, 0, 3);
        b.extend(var_record("X", 0));
        push_i32(&mut b, 999);
        b.extend_from_slice(&[105, 253, 255, 0, 0, 0, 0, 0]);
        b.extend_from_slice(&200.0f64.to_le_bytes());
        b.extend_from_slice(&[252, 0, 0, 0, 0, 0, 0, 0]);
        b
    }

    /// Hand-built compression 2 (`$FL3`) fixture over the same bytecode stream.
    fn comp2_fixture() -> Vec<u8> {
        let mut code = vec![105u8, 253, 255, 0, 0, 0, 0, 0];
        code.extend_from_slice(&200.0f64.to_le_bytes());
        code.extend_from_slice(&[252, 0, 0, 0, 0, 0, 0, 0]);

        let mut enc = ZlibEncoder::new(Vec::new(), Compression::default());
        enc.write_all(&code).unwrap();
        let compressed = enc.finish().unwrap();

        let mut b = raw_header(b"$FL3", 1, 2, 0, 3);
        b.extend(var_record("X", 0));
        push_i32(&mut b, 999);
        push_i64(&mut b, 0); // zheader offset
        push_i64(&mut b, 24 + compressed.len() as i64); // ztrailer offset
        push_i64(&mut b, 32 + 24); // ztrailer length
        b.extend_from_slice(&compressed);
        push_i64(&mut b, 100); // trailer bias (ignored by reader)
        push_i64(&mut b, 0);
        push_i64(&mut b, code.len() as i64); // block size
        push_i64(&mut b, 1); // n_blocks
        push_i64(&mut b, 0); // uncompressed offset
        push_i64(&mut b, 0); // compressed offset (relative to zlib stream start)
        push_i32(&mut b, code.len() as i32); // uncompressed size
        push_i32(&mut b, compressed.len() as i32); // compressed size
        b
    }

    fn read_bytes(bytes: &[u8], name: &str) -> Dataset {
        let path = temp_path(name);
        std::fs::write(&path, bytes).unwrap();
        let ds = SavReader.read_path(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        ds
    }

    #[test]
    fn reads_compression0() {
        let ds = read_bytes(&comp0_fixture(), "c0_dataset.sav");
        assert_eq!(ds.n_vars(), 2);
        assert_eq!(ds.n_rows(), 2);
        assert_eq!(ds.numeric_slice("X").unwrap(), &[Some(1.0), None]);
        assert_eq!(
            ds.text_slice("S").unwrap(),
            &[Some("ab".into()), Some("cd".into())]
        );
    }

    #[test]
    fn reads_compression1_bytecode() {
        let ds = read_bytes(&comp1_fixture(), "c1_dataset.sav");
        assert_eq!(ds.numeric_slice("X").unwrap(), &[Some(5.0), Some(200.0), None]);
    }

    #[test]
    fn reads_compression2_zlib() {
        let ds = read_bytes(&comp2_fixture(), "c2_dataset.sav");
        assert_eq!(ds.numeric_slice("X").unwrap(), &[Some(5.0), Some(200.0), None]);
    }

    #[test]
    fn rejects_non_ieee_bias() {
        let mut bytes = comp0_fixture();
        bytes[84..92].copy_from_slice(&500.0f64.to_le_bytes());
        let path = temp_path("bad_bias");
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            SavReader.read_path(&path),
            Err(SocStatError::UnsupportedFormat(_))
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_ebcdic() {
        let mut bytes = comp0_fixture();
        bytes[0..4].copy_from_slice(&[0x5b, 0xc6, 0xd3, 0xf2]);
        let path = temp_path("ebcdic");
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            SavReader.read_path(&path),
            Err(SocStatError::UnsupportedFormat(_))
        ));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn rejects_unknown_dict_record() {
        let mut bytes = raw_header(b"$FL2", 1, 1, 0, 1);
        push_i32(&mut bytes, 99); // bogus record type
        let path = temp_path("unknown_rec");
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(SavReader.read_path(&path), Err(SocStatError::Sav(_))));
        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------
    // Round-trips
    // -----------------------------------------------------------------

    fn sample_ds() -> Dataset {
        let mut ds = Dataset::new();
        ds.set_name("demo");
        ds.add_var(
            Variable::numeric("a_very_long_numeric_name")
                .label("Main score")
                .measure(MeasureType::Scale)
                .missing_discrete(&[-1.0, -2.0, 99.0])
                .value_label("1", "one")
                .value_label("2", "two"),
        )
        .unwrap();
        ds.add_var(
            Variable::numeric("income")
                .measure(MeasureType::Ordinal)
                .missing_range(0.0, 100.0, Some(999.0)),
        )
        .unwrap();
        ds.add_var(
            Variable::numeric("w")
                .measure(MeasureType::Nominal)
                .weight()
                .missing_range(-f64::MAX, 0.0, None),
        )
        .unwrap();
        ds.add_var(
            Variable::text("g")
                .width(6)
                .value_label("M", "Male")
                .value_label("F", "Female"),
        )
        .unwrap();
        ds.add_var(Variable::text("long_text_var").width(16).label("Wide text"))
            .unwrap();
        let rows: Vec<Vec<Value>> = vec![
            vec![
                Value::Number(1.0),
                Value::Number(50.0),
                Value::Number(2.0),
                Value::Text("M".into()),
                Value::Text("hello world".into()),
            ],
            vec![
                Value::Number(-1.0),
                Value::Number(3.141592653589793),
                Value::Number(2.0),
                Value::Text("F".into()),
                Value::Text("héllo wörld".into()),
            ],
            vec![
                Value::Number(99.0),
                Value::Number(0.1),
                Value::Missing,
                Value::Text("F".into()),
                Value::Missing,
            ],
            vec![
                Value::Number(2.0),
                Value::Number(1e-7),
                Value::Number(1.0),
                Value::Text("U".into()),
                Value::Text("pad to width y".into()),
            ],
            vec![Value::Missing; 5],
        ];
        for r in rows {
            ds.push_row(r).unwrap();
        }
        ds
    }

    #[test]
    fn roundtrip_full_fidelity() {
        let ds = sample_ds();
        let path = temp_path("full");
        crate::write().sav(&ds, &path).unwrap();

        // Builder + extension auto-detection.
        let ds2 = crate::read().auto(&path).unwrap();

        assert_eq!(ds2.n_vars(), 5);
        assert_eq!(ds2.n_rows(), 5);

        // Metadata: names (incl. long names), labels, missing rules,
        // measures, widths, weights, value labels —byte-for-byte.
        let expect = ds.variables().to_vec();
        let got = ds2.variables().to_vec();
        assert_eq!(got, expect);

        // Weight variable index preserved.
        assert_eq!(ds2.weight_var_index(), Some(2));

        // Data: numeric precision bit-for-bit.
        for (a, b) in ["a_very_long_numeric_name", "income", "w"]
            .iter()
            .map(|n| {
                (
                    ds.numeric_slice(n).unwrap(),
                    ds2.numeric_slice(n).unwrap(),
                )
            })
        {
            let abits: Vec<Option<u64>> = a.iter().map(|o| o.map(|f| f.to_bits())).collect();
            let bbits: Vec<Option<u64>> = b.iter().map(|o| o.map(|f| f.to_bits())).collect();
            assert_eq!(bbits, abits);
        }

        // Text values unchanged (including multi-byte UTF-8 and multi-segment).
        assert_eq!(ds.text_slice("g").unwrap(), ds2.text_slice("g").unwrap());
        assert_eq!(
            ds.text_slice("long_text_var").unwrap(),
            ds2.text_slice("long_text_var").unwrap()
        );
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn roundtrip_long_string_width16() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::text("msg").width(16)).unwrap();
        ds.push_row(vec![Value::Text("0123456789abcdef".into())]).unwrap();
        ds.push_row(vec![Value::Text("short".into())]).unwrap();
        ds.push_row(vec![Value::Missing]).unwrap();
        let out = temp_path("wide");
        crate::write().sav(&ds, &out).unwrap();
        let ds2 = crate::read().sav(&out).unwrap();
        assert_eq!(
            ds.text_slice("msg").unwrap(),
            ds2.text_slice("msg").unwrap()
        );
        assert_eq!(ds2.variables()[0].width, 16);
        let _ = std::fs::remove_file(&out);
    }

    #[test]
    fn rejects_text_var_missing_spec() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::text("s").missing_discrete(&[-1.0])).unwrap();
        let out = temp_path("text_missing");
        assert!(matches!(
            crate::write().sav(&ds, &out),
            Err(SocStatError::Sav(_))
        ));
    }

    #[test]
    fn rejects_more_than_three_discrete() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x").missing_discrete(&[-1.0, -2.0, -3.0, -4.0]))
            .unwrap();
        let out = temp_path("too_many_missing");
        assert!(matches!(
            crate::write().sav(&ds, &out),
            Err(SocStatError::Sav(_))
        ));
    }

    #[test]
    fn rejects_labels_on_wide_string() {
        let mut ds = Dataset::new();
        ds.add_var(
            Variable::text("s")
                .width(8)
                .value_label("X", "eh"),
        )
        .unwrap();
        let out = temp_path("wide_label");
        assert!(matches!(
            crate::write().sav(&ds, &out),
            Err(SocStatError::Sav(_))
        ));
    }

    #[test]
    fn rejects_nonnumeric_label_key() {
        let mut ds = Dataset::new();
        ds.add_var(
            Variable::numeric("x").value_label("not-a-number", "label"),
        )
        .unwrap();
        let out = temp_path("bad_key");
        assert!(matches!(
            crate::write().sav(&ds, &out),
            Err(SocStatError::Sav(_))
        ));
    }

    #[test]
    fn empty_string_datasource_roundtrip() {
        let ds = Dataset::new();
        let out = temp_path("empty");
        crate::write().sav(&ds, &out).unwrap();
        let ds2 = crate::read().sav(&out).unwrap();
        assert_eq!(ds2.n_vars(), 0);
        assert_eq!(ds2.n_rows(), 0);
        let _ = std::fs::remove_file(&out);
    }
}