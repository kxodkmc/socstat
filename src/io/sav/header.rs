//! The 176-byte file header record and endianness / float-format probing.

use super::{truncated, unsupported, Cursor, sav_err};
use crate::error::SocStatResult;

/// Size of the file header record in bytes.
pub(super) const HEADER_SIZE: usize = 176;

/// SPSS system missing value: `-DBL_MAX`.
pub(super) const SYSMIS: f64 = -f64::MAX;

/// SPSS-21-preceding encoding of the "LOWEST" (unbounded) missing-range bound.
pub(super) const OLD_LOWEST_BITS: u64 = 0xffe_ffff_ffff_fffe;

/// Body of the record-type field. `$FL2` = compression 0/1, `$FL3` = zlib.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum RecTypeFamily {
    Fl2,
    Fl3,
}

/// Data compression scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Compression {
    None,
    Bytecode,
    Zlib,
}

/// Parsed file header.
pub(super) struct FileHeader {
    pub compression: Compression,
    /// 1-based dictionary slot of the weight variable; 0 = none.
    pub weight_index: i32,
    /// Number of cases, or `-1` if unknown.
    pub ncases: i32,
    /// Compression bias (must be 100.0 for IEEE-754 confirmation).
    pub bias: f64,
    /// Integer/float endianness of the file.
    pub le: bool,
}

/// Parse and validate the 176-byte file header.
pub(super) fn parse(data: &[u8]) -> SocStatResult<FileHeader> {
    if data.len() < HEADER_SIZE {
        return Err(truncated(0, HEADER_SIZE));
    }

    // --- Record type: ASCII `$FL2` / `$FL3` vs EBCDIC variants ---
    let rt = &data[0..4];
    let rec_type = if rt == b"$FL2" {
        RecTypeFamily::Fl2
    } else if rt == b"$FL3" {
        RecTypeFamily::Fl3
    } else if rt[0] == 0x5b && rt[1] == 0xc6 && rt[2] == 0xd3 {
        return Err(unsupported("EBCDIC-encoded system files are not supported"));
    } else {
        return Err(sav_err("not an SPSS system file (record type not '$FL2'/'$FL3')"));
    };

    // --- Endianness via the layout code (offset 64) ---
    let layout = &data[64..68];
    let le = if u32::from_le_bytes(layout.try_into().unwrap()) == 2 {
        true
    } else if u32::from_be_bytes(layout.try_into().unwrap()) == 2 {
        false
    } else {
        return Err(unsupported("unrecognized layout code; not an SPSS system file"));
    };

    let mut c = Cursor::new(data, le);
    c.skip(4)?; // rec_type
    c.skip(60)?; // prod_name
    c.skip(4)?; // layout_code (already used)
    c.skip(4)?; // nominal_case_size — unreliable on write, ignored on read
    let compression_field = c.read_i32()?;
    let weight_index = c.read_i32()?;
    let ncases = c.read_i32()?;
    let bias = c.read_f64()?;

    // IEEE-754 assumption check; also probes IBM/VAX float formats.
    if bias != 100.0 {
        return Err(unsupported(format!(
            "non-IEEE-754 floating point (bias = {bias}); IBM/VAX float formats are not supported"
        )));
    }

    let compression = match (rec_type, compression_field) {
        (RecTypeFamily::Fl2, 0) => Compression::None,
        (RecTypeFamily::Fl2, 1) => Compression::Bytecode,
        (RecTypeFamily::Fl3, 2) => Compression::Zlib,
        (RecTypeFamily::Fl2, field) => {
            return Err(sav_err(format!(
                "record type '$FL2' requires compression 0 or 1, got {field}"
            )))
        }
        (RecTypeFamily::Fl3, field) => {
            return Err(sav_err(format!(
                "record type '$FL3' requires compression 2, got {field}"
            )))
        }
    };

    Ok(FileHeader { compression, weight_index, ncases, bias, le })
}

/// Serialize a header for a bytecode (`$FL2`, compression 1) file.
pub(super) fn write(
    compression: Compression,
    weight_index: i32,
    ncases: i32,
    case_size: i32,
    file_label: &str,
) -> Vec<u8> {
    let mut out = [0u8; HEADER_SIZE];
    out[0..4].copy_from_slice(b"$FL2");
    let prod = b"@(#) SPSS DATA FILE        socstat";
    let prod = &prod[..prod.len().min(60)];
    out[4..4 + prod.len()].copy_from_slice(prod);
    for b in out[4..64].iter_mut() {
        if *b == 0 {
            *b = b' ';
        }
    }
    let cc: i32 = match compression {
        Compression::None => 0,
        Compression::Bytecode => 1,
        Compression::Zlib => 2,
    };
    out[64..68].copy_from_slice(&2i32.to_le_bytes()); // layout code: little-endian
    out[68..72].copy_from_slice(&case_size.to_le_bytes()); // nominal case size
    out[72..76].copy_from_slice(&cc.to_le_bytes());
    out[76..80].copy_from_slice(&weight_index.to_le_bytes());
    out[80..84].copy_from_slice(&ncases.to_le_bytes());
    out[84..92].copy_from_slice(&100.0f64.to_le_bytes()); // bias
    out[92..101].copy_from_slice(b"01 Jan 70");
    out[101..109].copy_from_slice(b"00:00:00");
    let label = file_label.as_bytes();
    for (i, b) in out[109..173].iter_mut().enumerate() {
        *b = label.get(i).copied().unwrap_or(b' ');
    }
    out.to_vec()
}