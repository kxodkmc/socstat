//! CSV reader and writer with **column-type inference**.
//!
//! # Type inference
//!
//! The reader samples up to 100 data rows to determine each column's type:
//! if **all** sampled values in a column parse as `f64`, the column is
//! `Numeric`; otherwise it is `Text`. Empty fields are always treated
//! as missing regardless of column type.
//!
//! # Writing
//!
//! Missing values are written as empty strings.

use std::fs::File;
use std::path::Path;

use crate::data::variable::{DataType, MeasureType, Variable};
use crate::data::{ColumnData, Dataset};
use crate::error::SocStatResult;

use super::{Reader, Writer};

pub struct CsvReader;
pub struct CsvWriter;

/// Number of rows to sample for type inference.
const TYPE_INFERENCE_SAMPLE: usize = 100;

impl Reader for CsvReader {
    fn read_path(&self, path: &Path) -> SocStatResult<Dataset> {
        let file = File::open(path)?;
        let mut rdr = csv::ReaderBuilder::new()
            .has_headers(true)
            .flexible(true)
            .from_reader(file);

        // --- Parse header into column names ---
        let header = rdr.headers()?.clone();
        let names: Vec<String> = header.iter().map(|s| s.to_string()).collect();
        let n_vars = names.len();

        // --- Collect all rows as StringRecords ---
        let mut records: Vec<csv::StringRecord> = Vec::new();
        let mut rec = csv::StringRecord::new();
        while rdr.read_record(&mut rec)? {
            records.push(rec.clone());
        }

        // --- Infer column types by sampling ---
        let sample_n = records.len().min(TYPE_INFERENCE_SAMPLE);
        let types: Vec<DataType> = (0..n_vars)
            .map(|col| infer_type(&records, col, sample_n))
            .collect();

        // --- Build variables ---
        let mut ds = Dataset::new();
        for (i, name) in names.iter().enumerate() {
            let var = match types[i] {
                DataType::Numeric => Variable::numeric(name).measure(MeasureType::Scale),
                DataType::Text => Variable::text(name).measure(MeasureType::Nominal),
            };
            ds.add_var(var)?;
        }

        // --- Build columns ---
        for (col_idx, _) in names.iter().enumerate() {
            let col_data = match types[col_idx] {
                DataType::Numeric => {
                    let v: Vec<Option<f64>> = records.iter()
                        .map(|r| parse_numeric(r.get(col_idx).unwrap_or("")))
                        .collect();
                    ColumnData::Numeric(v)
                }
                DataType::Text => {
                    let v: Vec<Option<String>> = records.iter()
                        .map(|r| parse_text(r.get(col_idx).unwrap_or("")))
                        .collect();
                    ColumnData::Text(v)
                }
            };
            // Replace the placeholder column created by add_var.
            if let Some(c) = ds.column_mut(col_idx) {
                *c = col_data;
            }
        }

        Ok(ds)
    }
}

impl Writer for CsvWriter {
    fn write_path(&self, ds: &Dataset, path: &Path) -> SocStatResult<()> {
        let file = File::create(path)?;
        let mut wtr = csv::WriterBuilder::new()
            .has_headers(true)
            .from_writer(file);

        // Header
        let header: Vec<&str> = ds.variables().iter().map(|v| v.name.as_str()).collect();
        wtr.write_record(&header)?;

        // Data rows
        let n_rows = ds.n_rows();
        let n_vars = ds.n_vars();
        let cols = ds.columns();
        for row_idx in 0..n_rows {
            let row: Vec<String> = (0..n_vars)
                .map(|col_idx| {
                    let col = &cols[col_idx];
                    col.get_value(row_idx)
                        .map(|v| v.display())
                        .unwrap_or_default()
                })
                .collect();
            wtr.write_record(&row)?;
        }

        wtr.flush()?;
        Ok(())
    }
}

// --- Helpers ---

fn infer_type(records: &[csv::StringRecord], col: usize, sample_n: usize) -> DataType {
    for rec in records.iter().take(sample_n) {
        let s = rec.get(col).unwrap_or("");
        if s.is_empty() { continue; } // skip missing
        if s.parse::<f64>().is_err() {
            return DataType::Text;
        }
    }
    DataType::Numeric
}

fn parse_numeric(s: &str) -> Option<f64> {
    if s.is_empty() { return None; }
    s.parse::<f64>().ok()
}

fn parse_text(s: &str) -> Option<String> {
    if s.is_empty() { None } else { Some(s.to_string()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_test_csv(path: &Path, content: &str) {
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
    }

    #[test]
    fn csv_type_inference() {
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_csv_inference.csv");
        write_test_csv(&path, "name,age,score\nAlice,25,85.5\nBob,30,92.0\nCarol,,78.0\n");

        let ds = CsvReader.read_path(&path).unwrap();
        assert_eq!(ds.n_vars(), 3);
        assert_eq!(ds.n_rows(), 3);

        // name should be Text
        assert!(ds.column_by_name("name").unwrap().as_text().is_some());
        // age and score should be Numeric
        assert!(ds.column_by_name("age").unwrap().as_numeric().is_some());
        assert!(ds.column_by_name("score").unwrap().as_numeric().is_some());

        // Carol's age should be missing
        let ages = ds.numeric_slice("age").unwrap();
        assert!(ages[2].is_none());
    }

    #[test]
    fn csv_mixed_column_becomes_text() {
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_csv_mixed.csv");
        write_test_csv(&path, "code\n123\nABC\n456\n");

        let ds = CsvReader.read_path(&path).unwrap();
        // "ABC" is not numeric → entire column becomes Text
        assert!(ds.column_by_name("code").unwrap().as_text().is_some());
    }

    #[test]
    fn csv_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_csv_roundtrip2.csv");
        write_test_csv(&path, "x,y\n1,hello\n2,world\n3,\n");

        let ds = CsvReader.read_path(&path).unwrap();
        assert_eq!(ds.n_rows(), 3);

        let out = dir.join("socstat_csv_roundtrip2_out.csv");
        CsvWriter.write_path(&ds, &out).unwrap();
        let ds2 = CsvReader.read_path(&out).unwrap();
        assert_eq!(ds2.n_rows(), 3);
        assert_eq!(ds2.n_vars(), 2);
    }
}
