//! CSV reader and writer with **column-type inference**.
//!
//! # Type inference
//!
//! The reader infers each column's type across **all** data rows: if every
//! value in a column parses as `f64`, the column is `Numeric`; otherwise it
//! is `Text`. Empty fields are always treated as missing regardless of column
//! type. Scanning all rows (not a sample) ensures a non-numeric value in a
//! later row is never silently dropped (BUG-002).
//!
//! # Writing
//!
//! Missing values are written as empty strings.

use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use crate::data::variable::{DataType, MeasureType, MissingSpec, Variable};
use crate::data::{ColumnData, Dataset};
use crate::error::SocStatResult;

use super::{Reader, Writer};

pub struct CsvReader;
pub struct CsvWriter;

/// Per-column user-defined missing-value rules applied when reading a CSV.
///
/// Raw values are stored in the column as-is; the missing spec is attached to
/// the variable so downstream statistics exclude them (matching SPSS and the
/// rest of the crate). See [`ReadBuilder::csv_with_options`].
#[derive(Debug, Clone, Default)]
pub struct CsvReaderOptions {
    missing: BTreeMap<String, MissingSpec>,
}

impl CsvReaderOptions {
    /// Start with no user-defined missing values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Treat `vals` as discrete user-missing values for a numeric column.
    pub fn missing_discrete(mut self, var: &str, vals: &[f64]) -> Self {
        self.missing.insert(var.to_string(), MissingSpec::Discrete(vals.to_vec()));
        self
    }

    /// Treat `[low, high]` (optionally plus one discrete value) as a
    /// user-missing range for a numeric column.
    pub fn missing_range(mut self, var: &str, low: f64, high: f64, discrete: Option<f64>) -> Self {
        self.missing.insert(var.to_string(), MissingSpec::Range { low, high, discrete });
        self
    }
}

impl Reader for CsvReader {
    fn read_path(&self, path: &Path) -> SocStatResult<Dataset> {
        self.read_path_with_options(path, &CsvReaderOptions::default())
    }
}

impl CsvReader {
    /// Read a CSV with per-column missing-value options.
    pub fn read_path_with_options(
        &self,
        path: &Path,
        options: &CsvReaderOptions,
    ) -> SocStatResult<Dataset> {
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

        // --- Infer column types across all rows ---
        let types: Vec<DataType> = (0..n_vars)
            .map(|col| infer_type(&records, col))
            .collect();

        // --- Build variables ---
        let mut ds = Dataset::new();
        for (i, name) in names.iter().enumerate() {
            let var = match types[i] {
                DataType::Numeric => {
                    let mut v = Variable::numeric(name).measure(MeasureType::Scale);
                    if let Some(spec) = options.missing.get(name) {
                        v.missing = spec.clone();
                    }
                    v
                }
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

fn infer_type(records: &[csv::StringRecord], col: usize) -> DataType {
    for rec in records {
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
    fn csv_inference_scans_all_rows() {
        // BUG-002: a non-numeric value beyond any sampling window must not be
        // silently dropped. Build 105 rows: 100 numeric, then one "ABC".
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_csv_full_scan.csv");
        let mut content = String::from("code\n");
        for i in 0..100 {
            content.push_str(&format!("{i}\n"));
        }
        content.push_str("ABC\n");
        write_test_csv(&path, &content);

        let ds = CsvReader.read_path(&path).unwrap();
        // The whole column is Text, so "ABC" is preserved rather than dropped.
        assert!(ds.column_by_name("code").unwrap().as_text().is_some());
        let col = ds.column_by_name("code").unwrap();
        assert_eq!(col.len(), 101);
        if let Some(v) = col.as_text() {
            assert_eq!(v[v.len() - 1].as_deref(), Some("ABC"));
        }
    }

    #[test]
    fn csv_missing_value_options() {
        // BUG-004: user-defined missing values read from CSV attach to the
        // variable so downstream statistics exclude them.
        let dir = std::env::temp_dir();
        let path = dir.join("socstat_csv_missing_opts.csv");
        write_test_csv(&path, "income,age\n10,3\n20,40\n-999,50\n30,60\n");

        let opts = CsvReaderOptions::new()
            .missing_discrete("income", &[-999.0])
            .missing_range("age", 0.0, 5.0, None);
        let ds = CsvReader.read_path_with_options(&path, &opts).unwrap();

        // Raw values are preserved in the column...
        assert_eq!(ds.numeric_values("income").unwrap(), vec![10.0, 20.0, 30.0]);
        assert_eq!(ds.numeric_values("age").unwrap(), vec![40.0, 50.0, 60.0]);
        // ...and the missing spec is attached to the variable.
        assert!(ds.variable("income").unwrap().is_user_missing(-999.0));
        assert!(ds.variable("age").unwrap().is_user_missing(3.0));
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
