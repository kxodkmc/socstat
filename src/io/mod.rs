//! I/O module — reading and writing [`Dataset`] from/to various formats.
//!
//! # Feature gates
//!
//! | Format  | Feature  | Default |
//! |---------|----------|---------|
//! | CSV     | `csv`    | yes     |
//! | JSON    | `csv`    | yes     |
//! | SPSS .sav | `sav`  | no      |

use std::path::Path;

use crate::data::Dataset;
use crate::error::SocStatResult;

#[cfg(feature = "csv")]
pub mod csv;
#[cfg(feature = "csv")]
pub mod json;
#[cfg(feature = "sav")]
pub mod sav;

/// A format reader.
pub trait Reader {
    fn read_path(&self, path: &Path) -> SocStatResult<Dataset>;
}

/// A format writer.
pub trait Writer {
    fn write_path(&self, dataset: &Dataset, path: &Path) -> SocStatResult<()>;
}

/// Entry point for reading datasets.
///
/// ```no_run
/// use socstat::prelude::*;
/// fn main() -> SocStatResult<()> {
///     let ds = socstat::read().csv("data.csv")?;
///     Ok(())
/// }
/// ```
pub fn read() -> ReadBuilder { ReadBuilder }

/// Entry point for writing datasets.
///
/// ```no_run
/// use socstat::prelude::*;
/// fn main() -> SocStatResult<()> {
///     let ds = Dataset::new();
///     socstat::write().csv(&ds, "out.csv")?;
///     Ok(())
/// }
/// ```
pub fn write() -> WriteBuilder { WriteBuilder }

pub struct ReadBuilder;
pub struct WriteBuilder;

impl ReadBuilder {
    #[cfg(feature = "csv")]
    pub fn csv(&self, path: impl AsRef<Path>) -> SocStatResult<Dataset> {
        self::csv::CsvReader.read_path(path.as_ref())
    }

    /// Read a CSV with per-column user-defined missing-value rules.
    ///
    /// ```no_run
    /// use socstat::io::csv::CsvReaderOptions;
    /// fn main() -> socstat::error::SocStatResult<()> {
    ///     let opts = CsvReaderOptions::new()
    ///         .missing_discrete("income", &[-999.0])
    ///         .missing_range("age", 0.0, 5.0, None);
    ///     let ds = socstat::read().csv_with_options("data.csv", &opts)?;
    ///     Ok(())
    /// }
    /// ```
    #[cfg(feature = "csv")]
    pub fn csv_with_options(
        &self,
        path: impl AsRef<Path>,
        options: &self::csv::CsvReaderOptions,
    ) -> SocStatResult<Dataset> {
        self::csv::CsvReader.read_path_with_options(path.as_ref(), options)
    }

    #[cfg(feature = "csv")]
    pub fn json(&self, path: impl AsRef<Path>) -> SocStatResult<Dataset> {
        self::json::JsonReader.read_path(path.as_ref())
    }

    #[cfg(feature = "sav")]
    pub fn sav(&self, path: impl AsRef<Path>) -> SocStatResult<Dataset> {
        self::sav::SavReader.read_path(path.as_ref())
    }

    /// Auto-detect format by file extension.
    pub fn auto(&self, path: impl AsRef<Path>) -> SocStatResult<Dataset> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        self.read_by_ext(&ext, path)
    }

    #[allow(unreachable_patterns, unused_variables)]
    fn read_by_ext(&self, fmt: &str, path: &Path) -> SocStatResult<Dataset> {
        match fmt {
            #[cfg(feature = "csv")]
            "csv" => self::csv::CsvReader.read_path(path),
            #[cfg(feature = "csv")]
            "json" => self::json::JsonReader.read_path(path),
            #[cfg(feature = "sav")]
            "sav" => self::sav::SavReader.read_path(path),
            _ => Err(crate::error::SocStatError::Other(
                format!("format '{fmt}' not available (feature not enabled)")
            )),
        }
    }
}

impl WriteBuilder {
    #[cfg(feature = "csv")]
    pub fn csv(&self, ds: &Dataset, path: impl AsRef<Path>) -> SocStatResult<()> {
        self::csv::CsvWriter.write_path(ds, path.as_ref())
    }

    #[cfg(feature = "csv")]
    pub fn json(&self, ds: &Dataset, path: impl AsRef<Path>) -> SocStatResult<()> {
        self::json::JsonWriter.write_path(ds, path.as_ref())
    }

    #[cfg(feature = "sav")]
    pub fn sav(&self, ds: &Dataset, path: impl AsRef<Path>) -> SocStatResult<()> {
        self::sav::SavWriter.write_path(ds, path.as_ref())
    }

    /// Auto-detect format by file extension.
    pub fn auto(&self, ds: &Dataset, path: impl AsRef<Path>) -> SocStatResult<()> {
        let path = path.as_ref();
        let ext = path.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        self.write_by_ext(&ext, ds, path)
    }

    #[allow(unreachable_patterns, unused_variables)]
    fn write_by_ext(&self, fmt: &str, ds: &Dataset, path: &Path) -> SocStatResult<()> {
        match fmt {
            #[cfg(feature = "csv")]
            "csv" => self::csv::CsvWriter.write_path(ds, path),
            #[cfg(feature = "csv")]
            "json" => self::json::JsonWriter.write_path(ds, path),
            #[cfg(feature = "sav")]
            "sav" => self::sav::SavWriter.write_path(ds, path),
            _ => Err(crate::error::SocStatError::Other(
                format!("format '{fmt}' not available (feature not enabled)")
            )),
        }
    }
}
