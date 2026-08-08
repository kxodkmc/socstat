//! `socstat` — a lightweight, professional statistical analysis library for Rust.
//!
//! Provides SPSS-equivalent core capabilities as an embeddable SDK, designed
//! for integration into any software that needs data analysis.
//!
//! # Quick Start
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let mut ds = socstat::read().csv("data.csv")?;
//!
//!     // Descriptive statistics
//!     let d = ds.descriptive("income")?;
//!     println!("Mean: {:.2}, Std: {:.2}", d.mean, d.std_dev);
//!
//!     // Frequency table
//!     let freq = ds.frequencies("gender")?;
//!     for row in freq.iter() {
//!         println!("{}: {} ({:.1}%)", row.value, row.count, row.valid_percent);
//!     }
//!
//!     // Compute a new variable
//!     ds.compute("bmi", |row| {
//!         let w = row.numeric("weight")?;
//!         let h = row.numeric("height")?;
//!         Some(w / (h * h))
//!     })?;
//!
//!     // Filter cases
//!     ds.filter(|row| row.numeric("age") > Some(18.0))?;
//!
//!     socstat::write().json(&ds, "out.json")?;
//!     Ok(())
//! }
//! ```
//!
//! # Features
//!
//! | Feature    | Description              | Default |
//! |------------|--------------------------|---------|
//! | `csv`      | CSV + JSON I/O           | yes     |
//! | `excel`    | Excel (.xlsx) I/O        | no      |
//! | `datetime` | Date/time value support  | no      |
//! | `full`     | All of the above         | no      |

pub mod data;
pub mod dist;
pub mod error;
pub mod io;
pub mod stats;

pub use io::{read, write};

/// Pre-exports for convenience: `use socstat::prelude::*;`
pub mod prelude {
    pub use crate::data::{
        ColumnData, DataType, Dataset, MeasureType, MissingSpec, RowView,
        Value, ValueFormat, Variable,
    };
    pub use crate::dist::{
        ChiSquaredDist, Distribution, FDist, NormalDist, StudentsTDist,
    };
    pub use crate::error::{SocStatError, SocStatResult};
    pub use crate::stats::{
        ChiSquareTest, Crosstab, Descriptive, FrequencyRow, FrequencyTable,
        IndependentTTest, MannWhitneyUTest, OneWayAnova, StatsExt, TTestModel,
    };
    pub use crate::{read, write};
}
