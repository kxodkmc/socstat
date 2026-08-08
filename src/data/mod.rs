//! Data model: [`Value`], [`Variable`], [`ColumnData`], [`Dataset`],
//! and data transformations ([`compute`][Dataset::compute],
//! [`recode`][Dataset::recode], [`filter`][Dataset::filter],
//! [`sort_by`][Dataset::sort_by]).

pub mod column;
pub mod dataset;
pub mod transform;
pub mod value;
pub mod variable;

pub use column::ColumnData;
pub use dataset::Dataset;
pub use transform::RowView;
pub use value::Value;
pub use variable::{DataType, MeasureType, MissingSpec, ValueFormat, Variable};
