//! Typed column storage — the physical data layer.
//!
//! Each column stores data in a contiguous `Vec<Option<T>>`, not per-cell
//! heap allocations. For a 100k-row dataset this means one allocation per
//! column instead of 100k enum allocations — an order of magnitude less
//! memory pressure and far better cache locality.
//!
//! `ColumnData` has two variants:
//! - [`ColumnData::Numeric`] — `Vec<Option<f64>>`, the workhorse for all
//!   statistical operations.
//! - [`ColumnData::Text`] — `Vec<Option<String>>`, for labels and
//!   free-form text.
//!
//! Missing values are represented as `None` inside the `Option`, not as
//! a separate sentinel. This is unambiguous and zero-cost for valid
//! values.

use super::value::Value;
use super::variable::DataType;
use serde::{Deserialize, Serialize};

/// Physical, typed storage for a single column's data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColumnData {
    /// Numeric values stored as `f64`. `None` = missing.
    Numeric(Vec<Option<f64>>),
    /// Text values. `None` = missing.
    Text(Vec<Option<String>>),
}

impl ColumnData {
    /// Create an empty column of the given type.
    pub fn empty(data_type: DataType) -> Self {
        match data_type {
            DataType::Numeric => Self::Numeric(Vec::new()),
            DataType::Text => Self::Text(Vec::new()),
        }
    }

    /// Create a column pre-allocated with `n` missing values.
    pub fn new_missing(data_type: DataType, n: usize) -> Self {
        match data_type {
            DataType::Numeric => Self::Numeric(vec![None; n]),
            DataType::Text => Self::Text(vec![None; n]),
        }
    }

    /// Number of rows (including missing).
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::Numeric(v) => v.len(),
            Self::Text(v) => v.len(),
        }
    }

    /// Whether this column has zero rows.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The [`DataType`] of this column's storage.
    pub fn data_type(&self) -> DataType {
        match self {
            Self::Numeric(_) => DataType::Numeric,
            Self::Text(_) => DataType::Text,
        }
    }

    /// Borrow as a numeric slice. Returns `None` if the column is `Text`.
    #[inline]
    pub fn as_numeric(&self) -> Option<&[Option<f64>]> {
        match self {
            Self::Numeric(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow as a text slice. Returns `None` if the column is `Numeric`.
    #[inline]
    pub fn as_text(&self) -> Option<&[Option<String>]> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow as a mutable numeric slice.
    #[inline]
    pub fn as_numeric_mut(&mut self) -> Option<&mut [Option<f64>]> {
        match self {
            Self::Numeric(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow as a mutable text slice.
    #[inline]
    pub fn as_text_mut(&mut self) -> Option<&mut [Option<String>]> {
        match self {
            Self::Text(v) => Some(v),
            _ => None,
        }
    }

    /// Push a [`Value`] into this column, converting as needed.
    /// Returns an error on type mismatch.
    pub fn push_value(&mut self, val: Value) -> SocStatResult<()> {
        match (self, val) {
            (Self::Numeric(v), Value::Number(n)) if n.is_finite() => v.push(Some(n)),
            (Self::Numeric(v), Value::Number(_)) => v.push(None), // NaN/Inf → missing
            (Self::Numeric(_), Value::Text(_)) => {
                return Err(SocStatError::TypeMismatch {
                    var: String::new(),
                    expected: "Numeric",
                    actual: "Text",
                })
            }
            (Self::Numeric(v), Value::Missing) => v.push(None),

            (Self::Text(v), Value::Text(s)) => v.push(Some(s)),
            (Self::Text(v), Value::Number(n)) => v.push(Some(fmt_number(&n))),
            (Self::Text(v), Value::Missing) => v.push(None),
        }
        Ok(())
    }

    /// Get the cell at `row_idx` as a [`Value`].
    pub fn get_value(&self, row_idx: usize) -> Option<Value> {
        match self {
            Self::Numeric(v) => v.get(row_idx).copied().map(Value::from),
            Self::Text(v) => v.get(row_idx).cloned().map(Value::from),
        }
    }

    /// Count valid (non-missing) values.
    pub fn n_valid(&self) -> usize {
        match self {
            Self::Numeric(v) => v.iter().filter(|x| x.is_some()).count(),
            Self::Text(v) => v.iter().filter(|x| x.is_some()).count(),
        }
    }

    /// Count missing values.
    #[inline]
    pub fn n_missing(&self) -> usize {
        self.len() - self.n_valid()
    }

    /// Truncate or extend with missing values to exactly `new_len` rows.
    pub fn resize(&mut self, new_len: usize) {
        match self {
            Self::Numeric(v) => v.resize(new_len, None),
            Self::Text(v) => v.resize(new_len, None),
        }
    }
}

/// Format an f64 for display: integers without decimal point, otherwise
/// minimal representation.
fn fmt_number(n: &f64) -> String {
    if n.fract() == 0.0 {
        format!("{}", *n as i64)
    } else {
        format!("{}", n)
    }
}

use crate::error::SocStatError;
use crate::error::SocStatResult;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_basic() {
        let col = ColumnData::Numeric(vec![Some(1.0), None, Some(3.0)]);
        assert_eq!(col.len(), 3);
        assert_eq!(col.n_valid(), 2);
        assert_eq!(col.n_missing(), 1);
        assert_eq!(col.data_type(), DataType::Numeric);
    }

    #[test]
    fn text_basic() {
        let col = ColumnData::Text(vec![Some("a".into()), None, Some("b".into())]);
        assert_eq!(col.n_valid(), 2);
        assert_eq!(col.n_missing(), 1);
    }

    #[test]
    fn push_value_roundtrip() {
        let mut col = ColumnData::empty(DataType::Numeric);
        col.push_value(Value::Number(1.0)).unwrap();
        col.push_value(Value::Missing).unwrap();
        col.push_value(Value::Number(f64::NAN)).unwrap();
        assert_eq!(col.as_numeric().unwrap(), &[Some(1.0), None, None]);
    }

    #[test]
    fn push_text_into_numeric_errors() {
        let mut col = ColumnData::empty(DataType::Numeric);
        assert!(col.push_value(Value::Text("x".into())).is_err());
    }

    #[test]
    fn get_value_back() {
        let col = ColumnData::Numeric(vec![Some(1.0), None]);
        assert_eq!(col.get_value(0), Some(Value::Number(1.0)));
        assert_eq!(col.get_value(1), Some(Value::Missing));
        assert_eq!(col.get_value(5), None);
    }

    #[test]
    fn resize_behavior() {
        let mut col = ColumnData::Numeric(vec![Some(1.0), Some(2.0)]);
        col.resize(4);
        assert_eq!(col.len(), 4);
        assert_eq!(col.as_numeric().unwrap()[3], None);

        col.resize(1);
        assert_eq!(col.len(), 1);
    }
}
