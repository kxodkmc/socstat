//! [`Dataset`] — the central data container.
//!
//! ## Design
//!
//! Data is stored **column-major** with **typed columns**: each column is a
//! [`ColumnData`] (either `Vec<Option<f64>>` or `Vec<Option<String>>`),
//! giving contiguous memory per variable. This is dramatically more
//! cache-friendly and memory-efficient than per-cell enum allocation.
//!
//! ## Metadata
//!
//! Each column has a [`Variable`] definition (name, label, type, measure,
//! missing-value rules, value labels). The dataset tracks an optional
//! case-weight variable for weighted analysis.
//!
//! ## Transformations
//!
//! Compute, recode, filter, and sort are first-class operations — see
//! the [`transform`][super::transform] module.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::error::{SocStatError, SocStatResult};

use super::column::ColumnData;
use super::value::Value;
use super::variable::{DataType, Variable};

/// A dataset: metadata + typed columnar data.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Dataset {
    pub(crate) variables: Vec<Variable>,
    pub(crate) columns: Vec<ColumnData>,
    /// Index of the case-weight variable, if any.
    pub(crate) weight_var: Option<usize>,
    /// Dataset name (for display/reporting).
    pub(crate) name: Option<String>,
    /// Arbitrary metadata key-value pairs.
    pub(crate) metadata: BTreeMap<String, String>,
}

impl Dataset {
    /// Create an empty dataset.
    pub fn new() -> Self {
        Self::default()
    }

    // --- Metadata ---

    /// Dataset name.
    pub fn name(&self) -> Option<&str> { self.name.as_deref() }

    /// Set the dataset name.
    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Get a metadata value.
    pub fn meta(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(|s| s.as_str())
    }

    /// Set a metadata key-value pair.
    pub fn set_meta(&mut self, key: &str, value: impl Into<String>) {
        self.metadata.insert(key.into(), value.into());
    }

    // --- Variables ---

    /// Number of variables.
    #[inline]
    pub fn n_vars(&self) -> usize { self.variables.len() }

    /// Borrow all variable definitions.
    pub fn variables(&self) -> &[Variable] { &self.variables }

    /// Internal: borrow all columns (for I/O writers).
    #[allow(dead_code)]
    pub(crate) fn columns(&self) -> &[ColumnData] { &self.columns }

    /// Borrow a variable by name.
    pub fn variable(&self, name: &str) -> Option<&Variable> {
        self.variables.iter().find(|v| v.name == name)
    }

    /// Find the index of a variable by name.
    pub fn index_of(&self, name: &str) -> SocStatResult<usize> {
        self.variables.iter()
            .position(|v| v.name == name)
            .ok_or_else(|| SocStatError::VariableNotFound(name.into()))
    }

    /// Iterate over variable names.
    pub fn var_names(&self) -> impl Iterator<Item = &str> {
        self.variables.iter().map(|v| v.name.as_str())
    }

    // --- Dimensions ---

    /// Number of rows.
    #[inline]
    pub fn n_rows(&self) -> usize {
        self.columns.first().map(|c| c.len()).unwrap_or(0)
    }

    // --- Column access ---

    /// Borrow a column's typed data by index.
    pub fn column(&self, idx: usize) -> SocStatResult<&ColumnData> {
        self.columns.get(idx)
            .ok_or(SocStatError::VariableIndexOutOfBounds { index: idx, len: self.columns.len() })
    }

    /// Borrow a column's typed data by variable name.
    pub fn column_by_name(&self, name: &str) -> SocStatResult<&ColumnData> {
        let idx = self.index_of(name)?;
        self.column(idx)
    }

    /// Borrow a column's numeric data slice by name.
    /// Returns `Err` if the variable is not numeric.
    pub fn numeric_slice(&self, name: &str) -> SocStatResult<&[Option<f64>]> {
        let idx = self.index_of(name)?;
        let col = self.column(idx)?;
        col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
            var: name.into(),
            expected: "Numeric",
            actual: "Text",
        })
    }

    /// Borrow a column's text data slice by name.
    pub fn text_slice(&self, name: &str) -> SocStatResult<&[Option<String>]> {
        let idx = self.index_of(name)?;
        let col = self.column(idx)?;
        col.as_text().ok_or_else(|| SocStatError::TypeMismatch {
            var: name.into(),
            expected: "Text",
            actual: "Numeric",
        })
    }

    /// Extract valid numeric values (non-missing) as a `Vec<f64>`.
    pub fn numeric_values(&self, name: &str) -> SocStatResult<Vec<f64>> {
        let idx = self.index_of(name)?;
        let var = &self.variables[idx];
        let col = self.column(idx)?;
        let slice = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
            var: name.into(), expected: "Numeric", actual: "Text",
        })?;
        Ok(slice.iter()
            .filter_map(|o| o.filter(|n| !var.is_user_missing(*n)))
            .collect())
    }

    /// Count valid (non-missing, non-user-missing) values.
    pub fn n_valid(&self, name: &str) -> SocStatResult<usize> {
        let idx = self.index_of(name)?;
        let var = &self.variables[idx];
        let col = self.column(idx)?;
        Ok(match col {
            ColumnData::Numeric(v) => v.iter()
                .filter_map(|o| o.filter(|n| !var.is_user_missing(*n)))
                .count(),
            ColumnData::Text(v) => v.iter().filter(|o| o.is_some()).count(),
        })
    }

    /// Count missing values (system + user missing).
    pub fn n_missing(&self, name: &str) -> SocStatResult<usize> {
        let n = self.n_rows();
        Ok(n - self.n_valid(name)?)
    }

    // --- Construction ---

    /// Add a variable definition. The column is appended with missing values
    /// backfilled to current row count.
    pub fn add_var(&mut self, var: Variable) -> SocStatResult<()> {
        self.check_name(&var.name)?;
        let n = self.n_rows();
        let mut col = ColumnData::new_missing(var.data_type, n);
        let _ = &mut col; // silence
        self.columns.push(col);
        if var.is_weight {
            self.weight_var = Some(self.variables.len());
        }
        self.variables.push(var);
        Ok(())
    }

    /// Push a row of values. Length must match variable count.
    pub fn push_row(&mut self, row: Vec<Value>) -> SocStatResult<()> {
        let n_vars = self.n_vars();
        if row.len() != n_vars {
            return Err(SocStatError::RowLengthMismatch {
                expected: n_vars, got: row.len(),
            });
        }
        for (i, val) in row.into_iter().enumerate() {
            self.columns[i].push_value(val)?;
        }
        Ok(())
    }

    /// Set (replace) a column's data by name. Length must match n_rows.
    /// Set (replace) an entire column by name. The column length must
    /// match the current row count. For I/O readers and transforms.
    #[allow(dead_code)]
    pub(crate) fn set_column(&mut self, name: &str, data: ColumnData) -> SocStatResult<()> {
        let idx = self.index_of(name)?;
        let n = self.n_rows();
        if data.len() != n {
            return Err(SocStatError::ColumnLengthMismatch {
                expected: n, got: data.len(),
            });
        }
        self.columns[idx] = data;
        Ok(())
    }

    /// Get a mutable reference to a column by index (internal use).
    #[allow(dead_code)]
    pub(crate) fn column_mut(&mut self, idx: usize) -> Option<&mut ColumnData> {
        self.columns.get_mut(idx)
    }

    /// Get a mutable reference to all columns (internal use).
    pub(crate) fn columns_mut(&mut self) -> &mut [ColumnData] {
        &mut self.columns
    }

    // --- Case weights ---

    /// The index of the case-weight variable, if set.
    pub fn weight_var_index(&self) -> Option<usize> { self.weight_var }

    /// Set the case-weight variable by name.
    pub fn set_weight(&mut self, name: &str) -> SocStatResult<()> {
        let idx = self.index_of(name)?;
        if self.variables[idx].data_type != DataType::Numeric {
            return Err(SocStatError::TypeMismatch {
                var: name.into(), expected: "Numeric", actual: "Text",
            });
        }
        self.weight_var = Some(idx);
        Ok(())
    }

    /// Get the weight values for all cases (or `None` if unweighted).
    pub fn weights(&self) -> Option<Vec<f64>> {
        let idx = self.weight_var?;
        let col = self.columns.get(idx)?;
        col.as_numeric().map(|slice| {
            slice.iter().map(|o| o.unwrap_or(0.0)).collect()
        })
    }

    // --- Internal helpers ---

    fn check_name(&self, name: &str) -> SocStatResult<()> {
        if self.variables.iter().any(|v| v.name == name) {
            Err(SocStatError::DuplicateVariable(name.into()))
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Dataset {
        let mut ds = Dataset::new();
        ds.set_name("test");
        ds.add_var(Variable::numeric("age").label("Age")).unwrap();
        ds.add_var(
            Variable::text("gender")
                .value_label("M", "Male")
                .value_label("F", "Female"),
        ).unwrap();
        ds.add_var(Variable::numeric("score").missing_discrete(&[-1.0])).unwrap();
        ds.push_row(vec![Value::Number(25.0), Value::Text("M".into()), Value::Number(85.0)]).unwrap();
        ds.push_row(vec![Value::Number(30.0), Value::Text("F".into()), Value::Number(-1.0)]).unwrap();
        ds.push_row(vec![Value::Missing, Value::Missing, Value::Missing]).unwrap();
        ds
    }

    #[test]
    fn structure() {
        let ds = sample();
        assert_eq!(ds.n_vars(), 3);
        assert_eq!(ds.n_rows(), 3);
        assert_eq!(ds.name(), Some("test"));
    }

    #[test]
    fn numeric_slice_access() {
        let ds = sample();
        let age = ds.numeric_slice("age").unwrap();
        assert_eq!(age, &[Some(25.0), Some(30.0), None]);
    }

    #[test]
    fn numeric_values_filters_user_missing() {
        let ds = sample();
        // score has -1.0 as user-missing
        let scores = ds.numeric_values("score").unwrap();
        assert_eq!(scores, vec![85.0]); // -1.0 filtered, Missing filtered
    }

    #[test]
    fn n_valid_with_user_missing() {
        let ds = sample();
        assert_eq!(ds.n_valid("age").unwrap(), 2); // 2 valid, 1 system missing
        assert_eq!(ds.n_missing("age").unwrap(), 1);
        assert_eq!(ds.n_valid("score").unwrap(), 1); // 85.0 only (-1 is user-missing)
    }

    #[test]
    fn weights() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x")).unwrap();
        ds.add_var(Variable::numeric("w").weight()).unwrap();
        ds.push_row(vec![Value::Number(1.0), Value::Number(2.0)]).unwrap();
        ds.push_row(vec![Value::Number(3.0), Value::Number(5.0)]).unwrap();
        assert_eq!(ds.weights(), Some(vec![2.0, 5.0]));
    }

    #[test]
    fn duplicate_name_rejected() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x")).unwrap();
        assert!(ds.add_var(Variable::numeric("x")).is_err());
    }

    #[test]
    fn metadata_kv() {
        let mut ds = Dataset::new();
        ds.set_meta("source", "survey_2024");
        assert_eq!(ds.meta("source"), Some("survey_2024"));
        assert_eq!(ds.meta("nonexistent"), None);
    }
}
