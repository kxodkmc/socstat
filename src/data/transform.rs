//! Data transformations: compute, recode, filter, sort.
//!
//! These are first-class operations — real analysis always requires
//! reshaping data before running statistics.
//!
//! ## Compute
//!
//! ```ignore
//! ds.compute("bmi", |row| {
//!     let h = row.numeric("height")?;
//!     let w = row.numeric("weight")?;
//!     Some(w / (h * h))
//! })?;
//! ```
//!
//! ## Recode
//!
//! ```ignore
//! ds.recode("age", |v| match v {
//!     Some(n) if n < 18.0 => Some(1.0),   // "under 18"
//!     Some(n) if n < 65.0 => Some(2.0),   // "adult"
//!     Some(_) => Some(3.0),              // "senior"
//!     None => None,
//! })?;
//! ```
//!
//! ## Filter
//!
//! ```ignore
//! ds.filter(|row| row.numeric("age") > Some(18.0))?;
//! ```

use crate::error::{SocStatError, SocStatResult};

use super::column::ColumnData;
use super::dataset::Dataset;
use super::value::Value;
use super::variable::{MeasureType, Variable};

/// A read-only view into a single row of a dataset.
/// Passed to closure-based transformations for ergonomic cell access.
pub struct RowView<'a> {
    dataset: &'a Dataset,
    row_idx: usize,
}

impl<'a> RowView<'a> {
    pub(crate) fn new(dataset: &'a Dataset, row_idx: usize) -> Self {
        Self { dataset, row_idx }
    }

    /// Get a numeric value from a variable in this row.
    /// Returns `None` for missing values or if the variable doesn't exist
    /// or isn't numeric. This is designed for ergonomic use in compute/filter
    /// closures — no `?` needed.
    pub fn numeric(&self, name: &str) -> Option<f64> {
        self.dataset.column_by_name(name)
            .ok()
            .and_then(|c| c.as_numeric())
            .and_then(|s| s.get(self.row_idx).copied().flatten())
    }

    /// Get a text value from a variable in this row.
    /// Same ergonomics as [`numeric`](Self::numeric).
    pub fn text(&self, name: &str) -> Option<&str> {
        self.dataset.column_by_name(name)
            .ok()
            .and_then(|c| c.as_text())
            .and_then(|s| s.get(self.row_idx).and_then(|o| o.as_deref()))
    }

    /// Get a raw cell value.
    pub fn value(&self, name: &str) -> SocStatResult<Value> {
        let col = self.dataset.column_by_name(name)?;
        Ok(col.get_value(self.row_idx).unwrap_or(Value::Missing))
    }
}

impl Dataset {
    /// Create a new numeric variable by computing values per row.
    ///
    /// The closure receives a [`RowView`] for the current row and returns
    /// `Some(f64)` for a valid value or `None` for missing.
    pub fn compute<F>(&mut self, name: &str, f: F) -> SocStatResult<()>
    where
        F: Fn(&RowView) -> Option<f64>,
    {
        // Collect first (immutable borrow), then mutate.
        let n = self.n_rows();
        let values: Vec<Option<f64>> = (0..n)
            .map(|i| f(&RowView::new(self, i)))
            .collect();

        let var = Variable::numeric(name).measure(MeasureType::Scale);
        self.add_var(var)?;
        let idx = self.index_of(name)?;
        self.columns[idx] = ColumnData::Numeric(values);
        Ok(())
    }

    /// Create a new text variable by computing string values per row.
    pub fn compute_text<F>(&mut self, name: &str, f: F) -> SocStatResult<()>
    where
        F: Fn(&RowView) -> Option<String>,
    {
        let n = self.n_rows();
        let values: Vec<Option<String>> = (0..n)
            .map(|i| f(&RowView::new(self, i)))
            .collect();

        let var = Variable::text(name);
        self.add_var(var)?;
        let idx = self.index_of(name)?;
        self.columns[idx] = ColumnData::Text(values);
        Ok(())
    }

    /// Recode a numeric variable in-place using a mapping closure.
    pub fn recode<F>(&mut self, name: &str, f: F) -> SocStatResult<()>
    where
        F: Fn(Option<f64>) -> Option<f64>,
    {
        let idx = self.index_of(name)?;
        let col = self.column(idx)?;
        if col.as_numeric().is_none() {
            return Err(SocStatError::TypeMismatch {
                var: name.into(), expected: "Numeric", actual: "Text",
            });
        }

        // Collect new values (immutable borrow ends before mutation).
        let old: Vec<Option<f64>> = self.column(idx)?
            .as_numeric().unwrap().to_vec();
        let new: Vec<Option<f64>> = old.into_iter().map(f).collect();

        self.columns[idx] = ColumnData::Numeric(new);
        Ok(())
    }

    /// Recode a numeric variable into a **new** variable, keeping the source
    /// column intact. Mirrors SPSS `RECODE ... INTO new_var` (UX-001).
    ///
    /// ```ignore
    /// ds.recode_into("age", "age_group", |v| match v {
    ///     Some(n) if n < 18.0 => Some(1.0), // "under 18"
    ///     Some(_) => Some(2.0),             // "adult"
    ///     None => None,
    /// })?;
    /// ```
    pub fn recode_into<F>(&mut self, src: &str, dst: &str, f: F) -> SocStatResult<()>
    where
        F: Fn(Option<f64>) -> Option<f64>,
    {
        let idx = self.index_of(src)?;
        let col = self.column(idx)?;
        if col.as_numeric().is_none() {
            return Err(SocStatError::TypeMismatch {
                var: src.into(), expected: "Numeric", actual: "Text",
            });
        }

        // Collect the mapped values before mutating the dataset.
        let old: Vec<Option<f64>> = self.column(idx)?
            .as_numeric().unwrap().to_vec();
        let new: Vec<Option<f64>> = old.into_iter().map(f).collect();

        let var = Variable::numeric(dst).measure(MeasureType::Scale);
        self.add_var(var)?;
        let didx = self.index_of(dst)?;
        self.columns[didx] = ColumnData::Numeric(new);
        Ok(())
    }

    /// Filter cases: keep only rows where the predicate returns true.
    pub fn filter<F>(&mut self, predicate: F) -> SocStatResult<usize>
    where
        F: Fn(&RowView) -> bool,
    {
        let n = self.n_rows();
        let keep: Vec<bool> = (0..n)
            .map(|i| predicate(&RowView::new(self, i)))
            .collect();
        let kept = keep.iter().filter(|&&b| b).count();

        for col in self.columns_mut() {
            match col {
                ColumnData::Numeric(v) => {
                    let filtered: Vec<Option<f64>> = v.iter()
                        .zip(&keep)
                        .filter(|(_, k)| **k)
                        .map(|(val, _)| *val)
                        .collect();
                    *v = filtered;
                }
                ColumnData::Text(v) => {
                    let filtered: Vec<Option<String>> = v.iter()
                        .zip(&keep)
                        .filter(|(_, k)| **k)
                        .map(|(val, _)| val.clone())
                        .collect();
                    *v = filtered;
                }
            }
        }
        Ok(kept)
    }

    /// Sort cases by a numeric variable (ascending by default).
    pub fn sort_by(&mut self, name: &str, descending: bool) -> SocStatResult<()> {
        let idx = self.index_of(name)?;
        let col = self.column(idx)?;
        if col.as_numeric().is_none() {
            return Err(SocStatError::TypeMismatch {
                var: name.into(), expected: "Numeric", actual: "Text",
            });
        }

        // Build sort indices from the numeric column.
        let slice = self.column(idx)?.as_numeric().unwrap();
        let mut indices: Vec<usize> = (0..slice.len()).collect();
        indices.sort_by(|&a, &b| {
            let va = slice[a].unwrap_or(f64::NEG_INFINITY);
            let vb = slice[b].unwrap_or(f64::NEG_INFINITY);
            if descending {
                vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
            } else {
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            }
        });

        // Apply the permutation to every column.
        for col in self.columns_mut() {
            match col {
                ColumnData::Numeric(v) => {
                    let sorted: Vec<Option<f64>> = indices.iter()
                        .map(|&i| v[i]).collect();
                    *v = sorted;
                }
                ColumnData::Text(v) => {
                    let sorted: Vec<Option<String>> = indices.iter()
                        .map(|&i| v[i].clone()).collect();
                    *v = sorted;
                }
            }
        }
        Ok(())
    }

    /// Select specific columns, dropping all others.
    pub fn keep(&mut self, names: &[&str]) -> SocStatResult<()> {
        let indices: Vec<usize> = names.iter()
            .map(|n| self.index_of(n))
            .collect::<SocStatResult<Vec<_>>>()?;

        self.variables = indices.iter()
            .map(|&i| self.variables[i].clone())
            .collect();
        self.columns = indices.iter()
            .map(|&i| self.columns[i].clone())
            .collect();
        // Fix weight_var index
        self.weight_var = self.weight_var
            .and_then(|w| indices.iter().position(|&i| i == w));
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Dataset {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("age")).unwrap();
        ds.add_var(Variable::numeric("weight_kg")).unwrap();
        ds.add_var(Variable::numeric("height_m")).unwrap();
        ds.push_row(vec![Value::Number(25.0), Value::Number(70.0), Value::Number(1.75)]).unwrap();
        ds.push_row(vec![Value::Number(30.0), Value::Number(85.0), Value::Number(1.80)]).unwrap();
        ds.push_row(vec![Value::Missing, Value::Number(60.0), Value::Number(1.65)]).unwrap();
        ds
    }

    #[test]
    fn compute_bmi() {
        let mut ds = sample();
        ds.compute("bmi", |row| {
            let w = row.numeric("weight_kg")?;
            let h = row.numeric("height_m")?;
            Some(w / (h * h))
        }).unwrap();

        let bmi = ds.numeric_slice("bmi").unwrap();
        // row 0: 70 / 1.75^2 = 22.857...
        assert!((bmi[0].unwrap() - 22.857).abs() < 0.01);
        // row 2: age is missing but BMI should still compute
        assert!((bmi[2].unwrap() - 22.038).abs() < 0.01);
    }

    #[test]
    fn recode_age_groups() {
        let mut ds = sample();
        ds.recode("age", |v| match v {
            Some(n) if n < 30.0 => Some(1.0), // young
            Some(_) => Some(2.0),            // adult
            None => None,
        }).unwrap();
        let groups = ds.numeric_values("age").unwrap();
        assert_eq!(groups, vec![1.0, 2.0]); // row 2 (missing) is excluded
    }

    #[test]
    fn recode_into_keeps_source() {
        let mut ds = sample();
        ds.recode_into("age", "age_group", |v| match v {
            Some(n) if n < 30.0 => Some(1.0),
            Some(_) => Some(2.0),
            None => None,
        }).unwrap();
        // Source column is unchanged.
        assert_eq!(ds.numeric_values("age").unwrap(), vec![25.0, 30.0]);
        // New column holds the recoded values, aligned by row.
        let slices = ds.numeric_slice("age_group").unwrap();
        assert_eq!(slices, &[Some(1.0), Some(2.0), None]);
    }

    #[test]
    fn filter_adults() {
        let mut ds = sample();
        let kept = ds.filter(|row| {
            row.numeric("age") > Some(18.0)
        }).unwrap();
        assert_eq!(kept, 2); // rows 0 and 1 (row 2 has missing age)
        assert_eq!(ds.n_rows(), 2);
    }

    #[test]
    fn sort_ascending() {
        let mut ds = sample();
        ds.sort_by("age", false).unwrap();
        let ages = ds.numeric_slice("age").unwrap();
        // Missing (treated as -inf) should come first in ascending order
        assert!(ages[0].is_none());
        assert_eq!(ages[1], Some(25.0));
        assert_eq!(ages[2], Some(30.0));
    }

    #[test]
    fn sort_descending() {
        let mut ds = sample();
        ds.sort_by("age", true).unwrap();
        let ages = ds.numeric_slice("age").unwrap();
        assert_eq!(ages[0], Some(30.0));
        assert_eq!(ages[1], Some(25.0));
        assert!(ages[2].is_none());
    }

    #[test]
    fn keep_columns() {
        let mut ds = sample();
        ds.keep(&["age", "weight_kg"]).unwrap();
        assert_eq!(ds.n_vars(), 2);
        assert!(ds.index_of("height_m").is_err());
    }
}
