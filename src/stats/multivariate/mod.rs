//! Multivariate statistics: principal component analysis (PCA) and
//! reliability (Cronbach's alpha).
//!
//! ## Missing data
//!
//! Both analyses operate on a **strict listwise-deleted** working matrix
//! (Hard Rule 4): a case is kept only if every selected variable has a valid,
//! finite, non-missing value. PCA requires a positive-definite analysis
//! matrix, so pairwise deletion would corrupt the eigenstructure; reliability
//! needs complete rows because it sums items into a scale score.
//!
//! ## Weights
//!
//! Frequency weights (case weights) are honored, matching the rest of the
//! crate: a case counts as `weight` replicates. A weight that is missing or
//! non-positive excludes the case. When no explicit weight variable is given,
//! the dataset's configured case-weight variable (if any) is used.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let ds = socstat::read().csv("data.csv")?;
//!
//!     // PCA on the correlation matrix of the numeric variables.
//!     let pca = ds.pca(&["height", "weight", "age"], PcaMatrix::Correlation)?;
//!     for c in &pca.components {
//!         println!("λ = {:.3} ({:.1}%)", c.eigenvalue, c.explained_variance_ratio * 100.0);
//!     }
//!
//!     // Cronbach's alpha for a questionnaire scale.
//!     let rel = ds.reliability(&["q1", "q2", "q3", "q4"])?;
//!     println!("α = {:.3}", rel.alpha);
//!     Ok(())
//! }
//! ```
//!
//! Both result types derive `Serialize`/`Deserialize` (Hard Rule 1), and
//! [`PcaResult`] carries its training means/stds so [`scores`][PcaResult::scores]
//! keeps working after a JSON round-trip.

pub mod pca;
pub mod reliability;

pub use pca::{PcaComponent, PcaMatrix, PcaResult};
pub use reliability::{ItemStatistic, ReliabilityResult};

use nalgebra::{DMatrix, DVector};

use crate::data::Dataset;
use crate::error::{SocStatError, SocStatResult};

/// Perform strict listwise deletion over `vars`, returning the working matrix
/// (cases × variables), the frequency weight per retained case, and the
/// variable names.
///
/// A row is kept only when every selected variable holds a finite, non-missing
/// value (system and user-defined missing both excluded) and the case weight
/// is finite and strictly positive.
///
/// `weight_var` names an explicit weight column; `None` falls back to the
/// dataset's configured case-weight variable (if any), otherwise unweighted.
pub(crate) fn listwise_clean(
    dataset: &Dataset,
    vars: &[&str],
    weight_var: Option<&str>,
) -> SocStatResult<(DMatrix<f64>, DVector<f64>, Vec<String>)> {
    if vars.is_empty() {
        return Err(SocStatError::InsufficientData(
            "no variables provided".into(),
        ));
    }

    // Resolve each column once, converting user-missing values to None so the
    // row scan below never re-checks a variable's missing spec.
    let mut cleaned: Vec<(String, Vec<Option<f64>>)> = Vec::with_capacity(vars.len());
    for &name in vars {
        let idx = dataset.index_of(name)?;
        let var = &dataset.variables()[idx];
        let col = dataset.column(idx)?;
        let slice = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
            var: name.to_string(),
            expected: "Numeric",
            actual: "Text",
        })?;
        cleaned.push((
            name.to_string(),
            slice.iter().map(|&o| o.filter(|v| !var.is_user_missing(*v))).collect(),
        ));
    }

    let weights: Vec<f64> = match weight_var {
        Some(name) => {
            let idx = dataset.index_of(name)?;
            let var = &dataset.variables()[idx];
            let col = dataset.column(idx)?;
            let slice = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
                var: name.to_string(),
                expected: "Numeric",
                actual: "Text",
            })?;
            slice
                .iter()
                .map(|&o| o.filter(|v| !var.is_user_missing(*v)).unwrap_or(0.0))
                .collect()
        }
        None => dataset.weights().unwrap_or_else(|| vec![1.0; dataset.n_rows()]),
    };

    let n_rows = dataset.n_rows();
    let p = cleaned.len();
    let mut rows: Vec<Vec<f64>> = Vec::new();
    let mut kept_weights: Vec<f64> = Vec::new();
    for r in 0..n_rows {
        let mut row = Vec::with_capacity(p);
        let mut complete = true;
        for (_, col) in &cleaned {
            match col.get(r).copied().flatten() {
                Some(v) if v.is_finite() => row.push(v),
                _ => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let w = weights.get(r).copied().unwrap_or(0.0);
        if !(w.is_finite() && w > 0.0) {
            continue;
        }
        rows.push(row);
        kept_weights.push(w);
    }

    if rows.len() < 2 {
        return Err(SocStatError::InsufficientData(format!(
            "fewer than 2 complete cases for {} variable(s) after listwise deletion",
            p
        )));
    }

    let n = rows.len();
    let matrix = DMatrix::from_fn(n, p, |r, c| rows[r][c]);
    let var_names = cleaned.into_iter().map(|(name, _)| name).collect();
    Ok((matrix, DVector::from_vec(kept_weights), var_names))
}

/// Compute weighted means, weighted sample standard deviations, and the
/// weighted analysis matrix from a cleaned data matrix.
///
/// The covariance is
/// `C = (X - mean)ᵀ W (X - mean) / (ΣW − 1)` (two-pass, sample denominator).
/// For [`PcaMatrix::Correlation`] the covariance is rescaled to a correlation
/// matrix: `C_ij / (std_i · std_j)`.
pub(crate) fn compute_weighted_covariance(
    matrix: &DMatrix<f64>,
    weights: &DVector<f64>,
    mode: PcaMatrix,
) -> (DVector<f64>, DVector<f64>, DMatrix<f64>) {
    let n = matrix.nrows();
    let p = matrix.ncols();
    let w_sum = weights.sum();

    let mut means = DVector::zeros(p);
    for j in 0..p {
        let col = matrix.column(j);
        means[j] = col.iter().zip(weights.iter()).map(|(x, w)| x * w).sum::<f64>() / w_sum;
    }

    let mut cov = DMatrix::zeros(p, p);
    for i in 0..p {
        for j in 0..p {
            let mut s = 0.0;
            for r in 0..n {
                s += weights[r] * (matrix[(r, i)] - means[i]) * (matrix[(r, j)] - means[j]);
            }
            cov[(i, j)] = s / (w_sum - 1.0);
        }
    }

    let stds = DVector::from_fn(p, |j, _| cov[(j, j)].sqrt());
    if mode == PcaMatrix::Correlation {
        for i in 0..p {
            for j in 0..p {
                cov[(i, j)] /= stds[i] * stds[j];
            }
        }
    }
    (means, stds, cov)
}
