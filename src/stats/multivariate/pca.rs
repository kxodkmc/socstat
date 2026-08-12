//! Principal component analysis (PCA).
//!
//! PCA decomposes an analysis matrix built from the data — either the
//! correlation or the covariance matrix — via singular value decomposition
//! (SVD). For a symmetric positive semi-definite matrix the singular values
//! equal the eigenvalues, so SVD on the analysis matrix yields the principal
//! component eigenvalues and eigenvectors directly.
//!
//! This "analysis matrix → SVD" path (rather than SVD of the raw data matrix)
//! is what makes frequency weights natural to support and mirrors SPSS's
//! FACTOR "Analysis Matrix" behaviour.
//!
//! ## Scoring new data
//!
//! [`PcaResult::scores`] standardizes new observations using the **training**
//! means and standard deviations stored in the result (never recomputing them
//! on the new data — data leakage is prevented by construction). A row with a
//! missing value on any fitted variable cannot be scored and yields `NaN`.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let ds = socstat::read().csv("data.csv")?;
//!     let pca = PcaResult::compute(&ds, &["height", "weight", "age"], PcaMatrix::Correlation)?;
//!     println!("{} components explain {:.2}% of variance",
//!              pca.components.len(), pca.components.last().map(|c| c.cumulative_variance_ratio * 100.0).unwrap_or(0.0));
//!     let scores = pca.scores(&ds)?;
//!     println!("first case on component 1: {:.3}", scores[(0, 0)]);
//!     Ok(())
//! }
//! ```

use nalgebra::{DMatrix, SVD};
use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::error::{SocStatError, SocStatResult};
use crate::stats::regression::cleaned_numeric_column;

use super::{compute_weighted_covariance, listwise_clean};

/// Eigenvalues below this magnitude are treated as numerical noise and the
/// corresponding component is dropped.
const ZERO_EIGENVALUE: f64 = 1e-10;

/// The analysis matrix a PCA is computed on.
///
/// This mirrors SPSS's FACTOR analysis-matrix option.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PcaMatrix {
    /// Standardize each variable to unit variance and decompose the
    /// correlation matrix. Default, and the SPSS FACTOR default.
    #[default]
    Correlation,
    /// Center each variable (no rescaling) and decompose the covariance
    /// matrix. Variables with larger variance dominate.
    Covariance,
}

/// A single principal component.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaComponent {
    /// Variance explained by this component (singular value of the analysis
    /// matrix). Components are ordered by descending eigenvalue.
    pub eigenvalue: f64,
    /// Share of total variance: `eigenvalue / total_variance`.
    pub explained_variance_ratio: f64,
    /// Cumulative explained-variance ratio through this component.
    pub cumulative_variance_ratio: f64,
    /// Unit-length eigenvector (column of V from the SVD). Sign is arbitrary
    /// but consistent within a fitted result, so it is safe to reuse for
    /// [`scores`][PcaResult::scores].
    pub eigenvector: Vec<f64>,
    /// SPSS-style loading: `eigenvector · sqrt(eigenvalue)`. This is the
    /// correlation of each variable with the component when the analysis
    /// matrix is a correlation matrix.
    pub loadings: Vec<f64>,
}

/// Result of a PCA fit.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PcaResult {
    /// Variable names, in analysis order.
    pub variables: Vec<String>,
    /// Effective sample size (sum of weights when weighted).
    pub n: f64,
    /// Which analysis matrix was used.
    pub matrix: PcaMatrix,
    /// Principal components, descending by eigenvalue.
    pub components: Vec<PcaComponent>,
    /// Training-set weighted means, one per variable.
    pub means: Vec<f64>,
    /// Training-set weighted standard deviations, one per variable.
    pub stds: Vec<f64>,
    /// Sum of all eigenvalues (trace of the analysis matrix).
    pub total_variance: f64,
}

impl PcaResult {
    /// Compute a PCA over the numeric variables `vars` of `dataset`.
    ///
    /// Missing values are excluded by strict listwise deletion; the dataset's
    /// case-weight variable is honored when set. At least two variables and
    /// two effective cases are required. In correlation mode every variable
    /// must have non-zero variance.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let pca = PcaResult::compute(&ds, &["v1", "v2", "v3"], PcaMatrix::Correlation)?;
    ///     Ok(())
    /// }
    /// ```
    pub fn compute(dataset: &Dataset, vars: &[&str], mode: PcaMatrix) -> SocStatResult<Self> {
        if vars.len() < 2 {
            return Err(SocStatError::InsufficientData(
                "PCA requires at least two variables".into(),
            ));
        }
        let (matrix, weights, var_names) = listwise_clean(dataset, vars, None)?;
        let n = weights.sum();
        if n <= 1.0 {
            return Err(SocStatError::InsufficientData(
                "PCA requires more than one effective case".into(),
            ));
        }

        let (means, stds, analysis) = compute_weighted_covariance(&matrix, &weights, mode);
        if mode == PcaMatrix::Correlation {
            for (j, &s) in stds.iter().enumerate() {
                if s == 0.0 {
                    return Err(SocStatError::Computation(format!(
                        "variable '{}' has zero variance; PCA on the correlation matrix is undefined",
                        var_names[j]
                    )));
                }
            }
        }

        // A symmetric positive semi-definite matrix has singular values equal
        // to its eigenvalues, so no sign handling is needed. v_t rows are the
        // eigenvectors (columns of V), in descending eigenvalue order.
        let svd = SVD::new(analysis, false, true);
        let singular_values = svd.singular_values.clone();
        let v_t = svd.v_t.expect("Vᵀ is computed because compute_v = true");

        let total_variance: f64 = singular_values.iter().sum();
        let mut components = Vec::new();
        let mut cumulative = 0.0;
        for (i, &eigenvalue) in singular_values.iter().enumerate() {
            if eigenvalue < ZERO_EIGENVALUE {
                break;
            }
            let eigenvector: Vec<f64> = v_t.row(i).iter().copied().collect();
            let ratio = eigenvalue / total_variance;
            cumulative += ratio;
            let sqrt_eig = eigenvalue.sqrt();
            let loadings = eigenvector.iter().map(|&v| v * sqrt_eig).collect();
            components.push(PcaComponent {
                eigenvalue,
                explained_variance_ratio: ratio,
                cumulative_variance_ratio: cumulative,
                eigenvector,
                loadings,
            });
        }
        if components.is_empty() {
            return Err(SocStatError::Computation(
                "PCA produced no positive eigenvalues; all variables are constant".into(),
            ));
        }

        Ok(Self {
            variables: var_names,
            n,
            matrix: mode,
            components,
            means: means.iter().copied().collect(),
            stds: stds.iter().copied().collect(),
            total_variance,
        })
    }

    /// Score a dataset using the fitted components.
    ///
    /// New observations are preprocessed with the **training** means and
    /// standard deviations stored in this result — they are never recomputed
    /// on the new data. Covariance-mode PCA centers; correlation-mode PCA
    /// centers and divides by the training std. The result is an `n × q`
    /// matrix (one column per retained component). A row that is missing a
    /// variable used by the fit produces `NaN` on every component (unscorable).
    pub fn scores(&self, dataset: &Dataset) -> SocStatResult<DMatrix<f64>> {
        let cleaned: Vec<Vec<Option<f64>>> = self
            .variables
            .iter()
            .map(|v| cleaned_numeric_column(dataset, v))
            .collect::<SocStatResult<Vec<_>>>()?;

        let n = dataset.n_rows();
        let p = self.variables.len();
        let q = self.components.len();

        let processed = DMatrix::from_fn(n, p, |r, c| match cleaned[c].get(r).copied().flatten() {
            Some(x) => {
                let centered = x - self.means[c];
                if self.matrix == PcaMatrix::Correlation {
                    centered / self.stds[c]
                } else {
                    centered
                }
            }
            None => f64::NAN,
        });

        let projection = DMatrix::from_fn(p, q, |i, j| self.components[j].eigenvector[i]);
        Ok(&processed * &projection)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::data::{Value, Variable};
    use crate::stats::StatsExt;

    fn num_dataset(rows: &[&[f64]]) -> Dataset {
        let p = rows[0].len();
        let mut d = Dataset::new();
        for j in 0..p {
            d.add_var(Variable::numeric(&format!("v{j}"))).unwrap();
        }
        for row in rows {
            d.push_row(row.iter().map(|&x| Value::Number(x)).collect()).unwrap();
        }
        d
    }

    #[test]
    fn three_perfectly_correlated_variables() {
        // Correlation matrix = all ones → eigenvalues {3, 0, 0}.
        let d = num_dataset(&[
            &[1.0, 2.0, 3.0],
            &[2.0, 4.0, 6.0],
            &[3.0, 6.0, 9.0],
            &[4.0, 8.0, 12.0],
        ]);
        let r = PcaResult::compute(&d, &["v0", "v1", "v2"], PcaMatrix::Correlation).unwrap();
        assert_abs_diff_eq!(r.n, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.total_variance, 3.0, epsilon = 1e-6);
        assert_eq!(r.components.len(), 1, "zero eigenvalues are dropped");
        assert_abs_diff_eq!(r.components[0].eigenvalue, 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[0].explained_variance_ratio, 1.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[0].cumulative_variance_ratio, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn two_variable_known_eigenstructure() {
        // x = 1..5, y = (1,4,2,5,3): corr(x, y) = 0.5 (see regression tests).
        // Eigenvalues of [[1, .5], [.5, 1]] are 1.5 and 0.5.
        let d = num_dataset(&[
            &[1.0, 1.0],
            &[2.0, 4.0],
            &[3.0, 2.0],
            &[4.0, 5.0],
            &[5.0, 3.0],
        ]);
        let r = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Correlation).unwrap();
        assert_eq!(r.components.len(), 2);
        assert_abs_diff_eq!(r.components[0].eigenvalue, 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[1].eigenvalue, 0.5, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[0].explained_variance_ratio, 0.75, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[1].cumulative_variance_ratio, 1.0, epsilon = 1e-6);
        // Loadings for the first component: sqrt(1.5) * [1, 1]/sqrt(2) = sqrt(0.75).
        let l = (0.75_f64).sqrt();
        assert_abs_diff_eq!(r.components[0].loadings[0].abs(), l, epsilon = 1e-6);
        assert_abs_diff_eq!(r.components[0].loadings[1].abs(), l, epsilon = 1e-6);
        // Eigenvector of PC1 is the "size" direction [1, 1]/sqrt(2).
        let ev = &r.components[0].eigenvector;
        let inv2 = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(ev[0].abs(), inv2, epsilon = 1e-6);
        assert_abs_diff_eq!(ev[1].abs(), inv2, epsilon = 1e-6);
    }

    #[test]
    fn covariance_vs_correlation_mode() {
        // v1 = 10·v0 ⇒ one dimension of variance. In covariance mode the
        // single eigenvalue equals var(v0) + var(v1) = 2.5 + 250 = 252.5.
        let d = num_dataset(&[
            &[1.0, 10.0],
            &[2.0, 20.0],
            &[3.0, 30.0],
            &[4.0, 40.0],
            &[5.0, 50.0],
        ]);
        let cov = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_abs_diff_eq!(cov.total_variance, 252.5, epsilon = 1e-6);
        assert_eq!(cov.components.len(), 1);
        assert_abs_diff_eq!(cov.components[0].eigenvalue, 252.5, epsilon = 1e-6);

        // Correlation mode standardizes: one component with eigenvalue 2.
        let corr = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Correlation).unwrap();
        assert_abs_diff_eq!(corr.total_variance, 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(corr.components[0].eigenvalue, 2.0, epsilon = 1e-6);
    }

    #[test]
    fn weighted_equals_frequency_replication() {
        let mut dw = Dataset::new();
        dw.add_var(Variable::numeric("v0")).unwrap();
        dw.add_var(Variable::numeric("v1")).unwrap();
        dw.add_var(Variable::numeric("w").weight()).unwrap();
        for (v0, v1) in [(1.0, 2.0), (2.0, 3.0), (3.0, 6.0)] {
            dw.push_row(vec![Value::Number(v0), Value::Number(v1), Value::Number(2.0)]).unwrap();
        }
        let weighted = PcaResult::compute(&dw, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_abs_diff_eq!(weighted.n, 6.0, epsilon = 1e-12);

        let d2 = num_dataset(&[
            &[1.0, 2.0],
            &[1.0, 2.0],
            &[2.0, 3.0],
            &[2.0, 3.0],
            &[3.0, 6.0],
            &[3.0, 6.0],
        ]);
        let unweighted = PcaResult::compute(&d2, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_abs_diff_eq!(weighted.total_variance, unweighted.total_variance, epsilon = 1e-9);
        assert_abs_diff_eq!(
            weighted.components[0].eigenvalue,
            unweighted.components[0].eigenvalue,
            epsilon = 1e-9
        );
        for (a, b) in weighted.components[0].loadings.iter().zip(&unweighted.components[0].loadings) {
            assert_abs_diff_eq!(a.abs(), b.abs(), epsilon = 1e-9);
        }
    }

    #[test]
    fn scores_reuse_training_stats_and_match_manual() {
        // v1 = 2·v0 ⇒ one component whose eigenvector is [1, 2]/sqrt(5).
        let d = num_dataset(&[
            &[1.0, 2.0],
            &[2.0, 4.0],
            &[3.0, 6.0],
            &[4.0, 8.0],
        ]);
        let r = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        let scores = r.scores(&d).unwrap();
        assert_eq!(scores.nrows(), 4);
        assert_eq!(scores.ncols(), 1);

        let ev = &r.components[0].eigenvector;
        let manual_row0 = (-1.5) * ev[0] + (-3.0) * ev[1]; // row 0 centered on training means [2.5, 5.0]
        assert_abs_diff_eq!(scores[(0, 0)], manual_row0, epsilon = 1e-9);

        // Scores are centered (mean ≈ 0) because preprocessing used the
        // training means and the eigenvector is unit length.
        let mean: f64 = (0..4).map(|r| scores[(r, 0)]).sum::<f64>() / 4.0;
        assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn scores_missing_row_yields_nan() {
        let d = num_dataset(&[&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0]]);
        let r = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Correlation).unwrap();

        let mut d2 = Dataset::new();
        d2.add_var(Variable::numeric("v0")).unwrap();
        d2.add_var(Variable::numeric("v1")).unwrap();
        d2.push_row(vec![Value::Number(1.5), Value::Number(3.0)]).unwrap();
        d2.push_row(vec![Value::Number(2.5), Value::Missing]).unwrap();
        let s = r.scores(&d2).unwrap();
        assert!(s[(0, 0)].is_finite());
        assert!(s[(1, 0)].is_nan());
    }

    #[test]
    fn serde_round_trip_preserves_scores() {
        let d = num_dataset(&[
            &[1.0, 2.0, 3.0],
            &[2.0, 3.0, 5.0],
            &[3.0, 5.0, 4.0],
            &[4.0, 6.0, 6.0],
            &[5.0, 7.0, 8.0],
        ]);
        let r = PcaResult::compute(&d, &["v0", "v1", "v2"], PcaMatrix::Correlation).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: PcaResult = serde_json::from_str(&json).unwrap();

        let a = r.scores(&d).unwrap();
        let b = back.scores(&d).unwrap();
        for i in 0..a.nrows() {
            for j in 0..a.ncols() {
                assert_abs_diff_eq!(a[(i, j)], b[(i, j)], epsilon = 1e-12);
            }
        }
    }

    #[test]
    fn listwise_deletion_excludes_incomplete_rows() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("v0")).unwrap();
        d.add_var(Variable::numeric("v1")).unwrap();
        d.push_row(vec![Value::Number(1.0), Value::Number(2.0)]).unwrap();
        d.push_row(vec![Value::Number(2.0), Value::Missing]).unwrap();
        d.push_row(vec![Value::Number(3.0), Value::Number(6.0)]).unwrap();
        d.push_row(vec![Value::Missing, Value::Number(8.0)]).unwrap();
        d.push_row(vec![Value::Number(4.0), Value::Number(8.0)]).unwrap();
        let r = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_abs_diff_eq!(r.n, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn user_missing_excluded() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("v0")).unwrap();
        d.add_var(Variable::numeric("v1").missing_discrete(&[-9.0])).unwrap();
        for (v0, v1) in [(1.0, 2.0), (2.0, -9.0), (3.0, 6.0), (4.0, 8.0)] {
            d.push_row(vec![Value::Number(v0), Value::Number(v1)]).unwrap();
        }
        let r = PcaResult::compute(&d, &["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_abs_diff_eq!(r.n, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn errors() {
        // Fewer than two variables.
        let d1 = num_dataset(&[&[1.0], &[2.0], &[3.0]]);
        assert!(PcaResult::compute(&d1, &["v0"], PcaMatrix::Correlation).is_err());

        // Text variable.
        let mut dtext = Dataset::new();
        dtext.add_var(Variable::numeric("v0")).unwrap();
        dtext.add_var(Variable::text("t")).unwrap();
        dtext.push_row(vec![Value::Number(1.0), Value::Text("a".into())]).unwrap();
        dtext.push_row(vec![Value::Number(2.0), Value::Text("b".into())]).unwrap();
        assert!(PcaResult::compute(&dtext, &["v0", "t"], PcaMatrix::Correlation).is_err());

        // Zero variance in correlation mode.
        let dconst = num_dataset(&[&[1.0, 2.0], &[1.0, 3.0], &[1.0, 4.0], &[1.0, 5.0]]);
        assert!(PcaResult::compute(&dconst, &["v0", "v1"], PcaMatrix::Correlation).is_err());

        // Not enough complete cases.
        let dmiss = num_dataset(&[&[1.0, 2.0]]);
        assert!(PcaResult::compute(&dmiss, &["v0", "v1"], PcaMatrix::Covariance).is_err());
    }

    #[test]
    fn stats_ext_pca() {
        let d = num_dataset(&[&[1.0, 2.0], &[2.0, 4.0], &[3.0, 6.0], &[4.0, 8.0]]);
        let r = d.pca(&["v0", "v1"], PcaMatrix::Covariance).unwrap();
        assert_eq!(r.components.len(), 1);
    }
}
