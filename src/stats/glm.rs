//! Generalized linear models via iteratively reweighted least squares (IRLS).
//!
//! Implements binary logistic regression (binomial family, logit link) as the
//! first GLM. The [`GlmFamily`] trait abstracts the variance function and link
//! function so Poisson / Gamma families can be added later without touching the
//! core [`irls_fit`] solver.
//!
//! # The IRLS algorithm
//!
//! Logistic regression models the log-odds (logit) of the probability π:
//!
//! ```text
//! g(π) = log(π / (1 − π)) = xᵀβ          (link)
//! π    = 1 / (1 + e^(−xᵀβ))               (inverse link)
//! ```
//!
//! The maximum-likelihood estimate has no closed form, so it is found by
//! iterating a weighted least-squares problem. At each step, with fitted means
//! μ and linear predictors η = Xβ:
//!
//! ```text
//! working response:  zᵢ = ηᵢ + (yᵢ − μᵢ) · g′(μᵢ)
//! weights:           wᵢ = caseᵢ / (g′(μᵢ)² · V(μᵢ))
//! update:            β ← (XᵀWX)⁻¹ XᵀWz
//! ```
//!
//! For the canonical logit link `g′(μ) = 1/(μ(1−μ))` and `V(μ) = μ(1−μ)`, so the
//! weights collapse to `wᵢ = caseᵢ · μᵢ(1−μᵢ)` — Fisher scoring for a Bernoulli
//! likelihood. This is Newton–Raphson on the log-likelihood. The normal
//! equations are solved with Cholesky decomposition, falling back to LU; a
//! singular system returns [`SocStatError::SingularMatrix`].
//!
//! # Weights, missing values, and numerical safety
//!
//! Case weights are **frequency weights** (each case counts as `weight`
//! replicates), matching the rest of the crate. Missing values (including
//! user-defined missing values at the dataset level) are excluded by listwise
//! deletion. The inverse link is evaluated with a numerically stable
//! expit, fitted probabilities are clamped away from 0/1 when computing
//! variances, and complete separation (coefficients diverging to ±∞) is
//! detected and reported as [`SocStatError::CompleteSeparation`] instead of
//! panicking.
//!
//! Every public result struct derives `Serialize`/`Deserialize`
//! (Hard Rule 1) and keeps its variable names, so a fitted model can be
//! reused for [`predict`](LogisticRegressionResult::predict) after a JSON
//! round-trip.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let ds = socstat::read().csv("data.csv")?;
//!     let model = ds.logistic_regression("defaulted", &["age", "income"])?;
//!     for c in &model.coefficients {
//!         println!("{}: β = {:.4} (z = {:.3}, p = {:.4})", c.name, c.estimate, c.z_statistic, c.p_value);
//!     }
//!     println!("AIC = {:.2}, residual deviance = {:.2}", model.aic, model.residual_deviance);
//!     let probs = model.predict(&ds)?;
//!     let cm = model.confusion_matrix(&ds, 0.5)?;
//!     println!("accuracy = {:.3}, F1 = {:.3}", cm.accuracy, cm.f1);
//!     Ok(())
//! }
//! ```

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::data::{ColumnData, Dataset, RowView};
use crate::dist::{Distribution, NormalDist};
use crate::error::{SocStatError, SocStatResult};

use super::regression::cleaned_numeric_column;

// ---------------------------------------------------------------------------
// GLM family abstraction
// ---------------------------------------------------------------------------

/// A GLM family: the variance function `V(μ)`, the link `g(μ)` and its
/// derivative `g′(μ)`, plus starting values for the fitted means.
///
/// Canonical families (binomial → logit, Poisson → log, Gamma → inverse) are
/// intended to be added here. [`irls_fit`] is generic over this trait.
pub trait GlmFamily {
    /// Variance function `V(μ)` of the response distribution.
    fn variance(&self, mean: f64) -> f64;

    /// Link function `g(μ)` mapping the mean to the linear predictor.
    fn link(&self, mean: f64) -> f64;

    /// Inverse link `g⁻¹(η)` mapping the linear predictor to the mean.
    fn inverse_link(&self, linear_pred: f64) -> f64;

    /// Derivative of the link function with respect to the mean, `g′(μ)`.
    /// Used to build the working response and the IRLS weights.
    fn deriv_link(&self, mean: f64) -> f64;

    /// Starting values for the fitted means. Should equal `g⁻¹(0)` so the
    /// initial `β = 0` (hence `η = 0`) is consistent with `μ`.
    fn initialize(&self, y: &[f64]) -> Vec<f64>;
}

/// The binomial family with the canonical logit link, used for binary
/// logistic regression.
pub struct BinomialFamily;

impl GlmFamily for BinomialFamily {
    /// `V(μ) = μ(1−μ)`.
    fn variance(&self, mean: f64) -> f64 {
        let m = mean.clamp(1e-12, 1.0 - 1e-12);
        m * (1.0 - m)
    }

    /// `g(μ) = log(μ / (1−μ))`.
    fn link(&self, mean: f64) -> f64 {
        let m = mean.clamp(1e-12, 1.0 - 1e-12);
        (m / (1.0 - m)).ln()
    }

    /// `g⁻¹(η) = 1 / (1 + e^(−η))`, evaluated stably for large `|η|`.
    fn inverse_link(&self, linear_pred: f64) -> f64 {
        if linear_pred >= 0.0 {
            let e = (-linear_pred).exp();
            1.0 / (1.0 + e)
        } else {
            let e = linear_pred.exp();
            e / (1.0 + e)
        }
    }

    /// `g′(μ) = 1 / (μ(1−μ))`.
    fn deriv_link(&self, mean: f64) -> f64 {
        let m = mean.clamp(1e-12, 1.0 - 1e-12);
        1.0 / (m * (1.0 - m))
    }

    /// Start every fitted mean at 0.5, the midpoint of the probability scale
    /// (and equal to `g⁻¹(0)`).
    fn initialize(&self, y: &[f64]) -> Vec<f64> {
        vec![0.5; y.len()]
    }
}

// ---------------------------------------------------------------------------
// IRLS solver
// ---------------------------------------------------------------------------

/// Maximum number of IRLS iterations before giving up.
const MAX_ITERATIONS: usize = 25;

/// Convergence tolerance on the largest absolute coefficient change.
const CONVERGENCE_TOLERANCE: f64 = 1e-8;

/// Coefficients beyond this magnitude indicate perfect separation.
const SEPARATION_BOUND: f64 = 1e6;

/// A converged IRLS fit: coefficients, the inverse normal-equations matrix
/// `(XᵀWX)⁻¹` (the covariance basis), and iteration bookkeeping.
struct IrlsFit {
    beta: DVector<f64>,
    /// `(XᵀWX)⁻¹` evaluated at the final fitted probabilities.
    xtwx_inv: DMatrix<f64>,
    iterations: usize,
    converged: bool,
}

/// Solve `XtWX β = XtWz` with Cholesky, falling back to LU (P4 pattern).
fn solve_normal_equations(xtwx: &DMatrix<f64>, xtwz: &DVector<f64>) -> SocStatResult<DVector<f64>> {
    if let Some(chol) = xtwx.clone().cholesky() {
        Ok(chol.solve(xtwz))
    } else {
        let lu = xtwx.clone().lu();
        lu.solve(xtwz).ok_or_else(|| {
            SocStatError::SingularMatrix(
                "the weighted normal-equations matrix is singular (check for multicollinearity)"
                    .into(),
            )
        })
    }
}

/// Invert `XtWX` with Cholesky, falling back to LU.
fn invert_normal_equations(xtwx: &DMatrix<f64>) -> SocStatResult<DMatrix<f64>> {
    let k = xtwx.nrows();
    if let Some(chol) = xtwx.clone().cholesky() {
        Ok(chol.inverse())
    } else {
        let lu = xtwx.clone().lu();
        lu.solve(&DMatrix::identity(k, k)).ok_or_else(|| {
            SocStatError::SingularMatrix(
                "could not invert the weighted normal-equations matrix".into(),
            )
        })
    }
}

/// Fit a GLM by iteratively reweighted least squares.
///
/// `x` is the full design matrix (intercept column included), `y` the
/// response, and `case_w` the frequency weights (pass all ones when
/// unweighted). Returns the coefficient vector, `(XᵀWX)⁻¹` at convergence,
/// and iteration bookkeeping.
///
/// Errors: [`SocStatError::ConvergenceNotReached`] if the coefficient changes
/// fail to shrink below tolerance; [`SocStatError::CompleteSeparation`] when
/// the data perfectly separate (fitted probabilities collapse to 0/1 and the
/// coefficients diverge); [`SocStatError::SingularMatrix`] on collinearity.
fn irls_fit(
    family: &dyn GlmFamily,
    x: &DMatrix<f64>,
    y: &[f64],
    case_w: &[f64],
) -> SocStatResult<IrlsFit> {
    let (n, p) = (x.nrows(), x.ncols());
    let mut beta = DVector::zeros(p);
    let mut mu = family.initialize(y);
    let mut eta = x * &beta;

    let mut iterations = 0;
    let mut converged = false;
    // Complete separation makes Newton steps fail to shrink: track whether the
    // largest coefficient change stops decreasing while still large.
    let mut stagnation_count = 0;
    let mut prev_max_diff = f64::INFINITY;

    for iter in 0..MAX_ITERATIONS {
        iterations = iter + 1;

        // Working response and weights at the current μ.
        let mut xtwx = DMatrix::zeros(p, p);
        let mut xtwz = DVector::zeros(p);
        let mut max_weight = 0.0f64;
        for i in 0..n {
            let m = mu[i].clamp(1e-12, 1.0 - 1e-12);
            let gd = family.deriv_link(m);
            let var = family.variance(m);
            let w = case_w[i] / (gd * gd * var);
            max_weight = max_weight.max(w);
            let z = eta[i] + (y[i] - m) * gd;
            for a in 0..p {
                xtwz[a] += x[(i, a)] * w * z;
                for b in 0..p {
                    xtwx[(a, b)] += x[(i, a)] * x[(i, b)] * w;
                }
            }
        }

        // All weights collapsed to ~0 → every fitted probability is pinned at
        // 0 or 1 → complete separation (or a degenerate response).
        if max_weight < 1e-12 {
            return Err(SocStatError::CompleteSeparation(
                "the data are completely separated: fitted probabilities are pinned at 0 or 1 \
                 and the maximum-likelihood coefficients do not exist"
                    .into(),
            ));
        }

        let beta_new = solve_normal_equations(&xtwx, &xtwz)?;
        let max_diff = (&beta_new - &beta)
            .iter()
            .fold(0.0f64, |m, d| m.max(d.abs()));

        beta = beta_new;

        // Coefficients diverging → separation, bail out early.
        let max_abs_beta = beta.iter().fold(0.0f64, |m, b| m.max(b.abs()));
        if max_abs_beta > SEPARATION_BOUND {
            return Err(SocStatError::CompleteSeparation(
                "the data are completely separated: coefficients diverge to infinity".into(),
            ));
        }

        if max_diff < CONVERGENCE_TOLERANCE {
            converged = true;
            break;
        }

        // Divergence detection: for a finite MLE, Newton steps shrink rapidly.
        // Under (quasi-)complete separation the log-likelihood is unbounded, so
        // the steps stop shrinking and the coefficients drift toward infinity
        // at a constant rate. If the largest step change stops falling by more
        // than ~10% for several iterations while still being far from the
        // tolerance, the fit is separated rather than merely slow.
        if max_diff > CONVERGENCE_TOLERANCE * 1e3 && max_diff >= 0.9 * prev_max_diff {
            stagnation_count += 1;
        } else {
            stagnation_count = 0;
        }
        if stagnation_count >= 6 {
            return Err(SocStatError::CompleteSeparation(
                "the data are quasi-separated: coefficient updates no longer shrink and the \
                 maximum-likelihood coefficients diverge"
                    .into(),
            ));
        }
        prev_max_diff = max_diff;

        // Update the linear predictor and fitted means for the next iteration.
        eta = x * &beta;
        for i in 0..n {
            mu[i] = family.inverse_link(eta[i]);
        }
    }

    if !converged {
        return Err(SocStatError::ConvergenceNotReached(iterations));
    }

    // Recompute the information matrix at the converged fit for standard errors.
    let mut xtwx = DMatrix::zeros(p, p);
    for i in 0..n {
        let m = mu[i].clamp(1e-12, 1.0 - 1e-12);
        let gd = family.deriv_link(m);
        let var = family.variance(m);
        let w = case_w[i] / (gd * gd * var);
        for a in 0..p {
            for b in 0..p {
                xtwx[(a, b)] += x[(i, a)] * x[(i, b)] * w;
            }
        }
    }
    let xtwx_inv = invert_normal_equations(&xtwx)?;

    Ok(IrlsFit { beta, xtwx_inv, iterations, converged })
}

// ---------------------------------------------------------------------------
// Result structs (Hard Rule 1: all serializable)
// ---------------------------------------------------------------------------

/// A single logistic-regression coefficient with Wald diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticCoefficient {
    /// "Intercept" for the constant term, otherwise the variable name.
    pub name: String,
    /// Point estimate (log-odds).
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// Wald z statistic = estimate / std_error.
    pub z_statistic: f64,
    /// Two-sided p-value under the standard normal.
    pub p_value: f64,
    /// 95% confidence interval `(lower, upper)`, normal-based.
    pub ci_95: (f64, f64),
}

/// Result of fitting a binary logistic regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogisticRegressionResult {
    /// Model formula, e.g. `defaulted ~ age + income`.
    pub model_formula: String,
    /// Name of the binary dependent variable.
    pub dep_var: String,
    /// Effective sample size used for the fit (sum of weights).
    pub n: f64,
    /// Intercept plus one entry per predictor, in model order.
    pub coefficients: Vec<LogisticCoefficient>,
    /// Log-likelihood of the fitted model.
    pub log_likelihood: f64,
    /// Log-likelihood of the intercept-only (null) model.
    pub null_log_likelihood: f64,
    /// Akaike information criterion: `−2ℓ + 2k`.
    pub aic: f64,
    /// Deviance of the intercept-only model.
    pub null_deviance: f64,
    /// Deviance of the fitted model (`−2ℓ` for binary data).
    pub residual_deviance: f64,
    /// `(df_null, df_residual)` — degrees of freedom of the two deviances.
    pub degrees_of_freedom: (usize, usize),
    /// McFadden's pseudo R² `= 1 − ℓ/ℓ₀`.
    pub mcfadden_r2: f64,
    /// Cox–Snell pseudo R² `= 1 − exp(2(ℓ₀ − ℓ)/n)`.
    pub cox_snell_r2: f64,
    /// Number of IRLS iterations used.
    pub iterations: usize,
    /// Whether the IRLS algorithm converged within the iteration budget.
    pub converged: bool,
}

/// Confusion matrix for a fitted classifier at a probability threshold.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfusionMatrix {
    /// Probability threshold used to label predicted positives.
    pub threshold: f64,
    /// True positives (weighted count).
    pub true_positive: f64,
    /// False positives (weighted count).
    pub false_positive: f64,
    /// True negatives (weighted count).
    pub true_negative: f64,
    /// False negatives (weighted count).
    pub false_negative: f64,
    /// Accuracy = (TP + TN) / total.
    pub accuracy: f64,
    /// Precision = TP / (TP + FP).
    pub precision: f64,
    /// Recall (sensitivity) = TP / (TP + FN).
    pub recall: f64,
    /// F1 score = harmonic mean of precision and recall.
    pub f1: f64,
}

// ---------------------------------------------------------------------------
// Fitting entry points
// ---------------------------------------------------------------------------

/// Fit a logistic regression model from typed columns.
///
/// `dep` is the binary (0/1) dependent variable; `indep` lists `(name, column)`
/// pairs for the predictors. `None` cells are dropped by listwise deletion.
/// `weights` are optional frequency weights.
///
/// # Example
///
/// ```no_run
/// use socstat::prelude::*;
/// fn main() -> SocStatResult<()> {
///     let ds = socstat::read().csv("data.csv")?;
///     let dep = ds.column_by_name("defaulted")?;
///     let age = ds.column_by_name("age")?;
///     let model = socstat::stats::glm::logistic_regression(
///         "defaulted", dep, &[("age", age)], None,
///     )?;
///     println!("AIC = {:.2}", model.aic);
///     Ok(())
/// }
/// ```
pub fn logistic_regression(
    dep_name: &str,
    dep: &ColumnData,
    indep: &[(&str, &ColumnData)],
    weights: Option<&[f64]>,
) -> SocStatResult<LogisticRegressionResult> {
    let dep_slice = dep.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
        var: dep_name.to_string(),
        expected: "Numeric",
        actual: "Text",
    })?;
    let mut indep_slices: Vec<(&str, &[Option<f64>])> = Vec::with_capacity(indep.len());
    for (name, col) in indep {
        let s = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
            var: (*name).to_string(),
            expected: "Numeric",
            actual: "Text",
        })?;
        indep_slices.push((*name, s));
    }
    fit_logistic(dep_name, dep_slice, &indep_slices, weights)
}

/// IRLS fit over aligned, already-cleaned numeric slices.
fn fit_logistic(
    dep_name: &str,
    dep: &[Option<f64>],
    indep: &[(&str, &[Option<f64>])],
    weights: Option<&[f64]>,
) -> SocStatResult<LogisticRegressionResult> {
    let n = dep.len();
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: w.len(),
        });
    }
    for (_, col) in indep {
        if col.len() != n {
            return Err(SocStatError::ColumnLengthMismatch {
                expected: n,
                got: col.len(),
            });
        }
    }
    let p = indep.len();
    if p == 0 {
        return Err(SocStatError::InsufficientData(
            "logistic regression requires at least one predictor variable".into(),
        ));
    }
    let k = p + 1;

    // Listwise deletion + binary-outcome validation + weight validation.
    let mut rows: Vec<(f64, Vec<f64>, f64)> = Vec::new();
    for i in 0..n {
        let Some(y) = dep[i] else { continue };
        if !y.is_finite() {
            continue;
        }
        if y != 0.0 && y != 1.0 {
            return Err(SocStatError::Computation(format!(
                "logistic regression requires a binary dependent variable taking only 0 or 1; \
                 found {y} in '{dep_name}'"
            )));
        }
        let mut xs = Vec::with_capacity(p);
        let mut complete = true;
        for (_, col) in indep {
            match col[i] {
                Some(v) if v.is_finite() => xs.push(v),
                _ => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        rows.push((y, xs, w));
    }
    if rows.is_empty() {
        return Err(SocStatError::InsufficientData(
            "no complete cases for logistic regression after listwise deletion".into(),
        ));
    }

    let n_eff: f64 = rows.iter().map(|r| r.2).sum();
    if n_eff <= k as f64 {
        return Err(SocStatError::InsufficientData(format!(
            "sample size too small for logistic regression: need more than {k} cases (weighted), \
             found {n_eff}"
        )));
    }

    // Both outcome classes must be present, else the intercept diverges.
    let n_ones: f64 = rows.iter().map(|r| r.0 * r.2).sum();
    if n_ones <= 0.0 || (n_eff - n_ones) <= 0.0 {
        return Err(SocStatError::CompleteSeparation(format!(
            "the dependent variable '{dep_name}' has only one observed class"
        )));
    }

    // Build the design matrix (intercept column of ones).
    let nr = rows.len();
    let mut x = DMatrix::from_element(nr, k, 0.0);
    let mut y = Vec::with_capacity(nr);
    let mut case_w = Vec::with_capacity(nr);
    for (r, (yi, xs, w)) in rows.iter().enumerate() {
        x[(r, 0)] = 1.0;
        for (j, v) in xs.iter().enumerate() {
            x[(r, j + 1)] = *v;
        }
        y.push(*yi);
        case_w.push(*w);
    }

    let fit = irls_fit(&BinomialFamily, &x, &y, &case_w)?;
    let beta = fit.beta;
    let xtwx_inv = fit.xtwx_inv;

    // Coefficient diagnostics: Wald z with a normal-based 95% CI.
    let normal = NormalDist::standard();
    let zcrit = normal.inverse_cdf(0.975);
    let mut coefficients = Vec::with_capacity(k);
    for i in 0..k {
        let name = if i == 0 {
            "Intercept".to_string()
        } else {
            indep[i - 1].0.to_string()
        };
        let estimate = beta[i];
        let se = xtwx_inv[(i, i)].sqrt();
        let (z_statistic, p_value, ci_95) = if se.is_finite() && se > 0.0 {
            let z = estimate / se;
            let p = 2.0 * (1.0 - normal.cdf(z.abs()));
            (z, p, (estimate - zcrit * se, estimate + zcrit * se))
        } else if estimate.abs() > 0.0 {
            let z = estimate.signum() * f64::INFINITY;
            (z, 0.0, (estimate, estimate))
        } else {
            (0.0, 1.0, (0.0, 0.0))
        };
        coefficients.push(LogisticCoefficient {
            name,
            estimate,
            std_error: se,
            z_statistic,
            p_value,
            ci_95,
        });
    }

    // Log-likelihoods via numerically stable log-odds forms.
    //   log π      = −softplus(−η)
    //   log(1 − π) = −softplus(η)
    let eta_vec = x * &beta;
    let mut log_likelihood = 0.0;
    for i in 0..nr {
        log_likelihood += case_w[i] * bernoulli_log_like(y[i], eta_vec[i]);
    }

    // Null model: intercept-only, fitted mean = weighted sample proportion.
    let p_bar = n_ones / n_eff;
    let eta_null = (p_bar / (1.0 - p_bar)).ln();
    let null_log_likelihood: f64 = y
        .iter()
        .zip(&case_w)
        .map(|(&yi, &w)| w * bernoulli_log_like(yi, eta_null))
        .sum();

    let aic = -2.0 * log_likelihood + 2.0 * k as f64;
    let residual_deviance = -2.0 * log_likelihood;
    let null_deviance = -2.0 * null_log_likelihood;
    let mcfadden_r2 = 1.0 - log_likelihood / null_log_likelihood;
    let cox_snell_r2 = 1.0 - ((2.0 / n_eff) * (null_log_likelihood - log_likelihood)).exp();

    let formula = if p == 1 {
        format!("{dep_name} ~ {}", indep[0].0)
    } else {
        let names: Vec<&str> = indep.iter().map(|(n, _)| *n).collect();
        format!("{dep_name} ~ {}", names.join(" + "))
    };

    Ok(LogisticRegressionResult {
        model_formula: formula,
        dep_var: dep_name.to_string(),
        n: n_eff,
        coefficients,
        log_likelihood,
        null_log_likelihood,
        aic,
        null_deviance,
        residual_deviance,
        degrees_of_freedom: (n_eff as usize - 1, n_eff as usize - k),
        mcfadden_r2,
        cox_snell_r2,
        iterations: fit.iterations,
        converged: fit.converged,
    })
}

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------

/// True for a usable frequency weight: finite and strictly positive.
fn positive_weight(w: f64) -> bool {
    w.is_finite() && w > 0.0
}

/// `y·log π + (1−y)·log(1−π)` for a Bernoulli observation, computed from the
/// linear predictor with softplus to avoid `log(0)` overflow.
fn bernoulli_log_like(y: f64, eta: f64) -> f64 {
    -y * softplus(-eta) - (1.0 - y) * softplus(eta)
}

/// `ln(1 + e^x)`, stable for large x.
fn softplus(x: f64) -> f64 {
    if x > 30.0 {
        x
    } else {
        (1.0_f64 + x.exp()).ln()
    }
}

impl LogisticRegressionResult {
    /// Fit a model from a dataset, resolving variables by name.
    ///
    /// The dependent variable must be numeric and binary (0/1). User-defined
    /// and system missing values are excluded by listwise deletion; case
    /// weights are honored when the dataset has a weight variable set.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let model = LogisticRegressionResult::fit(&ds, "defaulted", &["age", "income"])?;
    ///     println!("AIC = {:.2}", model.aic);
    ///     Ok(())
    /// }
    /// ```
    pub fn fit(dataset: &Dataset, dep_var: &str, indep_vars: &[&str]) -> SocStatResult<Self> {
        let dep_clean = cleaned_numeric_column(dataset, dep_var)?;
        let mut indep_clean: Vec<(&str, Vec<Option<f64>>)> = Vec::with_capacity(indep_vars.len());
        for v in indep_vars {
            indep_clean.push((v, cleaned_numeric_column(dataset, v)?));
        }
        let indep_refs: Vec<(&str, &[Option<f64>])> = indep_clean
            .iter()
            .map(|(name, col)| (*name, col.as_slice()))
            .collect();
        let weights = dataset.weights();
        fit_logistic(dep_var, &dep_clean, &indep_refs, weights.as_deref())
    }

    /// Predicted probabilities `P(y = 1)` for every row of a dataset.
    ///
    /// Predictors are resolved by the coefficient names recorded at fit time
    /// (so prediction keeps working after a serialization round-trip). Rows
    /// with a missing or user-missing predictor produce `None`.
    pub fn predict(&self, dataset: &Dataset) -> SocStatResult<Vec<Option<f64>>> {
        let predictors: Vec<(String, Vec<Option<f64>>)> = self
            .coefficients
            .iter()
            .skip(1)
            .map(|c| Ok((c.name.clone(), cleaned_numeric_column(dataset, &c.name)?)))
            .collect::<SocStatResult<_>>()?;
        let intercept = self.coefficients.first().map(|c| c.estimate).unwrap_or(0.0);
        let n = dataset.n_rows();
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut eta = intercept;
            let mut complete = true;
            for (c, (_, col)) in self.coefficients.iter().skip(1).zip(&predictors) {
                match col.get(i).copied().flatten() {
                    Some(x) => eta += c.estimate * x,
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            out.push(if complete { Some(expit(eta)) } else { None });
        }
        Ok(out)
    }

    /// Predict the probability for a single row via a [`RowView`].
    ///
    /// Errors if a predictor is missing, non-numeric, or not found in the row.
    pub fn predict_row(&self, row: &RowView) -> SocStatResult<f64> {
        let intercept = self.coefficients.first().map(|c| c.estimate).unwrap_or(0.0);
        let mut eta = intercept;
        for c in self.coefficients.iter().skip(1) {
            let x = row
                .numeric(&c.name)
                .ok_or_else(|| SocStatError::MissingNumber(c.name.clone()))?;
            eta += c.estimate * x;
        }
        Ok(expit(eta))
    }

    /// Confusion matrix against the observed outcomes in a dataset, at a
    /// given probability threshold.
    ///
    /// Rows where either the outcome or a predictor is missing are excluded.
    /// Cells are counted with case weights when the dataset has a weight
    /// variable set.
    pub fn confusion_matrix(&self, dataset: &Dataset, threshold: f64) -> SocStatResult<ConfusionMatrix> {
        let y_clean = cleaned_numeric_column(dataset, &self.dep_var)?;
        let probs = self.predict(dataset)?;
        let weights = dataset.weights();

        let (mut tp, mut fp, mut tn, mut fn_) = (0.0, 0.0, 0.0, 0.0);
        for i in 0..dataset.n_rows() {
            let (Some(y), Some(p)) = (y_clean[i], probs[i]) else { continue };
            let w = weights.as_ref().map(|ws| ws[i]).unwrap_or(1.0);
            if !positive_weight(w) {
                continue;
            }
            match (y > 0.5, p > threshold) {
                (true, true) => tp += w,
                (false, true) => fp += w,
                (false, false) => tn += w,
                (true, false) => fn_ += w,
            }
        }

        let total = tp + fp + tn + fn_;
        let accuracy = if total > 0.0 { (tp + tn) / total } else { f64::NAN };
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { f64::NAN };
        let recall = if tp + fn_ > 0.0 { tp / (tp + fn_) } else { f64::NAN };
        let f1 = if precision.is_finite() && recall.is_finite() && (precision + recall) > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            f64::NAN
        };

        Ok(ConfusionMatrix {
            threshold,
            true_positive: tp,
            false_positive: fp,
            true_negative: tn,
            false_negative: fn_,
            accuracy,
            precision,
            recall,
            f1,
        })
    }
}

/// `1 / (1 + e^(−η))`, stable for large `|η|`.
fn expit(eta: f64) -> f64 {
    BinomialFamily.inverse_link(eta)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    use crate::data::Value;
    use crate::data::Variable;
    use crate::stats::StatsExt;

    fn dataset(y: &[f64], x1: &[f64], x2: &[f64]) -> Dataset {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("y")).unwrap();
        d.add_var(Variable::numeric("x1")).unwrap();
        d.add_var(Variable::numeric("x2")).unwrap();
        for i in 0..y.len() {
            d.push_row(vec![Value::Number(y[i]), Value::Number(x1[i]), Value::Number(x2[i])])
                .unwrap();
        }
        d
    }

    /// The mtcars `am ~ wt + hp` dataset as a `Dataset` (variables `y`, `x1`, `x2`).
    fn mtcars() -> Dataset {
        let (mut am, mut wt, mut hp) = (Vec::new(), Vec::new(), Vec::new());
        for (a, w, h) in MTCARS {
            am.push(a);
            wt.push(w);
            hp.push(h);
        }
        dataset(&am, &wt, &hp)
    }

    // ---- Family math ----

    #[test]
    fn binomial_inverse_link_round_trip() {
        let fam = BinomialFamily;
        for m in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let eta = fam.link(m);
            assert_abs_diff_eq!(fam.inverse_link(eta), m, epsilon = 1e-12);
        }
    }

    #[test]
    fn binomial_inverse_link_stable_for_large_eta() {
        let fam = BinomialFamily;
        // No overflow / NaN for extreme linear predictors.
        assert_eq!(fam.inverse_link(1000.0), 1.0);
        assert!(fam.inverse_link(-1000.0).is_finite());
        assert_eq!(fam.inverse_link(-1000.0), 0.0);
    }

    #[test]
    fn binomial_variance_and_deriv() {
        let fam = BinomialFamily;
        assert_abs_diff_eq!(fam.variance(0.5), 0.25, epsilon = 1e-12);
        assert_abs_diff_eq!(fam.deriv_link(0.5), 4.0, epsilon = 1e-12);
    }

    // ---- Known-value comparison with R ----
    //
    // mtcars (32 rows), R: glm(am ~ wt + hp, family = binomial)
    //     (Intercept)  18.866276  7.443545   2.535  0.01125
    //     wt           -8.083463  3.068668  -2.634  0.00843
    //     hp            0.036256  0.017734   2.044  0.04095
    //     Null deviance: 43.230 on 31 df; residual deviance: 10.059 on 29 df
    //     AIC: 16.059

    // 32 (am, wt, hp) rows from the R mtcars dataset.
    const MTCARS: [(f64, f64, f64); 32] = [
        (1.0, 2.62, 110.0),
        (1.0, 2.875, 110.0),
        (1.0, 2.32, 93.0),
        (0.0, 3.215, 110.0),
        (0.0, 3.44, 175.0),
        (0.0, 3.46, 105.0),
        (0.0, 3.57, 245.0),
        (0.0, 3.19, 62.0),
        (0.0, 3.15, 95.0),
        (0.0, 3.44, 123.0),
        (0.0, 3.44, 123.0),
        (0.0, 4.07, 180.0),
        (0.0, 3.73, 180.0),
        (0.0, 3.78, 180.0),
        (0.0, 5.25, 205.0),
        (0.0, 5.424, 215.0),
        (0.0, 5.345, 230.0),
        (1.0, 2.2, 66.0),
        (1.0, 1.615, 52.0),
        (1.0, 1.835, 65.0),
        (0.0, 2.465, 97.0),
        (0.0, 3.52, 150.0),
        (0.0, 3.435, 150.0),
        (0.0, 3.84, 245.0),
        (0.0, 3.845, 175.0),
        (1.0, 1.935, 66.0),
        (1.0, 2.14, 91.0),
        (1.0, 1.513, 113.0),
        (1.0, 3.17, 264.0),
        (1.0, 2.77, 175.0),
        (1.0, 3.57, 335.0),
        (1.0, 2.78, 109.0),
    ];

    #[test]
    fn logistic_matches_r_known_values() {
        let m = LogisticRegressionResult::fit(&mtcars(), "y", &["x1", "x2"]).unwrap();

        assert_eq!(m.coefficients.len(), 3);
        assert_eq!(m.coefficients[0].name, "Intercept");
        assert_eq!(m.coefficients[1].name, "x1");
        assert_eq!(m.coefficients[2].name, "x2");

        assert_abs_diff_eq!(m.coefficients[0].estimate, 18.866276, epsilon = 1e-3);
        assert_abs_diff_eq!(m.coefficients[1].estimate, -8.083463, epsilon = 1e-3);
        assert_abs_diff_eq!(m.coefficients[2].estimate, 0.036256, epsilon = 1e-3);
        assert_abs_diff_eq!(m.coefficients[0].std_error, 7.443545, epsilon = 1e-2);
        assert_abs_diff_eq!(m.coefficients[1].std_error, 3.068668, epsilon = 1e-2);
        assert_abs_diff_eq!(m.coefficients[2].std_error, 0.017734, epsilon = 1e-3);
        assert_abs_diff_eq!(m.coefficients[0].z_statistic, 2.535, epsilon = 1e-2);
        assert_abs_diff_eq!(m.coefficients[1].z_statistic, -2.634, epsilon = 1e-2);
        assert_abs_diff_eq!(m.coefficients[2].z_statistic, 2.044, epsilon = 1e-2);

        assert_abs_diff_eq!(m.null_deviance, 43.230, epsilon = 1e-2);
        assert_abs_diff_eq!(m.residual_deviance, 10.059, epsilon = 1e-2);
        assert_abs_diff_eq!(m.aic, 16.059, epsilon = 1e-2);
        assert_eq!(m.degrees_of_freedom, (31, 29));
        assert!(m.converged);
    }

    // ---- Perfect separation ----
    //
    // response = rep(0,6), rep(1,6); treatment = rep(0,6), rep(1,6).
    // The predictor perfectly classifies the outcome → coefficients diverge.
    // R emits a "fitted probabilities numerically 0 or 1" warning and huge
    // coefficients; we must error, not panic.

    #[test]
    fn logistic_perfect_separation_errors() {
        let treat = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let resp = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let d = dataset(&resp, &treat, &[0.0; 12]);
        let res = LogisticRegressionResult::fit(&d, "y", &["x1"]);
        assert!(matches!(res, Err(SocStatError::CompleteSeparation(_))));
    }

    #[test]
    fn logistic_single_class_errors() {
        let d = dataset(&[1.0, 1.0, 1.0, 1.0], &[1.0, 2.0, 3.0, 4.0], &[0.0; 4]);
        assert!(matches!(
            LogisticRegressionResult::fit(&d, "y", &["x1"]),
            Err(SocStatError::CompleteSeparation(_))
        ));
    }

    // ---- Non-binary outcome is rejected ----

    #[test]
    fn logistic_non_binary_outcome_errors() {
        let d = dataset(&[0.0, 0.5, 1.0], &[1.0, 2.0, 3.0], &[0.0; 3]);
        let res = LogisticRegressionResult::fit(&d, "y", &["x1"]);
        assert!(matches!(res, Err(SocStatError::Computation(_))));
    }

    // ---- Synthetic convergence: recover a known coefficient vector ----

    #[test]
    fn logistic_recovers_known_coefficients() {
        // π = expit(−0.5 + 1.5·x); simulate 400 Bernoulli draws.
        let mut y = Vec::new();
        let mut x = Vec::new();
        let mut state: u64 = 0x1234_5678_9abc_def0;
        let mut rng = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64
        };
        for _ in 0..400 {
            let xi = rng() * 4.0 - 2.0;
            let p = expit(-0.5 + 1.5 * xi);
            let yi = if rng() < p { 1.0 } else { 0.0 };
            x.push(xi);
            y.push(yi);
        }
        let d = dataset(&y, &x, &[0.0; 400]);
        let m = LogisticRegressionResult::fit(&d, "y", &["x1"]).unwrap();
        assert_abs_diff_eq!(m.coefficients[0].estimate, -0.5, epsilon = 0.4);
        assert_abs_diff_eq!(m.coefficients[1].estimate, 1.5, epsilon = 0.4);
        assert!(m.converged);
    }

    // ---- Weighted fit equals frequency-expanded fit ----
    //
    // Overlapping data (both outcomes at each x) so the MLE exists.

    #[test]
    fn logistic_weighted_equals_frequency() {
        let x = vec![0.0, 0.0, 1.0, 1.0];
        let y = vec![0.0, 1.0, 1.0, 0.0];
        let w = vec![2.0, 2.0, 2.0, 2.0];

        let d1 = dataset(&y, &x, &[0.0; 4]);
        let m1 = {
            let dep = d1.column_by_name("y").unwrap();
            let x1 = d1.column_by_name("x1").unwrap();
            logistic_regression("y", dep, &[("x1", x1)], Some(&w)).unwrap()
        };
        let d2 = dataset(
            &[0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            &[0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            &[0.0; 8],
        );
        let m2 = LogisticRegressionResult::fit(&d2, "y", &["x1"]).unwrap();

        assert_abs_diff_eq!(m1.coefficients[0].estimate, m2.coefficients[0].estimate, epsilon = 1e-8);
        assert_abs_diff_eq!(m1.coefficients[1].estimate, m2.coefficients[1].estimate, epsilon = 1e-8);
        assert_abs_diff_eq!(m1.coefficients[0].std_error, m2.coefficients[0].std_error, epsilon = 1e-8);
        assert_abs_diff_eq!(m1.n, m2.n, epsilon = 1e-12);
    }

    // ---- Missing data handling ----

    #[test]
    fn logistic_listwise_deletion() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("y")).unwrap();
        d.add_var(Variable::numeric("x")).unwrap();
        for (y, x) in [
            (Value::Number(0.0), 0.0),
            (Value::Number(1.0), 1.0),
            (Value::Missing, 2.0),
            (Value::Number(1.0), 3.0),
            (Value::Number(0.0), 4.0),
        ] {
            d.push_row(vec![y, Value::Number(x)]).unwrap();
        }
        let m = LogisticRegressionResult::fit(&d, "y", &["x"]).unwrap();
        assert_eq!(m.n, 4.0);

        let pred = m.predict(&d).unwrap();
        assert!(pred[0].is_some());
        assert!(pred[1].is_some());
        assert!(pred[2].is_some()); // y missing, but x present → predictable
        assert!(pred[3].is_some());
        assert!(pred[4].is_some());
    }

    #[test]
    fn logistic_user_missing_excluded() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("y")).unwrap();
        d.add_var(Variable::numeric("x").missing_discrete(&[-9.0])).unwrap();
        for (y, x) in [(0.0, 0.0), (1.0, 1.0), (0.0, 2.0), (1.0, -9.0)] {
            d.push_row(vec![Value::Number(y), Value::Number(x)]).unwrap();
        }
        let m = LogisticRegressionResult::fit(&d, "y", &["x"]).unwrap();
        assert_eq!(m.n, 3.0);
    }

    // ---- Prediction and confusion matrix (on mtcars) ----

    #[test]
    fn predict_row_matches() {
        let d = mtcars();
        let m = LogisticRegressionResult::fit(&d, "y", &["x1", "x2"]).unwrap();
        let row = RowView::new(&d, 2);
        let p = m.predict_row(&row).unwrap();
        assert!((0.0..=1.0).contains(&p));
        let probs = m.predict(&d).unwrap();
        assert_abs_diff_eq!(probs[2].unwrap(), p, epsilon = 1e-15);
    }

    #[test]
    fn confusion_matrix_basic() {
        let d = mtcars();
        let m = LogisticRegressionResult::fit(&d, "y", &["x1", "x2"]).unwrap();
        let cm = m.confusion_matrix(&d, 0.5).unwrap();

        // Verified against R on mtcars: TP=12, FP=1, TN=18, FN=1, acc=30/32.
        assert_eq!(cm.true_positive + cm.false_positive + cm.true_negative + cm.false_negative, 32.0);
        assert_abs_diff_eq!(cm.true_positive, 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cm.false_positive, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cm.true_negative, 18.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cm.false_negative, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(cm.accuracy, 30.0 / 32.0, epsilon = 1e-12);
    }

    // ---- Serialization ----

    #[test]
    fn logistic_serde_round_trip_keeps_prediction_working() {
        let d = mtcars();
        let m = LogisticRegressionResult::fit(&d, "y", &["x1", "x2"]).unwrap();

        let json = serde_json::to_string(&m).unwrap();
        let back: LogisticRegressionResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.coefficients[1].estimate, m.coefficients[1].estimate, epsilon = 1e-15);

        let predicted = back.predict(&d).unwrap();
        let original = m.predict(&d).unwrap();
        for (a, b) in predicted.iter().zip(&original) {
            assert_abs_diff_eq!(a.unwrap(), b.unwrap(), epsilon = 1e-15);
        }
    }

    // ---- Dataset-level trait API ----

    #[test]
    fn logistic_regression_trait_works() {
        let d = mtcars();
        let m = d.logistic_regression("y", &["x1", "x2"]).unwrap();
        assert!(m.converged);
        assert_eq!(m.model_formula, "y ~ x1 + x2");
    }

    #[test]
    fn logistic_regression_missing_variable_errors() {
        let d = dataset(&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0], &[0.0; 3]);
        assert!(d.logistic_regression("y", &["nope"]).is_err());
    }

    #[test]
    fn logistic_regression_insufficient_data() {
        let d = dataset(&[0.0, 1.0], &[1.0, 2.0], &[0.0; 2]);
        assert!(LogisticRegressionResult::fit(&d, "y", &["x1"]).is_err());
    }

    #[test]
    fn logistic_regression_no_predictor_errors() {
        let d = dataset(&[0.0, 1.0, 1.0], &[1.0, 2.0, 3.0], &[0.0; 3]);
        assert!(LogisticRegressionResult::fit(&d, "y", &[]).is_err());
    }
}
