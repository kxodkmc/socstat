//! Correlation and linear regression.
//!
//! Implements the Pearson / Spearman / Kendall correlation coefficients with
//! significance tests, and ordinary least squares (OLS) linear regression
//! with full model diagnostics.
//!
//! All statistics are **weight-aware** (frequency weights: each case counts
//! as `weight` replicates, matching the rest of the crate) and handle missing
//! values by listwise deletion. The column-level entry points treat `None` as
//! missing; the [`LinearRegressionResult::fit`] dataset entry point also
//! excludes user-defined missing values.
//!
//! Numerical stability: the OLS normal equations are solved with Cholesky
//! decomposition and fall back to LU; a singular design matrix returns
//! [`SocStatError::SingularMatrix`] instead of panicking. Weighted sums use
//! the two-pass (deviations-around-the-mean) form to avoid cancellation.
//!
//! Every public result struct derives `Serialize`/`Deserialize`
//! (Hard Rule 1). Coefficients carry their variable names, so a fitted model
//! can be reused for [`predict`](LinearRegressionResult::predict) even after
//! a JSON round-trip.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let ds = socstat::read().csv("data.csv")?;
//!     let model = ds.regression("income", &["age", "education"])?;
//!     println!("R² = {:.3}, adj R² = {:.3}", model.r_squared, model.adj_r_squared);
//!     for c in &model.coefficients {
//!         println!("{}: beta = {:.4} (p = {:.4})", c.name, c.estimate, c.p_value);
//!     }
//!     let predicted = model.predict(&ds)?;
//!     Ok(())
//! }
//! ```

use nalgebra::{DMatrix, DVector};
use serde::{Deserialize, Serialize};

use crate::data::{ColumnData, Dataset, RowView};
use crate::dist::{Distribution, FDist, NormalDist, StudentsTDist};
use crate::error::{SocStatError, SocStatResult};

// ---------------------------------------------------------------------------
// Result structs (Hard Rule 1: all serializable)
// ---------------------------------------------------------------------------

/// The correlation coefficient family to compute.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CorrelationMethod {
    /// Pearson product-moment correlation (interval/ratio data).
    #[default]
    Pearson,
    /// Spearman's rank correlation (monotonic association).
    Spearman,
    /// Kendall's tau-b (rank correlation robust to ties).
    Kendall,
}

/// A single correlation coefficient together with its significance test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationResult {
    /// The correlation coefficient in [-1, 1].
    pub coefficient: f64,
    /// Two-sided p-value under the null hypothesis of no association.
    pub p_value: f64,
}

/// Correlation result for a pair of variables.
///
/// Only the field matching the requested [`CorrelationMethod`] is populated;
/// the others are `None`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationPair {
    /// Name of the first variable.
    pub var1: String,
    /// Name of the second variable.
    pub var2: String,
    /// Effective sample size (sum of weights when weighted).
    pub n: f64,
    /// Pearson result, when requested.
    pub pearson: Option<CorrelationResult>,
    /// Spearman result, when requested.
    pub spearman: Option<CorrelationResult>,
    /// Kendall result, when requested.
    pub kendall: Option<CorrelationResult>,
}

/// A single regression coefficient with its full diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Coefficient {
    /// "Intercept" for the constant term, otherwise the variable name.
    pub name: String,
    /// Point estimate (β).
    pub estimate: f64,
    /// Standard error of the estimate.
    pub std_error: f64,
    /// t statistic = estimate / std_error.
    pub t_statistic: f64,
    /// Two-sided p-value for the t test.
    pub p_value: f64,
    /// 95% confidence interval `(lower, upper)`, t-based.
    pub ci_95: (f64, f64),
}

/// Result of fitting a linear regression model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearRegressionResult {
    /// Model formula, e.g. `income ~ age + education`.
    pub model_formula: String,
    /// Effective sample size used for the fit (sum of weights when weighted).
    pub n: f64,
    /// Coefficient of determination R².
    pub r_squared: f64,
    /// Adjusted R² (penalizes the number of predictors).
    pub adj_r_squared: f64,
    /// Overall F statistic (undefined for NaN when the model has no predictors).
    pub f_statistic: f64,
    /// p-value of the overall F test.
    pub f_p_value: f64,
    /// Residual standard error (sqrt of residual mean square).
    pub residuals_std_error: f64,
    /// `(df_model, df_residual)` — degrees of freedom of the F test.
    pub degrees_of_freedom: (usize, usize),
    /// Intercept plus one entry per predictor, in model order.
    pub coefficients: Vec<Coefficient>,
}

// ---------------------------------------------------------------------------
// Rank helpers
// ---------------------------------------------------------------------------

/// Average ranks with ties (mid-ranks). Inputs must be finite.
fn rank_data(data: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && data[idx[j + 1]] == data[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Average ranks treating weights as frequency weights (each case counts as
/// `weight` replicates). Mirrors the tie handling in the Mann–Whitney test.
fn weighted_ranks(data: &[f64], w: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
    let mut ranks = vec![0.0; data.len()];
    let mut cum = 0.0f64;
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        let mut block_w = 0.0;
        while j < idx.len() && data[idx[j]] == data[idx[i]] {
            block_w += w[idx[j]];
            j += 1;
        }
        let avg = cum + (block_w + 1.0) / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        cum += block_w;
        i = j;
    }
    ranks
}

// ---------------------------------------------------------------------------
// Correlation
// ---------------------------------------------------------------------------

/// True for a usable frequency weight: finite and strictly positive.
fn positive_weight(w: f64) -> bool {
    w.is_finite() && w > 0.0
}

/// Row-wise aligned numeric pairs with optional frequency weights.
type AlignedData = (Vec<f64>, Vec<f64>, Option<Vec<f64>>);

/// Align two numeric slices row-wise, dropping rows where either value is
/// missing or non-finite, or the weight is non-positive. Returns weights as
/// `Some` only when the caller supplied them.
pub(crate) fn align_slices(
    x: &[Option<f64>],
    y: &[Option<f64>],
    weights: Option<&[f64]>,
) -> SocStatResult<AlignedData> {
    if x.len() != y.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: x.len(),
            got: y.len(),
        });
    }
    if let Some(w) = weights
        && w.len() != x.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: x.len(),
            got: w.len(),
        });
    }

    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut ws = Vec::new();
    for i in 0..x.len() {
        let (Some(a), Some(b)) = (x[i], y[i]) else { continue };
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        xs.push(a);
        ys.push(b);
        ws.push(w);
    }
    if xs.is_empty() {
        return Err(SocStatError::InsufficientData(
            "no valid pairs for correlation".into(),
        ));
    }
    let w_out = if weights.is_some() { Some(ws) } else { None };
    Ok((xs, ys, w_out))
}

/// Weighted sums of squared deviations and cross-deviation around the means.
/// `w` is empty for unweighted data.
fn central_sums(x: &[f64], y: &[f64], w: &[f64], n_eff: f64) -> (f64, f64, f64) {
    let weighted = !w.is_empty();
    let (mx, my) = if weighted {
        let mx = x.iter().zip(w).map(|(xi, wi)| xi * wi).sum::<f64>() / n_eff;
        let my = y.iter().zip(w).map(|(yi, wi)| yi * wi).sum::<f64>() / n_eff;
        (mx, my)
    } else {
        let mx = x.iter().sum::<f64>() / n_eff;
        let my = y.iter().sum::<f64>() / n_eff;
        (mx, my)
    };
    if weighted {
        x.iter().zip(y).zip(w).fold((0.0, 0.0, 0.0), |(sxx, syy, sxy), ((xi, yi), wi)| {
            let dx = xi - mx;
            let dy = yi - my;
            (sxx + wi * dx * dx, syy + wi * dy * dy, sxy + wi * dx * dy)
        })
    } else {
        x.iter().zip(y).fold((0.0, 0.0, 0.0), |(sxx, syy, sxy), (xi, yi)| {
            let dx = xi - mx;
            let dy = yi - my;
            (sxx + dx * dx, syy + dy * dy, sxy + dx * dy)
        })
    }
}

/// Pearson product-moment correlation with a two-sided significance test
/// (t statistic with df = n_eff − 2).
///
/// `weights` are optional frequency weights; `None` means unweighted. The two
/// slices must have equal length and contain at least 3 valid values.
///
/// # Example
///
/// ```no_run
/// use socstat::stats::regression::pearson;
/// let r = pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.0, 5.0], None).unwrap();
/// println!("r = {:.4}, p = {:.4}", r.coefficient, r.p_value);
/// ```
pub fn pearson(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
) -> SocStatResult<CorrelationResult> {
    let n = x.len();
    if n != y.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let w = match weights {
        Some(w) if w.len() == n => w,
        Some(w) => {
            return Err(SocStatError::ColumnLengthMismatch {
                expected: n,
                got: w.len(),
            })
        }
        None => &[][..],
    };
    let n_eff: f64 = if w.is_empty() { n as f64 } else { w.iter().sum() };
    if n_eff < 3.0 {
        return Err(SocStatError::InsufficientData(
            "correlation requires at least 3 valid cases".into(),
        ));
    }

    let (sxx, syy, sxy) = central_sums(x, y, w, n_eff);
    let denom = (sxx * syy).sqrt();
    if denom <= 0.0 {
        return Err(SocStatError::Computation(
            "correlation is undefined: one variable has zero variance".into(),
        ));
    }
    let r = (sxy / denom).clamp(-1.0, 1.0);

    let df = n_eff - 2.0;
    let dist = StudentsTDist::new(df)?;
    let t = r * (df / (1.0 - r * r)).sqrt();
    let p_value = two_sided_tail(&dist, t);
    Ok(CorrelationResult { coefficient: r, p_value })
}

/// Spearman's rank correlation with significance test (t approximation).
///
/// Ties are handled with average ranks; weights are treated as frequency
/// weights. See [`pearson`] for the underlying computation on the ranks.
pub fn spearman(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
) -> SocStatResult<CorrelationResult> {
    let n = x.len();
    if n != y.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    match weights {
        Some(w) if w.len() == n => {
            let rx = weighted_ranks(x, w);
            let ry = weighted_ranks(y, w);
            pearson(&rx, &ry, Some(w))
        }
        Some(w) => Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: w.len(),
        }),
        None => {
            let rx = rank_data(x);
            let ry = rank_data(y);
            pearson(&rx, &ry, None)
        }
    }
}

/// Weighted size of each tie block (block of equal values).
fn tie_block_sizes(data: &[f64], w: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
    let mut sizes = Vec::new();
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        let mut sw = 0.0;
        while j < idx.len() && data[idx[j]] == data[idx[i]] {
            sw += if w.is_empty() { 1.0 } else { w[idx[j]] };
            j += 1;
        }
        if j - i > 1 {
            sizes.push(sw);
        }
        i = j;
    }
    sizes
}

/// Kendall's tau-b with a tie-corrected asymptotic significance test.
///
/// Computes concordant / discordant pairs in O(n²); for large samples prefer
/// [`pearson`] or [`spearman`]. Weights are treated as frequency weights; the
/// variance uses the effective sample size (asymptotic approximation).
pub fn kendall(
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
) -> SocStatResult<CorrelationResult> {
    let n = x.len();
    if n != y.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: y.len(),
        });
    }
    let w = match weights {
        Some(w) if w.len() == n => w,
        Some(w) => {
            return Err(SocStatError::ColumnLengthMismatch {
                expected: n,
                got: w.len(),
            })
        }
        None => &[][..],
    };
    let weighted = !w.is_empty();
    let n_eff: f64 = if weighted { w.iter().sum() } else { n as f64 };
    if n_eff < 3.0 {
        return Err(SocStatError::InsufficientData(
            "Kendall correlation requires at least 3 valid cases".into(),
        ));
    }

    let (mut nc, mut nd) = (0.0f64, 0.0f64);
    for i in 0..n {
        for j in (i + 1)..n {
            let wij = if weighted { w[i] * w[j] } else { 1.0 };
            let dx = x[i] - x[j];
            let dy = y[i] - y[j];
            if (dx > 0.0 && dy > 0.0) || (dx < 0.0 && dy < 0.0) {
                nc += wij;
            } else if (dx > 0.0 && dy < 0.0) || (dx < 0.0 && dy > 0.0) {
                nd += wij;
            }
        }
    }

    let tx_blocks = tie_block_sizes(x, w);
    let ty_blocks = tie_block_sizes(y, w);
    let tx: f64 = tx_blocks.iter().map(|t| t * (t - 1.0) / 2.0).sum();
    let ty: f64 = ty_blocks.iter().map(|t| t * (t - 1.0) / 2.0).sum();

    let denom = ((nc + nd + tx) * (nc + nd + ty)).sqrt();
    let tau = if denom > 0.0 { (nc - nd) / denom } else { 0.0 };

    // Kendall's tie-corrected variance of (C − D), generalized to weights.
    let s1x: f64 = tx_blocks.iter().map(|t| t * (t - 1.0) * (2.0 * t + 5.0)).sum();
    let s1y: f64 = ty_blocks.iter().map(|t| t * (t - 1.0) * (2.0 * t + 5.0)).sum();
    let s2x: f64 = tx_blocks.iter().map(|t| t * (t - 1.0) * (t - 2.0)).sum();
    let s2y: f64 = ty_blocks.iter().map(|t| t * (t - 1.0) * (t - 2.0)).sum();
    let var_s = (n_eff * (n_eff - 1.0) * (2.0 * n_eff + 5.0) - s1x - s1y) / 18.0
        + (s2x * s2y) / (9.0 * n_eff * (n_eff - 1.0) * (n_eff - 2.0));

    let p_value = if var_s > 0.0 {
        let z = (nc - nd) / var_s.sqrt();
        let normal = NormalDist::standard();
        2.0 * (1.0 - normal.cdf(z.abs()))
    } else {
        f64::NAN
    };

    Ok(CorrelationResult { coefficient: tau, p_value })
}

/// Correlation for a pair of typed columns.
///
/// `None` values are treated as missing and dropped pairwise; non-numeric
/// columns are rejected with a type error.
pub fn correlation_pair(
    var1: &str,
    var2: &str,
    x: &ColumnData,
    y: &ColumnData,
    method: CorrelationMethod,
    weights: Option<&[f64]>,
) -> SocStatResult<CorrelationPair> {
    let xs = x.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
        var: var1.to_string(),
        expected: "Numeric",
        actual: "Text",
    })?;
    let ys = y.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
        var: var2.to_string(),
        expected: "Numeric",
        actual: "Text",
    })?;
    let (x, y, w) = align_slices(xs, ys, weights)?;
    correlation_pair_aligned(var1, var2, &x, &y, w.as_deref(), method)
}

/// Build a [`CorrelationPair`] from already-aligned numeric slices.
pub(crate) fn correlation_pair_aligned(
    var1: &str,
    var2: &str,
    x: &[f64],
    y: &[f64],
    weights: Option<&[f64]>,
    method: CorrelationMethod,
) -> SocStatResult<CorrelationPair> {
    let n = weights.map(|w| w.iter().sum::<f64>()).unwrap_or(x.len() as f64);
    let mut pair = CorrelationPair {
        var1: var1.to_string(),
        var2: var2.to_string(),
        n,
        pearson: None,
        spearman: None,
        kendall: None,
    };
    let result = match method {
        CorrelationMethod::Pearson => pearson(x, y, weights),
        CorrelationMethod::Spearman => spearman(x, y, weights),
        CorrelationMethod::Kendall => kendall(x, y, weights),
    }?;
    match method {
        CorrelationMethod::Pearson => pair.pearson = Some(result),
        CorrelationMethod::Spearman => pair.spearman = Some(result),
        CorrelationMethod::Kendall => pair.kendall = Some(result),
    }
    Ok(pair)
}

// ---------------------------------------------------------------------------
// Linear regression
// ---------------------------------------------------------------------------

/// Extract a variable's numeric values with user-missing values converted to
/// `None`, so dataset-level analyses exclude them (Hard Rule 4).
pub(crate) fn cleaned_numeric_column(ds: &Dataset, name: &str) -> SocStatResult<Vec<Option<f64>>> {
    let idx = ds.index_of(name)?;
    let var = &ds.variables()[idx];
    let col = ds.column(idx)?;
    let slice = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
        var: name.to_string(),
        expected: "Numeric",
        actual: "Text",
    })?;
    Ok(slice
        .iter()
        .map(|o| o.filter(|v| !var.is_user_missing(*v)))
        .collect())
}

/// Fit a linear regression model from typed columns.
///
/// `dep` is the numeric dependent variable; `indep` lists `(name, column)`
/// pairs for the predictors. `None` cells are dropped by listwise deletion.
///
/// # Example
///
/// ```no_run
/// use socstat::prelude::*;
/// fn main() -> SocStatResult<()> {
///     let ds = socstat::read().csv("data.csv")?;
///     let dep = ds.column_by_name("income")?;
///     let age = ds.column_by_name("age")?;
///     let model = socstat::stats::regression::linear_regression(
///         "income", dep, &[("age", age)], None,
///     )?;
///     Ok(())
/// }
/// ```
pub fn linear_regression(
    dep_name: &str,
    dep: &ColumnData,
    indep: &[(&str, &ColumnData)],
    weights: Option<&[f64]>,
) -> SocStatResult<LinearRegressionResult> {
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
    fit_ols(dep_name, dep_slice, &indep_slices, weights)
}

/// OLS solver over aligned, already-cleaned numeric slices.
fn fit_ols(
    dep_name: &str,
    dep: &[Option<f64>],
    indep: &[(&str, &[Option<f64>])],
    weights: Option<&[f64]>,
) -> SocStatResult<LinearRegressionResult> {
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
            "linear regression requires at least one predictor variable".into(),
        ));
    }
    let k = p + 1;

    // Listwise deletion + weight validation in one pass.
    let mut rows: Vec<(f64, Vec<f64>, f64)> = Vec::new();
    for i in 0..n {
        let Some(y) = dep[i] else { continue };
        if !y.is_finite() {
            continue;
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
            "no complete cases for regression after listwise deletion".into(),
        ));
    }
    let n_eff: f64 = rows.iter().map(|r| r.2).sum();
    if n_eff <= k as f64 {
        return Err(SocStatError::InsufficientData(format!(
            "sample size too small for regression: need more than {k} cases (weighted), found {n_eff}"
        )));
    }

    // Design matrix with rows scaled by sqrt(weight), so plain OLS on the
    // scaled system is exactly weighted least squares (frequency weights).
    let nr = rows.len();
    let mut xw = DMatrix::from_element(nr, k, 0.0);
    let mut yw = DVector::zeros(nr);
    for (r, (y, xs, w)) in rows.iter().enumerate() {
        let s = w.sqrt();
        xw[(r, 0)] = s;
        for (j, v) in xs.iter().enumerate() {
            xw[(r, j + 1)] = s * v;
        }
        yw[r] = s * y;
    }

    // Normal equations X'X β = X'y.
    let xtx = xw.tr_mul(&xw);
    let xty = xw.tr_mul(&yw);

    // Solve with Cholesky, fall back to LU; singular → error, never panic.
    let (beta, xtx_inv) = if let Some(chol) = xtx.clone().cholesky() {
        (chol.solve(&xty), chol.inverse())
    } else {
        let lu = xtx.clone().lu();
        let beta = lu.solve(&xty).ok_or_else(|| {
            SocStatError::SingularMatrix("perfect multicollinearity detected in the design matrix".into())
        })?;
        let inv = lu.solve(&DMatrix::identity(k, k)).ok_or_else(|| {
            SocStatError::SingularMatrix("could not invert the normal-equations matrix".into())
        })?;
        (beta, inv)
    };

    // Goodness of fit.
    let y_pred = &xw * &beta;
    let residuals = &yw - &y_pred;
    let sse = residuals.dot(&residuals);

    let y_mean = rows.iter().map(|(y, _, w)| y * w).sum::<f64>() / n_eff;
    let sst: f64 = rows.iter().map(|(y, _, w)| w * (y - y_mean).powi(2)).sum();
    if sst <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "dependent variable has zero variance".into(),
        ));
    }

    let r2 = (1.0 - sse / sst).clamp(0.0, 1.0);
    let df_residual = n_eff - k as f64;
    let mse = sse / df_residual;
    let rmse = mse.sqrt();
    let adj_r2 = 1.0 - (1.0 - r2) * ((n_eff - 1.0) / df_residual);

    // Overall F test.
    let df_model = (k - 1) as f64;
    let ssr = (sst - sse).max(0.0);
    let (f_stat, f_p_value) = if mse > 0.0 {
        let f = (ssr / df_model) / mse;
        let p = 1.0 - FDist::new(df_model, df_residual)?.cdf(f);
        (f, p)
    } else if ssr > 0.0 {
        (f64::INFINITY, 0.0)
    } else {
        (f64::NAN, f64::NAN)
    };

    // Coefficient standard errors, t tests, and 95% confidence intervals.
    let t_dist = StudentsTDist::new(df_residual)?;
    let tcrit = t_dist.inverse_cdf(0.975);
    let cov = &xtx_inv * mse;
    let mut coefficients = Vec::with_capacity(k);
    for i in 0..k {
        let name = if i == 0 {
            "Intercept".to_string()
        } else {
            indep[i - 1].0.to_string()
        };
        let estimate = beta[i];
        let se = cov[(i, i)].sqrt();
        let (t_statistic, p_value, ci_95) = if se.is_finite() && se > 0.0 {
            let t = estimate / se;
            (t, two_sided_tail(&t_dist, t), (estimate - tcrit * se, estimate + tcrit * se))
        } else if estimate.abs() > 0.0 {
            let t = estimate.signum() * f64::INFINITY;
            (t, 0.0, (estimate, estimate))
        } else {
            (0.0, 1.0, (0.0, 0.0))
        };
        coefficients.push(Coefficient {
            name,
            estimate,
            std_error: se,
            t_statistic,
            p_value,
            ci_95,
        });
    }

    let formula = if p == 1 {
        format!("{dep_name} ~ {}", indep[0].0)
    } else {
        let names: Vec<&str> = indep.iter().map(|(n, _)| *n).collect();
        format!("{dep_name} ~ {}", names.join(" + "))
    };

    Ok(LinearRegressionResult {
        model_formula: formula,
        n: n_eff,
        r_squared: r2,
        adj_r_squared: adj_r2,
        f_statistic: f_stat,
        f_p_value,
        residuals_std_error: rmse,
        degrees_of_freedom: (k - 1, df_residual as usize),
        coefficients,
    })
}

// ---------------------------------------------------------------------------
// Model helpers
// ---------------------------------------------------------------------------

/// Two-sided tail probability for a symmetric distribution.
fn two_sided_tail(dist: &impl Distribution, stat: f64) -> f64 {
    2.0 * (1.0 - dist.cdf(stat.abs()))
}

impl LinearRegressionResult {
    /// Fit a model from a dataset, resolving variables by name.
    ///
    /// User-defined missing values (per each variable's missing spec) and
    /// system missing values are excluded by listwise deletion; case weights
    /// are honored when the dataset has a weight variable set.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let model = LinearRegressionResult::fit(&ds, "income", &["age", "education"])?;
    ///     println!("R² = {:.3}", model.r_squared);
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
        fit_ols(dep_var, &dep_clean, &indep_refs, weights.as_deref())
    }

    /// Predict for every row of a dataset.
    ///
    /// Predictor variables are resolved by the coefficient names recorded at
    /// fit time (so prediction keeps working after a serialization round-trip).
    /// Rows with a missing or user-missing predictor produce `None`.
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
            let mut value = intercept;
            let mut complete = true;
            for (c, (_, col)) in self.coefficients.iter().skip(1).zip(&predictors) {
                match col.get(i).copied().flatten() {
                    Some(x) => value += c.estimate * x,
                    None => {
                        complete = false;
                        break;
                    }
                }
            }
            out.push(if complete { Some(value) } else { None });
        }
        Ok(out)
    }

    /// Predict a single row via a [`RowView`].
    ///
    /// Errors if a predictor is missing, non-numeric, or not found in the row.
    pub fn predict_row(&self, row: &RowView) -> SocStatResult<f64> {
        let intercept = self.coefficients.first().map(|c| c.estimate).unwrap_or(0.0);
        let mut value = intercept;
        for c in self.coefficients.iter().skip(1) {
            let x = row
                .numeric(&c.name)
                .ok_or_else(|| SocStatError::MissingNumber(c.name.clone()))?;
            value += c.estimate * x;
        }
        Ok(value)
    }
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

    fn num_col(values: &[Option<f64>]) -> ColumnData {
        ColumnData::Numeric(values.to_vec())
    }

    fn dataset(x: &[f64], y: &[f64]) -> Dataset {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("x")).unwrap();
        d.add_var(Variable::numeric("y")).unwrap();
        for i in 0..x.len() {
            d.push_row(vec![Value::Number(x[i]), Value::Number(y[i])]).unwrap();
        }
        d
    }

    // ---- Rank helpers ----

    #[test]
    fn rank_data_basic() {
        assert_eq!(rank_data(&[3.0, 1.0, 2.0]), vec![3.0, 1.0, 2.0]);
        assert_eq!(rank_data(&[1.0, 2.0, 3.0]), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn rank_data_ties() {
        assert_eq!(rank_data(&[1.0, 1.0, 2.0]), vec![1.5, 1.5, 3.0]);
        assert_eq!(rank_data(&[2.0, 2.0, 2.0]), vec![2.0, 2.0, 2.0]);
    }

    #[test]
    fn weighted_ranks_matches_frequency() {
        // weight 2 twice = 4 replicates of value 1 (ranks 1..4 → 2.5), then one 5th case
        assert_eq!(weighted_ranks(&[1.0, 1.0, 2.0], &[2.0, 2.0, 1.0]), vec![2.5, 2.5, 5.0]);
    }

    // ---- Pearson ----

    #[test]
    fn pearson_perfect_positive() {
        let r = pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 6.0, 8.0, 10.0], None).unwrap();
        assert_abs_diff_eq!(r.coefficient, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn pearson_known_value() {
        // R: cor.test(1:5, c(1,4,2,5,3)) → r = 0.5, p = 0.3910
        let r = pearson(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1.0, 4.0, 2.0, 5.0, 3.0], None).unwrap();
        assert_abs_diff_eq!(r.coefficient, 0.5, epsilon = 1e-9);
        assert_abs_diff_eq!(r.p_value, 0.391_0, epsilon = 1e-3);
    }

    #[test]
    fn pearson_undefined_for_constant() {
        assert!(pearson(&[1.0, 2.0, 3.0], &[5.0, 5.0, 5.0], None).is_err());
    }

    #[test]
    fn pearson_insufficient_data() {
        assert!(pearson(&[1.0, 2.0], &[3.0, 4.0], None).is_err());
    }

    #[test]
    fn pearson_weighted_equals_frequency() {
        // weight 2 on 3 cases == 6 unweighted cases
        let w = vec![2.0, 2.0, 2.0];
        let a = pearson(&[1.0, 2.0, 3.0], &[2.0, 3.0, 6.0], Some(&w)).unwrap();
        let b = pearson(
            &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0],
            &[2.0, 2.0, 3.0, 3.0, 6.0, 6.0],
            None,
        )
        .unwrap();
        assert_abs_diff_eq!(a.coefficient, b.coefficient, epsilon = 1e-12);
        assert_abs_diff_eq!(a.p_value, b.p_value, epsilon = 1e-9);
    }

    // ---- Spearman ----

    #[test]
    fn spearman_with_ties() {
        // ρ = 9/√90 = 0.9486833; t = ρ√(3/(1−ρ²)) = 5.196, df=3 → p = 0.01385
        // (t-approximation; exact t₃ tail = 0.0138468)
        let r = spearman(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 2.0, 3.0, 4.0, 4.0], None).unwrap();
        assert_abs_diff_eq!(r.coefficient, 0.948_683_3, epsilon = 1e-6);
        assert_abs_diff_eq!(r.p_value, 0.013_85, epsilon = 1e-4);
    }

    // ---- Kendall ----

    #[test]
    fn kendall_known_value() {
        // x=1..5, y=c(2,2,3,4,4): C=8, D=0, ty=2 → τ_b = 8/√80 = 0.8944272
        let r = kendall(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 2.0, 3.0, 4.0, 4.0], None).unwrap();
        let expected_tau = 8.0 / (80.0_f64).sqrt();
        assert_abs_diff_eq!(r.coefficient, expected_tau, epsilon = 1e-9);
        assert!(r.p_value.is_finite());
        assert!(r.p_value > 0.0 && r.p_value < 0.05);
    }

    // ---- Linear regression ----

    #[test]
    fn linear_regression_known_value() {
        // R: lm(y ~ x) with x=1:5, y=c(2,4,5,4,5)
        let d = dataset(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.0, 5.0]);
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();

        assert_eq!(m.coefficients.len(), 2);
        assert_eq!(m.coefficients[0].name, "Intercept");
        assert_eq!(m.coefficients[1].name, "x");

        // Coefficients
        assert_abs_diff_eq!(m.coefficients[0].estimate, 2.2, epsilon = 1e-12);
        assert_abs_diff_eq!(m.coefficients[1].estimate, 0.6, epsilon = 1e-12);
        // Standard errors
        assert_abs_diff_eq!(m.coefficients[0].std_error, 0.938_083, epsilon = 1e-6);
        assert_abs_diff_eq!(m.coefficients[1].std_error, 0.282_843, epsilon = 1e-6);
        // t statistics
        assert_abs_diff_eq!(m.coefficients[0].t_statistic, 2.345_208, epsilon = 1e-5);
        assert_abs_diff_eq!(m.coefficients[1].t_statistic, 2.121_320, epsilon = 1e-5);
        // p-values (exact t₃ tails: t=2.34521 → 0.10058, t=2.12132 → 0.12403)
        assert_abs_diff_eq!(m.coefficients[0].p_value, 0.100_6, epsilon = 2e-4);
        assert_abs_diff_eq!(m.coefficients[1].p_value, 0.124_0, epsilon = 2e-4);

        // Model fit
        assert_abs_diff_eq!(m.r_squared, 0.6, epsilon = 1e-12);
        assert_abs_diff_eq!(m.adj_r_squared, 0.466_666_7, epsilon = 1e-6);
        assert_abs_diff_eq!(m.residuals_std_error, 0.894_427, epsilon = 1e-6);
        assert_abs_diff_eq!(m.f_statistic, 4.5, epsilon = 1e-12);
        // F(1,3) tail = P(|t₃| > √4.5) = 0.12403
        assert_abs_diff_eq!(m.f_p_value, 0.124_0, epsilon = 2e-4);
        assert_eq!(m.degrees_of_freedom, (1, 3));
        assert_abs_diff_eq!(m.n, 5.0, epsilon = 1e-12);

        // 95% CI, t(0.975, 3) = 3.182446
        let tcrit = 3.182_446;
        assert_abs_diff_eq!(m.coefficients[1].ci_95.0, 0.6 - tcrit * 0.282_843, epsilon = 1e-5);
        assert_abs_diff_eq!(m.coefficients[1].ci_95.1, 0.6 + tcrit * 0.282_843, epsilon = 1e-5);
    }

    #[test]
    fn linear_regression_perfect_fit() {
        let d = dataset(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5.0, 7.0, 9.0, 11.0, 13.0]);
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();
        assert_abs_diff_eq!(m.coefficients[0].estimate, 3.0, epsilon = 1e-9);
        assert_abs_diff_eq!(m.coefficients[1].estimate, 2.0, epsilon = 1e-9);
        assert_abs_diff_eq!(m.r_squared, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(m.f_p_value, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn linear_regression_weighted_equals_frequency() {
        // Weighted with w=2 must equal the unweighted doubled dataset.
        let w = vec![2.0, 2.0, 2.0];
        let d1 = dataset(&[1.0, 2.0, 3.0], &[2.0, 3.0, 6.0]);
        let m1 = {
            let dep = d1.column_by_name("y").unwrap();
            let x = d1.column_by_name("x").unwrap();
            linear_regression("y", dep, &[("x", x)], Some(&w)).unwrap()
        };
        let d2 = dataset(&[1.0, 1.0, 2.0, 2.0, 3.0, 3.0], &[2.0, 2.0, 3.0, 3.0, 6.0, 6.0]);
        let m2 = LinearRegressionResult::fit(&d2, "y", &["x"]).unwrap();
        assert_abs_diff_eq!(m1.coefficients[0].estimate, m2.coefficients[0].estimate, epsilon = 1e-9);
        assert_abs_diff_eq!(m1.coefficients[1].estimate, m2.coefficients[1].estimate, epsilon = 1e-9);
        assert_abs_diff_eq!(m1.coefficients[0].std_error, m2.coefficients[0].std_error, epsilon = 1e-9);
        assert_abs_diff_eq!(m1.r_squared, m2.r_squared, epsilon = 1e-9);
    }

    #[test]
    fn linear_regression_singular_matrix() {
        let d = dataset(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.0, 5.0]);
        let x = d.column_by_name("x").unwrap();
        let y = d.column_by_name("y").unwrap();
        // x1 and x2 identical → perfect multicollinearity.
        let res = linear_regression("y", y, &[("x1", x), ("x2", x)], None);
        assert!(matches!(res, Err(SocStatError::SingularMatrix(_))));
    }

    #[test]
    fn linear_regression_missing_data_listwise() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("x")).unwrap();
        d.add_var(Variable::numeric("y")).unwrap();
        d.push_row(vec![Value::Number(1.0), Value::Number(2.0)]).unwrap();
        d.push_row(vec![Value::Number(2.0), Value::Missing]).unwrap();
        d.push_row(vec![Value::Missing, Value::Number(5.0)]).unwrap();
        d.push_row(vec![Value::Number(4.0), Value::Number(4.0)]).unwrap();
        d.push_row(vec![Value::Number(5.0), Value::Number(5.0)]).unwrap();
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();
        assert_abs_diff_eq!(m.n, 3.0, epsilon = 1e-12);

        let predicted = m.predict(&d).unwrap();
        assert!(predicted[0].is_some());
        assert!(predicted[1].is_some()); // y missing but x present → still predictable
        assert!(predicted[2].is_none()); // x missing
        assert!(predicted[3].is_some());
        assert!(predicted[4].is_some());
    }

    #[test]
    fn linear_regression_insufficient_data() {
        let d = dataset(&[1.0, 2.0], &[2.0, 4.0]);
        assert!(LinearRegressionResult::fit(&d, "y", &["x"]).is_err());
    }

    #[test]
    fn linear_regression_no_predictor_errors() {
        let d = dataset(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!(LinearRegressionResult::fit(&d, "y", &[]).is_err());
    }

    #[test]
    fn linear_regression_user_missing_excluded() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("y")).unwrap();
        d.add_var(Variable::numeric("x").missing_discrete(&[-9.0])).unwrap();
        for (y, x) in [(2.0, 1.0), (4.0, 2.0), (6.0, 3.0), (8.0, -9.0)] {
            d.push_row(vec![Value::Number(y), Value::Number(x)]).unwrap();
        }
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();
        assert_abs_diff_eq!(m.n, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(m.coefficients[1].estimate, 2.0, epsilon = 1e-9);
    }

    #[test]
    fn predict_row_matches() {
        let d = dataset(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();
        let row = RowView::new(&d, 1);
        assert_abs_diff_eq!(m.predict_row(&row).unwrap(), 4.0, epsilon = 1e-9);
    }

    #[test]
    fn serde_round_trip_keeps_prediction_working() {
        let d = dataset(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.0, 5.0]);
        let m = LinearRegressionResult::fit(&d, "y", &["x"]).unwrap();
        let json = serde_json::to_string(&m).unwrap();
        let back: LinearRegressionResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.coefficients[1].estimate, 0.6, epsilon = 1e-15);
        // Predictors are reconstructed from serialized coefficient names.
        let predicted = back.predict(&d).unwrap();
        let original = m.predict(&d).unwrap();
        for (a, b) in predicted.iter().zip(&original) {
            assert_abs_diff_eq!(a.unwrap(), b.unwrap(), epsilon = 1e-12);
        }
    }

    // ---- Dataset-level API ----

    #[test]
    fn correlation_trait_returns_pairs() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("a")).unwrap();
        d.add_var(Variable::numeric("b")).unwrap();
        d.add_var(Variable::numeric("c")).unwrap();
        d.push_row(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)]).unwrap();
        d.push_row(vec![Value::Number(2.0), Value::Number(4.0), Value::Number(5.0)]).unwrap();
        d.push_row(vec![Value::Number(3.0), Value::Number(6.0), Value::Number(7.0)]).unwrap();
        let pairs = d.correlation(&["a", "b", "c"], CorrelationMethod::Pearson).unwrap();
        assert_eq!(pairs.len(), 3); // (a,b), (a,c), (b,c)
        for p in &pairs {
            assert!(p.pearson.is_some());
            assert!(p.spearman.is_none());
            assert!(p.kendall.is_none());
            assert_abs_diff_eq!(p.n, 3.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn correlation_trait_text_variable_errors() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("x")).unwrap();
        d.add_var(Variable::text("t")).unwrap();
        d.push_row(vec![Value::Number(1.0), Value::Text("a".into())]).unwrap();
        assert!(d.correlation(&["x", "t"], CorrelationMethod::Pearson).is_err());
    }

    #[test]
    fn correlation_pair_rejects_text() {
        let t = ColumnData::Text(vec![Some("a".into())]);
        let n = num_col(&[Some(1.0)]);
        assert!(correlation_pair("x", "t", &n, &t, CorrelationMethod::Pearson, None).is_err());
    }

    #[test]
    fn regression_trait_works() {
        let d = dataset(&[1.0, 2.0, 3.0, 4.0, 5.0], &[2.0, 4.0, 5.0, 4.0, 5.0]);
        let m = d.regression("y", &["x"]).unwrap();
        assert_abs_diff_eq!(m.coefficients[1].estimate, 0.6, epsilon = 1e-12);
    }

    #[test]
    fn regression_trait_missing_variable_errors() {
        let d = dataset(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]);
        assert!(d.regression("y", &["nope"]).is_err());
    }
}
