//! Statistics module — descriptive stats, frequency tables, crosstabs,
//! hypothesis testing, correlation, linear and logistic regression, plus
//! multivariate analysis (PCA and reliability).
//!
//! The [`StatsExt`] trait extends [`Dataset`](crate::data::Dataset) with
//! analysis methods. Statistics automatically use case weights when set.
//!
//! ## Weights
//!
//! All statistics (including the tests in [`tests`]) treat weights as
//! **frequency weights** (case weights). Complex sampling weights
//! (probability weights) are not supported in this version.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//! let ds = socstat::read().csv("data.csv")?;
//! let d = ds.descriptive("income")?;
//! println!("Mean: {:.2}, Std: {:.2}", d.mean, d.std_dev);
//! let freq = ds.frequencies("gender")?;
//! let cross = ds.crosstab("gender", "education")?;
//! let t = ds.independent_t_test("income", "gender")?;
//! println!("t = {:.3}, p = {:.5}", t.equal_variances.t_statistic, t.equal_variances.p_value);
//! # Ok(())
//! }
//! ```

pub mod crosstab;
pub mod descriptive;
pub mod frequencies;
pub mod glm;
pub mod multivariate;
pub mod regression;
pub mod tests;

pub use crosstab::Crosstab;
pub use descriptive::Descriptive;
pub use frequencies::{FrequencyRow, FrequencyTable};
pub use glm::{BinomialFamily, ConfusionMatrix, GlmFamily, LogisticCoefficient, LogisticRegressionResult};
pub use multivariate::{ItemStatistic, PcaComponent, PcaMatrix, PcaResult, ReliabilityResult};
pub use regression::{
    Coefficient, CorrelationMethod, CorrelationPair, CorrelationResult, LinearRegressionResult,
};
pub use tests::{
    ChiSquareTest, Effect, GroupSummary, IndependentTTest, LeveneResult, MannWhitneyUTest,
    OneWayAnova, RankSummary, TTestModel,
};

use crate::data::Dataset;
use crate::error::SocStatResult;

/// Extension trait providing statistical analysis methods on [`Dataset`].
///
/// Statistics automatically use case weights when a weight variable
/// is set (see [`Dataset::set_weight`]).
pub trait StatsExt {
    /// Compute comprehensive descriptive statistics for a numeric variable.
    /// Returns an error if the variable is not numeric or doesn't exist.
    fn descriptive(&self, var: &str) -> SocStatResult<Descriptive>;

    /// Build a frequency table for any variable.
    fn frequencies(&self, var: &str) -> SocStatResult<FrequencyTable>;

    /// Build a crosstabulation (contingency table) for two variables.
    fn crosstab(&self, row_var: &str, col_var: &str) -> SocStatResult<Crosstab>;

    /// Independent-samples t-test of a numeric dependent variable between
    /// the two groups defined by `group_var`. Reports pooled, Welch, and
    /// Levene's results. Errors if the grouping variable has ≠ 2 groups.
    fn independent_t_test(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<IndependentTTest>;

    /// One-way ANOVA of a numeric dependent variable across the groups of
    /// `factor_var`.
    fn one_way_anova(&self, dep_var: &str, factor_var: &str) -> SocStatResult<OneWayAnova>;

    /// Pearson chi-square test of independence between two categorical
    /// variables.
    fn chi_square_test(&self, var1: &str, var2: &str) -> SocStatResult<ChiSquareTest>;

    /// Mann–Whitney U test of a numeric dependent variable between the two
    /// groups defined by `group_var`.
    fn mann_whitney_u_test(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<MannWhitneyUTest>;

    /// Correlation between every pair of the given numeric variables.
    ///
    /// Returns one [`CorrelationPair`] per unordered pair (upper triangle).
    /// Only the coefficient matching `method` is populated. Missing values
    /// (including user-missing values) are excluded pairwise; case weights
    /// are honored.
    ///
    /// # Example
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     for p in ds.correlation(&["height", "weight", "age"], CorrelationMethod::Pearson)? {
    ///         if let Some(r) = &p.pearson {
    ///             println!("{} ~ {}: r = {:.3}, p = {:.4}", p.var1, p.var2, r.coefficient, r.p_value);
    ///         }
    ///     }
    ///     Ok(())
    /// }
    /// ```
    fn correlation(
        &self,
        vars: &[&str],
        method: CorrelationMethod,
    ) -> SocStatResult<Vec<CorrelationPair>>;

    /// Correlation between exactly two variables (UX-004).
    ///
    /// Convenience wrapper over [`correlation`](Self::correlation) for the
    /// common two-variable case; returns the single [`CorrelationPair`].
    fn correlation_pair(
        &self,
        var1: &str,
        var2: &str,
        method: CorrelationMethod,
    ) -> SocStatResult<CorrelationPair>;

    /// Fit a linear regression of `dep_var` on `indep_vars` (OLS).
    ///
    /// An intercept is always included. User-missing and system-missing
    /// values are excluded by listwise deletion; case weights are honored.
    /// A singular design matrix returns `SocStatError::SingularMatrix`.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let model = ds.regression("income", &["age", "education"])?;
    ///     println!("R² = {:.3}, F = {:.2}, p = {:.4}",
    ///              model.r_squared, model.f_statistic, model.f_p_value);
    ///     Ok(())
    /// }
    /// ```
    fn regression(&self, dep_var: &str, indep_vars: &[&str]) -> SocStatResult<LinearRegressionResult>;

    /// Fit a binary logistic regression of the 0/1 outcome `dep_var` on
    /// `indep_vars` using iteratively reweighted least squares.
    ///
    /// An intercept is always included. The dependent variable must be
    /// numeric and binary (0/1). User-missing and system-missing values are
    /// excluded by listwise deletion; case weights are honored. Perfectly
    /// separated data return `SocStatError::CompleteSeparation`; a singular
    /// weighted design matrix returns `SocStatError::SingularMatrix`.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let model = ds.logistic_regression("defaulted", &["age", "income"])?;
    ///     println!("AIC = {:.2}, residual deviance = {:.2}",
    ///              model.aic, model.residual_deviance);
    ///     Ok(())
    /// }
    /// ```
    fn logistic_regression(
        &self,
        dep_var: &str,
        indep_vars: &[&str],
    ) -> SocStatResult<LogisticRegressionResult>;

    /// Principal component analysis of `vars` on the given analysis matrix.
    ///
    /// Missing values are excluded by strict listwise deletion; the dataset's
    /// case-weight variable is honored when set. The returned [`PcaResult`]
    /// stores the training means/stds, so [`PcaResult::scores`] can score new
    /// data without re-estimating the standardization.
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let pca = ds.pca(&["height", "weight", "age"], PcaMatrix::Correlation)?;
    ///     for c in &pca.components {
    ///         println!("λ = {:.3} ({:.1}%)", c.eigenvalue, c.explained_variance_ratio * 100.0);
    ///     }
    ///     Ok(())
    /// }
    /// ```
    fn pca(&self, vars: &[&str], mode: PcaMatrix) -> SocStatResult<PcaResult>;

    /// Cronbach's alpha reliability analysis of the numeric items `vars`.
    ///
    /// Missing values are excluded by strict listwise deletion; the dataset's
    /// case-weight variable is honored when set. Reports the overall alpha
    /// plus per-item diagnostics (corrected item-total correlation and
    /// alpha-if-deleted).
    ///
    /// ```no_run
    /// use socstat::prelude::*;
    /// fn main() -> SocStatResult<()> {
    ///     let ds = socstat::read().csv("data.csv")?;
    ///     let rel = ds.reliability(&["q1", "q2", "q3"])?;
    ///     println!("α = {:.3}", rel.alpha);
    ///     Ok(())
    /// }
    /// ```
    fn reliability(&self, vars: &[&str]) -> SocStatResult<ReliabilityResult>;
}

impl StatsExt for Dataset {
    fn descriptive(&self, var: &str) -> SocStatResult<Descriptive> {
        let idx = self.index_of(var)?;
        let var_def = &self.variables()[idx];
        let col = self.column(idx)?;
        let slice = col.as_numeric().ok_or_else(|| crate::error::SocStatError::TypeMismatch {
            var: var.into(),
            expected: "Numeric",
            actual: "Text",
        })?;

        // Extract valid values, excluding system-missing and user-missing
        // values (Hard Rule 4). When a weight variable is set, drop the
        // weight of every excluded case so the weight slice stays aligned
        // with the data slice — otherwise `compute()` would silently fall
        // back to an unweighted result (BUG-001).
        let weights = self.weights();
        let (data, aligned_weights): (Vec<f64>, Option<Vec<f64>>) = match &weights {
            Some(w) => {
                let (d, aw): (Vec<f64>, Vec<f64>) = slice
                    .iter()
                    .zip(w.iter())
                    .filter_map(|(val, wt)| {
                        let v = (*val)?;
                        if var_def.is_user_missing(v) || !(wt.is_finite() && *wt > 0.0) {
                            None
                        } else {
                            Some((v, *wt))
                        }
                    })
                    .unzip();
                (d, if aw.is_empty() { None } else { Some(aw) })
            }
            None => {
                let d: Vec<f64> = slice
                    .iter()
                    .filter_map(|o| o.filter(|v| !var_def.is_user_missing(*v)))
                    .collect();
                (d, None)
            }
        };

        if data.is_empty() {
            return Err(crate::error::SocStatError::InsufficientData(
                format!("no valid values in variable '{var}'"),
            ));
        }

        Ok(descriptive::compute(&data, aligned_weights.as_deref()))
    }

    fn frequencies(&self, var: &str) -> SocStatResult<FrequencyTable> {
        let idx = self.index_of(var)?;
        let col = self.column(idx)?;
        let var_def = &self.variables()[idx];
        frequencies::build(col, &var_def.value_labels)
    }

    fn crosstab(&self, row_var: &str, col_var: &str) -> SocStatResult<Crosstab> {
        let row_col = self.column_by_name(row_var)?;
        let col_col = self.column_by_name(col_var)?;
        crosstab::build(row_col, col_col)
    }

    fn independent_t_test(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<IndependentTTest> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let group = self.column_by_name(group_var)?;
        tests::independent_t_test(dep, group, self.weights().as_deref())
    }

    fn one_way_anova(&self, dep_var: &str, factor_var: &str) -> SocStatResult<OneWayAnova> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let factor = self.column_by_name(factor_var)?;
        tests::one_way_anova(dep, factor, self.weights().as_deref())
    }

    fn chi_square_test(&self, var1: &str, var2: &str) -> SocStatResult<ChiSquareTest> {
        let c1 = self.column_by_name(var1)?;
        let c2 = self.column_by_name(var2)?;
        tests::chi_square_test(c1, c2, self.weights().as_deref())
    }

    fn mann_whitney_u_test(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<MannWhitneyUTest> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let group = self.column_by_name(group_var)?;
        tests::mann_whitney_u_test(dep, group, self.weights().as_deref())
    }

    fn correlation(
        &self,
        vars: &[&str],
        method: CorrelationMethod,
    ) -> SocStatResult<Vec<CorrelationPair>> {
        if vars.len() < 2 {
            return Err(crate::error::SocStatError::InsufficientData(
                "correlation requires at least two variables".into(),
            ));
        }
        let weights = self.weights();
        let mut out = Vec::with_capacity(vars.len() * (vars.len() - 1) / 2);
        for i in 0..vars.len() {
            let x = regression::cleaned_numeric_column(self, vars[i])?;
            for j in (i + 1)..vars.len() {
                let y = regression::cleaned_numeric_column(self, vars[j])?;
                let (x, y, w) = regression::align_slices(&x, &y, weights.as_deref())?;
                out.push(regression::correlation_pair_aligned(
                    vars[i], vars[j], &x, &y, w.as_deref(), method,
                )?);
            }
        }
        Ok(out)
    }

    fn correlation_pair(
        &self,
        var1: &str,
        var2: &str,
        method: CorrelationMethod,
    ) -> SocStatResult<CorrelationPair> {
        let x = regression::cleaned_numeric_column(self, var1)?;
        let y = regression::cleaned_numeric_column(self, var2)?;
        let weights = self.weights();
        let (x, y, w) = regression::align_slices(&x, &y, weights.as_deref())?;
        regression::correlation_pair_aligned(var1, var2, &x, &y, w.as_deref(), method)
    }

    fn regression(&self, dep_var: &str, indep_vars: &[&str]) -> SocStatResult<LinearRegressionResult> {
        LinearRegressionResult::fit(self, dep_var, indep_vars)
    }

    fn logistic_regression(
        &self,
        dep_var: &str,
        indep_vars: &[&str],
    ) -> SocStatResult<LogisticRegressionResult> {
        LogisticRegressionResult::fit(self, dep_var, indep_vars)
    }

    fn pca(&self, vars: &[&str], mode: PcaMatrix) -> SocStatResult<PcaResult> {
        PcaResult::compute(self, vars, mode)
    }

    fn reliability(&self, vars: &[&str]) -> SocStatResult<ReliabilityResult> {
        ReliabilityResult::compute(self, vars)
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;
    use crate::data::{Value, Variable};
    use approx::assert_abs_diff_eq;

    #[test]
    fn descriptive_weights_aligned_after_missing() {
        // BUG-001: weights must stay aligned with the valid values so a
        // missing value does not silently drop the weights.
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x")).unwrap();
        ds.add_var(Variable::numeric("w").weight()).unwrap();
        ds.push_row(vec![Value::Number(10.0), Value::Number(1.0)]).unwrap();
        ds.push_row(vec![Value::Missing, Value::Number(2.0)]).unwrap();
        ds.push_row(vec![Value::Number(20.0), Value::Number(3.0)]).unwrap();

        let d = ds.descriptive("x").unwrap();
        // Weighted mean = (10*1 + 20*3) / (1+3) = 70/4 = 17.5; n = 4.
        assert_abs_diff_eq!(d.n, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean, 17.5, epsilon = 1e-12);
    }

    #[test]
    fn descriptive_user_missing_excluded() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x").missing_discrete(&[-1.0])).unwrap();
        ds.push_row(vec![Value::Number(1.0)]).unwrap();
        ds.push_row(vec![Value::Number(-1.0)]).unwrap();
        ds.push_row(vec![Value::Number(3.0)]).unwrap();
        let d = ds.descriptive("x").unwrap();
        assert_abs_diff_eq!(d.n, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.mean, 2.0, epsilon = 1e-12);
    }

    #[test]
    fn correlation_pair_shortcut() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("a")).unwrap();
        ds.add_var(Variable::numeric("b")).unwrap();
        for (a, b) in [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0), (4.0, 8.0), (5.0, 10.0)] {
            ds.push_row(vec![Value::Number(a), Value::Number(b)]).unwrap();
        }
        let p = ds.correlation_pair("a", "b", CorrelationMethod::Pearson).unwrap();
        assert!(p.pearson.is_some());
        assert_abs_diff_eq!(p.pearson.as_ref().unwrap().coefficient, 1.0, epsilon = 1e-12);
    }
}
