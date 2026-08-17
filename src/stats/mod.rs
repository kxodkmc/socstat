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
pub mod normality;
pub mod posthoc;
pub mod regression;
pub(crate) mod shared;
pub mod tests;

pub use crosstab::Crosstab;
pub use descriptive::Descriptive;
pub use frequencies::{FrequencyRow, FrequencyTable};
pub use glm::{BinomialFamily, ConfusionMatrix, GlmFamily, LogisticCoefficient, LogisticRegressionResult};
pub use multivariate::{ItemStatistic, PcaComponent, PcaMatrix, PcaResult, ReliabilityResult};
pub use normality::{KolmogorovSmirnovResult, KsTestType, ShapiroWilkResult};
pub use posthoc::{PostHocComparison, PostHocMethod, PostHocResult};
pub use regression::{
    Coefficient, CorrelationMethod, CorrelationPair, CorrelationResult, LinearRegressionResult,
    PartialCorrelationResult, VifResult,
};
pub use tests::{
    Alternative, ChiSquareTest, Effect, FisherExactTest, GroupSummary, IndependentTTest,
    KruskalWallisResult, LeveneResult, MannWhitneyUTest, OneWayAnova, PairedTTest, RankSummary,
    TTestModel, WilcoxonSignedRankResult,
};

use crate::data::{ColumnData, Dataset};
use crate::error::SocStatResult;

/// Split a cleaned numeric column into finite values with aligned weights,
/// dropping rows whose value is missing or whose weight is non-positive.
fn aligned_numeric(
    cleaned: &[Option<f64>],
    weights: Option<&[f64]>,
) -> (Vec<f64>, Option<Vec<f64>>) {
    match weights {
        Some(ws) => {
            let (data, w): (Vec<f64>, Vec<f64>) = cleaned
                .iter()
                .zip(ws.iter())
                .filter_map(|(v, wt)| {
                    let v = (*v)?;
                    if !wt.is_finite() || *wt <= 0.0 {
                        None
                    } else {
                        Some((v, *wt))
                    }
                })
                .unzip();
            (data, Some(w))
        }
        None => {
            let data: Vec<f64> = cleaned.iter().flatten().copied().collect();
            (data, None)
        }
    }
}

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

    /// Paired-samples t-test of the mean difference between two numeric
    /// variables (each row is one paired observation).
    ///
    /// Uses the closed-form dependent t: t = d̄ / (s_d / √n) with df = n−1. Rows
    /// where either value is missing (or the weight is non-positive) are dropped
    /// pairwise; case weights are honored. Reports the Pearson correlation of
    /// the paired observations as well.
    fn paired_t_test(&self, var1: &str, var2: &str) -> SocStatResult<PairedTTest>;

    /// Fisher's exact test on a 2×2 table.
    ///
    /// The two variables are cross-tabulated (weights rounded to counts); each
    /// must have exactly two categories. Reports the two-sided, less-than, and
    /// greater-than p-values plus the odds ratio.
    fn fisher_exact_test(
        &self,
        var1: &str,
        var2: &str,
        alternative: Alternative,
    ) -> SocStatResult<FisherExactTest>;

    /// Wilcoxon signed-rank test on paired observations (differences of the
    /// two numeric variables).
    ///
    /// Uses the asymptotic normal approximation (R's default `correct = TRUE`).
    /// Requires at least 10 non-zero differences. Rows with missing values or
    /// non-positive weights are dropped pairwise; case weights are honored.
    fn wilcoxon_signed_rank_test(
        &self,
        var1: &str,
        var2: &str,
    ) -> SocStatResult<WilcoxonSignedRankResult>;

    /// Kruskal–Wallis rank-sum test of a numeric dependent variable across the
    /// groups of a factor variable.
    ///
    /// The H statistic uses average ranks (mid-ranks across ties) with the
    /// standard tie correction. Case weights are honored. For small groups the
    /// chi-squared approximation may be unreliable, and a `warning` is set.
    fn kruskal_wallis_test(&self, dep_var: &str, factor_var: &str) -> SocStatResult<KruskalWallisResult>;

    /// Shapiro–Wilk test of normality for a numeric variable.
    ///
    /// Computes the W statistic and an approximate p-value (Royston AS R94)
    /// for effective sample sizes 3..=5000; case weights are treated as
    /// frequency weights.
    fn shapiro_wilk(&self, var: &str) -> SocStatResult<ShapiroWilkResult>;

    /// One-sample Kolmogorov–Smirnov normality test for a numeric variable.
    ///
    /// `test_type` selects either a fully-specified `N(mean, std_dev)` or the
    /// Lilliefors variant (parameters estimated from the sample). Case weights
    /// are honored.
    fn ks_normality_test(&self, var: &str, test_type: KsTestType) -> SocStatResult<KolmogorovSmirnovResult>;

    /// Variance inflation factors for the given predictor variables.
    ///
    /// Each predictor is regressed on the others; reports `VIF = 1/(1−R²)`,
    /// tolerance, and the per-variable R². Requires at least two predictors;
    /// perfect collinearity returns `SocStatError::SingularMatrix`.
    fn vif(&self, indep_vars: &[&str]) -> SocStatResult<Vec<VifResult>>;

    /// Partial correlation of `var1` and `var2` whilst controlling for
    /// `control_vars` (residual method).
    ///
    /// `df = n − k − 2`; case weights are honored. Control variables are
    /// optional, but at least one is expected (use
    /// [`correlation_pair`](Self::correlation_pair) otherwise).
    fn partial_correlation(
        &self,
        var1: &str,
        var2: &str,
        control_vars: &[&str],
        method: CorrelationMethod,
    ) -> SocStatResult<PartialCorrelationResult>;

    /// ANOVA post-hoc comparisons (Bonferroni / Tukey HSD / Scheffé) of a
    /// numeric dependent variable across the groups of `factor_var`.
    ///
    /// Uses the pooled within-group variance from [`one_way_anova`](Self::one_way_anova)
    /// as the error term. Reports adjusted p-values and confidence intervals
    /// for every pair of groups.
    fn post_hoc(&self, dep_var: &str, factor_var: &str, method: PostHocMethod)
        -> SocStatResult<PostHocResult>;
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
            let x = shared::cleaned_numeric_column(self, vars[i])?;
            for j in (i + 1)..vars.len() {
                let y = shared::cleaned_numeric_column(self, vars[j])?;
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
        let x = shared::cleaned_numeric_column(self, var1)?;
        let y = shared::cleaned_numeric_column(self, var2)?;
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

    fn paired_t_test(&self, var1: &str, var2: &str) -> SocStatResult<PairedTTest> {
        self.numeric_slice(var1)?;
        self.numeric_slice(var2)?;
        let v1 = self.column_by_name(var1)?;
        let v2 = self.column_by_name(var2)?;
        tests::paired_t_test(v1, v2, self.weights().as_deref())
    }

    fn fisher_exact_test(
        &self,
        var1: &str,
        var2: &str,
        alternative: Alternative,
    ) -> SocStatResult<FisherExactTest> {
        let c1 = self.column_by_name(var1)?;
        let c2 = self.column_by_name(var2)?;
        let table = tests::fisher_table_from_columns(c1, c2, self.weights().as_deref())?;
        tests::fisher_exact(table, alternative)
    }

    fn wilcoxon_signed_rank_test(
        &self,
        var1: &str,
        var2: &str,
    ) -> SocStatResult<WilcoxonSignedRankResult> {
        self.numeric_slice(var1)?;
        self.numeric_slice(var2)?;
        let v1 = self.column_by_name(var1)?;
        let v2 = self.column_by_name(var2)?;
        tests::wilcoxon_signed_rank(v1, v2, self.weights().as_deref())
    }

    fn kruskal_wallis_test(
        &self,
        dep_var: &str,
        factor_var: &str,
    ) -> SocStatResult<KruskalWallisResult> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let factor = self.column_by_name(factor_var)?;
        tests::kruskal_wallis(dep, factor, self.weights().as_deref())
    }

    fn shapiro_wilk(&self, var: &str) -> SocStatResult<ShapiroWilkResult> {
        let cleaned = shared::cleaned_numeric_column(self, var)?;
        let weights = self.weights();
        let (data, w) = aligned_numeric(&cleaned, weights.as_deref());
        normality::shapiro_wilk(&data, w.as_deref())
    }

    fn ks_normality_test(
        &self,
        var: &str,
        test_type: KsTestType,
    ) -> SocStatResult<KolmogorovSmirnovResult> {
        let cleaned = shared::cleaned_numeric_column(self, var)?;
        let weights = self.weights();
        let (data, w) = aligned_numeric(&cleaned, weights.as_deref());
        normality::ks_test(&data, w.as_deref(), test_type)
    }

    fn vif(&self, indep_vars: &[&str]) -> SocStatResult<Vec<VifResult>> {
        if indep_vars.len() < 2 {
            return Err(crate::error::SocStatError::InsufficientData(
                "VIF needs at least two predictor variables".into(),
            ));
        }
        let cols: Vec<ColumnData> = indep_vars
            .iter()
            .map(|v| Ok(ColumnData::Numeric(shared::cleaned_numeric_column(self, v)?)))
            .collect::<crate::error::SocStatResult<_>>()?;
        let refs: Vec<(&str, &ColumnData)> = indep_vars.iter().zip(&cols).map(|(n, c)| (*n, c)).collect();
        regression::variance_inflation_factors(&refs, self.weights().as_deref())
    }

    fn partial_correlation(
        &self,
        var1: &str,
        var2: &str,
        control_vars: &[&str],
        method: CorrelationMethod,
    ) -> SocStatResult<PartialCorrelationResult> {
        let x = ColumnData::Numeric(shared::cleaned_numeric_column(self, var1)?);
        let y = ColumnData::Numeric(shared::cleaned_numeric_column(self, var2)?);
        let controls: Vec<ColumnData> = control_vars
            .iter()
            .map(|v| Ok(ColumnData::Numeric(shared::cleaned_numeric_column(self, v)?)))
            .collect::<crate::error::SocStatResult<_>>()?;
        let refs: Vec<(&str, &ColumnData)> =
            control_vars.iter().zip(&controls).map(|(n, c)| (*n, c)).collect();
        regression::partial_correlation(
            var1,
            var2,
            &x,
            &y,
            &refs,
            self.weights().as_deref(),
            method,
        )
    }

    fn post_hoc(
        &self,
        dep_var: &str,
        factor_var: &str,
        method: PostHocMethod,
    ) -> SocStatResult<PostHocResult> {
        let aov = self.one_way_anova(dep_var, factor_var)?;
        let dep = self.column_by_name(dep_var)?;
        let factor = self.column_by_name(factor_var)?;
        let groups = shared::split_groups(dep, factor, self.weights().as_deref())?;
        posthoc::post_hoc(&groups, aov.within_groups.ms, aov.within_groups.df, method)
    }
}

#[cfg(test)]
mod stats_tests {
    use super::*;
    use crate::data::{Value, Variable};
    use crate::dist::Distribution;
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

    #[test]
    fn new_test_methods_via_trait() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("x")).unwrap();
        ds.add_var(Variable::numeric("y")).unwrap();
        ds.add_var(Variable::numeric("g")).unwrap();
        ds.add_var(Variable::numeric("f")).unwrap();
        // Paired / Wilcoxon / Kruskal data.
        for (x, y, g, f) in [
            (8.0, 6.0, 1.0, 1.0),
            (7.0, 4.0, 1.0, 1.0),
            (6.0, 5.0, 2.0, 1.0),
            (9.0, 5.0, 2.0, 2.0),
            (10.0, 5.0, 3.0, 2.0),
            (11.0, 8.0, 3.0, 2.0),
        ] {
            ds.push_row(vec![Value::Number(x), Value::Number(y), Value::Number(g), Value::Number(f)])
                .unwrap();
        }
        // Paired t-test.
        let pt = ds.paired_t_test("x", "y").unwrap();
        assert!(pt.p_value.is_finite());
        // Kruskal–Wallis.
        let kw = ds.kruskal_wallis_test("x", "f").unwrap();
        assert_eq!(kw.group_stats.len(), 2);
        assert!(kw.h_statistic.is_finite());
    }

    #[test]
    fn wilcoxon_and_fisher_via_trait() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("a")).unwrap();
        ds.add_var(Variable::numeric("b")).unwrap();
        // 20 positive differences → Wilcoxon valid.
        for i in 1..=20 {
            ds.push_row(vec![Value::Number(i as f64), Value::Number(0.0)]).unwrap();
        }
        ds.add_var(Variable::numeric("r")).unwrap();
        ds.add_var(Variable::numeric("c")).unwrap();
        // 2×2 categories for Fisher.
        for i in 0..20 {
            ds.push_row(vec![
                Value::Number(i as f64),
                Value::Number(0.0),
                Value::Number(if i < 10 { 1.0 } else { 2.0 }),
                Value::Number(if i % 2 == 0 { 1.0 } else { 2.0 }),
            ])
            .unwrap();
        }
        let w = ds.wilcoxon_signed_rank_test("a", "b").unwrap();
        assert!(w.p_value.is_finite());
        assert!(w.w_positive > w.w_negative);

        let ft = ds.fisher_exact_test("r", "c", Alternative::TwoSided).unwrap();
        assert!(ft.p_value_two_sided.is_finite());
        assert!(ft.odds_ratio.is_finite());
    }

    #[test]
    fn normality_tests_via_trait() {
        // Near-normal column: both normality tests should be non-significant.
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("z")).unwrap();
        for i in 1..=40 {
            // A low-discrepancy "normal-ish" bump via inverse CDF of a lattice.
            let p = (i as f64 - 0.5) / 40.0;
            let v = crate::dist::NormalDist::standard().inverse_cdf(p);
            ds.push_row(vec![Value::Number(v)]).unwrap();
        }
        let sw = ds.shapiro_wilk("z").unwrap();
        assert!(sw.w_statistic > 0.9 && sw.p_value > 0.05);
        let k = ds.ks_normality_test("z", KsTestType::Lilliefors).unwrap();
        assert!(k.p_value > 0.05);
    }

    #[test]
    fn multicollinearity_tests_via_trait() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("y")).unwrap();
        ds.add_var(Variable::numeric("x")).unwrap();
        ds.add_var(Variable::numeric("c")).unwrap();
        ds.add_var(Variable::numeric("d")).unwrap();
        // u, v are small independent resids so x, y are not perfectly
        // collinear with c (keeps VIF finite and partial correlation defined).
        let u = [0.3, -0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3];
        let v = [-0.3, 0.5, -0.4, 0.6, -0.5, 0.4, -0.6, 0.5];
        for i in 1..=8 {
            let c = i as f64;
            ds.push_row(vec![
                Value::Number(c + 1.3 + v[i - 1]),
                Value::Number(c + 1.3 + u[i - 1]),
                Value::Number(c),
                Value::Number((i % 3) as f64),
            ])
            .unwrap();
        }
        // VIF: need ≥ 2 predictors; returns one entry per predictor, all ≥ 1.
        let vif = ds.vif(&["x", "c", "d"]).unwrap();
        assert_eq!(vif.len(), 3);
        for v in &vif {
            assert!(v.vif >= 1.0);
            assert!(v.vif.is_finite());
        }
        // Partial correlation controlling for the shared confounder is well-defined.
        let pc = ds.partial_correlation("y", "x", &["c"], CorrelationMethod::Pearson).unwrap();
        assert_eq!(pc.controlling_for, vec!["c".to_string()]);
        assert_abs_diff_eq!(pc.n, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pc.df, 5.0, epsilon = 1e-12);
        assert!(pc.coefficient.is_finite() && pc.coefficient.abs() <= 1.0);
    }

    #[test]
    fn post_hoc_via_trait() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("y")).unwrap();
        ds.add_var(Variable::numeric("g")).unwrap();
        // Three well-separated groups.
        for (y, g) in [
            (1.0, 1.0), (2.0, 1.0), (3.0, 1.0), (4.0, 1.0),
            (11.0, 2.0), (12.0, 2.0), (13.0, 2.0), (14.0, 2.0),
            (21.0, 3.0), (22.0, 3.0), (23.0, 3.0), (24.0, 3.0),
        ] {
            ds.push_row(vec![Value::Number(y), Value::Number(g)]).unwrap();
        }
        for method in [PostHocMethod::Bonferroni, PostHocMethod::Tukey, PostHocMethod::Scheffe] {
            let r = ds.post_hoc("y", "g", method).unwrap();
            assert_eq!(r.comparisons.len(), 3);
            assert_eq!(r.n_groups, 3);
            for c in &r.comparisons {
                assert!(c.p_value.is_finite());
                assert!(c.p_value < 0.05);
                assert!(c.ci_95.0 <= c.mean_difference && c.mean_difference <= c.ci_95.1);
            }
        }
    }
}
