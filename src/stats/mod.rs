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
//! let t = ds.ttest_independent("income", "gender")?;
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
    fn ttest_independent(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<IndependentTTest>;

    /// One-way ANOVA of a numeric dependent variable across the groups of
    /// `factor_var`.
    fn anova_one_way(&self, dep_var: &str, factor_var: &str) -> SocStatResult<OneWayAnova>;

    /// Pearson chi-square test of independence between two categorical
    /// variables.
    fn chi_square_test(&self, var1: &str, var2: &str) -> SocStatResult<ChiSquareTest>;

    /// Mann–Whitney U test of a numeric dependent variable between the two
    /// groups defined by `group_var`.
    fn mann_whitney_u(
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
        let col = self.column_by_name(var)?;
        let slice = col.as_numeric().ok_or_else(|| crate::error::SocStatError::TypeMismatch {
            var: var.into(),
            expected: "Numeric",
            actual: "Text",
        })?;

        // Extract valid values
        let data: Vec<f64> = slice.iter()
            .filter_map(|o| *o)
            .collect();

        if data.is_empty() {
            return Err(crate::error::SocStatError::Computation(
                format!("no valid values in variable '{var}'")
            ));
        }

        // Get weights if available
        let weights = self.weights();

        Ok(descriptive::compute(&data, weights.as_deref()))
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

    fn ttest_independent(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<IndependentTTest> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let group = self.column_by_name(group_var)?;
        tests::independent_ttest(dep, group, self.weights().as_deref())
    }

    fn anova_one_way(&self, dep_var: &str, factor_var: &str) -> SocStatResult<OneWayAnova> {
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

    fn mann_whitney_u(
        &self,
        dep_var: &str,
        group_var: &str,
    ) -> SocStatResult<MannWhitneyUTest> {
        self.numeric_slice(dep_var)?;
        let dep = self.column_by_name(dep_var)?;
        let group = self.column_by_name(group_var)?;
        tests::mann_whitney_u(dep, group, self.weights().as_deref())
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
