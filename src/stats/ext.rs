//! The [`StatsExt`] extension trait: every statistical analysis reachable
//! from a [`Dataset`](crate::data::Dataset), with doc-tested signatures.
//!
//! The trait lives in its own file so the method surface is readable at a
//! glance; the implementation over [`Dataset`](crate::data::Dataset)
//! is in [`super::ext_impl`].

use crate::error::SocStatResult;

use super::anova::{FactorialAnova, SsType};
use super::crosstab::Crosstab;
use super::descriptive::Descriptive;
use super::friedman::FriedmanResult;
use super::frequencies::FrequencyTable;
use super::glm::LogisticRegressionResult;
use super::gof::ChiSquareGof;
use super::ks_two_sample::TwoSampleKsResult;
use super::mcnemar::McNemarResult;
use super::multivariate::{PcaMatrix, PcaResult, ReliabilityResult};
use super::normality::{KolmogorovSmirnovResult, KsTestType, ShapiroWilkResult};
use super::onesample::OneSampleTTest;
use super::posthoc::{PostHocMethod, PostHocResult};
use super::regression::{
    CorrelationMethod, CorrelationPair, LinearRegressionResult,
    PartialCorrelationResult, VifResult,
};
use super::tests::{
    ChiSquareTest, FisherExactTest, IndependentTTest, KruskalWallisResult,
    MannWhitneyUTest, OneWayAnova, PairedTTest, WilcoxonSignedRankResult,
};

/// Extension trait providing statistical analysis methods on
/// [`Dataset`](crate::data::Dataset).
///
/// Statistics automatically use case weights when a weight variable
/// is set (see [`Dataset::set_weight`](crate::data::Dataset::set_weight)).
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

    /// One-sample t-test of a numeric variable's mean against `mu0`
    /// (R: `t.test(x, mu = mu0)`).
    fn one_sample_t_test(&self, var: &str, mu0: f64) -> SocStatResult<OneSampleTTest>;

    /// One-way ANOVA of a numeric dependent variable across the groups of
    /// `factor_var`.
    fn one_way_anova(&self, dep_var: &str, factor_var: &str) -> SocStatResult<OneWayAnova>;

    /// Pearson chi-square test of independence between two categorical
    /// variables.
    fn chi_square_test(&self, var1: &str, var2: &str) -> SocStatResult<ChiSquareTest>;

    /// Chi-square goodness-of-fit test of a categorical variable's counts
    /// against `probs` (uniform when `None`)
    /// (R: `chisq.test(table(x), p = probs)`).
    fn chi_square_gof_test(
        &self,
        var: &str,
        probs: Option<&[f64]>,
    ) -> SocStatResult<ChiSquareGof>;

    /// McNemar test for paired binary outcomes in `before_var` vs
    /// `after_var` (R: `mcnemar.test`). Uses the exact binomial p-value
    /// when the discordant pairs total < 25, otherwise the chi-square
    /// approximation with continuity correction.
    fn mcnemar_test(
        &self,
        before_var: &str,
        after_var: &str,
    ) -> SocStatResult<McNemarResult>;

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

    /// Friedman rank-sum test for repeated measurements: each variable in
    /// `treatment_vars` is one treatment, each row one subject (block)
    /// (R: `friedman.test` on a wide matrix). Rows with any missing value
    /// are dropped. Reports per-treatment rank summaries, the χ²(k−1)
    /// p-value with tie correction, and Kendall's W.
    fn friedman_test(&self, treatment_vars: &[&str]) -> SocStatResult<FriedmanResult>;

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

    /// Two-sample Kolmogorov–Smirnov test comparing the distributions of
    /// two independent numeric variables (R: `ks.test(x, y)`, samples are
    /// not row-aligned; missing values are dropped per-variable).
    /// p-value from the asymptotic Kolmogorov distribution with Stephens'
    /// finite-sample adjustment.
    fn ks_two_sample_test(
        &self,
        var1: &str,
        var2: &str,
    ) -> SocStatResult<TwoSampleKsResult>;

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

    /// Multifactor (factorial) ANOVA of `dep_var` on two or more factors.
    ///
    /// Factors are dummy-coded with all two-way interactions; `ss_type` selects
    /// Type I (sequential) or Type II (marginal) sums of squares. Reports each
    /// effect's SS/df/MS/F, η² and partial-η², plus the overall model fit.
    /// See [`factorial_anova`](crate::stats::anova::factorial_anova).
    fn factorial_anova(
        &self,
        dep_var: &str,
        factors: &[&str],
        ss_type: SsType,
    ) -> SocStatResult<FactorialAnova>;
}
