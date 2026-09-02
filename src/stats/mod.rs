//! Statistics module — descriptive stats, frequency tables, crosstabs,
//! hypothesis testing, correlation, linear and logistic regression, plus
//! multivariate analysis (PCA and reliability).
//!
//! The [`StatsExt`] trait (declared in [`ext`], implemented in [`ext_impl`])
//! extends [`Dataset`] with the main analysis methods.
//! Statistics automatically use case weights when set.
//!
//! ## Capabilities
//!
//! - **Descriptive** (`descriptive`), **frequencies**, **crosstab**.
//! - **Tests** ([`tests`]): independent & paired & one-sample `t`, one-way
//!   ANOVA, chi-square (independence + goodness-of-fit), McNemar, Mann–
//!   Whitney U, Wilcoxon signed-rank, Fisher's exact, Kruskal–Wallis.
//! - **Normality & distributions** ([`normality`], [`ks_two_sample`]):
//!   Shapiro–Wilk, one-sample K-S (incl. Lilliefors), two-sample K-S.
//! - **Repeated measures** ([`friedman`]): the Friedman rank-sum test with
//!   Kendall's W.
//! - **Correlation & collinearity** ([`regression`]): Pearson / Spearman /
//!   Kendall, VIF, partial correlation, and (multi)linear & logistic
//!   regression.
//! - **ANOVA follow-ups** ([`posthoc`]): Bonferroni, Tukey HSD, Scheffé,
//!   Games–Howell.
//! - **Multifactor ANOVA** ([`anova`]): Type I / Type II sums of squares.
//! - **Multivariate** ([`multivariate`]): PCA and Cronbach's α reliability.
//!
//! ## Weights
//!
//! All statistics treat weights as **frequency weights** (case weights):
//! each case counts as `weight` replicates. Complex sampling weights
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
//! let sw = ds.shapiro_wilk("income")?; // normality check
//! let aov = ds.factorial_anova("income", &["gender", "education"], SsType::TypeII)?;
//! # Ok(())
//! }
//! ```

mod ext;
mod ext_impl;
pub use ext::StatsExt;

pub mod anova;
pub mod crosstab;
pub mod descriptive;
pub mod friedman;
pub mod glm;
pub mod gof;
pub mod ks_two_sample;
pub mod mcnemar;
pub mod multivariate;
pub mod normality;
pub mod onesample;
pub mod posthoc;
pub mod regression;
pub(crate) mod shared;
pub mod tests;

pub use anova::{AnovaEffect, FactorialAnova, SsType};
pub use crosstab::Crosstab;
pub use descriptive::Descriptive;
pub use friedman::{FriedmanResult, FriedmanTreatment};
pub use frequencies::{FrequencyRow, FrequencyTable};
pub mod frequencies;
pub use glm::{BinomialFamily, ConfusionMatrix, GlmFamily, LogisticCoefficient, LogisticRegressionResult};
pub use gof::ChiSquareGof;
pub use ks_two_sample::TwoSampleKsResult;
pub use mcnemar::McNemarResult;
pub use multivariate::{ItemStatistic, PcaComponent, PcaMatrix, PcaResult, ReliabilityResult};
pub use normality::{KolmogorovSmirnovResult, KsTestType, ShapiroWilkResult};
pub use onesample::OneSampleTTest;
pub use posthoc::{PostHocComparison, PostHocMethod, PostHocResult};
pub use regression::{
    Coefficient, CorrelationMethod, CorrelationPair, CorrelationResult, LinearRegressionResult,
    PartialCorrelationResult, VifResult,
};
pub use tests::{
    ChiSquareTest, Effect, FisherExactTest, GroupSummary, IndependentTTest,
    KruskalWallisResult, LeveneResult, MannWhitneyUTest, OneWayAnova, PairedTTest, RankSummary,
    TTestModel, WilcoxonSignedRankResult,
};
