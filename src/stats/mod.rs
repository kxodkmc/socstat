//! Statistics module — descriptive stats, frequency tables, crosstabs,
//! and hypothesis testing.
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
pub mod tests;

pub use crosstab::Crosstab;
pub use descriptive::Descriptive;
pub use frequencies::{FrequencyRow, FrequencyTable};
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
}
