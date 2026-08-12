//! Reliability analysis — Cronbach's alpha.
//!
//! Measures internal consistency of a multi-item scale. Cronbach's alpha is
//!
//! ```text
//! α = k/(k − 1) · (1 − Σᵢ Var(xᵢ) / Var(Σᵢ xᵢ))
//! ```
//!
//! where `k` is the number of items. It can be negative when items are
//! negatively correlated — a sign the scale is poorly constructed.
//!
//! Items are summed per case (listwise-deleted), so a case is retained only if
//! every item is valid. Frequency weights are honored.
//!
//! # Example
//!
//! ```no_run
//! use socstat::prelude::*;
//! fn main() -> SocStatResult<()> {
//!     let ds = socstat::read().csv("data.csv")?;
//!     let rel = ReliabilityResult::compute(&ds, &["q1", "q2", "q3", "q4"])?;
//!     println!("α = {:.3} (standardized {:.3})", rel.alpha, rel.standardized_alpha);
//!     for item in &rel.item_statistics {
//!         println!("{}: corrected r = {:.3}, α if deleted = {:.3}",
//!                  item.item, item.corrected_item_total_correlation, item.alpha_if_deleted);
//!     }
//!     Ok(())
//! }
//! ```

use serde::{Deserialize, Serialize};

use crate::data::Dataset;
use crate::error::{SocStatError, SocStatResult};

use super::{compute_weighted_covariance, listwise_clean, PcaMatrix};

/// Diagnostics for a single scale item.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ItemStatistic {
    /// Item (variable) name.
    pub item: String,
    /// Weighted mean of the item.
    pub mean: f64,
    /// Weighted sample standard deviation of the item.
    pub std_dev: f64,
    /// Corrected item-total correlation: correlation of the item with the
    /// sum of the *other* items. `NaN` when undefined (constant item).
    pub corrected_item_total_correlation: f64,
    /// Cronbach's alpha with this item removed. `NaN` for 2-item scales.
    pub alpha_if_deleted: f64,
}

/// Result of a reliability (Cronbach's alpha) analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityResult {
    /// Item (variable) names, in analysis order.
    pub items: Vec<String>,
    /// Effective sample size (sum of weights when weighted).
    pub n: f64,
    /// Number of complete cases retained by listwise deletion.
    pub n_cases: usize,
    /// Cronbach's alpha.
    pub alpha: f64,
    /// Cronbach's alpha computed from the correlation matrix of the items
    /// (equivalent to alpha on standardized items). `NaN` if any item has
    /// zero variance.
    pub standardized_alpha: f64,
    /// Scale mean (sum of item means).
    pub scale_mean: f64,
    /// Scale variance (variance of the summed item scores).
    pub scale_variance: f64,
    /// Per-item diagnostics.
    pub item_statistics: Vec<ItemStatistic>,
}

impl ReliabilityResult {
    /// Compute Cronbach's alpha over the numeric items `vars`.
    ///
    /// Missing values are excluded by strict listwise deletion; the dataset's
    /// case-weight variable is honored when set. At least two items and two
    /// effective cases are required. Returns an error if the total (summed)
    /// score has zero variance, since alpha is then undefined.
    pub fn compute(dataset: &Dataset, vars: &[&str]) -> SocStatResult<Self> {
        if vars.len() < 2 {
            return Err(SocStatError::InsufficientData(
                "reliability analysis requires at least two items".into(),
            ));
        }
        let (matrix, weights, item_names) = listwise_clean(dataset, vars, None)?;
        let n_eff = weights.sum();
        if n_eff <= 1.0 {
            return Err(SocStatError::InsufficientData(
                "reliability analysis requires more than one effective case".into(),
            ));
        }
        let k = matrix.ncols();
        let n_cases = matrix.nrows();

        // Sample covariance matrix (denominator n − 1) with weighted means.
        let (means, stds, cov) = compute_weighted_covariance(&matrix, &weights, PcaMatrix::Covariance);

        let sum_var: f64 = (0..k).map(|j| cov[(j, j)]).sum();
        let mut var_total = 0.0;
        for i in 0..k {
            for j in 0..k {
                var_total += cov[(i, j)];
            }
        }
        if var_total <= 0.0 {
            return Err(SocStatError::Computation(
                "the total (summed) score has zero variance; Cronbach's alpha is undefined".into(),
            ));
        }

        let alpha = k as f64 / (k - 1) as f64 * (1.0 - sum_var / var_total);

        // Standardized alpha from the correlation matrix: S = 1ᵀ R 1.
        let standardized_alpha = if stds.iter().any(|&s| s == 0.0) {
            f64::NAN
        } else {
            let mut s = 0.0;
            for i in 0..k {
                for j in 0..k {
                    s += cov[(i, j)] / (stds[i] * stds[j]);
                }
            }
            k as f64 / (k - 1) as f64 * (1.0 - k as f64 / s)
        };

        let scale_mean: f64 = means.iter().sum();

        let mut item_statistics = Vec::with_capacity(k);
        for i in 0..k {
            let mut cov_i_total = 0.0;
            for j in 0..k {
                cov_i_total += cov[(i, j)];
            }
            let var_rest = var_total + cov[(i, i)] - 2.0 * cov_i_total;
            let cov_rest = cov_i_total - cov[(i, i)];
            let corrected = if var_rest > 0.0 && cov[(i, i)] > 0.0 {
                (cov_rest / (stds[i] * var_rest.sqrt())).clamp(-1.0, 1.0)
            } else {
                f64::NAN
            };
            let alpha_if_deleted = if k > 2 {
                let sum_var_excl = sum_var - cov[(i, i)];
                (k - 1) as f64 / (k - 2) as f64 * (1.0 - sum_var_excl / var_rest)
            } else {
                f64::NAN
            };
            item_statistics.push(ItemStatistic {
                item: item_names[i].clone(),
                mean: means[i],
                std_dev: stds[i],
                corrected_item_total_correlation: corrected,
                alpha_if_deleted,
            });
        }

        Ok(Self {
            items: item_names,
            n: n_eff,
            n_cases,
            alpha,
            standardized_alpha,
            scale_mean,
            scale_variance: var_total,
            item_statistics,
        })
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
    fn identical_items_have_alpha_one() {
        let d = num_dataset(&[
            &[1.0, 1.0, 1.0],
            &[2.0, 2.0, 2.0],
            &[3.0, 3.0, 3.0],
            &[4.0, 4.0, 4.0],
        ]);
        let r = ReliabilityResult::compute(&d, &["v0", "v1", "v2"]).unwrap();
        assert_abs_diff_eq!(r.alpha, 1.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r.standardized_alpha, 1.0, epsilon = 1e-9);
        // Each item is perfectly correlated with the rest of the scale.
        for item in &r.item_statistics {
            assert_abs_diff_eq!(item.corrected_item_total_correlation, 1.0, epsilon = 1e-9);
            assert_abs_diff_eq!(item.alpha_if_deleted, 1.0, epsilon = 1e-9);
        }
    }

    #[test]
    fn known_values() {
        let d = num_dataset(&[
            &[1.0, 2.0, 4.0],
            &[2.0, 3.0, 3.0],
            &[3.0, 5.0, 4.0],
            &[4.0, 6.0, 6.0],
            &[5.0, 7.0, 8.0],
        ]);
        let r = ReliabilityResult::compute(&d, &["v0", "v1", "v2"]).unwrap();
        assert_abs_diff_eq!(r.n, 5.0, epsilon = 1e-12);
        assert_eq!(r.n_cases, 5);
        assert_abs_diff_eq!(r.alpha, 0.956_376, epsilon = 1e-5);
        assert_abs_diff_eq!(r.scale_mean, 12.6, epsilon = 1e-9);
        assert_abs_diff_eq!(r.scale_variance, 29.8, epsilon = 1e-6);

        // Item means and stds.
        assert_abs_diff_eq!(r.item_statistics[0].mean, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.item_statistics[0].std_dev, 2.5_f64.sqrt(), epsilon = 1e-9);

        // Corrected item-total correlation and alpha-if-deleted for item 0.
        assert_abs_diff_eq!(r.item_statistics[0].corrected_item_total_correlation, 0.970_166, epsilon = 1e-4);
        assert_abs_diff_eq!(r.item_statistics[0].alpha_if_deleted, 0.915_033, epsilon = 1e-4);
    }

    #[test]
    fn two_item_scale() {
        let d = num_dataset(&[
            &[1.0, 2.0],
            &[2.0, 4.0],
            &[3.0, 5.0],
            &[4.0, 6.0],
        ]);
        let r = ReliabilityResult::compute(&d, &["v0", "v1"]).unwrap();
        assert!(r.alpha.is_finite());
        assert!(r.alpha > 0.0 && r.alpha <= 1.0);
        assert!(r.item_statistics[0].alpha_if_deleted.is_nan(), "alpha-if-deleted is undefined for 2 items");
        assert!(r.item_statistics[0].corrected_item_total_correlation.is_finite());
    }

    #[test]
    fn weighted_equals_frequency_replication() {
        let mut dw = Dataset::new();
        dw.add_var(Variable::numeric("v0")).unwrap();
        dw.add_var(Variable::numeric("v1")).unwrap();
        dw.add_var(Variable::numeric("v2")).unwrap();
        dw.add_var(Variable::numeric("w").weight()).unwrap();
        for (v0, v1, v2) in [(1.0, 2.0, 4.0), (2.0, 3.0, 3.0), (3.0, 5.0, 4.0)] {
            dw.push_row(vec![
                Value::Number(v0),
                Value::Number(v1),
                Value::Number(v2),
                Value::Number(2.0),
            ])
            .unwrap();
        }
        let weighted = ReliabilityResult::compute(&dw, &["v0", "v1", "v2"]).unwrap();

        let d2 = num_dataset(&[
            &[1.0, 2.0, 4.0],
            &[1.0, 2.0, 4.0],
            &[2.0, 3.0, 3.0],
            &[2.0, 3.0, 3.0],
            &[3.0, 5.0, 4.0],
            &[3.0, 5.0, 4.0],
        ]);
        let unweighted = ReliabilityResult::compute(&d2, &["v0", "v1", "v2"]).unwrap();
        assert_abs_diff_eq!(weighted.n, 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(weighted.alpha, unweighted.alpha, epsilon = 1e-9);
        assert_abs_diff_eq!(weighted.standardized_alpha, unweighted.standardized_alpha, epsilon = 1e-9);
        for (a, b) in weighted.item_statistics.iter().zip(&unweighted.item_statistics) {
            assert_abs_diff_eq!(a.mean, b.mean, epsilon = 1e-9);
            assert_abs_diff_eq!(a.std_dev, b.std_dev, epsilon = 1e-9);
            assert_abs_diff_eq!(a.corrected_item_total_correlation, b.corrected_item_total_correlation, epsilon = 1e-9);
        }
    }

    #[test]
    fn listwise_deletion() {
        let mut d = Dataset::new();
        d.add_var(Variable::numeric("v0")).unwrap();
        d.add_var(Variable::numeric("v1")).unwrap();
        d.add_var(Variable::numeric("v2")).unwrap();
        d.push_row(vec![Value::Number(1.0), Value::Number(2.0), Value::Number(3.0)]).unwrap();
        d.push_row(vec![Value::Number(2.0), Value::Missing, Value::Number(3.0)]).unwrap();
        d.push_row(vec![Value::Number(3.0), Value::Number(4.0), Value::Number(5.0)]).unwrap();
        d.push_row(vec![Value::Number(4.0), Value::Number(5.0), Value::Number(6.0)]).unwrap();
        let r = ReliabilityResult::compute(&d, &["v0", "v1", "v2"]).unwrap();
        assert_eq!(r.n_cases, 3);
        assert_abs_diff_eq!(r.n, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn serde_round_trip() {
        let d = num_dataset(&[
            &[1.0, 2.0, 4.0],
            &[2.0, 3.0, 3.0],
            &[3.0, 5.0, 4.0],
            &[4.0, 6.0, 6.0],
            &[5.0, 7.0, 8.0],
        ]);
        let r = ReliabilityResult::compute(&d, &["v0", "v1", "v2"]).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: ReliabilityResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.alpha, r.alpha, epsilon = 1e-15);
        assert_abs_diff_eq!(back.item_statistics[0].corrected_item_total_correlation,
            r.item_statistics[0].corrected_item_total_correlation, epsilon = 1e-15);
    }

    #[test]
    fn errors() {
        // Fewer than two items.
        let d1 = num_dataset(&[&[1.0], &[2.0], &[3.0]]);
        assert!(ReliabilityResult::compute(&d1, &["v0"]).is_err());

        // Text variable.
        let mut dtext = Dataset::new();
        dtext.add_var(Variable::numeric("v0")).unwrap();
        dtext.add_var(Variable::text("t")).unwrap();
        dtext.push_row(vec![Value::Number(1.0), Value::Text("a".into())]).unwrap();
        dtext.push_row(vec![Value::Number(2.0), Value::Text("b".into())]).unwrap();
        assert!(ReliabilityResult::compute(&dtext, &["v0", "t"]).is_err());

        // Not enough complete cases.
        let dmiss = num_dataset(&[&[1.0, 2.0]]);
        assert!(ReliabilityResult::compute(&dmiss, &["v0", "v1"]).is_err());

        // Zero-variance total score (all rows identical).
        let dconst = num_dataset(&[&[1.0, 2.0], &[1.0, 2.0], &[1.0, 2.0]]);
        assert!(ReliabilityResult::compute(&dconst, &["v0", "v1"]).is_err());
    }

    #[test]
    fn stats_ext_reliability() {
        let d = num_dataset(&[
            &[1.0, 2.0, 4.0],
            &[2.0, 3.0, 3.0],
            &[3.0, 5.0, 4.0],
            &[4.0, 6.0, 6.0],
            &[5.0, 7.0, 8.0],
        ]);
        let r = d.reliability(&["v0", "v1", "v2"]).unwrap();
        assert_abs_diff_eq!(r.alpha, 0.956_376, epsilon = 1e-5);
    }
}
