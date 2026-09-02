//! One-sample t-test: compares the mean of a numeric sample against a
//! reference value `mu0`.
//!
//! `t = (mean - mu0) / (s / sqrt(n))` with `n - 1` degrees of freedom
//! (the same statistic R's `t.test(x, mu = ...)` reports). Weights are
//! **frequency weights** (each case counts as `weight` replicates); a case
//! with a non-finite value or a non-positive weight is excluded.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use serde::{Deserialize, Serialize};

use crate::dist::{Distribution, StudentsTDist};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{WeightedSummary, two_sided_tail};

/// Result of a one-sample t-test against a reference mean.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneSampleTTest {
    /// Reference mean under the null hypothesis.
    pub mu0: f64,
    pub n: f64,
    pub mean: f64,
    pub std_dev: f64,
    pub std_error: f64,
    pub t_statistic: f64,
    pub df: f64,
    pub p_value: f64,
    /// 95% confidence interval for the population mean,
    /// `mean ± t(0.975, df) · std_error`.
    pub ci_95: (f64, f64),
}

/// One-sample t-test of `data` against `mu0`.
///
/// Two-sided p-value and a 95% CI for the mean. Requires at least 2
/// usable observations with non-zero variance.
pub fn one_sample_t_test(
    data: &[f64],
    weights: Option<&[f64]>,
    mu0: f64,
) -> SocStatResult<OneSampleTTest> {
    if let Some(w) = weights
        && w.len() != data.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: data.len(),
            got: w.len(),
        });
    }

    let pairs: Vec<(f64, f64)> = (0..data.len())
        .map(|i| (data[i], weights.map(|ww| ww[i]).unwrap_or(1.0)))
        .collect();

    let ws = WeightedSummary::compute(&pairs)?;
    let std_dev = ws.variance().sqrt();
    if ws.n < 2.0 {
        return Err(SocStatError::InsufficientData(
            "one-sample t-test needs at least 2 observations".into(),
        ));
    }
    if std_dev <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "one-sample t-test needs non-zero variance".into(),
        ));
    }
    let n = ws.n;
    let mean = ws.mean;
    let std_error = std_dev / n.sqrt();
    let df = n - 1.0;
    let t = (mean - mu0) / std_error;

    let dist = StudentsTDist::new(df)?;
    let p = two_sided_tail(&dist, t);
    let t_crit = dist.inverse_cdf(0.975);
    let ci = (mean - t_crit * std_error, mean + t_crit * std_error);

    Ok(OneSampleTTest {
        mu0,
        n,
        mean,
        std_dev,
        std_error,
        t_statistic: t,
        df,
        p_value: p,
        ci_95: ci,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// InsectSprays spray A (R dataset): reference values from R/scipy.
    #[test]
    fn one_sample_t_matches_reference() {
        let a = [
            10.0, 7.0, 20.0, 14.0, 14.0, 12.0, 10.0, 23.0, 17.0, 20.0, 14.0, 13.0,
        ];
        let r = one_sample_t_test(&a, None, 10.0).unwrap();
        assert_abs_diff_eq!(r.n, 12.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.mean, 14.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r.std_dev, 4.719_399_037_242_694, epsilon = 1e-12);
        assert_abs_diff_eq!(r.std_error, 1.362_373_152_282_665_2, epsilon = 1e-12);
        assert_abs_diff_eq!(r.t_statistic, 3.303_059_805_942_462, epsilon = 1e-12);
        assert_abs_diff_eq!(r.df, 11.0, epsilon = 1e-12);
        // R: t.test(sprayA, mu = 10) → p = 0.007039667
        assert_abs_diff_eq!(r.p_value, 0.007_039_667_367_947_395, epsilon = 1e-12);
        assert_abs_diff_eq!(r.ci_95.0, 11.501_436_909_330_426, epsilon = 1e-9);
        assert_abs_diff_eq!(r.ci_95.1, 17.498_563_090_669_574, epsilon = 1e-9);
    }

    #[test]
    fn one_sample_t_small_n() {
        let x = [2.8, 3.1, 3.5, 2.9, 3.8];
        let r = one_sample_t_test(&x, None, 3.0).unwrap();
        // scipy ttest_1samp: t = 1.169286807596, p = 0.307212781760
        assert_abs_diff_eq!(r.t_statistic, 1.169_286_807_596, epsilon = 1e-10);
        assert_abs_diff_eq!(r.p_value, 0.307_212_781_760, epsilon = 1e-10);
        assert_abs_diff_eq!(r.ci_95.0, 2.697_614_970_788, epsilon = 1e-9);
        assert_abs_diff_eq!(r.ci_95.1, 3.742_385_029_212, epsilon = 1e-9);
    }

    #[test]
    fn one_sample_t_weighted_equals_frequency_expansion() {
        let x = [10.0, 20.0, 30.0];
        let w = [1.0, 3.0, 2.0];
        let expanded = [
            10.0, 20.0, 20.0, 20.0, 30.0, 30.0,
        ];
        let a = one_sample_t_test(&x, Some(&w), 0.0).unwrap();
        let b = one_sample_t_test(&expanded, None, 0.0).unwrap();
        assert_abs_diff_eq!(a.n, b.n, epsilon = 1e-12);
        assert_abs_diff_eq!(a.mean, b.mean, epsilon = 1e-12);
        assert_abs_diff_eq!(a.t_statistic, b.t_statistic, epsilon = 1e-12);
        assert_abs_diff_eq!(a.p_value, b.p_value, epsilon = 1e-12);
        assert_abs_diff_eq!(a.ci_95.0, b.ci_95.0, epsilon = 1e-12);
    }

    #[test]
    fn one_sample_t_edge_cases() {
        // Missing/NaN values and non-positive weights are excluded.
        let x = [1.0, f64::NAN, 2.0, 3.0, 4.0];
        let w = [1.0, 1.0, 1.0, 0.0, 2.0];
        let r = one_sample_t_test(&x, Some(&w), 2.0).unwrap();
        // Rows kept: (1, w=1), (2, w=1), (4, w=2) → n = 4.
        assert_abs_diff_eq!(r.n, 4.0, epsilon = 1e-12);
        // Too few observations.
        assert!(one_sample_t_test(&[1.0, f64::NAN], None, 0.0).is_err());
        // Zero variance.
        assert!(one_sample_t_test(&[2.0, 2.0, 2.0], None, 0.0).is_err());
        // Weight-length mismatch.
        assert!(one_sample_t_test(&[1.0, 2.0], Some(&[1.0]), 0.0).is_err());
    }

    #[test]
    fn one_sample_t_serde_round_trip() {
        let r = one_sample_t_test(&[1.0, 2.0, 3.0, 4.5], None, 2.0).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: OneSampleTTest = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.t_statistic, r.t_statistic, epsilon = 1e-15);
        assert_abs_diff_eq!(back.p_value, r.p_value, epsilon = 1e-15);
    }
}
