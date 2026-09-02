//! Two-sample Kolmogorov–Smirnov test: compares the empirical distribution
//! functions of two independent samples.
//!
//! The statistic `D = sup |F₁(x) − F₂(x)|` is evaluated on the pooled,
//! tie-merged grid (the same convention as R's `ks.test` and SciPy's
//! `ks_2samp`). The p-value uses the asymptotic Kolmogorov series with
//! Stephens' (1970) finite-sample adjustment
//! `λ = (√ne + 0.12 + 0.11/√ne) · D`, `ne = n₁n₂/(n₁+n₂)`.
//!
//! Weights are **frequency weights** (each case counts as `weight`
//! replicates); non-finite values and non-positive weights are excluded
//! per-sample, so the two samples need not be row-aligned.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use serde::{Deserialize, Serialize};

use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{kolmogorov_two_sided_p, positive_weight, stephens_lambda};

/// Result of a two-sample Kolmogorov–Smirnov test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwoSampleKsResult {
    /// Effective size of sample 1 (sum of weights).
    pub n1: f64,
    /// Effective size of sample 2 (sum of weights).
    pub n2: f64,
    /// Harmonic effective size `n₁n₂/(n₁+n₂)`.
    pub n_eff: f64,
    pub d_statistic: f64,
    pub p_value: f64,
    /// Whether duplicate values were present in the pooled sample.
    pub has_ties: bool,
    /// Set when the effective sample size is small enough that the
    /// asymptotic p-value may be unreliable.
    pub warning: Option<String>,
}

/// Two-sample Kolmogorov–Smirnov test between `x` and `y`.
///
/// Both samples are filtered independently (non-finite values and
/// non-positive weights dropped), so paired rows are not required.
pub fn ks_two_sample_test(
    x: &[f64],
    wx: Option<&[f64]>,
    y: &[f64],
    wy: Option<&[f64]>,
) -> SocStatResult<TwoSampleKsResult> {
    let weight_at = |w: Option<&[f64]>, i: usize| w.map(|ww| ww[i]).unwrap_or(1.0);

    if let Some(w) = wx
        && w.len() != x.len()
    {
        return Err(SocStatError::ColumnLengthMismatch { expected: x.len(), got: w.len() });
    }
    if let Some(w) = wy
        && w.len() != y.len()
    {
        return Err(SocStatError::ColumnLengthMismatch { expected: y.len(), got: w.len() });
    }

    let mut xs: Vec<(f64, f64)> = (0..x.len())
        .filter(|&i| x[i].is_finite() && positive_weight(weight_at(wx, i)))
        .map(|i| (x[i], weight_at(wx, i)))
        .collect();
    let mut ys: Vec<(f64, f64)> = (0..y.len())
        .filter(|&i| y[i].is_finite() && positive_weight(weight_at(wy, i)))
        .map(|i| (y[i], weight_at(wy, i)))
        .collect();
    xs.sort_by(|a, b| a.0.total_cmp(&b.0));
    ys.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n1: f64 = xs.iter().map(|&(_, w)| w).sum();
    let n2: f64 = ys.iter().map(|&(_, w)| w).sum();
    if n1 < 1.0 || n2 < 1.0 {
        return Err(SocStatError::InsufficientData(
            "two-sample K-S test needs at least one observation per sample".into(),
        ));
    }

    // Pooled distinct values in ascending order; D changes only there.
    let mut pool: Vec<f64> = xs.iter().map(|&(v, _)| v).chain(ys.iter().map(|&(v, _)| v)).collect();
    pool.sort_by(|a, b| a.total_cmp(b));
    pool.dedup();
    let has_ties = pool.len() < xs.len() + ys.len();

    let mut cum1 = 0.0_f64;
    let mut cum2 = 0.0_f64;
    let mut xi = 0;
    let mut yi = 0;
    let mut d = 0.0_f64;
    for &v in &pool {
        while xi < xs.len() && xs[xi].0 == v {
            cum1 += xs[xi].1;
            xi += 1;
        }
        while yi < ys.len() && ys[yi].0 == v {
            cum2 += ys[yi].1;
            yi += 1;
        }
        d = d.max((cum1 / n1 - cum2 / n2).abs());
    }

    let n_eff = n1 * n2 / (n1 + n2);
    let lambda = stephens_lambda(n_eff, d);
    let p_value = kolmogorov_two_sided_p(lambda);

    let warning = if n_eff < 10.0 {
        Some(format!(
            "effective sample size ne = {:.1} is small; the asymptotic p-value is approximate",
            n_eff
        ))
    } else {
        None
    };

    Ok(TwoSampleKsResult { n1, n2, n_eff, d_statistic: d, p_value, has_ties, warning })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Pooled-grid ECDF comparison with SciPy `ks_2samp` (v1.14.1):
    /// a = [1,1,1,2,2,2,2.5,2.5,2.5,4,4,4,7,7,7,3,5,9,6,8] (n=20),
    /// b = [1.5,1.5,1.5,3,3,3,3.5,3.5,3.5,6,6,6,2,4,8,5,7.5,9.5] (n=18).
    /// D = 0.22777777..., p(Stephens-adjusted) = 0.6498235078.
    #[test]
    fn ks_two_sample_with_ties_matches_scipy() {
        let a: Vec<f64> = [1.0, 2.0, 2.5, 4.0, 7.0]
            .iter().flat_map(|&v| [v, v, v])
            .chain([3.0, 5.0, 9.0, 6.0, 8.0])
            .collect();
        let b: Vec<f64> = [1.5, 3.0, 3.5, 6.0]
            .iter().flat_map(|&v| [v, v, v])
            .chain([2.0, 4.0, 8.0, 5.0, 7.5, 9.5])
            .collect();
        assert_eq!(a.len(), 20);
        assert_eq!(b.len(), 18);

        let r = ks_two_sample_test(&a, None, &b, None).unwrap();
        assert!(r.has_ties);
        assert_abs_diff_eq!(r.d_statistic, 0.227_777_777_777_777_8, epsilon = 1e-12);
        assert_abs_diff_eq!(r.n_eff, 20.0 * 18.0 / 38.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.649_823_507_830_415_3, epsilon = 1e-6);
    }

    /// Identical samples → D = 0, p = 1.
    #[test]
    fn ks_two_sample_identical() {
        let v = [1.0, 2.0, 3.0, 4.0, 5.0];
        let r = ks_two_sample_test(&v, None, &v, None).unwrap();
        assert_abs_diff_eq!(r.d_statistic, 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r.p_value, 1.0, epsilon = 1e-12);
    }

    /// Disjoint samples → D = 1.
    #[test]
    fn ks_two_sample_disjoint() {
        let r = ks_two_sample_test(&[1.0, 2.0, 3.0], None, &[10.0, 11.0, 12.0], None).unwrap();
        assert_abs_diff_eq!(r.d_statistic, 1.0, epsilon = 1e-12);
    }

    /// Frequency weights must reproduce the replicate-expanded result.
    #[test]
    fn ks_two_sample_weights_match_expansion() {
        let x = [-0.49, -0.0, 0.73, -1.43, 1.31];
        let y = [2.04, -0.74, 0.63, 1.42, 0.1, 0.93];
        let wx = [2.0, 3.0, 1.0, 1.0, 2.0];
        let wy = [2.0; 6];
        let expanded_x: Vec<f64> = x.iter()
            .zip(&wx)
            .flat_map(|(&v, &w)| std::iter::repeat_n(v, w as usize))
            .collect();
        let expanded_y: Vec<f64> = y.iter()
            .flat_map(|&v| std::iter::repeat_n(v, 2))
            .collect();

        let weighted = ks_two_sample_test(&x, Some(&wx), &y, Some(&wy)).unwrap();
        let expanded = ks_two_sample_test(&expanded_x, None, &expanded_y, None).unwrap();
        assert_abs_diff_eq!(weighted.d_statistic, expanded.d_statistic, epsilon = 1e-12);
        assert_abs_diff_eq!(weighted.p_value, expanded.p_value, epsilon = 1e-9);
        assert_abs_diff_eq!(weighted.n1, expanded.n1, epsilon = 1e-12);
        assert_abs_diff_eq!(weighted.n2, expanded.n2, epsilon = 1e-12);
        assert_abs_diff_eq!(weighted.n_eff, expanded.n_eff, epsilon = 1e-12);
    }

    /// Non-finite values and zero weights are dropped per-sample.
    #[test]
    fn ks_two_sample_drops_missing_and_zero_weight() {
        let x = [1.0, f64::NAN, 2.0, 3.0];
        let y = [4.0, 5.0, f64::NAN, 6.0];
        let w = [1.0, 1.0, 1.0, 0.0];
        let r = ks_two_sample_test(&x, Some(&w), &y, None).unwrap();
        assert_abs_diff_eq!(r.n1, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.n2, 3.0, epsilon = 1e-12);
    }

    #[test]
    fn ks_two_sample_edge_cases() {
        // Empty sample.
        assert!(ks_two_sample_test(&[], None, &[1.0], None).is_err());
        // All-zero weights.
        assert!(ks_two_sample_test(&[1.0], Some(&[0.0]), &[1.0], None).is_err());
        // Weight-length mismatch.
        assert!(ks_two_sample_test(&[1.0], Some(&[]), &[1.0], None).is_err());
        // Small effective size → warning.
        let r = ks_two_sample_test(&[1.0, 2.0], None, &[3.0], None).unwrap();
        assert!(r.warning.is_some());
        // Serde round-trip.
        let r = ks_two_sample_test(&[1.0, 2.0, 3.0], None, &[4.0, 5.0, 6.0], None).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: TwoSampleKsResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.d_statistic, r.d_statistic, epsilon = 1e-15);
    }
}
