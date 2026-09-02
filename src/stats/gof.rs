//! Chi-square goodness-of-fit test for a categorical variable: compares
//! observed category counts against expected counts under a hypothesized
//! distribution.
//!
//! `chi2 = Σ (O_i − E_i)² / E_i` with `E_i = N · p_i` and `k − 1` degrees
//! of freedom (the same statistic as R's `chisq.test(x, p = ...)`). When no
//! `p` is supplied, equal probabilities `1/k` are used. Weights are
//! **frequency weights** (each case counts as `weight` replicates); a case
//! with a missing value or a non-positive weight is excluded.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::data::ColumnData;
use crate::dist::{ChiSquaredDist, Distribution};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{extract_labels, positive_weight};

/// Result of a chi-square goodness-of-fit test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareGof {
    /// Category labels, lexicographically sorted.
    pub categories: Vec<String>,
    /// Observed (weighted) counts, aligned with `categories`.
    pub observed: Vec<f64>,
    /// Expected counts under the null, aligned with `categories`.
    pub expected: Vec<f64>,
    pub n: f64,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
    /// Set when an expected count is below 5, following R's convention of
    /// warning that the chi-square approximation may be unreliable.
    pub warning: Option<String>,
}

/// Chi-square goodness-of-fit test of a categorical variable.
///
/// `expected_probs` holds the null-hypothesis probability of each observed
/// category (aligned with the lexicographically sorted category labels);
/// `None` tests for equal probabilities. Probabilities must be positive and
/// sum to 1.
pub fn chi_square_gof(
    var: &ColumnData,
    weights: Option<&[f64]>,
    expected_probs: Option<&[f64]>,
) -> SocStatResult<ChiSquareGof> {
    let n = var.len();
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: w.len(),
        });
    }

    let labels = extract_labels(var);
    let mut counts: BTreeMap<String, f64> = BTreeMap::new();
    for (i, label) in labels.iter().enumerate() {
        let Some(label) = label else { continue };
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        *counts.entry(label.clone()).or_default() += w;
    }

    let categories: Vec<String> = counts.keys().cloned().collect();
    let k = categories.len();
    if k < 2 {
        return Err(SocStatError::InsufficientData(
            "chi-square goodness-of-fit needs at least 2 categories".into(),
        ));
    }

    let observed: Vec<f64> = categories.iter().map(|c| counts[c]).collect();
    let n_total: f64 = observed.iter().sum();

    let probs: Vec<f64> = match expected_probs {
        None => vec![1.0 / k as f64; k],
        Some(p) => {
            if p.len() != k {
                return Err(SocStatError::InvalidInput(format!(
                    "expected {} probabilities for the {} observed categories",
                    k, k
                )));
            }
            if !p.iter().all(|q| q.is_finite() && *q > 0.0) {
                return Err(SocStatError::InvalidInput(
                    "expected probabilities must all be positive".into(),
                ));
            }
            let sum: f64 = p.iter().sum();
            if (sum - 1.0).abs() > 1e-8 {
                return Err(SocStatError::InvalidInput(
                    "expected probabilities must sum to 1".into(),
                ));
            }
            p.to_vec()
        }
    };

    let expected: Vec<f64> = probs.iter().map(|q| q * n_total).collect();
    let chi_square: f64 = observed
        .iter()
        .zip(&expected)
        .map(|(&o, &e)| (o - e) * (o - e) / e)
        .sum();

    let df = (k - 1) as f64;
    let dist = ChiSquaredDist::new(df)?;
    let p_value = (1.0 - dist.cdf(chi_square)).clamp(0.0, 1.0);

    let warning = if expected.iter().any(|&e| e < 5.0) {
        Some("expected count below 5: the chi-square approximation may be unreliable".into())
    } else {
        None
    };

    Ok(ChiSquareGof {
        categories,
        observed,
        expected,
        n: n_total,
        chi_square,
        df,
        p_value,
        warning,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn text_col(labels: &[&str]) -> ColumnData {
        ColumnData::Text(labels.iter().map(|s| Some(s.to_string())).collect())
    }

    /// Mendel's classic 9:3:3:1 segregation data.
    /// R: chisq.test(c(315,108,101,32), p=c(9,3,3,1)/16)
    /// → chi2 = 0.470024, df = 3, p = 0.9254259.
    #[test]
    fn gof_mendel_matches_reference() {
        // Labels sort lexicographically (BTreeMap order): round-green,
        // round-yellow, wrinkled-green, wrinkled-yellow.
        let sorted_labels: Vec<String> = {
            let mut v: Vec<String> = [
                "round-yellow".to_string(),
                "round-green".to_string(),
                "wrinkled-yellow".to_string(),
                "wrinkled-green".to_string(),
            ]
            .to_vec();
            v.sort();
            v
        };
        let probs_by_label: std::collections::BTreeMap<&str, f64> = [
            ("round-yellow", 9.0 / 16.0),
            ("round-green", 3.0 / 16.0),
            ("wrinkled-yellow", 3.0 / 16.0),
            ("wrinkled-green", 1.0 / 16.0),
        ]
        .into_iter()
        .collect();
        let p: Vec<f64> = sorted_labels.iter().map(|l| probs_by_label[l.as_str()]).collect();
        // observed counts in label order
        let obs_by_label: std::collections::BTreeMap<&str, usize> = [
            ("round-yellow", 315),
            ("round-green", 108),
            ("wrinkled-yellow", 101),
            ("wrinkled-green", 32),
        ]
        .into_iter()
        .collect();

        // Build a column where each label repeats `count` times.
        let mut labels: Vec<&str> = Vec::new();
        for (label, count) in &obs_by_label {
            labels.extend(std::iter::repeat_n(*label, *count));
        }
        let col = text_col(&labels);

        let r = chi_square_gof(&col, None, Some(&p)).unwrap();
        assert_eq!(r.categories, sorted_labels);
        let obs_sorted: Vec<f64> = sorted_labels.iter().map(|l| obs_by_label[l.as_str()] as f64).collect();
        assert_eq!(r.observed, obs_sorted);
        assert_abs_diff_eq!(r.chi_square, 0.470_023_980_815_347_7, epsilon = 1e-12);
        assert_abs_diff_eq!(r.df, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.925_425_895_103_616, epsilon = 1e-12);
        assert!(r.warning.is_none());
    }

    /// Equal-probability test (R: chisq.test(c(30,45,28,52,41,34))).
    #[test]
    fn gof_equal_probs_matches_reference() {
        let obs = [30, 45, 28, 52, 41, 34];
        let mut labels: Vec<&str> = Vec::new();
        for (i, &count) in obs.iter().enumerate() {
            labels.extend(std::iter::repeat_n(["a", "b", "c", "d", "e", "f"][i], count));
        }
        let col = text_col(&labels);
        let r = chi_square_gof(&col, None, None).unwrap();
        assert_abs_diff_eq!(r.chi_square, 11.304_347_826_086_959, epsilon = 1e-12);
        assert_abs_diff_eq!(r.df, 5.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.045_668_641_315_982_13, epsilon = 1e-12);
        // Expected 230/6 ≈ 38.33 < 5? No: 38.33 ≥ 5 → no warning.
        assert!(r.warning.is_none());
    }

    #[test]
    fn gof_weighted_equals_frequency_expansion() {
        let col = text_col(&["x", "y", "x", "y"]);
        let w = [2.0, 3.0, 1.0, 1.0];
        let r = chi_square_gof(&col, Some(&w), None).unwrap();
        // x: 2+1=3, y: 3+1=4; E = 3.5 each; chi2 = 0.5/3.5 + 0.5/3.5.
        assert_abs_diff_eq!(r.observed[0], 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.observed[1], 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.n, 7.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.chi_square, (0.25 + 0.25) / 3.5, epsilon = 1e-12);
        // Expected counts 3.5 < 5 → warning.
        assert!(r.warning.is_some());
    }

    #[test]
    fn gof_input_validation() {
        let col = text_col(&["a", "b", "a"]);
        // Wrong probability length.
        assert!(chi_square_gof(&col, None, Some(&[0.5])).is_err());
        // Negative probability.
        assert!(chi_square_gof(&col, None, Some(&[1.5, -0.5])).is_err());
        // Probabilities do not sum to 1.
        assert!(chi_square_gof(&col, None, Some(&[0.5, 0.4])).is_err());
        // Weight length mismatch.
        assert!(chi_square_gof(&col, Some(&[1.0]), None).is_err());
        // Single category → not testable.
        let one = text_col(&["a", "a"]);
        assert!(chi_square_gof(&one, None, None).is_err());
    }

    #[test]
    fn gof_serde_round_trip() {
        let col = text_col(&["a", "b", "a", "b", "a"]);
        let r = chi_square_gof(&col, None, None).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: ChiSquareGof = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.chi_square, r.chi_square, epsilon = 1e-15);
        assert_eq!(back.warning, r.warning);
    }
}
