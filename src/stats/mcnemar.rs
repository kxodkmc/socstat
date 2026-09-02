//! McNemar's test: compares two paired binary proportions from a 2×2
//! contingency table of before/after or matched-pair outcomes.
//!
//! The test evaluates only the discordant pairs `b` (first level → second
//! level) and `c` (second level → first level):
//!
//! - Exact binomial p-value `2 · min(P(X ≤ min(b,c)), 1 − P(X ≤ min(b,c)−1))`
//!   with `X ~ Bin(b + c, 0.5)`, used automatically when `b + c < 25`
//!   (the statsmodels/SPSS convention) and the counts are integers.
//! - Otherwise chi-square `(max(|b − c| − 1, 0))² / (b + c)` with continuity
//!   correction (R's default `correct = TRUE`), `df = 1`.
//!
//! Weights are **frequency weights** (each case counts as `weight`
//! replicates); a row with a missing value in either variable or a
//! non-positive weight is dropped pair-wise.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::data::ColumnData;
use crate::dist::{ChiSquaredDist, Distribution};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{extract_labels, ln_gamma, positive_weight};

/// Below this many discordant pairs the binomial exact test is preferred.
const EXACT_MAX_DISCORDANT: f64 = 25.0;

/// Result of McNemar's test on a paired 2×2 table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McNemarResult {
    /// Observed (weighted) counts `[[a, b], [c, d]]`, rows = first variable's
    /// levels, cols = second variable's levels (levels lexicographically
    /// ordered).
    pub table: [[f64; 2]; 2],
    pub row_labels: [String; 2],
    pub col_labels: [String; 2],
    /// Total number of (weighted) pairs retained.
    pub n: f64,
    /// The discordant pair counts `(b, c)`.
    pub discordant: (f64, f64),
    /// Whether the p-value comes from the exact binomial distribution.
    pub exact: bool,
    /// McNemar chi-square statistic. With `exact` this is the uncorrected
    /// `(b − c)² / (b + c)` reported for reference only.
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
    /// Set when the exact binomial was unavailable (fractional weighted
    /// discordant counts) and the chi-square approximation was used instead.
    pub warning: Option<String>,
}

/// McNemar's test on two paired binary variables.
///
/// Both variables must have exactly two categories. Non-integer weighted
/// discordant counts force the chi-square approximation with a `warning`.
pub fn mcnemar_test(
    v1: &ColumnData,
    v2: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<McNemarResult> {
    let n = v1.len();
    if n != v2.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: v2.len(),
        });
    }
    if let Some(w) = weights
        && w.len() != n
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: w.len(),
        });
    }

    let l1 = extract_labels(v1);
    let l2 = extract_labels(v2);
    let rows: Vec<String> = l1.iter().flatten().cloned().collect::<BTreeSet<_>>()
        .into_iter().collect();
    let cols: Vec<String> = l2.iter().flatten().cloned().collect::<BTreeSet<_>>()
        .into_iter().collect();
    if rows.len() != 2 || cols.len() != 2 {
        return Err(SocStatError::InsufficientData(
            "McNemar's test requires both variables to have exactly two categories".into(),
        ));
    }

    let mut counts = [[0.0_f64; 2]; 2];
    for i in 0..n {
        let (Some(a), Some(b)) = (&l1[i], &l2[i]) else { continue };
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        let ri = if *a == rows[0] { 0 } else { 1 };
        let ci = if *b == cols[0] { 0 } else { 1 };
        counts[ri][ci] += w;
    }

    let n_total: f64 = counts.iter().flat_map(|r| r.iter()).sum();
    let (b, c) = (counts[0][1], counts[1][0]);
    let n_dis = b + c;

    if n_dis <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "McNemar's test needs at least one discordant pair".into(),
        ));
    }

    let chi_square_uncorrected = (b - c) * (b - c) / n_dis;
    let can_exact = b.fract() == 0.0 && c.fract() == 0.0;
    let use_exact = can_exact && n_dis < EXACT_MAX_DISCORDANT;

    let (exact, chi_square, p_value, warning) = if use_exact {
        let p = exact_binomial_two_sided(b.round() as u64, c.round() as u64);
        (true, chi_square_uncorrected, p, None)
    } else {
        let corrected = ((b - c).abs() - 1.0).max(0.0);
        let stat = corrected * corrected / n_dis;
        let p = (1.0 - ChiSquaredDist::new(1.0)?.cdf(stat)).clamp(0.0, 1.0);
        let warning = if n_dis < EXACT_MAX_DISCORDANT {
            Some(
                "fractional weighted discordant counts: the chi-square \
                 approximation was used instead of the exact binomial test"
                    .into(),
            )
        } else {
            None
        };
        (false, stat, p, warning)
    };

    Ok(McNemarResult {
        table: counts,
        row_labels: [rows[0].clone(), rows[1].clone()],
        col_labels: [cols[0].clone(), cols[1].clone()],
        n: n_total,
        discordant: (b, c),
        exact,
        chi_square,
        df: 1.0,
        p_value,
        warning,
    })
}

/// Two-sided exact p-value for the sign test of `b` vs `c` discordant pairs:
/// `2 · min(P(X ≤ k), 1 − P(X ≤ k − 1))` with `X ~ Bin(b + c, 0.5)`,
/// `k = min(b, c)`.
fn exact_binomial_two_sided(b: u64, c: u64) -> f64 {
    let n = (b + c) as usize;
    let k = b.min(c) as usize;
    let cdf = binom_cdf_half(k, n);
    let p_lower = cdf;
    let p_upper = 1.0 - binom_cdf_half(k.saturating_sub(1), n);
    (2.0 * p_lower.min(p_upper)).min(1.0)
}

/// `P(X ≤ k)` for `X ~ Bin(n, 0.5)`, computed via log-gamma coefficients.
fn binom_cdf_half(k: usize, n: usize) -> f64 {
    let ln_half = -(n as f64) * std::f64::consts::LN_2;
    (0..=k).fold(0.0, |acc, i| {
        let ln_choose = ln_gamma(n as f64 + 1.0)
            - ln_gamma(i as f64 + 1.0)
            - ln_gamma((n - i) as f64 + 1.0);
        acc + (ln_choose + ln_half).exp()
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn text_col(labels: &[&str]) -> ColumnData {
        ColumnData::Text(labels.iter().map(|s| Some(s.to_string())).collect())
    }

    /// b = 15, c = 10 (25 discordant pairs): chi-square with continuity
    /// correction. Reference values from R/statsmodels.
    /// corrected chi2 = (5-1)^2/25 = 0.64, p = 0.4237108.
    #[test]
    fn mcnemar_chi_square_corrected_matches_reference() {
        // (before, after) rows: 5×(no,no), 15×(no,yes), 10×(yes,no), 5×(yes,yes).
        let pairs: Vec<(&str, &str)> = std::iter::repeat_n(("no", "no"), 5)
            .chain(std::iter::repeat_n(("no", "yes"), 15))
            .chain(std::iter::repeat_n(("yes", "no"), 10))
            .chain(std::iter::repeat_n(("yes", "yes"), 5))
            .collect();
        let before: Vec<&str> = pairs.iter().map(|p| p.0).collect();
        let after: Vec<&str> = pairs.iter().map(|p| p.1).collect();
        let r = mcnemar_test(&text_col(&before), &text_col(&after), None).unwrap();
        assert!(!r.exact);
        assert_abs_diff_eq!(r.discordant.0, 15.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.discordant.1, 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.chi_square, 0.64, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.423_710_797_166_793_6, epsilon = 1e-12);
        assert_eq!(r.df, 1.0);
    }

    /// b = 3, c = 12 (15 discordant pairs): exact binomial.
    /// p = 2 * P(Bin(15, 0.5) <= 3) = 2 * 576/32768 = 0.03515625.
    #[test]
    fn mcnemar_exact_binomial_matches_reference() {
        // 2×(no,no), 3×(no,yes), 12×(yes,no), 1×(yes,yes).
        let pairs: Vec<(&str, &str)> = std::iter::repeat_n(("no", "no"), 2)
            .chain(std::iter::repeat_n(("no", "yes"), 3))
            .chain(std::iter::repeat_n(("yes", "no"), 12))
            .chain(std::iter::repeat_n(("yes", "yes"), 1))
            .collect();
        let before: Vec<&str> = pairs.iter().map(|p| p.0).collect();
        let after: Vec<&str> = pairs.iter().map(|p| p.1).collect();
        let r = mcnemar_test(&text_col(&before), &text_col(&after), None).unwrap();
        assert!(r.exact);
        assert_abs_diff_eq!(r.p_value, 0.035_156_25, epsilon = 1e-12);
        // Uncorrected chi2 reported for reference: (3-12)^2/15 = 5.4.
        assert_abs_diff_eq!(r.chi_square, 5.4, epsilon = 1e-12);
    }

    /// b = c = 6: no evidence against equal proportions, p = 1.
    #[test]
    fn mcnemar_symmetric_gives_p_one() {
        let before = text_col(&["no", "no", "no", "no", "no", "no",
                                "yes", "yes", "yes", "yes", "yes", "yes"]);
        let after = text_col(&["yes", "yes", "yes", "yes", "yes", "yes",
                               "no", "no", "no", "no", "no", "no"]);
        let r = mcnemar_test(&before, &after, None).unwrap();
        assert!(r.exact);
        assert_abs_diff_eq!(r.p_value, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mcnemar_weighted_equals_frequency_expansion() {
        // 2×(no→yes) and 1×(yes→no) with frequency weights.
        let before = text_col(&["no", "no", "yes"]);
        let after = text_col(&["yes", "no", "no"]);
        let w = [2.0, 1.0, 1.0];
        let r = mcnemar_test(&before, &after, Some(&w)).unwrap();
        // b = 2, c = 1 → 3 discordant pairs, exact.
        assert!(r.exact);
        assert_abs_diff_eq!(r.discordant.0, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.discordant.1, 1.0, epsilon = 1e-12);
        // 2 * P(Bin(3, 0.5) <= 1) = 2 * 0.5 = 1.
        assert_abs_diff_eq!(r.p_value, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mcnemar_fractional_weights_fall_back_to_chi_square() {
        // 3×(no→yes) and 1×(yes→no), each weighted 0.5 → b = 1.5, c = 0.5.
        let before = text_col(&["no", "no", "no", "yes"]);
        let after = text_col(&["yes", "yes", "yes", "no"]);
        let w = [0.5, 0.5, 0.5, 0.5];
        let r = mcnemar_test(&before, &after, Some(&w)).unwrap();
        // b = 1.5, c = 0.5: fractional → chi-square with warning.
        assert!(!r.exact);
        assert!(r.warning.is_some());
        // Corrected: (|1.5-0.5| - 1)^2 / 2 = 0.
        assert_abs_diff_eq!(r.chi_square, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mcnemar_edge_cases() {
        // Concordant-only data → error (no discordant pairs).
        let a = text_col(&["no", "yes"]);
        let b = text_col(&["no", "yes"]);
        assert!(mcnemar_test(&a, &b, None).is_err());
        // Three categories → error.
        let three = text_col(&["no", "maybe", "yes"]);
        assert!(mcnemar_test(&three, &b, None).is_err());
        // Length mismatch.
        let short = text_col(&["no"]);
        assert!(mcnemar_test(&short, &b, None).is_err());
        // Missing rows pair-wise dropped: 2 usable pairs remain (1 discordant).
        let before = text_col(&["no", "yes", "no"]);
        let after = ColumnData::Text(vec![Some("yes".into()), None, Some("no".into())]);
        let r = mcnemar_test(&before, &after, None).unwrap();
        assert_abs_diff_eq!(r.n, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.discordant.0, 1.0, epsilon = 1e-12);
    }

    #[test]
    fn mcnemar_serde_round_trip() {
        let before = text_col(&["no", "yes", "no"]);
        let after = text_col(&["yes", "yes", "no"]);
        let r = mcnemar_test(&before, &after, None).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: McNemarResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.p_value, r.p_value, epsilon = 1e-15);
        assert_eq!(back.exact, r.exact);
    }
}
