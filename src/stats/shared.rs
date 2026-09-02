//! Shared internal helpers reused across the analysis modules.
//!
//! Keeps the numerically-stable weighted summary, group splitting, ranking,
//! pairing/alignment, and user-missing handling in one place so [`tests`],
//! [`regression`], and the newer analysis modules all reuse the same code.
//!
//! All weights here are **frequency weights**: each case counts as `weight`
//! replicates, and a case with a weight ≤ 0 (or a missing/non-finite weight)
//! is excluded.

use std::collections::BTreeMap;

use crate::data::{ColumnData, Dataset};
use crate::dist::Distribution;
use crate::error::{SocStatError, SocStatResult};

/// True for a usable frequency weight: finite and strictly positive.
/// NaN weights must be excluded rather than silently treated as valid.
pub(crate) fn positive_weight(w: f64) -> bool {
    w.is_finite() && w > 0.0
}

/// Format a number the way group labels are shown: integers without a
/// trailing `.0`.
pub(crate) fn format_num(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

/// Two-sided tail probability for a symmetric distribution.
pub(crate) fn two_sided_tail(dist: &impl Distribution, stat: f64) -> f64 {
    2.0 * (1.0 - dist.cdf(stat.abs()))
}

/// Map a column to its display labels: text values pass through, numeric
/// values are formatted like group labels.
pub(crate) fn extract_labels(col: &ColumnData) -> Vec<Option<String>> {
    match col {
        ColumnData::Numeric(v) => v.iter().map(|o| o.map(format_num)).collect(),
        ColumnData::Text(v) => v.clone(),
    }
}

// ---------------------------------------------------------------------------
// Group splitting
// ---------------------------------------------------------------------------

/// One group split out of the dependent variable.
#[derive(Debug, Clone)]
pub(crate) struct GroupedData {
    pub(crate) label: String,
    /// (value, weight) pairs, weight already validated > 0.
    pub(crate) pairs: Vec<(f64, f64)>,
}

/// Split a numeric dependent column into groups by a grouping column,
/// dropping rows where either value is missing or the weight is ≤ 0.
///
/// Groups are returned in lexicographic label order (the same convention as
/// [`crosstab`](crate::stats::crosstab)), so results are independent of the
/// input row order.
pub(crate) fn split_groups(
    dep: &ColumnData,
    group: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<Vec<GroupedData>> {
    let dep_slice = dep.as_numeric().ok_or(SocStatError::TypeMismatch {
        var: String::new(),
        expected: "Numeric",
        actual: "Text",
    })?;
    let n = dep_slice.len();
    if n != group.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: group.len(),
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

    let group_labels: Vec<Option<String>> = match group {
        ColumnData::Numeric(v) => v.iter().map(|o| o.map(format_num)).collect(),
        ColumnData::Text(v) => v.clone(),
    };

    // Group by label in a BTreeMap so groups come out lexicographically
    // sorted, independent of appearance order.
    let mut acc: BTreeMap<String, Vec<(f64, f64)>> = BTreeMap::new();
    for i in 0..n {
        let Some(x) = dep_slice[i] else { continue };
        let Some(label) = &group_labels[i] else { continue };
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        acc.entry(label.clone()).or_default().push((x, w));
    }

    Ok(acc.into_iter().map(|(label, pairs)| GroupedData { label, pairs }).collect())
}

// ---------------------------------------------------------------------------
// Weighted two-pass summaries
// ---------------------------------------------------------------------------

/// Weighted summary statistics computed with a two-pass algorithm.
///
/// Pass 1 computes the weighted mean; pass 2 accumulates weighted squared
/// deviations around the mean. This is numerically stable for data with a
/// large mean and a tiny variance (catastrophic cancellation avoided).
#[derive(Debug, Clone)]
pub(crate) struct WeightedSummary {
    /// Effective sample size (sum of weights).
    pub(crate) n: f64,
    pub(crate) mean: f64,
    /// Sum of squared deviations around the mean (weighted).
    pub(crate) sum_squares: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
}

impl WeightedSummary {
    /// Compute from (value, weight) pairs. Pairs with a non-positive weight
    /// or a non-finite value are excluded.
    pub(crate) fn compute(pairs: &[(f64, f64)]) -> SocStatResult<Self> {
        let mut n_valid = 0usize;
        let mut sum_w = 0.0;
        let mut sum_wx = 0.0;
        let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);

        for &(x, w) in pairs {
            if !x.is_finite() || !positive_weight(w) {
                continue;
            }
            n_valid += 1;
            sum_w += w;
            sum_wx += w * x;
            min = min.min(x);
            max = max.max(x);
        }

        if n_valid == 0 {
            return Err(SocStatError::InsufficientData(
                "no valid (weighted) cases to analyze".into(),
            ));
        }

        let mean = sum_wx / sum_w;
        let mut sum_squares = 0.0;
        for &(x, w) in pairs {
            if !x.is_finite() || !positive_weight(w) {
                continue;
            }
            let d = x - mean;
            sum_squares += w * d * d;
        }

        Ok(Self { n: sum_w, mean, sum_squares, min, max })
    }

    /// Sample variance (denominator n−1).
    pub(crate) fn variance(&self) -> f64 {
        if self.n > 1.0 {
            self.sum_squares / (self.n - 1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Paired alignment
// ---------------------------------------------------------------------------

/// Row-aligned paired data with optional weights. `weights` is empty when the
/// caller supplied none.
#[derive(Debug, Clone)]
pub(crate) struct PairedData {
    pub(crate) v1: Vec<f64>,
    pub(crate) v2: Vec<f64>,
    pub(crate) weights: Vec<f64>,
    /// Number of valid (weighted) pairs kept.
    pub(crate) n: usize,
}

/// Align two paired numeric slices row-wise, dropping rows where either value
/// is missing/non-finite or the weight is non-positive.
pub(crate) fn align_paired_slices(
    v1: &[Option<f64>],
    v2: &[Option<f64>],
    weights: Option<&[f64]>,
) -> SocStatResult<PairedData> {
    if v1.len() != v2.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: v1.len(),
            got: v2.len(),
        });
    }
    if let Some(w) = weights
        && w.len() != v1.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: v1.len(),
            got: w.len(),
        });
    }
    let mut data = PairedData { v1: Vec::new(), v2: Vec::new(), weights: Vec::new(), n: 0 };
    for i in 0..v1.len() {
        let (Some(a), Some(b)) = (v1[i], v2[i]) else { continue };
        if !a.is_finite() || !b.is_finite() {
            continue;
        }
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        data.v1.push(a);
        data.v2.push(b);
        data.weights.push(w);
        data.n += 1;
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// Ranking across groups
// ---------------------------------------------------------------------------

/// Average-rank all groups together (mid-ranks across ties), treating each
/// case as `weight` replicates. Returns each group's rank sum, whether ties
/// were present, and the tie-correction sum Σ(tⱼ³ − tⱼ).
pub(crate) fn rank_all_groups(groups: &[GroupedData]) -> (Vec<f64>, bool, f64) {
    // Pool: (value, group_index, weight).
    let mut all: Vec<(f64, usize, f64)> = groups
        .iter()
        .enumerate()
        .flat_map(|(gi, g)| g.pairs.iter().map(move |(v, w)| (*v, gi, *w)))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n = all.len();
    let mut rank_sums = vec![0.0_f64; groups.len()];
    let mut has_ties = false;
    let mut tie_adj = 0.0_f64;

    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && all[j].0 == all[i].0 {
            j += 1;
        }
        let count = j - i;
        // mid-rank over weighted replicates: average rank = (i+1 + j) / 2.
        let avg_rank = (i as f64 + 1.0 + j as f64) / 2.0;
        for k in i..j {
            rank_sums[all[k].1] += avg_rank * all[k].2;
        }
        if count > 1 {
            has_ties = true;
            tie_adj += (count as f64).powi(3) - count as f64;
        }
        i = j;
    }
    (rank_sums, has_ties, tie_adj)
}

/// Average ranks with ties (mid-ranks) for a finite numeric slice.
/// Two-sided tail probability of the Kolmogorov distribution at `lambda`,
/// from the alternating series `2 Σ (−1)^(j−1) exp(−2 j² λ²)`.
///
/// `λ ≤ 0` (an exactly-reproduced distribution) gives p = 1; for very small
/// λ the series converges slowly, so ~`1/λ` terms are needed — 100 covers
/// λ down to 0.01 with margin.
pub(crate) fn kolmogorov_two_sided_p(lambda: f64) -> f64 {
    if lambda <= 0.0 {
        return 1.0;
    }
    let mut p = 0.0;
    for j in 1..=100 {
        let sign = if j % 2 == 1 { 1.0 } else { -1.0 };
        p += sign * (-2.0 * (j as f64).powi(2) * lambda * lambda).exp();
    }
    (2.0 * p).clamp(0.0, 1.0)
}

/// Stephens' finite-sample adjustment factor for a Kolmogorov–Smirnov
/// statistic with effective sample size `n_eff`: multiply the statistic by
/// `sqrt(n_eff) + 0.12 + 0.11 / sqrt(n_eff)` before the asymptotic series.
pub(crate) fn stephens_lambda(n_eff: f64, d: f64) -> f64 {
    let root = n_eff.sqrt();
    (root + 0.12 + 0.11 / root) * d
}

pub(crate) fn rank_data(data: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..data.len()).collect();
    idx.sort_by(|&a, &b| data[a].total_cmp(&data[b]));
    let mut ranks = vec![0.0; data.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && data[idx[j + 1]] == data[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for k in i..=j {
            ranks[idx[k]] = avg;
        }
        i = j + 1;
    }
    ranks
}

// ---------------------------------------------------------------------------
// Dataset / missing handling
// ---------------------------------------------------------------------------

/// Extract a variable's numeric values with user-missing values converted to
/// `None`, so dataset-level analyses exclude them (Hard Rule 4).
pub(crate) fn cleaned_numeric_column(ds: &Dataset, name: &str) -> SocStatResult<Vec<Option<f64>>> {
    let idx = ds.index_of(name)?;
    let var = &ds.variables()[idx];
    let col = ds.column(idx)?;
    let slice = col.as_numeric().ok_or_else(|| SocStatError::TypeMismatch {
        var: name.to_string(),
        expected: "Numeric",
        actual: "Text",
    })?;
    Ok(slice
        .iter()
        .map(|o| o.filter(|v| !var.is_user_missing(*v)))
        .collect())
}

// ---------------------------------------------------------------------------
// Special functions (kept statrs-free per the crate architecture)
// ---------------------------------------------------------------------------

/// Natural logarithm of the gamma function (Lanczos approximation).
///
/// Used to compute binomial coefficients in log space (e.g. Fisher's exact
/// test) without overflow. Kept as a standalone scalar helper so no module
/// outside [`dist`](crate::dist) depends on external special-function crates.
#[allow(clippy::excessive_precision, clippy::inconsistent_digit_grouping)] // verbatim g=7 Lanczos constants
pub(crate) fn ln_gamma(z: f64) -> f64 {
    const C: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if z < 0.5 {
        // Reflection formula: Γ(z)Γ(1−z) = π / sin(πz).
        std::f64::consts::PI.ln() - (std::f64::consts::PI * z).sin().ln() - ln_gamma(1.0 - z)
    } else {
        let x = z - 1.0;
        let mut acc = C[0];
        for (i, c) in C.iter().enumerate().skip(1) {
            acc += c / (x + i as f64);
        }
        let t = x + 7.5;
        let half_ln_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
        half_ln_2pi + (x + 0.5) * t.ln() - t + acc.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ln_gamma_matches_known_values() {
        // ln Γ(1) = 0, ln Γ(2) = 0, ln Γ(0.5) = ½ln π.
        assert_abs_diff_eq!(ln_gamma(1.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ln_gamma(2.0), 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ln_gamma(0.5), 0.5 * std::f64::consts::PI.ln(), epsilon = 1e-12);
        // ln Γ(5) = ln(24) ≈ 3.178...
        assert_abs_diff_eq!(ln_gamma(5.0), 24.0_f64.ln(), epsilon = 1e-10);
    }

    #[test]
    fn rank_all_groups_mid_ranks() {
        // Group A values [1, 3], group B values [2], group C values [3].
        // Combined sorted 1,2,3,3 → ranks 1,2,3.5,3.5.
        let groups = vec![
            GroupedData { label: "A".into(), pairs: vec![(1.0, 1.0), (3.0, 1.0)] },
            GroupedData { label: "B".into(), pairs: vec![(2.0, 1.0)] },
            GroupedData { label: "C".into(), pairs: vec![(3.0, 1.0)] },
        ];
        let (sums, has_ties, tie_adj) = rank_all_groups(&groups);
        assert_abs_diff_eq!(sums[0], 4.5, epsilon = 1e-12);
        assert_abs_diff_eq!(sums[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(sums[2], 3.5, epsilon = 1e-12);
        assert!(has_ties);
        assert_abs_diff_eq!(tie_adj, (2.0_f64).powi(3) - 2.0, epsilon = 1e-12);
    }

    #[test]
    fn align_paired_slices_drops_missing_and_bad_weight() {
        let v1 = [Some(1.0), Some(2.0), None, Some(f64::NAN), Some(5.0)];
        let v2 = [Some(1.0), Some(4.0), Some(3.0), Some(4.0), Some(5.0)];
        let w = [1.0, 2.0, 1.0, 1.0, 0.0];
        let d = align_paired_slices(&v1, &v2, Some(&w)).unwrap();
        // Rows 0 and 1 survive; rows 2 (missing), 3 (NaN), 4 (weight 0) drop.
        assert_eq!(d.n, 2);
        assert_abs_diff_eq!(d.v1[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.v1[1], 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.weights[0], 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(d.weights[1], 2.0, epsilon = 1e-12);
    }
}