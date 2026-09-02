//! Friedman rank-sum test for repeated measurements: `k` treatments each
//! observed on the same `n` subjects (blocks).
//!
//! Observations are ranked within each block (mid-ranks for ties) and
//!
//! `Q = [12/(n k (k+1)) · Σ Rⱼ² − 3 n (k+1)] / [1 − Σ(t³−t)/(n k (k²−1))]`
//!
//! is referred to χ²(k−1) (the same statistic as R's `friedman.test` and
//! SciPy's `friedmanchisquare`). Kendall's coefficient of concordance
//! `W = Q / (n (k−1))` is reported as the effect size.
//!
//! Weights are **frequency weights** (a block counts as `weight` replicates).
//! A block with a missing value in any treatment is dropped list-wise.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use serde::{Deserialize, Serialize};

use crate::dist::{ChiSquaredDist, Distribution};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::positive_weight;

/// Per-treatment summary of the Friedman test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanTreatment {
    pub label: String,
    /// Sum of within-block ranks received by this treatment.
    pub rank_sum: f64,
    pub mean_rank: f64,
    /// Unweighted mean of the raw observations (descriptive).
    pub mean: f64,
}

/// Result of the Friedman rank-sum test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FriedmanResult {
    /// Number of complete blocks retained.
    pub n: usize,
    /// Number of treatments.
    pub k: usize,
    pub treatments: Vec<FriedmanTreatment>,
    /// Number of within-block tie groups of size > 1, summed over blocks.
    pub n_tie_groups: usize,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
    /// Kendall's W coefficient of concordance (effect size, in [0, 1]).
    pub kendall_w: f64,
    /// Set when `n` or `k` is small enough that the χ² approximation may be
    /// unreliable (R's documentation threshold: n > 15 or k > 4).
    pub warning: Option<String>,
}

/// Friedman test on a complete block matrix `data[block][treatment]` with
/// optional per-block frequency weights.
///
/// Every block must have the same number of treatments; blocks containing
/// non-finite values are dropped, as are blocks with non-positive weights.
pub fn friedman_test(
    data: &[&[f64]],
    labels: &[&str],
    weights: Option<&[f64]>,
) -> SocStatResult<FriedmanResult> {
    let k = data.first().map_or(0, |b| b.len());
    if k < 3 {
        return Err(SocStatError::InsufficientData(
            "Friedman test needs at least 3 treatments".into(),
        ));
    }

    if let Some(w) = weights
        && w.len() != data.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: data.len(),
            got: w.len(),
        });
    }

    // Retain complete, weighted blocks.
    let mut blocks: Vec<(Vec<f64>, f64)> = Vec::with_capacity(data.len());
    for (i, block) in data.iter().enumerate() {
        if block.len() != k {
            return Err(SocStatError::InvalidInput(format!(
                "Friedman test: block {i} has {} treatments, expected {k}",
                block.len()
            )));
        }
        let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
        if block.iter().all(|v| v.is_finite()) && positive_weight(w) {
            blocks.push((block.to_vec(), w));
        }
    }
    let n = blocks.len();
    if n < 2 {
        return Err(SocStatError::InsufficientData(
            "Friedman test needs at least 2 complete blocks".into(),
        ));
    }

    // Rank within each block (mid-ranks); accumulate per-treatment rank sums
    // and the tie correction in a single pass over the blocks. Under
    // frequency weights a block counts as `w` replicate rows, so the
    // effective sample size is the weight sum, not the block count.
    let weight_sum: f64 = blocks.iter().map(|(_, w)| *w).sum();
    let mut rank_sums = vec![0.0_f64; k];
    let mut raw_sums = vec![0.0_f64; k];
    let mut tie_term = 0.0_f64;
    let mut n_tie_groups = 0usize;
    for (block, w) in &blocks {
        let ranks = midranks(block);
        for j in 0..k {
            rank_sums[j] += w * ranks[j];
            raw_sums[j] += block[j];
        }
        let (t3_minus_t, groups) = tie_stats(block);
        tie_term += w * t3_minus_t;
        n_tie_groups += groups;
    }

    // Q with the tie correction (R friedman.test / scipy friedmanchisquare).
    let n_f = weight_sum;
    let k_f = k as f64;
    let ssbn: f64 = rank_sums.iter().map(|r| r * r).sum();
    let uncorrected = 12.0 / (n_f * k_f * (k_f + 1.0)) * ssbn - 3.0 * n_f * (k_f + 1.0);
    let denom = 1.0 - tie_term / (n_f * k_f * (k_f * k_f - 1.0));
    let chi_square = if denom > 0.0 { uncorrected / denom } else { uncorrected };

    let df = k_f - 1.0;
    let p_value = 1.0 - ChiSquaredDist::new(df)?.cdf(chi_square);
    let kendall_w = (chi_square / (n_f * (k_f - 1.0))).clamp(0.0, 1.0);

    let warning = if n <= 15 && k <= 4 {
        Some(format!(
            "n = {n} and k = {k}: the chi-squared approximation may be inaccurate; \
             consult exact Friedman tables",
        ))
    } else {
        None
    };

    let treatments = (0..k)
        .map(|j| FriedmanTreatment {
            label: labels.get(j).map_or_else(|| format!("T{}", j + 1), |s| s.to_string()),
            rank_sum: rank_sums[j],
            mean_rank: rank_sums[j] / n_f,
            mean: raw_sums[j] / n_f,
        })
        .collect();

    Ok(FriedmanResult {
        n,
        k,
        treatments,
        n_tie_groups,
        chi_square,
        df,
        p_value,
        kendall_w,
        warning,
    })
}

/// Mid-ranks (average ranks for ties) of a slice.
fn midranks(values: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0.0; values.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && values[idx[j + 1]] == values[idx[i]] {
            j += 1;
        }
        let avg = ((i + 1) + (j + 1)) as f64 / 2.0;
        for &p in &idx[i..=j] {
            ranks[p] = avg;
        }
        i = j + 1;
    }
    ranks
}

/// Tie summary of a slice: `Σ (t³ − t)` over tie groups of size `t > 1`,
/// and the number of such groups.
fn tie_stats(values: &[f64]) -> (f64, usize) {
    let mut sorted: Vec<f64> = values.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut sum = 0.0;
    let mut groups = 0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j + 1 < sorted.len() && sorted[j + 1] == sorted[i] {
            j += 1;
        }
        if j > i {
            let t = (j - i + 1) as f64;
            sum += t * t * t - t;
            groups += 1;
        }
        i = j + 1;
    }
    (sum, groups)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn rounding_times() -> Vec<Vec<f64>> {
        [
            [5.40, 5.50, 5.55], [5.85, 5.70, 5.75], [5.20, 5.60, 5.50],
            [5.55, 5.50, 5.40], [5.90, 5.85, 5.70], [5.45, 5.55, 5.60],
            [5.40, 5.40, 5.35], [5.45, 5.50, 5.35], [5.25, 5.15, 5.00],
            [5.85, 5.80, 5.70], [5.25, 5.20, 5.10], [5.65, 5.55, 5.45],
            [5.60, 5.35, 5.45], [5.05, 5.00, 4.95], [5.50, 5.50, 5.40],
            [5.45, 5.55, 5.50], [5.55, 5.55, 5.35], [5.45, 5.50, 5.55],
            [5.50, 5.45, 5.25], [5.65, 5.60, 5.40], [5.70, 5.65, 5.55],
            [6.30, 6.30, 6.25],
        ].iter().map(|r| r.to_vec()).collect()
    }

    /// R `friedman.test(RoundingTimes)` (R docs dataset):
    /// chi2 = 11.142857, df = 2, p = 0.00380504, Rj = (53, 47, 32).
    #[test]
    fn friedman_rounding_times_matches_r() {
        let mat = rounding_times();
        let blocks: Vec<&[f64]> = mat.iter().map(|b| b.as_slice()).collect();
        let r = friedman_test(&blocks, &["Out", "Narrow", "Wide"], None).unwrap();
        assert_eq!(r.n, 22);
        assert_eq!(r.k, 3);
        assert_eq!(r.n_tie_groups, 4);
        assert_abs_diff_eq!(r.treatments[0].rank_sum, 53.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.treatments[1].rank_sum, 47.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.treatments[2].rank_sum, 32.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.chi_square, 11.142_857_142_857_132, epsilon = 1e-9);
        assert_abs_diff_eq!(r.df, 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.003_805_040_775_511_383, epsilon = 1e-9);
        assert_abs_diff_eq!(r.kendall_w, 0.253_246_753_246_753, epsilon = 1e-12);
        // n = 22 > 15 → the asymptotic chi-square is reliable: no warning.
        assert!(r.warning.is_none());
    }

    /// scipy `friedmanchisquare` on a tie-heavy 4×4 matrix (independent anchor):
    /// Q = 2.13157894736842, df = 3, p = 0.545550592398322, Rj = (8, 8.5, 11, 12.5).
    #[test]
    fn friedman_ties_match_scipy() {
        let mat: Vec<Vec<f64>> = vec![
            vec![4.0, 3.0, 2.0, 1.0],
            vec![1.0, 2.0, 3.5, 3.5],
            vec![2.0, 1.0, 3.0, 4.0],
            vec![1.0, 2.0, 2.0, 3.0],
        ];
        let blocks: Vec<&[f64]> = mat.iter().map(|b| b.as_slice()).collect();
        let r = friedman_test(&blocks, &["A", "B", "C", "D"], None).unwrap();
        assert_eq!(r.n, 4);
        assert_eq!(r.k, 4);
        assert_eq!(r.n_tie_groups, 2);
        assert_abs_diff_eq!(r.chi_square, 2.131_578_947_368_419_5, epsilon = 1e-9);
        assert_abs_diff_eq!(r.df, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 0.545_550_592_398_322_4, epsilon = 1e-9);
        assert_abs_diff_eq!(r.kendall_w, 0.177_631_578_947_368_3, epsilon = 1e-9);
        assert_abs_diff_eq!(r.treatments[0].rank_sum, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.treatments[1].rank_sum, 8.5, epsilon = 1e-12);
    }

    /// Frequency weights must reproduce the replicate-expanded test.
    #[test]
    fn friedman_weights_match_expansion() {
        let mat: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![3.0, 2.0, 1.0],
            vec![1.0, 3.0, 2.0],
        ];
        let blocks: Vec<&[f64]> = mat.iter().map(|b| b.as_slice()).collect();
        let w = [2.0, 1.0, 3.0];
        let mut expanded: Vec<Vec<f64>> = Vec::new();
        for (row, &wt) in mat.iter().zip(&w) {
            for _ in 0..wt as usize {
                expanded.push(row.clone());
            }
        }
        let exp_blocks: Vec<&[f64]> = expanded.iter().map(|b| b.as_slice()).collect();

        let weighted = friedman_test(&blocks, &["A", "B", "C"], Some(&w)).unwrap();
        let expanded = friedman_test(&exp_blocks, &["A", "B", "C"], None).unwrap();
        // Weighted n counts blocks (3); the chi-square and p-value must match
        // the frequency-expanded data (6 replicate rows).
        assert_eq!(weighted.n, 3);
        assert_eq!(expanded.n, 6);
        assert_abs_diff_eq!(weighted.chi_square, expanded.chi_square, epsilon = 1e-9);
        assert_abs_diff_eq!(weighted.p_value, expanded.p_value, epsilon = 1e-9);
    }

    /// Blocks with missing values are dropped list-wise.
    #[test]
    fn friedman_drops_incomplete_blocks() {
        let mat: Vec<Vec<f64>> = vec![
            vec![1.0, 2.0, 3.0],
            vec![f64::NAN, 2.0, 1.0],
            vec![3.0, 1.0, 2.0],
        ];
        let blocks: Vec<&[f64]> = mat.iter().map(|b| b.as_slice()).collect();
        let r = friedman_test(&blocks, &["A", "B", "C"], None).unwrap();
        assert_eq!(r.n, 2);
    }

    #[test]
    fn friedman_edge_cases() {
        // Fewer than 3 treatments.
        let two = [vec![1.0, 2.0]];
        let blocks: Vec<&[f64]> = two.iter().map(|b| b.as_slice()).collect();
        assert!(friedman_test(&blocks, &["A", "B"], None).is_err());
        // Fewer than 2 complete blocks.
        let one = [vec![f64::NAN, 2.0, 3.0]];
        let b1: Vec<&[f64]> = one.iter().map(|b| b.as_slice()).collect();
        assert!(friedman_test(&b1, &["A", "B", "C"], None).is_err());
        // Ragged block.
        let ragged = [vec![1.0, 2.0, 3.0], vec![1.0, 2.0]];
        let b2: Vec<&[f64]> = ragged.iter().map(|b| b.as_slice()).collect();
        assert!(friedman_test(&b2, &["A", "B", "C"], None).is_err());
        // Weight-length mismatch.
        let mat = [vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]];
        let b3: Vec<&[f64]> = mat.iter().map(|b| b.as_slice()).collect();
        assert!(friedman_test(&b3, &["A", "B", "C"], Some(&[1.0])).is_err());
        // Serde round-trip.
        let r = friedman_test(&b3, &["A", "B", "C"], None).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: FriedmanResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.chi_square, r.chi_square, epsilon = 1e-15);
    }
}
