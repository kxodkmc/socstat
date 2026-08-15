//! Hypothesis testing: Student's t, one-way ANOVA, chi-square, and
//! the Mann–Whitney U nonparametric test.
//!
//! All statistics are **weighted-aware** and treat weights as **frequency
//! weights** (case weights): each case counts as `weight` replicates.
//! Complex sampling weights (probability weights) are **not** supported in
//! this version. A case with a weight ≤ 0 (or a missing weight) is excluded.
//!
//! Numerical stability: weighted sums of squares are always computed with a
//! two-pass algorithm (weighted mean first, then squared deviations around the
//! mean), avoiding the catastrophic cancellation of `Σx² − (Σx)²/n`.
//!
//! Every public result struct derives `Serialize`/`Deserialize` so hosts can
//! ship results as JSON/FFI payloads (Hard Rule 1).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::data::ColumnData;
use crate::dist::{ChiSquaredDist, Distribution, FDist, NormalDist, StudentsTDist};
use crate::error::{SocStatError, SocStatResult};

// ---------------------------------------------------------------------------
// Result structs (Hard Rule 1: all serializable)
// ---------------------------------------------------------------------------

/// Summary statistics for a single group within a test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupSummary {
    /// Group label (value of the grouping variable).
    pub label: String,
    /// Effective sample size (sum of weights).
    pub n: f64,
    /// Weighted mean.
    pub mean: f64,
    /// Sample standard deviation (denominator n-1).
    pub std_dev: f64,
    /// Sample variance.
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    /// Standard error of the mean: std_dev / sqrt(n).
    pub std_error: f64,
}

/// Levene's test of equality of variances (based on group means).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeveneResult {
    pub f_statistic: f64,
    pub df1: f64,
    pub df2: f64,
    pub p_value: f64,
}

/// One t-test model (equal-variances pooled or Welch).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TTestModel {
    pub t_statistic: f64,
    pub df: f64,
    pub p_value: f64,
    /// Mean of group 1 minus mean of group 2.
    pub mean_difference: f64,
    /// Standard error of the difference.
    pub std_error: f64,
    /// 95% confidence interval for the mean difference.
    pub ci_95: (f64, f64),
}

/// Independent-samples t-test, reporting both the pooled and the
/// Welch (unequal-variances) models together with Levene's test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndependentTTest {
    pub group_stats: Vec<GroupSummary>,
    /// Levene's test of variance homogeneity.
    pub levene_test: LeveneResult,
    /// Model assuming equal variances (pooled).
    pub equal_variances: TTestModel,
    /// Welch's model assuming unequal variances.
    pub unequal_variances: TTestModel,
}

/// One row of an ANOVA table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Effect {
    pub ss: f64,
    pub df: f64,
    pub ms: f64,
    /// F statistic; `None` for the within/total rows where it is undefined.
    pub f: Option<f64>,
    /// p value; `None` where F is undefined.
    pub p_value: Option<f64>,
}

/// One-way (single-factor) ANOVA result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OneWayAnova {
    pub group_stats: Vec<GroupSummary>,
    pub between_groups: Effect,
    pub within_groups: Effect,
    pub total: Effect,
    pub f_statistic: f64,
    pub p_value: f64,
    /// Effect size η² = SS_between / SS_total.
    pub eta_squared: f64,
}

/// Pearson chi-square test of independence for two categorical variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChiSquareTest {
    pub row_labels: Vec<String>,
    pub col_labels: Vec<String>,
    /// Observed (weighted) cell counts [row][col].
    pub observed: Vec<Vec<f64>>,
    /// Expected counts under independence [row][col].
    pub expected: Vec<Vec<f64>>,
    pub row_totals: Vec<f64>,
    pub col_totals: Vec<f64>,
    /// Grand total (sum of weights).
    pub n: f64,
    pub chi_square: f64,
    pub df: f64,
    pub p_value: f64,
}

/// Rank summary for one group in the Mann–Whitney U test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RankSummary {
    pub label: String,
    /// Effective sample size (sum of weights).
    pub n: f64,
    /// Sum of ranks (each weighted case contributes its weight).
    pub rank_sum: f64,
    pub mean_rank: f64,
}

/// Mann–Whitney U test (asymptotic normal approximation with tie correction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MannWhitneyUTest {
    pub group_stats: Vec<RankSummary>,
    /// Reported U statistic (the smaller of U1, U2).
    pub u_statistic: f64,
    /// Standardized normal deviate.
    pub z_score: f64,
    pub p_value: f64,
    pub n1: f64,
    pub n2: f64,
    /// Whether ties were present (tie correction applied to the variance).
    pub has_ties: bool,
}

// ---------------------------------------------------------------------------
// Numerically stable weighted summaries
// ---------------------------------------------------------------------------

/// Weighted summary statistics computed with a two-pass algorithm.
///
/// Pass 1 computes the weighted mean; pass 2 accumulates weighted squared
/// deviations around the mean. This is numerically stable for data with a
/// large mean and a tiny variance (catastrophic cancellation avoided).
#[derive(Debug, Clone)]
struct WeightedSummary {
    /// Effective sample size (sum of weights).
    n: f64,
    mean: f64,
    /// Sum of squared deviations around the mean (weighted).
    sum_squares: f64,
    min: f64,
    max: f64,
}

impl WeightedSummary {
    /// Compute from (value, weight) pairs. Pairs with a non-positive weight
    /// or a non-finite value are excluded.
    fn compute(pairs: &[(f64, f64)]) -> SocStatResult<Self> {
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
    fn variance(&self) -> f64 {
        if self.n > 1.0 {
            self.sum_squares / (self.n - 1.0)
        } else {
            0.0
        }
    }
}

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// One group split out of the dependent variable.
#[derive(Debug, Clone)]
struct GroupedData {
    label: String,
    /// (value, weight) pairs, weight already validated > 0.
    pairs: Vec<(f64, f64)>,
}

fn format_num(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

/// True for a usable frequency weight: finite and strictly positive.
/// NaN weights must be excluded rather than silently treated as valid.
fn positive_weight(w: f64) -> bool {
    w.is_finite() && w > 0.0
}

/// Split a numeric dependent column into groups by a grouping column,
/// dropping rows where either value is missing or the weight is ≤ 0.
fn split_groups(
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

    let mut groups: Vec<GroupedData> = Vec::new();
    let mut index: BTreeMap<String, usize> = BTreeMap::new();

    for i in 0..n {
        let Some(x) = dep_slice[i] else { continue };
        let Some(label) = &group_labels[i] else { continue };
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        let idx = if let Some(&idx) = index.get(label) {
            idx
        } else {
            let idx = groups.len();
            index.insert(label.clone(), idx);
            groups.push(GroupedData { label: label.clone(), pairs: Vec::new() });
            idx
        };
        groups[idx].pairs.push((x, w));
    }

    Ok(groups)
}

fn group_summary(label: &str, ws: &WeightedSummary) -> GroupSummary {
    let variance = ws.variance();
    let std_dev = variance.sqrt();
    GroupSummary {
        label: label.to_string(),
        n: ws.n,
        mean: ws.mean,
        std_dev,
        variance,
        min: ws.min,
        max: ws.max,
        std_error: std_dev / ws.n.sqrt(),
    }
}

// ---------------------------------------------------------------------------
// Independent-samples t-test
// ---------------------------------------------------------------------------

/// Independent-samples t-test (Student + Welch) with Levene's test.
///
/// `dep` must be a numeric column; `group` splits cases into **exactly two**
/// groups. `weights` are optional frequency weights aligned by row index.
pub fn independent_t_test(
    dep: &ColumnData,
    group: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<IndependentTTest> {
    let groups = split_groups(dep, group, weights)?;
    if groups.len() != 2 {
        return Err(SocStatError::InsufficientData(format!(
            "independent t-test requires exactly two groups, found {}",
            groups.len()
        )));
    }

    let g1 = &groups[0];
    let g2 = &groups[1];
    let s1 = WeightedSummary::compute(&g1.pairs)?;
    let s2 = WeightedSummary::compute(&g2.pairs)?;

    if s1.n <= 1.0 || s2.n <= 1.0 {
        return Err(SocStatError::InsufficientData(
            "each t-test group needs at least two valid cases".into(),
        ));
    }

    let n1 = s1.n;
    let n2 = s2.n;
    let var1 = s1.variance();
    let var2 = s2.variance();
    let mean_diff = s1.mean - s2.mean;

    // Equal variances (pooled).
    let pooled_var = ((n1 - 1.0) * var1 + (n2 - 1.0) * var2) / (n1 + n2 - 2.0);
    let se_eq = (pooled_var * (1.0 / n1 + 1.0 / n2)).sqrt();
    // A zero (or non-finite) standard error means at least one group has zero
    // variance; the t statistic is NaN/Inf and the statrs p-value machinery
    // would panic. Return an error instead of crashing the host (BUG-003).
    if !(se_eq.is_finite() && se_eq > 0.0) {
        return Err(SocStatError::InsufficientData(
            "independent t-test is undefined: one or both groups have zero variance".into(),
        ));
    }
    let t_eq = mean_diff / se_eq;
    let df_eq = n1 + n2 - 2.0;
    let dist_eq = StudentsTDist::new(df_eq)?;
    let p_eq = two_sided_tail(&dist_eq, t_eq);
    let tcrit_eq = dist_eq.inverse_cdf(0.975);
    let ci_eq = (mean_diff - tcrit_eq * se_eq, mean_diff + tcrit_eq * se_eq);

    // Unequal variances (Welch–Satterthwaite).
    let se_uw = (var1 / n1 + var2 / n2).sqrt();
    let t_uw = mean_diff / se_uw;
    let df_uw = (var1 / n1 + var2 / n2).powi(2)
        / ((var1 / n1).powi(2) / (n1 - 1.0) + (var2 / n2).powi(2) / (n2 - 1.0));
    let (p_uw, ci_uw) = if df_uw.is_finite() && df_uw > 0.0 {
        let dist = StudentsTDist::new(df_uw)?;
        let p = two_sided_tail(&dist, t_uw);
        let tc = dist.inverse_cdf(0.975);
        (p, (mean_diff - tc * se_uw, mean_diff + tc * se_uw))
    } else {
        (f64::NAN, (f64::NAN, f64::NAN))
    };

    Ok(IndependentTTest {
        group_stats: vec![group_summary(&g1.label, &s1), group_summary(&g2.label, &s2)],
        levene_test: levene_based_on_mean(&groups)?,
        equal_variances: TTestModel {
            t_statistic: t_eq,
            df: df_eq,
            p_value: p_eq,
            mean_difference: mean_diff,
            std_error: se_eq,
            ci_95: ci_eq,
        },
        unequal_variances: TTestModel {
            t_statistic: t_uw,
            df: df_uw,
            p_value: p_uw,
            mean_difference: mean_diff,
            std_error: se_uw,
            ci_95: ci_uw,
        },
    })
}

/// Levene's test based on group means: ANOVA on |x − mean_group|.
fn levene_based_on_mean(groups: &[GroupedData]) -> SocStatResult<LeveneResult> {
    let mut abs_dev: Vec<Vec<(f64, f64)>> = Vec::with_capacity(groups.len());
    let mut pooled: Vec<(f64, f64)> = Vec::new();

    for g in groups {
        let ws = WeightedSummary::compute(&g.pairs)?;
        let zd: Vec<(f64, f64)> = g.pairs
            .iter()
            .filter(|(x, w)| x.is_finite() && *w > 0.0)
            .map(|(x, w)| ((x - ws.mean).abs(), *w))
            .collect();
        pooled.extend_from_slice(&zd);
        abs_dev.push(zd);
    }

    let total = WeightedSummary::compute(&pooled)?;
    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for zd in &abs_dev {
        let zg = WeightedSummary::compute(zd)?;
        ss_between += zg.n * (zg.mean - total.mean).powi(2);
        ss_within += zg.sum_squares;
    }

    let k = groups.len() as f64;
    let df1 = k - 1.0;
    let df2 = total.n - k;
    if df1 <= 0.0 || df2 <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "Levene's test needs at least two groups with more cases than groups".into(),
        ));
    }

    let f = if ss_within > 0.0 {
        (ss_between / df1) / (ss_within / df2)
    } else if ss_between > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let p_value = if f.is_nan() {
        f64::NAN
    } else {
        1.0 - FDist::new(df1, df2)?.cdf(f)
    };

    Ok(LeveneResult { f_statistic: f, df1, df2, p_value })
}

// ---------------------------------------------------------------------------
// One-way ANOVA
// ---------------------------------------------------------------------------

/// One-way ANOVA of `dep` (numeric) across the groups of `factor`.
pub fn one_way_anova(
    dep: &ColumnData,
    factor: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<OneWayAnova> {
    let groups = split_groups(dep, factor, weights)?;
    if groups.len() < 2 {
        return Err(SocStatError::InsufficientData(format!(
            "one-way ANOVA requires at least two groups, found {}",
            groups.len()
        )));
    }

    let summaries: Vec<WeightedSummary> = groups
        .iter()
        .map(|g| WeightedSummary::compute(&g.pairs))
        .collect::<SocStatResult<_>>()?;

    let all_pairs: Vec<(f64, f64)> = groups.iter().flat_map(|g| g.pairs.iter().copied()).collect();
    let total = WeightedSummary::compute(&all_pairs)?;
    let grand_mean = total.mean;

    let mut ss_between = 0.0;
    let mut ss_within = 0.0;
    for ws in &summaries {
        ss_between += ws.n * (ws.mean - grand_mean).powi(2);
        ss_within += ws.sum_squares;
    }

    let k = groups.len() as f64;
    let n_total = total.n;
    let df_between = k - 1.0;
    let df_within = n_total - k;
    if df_within <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "ANOVA needs more valid cases than groups".into(),
        ));
    }

    let ms_between = ss_between / df_between;
    let ms_within = ss_within / df_within;
    let f = if ss_within > 0.0 {
        ms_between / ms_within
    } else if ss_between > 0.0 {
        f64::INFINITY
    } else {
        f64::NAN
    };
    let p_value = if f.is_nan() {
        f64::NAN
    } else {
        1.0 - FDist::new(df_between, df_within)?.cdf(f)
    };
    let eta_squared = if ss_between + ss_within > 0.0 {
        ss_between / (ss_between + ss_within)
    } else {
        0.0
    };

    Ok(OneWayAnova {
        group_stats: groups
            .iter()
            .zip(&summaries)
            .map(|(g, s)| group_summary(&g.label, s))
            .collect(),
        between_groups: Effect {
            ss: ss_between,
            df: df_between,
            ms: ms_between,
            f: if f.is_finite() { Some(f) } else { None },
            p_value: if f.is_finite() { Some(p_value) } else { None },
        },
        within_groups: Effect {
            ss: ss_within,
            df: df_within,
            ms: ms_within,
            f: None,
            p_value: None,
        },
        total: Effect {
            ss: total.sum_squares,
            df: n_total - 1.0,
            ms: total.sum_squares / (n_total - 1.0),
            f: None,
            p_value: None,
        },
        f_statistic: f,
        p_value,
        eta_squared,
    })
}

// ---------------------------------------------------------------------------
// Chi-square test of independence
// ---------------------------------------------------------------------------

/// Pearson chi-square test of independence between two categorical columns.
///
/// Both columns may be numeric or text; numeric values are binned by their
/// display value. Observed and expected counts are weight-aware.
pub fn chi_square_test(
    var1: &ColumnData,
    var2: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<ChiSquareTest> {
    let n = var1.len();
    if n != var2.len() {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: n,
            got: var2.len(),
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

    let v1 = extract_labels(var1);
    let v2 = extract_labels(var2);

    let mut row_index: BTreeMap<String, usize> = BTreeMap::new();
    let mut col_index: BTreeMap<String, usize> = BTreeMap::new();
    let mut counts: Vec<Vec<f64>> = Vec::new();
    let mut row_totals: Vec<f64> = Vec::new();
    let mut col_totals: Vec<f64> = Vec::new();
    let mut grand_total = 0.0f64;

    for i in 0..n {
        let (Some(a), Some(b)) = (&v1[i], &v2[i]) else { continue };
        let w = weights.map(|ws| ws[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        let ri = if let Some(&ri) = row_index.get(a) {
            ri
        } else {
            let ri = row_index.len();
            row_index.insert(a.clone(), ri);
            counts.push(vec![0.0; col_index.len()]);
            row_totals.push(0.0);
            ri
        };
        let ci = if let Some(&ci) = col_index.get(b) {
            ci
        } else {
            let ci = col_index.len();
            col_index.insert(b.clone(), ci);
            for row in counts.iter_mut() {
                row.push(0.0);
            }
            col_totals.push(0.0);
            ci
        };
        counts[ri][ci] += w;
        row_totals[ri] += w;
        col_totals[ci] += w;
        grand_total += w;
    }

    if grand_total <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "chi-square test needs at least one valid pair of cases".into(),
        ));
    }
    let (n_rows, n_cols) = (counts.len(), if counts.is_empty() { 0 } else { counts[0].len() });
    if n_rows < 2 || n_cols < 2 {
        return Err(SocStatError::InsufficientData(
            "chi-square test requires at least two categories per variable".into(),
        ));
    }

    let mut expected = vec![vec![0.0f64; n_cols]; n_rows];
    let mut chi_square = 0.0f64;
    for (i, row) in counts.iter().enumerate() {
        for (j, &obs) in row.iter().enumerate() {
            let e = row_totals[i] * col_totals[j] / grand_total;
            expected[i][j] = e;
            if e > 0.0 {
                chi_square += (obs - e).powi(2) / e;
            } else if obs > 0.0 {
                return Err(SocStatError::Computation(
                    "zero expected count for a non-empty cell".into(),
                ));
            }
        }
    }

    let df = (n_rows - 1) as f64 * (n_cols - 1) as f64;
    let p_value = 1.0 - ChiSquaredDist::new(df)?.cdf(chi_square);

    Ok(ChiSquareTest {
        row_labels: row_index.keys().cloned().collect(),
        col_labels: col_index.keys().cloned().collect(),
        observed: counts,
        expected,
        row_totals,
        col_totals,
        n: grand_total,
        chi_square,
        df,
        p_value,
    })
}

fn extract_labels(col: &ColumnData) -> Vec<Option<String>> {
    match col {
        ColumnData::Numeric(v) => v.iter().map(|o| o.map(format_num)).collect(),
        ColumnData::Text(v) => v.clone(),
    }
}

// ---------------------------------------------------------------------------
// Mann–Whitney U test
// ---------------------------------------------------------------------------

/// Mann–Whitney U test between two groups.
///
/// Uses the asymptotic normal approximation with the standard tie correction
/// (this matches SPSS's asymptotic significance). Does **not** report exact
/// p-values, so results for tiny samples may differ from R's `wilcox.test`
/// which defaults to an exact test.
pub fn mann_whitney_u_test(
    dep: &ColumnData,
    group: &ColumnData,
    weights: Option<&[f64]>,
) -> SocStatResult<MannWhitneyUTest> {
    let groups = split_groups(dep, group, weights)?;
    if groups.len() != 2 {
        return Err(SocStatError::InsufficientData(format!(
            "Mann–Whitney U requires exactly two groups, found {}",
            groups.len()
        )));
    }

    let n1: f64 = groups[0].pairs.iter().map(|(_, w)| w).sum();
    let n2: f64 = groups[1].pairs.iter().map(|(_, w)| w).sum();
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err(SocStatError::InsufficientData(
            "each Mann–Whitney group needs at least one valid case".into(),
        ));
    }

    // Pool (value, weight, group) and sort by value.
    let mut pool: Vec<(f64, f64, usize)> = Vec::new();
    for (gi, g) in groups.iter().enumerate() {
        for &(x, w) in &g.pairs {
            pool.push((x, w, gi));
        }
    }
    pool.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // Average-rank ties, treating each weighted case as `weight` replicates.
    let mut cum = 0.0f64;
    let mut rank_sums = [0.0f64, 0.0f64];
    let mut has_ties = false;
    let mut tie_weights: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < pool.len() {
        let mut j = i;
        let mut block_w = 0.0;
        while j < pool.len() && pool[j].0 == pool[i].0 {
            block_w += pool[j].1;
            j += 1;
        }
        if j - i > 1 {
            has_ties = true;
            tie_weights.push(block_w);
        }
        let avg_rank = cum + (block_w + 1.0) / 2.0;
        for k in i..j {
            rank_sums[pool[k].2] += pool[k].1 * avg_rank;
        }
        cum += block_w;
        i = j;
    }

    let n_total = n1 + n2;
    let u1 = rank_sums[0] - n1 * (n1 + 1.0) / 2.0;
    let u2 = n1 * n2 - u1;
    let u_statistic = u1.min(u2);

    let mean_u = n1 * n2 / 2.0;
    let var_u = if has_ties && n_total > 1.0 {
        let tie_adj: f64 = tie_weights.iter().map(|t| t.powi(3) - t).sum();
        (n1 * n2 / (n_total * (n_total - 1.0)))
            * ((n_total.powi(3) - n_total) - tie_adj)
            / 12.0
    } else {
        n1 * n2 * (n_total + 1.0) / 12.0
    };
    if var_u <= 0.0 {
        return Err(SocStatError::Computation(
            "Mann–Whitney U variance is zero; cannot compute the test".into(),
        ));
    }

    let z_score = (u_statistic - mean_u) / var_u.sqrt();
    let normal = NormalDist::standard();
    let p_value = 2.0 * (1.0 - normal.cdf(z_score.abs()));

    Ok(MannWhitneyUTest {
        group_stats: vec![
            RankSummary {
                label: groups[0].label.clone(),
                n: n1,
                rank_sum: rank_sums[0],
                mean_rank: rank_sums[0] / n1,
            },
            RankSummary {
                label: groups[1].label.clone(),
                n: n2,
                rank_sum: rank_sums[1],
                mean_rank: rank_sums[1] / n2,
            },
        ],
        u_statistic,
        z_score,
        p_value,
        n1,
        n2,
        has_ties,
    })
}

/// Two-sided tail probability for a symmetric distribution.
fn two_sided_tail(dist: &impl Distribution, stat: f64) -> f64 {
    2.0 * (1.0 - dist.cdf(stat.abs()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod unit {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn num_col(values: &[Option<f64>]) -> ColumnData {
        ColumnData::Numeric(values.to_vec())
    }

    fn text_col(values: &[Option<&str>]) -> ColumnData {
        ColumnData::Text(values.iter().map(|o| o.map(|s| s.to_string())).collect())
    }

    // ---- WeightedSummary: numerical stability ----

    #[test]
    fn weighted_summary_large_mean_small_variance() {
        // Classic catastrophic-cancellation case for the one-pass formula
        // Σx² − (Σx)²/n: mean 1e6, variance must stay small and non-negative.
        let pairs = vec![
            (1_000_000.1, 1.0),
            (1_000_000.2, 1.0),
        ];
        let ws = WeightedSummary::compute(&pairs).unwrap();
        assert_abs_diff_eq!(ws.mean, 1_000_000.15, epsilon = 1e-9);
        let var = ws.variance();
        assert!(var >= 0.0, "variance must not be negative, got {var}");
        assert_abs_diff_eq!(var, 0.005, epsilon = 1e-6);
    }

    #[test]
    fn weighted_summary_matches_hand_computed() {
        // weights [1,2,3] over values [1,2,3]
        let pairs = vec![(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)];
        let ws = WeightedSummary::compute(&pairs).unwrap();
        assert_abs_diff_eq!(ws.n, 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(ws.mean, 14.0 / 6.0, epsilon = 1e-12);
        // mean = 7/3; ss = 1*(1-7/3)^2 + 2*(2-7/3)^2 + 3*(3-7/3)^2
        let expected_ss = (4.0f64 / 3.0).powi(2)
            + 2.0 * (1.0f64 / 3.0).powi(2)
            + 3.0 * (2.0f64 / 3.0).powi(2);
        assert_abs_diff_eq!(ws.sum_squares, expected_ss, epsilon = 1e-12);
    }

    #[test]
    fn weighted_summary_rejects_empty() {
        assert!(WeightedSummary::compute(&[]).is_err());
        assert!(WeightedSummary::compute(&[(1.0, 0.0)]).is_err());
    }

    // ---- Independent t-test ----

    #[test]
    fn ttest_matches_hand_computed() {
        let dep = num_col(&[Some(5.0), Some(6.0), Some(7.0), Some(8.0),
                            Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let grp = num_col(&[Some(1.0); 4].into_iter().chain(vec![Some(2.0); 4]).collect::<Vec<_>>());
        let r = independent_t_test(&dep, &grp, None).unwrap();

        assert_eq!(r.group_stats.len(), 2);
        let g1 = &r.group_stats[0];
        assert_eq!(g1.label, "1");
        assert_abs_diff_eq!(g1.n, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(g1.mean, 6.5, epsilon = 1e-12);
        assert_abs_diff_eq!(g1.variance, 5.0 / 3.0, epsilon = 1e-12);
        let g2 = &r.group_stats[1];
        assert_eq!(g2.label, "2");
        assert_abs_diff_eq!(g2.mean, 2.5, epsilon = 1e-12);

        // Pooled model
        let t = &r.equal_variances;
        assert_abs_diff_eq!(t.mean_difference, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.df, 6.0, epsilon = 1e-12);
        assert_abs_diff_eq!(t.std_error, 0.912_870_929_175_276_9, epsilon = 1e-12);
        assert_abs_diff_eq!(t.t_statistic, 4.381_780_460_041_329, epsilon = 1e-9);
        // Reference (R t.test): two-sided p = 0.004659215
        assert_abs_diff_eq!(t.p_value, 0.004_659_214_943_9, epsilon = 1e-6);
        // 95% CI: 4 ± t(0.975,6) * se ; t(0.975,6) = 2.446911851144969
        let tcrit = 2.446_911_851_144_969;
        assert_abs_diff_eq!(t.ci_95.0, 4.0 - tcrit * t.std_error, epsilon = 1e-9);
        assert_abs_diff_eq!(t.ci_95.1, 4.0 + tcrit * t.std_error, epsilon = 1e-9);

        // Welch model: for equal-sized groups with equal variance it coincides
        assert_abs_diff_eq!(r.unequal_variances.df, 6.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r.unequal_variances.t_statistic, t.t_statistic, epsilon = 1e-9);
    }

    #[test]
    fn ttest_rejects_non_two_groups() {
        let dep = num_col(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let grp = num_col(&[Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
        assert!(independent_t_test(&dep, &grp, None).is_ok());
        let grp3 = num_col(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        assert!(independent_t_test(&dep, &grp3, None).is_err());
    }

    #[test]
    fn ttest_zero_variance_errors_not_panics() {
        // BUG-003: identical groups (zero variance) must return an error, not
        // panic inside statrs.
        let dep = num_col(&[Some(10.0); 10]);
        let grp_vals: Vec<Option<f64>> = [Some(1.0); 5].into_iter()
            .chain(vec![Some(2.0); 5])
            .collect();
        let grp = num_col(&grp_vals);
        assert!(independent_t_test(&dep, &grp, None).is_err());
    }

    #[test]
    fn ttest_weights_are_frequency() {
        // Duplicating each case with a weight of 2 must equal the unweighted
        // result on the doubled dataset.
        let dep_a = num_col(&[Some(1.0), Some(2.0), Some(3.0),
                              Some(5.0), Some(6.0), Some(7.0)]);
        let grp_a = num_col(&[Some(1.0), Some(1.0), Some(1.0),
                              Some(2.0), Some(2.0), Some(2.0)]);
        let w = vec![2.0, 2.0, 2.0, 2.0, 2.0, 2.0];
        let r_w = independent_t_test(&dep_a, &grp_a, Some(&w)).unwrap();

        let dep_b = num_col(&[
            Some(1.0), Some(1.0), Some(2.0), Some(2.0), Some(3.0), Some(3.0),
            Some(5.0), Some(5.0), Some(6.0), Some(6.0), Some(7.0), Some(7.0),
        ]);
        let grp_b = num_col(&[
            Some(1.0); 6].into_iter().chain(vec![Some(2.0); 6]).collect::<Vec<_>>());
        let r_u = independent_t_test(&dep_b, &grp_b, None).unwrap();

        assert_abs_diff_eq!(r_w.equal_variances.t_statistic,
                            r_u.equal_variances.t_statistic, epsilon = 1e-9);
        assert_abs_diff_eq!(r_w.equal_variances.df, r_u.equal_variances.df, epsilon = 1e-9);
        assert_abs_diff_eq!(r_w.levene_test.f_statistic,
                            r_u.levene_test.f_statistic, epsilon = 1e-9);
    }

    // ---- ANOVA ----

    #[test]
    fn anova_matches_hand_computed() {
        // Group A: [4,6,8], Group B: [1,3,5]
        let dep = num_col(&[Some(4.0), Some(6.0), Some(8.0),
                            Some(1.0), Some(3.0), Some(5.0)]);
        let fct = num_col(&[Some(1.0), Some(1.0), Some(1.0),
                            Some(2.0), Some(2.0), Some(2.0)]);
        let r = one_way_anova(&dep, &fct, None).unwrap();

        // Group means: A=6, B=3. Grand mean = 4.5
        // SS_between = 3*(6-4.5)^2 + 3*(3-4.5)^2 = 6.75 + 6.75 = 13.5
        assert_abs_diff_eq!(r.between_groups.ss, 13.5, epsilon = 1e-12);
        assert_abs_diff_eq!(r.between_groups.df, 1.0, epsilon = 1e-12);
        // SS_within = (4+0+4) + (4+0+4) = 16
        assert_abs_diff_eq!(r.within_groups.ss, 16.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.within_groups.df, 4.0, epsilon = 1e-12);
        // MS_between = 13.5, MS_within = 4.0, F = 3.375
        assert_abs_diff_eq!(r.f_statistic, 3.375, epsilon = 1e-12);
        // Reference (R aov): pf(3.375, 1, 4, lower=FALSE) = 0.140065985
        assert_abs_diff_eq!(r.p_value, 0.140_065_984_912, epsilon = 1e-6);
        // Total SS = Σ(x − 4.5)² = 29.5
        assert_abs_diff_eq!(r.total.ss, 29.5, epsilon = 1e-12);
        // η² = 13.5 / 29.5 = 0.457627…
        assert_abs_diff_eq!(r.eta_squared, 13.5 / 29.5, epsilon = 1e-12);
    }

    #[test]
    fn anova_eta_squared_range() {
        let dep = num_col(&[Some(1.0), Some(2.0), Some(10.0), Some(20.0)]);
        let fct = num_col(&[Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
        let r = one_way_anova(&dep, &fct, None).unwrap();
        assert!((0.0..=1.0).contains(&r.eta_squared));
    }

    // ---- Chi-square ----

    #[test]
    fn chi_square_matches_hand_computed() {
        // 2x2 with each cell expected = 15, observed [10 20; 20 10]
        let v1 = text_col(&[Some("A"), Some("A"), Some("B"), Some("B")]);
        let v2 = text_col(&[
            Some("X"), Some("Y"), Some("X"), Some("Y"),
        ]);
        // To get counts [10,20;20,10] use weights.
        let w = vec![10.0, 20.0, 20.0, 10.0];
        let r = chi_square_test(&v1, &v2, Some(&w)).unwrap();

        assert_abs_diff_eq!(r.n, 60.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.row_totals[0], 30.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.col_totals[0], 30.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.expected[0][0], 15.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.chi_square, 100.0 / 15.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.df, 1.0, epsilon = 1e-12);
        // Reference (R chisq.test): p = 0.009823275
        assert_abs_diff_eq!(r.p_value, 0.009_823_274_508, epsilon = 1e-6);
    }

    #[test]
    fn chi_square_requires_two_categories() {
        let v1 = text_col(&[Some("A"), Some("A"), Some("A")]);
        let v2 = text_col(&[Some("X"), Some("Y"), Some("X")]);
        assert!(chi_square_test(&v1, &v2, None).is_err());
    }

    // ---- Mann–Whitney U ----

    #[test]
    fn mann_whitney_no_ties() {
        let dep = num_col(&[Some(1.0), Some(2.0), Some(3.0),
                            Some(4.0), Some(5.0), Some(6.0)]);
        let grp = num_col(&[Some(1.0), Some(1.0), Some(1.0),
                            Some(2.0), Some(2.0), Some(2.0)]);
        let r = mann_whitney_u_test(&dep, &grp, None).unwrap();

        // Ranks of group 1 are 1,2,3 → R1=6 → U1 = 6 - 6 = 0; U2 = 9.
        assert_abs_diff_eq!(r.u_statistic, 0.0, epsilon = 1e-12);
        assert!(!r.has_ties);
        assert_abs_diff_eq!(r.n1, 3.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.n2, 3.0, epsilon = 1e-12);
        // var = 3*3*7/12 = 5.25; mean = 4.5; z = (0-4.5)/sqrt(5.25)
        let expected_z = -4.5 / 5.25_f64.sqrt();
        assert_abs_diff_eq!(r.z_score, expected_z, epsilon = 1e-12);
        // Reference (R wilcox.test(1:3, 4:6, correct=FALSE)): p = 0.049534613
        assert_abs_diff_eq!(r.p_value, 0.049_534_613_436, epsilon = 1e-6);
    }

    #[test]
    fn mann_whitney_with_ties() {
        let dep = num_col(&[Some(1.0), Some(2.0), Some(2.0),
                            Some(2.0), Some(3.0), Some(4.0)]);
        let grp = num_col(&[Some(1.0), Some(1.0), Some(1.0),
                            Some(2.0), Some(2.0), Some(2.0)]);
        let r = mann_whitney_u_test(&dep, &grp, None).unwrap();
        assert!(r.has_ties);
        // Tie correction must produce a finite, non-NaN result.
        assert!(r.z_score.is_finite());
        assert!(r.p_value.is_finite());
    }

    // ---- Serde round-trips (Hard Rule 1) ----

    #[test]
    fn serde_round_trip_all_results() {
        let dep = num_col(&[Some(5.0), Some(6.0), Some(7.0), Some(8.0),
                            Some(1.0), Some(2.0), Some(3.0), Some(4.0)]);
        let grp = num_col(&[Some(1.0); 4].into_iter().chain(vec![Some(2.0); 4]).collect::<Vec<_>>());
        let ttest = independent_t_test(&dep, &grp, None).unwrap();
        let json = serde_json::to_string(&ttest).unwrap();
        let back: IndependentTTest = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.equal_variances.t_statistic,
                            ttest.equal_variances.t_statistic, epsilon = 1e-15);

        let fct = num_col(&[Some(1.0), Some(1.0), Some(2.0), Some(2.0),
                            Some(1.0), Some(1.0), Some(2.0), Some(2.0)]);
        let anova = one_way_anova(&dep, &fct, None).unwrap();
        let json = serde_json::to_string(&anova).unwrap();
        let back: OneWayAnova = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.f_statistic, anova.f_statistic, epsilon = 1e-15);

        let ct = chi_square_test(&grp, &fct, None).unwrap();
        let json = serde_json::to_string(&ct).unwrap();
        let back: ChiSquareTest = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.chi_square, ct.chi_square, epsilon = 1e-15);

        let mwu = mann_whitney_u_test(&dep, &grp, None).unwrap();
        let json = serde_json::to_string(&mwu).unwrap();
        let back: MannWhitneyUTest = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.u_statistic, mwu.u_statistic, epsilon = 1e-15);
    }

    // ---- Extreme-value checks on the p-value plumbing ----

    #[test]
    fn p_value_edge_cases() {
        // t = 0 → p = 1
        let dist = StudentsTDist::new(5.0).unwrap();
        assert_abs_diff_eq!(two_sided_tail(&dist, 0.0), 1.0, epsilon = 1e-12);
        // very large t → p → 0
        assert!(two_sided_tail(&dist, 1e10) < 1e-15);
        // df = 1 (Cauchy) still well-behaved
        let dist1 = StudentsTDist::new(1.0).unwrap();
        assert!(two_sided_tail(&dist1, 0.0) == 1.0);
    }
}
