//! ANOVA post-hoc comparisons: Bonferroni, Tukey HSD (Tukey–Kramer), Scheffé,
//! and Games–Howell. The first three work on a set of groups and the pooled
//! within-group variance (`MS_within`) from a one-way ANOVA; Games–Howell
//! instead uses per-pair variances and Welch–Satterthwaite degrees of freedom,
//! so it applies when group variances are unequal (the standard replacement
//! when Tukey's homogeneity assumption fails).
//!
//! p-values and confidence intervals are adjusted for multiple comparisons;
//! the statistics differ by method (Bonferroni→t, Tukey→q, Scheffé→F,
//! Games–Howell→q). Every public result type derives `Serialize`/`Deserialize`
//! (Hard Rule 1).

use serde::{Deserialize, Serialize};

use crate::dist::{Distribution, FDist, StudentsTDist, ptukey};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{GroupedData, WeightedSummary, two_sided_tail};

/// The post-hoc method to apply.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostHocMethod {
    /// Pairwise t-tests with a Bonferroni multiple-comparison correction.
    Bonferroni,
    /// Tukey's Honestly Significant Difference (Tukey–Kramer for unequal n).
    Tukey,
    /// Scheffé's test for all contrasts.
    Scheffe,
    /// Games–Howell: pairwise comparisons with per-pair variances and
    /// Welch–Satterthwaite df; robust to unequal group variances.
    GamesHowell,
}

/// One pairwise comparison from a post-hoc test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostHocComparison {
    pub group1: String,
    pub group2: String,
    /// `x̄(group2) − x̄(group1)` (signed difference of means).
    pub mean_difference: f64,
    pub std_error: f64,
    /// Test statistic: Bonferroni→t, Tukey→q, Scheffé→F, Games–Howell→q.
    pub statistic: f64,
    /// Multiple-comparison adjusted p-value.
    pub p_value: f64,
    /// Adjusted 95% confidence interval for the mean difference.
    pub ci_95: (f64, f64),
    /// Per-pair degrees of freedom. Set only by [`PostHocMethod::GamesHowell`]
    /// (Welch–Satterthwaite); `None` for pooled-variance methods, whose error
    /// degrees of freedom are the result's `df_within`.
    #[serde(default)]
    pub df: Option<f64>,
}

/// Result of a post-hoc comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PostHocResult {
    pub method: PostHocMethod,
    pub comparisons: Vec<PostHocComparison>,
    pub n_groups: usize,
    /// Pooled within-group variance used as the error term.
    pub ms_within: f64,
    pub df_within: f64,
}

/// Run a post-hoc comparison over pre-split groups.
///
/// `groups` are the split groups (labels plus weighted `(value, weight)`
/// pairs), and `ms_within`/`df_within` are the error terms from a one-way
/// ANOVA. Requires at least two groups and `df_within ≥ 1`. The public entry
/// point for datasets is [`StatsExt::post_hoc`](crate::stats::StatsExt::post_hoc).
pub(crate) fn post_hoc(
    groups: &[GroupedData],
    ms_within: f64,
    df_within: f64,
    method: PostHocMethod,
) -> SocStatResult<PostHocResult> {
    let m = groups.len();
    if m < 2 {
        return Err(SocStatError::InsufficientData(
            "post-hoc tests need at least two groups".into(),
        ));
    }
    if df_within < 1.0 || !(ms_within.is_finite() && ms_within > 0.0) {
        return Err(SocStatError::InsufficientData(
            "post-hoc tests need a positive within-group variance (df ≥ 1)".into(),
        ));
    }

    let summaries: Vec<WeightedSummary> = groups
        .iter()
        .map(|g| WeightedSummary::compute(&g.pairs))
        .collect::<SocStatResult<_>>()?;

    // Tukey's critical value depends only on (m, df_within): compute it once
    // for all pairs instead of once per pair.
    let tukey_q_crit = (method == PostHocMethod::Tukey)
        .then(|| inverse_ptukey(0.95, m, df_within));

    let k_comp = m * (m - 1) / 2;

    let mut comparisons = Vec::with_capacity(k_comp);
    for i in 0..m {
        let ni = summaries[i].n;
        for j in (i + 1)..m {
            let nj = summaries[j].n;
            if ni <= 0.0 || nj <= 0.0 {
                return Err(SocStatError::InsufficientData(
                    "post-hoc tests need at least one case per group".into(),
                ));
            }
            let diff = summaries[j].mean - summaries[i].mean;
            let adiff = diff.abs();

            let (statistic, std_error, p_value, ci_95, pair_df) = match method {
                PostHocMethod::Bonferroni => {
                    let se = (ms_within * (1.0 / ni + 1.0 / nj)).sqrt();
                    let t = adiff / se;
                    let dist = StudentsTDist::new(df_within)?;
                    let p_raw = two_sided_tail(&dist, t);
                    let p = (p_raw * k_comp as f64).min(1.0);
                    // Bonferroni-adjusted critical value for the CI.
                    let crit = dist.inverse_cdf(1.0 - 0.025 / k_comp as f64);
                    (t, se, p, (diff - crit * se, diff + crit * se), None)
                }
                PostHocMethod::Tukey => {
                    // Tukey–Kramer: se = √(MS_within/2 · (1/nᵢ + 1/nⱼ)).
                    let se = (ms_within / 2.0 * (1.0 / ni + 1.0 / nj)).sqrt();
                    let q = adiff / se;
                    let p = 1.0 - ptukey(q, m, df_within);
                    let q_crit = tukey_q_crit.unwrap();
                    (q, se, p, (diff - q_crit * se, diff + q_crit * se), None)
                }
                PostHocMethod::Scheffe => {
                    let se = (ms_within * (1.0 / ni + 1.0 / nj)).sqrt();
                    // For any contrast, Scheffé's F = t²/(k−1) ~ F(k−1, df_within).
                    let f = diff * diff / (se * se) / (m - 1) as f64;
                    let dist = FDist::new((m - 1) as f64, df_within)?;
                    let p = 1.0 - dist.cdf(f);
                    // Scheffé critical value S* = √((m−1)·F(0.95, m−1, df)).
                    let sc = ((m - 1) as f64 * dist.inverse_cdf(0.95)).sqrt();
                    (f, se, p, (diff - sc * se, diff + sc * se), None)
                }
                PostHocMethod::GamesHowell => {
                    if ni < 2.0 || nj < 2.0 {
                        return Err(SocStatError::InsufficientData(
                            "Games–Howell needs at least two cases per group".into(),
                        ));
                    }
                    // Games–Howell: per-pair variances, no pooled MS required
                    // for the test itself (ms/df stay as reported references).
                    let vi = summaries[i].variance() / ni;
                    let vj = summaries[j].variance() / nj;
                    let se = (0.5 * (vi + vj)).sqrt();
                    if !(se.is_finite() && se > 0.0) {
                        return Err(SocStatError::InsufficientData(
                            "Games–Howell needs non-zero variance in each group pair".into(),
                        ));
                    }
                    let q = adiff / se;
                    let df_pair = (vi + vj).powi(2) / (vi * vi / (ni - 1.0) + vj * vj / (nj - 1.0));
                    let p = 1.0 - ptukey(q, m, df_pair);
                    let q_crit = inverse_ptukey(0.95, m, df_pair);
                    (q, se, p, (diff - q_crit * se, diff + q_crit * se), Some(df_pair))
                }
            };

            comparisons.push(PostHocComparison {
                group1: groups[i].label.clone(),
                group2: groups[j].label.clone(),
                mean_difference: diff,
                std_error,
                statistic,
                p_value,
                ci_95,
                df: pair_df,
            });
        }
    }

    Ok(PostHocResult { method, comparisons, n_groups: m, ms_within, df_within })
}

/// Inverse studentized-range CDF (Illinois method): the `q` with
/// `ptukey(q) = p`. The secant steps converge in ~10 evaluations, versus
/// ~60 for plain bisection, which matters because each `ptukey` evaluation
/// is a nested quadrature.
fn inverse_ptukey(p: f64, k: usize, df: f64) -> f64 {
    let target = p.clamp(0.0, 1.0);
    if target <= 0.0 {
        return 0.0;
    }
    // Bracket the root: ptukey(0) = 0 < target; expand hi until it covers.
    let (mut lo, mut flo) = (0.0, 0.0);
    let mut hi = 4.0;
    let mut fhi = ptukey(hi, k, df);
    while fhi < target {
        (lo, flo) = (hi, fhi);
        hi *= 2.0;
        fhi = ptukey(hi, k, df);
        if !fhi.is_finite() {
            return hi; // degenerate df; the caller's p-value is NaN anyway
        }
    }
    for _ in 0..48 {
        if !(fhi - flo).is_finite() || fhi - flo <= 0.0 {
            break;
        }
        // Secant step inside the bracket.
        let m = lo + (target - flo) / (fhi - flo) * (hi - lo);
        if !(m.is_finite() && m > lo && m < hi) {
            // Secant step left the bracket (flat or super-steep CDF) → bisect.
            let mid = 0.5 * (lo + hi);
            let fmid = ptukey(mid, k, df);
            if fmid < target { (lo, flo) = (mid, fmid); } else { (hi, fhi) = (mid, fmid); }
            continue;
        }
        let fm = ptukey(m, k, df);
        if (fm - target).abs() <= 1e-12 {
            return m;
        }
        if fm < target {
            (lo, flo) = (m, fm);
            fhi *= 0.5; // Illinois: deflate the retained endpoint
        } else {
            (hi, fhi) = (m, fm);
            flo *= 0.5;
        }
    }
    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use crate::stats::shared::GroupedData;

    fn group(label: &str, values: &[f64]) -> GroupedData {
        GroupedData { label: label.into(), pairs: values.iter().map(|&v| (v, 1.0)).collect() }
    }

    // Balanced one-way ANOVA data: groups with clearly distinct means.
    // Group A: 1..6, B: 11..16, C: 21..26. MS_within = pooled variance.
    fn balanced_groups() -> Vec<GroupedData> {
        vec![
            group("A", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            group("B", &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0]),
            group("C", &[21.0, 22.0, 23.0, 24.0, 25.0, 26.0]),
        ]
    }

    /// MS_within for balanced data where each group has identical spread:
    /// values step by 1, so the within-group sample variance is 3.5 each.
    fn ms_within_of(balanced: &[GroupedData]) -> f64 {
        let var = |g: &GroupedData| {
            let s = WeightedSummary::compute(&g.pairs).unwrap();
            s.variance()
        };
        let pooled: f64 = balanced.iter().map(|g| (g.pairs.len() - 1) as f64 * var(g)).sum();
        pooled / (balanced.iter().map(|g| g.pairs.len()).sum::<usize>() as f64 - balanced.len() as f64)
    }

    #[test]
    fn bonferroni_adjusts_p_and_clamps() {
        let g = balanced_groups();
        let ms = ms_within_of(&g);
        let r = post_hoc(&g, ms, 15.0, PostHocMethod::Bonferroni).unwrap();
        assert_eq!(r.comparisons.len(), 3); // m(m−1)/2 = 3 pairs
        assert_eq!(r.method, PostHocMethod::Bonferroni);
        assert_eq!(r.n_groups, 3);
        assert_abs_diff_eq!(r.df_within, 15.0, epsilon = 1e-12);
        // All pair differences are huge → all p-values near 0 (before clamp).
        for c in &r.comparisons {
            assert!(c.p_value <= 1.0);
            assert!(c.p_value < 0.01);
        }
        // Ordered by group: diff of A→B = 10, A→C = 20, B→C = 10.
        assert_abs_diff_eq!(r.comparisons[0].mean_difference, 10.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.comparisons[1].mean_difference, 20.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.comparisons[2].mean_difference, 10.0, epsilon = 1e-12);
        // ci brackets the true difference.
        for c in &r.comparisons {
            assert!(c.ci_95.0 <= c.mean_difference && c.mean_difference <= c.ci_95.1);
        }
    }

    #[test]
    fn tukey_p_is_source_cdf_complement() {
        let g = balanced_groups();
        let ms = ms_within_of(&g);
        let r = post_hoc(&g, ms, 15.0, PostHocMethod::Tukey).unwrap();
        for c in &r.comparisons {
            // statistic is the studentized q.
            let expect_q = c.std_error > 0.0;
            assert!(expect_q);
            // p = 1 − ptukey(statistic, m, df).
            let recomputed = 1.0 - ptukey(c.statistic, r.n_groups, r.df_within);
            assert_abs_diff_eq!(c.p_value, recomputed, epsilon = 1e-9);
            assert!(c.p_value.is_finite());
        }
    }

    #[test]
    fn scheffe_p_is_f_complement() {
        let g = balanced_groups();
        let ms = ms_within_of(&g);
        let r = post_hoc(&g, ms, 15.0, PostHocMethod::Scheffe).unwrap();
        let fdist = FDist::new((r.n_groups - 1) as f64, r.df_within).unwrap();
        for c in &r.comparisons {
            let recomputed = 1.0 - fdist.cdf(c.statistic);
            assert_abs_diff_eq!(c.p_value, recomputed, epsilon = 1e-9);
            assert!(c.p_value.is_finite());
        }
    }

    #[test]
    fn scheffe_statistic_is_t_squared_over_k_minus_1() {
        // Regression for BUG-5: Scheffé F must be t²/(k−1), not t².
        let g = balanced_groups();
        let ms = ms_within_of(&g);
        let k = g.len();
        let r = post_hoc(&g, ms, 15.0, PostHocMethod::Scheffe).unwrap();
        for c in &r.comparisons {
            let t = c.mean_difference / c.std_error;
            let expected_f = t * t / (k - 1) as f64;
            assert_abs_diff_eq!(c.statistic, expected_f, epsilon = 1e-12);
        }
    }

    #[test]
    fn post_hoc_edge_cases() {
        // Fewer than two groups → error.
        let one = vec![group("A", &[1.0, 2.0, 3.0])];
        assert!(post_hoc(&one, 1.0, 2.0, PostHocMethod::Tukey).is_err());
        // Non-positive df → error.
        let g = balanced_groups();
        assert!(post_hoc(&g, 1.0, 0.0, PostHocMethod::Tukey).is_err());
        // Serde round-trip.
        let ms = ms_within_of(&g);
        let r = post_hoc(&g, ms, 15.0, PostHocMethod::Tukey).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: PostHocResult = serde_json::from_str(&json).unwrap();
        assert_eq!(back.comparisons.len(), r.comparisons.len());
        assert_abs_diff_eq!(back.comparisons[0].p_value, r.comparisons[0].p_value, epsilon = 1e-15);
    }

    /// R InsectSprays (sprays A–D; the Schlegel 2020 worked example covers
    /// all six) with PMCMRplus::gamesHowellTest as reference anchors.
    #[test]
    fn games_howell_matches_reference() {
        let sprays = [
            ("A", [10.0, 7.0, 20.0, 14.0, 14.0, 12.0, 10.0, 23.0, 17.0, 20.0, 14.0, 13.0]),
            ("B", [11.0, 17.0, 21.0, 11.0, 16.0, 14.0, 17.0, 17.0, 19.0, 21.0, 7.0, 13.0]),
            ("C", [0.0, 1.0, 7.0, 2.0, 3.0, 1.0, 2.0, 1.0, 3.0, 0.0, 1.0, 4.0]),
            ("D", [3.0, 5.0, 12.0, 6.0, 4.0, 3.0, 5.0, 5.0, 5.0, 5.0, 2.0, 4.0]),
        ];
        let groups: Vec<GroupedData> =
            sprays.iter().map(|(l, v)| group(l, v)).collect();
        let ms = ms_within_of(&groups);
        let r = post_hoc(&groups, ms, 44.0, PostHocMethod::GamesHowell).unwrap();

        assert_eq!(r.method, PostHocMethod::GamesHowell);
        assert_eq!(r.comparisons.len(), 6);
        // Pooled-variance methods keep df = None; Games–Howell reports the
        // per-pair Welch df.
        let pooled = post_hoc(&groups, ms, 44.0, PostHocMethod::Bonferroni).unwrap();
        assert!(pooled.comparisons.iter().all(|c| c.df.is_none()));

        // A vs B (reference: x̄B − x̄A = 0.833333, se 1.299281, q 0.641380,
        // df 21.784431, p 0.997248).
        let ab = &r.comparisons[0];
        assert_eq!((ab.group1.as_str(), ab.group2.as_str()), ("A", "B"));
        assert_abs_diff_eq!(ab.mean_difference, 0.833_333_333_333_333, epsilon = 1e-9);
        assert_abs_diff_eq!(ab.std_error, 1.299_281, epsilon = 1e-6);
        assert_abs_diff_eq!(ab.statistic, 0.641_380, epsilon = 1e-6);
        assert_abs_diff_eq!(ab.df.unwrap(), 21.784_431, epsilon = 1e-6);
        assert_abs_diff_eq!(ab.p_value, 0.968_229_570_976, epsilon = 1e-6);
        assert!(ab.ci_95.0 <= ab.mean_difference && ab.mean_difference <= ab.ci_95.1);

        // B vs C: the largest separation (x̄C − x̄B = -13.25, df 15.499347,
        // p = 2.85e-7).
        let bc = &r.comparisons[3];
        assert_eq!((bc.group1.as_str(), bc.group2.as_str()), ("B", "C"));
        assert_abs_diff_eq!(bc.mean_difference, -13.25, epsilon = 1e-9);
        assert_abs_diff_eq!(bc.std_error, 0.960_554, epsilon = 1e-6);
        assert_abs_diff_eq!(bc.df.unwrap(), 15.499_347, epsilon = 1e-6);
        assert_abs_diff_eq!(bc.p_value, 2.85e-7, epsilon = 0.5e-6);

        // C vs D: borderline pair (x̄D − x̄C = 2.833333, q 4.353254,
        // p 0.027105 with k = 4).
        let cd = &r.comparisons[5];
        assert_eq!((cd.group1.as_str(), cd.group2.as_str()), ("C", "D"));
        assert_abs_diff_eq!(cd.mean_difference, 2.833_333_333_333_333, epsilon = 1e-9);
        assert_abs_diff_eq!(cd.statistic, 4.353_254, epsilon = 1e-6);
        assert_abs_diff_eq!(cd.p_value, 0.027_105_263_340, epsilon = 1e-6);

        // Serde round-trip keeps the per-pair df.
        let json = serde_json::to_string(&r).unwrap();
        let back: PostHocResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.comparisons[0].df.unwrap(), r.comparisons[0].df.unwrap(), epsilon = 1e-15);
        // Old JSON without the df field still deserializes (serde default).
        let legacy = json.replace("\"df\":", "\"df_unused\":");
        let back2: PostHocResult = serde_json::from_str(&legacy).unwrap();
        assert!(back2.comparisons[0].df.is_none());
    }

    #[test]
    fn games_howell_edge_cases() {
        // Single-case group → error (per-pair df undefined).
        let g = vec![
            group("A", &[1.0, 2.0, 3.0]),
            group("B", &[1.0, 2.0, 3.0]),
            group("C", &[5.0]),
        ];
        assert!(post_hoc(&g, 1.0, 2.0, PostHocMethod::GamesHowell).is_err());
        // Zero-variance pair → error.
        let z = vec![group("A", &[2.0, 2.0, 2.0]), group("B", &[2.0, 2.0, 2.0])];
        assert!(post_hoc(&z, 1.0, 2.0, PostHocMethod::GamesHowell).is_err());
    }
}