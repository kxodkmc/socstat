//! Normality tests: Shapiro–Wilk and Kolmogorov–Smirnov.
//!
//! [`shapiro_wilk`] implements the Shapiro–Wilk W test for `3 ≤ n ≤ 5000`
//! using Royston's Remark AS R94 (the same algorithm as R's `shapiro.test`).
//! [`ks_test`] implements the one-sample Kolmogorov–Smirnov test against a
//! specified `N(μ, σ²)` and the Lilliefors version for an estimated
//! `N(x̄, s²)` (with the Dallal–Wilkinson approximate p-value).
//!
//! Weights are treated as **frequency weights** (each case counts as `weight`
//! replicates); a case with a non-positive or non-finite weight is excluded.
//! For Shapiro–Wilk the frequency expansion is limited to `n ≤ 5000`, matching
//! the test's supported range.
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use serde::{Deserialize, Serialize};

use crate::dist::{Distribution, NormalDist};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{kolmogorov_two_sided_p, positive_weight, stephens_lambda};

/// Result of the Shapiro–Wilk normality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapiroWilkResult {
    /// Effective sample size (sum of weights after frequency expansion).
    pub n: f64,
    /// The Shapiro–Wilk W statistic (in [0, 1]; closer to 1 is more normal).
    pub w_statistic: f64,
    /// Approximate two-sided p-value under the null of normality.
    pub p_value: f64,
}

/// The kind of one-sample Kolmogorov–Smirnov normality test.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum KsTestType {
    /// Test against a fully specified `N(mean, std_dev)`. P-values use the
    /// Kolmogorov distribution (exact for a known distribution).
    OneSample { mean: f64, std_dev: f64 },
    /// Lilliefors test: `μ` and `σ` are estimated from the sample. P-values
    /// use the Dallal–Wilkinson approximation (only reliable for `p < 0.1`).
    Lilliefors,
}

/// Result of a one-sample Kolmogorov–Smirnov normality test.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KolmogorovSmirnovResult {
    /// Effective sample size (sum of weights).
    pub n: f64,
    /// The K-S D statistic (supremum distance between CDFs).
    pub d_statistic: f64,
    pub p_value: f64,
    pub test: KsTestType,
    /// True when `p_value` is only a coarse approximation (Lilliefors
    /// `p ≥ 0.1` is rounded to the conservative value `0.1`).
    pub p_is_approx: bool,
}

// ---------------------------------------------------------------------------
// AS R94 polynomial coefficients (Royston, 1995)
// ---------------------------------------------------------------------------

const G: [f64; 2] = [-2.273, 0.459];
const C1: [f64; 6] = [0.0, 0.221157, -0.147981, -2.07119, 4.434_685, -2.706_056];
const C2: [f64; 6] = [0.0, 0.042981, -0.293762, -1.752_461, 5.682_633, -3.582_633];
const C3: [f64; 4] = [0.544, -0.39978, 0.025054, -6.714e-4];
const C4: [f64; 4] = [1.3822, -0.77857, 0.062767, -0.0020322];
const C5: [f64; 4] = [-1.5861, -0.31082, -0.083751, 0.0038915];
const C6: [f64; 3] = [-0.4803, -0.082676, 0.0030302];

/// Evaluate `cc[0] + cc[1]·x + cc[2]·x² + …` by Horner's rule.
fn poly(cc: &[f64], x: f64) -> f64 {
    let mut res = cc[cc.len() - 1];
    for &c in cc[..cc.len() - 1].iter().rev() {
        res = res * x + c;
    }
    res
}

// ---------------------------------------------------------------------------
// Shapiro–Wilk (AS R94)
// ---------------------------------------------------------------------------

/// Shapiro–Wilk W test of normality.
///
/// `data` must contain at least 3 and at most 5000 valid values; non-finite
/// values are dropped. `weights` are optional frequency weights: each value is
/// rounded to `weight` replicates, so the effective sample size may not exceed
/// 5000.
///
/// # Example
///
/// ```no_run
/// use socstat::stats::normality::shapiro_wilk;
/// let data = [3.2, 4.8, 5.1, 4.9, 6.3, 5.7];
/// let r = shapiro_wilk(&data, None).unwrap();
/// println!("W = {:.4}, p = {:.4}", r.w_statistic, r.p_value);
/// ```
pub fn shapiro_wilk(data: &[f64], weights: Option<&[f64]>) -> SocStatResult<ShapiroWilkResult> {
    let mut x = expand_frequency_weights(data, weights)?;
    if x.len() > 5000 {
        return Err(SocStatError::InsufficientData(
            "Shapiro-Wilk supports n ≤ 5000; use ks_test (Lilliefors) for larger samples".into(),
        ));
    }
    if x.len() < 3 {
        return Err(SocStatError::InsufficientData(
            "Shapiro-Wilk needs at least 3 valid values".into(),
        ));
    }
    x.sort_by(|a, b| a.total_cmp(b));

    let w = swilk_w_statistic(&x)?;
    let p = swilk_p_value(w, x.len());
    Ok(ShapiroWilkResult { n: x.len() as f64, w_statistic: w, p_value: p })
}

/// Expand frequency weights into integer replicates (rounded), keeping only
/// finite values with a positive weight.
fn expand_frequency_weights(data: &[f64], weights: Option<&[f64]>) -> SocStatResult<Vec<f64>> {
    if let Some(w) = weights
        && w.len() != data.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: data.len(),
            got: w.len(),
        });
    }
    let Some(w) = weights else {
        return Ok(data.iter().filter(|v| v.is_finite()).copied().collect());
    };

    let mut reps = 0usize;
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() || !positive_weight(w[i]) {
            continue;
        }
        reps = reps.saturating_add(w[i].round() as usize);
        if reps > 5000 {
            return Err(SocStatError::InsufficientData(
                "Shapiro-Wilk sample exceeds 5000 after weighting".into(),
            ));
        }
    }
    let mut out = Vec::with_capacity(reps);
    for (i, &v) in data.iter().enumerate() {
        if !v.is_finite() || !positive_weight(w[i]) {
            continue;
        }
        let k = w[i].round() as usize;
        out.extend(std::iter::repeat_n(v, k));
    }
    Ok(out)
}

/// Compute the W statistic from sorted `x` (length 3..=5000), faithful to
/// R's `swilk.c` (AS R94).
fn swilk_w_statistic(x: &[f64]) -> SocStatResult<f64> {
    let n = x.len();
    let nn2 = n / 2;
    let an = n as f64;

    // Coefficients a[1..=nn2] (1-indexed, matching AS R94).
    let mut a = vec![0.0_f64; nn2 + 1];
    if n == 3 {
        a[1] = std::f64::consts::FRAC_1_SQRT_2;
    } else {
        let an25 = an + 0.25;
        let normal = NormalDist::standard();
        let mut summ2 = 0.0;
        for (i, slot) in a.iter_mut().enumerate().take(nn2 + 1).skip(1) {
            *slot = normal.inverse_cdf((i as f64 - 0.375) / an25);
            summ2 += *slot * *slot;
        }
        summ2 *= 2.0;
        let ssumm2 = summ2.sqrt();
        let rsn = 1.0 / an.sqrt();
        let a1 = poly(&C1, rsn) - a[1] / ssumm2;

        let (i1, fac) = if n > 5 {
            let a2 = -a[2] / ssumm2 + poly(&C2, rsn);
            let radicand = (summ2 - 2.0 * a[1] * a[1] - 2.0 * a[2] * a[2])
                / (1.0 - 2.0 * a1 * a1 - 2.0 * a2 * a2);
            a[2] = a2;
            (3usize, radicand.sqrt())
        } else {
            let radicand = (summ2 - 2.0 * a[1] * a[1]) / (1.0 - 2.0 * a1 * a1);
            (2usize, radicand.sqrt())
        };
        if !(fac.is_finite() && fac > 0.0) {
            return Err(SocStatError::Computation(
                "Shapiro-Wilk coefficients could not be normalized".into(),
            ));
        }
        a[1] = a1;
        for slot in a[i1..=nn2].iter_mut() {
            *slot /= -fac;
        }
    }

    // W as the squared correlation of the (antisymmetric) coefficients with
    // the order statistics. Scaling by the range keeps W near 1 accurate.
    let range = x[n - 1] - x[0];
    if range <= 1e-19 {
        return Err(SocStatError::InsufficientData(
            "Shapiro-Wilk is undefined for constant data".into(),
        ));
    }
    let mut sax = 0.0;
    for (&c, (&hi, &lo)) in a[1..=nn2].iter().zip(x.iter().rev().zip(x.iter())) {
        sax += c * (hi - lo) / range;
    }
    let xbar = x.iter().sum::<f64>() / an / range;
    let mut ss = 0.0;
    for &xi in x {
        let v = xi / range - xbar;
        ss += v * v;
    }
    // Σa² = 1 over the full coefficient vector (by construction), so
    // W = (Σ a·x)² / Σ(x - x̄)².
    Ok(((sax * sax) / ss).clamp(0.0, 1.0))
}

/// p-value for a given W and n (exact for n = 3, AS R94 otherwise).
fn swilk_p_value(w: f64, n: usize) -> f64 {
    let w = w.clamp(0.0, 1.0);
    let w1 = 1.0 - w;
    let an = n as f64;
    let normal = NormalDist::standard();

    if n == 3 {
        let pi6 = 6.0 / std::f64::consts::PI;
        let stqr = (3.0_f64 / 4.0).sqrt().asin();
        return (pi6 * (w.sqrt().asin() - stqr)).max(0.0);
    }

    // y0 = ln(1 - W); may be -∞ for W = 1, which is handled by the arithmetic.
    let y0 = w1.ln();
    let (y, m, s) = if n <= 11 {
        let gamma = poly(&G, an);
        if y0 >= gamma {
            // Extremely non-normal small sample: p is effectively 0.
            return 1e-99;
        }
        (-(gamma - y0).ln(), poly(&C3, an), poly(&C4, an).exp())
    } else {
        (y0, poly(&C5, an.ln()), poly(&C6, an.ln()).exp())
    };

    // r_to_s/m normalizing transform → upper-tail p.
    let z = (y - m) / s;
    (1.0 - normal.cdf(z)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Kolmogorov–Smirnov (one-sample + Lilliefors)
// ---------------------------------------------------------------------------

/// One-sample Kolmogorov–Smirnov test against a normal distribution.
///
/// `test_type` selects either a fully specified `N(mean, std_dev)` (p-value
/// from the Kolmogorov distribution) or the Lilliefors version (parameters
/// estimated from the sample; Dallal–Wilkinson p-value). Non-finite values and
/// non-positive weights are excluded.
pub fn ks_test(
    data: &[f64],
    weights: Option<&[f64]>,
    test_type: KsTestType,
) -> SocStatResult<KolmogorovSmirnovResult> {
    if let Some(w) = weights
        && w.len() != data.len()
    {
        return Err(SocStatError::ColumnLengthMismatch {
            expected: data.len(),
            got: w.len(),
        });
    }

    let mut vals: Vec<(f64, f64)> = (0..data.len())
        .filter_map(|i| {
            let v = data[i];
            if !v.is_finite() {
                return None;
            }
            let w = weights.map(|ww| ww[i]).unwrap_or(1.0);
            positive_weight(w).then_some((v, w))
        })
        .collect();
    let n_eff: f64 = vals.iter().map(|&(_, w)| w).sum();
    if n_eff < 5.0 {
        return Err(SocStatError::InsufficientData(
            "K-S test needs at least 5 valid (weighted) cases".into(),
        ));
    }
    vals.sort_by(|a, b| a.0.total_cmp(&b.0));

    let (mu, sigma) = match test_type {
        KsTestType::OneSample { mean, std_dev } => (mean, std_dev),
        KsTestType::Lilliefors => {
            let mean = vals.iter().map(|&(v, w)| v * w).sum::<f64>() / n_eff;
            let var = vals.iter().map(|&(v, w)| w * (v - mean).powi(2)).sum::<f64>()
                / (n_eff - 1.0);
            (mean, var.sqrt())
        }
    };
    if !(sigma.is_finite() && sigma > 0.0) {
        return Err(SocStatError::InsufficientData(
            "K-S test is undefined for zero variance data".into(),
        ));
    }
    let normal = NormalDist::new(mu, sigma)?;

    // Supremum D between the weighted empirical CDF and the normal CDF.
    let mut cum = 0.0_f64;
    let (mut d_plus, mut d_minus) = (0.0_f64, 0.0_f64);
    for &(v, w) in &vals {
        let f = normal.cdf(v);
        cum += w;
        d_plus = d_plus.max(cum / n_eff - f);
        d_minus = d_minus.max(f - (cum - w) / n_eff);
    }
    let d = d_plus.max(d_minus);

    let (p_value, p_is_approx) = match test_type {
        KsTestType::OneSample { .. } => {
            // Kolmogorov distribution with Stephens' finite-size adjustment.
            let lambda = stephens_lambda(n_eff, d);
            (kolmogorov_two_sided_p(lambda), false)
        }
        KsTestType::Lilliefors => {
            // Dallal & Wilkinson (1986) approximation.
            let (d_star, n_star) = if n_eff > 100.0 {
                (d * (n_eff / 100.0).powf(0.49), 100.0_f64)
            } else {
                (d, n_eff)
            };
            let ln_p = -7.01256 * d_star * d_star * (n_star + 2.78019)
                + 2.99587 * d_star * (n_star + 2.78019).sqrt()
                - 0.122119
                + 0.974598 / n_star.sqrt()
                + 1.67997 / n_star;
            let p = ln_p.exp();
            if p < 0.1 {
                (p.clamp(0.0, 1.0), false)
            } else {
                // Approx unreliable here; return the conservative value.
                (0.1, true)
            }
        }
    };

    Ok(KolmogorovSmirnovResult {
        n: n_eff,
        d_statistic: d,
        p_value,
        test: test_type,
        p_is_approx,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    /// Standard-normal order statistics; a sample of these should read as
    /// essentially normal (W ≈ 1).
    fn normal_quantiles(n: usize) -> Vec<f64> {
        let normal = NormalDist::standard();
        (1..=n).map(|i| normal.inverse_cdf((i as f64 - 0.375) / (n as f64 + 0.25))).collect()
    }

    #[test]
    fn shapiro_accepts_normal_large_sample() {
        let x = normal_quantiles(50);
        let r = shapiro_wilk(&x, None).unwrap();
        assert_abs_diff_eq!(r.n, 50.0, epsilon = 1e-12);
        assert!(r.w_statistic > 0.98);
        assert!(r.p_value > 0.5);
    }

    #[test]
    fn shapiro_rejects_skewed_sample() {
        // Strong right skew (values from 1 up to ~1800): clearly non-normal.
        let x: Vec<f64> = (0..15).map(|i| (i as f64 / 2.0).exp()).collect();
        let r = shapiro_wilk(&x, None).unwrap();
        assert!(r.w_statistic < 0.85);
        assert!(r.p_value < 0.001);
    }

    #[test]
    fn shapiro_n3_exact() {
        // [1,2,3]: mean 2, W = 0.5*(3-1)^2 / 2 = 1 → p = 1 (exact).
        let r = shapiro_wilk(&[1.0, 2.0, 3.0], None).unwrap();
        assert_abs_diff_eq!(r.w_statistic, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.p_value, 1.0, epsilon = 1e-12);
        // [1,1,2]: W < 1 and p in (0, 1).
        let r2 = shapiro_wilk(&[1.0, 1.0, 2.0], None).unwrap();
        assert!(r2.p_value > 0.0 && r2.p_value < 1.0);
    }

    #[test]
    fn shapiro_weighted_equals_frequency() {
        // weight 3 on a normal sample must match tripling the points.
        let x = normal_quantiles(20);
        let w = vec![3.0; 20];
        let rw = shapiro_wilk(&x, Some(&w)).unwrap();
        let mut x3 = Vec::new();
        for &v in &x {
            x3.extend(std::iter::repeat_n(v, 3));
        }
        let ru = shapiro_wilk(&x3, None).unwrap();
        assert_abs_diff_eq!(rw.w_statistic, ru.w_statistic, epsilon = 1e-10);
        assert!((rw.p_value - ru.p_value).abs() < 1e-6);
    }

    #[test]
    fn shapiro_edge_cases() {
        assert!(shapiro_wilk(&[1.0, 2.0], None).is_err());
        assert!(shapiro_wilk(&[5.0, 5.0, 5.0], None).is_err()); // constant → zero range
        // Expanded sample beyond 5000 is rejected, not allocated.
        assert!(shapiro_wilk(&[1.0], Some(&[6000.0])).is_err());
    }

    #[test]
    fn ks_one_sample_normal_passes() {
        let x = normal_quantiles(40);
        let r = ks_test(&x, None, KsTestType::OneSample { mean: 0.0, std_dev: 1.0 }).unwrap();
        assert!(r.d_statistic < 0.2);
        assert!(r.p_value > 0.1);
        assert!(!r.p_is_approx);
    }

    #[test]
    fn ks_lilliefors_rejects_skewed() {
        let x: Vec<f64> = (0..20).map(|i| (i as f64 / 4.0).exp()).collect();
        let r = ks_test(&x, None, KsTestType::Lilliefors).unwrap();
        assert!(r.p_value < 0.05);
    }

    #[test]
    fn ks_edge_cases() {
        assert!(ks_test(&[1.0, 2.0, 3.0, 4.0], None, KsTestType::Lilliefors).is_err());
        assert!(ks_test(
            &[1.0, 1.0, 1.0, 1.0, 1.0],
            None,
            KsTestType::Lilliefors
        )
        .is_err()); // zero variance
        // Serde round-trip.
        let r = ks_test(
            &normal_quantiles(30),
            None,
            KsTestType::OneSample { mean: 0.0, std_dev: 1.0 },
        )
        .unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: KolmogorovSmirnovResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.d_statistic, r.d_statistic, epsilon = 1e-15);
    }

    #[test]
    fn shapiro_serde_round_trip() {
        let x = normal_quantiles(30);
        let r = shapiro_wilk(&x, None).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: ShapiroWilkResult = serde_json::from_str(&json).unwrap();
        assert_abs_diff_eq!(back.w_statistic, r.w_statistic, epsilon = 1e-15);
    }
}