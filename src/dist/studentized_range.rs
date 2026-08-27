//! Studentized range distribution.
//!
//! The distribution of the studentized range `Q = (max X − min X) / s`, where
//! `X` is a sample of `k` standard-normal order statistics and `s` is an
//! independent `χ²_ν/ν` estimate of the scale. Implements `P(Q ≤ q)` by a
//! double composite Gauss–Legendre quadrature of the CDF integral, keeping the
//! outer integrand in log space so it stays finite for any degrees of freedom
//! (a plain `s^(ν−1)` factor would overflow to `inf` for large `ν`). This is
//! the distribution behind Tukey's honest-significant-difference post-hoc
//! tests.

use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::ln_gamma;

/// 16-point Gauss–Legendre nodes on [−1, 1] (verbatim published constants).
#[allow(clippy::excessive_precision)]
const GL16_NODES: [f64; 16] = [
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792590, -0.0950125098376374,
    0.0950125098376374, 0.2816035507792590, 0.4580167776572274, 0.6178762444026438,
    0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499,
];
/// 16-point Gauss–Legendre weights (verbatim published constants; the total
/// must be exactly 2.0).
#[allow(clippy::excessive_precision)]
const GL16_WEIGHTS: [f64; 16] = [
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495959888165767, 0.1691565193950025, 0.1826034150449236, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449236, 0.1691565193950025, 0.1495959888165767,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
];

/// Panels over the inner `z ∈ [−8, 8]` integral; the standard-normal density
/// is smooth so a modest count resolves it below 1e-9.
const INNER_PANELS: usize = 2;
/// Panels over the outer `t ∈ (0, 1)` integral. The integrand is a narrow bump
/// near `s ≈ 1` (i.e. `t ≈ 0.5`) that sharpens as `ν` grows; enough uniform
/// panels keep it resolved for large degrees of freedom.
const OUTER_PANELS: usize = 128;

/// Numerically integrate `f` over `[a, b]` with a single 16-point
/// Gauss–Legendre panel.
fn gl16<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64) -> f64 {
    let half = (b - a) / 2.0;
    let mid = (a + b) / 2.0;
    GL16_NODES
        .iter()
        .zip(GL16_WEIGHTS.iter())
        .map(|(&node, &wt)| wt * f(half * node + mid))
        .sum::<f64>()
        * half
}

/// Composite 16-point Gauss–Legendre over `[a, b]` split into `panels`
/// sub-intervals. Subdividing keeps the rule accurate for peaked integrands.
fn gl16_composite<F: Fn(f64) -> f64>(f: &F, a: f64, b: f64, panels: usize) -> f64 {
    let h = (b - a) / panels as f64;
    (0..panels)
        .map(|p| {
            let lo = a + p as f64 * h;
            gl16(f, lo, lo + h)
        })
        .sum()
}

/// CDF of the studentized range: `P(Q ≤ q)` for `k` groups and `df` degrees
/// of freedom.
///
/// `q ≤ 0` gives 0, and `k ≥ 2` is required (a single group has constant
/// range 0). Uses the integral
///
/// ```text
/// F(q) = k ν^(ν/2) / (Γ(ν/2) 2^(ν/2−1))
///        ∫₀^∞ s^(ν−1) e^(−ν s²/2) [ ∫ φ(z) (Φ(z + q·s) − Φ(z))^(k−1) dz ] ds
/// ```
///
/// The outer `s ∈ (0, ∞)` integral is mapped to `t ∈ (0, 1)` via `s = t/(1−t)`
/// and evaluated in log space (`(ν−1)ln s − νs²/2`) so it never overflows to
/// `inf`, which previously produced `NaN` for `df ≥ ~72`. The inner `z`
/// integral is truncated to `[−8, 8]` (the standard-normal tail is negligible
/// there) and computed with a composite rule for accuracy.
pub fn ptukey(q: f64, k: usize, df: f64) -> f64 {
    if q <= 0.0 {
        return 0.0;
    }
    if k < 2 {
        return 1.0;
    }
    if !(df.is_finite() && df > 0.0) {
        return f64::NAN;
    }
    let nu = df;
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Inner integral I(s) = ∫ φ(z) (Φ(z + q·s) − Φ(z))^(k−1) dz over z ∈ [−8, 8].
    let inner = |s: f64| -> f64 {
        gl16_composite(
            &|z: f64| {
                let diff = normal.cdf(z + q * s) - normal.cdf(z);
                normal.pdf(z) * diff.powi(k as i32 - 1)
            },
            -8.0,
            8.0,
            INNER_PANELS,
        )
    };

    // Outer integral over t ∈ (0, 1): term = exp((ν−1)ln s − νs²/2) · I(s) / (1−t)².
    // Accumulate log-terms and log-sum-exp, so individual terms may underflow
    // to zero (or the extreme nodes to −inf) without corrupting the result.
    let panel_half = 1.0 / OUTER_PANELS as f64 / 2.0; // half the width of one panel
    let mut log_terms: Vec<f64> = Vec::with_capacity(OUTER_PANELS * 16);
    for p in 0..OUTER_PANELS {
        // Panel interval [lo, lo + 2·panel_half]; Gauss–Legendre maps the
        // node ∈ [−1, 1] via mid + node·panel_half.
        let lo = p as f64 / OUTER_PANELS as f64;
        for (&node, &wt) in GL16_NODES.iter().zip(GL16_WEIGHTS.iter()) {
            let ti = lo + panel_half * (node + 1.0);
            if ti <= 0.0 || ti >= 1.0 {
                continue;
            }
            let s = ti / (1.0 - ti);
            let is = inner(s);
            if is <= 0.0 || !is.is_finite() {
                continue;
            }
            let exp_log = (nu - 1.0) * s.ln() - nu * 0.5 * s * s;
            let log_term = (wt * panel_half).ln() + exp_log + is.ln() - 2.0 * (1.0 - ti).ln();
            log_terms.push(log_term);
        }
    }

    let log_sum = if log_terms.is_empty() {
        f64::NEG_INFINITY
    } else {
        let max = log_terms.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        if !max.is_finite() {
            return 0.0; // every term underflowed → integral is subnormal-zero
        }
        max + log_terms.iter().map(|lt| (lt - max).exp()).sum::<f64>().ln()
    };

    // Leading constant log K = ln k + (ν/2)ln ν − lnΓ(ν/2) − (ν/2 − 1)ln 2.
    let log_k = (k as f64).ln() + (nu / 2.0) * nu.ln() - ln_gamma(nu / 2.0)
        - (nu / 2.0 - 1.0) * (2.0_f64).ln();

    (log_k + log_sum).exp().clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ptukey_k2_matches_t_identity() {
        // For k = 2, Q/√2 = |t_df|, so ptukey(q, 2, df) = 2·T_df(q/√2) − 1
        // exactly. This is the strongest oracle for the CDF and its quadrature.
        use crate::dist::{Distribution, StudentsTDist};
        let q = 3.0;
        for df in [8.0, 30.0, 60.0, 117.0] {
            let got = ptukey(q, 2, df);
            let t = StudentsTDist::new(df).unwrap();
            let expected = 2.0 * t.cdf(q / 2.0_f64.sqrt()) - 1.0;
            assert_abs_diff_eq!(got, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn ptukey_finite_and_accurate_at_large_df() {
        // Regression for BUG-2: df ≥ ~72 used to overflow s^(ν−1) → NaN.
        // Now finite, in range, and the k=3 df=20 value matches scipy 0.88924.
        assert_abs_diff_eq!(ptukey(3.0, 3, 20.0), 0.8892432, epsilon = 1e-5);
        for df in [60.0, 80.0, 117.0, 500.0] {
            let v = ptukey(3.0, 3, df);
            assert!(v.is_finite(), "ptukey NaN at df={df}");
            assert!((0.0..=1.0).contains(&v));
        }
        // CDF reaches 1 (saturates) for a large q, not 0.985 or 0.999.
        let sat = ptukey(200.0, 3, 60.0);
        assert!(sat > 0.999999, "CDF did not saturate near 1: {sat}");
    }

    #[test]
    fn ptukey_bounds_and_monotonic() {
        assert_abs_diff_eq!(ptukey(0.0, 3, 10.0), 0.0, epsilon = 1e-12);
        assert!(ptukey(50.0, 3, 10.0) > 0.999);
        // Monotone non-decreasing in q.
        let mut prev = -1.0;
        for q in (10..=100).step_by(10) {
            let v = ptukey(q as f64 / 10.0, 4, 8.0);
            assert!(v >= prev && v <= 1.0);
            prev = v;
        }
        // k < 2 → the range is always 0 → CDF saturates at 1.
        assert_abs_diff_eq!(ptukey(1.0, 1, 5.0), 1.0, epsilon = 1e-12);
    }

    #[test]
    fn gl16_weights_sum_to_two() {
        // The published Gauss–Legendre weights must integrate a constant
        // exactly; a subtle typo here caused a systematic ~1e-4 CDF bias
        // (BUG-2a). Guard it.
        assert_abs_diff_eq!(GL16_WEIGHTS.iter().sum::<f64>(), 2.0, epsilon = 1e-12);
    }
}