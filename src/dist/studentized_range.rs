//! Studentized range distribution.
//!
//! The distribution of the studentized range `Q = (max X − min X) / s`, where
//! `X` is a sample of `k` standard-normal order statistics and `s` is an
//! independent `χ²_ν/ν` estimate of the scale. Implements `P(Q ≤ q)` by a
//! double 16-point Gauss–Legendre quadrature of the CDF integral (the same
//! approach as R's `ptukey`). This is the distribution behind Tukey's
//! honest-significant-difference post-hoc tests.

use statrs::distribution::{Continuous, ContinuousCDF, Normal};
use statrs::function::gamma::gamma;

/// 16-point Gauss–Legendre nodes on [−1, 1] (verbatim published constants).
#[allow(clippy::excessive_precision)]
const GL16_NODES: [f64; 16] = [
    -0.9894009349916499, -0.9445750230732326, -0.8656312023878318, -0.7554044083550030,
    -0.6178762444026438, -0.4580167776572274, -0.2816035507792590, -0.0950125098376374,
    0.0950125098376374, 0.2816035507792590, 0.4580167776572274, 0.6178762444026438,
    0.7554044083550030, 0.8656312023878318, 0.9445750230732326, 0.9894009349916499,
];
/// 16-point Gauss–Legendre weights (verbatim published constants).
#[allow(clippy::excessive_precision)]
const GL16_WEIGHTS: [f64; 16] = [
    0.0271524594117541, 0.0622535239386479, 0.0951585116824928, 0.1246289712555339,
    0.1495459909365477, 0.1691565193950025, 0.1826034150449906, 0.1894506104550685,
    0.1894506104550685, 0.1826034150449906, 0.1691565193950025, 0.1495459909365477,
    0.1246289712555339, 0.0951585116824928, 0.0622535239386479, 0.0271524594117541,
];

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
/// F(q) = √(2π) k ν^(ν/2) / (Γ(ν/2) 2^(ν/2−1))
///        ∫₀^∞ s^(ν−1) φ(√ν s) [ ∫ φ(z) (Φ(z + q·s) − Φ(z))^(k−1) dz ] ds
/// ```
///
/// The outer `s ∈ (0, ∞)` integral is mapped to `t ∈ (0, 1)` via `s = t/(1−t)`
/// and the inner `z` integral is truncated to `[−8, 8]` (the standard-normal
/// tail there is negligible).
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
    let normal = Normal::new(0.0, 1.0).unwrap();
    let nu = df;

    // Leading constant.
    let coef = (2.0 * std::f64::consts::PI).sqrt() * k as f64 * nu.powf(nu / 2.0)
        / (gamma(nu / 2.0) * 2.0_f64.powf(nu / 2.0 - 1.0));

    let outer = |t: f64| -> f64 {
        if t <= 0.0 || t >= 1.0 {
            return 0.0;
        }
        let s = t / (1.0 - t);
        let s_nu = s.powf(nu - 1.0);
        let phi_s = normal.pdf((nu).sqrt() * s);

        let inner = gl16(
            &|z| {
                let diff = normal.cdf(z + q * s) - normal.cdf(z);
                normal.pdf(z) * diff.powi(k as i32 - 1)
            },
            -8.0,
            8.0,
        );

        // ds/dt = 1/(1−t)².
        s_nu * phi_s * inner / (1.0 - t).powi(2)
    };

    // Outer s ∈ (0, ∞) integral mapped to t ∈ (0, 1); composite quadrature
    // keeps it accurate across the peaked integrand for any df.
    let integral = gl16_composite(&outer, 0.0, 1.0, 128);
    (coef * integral).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn ptukey_k2_matches_t_identity() {
        // For k = 2, Q/√2 = |t_df|, so ptukey(q, 2, df) = 2·T_df(q/√2) − 1
        // exactly. Checks both the CDF formula and the quadrature accuracy at
        // moderate dfs (where the integrand is resolvable by the 16-point rule).
        use crate::dist::{Distribution, StudentsTDist};
        let q = 3.0;
        for df in [8.0, 15.0, 30.0] {
            let got = ptukey(q, 2, df);
            let t = StudentsTDist::new(df).unwrap();
            let expected = 2.0 * t.cdf(q / 2.0_f64.sqrt()) - 1.0;
            assert_abs_diff_eq!(got, expected, epsilon = 1e-2);
        }
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
}