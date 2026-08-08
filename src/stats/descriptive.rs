//! Descriptive statistics — the most commonly used summary measures.
//!
//! Supports both unweighted and frequency-weighted computation.
//! When a [`Dataset`](crate::data::Dataset) has a weight variable set
//! (via [`set_weight`](crate::data::Dataset::set_weight)), all statistics
//! automatically use frequency weights.

use serde::{Deserialize, Serialize};

use crate::dist::{Distribution, StudentsTDist};

/// Comprehensive descriptive statistics for a single variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Descriptive {
    /// Number of valid (non-missing) observations.
    /// For weighted data, this is the sum of weights.
    pub n: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Sample standard deviation (denominator = n-1).
    pub std_dev: f64,
    /// Sample variance.
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub sum: f64,
    /// 50th percentile (median).
    pub median: f64,
    /// Fisher-Pearson coefficient of skewness.
    pub skewness: f64,
    /// Excess kurtosis (kurtosis - 3).
    pub kurtosis: f64,
    /// 25th percentile.
    pub q1: f64,
    /// 75th percentile.
    pub q3: f64,
    /// Standard error of the mean: std_dev / sqrt(n).
    pub sem: f64,
    /// 95% confidence interval for the mean (t-based).
    pub ci_95: (f64, f64),
}

/// Compute descriptive statistics from a slice of valid numeric values.
///
/// If `weights` is `Some`, they are used as frequency weights.
/// The lengths of `data` and `weights` must match.
pub fn compute(data: &[f64], weights: Option<&[f64]>) -> Descriptive {
    let n = data.len();
    let w = match weights {
        Some(w) if w.len() == n => w,
        _ => &[][..],
    };
    let weighted = !w.is_empty();

    // Effective N
    let n_eff: f64 = if weighted { w.iter().sum() } else { n as f64 };

    // Sum and mean
    let (sum, mean) = if weighted {
        let s: f64 = data.iter().zip(w).map(|(x, wi)| x * wi).sum();
        let m = s / n_eff;
        (s, m)
    } else {
        let s: f64 = data.iter().sum();
        let m = s / n_eff;
        (s, m)
    };

    // Central moments m2, m3, m4
    let (m2, m3, m4) = if weighted {
        let (s2, s3, s4): (f64, f64, f64) = data.iter().zip(w)
            .map(|(x, wi)| {
                let d = x - mean;
                let d2 = d * d;
                (wi * d2, wi * d2 * d, wi * d2 * d2)
            })
            .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        (s2 / n_eff, s3 / n_eff, s4 / n_eff)
    } else {
        let (s2, s3, s4): (f64, f64, f64) = data.iter()
            .map(|x| {
                let d = x - mean;
                let d2 = d * d;
                (d2, d2 * d, d2 * d2)
            })
            .fold((0.0, 0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1, a.2 + b.2));
        (s2 / n_eff, s3 / n_eff, s4 / n_eff)
    };

    // Sample variance: m2 * n / (n-1)
    let variance = if n_eff > 1.0 {
        m2 * n_eff / (n_eff - 1.0)
    } else {
        0.0
    };
    let std_dev = variance.sqrt();

    // Skewness: m3 / m2^(3/2)
    let skewness = if m2 > 0.0 {
        m3 / m2.powf(1.5)
    } else {
        0.0
    };

    // Excess kurtosis: m4 / m2^2 - 3
    let kurtosis = if m2 > 0.0 {
        m4 / (m2 * m2) - 3.0
    } else {
        0.0
    };

    // Min, max
    let (min, max) = data.iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(mn, mx), x| {
            (mn.min(*x), mx.max(*x))
        });

    // Percentiles (need sorted data)
    // For weighted percentiles, we use the standard approach:
    // sort by value, then find the value at position p * n_eff in cumulative weight.
    let (median, q1, q3) = if weighted {
        // Build (value, weight) pairs and sort
        let mut pairs: Vec<(f64, f64)> = data.iter().zip(w)
            .map(|(x, wi)| (*x, *wi))
            .collect();
        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        (
            weighted_percentile(&pairs, 0.50),
            weighted_percentile(&pairs, 0.25),
            weighted_percentile(&pairs, 0.75),
        )
    } else {
        let mut sorted = data.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        (
            percentile(&sorted, 0.50),
            percentile(&sorted, 0.25),
            percentile(&sorted, 0.75),
        )
    };

    // Standard error of the mean
    let sem = std_dev / n_eff.sqrt();

    // 95% CI: mean ± t(0.975, n-1) * sem
    let ci_95 = if n_eff > 1.0 {
        let df = n_eff - 1.0;
        let t_crit = StudentsTDist::new(df)
            .ok()
            .map(|d| d.inverse_cdf(0.975))
            .unwrap_or(1.96);
        (mean - t_crit * sem, mean + t_crit * sem)
    } else {
        (mean, mean)
    };

    Descriptive {
        n: n_eff,
        mean,
        std_dev,
        variance,
        min,
        max,
        range: max - min,
        sum,
        median,
        skewness,
        kurtosis,
        q1,
        q3,
        sem,
        ci_95,
    }
}

/// Linear-interpolation percentile from a sorted slice.
/// `p` is in [0, 1]. Uses the same method as NumPy default (Type 7).
fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = lo + 1;
    if hi >= sorted.len() {
        return sorted[sorted.len() - 1];
    }
    let frac = idx - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Weighted percentile from sorted (value, weight) pairs.
fn weighted_percentile(pairs: &[(f64, f64)], p: f64) -> f64 {
    if pairs.is_empty() {
        return f64::NAN;
    }
    let total: f64 = pairs.iter().map(|(_, w)| w).sum();
    let target = p * total;
    let mut cum = 0.0;
    for &(val, w) in pairs {
        cum += w;
        if cum >= target {
            return val;
        }
    }
    pairs.last().unwrap().0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn basic_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = compute(&data, None);
        assert_abs_diff_eq!(d.mean, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d.std_dev, 1.5811, epsilon = 1e-3);
        assert_abs_diff_eq!(d.variance, 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(d.min, 1.0, epsilon = 0.0);
        assert_abs_diff_eq!(d.max, 5.0, epsilon = 0.0);
        assert_abs_diff_eq!(d.median, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d.sum, 15.0, epsilon = 1e-10);
        assert_eq!(d.n, 5.0);
    }

    #[test]
    fn skewness_symmetric_data() {
        // Symmetric data → skewness ≈ 0
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = compute(&data, None);
        assert!(d.skewness.abs() < 1e-10);
    }

    #[test]
    fn kurtosis_uniform_data() {
        // Uniform distribution → excess kurtosis ≈ -1.2
        let data: Vec<f64> = (1..=6).map(|x| x as f64).collect();
        let d = compute(&data, None);
        assert!(d.kurtosis < 0.0, "uniform should have negative excess kurtosis");
    }

    #[test]
    fn quartiles_even_count() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let d = compute(&data, None);
        // Type 7 (linear): idx = p*(n-1)
        // Q1: 0.25*7=1.75 → sorted[1]*0.25 + sorted[2]*0.75 = 2*0.25+3*0.75 = 2.75
        assert_abs_diff_eq!(d.q1, 2.75, epsilon = 1e-10);
        // Q3: 0.75*7=5.25 → sorted[5]*0.75 + sorted[6]*0.25 = 6*0.75+7*0.25 = 6.25
        assert_abs_diff_eq!(d.q3, 6.25, epsilon = 1e-10);
        // Median: 0.5*7=3.5 → sorted[3]*0.5 + sorted[4]*0.5 = 4*0.5+5*0.5 = 4.5
        assert_abs_diff_eq!(d.median, 4.5, epsilon = 1e-10);
    }

    #[test]
    fn weighted_mean() {
        let data = vec![1.0, 2.0, 3.0];
        let weights = vec![1.0, 2.0, 3.0]; // weighted mean = (1+4+9)/6 = 14/6
        let d = compute(&data, Some(&weights));
        assert_abs_diff_eq!(d.mean, 14.0 / 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(d.n, 6.0, epsilon = 1e-10);
    }

    #[test]
    fn weighted_matches_unweighted_when_unit_weights() {
        let data = vec![10.0, 20.0, 30.0];
        let weights = vec![1.0, 1.0, 1.0];
        let d_w = compute(&data, Some(&weights));
        let d_u = compute(&data, None);
        assert_abs_diff_eq!(d_w.mean, d_u.mean, epsilon = 1e-10);
        assert_abs_diff_eq!(d_w.variance, d_u.variance, epsilon = 1e-10);
        assert_abs_diff_eq!(d_w.skewness, d_u.skewness, epsilon = 1e-10);
    }

    #[test]
    fn ci_95_brackets_mean() {
        let data = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0];
        let d = compute(&data, None);
        assert!(d.ci_95.0 < d.mean);
        assert!(d.ci_95.1 > d.mean);
    }

    #[test]
    fn single_value() {
        let data = vec![42.0];
        let d = compute(&data, None);
        assert_abs_diff_eq!(d.mean, 42.0, epsilon = 1e-10);
        assert!(d.std_dev.is_nan() || d.std_dev == 0.0);
        assert_abs_diff_eq!(d.min, 42.0, epsilon = 0.0);
    }
}
