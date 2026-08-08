//! Normal (Gaussian) distribution.

use statrs::distribution::{Continuous, ContinuousCDF, Normal};

use super::Distribution;
use crate::error::{SocStatError, SocStatResult};

/// Normal distribution with mean μ and standard deviation σ.
pub struct NormalDist {
    inner: Normal,
}

impl NormalDist {
    /// Create a Normal(μ, σ) distribution.
    /// Returns an error if σ ≤ 0.
    pub fn new(mean: f64, std_dev: f64) -> SocStatResult<Self> {
        if std_dev <= 0.0 {
            return Err(SocStatError::Computation(
                "std_dev must be positive".into(),
            ));
        }
        let inner = Normal::new(mean, std_dev)
            .map_err(|e| SocStatError::Computation(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Standard normal: N(0, 1).
    pub fn standard() -> Self {
        Self { inner: Normal::new(0.0, 1.0).unwrap() }
    }
}

impl Distribution for NormalDist {
    #[inline]
    fn pdf(&self, x: f64) -> f64 { self.inner.pdf(x) }

    #[inline]
    fn cdf(&self, x: f64) -> f64 { self.inner.cdf(x) }

    #[inline]
    fn inverse_cdf(&self, p: f64) -> f64 { self.inner.inverse_cdf(p) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn standard_normal_cdf() {
        let n = NormalDist::standard();
        assert_abs_diff_eq!(n.cdf(0.0), 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(n.cdf(1.96), 0.975, epsilon = 1e-3);
        assert_abs_diff_eq!(n.cdf(-1.96), 0.025, epsilon = 1e-3);
    }

    #[test]
    fn standard_normal_inverse_cdf() {
        let n = NormalDist::standard();
        assert_abs_diff_eq!(n.inverse_cdf(0.5), 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(n.inverse_cdf(0.975), 1.96, epsilon = 1e-2);
        assert_abs_diff_eq!(n.inverse_cdf(0.025), -1.96, epsilon = 1e-2);
    }

    #[test]
    fn nonzero_mean_variance() {
        let n = NormalDist::new(5.0, 2.0).unwrap();
        assert_abs_diff_eq!(n.cdf(5.0), 0.5, epsilon = 1e-10);
        // P(X ≤ 7) = P(Z ≤ 1) ≈ 0.8413
        assert_abs_diff_eq!(n.cdf(7.0), 0.8413, epsilon = 1e-3);
    }

    #[test]
    fn negative_std_dev_errors() {
        assert!(NormalDist::new(0.0, -1.0).is_err());
        assert!(NormalDist::new(0.0, 0.0).is_err());
    }
}
