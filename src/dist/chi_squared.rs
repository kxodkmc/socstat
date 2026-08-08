//! Chi-square (χ²) distribution.

use statrs::distribution::{ChiSquared, Continuous, ContinuousCDF};

use super::Distribution;
use crate::error::{SocStatError, SocStatResult};

/// Chi-square distribution with `k` degrees of freedom.
pub struct ChiSquaredDist {
    inner: ChiSquared,
}

impl ChiSquaredDist {
    /// Create a χ² distribution with `k` degrees of freedom.
    /// Returns an error if k ≤ 0.
    pub fn new(k: f64) -> SocStatResult<Self> {
        if k <= 0.0 {
            return Err(SocStatError::Computation(
                "df must be positive".into(),
            ));
        }
        let inner = ChiSquared::new(k)
            .map_err(|e| SocStatError::Computation(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Distribution for ChiSquaredDist {
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
    fn chi2_cdf_at_zero() {
        let chi = ChiSquaredDist::new(3.0).unwrap();
        assert_abs_diff_eq!(chi.cdf(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn chi2_critical_value() {
        // χ²(1) at α=0.05 → critical ≈ 3.841
        let chi = ChiSquaredDist::new(1.0).unwrap();
        assert_abs_diff_eq!(chi.inverse_cdf(0.95), 3.841, epsilon = 1e-2);

        // χ²(3) at α=0.05 → critical ≈ 7.815
        let chi = ChiSquaredDist::new(3.0).unwrap();
        assert_abs_diff_eq!(chi.inverse_cdf(0.95), 7.815, epsilon = 1e-2);
    }

    #[test]
    fn invalid_df_errors() {
        assert!(ChiSquaredDist::new(0.0).is_err());
        assert!(ChiSquaredDist::new(-1.0).is_err());
    }
}
