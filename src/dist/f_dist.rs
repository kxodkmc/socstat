//! F (Fisher-Snedecor) distribution.

use statrs::distribution::{Continuous, ContinuousCDF, FisherSnedecor};

use super::Distribution;
use crate::error::{SocStatError, SocStatResult};

/// F distribution with `df1` (numerator) and `df2` (denominator) DOF.
pub struct FDist {
    inner: FisherSnedecor,
}

impl FDist {
    /// Create an F distribution with `d1` and `d2` degrees of freedom.
    /// Returns an error if either is ≤ 0.
    pub fn new(d1: f64, d2: f64) -> SocStatResult<Self> {
        if d1 <= 0.0 || d2 <= 0.0 {
            return Err(SocStatError::Computation(
                "degrees of freedom must be positive".into(),
            ));
        }
        let inner = FisherSnedecor::new(d1, d2)
            .map_err(|e| SocStatError::Computation(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Distribution for FDist {
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
    fn f_cdf_at_zero() {
        let f = FDist::new(3.0, 10.0).unwrap();
        assert_abs_diff_eq!(f.cdf(0.0), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn f_critical_value() {
        // F(3, 10) at α=0.05 → critical ≈ 3.708
        let f = FDist::new(3.0, 10.0).unwrap();
        assert_abs_diff_eq!(f.inverse_cdf(0.95), 3.708, epsilon = 1e-2);
    }

    #[test]
    fn invalid_dof_errors() {
        assert!(FDist::new(0.0, 10.0).is_err());
        assert!(FDist::new(3.0, 0.0).is_err());
    }
}
