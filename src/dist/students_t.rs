//! Student's t distribution.

use statrs::distribution::{Continuous, ContinuousCDF, StudentsT};

use super::Distribution;
use crate::error::{SocStatError, SocStatResult};

/// Student's t distribution with `df` degrees of freedom.
pub struct StudentsTDist {
    inner: StudentsT,
}

impl StudentsTDist {
    /// Create a t distribution with `df` degrees of freedom.
    /// Returns an error if df ≤ 0.
    pub fn new(df: f64) -> SocStatResult<Self> {
        if df <= 0.0 {
            return Err(SocStatError::Computation(
                "df must be positive".into(),
            ));
        }
        let inner = StudentsT::new(0.0, 1.0, df)
            .map_err(|e| SocStatError::Computation(e.to_string()))?;
        Ok(Self { inner })
    }
}

impl Distribution for StudentsTDist {
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
    fn t_cdf_symmetric() {
        let t = StudentsTDist::new(10.0).unwrap();
        assert_abs_diff_eq!(t.cdf(0.0), 0.5, epsilon = 1e-10);
        // P(T ≤ t) + P(T ≤ -t) = 1
        let p = t.cdf(1.5);
        let p_neg = t.cdf(-1.5);
        assert_abs_diff_eq!(p + p_neg, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn t_critical_values() {
        // df=∞ → t = z
        let t_large = StudentsTDist::new(100000.0).unwrap();
        assert_abs_diff_eq!(t_large.inverse_cdf(0.975), 1.96, epsilon = 1e-2);

        // df=10, two-tailed α=0.05 → t_crit ≈ 2.228
        let t = StudentsTDist::new(10.0).unwrap();
        assert_abs_diff_eq!(t.inverse_cdf(0.975), 2.228, epsilon = 1e-2);
    }

    #[test]
    fn invalid_df_errors() {
        assert!(StudentsTDist::new(0.0).is_err());
        assert!(StudentsTDist::new(-1.0).is_err());
    }
}
