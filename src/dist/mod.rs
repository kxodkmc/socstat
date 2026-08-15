//! Statistical distributions — the foundation for all hypothesis tests.
//!
//! Provides a clean [`Distribution`] trait wrapping `statrs` internally,
//! so the rest of the library never depends on statrs' API directly.
//! Each distribution implements `pdf`, `cdf`, and `inverse_cdf`.
//!
//! # Example
//!
//! ```no_run
//! use socstat::dist::{Distribution, NormalDist};
//!
//! let n = NormalDist::standard();
//! assert!((n.cdf(0.0) - 0.5).abs() < 1e-10);
//! assert!((n.inverse_cdf(0.975) - 1.96).abs() < 0.01);
//! ```

mod chi_squared;
mod f_dist;
mod normal;
mod students_t;

pub use chi_squared::ChiSquaredDist;
pub use f_dist::FDist;
pub use normal::NormalDist;
pub use students_t::StudentsTDist;

/// A continuous probability distribution.
pub trait Distribution {
    /// Probability density function at `x`.
    fn pdf(&self, x: f64) -> f64;

    /// Cumulative distribution function: P(X ≤ x).
    fn cdf(&self, x: f64) -> f64;

    /// Inverse CDF (quantile function): returns x such that P(X ≤ x) = p.
    /// For `p` outside the open interval (0, 1) the behavior is
    /// distribution-specific: the underlying quantile may clamp, return NaN,
    /// or panic. Pass only `p` in `(0, 1)` for well-defined results.
    fn inverse_cdf(&self, p: f64) -> f64;
}
