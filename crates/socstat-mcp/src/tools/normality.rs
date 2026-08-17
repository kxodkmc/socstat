//! Normality-testing tools: Shapiro–Wilk and one-sample Kolmogorov–Smirnov
//! (including the Lilliefors variant).

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::describe::VarRequest;
use super::to_value;
use crate::state::SharedState;

/// Parameters for a one-sample Kolmogorov–Smirnov normality test.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KsRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variable to test.
    pub var: String,
    /// Variant: `lilliefors` (μ, σ estimated from the sample) or `one_sample`
    /// (test against a fully specified normal, using `mean` / `std_dev`).
    #[schemars(default = "default_test_type")]
    pub test_type: String,
    /// Mean of the hypothesized normal when `test_type` is `one_sample`.
    #[schemars(default = "default_zero")]
    pub mean: f64,
    /// Standard deviation of the hypothesized normal when `test_type` is
    /// `one_sample`.
    #[schemars(default = "default_one")]
    pub std_dev: f64,
}

fn default_test_type() -> String { "lilliefors".into() }
fn default_zero() -> f64 { 0.0 }
fn default_one() -> f64 { 1.0 }

fn parse_ks_type(s: &str, mean: f64, std_dev: f64) -> Result<KsTestType, String> {
    match s.to_ascii_lowercase().as_str() {
        "lilliefors" => Ok(KsTestType::Lilliefors),
        "one_sample" | "onesample" | "one-sample" => {
            Ok(KsTestType::OneSample { mean, std_dev })
        }
        other => Err(format!(
            "unknown test_type '{other}'; expected 'lilliefors' or 'one_sample'"
        )),
    }
}

/// Shapiro–Wilk test of normality for a numeric variable (Royston AS R94).
pub fn shapiro_wilk(state: &SharedState, req: VarRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.shapiro_wilk(&req.var).map_err(|e| e.to_string())?)
}

/// One-sample Kolmogorov–Smirnov normality test (Lilliefors or against a
/// fully-specified normal).
pub fn ks_normality_test(state: &SharedState, req: KsRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let test_type = parse_ks_type(&req.test_type, req.mean, req.std_dev)?;
    to_value(&ds.ks_normality_test(&req.var, test_type).map_err(|e| e.to_string())?)
}