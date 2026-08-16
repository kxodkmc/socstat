//! Correlation and regression tools: linear OLS, logistic, correlations.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::to_value;
use crate::state::SharedState;

/// Correlation between two variables.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorrelationRequest {
    /// Dataset name.
    pub dataset: String,
    /// First numeric variable.
    pub var1: String,
    /// Second numeric variable.
    pub var2: String,
    /// Correlation method: `pearson`, `spearman`, or `kendall`.
    pub method: String,
}

/// Correlation matrix across several variables.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct CorrelationMatrixRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variables (at least two).
    pub vars: Vec<String>,
    /// Correlation method: `pearson`, `spearman`, or `kendall`.
    pub method: String,
}

/// Parameters for a regression fit.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RegressionRequest {
    /// Dataset name.
    pub dataset: String,
    /// Dependent variable.
    pub dep_var: String,
    /// Independent variables (intercept is always included).
    pub indep_vars: Vec<String>,
}

fn parse_method(s: &str) -> Result<CorrelationMethod, String> {
    match s.to_ascii_lowercase().as_str() {
        "pearson" => Ok(CorrelationMethod::Pearson),
        "spearman" => Ok(CorrelationMethod::Spearman),
        "kendall" => Ok(CorrelationMethod::Kendall),
        other => Err(format!(
            "unknown correlation method '{other}'; expected 'pearson', 'spearman', or 'kendall'"
        )),
    }
}

/// Pearson, Spearman, or Kendall correlation between two variables.
pub fn correlation_pair(state: &SharedState, req: CorrelationRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let method = parse_method(&req.method)?;
    to_value(&ds.correlation_pair(&req.var1, &req.var2, method).map_err(|e| e.to_string())?)
}

/// Correlation between every pair of the given variables.
pub fn correlation(state: &SharedState, req: CorrelationMatrixRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let method = parse_method(&req.method)?;
    let vars: Vec<&str> = req.vars.iter().map(|s| s.as_str()).collect();
    to_value(&ds.correlation(&vars, method).map_err(|e| e.to_string())?)
}

/// Fit a linear regression (OLS) of `dep_var` on `indep_vars`.
pub fn linear_regression(state: &SharedState, req: RegressionRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let vars: Vec<&str> = req.indep_vars.iter().map(|s| s.as_str()).collect();
    to_value(&ds.regression(&req.dep_var, &vars).map_err(|e| e.to_string())?)
}

/// Fit a binary logistic regression of a 0/1 outcome.
pub fn logistic_regression(state: &SharedState, req: RegressionRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let vars: Vec<&str> = req.indep_vars.iter().map(|s| s.as_str()).collect();
    to_value(&ds.logistic_regression(&req.dep_var, &vars).map_err(|e| e.to_string())?)
}