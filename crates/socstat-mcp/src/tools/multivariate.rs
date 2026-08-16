//! Multivariate tools: PCA and Cronbach's alpha reliability.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::to_value;
use crate::state::SharedState;

/// A list-of-variables analysis request.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarsRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variables to analyze.
    pub vars: Vec<String>,
}

/// PCA over a set of numeric variables.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PcaRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variables.
    pub vars: Vec<String>,
    /// Analysis matrix: `correlation` (default) or `covariance`.
    pub matrix: String,
}

fn parse_matrix(s: &str) -> Result<PcaMatrix, String> {
    match s.to_ascii_lowercase().as_str() {
        "correlation" => Ok(PcaMatrix::Correlation),
        "covariance" => Ok(PcaMatrix::Covariance),
        other => Err(format!("unknown analysis matrix '{other}'; expected 'correlation' or 'covariance'")),
    }
}

/// Principal component analysis of the given variables.
pub fn pca(state: &SharedState, req: PcaRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let matrix = parse_matrix(&req.matrix)?;
    let vars: Vec<&str> = req.vars.iter().map(|s| s.as_str()).collect();
    to_value(&ds.pca(&vars, matrix).map_err(|e| e.to_string())?)
}

/// Cronbach's alpha reliability of the given scale items.
pub fn reliability(state: &SharedState, req: VarsRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let vars: Vec<&str> = req.vars.iter().map(|s| s.as_str()).collect();
    to_value(&ds.reliability(&vars).map_err(|e| e.to_string())?)
}