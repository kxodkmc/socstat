//! Descriptive-statistics tools: descriptive, frequencies, crosstab.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::to_value;
use crate::state::SharedState;

/// A single-variable analysis request.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct VarRequest {
    /// Dataset name.
    pub dataset: String,
    /// Variable to analyze.
    pub var: String,
}

/// A two-variable analysis request.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TwoVarRequest {
    /// Dataset name.
    pub dataset: String,
    /// Row variable.
    pub row_var: String,
    /// Column variable.
    pub col_var: String,
}

/// Comprehensive descriptive statistics for a numeric variable.
pub fn descriptive(state: &SharedState, req: VarRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.descriptive(&req.var).map_err(|e| e.to_string())?)
}

/// Frequency table for any variable.
pub fn frequencies(state: &SharedState, req: VarRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.frequencies(&req.var).map_err(|e| e.to_string())?)
}

/// Crosstabulation (contingency table) of two variables.
pub fn crosstab(state: &SharedState, req: TwoVarRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.crosstab(&req.row_var, &req.col_var).map_err(|e| e.to_string())?)
}