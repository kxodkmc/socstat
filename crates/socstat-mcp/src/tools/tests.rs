//! Hypothesis-testing tools: t-test, ANOVA, chi-square, Mann–Whitney U.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::to_value;
use crate::state::SharedState;

/// A dependent-by-group analysis request.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ByGroupRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric dependent variable.
    pub dep_var: String,
    /// Grouping variable (2 groups for t-test / Mann–Whitney).
    pub group_var: String,
}

/// Independent-samples t-test of `dep_var` between the two groups.
pub fn independent_t_test(state: &SharedState, req: ByGroupRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.independent_t_test(&req.dep_var, &req.group_var).map_err(|e| e.to_string())?)
}

/// One-way ANOVA of `dep_var` across the groups of a factor.
pub fn one_way_anova(state: &SharedState, req: ByGroupRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.one_way_anova(&req.dep_var, &req.group_var).map_err(|e| e.to_string())?)
}

/// A two-variable independence test request.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct TwoVarRequest {
    /// Dataset name.
    pub dataset: String,
    /// First (categorical) variable.
    pub var1: String,
    /// Second (categorical) variable.
    pub var2: String,
}

/// Pearson chi-square test of independence between two categorical variables.
pub fn chi_square_test(state: &SharedState, req: TwoVarRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.chi_square_test(&req.var1, &req.var2).map_err(|e| e.to_string())?)
}

/// Mann–Whitney U test of `dep_var` between two groups.
pub fn mann_whitney_u_test(state: &SharedState, req: ByGroupRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    to_value(&ds.mann_whitney_u_test(&req.dep_var, &req.group_var).map_err(|e| e.to_string())?)
}