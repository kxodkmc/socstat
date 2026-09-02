//! ANOVA tools: post-hoc comparisons over a factor, and multifactor (factorial)
//! ANOVA with Type I / Type II sums of squares.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

use socstat::prelude::*;

use super::to_value;
use crate::state::SharedState;

/// Parameters for a post-hoc comparison.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PostHocRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric dependent variable.
    pub dep_var: String,
    /// Factor variable defining the groups (2+ groups).
    pub factor_var: String,
    /// Method: `bonferroni`, `tukey`, `scheffe`, or `games_howell`.
    #[schemars(default = "default_method")]
    pub method: String,
}

fn default_method() -> String { "bonferroni".into() }

fn parse_method(s: &str) -> Result<PostHocMethod, String> {
    match s.to_ascii_lowercase().as_str() {
        "bonferroni" => Ok(PostHocMethod::Bonferroni),
        "tukey" => Ok(PostHocMethod::Tukey),
        "scheffe" | "scheffé" => Ok(PostHocMethod::Scheffe),
        "games_howell" | "games-howell" | "gameshowell" => Ok(PostHocMethod::GamesHowell),
        other => Err(format!(
            "unknown post-hoc method '{other}'; expected 'bonferroni', 'tukey', 'scheffe', or 'games_howell'"
        )),
    }
}

/// Parameters for a multifactor (factorial) ANOVA.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FactorialAnovaRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric dependent variable.
    pub dep_var: String,
    /// Two or more factor variables (only two-way interactions are modelled).
    pub factors: Vec<String>,
    /// Sums-of-squares type: `type_i` (sequential) or `type_ii` (marginal).
    #[schemars(default = "default_ss_type")]
    pub ss_type: String,
}

fn default_ss_type() -> String { "type_ii".into() }

fn parse_ss_type(s: &str) -> Result<SsType, String> {
    match s.to_ascii_lowercase().replace('-', "_").as_str() {
        "type_i" | "type1" => Ok(SsType::TypeI),
        "type_ii" | "type2" => Ok(SsType::TypeII),
        other => Err(format!(
            "unknown ss_type '{other}'; expected 'type_i' or 'type_ii'"
        )),
    }
}

/// ANOVA post-hoc comparisons of `dep_var` across the groups of `factor_var`
/// (Bonferroni / Tukey HSD / Scheffé / Games–Howell).
pub fn post_hoc(state: &SharedState, req: PostHocRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let method = parse_method(&req.method)?;
    to_value(&ds.post_hoc(&req.dep_var, &req.factor_var, method).map_err(|e| e.to_string())?)
}

/// Multifactor (factorial) ANOVA of `dep_var` on two or more factors, with
/// Type I or Type II sums of squares and two-way interactions.
pub fn factorial_anova(state: &SharedState, req: FactorialAnovaRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let ss_type = parse_ss_type(&req.ss_type)?;
    let factors: Vec<&str> = req.factors.iter().map(|s| s.as_str()).collect();
    to_value(&ds.factorial_anova(&req.dep_var, &factors, ss_type).map_err(|e| e.to_string())?)
}