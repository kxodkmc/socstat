//! Data-transformation tools: recode, filter, sort, keep, set weight, compute.
//!
//! socstat's transforms are closure-based Rust APIs; these tools bridge that
//! gap by capturing the JSON-provided, data-driven parameters inside the
//! closure, so the whole socstat engine stays intact behind the boundary.

use std::collections::HashMap;

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use socstat::data::RowView;

use crate::state::SharedState;

fn require_mut(state: &SharedState, name: &str) -> Result<socstat::data::Dataset, String> {
    state.require(name)
}

fn persist(state: &SharedState, name: &str, ds: socstat::data::Dataset) -> Result<Value, String> {
    state.replace(name, ds)?;
    super::data::info(state, name)
}

/// One discrete recode rule: `from` maps to `to`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct MappingEntry {
    /// Source numeric value.
    pub from: f64,
    /// Target numeric value.
    pub to: f64,
}

/// Parameters to group a numeric variable into a new variable.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct RecodeRequest {
    /// Dataset name.
    pub dataset: String,
    /// Source numeric variable.
    pub src: String,
    /// New variable to hold the recoded values (source is kept).
    pub dst: String,
    /// Discrete value mapping.
    pub mapping: Vec<MappingEntry>,
}

/// Recode a numeric variable into a new variable via a discrete mapping.
pub fn recode(state: &SharedState, req: RecodeRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    let map: HashMap<u64, f64> = req
        .mapping
        .iter()
        .map(|m| (m.from.to_bits(), m.to))
        .collect();
    ds.recode_into(&req.src, &req.dst, |v| v.and_then(|x| map.get(&x.to_bits()).copied()))
        .map_err(|e| e.to_string())?;
    persist(state, &req.dataset, ds)
}

/// Parameters to keep only rows where `var` satisfies `op value`.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct FilterRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variable to test.
    pub var: String,
    /// Comparison operator: `gt`, `ge`, `lt`, `le`, `eq`, `ne`.
    pub op: String,
    /// Comparison threshold.
    pub value: f64,
}

/// Keep only rows where the predicate holds.
pub fn filter(state: &SharedState, req: FilterRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    let op = req.op.to_ascii_lowercase();
    validate_op(&op)?;
    let kept = ds
        .filter(|row| row.numeric(&req.var).is_some_and(|v| compare(v, &op, req.value)))
        .map_err(|e| e.to_string())?;
    let info = persist(state, &req.dataset, ds)?;
    Ok(json!({ "kept": kept, "dataset": info }))
}

fn validate_op(op: &str) -> Result<(), String> {
    if matches!(op, "gt" | "ge" | "lt" | "le" | "eq" | "ne") {
        Ok(())
    } else {
        Err(format!(
            "unknown comparison operator '{op}'; expected 'gt', 'ge', 'lt', 'le', 'eq', or 'ne'"
        ))
    }
}

fn compare(v: f64, op: &str, threshold: f64) -> bool {
    match op {
        "gt" => v > threshold,
        "ge" => v >= threshold,
        "lt" => v < threshold,
        "le" => v <= threshold,
        "eq" => v == threshold,
        "ne" => v != threshold,
        _ => unreachable!("operator is validated before use"),
    }
}

/// Parameters to reorder rows by a numeric column.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SortRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variable to sort by.
    pub var: String,
    /// Sort descending when true, ascending otherwise.
    #[schemars(default = "default_false")]
    pub descending: bool,
}

fn default_false() -> bool { false }

/// Sort rows by a numeric variable.
pub fn sort(state: &SharedState, req: SortRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    ds.sort_by(&req.var, req.descending).map_err(|e| e.to_string())?;
    persist(state, &req.dataset, ds)
}

/// Parameters to keep only selected columns.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct KeepRequest {
    /// Dataset name.
    pub dataset: String,
    /// Variables to keep (all others dropped).
    pub vars: Vec<String>,
}

/// Drop all columns except those listed.
pub fn keep(state: &SharedState, req: KeepRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    let names: Vec<&str> = req.vars.iter().map(|s| s.as_str()).collect();
    ds.keep(&names).map_err(|e| e.to_string())?;
    persist(state, &req.dataset, ds)
}

/// Parameters to set the case-weight variable.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct SetWeightRequest {
    /// Dataset name.
    pub dataset: String,
    /// Numeric variable to use as the frequency weight.
    pub var: String,
}

/// Set the case-weight variable. There is no way to clear a weight once set
/// (re-set it to another variable, or reload the dataset).
pub fn set_weight(state: &SharedState, req: SetWeightRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    ds.set_weight(&req.var).map_err(|e| e.to_string())?;
    persist(state, &req.dataset, ds)
}

/// An operand in [`ComputeRequest`]: either a column name or a constant.
#[derive(Debug, Clone, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Operand {
    /// Reference a numeric column by name.
    Column(String),
    /// A literal numeric constant.
    Constant(f64),
}

fn operand_value(op: &Operand, row: &RowView) -> Option<f64> {
    match op {
        Operand::Column(name) => row.numeric(name),
        Operand::Constant(n) => Some(*n),
    }
}

/// Parameters to compute a new numeric variable elementwise.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ComputeRequest {
    /// Dataset name.
    pub dataset: String,
    /// Name of the new variable.
    pub new_var: String,
    /// Left operand (column or constant).
    pub left: Operand,
    /// Operator: `+`, `-`, `*`, `/`.
    pub operator: String,
    /// Right operand (column or constant).
    pub right: Operand,
}

/// Create a new numeric variable as `left op right` elementwise.
pub fn compute(state: &SharedState, req: ComputeRequest) -> Result<Value, String> {
    let mut ds = require_mut(state, &req.dataset)?;
    let (left, right, op) = (req.left, req.right, req.operator.clone());
    if !matches!(op.as_str(), "+" | "-" | "*" | "/") {
        return Err(format!(
            "unknown operator '{op}'; expected '+', '-', '*', or '/'"
        ));
    }
    ds.compute(&req.new_var, |row| {
        let a = operand_value(&left, row)?;
        let b = operand_value(&right, row)?;
        match op.as_str() {
            "+" => Some(a + b),
            "-" => Some(a - b),
            "*" => Some(a * b),
            "/" => Some(a / b),
            _ => None,
        }
    })
    .map_err(|e| e.to_string())?;
    persist(state, &req.dataset, ds)
}