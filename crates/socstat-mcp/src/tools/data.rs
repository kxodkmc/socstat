//! Dataset management tools: load, inspect, preview, and drop datasets.

use rmcp::schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{json, Value};

use crate::state::SharedState;

/// Parameters to load a dataset from a file into shared state.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct LoadRequest {
    /// Name to store the dataset under (referenced by later tools).
    pub name: String,
    /// Path to a `.csv` or `.json` file (format by extension; `.sav` needs the
    /// `sav` feature enabled at build time).
    pub path: String,
}

/// A dataset name reference.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct ByName {
    /// Dataset name.
    pub dataset: String,
}

/// Parameters for previewing the first rows of a dataset.
#[derive(Debug, Deserialize, JsonSchema)]
pub struct PreviewRequest {
    /// Dataset name.
    pub dataset: String,
    /// Maximum number of rows to return.
    #[schemars(default = "default_rows")]
    pub rows: usize,
}

fn default_rows() -> usize { 10 }

/// List all loaded datasets with their dimensions.
pub fn list(state: &SharedState) -> Value {
    let datasets: Vec<Value> = state
        .names()
        .iter()
        .map(|name| {
            let ds = state.require(name).unwrap_or_default();
            json!({ "name": name, "n_rows": ds.n_rows(), "n_vars": ds.n_vars() })
        })
        .collect();
    json!({ "datasets": datasets })
}

/// Load a file into shared state and return its schema.
pub fn load(state: &SharedState, req: LoadRequest) -> Result<Value, String> {
    let ds = socstat::read()
        .auto(&req.path)
        .map_err(|e| format!("failed to load '{}': {e}", req.path))?;
    state.load(req.name.clone(), ds);
    info(state, &req.name)
}

/// Describe the variables and shape of a dataset.
pub fn info(state: &SharedState, name: &str) -> Result<Value, String> {
    let ds = state.require(name)?;
    let variables: Vec<Value> = ds
        .variables()
        .iter()
        .map(|v| {
            let n_valid = ds.n_valid(&v.name).unwrap_or(0);
            json!({
                "name": v.name,
                "label": v.label,
                "data_type": format!("{:?}", v.data_type),
                "measure": format!("{:?}", v.measure),
                "n_valid": n_valid,
                "n_missing": ds.n_rows().saturating_sub(n_valid),
            })
        })
        .collect();
    let weight = ds
        .weight_var_index()
        .and_then(|i| ds.variables().get(i))
        .map(|v| v.name.clone());
    Ok(json!({
        "name": name,
        "n_rows": ds.n_rows(),
        "n_vars": ds.n_vars(),
        "weight_var": weight,
        "variables": variables,
    }))
}

/// Preview the first `rows` rows of a dataset as typed cell values.
pub fn preview(state: &SharedState, req: PreviewRequest) -> Result<Value, String> {
    let ds = state.require(&req.dataset)?;
    let columns: Vec<&str> = ds.var_names().collect();
    let limit = req.rows.min(ds.n_rows());
    let mut rows = Vec::with_capacity(limit);
    for i in 0..limit {
        let mut row = Vec::with_capacity(columns.len());
        for col in &columns {
            let cell = ds
                .column_by_name(col)
                .ok()
                .and_then(|c| c.get_value(i));
            row.push(match cell {
                Some(socstat::data::Value::Number(n)) => json!(n),
                Some(socstat::data::Value::Text(s)) => json!(s),
                _ => Value::Null,
            });
        }
        rows.push(Value::Array(row));
    }
    Ok(json!({
        "dataset": req.dataset,
        "columns": columns,
        "rows": rows,
        "n_rows_shown": limit,
        "n_rows_total": ds.n_rows(),
    }))
}

/// Drop a dataset from shared state.
pub fn drop(state: &SharedState, name: &str) -> Result<Value, String> {
    if state.remove(name) {
        Ok(json!({ "removed": name }))
    } else {
        Err(format!("dataset '{name}' not found"))
    }
}