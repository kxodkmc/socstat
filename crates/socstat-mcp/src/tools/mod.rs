//! Tool parameter structs and the pure business logic behind each MCP tool.
//!
//! Each submodule defines the [`JsonSchema`] parameter structs consumed by the
//! `#[tool]` handlers in `server.rs` and the `pub fn` helpers those handlers
//! call. Keeping the logic here (instead of inside the macros) keeps the server
//! declaration thin and the analysis code unit-testable.

pub mod anova;
pub mod data;
pub mod describe;
pub mod multivariate;
pub mod normality;
pub mod regression;
pub mod tests;
pub mod transform;

use serde::Serialize;
use serde_json::Value;

/// Serialize any socstat result into a JSON value, mapping serde errors to a
/// user-friendly message.
pub fn to_value<T: Serialize>(v: &T) -> Result<Value, String> {
    serde_json::to_value(v).map_err(|e| format!("failed to serialize result: {e}"))
}