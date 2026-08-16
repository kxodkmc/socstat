//! `socstat-mcp` — an AI-friendly MCP server built on [`socstat`].
//!
//! Exposes socstat's statistical analyses as Model Context Protocol tools over
//! stdio. Datasets are held in shared, stateful storage so an AI host loads
//! data once and references it by name in every analysis tool.
//!
//! # Embedding
//!
//! ```no_run
//! use socstat_mcp::{SharedState, SocstatMcpServer};
//! use rmcp::ServiceExt;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let server = SocstatMcpServer::new(SharedState::arc());
//! let running = server.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
//! running.waiting().await?;
//! # Ok(())
//! # }
//! ```

pub mod server;
pub mod state;
mod tools;

pub use server::SocstatMcpServer;
pub use state::SharedState;

#[cfg(test)]
mod tests {
    use socstat::data::{Dataset, Value, Variable};

    use crate::tools::{data, describe, transform};
    use crate::SharedState;

    fn sample_state() -> SharedState {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("age").label("Age")).unwrap();
        ds.add_var(
            Variable::text("gender").value_label("M", "Male").value_label("F", "Female"),
        ).unwrap();
        ds.add_var(Variable::numeric("score")).unwrap();
        for (age, gender, score) in [(25.0, "M", 85.0), (30.0, "F", 92.0), (35.0, "M", 78.0), (28.0, "F", 95.0)] {
            ds.push_row(vec![
                Value::Number(age),
                Value::Text(gender.into()),
                Value::Number(score),
            ]).unwrap();
        }
        let state = SharedState::new();
        state.load("sample".into(), ds);
        state
    }

    #[test]
    fn info_lists_all_variables() {
        let info = data::info(&sample_state(), "sample").unwrap();
        assert_eq!(info["n_rows"], 4);
        assert_eq!(info["n_vars"], 3);
        assert_eq!(info["variables"].as_array().unwrap().len(), 3);
        assert_eq!(info["variables"][0]["name"], "age");
    }

    #[test]
    fn require_missing_dataset_errors() {
        let state = sample_state();
        assert!(data::info(&state, "nope").is_err());
    }

    #[test]
    fn descriptive_reports_mean() {
        let out = describe::descriptive(
            &sample_state(),
            describe::VarRequest { dataset: "sample".into(), var: "age".into() },
        ).unwrap();
        assert!((out["mean"].as_f64().unwrap() - 29.5).abs() < 1e-9);
        assert_eq!(out["n"].as_f64().unwrap(), 4.0);
    }

    #[test]
    fn recode_creates_new_variable() {
        let state = sample_state();
        let req = transform::RecodeRequest {
            dataset: "sample".into(),
            src: "age".into(),
            dst: "adult".into(),
            mapping: vec![
                transform::MappingEntry { from: 25.0, to: 1.0 },
                transform::MappingEntry { from: 28.0, to: 2.0 },
                transform::MappingEntry { from: 30.0, to: 2.0 },
                transform::MappingEntry { from: 35.0, to: 3.0 },
            ],
        };
        let info = transform::recode(&state, req).unwrap();
        let names: Vec<&str> = info["variables"]
            .as_array().unwrap()
            .iter().filter_map(|v| v["name"].as_str())
            .collect();
        assert!(names.contains(&"adult"));
    }

    #[test]
    fn filter_keeps_matching_rows() {
        let state = sample_state();
        let req = transform::FilterRequest {
            dataset: "sample".into(),
            var: "age".into(),
            op: "ge".into(),
            value: 30.0,
        };
        let out = transform::filter(&state, req).unwrap();
        assert_eq!(out["kept"], 2);
        assert_eq!(out["dataset"]["n_rows"], 2);
    }

    #[test]
    fn compute_creates_arithmetic_variable() {
        let state = sample_state();
        let req = transform::ComputeRequest {
            dataset: "sample".into(),
            new_var: "double_age".into(),
            left: transform::Operand::Column("age".into()),
            operator: "*".into(),
            right: transform::Operand::Constant(2.0),
        };
        let info = transform::compute(&state, req).unwrap();
        let names: Vec<&str> = info["variables"]
            .as_array().unwrap()
            .iter().filter_map(|v| v["name"].as_str())
            .collect();
        assert!(names.contains(&"double_age"));
    }

    // --- Verification tests for the reviewed issues -------------------------

    // A typo like "gre" must be rejected, not silently drop every row (which
    // would leave an empty dataset with no error signal).
    #[test]
    fn filter_invalid_op_errors() {
        let req = transform::FilterRequest {
            dataset: "sample".into(),
            var: "age".into(),
            op: "gre".into(), // typo for "ge"
            value: 30.0,
        };
        assert!(transform::filter(&sample_state(), req).is_err());
    }

    // Every advertised operator must still be accepted.
    #[test]
    fn filter_all_valid_ops_succeed() {
        for op in ["gt", "ge", "lt", "le", "eq", "ne"] {
            let req = transform::FilterRequest {
                dataset: "sample".into(),
                var: "age".into(),
                op: op.into(),
                value: 30.0,
            };
            assert!(transform::filter(&sample_state(), req).is_ok(), "op '{op}' should succeed");
        }
    }

    // An unsupported operator must not silently produce an all-missing column.
    #[test]
    fn compute_invalid_operator_errors() {
        let req = transform::ComputeRequest {
            dataset: "sample".into(),
            new_var: "bad".into(),
            left: transform::Operand::Column("age".into()),
            operator: "^".into(), // not in { + - * / }
            right: transform::Operand::Constant(2.0),
        };
        assert!(transform::compute(&sample_state(), req).is_err());
    }
}