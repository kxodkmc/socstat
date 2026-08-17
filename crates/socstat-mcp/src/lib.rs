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

    use crate::tools::{anova, data, describe, multivariate, normality, regression, tests, transform};
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

    // --- Verification tests for the newly exposed analysis tools -----------

    fn paired_state() -> SharedState {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("pre")).unwrap();
        ds.add_var(Variable::numeric("post")).unwrap();
        for (pre, post) in [(5.0, 6.0), (6.0, 7.0), (7.0, 9.0), (4.0, 5.0), (8.0, 8.0)] {
            ds.push_row(vec![Value::Number(pre), Value::Number(post)]).unwrap();
        }
        let state = SharedState::new();
        state.load("pair".into(), ds);
        state
    }

    #[test]
    fn paired_t_test_reports_mean_difference() {
        let req = tests::TwoVarRequest {
            dataset: "pair".into(),
            var1: "pre".into(),
            var2: "post".into(),
        };
        let out = tests::paired_t_test(&paired_state(), req).unwrap();
        assert!(out["n"].as_f64().unwrap() > 0.0);
        assert!(out["t_statistic"].as_f64().unwrap().is_finite());
        assert!(out["p_value"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn wilcoxon_signed_rank_needs_ten_differences() {
        // Only 5 paired rows -> fewer than 10 non-zero differences -> error.
        let req = tests::TwoVarRequest {
            dataset: "pair".into(),
            var1: "pre".into(),
            var2: "post".into(),
        };
        assert!(tests::wilcoxon_signed_rank_test(&paired_state(), req).is_err());
    }

    #[test]
    fn fisher_exact_accepts_valid_alternative() {
        let mut ds = Dataset::new();
        ds.add_var(Variable::text("g").value_label("A", "A").value_label("B", "B")).unwrap();
        ds.add_var(Variable::text("h").value_label("X", "X").value_label("Y", "Y")).unwrap();
        for (g, h) in [("A", "X"), ("A", "Y"), ("B", "X"), ("B", "Y"), ("A", "X")] {
            ds.push_row(vec![Value::Text(g.into()), Value::Text(h.into())]).unwrap();
        }
        let state = SharedState::new();
        state.load("t".into(), ds);
        let req = tests::FisherRequest {
            dataset: "t".into(),
            var1: "g".into(),
            var2: "h".into(),
            alternative: "two-sided".into(),
        };
        let out = tests::fisher_exact_test(&state, req).unwrap();
        assert!(out["odds_ratio"].as_f64().unwrap().is_finite());
        assert!(out["p_value_two_sided"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn fisher_exact_rejects_bad_alternative() {
        let state = paired_state();
        let req = tests::FisherRequest {
            dataset: "pair".into(),
            var1: "pre".into(),
            var2: "post".into(),
            alternative: "sideways".into(), // invalid
        };
        assert!(tests::fisher_exact_test(&state, req).is_err());
    }

    fn grouped_state() -> SharedState {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("score")).unwrap();
        ds.add_var(Variable::text("group")).unwrap();
        for (score, group) in [
            (1.0, "a"), (2.0, "a"), (3.0, "a"),
            (11.0, "b"), (12.0, "b"), (13.0, "b"),
            (21.0, "c"), (22.0, "c"), (23.0, "c"),
        ] {
            ds.push_row(vec![Value::Number(score), Value::Text(group.into())]).unwrap();
        }
        let state = SharedState::new();
        state.load("grp".into(), ds);
        state
    }

    #[test]
    fn kruskal_wallis_reports_groups() {
        let req = tests::ByGroupRequest {
            dataset: "grp".into(),
            dep_var: "score".into(),
            group_var: "group".into(),
        };
        let out = tests::kruskal_wallis_test(&grouped_state(), req).unwrap();
        assert_eq!(out["group_stats"].as_array().unwrap().len(), 3);
        assert!(out["h_statistic"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn shapiro_wilk_reports_w_statistic() {
        let state = grouped_state();
        let req = describe::VarRequest { dataset: "grp".into(), var: "score".into() };
        let out = normality::shapiro_wilk(&state, req).unwrap();
        assert!(out["w_statistic"].as_f64().unwrap() > 0.0);
        assert!(out["p_value"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn ks_normality_supports_both_modes() {
        let state = grouped_state();
        let lillie = normality::KsRequest {
            dataset: "grp".into(),
            var: "score".into(),
            test_type: "lilliefors".into(),
            mean: 0.0,
            std_dev: 1.0,
        };
        assert!(normality::ks_normality_test(&state, lillie).unwrap()["d_statistic"].as_f64().unwrap() > 0.0);
        let one = normality::KsRequest {
            dataset: "grp".into(),
            var: "score".into(),
            test_type: "one_sample".into(),
            mean: 10.0,
            std_dev: 8.0,
        };
        assert!(normality::ks_normality_test(&state, one).is_ok());
        // Invalid test_type must error, not silently pass.
        let bad = normality::KsRequest {
            dataset: "grp".into(),
            var: "score".into(),
            test_type: "watson".into(),
            mean: 0.0,
            std_dev: 1.0,
        };
        assert!(normality::ks_normality_test(&state, bad).is_err());
    }

    #[test]
    fn post_hoc_returns_comparisons() {
        let req = anova::PostHocRequest {
            dataset: "grp".into(),
            dep_var: "score".into(),
            factor_var: "group".into(),
            method: "tukey".into(),
        };
        let out = anova::post_hoc(&grouped_state(), req).unwrap();
        assert_eq!(out["comparisons"].as_array().unwrap().len(), 3); // 3 pairs
        assert_eq!(out["n_groups"].as_u64().unwrap(), 3);
        assert!(out["comparisons"][0]["p_value"].as_f64().unwrap().is_finite());
    }

    #[test]
    fn post_hoc_distinguishes_valid_and_invalid_methods() {
        let state = grouped_state();
        let ok = anova::PostHocRequest {
            dataset: "grp".into(),
            dep_var: "score".into(),
            factor_var: "group".into(),
            method: "bonferroni".into(),
        };
        assert!(anova::post_hoc(&state, ok).is_ok());
        let bad = anova::PostHocRequest {
            dataset: "grp".into(),
            dep_var: "score".into(),
            factor_var: "group".into(),
            method: "fisher".into(),
        };
        assert!(anova::post_hoc(&state, bad).is_err());
    }

    fn factorial_state() -> SharedState {
        let mut ds = Dataset::new();
        ds.add_var(Variable::text("a")).unwrap();
        ds.add_var(Variable::text("b")).unwrap();
        ds.add_var(Variable::numeric("y")).unwrap();
        for (a, b, y) in [
            ("1", "1", 10.0), ("1", "1", 11.0),
            ("1", "2", 20.0), ("1", "2", 21.0),
            ("2", "1", 30.0), ("2", "1", 31.0),
            ("2", "2", 40.0), ("2", "2", 41.0),
        ] {
            ds.push_row(vec![Value::Text(a.into()), Value::Text(b.into()), Value::Number(y)])
                .unwrap();
        }
        let state = SharedState::new();
        state.load("fact".into(), ds);
        state
    }

    #[test]
    fn factorial_anova_reports_effects() {
        let req = anova::FactorialAnovaRequest {
            dataset: "fact".into(),
            dep_var: "y".into(),
            factors: vec!["a".into(), "b".into()],
            ss_type: "type_ii".into(),
        };
        let out = anova::factorial_anova(&factorial_state(), req).unwrap();
        let effects = out["effects"].as_array().unwrap();
        assert!(effects.iter().any(|e| e["source"] == "Error"));
        assert!(out["r_squared"].as_f64().unwrap() > 0.0);
        // Invalid ss_type must error.
        let bad = anova::FactorialAnovaRequest {
            dataset: "fact".into(),
            dep_var: "y".into(),
            factors: vec!["a".into(), "b".into()],
            ss_type: "type_iii".into(),
        };
        assert!(anova::factorial_anova(&factorial_state(), bad).is_err());
    }

    fn regression_state() -> SharedState {
        let mut ds = Dataset::new();
        ds.add_var(Variable::numeric("y")).unwrap();
        ds.add_var(Variable::numeric("x")).unwrap();
        ds.add_var(Variable::numeric("c")).unwrap();
        ds.add_var(Variable::numeric("d")).unwrap();
        let u = [0.3, -0.4, 0.5, -0.2, 0.3, -0.5, 0.4, -0.3];
        let v = [-0.3, 0.5, -0.4, 0.6, -0.5, 0.4, -0.6, 0.5];
        for i in 1..=8 {
            let c = i as f64;
            ds.push_row(vec![
                Value::Number(c + 1.3 + v[i - 1]),
                Value::Number(c + 1.3 + u[i - 1]),
                Value::Number(c),
                Value::Number((i % 3) as f64),
            ])
            .unwrap();
        }
        let state = SharedState::new();
        state.load("reg".into(), ds);
        state
    }

    #[test]
    fn vif_returns_one_row_per_predictor() {
        let req = multivariate::VarsRequest {
            dataset: "reg".into(),
            vars: vec!["x".into(), "c".into(), "d".into()],
        };
        let out = regression::vif(&regression_state(), req).unwrap();
        let rows = out.as_array().unwrap();
        assert_eq!(rows.len(), 3);
        for r in rows {
            assert!(r["vif"].as_f64().unwrap() >= 1.0);
            assert!(r["vif"].as_f64().unwrap().is_finite());
        }
    }

    #[test]
    fn partial_correlation_controls_for_confounder() {
        let req = regression::PartialCorrRequest {
            dataset: "reg".into(),
            var1: "y".into(),
            var2: "x".into(),
            controls: vec!["c".into()],
            method: "pearson".into(),
        };
        let out = regression::partial_correlation(&regression_state(), req).unwrap();
        assert_eq!(out["controlling_for"], serde_json::json!(["c"]));
        assert!(out["coefficient"].as_f64().unwrap().abs() <= 1.0);
        // Empty controls list must error (core expects at least one).
        let bad = regression::PartialCorrRequest {
            dataset: "reg".into(),
            var1: "y".into(),
            var2: "x".into(),
            controls: vec![],
            method: "pearson".into(),
        };
        assert!(regression::partial_correlation(&regression_state(), bad).is_err());
    }
}