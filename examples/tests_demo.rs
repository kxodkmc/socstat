//! Example: P3 hypothesis tests — t-test, ANOVA, chi-square, Mann–Whitney U.
//!
//! Builds a dataset (and round-trips it through CSV), then runs every test and
//! prints the fully serialized JSON results — the shape hosts receive.
//!
//! Run with: `cargo run --example tests_demo`

use socstat::prelude::*;

fn main() -> SocStatResult<()> {
    // --- Build a dataset in memory, mirroring the ToothGrowth style ---
    let mut ds = Dataset::new();
    ds.add_var(Variable::text("supp").label("Supplement type"))?;
    ds.add_var(Variable::numeric("dose").label("Dose (mg/day)").measure(MeasureType::Scale))?;
    ds.add_var(Variable::numeric("len").label("Tooth length").measure(MeasureType::Scale))?;
    ds.add_var(Variable::text("outcome").label("Outcome"))?;
    ds.add_var(Variable::numeric("w").label("Case weight").weight())?;

    let data: &[(&str, f64, f64, &str)] = &[
        // (supp, dose, len, outcome)
        ("VC", 0.5, 4.2, "short"),
        ("VC", 0.5, 11.5, "short"),
        ("VC", 0.5, 7.3, "short"),
        ("VC", 1.0, 16.5, "short"),
        ("VC", 1.0, 15.2, "short"),
        ("VC", 2.0, 33.9, "long"),
        ("VC", 2.0, 32.5, "long"),
        ("VC", 2.0, 26.4, "long"),
        ("OJ", 0.5, 15.2, "short"),
        ("OJ", 0.5, 21.5, "long"),
        ("OJ", 1.0, 25.5, "long"),
        ("OJ", 1.0, 26.4, "long"),
        ("OJ", 2.0, 29.5, "long"),
        ("OJ", 2.0, 30.9, "long"),
    ];
    for (supp, dose, len, outcome) in data {
        ds.push_row(vec![
            Value::Text((*supp).into()),
            Value::Number(*dose),
            Value::Number(*len),
            Value::Text((*outcome).into()),
            Value::Number(1.0),
        ])?;
    }

    // --- Honor the "load a CSV" workflow: write then read back ---
    let csv_path = std::env::temp_dir().join("socstat_tests_demo.csv");
    socstat::write().csv(&ds, &csv_path)?;
    let ds = socstat::read().csv(&csv_path)?;
    println!("Dataset: {} vars, {} rows (loaded from {})\n", ds.n_vars(), ds.n_rows(), csv_path.display());

    // --- Independent-samples t-test: len by supp ---
    let t = ds.independent_t_test("len", "supp")?;
    println!("=== Independent t-test: len by supp ===");
    for g in &t.group_stats {
        println!("  {}  n={:.1}  mean={:.3}  sd={:.3}", g.label, g.n, g.mean, g.std_dev);
    }
    println!("  Levene F={:.4}  p={:.4}", t.levene_test.f_statistic, t.levene_test.p_value);
    println!("  pooled:   t={:.4}  df={:.1}  p={:.4}", t.equal_variances.t_statistic, t.equal_variances.df, t.equal_variances.p_value);
    println!("  Welch:    t={:.4}  df={:.1}  p={:.4}", t.unequal_variances.t_statistic, t.unequal_variances.df, t.unequal_variances.p_value);
    println!("  {}\n", serde_json::to_string_pretty(&t).unwrap());

    // --- One-way ANOVA: len by dose (3 groups) ---
    let a = ds.one_way_anova("len", "dose")?;
    println!("=== One-way ANOVA: len by dose ===");
    for g in &a.group_stats {
        println!("  dose {}  n={:.1}  mean={:.3}", g.label, g.n, g.mean);
    }
    println!("  F({:.0},{:.0}) = {:.4}  p = {:.4}  η² = {:.4}", a.between_groups.df, a.within_groups.df, a.f_statistic, a.p_value, a.eta_squared);
    println!("  {}\n", serde_json::to_string_pretty(&a).unwrap());

    // --- Chi-square: supp x outcome ---
    let c = ds.chi_square_test("supp", "outcome")?;
    println!("=== Chi-square: supp x outcome ===");
    println!("  χ²({:.0}) = {:.4}  p = {:.4}  n = {:.1}", c.df, c.chi_square, c.p_value, c.n);
    println!("  {}\n", serde_json::to_string_pretty(&c).unwrap());

    // --- Mann–Whitney U: len by supp ---
    let m = ds.mann_whitney_u_test("len", "supp")?;
    println!("=== Mann–Whitney U: len by supp ===");
    for g in &m.group_stats {
        println!("  {}  n={:.1}  mean_rank={:.3}", g.label, g.n, g.mean_rank);
    }
    println!("  U = {:.1}  z = {:.4}  p = {:.4}  (ties: {})", m.u_statistic, m.z_score, m.p_value, m.has_ties);
    println!("  {}\n", serde_json::to_string_pretty(&m).unwrap());

    Ok(())
}
