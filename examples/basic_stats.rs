//! Example: dataset, transformations, descriptive stats, frequencies, crosstabs.
//!
//! Run with: `cargo run --example basic_stats`

use socstat::prelude::*;

fn main() -> SocStatResult<()> {
    // --- Build a dataset ---
    let mut ds = Dataset::new();
    ds.add_var(Variable::numeric("age").label("Age in years"))?;
    ds.add_var(
        Variable::text("gender")
            .label("Gender")
            .value_label("M", "Male")
            .value_label("F", "Female"),
    )?;
    ds.add_var(Variable::numeric("score").label("Test score"))?;

    ds.push_row(vec![Value::Number(25.0), Value::Text("M".into()), Value::Number(85.0)])?;
    ds.push_row(vec![Value::Number(30.0), Value::Text("F".into()), Value::Number(92.0)])?;
    ds.push_row(vec![Value::Number(35.0), Value::Text("M".into()), Value::Missing])?;
    ds.push_row(vec![Value::Number(40.0), Value::Text("F".into()), Value::Number(78.0)])?;
    ds.push_row(vec![Value::Number(28.0), Value::Text("M".into()), Value::Number(95.0)])?;
    ds.push_row(vec![Value::Missing, Value::Text("F".into()), Value::Number(88.0)])?;

    println!("=== Dataset: {} vars, {} rows ===\n", ds.n_vars(), ds.n_rows());

    // --- Descriptive statistics ---
    println!("--- Descriptive: age ---");
    let d = ds.descriptive("age")?;
    println!("  n={:.0}  mean={:.2}  std={:.2}  median={:.1}", d.n, d.mean, d.std_dev, d.median);
    println!("  min={:.1}  max={:.1}  range={:.1}", d.min, d.max, d.range);
    println!("  Q1={:.1}  Q3={:.1}  IQR={:.1}", d.q1, d.q3, d.q3 - d.q1);
    println!("  skew={:.3}  kurt={:.3}", d.skewness, d.kurtosis);
    println!("  95% CI: [{:.2}, {:.2}]\n", d.ci_95.0, d.ci_95.1);

    println!("--- Descriptive: score ---");
    let d = ds.descriptive("score")?;
    println!("  n={:.0}  mean={:.2}  std={:.2}", d.n, d.mean, d.std_dev);

    // --- Frequency table ---
    println!("\n--- Frequencies: gender ---");
    let freq = ds.frequencies("gender")?;
    println!("  {:<10} {:>5} {:>8} {:>10} {:>10}", "Value", "Count", "Percent", "Valid%", "Cum%");
    for row in freq.iter() {
        println!("  {:<10} {:>5} {:>7.1}% {:>9.1}% {:>9.1}%",
            row.value, row.count, row.percent, row.valid_percent, row.cumulative);
    }
    println!("  Valid={}, Missing={}", freq.n_valid, freq.n_missing);

    // --- Crosstab ---
    println!("\n--- Crosstab: gender × (age>30) ---");
    ds.compute("age_grp", |row| {
        row.numeric("age").map(|a| if a > 30.0 { 2.0 } else { 1.0 })
    })?;
    let ct = ds.crosstab("gender", "age_grp")?;
    print!("  {:<8}", "");
    for cl in &ct.col_labels { print!(" {:>8}", cl); }
    println!(" {:>8}", "Total");
    for (i, rl) in ct.row_labels.iter().enumerate() {
        print!("  {:<8}", rl);
        for &c in &ct.counts[i] { print!(" {:>8}", c); }
        println!(" {:>8}", ct.row_totals[i]);
    }

    // --- CSV round-trip ---
    let csv_path = std::env::temp_dir().join("socstat_example.csv");
    socstat::write().csv(&ds, &csv_path)?;
    let ds2 = socstat::read().csv(&csv_path)?;
    println!("\n=== CSV round-trip: {} vars, {} rows ===", ds2.n_vars(), ds2.n_rows());

    // --- Distribution functions ---
    println!("\n--- Distribution: Normal(0,1) ---");
    let z = NormalDist::standard();
    println!("  P(Z ≤ 1.96) = {:.4}", z.cdf(1.96));
    println!("  P(Z ≤ -1.96) = {:.4}", z.cdf(-1.96));
    println!("  z(0.975) = {:.4}", z.inverse_cdf(0.975));

    let t = StudentsTDist::new(10.0)?;
    println!("\n--- Distribution: t(10) ---");
    println!("  t(0.975, 10) = {:.4}", t.inverse_cdf(0.975));

    Ok(())
}
