//! Multifactor (factorial) ANOVA with Type I (sequential) and Type II
//! (marginal) sums of squares.
//!
//! Factors are dummy-coded (each factor → k−1 indicator columns, a base level
//! is absorbed into the intercept) and all pairwise two-way interactions are
//! built as products of the main-effect dummies. The design is then fitted
//! with OLS; each effect's sum of squares is the change in regression sum of
//! squares between nested models. Three-way and higher interactions are not
//! supported (Hard Rule 4 keeps the scope honest about what is modelled).
//!
//! Every public result type derives `Serialize`/`Deserialize` (Hard Rule 1).

use std::collections::BTreeSet;

use serde::{Deserialize, Serialize};

use crate::data::{ColumnData, Dataset};
use crate::dist::{Distribution, FDist};
use crate::error::{SocStatError, SocStatResult};

use crate::stats::shared::{cleaned_numeric_column, extract_labels, positive_weight};
use crate::stats::regression::linear_regression;

/// Whether Type I (sequential) or Type II (marginal) sums of squares are used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SsType {
    /// Sequential SS: factors enter in the given order, then two-way
    /// interactions. Matches R's `anova()` default for ordered terms.
    TypeI,
    /// Marginal SS obeying the principle of marginality: each term is tested
    /// against the model of all main effects (and its own interaction).
    TypeII,
}

/// One row of a factorial ANOVA table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnovaEffect {
    pub source: String,
    pub ss: f64,
    pub df: f64,
    pub ms: f64,
    /// F statistic; `None` for the error/total rows where it is undefined.
    pub f: Option<f64>,
    /// p-value; `None` where F is undefined.
    pub p_value: Option<f64>,
    /// η² = SS_effect / SS_total.
    pub eta_squared: Option<f64>,
    /// Partial η² = SS_effect / (SS_effect + SS_error).
    pub partial_eta_squared: Option<f64>,
}

/// Result of a factorial ANOVA.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorialAnova {
    pub factors: Vec<String>,
    pub dependent_var: String,
    pub ss_type: SsType,
    pub effects: Vec<AnovaEffect>,
    pub r_squared: f64,
    pub adj_r_squared: f64,
    /// Effective sample size (sum of weights).
    pub n: f64,
}

/// One model term (a factor's main effect or a two-way interaction), as a set
/// of dummy-column indices.
struct Term {
    name: String,
    indices: Vec<usize>,
}

/// Multifactor ANOVA of `dep_var` on `factors` (`factors.len() ≥ 2`), with
/// either Type I or Type II sums of squares.
///
/// Only two-way interactions are modelled. Missing values are excluded by
/// listwise deletion; case weights are honored. Perfectly collinear dummy
/// columns (e.g. empty cells) return `SocStatError::SingularMatrix`.
pub fn factorial_anova(
    ds: &Dataset,
    dep_var: &str,
    factors: &[&str],
    ss_type: SsType,
) -> SocStatResult<FactorialAnova> {
    if factors.len() < 2 {
        return Err(SocStatError::Other(
            "use one_way_anova for a single factor".into(),
        ));
    }
    let dep = cleaned_numeric_column(ds, dep_var)?;
    let weights = ds.weights();
    let n = dep.len();
    if let Some(w) = &weights
        && w.len() != n
    {
        return Err(SocStatError::ColumnLengthMismatch { expected: n, got: w.len() });
    }

    // Factor display labels per column.
    let mut flabels = Vec::with_capacity(factors.len());
    for &f in factors {
        let col = ds.column_by_name(f)?;
        if col.len() != n {
            return Err(SocStatError::ColumnLengthMismatch { expected: n, got: col.len() });
        }
        flabels.push(extract_labels(col));
    }

    // Listwise-align y, all factor labels, and weights.
    let mut y: Vec<f64> = Vec::new();
    let mut row_labels: Vec<Vec<String>> = Vec::new();
    let mut wa: Vec<f64> = Vec::new();
    for i in 0..n {
        let Some(yv) = dep[i] else { continue };
        if !yv.is_finite() {
            continue;
        }
        let mut labs = Vec::with_capacity(factors.len());
        let mut complete = true;
        for col in &flabels {
            match &col[i] {
                Some(s) => labs.push(s.clone()),
                None => {
                    complete = false;
                    break;
                }
            }
        }
        if !complete {
            continue;
        }
        let w = weights.as_ref().map(|ww| ww[i]).unwrap_or(1.0);
        if !positive_weight(w) {
            continue;
        }
        y.push(yv);
        row_labels.push(labs);
        wa.push(w);
    }
    let n_valid = y.len();
    let n_eff: f64 = wa.iter().sum();

    // Levels per factor (sorted), from the aligned rows.
    let mut levels: Vec<Vec<String>> = Vec::with_capacity(factors.len());
    for f in 0..factors.len() {
        let set: BTreeSet<&str> = row_labels.iter().map(|r| r[f].as_str()).collect();
        if set.len() < 2 {
            return Err(SocStatError::InsufficientData(format!(
                "factor '{}' has fewer than two levels",
                factors[f]
            )));
        }
        levels.push(set.into_iter().map(str::to_string).collect());
    }

    // Build main-effect dummies and their terms.
    let mut predictors: Vec<(String, Vec<f64>)> = Vec::new();
    let mut main_terms: Vec<Term> = Vec::with_capacity(factors.len());
    for f in 0..factors.len() {
        let mut indices = Vec::new();
        for level in &levels[f][1..] {
            let name = format!("{}={}", factors[f], level);
            let dummy: Vec<f64> = row_labels
                .iter()
                .map(|r| if r[f] == *level { 1.0 } else { 0.0 })
                .collect();
            indices.push(predictors.len());
            predictors.push((name, dummy));
        }
        main_terms.push(Term { name: factors[f].to_string(), indices });
    }

    // Build two-way interaction dummies.
    let mut inter_terms: Vec<Term> = Vec::new();
    for a in 0..factors.len() {
        for b in (a + 1)..factors.len() {
            let mut indices = Vec::new();
            for &da in &main_terms[a].indices {
                for &db in &main_terms[b].indices {
                    let (_, ca) = &predictors[da];
                    let (_, cb) = &predictors[db];
                    let prod: Vec<f64> = ca.iter().zip(cb).map(|(&x, &y2)| x * y2).collect();
                    indices.push(predictors.len());
                    predictors.push((format!("{}*{}", factors[a], factors[b]), prod));
                }
            }
            inter_terms.push(Term { name: format!("{} * {}", factors[a], factors[b]), indices });
        }
    }

    if n_valid == 0 {
        return Err(SocStatError::InsufficientData("no valid cases".into()));
    }
    let ybar = y.iter().zip(&wa).map(|(&v, &w)| v * w).sum::<f64>() / n_eff;
    let ss_total = y.iter().zip(&wa).map(|(&v, &w)| w * (v - ybar).powi(2)).sum::<f64>();
    if ss_total <= 0.0 {
        return Err(SocStatError::InsufficientData("dependent variable has zero variance".into()));
    }

    // Wrap columns and fit-resolution helper.
    let cols: Vec<ColumnData> = predictors
        .iter()
        .map(|(_, v)| ColumnData::Numeric(v.iter().map(|&x| Some(x)).collect()))
        .collect();
    let ycol = ColumnData::Numeric(y.iter().map(|&v| Some(v)).collect::<Vec<_>>());
    let wref = Some(&wa[..]);

    let ss_model = |indices: &[usize]| -> SocStatResult<f64> {
        if indices.is_empty() {
            return Ok(0.0);
        }
        let refs: Vec<(&str, &ColumnData)> = indices
            .iter()
            .map(|&idx| (predictors[idx].0.as_str(), &cols[idx]))
            .collect();
        let m = linear_regression(dep_var, &ycol, &refs, wref)?;
        Ok(m.r_squared * ss_total)
    };

    // Full model: error term + overall fit diagnostics.
    let full_indices: Vec<usize> = main_terms
        .iter()
        .chain(&inter_terms)
        .flat_map(|t| t.indices.iter().copied())
        .collect();
    let full_refs: Vec<(&str, &ColumnData)> = full_indices
        .iter()
        .map(|&idx| (predictors[idx].0.as_str(), &cols[idx]))
        .collect();
    let full_model = linear_regression(dep_var, &ycol, &full_refs, wref)?;
    let full_ss = full_model.r_squared * ss_total;
    let total_params = 1 + predictors.len();
    let df_error = n_eff - total_params as f64;
    if df_error <= 0.0 {
        return Err(SocStatError::InsufficientData(format!(
            "sample size too small for factorial ANOVA: have {n_eff} cases vs {total_params} parameters"
        )));
    }
    let ss_error = (ss_total - full_ss).max(0.0);
    let ms_error = ss_error / df_error;

    let make_effect = |source: &str, ss: f64, df: f64| -> AnovaEffect {
        let ss = ss.max(0.0);
        let ms = ss / df;
        let (f, p) = if ms_error > 0.0 {
            let f = ms / ms_error;
            let p = 1.0 - FDist::new(df, df_error).map(|d| d.cdf(f)).unwrap_or(f64::NAN);
            (Some(f), Some(p))
        } else {
            (None, None)
        };
        let eta = (ss / ss_total).clamp(0.0, 1.0);
        let peta = (ss / (ss + ss_error)).clamp(0.0, 1.0);
        AnovaEffect {
            source: source.to_string(),
            ss,
            df,
            ms,
            f,
            p_value: p,
            eta_squared: Some(eta),
            partial_eta_squared: Some(peta),
        }
    };

    let mut effects = Vec::with_capacity(main_terms.len() + inter_terms.len() + 2);
    match ss_type {
        SsType::TypeI => {
            // Sequential: factors in order, then interactions.
            let mut cum: Vec<usize> = Vec::new();
            let mut prev = 0.0_f64;
            for t in &main_terms {
                cum.extend_from_slice(&t.indices);
                let cur = ss_model(&cum)?;
                effects.push(make_effect(&t.name, cur - prev, t.indices.len() as f64));
                prev = cur;
            }
            for t in &inter_terms {
                cum.extend_from_slice(&t.indices);
                let cur = ss_model(&cum)?;
                effects.push(make_effect(&t.name, cur - prev, t.indices.len() as f64));
                prev = cur;
            }
        }
        SsType::TypeII => {
            // Marginal: each main effect against the all-main-effects model,
            // each interaction against all main effects plus itself.
            let all_mains: Vec<usize> =
                main_terms.iter().flat_map(|t| t.indices.iter().copied()).collect();
            let mains_ss = ss_model(&all_mains)?;
            for t in &main_terms {
                let reduced: Vec<usize> = all_mains
                    .iter()
                    .copied()
                    .filter(|i| !t.indices.contains(i))
                    .collect();
                let ss = ss_model(&all_mains)? - ss_model(&reduced)?;
                effects.push(make_effect(&t.name, ss, t.indices.len() as f64));
            }
            for t in &inter_terms {
                let full_set: Vec<usize> =
                    all_mains.iter().chain(t.indices.iter()).copied().collect();
                let ss = ss_model(&full_set)? - mains_ss;
                effects.push(make_effect(&t.name, ss, t.indices.len() as f64));
            }
        }
    }

    effects.push(AnovaEffect {
        source: "Error".into(),
        ss: ss_error,
        df: df_error,
        ms: ms_error,
        f: None,
        p_value: None,
        eta_squared: None,
        partial_eta_squared: None,
    });
    effects.push(AnovaEffect {
        source: "Total".into(),
        ss: ss_total,
        df: n_eff,
        ms: ss_total / n_eff,
        f: None,
        p_value: None,
        eta_squared: None,
        partial_eta_squared: None,
    });

    Ok(FactorialAnova {
        factors: factors.iter().map(|s| s.to_string()).collect(),
        dependent_var: dep_var.to_string(),
        ss_type,
        effects,
        r_squared: full_model.r_squared,
        adj_r_squared: full_model.adj_r_squared,
        n: n_eff,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Value, Variable};
    use crate::stats::StatsExt;
    use approx::assert_abs_diff_eq;

    /// Balanced 2×2 design: cell means 10.5/20.5/30.5/40.5, additive (no
    /// interaction). SS(A)=800, SS(B)=200, SS(AB)=0, SS_err=2, SS_total=1002.
    fn balanced_dataset() -> Dataset {
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
        ds
    }

    fn effect<'a>(r: &'a FactorialAnova, source: &str) -> &'a AnovaEffect {
        r.effects.iter().find(|e| e.source == source).expect("effect present")
    }

    #[test]
    fn balanced_type1_matches_hand_sums() {
        let ds = balanced_dataset();
        let r = ds.factorial_anova("y", &["a", "b"], SsType::TypeI).unwrap();

        let a = effect(&r, "a");
        assert_abs_diff_eq!(a.ss, 800.0, epsilon = 1e-6);
        assert_abs_diff_eq!(a.df, 1.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a.ms, 800.0, epsilon = 1e-6);
        assert_abs_diff_eq!(a.eta_squared.unwrap(), 800.0 / 1002.0, epsilon = 1e-6);
        assert_abs_diff_eq!(a.partial_eta_squared.unwrap(), 800.0 / 802.0, epsilon = 1e-6);
        assert!(a.p_value.unwrap().is_finite());
        assert!(a.p_value.unwrap() < 0.001); // F = 800/0.5 = 1600, df (1,4)

        let b = effect(&r, "b");
        assert_abs_diff_eq!(b.ss, 200.0, epsilon = 1e-6);

        let ab = effect(&r, "a * b");
        assert!(ab.ss.abs() < 1e-6);

        let err = effect(&r, "Error");
        assert_abs_diff_eq!(err.ss, 2.0, epsilon = 1e-6);
        assert_abs_diff_eq!(err.df, 4.0, epsilon = 1e-12);
        assert_abs_diff_eq!(err.ms, 0.5, epsilon = 1e-6);

        let tot = effect(&r, "Total");
        assert_abs_diff_eq!(tot.ss, 1002.0, epsilon = 1e-6);
        assert_abs_diff_eq!(r.n, 8.0, epsilon = 1e-12);
        assert_abs_diff_eq!(r.r_squared, 1000.0 / 1002.0, epsilon = 1e-6);
    }

    #[test]
    fn balanced_type2_agrees_with_type1() {
        let ds = balanced_dataset();
        let t1 = ds.factorial_anova("y", &["a", "b"], SsType::TypeI).unwrap();
        let t2 = ds.factorial_anova("y", &["a", "b"], SsType::TypeII).unwrap();
        // Balanced orthogonal design: the two types give identical SS.
        for src in ["a", "b", "a * b"] {
            assert_abs_diff_eq!(
                effect(&t1, src).ss,
                effect(&t2, src).ss,
                epsilon = 1e-6
            );
        }
        assert_abs_diff_eq!(effect(&t2, "a").ss, 800.0, epsilon = 1e-6);
        assert_abs_diff_eq!(effect(&t2, "b").ss, 200.0, epsilon = 1e-6);
    }

    #[test]
    fn anova_edge_cases() {
        // Single factor → error.
        let ds = balanced_dataset();
        assert!(ds.factorial_anova("y", &["a"], SsType::TypeI).is_err());
        // Factor with one level → error.
        let mut ds1 = Dataset::new();
        ds1.add_var(Variable::text("a")).unwrap();
        ds1.add_var(Variable::text("b")).unwrap();
        ds1.add_var(Variable::numeric("y")).unwrap();
        ds1.push_row(vec![Value::Text("1".into()), Value::Text("1".into()), Value::Number(1.0)])
            .unwrap();
        ds1.push_row(vec![Value::Text("1".into()), Value::Text("1".into()), Value::Number(2.0)])
            .unwrap();
        ds1.push_row(vec![Value::Text("1".into()), Value::Text("2".into()), Value::Number(3.0)])
            .unwrap();
        assert!(ds1.factorial_anova("y", &["a", "b"], SsType::TypeI).is_err());
        // Serde round-trip.
        let r = ds.factorial_anova("y", &["a", "b"], SsType::TypeII).unwrap();
        let json = serde_json::to_string(&r).unwrap();
        let back: FactorialAnova = serde_json::from_str(&json).unwrap();
        assert_eq!(back.effects.len(), r.effects.len());
        assert_abs_diff_eq!(
            back.effects[0].ss,
            r.effects[0].ss,
            epsilon = 1e-15
        );
    }
}