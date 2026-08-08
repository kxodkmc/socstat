//! Crosstabulation (contingency tables) — counts and percentages
//! for the intersection of two categorical variables.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::data::ColumnData;
use crate::error::SocStatResult;

/// A crosstab result: observed counts, expected counts, and percentages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crosstab {
    /// Row variable category labels.
    pub row_labels: Vec<String>,
    /// Column variable category labels.
    pub col_labels: Vec<String>,
    /// Observed cell counts [row][col].
    pub counts: Vec<Vec<usize>>,
    /// Expected counts under independence: (row_total * col_total) / grand_total.
    pub expected: Vec<Vec<f64>>,
    /// Percentage of row total [row][col].
    pub row_pcts: Vec<Vec<f64>>,
    /// Percentage of column total [row][col].
    pub col_pcts: Vec<Vec<f64>>,
    /// Percentage of grand total [row][col].
    pub total_pcts: Vec<Vec<f64>>,
    /// Grand total (n).
    pub n: usize,
    /// Row marginal totals.
    pub row_totals: Vec<usize>,
    /// Column marginal totals.
    pub col_totals: Vec<usize>,
}

/// Build a crosstab from two columns.
pub fn build(row_col: &ColumnData, col_col: &ColumnData) -> SocStatResult<Crosstab> {
    let n = row_col.len();
    if n != col_col.len() {
        return Err(crate::error::SocStatError::ColumnLengthMismatch {
            expected: n,
            got: col_col.len(),
        });
    }

    // Extract values as strings for both columns, skipping rows where either is missing
    let row_vals = extract_values(row_col);
    let col_vals = extract_values(col_col);

    // Build unique sorted label lists
    let mut row_map: BTreeMap<String, usize> = BTreeMap::new();
    let mut col_map: BTreeMap<String, usize> = BTreeMap::new();

    for (r, c) in row_vals.iter().zip(col_vals.iter()) {
        if let (Some(rv), Some(cv)) = (r, c) {
            if !row_map.contains_key(rv) {
                row_map.insert(rv.clone(), row_map.len());
            }
            if !col_map.contains_key(cv) {
                col_map.insert(cv.clone(), col_map.len());
            }
        }
    }

    let n_rows = row_map.len();
    let n_cols = col_map.len();
    let row_labels: Vec<String> = row_map.keys().cloned().collect();
    let col_labels: Vec<String> = col_map.keys().cloned().collect();

    // Count
    let mut counts = vec![vec![0usize; n_cols]; n_rows];
    let mut grand_total = 0usize;

    for (r, c) in row_vals.iter().zip(col_vals.iter()) {
        if let (Some(rv), Some(cv)) = (r, c) {
            let ri = row_map[rv];
            let ci = col_map[cv];
            counts[ri][ci] += 1;
            grand_total += 1;
        }
    }

    // Marginal totals
    let row_totals: Vec<usize> = counts.iter()
        .map(|r| r.iter().sum())
        .collect();
    let col_totals: Vec<usize> = (0..n_cols)
        .map(|c| counts.iter().map(|r| r[c]).sum())
        .collect();

    // Expected counts and percentages
    let gt = grand_total as f64;
    let mut expected = vec![vec![0.0f64; n_cols]; n_rows];
    let mut row_pcts = vec![vec![0.0f64; n_cols]; n_rows];
    let mut col_pcts = vec![vec![0.0f64; n_cols]; n_rows];
    let mut total_pcts = vec![vec![0.0f64; n_cols]; n_rows];

    for (ri, r) in counts.iter().enumerate() {
        for (ci, &count) in r.iter().enumerate() {
            let rt = row_totals[ri] as f64;
            let ct = col_totals[ci] as f64;
            expected[ri][ci] = if gt > 0.0 { rt * ct / gt } else { 0.0 };
            row_pcts[ri][ci] = if rt > 0.0 { count as f64 / rt * 100.0 } else { 0.0 };
            col_pcts[ri][ci] = if ct > 0.0 { count as f64 / ct * 100.0 } else { 0.0 };
            total_pcts[ri][ci] = if gt > 0.0 { count as f64 / gt * 100.0 } else { 0.0 };
        }
    }

    Ok(Crosstab {
        row_labels,
        col_labels,
        counts,
        expected,
        row_pcts,
        col_pcts,
        total_pcts,
        n: grand_total,
        row_totals,
        col_totals,
    })
}

/// Extract values from a column as strings, preserving None for missing.
fn extract_values(col: &ColumnData) -> Vec<Option<String>> {
    match col {
        ColumnData::Numeric(v) => v.iter().map(|opt| opt.map(format_num)).collect(),
        ColumnData::Text(v) => v.to_vec(),
    }
}

fn format_num(x: f64) -> String {
    if x.fract() == 0.0 {
        format!("{}", x as i64)
    } else {
        format!("{}", x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_crosstab() {
        // 2x2: Gender × Preference
        let row = ColumnData::Text(vec![
            Some("M".into()), Some("M".into()),
            Some("F".into()), Some("F".into()),
        ]);
        let col = ColumnData::Text(vec![
            Some("Yes".into()), Some("No".into()),
            Some("Yes".into()), Some("No".into()),
        ]);

        let ct = build(&row, &col).unwrap();
        assert_eq!(ct.n, 4);
        assert_eq!(ct.row_labels, vec!["F", "M"]); // BTreeMap sorts alphabetically
        assert_eq!(ct.col_labels, vec!["No", "Yes"]);

        // Each cell should have count 1
        for r in &ct.counts {
            for &c in r {
                assert_eq!(c, 1);
            }
        }

        // Expected counts = 1.0 each (uniform)
        for r in &ct.expected {
            for &e in r {
                assert!((e - 1.0).abs() < 1e-10);
            }
        }

        // Row percentages = 50% each
        for r in &ct.row_pcts {
            for &p in r {
                assert!((p - 50.0).abs() < 0.01);
            }
        }
    }

    #[test]
    fn crosstab_with_missing() {
        let row = ColumnData::Text(vec![
            Some("A".into()), Some("B".into()), Some("A".into()), None,
        ]);
        let col = ColumnData::Text(vec![
            Some("X".into()), Some("X".into()), Some("Y".into()), Some("X".into()),
        ]);

        let ct = build(&row, &col).unwrap();
        // Row 3 has missing row value → skipped
        assert_eq!(ct.n, 3);
        assert_eq!(ct.row_labels, vec!["A", "B"]);
        assert_eq!(ct.col_labels, vec!["X", "Y"]);
    }

    #[test]
    fn length_mismatch_errors() {
        let row = ColumnData::Numeric(vec![Some(1.0), Some(2.0)]);
        let col = ColumnData::Numeric(vec![Some(1.0)]);
        assert!(build(&row, &col).is_err());
    }
}
