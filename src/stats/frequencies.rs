//! Frequency tables — count and percentage distributions for any variable.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::data::ColumnData;
use crate::error::SocStatResult;

/// A single row in a frequency table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyRow {
    /// The value (formatted as string for display).
    pub value: String,
    /// Raw count of occurrences.
    pub count: usize,
    /// Percentage of total (including missing): count / total * 100.
    pub percent: f64,
    /// Percentage of valid (non-missing): count / n_valid * 100.
    pub valid_percent: f64,
    /// Cumulative valid_percent up to and including this row.
    pub cumulative: f64,
}

/// A complete frequency table for a variable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyTable {
    pub rows: Vec<FrequencyRow>,
    pub n_valid: usize,
    pub n_missing: usize,
    pub total: usize,
}

impl FrequencyTable {
    /// Iterate over the rows.
    pub fn iter(&self) -> impl Iterator<Item = &FrequencyRow> {
        self.rows.iter()
    }
}

/// Build a frequency table from a column.
pub fn build(col: &ColumnData, value_labels: &BTreeMap<String, String>) -> SocStatResult<FrequencyTable> {
    let total = col.len();
    if total == 0 {
        return Ok(FrequencyTable {
            rows: vec![],
            n_valid: 0,
            n_missing: 0,
            total: 0,
        });
    }

    // Count occurrences of each value
    // Use BTreeMap for sorted output (numeric sort for numbers, lexicographic for strings)
    let mut counts: BTreeMap<String, (String, usize)> = BTreeMap::new();
    let mut n_missing = 0usize;

    match col {
        ColumnData::Numeric(v) => {
            for &opt in v {
                match opt {
                    Some(x) => {
                        let display = format_num(x);
                        let label = value_labels.get(&display)
                            .cloned()
                            .unwrap_or_else(|| display.clone());
                        counts.entry(display.clone())
                            .and_modify(|(_, c)| *c += 1)
                            .or_insert((label, 1));
                    }
                    None => n_missing += 1,
                }
            }
        }
        ColumnData::Text(v) => {
            for opt in v {
                match opt {
                    Some(s) => {
                        let label = value_labels.get(s)
                            .cloned()
                            .unwrap_or_else(|| s.clone());
                        counts.entry(s.clone())
                            .and_modify(|(_, c)| *c += 1)
                            .or_insert((label, 1));
                    }
                    None => n_missing += 1,
                }
            }
        }
    }

    let n_valid = total - n_missing;

    // Build rows with percentages
    let mut rows: Vec<FrequencyRow> = Vec::with_capacity(counts.len());
    let mut cumulative = 0.0f64;

    for (_display, (label, count)) in counts {
        let percent = count as f64 / total as f64 * 100.0;
        let valid_percent = if n_valid > 0 {
            count as f64 / n_valid as f64 * 100.0
        } else {
            0.0
        };
        cumulative += valid_percent;

        rows.push(FrequencyRow {
            value: label,
            count,
            percent,
            valid_percent,
            cumulative,
        });
    }

    Ok(FrequencyTable {
        rows,
        n_valid,
        n_missing,
        total,
    })
}

/// Format a number for display: integers without decimal point, floats trimmed.
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
    use crate::data::ColumnData;

    #[test]
    fn numeric_frequency() {
        let col = ColumnData::Numeric(vec![
            Some(1.0), Some(2.0), Some(1.0), Some(3.0), None, Some(1.0),
        ]);
        let labels = BTreeMap::new();
        let table = build(&col, &labels).unwrap();

        assert_eq!(table.total, 6);
        assert_eq!(table.n_valid, 5);
        assert_eq!(table.n_missing, 1);

        // Three distinct values: 1, 2, 3
        assert_eq!(table.rows.len(), 3);

        // Value "1" appears 3 times → 50% of total, 60% of valid
        let r0 = &table.rows[0];
        assert_eq!(r0.count, 3);
        assert!((r0.valid_percent - 60.0).abs() < 0.01);
        assert!((r0.cumulative - 60.0).abs() < 0.01);

        // Cumulative should end at 100%
        let last = table.rows.last().unwrap();
        assert!((last.cumulative - 100.0).abs() < 0.01);
    }

    #[test]
    fn text_frequency() {
        let col = ColumnData::Text(vec![
            Some("A".into()), Some("B".into()), Some("A".into()), None,
        ]);
        let labels = BTreeMap::new();
        let table = build(&col, &labels).unwrap();

        assert_eq!(table.n_valid, 3);
        assert_eq!(table.n_missing, 1);
        assert_eq!(table.rows.len(), 2);

        // "A" appears 2 times, "B" appears 1 time
        // BTreeMap sorts alphabetically: A before B
        assert_eq!(table.rows[0].value, "A");
        assert_eq!(table.rows[0].count, 2);
        assert_eq!(table.rows[1].value, "B");
        assert_eq!(table.rows[1].count, 1);
    }

    #[test]
    fn value_labels_applied() {
        let col = ColumnData::Numeric(vec![Some(1.0), Some(2.0), Some(1.0)]);
        let mut labels = BTreeMap::new();
        labels.insert("1".to_string(), "Male".to_string());
        labels.insert("2".to_string(), "Female".to_string());

        let table = build(&col, &labels).unwrap();
        assert_eq!(table.rows[0].value, "Male");
        assert_eq!(table.rows[1].value, "Female");
    }

    #[test]
    fn empty_column() {
        let col = ColumnData::Numeric(vec![]);
        let table = build(&col, &BTreeMap::new()).unwrap();
        assert_eq!(table.total, 0);
        assert!(table.rows.is_empty());
    }
}
