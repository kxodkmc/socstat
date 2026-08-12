//! [`Variable`] — column metadata: name, label, type, measurement level,
//! missing-value rules, value labels, and display format.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// The storage type of a variable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum DataType {
    #[default]
    Numeric,
    Text,
}

/// Measurement level — mirrors SPSS's three levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum MeasureType {
    #[default]
    Nominal,
    Ordinal,
    Scale,
}

/// User-defined missing-value rules for a variable.
///
/// SPSS allows up to 3 discrete missing values, or one continuous range
/// plus one discrete value.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum MissingSpec {
    /// No user-defined missing values.
    #[default]
    None,
    /// Up to 3 discrete missing values.
    Discrete(Vec<f64>),
    /// A range [low, high] optionally plus one discrete value.
    Range { low: f64, high: f64, discrete: Option<f64> },
}

/// Display format hint for numeric variables.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub enum ValueFormat {
    /// General numeric (default).
    #[default]
    General,
    /// Fixed decimal: `width` total chars, `decimals` after point.
    Fixed { width: usize, decimals: usize },
    /// Scientific notation.
    Scientific { width: usize, decimals: usize },
    /// Date (epoch days).
    Date,
    /// Date + time (epoch seconds).
    DateTime,
    /// Percentage.
    Percent { decimals: usize },
    /// Currency.
    Currency { decimals: usize },
}

/// Column metadata.
#[derive(Debug, Clone, PartialEq, Default, Serialize, Deserialize)]
pub struct Variable {
    pub name: String,
    pub label: Option<String>,
    pub data_type: DataType,
    pub measure: MeasureType,
    pub format: ValueFormat,
    pub missing: MissingSpec,
    pub value_labels: BTreeMap<String, String>,
    /// String width (0 for numeric).
    pub width: usize,
    /// Role: is this a case weight?
    pub is_weight: bool,
}

impl Variable {
    /// Create a numeric variable.
    pub fn numeric(name: &str) -> Self {
        Self {
            name: name.to_string(),
            label: None,
            data_type: DataType::Numeric,
            measure: MeasureType::Scale,
            format: ValueFormat::General,
            missing: MissingSpec::None,
            value_labels: BTreeMap::new(),
            width: 0,
            is_weight: false,
        }
    }

    /// Create a text variable.
    pub fn text(name: &str) -> Self {
        Self {
            name: name.to_string(),
            label: None,
            data_type: DataType::Text,
            measure: MeasureType::Nominal,
            format: ValueFormat::General,
            missing: MissingSpec::None,
            value_labels: BTreeMap::new(),
            width: 255,
            is_weight: false,
        }
    }

    /// Set the variable label.
    pub fn label(mut self, label: &str) -> Self {
        self.label = Some(label.to_string());
        self
    }

    /// Set the measurement level.
    pub fn measure(mut self, m: MeasureType) -> Self {
        self.measure = m;
        self
    }

    /// Set string width.
    pub fn width(mut self, w: usize) -> Self {
        self.width = w;
        self
    }

    /// Set the display format.
    pub fn format(mut self, f: ValueFormat) -> Self {
        self.format = f;
        self
    }

    /// Add a value label (e.g. `1 = "Male"`).
    pub fn value_label(mut self, value: &str, label: &str) -> Self {
        self.value_labels.insert(value.to_string(), label.to_string());
        self
    }

    /// Set discrete missing values.
    pub fn missing_discrete(mut self, vals: &[f64]) -> Self {
        self.missing = MissingSpec::Discrete(vals.to_vec());
        self
    }

    /// Set a missing range.
    pub fn missing_range(mut self, low: f64, high: f64, discrete: Option<f64>) -> Self {
        self.missing = MissingSpec::Range { low, high, discrete };
        self
    }

    /// Mark as the case weight variable.
    pub fn weight(mut self) -> Self {
        self.is_weight = true;
        self
    }

    /// Check if a numeric value should be treated as user-missing.
    pub fn is_user_missing(&self, n: f64) -> bool {
        match &self.missing {
            MissingSpec::None => false,
            MissingSpec::Discrete(vals) => vals.contains(&n),
            MissingSpec::Range { low, high, discrete } => {
                (n >= *low && n <= *high) || *discrete == Some(n)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn numeric_builder() {
        let v = Variable::numeric("age")
            .label("Age")
            .measure(MeasureType::Scale);
        assert_eq!(v.name, "age");
        assert_eq!(v.label.as_deref(), Some("Age"));
        assert_eq!(v.measure, MeasureType::Scale);
    }

    #[test]
    fn string_builder() {
        let v = Variable::text("gender")
            .value_label("M", "Male")
            .value_label("F", "Female");
        assert_eq!(v.data_type, DataType::Text);
        assert_eq!(v.value_labels.get("M"), Some(&"Male".to_string()));
    }

    #[test]
    fn missing_discrete() {
        let v = Variable::numeric("x").missing_discrete(&[-1.0, -9.0]);
        assert!(v.is_user_missing(-1.0));
        assert!(v.is_user_missing(-9.0));
        assert!(!v.is_user_missing(0.0));
    }

    #[test]
    fn missing_range() {
        let v = Variable::numeric("income")
            .missing_range(0.0, 100.0, Some(999.0));
        assert!(v.is_user_missing(50.0));
        assert!(v.is_user_missing(999.0));
        assert!(!v.is_user_missing(101.0));
    }
}
