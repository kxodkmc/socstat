//! [`Value`] — a transient cell-level type for building and inspecting data.
//!
//! `Value` is **not** the storage type. Columns store data in typed
//! contiguous vectors ([`ColumnData`]). `Value` exists for row-oriented
//! operations: building rows programmatically, reading I/O, display.

use std::fmt;

/// A single cell value — transient, for row-level operations.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    Number(f64),
    Text(String),
    Missing,
}

impl Value {
    #[inline]
    pub fn as_number(&self) -> Option<f64> {
        match self {
            Value::Number(n) if n.is_finite() => Some(*n),
            _ => None,
        }
    }

    #[inline]
    pub fn as_text(&self) -> Option<&str> {
        match self {
            Value::Text(s) => Some(s),
            _ => None,
        }
    }

    #[inline]
    pub fn is_missing(&self) -> bool {
        matches!(self, Value::Missing)
    }

    /// A display string: empty for missing, numbers formatted cleanly.
    pub fn display(&self) -> String {
        match self {
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            Value::Text(s) => s.clone(),
            Value::Missing => String::new(),
        }
    }
}

// --- Conversions ---

impl From<f64> for Value {
    fn from(n: f64) -> Self {
        if n.is_finite() { Value::Number(n) } else { Value::Missing }
    }
}

impl From<&str> for Value {
    fn from(s: &str) -> Self {
        if s.is_empty() { Value::Missing } else { Value::Text(s.into()) }
    }
}

impl From<String> for Value {
    fn from(s: String) -> Self {
        if s.is_empty() { Value::Missing } else { Value::Text(s) }
    }
}

impl From<Option<f64>> for Value {
    fn from(o: Option<f64>) -> Self {
        o.map(Value::from).unwrap_or(Value::Missing)
    }
}

impl From<Option<String>> for Value {
    fn from(o: Option<String>) -> Self {
        o.map(Value::from).unwrap_or(Value::Missing)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.display())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(Value::from(1.0), Value::Number(1.0));
        assert_eq!(Value::from(f64::NAN), Value::Missing);
        assert_eq!(Value::from(""), Value::Missing);
        assert_eq!(Value::from(Some(2.5)), Value::Number(2.5));
        assert_eq!(Value::from(None::<f64>), Value::Missing);
    }

    #[test]
    fn display_clean() {
        assert_eq!(Value::Number(42.0).display(), "42");
        assert_eq!(Value::Number(2.5).display(), "2.5");
        assert_eq!(Value::Text("hi".into()).display(), "hi");
        assert_eq!(Value::Missing.display(), "");
    }
}
