//! Error types for the socstat crate.

use thiserror::Error;

/// The unified error type for all socstat operations.
#[derive(Debug, Error)]
pub enum SocStatError {
    #[error("variable not found: {0}")]
    VariableNotFound(String),

    #[error("variable index out of bounds: index {index}, len {len}")]
    VariableIndexOutOfBounds { index: usize, len: usize },

    #[error("type mismatch on variable '{var}': expected {expected}, got {actual}")]
    TypeMismatch { var: String, expected: &'static str, actual: &'static str },

    #[error("missing value in variable '{0}' where a number is required")]
    MissingNumber(String),

    #[error("variable '{0}' already exists")]
    DuplicateVariable(String),

    #[error("row length mismatch: expected {expected}, got {got}")]
    RowLengthMismatch { expected: usize, got: usize },

    #[error("column length mismatch: expected {expected}, got {got}")]
    ColumnLengthMismatch { expected: usize, got: usize },

    #[error("computation error: {0}")]
    Computation(String),

    #[error("insufficient data: {0}")]
    InsufficientData(String),

    #[error("singular matrix: {0}")]
    SingularMatrix(String),

    #[error("convergence not reached after {0} iterations")]
    ConvergenceNotReached(usize),

    #[error("complete separation: {0}")]
    CompleteSeparation(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[cfg(feature = "csv")]
    #[error("csv error: {0}")]
    Csv(#[from] csv::Error),

    #[cfg(feature = "csv")]
    #[error("json error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("{0}")]
    Other(String),
}

pub type SocStatResult<T> = std::result::Result<T, SocStatError>;
