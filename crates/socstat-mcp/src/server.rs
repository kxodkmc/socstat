//! The MCP server: a [`ServerHandler`] exposing socstat's analyses as tools.
//!
//! Tools are declared with `#[tool]` here as thin wrappers over the pure
//! helpers in [`tools`]; each returns structured JSON via [`Json`].

use std::borrow::Cow;
use std::sync::Arc;

use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::ProtocolVersion;
use rmcp::{Json, ServerHandler, tool, tool_handler, tool_router};
use serde_json::Value;
use tokio::sync::Mutex;

use crate::state::SharedState;
use crate::tools::{anova, data, describe, multivariate, normality, regression, tests, transform};

/// The MCP service. Holds shared, stateful dataset storage.
///
/// `gate` serializes tool dispatches: rmcp may run concurrent tool calls, but
/// a stateful workflow (load → analyze) must be strictly ordered.
pub struct SocstatMcpServer {
    state: Arc<SharedState>,
    gate: Mutex<()>,
}

impl SocstatMcpServer {
    /// Create a server around a shared dataset store.
    pub fn new(state: Arc<SharedState>) -> Self {
        Self { state, gate: Mutex::new(()) }
    }
}

#[tool_router]
impl SocstatMcpServer {
    // --- Data management ----------------------------------------------------

    #[tool(
        description = "List all loaded datasets with their row and variable counts",
        annotations(title = "List datasets", read_only_hint = true)
    )]
    pub async fn list_datasets(&self) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(data::list(&self.state)))
    }

    #[tool(
        description = "Load a dataset from a CSV or JSON file into shared state, then return its schema (SPSS .sav is supported only when built with the `sav` feature)",
        annotations(title = "Load dataset", destructive_hint = false)
    )]
    pub async fn load_dataset(&self, Parameters(req): Parameters<data::LoadRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(data::load(&self.state, req)?))
    }

    #[tool(
        description = "Describe the variables (name, label, type, measure, missing counts) and shape of a dataset",
        annotations(title = "Dataset info", read_only_hint = true)
    )]
    pub async fn dataset_info(&self, Parameters(req): Parameters<data::ByName>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(data::info(&self.state, &req.dataset)?))
    }

    #[tool(
        description = "Preview the first rows of a dataset as typed cell values",
        annotations(title = "Preview rows", read_only_hint = true)
    )]
    pub async fn preview(&self, Parameters(req): Parameters<data::PreviewRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(data::preview(&self.state, req)?))
    }

    #[tool(
        description = "Drop a loaded dataset from shared memory",
        annotations(title = "Drop dataset", destructive_hint = true)
    )]
    pub async fn drop_dataset(&self, Parameters(req): Parameters<data::ByName>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(data::drop(&self.state, &req.dataset)?))
    }

    // --- Transforms ---------------------------------------------------------

    #[tool(
        description = "Recode a numeric variable into a new variable via a discrete value mapping (source is kept)",
        annotations(title = "Recode", destructive_hint = false)
    )]
    pub async fn recode(&self, Parameters(req): Parameters<transform::RecodeRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::recode(&self.state, req)?))
    }

    #[tool(
        description = "Keep only rows where a numeric variable satisfies a comparison (op: gt/ge/lt/le/eq/ne)",
        annotations(title = "Filter rows", destructive_hint = true)
    )]
    pub async fn filter(&self, Parameters(req): Parameters<transform::FilterRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::filter(&self.state, req)?))
    }

    #[tool(
        description = "Sort rows by a numeric variable (ascending or descending)",
        annotations(title = "Sort rows", destructive_hint = true)
    )]
    pub async fn sort(&self, Parameters(req): Parameters<transform::SortRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::sort(&self.state, req)?))
    }

    #[tool(
        description = "Keep only the listed columns, dropping all others",
        annotations(title = "Keep columns", destructive_hint = true)
    )]
    pub async fn keep(&self, Parameters(req): Parameters<transform::KeepRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::keep(&self.state, req)?))
    }

    #[tool(
        description = "Set the case-weight variable; all later statistics use it as frequency weights",
        annotations(title = "Set weight", destructive_hint = false)
    )]
    pub async fn set_weight(&self, Parameters(req): Parameters<transform::SetWeightRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::set_weight(&self.state, req)?))
    }

    #[tool(
        description = "Compute a new numeric variable elementwise as `left op right`, where each operand is a column or a constant and op is + - * /",
        annotations(title = "Compute variable", destructive_hint = false)
    )]
    pub async fn compute(&self, Parameters(req): Parameters<transform::ComputeRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(transform::compute(&self.state, req)?))
    }

    // --- Descriptive statistics --------------------------------------------

    #[tool(
        description = "Comprehensive descriptive statistics for a numeric variable (mean, std, median, quartiles, skew, kurtosis, CI)",
        annotations(title = "Descriptive statistics", read_only_hint = true)
    )]
    pub async fn descriptive(&self, Parameters(req): Parameters<describe::VarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(describe::descriptive(&self.state, req)?))
    }

    #[tool(
        description = "Build a frequency table (counts and percentages) for any variable",
        annotations(title = "Frequency table", read_only_hint = true)
    )]
    pub async fn frequencies(&self, Parameters(req): Parameters<describe::VarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(describe::frequencies(&self.state, req)?))
    }

    #[tool(
        description = "Build a crosstabulation (contingency table) of two variables",
        annotations(title = "Crosstab", read_only_hint = true)
    )]
    pub async fn crosstab(&self, Parameters(req): Parameters<describe::TwoVarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(describe::crosstab(&self.state, req)?))
    }

    // --- Hypothesis tests ---------------------------------------------------

    #[tool(
        description = "Independent-samples t-test of a numeric variable between two groups (pooled, Welch, Levene)",
        annotations(title = "Independent t-test", read_only_hint = true)
    )]
    pub async fn independent_t_test(&self, Parameters(req): Parameters<tests::ByGroupRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::independent_t_test(&self.state, req)?))
    }

    #[tool(
        description = "One-way ANOVA of a numeric variable across the groups of a factor",
        annotations(title = "One-way ANOVA", read_only_hint = true)
    )]
    pub async fn one_way_anova(&self, Parameters(req): Parameters<tests::ByGroupRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::one_way_anova(&self.state, req)?))
    }

    #[tool(
        description = "Pearson chi-square test of independence between two categorical variables",
        annotations(title = "Chi-square test", read_only_hint = true)
    )]
    pub async fn chi_square_test(&self, Parameters(req): Parameters<tests::TwoVarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::chi_square_test(&self.state, req)?))
    }

    #[tool(
        description = "Mann-Whitney U test of a numeric variable between two groups (nonparametric)",
        annotations(title = "Mann-Whitney U", read_only_hint = true)
    )]
    pub async fn mann_whitney_u_test(&self, Parameters(req): Parameters<tests::ByGroupRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::mann_whitney_u_test(&self.state, req)?))
    }

    #[tool(
        description = "Paired-samples t-test of the mean difference between two numeric variables (each row is one paired observation)",
        annotations(title = "Paired t-test", read_only_hint = true)
    )]
    pub async fn paired_t_test(&self, Parameters(req): Parameters<tests::TwoVarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::paired_t_test(&self.state, req)?))
    }

    #[tool(
        description = "Fisher's exact test of independence on a 2x2 table from two categorical variables that each have exactly two categories (alternative: two-sided, less, or greater)",
        annotations(title = "Fisher's exact test", read_only_hint = true)
    )]
    pub async fn fisher_exact_test(&self, Parameters(req): Parameters<tests::FisherRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::fisher_exact_test(&self.state, req)?))
    }

    #[tool(
        description = "Wilcoxon signed-rank test on paired observations of two numeric variables (nonparametric)",
        annotations(title = "Wilcoxon signed-rank", read_only_hint = true)
    )]
    pub async fn wilcoxon_signed_rank_test(&self, Parameters(req): Parameters<tests::TwoVarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::wilcoxon_signed_rank_test(&self.state, req)?))
    }

    #[tool(
        description = "Kruskal-Wallis H test of a numeric variable across the groups of a factor (nonparametric, 2+ groups)",
        annotations(title = "Kruskal-Wallis", read_only_hint = true)
    )]
    pub async fn kruskal_wallis_test(&self, Parameters(req): Parameters<tests::ByGroupRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(tests::kruskal_wallis_test(&self.state, req)?))
    }

    // --- Normality tests ---------------------------------------------------

    #[tool(
        description = "Shapiro-Wilk test of normality for a numeric variable",
        annotations(title = "Shapiro-Wilk", read_only_hint = true)
    )]
    pub async fn shapiro_wilk(&self, Parameters(req): Parameters<describe::VarRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(normality::shapiro_wilk(&self.state, req)?))
    }

    #[tool(
        description = "One-sample Kolmogorov-Smirnov normality test (test_type: lilliefors or one_sample; one_sample needs mean and std_dev)",
        annotations(title = "Kolmogorov-Smirnov test", read_only_hint = true)
    )]
    pub async fn ks_normality_test(&self, Parameters(req): Parameters<normality::KsRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(normality::ks_normality_test(&self.state, req)?))
    }

    // --- Correlation & regression ------------------------------------------

    #[tool(
        description = "Correlation between two variables (method: pearson, spearman, or kendall)",
        annotations(title = "Correlation (pair)", read_only_hint = true)
    )]
    pub async fn correlation_pair(&self, Parameters(req): Parameters<regression::CorrelationRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::correlation_pair(&self.state, req)?))
    }

    #[tool(
        description = "Correlation between every pair of the given variables (method: pearson, spearman, or kendall)",
        annotations(title = "Correlation matrix", read_only_hint = true)
    )]
    pub async fn correlation(&self, Parameters(req): Parameters<regression::CorrelationMatrixRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::correlation(&self.state, req)?))
    }

    #[tool(
        description = "Fit a linear regression (OLS) of a dependent variable on independent variables",
        annotations(title = "Linear regression", read_only_hint = true)
    )]
    pub async fn linear_regression(&self, Parameters(req): Parameters<regression::RegressionRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::linear_regression(&self.state, req)?))
    }

    #[tool(
        description = "Fit a binary logistic regression of a 0/1 outcome on independent variables",
        annotations(title = "Logistic regression", read_only_hint = true)
    )]
    pub async fn logistic_regression(&self, Parameters(req): Parameters<regression::RegressionRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::logistic_regression(&self.state, req)?))
    }

    #[tool(
        description = "Variance inflation factors for the given predictors (multicollinearity diagnostics; needs at least two predictors)",
        annotations(title = "Variance inflation factors", read_only_hint = true)
    )]
    pub async fn vif(&self, Parameters(req): Parameters<multivariate::VarsRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::vif(&self.state, req)?))
    }

    #[tool(
        description = "Partial correlation of two variables whilst controlling for one or more control variables (method: pearson, spearman, or kendall)",
        annotations(title = "Partial correlation", read_only_hint = true)
    )]
    pub async fn partial_correlation(&self, Parameters(req): Parameters<regression::PartialCorrRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(regression::partial_correlation(&self.state, req)?))
    }

    // --- ANOVA follow-ups & multifactor ANOVA ------------------------------

    #[tool(
        description = "ANOVA post-hoc comparisons of a numeric variable across the groups of a factor (method: bonferroni, tukey, or scheffe)",
        annotations(title = "Post-hoc comparisons", read_only_hint = true)
    )]
    pub async fn post_hoc(&self, Parameters(req): Parameters<anova::PostHocRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(anova::post_hoc(&self.state, req)?))
    }

    #[tool(
        description = "Multifactor (factorial) ANOVA of a numeric variable on two or more factors with two-way interactions (ss_type: type_i or type_ii)",
        annotations(title = "Factorial ANOVA", read_only_hint = true)
    )]
    pub async fn factorial_anova(&self, Parameters(req): Parameters<anova::FactorialAnovaRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(anova::factorial_anova(&self.state, req)?))
    }

    // --- Multivariate -------------------------------------------------------

    #[tool(
        description = "Principal component analysis of numeric variables (matrix: correlation or covariance)",
        annotations(title = "PCA", read_only_hint = true)
    )]
    pub async fn pca(&self, Parameters(req): Parameters<multivariate::PcaRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(multivariate::pca(&self.state, req)?))
    }

    #[tool(
        description = "Cronbach's alpha reliability analysis of scale items",
        annotations(title = "Reliability (alpha)", read_only_hint = true)
    )]
    pub async fn reliability(&self, Parameters(req): Parameters<multivariate::VarsRequest>) -> Result<Json<Value>, String> {
        let _guard = self.gate.lock().await;
        Ok(Json(multivariate::reliability(&self.state, req)?))
    }
}

// `#[tool_handler]` generates call_tool / list_tools / get_tool / get_info
// (with the tools capability and server metadata below). We only override the
// protocol-version set to advertise both the 2026-07-28 and 2025-11-25 specs.
#[tool_handler(
    name = "socstat-mcp",
    instructions = "Statistical analysis over named datasets. Workflow: 1) load_dataset to register a dataset, 2) dataset_info / preview to inspect it, 3) run any analysis tool by dataset name. Results are returned as JSON."
)]
impl ServerHandler for SocstatMcpServer {
    fn supported_protocol_versions(&self) -> Cow<'static, [ProtocolVersion]> {
        Cow::Borrowed(&[ProtocolVersion::V_2026_07_28, ProtocolVersion::V_2025_11_25])
    }
}