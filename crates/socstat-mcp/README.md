# socstat-mcp

基于 [`socstat`](../../README.md) 的 AI 友好型 MCP（Model Context Protocol）服务器。它把 socstat 的统计分
析能力以 **MCP tool** 的形式暴露给 AI 主机（如 Claude、Cursor 等），通过 **stdio** 传输协议交互。

socstat-mcp 是工作区的**可选成员**：核心 `socstat` crate 保持轻量，绝不会编译 `rmcp` / `tokio`；只有需要
MCP 服务时才构建本 crate。

---

## 目录

- [工作原理](#工作原理)
- [特性开关](#特性开关)
- [构建与运行](#构建与运行)
- [MCP 客户端配置](#mcp-客户端配置)
- [工具参考](#工具参考)
- [典型工作流](#典型工作流)
- [作为库嵌入](#作为库嵌入)
- [设计说明](#设计说明)
- [开发与验证](#开发与验证)

---

## 工作原理

MCP 工具调用本身是无状态的。socstat-mcp 通过一个**共享、有状态**的数据集存储，让 AI 主机遵循真实的分析
工作流：

1. `load_dataset` 一次性将数据文件加载进共享内存，并按名注册；
2. 后续所有分析工具都**按数据集名称**引用该数据；
3. 全程无需重复加载数据。

```text
                    ┌───────────────────────────────────────────────┐
  AI 主机 (Claude/   │  stdio  (stdin / stdout)                     │
  Cursor/ ... ) ────▶│                                               │
                    │  +------------------+                          │
                    │  |  SocstatMcpServer|  (34 个 tool)            │
                    │  +--------+---------+                          │
                    │           │                                    │
                    │  +--------v---------+                          │
                    │  |   SharedState    |  按名注册 / 读取 Dataset │
                    │  |  (Dataset 存储)  |                          │
                    │  +------------------+                          │
                    └───────────────────────────────────────────────┘
```

---

## 特性开关

| 特性 | 内容 | 默认 |
|------|------|------|
| `csv` | 允许加载 CSV / JSON 格式数据 | 是 |
| `sav` | 允许加载 SPSS `.sav` 格式数据 | 否 |
| `full` | `csv` + `sav` | 否 |

> **注意**：`.sav` 读取仅在启用 `sav` 特性时可用。默认构建下调用 `load_dataset` 加载 `.sav`
> 文件会返回类似 `format 'sav' not available (feature not enabled)` 的错误。

---

## 构建与运行

```bash
# 默认构建（CSV / JSON）
cargo build -p socstat-mcp

# 启用 SPSS .sav 支持
cargo build -p socstat-mcp --features sav   # 或 --features full
```

构建产物为可执行文件 `socstat-mcp`，由 MCP 客户端以子进程方式启动，通过 stdin/stdout 通信：

```bash
socstat-mcp
```

---

## MCP 客户端配置

在支持 MCP 的客户端（Claude Desktop、Cursor 等）中，将 `socstat-mcp` 注册为一个 stdio 风格的服务。
`command` 应为可执行文件的路径（或确保其在 `PATH` 中）：

```json
{
  "mcpServers": {
    "socstat": {
      "command": "/absolute/path/to/socstat-mcp",
      "args": []
    }
  }
}
```

启动后，客户端会列举出本 crate 暴露的全部工具（见下节）。

---

## 工具参考

共 **34 个工具**，按功能分组。除 `list_datasets` 外，其余工具均需 `dataset` 参数（已加载的数据集名称）。

### 数据管理

| 工具 | 参数 | 说明 |
|------|------|------|
| `list_datasets` | — | 列出所有已加载数据集及其行数、变量数 |
| `load_dataset` | `name`, `path` | 从文件（按扩展名识别格式）加载数据集并注册，返回 schema |
| `dataset_info` | `dataset` | 描述数据集的形状与各变量（名称、标签、类型、测量水平、有效/缺失计数） |
| `preview` | `dataset`, `rows`(默认 10) | 预览前若干行的类型化单元格值 |
| `drop_dataset` | `dataset` | 从共享内存移除一个数据集 |

### 数据变换

| 工具 | 参数 | 说明 |
|------|------|------|
| `recode` | `dataset`, `src`, `dst`, `mapping[]` | 按离散映射把数值变量重编码进新变量（保留源变量） |
| `filter` | `dataset`, `var`, `op`, `value` | 仅保留满足比较条件的行；返回保留数量与新数据集信息 |
| `sort` | `dataset`, `var`, `descending` | 按数值变量升序 / 降序排序 |
| `keep` | `dataset`, `vars[]` | 仅保留指定变量，丢弃其余列 |
| `set_weight` | `dataset`, `var` | 设置个案权重变量（后续统计按频率权重计算） |
| `compute` | `dataset`, `new_var`, `left`, `operator`, `right` | 逐行计算新数值变量 `left op right` |

`filter` 的 `op` 取值：`gt`、`ge`、`lt`、`le`、`eq`、`ne`。非法操作符会返回错误，不会静默清空数据。

`compute` 的 `operator` 取值：`+`、`-`、`*`、`/`。`left` / `right` 既可以是列名（字符串），也可以是数值
常量（数字）。非法操作符会返回错误，不会静默生成空列。

`set_weight` 一经设置即无法清除，只能改设其他变量或重新加载数据集。

### 描述统计

| 工具 | 参数 | 说明 |
|------|------|------|
| `descriptive` | `dataset`, `var` | 数值变量的描述统计：均值、标准差、中位数、四分位数、偏度、峰度、95% 置信区间等 |
| `frequencies` | `dataset`, `var` | 任意变量的频数表（计数与百分比） |
| `crosstab` | `dataset`, `row_var`, `col_var` | 两变量的交叉表（列联表） |

### 假设检验

| 工具 | 参数 | 说明 |
|------|------|------|
| `independent_t_test` | `dataset`, `dep_var`, `group_var` | 独立样本 t 检验（合并方差 / Welch / Levene） |
| `one_way_anova` | `dataset`, `dep_var`, `group_var` | 单因素 ANOVA |
| `chi_square_test` | `dataset`, `var1`, `var2` | 两分类变量的卡方独立性检验 |
| `mann_whitney_u_test` | `dataset`, `dep_var`, `group_var` | Mann–Whitney U 非参数检验 |
| `paired_t_test` | `dataset`, `var1`, `var2` | 配对样本 t 检验（每一行为一对观测） |
| `fisher_exact_test` | `dataset`, `var1`, `var2`, `alternative` | 2×2 表 Fisher 精确检验（`alternative`: `two-sided` / `less` / `greater`） |
| `wilcoxon_signed_rank_test` | `dataset`, `var1`, `var2` | 配对观测的 Wilcoxon 符号秩检验（需 ≥10 个非零差值） |
| `kruskal_wallis_test` | `dataset`, `dep_var`, `group_var` | Kruskal–Wallis H 非参数检验（2+ 组） |

### 正态性检验

| 工具 | 参数 | 说明 |
|------|------|------|
| `shapiro_wilk` | `dataset`, `var` | 数值变量的 Shapiro–Wilk 正态性检验（Royston AS R94） |
| `ks_normality_test` | `dataset`, `var`, `test_type`, `mean`, `std_dev` | 单样本 K-S 正态性检验（`test_type`: `lilliefors` 或 `one_sample`） |

### ANOVA 后续检验与多因素 ANOVA

| 工具 | 参数 | 说明 |
|------|------|------|
| `post_hoc` | `dataset`, `dep_var`, `factor_var`, `method` | ANOVA 事后多重比较（`method`: `bonferroni` / `tukey` / `scheffe`） |
| `factorial_anova` | `dataset`, `dep_var`, `factors[]`, `ss_type` | 多因素（析因）ANOVA，含二阶交互（`ss_type`: `type_i` / `type_ii`） |

### 相关与回归

| 工具 | 参数 | 说明 |
|------|------|------|
| `correlation_pair` | `dataset`, `var1`, `var2`, `method` | 两变量的相关系数（`pearson` / `spearman` / `kendall`） |
| `correlation` | `dataset`, `vars[]`, `method` | 给定变量两两之间的相关矩阵（上三角） |
| `vif` | `dataset`, `vars[]` | 方差膨胀因子（多重共线性诊断，需 ≥2 个预测变量） |
| `partial_correlation` | `dataset`, `var1`, `var2`, `controls[]`, `method` | 控制变量的偏相关（残差法，`controls` 至少 1 个） |
| `linear_regression` | `dataset`, `dep_var`, `indep_vars[]` | 线性回归（OLS，恒含截距） |
| `logistic_regression` | `dataset`, `dep_var`, `indep_vars[]` | 二元逻辑回归（因变量须为 0/1） |

### 多变量分析

| 工具 | 参数 | 说明 |
|------|------|------|
| `pca` | `dataset`, `vars[]`, `matrix` | 主成分分析（`matrix`: `correlation` 或 `covariance`） |
| `reliability` | `dataset`, `vars[]` | Cronbach α 信度分析 |

---

## 典型工作流

一个完整的 AI 驱动分析流程示例：

```text
1. load_dataset   { "name": "survey", "path": "/data/survey.csv" }
2. dataset_info   { "dataset": "survey" }
3. preview        { "dataset": "survey", "rows": 5 }
4. descriptive    { "dataset": "survey", "var": "income" }
5. frequencies    { "dataset": "survey", "var": "gender" }
6. crosstab       { "dataset": "survey", "row_var": "gender", "col_var": "education" }
7. independent_t_test { "dataset": "survey", "dep_var": "income", "group_var": "gender" }
8. linear_regression  { "dataset": "survey", "dep_var": "income", "indep_vars": ["age", "education"] }
```

所有工具返回结构化 JSON，可直接被 AI 主机解读或转交下游处理。

---

## 作为库嵌入

除作为独立可执行文件外，也可把 socstat-mcp 作为库嵌入到自定义 Rust 程序中：

```rust,no_run
use socstat_mcp::{SharedState, SocstatMcpServer};
use rmcp::ServiceExt;

# async fn run() -> Result<(), Box<dyn std::error::Error>> {
let server = SocstatMcpServer::new(SharedState::arc());
let running = server.serve((tokio::io::stdin(), tokio::io::stdout())).await?;
running.waiting().await?;
# Ok(())
# }
```

创建服务器时传入一个 `Arc<SharedState>`，可用于在多个服务器实例或外部逻辑间共享同一份数据集存储。

---

## 设计说明

- **有状态会话**：`SharedState` 在服务器生命周期内持有命名数据集，`load_dataset` 一次加载、后续按名引用。
- **结果可序列化**：所有工具返回 `serde_json::Value`，来源是 socstat core 中已派生 `Serialize` 的结果类型，
  可直接以 JSON/FFI 载荷交付宿主。
- **权重透明**：分析工具透传 socstat 的 `StatsExt`，自动使用数据集上已设置的个案权重。
- **串行调度**：服务器用内部锁串行化工具分发，保证有状态工作流（加载 → 分析）严格有序，避免并发竞态。
- **错误友好**：数据集缺失、操作符非法、统计不适用等情况均返回明确错误信息，而非静默失败或损坏数据。

---

## 开发与验证

```bash
cargo build -p socstat-mcp            # 默认特性编译
cargo test -p socstat-mcp             # 单元测试 + 文档测试
cargo build -p socstat-mcp --features full   # 特性门控代码编译
cargo clippy -p socstat-mcp -- -D warnings   # 无新增警告
```

socstat 的完整能力与 API 说明见 [根 README](../../README.md) 与 [AGENTS.md](../../AGENTS.md)。