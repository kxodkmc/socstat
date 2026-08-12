# socstat

轻量级、可嵌入的 Rust 统计分析 SDK，为任意平台与应用的集成场景提供 SPSS 级别的核心统计分析能力。

- **可嵌入**：作为库被其他软件消费，公开 API 稳定、可序列化，不依赖具体调用环境。
- **轻量**：默认构建仅依赖 `nalgebra`、`statrs`、`serde`、`thiserror`；可选能力全部通过 Cargo feature 门控。

## 设计原则

- **结果可序列化**：所有公开结果类型与数据模型均派生 `Serialize` / `Deserialize`，可直接以 JSON/FFI 载荷交付宿主。
- **列式类型化存储**：数据按列、按类型连续存储（`ColumnData::Numeric(Vec<Option<f64>>)` / `Text(Vec<Option<String>>)`），缺失值一律使用 `None`，不使用 `NaN`、`-999` 等哨兵值。
- **统计正确性优先**：样本方差使用 n−1，用户缺失值剔除，权重生效，百分比守恒；数值计算采用两遍算法，避免灾难性抵消。
- **SPSS 语义为基线**：支持变量标签、值标签、缺失值规则、测量水平、个案权重，但不被 SPSS 局限。
- **宿主友好**：API 面向调用方设计，公开接口保持稳定，不泄露内部借用与实现细节；文本/数值列均可直接以 `&[Option<T>]` 切片访问。

## 快速开始

```toml
# Cargo.toml
[dependencies]
socstat = "0.1"

# 如需 SPSS .sav 读写
# socstat = { version = "0.1", features = ["sav"] }
```

```rust
use socstat::prelude::*;

fn main() -> SocStatResult<()> {
    let mut ds = socstat::read().csv("data.csv")?;

    // 描述统计
    let d = ds.descriptive("income")?;
    println!("Mean: {:.2}, Std: {:.2}", d.mean, d.std_dev);

    // 频数表
    let freq = ds.frequencies("gender")?;
    for row in freq.iter() {
        println!("{}: {} ({:.1}%)", row.value, row.count, row.valid_percent);
    }

    // 计算新变量
    ds.compute("bmi", |row| {
        let w = row.numeric("weight")?;
        let h = row.numeric("height")?;
        Some(w / (h * h))
    })?;

    // 筛选个案
    ds.filter(|row| row.numeric("age") > Some(18.0))?;

    socstat::write().json(&ds, "out.json")?;
    Ok(())
}
```

## 数据模型

数据按**列主序**存储：每个变量对应一列类型化连续内存，`None` 表示缺失。

| 类型 | 说明 |
|------|------|
| `Dataset` | 数据集：变量元数据 + 列式数据 + 可选的个案权重变量 + 名称/元数据键值对 |
| `Variable` | 变量元数据：名称、标签、类型、测量水平、缺失规则、值标签、宽度、是否权重 |
| `ColumnData` | 列存储：`Numeric(Vec<Option<f64>>)` / `Text(Vec<Option<String>>)` 两种变体 |
| `Value` | 瞬态行级值：`Number(f64)` / `Text(String)` / `Missing`，仅用于构建与读取，不作为存储 |
| `RowView` | 行级只读视图，供 `compute` / `filter` 等闭包使用，`row.numeric("x")` 无需 `?` |
| `DataType` | `Numeric` / `Text` |
| `MeasureType` | 测量水平：`Nominal` / `Ordinal` / `Scale` |
| `MissingSpec` | 用户缺失规则：无 / 离散（最多 3 个）/ 区间 `[low, high]`（可选加 1 个离散值） |
| `ValueFormat` | 显示格式：`General` / `Fixed` / `Scientific` / `Date` / `DateTime` / `Percent` / `Currency` |

### 构建数据集

```rust
let mut ds = Dataset::new();
ds.add_var(Variable::numeric("age").label("Age in years").measure(MeasureType::Scale))?;
ds.add_var(
    Variable::text("gender")
        .value_label("M", "Male")
        .value_label("F", "Female"),
)?;
ds.add_var(Variable::numeric("score").missing_discrete(&[-1.0]))?;

ds.push_row(vec![Value::Number(25.0), Value::Text("M".into()), Value::Number(85.0)])?;
```

`Variable` 提供链式构建器：`.label(..)`、`.measure(..)`、`.width(..)`、`.format(..)`、`.value_label(..)`、`.missing_discrete(&[..])`、`.missing_range(low, high, discrete)`、`.weight()`。

### 列访问

- `ds.numeric_slice("x")` / `ds.text_slice("x")` → `&[Option<f64>]` / `&[Option<String>]`
- `ds.numeric_values("x")` → `Vec<f64>`（剔除缺失与用户缺失）
- `ds.n_valid("x")` / `ds.n_missing("x")` → 有效/缺失计数（用户缺失计入缺失）

### 个案权重

权重语义为**频率权重**（个案权重）：每个个案按权重值重复计数，权重缺失或 ≤ 0 的个案从分析中剔除。

```rust
let mut ds = Dataset::new();
ds.add_var(Variable::numeric("income"))?;
ds.add_var(Variable::numeric("w").weight())?;   // 声明为权重变量
// 或运行时指定：ds.set_weight("w")
```

## 数据变换

| 方法 | 说明 |
|------|------|
| `compute(name, closure)` | 逐行计算新数值变量（闭包返回 `Option<f64>`） |
| `compute_text(name, closure)` | 逐行计算新文本变量 |
| `recode(name, closure)` | 原地重编码数值变量（闭包接收 `Option<f64>`） |
| `filter(predicate)` | 保留满足谓词的个案，返回保留数量 |
| `sort_by(name, desc)` | 按数值变量排序（缺失值视为 −∞） |
| `keep(names)` | 仅保留指定变量 |

```rust
ds.compute("bmi", |row| {
    let w = row.numeric("weight")?;
    let h = row.numeric("height")?;
    Some(w / (h * h))
})?;

ds.recode("age", |v| match v {
    Some(n) if n < 18.0 => Some(1.0),
    Some(n) if n < 65.0 => Some(2.0),
    Some(_) => Some(3.0),
    None => None,
})?;

ds.filter(|row| row.numeric("age") > Some(18.0))?;
```

## 统计分析

所有分析通过 `StatsExt` trait 提供（`use socstat::prelude::*;` 后直接调用），自动使用已设置的权重。

### 描述统计

`ds.descriptive(var)` 返回 `Descriptive`，包含：`n`、`mean`、`std_dev`、`variance`、`min`、`max`、`range`、`sum`、`median`、`q1`、`q3`、`sem`（标准误）、`skewness`、`kurtosis`（超额峰度）、`ci_95`（均值 95% 置信区间，t 分布）。

### 频数与交叉表

| 方法 | 说明 |
|------|------|
| `frequencies(var)` | 频数表：`FrequencyTable`，每行含 `value` / `count` / `percent` / `valid_percent` / `cumulative`，并给出 `n_valid` / `n_missing` / `total` |
| `crosstab(row, col)` | 交叉表：`Crosstab`，含 `counts`、`expected`、`row_pcts`、`col_pcts`、`total_pcts` 与行列合计 |

```rust
let freq = ds.frequencies("gender")?;
let cross = ds.crosstab("gender", "education")?;
```

### 假设检验

| 方法 | 说明 |
|------|------|
| `ttest_independent(dep, group)` | 独立样本 t 检验：组统计量 + Levene 方差齐性检验 + 合并方差（pooled）模型 + Welch 模型。要求分组变量恰好两个类别 |
| `anova_one_way(dep, factor)` | 单因素 ANOVA：组间/组内平方和、F、p、效应量 η²。要求至少两个组 |
| `chi_square_test(v1, v2)` | 卡方独立性检验：观测/期望频数、χ²、自由度、p。要求两变量均至少两个类别 |
| `mann_whitney_u(dep, group)` | Mann–Whitney U 非参数检验：秩和、U、z（含结校正）、渐近 p。要求分组变量恰好两个类别 |

```rust
let t = ds.ttest_independent("len", "supp")?;
println!("t = {:.4}, p = {:.4}", t.equal_variances.t_statistic, t.equal_variances.p_value);
println!("Welch: t = {:.4}, p = {:.4}", t.unequal_variances.t_statistic, t.unequal_variances.p_value);

let a = ds.anova_one_way("len", "dose")?;
println!("F = {:.4}, p = {:.4}, eta^2 = {:.4}", a.f_statistic, a.p_value, a.eta_squared);

let c = ds.chi_square_test("supp", "outcome")?;
let m = ds.mann_whitney_u("len", "supp")?;
```

**约束**：`ttest_independent` 与 `mann_whitney_u` 要求分组变量恰好两个类别；`anova_one_way` 要求至少两个组；`chi_square_test` 要求两个变量均至少两个类别。违规返回 `SocStatError::InsufficientData`。

**说明**：Mann–Whitney U 采用渐近正态近似（含结校正），与 SPSS 的渐近显著性一致；小样本下与 R `wilcox.test` 的精确检验可能不同。

### 相关分析

`ds.correlation(vars, method)` 对给定数值变量两两计算相关（上三角）。`CorrelationMethod` 支持 `Pearson` / `Spearman` / `Kendall`；结果 `Vec<CorrelationPair>` 中仅填充所选方法的系数。

```rust
for p in ds.correlation(&["height", "weight", "age"], CorrelationMethod::Pearson)? {
    if let Some(r) = &p.pearson {
        println!("{} ~ {}: r = {:.3}, p = {:.4}", p.var1, p.var2, r.coefficient, r.p_value);
    }
}
```

缺失值（含用户缺失）按两两剔除（pairwise）；权重生效。

### 线性回归（OLS）

`ds.regression(dep, &indeps)` 拟合普通最小二乘线性回归，恒含截距。报告：各系数（估计值、标准误、t、p）、`R²`、`adjusted R²`、整体 F 与 p、残差标准误、自由度、`model_formula`。

```rust
let model = ds.regression("income", &["age", "education"])?;
println!("R² = {:.3}, F = {:.2}, p = {:.4}", model.r_squared, model.f_statistic, model.f_p_value);
```

缺失值（系统缺失与用户缺失）按 listwise 剔除；权重生效。设计矩阵奇异时返回 `SocStatError::SingularMatrix`。

### 逻辑回归（IRLS）

`ds.logistic_regression(dep, &indeps)` 用迭代加权最小二乘拟合二元逻辑回归，恒含截距。因变量必须为 0/1 数值变量。报告：各系数（估计值、标准误、z、p）、对数似然与零模型对数似然、AIC、零偏差与残差偏差、自由度、迭代次数与收敛标志。

```rust
let model = ds.logistic_regression("defaulted", &["age", "income"])?;
println!("AIC = {:.2}, residual deviance = {:.2}", model.aic, model.residual_deviance);
```

数据完全分离时返回 `SocStatError::CompleteSeparation`；加权设计矩阵奇异时返回 `SocStatError::SingularMatrix`。

### 多变量分析

| 方法 | 说明 |
|------|------|
| `pca(vars, matrix)` | 主成分分析。`PcaMatrix` 可选 `Covariance` 或 `Correlation` 矩阵；报告各主成分的特征值、方差解释比、累计方差比、特征向量与载荷，并保存训练均值/标准差以对新数据评分（`PcaResult::scores`） |
| `reliability(vars)` | 信度分析（Cronbach α）：总体 α、标准化 α、量表均值/方差，以及逐条目的校正题总相关与删除后 α |

两者均采用严格 listwise 剔除（所有选定变量在该行都有有限有效值才保留），权重生效；样本不足时返回 `SocStatError::InsufficientData`。

```rust
let pca = ds.pca(&["height", "weight", "age"], PcaMatrix::Correlation)?;
for c in &pca.components {
    println!("λ = {:.3} ({:.1}%)", c.eigenvalue, c.explained_variance_ratio * 100.0);
}

let rel = ds.reliability(&["q1", "q2", "q3"])?;
println!("α = {:.3}", rel.alpha);
```

## 分布函数

`dist` 模块封装 `statrs`，通过统一的 `Distribution` trait（`pdf` / `cdf` / `inverse_cdf`）提供，其余模块不直接依赖 `statrs`。

```rust
use socstat::dist::{Distribution, NormalDist, StudentsTDist, FDist, ChiSquaredDist};

let n = NormalDist::standard();
assert!((n.cdf(0.0) - 0.5).abs() < 1e-10);

let t = StudentsTDist::new(10.0)?;
let f = FDist::new(3.0, 10.0)?;
let chi = ChiSquaredDist::new(5.0)?;
```

## 数据读写

使用构建器入口 `read()` / `write()`；`read().auto(path)` / `write().auto(ds, path)` 按扩展名自动识别格式。

```rust
let ds = socstat::read().csv("data.csv")?;
let ds = socstat::read().json("data.json")?;
let ds = socstat::read().sav("data.sav")?;   // 需启用 sav 特性
let ds = socstat::read().auto("data.csv")?;

socstat::write().csv(&ds, "out.csv")?;
socstat::write().json(&ds, "out.json")?;
socstat::write().sav(&ds, "out.sav")?;       // 需启用 sav 特性
```

| 格式 | 说明 |
|------|------|
| CSV | 读取通过采样自动推断列类型：某列所有采样值均可解析为 `f64` 则为数值列，否则为文本列；空串视为缺失，缺失值写为空串 |
| JSON | 数据集完整 JSON 互操作格式，适合 Web 管线与调试 |
| SPSS .sav | 遵循 GNU PSPP System File Format 规范；读取支持压缩方式 0/1/2（uncompressed / bytecode / zlib `$FL3`），写出为压缩方式 1（bytecode `$FL2`），SAS/PSPP/R 可读 |

**`.sav` 细节**：round-trip 保留变量标签、值标签、用户缺失（离散/区间）、度量水平、权重变量、长变量名（>8 字符）、字符串宽度；`f64` 数值逐位保精度。字符串按 UTF-8 写出；读取时对非 UTF-8 文件按 Latin-1 逐字节映射。EBCDIC 编码、非 IEEE-754 浮点（`bias != 100.0`）、超长字符串（宽度 > 255）明确报错。

## Cargo 特性

| 特性 | 内容 | 默认 |
|------|------|------|
| `csv` | CSV + JSON 读写 | 是 |
| `sav` | SPSS .sav 二进制读写 | 否 |
| `excel` | 已声明依赖（`calamine` + `rust_xlsxwriter`），读写逻辑尚未实现 | 否 |
| `datetime` | 已声明依赖（`chrono`），尚未实现 | 否 |
| `full` | `csv` + `excel` + `datetime` + `sav` | 否 |

## 错误处理

所有可能失败的 API 返回 `SocStatResult<T>`（`Result<T, SocStatError>`）。主要错误变体：

- `VariableNotFound`、`VariableIndexOutOfBounds`：变量不存在或索引越界
- `TypeMismatch`：类型不匹配（如对文本列做数值运算），不做静默转换
- `MissingNumber`、`DuplicateVariable`、`RowLengthMismatch`、`ColumnLengthMismatch`
- `Computation`、`InsufficientData`：计算失败或样本不足
- `SingularMatrix`、`ConvergenceNotReached`、`CompleteSeparation`：数值失败（回归等）
- `Io`、`Csv`、`Json`：底层 I/O 与解析错误
- `Sav`、`UnsupportedFormat`：`.sav` 格式错误与不支持格式

## 示例

```bash
cargo run --example basic_stats    # 数据模型、变换、描述统计、频数、交叉表、分布
cargo run --example tests_demo     # 假设检验：t、ANOVA、卡方、Mann–Whitney U，并输出 JSON
```

## 开发规范

项目约束见 `AGENTS.md`（Hard Rules、架构边界、实现路线）。提交前验证：

```bash
cargo build                 # 默认特性编译
cargo test                  # 单元测试 + 文档测试
cargo test --features full  # 特性门控代码
cargo clippy -- -D warnings # 无新增警告
```

## 已知限制

- `excel` 与 `datetime` 特性已声明依赖，但读写/支持逻辑尚未实现，启用后不会提供对应 API。
- 权重仅支持频率权重，不支持概率（复杂抽样）权重。
- Mann–Whitney U 只提供渐近近似，不提供精确 p 值。
- 结果中的极端退化情形（如完全常数组的 F/t 统计量）可能产生非有限值（`NaN`/`Inf`），严格 JSON 序列化无法表示此类值。
- `.sav` 写出方向不支持 zlib 压缩（`.zsav`）与 `.por` 便携格式；读入时忽略未知扩展记录与文件级文档记录。
