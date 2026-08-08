# socstat

轻量级、可嵌入的 Rust 统计分析 SDK，提供 SPSS 级别的核心统计分析能力，面向需要数据分析能力的任何平台与应用。

## 设计原则

- **结果可序列化**：所有公开结果类型（`Descriptive`、`FrequencyTable`、`Crosstab`、各类检验结果）均派生 `Serialize` / `Deserialize`，可直接以 JSON/FFI 载荷交付宿主。
- **列式类型化存储**：数据按列、按类型连续存储（`ColumnData::Numeric(Vec<Option<f64>>)` / `Text(Vec<Option<String>>)`），缺失值一律使用 `None`，不使用 `NaN`、`-999` 等哨兵值。
- **统计正确性优先**：样本方差使用 n−1，用户缺失值剔除，权重生效，百分比守恒。数值计算采用两遍算法，避免灾难性抵消。
- **SPSS 语义为基线**：支持变量标签、值标签、缺失值规则、测量水平、个案权重，但不被 SPSS 局限。
- **宿主友好**：API 面向调用方设计，公开接口保持稳定，不泄露内部借用与实现细节。

## 快速开始

```toml
# Cargo.toml
[dependencies]
socstat = "0.1"
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

## 核心概念

| 类型 | 说明 |
|------|------|
| `Dataset` | 数据集：变量元数据 + 列式数据 + 可选的个案权重变量 |
| `Variable` | 变量元数据：名称、标签、类型、测量水平、缺失规则、值标签、宽度 |
| `ColumnData` | 列存储：`Numeric` / `Text` 两种变体，`None` 表示缺失 |
| `Value` | 瞬态行级值：`Number(f64)` / `Text(String)` / `Missing`，仅用于构建与读取，不做存储 |
| `MissingSpec` | 用户缺失规则：离散值（最多 3 个）或范围 [low, high] 加一个离散值 |
| `MeasureType` | 测量水平：`Nominal` / `Ordinal` / `Scale` |

```rust
// 构建数据集
let mut ds = Dataset::new();
ds.add_var(Variable::numeric("age").label("Age").measure(MeasureType::Scale))?;
ds.add_var(
    Variable::text("gender")
        .value_label("M", "Male")
        .value_label("F", "Female"),
)?;
ds.add_var(Variable::numeric("score").missing_discrete(&[-1.0]))?;

ds.push_row(vec![Value::Number(25.0), Value::Text("M".into()), Value::Number(85.0)])?;
```

### 个案权重

权重语义为**频率权重**（个案权重）：每个个案按权重值重复计数，权重缺失或 ≤ 0 的个案从分析中剔除。

```rust
let mut ds = Dataset::new();
ds.add_var(Variable::numeric("income"))?;
ds.add_var(Variable::numeric("w").weight())?;   // 声明为权重变量
// 或运行时指定：ds.set_weight("w")
```

## 统计分析

所有分析通过 `StatsExt` trait 提供，自动使用已设置的权重。

| 方法 | 说明 |
|------|------|
| `descriptive(var)` | 描述统计：n、均值、标准差、方差、中位数、分位数、偏度、峰度、95% CI |
| `frequencies(var)` | 频数表：计数、百分比、有效百分比、累计百分比 |
| `crosstab(row, col)` | 交叉表：观测频数、期望频数、行列/总百分比 |

```rust
let d = ds.descriptive("income")?;
let freq = ds.frequencies("gender")?;
let cross = ds.crosstab("gender", "education")?;
```

## 假设检验

| 方法 | 说明 |
|------|------|
| `ttest_independent(dep, group)` | 独立样本 t 检验：Levene 方差齐性检验 + 合并方差模型 + Welch 模型 |
| `anova_one_way(dep, factor)` | 单因素 ANOVA：组间/组内平方和、F、p、效应量 η² |
| `chi_square_test(v1, v2)` | 卡方独立性检验：期望频数、χ²、自由度、p |
| `mann_whitney_u(dep, group)` | Mann–Whitney U 非参数检验：秩和、渐近正态近似（含结校正） |

```rust
let t = ds.ttest_independent("len", "supp")?;
println!("t = {:.4}, p = {:.4}", t.equal_variances.t_statistic, t.equal_variances.p_value);

let a = ds.anova_one_way("len", "dose")?;
println!("F = {:.4}, p = {:.4}, eta^2 = {:.4}", a.f_statistic, a.p_value, a.eta_squared);

let c = ds.chi_square_test("supp", "outcome")?;
let m = ds.mann_whitney_u("len", "supp")?;
```

**约束**：`ttest_independent` 与 `mann_whitney_u` 要求分组变量恰好两个类别；`anova_one_way` 要求至少两个组；`chi_square_test` 要求两个变量均至少两个类别。违规返回 `SocStatError::InsufficientData`。

**权重说明**：检验的权重为频率权重；复杂抽样权重（概率权重）当前版本不支持。Mann–Whitney U 采用渐近正态近似，与 SPSS 的渐近显著性一致；小样本下与 R `wilcox.test` 的精确检验可能不同。

## 数据变换

| 方法 | 说明 |
|------|------|
| `compute(name, closure)` | 逐行计算新数值变量 |
| `compute_text(name, closure)` | 逐行计算新文本变量 |
| `recode(name, closure)` | 原地重编码数值变量 |
| `filter(predicate)` | 保留满足谓词的个案，返回保留数量 |
| `sort_by(name, desc)` | 按数值变量排序（缺失值视为 −∞） |
| `keep(names)` | 仅保留指定变量 |

```rust
ds.compute("bmi", |row| {
    let w = row.numeric("weight")?;
    let h = row.numeric("height")?;
    Some(w / (h * h))
})?;

ds.filter(|row| row.numeric("age") > Some(18.0))?;
```

## 数据读写

使用构建器入口 `read()` / `write()`，按扩展名自动识别格式（`.auto()`）。

```rust
let ds = socstat::read().csv("data.csv")?;
let ds = socstat::read().json("data.json")?;
let ds = socstat::read().auto("data.csv")?;

socstat::write().csv(&ds, "out.csv")?;
socstat::write().json(&ds, "out.json")?;
```

CSV 读取通过采样自动推断列类型：某列所有采样值均可解析为 `f64` 则为数值列，否则为文本列。缺失值写为空串，读为空串视为缺失。

## 分布函数

`dist` 模块封装 `statrs`，通过统一的 `Distribution` trait（`pdf` / `cdf` / `inverse_cdf`）提供，其他模块不直接依赖 `statrs`。

```rust
let n = NormalDist::standard();
assert!((n.cdf(0.0) - 0.5).abs() < 1e-10);

let t = StudentsTDist::new(10.0)?;
let f = FDist::new(3.0, 10.0)?;
let chi = ChiSquaredDist::new(5.0)?;
```

## Cargo 特性

| 特性 | 内容 | 默认 |
|------|------|------|
| `csv` | CSV + JSON 读写 | 是 |
| `excel` | Excel (.xlsx) 读写（依赖已声明，实现未完成） | 否 |
| `datetime` | 日期/时间值支持 | 否 |
| `full` | 全部特性 | 否 |

## 错误处理

所有可能失败的 API 返回 `SocStatResult<T>`（`Result<T, SocStatError>`）。常见错误：

- `VariableNotFound`、`VariableIndexOutOfBounds`：变量不存在或越界
- `TypeMismatch`：文本列用于数值操作等类型错误（不静默转换）
- `MissingNumber`、`DuplicateVariable`
- `Computation`、`InsufficientData`：计算失败或样本不足
- `Io`、`Csv`、`Json`：底层 I/O 错误

## 开发规范

项目约束详见 `AGENTS.md`（Hard Rules、架构边界、路线图）。修改代码前必须阅读。

提交前验证：

```bash
cargo build                 # 默认特性编译
cargo test                  # 单元测试 + 文档测试
cargo test --features full  # 特性门控代码
cargo clippy -- -D warnings # 无新增警告
```

示例：

```bash
cargo run --example basic_stats
cargo run --example tests_demo
```

实现路线：

| 阶段 | 范围 | 状态 |
|------|------|------|
| P1 | 项目骨架 + data 模块 + error + CSV I/O | 完成 |
| P2 | dist + 描述统计 + 频数 + 交叉表 | 完成 |
| P7 | 数据变换 | 完成 |
| P3 | 假设检验（t、ANOVA、卡方、非参数） | 完成 |
| P4 | 回归（OLS/QR、Logistic） | 待实现 |
| P5 | 多变量（PCA、Cronbach α） | 待实现 |
| P6 | SPSS .sav 读写 | 待实现 |
| P8 | 集成测试 + 示例 + 文档 | 待实现 |

## 已知限制

- `excel` 特性已声明依赖但尚未实现读写逻辑。
- 权重仅支持频率权重，不支持概率权重。
- Mann–Whitney U 只提供渐近近似，不提供精确 p 值。
- 结果中的极端退化情形（如完全常数组的 F/t 统计量）可能产生非有限值（`NaN`/`Inf`），严格 JSON 序列化无法表示此类值。
