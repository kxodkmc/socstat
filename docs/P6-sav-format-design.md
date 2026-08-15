# P6 设计文档 — SPSS `.sav` 二进制读写

> 状态:设计稿(v1)
> 阶段:P6(SPSS `.sav` format compatibility)
> 依据:GNU PSPP《System File Format》(`https://www.gnu.org/software/pspp/pspp-dev/html_node/System-File-Format.html`,IBM 未公开 `.sav` 规范,该文档是事实标准)

---

## 1. 目标与范围

### 1.1 目标

为 `socstat` 增加 `io/sav` 模块,支持读取和写出 SPSS `.sav`(System File)格式的二进制文件,
并在 `sav` Cargo feature 下提供 `read().sav(...)` / `write().sav(...)` 入口。

### 1.2 范围

**第一版(本设计交付)支持:**

- 读取:压缩方式 0(无压缩)、1(字节码)、2(zlib / `$FL3`)全部支持。
- 写出:压缩方式 1(字节码,`$FL2`),与 haven 默认行为一致,SAS/PSPP/R 均可读。
- 编码:写出恒为 UTF-8(SPSS ≥ 16 默认);读取时对非 UTF-8 按 latin-1 逐字节映射。
  注:字节级 round-trip 仅对 UTF-8 文件成立;非 UTF-8 读入再写出,字节必变(模型只存 UTF-8 字符串)。
- 元数据:变量名(含 >8 字符长名)、变量标签、值标签、用户缺失(离散/区间)、度量水平
  (Nominal/Ordinal/Scale)、权重变量、字符串宽度。
- 字符串宽度 ≤ 255(SPSS 短字符串;写入时拆成 8 字节一段的多条变量记录)。

**明确不实现(报清晰错误或跳过):**

- EBCDIC 编码文件(rec_type 非 ASCII 变体)。
- IBM / VAX 浮点格式(通过 `bias != 100.0` 检测后拒绝)。
- Very Long String(宽度 > 255,需要 subtype 14 记录)——第一版返回 `UnsupportedFormat`。
- `.zsav`(`$FL3` 写方向)和 `.por`(SPSS Portable)。
- 未知扩展记录:`size * count` 跳过,不报错(规范明确建议容忍)。

### 1.3 不违反的硬规则

- Hard Rule 1:读入的 `Dataset`、`Variable` 元数据可被 serde 序列化。
  注意:当前 `data/` 模型(`Variable`/`ColumnData`/`Dataset`/`Value`)尚无 `Serialize`/`Deserialize`
  derive(仅 `stats/*` 结果类型有)——P6 交付时必须补齐,不能假设已存在。
- Hard Rule 2:数据一律按列主序存入 `ColumnData::Numeric(Vec<Option<f64>>)` /
  `Text(Vec<Option<String>>)`,绝不落地为 `Vec<Value>`;缺失 = `None`,绝不使用哨兵值。
- Hard Rule 3:`sav` feature 只有在 `src/io/sav.rs`(或 `src/io/sav/`)真正实现后才重新启用;
  实现完成前保持 `Cargo.toml` 与 `io/mod.rs` 中的 placeholder 为空,不得虚标能力。
- Hard Rule 4:数值全程以 IEEE-754 `f64` 原值保精度;字节码压缩只对整数走 bias 编码,
  其余走字面量(见 §7.3),杜绝精度损失。
- Hard Rule 6:SPSS 语义为基线——长名、度量水平、权重、用户缺失全部镜像。

---

## 2. `.sav` 文件结构(现状研究结论)

文件 = **文件头(176 字节)** + **字典记录** + **字典终止记录(999)** + **数据记录**。

### 2.1 文件头(File Header Record,176 字节)

```
char   rec_type[4];          // "$FL2"(压缩0/1) 或 "$FL3"(压缩2/zlib)
char   prod_name[60];        // 恒以 "@(#) SPSS DATA FILE" 开头
int32  layout_code;          // 通常=2,用于端序探测
int32  nominal_case_size;    // 每 case 数据元素数,不可靠,读取时不依赖
int32  compression;          // 0=无压缩 1=字节码 2=zlib
int32  weight_index;         // 权重变量字典索引(0 表示无),"字典索引+1"待真实文件校准
int32  ncases;               // 已知则 case 数,否则 -1
flt64  bias;                 // 压缩偏差,通常 100.0;兼作浮点格式探测
char   creation_date[9];     // "dd mmm yy",未知填 "01 Jan 70"
char   creation_time[8];     // "hh:mm:ss",未知填 "00:00:00"
char   file_label[64];       // 文件标签,右补空格
char   padding[3];           // 0
```

### 2.2 字典记录

| rec_type | 内容 | 说明 |
|---|---|---|
| 2 | 变量描述记录 | 每个变量一条(长字符串按 8 字节拆多条,续条 type=-1) |
| 3 + 4 | 值标签对 | rec 3 存 `(value[8], label_len, label[])` 列表,`label_len+label` 对齐到 **8 字节**的倍数;rec 4 列出适用的 1-based 变量字典索引。宽度 ≥ 8 的字符串变量不能出现在 rec 4(见 §7.5) |
| 6 | 文档记录 | 80 字符行,可选,忽略 |
| 7 | 扩展记录 | `int32 subtype, size, count, data[size*count]`,未知 subtype 直接跳过 |
| 999 | 字典终止 | 字典与数据的分界 |

常用扩展记录 subtype:

| subtype | 用途 | 本项目使用 |
|---|---|---|
| 11 | Variable Display Parameter:`measure(0/1/2/3), width, alignment` | 读度量水平 + 写度量水平/宽度 |
| 13 | Long Variable Names:`短名=长名`,09 分隔 | 读/写 >8 字符变量名 |
| 20 | Character Encoding | 读编码声明 |
| 14 | Very Long String | 不实现,拒绝 |
| 4 | Machine Integer Info | 忽略(编码回退线索) |

### 2.3 变量描述记录(rec_type=2)布局

```
int32  rec_type;            // 2
int32  type;                // 0=numeric;>0=字符串宽度;续条(长字符串后续段)=-1
int32  has_var_label;       // 0/1
int32  n_missing_values;    // 1/2/3=离散个数;-2=区间;-3=区间+离散值
int32  print;               // 显示格式(低字节=小数位,次低=宽度,再高=格式类型)
int32  write;               // 写格式,同上
char   name[8];             // SPSS 短名,右补空格
// 仅当 has_var_label=1:
int32  label_len;
char   label[];             // 对齐到 4 字节
// 仅当 n_missing_values != 0:
flt64  missing_values[];    // |n_missing_values| 个元素;区间上下界为 LOWEST/HIGHEST(见下)
```

> LOWEST/HIGHEST 表示无界区间。HIGHEST 恒为 `+DBL_MAX`;LOWEST 在 SPSS 21+ 为 `-DBL_MAX`,
> 在更早版本为次大负数 `0xffeffffffffffffe`(IEEE-754)。**读端必须把两种编码都识别为"无界下界"**,
> 否则旧文件会被解析成字面量下界,缺失判断错乱。

### 2.4 数据记录

**压缩 0(无压缩):** 每 case 依次为若干 8 字节元素;数值为 `flt64`,字符串右补空格。

**压缩 1(字节码):** 数据按 8 字节一块组织,每块内 8 个 1 字节命令码:

| 命令码 | 含义 |
|---|---|
| 0 | 忽略/填充 |
| 1–251 | 数值 = 码 − bias(如 bias=100 时,码 105 → 值 5) |
| 252 | EOF |
| 253 | 字面量,值在紧随当前命令块后的 8 字节区(按出现顺序) |
| 254 | 8 字节全空格字符串 |
| 255 | 系统缺失(SYSMIS = -DBL_MAX) |

字符串内的空字节(值 0)以"码 0 译作 null"处理。

**压缩 2(zlib):** `$FL3`。24 字节 zlib 头(3×int64:zheader_ofs/ztrailer_ofs/ztrailer_len),
随后若干 RFC1950 zlib 数据块,解压后为**压缩 1 的字节码流**;24 字节固定 trailer
(bias=100 的负整数、0、block_size、n_blocks)+ 每块 24 字节描述符。

> 实现捷径:压缩 2 = 读 trailer → 按描述符逐块 `flate2::Decompress` → 得到字节码流 → 复用压缩 1 解码器。

---

## 3. 与现有数据模型的映射

| `socstat` 模型 | `.sav` 载体 | 方向 | 备注 |
|---|---|---|---|
| `Variable.name`(可 >8 字符) | `name[8]` + ext 13 | 双向 | 长名在读取时覆盖短名;写出时自动生成唯一短名 |
| `Variable.label: Option<String>` | `has_var_label` + `label_len/label[]` | 双向 | label 补 4 字节对齐 |
| `DataType` / `Variable.width` | `type` 字段 | 双向 | numeric=0;文本=宽度;宽度>8 拆多段(续条 type=-1) |
| `MissingSpec::None` | `n_missing_values=0` | 双向 | |
| `MissingSpec::Discrete(vals)` | `n_missing_values = len(1..=3)` + `flt64[]` | 双向 | >3 个离散值:报错(SPSS 上限) |
| `MissingSpec::Range{low,high,discrete}` | `n_missing_values=-2`(或 -3)+ 2/3 个 `flt64` | 双向 | LOWEST/HIGHEST 用 ±DBL_MAX |
| `Variable.value_labels: BTreeMap<String,String>` | rec 3+4 | 双向 | 键:数值变量解析为 `f64`;文本变量直接写 bytes |
| `MeasureType::Nominal/Ordinal/Scale` | ext 11 `measure = 1/2/3` | 双向 | 未知(0)读入为 Nominal?见 §6.2 |
| `Variable.is_weight` | header `weight_index` | 双向 | 见 §8 待校准项 |
| 缺失值 | 数值 SYSMIS=-DBL_MAX;字节码 255 | 双向 | 模型内一律 `None` |
| `ValueFormat` | `print`/`write` 编码 | 读:保留通用;写:映射 F/E/PCT 等 | 低优先级(§9) |

> **字符串缺失值**:`.sav` 支持字符串变量的缺失值(存 flt64 槽内,右补空格到 8 字节),
> 但模型 `MissingSpec` 仅能表达 `f64`。第一版策略:读入时忽略字符串变量的缺失规格(槽内按
> 8 字节跳过);写出时若 `Text` 变量带 `MissingSpec`,返回 `Sav` 错误(数值缺失字节写进字符串变量无意义)。

### 3.1 值标签键转换(已知模型层限制)

模型层 `value_labels` 以**显示字符串**为键。`sav` 层约定:

- 数值变量:写时 `key.parse::<f64>()`(失败报 `Sav` 错误);读时 `f64` 格式化为字符串键。
- 文本变量:键为原始字符串(宽度 <8 时右补空格比对)。
- 文档说明:round-trip 中 `"1"` 与 `"1.0"` 这类键表示可能不逐字一致,这是模型层的固有表示限制,不以"数据损坏"处理。

### 3.2 长字符串(`Variable.width > 8`)

- 写出:宽度 `w` 拆成 `ceil(w/8)` 条变量记录,第 1 条含真实元数据,后续为 dummy(`type=-1`)。
- 读入:遇到 `type=-1` 的续条,把数据字节拼回前一条;恢复真实 `width` 与完整字符串。
- 值标签与缺失(长字符串)需要 ext 14/20(第一版拒绝 >255;8 < w ≤ 255 用标准记录)。

---

## 4. 模块结构

```
src/io/sav/
  mod.rs       SavReader + SavWriter(实现 io::Reader / io::Writer trait)
  header.rs    文件头解析/写出 + 端序 & 浮点格式探测
  records.rs   字典记录:变量记录、值标签记录、扩展记录、999 终止
  data.rs      数据记录:字节码解码/编码(压缩 0/1/2 共用)、zlib 读路径
src/io/mod.rs  #[cfg(feature = "sav")] pub mod sav; + 构建器接线
```

`io::Reader` / `io::Writer` trait 已在 `src/io/mod.rs` 定义,直接实现:

```rust
pub struct SavReader;
pub struct SavWriter;
impl Reader for SavReader  { fn read_path(&self, path: &Path) -> SocStatResult<Dataset>; }
impl Writer for SavWriter  { fn write_path(&self, ds: &Dataset, path: &Path) -> SocStatResult<()>; }
```

构建器接线(`ReadBuilder` / `WriteBuilder` 各加):

```rust
#[cfg(feature = "sav")]
pub fn sav(&self, path: impl AsRef<Path>) -> SocStatResult<Dataset> { ... }
```

`read_by_ext` / `write_by_ext` 增加 `"sav" => ...` 分支。

---

## 5. 依赖

| 依赖 | 理由 | 归属 |
|---|---|---|
| `flate2`(纯 Rust,miniz_oxide) | 读 `$FL3` / 压缩 2 需要 RFC1950 解压 | `sav` feature 内,默认构建不受影响 |

- **不引入** `byteorder`:端序用 std 的 `from_le_bytes` / `from_be_bytes` 处理。
- **不引入** 现有第三方 `.sav` 库(`spss_sav`、`ambers` 等):本模块是 SDK 的一等公民,
  需要与 `Dataset`/`Variable` 模型深度契合;外部库(尤其 Arrow 系)与"轻量"目标相悖。
  可参考其实现做正确性交叉验证。

`Cargo.toml` 变更:

```toml
[features]
default = ["csv"]
csv = ["dep:csv", "dep:serde_json"]
sav = ["dep:flate2"]            # 新增
full = ["csv", "sav"]            # sav 加入 full
# 注:excel/datetime 特性已于后续移除(Hard Rule 3:不声明未实现的能力),
# 待 src/io/excel.rs 与日期时间模块真实落地后再恢复。

[dependencies]
flate2 = { version = "1.0", optional = true }  # 新增(可选)
```

---

## 6. Reader 设计

### 6.1 读取流程

```
read_path
 ├─ 打开文件,读 176 字节 header
 ├─ 探测:rec_type → 编码族;layout_code → 整数端序;bias → 浮点格式
 │   (bias==100.0 的 IEEE-754 假设,否则报 UnsupportedFormat)
 ├─ 循环读字典:
 │   ├─ rec 2 → 变量记录(构建 Vec<RawVariable>,含续条标记)
 │   ├─ rec 3+4 → 值标签对(先暂存,等所有变量就绪后回填)
 │   ├─ rec 7 → 按 subtype 分派:11/13/20 消费,其余跳过
 │   ├─ rec 999 → 字典结束
 │   └─ 其他 → 跳过(规范:容忍变化)
  ├─ 数据:
  │   ├─ compression 0 → 直接读 8 字节元素
  │   ├─ compression 1 → 字节码解码
  │   └─ compression 2 → zlib 解压为字节码流 → 字节码解码
  │   (case 数:header `ncases >= 0` 时按该值;`ncases = -1` 时读至 EOF,压缩 1/2 遇码 252 即止)
  ├─ 装配:
  │   ├─ 长名覆盖短名(ext 13)
  │   ├─ 度量水平回填(ext 11,按"真实变量"而非续条计数;需按 `size*count` 判定
  │   │    每变量是 2 个字段(无 width)还是 3 个字段,见 §6.2)
  │   ├─ 值标签按字典索引回填(跳过指向续条/长字符串的项)
  │   ├─ 权重变量标记(weight_index:1-based 字典索引含续条,需换算成合并后的模型下标)
  │   └─ 按列主序构造 ColumnData → Dataset
  └─ 返回 Dataset
```

### 6.1a rec 7 分派细节

- **subtype 11**:count 可为"变量数 × 2"或"× 3";`width` 字段仅在 × 3 时存在。
  读取必须按 `size * count`(每变量 12 或 16 字节)分支,不能假设固定 3×。
- **subtype 13**:`短名=长名` 对,09 分隔,无尾部分隔符。
- **subtype 20**:`size=1`,`encoding[]` 为编码名,大小写不敏感匹配。

### 6.2 度量水平边界情况

`measure = 0`(Unknown)在模型里没有对应值。决定:**读入时保持变量默认度量**,
并在 `read_path` 的文档注释中说明"未知度量水平按变量类型默认(Numeric→Scale,Text→Nominal)"。
不新增枚举变体(避免扩散到所有分析代码)。

### 6.3 数据解码核心(压缩 1/2 共用)

```
逐 8 字节命令块:
  for code in block:
    0         → 若用于字符串位置,写入空字节;否则跳过(填充)
    1..=251   → 数值 = (code as f64) - bias
    253       → 取字面量区下一个 8 字节(flt64 或字符串 bytes)
    254       → 8 空格字符串
    255       → 系统缺失 → None
  块末尾,按块内 253 出现次数消费对应字面量
```

字符串列语义:字节按变量宽度切分,`\0` 与全空格截断到实际长度(SPSS 惯例为尾随空格截断),
空/全空格 → `None`;其余 trim 尾随空格后转 `String`。

### 6.4 编码处理

- ext 20 声明 `UTF-8`(大小写不敏感,含 `UTF8` 等 IANA 别名)→ 直接 `String::from_utf8`。
- 无声明或声明为其他 8-bit → 按 latin-1 逐字节映射(尽力解码;字节级 round-trip 仅 UTF-8 路径成立)。
- rec_type 为 EBCDIC 变体(`5b c6 d3 f2`)→ `UnsupportedFormat`。

---

## 7. Writer 设计

### 7.1 写出流程

```
write_path(ds, path)
 ├─ 预处理:检查类型映射可行性(离散缺失 ≤3;宽度 ≤255;值标签键可转 f64)
 ├─ 生成短名:≤8 字符且符合 SPSS 命名规则(name 规则见 §7.4)直接用;
 │   否则生成唯一短名 + 收集 (短名, 长名) 对
 ├─ 写 header($FL2, compression=1, bias=100.0, weight_index, ncases,
 │   creation_date/time, file_label, padding)
 ├─ 写变量记录(长字符串拆段,续条 type=-1)
 ├─ 写值标签对(rec 3 + 4;按字典索引,跳过续条)
 ├─ 写扩展记录(升序):
 │   11 → measure,width,alignment(每真实变量 3 个 int32)
 │   13 → 短名=长名 对(若有)
 │   20 → 声明 UTF-8
 ├─ 写 999 字典终止
 └─ 写数据(字节码编码,见 §7.3)
```

### 7.2 header 细节

- `rec_type[4] = "$FL2"`,`compression = 1`。
- `nominal_case_size` 计算:每个变量占用的数据元素数 = 字符串 `ceil(width/8)` 或 1。
- `ncases` 已知(内存 Dataset)→ 直接写;避免 seek-back。
- `weight_index`:写出权重变量的字典索引(含续条计数),待 §8 校准。
- `creation_date/time`:`01 Jan 70 00:00:00`(不引入 chrono 到默认路径;
  `datetime` feature 已移除,`sav` 时间戳不是优先级,第一版写占位)。

### 7.3 字节码编码

对每个数据元素:

| 元素类型 | 编码 |
|---|---|
| `None`(系统缺失) | 255 |
| 数值 x,且 `x` 为整数且 `-99 <= x <= 151`(即 `code = x + 100` 落在 1..=251) | 1..=251 |
| 其他数值 | 253 + 追加到字面量区(flt64 原值) |
| 全空格字符串(或空) | 254 |
| 其他字符串(≤8 字节段) | 253 + 字面量 bytes(右补空格到 8);`\0` 用码 0 |

> 整数范围校验:`x + bias` 必须落在 1..=251,即 `x ∈ [1-100, 251-100] = [-99, 151]`。
> 非整数一律走字面量,**保证 `f64` 逐位精度**(Hard Rule 4)。

### 7.4 SPSS 命名规则

- 短名:首字符必须为大写字母或 `@`;后续可为大小写字母、数字、`#`、`$`、`_`、`.`。
- 读取时短名直接可用;写入时对超长名:取合法前缀或生成 `A@xxxxx` 式唯一短名,长名进 ext 13。
- 重复冲突:追加序号直到唯一。

### 7.5 值标签写出

- 数值变量:键 `parse::<f64>()`;值写入 8 字节 flt64。
- 文本变量:键字符串右补空格到 8 字节;**仅支持宽度 < 8 的变量,宽度 ≥ 8 时报 `Sav` 错误**
  (规范禁止其在标准值标签记录中出现)。
- `label_len` 用 u8(SPSS 上限 255),超出报 `Sav` 错误。
- rec 3 的 `label_len + label` 对齐到 **8 字节**的倍数(区别于变量记录 label 的 4 字节对齐)。
- 同一套值标签可被多变量共享——第一版**不聚合**,每变量独立写一组(简单、正确优先)。

---

## 8. 待实测校准的开放项

1. **`weight_index` 的实际写法**:规范明确为"0=无;否则 = 字典索引(含续条)+ 1"。写/读端按此实现,
   但需用真实文件(pyreadstat/PSPP 生成)实测确认 SPSS 确按"含续条记录"计索引。
2. **字节码块的字面量顺序**:块内多个 253 按出现顺序消费,需对照真实文件验证。
3. **`nominal_case_size`**:读取时不依赖;写入时按 §7.2 计算(SPSS 可容忍该值)。
4. **值标签变量记录(rec 4)布局**:PSPP 文档记载为 `var_count + int32 索引数组`;
   但部分流行实现按 `(index, label_len, label)` 逐项解析,存在历史变体。实现时按文档版写出,
   读取做防御性解析(先读索引,若剩余字节够则再跳过 label_len+label),并以真实文件确认。
5. **LOWEST 旧编码**:SPSS 21 前用 `0xffeffffffffffffe` 表示无界下界,需与 `-DBL_MAX` 一并
   识别为"无界下界"(§2.3),用 fixture 覆盖旧/新两种形态。
6. **subtype 11 字段数**:真实文件可能按"每变量 2 个(无 width)"或"3 个"写,读取按
   `size * count` 判定(§6.1a),用 fixture 验证两种形态。

---

## 9. 不做/推迟清单(边界)

- `.zsav` 写出、`.por`、EBCDIC、IBM/VAX 浮点。
- `ValueFormat` 全映射:第一版读入一律归为 `General`,写出数值统一 F 格式;解析 `print` 编码
  中 F/E/PCT/DATE 等为后续增强项。
- 文件级 `file_label`/文档记录(rec 6)读入暂不落模型(模型无对应字段),写出占位。

---

## 10. 错误处理

`SocStatError` 新增两个变体:

```rust
#[error("sav format error: {0}")]
Sav(String),                      // 字段非法、记录错乱、值标签键不可转 f64 等

#[error("unsupported format: {0}")]
UnsupportedFormat(String),        // EBCDIC、IBM/VAX 浮点、宽度>255、续条孤立等
```

规则:能跳过的(未知扩展记录、未知 rec_type)尽量跳过不报错;结构性错误(字段越界、
孤立的续条、无法解析的字典索引)必须报 `Sav`。

---

## 11. 测试策略

### 11.1 Round-trip 单测(核心)

构造含以下特征的 Dataset → write → read → 逐字段断言相等:

- 数值 + 文本列、`None` 缺失(验证 SYSMIS / 255 与 `None` 双向)。
- 用户缺失:离散 3 个、区间、区间+离散。
- 变量标签、值标签(数值与文本)。
- 度量水平 3 种。
- 权重变量(`is_weight`)、长名(>8 字符)。
- `f64` 精度:整数与大量小数位数值,断言逐位相等(`to_bits` 比较)。
- 字符串宽度 8 / 16(跨多段)的文本列。

### 11.2 独立 fixture

- 手工构造二进制字节(压缩 0/1 各一份)做读入测试,验证端序探测、bias 压缩、字面量顺序。
- `$FL3`(压缩 2)fixture 验证 zlib 路径(可用 flate2 自造,或 PSPP 生成后提交)。

### 11.3 交叉验证(P8 集成阶段)

- 提交由 GNU PSPP 或 pyreadstat 生成的真实 `.sav` 样本,读取后与已知值比对。
- 反向:socstat 写出的文件由 PSPP/`pyreadstat`(CI 需要 Python)读回验证兼容性;
  本地至少用 PSPP 命令行做一次人工验证。

### 11.4 质量门槛

```bash
cargo build                 # 默认 features 编译干净
cargo test --features full  # 单元 + doctest + sav 门控测试全过
cargo clippy -- -D warnings # 无新告警
```

---

## 12. 实施步骤与里程碑

| # | 里程碑 | 内容 | 验收 |
|---|---|---|---|
| M1 | feature 接线 | `Cargo.toml` 加 `sav` feature(+`flate2`,入 `full`);`io/mod.rs` 挂模块与构建器 | `cargo build --features full` 过 |
| M2 | Reader 骨架 | header 解析 + 端序/浮点探测 + 变量记录读取 | 读入"压缩 0"手工 fixture |
| M3 | Reader 完整 | 值标签、ext 11/13/20、压缩 1/2 解码、列装配 | 读入压缩 1/2 fixture,标签/长名/缺失正确 |
| M4 | Writer | header/字典/字节码写出 + round-trip 单测全绿 | §11.1 全部用例通过 |
| M5 | 打磨 | 边界错误、文档注释、`--features full` + clippy 全绿 | §11.4 门槛通过 |
| M6 | 交叉验证(可并入 P8) | 真实文件样本 + PSPP 人工验证 | 外部软件可互读 |

预估总量:约 1200–1600 行(不含测试)+ 400–600 行测试。

---

## 13. 验收标准(完成定义)

1. `cargo build`、`cargo test --features full`、`cargo clippy -- -D warnings` 全部通过。
2. `read().sav()` / `write().sav()` 可经 `prelude` 或 `io` 直接使用,文档含 `no_run` 示例。
3. Round-trip 保真:标签、缺失规格、度量水平、权重、长名、`f64` 逐位精度。
4. 压缩 0/1/2 均可读;写出为压缩 1,PSPP 或 pyreadstat 可读回。
5. `AGENTS.md` 路线图中 P6 状态更新为 ✅,README/lib.rs feature 表加入 `sav`。
6. Hard Rules 全部满足(重点复查 Hard Rule 2 的列主序存储与 Hard Rule 3 的 feature 真实性)。
