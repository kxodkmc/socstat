# AGENTS.md — Guardrails for AI agents working on socstat

This file exists to keep any future AI agent (or human contributor) aligned with the
project's vision. Read it before modifying code. If a change would violate a
**Hard Rule**, stop and reconsider the design.

---

## 1. Mission (do not lose sight of this)

> **socstat is a lightweight, embeddable statistical-analysis SDK.** It gives any
> platform and any application SPSS-equivalent core capabilities — and aims to
> exceed them.

Implications:

- The library is consumed by **other software**, not just by `main()`.
  Public APIs must be host-friendly: stable, serializable, and free of assumptions
  about the calling environment.
- "Lightweight" is a feature. Feature-gate everything optional. Do not pull in heavy
  dependencies for the default build.
- Rich, ergonomic, discoverable APIs are the product. Prefer clean public surface
  over internal cleverness.

## 2. Hard Rules

1. **Results must be serializable.** Every public result type (`Descriptive`,
   `FrequencyTable`, `Crosstab`, and anything new) MUST derive `Serialize` /
   `Deserialize` from `serde`. Hosts consume results as JSON/FFI payloads.
   If a type cannot be `Deserialize`-able (e.g. it's cheap to recompute), still
   derive `Serialize` and document why.
2. **No per-cell enum storage.** Data is stored **column-major, typed, contiguous**
   (`ColumnData::Numeric(Vec<Option<f64>>)` / `Text(Vec<Option<String>>)`).
   `Value` is a *transient* row-level type only. Never store a `Vec<Value>` as a
   column. Missing = `None`, never a sentinel like `NaN` or `-999`.
3. **Every declared feature must actually work.** `Cargo.toml` features (`csv`,
   `sav`, `excel`, `datetime`) must never advertise a capability that is not
   implemented. If `src/io/sav.rs` does not exist, do NOT enable `sav` in default
   features, and do NOT leave `#[cfg(feature = "sav")]` blocks referencing missing
   files. Either implement it or remove the claim.
4. **Statistics must be statistically correct.** Percentages sum correctly, `n-1`
   for sample variance, user-missing values excluded, weights honored. Wrong
   statistics are a bug, not a shortcut.
5. **No silent data corruption.** Type mismatches return `SocStatError`, never
   silently coerce. Round-trips (write → read) must preserve data.
6. **SPSS semantics are the baseline, not the ceiling.** Mirror SPSS concepts
   (variable labels, value labels, missing specs, measurement levels, case
   weights) but never feel limited by SPSS. Better defaults are welcome.

## 3. Architecture (preserve these boundaries)

```
src/
  data/     Data model: Value, Variable, ColumnData, Dataset, RowView
  dist/     Probability distributions (thin wrapper over statrs)
  stats/    Analysis: descriptive, frequencies, crosstab (+ future tests)
  io/       Readers/Writers per format, gated by Cargo features
  error.rs  Unified SocStatError + SocStatResult
  lib.rs    Crate docs + prelude
```

- `stats/` functions take **`&ColumnData` slices or data**, never mutate the dataset.
  The `StatsExt` trait (in `stats/mod.rs`) bridges `Dataset` → analysis.
- `dist/` wraps `statrs` behind the `Distribution` trait so no other module
  depends on `statrs` directly.
- `io/` uses the `Reader`/`Writer` traits and builder entry points
  (`read()` / `write()` with `.csv()`, `.json()`, `.auto()`).
- New analysis goes under `stats/`; new formats go under `io/` with a matching
  Cargo feature. Everything public is re-exported through `prelude`.

## 4. API conventions

- **Ergonomics first**: builder methods on `Variable` (`Variable::numeric(name).label(..)`)
  and closure-based transforms (`compute`, `filter`, `recode`) are the established
  style — keep it.
- **Errors**: use `SocStatResult<T>` everywhere; add variants to `SocStatError`
  instead of `String`-ly typed errors where a distinct case exists.
- **Host-friendliness**: expose `&[Option<f64>]` / `&[Option<String>]` slices and
  `Vec<f64>` extraction helpers. Avoid APIs that force hosts to hold borrows of
  internals (e.g. prefer owned results). Do not make useful operations `pub(crate)`
  just because the current crate doesn't call them — the SDK surface is the product.
- **Docs**: every public item gets a `//!` / `///` doc comment with a runnable
  `no_run` example where reasonable. Crate-level `lib.rs` docs show a Quick Start.
- **No comments unless they explain "why"** or are the required module/API docs.

## 5. Dependencies & features

- Default build stays light: `nalgebra`, `statrs`, `serde`, `thiserror`, plus
  `csv`/`serde_json` via the `csv` feature.
- Optional heavy deps live behind features: `excel` → `calamine` + `rust_xlsxwriter`,
  `datetime` → `chrono`. The `sav` feature flag was removed (Hard Rule 3: no fake
  claims) until P6 actually implements `src/io/sav.rs`.
- Before adding a dependency, ask: can this be done with the stdlib or an
  existing dep? Is it worth the compile time for default users?
- Keep `full` = all features. Run a `--features full` build to verify feature
  coherence after any `Cargo.toml` change.

## 6. Verification (always run before finishing)

```bash
cargo build                 # default features compile clean
cargo test                  # all unit + doctests pass
cargo test --features full  # feature-gated code compiles & passes
cargo clippy -- -D warnings # no new warnings (if clippy is available)
```

Never mark a task done until `cargo build` and `cargo test` pass.

## 7. Implementation roadmap (implement phases in order)

This is the **explicit plan**. Work on the next incomplete phase; do not skip ahead.
Each phase is not "done" until its deliverable compiles under `cargo build` and
passes `cargo test` (and `--features full` where gated).

| Phase | Scope | Deliverable | Status |
|-------|-------|-------------|--------|
| P1 | Project skeleton + `data` module + `error` + CSV I/O | Read/write CSV, build `Dataset` | ✅ done |
| P2 | `dist` module + `stats/descriptive` + `frequencies` + `crosstab` | Full descriptive statistics | ✅ done |
| P7 | `data/transform`: compute / recode / filter / sort | Data transforms | ✅ done |
| P3 | `stats/tests`: t-test + ANOVA + chi-square + nonparametric | Full hypothesis-testing | ✅ done |
| P4 | `stats/regression`: linear (OLS/QR) + logistic (IRLS) | Full regression | ⬜ |
| P5 | `stats/multivariate`: PCA (SVD) + reliability (Cronbach α) | Multivariate analysis | ⬜ |
| P6 | `io/sav`: SPSS `.sav` binary read/write | Format compatibility | ⬜ |
| P8 | Integration tests + examples + docs | Production readiness | ⬜ |

### Phase invariants (apply to every phase)

- **P3+ statistics** must be: statistically correct, weighted-aware, and return
  `Serialize`-able result structs (Hard Rule 1).
- **P4 regression** may use `nalgebra` (already a core dep) for OLS/QR/IRLS
  numerics. Add no new dependency without a reason (Section 5).
- **P6 sav** goes under `io/` behind the existing `sav` feature, replacing the
  placeholder `#[cfg(feature = "sav")]` blocks in `src/io/mod.rs`. Binary
  round-trip tests must preserve labels, missing specs, and measure levels.
- **P8** means examples stay runnable (`cargo run --example …`), the crate-level
  Quick Start in `lib.rs` stays accurate, and `--features full` compiles clean.

### Direction of travel (spirit behind the phases)

1. **Serializable results** — all result types carry `serde` derives (highest priority).
2. **Statistical tests** — t-tests, chi-square, ANOVA, correlation, regression.
3. **Real I/O** — implement `sav` (SPSS) and `excel` behind their features so every
   declared feature is genuine.
4. **Binding readiness** — keep the API surface FFI/WASM-friendly; never leak Rust
   internals across the boundary.
5. **Surpass SPSS** — modern ergonomics, streaming, better error messages, and
   correctness are all acceptable places to beat SPSS, not imitate it.

## 8. Anti-goals (explicitly do NOT do these)

- Do NOT build a GUI, a CLI analysis app, or R-style interactive console as part of
  this crate. This is an SDK; consumers build their own frontends.
- Do NOT add a plugin/extension system that bloats the core.
- Do NOT trade correctness for speed in statistics (numerical stability and
  accuracy first).
- Do NOT change the storage model to row-major or per-cell enums.
- Do NOT put analysis logic into `data/` or data-model logic into `stats/`.
- Do NOT add `unsafe` without a written justification and a safety comment.
