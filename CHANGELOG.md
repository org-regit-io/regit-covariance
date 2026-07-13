<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.1] — 2026-07-13

### Changed

- Refreshed dependencies: `Cargo.lock` updated to the latest semver-compatible
  patch and minor versions.
- Bumped the workspace version to 1.0.1.
- No public API changes.

## [1.0.0] — 2026-05-07

First stable release. Public API frozen under semver guarantees.

The repository was reorganised into a Cargo workspace. The pure mathematical
core (`regit-covariance`) is now publishable to crates.io and compiles
cleanly to `wasm32-unknown-unknown`. The Yahoo Finance provider was extracted
to a sibling crate, and the demo server moved to a third crate marked
`publish = false`.

### Changed — repository structure

The single-crate layout (`src/`, `tests/`, `examples/`, `benches/`, `static/`)
was reorganised into a workspace with three crates under `crates/`:

```
crates/
  regit-covariance/          # PURE MATH. Publishable. WASM-compatible.
  regit-covariance-yahoo/    # Yahoo Finance provider. Native-only.
  regit-covariance-server/   # axum HTTP + SSE demo. publish = false.
```

Migration path for downstream code: imports formerly under
`regit_covariance::data::provider::yahoo` move to the standalone crate
`regit_covariance_yahoo`; the math API path (`regit_covariance::math::*`)
is unchanged.

### Added — `regit-covariance` (core)

#### Sample covariance (`src/math/sample_covariance.rs`)
- `CovarianceError` enum — `InsufficientObservations`, `EmptyMatrix`,
  `ZeroVariance { column }`, `InvalidHalfLife`; implements `Display`, `Debug`,
  `std::error::Error` via `thiserror`
- `CovarianceResult` struct — holds `correlation: DMatrix<f64>`, per-asset
  `std_devs: Vec<f64>`, `observations`, `assets`, `q = T/N`
- `correlation_matrix(returns: &DMatrix<f64>) -> Result<CovarianceResult, CovarianceError>` —
  equal-weight sample correlation `C = (1/T) X^T X` after column standardisation
  (population variance, zero mean, unit variance)
- `ewm_correlation_matrix(returns, half_life) -> Result<CovarianceResult, CovarianceError>` —
  exponentially weighted correlation; weights `w_i = decay^(T-1-i)`,
  `decay = exp(-ln(2)/half_life)`; converges to equal-weight as
  `half_life -> infinity`
- `covariance_from_correlation(correlation, std_devs) -> DMatrix<f64>` —
  reconstructs covariance from correlation and per-asset standard deviations

#### Eigendecomposition (`src/math/eigen.rs`)
- `EigenError` enum — `NotSquare { rows, cols }`, `EmptyMatrix`
- `EigenDecomposition` struct — holds `eigenvalues: DVector<f64>` (descending),
  `eigenvectors: DMatrix<f64>` (columns matched), dimension `n`
- `eigendecompose(matrix) -> Result<EigenDecomposition, EigenError>` —
  wraps `nalgebra::SymmetricEigen` with descending sort and matched eigenvectors
- `EigenDecomposition::reconstruct()`, `trace()`, `count_above(threshold)`

#### Marchenko-Pastur (`src/math/marchenko_pastur.rs`)
- `MpError` enum — `InvalidQ`, `EmptyEigenvalues`, `ConvergenceFailure { max_iter, residual }`
- `MpFit` struct — `sigma_sq`, `lambda_plus`, `lambda_minus`, `signal_count`,
  `noise_count`, `q`
- `mp_bounds(sigma_sq, q) -> (f64, f64)` — `(lambda_-, lambda_+)`
- `mp_density(lambda, sigma_sq, q) -> f64` — Marchenko-Pastur PDF, zero outside support
- `fit_sigma_sq(eigenvalues, q) -> Result<MpFit, MpError>` — fixed-point
  iteration on noise mean (Lopez de Prado 2018, Code Snippet 2.4); 1000 iter
  cap, `1e-10` convergence tolerance

#### Eigenvalue denoising (`src/math/denoise.rs`)
- `DenoiseMethod` enum — `Constant` (replace noise with mean), `Target`
  (replace noise with `1.0`, then trace-rescale)
- `DenoiseResult` struct — cleaned matrix, eigenvalues, trace, method
- `denoise(eigen, mp_fit, method) -> DenoiseResult` — partition spectrum at
  `lambda_+`, replace, reconstruct; both methods preserve trace exactly
- `renormalize_to_correlation(matrix) -> DMatrix<f64>` — restores unit
  diagonal via `D^-1 * matrix * D^-1`

#### Detoning (`src/math/detone.rs`)
- `DetoneResult` struct — detoned matrix, eigenvalues, removed eigenvalues, k
- `detone(eigen, k) -> DetoneResult` — zeros top `k` eigenvalues, reconstructs
- `readd_tones(denoised, detone_result, eigenvectors) -> DMatrix<f64>` —
  reconstructs full matrix combining denoised idiosyncratic spectrum with
  saved market-mode eigenvalues

#### Ledoit-Wolf shrinkage (`src/math/ledoit_wolf.rs`)
- `ShrinkageError` enum — `InsufficientObservations`, `EmptyReturns`
- `LinearShrinkageResult` struct — shrunk matrix, optimal intensity
- `linear_shrinkage(returns, sample_corr) -> Result<...>` — Ledoit-Wolf 2004
  with constant-correlation target; analytical optimal intensity
- `NonlinearShrinkageResult` struct — shrunk matrix, shrunk eigenvalues
- `nonlinear_shrinkage(eigen, num_obs) -> NonlinearShrinkageResult` —
  Ledoit-Wolf 2020 analytical formula via Hilbert-transform sum;
  trace-preserving rescale; no tuning parameters

#### Condition number (`src/math/condition.rs`)
- `MatrixHealth` enum — `Healthy` (`< 100`), `Acceptable` (`< 1000`), `Unstable`
- `ConditionReport` struct — `condition_number`, `health`, `lambda_max`, `lambda_min`
- `ConditionImprovement` struct — before, after, improvement factor
- `classify(kappa) -> MatrixHealth`, `condition_report(eigen) -> ConditionReport`,
  `compare(before, after) -> ConditionImprovement`

#### Value at Risk (`src/math/var.rs`)
- `VarError` enum — `DimensionMismatch`, `InvalidConfidence`, `ZeroVolatility`
- `VarResult` struct — `var`, `portfolio_volatility`, `portfolio_return`, `confidence`
- `parametric_var(weights, expected_returns, covariance, confidence) -> Result<VarResult, VarError>` —
  Gaussian VaR
- `cornish_fisher_var(weights, expected_returns, covariance, confidence, skewness, excess_kurtosis) -> Result<VarResult, VarError>` —
  Cornish-Fisher adjusted VaR
- Internal `normal_quantile(p)` — Abramowitz & Stegun 26.2.23 rational approximation

#### PRIIPs SRI (`src/math/sri.rs`)
- `SriError` enum — `InvalidVev`
- `SriResult` struct — `mrm: u8` (1-7), `vev: f64`
- `DivergenceFlag` enum — `Green`, `Yellow`, `Red`
- `DivergenceReport` struct — prescribed/kernel SRI, difference, both VEVs, flag
- `var_equivalent_volatility(var, z_alpha) -> f64`,
  `classify_mrm(vev) -> Result<SriResult, SriError>`,
  `divergence_report(prescribed_vev, kernel_vev) -> Result<DivergenceReport, SriError>`

#### Log returns (`src/data/returns.rs`)
- `ReturnsError` enum — `InvalidPrice`, `InsufficientPrices`, `EmptyPrices`
- `log_returns(prices) -> Result<DMatrix<f64>, ReturnsError>` —
  `r_{t,i} = ln(P_{t,i} / P_{t-1,i})`; output shape `(T-1) x N`

#### Tests
- **87 unit tests** in `src/math/*.rs` and `src/data/returns.rs` — analytical
  correctness, trace preservation, PSD, symmetry, eigenvalue ordering,
  reconstruction accuracy, MRM thresholds, edge cases
- **33 integration tests** across 5 suites in `tests/`:
  - `trace_preservation` — proptest: denoised trace = N, symmetry
  - `psd_check` — positive semi-definiteness across q ratios and seeds
  - `denoise_identity` — pure noise -> near-identity
  - `mp_analytical` — MP bounds match closed-form formula
  - `pipeline_full` — end-to-end synthetic pipeline

#### Benchmarks (`benches/eigendecompose.rs`)
- Criterion benchmarks for the eigendecomposition wrapper across matrix
  sizes (10, 50, 100, 500), measuring sort overhead vs raw nalgebra

#### Examples (`examples/quickstart.rs`)
- Library-usage demo: returns -> correlation -> eigen -> MP fit -> denoise ->
  condition comparison -> VaR

#### Crate metadata
- `description`, `keywords`, `categories`, `readme = "../../README.md"`
- Inherits version, edition, rust-version, authors, license, repository,
  homepage from `[workspace.package]`
- `Apache-2.0` licence; SPDX identifiers on every `.rs` file
- `#![forbid(unsafe_code)]` at the crate root
- WASM target support: builds for `wasm32-unknown-unknown` (release rlib ~280 KB)

### Added — `regit-covariance-yahoo`

#### Yahoo Finance provider (`src/lib.rs`)
- `YahooError` enum — covers HTTP failures, JSON decoding errors, missing
  fields, ISIN-resolution failures
- `TickerPrices` struct — ticker, fund name, currency, timestamps, close prices
- `fetch_prices(tickers, period_days)` — Yahoo Finance v8 chart API;
  rate-limited per-ticker fetch; corrupt-row detection; GBp filtering
- `align_prices(ticker_data) -> (Vec<i64>, DMatrix<f64>)` — intersects
  timestamps and produces a date-aligned price matrix
- `timestamps_to_dates(timestamps) -> Vec<String>` — UNIX-second to ISO-8601
- `resolve_isin(isin) -> Result<String, YahooError>` — ISIN to ticker via
  the Yahoo search API; prefers EUR-denominated exchanges

### Added — `regit-covariance-server`

#### HTTP API (`src/api/`)
- `SharedState = Arc<RwLock<AppState>>` — concurrent state for SSE + REST
- `AppState` struct — `results: HashMap<...>`, `event_logs: HashMap<...>`
- `ComputeRequest`, `ComputationResult`, `MpFitResponse`,
  `ConditionResponse`, `RiskMetricsResponse`, `SseEvent` — serialisable payloads
- `health()` — `GET /api/health`
- `compute(state, req)` — `POST /api/compute` triggers async pipeline
- `get_results(state, id)` — `GET /api/results/:id`
- `list_results(state)` — `GET /api/results`
- `resolve_isin(isin)` — `GET /api/resolve-isin/:isin`
- `stream(state, id)` — `GET /api/stream/:id` Server-Sent Events
- `run_pipeline(state, id, req)` — orchestrates fetch / generate -> log
  returns -> correlation -> eigen -> MP fit -> denoise -> condition ->
  VaR -> SRI -> divergence; emits SSE events at each stage

#### Static frontend (`static/index.html`)
- Single-page Chart.js visualisation: ticker / ISIN / synthetic input form,
  eigenvalue spectrum chart, Marchenko-Pastur fit overlay, correlation
  heatmaps (raw + denoised), SRI dashboard with Green/Yellow/Red divergence
  flag, condition-number improvement card

#### Binary entrypoint (`src/main.rs`)
- Tracing-subscriber configured at INFO; `PORT` env var (default 3000);
  binds to `0.0.0.0:$PORT`

### Added — Documentation

- `README.md` — full rewrite mirroring `regit-blackscholes` structure;
  added "Why this crate exists" narrative motivating the regulatory-noise
  problem (UCITS with N=500, T=252, q=0.504 -> 248 zero eigenvalues, SRI
  swing of 1-2 classes); added Methods table with module paths; added
  Algorithms table with primary citations; updated Architecture section to
  reflect the workspace layout
- `MATH.md` — complete rewrite, ~870 lines; plain-text formulas in code
  blocks (indexable for SEO and grep); per-module sections each mapping to
  a source file (`## Marchenko-Pastur — src/math/marchenko_pastur.rs`);
  inline primary citations on every section; full derivations of all 19
  components from sample correlation through divergence detection; algorithm
  references table at the end. Replaces the previous LaTeX-heavy 238-line draft
- `SECURITY.md` — added `## Supported Versions` table; expanded `## Scope`
  to distinguish numerical-correctness concerns (math modules) from
  network-and-serving concerns (api + Yahoo); explicit data disclaimer
- `CHANGELOG.md` — switched to per-module breakdown enumerating every public
  API; added crate-by-crate sections matching the new workspace layout

### Added — CI / release infrastructure

- `.github/workflows/ci.yml` — workspace-aware quality gate
  (`cargo fmt --all --check`, `cargo clippy --workspace --all-targets -- -D warnings`,
  `cargo test --workspace`, `cargo doc --workspace --no-deps`, `cargo deny check`);
  parallel `wasm` job that builds the core crate for `wasm32-unknown-unknown`
  on every PR
- `.github/workflows/release.yml` — tag-driven release pipeline
  (`v*.*.*`); pre-publish verification job (fmt + clippy + test + WASM); ordered
  publish to crates.io (core first, then yahoo); GitHub Release with
  changelog excerpt; `workflow_dispatch` with `dry_run` toggle for staging
- `Cargo.toml` (workspace root) — `[workspace.package]` for shared metadata,
  `[workspace.dependencies]` for pinned versions across all member crates,
  `[workspace.lints.clippy]` enforcing `pedantic = warn`, resolver "3"
- `justfile` — workspace-aware recipes (`fmt`, `lint`, `test`, `doc`,
  `build`, `serve`, `example`, `bench`, `wasm`, `deny`)

### Added — code quality and dependencies

- `nalgebra` 0.33 -> 0.34, `reqwest` 0.12 -> 0.13, `criterion` 0.5 -> 0.8;
  pinned `axum = 0.8.9`, `tokio = 1.52.2`, `proptest = 1.11.0`, `tower-http = 0.6.10`
- All `#[allow(clippy::cast_precision_loss)]` annotations removed from
  test code; replaced with lossless casts
  (`f64::from(u32::try_from(x).expect(...))` for bounded dimensions, and
  rewritten LCG random-number generators using `u32`-bounded right shifts so
  `f64::from(u32)` is provably exact)
- `clippy::pedantic` clean across the entire workspace and all targets

[1.0.0]: https://github.com/regit-io/regit-covariance/releases/tag/v1.0.0

---

*Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.*
