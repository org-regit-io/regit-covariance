# regit-covariance

Covariance matrix denoising for financial risk validation. Pure Rust.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://www.rust-lang.org)

## What it does

`regit-covariance` takes a returns matrix, applies Marchenko-Pastur denoising and Ledoit-Wolf shrinkage to the sample covariance, and produces validated risk metrics — VaR, SRI, divergence flags — that serve as an **independent second opinion against the prescribed PRIIPs methodology**. Covers the full pipeline from log returns to MRM classification.

Every algorithm is hand-rolled from primary paper sources. A regulator, quant auditor, or new engineer can open any source file and trace every number to a citable formula.

## Why this crate exists

The PRIIPs Regulatory Technical Standard ((EU) 2017/653, Annex II) prescribes a fixed recipe for computing the Summary Risk Indicator (SRI) on a Key Information Document (KID): build the sample covariance, derive a Cornish-Fisher VaR, look up the VaR-equivalent volatility against seven thresholds. The recipe assumes the sample covariance is a reliable estimate of the true covariance.

**For realistic funds, that assumption is wrong.**

Take a UCITS with `N = 500` underlyings and `T = 252` daily observations of usable history. The observation ratio `q = T/N = 0.504` is below 1, which means:

- The sample covariance matrix is **rank-deficient** with at least `248` exact-zero eigenvalues
- The remaining eigenvalues are **systematically distorted** by random matrix theory — small ones are biased downward, large ones upward
- The condition number is **infinite** before denoising; portfolio risk metrics computed on this matrix inherit the distortion
- The resulting SRI can swing **one or two classes** under resampling — meaning the prescribed methodology gives a different answer depending on which 252-day window you pick

A class-4 product marketed as a class-3 understates risk to retail investors. A class-3 product marketed as a class-4 carries a commercial penalty for no reason. Both are failures of the same underlying problem: the prescribed recipe does not account for estimation noise.

**This crate runs the same calculation a second time on a noise-cleaned covariance.** When the kernel SRI matches the prescribed SRI (Green flag), the regulatory number is reliable. When it differs by one class (Yellow), the product sits near a boundary and risk teams should be aware. When it differs by two or more (Red), the prescribed SRI is materially distorted and warrants investigation. Same regulation, same VaR formula, same thresholds — only the input covariance is denoised.

This is not an attempt to replace the regulatory methodology. It is a verification kernel that surfaces when blindly following the RTS produces a wrong conclusion.

This sits within the broader [Regit OS](https://www.regit.io) effort: tools for investment products that are regulation-compliant by default, while keeping the underlying calculations grounded in quant-proven math — Marchenko-Pastur, Ledoit-Wolf, primary-source derivations — alongside the regulatory recipe rather than instead of it.

## Quick start

```bash
git clone https://github.com/regit-io/regit-covariance.git
cd regit-covariance
cargo build --workspace --release

# Run the visualization server
cargo run -p regit-covariance-server --release
# Open http://localhost:3000, enter tickers (AAPL,MSFT,...) or an ISIN
# (LU1681043599), or switch to Synthetic mode for generated data.
```

Library-only consumers add the math crate to their `Cargo.toml`:

```toml
[dependencies]
regit-covariance = "1.0"
```

```rust
use regit_covariance::math::{
    sample_covariance::correlation_matrix,
    eigen::eigendecompose,
    marchenko_pastur::fit_sigma_sq,
    denoise::{denoise, DenoiseMethod},
    condition,
};

let cov = correlation_matrix(&returns)?;
let eigen = eigendecompose(&cov.correlation)?;
let mp = fit_sigma_sq(&eigen.eigenvalues, cov.q)?;
let cleaned = denoise(&eigen, &mp, DenoiseMethod::Target);

let eigen_clean = eigendecompose(&cleaned.matrix)?;
let improvement = condition::compare(&eigen, &eigen_clean);
println!("Condition number: {} -> {} ({}x improvement)",
    improvement.before.condition_number,
    improvement.after.condition_number,
    improvement.improvement_factor);
```

See [`examples/quickstart.rs`](examples/quickstart.rs) for a complete working example.

## Methods

| Method | Module | Reference |
|--------|--------|-----------|
| Sample correlation (equal-weight) | `src/math/sample_covariance.rs` | Lopez de Prado (2018), Ch. 2 |
| Exponentially weighted correlation | `src/math/sample_covariance.rs` | RiskMetrics Technical Document (1996) |
| Symmetric eigendecomposition | `src/math/eigen.rs` | Golub & Van Loan (2013), Ch. 8 |
| Marchenko-Pastur density & noise fit | `src/math/marchenko_pastur.rs` | Marchenko & Pastur (1967) |
| Eigenvalue denoising (Constant + Target) | `src/math/denoise.rs` | Lopez de Prado (2018), Ch. 2 |
| Detoning (market-mode removal) | `src/math/detone.rs` | Lopez de Prado (2018), Ch. 2 |
| Linear shrinkage | `src/math/ledoit_wolf.rs` | Ledoit & Wolf (2004) |
| Nonlinear analytical shrinkage | `src/math/ledoit_wolf.rs` | Ledoit & Wolf (2020) |
| Condition number monitoring | `src/math/condition.rs` | Golub & Van Loan (2013) |
| Parametric VaR (Gaussian) | `src/math/var.rs` | PRIIPs RTS, Annex II |
| Cornish-Fisher VaR | `src/math/var.rs` | Cornish & Fisher (1938) |
| VaR-Equivalent Volatility (VEV) | `src/math/sri.rs` | PRIIPs RTS, Annex II |
| PRIIPs SRI (1-7 scale) | `src/math/sri.rs` | EU Regulation 2017/653 |
| Divergence detection (prescribed vs kernel) | `src/math/sri.rs` | Implementation-specific |
| Log returns | `src/data/returns.rs` | Standard |

```
Returns (T x N)
  -> Sample correlation matrix
  -> Eigendecomposition
  -> Marchenko-Pastur noise filtering
  -> Eigenvalue replacement + reconstruction
  -> Denoised correlation matrix (PSD, trace-preserving)
  -> Risk metrics (VaR, SRI, divergence report)
```

## HTTP API

The binary ships an embedded server (`axum` + Server-Sent Events) for visualisation and pipeline streaming.

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Embedded Chart.js visualisation |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/compute` | Trigger computation |
| `GET` | `/api/results` | List computation IDs |
| `GET` | `/api/results/:id` | Get full result as JSON |
| `GET` | `/api/stream/:id` | SSE event stream |
| `GET` | `/api/isin/:isin` | Resolve ISIN to Yahoo ticker |

Compute request (live mode):

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "period_days": 365,
  "prescribed_sri": 5
}
```

Compute request (synthetic mode):

```json
{
  "num_assets": 50,
  "num_observations": 252,
  "seed": 42
}
```

The optional `prescribed_sri` field accepts the SRI (1-7) from a PRIIPs KID document, enabling direct comparison against the kernel's independent estimate.

The SSE stream emits `status`, `eigenvalues_raw`, `mp_fit`, `eigenvalues_denoised`, `condition_number`, `risk_metrics`, and `complete` events progressively as the pipeline runs.

## Architecture

The repository is a **Cargo workspace** with three crates. Splitting along the I/O boundary keeps the math core publishable to crates.io, WASM-compatible, and dependency-light, while the network and serving surfaces sit in clearly bounded sibling crates.

```
crates/
  regit-covariance/                # PURE MATH. Publishable. WASM-compatible.
    src/lib.rs                     # Public API
    src/math/
      sample_covariance.rs         # C = (1/T) X^T X, EWM
      eigen.rs                     # Eigendecomposition wrapper (descending sort)
      marchenko_pastur.rs          # MP density, noise variance fit (fixed-point)
      denoise.rs                   # Eigenvalue replacement (Constant + Target)
      detone.rs                    # Market mode removal
      ledoit_wolf.rs               # Linear (2004) + nonlinear analytical (2020)
      var.rs                       # Parametric + Cornish-Fisher VaR
      sri.rs                       # PRIIPs VEV / MRM / divergence
      condition.rs                 # Condition number monitoring
    src/data/returns.rs            # Log returns: r_t = ln(P_t / P_{t-1})
    tests/                         # 33 integration tests (proptest, PSD, MP)
    examples/quickstart.rs         # Library usage demo
    benches/eigendecompose.rs      # Criterion benchmarks

  regit-covariance-yahoo/          # Yahoo Finance provider. Native only.
    src/lib.rs                     # v8 chart API + ISIN search

  regit-covariance-server/         # axum + SSE demo server. publish = false.
    src/main.rs                    # Binary entrypoint
    src/api/                       # REST + SSE handlers, pipeline orchestration
    static/index.html              # Embedded Chart.js frontend
```

Dependency direction: `server -> { core, yahoo }`, `yahoo -> reqwest`, `core -> { nalgebra, serde, thiserror }`. The math core has no async, no network, no I/O — `cargo build -p regit-covariance --target wasm32-unknown-unknown --release` produces a clean WASM artefact (~280 KB rlib). One file, one mathematical operation; each function is pure and composable.

## Testing

```bash
cargo test --workspace                              # 120 tests
cargo run -p regit-covariance --example quickstart  # Library usage demo
```

**87 unit tests** covering analytical correctness, trace preservation, PSD, symmetry, eigenvalue ordering, reconstruction accuracy, PRIIPs SRI thresholds, and edge cases.

**33 integration tests** across 5 suites:
- `trace_preservation` — denoised trace = N, symmetry under Constant and Target methods
- `psd_check` — positive semi-definiteness across q ratios and seeds
- `denoise_identity` — pure noise -> near-identity after denoising
- `mp_analytical` — MP bounds match theoretical formulas
- `pipeline_full` — end-to-end: returns -> correlation -> eigen -> MP -> denoise -> VaR -> SRI

## Code quality

- `#![forbid(unsafe_code)]` in all math modules
- `clippy::pedantic` with zero warnings
- Every public function documented with mathematical references
- No `unwrap()` in library code
- Deterministic: same input produces bit-identical output

## Dependencies

The math core has a deliberately small dependency footprint — three runtime crates, each carrying a single, well-defined responsibility:

### `regit-covariance` (math core)

| Crate | Purpose | License |
|-------|---------|---------|
| `nalgebra` | Linear algebra; symmetric eigendecomposition | Apache-2.0 |
| `serde` | `#[derive(Serialize, Deserialize)]` on result types | Apache-2.0/MIT |
| `thiserror` | Error type derivation | Apache-2.0/MIT |

No async runtime, no network, no I/O. Compatible with `wasm32-unknown-unknown`. Implementing a symmetric eigensolver from scratch would be a multi-thousand-line effort that this crate explicitly declines — `nalgebra` is the right place for that work and is itself Apache-2.0 with a clean dependency graph.

### `regit-covariance-yahoo` (Yahoo Finance provider) — adds

| Crate | Purpose | License |
|-------|---------|---------|
| `reqwest` | HTTPS client for the Yahoo v8 chart and search APIs | Apache-2.0/MIT |
| `tokio` | Async runtime | MIT |
| `chrono` | Date alignment | Apache-2.0/MIT |
| `tracing` | Structured logging | MIT |

### `regit-covariance-server` (axum demo) — adds

| Crate | Purpose | License |
|-------|---------|---------|
| `axum` | HTTP server + SSE | MIT |
| `tower-http` | Static file serving | MIT |
| `async-stream`, `futures` | SSE plumbing | Apache-2.0/MIT |
| `tracing-subscriber` | Logging output | MIT |

License policy enforced via `cargo-deny`. No copyleft dependencies anywhere in the workspace.

## Data disclaimer

Market data is fetched from Yahoo Finance and is provided for **educational and research purposes only**. Yahoo Finance data should not be used for production regulatory reporting, live trading decisions, or as the sole basis for investment advice. For production use in regulated environments, source market data from a licensed provider (Bloomberg, Refinitiv, etc.).

## Algorithms

All implemented from primary paper sources. No ports from Python, no reading existing Rust crates.

| Algorithm | Primary reference |
|-----------|-------------------|
| Sample covariance / correlation | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2 |
| Exponentially weighted moments | RiskMetrics Technical Document (J.P. Morgan / Reuters, 1996) |
| Symmetric eigendecomposition | Golub & Van Loan, *Matrix Computations*, 4th ed. (Johns Hopkins, 2013) |
| Marchenko-Pastur law | Marchenko & Pastur, *Matematicheskii Sbornik* 114(4):507-536 (1967) |
| RMT noise filtering of correlation matrices | Laloux, Cizeau, Bouchaud & Potters, *Physical Review Letters* 83(7):1467-1470 (1999) |
| Cleaning correlation matrices (review) | Bun, Bouchaud & Potters, *Physics Reports* 666:1-109 (2017) |
| Eigenvalue replacement (Constant + Target) | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2, Code Snippet 2.6 |
| Detoning | Lopez de Prado, *Advances in Financial Machine Learning* (Wiley, 2018), Ch. 2, Section 2.6 |
| Linear shrinkage | Ledoit & Wolf, *Journal of Multivariate Analysis* 88(2):365-411 (2004) |
| Nonlinear analytical shrinkage | Ledoit & Wolf, *Annals of Statistics* 48(5):3043-3065 (2020) |
| QuEST function | Ledoit & Wolf, *Journal of Multivariate Analysis* 159:55-77 (2017) |
| Statistical physics view of risk | Bouchaud & Potters, *Theory of Financial Risk and Derivative Pricing*, 2nd ed. (Cambridge, 2003) |
| Machine learning for asset managers | Lopez de Prado, *Machine Learning for Asset Managers* (Cambridge, 2020) |
| Standard normal quantile | Abramowitz & Stegun, *Handbook of Mathematical Functions*, formula 26.2.23 (1964) |
| Cornish-Fisher expansion | Cornish & Fisher, *Revue de l'Institut International de Statistique* 5(4):307-320 (1938) |
| PRIIPs SRI / VEV / MRM | Commission Delegated Regulation (EU) 2017/653, Annex II |
| PRIIPs Regulation (parent act) | Regulation (EU) 1286/2014 |

## Documentation

- [MATH.md](MATH.md) — Full mathematical derivations (sample covariance through divergence detection)
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [SECURITY.md](SECURITY.md) — Vulnerability disclosure policy

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

```
Copyright 2026 Regit.io — Nicolas Koenig
```

---

Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.
