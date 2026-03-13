# regit-covariance

Covariance matrix denoising for financial risk validation. Pure Rust.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.94%2B-orange.svg)](https://www.rust-lang.org)

## What it does

`regit-covariance` takes a returns matrix, estimates the covariance structure, applies denoising using random matrix theory, and produces validated risk metrics that serve as an independent second opinion against prescribed PRIIPs methodology.

For a fund with N = 500 assets and T = 252 trading days, the sample covariance matrix is rank-deficient with at least 248 zero eigenvalues. The remaining eigenvalues are systematically distorted. Risk metrics computed on this matrix inherit that distortion. This crate fixes it.

## Mathematical pipeline

```
Returns (T x N)
  -> Sample correlation matrix
  -> Eigendecomposition
  -> Marchenko-Pastur noise filtering
  -> Eigenvalue replacement + reconstruction
  -> Denoised correlation matrix (PSD, trace-preserving)
  -> Risk metrics (VaR, SRI, divergence report)
```

### Methods implemented

| Method | Reference |
|--------|-----------|
| Sample correlation (equal-weight + EWM) | Standard |
| Marchenko-Pastur density & noise fitting | Marchenko & Pastur (1967) |
| Eigenvalue denoising (constant + target) | Lopez de Prado (2018), Ch. 2 |
| Detoning (market mode removal) | Lopez de Prado (2018), Ch. 2 |
| Linear shrinkage | Ledoit & Wolf (2004) |
| Nonlinear shrinkage (analytical) | Ledoit & Wolf (2020) |
| Parametric VaR (Gaussian) | PRIIPs RTS, Annex II |
| Cornish-Fisher VaR | Cornish & Fisher (1938) |
| PRIIPs SRI (1-7 scale) | EU Regulation 2017/653 |

## Quick start

```bash
# Clone and build
git clone https://github.com/regit-io/regit-covariance.git
cd regit-covariance
cargo build --release

# Run the server (includes embedded visualization)
cargo run --release

# Open http://localhost:3000 in your browser
# Enter tickers or ISINs (e.g. AAPL,MSFT,GOOGL or LU1681043599)
# and click "Run Pipeline"
# Or switch to Synthetic mode for generated data
```

ISINs are automatically resolved to Yahoo Finance tickers via the search API, with preference for EUR-listed exchanges.

## API

### REST endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Embedded visualization (Chart.js) |
| `GET` | `/api/health` | Health check |
| `POST` | `/api/compute` | Trigger computation |
| `GET` | `/api/results` | List computation IDs |
| `GET` | `/api/results/:id` | Get full result as JSON |
| `GET` | `/api/stream/:id` | SSE event stream |

### Compute request

**Live mode** (fetches prices from Yahoo Finance):

```json
{
  "tickers": ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],
  "period_days": 365,
  "prescribed_sri": 5
}
```

**Synthetic mode** (generated data):

```json
{
  "num_assets": 50,
  "num_observations": 252,
  "seed": 42
}
```

The optional `prescribed_sri` field accepts the SRI (1-7) from a PRIIPs KID document, enabling direct comparison against the kernel's independent estimate. For multi-asset portfolios, per-asset SRI breakdown is included in the `risk_metrics` event.

### SSE events

The stream delivers events progressively:

1. `status` - Pipeline stage updates
2. `eigenvalues_raw` - Raw eigenvalue spectrum
3. `mp_fit` - Marchenko-Pastur fit (noise variance, bounds, signal/noise partition)
4. `eigenvalues_denoised` - Cleaned eigenvalues
5. `condition_number` - Before/after conditioning with health classification
6. `risk_metrics` - VaR, SRI, divergence report
7. `complete` - Full result payload

## Library usage

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

Every function is pure: data in, result out, no side effects. See [`examples/quickstart.rs`](examples/quickstart.rs) for a complete working example.

## Architecture

```
src/
  lib.rs                    # Public API
  main.rs                   # axum server + SSE

  math/
    sample_covariance.rs    # C = (1/T) X'X, EWM
    eigen.rs                # Eigendecomposition wrapper
    marchenko_pastur.rs     # MP density, noise fitting
    denoise.rs              # Eigenvalue replacement
    detone.rs               # Market mode removal
    ledoit_wolf.rs          # Linear + nonlinear shrinkage
    var.rs                  # Parametric + Cornish-Fisher VaR
    sri.rs                  # PRIIPs SRI (1-7)
    condition.rs            # Condition number monitoring

  data/
    returns.rs              # Log returns: r_t = ln(P_t / P_{t-1})
    provider/
      yahoo.rs              # Yahoo Finance v8 chart API

  api/
    routes.rs               # REST endpoints
    sse.rs                  # Server-Sent Events
    compute.rs              # Pipeline orchestration
    state.rs                # Shared state
```

One file, one mathematical operation. Each function is pure and composable.

## Testing

```bash
cargo test                        # 120 tests
cargo run --example quickstart    # Library usage demo
```

**87 unit tests** covering analytical correctness, trace preservation, PSD, symmetry, eigenvalue ordering, reconstruction accuracy, PRIIPs SRI thresholds, and edge cases.

**33 integration tests** across 5 test suites:
- `trace_preservation` — proptest: denoised trace = N, symmetry (randomized dimensions)
- `psd_check` — positive semi-definiteness across q ratios and seeds
- `denoise_identity` — pure noise → near-identity after denoising
- `mp_analytical` — MP bounds match theoretical formulas
- `pipeline_full` — end-to-end: returns → correlation → eigen → MP → denoise → VaR → SRI

## Code quality

- `#![forbid(unsafe_code)]` in all math modules
- `clippy::pedantic` with zero warnings
- Every public function documented with mathematical references
- No `unwrap()` in library code
- Deterministic: same input produces bit-identical output

## Dependencies

| Crate | Purpose | License |
|-------|---------|---------|
| `nalgebra` | Linear algebra, eigendecomposition | Apache-2.0 |
| `axum` | HTTP server + SSE | MIT |
| `tokio` | Async runtime | MIT |
| `serde` | Serialization | Apache-2.0/MIT |
| `thiserror` | Error types | Apache-2.0/MIT |
| `reqwest` | HTTP client (Yahoo Finance) | Apache-2.0/MIT |
| `chrono` | Date handling | Apache-2.0/MIT |

License policy enforced via `cargo-deny`. No copyleft dependencies.

## Data disclaimer

Market data is fetched from Yahoo Finance and is provided for **educational and research purposes only**. Yahoo Finance data should not be used for production regulatory reporting, live trading decisions, or as the sole basis for investment advice. For production use in regulated environments, source market data from a licensed provider (Bloomberg, Refinitiv, etc.).

## References

- Marchenko, V. A., & Pastur, L. A. (1967). Distribution of eigenvalues for some sets of random matrices. *Matematicheskii Sbornik*, 114(4), 507-536.
- Ledoit, O., & Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *Journal of Multivariate Analysis*, 88(2), 365-411.
- Ledoit, O., & Wolf, M. (2020). Analytical nonlinear shrinkage of large-dimensional covariance matrices. *Annals of Statistics*, 48(5), 3043-3065.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley. Chapters 2-3.
- Cornish, E. A., & Fisher, R. A. (1938). Moments and cumulants in the specification of distributions. *Revue de l'Institut International de Statistique*, 5(4), 307-320.
- PRIIPs Delegated Regulation (EU) 2017/653, Annex II.

## Documentation

- [MATH.md](MATH.md) — Full mathematical derivations (Marchenko-Pastur through SRI classification)
- [CHANGELOG.md](CHANGELOG.md) — Release history
- [SECURITY.md](SECURITY.md) — Vulnerability disclosure policy

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE).

---

Part of [Regit OS](https://www.regit.io) — the operating system for investment products. From Luxembourg.
