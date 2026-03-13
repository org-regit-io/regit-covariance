<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] — 2026-03-13

Initial release.

### Added

- **Sample covariance estimation** — equal-weight and exponentially weighted (EWM) correlation matrices
- **Eigendecomposition** — sorted descending, with eigenvector matrix
- **Marchenko-Pastur filtering** — MP density, noise variance fitting via fixed-point iteration, signal/noise partition
- **Eigenvalue denoising** — constant and target shrinkage methods, trace-preserving reconstruction
- **Detoning** — configurable market mode removal (top k eigenmodes) with optional re-addition
- **Ledoit-Wolf shrinkage** — linear (2004) and nonlinear analytical (2020) estimators
- **Condition number monitoring** — before/after comparison with health classification (healthy/acceptable/unstable)
- **Parametric VaR** — Gaussian, at configurable confidence levels
- **Cornish-Fisher VaR** — skewness and excess kurtosis adjustment from actual portfolio return series
- **PRIIPs SRI classification** — MRM 1-7 scale per EU Delegated Regulation 2017/653
- **VaR-Equivalent Volatility (VEV)** — mapping VaR to the SRI scale
- **Divergence detection** — prescribed vs kernel SRI comparison with green/amber/red flagging
- **Per-asset SRI breakdown** — individual annualized volatility and MRM from denoised covariance diagonal
- **ISIN resolution** — automatic ISIN-to-ticker mapping via Yahoo Finance search API, with EUR-exchange preference
- **HTTP API** — `POST /api/compute`, `GET /api/results/:id`, `GET /api/stream/:id` via axum
- **SSE streaming** — progressive event delivery (eigenvalues, MP fit, denoised spectrum, condition, risk metrics)
- **Static frontend** — embedded single-page visualization with Chart.js (eigenvalue spectrum, correlation heatmaps, SRI dashboard)
- **Yahoo Finance provider** — price fetching with date alignment, corrupt data detection, GBp listing avoidance
- **120 tests** — 87 unit tests + 33 integration tests (proptest property-based, PSD checks, MP analytical, full pipeline)
- **MATH.md** — full mathematical derivations from Marchenko-Pastur through to SRI classification
- **CI** — GitHub Actions: fmt, clippy (pedantic), test, doc, cargo-deny

[0.1.0]: https://github.com/regit-io/regit-covariance/releases/tag/v0.1.0
