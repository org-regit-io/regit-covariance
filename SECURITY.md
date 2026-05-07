<!-- Copyright 2026 Regit.io — Nicolas Koenig -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a Vulnerability

If you discover a security vulnerability in `regit-covariance`, please report it responsibly:

1. **Do not** open a public GitHub issue
2. Email **nicolas.koenig@regit.io** with a description of the vulnerability
3. Include steps to reproduce if possible
4. We will acknowledge receipt within 48 hours and provide a timeline for a fix

## Scope

This crate has two distinct security surfaces.

### Numerical correctness (math modules)

The `math/` modules perform pure mathematical computation — eigendecomposition, Marchenko-Pastur fitting, denoising, Ledoit-Wolf shrinkage, VaR, SRI classification. These functions take numeric input and produce numeric output with no I/O or external state. The primary security concern here is **numerical correctness**: an error in pricing, denoising, or risk classification could lead to incorrect financial decisions or misclassified PRIIPs SRIs published in retail KIDs.

If you find a numerical accuracy issue that falls outside the documented tolerance bounds (typically `1e-10` for analytical tests, looser for randomised proptest invariants), please report it using the same process above.

### Network and serving surface (`api/` + `data/provider/yahoo.rs`)

The `api/` module exposes an HTTP server (axum) and the `data/provider/yahoo.rs` module makes outbound HTTP calls to Yahoo Finance. Concerns relevant here:

- **Input validation** — `POST /api/compute` accepts ticker lists, ISINs, weights, and synthetic-mode parameters. All inputs are validated for length and finiteness before reaching the math layer.
- **Server-Sent Events** — long-lived streams are unbounded by design (one per `/api/stream/:id` connection); deployments should bound concurrency at the reverse-proxy layer.
- **Outbound calls** — Yahoo Finance is queried over HTTPS with a default `reqwest` client (TLS via `rustls` / `native-tls`). No credentials are sent. Responses are JSON-decoded with `serde`; corrupt or hostile payloads should fail closed (return `YahooError`).
- **No persistence** — computation results are held in `Arc<RwLock<HashMap>>` in process memory only. There is no database, file write, or shared cache. Restarts clear all state.

The crate is **not** intended for direct exposure to the public internet. Deployments should sit behind authentication, rate-limiting, and TLS termination managed by an operator-controlled reverse proxy.

### Data disclaimer (production use)

Yahoo Finance market data is for **educational and research purposes only**. This crate must not be used as the sole basis for production regulatory reporting or live trading decisions. Numerical correctness of the kernel does not equate to fitness for regulated production use — the upstream data source is unsupported for those purposes. See [README.md](README.md#data-disclaimer) for full details.

## Dependencies

License and supply-chain concerns are policed via `cargo-deny` (`deny.toml` in the repository root). The advisories database is checked in CI on every push. Dependency upgrades that introduce non-allowed licences or active advisories are rejected at the gate.
