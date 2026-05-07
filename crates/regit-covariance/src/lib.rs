// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! # regit-covariance
//!
//! Covariance matrix denoising for financial risk validation.
//!
//! This crate implements Marchenko-Pastur filtering, Ledoit-Wolf shrinkage,
//! and detoning for large-dimensional covariance matrices. Built to validate
//! PRIIPs risk metrics against prescribed regulatory methodology.
//!
//! Pure mathematical core: no I/O, no async, no network. Compatible with
//! `wasm32-unknown-unknown` and `wasm32-wasi` targets. The full pipeline
//! (returns -> correlation -> denoise -> `VaR` -> SRI) runs synchronously
//! and deterministically.
//!
//! For market-data ingestion (Yahoo Finance) see the companion crate
//! `regit-covariance-yahoo`. For an HTTP demo server see
//! `regit-covariance-server`.
//!
//! Part of [Regit OS](https://www.regit.io), the operating system for
//! investment products.
//!
//! ## Modules
//!
//! - [`data`] — Log-return computation from price levels.
//! - [`math`] — Covariance estimation, denoising, shrinkage, risk metrics.

#![forbid(unsafe_code)]

pub mod data;
pub mod math;
