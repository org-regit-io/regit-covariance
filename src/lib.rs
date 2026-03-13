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
//! Part of [Regit OS](https://www.regit.io), the operating system for
//! investment products.
//!
//! ## Modules
//!
//! - [`data`] — Universe definition, price providers, caching, log returns.
//! - [`math`] — Covariance estimation, denoising, shrinkage, risk metrics.
//! - [`api`] — HTTP API and Server-Sent Events streaming.

pub mod api;
pub mod data;
pub mod math;
