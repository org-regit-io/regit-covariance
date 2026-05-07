// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Data transformation primitives.
//!
//! Currently exposes log-return computation from price levels. Market-data
//! ingestion (e.g. Yahoo Finance) lives in the separate
//! `regit-covariance-yahoo` crate to keep this core crate WASM-compatible
//! and dependency-light.

pub mod returns;
