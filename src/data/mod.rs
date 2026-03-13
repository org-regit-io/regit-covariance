// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Data ingestion pipeline.
//!
//! Handles universe definition, price fetching via pluggable providers,
//! local Parquet caching, and log returns computation.
//!
//! The crate is a computation engine, not a data provider. Institutions
//! source market data from their licensed providers; this module accepts
//! that data through a `PriceProvider` trait defined in the `provider`
//! submodule.

pub mod provider;
pub mod returns;
