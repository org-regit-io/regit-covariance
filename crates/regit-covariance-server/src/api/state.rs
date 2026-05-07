// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Application state shared across handlers.
//!
//! Holds computation results in an `Arc<RwLock<>>` so that SSE streams
//! and REST endpoints can read the latest state concurrently.

use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

/// Shared application state.
pub type SharedState = Arc<RwLock<AppState>>;

/// Top-level application state.
#[derive(Debug, Default)]
pub struct AppState {
    /// Completed computation results, keyed by computation ID.
    pub results: HashMap<String, ComputationResult>,
    /// SSE event log per computation, for replay to new clients.
    pub event_logs: HashMap<String, Vec<SseEvent>>,
}

/// A single SSE event for streaming.
#[derive(Debug, Clone, Serialize)]
pub struct SseEvent {
    /// Event type (e.g., "status", "eigenvalues\_raw", "risk\_metrics").
    pub event: String,
    /// JSON payload.
    pub data: serde_json::Value,
}

/// Full computation result.
#[derive(Debug, Clone, Serialize)]
pub struct ComputationResult {
    /// Unique computation ID.
    pub id: String,
    /// Status: "running", "complete", "error".
    pub status: String,
    /// Number of assets (N).
    pub num_assets: usize,
    /// Number of observations (T).
    pub num_observations: usize,
    /// Raw eigenvalues (descending).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eigenvalues_raw: Option<Vec<f64>>,
    /// Denoised eigenvalues.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eigenvalues_denoised: Option<Vec<f64>>,
    /// Marchenko-Pastur fit results.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mp_fit: Option<MpFitResponse>,
    /// Condition number before/after.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub condition: Option<ConditionResponse>,
    /// Risk metrics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub risk_metrics: Option<RiskMetricsResponse>,
}

/// Marchenko-Pastur fit summary for API response.
#[derive(Debug, Clone, Serialize)]
pub struct MpFitResponse {
    /// Fitted noise variance.
    pub sigma_sq: f64,
    /// Upper MP bound.
    pub lambda_plus: f64,
    /// Lower MP bound.
    pub lambda_minus: f64,
    /// Number of signal eigenvalues.
    pub signal_count: usize,
    /// Number of noise eigenvalues.
    pub noise_count: usize,
}

/// Condition number summary for API response.
#[derive(Debug, Clone, Serialize)]
pub struct ConditionResponse {
    /// Condition number before denoising.
    pub before: f64,
    /// Condition number after denoising.
    pub after: f64,
    /// Improvement factor.
    pub improvement_factor: f64,
    /// Health classification after denoising.
    pub health: String,
}

/// Risk metrics summary for API response.
#[derive(Debug, Clone, Serialize)]
pub struct RiskMetricsResponse {
    /// Portfolio `VaR` (parametric).
    pub var_parametric: f64,
    /// Portfolio `VaR` (Cornish-Fisher).
    pub var_cornish_fisher: f64,
    /// Portfolio volatility.
    pub portfolio_volatility: f64,
    /// Prescribed SRI (from raw covariance).
    pub prescribed_sri: u8,
    /// Kernel SRI (from denoised covariance).
    pub kernel_sri: u8,
    /// Divergence flag: "green", "yellow", "red".
    pub divergence_flag: String,
}

/// Computation request from the client.
#[derive(Debug, Deserialize)]
pub struct ComputeRequest {
    /// Ticker symbols to fetch (live mode). If empty, uses synthetic data.
    #[serde(default)]
    pub tickers: Vec<String>,
    /// Portfolio weights (same order as tickers). If empty, equal-weight.
    #[serde(default)]
    pub weights: Vec<f64>,
    /// Prescribed SRI from KID (for divergence comparison). Optional.
    #[serde(default)]
    pub prescribed_sri: Option<u8>,
    /// Number of calendar days of history to fetch (default: 365).
    #[serde(default = "default_period_days")]
    pub period_days: u32,
    /// Number of assets to simulate (synthetic mode only).
    #[serde(default = "default_num_assets")]
    pub num_assets: usize,
    /// Number of observations to simulate (synthetic mode only).
    #[serde(default = "default_num_observations")]
    pub num_observations: usize,
    /// Random seed for reproducibility (synthetic mode only).
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_period_days() -> u32 {
    365
}

fn default_num_assets() -> usize {
    50
}

fn default_num_observations() -> usize {
    252
}

fn default_seed() -> u64 {
    42
}
