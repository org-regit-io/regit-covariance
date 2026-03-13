// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! REST endpoint handlers.
//!
//! - `GET /api/health` — Health check.
//! - `POST /api/compute` — Trigger a new computation.
//! - `GET /api/results/:id` — Retrieve computation results.

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::Json;
use serde_json::Value;

use crate::data::provider::yahoo;

use super::compute::run_pipeline;
use super::state::{ComputeRequest, SharedState};

/// Health check endpoint.
pub async fn health() -> Json<Value> {
    Json(serde_json::json!({
        "status": "ok",
        "service": "regit-covariance",
        "version": env!("CARGO_PKG_VERSION"),
    }))
}

/// Trigger a new computation.
///
/// Accepts a JSON body with either:
/// - `tickers` (live mode): fetches prices from Yahoo Finance
/// - `num_assets`, `num_observations`, `seed` (synthetic mode)
///
/// Returns the computation ID and initial status.
///
/// # Errors
///
/// Returns 500 if the pipeline fails.
pub async fn compute(
    State(state): State<SharedState>,
    Json(request): Json<ComputeRequest>,
) -> (StatusCode, Json<Value>) {
    let computation_id = if request.tickers.is_empty() {
        format!(
            "comp-{}-{}x{}",
            request.seed, request.num_assets, request.num_observations
        )
    } else {
        format!(
            "comp-live-{}",
            request
                .tickers
                .iter()
                .map(|t| t.replace('.', "_"))
                .collect::<Vec<_>>()
                .join("-")
        )
    };

    match run_pipeline(&request, &computation_id).await {
        Ok((result, events)) => {
            let response = serde_json::json!({
                "id": computation_id,
                "status": result.status,
                "num_assets": result.num_assets,
                "num_observations": result.num_observations,
            });

            let mut app_state = state.write().await;
            app_state.results.insert(computation_id.clone(), result);
            app_state.event_logs.insert(computation_id, events);

            (StatusCode::CREATED, Json(response))
        }
        Err(err) => {
            let response = serde_json::json!({
                "id": computation_id,
                "status": "error",
                "error": err,
            });
            (StatusCode::INTERNAL_SERVER_ERROR, Json(response))
        }
    }
}

/// Retrieve computation results by ID.
///
/// # Errors
///
/// Returns 404 if the computation ID is not found.
pub async fn get_results(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Json<Value>, StatusCode> {
    let app_state = state.read().await;
    match app_state.results.get(&id) {
        Some(result) => Ok(Json(serde_json::to_value(result).unwrap_or_default())),
        None => Err(StatusCode::NOT_FOUND),
    }
}

/// List all computation IDs.
pub async fn list_results(State(state): State<SharedState>) -> Json<Value> {
    let app_state = state.read().await;
    let ids: Vec<&String> = app_state.results.keys().collect();
    Json(serde_json::json!({ "computations": ids }))
}

/// Resolve an ISIN to a Yahoo Finance ticker symbol.
///
/// # Errors
///
/// Returns 404 if no matching ticker is found.
pub async fn resolve_isin(Path(isin): Path<String>) -> (StatusCode, Json<Value>) {
    match yahoo::resolve_isin(&isin).await {
        Ok(ticker) => (
            StatusCode::OK,
            Json(serde_json::json!({ "isin": isin, "ticker": ticker })),
        ),
        Err(e) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({ "isin": isin, "error": e.to_string() })),
        ),
    }
}
