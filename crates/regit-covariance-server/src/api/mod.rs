// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! HTTP API and Server-Sent Events streaming.
//!
//! Exposes computation results via REST endpoints and streams
//! computation stages in real time via SSE for live visualization.
//!
//! ## Endpoints
//!
//! | Method | Path                | Description                    |
//! |--------|---------------------|--------------------------------|
//! | GET    | `/`                 | Static frontend (index.html)   |
//! | GET    | `/api/health`       | Health check                   |
//! | POST   | `/api/compute`      | Trigger computation            |
//! | GET    | `/api/results`      | List all computation IDs       |
//! | GET    | `/api/results/:id`  | Get computation result         |
//! | GET    | `/api/stream/:id`   | SSE event stream               |
//!
//! Built on [`axum`] with [`tokio`] as the async runtime.

pub mod compute;
pub mod routes;
pub mod sse;
pub mod state;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post};
use tokio::sync::RwLock;
use tower_http::services::ServeFile;

use state::AppState;

/// Build the axum router with all endpoints.
pub fn router() -> Router {
    let state = Arc::new(RwLock::new(AppState::default()));

    Router::new()
        .route_service("/", ServeFile::new("static/index.html"))
        .route("/api/health", get(routes::health))
        .route("/api/compute", post(routes::compute))
        .route("/api/resolve-isin/{isin}", get(routes::resolve_isin))
        .route("/api/results", get(routes::list_results))
        .route("/api/results/{id}", get(routes::get_results))
        .route("/api/stream/{id}", get(sse::stream))
        .with_state(state)
}
