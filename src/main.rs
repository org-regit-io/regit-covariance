// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Binary entrypoint for regit-covariance.
//!
//! Starts the axum HTTP server, serves the static frontend, and exposes
//! REST + SSE endpoints for covariance denoising and risk validation.

use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    // Initialize tracing.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("info".parse().unwrap()))
        .init();

    let app = regit_covariance::api::router();

    let port = std::env::var("PORT").unwrap_or_else(|_| "3000".to_string());
    let addr = format!("0.0.0.0:{port}");
    tracing::info!("regit-covariance — https://www.regit.io");
    tracing::info!("Server starting on http://{addr}");
    tracing::info!("Open your browser to http://localhost:{port}");

    let listener = tokio::net::TcpListener::bind(&addr)
        .await
        .unwrap_or_else(|_| panic!("failed to bind to {addr}"));

    axum::serve(listener, app).await.expect("server error");
}
