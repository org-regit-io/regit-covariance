// Copyright 2026 Regit.io — Nicolas Koenig
// SPDX-License-Identifier: Apache-2.0

//! Server-Sent Events (SSE) streaming.
//!
//! When a client connects to `GET /api/stream/:id`, the server replays
//! all computation events for that ID with a short delay between each
//! event, simulating a live computation stream.

use std::convert::Infallible;
use std::time::Duration;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::Stream;

use super::state::SharedState;

/// SSE stream endpoint — replays computation events for a given ID.
///
/// New clients receive the full event sequence with a 200ms delay
/// between events for progressive visualization.
///
/// # Errors
///
/// Returns 404 if the computation ID is not found.
pub async fn stream(
    State(state): State<SharedState>,
    Path(id): Path<String>,
) -> Result<Sse<impl Stream<Item = Result<Event, Infallible>>>, impl IntoResponse> {
    let app_state = state.read().await;
    let events = match app_state.event_logs.get(&id) {
        Some(events) => events.clone(),
        None => return Err(StatusCode::NOT_FOUND),
    };
    drop(app_state);

    let stream = async_stream::stream! {
        for sse_event in events {
            let data = serde_json::to_string(&sse_event.data).unwrap_or_default();
            let event = Event::default()
                .event(sse_event.event)
                .data(data);
            yield Ok::<_, Infallible>(event);
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    };

    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}
