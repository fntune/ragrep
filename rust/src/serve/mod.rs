//! HTTP search service.

pub mod auth;
pub mod search;

use std::sync::Arc;

use anyhow::Result;
use axum::{extract::State, routing::get, Json, Router};
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::config::Config;
use crate::embed::{self, Embedder};
use crate::index::{bm25::Bm25, flat::Flat, store};

pub struct AppState {
    pub cfg: Config,
    pub chunks: Vec<crate::models::Chunk>,
    pub flat: Flat,
    pub bm25: Bm25,
    pub embedder: Box<dyn Embedder>,
}

impl AppState {
    pub fn load(cfg: Config) -> Result<Self> {
        let dir = cfg.index_dir();
        let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?;
        let runtime = store::load_runtime(&dir, embedder.dim())?;
        Ok(Self {
            cfg,
            chunks: runtime.chunks,
            flat: runtime.flat,
            bm25: runtime.bm25,
            embedder,
        })
    }
}

pub fn router(state: Arc<AppState>, auth_policy: auth::Policy) -> Router {
    auth::apply(
        Router::new()
            .route("/search", get(search::handle))
            .route("/health", get(health))
            .with_state(state)
            .layer(TraceLayer::new_for_http()),
        auth_policy,
    )
}

async fn health(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    Json(json!({
        "status": "ok",
        "chunks": state.chunks.len(),
        "provider": state.embedder.provider(),
        "model": state.embedder.model(),
    }))
}
