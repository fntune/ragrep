//! HTTP search service.

pub mod auth;
pub mod knowledge;
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
            .route(
                "/knowledge/records/:source",
                get(knowledge::list_records).post(knowledge::batch_records),
            )
            .route(
                "/knowledge/records/:source/:id",
                get(knowledge::get_record)
                    .put(knowledge::put_record)
                    .delete(knowledge::delete_record),
            )
            .route("/knowledge/search", get(knowledge::handle))
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;

    use super::*;
    use crate::models::Chunk;

    struct DummyEmbedder;

    impl Embedder for DummyEmbedder {
        fn provider(&self) -> &str {
            "dummy"
        }

        fn model(&self) -> &str {
            "dummy"
        }

        fn dim(&self) -> usize {
            2
        }

        fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        fn embed_documents(
            &self,
            texts: &[&str],
            _batch_size: usize,
            _checkpoint: Option<&std::path::Path>,
        ) -> Result<Vec<Vec<f32>>> {
            Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
        }
    }

    #[test]
    fn router_builds_knowledge_record_routes() {
        let dir = tempfile::tempdir().unwrap();
        let embeddings = [1.0f32, 0.0];
        let embeddings_path = dir.path().join("embeddings.bin");
        std::fs::write(&embeddings_path, bytemuck::cast_slice(&embeddings)).unwrap();
        let chunks = vec![Chunk {
            id: "freshdesk:123:0".into(),
            doc_id: "freshdesk:123".into(),
            content: "KYC content".into(),
            title: "KYC".into(),
            source: "freshdesk".into(),
            metadata: BTreeMap::new(),
        }];
        let state = Arc::new(AppState {
            cfg: Config::default(),
            chunks: chunks.clone(),
            flat: Flat::open(&embeddings_path, 2).unwrap(),
            bm25: Bm25::build(chunks.iter().map(|chunk| chunk.content.as_str())),
            embedder: Box::new(DummyEmbedder),
        });

        let _ = router(state, auth::Policy::Open);
    }
}
