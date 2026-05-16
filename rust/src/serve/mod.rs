//! HTTP search service.

pub mod auth;
pub mod knowledge;
pub mod search;

use std::sync::{Arc, RwLock, RwLockReadGuard};

use anyhow::{anyhow, bail, Result};
use axum::{extract::State, routing::get, Json, Router};
use serde::Serialize;
use serde_json::json;
use tower_http::trace::TraceLayer;

use crate::config::Config;
use crate::embed::{self, Embedder};
use crate::index::{bm25::Bm25, flat::Flat, store};

pub struct AppState {
    pub cfg: Config,
    runtime: RwLock<RuntimeState>,
    pub embedder: Box<dyn Embedder>,
}

pub struct RuntimeState {
    pub chunks: Vec<crate::models::Chunk>,
    pub flat: Option<Flat>,
    pub bm25: Bm25,
    pub generation: Option<store::RuntimeGeneration>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RuntimeSummary {
    pub chunks: usize,
    pub generation: Option<store::RuntimeGeneration>,
}

impl AppState {
    pub fn load(cfg: Config) -> Result<Self> {
        let dir = cfg.index_dir();
        let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?;
        let runtime = store::load_runtime(&dir, embedder.dim())?;
        Ok(Self {
            cfg,
            runtime: RwLock::new(RuntimeState::from_index(runtime)),
            embedder,
        })
    }

    pub fn runtime(&self) -> Result<RwLockReadGuard<'_, RuntimeState>> {
        self.runtime
            .read()
            .map_err(|_| anyhow!("runtime index lock poisoned"))
    }

    pub fn reload_runtime(&self) -> Result<RuntimeSummary> {
        let next = RuntimeState::load_or_empty(&self.cfg, self.embedder.dim())?;
        let summary = next.summary();
        *self
            .runtime
            .write()
            .map_err(|_| anyhow!("runtime index lock poisoned"))? = next;
        Ok(summary)
    }

    pub fn runtime_summary(&self) -> Result<RuntimeSummary> {
        Ok(self.runtime()?.summary())
    }
}

impl RuntimeState {
    fn from_index(index: store::RuntimeIndex) -> Self {
        Self {
            chunks: index.chunks,
            flat: Some(index.flat),
            bm25: index.bm25,
            generation: Some(index.generation),
        }
    }

    fn load_or_empty(cfg: &Config, dim: usize) -> Result<Self> {
        let index_dir = cfg.index_dir();
        match (
            store::chunks_exist(&index_dir),
            store::embeddings_exist(&index_dir),
            store::bm25_exists(&index_dir),
        ) {
            (true, true, true) => Ok(Self::from_index(store::load_runtime(&index_dir, dim)?)),
            (false, false, false) => Ok(Self::empty()),
            _ => bail!("incomplete runtime index in {}", index_dir.display()),
        }
    }

    fn empty() -> Self {
        Self {
            chunks: Vec::new(),
            flat: None,
            bm25: Bm25::build(std::iter::empty::<&str>()),
            generation: None,
        }
    }

    fn summary(&self) -> RuntimeSummary {
        RuntimeSummary {
            chunks: self.chunks.len(),
            generation: self.generation.clone(),
        }
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
            .route(
                "/knowledge/reload",
                get(knowledge::reload).post(knowledge::reload),
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
    let runtime = state.runtime_summary().ok();
    Json(json!({
        "status": "ok",
        "chunks": runtime.as_ref().map(|summary| summary.chunks).unwrap_or_default(),
        "generation": runtime.and_then(|summary| summary.generation),
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

    struct PanickingEmbedder;

    impl Embedder for PanickingEmbedder {
        fn provider(&self) -> &str {
            "panic"
        }

        fn model(&self) -> &str {
            "panic"
        }

        fn dim(&self) -> usize {
            2
        }

        fn embed_query(&self, _text: &str) -> Result<Vec<f32>> {
            panic!("empty runtimes must not call the embedder")
        }

        fn embed_documents(
            &self,
            _texts: &[&str],
            _batch_size: usize,
            _checkpoint: Option<&std::path::Path>,
        ) -> Result<Vec<Vec<f32>>> {
            panic!("empty runtimes must not call the embedder")
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
            runtime: RwLock::new(RuntimeState {
                chunks: chunks.clone(),
                flat: Some(Flat::open(&embeddings_path, 2).unwrap()),
                bm25: Bm25::build(chunks.iter().map(|chunk| chunk.content.as_str())),
                generation: None,
            }),
            embedder: Box::new(DummyEmbedder),
        });

        let _ = router(state, auth::Policy::Open);
    }

    #[test]
    fn reload_runtime_replaces_loaded_index() {
        let dir = tempfile::tempdir().unwrap();
        let old_chunks = vec![chunk("freshdesk:old:0", "old content")];
        write_index(dir.path(), &old_chunks);
        let runtime = store::load_runtime(dir.path(), 2).unwrap();
        let old_generation = runtime.generation.clone();
        let mut cfg = Config::default();
        cfg.data.index_dir = dir.path().display().to_string();
        let state = AppState {
            cfg,
            runtime: RwLock::new(RuntimeState::from_index(runtime)),
            embedder: Box::new(DummyEmbedder),
        };

        let new_chunks = vec![
            chunk("freshdesk:new:0", "new content"),
            chunk("freshdesk:other:0", "other content"),
        ];
        write_index(dir.path(), &new_chunks);
        let summary = state.reload_runtime().unwrap();

        assert_eq!(summary.chunks, 2);
        assert_ne!(summary.generation, Some(old_generation));
        assert_eq!(state.runtime_summary().unwrap().chunks, 2);
    }

    #[test]
    fn reload_runtime_can_clear_loaded_index() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![chunk("freshdesk:old:0", "old content")];
        write_index(dir.path(), &chunks);
        let mut cfg = Config::default();
        cfg.data.index_dir = dir.path().display().to_string();
        let state = AppState {
            cfg,
            runtime: RwLock::new(RuntimeState::from_index(
                store::load_runtime(dir.path(), 2).unwrap(),
            )),
            embedder: Box::new(DummyEmbedder),
        };

        store::clear_index(dir.path()).unwrap();
        let summary = state.reload_runtime().unwrap();

        assert_eq!(summary.chunks, 0);
        assert_eq!(summary.generation, None);
        assert_eq!(state.runtime_summary().unwrap().chunks, 0);
    }

    #[test]
    fn reload_runtime_rejects_incomplete_index() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![chunk("freshdesk:old:0", "old content")];
        write_index(dir.path(), &chunks);
        let mut cfg = Config::default();
        cfg.data.index_dir = dir.path().display().to_string();
        let state = AppState {
            cfg,
            runtime: RwLock::new(RuntimeState::from_index(
                store::load_runtime(dir.path(), 2).unwrap(),
            )),
            embedder: Box::new(DummyEmbedder),
        };
        std::fs::remove_file(store::embeddings_path(dir.path())).unwrap();

        let err = state.reload_runtime().unwrap_err();

        assert!(err.to_string().contains("incomplete runtime index"));
        assert_eq!(state.runtime_summary().unwrap().chunks, 1);
    }

    #[test]
    fn empty_runtime_search_returns_empty_without_embedding() {
        let state = AppState {
            cfg: Config::default(),
            runtime: RwLock::new(RuntimeState::empty()),
            embedder: Box::new(PanickingEmbedder),
        };
        let q = search::SearchQuery {
            q: "kyc".into(),
            mode: "semantic".into(),
            n: 3,
            source: None,
            filter: None,
            after: None,
            before: None,
            context: 0,
            full: true,
            scores: true,
            metadata: false,
        };

        let output = search::execute(&state, &q).unwrap();

        assert_eq!(output.hits.len(), 0);
    }

    fn chunk(id: &str, content: &str) -> Chunk {
        Chunk {
            id: id.into(),
            doc_id: id
                .rsplit_once(':')
                .map(|(doc_id, _)| doc_id)
                .unwrap_or(id)
                .into(),
            content: content.into(),
            title: id.into(),
            source: "freshdesk".into(),
            metadata: BTreeMap::new(),
        }
    }

    fn write_index(dir: &std::path::Path, chunks: &[Chunk]) {
        let bm25 = Bm25::build(chunks.iter().map(|chunk| chunk.content.as_str()));
        let embeddings = chunks.iter().map(|_| vec![1.0, 0.0]).collect::<Vec<_>>();
        store::save_index(dir, chunks, &bm25, &embeddings, 2).unwrap();
    }
}
