//! Embedding providers (HTTP only). Pipeline holds `Box<dyn Embedder>`;
//! switching `cfg.embedding.{provider, model_name}` swaps the concrete impl.

use std::path::Path;

use anyhow::{bail, Result};

pub mod cache;
pub mod gemini;
pub mod openai;
pub mod throttle;
pub mod voyage;

pub trait Embedder: Send + Sync {
    fn provider(&self) -> &str;
    fn model(&self) -> &str;
    fn dim(&self) -> usize;

    /// Embed a single query string. Result is L2-normalized.
    fn embed_query(&self, text: &str) -> Result<Vec<f32>>;

    /// Embed many document texts in batches. May checkpoint to `checkpoint`
    /// path every 500 batches and resume from it on restart.
    fn embed_documents(
        &self,
        texts: &[&str],
        batch_size: usize,
        checkpoint: Option<&Path>,
    ) -> Result<Vec<Vec<f32>>>;
}

/// Factory: route by provider string. Returns a boxed trait object so callers
/// can hold it without naming the concrete type.
pub fn make(provider: &str, model: &str) -> Result<Box<dyn Embedder>> {
    match provider {
        "voyage" => Ok(Box::new(voyage::Embedder::new(model)?)),
        "openai" => Ok(Box::new(openai::Embedder::new(model)?)),
        "gemini" => Ok(Box::new(gemini::Embedder::new(model)?)),
        other => bail!("embedding provider '{other}' not yet supported in the Rust port"),
    }
}

/// Backwards-compat free function used by `cli::search::run`. Builds a
/// throwaway embedder and runs one query.
pub fn embed_query(provider: &str, model: &str, text: &str) -> Result<Vec<f32>> {
    make(provider, model)?.embed_query(text)
}
