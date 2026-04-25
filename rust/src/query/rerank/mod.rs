pub mod voyage;

#[derive(Debug, Clone)]
pub struct Item {
    pub index: usize,
    pub score: f32,
}

/// Provider-agnostic rerank: route by provider string.
pub fn rerank(
    provider: &str,
    model: &str,
    query: &str,
    docs: &[&str],
    top_k: usize,
) -> anyhow::Result<Vec<Item>> {
    match provider {
        "voyage" => voyage::rerank(model, query, docs, top_k),
        other => anyhow::bail!("rerank provider '{other}' not yet supported in the Rust port"),
    }
}
