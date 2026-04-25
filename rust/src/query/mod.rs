//! Public engine API: `grep`, `semantic`, `hybrid`.

pub mod generate;
pub mod rerank;
pub mod retrieve;

use crate::models::{Chunk, SearchResult};

#[derive(Debug, Clone)]
pub struct Hit {
    pub rank: usize,
    pub result: SearchResult,
}

#[derive(Debug, Clone)]
pub struct GrepResult {
    pub query: String,
    pub total_matches: usize,
    pub hits: Vec<Hit>,
}

/// Case-insensitive substring search over chunks. Returns the top `n` matches
/// plus the total match count. Mirrors `search_grep` in `src/ragrep/search.py`.
pub fn grep(chunks: &[Chunk], term: &str, source: Option<&str>, n: usize) -> GrepResult {
    let needle = term.to_lowercase();
    let filtered = chunks
        .iter()
        .filter(|c| source.is_none_or(|s| c.source == s))
        .filter(|c| c.content.to_lowercase().contains(&needle));

    let mut total = 0usize;
    let mut hits = Vec::with_capacity(n);
    for c in filtered {
        total += 1;
        if hits.len() < n {
            hits.push(Hit {
                rank: hits.len() + 1,
                result: SearchResult {
                    chunk_id: c.id.clone(),
                    content: c.content.clone(),
                    title: c.title.clone(),
                    source: c.source.clone(),
                    metadata: c.metadata.clone(),
                    dense_score: 0.0,
                    bm25_score: 0.0,
                    rrf_score: 0.0,
                    rerank_score: 0.0,
                },
            });
        }
    }

    GrepResult {
        query: term.to_string(),
        total_matches: total,
        hits,
    }
}
