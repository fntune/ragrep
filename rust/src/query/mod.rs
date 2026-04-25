//! Public engine API: `grep`, `semantic`, `hybrid`.

pub mod generate;
pub mod rerank;
pub mod retrieve;

use crate::index::bm25::Bm25;
use crate::index::flat::Flat;
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

#[derive(Debug, Clone)]
pub struct SemanticResult {
    pub query: String,
    pub hits: Vec<Hit>,
}

/// Dense vector search via mmapped flat index. Returns top `n` hits with
/// `dense_score` populated. If `source` is set, fetches `n*3` then post-filters.
/// Mirrors `dense_search` in `src/ragrep/query/retrieve.py`.
pub fn semantic(
    flat: &Flat,
    chunks: &[Chunk],
    query: &str,
    query_embedding: &[f32],
    source: Option<&str>,
    n: usize,
) -> SemanticResult {
    let fetch_k = if source.is_some() { n * 3 } else { n };
    let raw = flat.search(query_embedding, fetch_k);

    let mut hits = Vec::with_capacity(n);
    for (idx, score) in raw {
        let chunk = &chunks[idx as usize];
        if let Some(s) = source {
            if chunk.source != s {
                continue;
            }
        }
        hits.push(Hit {
            rank: hits.len() + 1,
            result: SearchResult {
                chunk_id: chunk.id.clone(),
                content: chunk.content.clone(),
                title: chunk.title.clone(),
                source: chunk.source.clone(),
                metadata: chunk.metadata.clone(),
                dense_score: score,
                bm25_score: 0.0,
                rrf_score: 0.0,
                rerank_score: 0.0,
            },
        });
        if hits.len() >= n {
            break;
        }
    }

    SemanticResult {
        query: query.to_string(),
        hits,
    }
}

#[derive(Debug, Clone)]
pub struct HybridResult {
    pub query: String,
    pub hits: Vec<Hit>,
}

pub struct HybridOpts<'a> {
    pub n: usize,
    pub top_k_dense: usize,
    pub top_k_bm25: usize,
    /// Candidates passed to the reranker. Mirrors Python's
    /// `fetch_n = max(n*4, 20)` from `search.py::search_hybrid`.
    pub rerank_pool: usize,
    pub rrf_k: usize,
    pub rerank_provider: &'a str,
    pub rerank_model: &'a str,
    pub source: Option<&'a str>,
}

/// Hybrid retrieval: dense + BM25 → RRF fusion → rerank top-`n`.
/// Mirrors `retrieve()` + Voyage rerank from `src/ragrep/search.py`.
pub fn hybrid(
    flat: &Flat,
    bm25_idx: &Bm25,
    chunks: &[Chunk],
    query: &str,
    query_embedding: &[f32],
    opts: HybridOpts<'_>,
) -> anyhow::Result<HybridResult> {
    let dense_hits = retrieve::dense(flat, query_embedding, chunks, opts.top_k_dense, opts.source);
    let bm25_hits = retrieve::bm25(bm25_idx, chunks, query, opts.top_k_bm25, opts.source);
    let fused = retrieve::rrf(&dense_hits, &bm25_hits, opts.rrf_k);

    let pool_n = opts.rerank_pool.min(fused.len());
    let pre = &fused[..pool_n];
    let docs: Vec<&str> = pre
        .iter()
        .map(|(idx, _, _, _)| chunks[*idx].content.as_str())
        .collect();

    let rerank_items = rerank::rerank(
        opts.rerank_provider,
        opts.rerank_model,
        query,
        &docs,
        opts.n,
    )?;

    let mut hits = Vec::with_capacity(rerank_items.len());
    for (rank, item) in rerank_items.into_iter().enumerate() {
        let (idx, dense, bm25_s, rrf_s) = pre[item.index];
        let chunk = &chunks[idx];
        hits.push(Hit {
            rank: rank + 1,
            result: SearchResult {
                chunk_id: chunk.id.clone(),
                content: chunk.content.clone(),
                title: chunk.title.clone(),
                source: chunk.source.clone(),
                metadata: chunk.metadata.clone(),
                dense_score: dense,
                bm25_score: bm25_s,
                rrf_score: rrf_s,
                rerank_score: item.score,
            },
        });
    }

    Ok(HybridResult {
        query: query.to_string(),
        hits,
    })
}
