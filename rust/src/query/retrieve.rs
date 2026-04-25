//! Hybrid retrieval primitives: dense + BM25 + RRF fusion.
//!
//! Mirrors `src/ragrep/query/retrieve.py`.

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};

use crate::index::bm25::Bm25;
use crate::index::flat::Flat;
use crate::models::Chunk;
use crate::query::Filters;

pub type Scored = (usize, f32);

pub fn dense(
    flat: &Flat,
    query_emb: &[f32],
    chunks: &[Chunk],
    top_k: usize,
    filt: &Filters<'_>,
) -> Vec<Scored> {
    let has_filter = filt.source.is_some()
        || !filt.metadata.is_empty()
        || filt.after.is_some()
        || filt.before.is_some();
    let fetch_k = if has_filter { top_k * 5 } else { top_k };
    flat.search(query_emb, fetch_k)
        .into_iter()
        .filter(|(idx, _)| filt.matches(&chunks[*idx as usize]))
        .take(top_k)
        .map(|(idx, score)| (idx as usize, score))
        .collect()
}

pub fn bm25(
    bm25: &Bm25,
    chunks: &[Chunk],
    query: &str,
    top_k: usize,
    filt: &Filters<'_>,
) -> Vec<Scored> {
    let mut scores = bm25.scores(query);
    let has_filter = filt.source.is_some()
        || !filt.metadata.is_empty()
        || filt.after.is_some()
        || filt.before.is_some();
    if has_filter {
        for (i, c) in chunks.iter().enumerate() {
            if !filt.matches(c) {
                scores[i] = 0.0;
            }
        }
    }
    let mut indexed: Vec<Scored> = scores
        .into_iter()
        .enumerate()
        .filter(|(_, s)| *s > 0.0)
        .collect();
    let k = top_k.min(indexed.len());
    if k == 0 {
        return Vec::new();
    }
    indexed.select_nth_unstable_by(k - 1, |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal)
    });
    indexed.truncate(k);
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
    indexed
}

/// (chunk_idx, dense_score, bm25_score, rrf_score) sorted by rrf_score desc.
pub type Fused = (usize, f32, f32, f32);

const MISSING_RANK: usize = 1000;

pub fn rrf(dense_results: &[Scored], bm25_results: &[Scored], k: usize) -> Vec<Fused> {
    let dense_ranks: HashMap<usize, usize> = dense_results
        .iter()
        .enumerate()
        .map(|(rank, (idx, _))| (*idx, rank))
        .collect();
    let bm25_ranks: HashMap<usize, usize> = bm25_results
        .iter()
        .enumerate()
        .map(|(rank, (idx, _))| (*idx, rank))
        .collect();
    let dense_scores: HashMap<usize, f32> = dense_results.iter().copied().collect();
    let bm25_scores: HashMap<usize, f32> = bm25_results.iter().copied().collect();

    let all: HashSet<usize> = dense_ranks
        .keys()
        .chain(bm25_ranks.keys())
        .copied()
        .collect();

    let kf = k as f32;
    let mut scored: Vec<Fused> = all
        .into_iter()
        .map(|idx| {
            let d_rank = dense_ranks.get(&idx).copied().unwrap_or(MISSING_RANK);
            let b_rank = bm25_ranks.get(&idx).copied().unwrap_or(MISSING_RANK);
            let rrf_score = 1.0 / (kf + d_rank as f32) + 1.0 / (kf + b_rank as f32);
            let d = dense_scores.get(&idx).copied().unwrap_or(0.0);
            let b = bm25_scores.get(&idx).copied().unwrap_or(0.0);
            (idx, d, b, rrf_score)
        })
        .collect();

    scored.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap_or(Ordering::Equal));
    scored
}
