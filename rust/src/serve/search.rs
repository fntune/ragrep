//! `GET /search` handler.

use std::sync::Arc;

use anyhow::{bail, Context, Result};
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::AppState;
use crate::models::MetaValue;
use crate::query;
use crate::query::filters;

#[derive(Deserialize, Clone)]
pub struct SearchQuery {
    pub q: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_n")]
    pub n: usize,
    pub source: Option<String>,
    pub filter: Option<String>,
    pub after: Option<String>,
    pub before: Option<String>,
    #[serde(default = "default_context")]
    pub context: usize,
    #[serde(default)]
    pub full: bool,
    #[serde(default = "default_true")]
    pub scores: bool,
    #[serde(default)]
    pub metadata: bool,
}

fn default_mode() -> String {
    "grep".into()
}
fn default_n() -> usize {
    5
}
fn default_context() -> usize {
    300
}
fn default_true() -> bool {
    true
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Query(q): Query<SearchQuery>,
) -> Result<Json<serde_json::Value>, (StatusCode, Json<serde_json::Value>)> {
    let state2 = Arc::clone(&state);
    let q2 = q.clone();
    let result = tokio::task::spawn_blocking(move || do_search(&state2, q2))
        .await
        .map_err(|e| internal(format!("join: {e}")))?;
    match result {
        Ok(payload) => Ok(Json(payload)),
        Err(e) => Err(client_or_internal(e)),
    }
}

fn do_search(state: &AppState, q: SearchQuery) -> Result<serde_json::Value> {
    let output = execute(state, &q)?;
    let results = format_hits(&output.hits, &q);
    match output.total_matches {
        Some(total_matches) => Ok(json!({
            "query": output.query,
            "mode": output.mode,
            "total_matches": total_matches,
            "results": results,
        })),
        None => Ok(json!({
            "query": output.query,
            "mode": output.mode,
            "results": results,
        })),
    }
}

pub struct Output {
    pub query: String,
    pub mode: String,
    pub total_matches: Option<usize>,
    pub hits: Vec<query::Hit>,
}

pub fn execute(state: &AppState, q: &SearchQuery) -> Result<Output> {
    let filt_metadata = match &q.filter {
        Some(s) if !s.is_empty() => {
            let parts: Vec<String> = s.split(',').map(str::to_string).collect();
            filters::parse_filters(&parts)?
        }
        _ => Default::default(),
    };
    let after = q.after.as_deref().map(filters::parse_date).transpose()?;
    let before = q.before.as_deref().map(filters::parse_date).transpose()?;
    let filt = query::Filters {
        source: q.source.as_deref(),
        metadata: filt_metadata,
        after: after.as_deref(),
        before: before.as_deref(),
    };

    match q.mode.as_str() {
        "grep" => {
            let runtime = state.runtime()?;
            let r = query::grep(&runtime.chunks, &q.q, &filt, q.n);
            Ok(Output {
                query: r.query,
                mode: "grep".to_string(),
                total_matches: Some(r.total_matches),
                hits: r.hits,
            })
        }
        "semantic" => {
            if state.runtime()?.chunks.is_empty() {
                return Ok(empty_output(&q.q, "semantic", None));
            }
            let emb = state.embedder.embed_query(&q.q)?;
            let runtime = state.runtime()?;
            if runtime.chunks.is_empty() {
                return Ok(empty_output(&q.q, "semantic", None));
            }
            let flat = runtime
                .flat
                .as_ref()
                .context("runtime index missing embeddings")?;
            let r = query::semantic(flat, &runtime.chunks, &q.q, &emb, &filt, q.n);
            Ok(Output {
                query: r.query,
                mode: "semantic".to_string(),
                total_matches: None,
                hits: r.hits,
            })
        }
        "hybrid" => {
            if state.runtime()?.chunks.is_empty() {
                return Ok(empty_output(&q.q, "hybrid", None));
            }
            let emb = state.embedder.embed_query(&q.q)?;
            let runtime = state.runtime()?;
            if runtime.chunks.is_empty() {
                return Ok(empty_output(&q.q, "hybrid", None));
            }
            let flat = runtime
                .flat
                .as_ref()
                .context("runtime index missing embeddings")?;
            let cfg = &state.cfg;
            let opts = query::HybridOpts {
                n: q.n,
                top_k_dense: cfg.retrieval.top_k_dense,
                top_k_bm25: cfg.retrieval.top_k_bm25,
                rerank_pool: (q.n * 4).max(20),
                rrf_k: cfg.retrieval.rrf_k,
                rerank_provider: &cfg.reranker.provider,
                rerank_model: &cfg.reranker.model_name,
                filters: filt,
            };
            let r = query::hybrid(flat, &runtime.bm25, &runtime.chunks, &q.q, &emb, opts)?;
            Ok(Output {
                query: r.query,
                mode: "hybrid".to_string(),
                total_matches: None,
                hits: r.hits,
            })
        }
        m => bail!("invalid mode: {m}"),
    }
}

fn empty_output(query: &str, mode: &str, total_matches: Option<usize>) -> Output {
    Output {
        query: query.to_string(),
        mode: mode.to_string(),
        total_matches,
        hits: Vec::new(),
    }
}

#[derive(Serialize)]
struct OutHit<'a> {
    rank: usize,
    id: &'a str,
    source: &'a str,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    score: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rerank: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rrf: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bm25: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a std::collections::BTreeMap<String, MetaValue>>,
}

fn format_hits(hits: &[query::Hit], q: &SearchQuery) -> Vec<serde_json::Value> {
    hits.iter()
        .map(|h| {
            let r = &h.result;
            let title = truncate_title(&r.title);
            let (snippet_v, content_v) = if q.full {
                (None, Some(r.content.as_str()))
            } else if q.context > 0 {
                (Some(snippet(&r.content, q.context, &q.q)), None)
            } else {
                (None, None)
            };
            let out = OutHit {
                rank: h.rank,
                id: r.chunk_id.as_str(),
                source: r.source.as_str(),
                title,
                snippet: snippet_v,
                content: content_v,
                score: semantic_score(r.dense_score, q),
                rerank: hybrid_score(r.rerank_score, q),
                rrf: hybrid_score(r.rrf_score, q),
                dense: hybrid_score(r.dense_score, q),
                bm25: hybrid_score(r.bm25_score, q),
                metadata: if q.metadata { Some(&r.metadata) } else { None },
            };
            serde_json::to_value(&out).unwrap_or_else(|_| json!({}))
        })
        .collect()
}

fn truncate_title(title: &str) -> String {
    let chars: Vec<char> = title.chars().collect();
    if chars.len() <= 80 {
        title.to_string()
    } else {
        let mut s: String = chars[..77].iter().collect();
        s.push_str("...");
        s
    }
}

fn semantic_score(v: f32, q: &SearchQuery) -> Option<f64> {
    if q.scores && q.mode == "semantic" {
        Some(round_score(v))
    } else {
        None
    }
}

fn hybrid_score(v: f32, q: &SearchQuery) -> Option<f64> {
    if q.scores && q.mode == "hybrid" {
        Some(round_score(v))
    } else {
        None
    }
}

fn round_score(v: f32) -> f64 {
    ((v as f64) * 1000.0).round() / 1000.0
}

fn snippet(content: &str, length: usize, term: &str) -> String {
    let flat: String = content
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    let trimmed = flat.trim();
    if length == 0 || length >= trimmed.chars().count() {
        return trimmed.to_string();
    }
    let lower = trimmed.to_lowercase();
    if let Some(pos) = lower.find(&term.to_lowercase()) {
        let prefix_chars = trimmed[..pos].chars().count();
        let half_back = length / 4;
        let start = prefix_chars.saturating_sub(half_back);
        let chars: Vec<char> = trimmed.chars().collect();
        let end = (start + length).min(chars.len());
        let mut s: String = chars[start..end].iter().collect();
        if start > 0 {
            s.insert_str(0, "...");
        }
        if end < chars.len() {
            s.push_str("...");
        }
        return s;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mut s: String = chars[..length.min(chars.len())].iter().collect();
    if chars.len() > length {
        s.push_str("...");
    }
    s
}

fn internal(msg: String) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::INTERNAL_SERVER_ERROR,
        Json(json!({ "error": msg })),
    )
}

fn client_or_internal(e: anyhow::Error) -> (StatusCode, Json<serde_json::Value>) {
    let msg = format!("{e}");
    if msg.starts_with("invalid mode")
        || msg.contains("invalid filter")
        || msg.contains("invalid date")
    {
        (StatusCode::BAD_REQUEST, Json(json!({ "error": msg })))
    } else {
        internal(msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn query(mode: &str) -> SearchQuery {
        SearchQuery {
            q: "auth".to_string(),
            mode: mode.to_string(),
            n: 5,
            source: None,
            filter: None,
            after: None,
            before: None,
            context: 300,
            full: false,
            scores: true,
            metadata: false,
        }
    }

    #[test]
    fn semantic_scores_use_public_field_name() {
        let q = query("semantic");
        assert_eq!(semantic_score(0.12345, &q), Some(0.123));
        assert_eq!(hybrid_score(0.12345, &q), None);
    }

    #[test]
    fn hybrid_scores_include_zero_values() {
        let q = query("hybrid");
        assert_eq!(hybrid_score(0.0, &q), Some(0.0));
        assert_eq!(semantic_score(0.25, &q), None);
    }

    #[test]
    fn empty_output_has_no_hits() {
        let output = empty_output("kyc", "hybrid", None);

        assert_eq!(output.query, "kyc");
        assert_eq!(output.mode, "hybrid");
        assert_eq!(output.hits.len(), 0);
    }
}
