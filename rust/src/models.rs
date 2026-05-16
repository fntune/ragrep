use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    pub id: String,
    pub source: String,
    pub content: String,
    pub title: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, MetaValue>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chunk {
    pub id: String,
    pub doc_id: String,
    pub content: String,
    pub title: String,
    pub source: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, MetaValue>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MetaValue {
    Str(String),
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, Serialize)]
pub struct SearchResult {
    pub chunk_id: String,
    pub content: String,
    pub title: String,
    pub source: String,
    pub metadata: BTreeMap<String, MetaValue>,
    #[serde(default)]
    pub dense_score: f32,
    #[serde(default)]
    pub bm25_score: f32,
    #[serde(default)]
    pub rrf_score: f32,
    #[serde(default)]
    pub rerank_score: f32,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryResult {
    pub query: String,
    pub answer: String,
    pub sources: Vec<SearchResult>,
    #[serde(default)]
    pub timings: BTreeMap<String, f64>,
}

#[derive(Debug, Default, Clone, Serialize)]
pub struct IngestStats {
    pub documents: usize,
    pub chunks: usize,
    pub sources: BTreeMap<String, usize>,
    pub elapsed_s: f64,
}
