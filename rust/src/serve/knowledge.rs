//! Support-oriented knowledge search contract.

use std::sync::Arc;

use anyhow::Result;
use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::{search, AppState};
use crate::models::{MetaValue, SearchResult};
use crate::query;

#[derive(Deserialize, Clone)]
pub struct KnowledgeQuery {
    pub q: String,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default = "default_article_n")]
    pub article_n: usize,
    #[serde(default = "default_video_n")]
    pub video_n: usize,
    #[serde(default = "default_article_source")]
    pub article_source: String,
    #[serde(default = "default_video_source")]
    pub video_source: String,
    pub portal_id: Option<String>,
    pub folder_id: Option<String>,
    pub folder_name: Option<String>,
    pub article_filter: Option<String>,
    pub article_base_url: Option<String>,
}

fn default_mode() -> String {
    "hybrid".into()
}
fn default_article_n() -> usize {
    3
}
fn default_video_n() -> usize {
    1
}
fn default_article_source() -> String {
    "freshdesk".into()
}
fn default_video_source() -> String {
    "youtube".into()
}

#[derive(Serialize)]
pub struct KnowledgeResponse {
    pub query: String,
    pub knowledges: Vec<Article>,
    pub videos: Vec<Video>,
    pub youtube_search: YoutubeSearch,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub message: Option<&'static str>,
}

#[derive(Debug, Serialize)]
pub struct Article {
    pub id: String,
    pub title: String,
    pub content: String,
    pub link: String,
    pub updated_at: String,
    pub score: f64,
}

#[derive(Debug, Serialize)]
pub struct Video {
    pub video_id: String,
    pub title: String,
    pub description: String,
    pub video_url: String,
    pub thumbnail_url: String,
    pub score: f64,
}

#[derive(Serialize)]
pub struct YoutubeSearch {
    pub status: &'static str,
    pub count: usize,
    pub top_score: Option<f64>,
}

pub async fn handle(
    State(state): State<Arc<AppState>>,
    Query(q): Query<KnowledgeQuery>,
) -> Result<Json<KnowledgeResponse>, (StatusCode, Json<serde_json::Value>)> {
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

fn do_search(state: &AppState, q: KnowledgeQuery) -> Result<KnowledgeResponse> {
    let article_hits = if q.article_n == 0 {
        Vec::new()
    } else {
        let request = search_query(
            &q.q,
            &q.mode,
            q.article_n,
            &q.article_source,
            article_filter(&q),
        );
        search::execute(state, &request)?.hits
    };

    let video_hits = if q.video_n == 0 {
        Vec::new()
    } else {
        let request = search_query(&q.q, &q.mode, q.video_n, &q.video_source, None);
        search::execute(state, &request)?.hits
    };

    let knowledges: Vec<Article> = article_hits
        .iter()
        .map(|hit| article(hit, q.article_base_url.as_deref()))
        .collect();
    let videos: Vec<Video> = video_hits.iter().map(video).collect();
    let top_score = videos.first().map(|video| video.score);
    let message = if knowledges.is_empty() && videos.is_empty() {
        Some("No details found")
    } else {
        None
    };

    Ok(KnowledgeResponse {
        query: q.q,
        knowledges,
        videos,
        youtube_search: YoutubeSearch {
            status: "ok",
            count: video_hits.len(),
            top_score,
        },
        message,
    })
}

fn search_query(
    q: &str,
    mode: &str,
    n: usize,
    source: &str,
    filter: Option<String>,
) -> search::SearchQuery {
    search::SearchQuery {
        q: q.to_string(),
        mode: mode.to_string(),
        n,
        source: Some(source.to_string()),
        filter,
        after: None,
        before: None,
        context: 0,
        full: true,
        scores: true,
        metadata: true,
    }
}

fn article_filter(q: &KnowledgeQuery) -> Option<String> {
    let mut parts = Vec::new();
    if let Some(portal_id) = q.portal_id.as_deref().filter(|value| !value.is_empty()) {
        parts.push(format!("portal_id={portal_id}"));
    }
    if let Some(folder_id) = q.folder_id.as_deref().filter(|value| !value.is_empty()) {
        parts.push(format!("folder_id={folder_id}"));
    } else if let Some(folder_name) = q.folder_name.as_deref().filter(|value| !value.is_empty()) {
        parts.push(format!("folder_name={folder_name}"));
    }
    if let Some(filter) = q
        .article_filter
        .as_deref()
        .filter(|value| !value.is_empty())
    {
        parts.push(filter.to_string());
    }
    (!parts.is_empty()).then(|| parts.join(","))
}

fn article(hit: &query::Hit, article_base_url: Option<&str>) -> Article {
    let result = &hit.result;
    let id = record_id(&result.chunk_id);
    Article {
        id: id.clone(),
        title: result.title.clone(),
        content: result.content.clone(),
        link: meta_str(result, "link")
            .or_else(|| meta_str(result, "url"))
            .unwrap_or_else(|| article_link(article_base_url, &id)),
        updated_at: meta_str(result, "updated_at").unwrap_or_default(),
        score: score(result),
    }
}

fn video(hit: &query::Hit) -> Video {
    let result = &hit.result;
    let video_id = record_id(&result.chunk_id);
    Video {
        video_id: video_id.clone(),
        title: meta_str(result, "title").unwrap_or_else(|| result.title.clone()),
        description: meta_str(result, "description")
            .unwrap_or_else(|| truncate_description(&result.content)),
        video_url: meta_str(result, "video_url").unwrap_or_default(),
        thumbnail_url: meta_str(result, "thumbnail_url").unwrap_or_default(),
        score: score(result),
    }
}

fn record_id(chunk_id: &str) -> String {
    chunk_id
        .rsplit_once(':')
        .map(|(record_id, _)| record_id)
        .unwrap_or(chunk_id)
        .to_string()
}

fn meta_str(result: &SearchResult, key: &str) -> Option<String> {
    match result.metadata.get(key) {
        Some(MetaValue::Str(value)) if !value.is_empty() => Some(value.clone()),
        Some(MetaValue::Int(value)) => Some(value.to_string()),
        Some(MetaValue::Float(value)) => Some(value.to_string()),
        _ => None,
    }
}

fn article_link(base_url: Option<&str>, id: &str) -> String {
    base_url
        .filter(|base| !base.is_empty())
        .map(|base| format!("{base}{id}"))
        .unwrap_or_default()
}

fn score(result: &SearchResult) -> f64 {
    let value = if result.rerank_score != 0.0 {
        result.rerank_score
    } else if result.rrf_score != 0.0 {
        result.rrf_score
    } else if result.dense_score != 0.0 {
        result.dense_score
    } else {
        result.bm25_score
    };
    ((value as f64) * 1000.0).round() / 1000.0
}

fn truncate_description(content: &str) -> String {
    content.chars().take(200).collect()
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
    use std::collections::BTreeMap;

    use super::*;

    fn hit(chunk_id: &str, source: &str, metadata: BTreeMap<String, MetaValue>) -> query::Hit {
        query::Hit {
            rank: 1,
            result: SearchResult {
                chunk_id: chunk_id.to_string(),
                content: "content body".to_string(),
                title: "Result Title".to_string(),
                source: source.to_string(),
                metadata,
                dense_score: 0.81234,
                bm25_score: 0.0,
                rrf_score: 0.0,
                rerank_score: 0.0,
            },
        }
    }

    #[test]
    fn article_uses_record_id_and_support_fields() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "updated_at".to_string(),
            MetaValue::Str("2026-05-16".to_string()),
        );
        let hit = hit("123:0", "freshdesk", metadata);

        let article = article(&hit, Some("https://support.example/articles/"));

        assert_eq!(article.id, "123");
        assert_eq!(article.title, "Result Title");
        assert_eq!(article.link, "https://support.example/articles/123");
        assert_eq!(article.updated_at, "2026-05-16");
        assert_eq!(article.score, 0.812);
    }

    #[test]
    fn video_uses_metadata_contract() {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "title".to_string(),
            MetaValue::Str("Video Title".to_string()),
        );
        metadata.insert(
            "description".to_string(),
            MetaValue::Str("Video description".to_string()),
        );
        metadata.insert(
            "video_url".to_string(),
            MetaValue::Str("https://youtu.be/v1".to_string()),
        );
        metadata.insert(
            "thumbnail_url".to_string(),
            MetaValue::Str("https://img.example/v1.jpg".to_string()),
        );
        let hit = hit("video-1:0", "youtube", metadata);

        let video = video(&hit);

        assert_eq!(video.video_id, "video-1");
        assert_eq!(video.title, "Video Title");
        assert_eq!(video.description, "Video description");
        assert_eq!(video.video_url, "https://youtu.be/v1");
        assert_eq!(video.thumbnail_url, "https://img.example/v1.jpg");
    }
}
