//! Google Gemini embeddings — direct HTTP, no SDK.
//!
//! Mirrors `GeminiEmbedder` from `src/ragrep/ingest/embed.py:564`. Uses the
//! `batchEmbedContents` endpoint for the document path (up to 100 inputs
//! per call) and `embedContent` for one-shot queries.

use std::env;
use std::path::Path;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::throttle::Throttle;

const BASE: &str = "https://generativelanguage.googleapis.com/v1beta";
const ENV_KEY: &str = "GEMINI_API_KEY";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);
const PROVIDER: &str = "gemini";
const MAX_API_BATCH: usize = 100;
const MAX_ATTEMPTS: u32 = 5;

const TASK_QUERY: &str = "RETRIEVAL_QUERY";
const TASK_DOCUMENT: &str = "RETRIEVAL_DOCUMENT";

fn dim_for(model: &str) -> usize {
    match model {
        "gemini-embedding-001" | "text-embedding-004" => 3072,
        _ => 3072,
    }
}

pub struct Embedder {
    api_key: String,
    model: String,
    dim: usize,
}

impl Embedder {
    pub fn new(model: &str) -> Result<Self> {
        let api_key = env::var(ENV_KEY)
            .map_err(|_| anyhow!("{ENV_KEY} is not set (add to .env or export it)"))?;
        Ok(Self {
            api_key,
            model: model.to_string(),
            dim: dim_for(model),
        })
    }
}

impl super::Embedder for Embedder {
    fn provider(&self) -> &str {
        PROVIDER
    }
    fn model(&self) -> &str {
        &self.model
    }
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        let v = embed_one(&self.api_key, &self.model, text, TASK_QUERY)?;
        Ok(normalize(v))
    }

    fn embed_documents(
        &self,
        texts: &[&str],
        batch_size: usize,
        _checkpoint: Option<&Path>,
    ) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let api_batch = batch_size.clamp(1, MAX_API_BATCH);

        let mut all: Vec<Vec<f32>> = Vec::with_capacity(total);
        // Gemini's free tier has tighter limits than OpenAI; start moderate.
        let mut throttle = Throttle::new(2.0, 0.1, 60.0);
        let mut i = 0usize;

        while i < total {
            let end = (i + api_batch).min(total);
            let batch: Vec<&str> = texts[i..end].to_vec();

            if i > 0 {
                throttle.wait();
            }

            let mut attempt = 0u32;
            loop {
                attempt += 1;
                match embed_batch(&self.api_key, &self.model, &batch) {
                    Ok(vs) => {
                        all.extend(vs.into_iter().map(normalize));
                        throttle.on_success();
                        break;
                    }
                    Err(e) => {
                        let msg = format!("{e}");
                        if msg.contains("HTTP 429") || msg.contains("RESOURCE_EXHAUSTED") {
                            if attempt >= MAX_ATTEMPTS {
                                return Err(e).context(format!(
                                    "Gemini embed: rate-limit exhausted at {}/{}",
                                    all.len(),
                                    total
                                ));
                            }
                            throttle.on_rate_limit(None);
                            tracing::warn!(
                                target: "ragrep::embed",
                                "gemini rate-limited; sleeping {:.1}s (attempt {}/{})",
                                throttle.delay,
                                attempt,
                                MAX_ATTEMPTS
                            );
                            thread::sleep(Duration::from_secs_f32(throttle.delay));
                            continue;
                        }
                        return Err(e).context(format!(
                            "Gemini embed failed at {}/{}",
                            all.len(),
                            total
                        ));
                    }
                }
            }

            i = end;
        }

        Ok(all)
    }
}

#[derive(Serialize)]
struct Content<'a> {
    parts: Vec<Part<'a>>,
}

#[derive(Serialize)]
struct Part<'a> {
    text: &'a str,
}

#[derive(Serialize)]
struct EmbedOneRequest<'a> {
    model: String,
    content: Content<'a>,
    #[serde(rename = "taskType")]
    task_type: &'a str,
}

#[derive(Serialize)]
struct BatchRequest<'a> {
    requests: Vec<EmbedOneRequest<'a>>,
}

#[derive(Deserialize)]
struct EmbeddingValues {
    values: Vec<f32>,
}

#[derive(Deserialize)]
struct OneResponse {
    embedding: EmbeddingValues,
}

#[derive(Deserialize)]
struct BatchResponse {
    embeddings: Vec<EmbeddingValues>,
}

fn embed_one(api_key: &str, model: &str, text: &str, task_type: &str) -> Result<Vec<f32>> {
    let url = format!("{BASE}/models/{model}:embedContent?key={api_key}");
    let body = EmbedOneRequest {
        model: format!("models/{model}"),
        content: Content {
            parts: vec![Part { text }],
        },
        task_type,
    };
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building HTTP client")?;
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .context("sending Gemini embed request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().unwrap_or_default();
        bail!("Gemini embed failed: HTTP {status} — {txt}");
    }
    let payload: OneResponse = resp.json().context("parsing Gemini embed response")?;
    Ok(payload.embedding.values)
}

fn embed_batch(api_key: &str, model: &str, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let url = format!("{BASE}/models/{model}:batchEmbedContents?key={api_key}");
    let model_path = format!("models/{model}");
    let requests: Vec<EmbedOneRequest> = texts
        .iter()
        .map(|t| EmbedOneRequest {
            model: model_path.clone(),
            content: Content {
                parts: vec![Part { text: t }],
            },
            task_type: TASK_DOCUMENT,
        })
        .collect();
    let body = BatchRequest { requests };

    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building HTTP client")?;
    let resp = client
        .post(&url)
        .json(&body)
        .send()
        .context("sending Gemini batch embed request")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let txt = resp.text().unwrap_or_default();
        bail!("Gemini batch embed failed: HTTP {status} — {txt}");
    }
    let payload: BatchResponse = resp.json().context("parsing Gemini batch response")?;
    Ok(payload.embeddings.into_iter().map(|e| e.values).collect())
}

fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
