//! OpenAI embeddings — direct HTTP, no SDK.
//!
//! Mirrors `OpenAIEmbedder` from `src/ragrep/ingest/embed.py:393`.

use std::env;
use std::path::Path;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::throttle::Throttle;

const BASE: &str = "https://api.openai.com/v1";
const ENV_KEY: &str = "OPENAI_API_KEY";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const PROVIDER: &str = "openai";
/// OpenAI's `/v1/embeddings` accepts up to 2048 inputs. We cap at 64 to mirror
/// the Python implementation; smaller batches mean lower request bursts under
/// the same total throughput.
const MAX_API_BATCH: usize = 64;
const MAX_ATTEMPTS: u32 = 5;

/// Native dim per OpenAI model. `text-embedding-3-large` defaults to 3072
/// but supports the `dimensions` request param to truncate (we don't use
/// that; we pass the native dim through to the index).
fn dim_for(model: &str) -> usize {
    match model {
        "text-embedding-3-small" => 1536,
        "text-embedding-3-large" => 3072,
        "text-embedding-ada-002" => 1536,
        _ => 1536,
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
        let mut out = embed_call(&self.api_key, &self.model, &[text])?;
        Ok(normalize(out.swap_remove(0)))
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
        // OpenAI's normal tier has generous rate limits; start fast.
        let mut throttle = Throttle::new(0.5, 0.05, 30.0);
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
                match embed_call(&self.api_key, &self.model, &batch) {
                    Ok(vs) => {
                        all.extend(vs.into_iter().map(normalize));
                        throttle.on_success();
                        break;
                    }
                    Err(e) => {
                        let msg = format!("{e}");
                        if msg.contains("HTTP 429") || msg.contains("HTTP 503") {
                            if attempt >= MAX_ATTEMPTS {
                                return Err(e).context(format!(
                                    "OpenAI embed: rate-limit exhausted at {}/{}",
                                    all.len(),
                                    total
                                ));
                            }
                            throttle.on_rate_limit(None);
                            tracing::warn!(
                                target: "ragrep::embed",
                                "openai rate-limited; sleeping {:.1}s (attempt {}/{})",
                                throttle.delay,
                                attempt,
                                MAX_ATTEMPTS
                            );
                            thread::sleep(Duration::from_secs_f32(throttle.delay));
                            continue;
                        }
                        return Err(e).context(format!(
                            "OpenAI embed failed at {}/{}",
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
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
}

#[derive(Deserialize)]
struct EmbedResponse {
    data: Vec<EmbedItem>,
}

#[derive(Deserialize)]
struct EmbedItem {
    embedding: Vec<f32>,
    index: usize,
}

fn embed_call(api_key: &str, model: &str, inputs: &[&str]) -> Result<Vec<Vec<f32>>> {
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building HTTP client")?;
    let resp = client
        .post(format!("{BASE}/embeddings"))
        .bearer_auth(api_key)
        .json(&EmbedRequest {
            model,
            input: inputs,
        })
        .send()
        .context("sending embed request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("OpenAI embed failed: HTTP {status} — {body}");
    }
    let mut payload: EmbedResponse = resp.json().context("parsing embed response")?;
    if payload.data.is_empty() {
        bail!("OpenAI embed: empty data array");
    }
    payload.data.sort_by_key(|x| x.index);
    Ok(payload.data.into_iter().map(|x| x.embedding).collect())
}

/// L2-normalize. OpenAI returns un-normalized vectors; the rest of the
/// pipeline assumes ‖v‖ = 1 (cosine via inner product on a flat IP index).
fn normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
