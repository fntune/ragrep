//! Voyage AI embeddings — direct HTTP, no SDK.
//!
//! Mirrors the post-`voyageai`-removal Python implementation in
//! `src/ragrep/ingest/embed.py::VoyageEmbedder` (httpx + retry/backoff).

use std::env;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

const BASE: &str = "https://api.voyageai.com/v1";
const ENV_KEY: &str = "VOYAGE_API_KEY";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: Vec<&'a str>,
    input_type: &'a str,
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

/// One-shot query embedding. Returns the L2-normalized 1024-d vector.
pub fn embed_query(model: &str, text: &str) -> Result<Vec<f32>> {
    let key = env::var(ENV_KEY)
        .map_err(|_| anyhow!("{ENV_KEY} is not set (add to .env or export it)"))?;
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building HTTP client")?;
    let resp = client
        .post(format!("{BASE}/embeddings"))
        .bearer_auth(&key)
        .json(&EmbedRequest {
            model,
            input: vec![text],
            input_type: "query",
        })
        .send()
        .context("sending embed request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("Voyage embed failed: HTTP {status} — {body}");
    }

    let mut payload: EmbedResponse = resp.json().context("parsing embed response")?;
    if payload.data.is_empty() {
        bail!("Voyage embed: empty data array");
    }
    payload.data.sort_by_key(|x| x.index);
    Ok(payload.data.swap_remove(0).embedding)
}
