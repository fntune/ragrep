//! Voyage AI rerank — direct HTTP, no SDK.

use std::env;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::Item;

const BASE: &str = "https://api.voyageai.com/v1";
const ENV_KEY: &str = "VOYAGE_API_KEY";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Serialize)]
struct Request<'a> {
    model: &'a str,
    query: &'a str,
    documents: &'a [&'a str],
    top_k: usize,
}

#[derive(Deserialize)]
struct Response {
    data: Vec<Entry>,
}

#[derive(Deserialize)]
struct Entry {
    index: usize,
    relevance_score: f32,
}

pub fn rerank(model: &str, query: &str, docs: &[&str], top_k: usize) -> Result<Vec<Item>> {
    if docs.is_empty() {
        return Ok(Vec::new());
    }
    let key = env::var(ENV_KEY)
        .map_err(|_| anyhow!("{ENV_KEY} is not set (add to .env or export it)"))?;
    let client = reqwest::blocking::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .context("building HTTP client")?;

    let resp = client
        .post(format!("{BASE}/rerank"))
        .bearer_auth(&key)
        .json(&Request {
            model,
            query,
            documents: docs,
            top_k,
        })
        .send()
        .context("sending rerank request")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("Voyage rerank failed: HTTP {status} — {body}");
    }
    let payload: Response = resp.json().context("parsing rerank response")?;
    Ok(payload
        .data
        .into_iter()
        .map(|e| Item {
            index: e.index,
            score: e.relevance_score,
        })
        .collect())
}
