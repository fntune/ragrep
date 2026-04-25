//! Voyage AI embeddings — direct HTTP, no SDK.
//!
//! Mirrors the post-`voyageai`-removal Python implementation in
//! `src/ragrep/ingest/embed.py::VoyageEmbedder` (httpx + retry/backoff).

use std::env;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;
use std::thread;
use std::time::Duration;

use anyhow::{anyhow, bail, Context, Result};
use serde::{Deserialize, Serialize};

use super::throttle::Throttle;

const BASE: &str = "https://api.voyageai.com/v1";
const ENV_KEY: &str = "VOYAGE_API_KEY";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(30);
const PROVIDER: &str = "voyage";
/// Cap per-API-call batch to stay under free-tier 10K TPM.
const MAX_API_BATCH: usize = 16;
/// Checkpoint every N API batches.
const CHECKPOINT_EVERY: usize = 500;
/// Per-batch retry budget on transient errors (rate-limit etc.).
const MAX_ATTEMPTS: u32 = 5;

/// Voyage embed dimensions per model. All current Voyage embed models output 1024 dims.
fn dim_for(model: &str) -> usize {
    match model {
        "voyage-code-3"
        | "voyage-3"
        | "voyage-3-large"
        | "voyage-3-lite"
        | "voyage-multilingual-2"
        | "voyage-large-2-instruct"
        | "voyage-finance-2"
        | "voyage-law-2" => 1024,
        _ => 1024,
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
        let mut out = embed_call(&self.api_key, &self.model, &[text], "query")?;
        Ok(out.swap_remove(0))
    }

    fn embed_documents(
        &self,
        texts: &[&str],
        batch_size: usize,
        checkpoint: Option<&Path>,
    ) -> Result<Vec<Vec<f32>>> {
        let total = texts.len();
        let api_batch = batch_size.clamp(1, MAX_API_BATCH);

        let mut all: Vec<Vec<f32>> = Vec::with_capacity(total);
        let mut start = 0usize;
        if let Some(path) = checkpoint {
            if let Some(resumed) = read_checkpoint(path, total, self.dim)? {
                tracing::info!(target: "ragrep::embed", "resuming Voyage embed from checkpoint: {}/{}", resumed.len(), total);
                start = resumed.len();
                all = resumed;
            }
        }

        if start >= total {
            return Ok(all);
        }

        tracing::info!(
            target: "ragrep::embed",
            "embedding {} texts via Voyage (api_batch={api_batch}, start={start})",
            total
        );

        let mut throttle = Throttle::default();
        let mut batch_idx = 0usize;
        let mut i = start;
        while i < total {
            let end = (i + api_batch).min(total);
            let batch: Vec<&str> = texts[i..end].to_vec();

            // Throttle between batches; skip the very first request so we
            // don't pay 30s on a single-batch invocation.
            if i > start {
                throttle.wait();
            }

            let mut attempt = 0u32;
            loop {
                attempt += 1;
                match embed_call(&self.api_key, &self.model, &batch, "document") {
                    Ok(vs) => {
                        all.extend(vs);
                        throttle.on_success();
                        break;
                    }
                    Err(e) => {
                        if let Some(retry_after) = parse_429_retry_after(&e) {
                            if attempt >= MAX_ATTEMPTS {
                                save_checkpoint(checkpoint, &all, total, self.dim)?;
                                return Err(e).context(format!(
                                    "Voyage embed: rate-limit exhausted at {}/{}",
                                    all.len(),
                                    total
                                ));
                            }
                            throttle.on_rate_limit(retry_after);
                            tracing::warn!(
                                target: "ragrep::embed",
                                "rate-limited; sleeping {:.1}s (attempt {}/{}, retry-after={:?})",
                                throttle.delay,
                                attempt,
                                MAX_ATTEMPTS,
                                retry_after
                            );
                            thread::sleep(Duration::from_secs_f32(throttle.delay));
                            continue;
                        }
                        save_checkpoint(checkpoint, &all, total, self.dim)?;
                        return Err(e).context(format!(
                            "Voyage embed failed at {}/{}",
                            all.len(),
                            total
                        ));
                    }
                }
            }

            batch_idx += 1;
            if batch_idx.is_multiple_of(CHECKPOINT_EVERY) && all.len() < total {
                save_checkpoint(checkpoint, &all, total, self.dim)?;
            }

            i = end;
        }

        // Success: drop the checkpoint.
        if let Some(path) = checkpoint {
            let _ = std::fs::remove_file(path);
        }

        Ok(all)
    }
}

const CKPT_MAGIC: u32 = 0x5241_4332; // "RAG2"
const CKPT_HEADER_BYTES: usize = 4 + 4 + 4 + 4;

/// Try to read a checkpoint at `path`. Returns Some(prefix) if the checkpoint
/// matches the expected `total` and `dim`; otherwise None (and the file is
/// quietly ignored). Errors only on I/O / parse problems.
fn read_checkpoint(
    path: &Path,
    expected_total: usize,
    expected_dim: usize,
) -> Result<Option<Vec<Vec<f32>>>> {
    if !path.exists() {
        return Ok(None);
    }
    let file =
        File::open(path).with_context(|| format!("opening checkpoint {}", path.display()))?;
    let mut r = BufReader::new(file);
    let mut header = [0u8; CKPT_HEADER_BYTES];
    if r.read_exact(&mut header).is_err() {
        return Ok(None);
    }
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let n_done = u32::from_le_bytes(header[4..8].try_into().unwrap()) as usize;
    let total = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    if magic != CKPT_MAGIC || total != expected_total || dim != expected_dim {
        // Stale checkpoint; ignore.
        return Ok(None);
    }
    let mut bytes = vec![0u8; n_done * dim * 4];
    r.read_exact(&mut bytes)
        .with_context(|| format!("reading checkpoint body of {}", path.display()))?;
    let flat: &[f32] = bytemuck::cast_slice(&bytes);
    let out: Vec<Vec<f32>> = flat.chunks_exact(dim).map(|c| c.to_vec()).collect();
    Ok(Some(out))
}

/// Atomically write a checkpoint with the embeddings collected so far.
fn save_checkpoint(path: Option<&Path>, all: &[Vec<f32>], total: usize, dim: usize) -> Result<()> {
    let Some(path) = path else { return Ok(()) };
    if all.is_empty() {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let tmp = path.with_extension("bin.tmp");
    let file =
        File::create(&tmp).with_context(|| format!("creating checkpoint {}", tmp.display()))?;
    let mut w = BufWriter::new(file);
    w.write_all(&CKPT_MAGIC.to_le_bytes())?;
    w.write_all(&(all.len() as u32).to_le_bytes())?;
    w.write_all(&(total as u32).to_le_bytes())?;
    w.write_all(&(dim as u32).to_le_bytes())?;
    for v in all {
        w.write_all(bytemuck::cast_slice(v))?;
    }
    w.flush()?;
    drop(w);
    std::fs::rename(&tmp, path)
        .with_context(|| format!("renaming {} → {}", tmp.display(), path.display()))?;
    tracing::info!(
        target: "ragrep::embed",
        "checkpoint saved: {}/{} at {}",
        all.len(),
        total,
        path.display()
    );
    Ok(())
}

/// Pull `Retry-After` (in seconds) out of an embed error, if it was a 429.
/// Returns Some(secs) on rate-limit (with possibly None retry_after meaning
/// "use default backoff"), and None on any non-429 error.
fn parse_429_retry_after(e: &anyhow::Error) -> Option<Option<f32>> {
    let msg = format!("{e}");
    if !msg.contains("HTTP 429") {
        return None;
    }
    // Look for "retry-after: <num>" in the body.
    let lower = msg.to_lowercase();
    if let Some(idx) = lower.find("retry-after") {
        let tail = &msg[idx..];
        let secs: String = tail
            .chars()
            .skip_while(|c| !c.is_ascii_digit())
            .take_while(|c| c.is_ascii_digit() || *c == '.')
            .collect();
        if let Ok(v) = secs.parse::<f32>() {
            return Some(Some(v));
        }
    }
    Some(None)
}

#[derive(Serialize)]
struct EmbedRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
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

/// Single POST to /v1/embeddings. Returns embeddings sorted by their input index.
fn embed_call(
    api_key: &str,
    model: &str,
    inputs: &[&str],
    input_type: &str,
) -> Result<Vec<Vec<f32>>> {
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
            input_type,
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
    Ok(payload.data.into_iter().map(|x| x.embedding).collect())
}
