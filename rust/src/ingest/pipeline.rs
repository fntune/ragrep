//! Ingest orchestrator: scrape → normalize → chunk → embed → store.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::embed;
use crate::index::{bm25::Bm25, store};
use crate::ingest::{chunk, normalize};
use crate::models::IngestStats;

const CHECKPOINT_FILE: &str = ".embed_checkpoint.bin";

pub fn run(cfg: &Config, force: bool, source_filter: Option<&str>) -> Result<IngestStats> {
    let start = Instant::now();
    let raw_dir = cfg.raw_dir();
    let index_dir = cfg.index_dir();
    std::fs::create_dir_all(&index_dir)
        .with_context(|| format!("creating {}", index_dir.display()))?;

    // 1. Resolve embedder via factory
    let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?;
    tracing::info!(
        target: "ragrep::ingest",
        "embedder ready: provider={} model={} dim={}",
        embedder.provider(),
        embedder.model(),
        embedder.dim()
    );

    // 2. Load per-(provider, model) cache (empty if --force or file absent)
    let mut cache = if force {
        embed::cache::Cache::empty(embedder.provider(), embedder.model(), embedder.dim())
    } else {
        embed::cache::load(
            &index_dir,
            embedder.provider(),
            embedder.model(),
            embedder.dim(),
        )?
    };
    tracing::info!(
        target: "ragrep::ingest",
        "embed cache: {} entries",
        cache.len()
    );

    // 3. Normalize
    tracing::info!(target: "ragrep::ingest", "step 1/4 normalize: reading {}", raw_dir.display());
    let mut docs = normalize::normalize_all(&raw_dir)?;
    if let Some(s) = source_filter {
        docs.retain(|d| d.source == s);
        tracing::info!(target: "ragrep::ingest", "filtered to {} documents (source={s})", docs.len());
    }
    if docs.is_empty() {
        tracing::warn!(target: "ragrep::ingest", "no documents to ingest");
        return Ok(IngestStats::default());
    }

    // 4. Chunk
    tracing::info!(target: "ragrep::ingest", "step 2/4 chunk: {} docs", docs.len());
    let chunks = chunk::all(
        &docs,
        cfg.ingest.max_chunk_tokens,
        cfg.ingest.chunk_overlap_tokens,
    );

    // 5. Hash + diff vs cache
    tracing::info!(target: "ragrep::ingest", "step 3/4 embed: {} chunks", chunks.len());
    let hashes: Vec<embed::cache::Hash> = chunks
        .iter()
        .map(|c| embed::cache::hash(&c.content))
        .collect();
    let to_embed_idx: Vec<usize> = hashes
        .iter()
        .enumerate()
        .filter_map(|(i, h)| if cache.contains(h) { None } else { Some(i) })
        .collect();
    let cached_count = chunks.len() - to_embed_idx.len();
    tracing::info!(
        target: "ragrep::ingest",
        "embed plan: {} new, {} cached",
        to_embed_idx.len(),
        cached_count
    );

    // 6. Embed misses
    if !to_embed_idx.is_empty() {
        let texts: Vec<&str> = to_embed_idx
            .iter()
            .map(|i| chunks[*i].content.as_str())
            .collect();
        let checkpoint = index_dir.join(CHECKPOINT_FILE);
        let new_embeddings = embedder
            .embed_documents(&texts, cfg.ingest.batch_size, Some(&checkpoint))
            .context("embedding new chunks")?;
        for (j, idx) in to_embed_idx.iter().enumerate() {
            cache.insert(hashes[*idx], new_embeddings[j].clone());
        }
    }

    // 7. Assemble full embedding matrix in chunk order
    let mut embeddings: Vec<Vec<f32>> = Vec::with_capacity(chunks.len());
    for h in &hashes {
        embeddings.push(
            cache
                .get(h)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("embedding missing for hash after embed step"))?,
        );
    }

    // 8. Save chunks + BM25 + embeddings (atomic) + cache
    tracing::info!(target: "ragrep::ingest", "step 4/4 store: writing index files");
    let bm25 = Bm25::build(chunks.iter().map(|c| c.content.as_str()));
    store::save_index(&index_dir, &chunks, &bm25, &embeddings, embedder.dim())?;
    embed::cache::save(&cache, &index_dir)?;

    let elapsed = start.elapsed().as_secs_f64();

    let mut sources: BTreeMap<String, usize> = BTreeMap::new();
    for d in &docs {
        *sources.entry(d.source.clone()).or_insert(0) += 1;
    }

    let stats = IngestStats {
        documents: docs.len(),
        chunks: chunks.len(),
        sources,
        elapsed_s: elapsed,
    };

    tracing::info!(
        target: "ragrep::ingest",
        "done: {} docs → {} chunks ({} new, {} cached) in {:.1}s",
        stats.documents,
        stats.chunks,
        to_embed_idx.len(),
        cached_count,
        stats.elapsed_s
    );
    Ok(stats)
}
