//! Ingest orchestrator: scrape → normalize → chunk → embed → store.

use std::collections::BTreeMap;
use std::time::Instant;

use anyhow::{Context, Result};

use crate::config::Config;
use crate::embed;
use crate::index::{bm25::Bm25, store};
use crate::ingest::{chunk, normalize};
use crate::models::{Chunk, Document, IngestStats};

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
    let docs = if let Some(s) = source_filter {
        let docs = normalize::normalize_source(&raw_dir, s)?;
        tracing::info!(target: "ragrep::ingest", "filtered to {} documents (source={s})", docs.len());
        docs
    } else {
        normalize::normalize_all(&raw_dir)?
    };

    // 4. Chunk
    tracing::info!(target: "ragrep::ingest", "step 2/4 chunk: {} docs", docs.len());
    let replacement_chunks = chunk::all(
        &docs,
        cfg.ingest.max_chunk_tokens,
        cfg.ingest.chunk_overlap_tokens,
    );
    let chunks = runtime_chunks(&index_dir, source_filter, replacement_chunks)?;
    if chunks.is_empty() {
        tracing::warn!(target: "ragrep::ingest", "no chunks to ingest");
        if source_filter.is_some() {
            store::clear_index(&index_dir)?;
        }
        return Ok(ingest_stats(&docs, 0, start.elapsed().as_secs_f64()));
    }

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

    let stats = ingest_stats(&docs, chunks.len(), start.elapsed().as_secs_f64());

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

fn ingest_stats(docs: &[Document], chunks: usize, elapsed_s: f64) -> IngestStats {
    let mut sources: BTreeMap<String, usize> = BTreeMap::new();
    for doc in docs {
        *sources.entry(doc.source.clone()).or_insert(0) += 1;
    }

    IngestStats {
        documents: docs.len(),
        chunks,
        sources,
        elapsed_s,
    }
}

fn runtime_chunks(
    index_dir: &std::path::Path,
    source_filter: Option<&str>,
    replacement: Vec<Chunk>,
) -> Result<Vec<Chunk>> {
    let Some(source) = source_filter else {
        return Ok(replacement);
    };

    let existing = if store::chunks_exist(index_dir) {
        store::load_chunks(index_dir)?
    } else {
        Vec::new()
    };
    let preserved_count = existing
        .iter()
        .filter(|chunk| chunk.source != source)
        .count();
    let chunks = replace_source_chunks(existing, replacement, source);
    tracing::info!(
        target: "ragrep::ingest",
        "source-scoped publish: replaced source={source}, preserved {} existing chunks, runtime now has {} chunks",
        preserved_count,
        chunks.len()
    );
    Ok(chunks)
}

fn replace_source_chunks(
    existing: Vec<Chunk>,
    replacement: Vec<Chunk>,
    source: &str,
) -> Vec<Chunk> {
    let mut chunks: Vec<Chunk> = existing
        .into_iter()
        .filter(|chunk| chunk.source != source)
        .collect();
    chunks.extend(replacement);
    chunks
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chunk(id: &str, source: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            doc_id: id.to_string(),
            content: format!("{id} content"),
            title: format!("{id} title"),
            source: source.to_string(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn source_scoped_chunks_replace_only_selected_source() {
        let existing = vec![
            chunk("old-freshdesk", "freshdesk"),
            chunk("youtube-video", "youtube"),
            chunk("git-change", "git"),
        ];
        let replacement = vec![chunk("new-freshdesk", "freshdesk")];

        let chunks = replace_source_chunks(existing, replacement, "freshdesk");

        let ids: Vec<&str> = chunks.iter().map(|chunk| chunk.id.as_str()).collect();
        assert_eq!(ids, vec!["youtube-video", "git-change", "new-freshdesk"]);
    }

    #[test]
    fn source_scoped_chunks_can_clear_a_source() {
        let existing = vec![
            chunk("old-freshdesk", "freshdesk"),
            chunk("youtube-video", "youtube"),
        ];

        let chunks = replace_source_chunks(existing, Vec::new(), "freshdesk");

        let ids: Vec<&str> = chunks.iter().map(|chunk| chunk.id.as_str()).collect();
        assert_eq!(ids, vec!["youtube-video"]);
    }
}
