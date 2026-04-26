use std::path::PathBuf;
use std::{collections::BTreeMap, fs};

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::index::store;

#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: StatsArgs) -> Result<()> {
    let cfg = config::load(args.config.as_deref())?;
    let index_dir = cfg.index_dir();

    if !store::index_exists(&index_dir) {
        println!(
            "No complete Rust index found at {}. Run 'ragrep ingest' first.",
            index_dir.display()
        );
        return Ok(());
    }

    let chunks = store::load_chunks(&index_dir)?;
    let embedding_bytes = fs::metadata(store::embeddings_path(&index_dir))?.len();
    let shape = infer_embedding_shape(embedding_bytes, chunks.len());

    println!("Index: {}", index_dir.display());
    match shape {
        Some((vectors, dim)) => {
            println!("  Vectors: {vectors}");
            println!("  Dim:     {dim}");
        }
        None => {
            println!("  Vectors: unknown");
            println!("  Dim:     unknown");
        }
    }
    println!("  Chunks:  {}", chunks.len());

    let mut sources: BTreeMap<&str, usize> = BTreeMap::new();
    for chunk in &chunks {
        *sources.entry(chunk.source.as_str()).or_insert(0) += 1;
    }

    println!("  Sources:");
    let mut source_counts: Vec<(&str, usize)> = sources.into_iter().collect();
    source_counts.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (source, count) in source_counts {
        println!("    {source}: {count}");
    }

    for name in [store::FILE_EMBEDDINGS, store::FILE_CHUNKS, store::FILE_BM25] {
        let path = index_dir.join(name);
        if path.exists() {
            println!("  {name}: {:.1} MB", file_mb(&path)?);
        }
    }

    let cache_dir = index_dir.join("embed_cache");
    if cache_dir.is_dir() {
        let mut entries = fs::read_dir(&cache_dir)?
            .filter_map(|entry| entry.ok())
            .filter(|entry| entry.path().is_file())
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.file_name());
        for entry in entries {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            println!("  embed_cache/{name}: {:.1} MB", file_mb(&entry.path())?);
        }
    }

    Ok(())
}

fn infer_embedding_shape(bytes: u64, chunks: usize) -> Option<(usize, usize)> {
    if bytes == 0 || chunks == 0 {
        return None;
    }
    let row_bytes = chunks.checked_mul(std::mem::size_of::<f32>())? as u64;
    if bytes % row_bytes != 0 {
        return None;
    }
    Some((chunks, (bytes / row_bytes) as usize))
}

fn file_mb(path: &std::path::Path) -> Result<f64> {
    Ok(fs::metadata(path)?.len() as f64 / 1024.0 / 1024.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn infers_shape_from_chunk_count() {
        assert_eq!(infer_embedding_shape(10 * 1024 * 4, 10), Some((10, 1024)));
    }

    #[test]
    fn rejects_misaligned_shape() {
        assert_eq!(infer_embedding_shape(17, 10), None);
        assert_eq!(infer_embedding_shape(0, 10), None);
        assert_eq!(infer_embedding_shape(100, 0), None);
    }
}
