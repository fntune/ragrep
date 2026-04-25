use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use anyhow::{Context, Result};

use crate::index::bm25::Bm25;
use crate::index::flat::Flat;
use crate::models::Chunk;

pub const FILE_CHUNKS: &str = "chunks.msgpack";
pub const FILE_EMBEDDINGS: &str = "embeddings.bin";
pub const FILE_BM25: &str = "bm25.msgpack";

pub fn chunks_path(index_dir: &Path) -> std::path::PathBuf {
    index_dir.join(FILE_CHUNKS)
}
pub fn embeddings_path(index_dir: &Path) -> std::path::PathBuf {
    index_dir.join(FILE_EMBEDDINGS)
}
pub fn bm25_path(index_dir: &Path) -> std::path::PathBuf {
    index_dir.join(FILE_BM25)
}

pub fn load_chunks(index_dir: &Path) -> Result<Vec<Chunk>> {
    let path = chunks_path(index_dir);
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let chunks: Vec<Chunk> = rmp_serde::from_read(BufReader::new(file))
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(chunks)
}

pub fn load_flat(index_dir: &Path, dim: usize) -> Result<Flat> {
    let path = embeddings_path(index_dir);
    Flat::open(&path, dim).with_context(|| format!("mmapping {}", path.display()))
}

pub fn load_bm25(index_dir: &Path) -> Result<Bm25> {
    let path = bm25_path(index_dir);
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let bm25: Bm25 = rmp_serde::from_read(BufReader::new(file))
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(bm25)
}

pub fn save_bm25(index_dir: &Path, bm25: &Bm25) -> Result<()> {
    let path = bm25_path(index_dir);
    let tmp = path.with_extension("msgpack.tmp");
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, bm25)
        .with_context(|| format!("writing {}", tmp.display()))?;
    drop(writer);
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))?;
    Ok(())
}

pub fn save_chunks(index_dir: &Path, chunks: &[Chunk]) -> Result<()> {
    let path = chunks_path(index_dir);
    let tmp = path.with_extension("msgpack.tmp");
    let file = File::create(&tmp).with_context(|| format!("creating {}", tmp.display()))?;
    let mut writer = std::io::BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, &chunks)
        .with_context(|| format!("writing {}", tmp.display()))?;
    drop(writer);
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))?;
    Ok(())
}

pub fn chunks_exist(index_dir: &Path) -> bool {
    chunks_path(index_dir).exists()
}
pub fn embeddings_exist(index_dir: &Path) -> bool {
    embeddings_path(index_dir).exists()
}
pub fn bm25_exists(index_dir: &Path) -> bool {
    bm25_path(index_dir).exists()
}
