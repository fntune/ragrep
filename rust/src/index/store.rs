use std::fs::{File, OpenOptions};
use std::io::{BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use fs2::FileExt;

use crate::index::bm25::Bm25;
use crate::index::flat::Flat;
use crate::models::Chunk;

pub const FILE_CHUNKS: &str = "chunks.msgpack";
pub const FILE_EMBEDDINGS: &str = "embeddings.bin";
pub const FILE_BM25: &str = "bm25.msgpack";
const LOCK_FILE: &str = ".index.lock";

pub struct RuntimeIndex {
    pub chunks: Vec<Chunk>,
    pub flat: Flat,
    pub bm25: Bm25,
}

pub fn chunks_path(index_dir: &Path) -> PathBuf {
    index_dir.join(FILE_CHUNKS)
}
pub fn embeddings_path(index_dir: &Path) -> PathBuf {
    index_dir.join(FILE_EMBEDDINGS)
}
pub fn bm25_path(index_dir: &Path) -> PathBuf {
    index_dir.join(FILE_BM25)
}

pub fn load_chunks(index_dir: &Path) -> Result<Vec<Chunk>> {
    let _lock = lock(index_dir, LockMode::Shared)?;
    read_chunks(index_dir)
}

pub fn load_flat(index_dir: &Path, dim: usize) -> Result<Flat> {
    let _lock = lock(index_dir, LockMode::Shared)?;
    read_flat(index_dir, dim)
}

pub fn load_bm25(index_dir: &Path) -> Result<Bm25> {
    let _lock = lock(index_dir, LockMode::Shared)?;
    read_bm25(index_dir)
}

pub fn load_runtime(index_dir: &Path, dim: usize) -> Result<RuntimeIndex> {
    let _lock = lock(index_dir, LockMode::Shared)?;
    Ok(RuntimeIndex {
        chunks: read_chunks(index_dir)?,
        flat: read_flat(index_dir, dim)?,
        bm25: read_bm25(index_dir)?,
    })
}

fn read_chunks(index_dir: &Path) -> Result<Vec<Chunk>> {
    let path = chunks_path(index_dir);
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let chunks: Vec<Chunk> = rmp_serde::from_read(BufReader::new(file))
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(chunks)
}

fn read_flat(index_dir: &Path, dim: usize) -> Result<Flat> {
    let path = embeddings_path(index_dir);
    Flat::open(&path, dim).with_context(|| format!("mmapping {}", path.display()))
}

fn read_bm25(index_dir: &Path) -> Result<Bm25> {
    let path = bm25_path(index_dir);
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let bm25: Bm25 = rmp_serde::from_read(BufReader::new(file))
        .with_context(|| format!("parsing {}", path.display()))?;
    Ok(bm25)
}

pub fn save_index(
    index_dir: &Path,
    chunks: &[Chunk],
    bm25: &Bm25,
    embeddings: &[Vec<f32>],
    dim: usize,
) -> Result<()> {
    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("creating {}", index_dir.display()))?;
    let stage_dir = stage_dir(index_dir)?;
    let result = (|| -> Result<()> {
        write_chunks_file(&stage_dir.join(FILE_CHUNKS), chunks)?;
        write_bm25_file(&stage_dir.join(FILE_BM25), bm25)?;
        write_embeddings_file(&stage_dir.join(FILE_EMBEDDINGS), embeddings, dim)?;
        publish_stage(index_dir, &stage_dir)
    })();
    let cleanup = std::fs::remove_dir_all(&stage_dir);
    match (result, cleanup) {
        (Ok(()), Ok(())) | (Ok(()), Err(_)) => Ok(()),
        (Err(err), Ok(())) | (Err(err), Err(_)) => Err(err),
    }
}

pub fn save_bm25(index_dir: &Path, bm25: &Bm25) -> Result<()> {
    let path = bm25_path(index_dir);
    let tmp = temp_file_path(&path);
    write_bm25_file(&tmp, bm25)?;
    let _lock = lock(index_dir, LockMode::Exclusive)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))
}

pub fn save_chunks(index_dir: &Path, chunks: &[Chunk]) -> Result<()> {
    let path = chunks_path(index_dir);
    let tmp = temp_file_path(&path);
    write_chunks_file(&tmp, chunks)?;
    let _lock = lock(index_dir, LockMode::Exclusive)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))
}

pub fn save_embeddings(index_dir: &Path, embeddings: &[Vec<f32>], dim: usize) -> Result<()> {
    let path = embeddings_path(index_dir);
    let tmp = temp_file_path(&path);
    write_embeddings_file(&tmp, embeddings, dim)?;
    let _lock = lock(index_dir, LockMode::Exclusive)?;
    std::fs::rename(&tmp, &path).with_context(|| format!("renaming {}", tmp.display()))
}

pub fn clear_index(index_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("creating {}", index_dir.display()))?;
    let _lock = lock(index_dir, LockMode::Exclusive)?;
    for name in [FILE_CHUNKS, FILE_BM25, FILE_EMBEDDINGS] {
        let path = index_dir.join(name);
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => {}
            Err(err) => return Err(err).with_context(|| format!("removing {}", path.display())),
        }
    }
    Ok(())
}

fn publish_stage(index_dir: &Path, stage_dir: &Path) -> Result<()> {
    let _lock = lock(index_dir, LockMode::Exclusive)?;
    for name in [FILE_CHUNKS, FILE_BM25, FILE_EMBEDDINGS] {
        let src = stage_dir.join(name);
        let dst = index_dir.join(name);
        std::fs::rename(&src, &dst)
            .with_context(|| format!("renaming {} -> {}", src.display(), dst.display()))?;
    }
    Ok(())
}

fn write_chunks_file(path: &Path, chunks: &[Chunk]) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, &chunks)
        .with_context(|| format!("writing {}", path.display()))?;
    writer.flush()?;
    Ok(())
}

fn write_bm25_file(path: &Path, bm25: &Bm25) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    rmp_serde::encode::write_named(&mut writer, bm25)
        .with_context(|| format!("writing {}", path.display()))?;
    writer.flush()?;
    Ok(())
}

fn write_embeddings_file(path: &Path, embeddings: &[Vec<f32>], dim: usize) -> Result<()> {
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for v in embeddings {
        if v.len() != dim {
            anyhow::bail!("embedding dim mismatch: expected {dim}, got {}", v.len());
        }
        writer.write_all(bytemuck::cast_slice(v))?;
    }
    writer.flush()?;
    Ok(())
}

fn stage_dir(index_dir: &Path) -> Result<PathBuf> {
    let dir = index_dir.join(format!(
        ".publish-{}-{}",
        std::process::id(),
        unique_suffix()
    ));
    std::fs::create_dir(&dir).with_context(|| format!("creating {}", dir.display()))?;
    Ok(dir)
}

fn temp_file_path(path: &Path) -> PathBuf {
    path.with_extension(format!("tmp-{}-{}", std::process::id(), unique_suffix()))
}

fn unique_suffix() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default()
}

enum LockMode {
    Shared,
    Exclusive,
}

struct IndexLock(File);

fn lock(index_dir: &Path, mode: LockMode) -> Result<IndexLock> {
    std::fs::create_dir_all(index_dir)
        .with_context(|| format!("creating {}", index_dir.display()))?;
    let file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(index_dir.join(LOCK_FILE))
        .with_context(|| format!("opening {}", index_dir.join(LOCK_FILE).display()))?;
    match mode {
        LockMode::Shared => file.lock_shared(),
        LockMode::Exclusive => file.lock_exclusive(),
    }
    .with_context(|| format!("locking {}", index_dir.join(LOCK_FILE).display()))?;
    Ok(IndexLock(file))
}

impl Drop for IndexLock {
    fn drop(&mut self) {
        let _ = self.0.unlock();
    }
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

pub fn index_exists(index_dir: &Path) -> bool {
    chunks_exist(index_dir) && embeddings_exist(index_dir) && bm25_exists(index_dir)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn chunk(id: &str, source: &str, content: &str) -> Chunk {
        Chunk {
            id: id.to_string(),
            doc_id: id.to_string(),
            content: content.to_string(),
            title: format!("title {id}"),
            source: source.to_string(),
            metadata: BTreeMap::new(),
        }
    }

    #[test]
    fn save_index_publishes_complete_runtime_files() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![chunk("a", "freshdesk", "alpha")];
        let bm25 = Bm25::build(chunks.iter().map(|c| c.content.as_str()));

        save_index(dir.path(), &chunks, &bm25, &[vec![1.0, 0.0]], 2).unwrap();

        assert!(chunks_path(dir.path()).exists());
        assert!(bm25_path(dir.path()).exists());
        assert!(embeddings_path(dir.path()).exists());
        assert_eq!(load_chunks(dir.path()).unwrap()[0].id, "a");
        assert_eq!(
            std::fs::metadata(embeddings_path(dir.path()))
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn save_index_failure_keeps_previous_runtime_files() {
        let dir = tempfile::tempdir().unwrap();
        let old_chunks = vec![chunk("old", "freshdesk", "old content")];
        let old_bm25 = Bm25::build(old_chunks.iter().map(|c| c.content.as_str()));
        save_index(dir.path(), &old_chunks, &old_bm25, &[vec![1.0, 0.0]], 2).unwrap();

        let new_chunks = vec![chunk("new", "freshdesk", "new content")];
        let new_bm25 = Bm25::build(new_chunks.iter().map(|c| c.content.as_str()));
        let err = save_index(dir.path(), &new_chunks, &new_bm25, &[vec![1.0]], 2)
            .expect_err("dimension mismatch should fail before publish");

        assert!(err.to_string().contains("embedding dim mismatch"));
        assert_eq!(load_chunks(dir.path()).unwrap()[0].id, "old");
        assert_eq!(
            std::fs::metadata(embeddings_path(dir.path()))
                .unwrap()
                .len(),
            8
        );
    }

    #[test]
    fn clear_index_removes_runtime_files() {
        let dir = tempfile::tempdir().unwrap();
        let chunks = vec![chunk("a", "freshdesk", "alpha")];
        let bm25 = Bm25::build(chunks.iter().map(|c| c.content.as_str()));
        save_index(dir.path(), &chunks, &bm25, &[vec![1.0, 0.0]], 2).unwrap();

        clear_index(dir.path()).unwrap();

        assert!(!chunks_path(dir.path()).exists());
        assert!(!bm25_path(dir.path()).exists());
        assert!(!embeddings_path(dir.path()).exists());
    }
}
