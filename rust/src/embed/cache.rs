//! Per-(provider, model) embedding cache, on-disk binary format.
//!
//! Lives at `<index_dir>/embed_cache/{provider}--{model}.bin`. Compact layout
//! (~½ the size of msgpack-of-f64-lists, ~2× faster to load via mmap +
//! `bytemuck::cast_slice`):
//!
//! ```text
//! [0..4]    u32 magic   = 0x52414331  ("RAG1")
//! [4..8]    u32 version = 1
//! [8..12]   u32 dim
//! [12..20]  u64 n_entries
//! [20..]    n × (32 bytes sha256 || dim × 4 bytes f32, little-endian)
//! ```
//!
//! Multibyte ints are little-endian. The sha256 keys are raw bytes (the
//! `hash()` helper hashes UTF-8 source text). The f32 vectors are written
//! as raw bytes via `bytemuck::cast_slice`.

use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::{anyhow, bail, Context, Result};
use sha2::{Digest, Sha256};

const MAGIC: u32 = 0x5241_4331;
const VERSION: u32 = 1;
const HEADER_BYTES: usize = 4 + 4 + 4 + 8;
const HASH_BYTES: usize = 32;

pub type Hash = [u8; HASH_BYTES];

#[derive(Debug)]
pub struct Cache {
    provider: String,
    model: String,
    dim: usize,
    entries: HashMap<Hash, Vec<f32>>,
}

impl Cache {
    pub fn empty(provider: &str, model: &str, dim: usize) -> Self {
        Self {
            provider: provider.to_string(),
            model: model.to_string(),
            dim,
            entries: HashMap::new(),
        }
    }

    pub fn provider(&self) -> &str {
        &self.provider
    }
    pub fn model(&self) -> &str {
        &self.model
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn len(&self) -> usize {
        self.entries.len()
    }
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn get(&self, key: &Hash) -> Option<&Vec<f32>> {
        self.entries.get(key)
    }

    pub fn insert(&mut self, key: Hash, vec: Vec<f32>) {
        debug_assert_eq!(vec.len(), self.dim, "vector dim mismatch");
        self.entries.insert(key, vec);
    }

    pub fn contains(&self, key: &Hash) -> bool {
        self.entries.contains_key(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Hash, &Vec<f32>)> {
        self.entries.iter()
    }
}

/// Sha256 of the UTF-8 bytes of `text`. Stable content hash for cache keying.
pub fn hash(text: &str) -> Hash {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hasher.finalize().into()
}

pub fn cache_dir(index_dir: &Path) -> PathBuf {
    index_dir.join("embed_cache")
}

pub fn cache_path(index_dir: &Path, provider: &str, model: &str) -> PathBuf {
    cache_dir(index_dir).join(format!("{provider}--{model}.bin"))
}

/// Load a cache from disk. Returns an empty cache if the file doesn't exist.
pub fn load(index_dir: &Path, provider: &str, model: &str, expected_dim: usize) -> Result<Cache> {
    let path = cache_path(index_dir, provider, model);
    if !path.exists() {
        return Ok(Cache::empty(provider, model, expected_dim));
    }
    let file = File::open(&path).with_context(|| format!("opening {}", path.display()))?;
    let mut r = BufReader::new(file);

    let mut header = [0u8; HEADER_BYTES];
    r.read_exact(&mut header)
        .with_context(|| format!("reading header of {}", path.display()))?;
    let magic = u32::from_le_bytes(header[0..4].try_into().unwrap());
    let version = u32::from_le_bytes(header[4..8].try_into().unwrap());
    let dim = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let n = u64::from_le_bytes(header[12..20].try_into().unwrap()) as usize;

    if magic != MAGIC {
        bail!(
            "{}: bad magic 0x{:08x} (expected 0x{:08x})",
            path.display(),
            magic,
            MAGIC
        );
    }
    if version != VERSION {
        bail!(
            "{}: unsupported cache version {} (this binary supports {})",
            path.display(),
            version,
            VERSION
        );
    }
    if dim != expected_dim {
        bail!(
            "{}: dim mismatch — file has {}, embedder expects {}",
            path.display(),
            dim,
            expected_dim
        );
    }

    let row_bytes = HASH_BYTES + dim * 4;
    let mut entries = HashMap::with_capacity(n);
    let mut row = vec![0u8; row_bytes];
    for i in 0..n {
        r.read_exact(&mut row)
            .with_context(|| format!("reading entry {i} from {}", path.display()))?;
        let mut key = [0u8; HASH_BYTES];
        key.copy_from_slice(&row[..HASH_BYTES]);
        let vec_bytes = &row[HASH_BYTES..];
        let vec: Vec<f32> = bytemuck::cast_slice(vec_bytes).to_vec();
        entries.insert(key, vec);
    }

    Ok(Cache {
        provider: provider.to_string(),
        model: model.to_string(),
        dim,
        entries,
    })
}

/// Save the cache atomically (tempfile + rename).
pub fn save(cache: &Cache, index_dir: &Path) -> Result<()> {
    let dir = cache_dir(index_dir);
    fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
    let final_path = cache_path(index_dir, &cache.provider, &cache.model);
    let tmp_path = final_path.with_extension("bin.tmp");

    let file =
        File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?;
    let mut w = BufWriter::new(file);

    w.write_all(&MAGIC.to_le_bytes())?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&(cache.dim as u32).to_le_bytes())?;
    w.write_all(&(cache.entries.len() as u64).to_le_bytes())?;

    for (key, vec) in &cache.entries {
        if vec.len() != cache.dim {
            return Err(anyhow!(
                "vec dim mismatch for key {:?}: expected {}, got {}",
                &key[..8],
                cache.dim,
                vec.len()
            ));
        }
        w.write_all(key)?;
        w.write_all(bytemuck::cast_slice(vec))?;
    }
    w.flush()?;
    drop(w);

    fs::rename(&tmp_path, &final_path)
        .with_context(|| format!("renaming {} → {}", tmp_path.display(), final_path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn roundtrip_empty() {
        let dir = tempdir().unwrap();
        let c = Cache::empty("voyage", "voyage-code-3", 4);
        save(&c, dir.path()).unwrap();
        let loaded = load(dir.path(), "voyage", "voyage-code-3", 4).unwrap();
        assert_eq!(loaded.len(), 0);
        assert_eq!(loaded.dim(), 4);
    }

    #[test]
    fn roundtrip_with_entries() {
        let dir = tempdir().unwrap();
        let mut c = Cache::empty("voyage", "voyage-code-3", 4);
        let h1 = hash("hello");
        let h2 = hash("world");
        c.insert(h1, vec![1.0, 2.0, 3.0, 4.0]);
        c.insert(h2, vec![5.0, 6.0, 7.0, 8.0]);
        save(&c, dir.path()).unwrap();

        let loaded = load(dir.path(), "voyage", "voyage-code-3", 4).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&h1), Some(&vec![1.0, 2.0, 3.0, 4.0]));
        assert_eq!(loaded.get(&h2), Some(&vec![5.0, 6.0, 7.0, 8.0]));
    }

    #[test]
    fn missing_file_yields_empty() {
        let dir = tempdir().unwrap();
        let loaded = load(dir.path(), "voyage", "voyage-code-3", 1024).unwrap();
        assert!(loaded.is_empty());
        assert_eq!(loaded.dim(), 1024);
    }

    #[test]
    fn dim_mismatch_errors() {
        let dir = tempdir().unwrap();
        let c = Cache::empty("voyage", "v", 4);
        save(&c, dir.path()).unwrap();
        let err = load(dir.path(), "voyage", "v", 8).unwrap_err();
        assert!(format!("{err}").contains("dim mismatch"));
    }

    #[test]
    fn hash_is_stable() {
        let h1 = hash("hello");
        let h2 = hash("hello");
        assert_eq!(h1, h2);
        let h3 = hash("hello!");
        assert_ne!(h1, h3);
    }
}
