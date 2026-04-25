use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

const ENV_VAR_CONFIG: &str = "RAGREP_CONFIG";

#[derive(Debug, Default, Clone, Deserialize)]
pub struct Config {
    #[serde(default)]
    pub data: DataConfig,
    #[serde(default)]
    pub ingest: IngestConfig,
    #[serde(default)]
    pub embedding: EmbeddingConfig,
    #[serde(default)]
    pub reranker: RerankerConfig,
    #[serde(default)]
    pub retrieval: RetrievalConfig,
    #[serde(default)]
    pub generation: GenerationConfig,
    #[serde(default)]
    pub scrape: ScrapeConfig,
}

impl Config {
    pub fn raw_dir(&self) -> PathBuf {
        PathBuf::from(&self.data.raw_dir)
    }
    pub fn index_dir(&self) -> PathBuf {
        PathBuf::from(&self.data.index_dir)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DataConfig {
    #[serde(default = "default_raw_dir")]
    pub raw_dir: String,
    #[serde(default = "default_index_dir")]
    pub index_dir: String,
}

impl Default for DataConfig {
    fn default() -> Self {
        Self {
            raw_dir: default_raw_dir(),
            index_dir: default_index_dir(),
        }
    }
}

fn default_raw_dir() -> String {
    "data/raw".into()
}
fn default_index_dir() -> String {
    "data/index".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct IngestConfig {
    #[serde(default = "default_chunk_tokens")]
    pub max_chunk_tokens: usize,
    #[serde(default = "default_overlap_tokens")]
    pub chunk_overlap_tokens: usize,
    #[serde(default = "default_batch_size")]
    pub batch_size: usize,
}

impl Default for IngestConfig {
    fn default() -> Self {
        Self {
            max_chunk_tokens: default_chunk_tokens(),
            chunk_overlap_tokens: default_overlap_tokens(),
            batch_size: default_batch_size(),
        }
    }
}

fn default_chunk_tokens() -> usize {
    512
}
fn default_overlap_tokens() -> usize {
    64
}
fn default_batch_size() -> usize {
    32
}

#[derive(Debug, Clone, Deserialize)]
pub struct EmbeddingConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_model_name")]
    pub model_name: String,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model_name: default_model_name(),
        }
    }
}

fn default_provider() -> String {
    "voyage".into()
}
fn default_model_name() -> String {
    "voyage-code-3".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RerankerConfig {
    #[serde(default = "default_provider")]
    pub provider: String,
    #[serde(default = "default_rerank_model")]
    pub model_name: String,
}

impl Default for RerankerConfig {
    fn default() -> Self {
        Self {
            provider: default_provider(),
            model_name: default_rerank_model(),
        }
    }
}

fn default_rerank_model() -> String {
    "rerank-2.5".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct RetrievalConfig {
    #[serde(default = "default_top_k")]
    pub top_k_dense: usize,
    #[serde(default = "default_top_k")]
    pub top_k_bm25: usize,
    #[serde(default = "default_rerank_k")]
    pub top_k_rerank: usize,
    #[serde(default = "default_rrf_k")]
    pub rrf_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            top_k_dense: default_top_k(),
            top_k_bm25: default_top_k(),
            top_k_rerank: default_rerank_k(),
            rrf_k: default_rrf_k(),
        }
    }
}

fn default_top_k() -> usize {
    20
}
fn default_rerank_k() -> usize {
    5
}
fn default_rrf_k() -> usize {
    60
}

#[derive(Debug, Clone, Deserialize)]
pub struct GenerationConfig {
    #[serde(default = "default_gen_model")]
    pub model_name: String,
    #[serde(default = "default_ollama_url")]
    pub ollama_url: String,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            model_name: default_gen_model(),
            ollama_url: default_ollama_url(),
            temperature: default_temperature(),
            max_tokens: default_max_tokens(),
        }
    }
}

fn default_gen_model() -> String {
    "gemma3:4b".into()
}
fn default_ollama_url() -> String {
    "http://localhost:11434".into()
}
fn default_temperature() -> f32 {
    0.3
}
fn default_max_tokens() -> usize {
    1024
}

#[derive(Debug, Default, Clone, Deserialize)]
pub struct ScrapeConfig {
    #[serde(default)]
    pub slack: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub atlassian: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub gdrive: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub git: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub bitbucket: BTreeMap<String, toml::Value>,
    #[serde(default)]
    pub code: BTreeMap<String, toml::Value>,
}

/// Populate `std::env` from `.env` files. Searches CWD, then `~/.config/ragrep/.env`.
/// Existing env vars win (set-once semantics, mirroring `os.environ.setdefault`).
pub fn load_env_files() {
    let mut candidates: Vec<PathBuf> = vec![PathBuf::from(".env")];
    if let Some(home) = dirs_home() {
        candidates.push(home.join(".config").join("ragrep").join(".env"));
    }
    for p in candidates {
        let _ = dotenvy::from_path(&p);
    }
}

/// Resolve config.toml: explicit arg → RAGREP_CONFIG env → ./config.toml → ~/.config/ragrep/config.toml.
pub fn find_config_path(explicit: Option<&Path>) -> Option<PathBuf> {
    if let Some(p) = explicit {
        return Some(p.to_path_buf());
    }
    if let Ok(p) = std::env::var(ENV_VAR_CONFIG) {
        return Some(PathBuf::from(p));
    }
    let cwd = PathBuf::from("config.toml");
    if cwd.exists() {
        return Some(cwd);
    }
    if let Some(home) = dirs_home() {
        let xdg = home.join(".config").join("ragrep").join("config.toml");
        if xdg.exists() {
            return Some(xdg);
        }
    }
    None
}

/// Load config from TOML file. Falls back to defaults when no file is found.
pub fn load(path: Option<&Path>) -> Result<Config> {
    let resolved = find_config_path(path);
    match resolved {
        Some(p) if p.exists() => {
            let raw = fs::read_to_string(&p).with_context(|| format!("reading {}", p.display()))?;
            toml::from_str(&raw).with_context(|| format!("parsing {}", p.display()))
        }
        _ => Ok(Config::default()),
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var_os("HOME").map(PathBuf::from)
}
