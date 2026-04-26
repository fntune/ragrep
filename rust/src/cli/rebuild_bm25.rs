use std::path::PathBuf;
use std::time::Instant;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::index::{bm25::Bm25, store};

/// Rebuild bm25.msgpack from chunks.msgpack.
#[derive(Args, Debug)]
pub struct RebuildBm25Args {
    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: RebuildBm25Args) -> Result<()> {
    let cfg = config::load(args.config.as_deref())?;
    let dir = cfg.index_dir();

    let t = Instant::now();
    let chunks = store::load_chunks(&dir)?;
    eprintln!("loaded {} chunks in {:?}", chunks.len(), t.elapsed());

    let t = Instant::now();
    let bm25 = Bm25::build(chunks.iter().map(|c| c.content.as_str()));
    eprintln!("built BM25 in {:?}", t.elapsed());

    let t = Instant::now();
    store::save_bm25(&dir, &bm25)?;
    eprintln!(
        "wrote {} in {:?}",
        store::bm25_path(&dir).display(),
        t.elapsed()
    );
    Ok(())
}
