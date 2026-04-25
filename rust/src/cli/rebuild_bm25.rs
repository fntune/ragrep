use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

/// Rebuild bm25.msgpack from chunks.msgpack. Transitional bridge for
/// indexes that came from the Python release (where bm25.pkl was dropped
/// during migration).
#[derive(Args, Debug)]
pub struct RebuildBm25Args {
    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: RebuildBm25Args) -> Result<()> {
    eprintln!("ragrep rebuild-bm25: not yet implemented in the Rust port (Phase 1)");
    std::process::exit(1)
}
