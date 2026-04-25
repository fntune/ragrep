use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct IngestArgs {
    /// Re-embed everything (ignore the embedding cache).
    #[arg(long)]
    pub force: bool,

    /// Limit to one source type.
    #[arg(short, long)]
    pub source: Option<String>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: IngestArgs) -> Result<()> {
    eprintln!("ragrep ingest: not yet implemented in the Rust port (Phase 2)");
    std::process::exit(1)
}
