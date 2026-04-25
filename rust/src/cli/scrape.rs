use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct ScrapeArgs {
    /// Comma-separated sources: slack,atlassian,gdrive,git,bitbucket,code,files.
    #[arg(short, long)]
    pub source: Option<String>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: ScrapeArgs) -> Result<()> {
    eprintln!("ragrep scrape: not yet implemented in the Rust port (Phase 3)");
    std::process::exit(1)
}
