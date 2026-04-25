use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct StatsArgs {
    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: StatsArgs) -> Result<()> {
    eprintln!("ragrep stats: not yet implemented in the Rust port (Phase 4)");
    std::process::exit(1)
}
