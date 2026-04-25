use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct InspectArgs {
    /// Inspection mode.
    #[arg(value_parser = ["raw", "docs", "sample", "grep"])]
    pub mode: String,

    /// Filter to a single source type.
    #[arg(short, long)]
    pub source: Option<String>,

    /// Search string (for sample/grep modes).
    #[arg(long)]
    pub grep: Option<String>,

    /// Number of results to show.
    #[arg(short)]
    pub n: Option<usize>,

    /// Show full chunk content.
    #[arg(long)]
    pub full: bool,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: InspectArgs) -> Result<()> {
    eprintln!("ragrep inspect: not yet implemented in the Rust port (Phase 4)");
    std::process::exit(1)
}
