use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct SearchArgs {
    /// Search term (positional).
    pub term: String,

    /// Number of results.
    #[arg(short, long, default_value_t = 5)]
    pub n: usize,

    /// Filter to a single source type (slack, atlassian, gdrive, git, bitbucket, file, ...).
    #[arg(short, long)]
    pub source: Option<String>,

    /// Search mode.
    #[arg(short = 'm', long, default_value = "hybrid", value_parser = ["grep", "semantic", "hybrid"])]
    pub mode: String,

    /// Repeatable metadata filter, AND-combined, substring match.
    #[arg(short = 'f', long = "filter", value_name = "KEY=VAL")]
    pub filter: Vec<String>,

    /// Inclusive lower date bound (YYYY-MM-DD or relative: 3m, 2w, 90d, 1y).
    #[arg(long)]
    pub after: Option<String>,

    /// Exclusive upper date bound.
    #[arg(long)]
    pub before: Option<String>,

    /// Snippet length in chars (0 disables).
    #[arg(short, long, default_value_t = 200)]
    pub context: usize,

    /// Show full chunk content.
    #[arg(long)]
    pub full: bool,

    /// JSON output (for agents/scripts).
    #[arg(long)]
    pub json: bool,

    /// Include scores in JSON output.
    #[arg(long)]
    pub scores: bool,

    /// Include metadata in JSON output.
    #[arg(long)]
    pub metadata: bool,

    /// Server URL (also reads `RAGREP_SERVER` env). Omit for local mode.
    #[arg(long, env = "RAGREP_SERVER")]
    pub server: Option<String>,

    /// Path to config.toml (defaults to CWD, RAGREP_CONFIG, ~/.config/ragrep/).
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: SearchArgs) -> Result<()> {
    eprintln!("ragrep search: not yet implemented in the Rust port (Phase 1)");
    std::process::exit(1)
}
