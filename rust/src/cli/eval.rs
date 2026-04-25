use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Save results to file.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: EvalArgs) -> Result<()> {
    eprintln!("ragrep eval: not yet implemented in the Rust port (Phase 4)");
    std::process::exit(1)
}
