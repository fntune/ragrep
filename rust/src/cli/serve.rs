use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Port to bind.
    #[arg(long, default_value_t = 8321)]
    pub port: u16,

    /// Host to bind.
    #[arg(long, default_value = "0.0.0.0")]
    pub host: String,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(_args: ServeArgs) -> Result<()> {
    eprintln!("ragrep serve: not yet implemented in the Rust port (Phase 4)");
    std::process::exit(1)
}
