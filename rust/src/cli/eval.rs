use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::eval::harness;

#[derive(Args, Debug)]
pub struct EvalArgs {
    /// Save results to file.
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: EvalArgs) -> Result<()> {
    let cfg = config::load(args.config.as_deref())?;
    harness::evaluate(&cfg, args.output.as_deref())?;
    Ok(())
}
