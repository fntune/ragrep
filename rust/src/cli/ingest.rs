use std::path::PathBuf;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::ingest::pipeline;

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

pub fn run(args: IngestArgs) -> Result<()> {
    // Pre-warm rayon (chunk + future parallel-normalize benefit) the same way
    // cli::search::run does — fire-and-forget so its init overlaps with config load.
    std::thread::spawn(|| {
        use rayon::prelude::*;
        [0u8; 1].par_iter().for_each(|_| ());
    });

    init_tracing();
    let cfg = config::load(args.config.as_deref())?;
    let stats = pipeline::run(&cfg, args.force, args.source.as_deref())?;

    if stats.documents == 0 {
        eprintln!("(no documents found in {})", cfg.raw_dir().display());
        return Ok(());
    }

    println!(
        "\nIngestion complete: {} docs → {} chunks in {:.1}s",
        stats.documents, stats.chunks, stats.elapsed_s
    );
    for (source, count) in &stats.sources {
        println!("  {source}: {count}");
    }
    Ok(())
}

fn init_tracing() {
    use tracing_subscriber::{fmt, EnvFilter};
    let _ = fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .with_target(false)
        .with_writer(std::io::stderr)
        .try_init();
}
