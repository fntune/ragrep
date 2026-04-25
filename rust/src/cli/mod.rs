use anyhow::Result;
use clap::{Parser, Subcommand};

pub mod eval;
pub mod ingest;
pub mod inspect;
pub mod rebuild_bm25;
pub mod scrape;
pub mod search;
pub mod serve;
pub mod stats;

/// Top-level CLI: search is the implicit default subcommand. With no args,
/// shows the welcome splash. With a positional term, runs `search`. With
/// any explicit subcommand, dispatches to it.
#[derive(Parser, Debug)]
#[command(
    name = "ragrep",
    version,
    about = "ripgrep for your team's knowledge base — hybrid retrieval, self-hosted",
    arg_required_else_help = false,
)]
pub struct Cli {
    /// Search term (default subcommand). Required unless an explicit subcommand is used.
    pub term: Option<String>,

    /// Search options (apply when no explicit subcommand is given).
    #[command(flatten)]
    pub search_opts: SearchOpts,

    #[command(subcommand)]
    pub cmd: Option<Cmd>,
}

#[derive(Subcommand, Debug)]
pub enum Cmd {
    /// Run the HTTP search server (replaces FastAPI/uvicorn).
    Serve(serve::ServeArgs),
    /// Build the FAISS + BM25 index from raw scraped data.
    Ingest(ingest::IngestArgs),
    /// Scrape sources into data/raw/.
    Scrape(scrape::ScrapeArgs),
    /// Run the evaluation harness.
    Eval(eval::EvalArgs),
    /// Show index statistics.
    Stats(stats::StatsArgs),
    /// Inspect raw data and pipeline output.
    Inspect(inspect::InspectArgs),
    /// Rebuild bm25.msgpack from chunks.msgpack (transitional, for pickle-migrated indexes).
    #[command(name = "rebuild-bm25")]
    RebuildBm25(rebuild_bm25::RebuildBm25Args),
}

#[derive(clap::Args, Debug)]
pub struct SearchOpts {
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

    /// Server URL. Omit for local mode.
    #[arg(long, env = "RAGREP_SERVER")]
    pub server: Option<String>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<std::path::PathBuf>,
}

pub fn run() -> Result<()> {
    crate::config::load_env_files();
    let cli = Cli::parse();

    match cli.cmd {
        Some(Cmd::Serve(a)) => serve::run(a),
        Some(Cmd::Ingest(a)) => ingest::run(a),
        Some(Cmd::Scrape(a)) => scrape::run(a),
        Some(Cmd::Eval(a)) => eval::run(a),
        Some(Cmd::Stats(a)) => stats::run(a),
        Some(Cmd::Inspect(a)) => inspect::run(a),
        Some(Cmd::RebuildBm25(a)) => rebuild_bm25::run(a),
        None => match cli.term {
            Some(term) => search::run(search::SearchArgs {
                term,
                n: cli.search_opts.n,
                source: cli.search_opts.source,
                mode: cli.search_opts.mode,
                filter: cli.search_opts.filter,
                after: cli.search_opts.after,
                before: cli.search_opts.before,
                context: cli.search_opts.context,
                full: cli.search_opts.full,
                json: cli.search_opts.json,
                scores: cli.search_opts.scores,
                metadata: cli.search_opts.metadata,
                server: cli.search_opts.server,
                config: cli.search_opts.config,
            }),
            None => {
                crate::splash::print();
                Ok(())
            }
        },
    }
}
