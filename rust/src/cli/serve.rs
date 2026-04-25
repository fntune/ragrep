use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use clap::Args;

use crate::config;
use crate::serve;

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

pub fn run(args: ServeArgs) -> Result<()> {
    init_tracing();

    // Pre-warm rayon (the search-path's parallel inner-product loop needs it).
    std::thread::spawn(|| {
        use rayon::prelude::*;
        [0u8; 1].par_iter().for_each(|_| ());
    });

    let cfg = config::load(args.config.as_deref())?;
    tracing::info!(
        target: "ragrep::serve",
        "loading index from {}",
        cfg.index_dir().display()
    );
    let state = Arc::new(serve::AppState::load(cfg)?);
    tracing::info!(
        target: "ragrep::serve",
        "ready: {} chunks, embedder={}/{} dim={}",
        state.chunks.len(),
        state.embedder.provider(),
        state.embedder.model(),
        state.embedder.dim()
    );

    let app = serve::router(state);
    let bind = format!("{}:{}", args.host, args.port);

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()?;
    rt.block_on(async move {
        let listener = tokio::net::TcpListener::bind(&bind).await?;
        tracing::info!(target: "ragrep::serve", "listening on http://{bind}");
        axum::serve(listener, app).await?;
        Ok::<(), anyhow::Error>(())
    })?;

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
