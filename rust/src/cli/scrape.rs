use std::path::PathBuf;

use anyhow::{bail, Result};
use clap::Args;

use crate::config;
use crate::ingest::extract;
use crate::ingest::scrape;

#[derive(Args, Debug)]
pub struct ScrapeArgs {
    /// Comma-separated sources: slack,atlassian,gdrive,git,bitbucket,code,files.
    #[arg(short, long)]
    pub source: Option<String>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: ScrapeArgs) -> Result<()> {
    let cfg = config::load(args.config.as_deref())?;
    let raw_dir = cfg.raw_dir();
    let sources = parse_sources(args.source.as_deref())?;

    for source in sources {
        match source.as_str() {
            "git" => {
                scrape::git::scrape(&raw_dir, &cfg.scrape.git)?;
            }
            "code" => {
                scrape::code::scrape(&raw_dir, &cfg.scrape.code)?;
            }
            "slack" => {
                scrape::slack::scrape(&raw_dir, &cfg.scrape.slack)?;
            }
            "atlassian" => {
                scrape::atlassian::scrape(&raw_dir, &cfg.scrape.atlassian)?;
            }
            "gdrive" => bail!("ragrep scrape gdrive is not yet implemented in the Rust port"),
            "bitbucket" => {
                scrape::bitbucket::scrape(&raw_dir, &cfg.scrape.bitbucket)?;
            }
            "files" => {
                extract::extract_all(&raw_dir)?;
            }
            other => bail!("unknown scrape source: {other}"),
        }
    }

    println!("\nScrape complete. Raw data in {}/", raw_dir.display());
    Ok(())
}

fn parse_sources(raw: Option<&str>) -> Result<Vec<String>> {
    let sources = raw
        .map(|s| {
            s.split(',')
                .map(str::trim)
                .filter(|s| !s.is_empty())
                .map(ToString::to_string)
                .collect::<Vec<_>>()
        })
        .unwrap_or_else(|| {
            vec![
                "slack".to_string(),
                "atlassian".to_string(),
                "gdrive".to_string(),
                "git".to_string(),
                "bitbucket".to_string(),
                "code".to_string(),
                "files".to_string(),
            ]
        });

    if sources.is_empty() {
        bail!("no scrape sources selected");
    }
    Ok(sources)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_csv_sources() {
        assert_eq!(
            parse_sources(Some("git, code")).unwrap(),
            vec!["git".to_string(), "code".to_string()]
        );
    }
}
