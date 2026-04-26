use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use clap::Args;
use serde::Serialize;

use crate::config;
use crate::embed;
use crate::index::store;
use crate::models::{MetaValue, SearchResult};
use crate::query;
use crate::splash;

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

    /// Server URL. Omit for local mode.
    #[arg(long, env = "RAGREP_SERVER")]
    pub server: Option<String>,

    /// Path to config.toml.
    #[arg(long)]
    pub config: Option<PathBuf>,
}

pub fn run(args: SearchArgs) -> Result<()> {
    let wall_start = Instant::now();

    if let Some(server) = args.server.as_deref() {
        run_proxy(server, &args)?;
        eprintln!(
            "\n({:.2}s wall, server mode)",
            wall_start.elapsed().as_secs_f32()
        );
        return Ok(());
    }

    // Hide rayon's first-call thread-pool init (~6ms) behind config + msgpack
    // load so the search loop measures the steady-state ~2ms instead of ~8ms.
    std::thread::spawn(|| {
        use rayon::prelude::*;
        [0u8; 1].par_iter().for_each(|_| ());
    });

    let cfg = config::load(args.config.as_deref())?;
    let dir = cfg.index_dir();

    match args.mode.as_str() {
        "grep" => run_grep(&dir, &args),
        "semantic" => run_semantic(&dir, &cfg, &args),
        "hybrid" => run_hybrid(&dir, &cfg, &args),
        m => bail!("unknown mode: {m}"),
    }?;

    eprintln!(
        "\n({:.2}s wall, local mode)",
        wall_start.elapsed().as_secs_f32()
    );
    Ok(())
}

fn run_proxy(server: &str, args: &SearchArgs) -> Result<()> {
    use std::time::Duration;

    let url = format!("{}/search", server.trim_end_matches('/'));
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(180))
        .build()
        .context("building HTTP client")?;

    let n_str = args.n.to_string();
    let context_str = args.context.to_string();
    let full_str = args.full.to_string();
    let scores_str = args.scores.to_string();
    let metadata_str = args.metadata.to_string();
    let mut params: Vec<(&str, &str)> = vec![
        ("q", args.term.as_str()),
        ("mode", &args.mode),
        ("n", &n_str),
        ("context", &context_str),
        ("full", &full_str),
        ("scores", &scores_str),
        ("metadata", &metadata_str),
    ];
    if let Some(s) = args.source.as_deref() {
        params.push(("source", s));
    }
    let filter_joined = args.filter.join(",");
    if !filter_joined.is_empty() {
        params.push(("filter", &filter_joined));
    }
    if let Some(a) = args.after.as_deref() {
        params.push(("after", a));
    }
    if let Some(b) = args.before.as_deref() {
        params.push(("before", b));
    }

    let mut request = client.get(&url).query(&params);
    if let Some(token) = cloud_run_identity_token(server) {
        request = request.bearer_auth(token);
    }
    let resp = request
        .send()
        .with_context(|| format!("contacting server at {server}"))?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().unwrap_or_default();
        bail!("server returned {status}: {body}");
    }
    let body = resp.text().context("reading server response body")?;

    if args.json {
        println!("{body}");
        return Ok(());
    }

    let payload: ServerPayload =
        serde_json::from_str(&body).context("parsing server response (expected JSON)")?;

    if payload.mode == "grep" {
        if let Some(t) = payload.total_matches {
            println!("{} chunks match '{}'\n", t, args.term);
        }
    } else {
        println!(
            "Top {} results for '{}'\n",
            payload.results.len(),
            args.term
        );
    }
    for r in &payload.results {
        let score_str = if args.scores {
            server_scores_inline(r)
        } else {
            String::new()
        };
        println!(
            "  [{}] [{}] {}  {}",
            r.rank,
            r.source,
            truncate_title(&r.title),
            score_str
        );
        println!("      id: {}", r.id);
        if let Some(c) = &r.content {
            println!("      {}", c);
        } else if let Some(s) = &r.snippet {
            println!("      {}", s);
        }
    }
    Ok(())
}

fn cloud_run_identity_token(server: &str) -> Option<String> {
    if !server.contains(".run.app") {
        return None;
    }
    let output = std::process::Command::new("gcloud")
        .args(["auth", "print-identity-token"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let token = String::from_utf8(output.stdout).ok()?;
    let token = token.trim();
    if token.is_empty() {
        None
    } else {
        Some(token.to_string())
    }
}

#[derive(serde::Deserialize)]
struct ServerPayload {
    mode: String,
    #[serde(default)]
    total_matches: Option<usize>,
    results: Vec<ServerResult>,
}

#[derive(serde::Deserialize)]
struct ServerResult {
    rank: usize,
    id: String,
    source: String,
    title: String,
    #[serde(default)]
    snippet: Option<String>,
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    rerank: Option<f32>,
    #[serde(default)]
    rrf: Option<f32>,
    #[serde(default)]
    dense: Option<f32>,
    #[serde(default)]
    bm25: Option<f32>,
}

fn server_scores_inline(r: &ServerResult) -> String {
    let mut parts = Vec::new();
    if let Some(v) = r.rerank {
        parts.push(format!("rerank={v:.3}"));
    }
    if let Some(v) = r.rrf {
        parts.push(format!("rrf={v:.3}"));
    }
    if let Some(v) = r.dense {
        parts.push(format!("dense={v:.3}"));
    }
    if let Some(v) = r.bm25 {
        parts.push(format!("bm25={v:.3}"));
    }
    parts.join("  ")
}

fn run_grep(dir: &std::path::Path, args: &SearchArgs) -> Result<()> {
    if !store::chunks_exist(dir) {
        // Fall back to splash if the user has nothing to search.
        splash::print();
        return Ok(());
    }

    let chunks = store::load_chunks(dir)?;
    let after = args
        .after
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let before = args
        .before
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let filt = query::Filters {
        source: args.source.as_deref(),
        metadata: query::filters::parse_filters(&args.filter)?,
        after: after.as_deref(),
        before: before.as_deref(),
    };
    let result = query::grep(&chunks, &args.term, &filt, args.n);

    if args.json {
        print_json(
            &result.query,
            "grep",
            Some(result.total_matches),
            &result.hits,
            args,
        )?;
    } else {
        print_human(
            &result.query,
            "grep",
            Some(result.total_matches),
            &result.hits,
            args,
        );
    }
    Ok(())
}

fn run_semantic(
    dir: &std::path::Path,
    cfg: &crate::config::Config,
    args: &SearchArgs,
) -> Result<()> {
    if !store::chunks_exist(dir) || !store::embeddings_exist(dir) {
        splash::print();
        return Ok(());
    }

    let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?;
    let chunks = store::load_chunks(dir)?;
    let flat = store::load_flat(dir, embedder.dim())?;
    let query_emb = embedder.embed_query(&args.term)?;

    let after = args
        .after
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let before = args
        .before
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let filt = query::Filters {
        source: args.source.as_deref(),
        metadata: query::filters::parse_filters(&args.filter)?,
        after: after.as_deref(),
        before: before.as_deref(),
    };
    let result = query::semantic(&flat, &chunks, &args.term, &query_emb, &filt, args.n);

    if args.json {
        print_json(&result.query, "semantic", None, &result.hits, args)?;
    } else {
        print_human(&result.query, "semantic", None, &result.hits, args);
    }
    Ok(())
}

fn run_hybrid(dir: &std::path::Path, cfg: &crate::config::Config, args: &SearchArgs) -> Result<()> {
    if !store::chunks_exist(dir) || !store::embeddings_exist(dir) {
        splash::print();
        return Ok(());
    }
    if !store::bm25_exists(dir) {
        bail!("bm25.msgpack missing — run `ragrep rebuild-bm25` first");
    }

    let embedder = embed::make(&cfg.embedding.provider, &cfg.embedding.model_name)?;
    let chunks = store::load_chunks(dir)?;
    let flat = store::load_flat(dir, embedder.dim())?;
    let bm25_idx = store::load_bm25(dir)?;
    let query_emb = embedder.embed_query(&args.term)?;

    let after = args
        .after
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let before = args
        .before
        .as_deref()
        .map(query::filters::parse_date)
        .transpose()?;
    let result = query::hybrid(
        &flat,
        &bm25_idx,
        &chunks,
        &args.term,
        &query_emb,
        query::HybridOpts {
            n: args.n,
            top_k_dense: cfg.retrieval.top_k_dense,
            top_k_bm25: cfg.retrieval.top_k_bm25,
            // Match Python's `fetch_n = max(n*4, 20)` in search.py::search_hybrid.
            rerank_pool: (args.n * 4).max(20),
            rrf_k: cfg.retrieval.rrf_k,
            rerank_provider: &cfg.reranker.provider,
            rerank_model: &cfg.reranker.model_name,
            filters: query::Filters {
                source: args.source.as_deref(),
                metadata: query::filters::parse_filters(&args.filter)?,
                after: after.as_deref(),
                before: before.as_deref(),
            },
        },
    )?;

    if args.json {
        print_json(&result.query, "hybrid", None, &result.hits, args)?;
    } else {
        print_human(&result.query, "hybrid", None, &result.hits, args);
    }
    Ok(())
}

fn print_human(
    term: &str,
    mode: &str,
    total: Option<usize>,
    hits: &[query::Hit],
    args: &SearchArgs,
) {
    if mode == "grep" {
        if let Some(t) = total {
            println!("{} chunks match '{}'\n", t, term);
        }
    } else {
        println!("Top {} results for '{}'\n", hits.len(), term);
    }
    for h in hits {
        let r = &h.result;
        let score_str = if args.scores {
            scores_inline(r)
        } else {
            String::new()
        };
        println!(
            "  [{}] [{}] {}  {}",
            h.rank,
            r.source,
            truncate_title(&r.title),
            score_str
        );
        println!("      id: {}", r.chunk_id);
        if args.full {
            println!("      {}", r.content);
        } else if args.context > 0 {
            println!("      {}", snippet(&r.content, args.context, term));
        }
    }
}

fn scores_inline(r: &SearchResult) -> String {
    let mut parts = Vec::new();
    if r.rerank_score != 0.0 {
        parts.push(format!("rerank={:.3}", r.rerank_score));
    }
    if r.rrf_score != 0.0 {
        parts.push(format!("rrf={:.3}", r.rrf_score));
    }
    if r.dense_score != 0.0 {
        parts.push(format!("dense={:.3}", r.dense_score));
    }
    if r.bm25_score != 0.0 {
        parts.push(format!("bm25={:.3}", r.bm25_score));
    }
    parts.join("  ")
}

#[derive(Serialize)]
struct JsonHit<'a> {
    rank: usize,
    id: &'a str,
    source: &'a str,
    title: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    snippet: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rerank: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    rrf: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bm25: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    metadata: Option<&'a std::collections::BTreeMap<String, MetaValue>>,
}

#[derive(Serialize)]
struct JsonOut<'a> {
    query: &'a str,
    mode: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_matches: Option<usize>,
    results: Vec<JsonHit<'a>>,
}

fn print_json(
    query: &str,
    mode: &str,
    total: Option<usize>,
    hits: &[query::Hit],
    args: &SearchArgs,
) -> Result<()> {
    let context = if args.json && !args.full && args.context == 200 {
        // Mirror Python: in JSON mode, default to no snippet unless explicitly set.
        0
    } else {
        args.context
    };

    let results: Vec<JsonHit> = hits
        .iter()
        .map(|h| {
            let r = &h.result;
            let title = truncate_title(&r.title);
            let (snippet_v, content_v) = if args.full {
                (None, Some(r.content.as_str()))
            } else if context > 0 {
                (Some(snippet(&r.content, context, query)), None)
            } else {
                (None, None)
            };
            JsonHit {
                rank: h.rank,
                id: r.chunk_id.as_str(),
                source: r.source.as_str(),
                title,
                snippet: snippet_v,
                content: content_v,
                rerank: opt_score(r.rerank_score, args.scores),
                rrf: opt_score(r.rrf_score, args.scores),
                dense: opt_score(r.dense_score, args.scores),
                bm25: opt_score(r.bm25_score, args.scores),
                metadata: if args.metadata {
                    Some(&r.metadata)
                } else {
                    None
                },
            }
        })
        .collect();

    let out = JsonOut {
        query,
        mode,
        total_matches: total,
        results,
    };
    println!("{}", serde_json::to_string_pretty(&out)?);
    Ok(())
}

fn truncate_title(title: &str) -> String {
    let chars: Vec<char> = title.chars().collect();
    if chars.len() <= 80 {
        title.to_string()
    } else {
        let mut s: String = chars[..77].iter().collect();
        s.push_str("...");
        s
    }
}

fn opt_score(v: f32, on: bool) -> Option<f32> {
    if on && v != 0.0 {
        Some((v * 1000.0).round() / 1000.0)
    } else {
        None
    }
}

fn snippet(content: &str, length: usize, term: &str) -> String {
    let flat: String = content
        .chars()
        .map(|c| if c == '\n' { ' ' } else { c })
        .collect();
    let trimmed = flat.trim();
    if length == 0 || length >= trimmed.chars().count() {
        return trimmed.to_string();
    }
    let lower = trimmed.to_lowercase();
    if let Some(pos) = lower.find(&term.to_lowercase()) {
        // Approximate centering on `term`. Convert byte pos to char pos.
        let prefix_chars = trimmed[..pos].chars().count();
        let half_back = length / 4;
        let start = prefix_chars.saturating_sub(half_back);
        let chars: Vec<char> = trimmed.chars().collect();
        let end = (start + length).min(chars.len());
        let mut s: String = chars[start..end].iter().collect();
        if start > 0 {
            s.insert_str(0, "...");
        }
        if end < chars.len() {
            s.push_str("...");
        }
        return s;
    }
    let chars: Vec<char> = trimmed.chars().collect();
    let mut s: String = chars[..length.min(chars.len())].iter().collect();
    if chars.len() > length {
        s.push_str("...");
    }
    s
}
