use std::path::PathBuf;
use std::{
    collections::{BTreeMap, HashSet},
    fs,
    io::{BufRead, BufReader},
    time::{SystemTime, UNIX_EPOCH},
};

use anyhow::{Context, Result};
use clap::Args;
use sha2::{Digest, Sha256};

use crate::config;
use crate::ingest::{chunk, normalize};
use crate::models::Chunk;

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

pub fn run(args: InspectArgs) -> Result<()> {
    let cfg = config::load(args.config.as_deref())?;

    if args.mode == "raw" {
        return inspect_raw(&cfg.raw_dir());
    }

    let mut docs = normalize::normalize_all(&cfg.raw_dir())?;
    if let Some(source) = args.source.as_deref() {
        docs.retain(|doc| doc.source == source);
    }
    let chunks = chunk::all(
        &docs,
        cfg.ingest.max_chunk_tokens,
        cfg.ingest.chunk_overlap_tokens,
    );

    match args.mode.as_str() {
        "docs" => inspect_docs(&docs, &chunks),
        "sample" => inspect_sample(
            &chunks,
            args.grep.as_deref(),
            args.n.unwrap_or(3),
            args.full,
        ),
        "grep" => inspect_grep(&chunks, args.grep.as_deref(), args.n.unwrap_or(5)),
        _ => unreachable!("clap restricts inspect mode"),
    }
}

fn inspect_raw(raw_dir: &std::path::Path) -> Result<()> {
    let mut files = if raw_dir.is_dir() {
        fs::read_dir(raw_dir)
            .with_context(|| format!("reading {}", raw_dir.display()))?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.extension().is_some_and(|ext| ext == "jsonl"))
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    files.sort();

    println!("{:<30} {:>8} {:>8}", "file", "size", "records");
    println!("{}", "-".repeat(50));
    for path in files {
        let size = fs::metadata(&path)?.len();
        let records = if size == 0 {
            0
        } else {
            count_nonempty_lines(&path)?
        };
        let name = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>");
        println!("{name:<30} {:>8} {:>8}", human_size(size), records);
    }
    Ok(())
}

fn inspect_docs(docs: &[crate::models::Document], chunks: &[Chunk]) -> Result<()> {
    let doc_sources = count_sources(docs.iter().map(|d| d.source.as_str()));
    let chunk_sources = count_sources(chunks.iter().map(|c| c.source.as_str()));

    println!(
        "{:<12} {:>8} {:>8} {:>6}",
        "source", "docs", "chunks", "ratio"
    );
    println!("{}", "-".repeat(38));

    let mut rows: Vec<(&str, usize)> = doc_sources.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (source, docs_count) in rows {
        let chunk_count = chunk_sources.get(source).copied().unwrap_or(0);
        println!(
            "{source:<12} {docs_count:>8} {chunk_count:>8} {:>5.1}x",
            ratio(chunk_count, docs_count)
        );
    }
    println!("{}", "-".repeat(38));
    println!(
        "{:<12} {:>8} {:>8} {:>5.1}x",
        "TOTAL",
        docs.len(),
        chunks.len(),
        ratio(chunks.len(), docs.len())
    );

    if chunks.is_empty() {
        println!("\nchunk chars: n/a");
        println!("chunk tokens: n/a");
        println!("\nunique hashes: 0 / 0 chunks (0 dupes)");
        return Ok(());
    }

    let lens: Vec<usize> = chunks.iter().map(|c| c.content.len()).collect();
    let toks: Vec<usize> = chunks.iter().map(|c| approx_tokens(&c.content)).collect();
    println!(
        "\nchunk chars: min={} med={} p90={} max={}",
        lens.iter().min().unwrap(),
        percentile(&lens, 50.0),
        percentile(&lens, 90.0),
        lens.iter().max().unwrap()
    );
    println!(
        "chunk tokens: min={} med={} p90={} max={}",
        toks.iter().min().unwrap(),
        percentile(&toks, 50.0),
        percentile(&toks, 90.0),
        toks.iter().max().unwrap()
    );

    let hashes = chunks
        .iter()
        .map(|c| {
            let digest = Sha256::digest(c.content.as_bytes());
            digest.to_vec()
        })
        .collect::<HashSet<_>>();
    println!(
        "\nunique hashes: {} / {} chunks ({} dupes)",
        hashes.len(),
        chunks.len(),
        chunks.len() - hashes.len()
    );
    Ok(())
}

fn inspect_sample(chunks: &[Chunk], grep: Option<&str>, n: usize, full: bool) -> Result<()> {
    let sample: Vec<&Chunk> = if let Some(needle) = grep {
        let needle = needle.to_lowercase();
        let pool = chunks
            .iter()
            .filter(|chunk| chunk.content.to_lowercase().contains(&needle))
            .take(n)
            .collect::<Vec<_>>();
        if pool.is_empty() {
            println!("No chunks matching '{}'", grep.unwrap());
            return Ok(());
        }
        pool
    } else {
        sample_indices(chunks.len(), n)
            .into_iter()
            .map(|idx| &chunks[idx])
            .collect()
    };

    for (i, chunk) in sample.iter().enumerate() {
        println!("--- [{}/{}] {} ---", i + 1, sample.len(), chunk.id);
        println!("source:   {}", chunk.source);
        println!("title:    {}", chunk.title);
        println!("doc_id:   {}", chunk.doc_id);
        println!("metadata: {}", serde_json::to_string(&chunk.metadata)?);
        if full {
            println!("content:  ({} chars)", chunk.content.len());
            println!("{}", chunk.content);
        } else {
            println!(
                "content:  ({} chars, showing first 500)",
                chunk.content.len()
            );
            let preview = safe_prefix(&chunk.content, 500);
            println!("{preview}");
            if chunk.content.len() > preview.len() {
                println!("...");
            }
        }
        println!();
    }
    Ok(())
}

fn inspect_grep(chunks: &[Chunk], grep: Option<&str>, n: usize) -> Result<()> {
    let Some(needle) = grep else {
        println!("--grep required for grep mode");
        return Ok(());
    };
    let needle_lower = needle.to_lowercase();
    let mut matches = Vec::new();
    for chunk in chunks {
        if let Some(pos) = chunk.content.to_lowercase().find(&needle_lower) {
            matches.push((chunk, pos));
        }
    }

    println!("{} chunks match '{}'\n", matches.len(), needle);
    let source_counts = count_sources(matches.iter().map(|(chunk, _)| chunk.source.as_str()));
    let mut rows: Vec<(&str, usize)> = source_counts.into_iter().collect();
    rows.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    for (source, count) in rows {
        println!("  {source}: {count}");
    }

    if matches.is_empty() {
        return Ok(());
    }

    println!("\nTop {} matches:", n.min(matches.len()));
    for (chunk, pos) in matches.into_iter().take(n) {
        let start = pos.saturating_sub(40);
        let end = pos + needle.len() + 40;
        let snippet = safe_slice(&chunk.content, start, end).replace('\n', " ");
        println!(
            "  [{}] {}  ...{}...",
            chunk.source,
            truncate(&chunk.title, 50),
            snippet
        );
    }
    Ok(())
}

fn count_nonempty_lines(path: &std::path::Path) -> Result<usize> {
    let file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let reader = BufReader::new(file);
    let mut count = 0usize;
    for line in reader.lines() {
        if !line?.trim().is_empty() {
            count += 1;
        }
    }
    Ok(count)
}

fn count_sources<'a>(items: impl Iterator<Item = &'a str>) -> BTreeMap<&'a str, usize> {
    let mut counts = BTreeMap::new();
    for source in items {
        *counts.entry(source).or_insert(0) += 1;
    }
    counts
}

fn human_size(bytes: u64) -> String {
    if bytes > 1_048_576 {
        format!("{:.1}M", bytes as f64 / 1_048_576.0)
    } else {
        format!("{:.0}K", bytes as f64 / 1024.0)
    }
}

fn ratio(num: usize, den: usize) -> f64 {
    if den == 0 {
        0.0
    } else {
        num as f64 / den as f64
    }
}

fn approx_tokens(text: &str) -> usize {
    text.split_whitespace().count().max(text.len() / 4)
}

fn percentile(values: &[usize], p: f64) -> usize {
    if values.is_empty() {
        return 0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_unstable();
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (p / 100.0) * (sorted.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - lo as f64;
        (sorted[lo] as f64 + (sorted[hi] as f64 - sorted[lo] as f64) * frac) as usize
    }
}

fn sample_indices(len: usize, n: usize) -> Vec<usize> {
    if len == 0 || n == 0 {
        return Vec::new();
    }
    let take = n.min(len);
    let mut idx: Vec<usize> = (0..len).collect();
    let mut seed = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x5241_4752);
    for i in 0..take {
        seed = xorshift64(seed);
        let j = i + (seed as usize % (len - i));
        idx.swap(i, j);
    }
    idx.truncate(take);
    idx
}

fn xorshift64(mut x: u64) -> u64 {
    if x == 0 {
        x = 0x5241_4752;
    }
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    x
}

fn safe_prefix(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn safe_slice(s: &str, start: usize, end: usize) -> &str {
    let mut start = start.min(s.len());
    while start > 0 && !s.is_char_boundary(start) {
        start -= 1;
    }
    let mut end = end.min(s.len());
    while end < s.len() && !s.is_char_boundary(end) {
        end += 1;
    }
    &s[start..end]
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    safe_prefix(s, max).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_interpolates_like_numpy_style_values() {
        assert_eq!(percentile(&[1, 2, 3, 4], 50.0), 2);
        assert_eq!(percentile(&[10, 20, 30], 90.0), 28);
    }

    #[test]
    fn safe_prefix_respects_char_boundaries() {
        assert_eq!(safe_prefix("éclair", 1), "");
        assert_eq!(safe_prefix("éclair", 2), "é");
    }
}
