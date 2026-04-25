use std::error::Error;
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Instant;

use clap::Parser;
use ragrep::index;
use serde::Deserialize;

const EMBED_DIM: usize = 1024;
const TOP_K: usize = 10;

#[derive(Parser)]
#[command(name = "ragrep", about = "Phase 0.1 spike: mmap + brute-force IP search")]
struct Args {
    #[arg(long, value_name = "PATH")]
    query: PathBuf,

    #[arg(long, default_value = "data/index/embeddings.bin")]
    embeddings: PathBuf,

    #[arg(long, default_value_t = 0)]
    bench: usize,
}

#[derive(Deserialize)]
struct Query {
    query: String,
    embedding: Vec<f32>,
    top10: Vec<TopEntry>,
}

#[derive(Deserialize)]
struct TopEntry {
    idx: u32,
    score: f32,
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}

fn run() -> Result<ExitCode, Box<dyn Error>> {
    let args = Args::parse();

    let t = Instant::now();
    let flat = index::flat::Flat::open(&args.embeddings, EMBED_DIM)?;
    eprintln!("mmap: n={} dim={} load={:?}", flat.n, flat.dim, t.elapsed());

    let q: Query = serde_json::from_slice(&std::fs::read(&args.query)?)?;
    eprintln!("query: {:?}", q.query);

    if args.bench > 0 {
        let _ = flat.search(&q.embedding, TOP_K);
        let mut times = Vec::with_capacity(args.bench);
        for _ in 0..args.bench {
            let t = Instant::now();
            let _ = flat.search(&q.embedding, TOP_K);
            times.push(t.elapsed());
        }
        times.sort();
        let p50 = times[times.len() / 2];
        eprintln!("bench {}x: min={:?} p50={:?} max={:?}", args.bench, times[0], p50, times[times.len() - 1]);
        return Ok(ExitCode::SUCCESS);
    }

    let t = Instant::now();
    let results = flat.search(&q.embedding, TOP_K);
    eprintln!("search: {:?}", t.elapsed());

    println!("rank | rust_idx  | rust_score | py_idx | py_score   | match");
    let mut all_match = true;
    for (rank, ((idx, score), expected)) in results.iter().zip(q.top10.iter()).enumerate() {
        let m = if *idx == expected.idx { "OK" } else { "DIFF" };
        if m == "DIFF" {
            all_match = false;
        }
        println!(
            "{:4} | {:9} | {:10.6} | {:6} | {:10.6} | {}",
            rank + 1,
            idx,
            score,
            expected.idx,
            expected.score,
            m
        );
    }

    Ok(if all_match { ExitCode::SUCCESS } else { ExitCode::from(1) })
}
