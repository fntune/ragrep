//! Evaluation harness: measures cross-source retrieval quality per stage.

use std::collections::BTreeSet;
use std::path::Path;
use std::time::Instant;

use anyhow::{bail, Context, Result};
use serde::Serialize;

use crate::config::Config;
use crate::embed;
use crate::eval::entity::{EntityGraph, EvalCase};
use crate::index::{bm25::Bm25, flat::Flat, store};
use crate::models::Chunk;
use crate::query::{self, retrieve};

const MAX_CASES: usize = 50;

#[derive(Debug, Clone, Default, Serialize)]
pub struct StageMetrics {
    pub source_recall: f64,
    pub entity_recall: f64,
    pub mrr: f64,
    pub sources_found: BTreeSet<String>,
    pub n_hits: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct CaseResult {
    pub entity_key: String,
    pub query: String,
    pub expected_sources: BTreeSet<String>,
    pub n_ground_truth: usize,
    pub stages: std::collections::BTreeMap<String, StageMetrics>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EvalSummary {
    pub cases: usize,
    pub elapsed_s: f64,
    pub results: Vec<CaseResult>,
}

pub fn evaluate(config: &Config, output_path: Option<&Path>) -> Result<EvalSummary> {
    let rc = &config.retrieval;
    let index_dir = config.index_dir();
    if !store::index_exists(&index_dir) {
        bail!(
            "no complete Rust index found at {} (run `ragrep ingest` first)",
            index_dir.display()
        );
    }

    eprintln!("Loading index from {}", index_dir.display());
    let embedder = embed::make(&config.embedding.provider, &config.embedding.model_name)?;
    let chunks = store::load_chunks(&index_dir)?;
    let flat = store::load_flat(&index_dir, embedder.dim())?;
    let bm25 = store::load_bm25(&index_dir)?;

    eprintln!("Building entity graph from {} chunks", chunks.len());
    let graph = EntityGraph::build(&chunks);
    let cross_count = graph.cross_source_entities(2).len();
    eprintln!("Found {cross_count} cross-source entities");

    let cases = graph.eval_cases(2, MAX_CASES);
    if cases.is_empty() {
        eprintln!("No eval cases generated");
        let summary = EvalSummary {
            cases: 0,
            elapsed_s: 0.0,
            results: Vec::new(),
        };
        if let Some(path) = output_path {
            save_results(&summary, path)?;
        }
        return Ok(summary);
    }
    eprintln!("Generated {} eval cases", cases.len());

    let start = Instant::now();
    let mut results = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        let result = eval_case(
            case,
            embedder.as_ref(),
            &flat,
            &bm25,
            &chunks,
            rc.top_k_dense,
            rc.top_k_bm25,
            rc.rrf_k,
            rc.top_k_rerank,
            &config.reranker.provider,
            &config.reranker.model_name,
        )?;
        let rrf = result.stages.get("rrf").cloned().unwrap_or_default();
        let rerank = result.stages.get("rerank").cloned().unwrap_or_default();
        println!(
            "  [{:2}] src={:.0}% entity={:.0}% mrr={:.2} ({}→{} hits) — {}: {}",
            i + 1,
            rerank.source_recall * 100.0,
            rerank.entity_recall * 100.0,
            rerank.mrr,
            rrf.n_hits,
            rerank.n_hits,
            case.entity_key,
            truncate(&case.query, 60),
        );
        results.push(result);
    }

    let elapsed_s = start.elapsed().as_secs_f64();
    print_summary(&results, elapsed_s);

    let summary = EvalSummary {
        cases: results.len(),
        elapsed_s,
        results,
    };
    if let Some(path) = output_path {
        save_results(&summary, path)?;
    }
    Ok(summary)
}

#[allow(clippy::too_many_arguments)]
fn eval_case(
    case: &EvalCase,
    embedder: &dyn embed::Embedder,
    flat: &Flat,
    bm25: &Bm25,
    chunks: &[Chunk],
    top_k_dense: usize,
    top_k_bm25: usize,
    rrf_k: usize,
    top_k_rerank: usize,
    rerank_provider: &str,
    rerank_model: &str,
) -> Result<CaseResult> {
    let top_k_eval = top_k_dense + top_k_bm25;
    let mut result = CaseResult {
        entity_key: case.entity_key.clone(),
        query: case.query.clone(),
        expected_sources: case.expected_sources.clone(),
        n_ground_truth: case.ground_truth_indices.len(),
        stages: std::collections::BTreeMap::new(),
    };

    let query_emb = embedder.embed_query(&case.query)?;
    let filters = query::Filters::default();

    let dense = retrieve::dense(flat, &query_emb, chunks, top_k_dense, &filters);
    let dense_indices: Vec<usize> = dense.iter().map(|(idx, _)| *idx).collect();
    result.stages.insert(
        "dense".to_string(),
        score_stage(
            &dense_indices,
            &case.ground_truth_indices,
            &case.expected_sources,
            chunks,
            top_k_dense,
        ),
    );

    let bm25_results = retrieve::bm25(bm25, chunks, &case.query, top_k_bm25, &filters);
    let bm25_indices: Vec<usize> = bm25_results.iter().map(|(idx, _)| *idx).collect();
    result.stages.insert(
        "bm25".to_string(),
        score_stage(
            &bm25_indices,
            &case.ground_truth_indices,
            &case.expected_sources,
            chunks,
            top_k_bm25,
        ),
    );

    let fused = retrieve::rrf(&dense, &bm25_results, rrf_k);
    let rrf_indices: Vec<usize> = fused.iter().map(|(idx, _, _, _)| *idx).collect();
    result.stages.insert(
        "rrf".to_string(),
        score_stage(
            &rrf_indices,
            &case.ground_truth_indices,
            &case.expected_sources,
            chunks,
            top_k_eval,
        ),
    );

    let pool_n = top_k_eval.min(fused.len());
    let pool = &fused[..pool_n];
    let docs: Vec<&str> = pool
        .iter()
        .map(|(idx, _, _, _)| chunks[*idx].content.as_str())
        .collect();
    let reranked = query::rerank::rerank(
        rerank_provider,
        rerank_model,
        &case.query,
        &docs,
        top_k_rerank,
    )?;
    let rerank_indices: Vec<usize> = reranked
        .into_iter()
        .filter_map(|item| pool.get(item.index).map(|(idx, _, _, _)| *idx))
        .collect();
    result.stages.insert(
        "rerank".to_string(),
        score_stage(
            &rerank_indices,
            &case.ground_truth_indices,
            &case.expected_sources,
            chunks,
            top_k_rerank,
        ),
    );

    Ok(result)
}

fn score_stage(
    result_indices: &[usize],
    ground_truth: &BTreeSet<usize>,
    expected_sources: &BTreeSet<String>,
    chunks: &[Chunk],
    top_k: usize,
) -> StageMetrics {
    let top = &result_indices[..result_indices.len().min(top_k)];

    let found_sources: BTreeSet<String> = top
        .iter()
        .filter_map(|idx| chunks.get(*idx))
        .map(|chunk| chunk.source.clone())
        .collect();
    let sources_found: BTreeSet<String> = found_sources
        .intersection(expected_sources)
        .cloned()
        .collect();

    let gt_hits: Vec<usize> = top
        .iter()
        .copied()
        .filter(|idx| ground_truth.contains(idx))
        .collect();

    let mrr = top
        .iter()
        .position(|idx| ground_truth.contains(idx))
        .map(|pos| 1.0 / (pos + 1) as f64)
        .unwrap_or(0.0);

    StageMetrics {
        source_recall: if expected_sources.is_empty() {
            0.0
        } else {
            sources_found.len() as f64 / expected_sources.len() as f64
        },
        entity_recall: if ground_truth.is_empty() {
            0.0
        } else {
            gt_hits.len() as f64 / ground_truth.len() as f64
        },
        mrr,
        sources_found,
        n_hits: gt_hits.len(),
    }
}

fn print_summary(results: &[CaseResult], elapsed_s: f64) {
    println!(
        "\n{:>8}  {:>10}  {:>13}  {:>6}",
        "Stage", "SrcRecall", "EntityRecall", "MRR"
    );
    println!("{}", "-".repeat(45));

    for stage in ["dense", "bm25", "rrf", "rerank"] {
        let metrics: Vec<StageMetrics> = results
            .iter()
            .map(|r| r.stages.get(stage).cloned().unwrap_or_default())
            .collect();
        let avg_src = mean(metrics.iter().map(|m| m.source_recall));
        let avg_ent = mean(metrics.iter().map(|m| m.entity_recall));
        let avg_mrr = mean(metrics.iter().map(|m| m.mrr));
        println!(
            "  {:>6}  {:>9.1}%  {:>12.1}%  {:>6.3}",
            stage,
            avg_src * 100.0,
            avg_ent * 100.0,
            avg_mrr
        );
    }

    println!("\n{} cases evaluated in {:.1}s", results.len(), elapsed_s);
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let mut total = 0.0;
    let mut count = 0usize;
    for value in values {
        total += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        total / count as f64
    }
}

fn save_results(summary: &EvalSummary, path: &Path) -> Result<()> {
    if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let raw = serde_json::to_string_pretty(summary).context("serializing eval results")?;
    std::fs::write(path, raw).with_context(|| format!("writing {}", path.display()))?;
    println!("Results saved to {}", path.display());
    Ok(())
}

fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_string();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::{Chunk, MetaValue};
    use std::collections::BTreeMap;

    fn chunk(source: &str) -> Chunk {
        Chunk {
            id: source.to_string(),
            doc_id: source.to_string(),
            content: source.to_string(),
            title: source.to_string(),
            source: source.to_string(),
            metadata: BTreeMap::<String, MetaValue>::new(),
        }
    }

    #[test]
    fn score_stage_computes_source_entity_and_mrr() {
        let chunks = vec![chunk("git"), chunk("slack"), chunk("atlassian")];
        let expected_sources = ["git".to_string(), "atlassian".to_string()]
            .into_iter()
            .collect();
        let ground_truth = [0usize, 2usize].into_iter().collect();

        let metrics = score_stage(&[1, 2, 0], &ground_truth, &expected_sources, &chunks, 3);

        assert_eq!(metrics.source_recall, 1.0);
        assert_eq!(metrics.entity_recall, 1.0);
        assert_eq!(metrics.mrr, 0.5);
        assert_eq!(metrics.n_hits, 2);
    }

    #[test]
    fn score_stage_respects_top_k_window() {
        let chunks = vec![chunk("git"), chunk("atlassian")];
        let expected_sources = ["atlassian".to_string()].into_iter().collect();
        let ground_truth = [1usize].into_iter().collect();

        let metrics = score_stage(&[0, 1], &ground_truth, &expected_sources, &chunks, 1);

        assert_eq!(metrics.source_recall, 0.0);
        assert_eq!(metrics.entity_recall, 0.0);
        assert_eq!(metrics.mrr, 0.0);
    }
}
