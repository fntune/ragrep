"""Evaluation harness: measures cross-source retrieval quality per stage.

Uses the entity graph to generate ground truth automatically — no manual
curation needed. For each cross-source entity (Jira ticket, service name),
checks whether retrieval surfaces related chunks from multiple sources.

Metrics per stage (dense, BM25, RRF, rerank):
  - source_recall: fraction of expected sources in top-K
  - entity_recall: fraction of ground truth chunks in top-K
  - mrr: reciprocal rank of first ground truth hit
"""

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragrep.config import Config
from ragrep.eval.entity_graph import EntityGraph, EvalCase
from ragrep.ingest.embed import make_embedder
from ragrep.ingest.store import load_index
from ragrep.query.rerank import make_reranker
from ragrep.query.retrieve import _reciprocal_rank_fusion, bm25_search, dense_search

log = logging.getLogger(__name__)


@dataclass
class StageMetrics:
    """Metrics for a single retrieval stage on one eval case."""

    source_recall: float = 0.0  # fraction of expected sources in results
    entity_recall: float = 0.0  # fraction of ground truth chunks in results
    mrr: float = 0.0  # reciprocal rank of first ground truth hit
    sources_found: set[str] = field(default_factory=set)
    n_hits: int = 0


@dataclass
class CaseResult:
    """Eval result for one case across all stages."""

    entity_key: str
    query: str
    expected_sources: set[str]
    n_ground_truth: int
    stages: dict[str, StageMetrics] = field(default_factory=dict)


def _score_stage(
    result_indices: list[int],
    ground_truth: set[int],
    expected_sources: set[str],
    chunks: list,
    top_k: int,
) -> StageMetrics:
    """Score a retrieval stage's results against ground truth."""
    top = result_indices[:top_k]

    # Source recall
    found_sources = {chunks[i].source for i in top if i < len(chunks)}
    source_hits = found_sources & expected_sources
    source_recall = len(source_hits) / len(expected_sources) if expected_sources else 0

    # Entity recall
    gt_hits = [i for i in top if i in ground_truth]
    entity_recall = len(gt_hits) / len(ground_truth) if ground_truth else 0

    # MRR
    mrr = 0.0
    for rank, idx in enumerate(top, 1):
        if idx in ground_truth:
            mrr = 1.0 / rank
            break

    return StageMetrics(
        source_recall=source_recall,
        entity_recall=entity_recall,
        mrr=mrr,
        sources_found=source_hits,
        n_hits=len(gt_hits),
    )


def evaluate(config: Config, output_path: Path | None = None) -> None:
    """Run cross-source retrieval evaluation."""
    rc = config.retrieval

    # Load index
    log.info("Loading index from %s", config.index_dir)
    faiss_index, chunks, bm25 = load_index(config.index_dir)

    # Build entity graph
    log.info("Building entity graph from %d chunks", len(chunks))
    graph = EntityGraph.build(chunks)
    cross = graph.cross_source_entities()
    log.info("Found %d cross-source entities", len(cross))

    # Generate eval cases
    cases = graph.eval_cases(max_cases=50)
    if not cases:
        log.warning("No eval cases generated")
        return
    log.info("Generated %d eval cases", len(cases))

    # Load models
    embedder = make_embedder(config.embedding.provider, config.embedding.model_name, config.embedding.device)
    reranker = make_reranker(config.reranker.provider, config.reranker.model_name, config.reranker.device)

    # Run eval
    results: list[CaseResult] = []
    t0 = time.monotonic()

    for i, case in enumerate(cases, 1):
        result = _eval_case(
            case, embedder, reranker, faiss_index, bm25, chunks,
            top_k_dense=rc.top_k_dense, top_k_bm25=rc.top_k_bm25, rrf_k=rc.rrf_k,
            top_k_rerank=rc.top_k_rerank,
        )
        results.append(result)

        # Print per-case summary
        rrf = result.stages.get("rrf", StageMetrics())
        rerank = result.stages.get("rerank", StageMetrics())
        print(
            f"  [{i:2d}] src={rerank.source_recall:.0%} "
            f"entity={rerank.entity_recall:.0%} "
            f"mrr={rerank.mrr:.2f} "
            f"({rrf.n_hits}→{rerank.n_hits} hits) "
            f"— {case.entity_key}: {case.query[:60]}"
        )

    elapsed = time.monotonic() - t0

    # Summary
    _print_summary(results, elapsed)

    if output_path:
        _save_results(results, output_path)


def _eval_case(
    case: EvalCase,
    embedder: object,
    reranker: object,
    faiss_index: object,
    bm25: object,
    chunks: list,
    top_k_dense: int,
    top_k_bm25: int,
    rrf_k: int,
    top_k_rerank: int,
) -> CaseResult:
    """Evaluate a single case across all retrieval stages."""
    gt = case.ground_truth_indices
    expected = case.expected_sources
    top_k_eval = top_k_dense + top_k_bm25  # eval window for pre-rerank stages

    result = CaseResult(
        entity_key=case.entity_key,
        query=case.query,
        expected_sources=expected,
        n_ground_truth=len(gt),
    )

    # Embed query
    query_emb = embedder.embed_query(case.query)

    # Dense
    dense = dense_search(query_emb, faiss_index, chunks, top_k_dense)
    dense_indices = [idx for idx, _ in dense]
    result.stages["dense"] = _score_stage(dense_indices, gt, expected, chunks, top_k_dense)

    # BM25
    bm25_results = bm25_search(case.query, bm25, chunks, top_k_bm25)
    bm25_indices = [idx for idx, _ in bm25_results]
    result.stages["bm25"] = _score_stage(bm25_indices, gt, expected, chunks, top_k_bm25)

    # RRF
    fused = _reciprocal_rank_fusion(dense, bm25_results, rrf_k)
    rrf_indices = [idx for idx, _, _, _ in fused]
    result.stages["rrf"] = _score_stage(rrf_indices, gt, expected, chunks, top_k_eval)

    # Rerank (on top of RRF pool)
    from ragrep.models import SearchResult

    # Map chunk_id → chunk index for ground truth matching after rerank
    id_to_idx = {chunks[idx].id: idx for idx, _, _, _ in fused[:top_k_eval]}

    candidates = []
    for idx, d_score, b_score, rrf_score in fused[:top_k_eval]:
        chunk = chunks[idx]
        candidates.append(SearchResult(
            chunk_id=chunk.id,
            content=chunk.content,
            title=chunk.title,
            source=chunk.source,
            metadata=chunk.metadata,
            dense_score=d_score,
            bm25_score=b_score,
            rrf_score=rrf_score,
        ))

    reranked = reranker.rerank(case.query, candidates, top_k_rerank)
    rerank_indices = [id_to_idx[r.chunk_id] for r in reranked]
    result.stages["rerank"] = _score_stage(rerank_indices, gt, expected, chunks, top_k_rerank)

    return result


def _print_summary(results: list[CaseResult], elapsed: float) -> None:
    """Print aggregate metrics per stage."""
    stages = ["dense", "bm25", "rrf", "rerank"]

    print(f"\n{'Stage':>8s}  {'SrcRecall':>10s}  {'EntityRecall':>13s}  {'MRR':>6s}")
    print("-" * 45)

    for stage in stages:
        metrics = [r.stages.get(stage, StageMetrics()) for r in results]
        avg_src = np.mean([m.source_recall for m in metrics])
        avg_ent = np.mean([m.entity_recall for m in metrics])
        avg_mrr = np.mean([m.mrr for m in metrics])
        print(f"  {stage:>6s}  {avg_src:>10.1%}  {avg_ent:>13.1%}  {avg_mrr:>6.3f}")

    print(f"\n{len(results)} cases evaluated in {elapsed:.1f}s")


def _save_results(results: list[CaseResult], path: Path) -> None:
    """Save results as JSON for further analysis."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        case_data = {
            "entity_key": r.entity_key,
            "query": r.query,
            "expected_sources": sorted(r.expected_sources),
            "n_ground_truth": r.n_ground_truth,
            "stages": {},
        }
        for stage, m in r.stages.items():
            case_data["stages"][stage] = {
                "source_recall": round(m.source_recall, 3),
                "entity_recall": round(m.entity_recall, 3),
                "mrr": round(m.mrr, 3),
                "sources_found": sorted(m.sources_found),
                "n_hits": m.n_hits,
            }
        data.append(case_data)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Results saved to {path}")
