"""Hybrid retrieval: dense (FAISS) + sparse (BM25) with Reciprocal Rank Fusion."""

import logging

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ragrep.models import Chunk, SearchResult

log = logging.getLogger(__name__)


def _matches_metadata(metadata: dict, filters: dict[str, str]) -> bool:
    """Check all filters match (case-insensitive substring)."""
    for key, val in filters.items():
        chunk_val = metadata.get(key)
        if chunk_val is None or val.lower() not in str(chunk_val).lower():
            return False
    return True


def dense_search(
    query_embedding: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    chunks: list[Chunk],
    top_k: int,
    source_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> list[tuple[int, float]]:
    """Dense vector search via FAISS. Returns (chunk_index, score) pairs."""
    filtering = source_filter or metadata_filter
    fetch_k = top_k * 5 if metadata_filter else (top_k * 3 if source_filter else top_k)
    scores, indices = faiss_index.search(query_embedding, fetch_k)

    results = []
    for score, idx in zip(scores[0], indices[0], strict=True):
        if idx < 0:
            continue
        chunk = chunks[idx]
        if source_filter and chunk.source != source_filter:
            continue
        if metadata_filter and not _matches_metadata(chunk.metadata, metadata_filter):
            continue
        results.append((int(idx), float(score)))
        if len(results) >= top_k:
            break

    return results


def bm25_search(
    query: str,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k: int,
    source_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> list[tuple[int, float]]:
    """Sparse BM25 search. Returns (chunk_index, score) pairs."""
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)

    if source_filter or metadata_filter:
        for i, chunk in enumerate(chunks):
            if source_filter and chunk.source != source_filter:
                scores[i] = 0.0
            elif metadata_filter and not _matches_metadata(chunk.metadata, metadata_filter):
                scores[i] = 0.0

    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


def _reciprocal_rank_fusion(
    dense_results: list[tuple[int, float]],
    bm25_results: list[tuple[int, float]],
    k: int = 60,
) -> list[tuple[int, float, float, float]]:
    """Merge dense and BM25 results via RRF. Returns (idx, dense_score, bm25_score, rrf_score)."""
    # Build rank maps
    dense_ranks = {idx: rank for rank, (idx, _) in enumerate(dense_results)}
    bm25_ranks = {idx: rank for rank, (idx, _) in enumerate(bm25_results)}
    dense_scores = {idx: score for idx, score in dense_results}
    bm25_scores = {idx: score for idx, score in bm25_results}

    # All unique indices
    all_indices = set(dense_ranks.keys()) | set(bm25_ranks.keys())

    scored = []
    for idx in all_indices:
        d_rank = dense_ranks.get(idx, 1000)  # Penalize missing results
        b_rank = bm25_ranks.get(idx, 1000)
        rrf_score = 1.0 / (k + d_rank) + 1.0 / (k + b_rank)
        scored.append((idx, dense_scores.get(idx, 0.0), bm25_scores.get(idx, 0.0), rrf_score))

    scored.sort(key=lambda x: x[3], reverse=True)
    return scored


def _multi_signal_rrf(
    ranked_lists: list[tuple[str, list[tuple[int, float]]]],
    k: int = 60,
) -> list[tuple[int, dict[str, float], float]]:
    """Generalized RRF over N ranked result lists.

    Args:
        ranked_lists: list of (signal_name, [(chunk_idx, score), ...])
        k: RRF constant

    Returns:
        list of (chunk_idx, {signal_name: score}, rrf_score), sorted by rrf_score desc
    """
    rank_maps: dict[str, dict[int, int]] = {}
    score_maps: dict[str, dict[int, float]] = {}

    for name, results in ranked_lists:
        rank_maps[name] = {idx: rank for rank, (idx, _) in enumerate(results)}
        score_maps[name] = {idx: score for idx, score in results}

    all_indices: set[int] = set()
    for _, results in ranked_lists:
        all_indices.update(idx for idx, _ in results)

    scored = []
    for idx in all_indices:
        rrf_score = 0.0
        signal_scores: dict[str, float] = {}
        for name, _ in ranked_lists:
            rank = rank_maps[name].get(idx, 1000)
            rrf_score += 1.0 / (k + rank)
            signal_scores[name] = score_maps[name].get(idx, 0.0)
        scored.append((idx, signal_scores, rrf_score))

    scored.sort(key=lambda x: x[2], reverse=True)
    return scored


def retrieve_multi(
    query: str,
    query_embeddings: dict[str, np.ndarray],
    faiss_indexes: dict[str, faiss.IndexFlatIP],
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k_dense: int = 20,
    top_k_bm25: int = 20,
    top_k_final: int = 20,
    rrf_k: int = 60,
    source_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Multi-model hybrid retrieval with generalized RRF fusion."""
    ranked_lists: list[tuple[str, list[tuple[int, float]]]] = []

    for mkey, faiss_idx in faiss_indexes.items():
        emb = query_embeddings[mkey]
        results = dense_search(emb, faiss_idx, chunks, top_k_dense, source_filter, metadata_filter)
        ranked_lists.append((f"dense:{mkey}", results))

    bm25_results = bm25_search(query, bm25, chunks, top_k_bm25, source_filter, metadata_filter)
    ranked_lists.append(("bm25", bm25_results))

    log.debug("Multi-signal RRF: %d signals, %s", len(ranked_lists), [n for n, _ in ranked_lists])
    fused = _multi_signal_rrf(ranked_lists, k=rrf_k)

    results = []
    for idx, signal_scores, rrf_score in fused[:top_k_final]:
        chunk = chunks[idx]
        dense_score = max(
            (signal_scores.get(f"dense:{k}", 0.0) for k in faiss_indexes),
            default=0.0,
        )
        results.append(SearchResult(
            chunk_id=chunk.id,
            content=chunk.content,
            title=chunk.title,
            source=chunk.source,
            metadata=chunk.metadata,
            dense_score=dense_score,
            bm25_score=signal_scores.get("bm25", 0.0),
            rrf_score=rrf_score,
        ))

    return results


def retrieve(
    query: str,
    query_embedding: np.ndarray,
    faiss_index: faiss.IndexFlatIP,
    bm25: BM25Okapi,
    chunks: list[Chunk],
    top_k_dense: int = 20,
    top_k_bm25: int = 20,
    top_k_final: int = 20,
    rrf_k: int = 60,
    source_filter: str | None = None,
    metadata_filter: dict[str, str] | None = None,
) -> list[SearchResult]:
    """Hybrid retrieval with RRF fusion."""
    dense_results = dense_search(query_embedding, faiss_index, chunks, top_k_dense, source_filter, metadata_filter)
    bm25_results = bm25_search(query, bm25, chunks, top_k_bm25, source_filter, metadata_filter)

    log.debug("Dense: %d results, BM25: %d results", len(dense_results), len(bm25_results))

    fused = _reciprocal_rank_fusion(dense_results, bm25_results, rrf_k)

    results = []
    for idx, dense_score, bm25_score, rrf_score in fused[:top_k_final]:
        chunk = chunks[idx]
        results.append(SearchResult(
            chunk_id=chunk.id,
            content=chunk.content,
            title=chunk.title,
            source=chunk.source,
            metadata=chunk.metadata,
            dense_score=dense_score,
            bm25_score=bm25_score,
            rrf_score=rrf_score,
        ))

    return results
