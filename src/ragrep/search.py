"""Search functions for ragrep — return dicts, no I/O."""

import logging
import re
from datetime import date, timedelta

from ragrep.query.filter import MetadataFilter, matches_filters

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_date(s: str) -> str:
    """Parse 'YYYY-MM-DD' or relative '3m', '2w', '90d', '1y' to YYYY-MM-DD."""
    m = re.fullmatch(r"(\d+)([dwmy])", s.strip().lower())
    if m:
        n, unit = int(m.group(1)), m.group(2)
        delta = {"d": timedelta(days=n), "w": timedelta(weeks=n),
                 "m": timedelta(days=n * 30), "y": timedelta(days=n * 365)}[unit]
        return (date.today() - delta).isoformat()
    date.fromisoformat(s)  # validate
    return s


def snippet(content: str, length: int, term: str = "") -> str:
    """Extract a snippet from content, centered on term if found."""
    content = content.replace("\n", " ").strip()
    if length <= 0 or length >= len(content):
        return content
    if term:
        pos = content.lower().find(term.lower())
        if pos >= 0:
            start = max(0, pos - length // 4)
            end = min(len(content), start + length)
            s = content[start:end]
            return ("..." if start > 0 else "") + s + ("..." if end < len(content) else "")
    return content[:length] + ("..." if len(content) > length else "")


def format_result(
    rank: int,
    chunk_id: str,
    source: str,
    title: str,
    content: str,
    scores: dict[str, float],
    metadata: dict,
    term: str,
    context: int,
    full: bool,
    show_scores: bool = False,
    include_metadata: bool = False,
) -> dict:
    """Format a single search result as a dict."""
    json_title = title if len(title) <= 80 else title[:77] + "..."
    rec: dict = {"rank": rank, "id": chunk_id, "source": source, "title": json_title}
    if show_scores:
        rec.update({k: round(v, 3) for k, v in scores.items()})
    if full:
        rec["content"] = content
    elif context > 0:
        rec["snippet"] = snippet(content, context, term)
    if include_metadata:
        rec["metadata"] = metadata
    return rec


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def search_grep(
    config: object,
    term: str,
    source: str | None = None,
    n: int = 5,
    filters: MetadataFilter | None = None,
    after: str | None = None,
    before: str | None = None,
    context: int = 0,
    full: bool = False,
    scores: bool = False,
    metadata: bool = False,
) -> dict:
    """Substring search over chunks. Returns result dict."""
    from ragrep.ingest.store import index_exists, load_index

    has_index = index_exists(config.index_dir)
    if has_index:
        _, chunks, _ = load_index(config.index_dir)
    else:
        from ragrep.ingest.chunk import chunk_all
        from ragrep.ingest.normalize import normalize_all
        docs = normalize_all(config.raw_dir)
        chunks = chunk_all(docs, config.ingest.max_chunk_tokens, config.ingest.chunk_overlap_tokens)

    if source:
        chunks = [c for c in chunks if c.source == source]
    if filters or after or before:
        chunks = [c for c in chunks if matches_filters(c.metadata, filters or {}, after, before)]

    matches = [(c, c.content.lower().find(term.lower())) for c in chunks]
    matches = [(c, pos) for c, pos in matches if pos >= 0]

    results = []
    for c, _ in matches[:n]:
        results.append(format_result(
            len(results) + 1, c.id, c.source, c.title, c.content,
            {}, c.metadata, term, context, full,
            show_scores=scores, include_metadata=metadata,
        ))

    return {"query": term, "mode": "grep", "total_matches": len(matches), "results": results}


def search_semantic(
    config: object,
    term: str,
    source: str | None = None,
    n: int = 5,
    filters: MetadataFilter | None = None,
    after: str | None = None,
    before: str | None = None,
    context: int = 0,
    full: bool = False,
    scores: bool = False,
    metadata: bool = False,
) -> dict:
    """Dense vector search via FAISS. Returns result dict."""
    from ragrep.ingest.embed import make_embedder
    from ragrep.ingest.store import load_index
    from ragrep.query.retrieve import dense_search

    faiss_index, chunks, _ = load_index(config.index_dir)
    embedder = make_embedder(config.embedding.provider, config.embedding.model_name)
    query_emb = embedder.embed_query(term)

    fetch_n = n * 5 if (after or before) else n
    results_raw = dense_search(query_emb, faiss_index, chunks, fetch_n, source_filter=source, metadata_filter=filters)

    if after or before:
        results_raw = [(idx, s) for idx, s in results_raw if matches_filters(chunks[idx].metadata, {}, after, before)]
        results_raw = results_raw[:n]

    results = []
    for idx, score in results_raw:
        chunk = chunks[idx]
        results.append(format_result(
            len(results) + 1, chunk.id, chunk.source, chunk.title, chunk.content,
            {"score": score}, chunk.metadata, term, context, full,
            show_scores=scores, include_metadata=metadata,
        ))

    return {"query": term, "mode": "semantic", "results": results}


def search_hybrid(
    config: object,
    term: str,
    source: str | None = None,
    n: int = 5,
    filters: MetadataFilter | None = None,
    after: str | None = None,
    before: str | None = None,
    context: int = 0,
    full: bool = False,
    scores: bool = False,
    metadata: bool = False,
) -> dict:
    """Hybrid FAISS + BM25 + RRF + rerank. Returns result dict."""
    from ragrep.ingest.embed import make_embedder
    from ragrep.ingest.store import load_index
    from ragrep.query.rerank import make_reranker
    from ragrep.query.retrieve import retrieve

    faiss_index, chunks, bm25 = load_index(config.index_dir)
    embedder = make_embedder(config.embedding.provider, config.embedding.model_name)
    query_emb = embedder.embed_query(term)

    fetch_n = max(n * 8, 40) if (after or before) else max(n * 4, 20)
    candidates = retrieve(
        query=term,
        query_embedding=query_emb,
        faiss_index=faiss_index,
        bm25=bm25,
        chunks=chunks,
        top_k_dense=config.retrieval.top_k_dense,
        top_k_bm25=config.retrieval.top_k_bm25,
        top_k_final=fetch_n,
        rrf_k=config.retrieval.rrf_k,
        source_filter=source,
        metadata_filter=filters,
    )

    if after or before:
        candidates = [c for c in candidates if matches_filters(c.metadata, {}, after, before)]

    reranker = make_reranker(config.reranker.provider, config.reranker.model_name)
    results_raw = reranker.rerank(term, candidates, top_k=n)

    results = []
    for r in results_raw:
        results.append(format_result(
            len(results) + 1, r.chunk_id, r.source, r.title, r.content,
            {"rerank": r.rerank_score, "rrf": r.rrf_score, "dense": r.dense_score, "bm25": r.bm25_score},
            r.metadata, term, context, full,
            show_scores=scores, include_metadata=metadata,
        ))

    return {"query": term, "mode": "hybrid", "results": results}
