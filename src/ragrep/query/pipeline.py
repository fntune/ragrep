"""Query pipeline orchestrator."""

import logging
import time

from ragrep.config import Config
from ragrep.ingest.embed import make_embedder
from ragrep.ingest.store import load_index, load_multi_index, model_key
from ragrep.models import QueryResult, SearchResult
from ragrep.query.filter import MetadataFilter
from ragrep.query.generate import generate
from ragrep.query.prompt import build_prompt
from ragrep.query.rerank import make_reranker
from ragrep.query.retrieve import retrieve_multi

log = logging.getLogger(__name__)


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _boost_and_dedup(
    results: list[SearchResult], threshold: float, rrf_k: int,
) -> list[SearchResult]:
    """Group similar results, boost representatives via overlap RRF, dedup."""
    token_cache = [set(r.content.lower().split()) for r in results]

    # Greedy grouping: iterate by rerank_score desc, merge into first similar group
    groups: list[list[int]] = []
    for i in range(len(results)):
        merged = False
        for group in groups:
            if any(_jaccard(token_cache[i], token_cache[j]) > threshold for j in group):
                group.append(i)
                merged = True
                break
        if not merged:
            groups.append([i])

    # Boost + dedup: keep best rerank per group, add overlap RRF to rrf_score
    deduped = []
    for group in groups:
        rep = results[group[0]]
        overlap_rrf = sum(1.0 / (rrf_k + i) for i in range(len(group)))
        rep.rrf_score += overlap_rrf
        deduped.append(rep)

    deduped.sort(key=lambda r: r.rrf_score, reverse=True)
    return deduped


class QueryEngine:
    """Holds loaded models and index for repeated queries."""

    def __init__(self, config: Config):
        self.config = config
        log.info("Loading index from %s", config.index_dir)

        embedding_models = config.embedding.get_models()
        model_keys = [model_key(m.provider, m.model_name) for m in embedding_models]

        if len(embedding_models) > 1:
            self.faiss_indexes, self.chunks, self.bm25 = load_multi_index(config.index_dir, model_keys)
            self.embedders = {
                model_key(m.provider, m.model_name): make_embedder(m.provider, m.model_name, m.device)
                for m in embedding_models
            }
        else:
            fi, self.chunks, self.bm25 = load_index(config.index_dir)
            m = embedding_models[0]
            key = model_key(m.provider, m.model_name)
            self.faiss_indexes = {key: fi}
            self.embedders = {key: make_embedder(m.provider, m.model_name, m.device)}

        self.reranker = make_reranker(config.reranker.provider, config.reranker.model_name, config.reranker.device)
        log.info("Query engine ready (%d chunks, %d models)", len(self.chunks), len(self.embedders))

    def query(
        self,
        question: str,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
        no_generate: bool = False,
    ) -> QueryResult:
        """Run the full query pipeline."""
        timings: dict[str, float] = {}
        rc = self.config.retrieval
        gc = self.config.generation

        # 1. Embed query with each model
        t0 = time.monotonic()
        query_embeddings = {
            key: embedder.embed_query(question)
            for key, embedder in self.embedders.items()
        }
        timings["embed"] = time.monotonic() - t0

        # 2. Retrieve (multi-signal RRF)
        t0 = time.monotonic()
        candidates = retrieve_multi(
            query=question,
            query_embeddings=query_embeddings,
            faiss_indexes=self.faiss_indexes,
            bm25=self.bm25,
            chunks=self.chunks,
            top_k_dense=rc.top_k_dense,
            top_k_bm25=rc.top_k_bm25,
            top_k_final=rc.top_k_dense + rc.top_k_bm25,
            rrf_k=rc.rrf_k,
            source_filter=source_filter,
            metadata_filter=metadata_filter,
        )
        timings["retrieve"] = time.monotonic() - t0

        # 3. Rerank (inflate top_k when dedup will reduce count)
        t0 = time.monotonic()
        rerank_k = rc.top_k_rerank * 3 if rc.dedup_threshold > 0 else rc.top_k_rerank
        results = self.reranker.rerank(question, candidates, rerank_k)
        timings["rerank"] = time.monotonic() - t0

        # 4. Overlap boost + dedup
        if rc.dedup_threshold > 0:
            results = _boost_and_dedup(results, rc.dedup_threshold, rc.rrf_k)
            results = results[:rc.top_k_rerank]

        # 5. Generate
        if no_generate:
            return QueryResult(answer="", sources=results, query=question, timings=timings)

        t0 = time.monotonic()
        messages = build_prompt(question, results)
        answer = generate(
            messages=messages,
            model=gc.model_name,
            ollama_url=gc.ollama_url,
            temperature=gc.temperature,
            max_tokens=gc.max_tokens,
        )
        timings["generate"] = time.monotonic() - t0

        return QueryResult(answer=answer, sources=results, query=question, timings=timings)
