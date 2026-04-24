"""Reranking providers: Voyage AI (API) and sentence-transformers CrossEncoder (local)."""

import logging

from ragrep.models import SearchResult

log = logging.getLogger(__name__)

_RERANK_INSTRUCTION = (
    "Given a question about an internal company knowledge base, "
    "retrieve relevant documents that answer the question"
)

# Qwen3-Reranker requires chat-template formatting for proper scoring
_QUERY_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query "
    "and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
_DOC_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _format_query(query: str, instruction: str = _RERANK_INSTRUCTION) -> str:
    return f"{_QUERY_PREFIX}<Instruct>: {instruction}\n<Query>: {query}\n"


def _format_document(document: str) -> str:
    return f"<Document>: {document}{_DOC_SUFFIX}"


class VoyageReranker:
    """Voyage AI API reranker."""

    def __init__(self, model_name: str = "rerank-2.5"):
        import voyageai

        self.client = voyageai.Client()
        self.model = model_name
        log.info("Voyage reranker ready (model=%s)", model_name)

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Rerank via Voyage API. Returns top_k sorted by relevance."""
        if not candidates:
            return []

        docs = [c.content for c in candidates]
        result = self.client.rerank(query, docs, model=self.model, top_k=top_k)

        reranked: list[SearchResult] = []
        for r in result.results:
            candidate = candidates[r.index]
            candidate.rerank_score = r.relevance_score
            reranked.append(candidate)
        return reranked


class STReranker:
    """Local cross-encoder reranker (e.g. Qwen3-Reranker-0.6B-seq-cls)."""

    def __init__(self, model_name: str = "tomaarsen/Qwen3-Reranker-0.6B-seq-cls", device: str = "mps"):
        from sentence_transformers import CrossEncoder

        log.info("Loading ST reranker: %s (device=%s)", model_name, device)
        self.model = CrossEncoder(model_name, device=device)
        log.info("ST reranker loaded")

    def rerank(self, query: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        """Rerank candidates. Returns top_k sorted by rerank score."""
        if not candidates:
            return []

        formatted_query = _format_query(query)
        pairs = [(formatted_query, _format_document(c.content)) for c in candidates]
        scores = self.model.predict(pairs)

        for candidate, score in zip(candidates, scores, strict=True):
            candidate.rerank_score = float(score)

        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]


def make_reranker(provider: str, model_name: str, device: str = "mps") -> VoyageReranker | STReranker:
    """Factory: create reranker based on provider."""
    if provider == "voyage":
        return VoyageReranker(model_name)
    return STReranker(model_name, device)
