"""Collection search result contract."""

from dataclasses import dataclass

from ragrep.models import MetadataValue, SearchResult


@dataclass(frozen=True)
class RecordHit:
    """A search hit tied back to the owning collection record."""

    record_id: str
    chunk_id: str
    source: str
    title: str
    content: str
    metadata: dict[str, MetadataValue]
    score: float
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0

    @classmethod
    def from_search_result(cls, result: SearchResult) -> "RecordHit":
        return cls(
            record_id=_record_id(result.chunk_id),
            chunk_id=result.chunk_id,
            source=result.source,
            title=result.title,
            content=result.content,
            metadata=dict(result.metadata),
            score=_score(result),
            dense_score=result.dense_score,
            bm25_score=result.bm25_score,
            rrf_score=result.rrf_score,
            rerank_score=result.rerank_score,
        )


def _record_id(chunk_id: str) -> str:
    record_id, separator, _chunk_idx = chunk_id.rpartition(":")
    return record_id if separator else chunk_id


def _score(result: SearchResult) -> float:
    return result.rerank_score or result.rrf_score or max(result.dense_score, result.bm25_score)
