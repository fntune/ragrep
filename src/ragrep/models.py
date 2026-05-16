"""Core data models for the RAG pipeline."""

from dataclasses import dataclass, field

MetadataValue = str | int | float | bool


@dataclass(frozen=True)
class Document:
    """Normalized document from any source."""

    id: str
    source: str  # slack, atlassian, gdrive, git, file, bookmark, pin
    content: str
    title: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document, ready for embedding."""

    id: str  # "{doc_id}:{chunk_idx}"
    doc_id: str
    content: str
    title: str
    source: str
    metadata: dict[str, MetadataValue] = field(default_factory=dict)


@dataclass
class SearchResult:
    """A retrieved + scored result."""

    chunk_id: str
    content: str
    title: str
    source: str
    metadata: dict[str, MetadataValue]
    dense_score: float = 0.0
    bm25_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class QueryResult:
    """Final query output."""

    answer: str
    sources: list[SearchResult]
    query: str
    timings: dict[str, float] = field(default_factory=dict)


@dataclass
class IngestStats:
    """Statistics from the ingestion pipeline."""

    documents: int = 0
    chunks: int = 0
    sources: dict[str, int] = field(default_factory=dict)
    elapsed_s: float = 0.0
