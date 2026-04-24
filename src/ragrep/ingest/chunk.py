"""Recursive token-based document chunking."""

import logging

from ragrep.models import Chunk, Document

log = logging.getLogger(__name__)

# Sources that are typically small enough to skip chunking
_NO_CHUNK_SOURCES = {"bookmark", "pin"}

# Separators in priority order for recursive splitting
_SEPARATORS = ["\n\n", "\n", "\t", ". ", " "]

# Hard character limit per chunk (safety valve — 4K chars ≈ 1K tokens)
_MAX_CHARS = 4000


def _approx_tokens(text: str) -> int:
    """Approximate token count. Uses max(word_count, char_count/4) to handle tab-delimited data."""
    return max(len(text.split()), len(text) // 4)


def _recursive_split(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """Split text recursively at natural boundaries."""
    if _approx_tokens(text) <= max_tokens:
        return [text]

    # Try each separator level
    for sep in _SEPARATORS:
        parts = text.split(sep)
        if len(parts) == 1:
            continue

        chunks = []
        current = parts[0]

        for part in parts[1:]:
            candidate = current + sep + part
            if _approx_tokens(candidate) <= max_tokens:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                # Overlap: grab the tail of current chunk
                if overlap_tokens > 0:
                    words = current.split()
                    overlap_text = " ".join(words[-overlap_tokens:]) if len(words) > overlap_tokens else ""
                    current = (overlap_text + sep + part).strip() if overlap_text else part
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        if len(chunks) > 1:
            return chunks

    # Final fallback: hard split by words
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_tokens - overlap_tokens):
        chunk = " ".join(words[i : i + max_tokens])
        if chunk.strip():
            chunks.append(chunk.strip())
    return chunks


def chunk_document(doc: Document, max_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Chunk a document. Returns single-element list if it fits in max_tokens."""
    # Small sources: never chunk
    if doc.source in _NO_CHUNK_SOURCES or _approx_tokens(doc.content) <= max_tokens:
        return [Chunk(
            id=f"{doc.id}:0",
            doc_id=doc.id,
            content=doc.content,
            title=doc.title,
            source=doc.source,
            metadata={**doc.metadata, "chunk_idx": 0},
        )]

    parts = _recursive_split(doc.content, max_tokens, overlap_tokens)

    # Safety valve: hard-split any chunk that's still too long by character count
    safe_parts: list[str] = []
    for part in parts:
        if len(part) <= _MAX_CHARS:
            safe_parts.append(part)
        else:
            for i in range(0, len(part), _MAX_CHARS):
                segment = part[i : i + _MAX_CHARS].strip()
                if segment:
                    safe_parts.append(segment)

    return [
        Chunk(
            id=f"{doc.id}:{i}",
            doc_id=doc.id,
            content=part,
            title=doc.title,
            source=doc.source,
            metadata={**doc.metadata, "chunk_idx": i},
        )
        for i, part in enumerate(safe_parts)
    ]


def chunk_all(docs: list[Document], max_tokens: int = 512, overlap_tokens: int = 64) -> list[Chunk]:
    """Chunk all documents."""
    chunks: list[Chunk] = []
    multi_chunk_count = 0

    for doc in docs:
        doc_chunks = chunk_document(doc, max_tokens, overlap_tokens)
        if len(doc_chunks) > 1:
            multi_chunk_count += 1
        chunks.extend(doc_chunks)

    log.info(
        "Chunked %d documents → %d chunks (%d documents split into multiple chunks)",
        len(docs), len(chunks), multi_chunk_count,
    )
    return chunks
