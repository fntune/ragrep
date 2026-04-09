"""Ingestion pipeline orchestrator."""

import hashlib
import logging
import pickle
import time
from pathlib import Path

import faiss
import numpy as np

from ragrep.config import Config
from ragrep.ingest.chunk import chunk_all
from ragrep.ingest.embed import make_embedder
from ragrep.ingest.normalize import normalize_all
from ragrep.ingest.store import build_bm25_index, build_faiss_index, save_index
from ragrep.models import Chunk, IngestStats

log = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Stable content hash for embedding cache dedup."""
    return hashlib.sha256(text.encode()).hexdigest()


def _load_embed_cache(cache_path: Path, index_dir: Path) -> dict[str, np.ndarray]:
    """Load embedding cache. Bootstraps from existing FAISS index if no cache file yet."""
    if cache_path.exists():
        with open(cache_path, "rb") as f:
            cache: dict[str, np.ndarray] = pickle.load(f)
        log.info("Loaded embedding cache: %d entries", len(cache))
        return cache

    # Bootstrap from existing index (one-time migration)
    faiss_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.pkl"
    if not (faiss_path.exists() and chunks_path.exists()):
        return {}

    log.info("Bootstrapping embedding cache from existing index")
    idx = faiss.read_index(str(faiss_path))
    with open(chunks_path, "rb") as f:
        chunks: list[Chunk] = pickle.load(f)

    embeddings = idx.reconstruct_n(0, idx.ntotal)
    cache = {}
    for chunk, emb in zip(chunks, embeddings, strict=True):
        cache[_content_hash(chunk.content)] = emb
    log.info("Bootstrapped %d embeddings into cache", len(cache))
    return cache


def _save_embed_cache(cache_path: Path, cache: dict[str, np.ndarray]) -> None:
    """Persist embedding cache to disk."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(cache, f)
    size_mb = cache_path.stat().st_size / 1024 / 1024
    log.info("Saved embedding cache: %d entries (%.1f MB)", len(cache), size_mb)


def ingest(config: Config, force: bool = False, source_filter: str | None = None) -> IngestStats:
    """Run the full ingestion pipeline: normalize → chunk → embed → store.

    Default: incremental — reuses cached embeddings, only embeds new/changed chunks.
    With force=True: ignores cache, re-embeds everything from scratch.
    """
    start = time.monotonic()
    raw_dir = config.raw_dir
    index_dir = config.index_dir

    # 1. Normalize
    log.info("Step 1/4: Normalizing raw data from %s", raw_dir)
    docs = normalize_all(raw_dir)

    if source_filter:
        docs = [d for d in docs if d.source == source_filter]
        log.info("Filtered to %d documents (source=%s)", len(docs), source_filter)

    if not docs:
        log.warning("No documents to index")
        return IngestStats()

    # 2. Chunk
    log.info("Step 2/4: Chunking %d documents", len(docs))
    chunks = chunk_all(docs, config.ingest.max_chunk_tokens, config.ingest.chunk_overlap_tokens)

    # 3. Embed (incremental)
    log.info("Step 3/4: Embedding %d chunks", len(chunks))

    cache_path = index_dir / "embed_cache.pkl"
    cache: dict[str, np.ndarray] = {} if force else _load_embed_cache(cache_path, index_dir)

    hashes = [_content_hash(c.content) for c in chunks]
    to_embed_idx = [i for i, h in enumerate(hashes) if h not in cache]
    n_cached = len(chunks) - len(to_embed_idx)

    if to_embed_idx:
        log.info("Embedding %d new chunks (%d cached)", len(to_embed_idx), n_cached)
        embedder = make_embedder(config.embedding.provider, config.embedding.model_name, config.embedding.device)
        new_texts = [chunks[i].content for i in to_embed_idx]
        checkpoint_path = index_dir / ".embed_checkpoint.npz"
        new_embeddings = embedder.embed_documents(
            new_texts, config.ingest.batch_size, checkpoint_path=checkpoint_path,
        )
        for j, idx in enumerate(to_embed_idx):
            cache[hashes[idx]] = new_embeddings[j]
    else:
        log.info("All %d chunks cached, no embedding needed", len(chunks))

    # Assemble full embedding matrix in chunk order
    embeddings = np.array([cache[h] for h in hashes], dtype=np.float32)
    _save_embed_cache(cache_path, cache)

    # 4. Store
    log.info("Step 4/4: Building and saving index")
    faiss_index = build_faiss_index(embeddings)
    bm25 = build_bm25_index(chunks)
    save_index(index_dir, faiss_index, chunks, bm25)

    elapsed = time.monotonic() - start

    # Compute stats
    source_counts: dict[str, int] = {}
    for doc in docs:
        source_counts[doc.source] = source_counts.get(doc.source, 0) + 1

    stats = IngestStats(
        documents=len(docs),
        chunks=len(chunks),
        sources=source_counts,
        elapsed_s=elapsed,
    )

    log.info(
        "Ingestion complete: %d docs → %d chunks (%d new, %d cached) in %.1fs. Sources: %s",
        stats.documents, stats.chunks, len(to_embed_idx), n_cached, stats.elapsed_s,
        ", ".join(f"{k}={v}" for k, v in sorted(stats.sources.items())),
    )
    return stats
