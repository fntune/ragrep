"""High-level SDK for ragrep: ingest documents, query with hybrid RAG."""

from __future__ import annotations

import hashlib
import logging
import pickle
import shutil
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from ragrep.collection.catalog import Catalog, RecordInput
from ragrep.collection.search import RecordHit
from ragrep.models import Document, IngestStats, MetadataValue, QueryResult
from ragrep.query.filter import MetadataFilter

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedModel:
    """Embedding model specification."""

    provider: str
    model_name: str
    device: str = "mps"


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _load_cache(path: Path) -> dict[str, object]:
    if path.exists():
        with open(path, "rb") as f:
            cache: dict[str, object] = pickle.load(f)
        log.info("Loaded embed cache: %d entries", len(cache))
        return cache
    return {}


def _save_cache(path: Path, cache: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(cache, f)
    log.info("Saved embed cache: %d entries", len(cache))


class Index:
    """High-level SDK for ragrep: ingest documents, query with hybrid RAG."""

    def __init__(
        self,
        index_dir: str | Path,
        embedding_models: list[EmbedModel | dict],
        *,
        reranker_provider: str = "voyage",
        reranker_model: str = "rerank-2.5",
        max_chunk_tokens: int = 512,
        chunk_overlap_tokens: int = 64,
        dedup_threshold: float = 0.0,
    ) -> None:
        self._index_dir = Path(index_dir)
        self._models = [
            m if isinstance(m, EmbedModel) else EmbedModel(**m)
            for m in embedding_models
        ]
        if not self._models:
            raise ValueError("at least one embedding model required")
        self._reranker_provider = reranker_provider
        self._reranker_model = reranker_model
        self._max_chunk_tokens = max_chunk_tokens
        self._chunk_overlap_tokens = chunk_overlap_tokens
        self._dedup_threshold = dedup_threshold

    @property
    def catalog_path(self) -> Path:
        """Path for persisted collection records."""
        return self._index_dir / "records.jsonl"

    def exists(self) -> bool:
        """Check if the index has been built for all configured models."""
        from ragrep.ingest.store import model_key, multi_index_exists

        keys = [model_key(m.provider, m.model_name) for m in self._models]
        return multi_index_exists(self._index_dir, keys)

    def ingest(self, docs: list[Document], *, force: bool = False, batch: bool = False) -> IngestStats:
        """Chunk, embed (with per-model cache), and store documents."""
        start = time.monotonic()

        if not docs:
            self._clear_search_artifacts()
            return IngestStats()

        from ragrep.ingest.chunk import chunk_all
        from ragrep.ingest.store import build_bm25_index, save_multi_index

        chunks = chunk_all(docs, self._max_chunk_tokens, self._chunk_overlap_tokens)
        model_indexes = self._embed_with_cache(chunks, force, batch=batch)
        bm25 = build_bm25_index(chunks)
        save_multi_index(self._index_dir, model_indexes, chunks, bm25)

        elapsed = time.monotonic() - start
        source_counts: dict[str, int] = {}
        for doc in docs:
            source_counts[doc.source] = source_counts.get(doc.source, 0) + 1

        stats = IngestStats(
            documents=len(docs),
            chunks=len(chunks),
            sources=source_counts,
            elapsed_s=elapsed,
        )
        log.info("Ingestion complete: %d docs → %d chunks in %.1fs", stats.documents, stats.chunks, stats.elapsed_s)
        return stats

    def query(
        self,
        question: str,
        *,
        top_k: int = 5,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
        no_generate: bool = True,
    ) -> QueryResult:
        """Search the index with hybrid retrieval + reranking."""
        from ragrep.query.pipeline import QueryEngine

        config = self._build_config(top_k_rerank=top_k)
        engine = QueryEngine(config)
        return engine.query(
            question,
            source_filter=source_filter,
            metadata_filter=metadata_filter,
            no_generate=no_generate,
        )

    def upsert_records(
        self,
        records: Iterable[RecordInput],
        *,
        force: bool = False,
        batch: bool = False,
    ) -> IngestStats:
        """Upsert collection records, persist them, and rebuild the searchable index."""
        catalog = self._load_catalog()
        catalog.upsert(records)
        return self._save_and_reindex(catalog, force=force, batch=batch)

    def delete_records(
        self,
        ids: Iterable[str],
        *,
        force: bool = False,
        batch: bool = False,
    ) -> IngestStats:
        """Delete collection records by ID and rebuild the searchable index."""
        catalog = self._load_catalog()
        catalog.delete(ids)
        return self._save_and_reindex(catalog, force=force, batch=batch)

    def clear_records(
        self,
        *,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
        force: bool = False,
        batch: bool = False,
    ) -> IngestStats:
        """Delete collection records matching the optional filters and rebuild the searchable index."""
        catalog = self._load_catalog()
        catalog.clear(source=source_filter, filters=metadata_filter)
        return self._save_and_reindex(catalog, force=force, batch=batch)

    def fetch_records_metadata(self, ids: Iterable[str]) -> dict[str, dict[str, MetadataValue]]:
        """Fetch flattened record fields and metadata by record ID."""
        return self._load_catalog().fetch_metadata(ids)

    def list_record_ids(
        self,
        *,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[str]:
        """List record IDs matching the optional filters."""
        return self._load_catalog().list_ids(source=source_filter, filters=metadata_filter)

    def count_records(
        self,
        *,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> int:
        """Count records matching the optional filters."""
        return self._load_catalog().count(source=source_filter, filters=metadata_filter)

    def search_records(
        self,
        query: str,
        *,
        top_k: int = 5,
        source_filter: str | None = None,
        metadata_filter: MetadataFilter | None = None,
    ) -> list[RecordHit]:
        """Search collection records and return hits keyed by record ID."""
        if top_k <= 0:
            return []

        result = self.query(
            query,
            top_k=top_k * 3,
            source_filter=source_filter,
            metadata_filter=metadata_filter,
            no_generate=True,
        )
        hits: list[RecordHit] = []
        seen: set[str] = set()
        for source in result.sources:
            hit = RecordHit.from_search_result(source)
            if hit.record_id in seen:
                continue
            seen.add(hit.record_id)
            hits.append(hit)
            if len(hits) >= top_k:
                break
        return hits

    def _load_catalog(self) -> Catalog:
        return Catalog.load(self.catalog_path)

    def _save_and_reindex(self, catalog: Catalog, *, force: bool, batch: bool) -> IngestStats:
        catalog.save()
        return self.ingest(catalog.documents(), force=force, batch=batch)

    def _clear_search_artifacts(self) -> None:
        for name in ("faiss.index", "chunks.pkl", "bm25.pkl"):
            (self._index_dir / name).unlink(missing_ok=True)
        shutil.rmtree(self._index_dir / "models", ignore_errors=True)

        store = sys.modules.get("ragrep.ingest.store")
        if store is not None and hasattr(store, "clear_index_cache"):
            store.clear_index_cache(self._index_dir)

    def _embed_with_cache(self, chunks: list, force: bool, *, batch: bool = False) -> dict:
        """Embed chunks with all models, using per-model content-hash cache."""
        import faiss
        import numpy as np

        from ragrep.ingest.embed import make_embedder
        from ragrep.ingest.store import build_faiss_index, model_key

        cache_path = self._index_dir / "embed_cache.pkl"
        cache: dict[str, object] = {} if force else _load_cache(cache_path)

        texts = [c.content for c in chunks]
        hashes = [_content_hash(t) for t in texts]

        # Build per-model info: which chunks need embedding
        model_work: list[dict] = []
        for em in self._models:
            mkey = model_key(em.provider, em.model_name)
            cache_keys = [f"{mkey}:{h}" for h in hashes]
            to_embed_idx = [i for i, ck in enumerate(cache_keys) if ck not in cache]
            n_cached = len(chunks) - len(to_embed_idx)
            model_work.append({
                "em": em, "mkey": mkey, "cache_keys": cache_keys,
                "to_embed_idx": to_embed_idx, "n_cached": n_cached,
            })

        if batch:
            # Submit all batches upfront, then collect all (parallel server-side)
            pending: list[tuple[dict, object, object]] = []
            for mw in model_work:
                if not mw["to_embed_idx"]:
                    log.info("Model %s: all %d chunks cached", mw["mkey"], len(chunks))
                    pending.append((mw, None, None))
                    continue
                log.info("Model %s: submitting %d chunks (%d cached)", mw["mkey"], len(mw["to_embed_idx"]), mw["n_cached"])
                embedder = make_embedder(mw["em"].provider, mw["em"].model_name, mw["em"].device)
                if not hasattr(embedder, "submit_batch"):
                    raise ValueError(f"Provider {mw['em'].provider!r} does not support batch embedding")
                new_texts = [texts[i] for i in mw["to_embed_idx"]]
                handle = embedder.submit_batch(new_texts)
                pending.append((mw, embedder, handle))

            log.info("All %d batch jobs submitted, waiting for results...", sum(1 for _, _, h in pending if h))

            # Collect results (polls each until done), save cache after each
            for mw, embedder, handle in pending:
                if handle is None:
                    continue
                log.info("Collecting results for %s...", mw["mkey"])
                new_embeddings = embedder.collect_batch(handle)
                for j, idx in enumerate(mw["to_embed_idx"]):
                    cache[mw["cache_keys"][idx]] = new_embeddings[j]
                _save_cache(cache_path, cache)
        else:
            # Sync path: embed one model at a time
            for mw in model_work:
                if not mw["to_embed_idx"]:
                    log.info("Model %s: all %d chunks cached", mw["mkey"], len(chunks))
                    continue
                log.info("Model %s: embedding %d new chunks (%d cached)", mw["mkey"], len(mw["to_embed_idx"]), mw["n_cached"])
                embedder = make_embedder(mw["em"].provider, mw["em"].model_name, mw["em"].device)
                new_texts = [texts[i] for i in mw["to_embed_idx"]]
                checkpoint_path = self._index_dir / f".embed_checkpoint_{mw['mkey']}.npz"
                new_embeddings = embedder.embed_documents(new_texts, checkpoint_path=checkpoint_path)
                for j, idx in enumerate(mw["to_embed_idx"]):
                    cache[mw["cache_keys"][idx]] = new_embeddings[j]

        # Build FAISS indexes from cache
        model_indexes: dict[str, faiss.IndexFlatIP] = {}
        for mw in model_work:
            embeddings = np.array([cache[ck] for ck in mw["cache_keys"]], dtype=np.float32)
            model_indexes[mw["mkey"]] = build_faiss_index(embeddings)

        _save_cache(cache_path, cache)
        return model_indexes

    def _build_config(self, *, top_k_rerank: int = 5):
        """Assemble a Config from Index constructor params."""
        from ragrep.config import (
            Config,
            DataConfig,
            EmbeddingConfig,
            EmbeddingModelConfig,
            GenerationConfig,
            IngestConfig,
            RerankerConfig,
            RetrievalConfig,
            ScrapeConfig,
        )

        models = tuple(
            EmbeddingModelConfig(provider=m.provider, model_name=m.model_name, device=m.device)
            for m in self._models
        )

        return Config(
            data=DataConfig(raw_dir=str(self._index_dir / "raw"), index_dir=str(self._index_dir)),
            ingest=IngestConfig(max_chunk_tokens=self._max_chunk_tokens, chunk_overlap_tokens=self._chunk_overlap_tokens),
            embedding=EmbeddingConfig(models=models),
            reranker=RerankerConfig(provider=self._reranker_provider, model_name=self._reranker_model),
            retrieval=RetrievalConfig(top_k_rerank=top_k_rerank, dedup_threshold=self._dedup_threshold),
            generation=GenerationConfig(),
            scrape=ScrapeConfig(),
        )
