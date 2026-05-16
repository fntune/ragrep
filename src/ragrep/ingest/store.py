"""FAISS + BM25 index persistence."""

import logging
import pickle
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from ragrep.models import Chunk

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None

log = logging.getLogger(__name__)

_INDEX_FILES = ("faiss.index", "chunks.pkl", "bm25.pkl")
_LOCK_FILE = ".index.lock"


@contextmanager
def _locked(index_dir: Path, *, exclusive: bool):
    index_dir.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        yield
        return

    with open(index_dir / _LOCK_FILE, "a+b") as lock_file:
        mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(lock_file.fileno(), mode)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def _staging_dir(index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".publish-", dir=index_dir) as tmp:
        yield Path(tmp)


def _write_pickle(path: Path, value: object) -> None:
    with open(path, "wb") as f:
        pickle.dump(value, f)


def _remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _publish_index(index_dir: Path, stage_dir: Path, *, include_models: bool) -> None:
    with _locked(index_dir, exclusive=True):
        for name in _INDEX_FILES:
            (stage_dir / name).replace(index_dir / name)

        if include_models:
            models_path = index_dir / "models"
            _remove_path(models_path)
            (stage_dir / "models").replace(models_path)
        else:
            _remove_path(index_dir / "models")


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a flat inner-product index (cosine similarity on L2-normalized vectors)."""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    log.info("Built FAISS index: %d vectors, %d dims", index.ntotal, dim)
    return index


def build_bm25_index(chunks: list[Chunk]) -> BM25Okapi:
    """Build BM25 index from chunk texts."""
    tokenized = [chunk.content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    log.info("Built BM25 index: %d documents", len(tokenized))
    return bm25


def save_index(index_dir: Path, faiss_index: faiss.IndexFlatIP, chunks: list[Chunk], bm25: BM25Okapi) -> None:
    """Persist all index files to disk."""
    with _staging_dir(index_dir) as stage_dir:
        faiss.write_index(faiss_index, str(stage_dir / "faiss.index"))
        _write_pickle(stage_dir / "chunks.pkl", chunks)
        _write_pickle(stage_dir / "bm25.pkl", bm25)

        _publish_index(index_dir, stage_dir, include_models=False)

    log.info("Saved index to %s (faiss.index, chunks.pkl, bm25.pkl)", index_dir)
    clear_index_cache(index_dir)


_index_cache: dict[str, tuple[faiss.IndexFlatIP, list[Chunk], BM25Okapi]] = {}


def _cache_key(index_dir: Path) -> str:
    return str(Path(index_dir).resolve())


def clear_index_cache(index_dir: Path) -> None:
    """Drop cached index state for a path after the index files change."""
    _index_cache.pop(_cache_key(index_dir), None)


def load_index(index_dir: Path) -> tuple[faiss.IndexFlatIP, list[Chunk], BM25Okapi]:
    """Load all index files from disk. Caches by resolved path."""
    cache_key = _cache_key(index_dir)
    if cache_key in _index_cache:
        return _index_cache[cache_key]

    with _locked(index_dir, exclusive=False):
        faiss_index = faiss.read_index(str(index_dir / "faiss.index"))
        with open(index_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(index_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

    log.info("Loaded index: %d vectors, %d chunks", faiss_index.ntotal, len(chunks))
    _index_cache[cache_key] = (faiss_index, chunks, bm25)
    return faiss_index, chunks, bm25


def index_exists(index_dir: Path) -> bool:
    """Check if all index files exist."""
    if not index_dir.exists():
        return False
    with _locked(index_dir, exclusive=False):
        return all((index_dir / name).exists() for name in _INDEX_FILES)


def model_key(provider: str, model_name: str) -> str:
    """Stable directory name for a model's artifacts."""
    return f"{provider}--{model_name}"


def save_multi_index(
    index_dir: Path,
    model_indexes: dict[str, faiss.IndexFlatIP],
    chunks: list[Chunk],
    bm25: BM25Okapi,
) -> None:
    """Persist multi-model index to disk."""
    if not model_indexes:
        raise ValueError("at least one model index required")

    with _staging_dir(index_dir) as stage_dir:
        _write_pickle(stage_dir / "chunks.pkl", chunks)
        _write_pickle(stage_dir / "bm25.pkl", bm25)

        models_dir = stage_dir / "models"
        models_dir.mkdir(exist_ok=True)
        first_key = None
        for key, fi in model_indexes.items():
            if first_key is None:
                first_key = key
            model_dir = models_dir / key
            model_dir.mkdir(exist_ok=True)
            faiss.write_index(fi, str(model_dir / "faiss.index"))

        # Backward compat: top-level faiss.index from first model
        if first_key is not None:
            faiss.write_index(model_indexes[first_key], str(stage_dir / "faiss.index"))

        _publish_index(index_dir, stage_dir, include_models=True)

    log.info("Saved multi-model index to %s (%d models)", index_dir, len(model_indexes))
    clear_index_cache(index_dir)


def load_multi_index(
    index_dir: Path,
    model_keys: list[str] | None = None,
) -> tuple[dict[str, faiss.IndexFlatIP], list[Chunk], BM25Okapi]:
    """Load multi-model index from disk."""
    with _locked(index_dir, exclusive=False):
        with open(index_dir / "chunks.pkl", "rb") as f:
            chunks = pickle.load(f)
        with open(index_dir / "bm25.pkl", "rb") as f:
            bm25 = pickle.load(f)

        models_dir = index_dir / "models"
        if models_dir.exists() and model_keys:
            indexes = {}
            for key in model_keys:
                path = models_dir / key / "faiss.index"
                indexes[key] = faiss.read_index(str(path))
            log.info("Loaded multi-model index: %d models, %d chunks", len(indexes), len(chunks))
            return indexes, chunks, bm25

        # Fallback: single model
        fi = faiss.read_index(str(index_dir / "faiss.index"))
    log.info("Loaded single-model index: %d vectors, %d chunks", fi.ntotal, len(chunks))
    return {"default": fi}, chunks, bm25


def multi_index_exists(index_dir: Path, model_keys: list[str]) -> bool:
    """Check if all multi-model index files exist."""
    if not index_dir.exists():
        return False
    with _locked(index_dir, exclusive=False):
        if not (index_dir / "chunks.pkl").exists() or not (index_dir / "bm25.pkl").exists():
            return False
        models_dir = index_dir / "models"
        return all((models_dir / key / "faiss.index").exists() for key in model_keys)
