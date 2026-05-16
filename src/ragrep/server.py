"""FastAPI search service for ragrep."""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Index state (loaded at startup, held in memory)
# ---------------------------------------------------------------------------

_INDEX_DIR: Path | None = None
_CONFIG = None


def _resolve_index_dir() -> Path:
    """Find or download the index directory."""
    # 1. RAGREP_INDEX_DIR env var
    env_dir = os.environ.get("RAGREP_INDEX_DIR")
    if env_dir and (Path(env_dir) / "faiss.index").exists():
        log.info("Using index from RAGREP_INDEX_DIR=%s", env_dir)
        return Path(env_dir)

    # 2. CWD/data/index (when serving from a project clone)
    cwd_dir = Path.cwd() / "data" / "index"
    if (cwd_dir / "faiss.index").exists():
        log.info("Using local index at %s", cwd_dir)
        return cwd_dir

    # 3. Download from GCS
    bucket_name = os.environ.get("RAGREP_GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("No index found via RAGREP_INDEX_DIR or CWD/data/index/, and RAGREP_GCS_BUCKET is not set")
    tmp_dir = Path("/tmp/ragrep-index")  # noqa: S108
    if (tmp_dir / "faiss.index").exists():
        log.info("Using cached GCS index at %s", tmp_dir)
        return tmp_dir

    log.info("Downloading index from gs://%s/ to %s ...", bucket_name, tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    from google.cloud import storage

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for blob_name in ("faiss.index", "chunks.pkl", "bm25.pkl"):
        blob = bucket.blob(blob_name)
        dest = tmp_dir / blob_name
        log.info("  Downloading %s (%s MB)...", blob_name, f"{blob.size / 1e6:.0f}" if blob.size else "?")
        blob.download_to_filename(str(dest))

    log.info("Index download complete.")
    return tmp_dir


def _load_config():
    """Load ragrep config; load_config handles RAGREP_CONFIG / CWD / XDG search."""
    from ragrep.config import load_config

    return load_config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index at startup."""
    global _INDEX_DIR, _CONFIG

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env if present (CWD then ~/.config/ragrep/)
    from ragrep.config import load_env_files

    load_env_files()

    _INDEX_DIR = _resolve_index_dir()
    _CONFIG = _load_config()

    # Patch config to use the resolved index dir (frozen dataclass, bypass via object.__setattr__)
    object.__setattr__(_CONFIG.data, "index_dir", str(_INDEX_DIR))

    # Pre-load index into memory by importing it once (search functions use load_index which caches)
    from ragrep.ingest.store import load_index

    log.info("Loading index into memory from %s ...", _INDEX_DIR)
    _faiss, _chunks, _bm25 = load_index(_INDEX_DIR)
    log.info("Index loaded: %d chunks", len(_chunks))

    app.state.index_dir = _INDEX_DIR
    app.state.config = _CONFIG
    app.state.chunks_count = len(_chunks)

    yield


app = FastAPI(title="ragrep", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    chunks_count = getattr(app.state, "chunks_count", 0)
    return {"status": "ok", "chunks": chunks_count}


@app.get("/search")
async def search(
    q: str = Query(..., description="Search term"),
    mode: str = Query("grep", description="grep, semantic, or hybrid"),
    n: int = Query(5, description="Number of results"),
    source: str | None = Query(None, description="Source filter"),
    filter: str | None = Query(None, description="Metadata filters (key=val,key=val)"),
    after: str | None = Query(None, description="After date (YYYY-MM-DD or relative)"),
    before: str | None = Query(None, description="Before date (YYYY-MM-DD or relative)"),
    context: int = Query(300, description="Snippet length in chars"),
    full: bool = Query(False, description="Return full content"),
    scores: bool = Query(True, description="Include relevance scores"),
    metadata: bool = Query(False, description="Include metadata"),
):
    """Search the ragrep index."""
    from ragrep.query.filter import parse_filters
    from ragrep.search import parse_date, search_grep, search_hybrid, search_semantic

    # Parse filters
    filters = None
    if filter:
        try:
            raw_filters = [filter] if filter.lstrip().startswith("{") else filter.split(",")
            filters = parse_filters(raw_filters)
        except ValueError as e:
            return JSONResponse(status_code=400, content={"error": str(e)})

    # Parse dates
    try:
        after_date = parse_date(after) if after else None
        before_date = parse_date(before) if before else None
    except ValueError as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid date: {e}"})

    config = app.state.config

    try:
        search_fn = {"grep": search_grep, "semantic": search_semantic, "hybrid": search_hybrid}.get(mode)
        if not search_fn:
            return JSONResponse(status_code=400, content={"error": f"Invalid mode: {mode}"})

        result = search_fn(
            config=config,
            term=q,
            source=source,
            n=n,
            filters=filters,
            after=after_date,
            before=before_date,
            context=context,
            full=full,
            scores=scores,
            metadata=metadata,
        )
        return result
    except Exception as e:
        log.exception("Search failed")
        return JSONResponse(status_code=500, content={"error": str(e)})
