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
    # 1. Check local data/index/ relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    local_dir = project_root / "data" / "index"
    if (local_dir / "faiss.index").exists():
        log.info("Using local index at %s", local_dir)
        return local_dir

    # 2. Check RAGREP_INDEX_DIR env var
    env_dir = os.environ.get("RAGREP_INDEX_DIR")
    if env_dir and (Path(env_dir) / "faiss.index").exists():
        log.info("Using index from RAGREP_INDEX_DIR=%s", env_dir)
        return Path(env_dir)

    # 3. Download from GCS
    bucket_name = os.environ.get("RAGREP_GCS_BUCKET")
    if not bucket_name:
        raise RuntimeError("No index found locally or via RAGREP_INDEX_DIR, and RAGREP_GCS_BUCKET is not set")
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
    """Load ragrep config, falling back to defaults."""
    project_root = Path(__file__).resolve().parent.parent.parent
    config_path = project_root / "config.toml"
    if not config_path.exists():
        # Minimal config for server mode — use defaults
        config_path = None

    from ragrep.config import load_config

    if config_path:
        config = load_config(config_path)
    else:
        config = load_config()

    # Override index_dir with resolved path
    return config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load index at startup."""
    global _INDEX_DIR, _CONFIG

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    # Load .env if present
    project_root = Path(__file__).resolve().parent.parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                os.environ.setdefault(key.strip(), val.strip())

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
    from ragrep.search import parse_date, parse_filters, search_grep, search_hybrid, search_semantic

    # Parse filters
    filters = None
    if filter:
        try:
            filters = parse_filters(filter.split(","))
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
