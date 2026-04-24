"""Run RAG ingestion pipeline on Modal.

When using Voyage AI provider (default): runs on CPU, embeds via API.
When using sentence-transformers provider: runs on GPU (L4).

Setup (one-time):
    pip install modal && modal token new
    modal volume create ragrep-index                                     # or: RAGREP_MODAL_VOLUME=my-vol
    modal secret create voyage-api-key VOYAGE_API_KEY=pa-...             # or: RAGREP_VOYAGE_SECRET=my-secret

Usage:
    modal run --detach modal_ingest.py

Download index:
    modal volume get ragrep-index index data/index

Customize Modal names via env vars before running:
    RAGREP_MODAL_VOLUME   Modal volume for the output index (default: ragrep-index)
    RAGREP_VOYAGE_SECRET  Modal secret holding VOYAGE_API_KEY (default: voyage-api-key)
"""

import os

import modal

_VOLUME_NAME = os.environ.get("RAGREP_MODAL_VOLUME", "ragrep-index")
_VOYAGE_SECRET = os.environ.get("RAGREP_VOYAGE_SECRET", "voyage-api-key")

app = modal.App("ragrep-ingest")

vol = modal.Volume.from_name(_VOLUME_NAME, create_if_missing=True)

# Voyage provider image (CPU, no torch needed for embedding)
voyage_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "voyageai>=0.3.0",
        "faiss-cpu>=1.9.0",
        "rank-bm25>=0.2.2",
        "httpx>=0.27.0",
        "numpy>=1.26.0",
        "tokenizers>=0.21.0",
    )
    .add_local_python_source("src")
    .add_local_file("config.toml", remote_path="/root/config.toml")
    .add_local_dir("data/raw", remote_path="/root/data/raw")
)

# ST provider image (GPU, full torch stack)
st_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "sentence-transformers>=2.7.0",
        "transformers>=4.51.0",
        "torch>=2.5.0",
        "faiss-cpu>=1.9.0",
        "rank-bm25>=0.2.2",
        "httpx>=0.27.0",
        "huggingface_hub>=0.26.0",
    )
    .add_local_python_source("src")
    .add_local_file("config.toml", remote_path="/root/config.toml")
    .add_local_dir("data/raw", remote_path="/root/data/raw")
)


@app.function(
    image=voyage_image,
    timeout=3600,
    volumes={"/vol": vol},
    secrets=[modal.Secret.from_name(_VOYAGE_SECRET)],
)
def ingest_voyage() -> None:
    """Run ingestion with Voyage AI embeddings (CPU, API-based)."""
    _run_ingest()


@app.function(
    image=st_image,
    gpu="L4",
    timeout=3600,
    volumes={"/vol": vol},
)
def ingest_st() -> None:
    """Run ingestion with sentence-transformers on GPU."""
    import os

    os.environ["HF_HOME"] = "/vol/hf_cache"

    from ragrep.config import load_config

    config = load_config()
    # Patch device to cuda
    config = config.__class__(
        data=config.data,
        ingest=config.ingest,
        embedding=config.embedding.__class__(
            provider=config.embedding.provider,
            model_name=config.embedding.model_name,
            device="cuda",
        ),
        reranker=config.reranker,
        retrieval=config.retrieval,
        generation=config.generation,
        scrape=config.scrape,
    )
    _run_ingest(config)


def _run_ingest(config=None) -> None:
    """Shared ingest logic."""
    import logging
    import os
    import shutil
    from pathlib import Path

    os.chdir("/root")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("modal_ingest")

    if config is None:
        from ragrep.config import load_config
        config = load_config(Path("config.toml"))

    log.info("Starting ingestion on Modal (provider=%s)", config.embedding.provider)

    from ragrep.ingest.pipeline import ingest as run_ingest

    stats = run_ingest(config, force=True)

    if stats.documents == 0:
        log.error("No documents ingested!")
        return

    log.info(
        "Ingestion complete: %d docs → %d chunks in %.1fs",
        stats.documents, stats.chunks, stats.elapsed_s,
    )

    # Copy index to persistent volume
    index_dir = Path("data/index")
    vol_index = Path("/vol/index")
    if vol_index.exists():
        shutil.rmtree(vol_index)
    shutil.copytree(index_dir, vol_index)
    vol.commit()

    log.info("Index saved to volume at /vol/index")
    for name in ("faiss.index", "chunks.pkl", "bm25.pkl"):
        path = vol_index / name
        if path.exists():
            size_mb = path.stat().st_size / 1024 / 1024
            log.info("  %s: %.1f MB", name, size_mb)


@app.local_entrypoint()
def main() -> None:
    import tomllib
    from pathlib import Path

    with open(Path("config.toml"), "rb") as f:
        raw = tomllib.load(f)

    provider = raw.get("embedding", {}).get("provider", "voyage")
    if provider == "voyage":
        ingest_voyage.remote()
    else:
        ingest_st.remote()
