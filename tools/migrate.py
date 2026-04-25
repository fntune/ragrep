"""Migrate a Python-built data/index/ to the Rust-readable format.

Reads:   data/index/{faiss.index, chunks.pkl, embed_cache.pkl}
         data/index/.. (config.toml for embedding provider/model)
Writes:  data/index/embeddings.bin
         data/index/chunks.msgpack
         data/index/embed_cache/{provider}--{model}.bin   (binary, see Rust embed::cache)

Drops:   bm25.pkl (rebuild via `ragrep rebuild-bm25` after migration).

Run inside the existing Python venv (the one that wrote the pickles).
"""

import pickle
import struct
import sys
import tomllib
from pathlib import Path

import faiss
import msgpack
import numpy as np

CACHE_MAGIC = 0x5241_4331  # "RAG1"
CACHE_VERSION = 1
DEFAULT_PROVIDER = "voyage"
DEFAULT_MODEL = "voyage-code-3"


def resolve_embedding(repo_root: Path) -> tuple[str, str]:
    """Resolve (provider, model) from config.toml; fall back to voyage defaults."""
    cfg_path = repo_root / "config.toml"
    if not cfg_path.exists():
        return DEFAULT_PROVIDER, DEFAULT_MODEL
    with open(cfg_path, "rb") as f:
        cfg = tomllib.load(f)
    embedding = cfg.get("embedding", {})
    return (
        embedding.get("provider", DEFAULT_PROVIDER),
        embedding.get("model_name", DEFAULT_MODEL),
    )


def migrate_embeddings(index_dir: Path) -> None:
    src = index_dir / "faiss.index"
    dst = index_dir / "embeddings.bin"
    if not src.exists():
        sys.exit(f"missing: {src}")

    idx = faiss.read_index(str(src))
    embs = idx.reconstruct_n(0, idx.ntotal)
    if embs.dtype.name != "float32":
        embs = embs.astype("float32")
    embs.tofile(dst)

    expected = idx.ntotal * idx.d * 4
    actual = dst.stat().st_size
    if actual != expected:
        sys.exit(f"embeddings.bin size mismatch: expected {expected}, got {actual}")
    print(f"embeddings.bin: {actual:,} bytes ({idx.ntotal} × {idx.d} × 4)")


def migrate_chunks(index_dir: Path) -> None:
    src = index_dir / "chunks.pkl"
    dst = index_dir / "chunks.msgpack"
    if not src.exists():
        sys.exit(f"missing: {src}")

    with open(src, "rb") as f:
        chunks = pickle.load(f)

    serializable = [
        {
            "id": c.id,
            "doc_id": c.doc_id,
            "content": c.content,
            "title": c.title,
            "source": c.source,
            "metadata": dict(c.metadata),
        }
        for c in chunks
    ]
    with open(dst, "wb") as f:
        msgpack.pack(serializable, f, use_bin_type=True)

    print(f"chunks.msgpack: {len(chunks):,} chunks, {dst.stat().st_size:,} bytes")


def migrate_embed_cache(index_dir: Path, provider: str, model: str) -> None:
    """Convert embed_cache.pkl → embed_cache/{provider}--{model}.bin (binary).

    Format mirrors `rust/src/embed/cache.rs`:
        u32 magic (RAG1) + u32 version + u32 dim + u64 n + n*(32 sha256 + dim*4 f32)

    ~50% smaller than the previous msgpack format (msgpack-python serializes
    f32 lists as float64) and 2x faster to load on the Rust side.
    """
    src = index_dir / "embed_cache.pkl"
    if not src.exists():
        print(f"skipping embed_cache (no {src})")
        return

    out_dir = index_dir / "embed_cache"
    out_dir.mkdir(exist_ok=True)
    dst = out_dir / f"{provider}--{model}.bin"

    with open(src, "rb") as f:
        cache = pickle.load(f)

    if not cache:
        print(f"embed_cache: empty input, skipping")
        return

    # Discover dim from the first value.
    first_v = next(iter(cache.values()))
    if isinstance(first_v, np.ndarray):
        dim = int(first_v.shape[-1])
    else:
        dim = len(first_v)

    n = len(cache)
    tmp = dst.with_suffix(".bin.tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<I", CACHE_MAGIC))
        f.write(struct.pack("<I", CACHE_VERSION))
        f.write(struct.pack("<I", dim))
        f.write(struct.pack("<Q", n))
        for k, v in cache.items():
            # Key normalization: hex string (64 chars) → 32 raw bytes.
            if isinstance(k, str):
                key_bytes = bytes.fromhex(k)
            elif isinstance(k, (bytes, bytearray)):
                key_bytes = bytes(k)
            else:
                key_bytes = bytes(k)
            assert len(key_bytes) == 32, f"sha256 key must be 32 bytes, got {len(key_bytes)}"

            # Value normalization: any → float32 ndarray → little-endian bytes.
            if isinstance(v, np.ndarray):
                arr = v.astype("<f4", copy=False)
            else:
                arr = np.asarray(list(v), dtype="<f4")
            assert arr.shape == (dim,), f"vec dim mismatch: expected {dim}, got {arr.shape}"

            f.write(key_bytes)
            f.write(arr.tobytes())

    tmp.rename(dst)
    print(f"{dst.relative_to(index_dir.parent)}: {n:,} entries, {dst.stat().st_size:,} bytes")


def main() -> None:
    index_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index")
    if not index_dir.exists():
        sys.exit(f"missing: {index_dir}")

    repo_root = index_dir.parent
    provider, model = resolve_embedding(repo_root)

    print(f"migrating {index_dir} (provider={provider}, model={model}) ...")
    migrate_embeddings(index_dir)
    migrate_chunks(index_dir)
    migrate_embed_cache(index_dir, provider, model)

    bm25 = index_dir / "bm25.pkl"
    if bm25.exists():
        print(f"note: {bm25} not migrated (rebuild via `ragrep rebuild-bm25`)")

    # Clean up the now-orphan transitional msgpack cache if present.
    old_cache = index_dir / "embed_cache.msgpack"
    if old_cache.exists():
        print(f"note: {old_cache} is no longer used (Rust reads embed_cache/{provider}--{model}.bin); safe to delete")

    print("done.")


if __name__ == "__main__":
    main()
