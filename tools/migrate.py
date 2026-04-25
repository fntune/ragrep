"""Migrate a Python-built data/index/ to the Rust-readable format.

Reads:   data/index/{faiss.index, chunks.pkl, embed_cache.pkl}
Writes:  data/index/{embeddings.bin, chunks.msgpack, embed_cache.msgpack}

Drops:   bm25.pkl (rebuild via `ragrep rebuild-bm25` after migration).

Run inside the existing Python venv (the one that wrote the pickles).
"""

import pickle
import sys
from pathlib import Path

import faiss
import msgpack
import numpy as np


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


def migrate_embed_cache(index_dir: Path) -> None:
    src = index_dir / "embed_cache.pkl"
    dst = index_dir / "embed_cache.msgpack"
    if not src.exists():
        print(f"skipping embed_cache (no {src})")
        return

    with open(src, "rb") as f:
        cache = pickle.load(f)

    # Cache is dict[sha256_hex_str, np.ndarray | list[float]]. Emit as a list of
    # [bytes(32), list[float]] pairs — Rust loads it into HashMap<[u8;32], Vec<f32>>.
    serializable = []
    for k, v in cache.items():
        if isinstance(v, np.ndarray):
            v = v.astype("float32").tolist()
        else:
            v = list(map(float, v))
        if isinstance(k, str):
            k = bytes.fromhex(k)
        elif not isinstance(k, (bytes, bytearray)):
            k = bytes(k)
        serializable.append([k, v])

    with open(dst, "wb") as f:
        msgpack.pack(serializable, f, use_bin_type=True)

    print(f"embed_cache.msgpack: {len(serializable):,} entries, {dst.stat().st_size:,} bytes")


def main() -> None:
    index_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index")
    if not index_dir.exists():
        sys.exit(f"missing: {index_dir}")

    print(f"migrating {index_dir} ...")
    migrate_embeddings(index_dir)
    migrate_chunks(index_dir)
    migrate_embed_cache(index_dir)

    bm25 = index_dir / "bm25.pkl"
    if bm25.exists():
        print(f"note: {bm25} not migrated (rebuild via `ragrep rebuild-bm25`)")

    print("done.")


if __name__ == "__main__":
    main()
