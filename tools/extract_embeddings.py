"""Extract embeddings from a Python-built faiss.index into a raw f32 array.

Output: data/index/embeddings.bin (ntotal * dim * 4 bytes, no header).
Position i in the file corresponds to chunks[i] in chunks.pkl.
"""

import struct
import sys
from pathlib import Path

import faiss


def main() -> None:
    index_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "data/index")
    src = index_dir / "faiss.index"
    dst = index_dir / "embeddings.bin"

    if not src.exists():
        sys.exit(f"missing: {src}")

    idx = faiss.read_index(str(src))
    n, d = idx.ntotal, idx.d
    print(f"loaded faiss: ntotal={n}, d={d}, metric={idx.metric_type}")

    embs = idx.reconstruct_n(0, n)
    print(f"reconstructed: shape={embs.shape}, dtype={embs.dtype}, "
          f"first norm={float((embs[0] ** 2).sum() ** 0.5):.6f}")

    if embs.dtype.name != "float32":
        embs = embs.astype("float32")

    expected_bytes = n * d * 4
    embs.tofile(dst)
    actual_bytes = dst.stat().st_size

    if actual_bytes != expected_bytes:
        sys.exit(f"byte count mismatch: expected {expected_bytes}, got {actual_bytes}")

    # Sanity-print the first 4 floats from disk to confirm byte order
    with open(dst, "rb") as f:
        head = struct.unpack("4f", f.read(16))
    print(f"wrote: {dst} ({actual_bytes:,} bytes)")
    print(f"first 4 floats from disk: {head}")
    print(f"first 4 floats in memory: {tuple(embs[0, :4].tolist())}")


if __name__ == "__main__":
    main()
