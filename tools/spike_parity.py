"""Capture Python FAISS top-10 baselines for the Rust spike parity check.

For each query string, embed via Voyage, run faiss IP search, dump:
  data/spike/q_<slug>.json = { query, embedding[1024], top10: [{idx, score}] }
"""

import json
import re
import sys
from pathlib import Path

import faiss

from ragrep.config import load_env_files
from ragrep.ingest.embed import VoyageEmbedder

QUERIES = ["how does the auth flow work", "deploy", "incident"]
TOP_K = 10


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:32]


def main() -> None:
    load_env_files()
    out_dir = Path("data/spike")
    out_dir.mkdir(parents=True, exist_ok=True)

    idx = faiss.read_index("data/index/faiss.index")
    print(f"loaded faiss: ntotal={idx.ntotal}, d={idx.d}")
    embedder = VoyageEmbedder("voyage-code-3")

    for q in QUERIES:
        emb = embedder.embed_query(q)  # shape (1, 1024), L2-normalized
        scores, indices = idx.search(emb, TOP_K)
        top10 = [
            {"idx": int(i), "score": float(s)}
            for i, s in zip(indices[0], scores[0], strict=True)
        ]
        record = {
            "query": q,
            "embedding": emb[0].tolist(),
            "top10": top10,
        }
        path = out_dir / f"q_{slug(q)}.json"
        path.write_text(json.dumps(record))
        print(f"wrote {path}: top1 idx={top10[0]['idx']} score={top10[0]['score']:.6f}")


if __name__ == "__main__":
    sys.exit(main())
