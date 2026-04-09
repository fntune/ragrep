# ragrep

> Hybrid FAISS + BM25 RAG pipeline. Ingest any document source, build a searchable index, query with natural language.

```bash
make scrape SOURCE=slack    # pull data from a source
make ingest                  # build FAISS + BM25 index (incremental)
make query Q="how does X work"
```

Or use the Python SDK directly:

```python
from ragrep import Index, Document, EmbedModel

index = Index(
    "data/index",
    embedding_models=[EmbedModel(provider="voyage", model_name="voyage-code-3")],
)
index.ingest([Document(id="1", source="docs", content="...", title="...")])
results = index.query("how does auth work")
```

---

## What it is

ragrep is a self-hosted RAG pipeline with a focus on retrieval quality. It combines dense vector search (FAISS), sparse keyword search (BM25), and reranking into a single pipeline with a clean Python SDK.

**Key design points:**
- **Hybrid retrieval** — FAISS cosine + BM25, fused with Reciprocal Rank Fusion (k=60)
- **Multi-model embedding** — run multiple embedding models in parallel, fuse with RRF for better recall
- **Incremental ingestion** — content-hash cache means re-indexing only processes new or changed documents
- **Adaptive rate limiting** — learns the API's actual throughput boundary at runtime; checkpoint resumability for long embedding runs
- **Multi-source scrapers** — Slack, Confluence/Jira, Google Drive, Git commits, Bitbucket PRs, local files
- **CLI + HTTP server** — `ragrep "query"` as a local CLI, or serve over HTTP for multi-client use

---

## Installation

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/fntune/ragrep
cd ragrep
uv sync --extra full   # includes FAISS, Voyage AI, sentence-transformers
```

Copy `.env.example` to `.env` and fill in your API keys.

---

## Usage

### CLI

```bash
ragrep "how does the auth flow work"
ragrep "deployment process" -n 10          # top 10 results
ragrep "slack token" -s slack              # filter by source
ragrep "pricing logic" --after 3m          # last 3 months
ragrep "query" --json                      # compact JSON output
ragrep "query" --scores                    # include retrieval scores
ragrep "query" --server http://localhost:8321
```

### Makefile

```bash
make install                     # set up venv
make scrape                      # all sources
make scrape SOURCE=slack         # single source
make ingest                      # incremental build
make ingest FORCE=1              # re-embed everything
make query Q="how does X work"
make query                       # interactive REPL
make stats                       # index statistics
make eval                        # evaluation harness
make serve                       # HTTP server on :8321
```

### Python SDK

```python
from ragrep import Index, Document, EmbedModel

index = Index(
    index_dir="data/index",
    embedding_models=[
        EmbedModel(provider="voyage", model_name="voyage-code-3"),
    ],
    reranker_provider="voyage",
    reranker_model="rerank-2.5",
    dedup_threshold=0.5,
)

stats = index.ingest([
    Document(id="doc1", source="docs", content="...", title="Getting Started"),
    Document(id="doc2", source="code", content="...", title="auth.py"),
])

results = index.query("how does authentication work", top_k=5)
for r in results:
    print(r.score, r.title, r.snippet)
```

Multi-model embedding for better recall:

```python
index = Index(
    "data/index",
    embedding_models=[
        EmbedModel(provider="voyage", model_name="voyage-code-3"),
        EmbedModel(provider="sentence-transformers", model_name="Qwen3-Embedding-0.6B"),
    ],
)
```

---

## Data sources

| Source | What it ingests | Required credentials |
|--------|----------------|----------------------|
| `slack` | Messages, threads, pins, bookmarks, file contents | `SLACK_TOKEN` |
| `atlassian` | Confluence pages, Jira issues + comments | `CONFLUENCE_URL`, `JIRA_URL`, `ATLASSIAN_USERNAME`, `ATLASSIAN_API_TOKEN` |
| `gdrive` | Docs, Sheets, Slides, PDFs | `gcloud auth application-default login` |
| `git` | Commit messages from local repos | — (local filesystem) |
| `bitbucket` | PR descriptions + comments | `BITBUCKET_OAUTH_SECRET` |
| `files` | PDF, DOCX, PPTX, XLSX (extracted via Gemini) | `GOOGLE_API_KEY` |

Configure in `config.toml`:

```toml
[scrape.slack]
date_cutoff = "2024-01-01"

[scrape.git]
repos = ["../myrepo", "../otherrepo"]
since = "2024-01-01"

[scrape.bitbucket]
workspace = "your-workspace"
states = ["MERGED", "OPEN"]
since = "2024-01-01"
```

---

## Embedding providers

| Provider | Model | Notes |
|----------|-------|-------|
| `voyage` (default) | `voyage-code-3` | API, 1024d, code + text in same space |
| `sentence-transformers` | `Qwen3-Embedding-0.6B` | local, GPU/MPS recommended |

Rerankers: `voyage/rerank-2.5` (API) or `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` (local).

---

## HTTP server

```bash
make serve    # port 8321

curl "http://localhost:8321/search?q=auth+flow&mode=hybrid&n=5"
curl "http://localhost:8321/search?q=deploy&mode=grep"

# CLI against server
export RAGREP_SERVER=http://localhost:8321
ragrep "query"
```

Set `RAGREP_GCS_BUCKET` to auto-download the index from GCS at server startup.

---

## Remote ingestion (Modal)

```bash
pip install modal && modal token new
modal run --detach modal_ingest.py
modal volume get ragrep-index index data/index
```

Voyage embedder has adaptive rate limiting and checkpoint resumability. Set `NTFY_TOPIC` for hourly progress notifications via ntfy.sh.

---

## Architecture

```
scrape → normalize → chunk → embed (incremental) → store (FAISS + BM25)
                                                         |
                                           query → retrieve → rerank → generate
```

**Retrieval pipeline:**
1. Embed query with each configured model
2. FAISS cosine search + BM25 per model
3. RRF fusion (`k=60`) across all result lists
4. Voyage/cross-encoder reranking
5. Jaccard dedup on content overlap

**Incremental ingestion:** `embed_cache.pkl` maps `sha256(content) → vector`. Only new/changed chunks hit the embedding API. First run bootstraps from existing FAISS via `reconstruct_n`.

```
src/ragrep/
  __init__.py      public API: Index, Document, EmbedModel
  index.py         SDK: ingest() and query()
  models.py        Document, Chunk, SearchResult, QueryResult
  cli.py           CLI
  server.py        FastAPI HTTP server
  ingest/
    pipeline.py    orchestrator
    scrape_*.py    source scrapers
    embed.py       providers + adaptive rate limiting + checkpointing
    store.py       FAISS + BM25 persistence
  query/
    retrieve.py    hybrid retrieval + RRF
    rerank.py      reranking providers
    generate.py    Ollama generation
  eval/
    harness.py     evaluation with per-stage metrics (dense/BM25/RRF/rerank)
```

---

## Development

```bash
make check    # ruff + mypy
make test     # pytest
```

---

## License

MIT
