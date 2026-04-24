# ragrep

> **ripgrep for your team's knowledge base.**
> Hybrid retrieval, self-hosted, single command.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![GitHub stars](https://img.shields.io/github/stars/fntune/ragrep?style=social)](https://github.com/fntune/ragrep/stargazers)

Search across your Slack, Confluence, Jira, Google Drive, Git history, Bitbucket PRs, and local files from one CLI. Dense vectors + BM25 + reranking. No SaaS, no per-seat pricing, no API quotas you didn't choose.

```bash
curl -fsSL https://ragrep.cc/install.sh | sh

ragrep "how does the auth flow work"
```

---

## Why ragrep

- **Hybrid retrieval, not just embeddings.** FAISS cosine + BM25 fused with Reciprocal Rank Fusion, then optionally reranked. The combo recalls cross-source content that pure dense or pure sparse misses.
- **Self-hosted, no lock-in.** Your data stays on your laptop, your VM, or your Cloud Run. The index is three files (`faiss.index`, `chunks.pkl`, `bm25.pkl`) — back them up with `cp`.
- **Real ingestion that scales.** Content-hash deduped embedding cache means re-indexing only embeds what changed. Adaptive rate-limiting learns the embedding API's actual ceiling at runtime, with checkpoint resumability for multi-hour runs.

---

## Install

```bash
curl -fsSL https://ragrep.cc/install.sh | sh
```

The installer pulls [`uv`](https://docs.astral.sh/uv/) if you don't have it, then installs `ragrep` with the `[full]` extras (FAISS, embedding providers, retrieval).

Alternatives:

```bash
# Direct via uv
uv tool install 'ragrep[full] @ git+https://github.com/fntune/ragrep'

# From source
git clone https://github.com/fntune/ragrep && cd ragrep
uv sync --extra full
```

Then create `~/.config/ragrep/.env` (or a `.env` in your CWD) with your API keys — see [`.env.example`](./.env.example).

---

## 30-second demo

```bash
# Search modes — defaults to hybrid
ragrep "rate limit handling"
ragrep "deploy process" -m grep         # exact substring
ragrep "sales pipeline" -m semantic     # FAISS only

# Filter
ragrep "incident" -s slack              # one source
ragrep "release notes" --after 2w       # last 2 weeks
ragrep "auth" -f author=alice           # metadata filter
ragrep "config" -n 20                   # top 20

# Output
ragrep "auth" --json                    # JSON for agents/scripts
ragrep "auth" --scores                  # include retrieval scores
ragrep "auth" --full                    # full chunk content

# Server mode (no local index needed)
export RAGREP_SERVER=http://your-server:8321
ragrep "query"
```

---

## Compared to

|  | ragrep | LlamaIndex / LangChain | Pinecone / Vectara | ripgrep |
|---|---|---|---|---|
| Install | `curl ragrep.cc/install.sh \| sh` | `pip install` + write code | SaaS account | `brew install` |
| Hybrid retrieval | ✅ FAISS + BM25 + rerank | ⚠️ DIY composition | ⚠️ varies | ❌ exact match only |
| Multi-source ingest | ✅ Slack, Confluence, Drive, Git, files | DIY | DIY or paid connectors | filesystem only |
| Self-hosted | ✅ | ✅ (you assemble) | ❌ | ✅ |
| Pricing | free + your own embedding API | free + your own | per-vector / per-query | free |
| Best for | Search your team's knowledge fast | Building custom RAG pipelines | Production-grade managed RAG | Searching code |

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

```bash
ragrep scrape                       # all sources configured in config.toml
ragrep scrape --source slack,git    # subset
ragrep ingest                       # build the index (incremental)
ragrep ingest --force               # re-embed everything
ragrep stats                        # show index stats
```

Configure source-specific options in `config.toml`:

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

## Python SDK

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

## HTTP server

Serve search over HTTP for multi-client use, or to back an internal app:

```bash
make serve    # localhost:8321

curl "http://localhost:8321/search?q=auth+flow&mode=hybrid&n=5"
curl "http://localhost:8321/search?q=deploy&mode=grep"

# CLI against the server (no local index needed on the client)
export RAGREP_SERVER=http://localhost:8321
ragrep "query"
```

Set `RAGREP_GCS_BUCKET` to auto-download the index from GCS at server startup. Or set `RAGREP_INDEX_DIR` to point at a pre-positioned local index.

---

## Embedding providers

| Provider | Default model | Notes |
|----------|---------------|-------|
| `voyage` (default) | `voyage-code-3` | API, 1024d, code + text in same space |
| `sentence-transformers` | `Qwen3-Embedding-0.6B` | local, GPU/MPS recommended |

Rerankers: `voyage/rerank-2.5` (API) or `sentence-transformers/cross-encoder/ms-marco-MiniLM-L-6-v2` (local).

---

## Remote ingestion (Modal)

For embedding many sources in parallel without burning your laptop:

```bash
pip install modal && modal token new
modal volume create ragrep-index
modal secret create voyage-api-key VOYAGE_API_KEY=pa-...

modal run --detach modal_ingest.py
modal volume get ragrep-index index data/index
```

Override the volume / secret names via `RAGREP_MODAL_VOLUME` and `RAGREP_VOYAGE_SECRET` env vars before `modal run`. Voyage embedder has adaptive rate limiting and checkpoint resumability — multi-hour runs survive crashes.

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
4. Voyage / cross-encoder reranking
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
    harness.py     per-stage metrics (dense / BM25 / RRF / rerank)
```

---

## Development

```bash
make check    # ruff + mypy
make test     # pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup and conventions.

---

## License

MIT — see [LICENSE](LICENSE).
