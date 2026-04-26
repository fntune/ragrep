# ragrep

RAG pipeline for Aidaptive's internal knowledge base. Hybrid FAISS + BM25 retrieval with Voyage AI embeddings.

## Quick Start

```bash
# Local CLI (query only — downloads pre-built index ~792 MB)
git clone git@bitbucket.org:jarvisml/ragrep.git && cd ragrep
make setup                      # installs deps + downloads index from GCS
source .venv/bin/activate
ragrep "your search term" -m grep           # no API key needed
ragrep "how does X work"                    # hybrid mode, needs VOYAGE_API_KEY in .env

# Server mode (serves search over HTTP)
make serve                                  # local server on port 8321
curl "http://localhost:8321/search?q=miner&mode=grep&n=5"

# CLI via server (no local index needed)
export RAGREP_SERVER=http://localhost:8321
ragrep "search term" -m grep
```

**Cloud Run**: `make deploy-dev` deploys to `jarvis-ml-dev`. Server auto-downloads index from `gs://ragrep-index-jarvis-ml-dev/` at startup.

## Architecture

```
scrape → normalize → chunk → embed (incremental) → store (FAISS + BM25)
                                                        ↓
                                          query → retrieve → rerank → generate
```

## Pipeline Details

### Ingestion (`src/ingest/pipeline.py`)

1. **Normalize**: Raw JSON → `Document` (id, source, content, title, metadata)
2. **Chunk**: Token-based with overlap (512 tokens, 64 overlap)
3. **Embed**: Content-hash dedup + embedding cache. Only new/changed chunks hit the API.
4. **Store**: FAISS `IndexFlatIP` (cosine on L2-normalized vectors) + BM25Okapi + pickled chunks

**Incremental ingestion**: `embed_cache.pkl` maps `sha256(content) → embedding_vector`. First run auto-bootstraps from existing FAISS via `reconstruct_n`. `--force` ignores cache.

**Embedding ↔ chunk mapping**: Positional — `FAISS_vector[i] ↔ chunks[i]`. Built from the same `texts` list in the same order, saved together in `save_index()`.

### Embedding (`src/ingest/embed.py`)

**Providers** (factory pattern via `make_embedder()`):

| Provider | Model | Dim | Notes |
|----------|-------|-----|-------|
| `voyage` (default) | `voyage-code-3` | 1024 | API, code + text in same space |
| `sentence-transformers` | `Qwen3-Embedding-0.6B` | varies | Local, needs GPU/MPS |

**Voyage rate limiting** (`_AdaptiveThrottle`):
- Starts at 30s delay, ramps down 0.75x after 3 consecutive successes
- On rate limit: 1.5x backoff, sets floor (learned lower bound)
- Floor erodes after 10 successes near it (allows probing when limits improve)
- `_MAX_API_BATCH = 16` to stay under 10K TPM per request
- `voyageai.Client(max_retries=0)` — SDK default, means 1 attempt, no internal retries
- Catches `voyageai.error.RateLimitError`, extracts `retry-after` from headers
- Checkpoint every 500 batches to `.embed_checkpoint.npz` for crash recovery
- `NTFY_TOPIC` env var for hourly progress notifications via ntfy.sh

### Retrieval (`src/query/`)

1. **Dense**: FAISS inner-product search (cosine similarity on normalized vectors)
2. **Sparse**: BM25Okapi with whitespace tokenization
3. **Fusion**: Reciprocal Rank Fusion (k=60) merges dense + BM25 results
4. **Rerank**: Voyage `rerank-2.5` or local cross-encoder, top-5 by default
5. **Generate**: Ollama HTTP API, `gemma3:4b` default

### Scrapers (`src/ingest/scrape_*.py`)

| Source | Scraper | What it captures | Env vars |
|--------|---------|-----------------|----------|
| Slack | `scrape_slack.py` | Messages, threads, pins, bookmarks, files | `SLACK_TOKEN` |
| Atlassian | `scrape_atlassian.py` | Confluence pages, Jira issues + comments | `CONFLUENCE_URL`, `JIRA_URL`, `ATLASSIAN_*` |
| Google Drive | `scrape_gdrive.py` | Docs, Sheets, Slides, PDFs | `gcloud auth` |
| Git | `scrape_git.py` | Commit messages only (not source code) | — |
| Bitbucket | `scrape_bitbucket.py` | PR descriptions + comments | `BITBUCKET_OAUTH_SECRET` |
| Files | `extract.py` | PDF, DOCX, PPTX, XLSX via Gemini | `google-genai` |

**Git scraper limitation**: Only captures `git log` output (subject + body). Does not scrape source code files or diffs. Code file scraping is planned.

## Eval Harness (`src/eval/`)

### Entity Graph (`entity_graph.py`)

Extracts entities from chunks to build automatic ground truth:
- **Jira IDs**: Regex `[A-Z]{2,10}-\d{1,6}`, filtered for false positives (UTF-8, etc.)
- **Service names**: Pattern dict for miner, airflow, entity_store, flagr, syncer, openclaw, etc.
- **Cross-source clusters**: Entities mentioned in 2+ sources serve as eval cases
- **Query generation**: Jira entities use ticket title from atlassian chunk; service entities use "How does {service} work?"

### Metrics (`harness.py`)

Per-stage breakdown (dense, BM25, RRF, rerank):
- **source_recall**: fraction of expected sources in top-K results
- **entity_recall**: fraction of ground truth chunks in top-K
- **MRR**: reciprocal rank of first ground truth hit

### Baseline Results (Feb 2025)

```
  Stage   SrcRecall   EntityRecall     MRR
   dense       74.5%          16.4%   0.609
    bm25       81.4%          31.3%   0.582
     rrf       86.3%          34.6%   0.671
  rerank       60.8%          18.5%   0.715
```

Key finding: RRF fusion improves cross-source recall (86% vs 74%/81%). Reranker top-5 drops source diversity but improves MRR.

## Current Index Stats

- 32,710 docs → 68,866 chunks
- Sources: git=21K, gdrive=25K, file=15K, atlassian=4K, slack=3.5K, bookmark=72, pin=11
- Index size: 464 MB (faiss.index=270MB, chunks.pkl=98MB, bm25.pkl=98MB, embed_cache.pkl=270MB)
- 67,313 unique content hashes in embedding cache

## Commands

```bash
make setup                # One-command install: deps + download index from GCS (~792 MB)
make install              # Install deps only (uv sync --extra dev --extra serve)
make download-index       # Download pre-built index from GCS
make upload-index         # Upload local index to GCS (maintainer only)
make serve                # Run local search server on port 8321
make deploy-dev           # Deploy to Cloud Run (jarvis-ml-dev)
make scrape               # All sources (or SOURCE=slack,git)
make ingest               # Incremental (reuses cached embeddings)
make ingest FORCE=1       # Re-embed everything from scratch
make query Q="..."        # Single query (or omit Q for interactive REPL)
make eval                 # Entity graph eval, outputs data/eval_results.json
make stats                # Index statistics
make check                # Ruff lint + format + mypy
make test                 # pytest
make clean                # Remove data/index/
```

## Distribution

**Public install (planned)**: `curl ragrep.sh | sh` — vanity domain serves a bash installer that picks the right OS/arch binary from GitHub Releases. Same pattern as rustup/bun/deno. Domain `ragrep.sh` is the single-token namespace we own; it also enables `go install ragrep.sh/cmd/ragrep@latest` via a `go-import` meta tag if a Go rewrite happens.

Why not registries: `ragrep` is taken on PyPI (pierce403, "Local semantic code recall…"), crates.io ("A fast, natural language code search tool", v0.2.0, active), and npm (squat). Single-name registry installs are not recoverable. The vanity domain is the only path to a single-token install command.

Pre-publish blockers (must fix before public install path goes live):

- `make setup` downloads `gs://ragrep-index-jarvis-ml-dev/` — internal company data. The public installer must not inherit this; ship a binary-only flow and require users to bring their own sources/index.
- `cli.py:_PROJECT_ROOT` (`os.chdir`, `.env` load, `Path("config.toml")` default) assumes a dev clone. Breaks when installed outside the repo.
- README and AGENTS.md identify the project as Aidaptive-internal; needs generic rewrite for public release.

## Remote Execution

Embedding on oracle VM (`ssh oracle`): rsync project, run `run_ingest.sh` (sources .env, sets NTFY_TOPIC). Modal also supported via `modal_ingest.py` (Voyage=CPU, ST=GPU/L4).

## Config (`config.toml`)

Provider selected via `[embedding].provider` and `[reranker].provider` ("voyage" or "sentence-transformers"). Embedder/reranker created lazily — Voyage API key only needed when there are new chunks to embed.

## Dependencies

Main: `voyageai`, `sentence-transformers`, `torch`, `faiss-cpu`, `rank-bm25`, `httpx`
Scrape: `slack-sdk`, `aiohttp`, `google-auth`, `google-api-python-client`
Extract: `google-genai`, `pypdf`, `python-docx`, `openpyxl`, `python-pptx`
Dev: `pytest`, `ruff`, `mypy`
