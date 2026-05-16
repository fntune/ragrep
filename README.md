# ragrep

> ripgrep for your team's knowledge base.
> Rust binary, hybrid retrieval, self-hosted.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub stars](https://img.shields.io/github/stars/fntune/ragrep?style=social)](https://github.com/fntune/ragrep/stargazers)

Search Slack, Confluence, Jira, Google Drive, Git history, Bitbucket PRs, source code, and local files from one native CLI. Ragrep builds a local Rust index with memory-mapped vectors, BM25, and optional Voyage reranking. No hosted database and no per-seat service.

```bash
curl -fsSL https://ragrep.cc/install.sh | sh

ragrep "how does the auth flow work"
```

## Why Ragrep

- **Native binary.** Install one Rust executable from GitHub Releases.
- **Hybrid retrieval.** Dense vector search combines with BM25 through Reciprocal Rank Fusion, then reranks when configured.
- **Self-hosted.** Your index is local files: `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.
- **Incremental ingest.** Content-hash embedding cache means re-indexing only embeds changed chunks.
- **Agent-friendly.** CLI and server modes both support JSON output for scripts and local agents.

## Install

```bash
curl -fsSL https://ragrep.cc/install.sh | sh
```

The installer detects OS/arch, downloads `ragrep-${target}.tar.gz` from GitHub Releases, verifies its `.sha256`, and installs to `~/.local/bin/ragrep`.

Build from source:

```bash
git clone https://github.com/fntune/ragrep
cd ragrep
cargo build --manifest-path rust/Cargo.toml --release --bin ragrep
./rust/target/release/ragrep --help
```

Create `~/.config/ragrep/.env` or a local `.env` with provider credentials. See [`.env.example`](./.env.example).

## Search

```bash
ragrep "rate limit handling"             # hybrid by default
ragrep "deploy process" -m grep          # exact substring
ragrep "sales pipeline" -m semantic      # dense only

ragrep "incident" -s slack               # source filter
ragrep "release notes" --after 2w        # relative date filter
ragrep "auth" -f author=alice            # metadata filter
ragrep "config" -n 20                    # top 20

ragrep "auth" --json
ragrep "auth" --scores
ragrep "auth" --full
```

Run against a server without a local index:

```bash
export RAGREP_SERVER=http://your-server:8321
ragrep "query"
```

For Cloud Run `*.run.app` servers, the CLI attempts `gcloud auth print-identity-token` and sends it as a bearer token.

JSON score fields are stable for clients:

- `grep`: no score fields.
- `semantic --scores`: `score`.
- `hybrid --scores`: `rerank`, `rrf`, `dense`, and `bm25`.

## Data Sources

| Source | What it ingests | Required credentials |
|--------|------------------|----------------------|
| `slack` | Messages, threads, pins, bookmarks, file contents | `SLACK_TOKEN` |
| `atlassian` | Confluence pages, Jira issues + comments | `CONFLUENCE_URL`, `JIRA_URL`, `ATLASSIAN_USERNAME`, `ATLASSIAN_API_TOKEN` |
| `gdrive` | Docs, Sheets, Slides, text-like Drive files | `GOOGLE_APPLICATION_CREDENTIALS` service account/ADC, or `GOOGLE_ACCESS_TOKEN` |
| `git` | Commit messages, diffs, changed files, PR metadata from local repos | local filesystem |
| `code` | Tracked source files from configured repos | local filesystem |
| `bitbucket` | PR descriptions, comments, approvals | `BITBUCKET_ACCESS_TOKEN` or `BITBUCKET_OAUTH_SECRET` |
| `files` | Text files locally; PDFs/Office/images through Gemini extraction | `GEMINI_API_KEY` for multimodal extraction |

```bash
ragrep scrape
ragrep scrape --source slack,git,code
ragrep ingest
ragrep ingest --force
ragrep stats
ragrep inspect raw
ragrep eval
```

Configure source options in `config.toml`:

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

## Server

```bash
ragrep serve --host 0.0.0.0 --port 8321

curl "http://localhost:8321/health"
curl "http://localhost:8321/search?q=auth+flow&mode=grep&n=5"
curl "http://localhost:8321/knowledge/search?q=kyc&mode=grep&portal_id=80000083721"
```

Server index resolution:

- `RAGREP_INDEX_DIR` if it contains `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.
- `config.toml` `[data].index_dir`.
- `RAGREP_GCS_BUCKET`, downloaded at startup to `/tmp/ragrep-index` or `RAGREP_INDEX_CACHE_DIR`.

Authentication:

- Local `ragrep serve` is unauthenticated by default.
- Set `RAGREP_AUTH_TOKEN` to require `Authorization: Bearer ...`.
- On Cloud Run, prefer platform IAM auth and set `RAGREP_AUTH_MODE=cloud-run` as an explicit deployment marker.

The Dockerfile builds the Rust binary and runs:

```bash
ragrep serve --host 0.0.0.0 --port 8080
```

### Support Knowledge Records

Ragrep can own support-chatbot article and video records directly, backed by
`data/raw/freshdesk.jsonl` and `data/raw/youtube.jsonl`:

```bash
curl "http://localhost:8321/knowledge/records/freshdesk"
curl "http://localhost:8321/knowledge/records/youtube?playlist_id=PL123"
curl "http://localhost:8321/knowledge/records/freshdesk/123"

curl -X PUT "http://localhost:8321/knowledge/records/freshdesk/123" \
  -H "content-type: application/json" \
  -d '{"title":"KYC","content":"How KYC works","status":2,"updated_at":"2026-05-16T00:00:00Z"}'

curl -X POST "http://localhost:8321/knowledge/records/youtube" \
  -H "content-type: application/json" \
  -d '{"upsert":[{"video_id":"v1","title":"Deposits","description":"How deposits work","playlist_id":"PL123"}],"delete":["old-video"]}'

curl -X DELETE "http://localhost:8321/knowledge/records/freshdesk/123"
```

Write responses include `ingest` stats and `refresh_required`. Until the server
refresh endpoint lands, restart `ragrep serve` after writes to serve the newly
published index generation.

## Providers

Embedding providers are HTTP-only in the Rust runtime:

| Provider | Default model | Env var |
|----------|---------------|---------|
| `voyage` | `voyage-code-3` | `VOYAGE_API_KEY` |
| `openai` | configured in `config.toml` | `OPENAI_API_KEY` |
| `gemini` | configured in `config.toml` | `GEMINI_API_KEY` |

Reranking currently supports Voyage (`rerank-2.5`).

## Migration

Existing Python-built indexes use `faiss.index`, `chunks.pkl`, and `bm25.pkl`. The Rust runtime uses `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.

See [MIGRATION.md](./MIGRATION.md) for the old-index migration path.

## Development

```bash
make check
make test
make stats

# direct equivalents
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --no-deps
cargo test --manifest-path rust/Cargo.toml
```

Release artifacts are produced by `.github/workflows/rust.yml` for Linux, macOS, and Windows. `docs/install.sh` expects those artifact names exactly:

- `ragrep-x86_64-unknown-linux-gnu.tar.gz`
- `ragrep-aarch64-unknown-linux-gnu.tar.gz`
- `ragrep-x86_64-apple-darwin.tar.gz`
- `ragrep-aarch64-apple-darwin.tar.gz`
- `ragrep-x86_64-pc-windows-msvc.zip`

## License

MIT - see [LICENSE](LICENSE).
