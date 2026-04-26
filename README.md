# ragrep

> **ripgrep for your team's knowledge base.**
> Hybrid retrieval, self-hosted, single binary.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.89%2B-orange.svg)](https://www.rust-lang.org/)
[![GitHub stars](https://img.shields.io/github/stars/fntune/ragrep?style=social)](https://github.com/fntune/ragrep/stargazers)

Search Slack, Confluence, Jira, Google Drive, Git history, Bitbucket PRs, source code, and local files from one CLI. Ragrep builds a local hybrid index with dense vectors, BM25, and optional Voyage reranking. No SaaS database, no per-seat pricing.

```bash
curl -fsSL https://ragrep.cc/install.sh | sh

ragrep "how does the auth flow work"
```

## Why Ragrep

- **Hybrid retrieval.** Dense cosine search over `embeddings.bin` combines with BM25 via Reciprocal Rank Fusion, then reranks when configured.
- **Self-hosted.** Your data stays on your laptop, VM, or Cloud Run. The runtime index is `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.
- **Incremental ingest.** Content-hash embedding cache means re-indexing only embeds changed chunks.
- **Fast distribution.** The default install path is a native Rust binary from GitHub Releases. The old Python install remains available with `--legacy` during migration.

## Install

```bash
curl -fsSL https://ragrep.cc/install.sh | sh
```

The installer detects OS/arch, downloads `ragrep-${target}.tar.gz` from GitHub Releases, verifies its `.sha256`, and installs to `~/.local/bin/ragrep`.

Legacy Python install:

```bash
curl -fsSL https://ragrep.cc/install.sh | sh -s -- --legacy
```

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

## Data Sources

| Source | What it ingests | Required credentials |
|--------|------------------|----------------------|
| `slack` | Messages, threads, pins, bookmarks, file contents | `SLACK_TOKEN` |
| `atlassian` | Confluence pages, Jira issues + comments | `CONFLUENCE_URL`, `JIRA_URL`, `ATLASSIAN_USERNAME`, `ATLASSIAN_API_TOKEN` |
| `gdrive` | Docs, Sheets, Slides, text-like Drive files | `GOOGLE_APPLICATION_CREDENTIALS` service account/ADC, or `GOOGLE_ACCESS_TOKEN` |
| `git` | Commit messages, changed files, PR metadata from local repos | local filesystem |
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

## Providers

Embedding providers are HTTP-only in the Rust port:

| Provider | Default model | Env var |
|----------|---------------|---------|
| `voyage` | `voyage-code-3` | `VOYAGE_API_KEY` |
| `openai` | configured in `config.toml` | `OPENAI_API_KEY` |
| `gemini` | configured in `config.toml` | `GEMINI_API_KEY` |

Reranking currently supports Voyage (`rerank-2.5`). Local sentence-transformers remains a legacy Python capability and is not part of the Rust v1 runtime.

## Migration

Existing Python-built indexes use `faiss.index`, `chunks.pkl`, and `bm25.pkl`. The Rust runtime uses `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.

See [MIGRATION.md](./MIGRATION.md) for the migration path and the current sentence-transformers regression note.

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
