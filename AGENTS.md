# ragrep

Rust CLI and server for self-hosted hybrid retrieval over team knowledge sources.

## Quick Start

```bash
cargo build --manifest-path rust/Cargo.toml --release --bin ragrep
./rust/target/release/ragrep --help

ragrep "your search term" -m grep
ragrep "how does X work"
```

Server mode:

```bash
ragrep serve --host 0.0.0.0 --port 8321
curl "http://localhost:8321/search?q=auth&mode=grep&n=5"
```

CLI via server:

```bash
export RAGREP_SERVER=http://localhost:8321
ragrep "search term" -m grep
```

## Architecture

```text
scrape -> normalize -> chunk -> embed -> store
                                      |
                            query -> retrieve -> rerank
```

Runtime index files:

- `embeddings.bin`
- `chunks.msgpack`
- `bm25.msgpack`
- `embed_cache/{provider}--{model}.bin`

## Rust Layout

```text
rust/src/
  cli/          clap subcommands and terminal output
  serve/        axum HTTP server
  config.rs     config loading and .env discovery
  ingest/       scrape -> normalize -> chunk -> embed -> store
  query/        grep, semantic, hybrid retrieval
  eval/         entity-graph eval harness
  index/        mmap vector search, chunks, BM25 storage
docs/           landing page and installer
tools/          old-index migration helper
```

## Commands

```bash
make install              # build release binary
make scrape               # all configured sources, or SOURCE=slack,git
make ingest               # incremental index build
make ingest FORCE=1       # re-embed from scratch
make query Q="..."        # single query
make serve                # local server on port 8321
make eval                 # eval harness
make stats                # index statistics
make inspect MODE=raw     # inspect raw data or pipeline output
make check                # cargo fmt + clippy
make test                 # cargo test
```

## Public Contracts

- `ragrep scrape [--source <csv>] [--config <path>]` writes raw JSONL files consumed by the normalizers.
- `ragrep ingest [--force] [--source <source>] [--config <path>]` writes the Rust runtime index.
- `ragrep eval [--output <path>] [--config <path>]` is the eval entrypoint.
- `ragrep serve` is the production HTTP command.
- `/search` and `ragrep --json` score fields:
  - `grep`: no score fields.
  - `semantic --scores`: `score`.
  - `hybrid --scores`: `rerank`, `rrf`, `dense`, and `bm25`.

## Migration

Keep `tools/migrate.py`. It is the only retained Python file and exists solely to convert old index files into the Rust layout. Do not add new runtime features there.

## Verification

Before claiming completion, run the smallest command set that proves the change. For broad changes, use:

```bash
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --no-deps
cargo test --manifest-path rust/Cargo.toml
cargo build --manifest-path rust/Cargo.toml --release --bin ragrep
```
