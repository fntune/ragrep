# Rust Switch-Over Plan

## Status

Ragrep is now a Rust-first project. The runtime CLI, ingest pipeline, scrapers, eval harness, stats/inspect commands, and HTTP server live under `rust/src/`.

The tracked Python runtime and package metadata are removed. The only retained Python file is `tools/migrate.py`, which exists solely to convert old indexes.

## Runtime Shape

- CLI entrypoint: `ragrep <term>` plus explicit subcommands.
- Server entrypoint: `ragrep serve --host 0.0.0.0 --port 8080`.
- Runtime index files: `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`.
- Embedding cache: `embed_cache/{provider}--{model}.bin`.
- Install flow: `curl -fsSL https://ragrep.cc/install.sh | sh`.

## Public Contracts

- `ragrep scrape [--source <csv>] [--config <path>]` writes the raw JSONL filenames consumed by the Rust normalizers.
- `ragrep ingest [--force] [--source <source>] [--config <path>]` builds the Rust runtime index.
- `ragrep eval [--output <path>] [--config <path>]` is the eval entrypoint.
- `ragrep serve` is the production server command.
- `/search` score fields are stable:
  - `grep`: no score fields.
  - `semantic`: `score` when scores are enabled.
  - `hybrid`: `rerank`, `rrf`, `dense`, and `bm25` when scores are enabled.
- `/knowledge/search` returns support-oriented `knowledges`, `videos`, and
  `youtube_search` fields without changing the generic `/search` contract.

## Old Index Migration

Old Python-built indexes can be migrated with `tools/migrate.py`.

Input files:

- `faiss.index`
- `chunks.pkl`
- `embed_cache.pkl`

Output files:

- `embeddings.bin`
- `chunks.msgpack`
- `embed_cache/{provider}--{model}.bin`

After migration, run `ragrep rebuild-bm25` to write `bm25.msgpack`.

## Verification Gates

- `cargo fmt --manifest-path rust/Cargo.toml --check`
- `cargo clippy --manifest-path rust/Cargo.toml --all-targets --no-deps`
- `cargo test --manifest-path rust/Cargo.toml`
- `cargo build --manifest-path rust/Cargo.toml --release --bin ragrep`
- `ragrep --help`, `ragrep scrape --help`, `ragrep ingest --help`, and `ragrep serve --help`
- Server smoke: `curl "http://localhost:8321/search?q=auth&mode=grep&n=5"`
- Installer smoke: `sh docs/install.sh --help`
