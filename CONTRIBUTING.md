# Contributing to ragrep

Issues and PRs are welcome, especially for new scrapers, retrieval tuning, server deployment, and CLI rough edges.

## Setup

```bash
git clone https://github.com/fntune/ragrep
cd ragrep
cargo build --manifest-path rust/Cargo.toml --release --bin ragrep
./rust/target/release/ragrep --help
```

Copy `.env.example` to `.env` and fill in the API keys needed for the sources or providers you are working on. Most code paths do not require every key.

## Run The Checks

```bash
make check
make test

# direct commands
cargo fmt --manifest-path rust/Cargo.toml --check
cargo clippy --manifest-path rust/Cargo.toml --all-targets --no-deps
cargo test --manifest-path rust/Cargo.toml
```

PRs should leave these checks passing.

## Conventions

- Keep changes close to the relevant Rust module.
- Prefer direct, typed data flow over broad abstractions.
- Validate at API, env, IO, and filesystem boundaries.
- Do not swallow errors; surface source context with `anyhow::Context`.
- Add comments only for non-obvious constraints or compatibility behavior.
- Keep public JSON and CLI output stable for scripts.

## Where Things Live

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

## Adding A Scraper

1. Add or extend a module under `rust/src/ingest/scrape/`.
2. Wire the source into `rust/src/cli/scrape.rs` if it is new.
3. Keep the raw JSONL filename compatible with the normalizer.
4. Document required env vars in `.env.example` and `README.md`.
5. Add focused tests for record shape and boundary failures.

## Filing Issues

For bugs, include `ragrep --version`, the command you ran, and the full error output.

For feature requests, include the use case and concrete CLI or HTTP examples.
