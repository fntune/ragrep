# Python Index Migration

The Rust runtime does not read the old Python index files directly.

Python runtime layout:

- `faiss.index`
- `chunks.pkl`
- `bm25.pkl`
- `embed_cache.pkl`

Rust runtime layout:

- `embeddings.bin`
- `chunks.msgpack`
- `bm25.msgpack`
- `embed_cache/{provider}--{model}.bin`

## Migrate An Existing Index

Run the migration script inside the Python environment that can still import the old dependencies and read the pickles:

```bash
uv sync --extra full --extra dev
uv run python tools/migrate.py data/index
```

Then rebuild BM25 in the Rust format:

```bash
cargo run --manifest-path rust/Cargo.toml -- rebuild-bm25 --config config.toml
```

Check the result:

```bash
cargo run --manifest-path rust/Cargo.toml -- stats --config config.toml
cargo run --manifest-path rust/Cargo.toml -- "auth" -m grep -n 5 --config config.toml
```

## Fresh Rust Index

For new indexes, use the Rust pipeline directly:

```bash
cargo run --manifest-path rust/Cargo.toml -- scrape --config config.toml
cargo run --manifest-path rust/Cargo.toml -- ingest --config config.toml
```

`ragrep ingest` writes all runtime files directly:

- `data/index/embeddings.bin`
- `data/index/chunks.msgpack`
- `data/index/bm25.msgpack`
- `data/index/embed_cache/{provider}--{model}.bin`

## Notes

- Keep `tools/migrate.py` until the transition window closes. It is intentionally Python because pickle and FAISS migration are Python-owned legacy concerns.
- The Rust v1 runtime supports HTTP embedding providers: Voyage, OpenAI, and Gemini. It does not load local sentence-transformers models.
- If your Python index used sentence-transformers, create a fresh Rust index with an HTTP provider before comparing semantic or hybrid results. Grep parity is independent of embedding provider.
- Upload Rust server indexes to GCS with the Rust filenames: `embeddings.bin`, `chunks.msgpack`, and `bm25.msgpack`. The Cloud Run server does not download `faiss.index`, `chunks.pkl`, or `bm25.pkl`.
